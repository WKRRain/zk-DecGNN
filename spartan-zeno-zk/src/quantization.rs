//! 对称均匀定点量化模块 (Symmetric Uniform Fixed-Point Quantization)
//!
//! 文档 0 节要求:
//! - 采用 INT8 或 INT16 对称量化
//! - 全局缩放因子 S_scale (如 128.00 或 256.00)
//! - 所有小数在素数域 F_p 内被严格映射为整数
//! - 矩阵乘法后必须在电路内强制执行除法截断 (Truncation)
//!
//! ## 设计决策
//! arkworks 0.4 的 PrimeField 通过 BigInt 表示内部大整数。
//! 对于 bit-decomposition / range-check 我们直接操作 prime field 的
//! 标准表示 (Montgomery back to standard→bits).

use crate::curve_config::Fp;
use ark_ff::PrimeField;
use ark_std::Zero;

/// 全局缩放因子 (对标文档中的 S_scale = 256.00)
pub const SCALE: u64 = 256;

/// 定点数的总位宽 (L=64 位, 文档要求)
pub const TOTAL_BITS: usize = 64;

/// 符号位索引
pub const SIGN_BIT: usize = TOTAL_BITS - 1;

/// 兼容旧版 witness 模块的缩放因子类型。
pub type ScaleFactor = u16;

/// 旧版 witness 代码使用的反量化模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMode {
    /// 返回带缩放解释的物理浮点值。
    Scaled,
    /// 仅把域元素解释为有符号整数，不除以缩放因子。
    NoRescale,
}

fn bigint_to_le_bytes32(big: <Fp as PrimeField>::BigInt) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for (limb_idx, limb) in big.as_ref().iter().enumerate() {
        let limb_bytes = limb.to_le_bytes();
        let offset = limb_idx * 8;
        if offset >= bytes.len() {
            break;
        }
        let end = (offset + 8).min(bytes.len());
        bytes[offset..end].copy_from_slice(&limb_bytes[..end - offset]);
    }
    bytes
}

fn bigint_to_u128_saturating(big: <Fp as PrimeField>::BigInt) -> u128 {
    let mut value = 0u128;
    for (limb_idx, limb) in big.as_ref().iter().take(2).enumerate() {
        value |= (*limb as u128) << (64 * limb_idx);
    }
    if big.as_ref().iter().skip(2).any(|limb| *limb != 0) {
        u128::MAX
    } else {
        value
    }
}

/// 量化: 将物理世界的小数 (f64) 映射为素数域标量
///
/// input: 浮点值
/// 输出: input * SCALE 的整数映射, 截断到 F_p 内。
pub fn quantize(value: f64) -> Fp {
    let scaled = (value * SCALE as f64).round() as i64;

    if scaled >= 0 {
        Fp::from(scaled as u64)
    } else {
        // 负数: 计算 F_p 内的表示 (等价于 p + scaled)
        let neg_abs = (-scaled) as u64;
        let pos_zero: Fp = Fp::zero();
        // 利用 F_p 的算术: -neg_abs ≡ p - neg_abs
        let mut result = Fp::from(neg_abs);
        result = pos_zero - result;
        result
    }
}

/// 按指定缩放因子量化浮点数，兼容旧版 witness 接口。
pub fn quantize_fp(value: f64, scale: u64) -> Fp {
    let scaled = (value * scale as f64).round() as i64;
    if scaled >= 0 {
        Fp::from(scaled as u64)
    } else {
        -Fp::from((-scaled) as u64)
    }
}

/// 反量化: 将素数域标量还原为浮点数 (仅用于端侧验证/测试)
pub fn dequantize(scalar: &Fp, _num_classifications: usize) -> f64 {
    // 将 Fp 转为 BigInt → u64 (仅当值在小范围内有效)
    let bytes = bigint_to_le_bytes32(scalar.into_bigint());

    // 尝试解析为 i128
    let mut unsigned: u128 = 0;
    for (i, b) in bytes.iter().take(16).enumerate() {
        unsigned |= (*b as u128) << (8 * i);
    }

    // 如果 unsigned > p/2, 则为负数
    // 简化: 检查 unsigned 是否超过 2^63
    if unsigned > (1u128 << 63) {
        // 负数: -(p - unsigned) 但因为我们处于小规模, 近似处理
        let neg = -((unsigned as i128).wrapping_neg() as f64);
        neg / SCALE as f64
    } else {
        unsigned as f64 / SCALE as f64
    }
}

/// 将域元素解释为有符号整数/浮点值，兼容旧版 witness 接口。
pub fn dequantize_fp(scalar: Fp, mode: QuantizationMode) -> f64 {
    let bytes = bigint_to_le_bytes32(scalar.into_bigint());
    let mut unsigned: u128 = 0;
    for (i, b) in bytes.iter().take(16).enumerate() {
        unsigned |= (*b as u128) << (8 * i);
    }

    let signed = if unsigned > (1u128 << 63) {
        (unsigned as i128).wrapping_neg() as f64
    } else {
        unsigned as f64
    };

    match mode {
        QuantizationMode::NoRescale => signed,
        QuantizationMode::Scaled => signed / SCALE as f64,
    }
}

/// 带余除法截断 (Truncation with Remainder)
///
/// 对应文档 2.1 第四步:
/// 输入 U_raw 携带 S_scale^2 的缩放因子,
/// 输出商 Q 与余数 R, 满足 U_raw = Q * S_scale + R (mod p)
///
/// 在素数域中的计算:
/// - 先将 Fp 转为 BigInt → 解析为 i128 → 执行整数除法 → 映射回 Fp
pub struct TruncationResult {
    pub quotient: Fp,  // Q, 对应截断后的基准张量
    pub remainder: Fp, // R, 0 <= R < SCALE
}

/// 执行单个标量的带余除法截断
pub fn truncate_scalar(raw: &Fp) -> TruncationResult {
    // 将 raw 转换为 BigInt
    let big = raw.into_bigint();
    let raw_u = bigint_to_u128_saturating(big);
    let q = raw_u / SCALE as u128;
    let r = raw_u % SCALE as u128;

    let quotient = Fp::from(q as u64);
    let remainder = Fp::from(r as u64);

    TruncationResult {
        quotient,
        remainder,
    }
}

/// 整除截断 (直接获取商, 省去余数提取)
pub fn truncate(raw: &Fp) -> Fp {
    truncate_scalar(raw).quotient
}

/// 比特分解: 将 Fp 标量拆解为 L 位二进制比特 (LSB 在索引 0)
///
/// 用于 ReLU 的符号位提取与绝对值比特分解。
/// 文档 2.1 第四步要求 L=64 位定点数表示。
pub fn bit_decompose(value: &Fp) -> [bool; TOTAL_BITS] {
    let big = value.into_bigint();
    let mut bits = [false; TOTAL_BITS];

    // 从 BigInt 的 limb 中提取每一位
    let limbs = big.as_ref();
    for (limb_idx, limb) in limbs.iter().enumerate() {
        for bit_idx in 0..64 {
            let global_idx = limb_idx * 64 + bit_idx;
            if global_idx < TOTAL_BITS {
                bits[global_idx] = (limb >> bit_idx) & 1u64 == 1u64;
            }
        }
    }

    bits
}

/// 符号位提取: 返回 true 表示负数, false 表示非负数
///
/// 在定点数表示中, 第 SIGN_BIT 位为 1 代表负数。
/// 但这仅对 i64 物理表示有效。在 F_p 域内, "负数" 表现为 p + negative_value。
/// 因此我们检查 value 是否 > p/2 (即是否为 F_p 内的大数)。
pub fn extract_sign_bit(value: &Fp) -> bool {
    // 对 Fp 取 BigInt 表示
    let bytes = bigint_to_le_bytes32(value.into_bigint());

    // 如果最高字节 (bytes[31]) 的最高位为 1, 且在 F_p 内这代表 > p/2
    // 对于 ed25519 标量域, p ≈ 2^252, 最高有效位在 byte[31] 的 0-3 位
    // 简便判断: 检查 byte[31] 的第 4 位
    (bytes[31] >> 4) & 1u8 == 1u8
}

/// 符号位矩阵: 对二维 Tensor (Vec<Vec<Fp>>) 提取符号位矩阵
pub fn extract_sign_matrix(matrix: &[Vec<Fp>]) -> Vec<Vec<bool>> {
    matrix
        .iter()
        .map(|row| row.iter().map(extract_sign_bit).collect())
        .collect()
}

/// 绝对值比特矩阵群: 对二维 Tensor 生成 L-1 个比特矩阵 (绝对值部分)
///
/// 返回 Vec<Vec<Vec<bool>>> 形状为 [L-1][rows][cols]
pub fn bit_decompose_matrix(matrix: &[Vec<Fp>]) -> Vec<Vec<Vec<bool>>> {
    let rows = matrix.len();
    if rows == 0 {
        return Vec::new();
    }
    let cols = matrix[0].len();

    // 为每一位初始化矩阵
    let mut bit_matrices: Vec<Vec<Vec<bool>>> = (0..TOTAL_BITS - 1)
        .map(|_| vec![vec![false; cols]; rows])
        .collect();

    for (i, row) in matrix.iter().enumerate() {
        for (j, val) in row.iter().enumerate() {
            let bits = bit_decompose(val);
            for k in 0..TOTAL_BITS - 1 {
                bit_matrices[k][i][j] = bits[k];
            }
        }
    }

    bit_matrices
}

/// 施加重构约束: 从比特矩阵与符号位重构原始标量
///
/// 对应文档公式:
/// u = (1 - 2 * sign) * sum_{k=0}^{L-2} (2^k * B_k)  (mod p)
pub fn reconstruct_from_bits(sign: bool, abs_bits: &[bool], bit_idx_max: usize) -> Fp {
    let mut sum = Fp::zero();
    for k in 0..bit_idx_max {
        if k < abs_bits.len() && abs_bits[k] {
            sum += Fp::from(1u64 << k);
        }
    }

    if sign {
        // 负数: -sum mod p
        let zero = Fp::zero();
        zero - sum
    } else {
        sum
    }
}

/// ReLU 激活截断:
/// if sign_bit == 1 → 0
/// else → value
pub fn relu_activation(value: &Fp, sign_bit: bool) -> Fp {
    if sign_bit {
        Fp::zero()
    } else {
        *value
    }
}

/// 极大负数值 (用于 Mask 极小值替换)
/// 对应文档中的 C ≈ -10^4
pub fn large_negative() -> Fp {
    quantize(-10_000.0)
}

/// 量化一批 f64 值
pub fn quantize_batch(values: &[f64]) -> Vec<Fp> {
    values.iter().map(|v| quantize(*v)).collect()
}

/// 量化二维权重矩阵
pub fn quantize_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<Fp>> {
    matrix.iter().map(|row| quantize_batch(row)).collect()
}

/// 截断二维矩阵 (批量)
pub fn truncate_matrix(matrix: &[Vec<Fp>]) -> Vec<Vec<Fp>> {
    matrix
        .iter()
        .map(|row| row.iter().map(truncate).collect())
        .collect()
}

/// 范围检查约束辅助: 确保 0 <= value < SCALE
pub fn check_range(value: &Fp) -> bool {
    // 将 value 转为 u64 并与 SCALE 比较
    let big = value.into_bigint();
    let mut bytes = [0u8; 8];
    let limbs = big.as_ref();
    for (i, limb) in limbs.iter().enumerate() {
        if i < 8 {
            // 小端序写入 u64
            let limb_bytes = limb.to_le_bytes();
            for (j, byte) in limb_bytes.iter().enumerate() {
                let idx = i * 8 + j;
                if idx < 8 {
                    bytes[idx] = *byte;
                }
            }
        }
    }
    let val_u64 = u64::from_le_bytes(bytes);
    val_u64 < SCALE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_positive() {
        let v = quantize(1.5);
        let expected = Fp::from(384u64); // 1.5 * 256 = 384
        assert_eq!(v, expected);
    }

    #[test]
    fn test_quantize_zero() {
        let v = quantize(0.0);
        assert_eq!(v, Fp::zero());
    }

    #[test]
    fn test_truncate_basic() {
        // raw = 1000, SCALE = 256 → Q = 3, R = 232
        let raw = Fp::from(1000u64);
        let result = truncate_scalar(&raw);
        assert_eq!(result.quotient, Fp::from(3u64));
        assert_eq!(result.remainder, Fp::from(232u64));
    }

    #[test]
    fn test_truncate_exact() {
        let raw = Fp::from(256u64);
        let result = truncate_scalar(&raw);
        assert_eq!(result.quotient, Fp::from(1u64));
        assert_eq!(result.remainder, Fp::zero());
    }

    #[test]
    fn test_bit_decompose_small() {
        let val = Fp::from(5u64); // 101b
        let bits = bit_decompose(&val);
        assert!(bits[0]); // LSB
        assert!(!bits[1]);
        assert!(bits[2]);
        for bit in bits.iter().take(TOTAL_BITS).skip(3) {
            assert!(!*bit);
        }
    }

    #[test]
    fn test_extract_sign_positive() {
        let val = Fp::from(100u64);
        assert!(!extract_sign_bit(&val));
    }

    #[test]
    fn test_extract_sign_negative() {
        let neg = quantize(-5.0);
        assert!(extract_sign_bit(&neg));
    }

    #[test]
    fn test_relu_activation() {
        let val = Fp::from(42u64);
        assert_eq!(relu_activation(&val, false), val);
        assert_eq!(relu_activation(&val, true), Fp::zero());
    }

    #[test]
    fn test_relu_of_negative() {
        let neg = quantize(-1.0);
        let sign = extract_sign_bit(&neg);
        assert!(sign); // should be detected as negative
        let result = relu_activation(&neg, sign);
        assert_eq!(result, Fp::zero());
    }

    #[test]
    fn test_roundtrip_small() {
        let original = 2.5;
        let q = quantize(original);
        let dq = dequantize(&q, 10);
        // 因为是精确的 (2.5 * 256 = 640), 应当完全恢复
        assert!((dq - original).abs() < 0.01);
    }

    #[test]
    fn test_batch_quantize() {
        let vals = vec![1.0, -3.0, 0.5];
        let q = quantize_batch(&vals);
        assert_eq!(q[0], Fp::from(256u64));
        assert_eq!(q[2], Fp::from(128u64));
        // q[1] 为负数, 在 F_p 中为大数
    }

    #[test]
    fn test_large_negative_constant() {
        let ln = large_negative();
        // 应当非零
        assert!(!ln.is_zero());
    }
}
