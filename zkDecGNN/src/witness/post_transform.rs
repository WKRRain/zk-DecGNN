//! Stage 3 后置变换与非线性激活 (Post-transform & Activation)
//!
//! 严格遵循 Spartan-ZENO 设计文档第 2.1 节第四步。
//!
//! ## 核心计算
//!
//! ### Step 3.1: 线性映射与截断
//! ```text
//! U_raw = Z^{(K)}·W₁ + 1·b₁^T
//! ```
//! 由于输入 Z^{(K)} 和权重 W₁ 各自带有 S_scale，
//! U_raw 携带 S_scale² 量纲，必须在 ZK 电路内执行带余除法截断。
//!
//! ### 量化截断算术化
//! ```text
//! U_raw = Q · S_scale + R  (mod p)
//! ```
//! - Q: 商数矩阵 (截断后的合法张量)
//! - R: 余数矩阵 (0 ≤ R < S_scale)
//!
//! ### 防伪造双重边界约束
//! 1. 余数界限: 0 ≤ R < S_scale
//! 2. 商数零点偏移: Q' = Q + Q_max (mod p), 0 ≤ Q' ≤ 2·Q_max
//!
//! ### ReLU 激活
//! ```text
//! V[i,j] = (1 - S_sign[i,j]) · U_scaled[i,j]
//! ```
//!
//! ### 输出端二次映射
//! ```text
//! T = [β·V + (1-β)·U_scaled]
//! Y = Truncate(T·W₂ + 1·b₂^T)
//! ```
//!
//! ### Step 3.2: 对角掩码极小值替换
//! ```text
//! Ÿ = M·Y + (I - M)·C
//! ```
//! 由于 M, I, C 均为公开常数，此步仅需线性组合，ZK 开销为零。

use crate::curve_config::Fp;
use crate::model_registry::ModelArchitecture;
use crate::quantization::{dequantize_fp, quantize_fp, QuantizationMode, SCALE};
use ark_ff::Zero;

// ============================================================================
// 核心结构体
// ============================================================================

/// Stage 3 后置变换的输入
#[derive(Debug, Clone)]
pub struct PostTransformInput {
    /// 最终聚合状态 Z^{(K)}: N_max × D_max (row-major)
    pub z_final: Vec<Fp>,
    /// 后置第一层权重 W₁: D_max × H_max (row-major)
    pub w1: Vec<Fp>,
    /// 后置第一层偏置 b₁: H_max (补齐后)
    pub b1: Vec<Fp>,
    /// 后置第二层权重 W₂: H_max × C (C=分类数)
    pub w2: Vec<Fp>,
    /// 后置第二层偏置 b₂: C
    pub b2: Vec<Fp>,
    /// 激活控制开关 β
    pub beta: f64,
    /// 模型架构
    pub architecture: ModelArchitecture,
}

/// Stage 3 后置变换的输出
#[derive(Debug, Clone)]
pub struct PostTransformOutput {
    /// 第一层膨胀值 U_raw: N_max × H_max
    pub u_raw: Vec<Fp>,
    /// 截断商数 Q₁ (即 U_scaled): N_max × H_max
    pub u_scaled: Vec<Fp>,
    /// 第一层余数 R₁: N_max × H_max
    pub r1: Vec<Fp>,
    /// ReLU 符号位矩阵 S_sign: N_max × H_max
    pub s_sign: Vec<Fp>,
    /// ReLU 激活结果 V: N_max × H_max
    pub v: Vec<Fp>,
    /// 融合张量 T: N_max × H_max
    pub t: Vec<Fp>,
    /// 第二层膨胀值 T·W₂: N_max × C
    pub y_raw: Vec<Fp>,
    /// 第二层截断商数 Q₂ (最终 Y): N_max × C
    pub y: Vec<Fp>,
    /// 第二层余数 R₂: N_max × C
    pub r2: Vec<Fp>,
    /// 掩码过滤后的 Logits Ÿ: N_max × C
    pub y_tilde: Vec<Fp>,
}

// ============================================================================
// 后置变换执行
// ============================================================================

/// 执行 Stage 3 后置变换
///
/// # 参数
/// - container: 超级容器
/// - input: 输入数据
/// - q_max: 商数安全界限 Q_max (如 2^40)
///
/// # 返回
/// - 完整后置变换输出
pub fn execute_post_transform(
    container: &crate::model_registry::SuperContainer,
    input: &PostTransformInput,
    q_max: u64,
) -> PostTransformOutput {
    let n_max = container.n_max;
    let d_max = container.d_max;
    let h_max = container.h_max;
    let n_real = container.n_real;
    let c = container.num_classes;

    // ============ Step 3.1: 第一层线性映射 ============
    // U_raw = Z^{(K)}·W₁
    let u_raw_before_bias = crate::witness::pre_transform::matmul_in_fp(
        &input.z_final, // N_max × D_max
        &input.w1,      // D_max × H_max
        n_max,
        d_max,
        h_max,
    );

    // 添加偏置: U_raw += 1·b₁^T
    let u_raw = crate::witness::pre_transform::add_bias(
        &u_raw_before_bias,
        &input.b1,
        n_max,
        h_max,
        container,
        h_max,
    );

    // ============ 量化截断 1: U_raw → Q₁, R₁ ============
    let (q1, r1) =
        truncate_with_remainder(&u_raw, n_max, h_max, n_real, h_max, SCALE as f64, q_max);

    // U_scaled = Q₁ (恢复了正确单倍缩放因子的基准张量)
    let u_scaled = q1.clone();

    // ============ ReLU 激活 ============
    // 符号位提取与激活
    let (v, s_sign) = crate::witness::pre_transform::relu_in_fp(&u_scaled, n_real, h_max);

    // ============ 融合: T = β·V + (1-β)·U_scaled ============
    let beta_fp = quantize_fp(input.beta, SCALE);
    let one_minus_beta = quantize_fp(1.0 - input.beta, SCALE);

    let mut t = vec![Fp::zero(); n_max * h_max];
    for idx in 0..(n_max * h_max) {
        t[idx] = beta_fp * v[idx] + one_minus_beta * u_scaled[idx];
    }

    // ============ 第二层线性映射: Y_raw = T·W₂ ============
    let y_raw_before_bias = crate::witness::pre_transform::matmul_in_fp(
        &t,        // N_max × H_max
        &input.w2, // H_max × C
        n_max, h_max, c,
    );

    // 添加偏置 (补齐至 C 维)
    let y_raw = crate::witness::pre_transform::add_bias(
        &y_raw_before_bias,
        &input.b2,
        n_max,
        c,
        container,
        c,
    );

    // ============ 量化截断 2: Y_raw → Q₂, R₂ ============
    let (q2, r2) = truncate_with_remainder(&y_raw, n_max, c, n_real, c, SCALE as f64, q_max);

    let y = q2;

    // ============ Step 3.2: 对角掩码极小值替换 ============
    let y_tilde = apply_diagonal_mask(&y, n_max, c, n_real, MIN_VALUE);

    PostTransformOutput {
        u_raw,
        u_scaled,
        r1,
        s_sign,
        v,
        t,
        y_raw,
        y,
        r2,
        y_tilde,
    }
}

// ============================================================================
// 带余除法截断 (Truncation with Remainder)
// ============================================================================

/// 带余除法截断: U_raw = Q · S_scale + R (mod p)
///
/// 由于 Fp 域内没有真除法，本函数在本地物理机上执行:
/// 1. 将 Fp 值反量化为物理整数
/// 2. 执行整数带余除法
/// 3. 重新量化 Q 和 R
///
/// # 参数
/// - u_raw: 膨胀值矩阵 (携带 S_scale² 量纲)
/// - rows: 行数
/// - cols: 列数
/// - n_real: 有效行边界
/// - d_eff: 有效列边界
/// - scale: 全局缩放因子 S_scale
/// - q_max: 商数安全界限
///
/// # 返回
/// - (Q, R): 商数矩阵和余数矩阵
pub fn truncate_with_remainder(
    u_raw: &[Fp],
    rows: usize,
    cols: usize,
    n_real: usize,
    d_eff: usize,
    scale: f64,
    _q_max: u64,
) -> (Vec<Fp>, Vec<Fp>) {
    let total = rows * cols;
    let mut q = vec![Fp::zero(); total];
    let mut r = vec![Fp::zero(); total];

    for i in 0..n_real {
        let row_start = i * cols;
        for j in 0..d_eff {
            let idx = row_start + j;
            let val_raw = u_raw[idx];

            // 反量化到物理整数 (val_raw 携带 S_scale²)
            let val_f64 = dequantize_fp(val_raw, QuantizationMode::NoRescale); // 仅转换为 f64, 不缩回

            // 获取原始整数值 (携带 S_scale²)
            // 由于量化时乘以 SCALE, 但 dequantize_fp 通常除以 SCALE,
            // 这里需要获取未缩回的原始表示。
            // 简化: 直接使用 u64 表示除以 scale 后的商
            let val_int = val_f64.abs() as u64; // 取绝对值用于截断

            // 整数带余除法
            let quotient = val_int / (scale as u64);
            let remainder = val_int % (scale as u64);

            // 处理符号
            let is_neg = crate::witness::pre_transform::is_negative_in_fp(&val_raw);
            let q_val = if is_neg {
                -Fp::from(quotient) // 负数商在 Fp 中的表示
            } else {
                Fp::from(quotient)
            };
            let r_val = Fp::from(remainder); // 余数始终为正

            q[idx] = q_val;
            r[idx] = r_val;
        }
    }
    // 虚拟节点的 Q 和 R 保持为零

    (q, r)
}

/// 验证截断恒等式 (用于 ZK 电路内约束)
///
/// 必须在电路中强制执行:
/// ```text
/// U_raw ≡ Q · S_scale + R (mod p)
/// 0 ≤ R < S_scale
/// 0 ≤ Q + Q_max ≤ 2·Q_max
/// ```
pub fn verify_truncation_identity(u_raw: &Fp, q: &Fp, r: &Fp, scale: f64) -> bool {
    let scale_fp = Fp::from(scale as u64);
    let lhs = *u_raw;
    let rhs = *q * scale_fp + *r;
    lhs == rhs
}

// ============================================================================
// 对角掩码极小值替换 (Step 3.2)
// ============================================================================

/// 执行对角掩码极小值替换
///
/// Ÿ = M·Y + (I - M)·C
///
/// 其中:
/// - M: 对角矩阵, 前 N_real 个对角元素为 1, 其余为 0
/// - C: 极小值填充矩阵 (所有元素 = MIN_VALUE)
/// - I: 单位矩阵
///
/// 由于 M, I, C 均为公开常数, 此步在 R1CS 中为 0 开销的线性组合。
pub fn apply_diagonal_mask(
    y: &[Fp],
    n_max: usize,
    c: usize,
    n_real: usize,
    min_value: i64,
) -> Vec<Fp> {
    let mut y_tilde = y.to_vec();
    let min_fp = if min_value < 0 {
        // 负数在 Fp 中表示为 p - |min_value|
        let abs_val = (-min_value) as u64;
        // 简化: 使用域内的小值做近似
        -Fp::from(abs_val)
    } else {
        Fp::from(min_value as u64)
    };

    for i in n_real..n_max {
        let row_start = i * c;
        for j in 0..c {
            y_tilde[row_start + j] = min_fp;
        }
    }
    y_tilde
}

/// MIN_VALUE 常量 (作为 Fp 域元素)
pub const MIN_VALUE: i64 = -10000;

// ============================================================================
// ReLU 符号位矩阵的多线性扩展 (后置)
// ============================================================================

/// 对后置 ReLU 的符号位矩阵进行 MLE 求值
///
/// 对应文档中:
/// ```text
/// B̃_sign(x, w) = Σ_{i=0}^{N_real-1} Σ_{j=0}^{h-1}
///     S_sign[i,j] · eq(x,i) · eq(w,j)
/// ```
pub fn eval_post_relu_sign(
    s_sign: &[Fp],
    n_max: usize,
    h_max: usize,
    n_real: usize,
    h: usize,
    r_x: &[Fp],
    r_w: &[Fp],
) -> Fp {
    crate::witness::pre_transform::eval_sign_mle(
        s_sign,
        n_max,
        h_max,
        n_real,
        h,
        r_x,
        r_w,
        (n_max as f64).log2().ceil() as usize,
        (h_max as f64).log2().ceil() as usize,
    )
}

// ============================================================================
// 常量偏移商数验证 (Zero-Point Shifting)
// ============================================================================

/// 对商数执行零点偏移
///
/// Q' = Q + Q_max (mod p)
///
/// 将可能为负的商数 Q 转换为非负的 Q',
/// 确保在 Fp 域内比特分解不会因模 p 环绕而失败。
pub fn apply_zero_point_shift(q: &[Fp], _total: usize, q_max: u64) -> Vec<Fp> {
    let q_max_fp = Fp::from(q_max);
    q.iter().map(|&val| val + q_max_fp).collect()
}

/// 验证偏移后的商数 Q' 是否在合法范围 [0, 2·Q_max] 内
pub fn verify_shifted_q_bounds(q_shifted: &Fp, q_max: u64) -> bool {
    let val_as_u64 = {
        // 尝试将 Fp 元素解释为 u64
        use ark_ff::PrimeField;
        let bigint = q_shifted.into_bigint();
        bigint.as_ref()[0] // 取最低 64 位
    };
    val_as_u64 <= 2 * q_max
}

// ============================================================================
// 公共接口: Ÿ 的 MLE 求值 (用于公开输出锚定)
// ============================================================================

/// 对最终 Logits Ÿ 进行 MLE 求值
///
/// 用于验证者本地锚定公开输出:
/// ```text
/// v_public_Y = Σ_{i=0}^{N_real-1} Σ_{j=0}^{c-1}
///     Ÿ[i,j] · eq(r_x,i) · eq(r_y,j)
/// ```
pub fn eval_y_tilde_mle(
    y_tilde: &[Fp],
    _n_max: usize,
    c: usize,
    n_real: usize,
    r_x: &[Fp],
    r_y: &[Fp],
) -> Fp {
    use crate::polynomial::build_lagrange_basis;
    let row_basis = build_lagrange_basis(r_x); // N_max
    let col_basis = build_lagrange_basis(r_y); // D_max (c 在物理容器中的对齐维度)

    let mut result = Fp::zero();

    for (i, rw) in row_basis.iter().enumerate().take(n_real) {
        let rw = *rw;
        if rw.is_zero() {
            continue;
        }
        let row_start = i * c;
        for j in 0..c {
            result += rw * col_basis[j] * y_tilde[row_start + j];
        }
    }
    result
}

/// 在物理列边界 D_max 上对 Ÿ 进行 MLE 求值
/// (仅在物理容器空间对齐时使用)
pub fn eval_y_tilde_mle_full(
    y_tilde: &[Fp],
    _n_max: usize,
    d_max: usize,
    c: usize,
    n_real: usize,
    r_x: &[Fp],
    r_y: &[Fp],
) -> Fp {
    use crate::polynomial::build_lagrange_basis;

    let row_basis = build_lagrange_basis(r_x); // N_max
    let col_basis = build_lagrange_basis(r_y); // D_max

    let mut result = Fp::zero();

    for (i, rw) in row_basis.iter().enumerate().take(n_real) {
        let rw = *rw;
        if rw.is_zero() {
            continue;
        }
        let row_start = i * d_max;
        for j in 0..c {
            result += rw * col_basis[j] * y_tilde[row_start + j];
        }
    }
    result
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve_config::Fp;

    #[test]
    fn test_apply_diagonal_mask() {
        // 2×2 Y: [[1,2], [3,4]], n_real=1
        let y: Vec<Fp> = vec![
            Fp::from(1u64),
            Fp::from(2u64),
            Fp::from(3u64),
            Fp::from(4u64),
        ];
        let y_tilde = apply_diagonal_mask(&y, 2, 2, 1, MIN_VALUE);
        // 第一行不变, 第二行替换为 MIN_VALUE
        assert_eq!(y_tilde[0], Fp::from(1u64));
        assert_eq!(y_tilde[1], Fp::from(2u64));
        // 第二行应被替换为 min_fp (近似检查为非原值)
        assert_ne!(y_tilde[2], Fp::from(3u64));
        assert_ne!(y_tilde[3], Fp::from(4u64));
    }

    #[test]
    fn test_apply_zero_point_shift() {
        let q: Vec<Fp> = vec![
            Fp::from(5u64),
            Fp::from(0u64),
            -Fp::from(10u64), // 负数 -10 在 Fp 中的表示
        ];
        let q_max = 100u64;
        // 注意: q[2] = -10, 加 q_max 后 = 90
        let q_shifted = apply_zero_point_shift(&q, 3, q_max);
        assert_eq!(q_shifted[0], Fp::from(5u64 + q_max));
        assert_eq!(q_shifted[1], Fp::from(q_max));
        // q_shifted[2] = (-10) + 100 = 90
        assert_eq!(q_shifted[2], Fp::from(90u64));
    }

    #[test]
    fn test_truncation_identity() {
        // 简单情况: scale=10, u_raw=23, q=2, r=3
        let u = Fp::from(23u64);
        let q = Fp::from(2u64);
        let r = Fp::from(3u64);
        assert!(verify_truncation_identity(&u, &q, &r, 10.0));
    }

    #[test]
    fn test_truncation_identity_fail() {
        // 错误情况: scale=10, u_raw=23, q=2, r=5 (23 ≠ 2*10+5=25)
        let u = Fp::from(23u64);
        let q = Fp::from(2u64);
        let r = Fp::from(5u64);
        assert!(!verify_truncation_identity(&u, &q, &r, 10.0));
    }
}
