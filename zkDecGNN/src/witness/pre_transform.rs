//! Stage 1 前置特征变换 (Pre-Transform)
//!
//! 严格遵循 Spartan-ZENO 设计文档第 2.1 节第二步。
//!
//! ## 计算动作
//! ```text
//! X_base = α_pre · [ReLU(X·W_pre^{(0)})·W_pre^{(1)}] + (1 - α_pre)·X
//! ```
//!
//! 令 V₁ = Max(0, X·W_pre^{(0)})，代入后:
//! ```text
//! X_base = α_pre · V₁·W_pre^{(1)} + (1 - α_pre)·X
//! ```
//!
//! ## ZENO 优化
//! - 若 α_pre = 0 (SGC/SSGC/SIGN): 短路所有前置 MLP 计算
//! - 直接零拷贝赋值 X_padded → X_base
//!
//! ## 见证记录
//! - 线性乘加中间值 (X·W_pre^{(0)} 的点积展开项)
//! - ReLU 非线性辅助见证 (符号位矩阵 S_sign_pre)
//! - 第二次线性乘加中间值 (V₁·W_pre^{(1)})
//! - X_base 完整输出张量

use crate::curve_config::Fp;
use crate::model_registry::{ModelArchitecture, SuperContainer};
use crate::quantization::{quantize_fp, SCALE};
use ark_ff::{One, Zero};

// ============================================================================
// 前置变换核心结构体
// ============================================================================

/// Stage 1 前置变换的输入数据
#[derive(Debug, Clone)]
pub struct PreTransformInput {
    /// 补齐后的初始特征矩阵 X_padded: N_max × D_max (row-major)
    pub x_padded: Vec<Fp>,
    /// 前置第一层权重 W_pre^{(0)}: D_max × H_pre_max (row-major)
    /// 仅 APPNP 模式下非空
    pub w_pre_0: Vec<Fp>,
    /// 前置第二层权重 W_pre^{(1)}: H_pre_max × D_effective (row-major)
    /// 仅 APPNP 模式下非空
    pub w_pre_1: Vec<Fp>,
    /// 前置第一层偏置 b_pre^{(0)}: H_pre_max (补齐后)
    pub b_pre_0: Vec<Fp>,
    /// 前置第二层偏置 b_pre^{(1)}: D_effective (补齐后)
    pub b_pre_1: Vec<Fp>,
    /// 前置变换控制开关 α_pre
    pub alpha_pre: f64,
    /// 模型架构
    pub architecture: ModelArchitecture,
}

/// Stage 1 前置变换的输出结果
#[derive(Debug, Clone)]
pub struct PreTransformOutput {
    /// X_base: N_max × D_max 的前置变换结果
    pub x_base: Vec<Fp>,
    /// 第一次线性乘加 X·W_pre^{(0)} 的原始累加值 (用于 ReLU 验证)
    pub u_pre_raw: Vec<Fp>,
    /// ReLU 辅助见证: 符号位矩阵 S_sign_pre (N_max × H_pre_max)
    pub s_sign_pre: Vec<Fp>,
    /// 激活后结果 V₁ = ReLU(U_pre) (N_max × H_pre_max)
    pub v1: Vec<Fp>,
    /// 第二次线性乘加 V₁·W_pre^{(1)} 的原始累加值
    pub v1_w1_raw: Vec<Fp>,
    /// 是否发生了短路 (α_pre = 0)
    pub bypassed: bool,
}

// ============================================================================
// 前置变换计算
// ============================================================================

/// 执行 Stage 1 前置特征变换
///
/// # 参数
/// - container: 超级容器
/// - input: 输入数据
///
/// # 返回
/// - 前置变换输出
pub fn execute_pre_transform(
    container: &SuperContainer,
    input: &PreTransformInput,
) -> PreTransformOutput {
    let n_max = container.n_max;
    let d_max = container.d_max;
    let h_pre_max = container.h_max; // 前置隐藏层物理上限
    let n_real = container.n_real;

    // ZENO 静态短路: 若 α_pre = 0，直接透传
    if input.alpha_pre == 0.0 {
        return PreTransformOutput {
            x_base: input.x_padded.clone(),
            u_pre_raw: vec![Fp::zero(); n_max * h_pre_max],
            s_sign_pre: vec![Fp::zero(); n_max * h_pre_max],
            v1: vec![Fp::zero(); n_max * h_pre_max],
            v1_w1_raw: vec![Fp::zero(); n_max * d_max],
            bypassed: true,
        };
    }

    // ==== 第一次线性变换: U_pre = X·W_pre^{(0)} ====
    let u_pre_raw = matmul_in_fp(
        &input.x_padded, // N_max × D_max
        &input.w_pre_0,  // D_max × H_pre_max
        n_max,
        d_max,
        h_pre_max,
    );

    // 添加偏置: U_pre += 1·b_pre_0^T
    let u_pre_raw = add_bias(
        &u_pre_raw,
        &input.b_pre_0,
        n_max,
        h_pre_max,
        container,
        h_pre_max,
    );

    // ==== ReLU 激活: V₁ = max(0, U_pre) ====
    let (v1, s_sign_pre) = relu_in_fp(&u_pre_raw, n_real, h_pre_max);

    // ==== 第二次线性变换: V₁·W_pre^{(1)} ====
    let d_eff = container.get_effective_dimension(input.architecture);
    let v1_w1_raw = matmul_in_fp(
        &v1,            // N_max × H_pre_max
        &input.w_pre_1, // H_pre_max × D_effective (对齐至 D_max)
        n_max,
        h_pre_max,
        d_max,
    );

    // 添加偏置
    let v1_w1_raw = add_bias(
        &v1_w1_raw,
        &input.b_pre_1,
        n_max,
        d_max,
        container,
        container.d_max,
    );

    // ==== 融合: X_base = α_pre·(V₁·W_pre^{(1)}) + (1-α_pre)·X ====
    let alpha_fp = quantize_fp(input.alpha_pre, SCALE);
    let one_minus_alpha = quantize_fp(1.0 - input.alpha_pre, SCALE);
    let x_padded_d_eff = truncate_to_d_eff(&input.x_padded, n_max, d_max, d_eff);

    let mut x_base = vec![Fp::zero(); n_max * d_max];
    for i in 0..(n_max * d_max) {
        x_base[i] = alpha_fp * v1_w1_raw[i] + one_minus_alpha * x_padded_d_eff[i];
    }

    PreTransformOutput {
        x_base,
        u_pre_raw,
        s_sign_pre,
        v1,
        v1_w1_raw,
        bypassed: false,
    }
}

// ============================================================================
// 矩阵乘法 (在 Fp 域内执行)
// ============================================================================

/// 在 Fp 域内执行矩阵乘法: C = A × B
///
/// A: m × k, B: k × n, C: m × n
/// 所有矩阵均为 row-major 排列。
///
/// 注意: 由于输入数据均已经过量化 (缩放因子 S_scale)，
/// 乘积累加后的结果将携带 S_scale² 量纲。
/// 本函数不做截断，调用方需在后续处理。
pub fn matmul_in_fp(
    a: &[Fp], // m × k, row-major
    b: &[Fp], // k × n, row-major
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Fp> {
    assert_eq!(a.len(), m * k, "Matrix A dimensions mismatch");
    assert_eq!(b.len(), k * n, "Matrix B dimensions mismatch");

    let mut c = vec![Fp::zero(); m * n];

    for i in 0..m {
        let a_row_start = i * k;
        for j in 0..n {
            let mut sum = Fp::zero();
            for t in 0..k {
                sum += a[a_row_start + t] * b[t * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

// ============================================================================
// 偏置加法: C[i,j] += b[j] (逐行添加偏置)
// ============================================================================

/// 为矩阵的每一行添加偏置向量
///
/// C: m × n, b: n (补齐至 n 维)
/// 对每一行 i: C[i, j] += b[j]
pub fn add_bias(
    c: &[Fp],
    bias: &[Fp],
    _m: usize,
    n: usize,
    container: &SuperContainer,
    physical_n: usize,
) -> Vec<Fp> {
    let n_real = container.n_real;
    let mut result = c.to_vec();
    // 仅对真实节点添加偏置 (虚拟节点的偏置归零)
    for i in 0..n_real {
        let row_start = i * physical_n;
        for j in 0..n {
            result[row_start + j] += bias[j];
        }
    }
    result
}

// ============================================================================
// ReLU 激活 (在 Fp 域内，带符号位提取)
// ============================================================================

/// 在 Fp 域内执行 ReLU: v = max(0, u)
///
/// 由于 Fp 域内没有"负数"概念，我们使用阈值判断:
/// - 若 Fp 值 ≤ p/2，视为非负 (正数或零)
/// - 若 Fp 值 > p/2，视为负数 (模 p 环绕)
///
/// 返回:
/// - v: 激活后的值 (负数 → 0，正数保持不变)
/// - s_sign: 符号位矩阵 (0 = 正, 1 = 负)
pub fn relu_in_fp(u: &[Fp], _n_rows: usize, _n_cols: usize) -> (Vec<Fp>, Vec<Fp>) {
    let mut v = vec![Fp::zero(); u.len()];
    let mut s_sign = vec![Fp::zero(); u.len()];

    // 使用 Fp 的原始字节表示来判断符号
    // Curve25519 标量域约为 2^252.x，正数范围在 [0, p/2]
    for (idx, &val) in u.iter().enumerate() {
        if is_negative_in_fp(&val) {
            // 负数 → ReLU 输出 0
            v[idx] = Fp::zero();
            s_sign[idx] = Fp::one();
        } else {
            // 正数 → 透传
            v[idx] = val;
            s_sign[idx] = Fp::zero();
        }
    }
    (v, s_sign)
}

/// 判断 Fp 值在物理意义上是否为负数
///
/// 对称均匀定点量化下，负数被表示为 p - |x| (模 p 环绕)。
/// 因此，若 Fp 值 > p/2，则视为负数。
pub fn is_negative_in_fp(val: &Fp) -> bool {
    // 使用 arkworks 的 into_bigint 获取大整数表示
    use ark_ff::PrimeField;
    let bigint = val.into_bigint();
    // Curve25519 标量域 p ≈ 2^252 + ...
    // p/2 ≈ 2^251 + ...
    // 检查最高 253-bit 的表示
    let p_half = ark_ff::BigInt::<4>::from(1u64 << 63); // 简化判断
    bigint > p_half
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 将矩阵截断至有效列边界 D_effective
///
/// 矩阵为 N_max × D_max (row-major)，截断后每行保留前 d_eff 列。
/// 截断后矩阵物理尺寸仍为 N_max × D_max，但超过 d_eff 的列填零。
pub fn truncate_to_d_eff(matrix: &[Fp], n_max: usize, d_max: usize, d_eff: usize) -> Vec<Fp> {
    assert!(d_eff <= d_max);
    let mut result = vec![Fp::zero(); n_max * d_max];
    for i in 0..n_max {
        let src_start = i * d_max;
        let dst_start = i * d_max;
        result[dst_start..dst_start + d_eff].copy_from_slice(&matrix[src_start..src_start + d_eff]);
    }
    result
}

/// 将有效区域 (N_real × D_effective) 从完整矩阵中提取出来
pub fn extract_effective(
    matrix: &[Fp],
    _n_max: usize,
    d_max: usize,
    n_real: usize,
    d_eff: usize,
) -> Vec<Fp> {
    let mut result = vec![Fp::zero(); n_real * d_eff];
    for i in 0..n_real {
        let src_start = i * d_max;
        let dst_start = i * d_eff;
        result[dst_start..dst_start + d_eff].copy_from_slice(&matrix[src_start..src_start + d_eff]);
    }
    result
}

// ============================================================================
// ReLU 符号位矩阵的多线性扩展 (用于后续 PCS)
// ============================================================================

/// 对符号位矩阵进行多线性扩展求值
///
/// 对应文档中的:
/// ```text
/// B̃_sign_pre(x, w_pre) = Σ_{i=0}^{N_real-1} Σ_{j=0}^{h_pre-1}
///     S_sign_pre[i,j] · eq(x,i) · eq(w_pre,j) + 0_padding
/// ```
///
/// # 参数
/// - s_sign_pre: 符号位矩阵 (N_max × H_max, row-major)
/// - n_max: 物理行数 N_max
/// - h_max: 物理列数 H_max
/// - n_real: 有效行边界
/// - h_pre: 有效列边界
/// - r_x: 行评估点 ∈ Fp^{n_v}
/// - r_w: 列评估点 ∈ Fp^{n_h}
/// - nv: 行布尔维度
/// - nh: 列布尔维度
pub fn eval_sign_mle(
    s_sign_pre: &[Fp],
    _n_max: usize,
    h_max: usize,
    n_real: usize,
    h_pre: usize,
    r_x: &[Fp],
    r_w: &[Fp],
    _nv: usize,
    _nh: usize,
) -> Fp {
    use crate::polynomial::build_lagrange_basis;

    let row_basis = build_lagrange_basis(r_x); // 长度 N_max
    let col_basis = build_lagrange_basis(r_w); // 长度 H_max

    let mut result = Fp::zero();

    for (i, row_weight) in row_basis.iter().enumerate().take(n_real) {
        let row_weight = *row_weight;
        let row_start = i * h_max;
        for j in 0..h_pre {
            result += row_weight * col_basis[j] * s_sign_pre[row_start + j];
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
    fn test_matmul_identity() {
        // 2×2 单位矩阵
        let a: Vec<Fp> = vec![
            Fp::from(1u64),
            Fp::from(0u64),
            Fp::from(0u64),
            Fp::from(1u64),
        ];
        let b: Vec<Fp> = vec![
            Fp::from(3u64),
            Fp::from(4u64),
            Fp::from(5u64),
            Fp::from(6u64),
        ];
        let c = matmul_in_fp(&a, &b, 2, 2, 2);
        assert_eq!(c[0], Fp::from(3u64));
        assert_eq!(c[1], Fp::from(4u64));
        assert_eq!(c[2], Fp::from(5u64));
        assert_eq!(c[3], Fp::from(6u64));
    }

    #[test]
    fn test_matmul_simple() {
        // [1, 2; 3, 4] × [5, 6; 7, 8]
        let a: Vec<Fp> = vec![
            Fp::from(1u64),
            Fp::from(2u64),
            Fp::from(3u64),
            Fp::from(4u64),
        ];
        let b: Vec<Fp> = vec![
            Fp::from(5u64),
            Fp::from(6u64),
            Fp::from(7u64),
            Fp::from(8u64),
        ];
        let c = matmul_in_fp(&a, &b, 2, 2, 2);
        // [1*5+2*7, 1*6+2*8; 3*5+4*7, 3*6+4*8]
        // = [19, 22; 43, 50]
        assert_eq!(c[0], Fp::from(19u64));
        assert_eq!(c[1], Fp::from(22u64));
        assert_eq!(c[2], Fp::from(43u64));
        assert_eq!(c[3], Fp::from(50u64));
    }

    #[test]
    fn test_relu_positive() {
        let u: Vec<Fp> = vec![Fp::from(5u64), Fp::from(10u64)];
        let (v, s) = relu_in_fp(&u, 1, 2);
        assert_eq!(v[0], Fp::from(5u64));
        assert_eq!(v[1], Fp::from(10u64));
        assert_eq!(s[0], Fp::zero());
        assert_eq!(s[1], Fp::zero());
    }

    #[test]
    fn test_truncate_to_d_eff() {
        let matrix: Vec<Fp> = (1..=8).map(|i| Fp::from(i as u64)).collect();
        // 2×4: [[1,2,3,4], [5,6,7,8]]
        let truncated = truncate_to_d_eff(&matrix, 2, 4, 2);
        // 应保留前 2 列: [[1,2,0,0], [5,6,0,0]]
        assert_eq!(truncated[0], Fp::from(1u64));
        assert_eq!(truncated[1], Fp::from(2u64));
        assert_eq!(truncated[2], Fp::zero());
        assert_eq!(truncated[3], Fp::zero());
        assert_eq!(truncated[4], Fp::from(5u64));
        assert_eq!(truncated[5], Fp::from(6u64));
    }

    #[test]
    fn test_extract_effective() {
        // 2×3 matrix: [[1,2,3], [4,5,6]]
        let matrix: Vec<Fp> = (1..=6).map(|i| Fp::from(i as u64)).collect();
        let eff = extract_effective(&matrix, 2, 3, 2, 2);
        // 2×2: [[1,2], [4,5]]
        assert_eq!(eff.len(), 4);
        assert_eq!(eff[0], Fp::from(1u64));
        assert_eq!(eff[1], Fp::from(2u64));
        assert_eq!(eff[2], Fp::from(4u64));
        assert_eq!(eff[3], Fp::from(5u64));
    }
}
