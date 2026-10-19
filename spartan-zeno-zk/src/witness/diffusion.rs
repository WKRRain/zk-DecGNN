//! Stage 2 扩散与聚合追踪 (Diffusion & Aggregation Tracking)
//!
//! 严格遵循 Spartan-ZENO 设计文档第 2.1 节第三步。
//!
//! ## 核心循环 (k = 1 → K)
//!
//! ### Stage 2.1: 扩散更新
//! ```text
//! S^{(k)} = λ₁·(Â·S^{(k-1)}) + λ₂·X_base
//! ```
//!
//! ### Stage 2.2: 聚合更新
//! ```text
//! Z^{(k)} = γ₁·Z^{(k-1)} ⊕ (γ₂·S^{(k)} + γ₃·X_base)
//! ```
//!
//! 其中 ⊕ 由客户端请求的 VK 决定:
//! - VK_SGC / VK_SSGC / VK_APPNP: ⊕ = 加法 (逐元素求和)
//! - VK_SIGN: ⊕ = 拼接 (写入超级容器指定偏移列)
//!
//! ## ZENO 剪枝
//! 循环完成后，利用静态掩码 M 抹除所有超过 N_real 的无效数据。

use crate::curve_config::Fp;
use crate::model_registry::ModelArchitecture;
use crate::quantization::quantize_fp;
use crate::quantization::SCALE;
use ark_ff::{One, Zero};

// ============================================================================
// 核心结构体
// ============================================================================

/// 扩散与聚合循环的输入
#[derive(Debug, Clone)]
pub struct DiffusionInput {
    /// 前置变换结果 X_base: N_max × D_max (row-major)
    pub x_base: Vec<Fp>,
    /// 归一化邻接矩阵 Â: N_max × N_max (row-major), CSR 或稠密
    pub a_hat: Vec<Fp>,
    /// 扩散系数 λ₁ (APPNP 中的 teleport 概率)
    pub lambda_1: f64,
    /// 扩散系数 λ₂
    pub lambda_2: f64,
    /// 聚合系数 γ₁
    pub gamma_1: f64,
    /// 聚合系数 γ₂
    pub gamma_2: f64,
    /// 聚合系数 γ₃
    pub gamma_3: f64,
    /// 总迭代轮数 K
    pub k: usize,
    /// 模型架构 (决定 ⊕ 的物理连线)
    pub architecture: ModelArchitecture,
}

/// 扩散与聚合循环的输出
#[derive(Debug, Clone)]
pub struct DiffusionOutput {
    /// 每轮扩散状态 S^{(1)}, S^{(2)}, ..., S^{(K)}
    /// 每个为 N_max × D_max (row-major)
    pub s_states: Vec<Vec<Fp>>,
    /// 每轮聚合状态 Z^{(1)}, Z^{(2)}, ..., Z^{(K)}
    /// 每个为 N_max × D_max (row-major)
    pub z_states: Vec<Vec<Fp>>,
    /// 最终聚合结果 Z^{(K)}
    pub z_final: Vec<Fp>,
    /// Â·S^{(k-1)} 的中间值 (每轮)
    pub a_s_product: Vec<Vec<Fp>>,
}

// ============================================================================
// 扩散与聚合循环执行
// ============================================================================

/// 执行 Stage 2 的 K 轮扩散与聚合循环
///
/// # 参数
/// - container: 超级容器
/// - input: 输入数据
///
/// # 返回
/// - 循环输出
pub fn execute_diffusion_loop(
    container: &crate::model_registry::SuperContainer,
    input: &DiffusionInput,
) -> DiffusionOutput {
    let n_max = container.n_max;
    let d_max = container.d_max;
    let _n_real = container.n_real;
    let k = input.k;

    // 初始化
    let mut s_states: Vec<Vec<Fp>> = Vec::with_capacity(k);
    let mut z_states: Vec<Vec<Fp>> = Vec::with_capacity(k);
    let mut a_s_products: Vec<Vec<Fp>> = Vec::with_capacity(k);

    // S^{(0)} = X_base, Z^{(0)} = X_base
    let mut s_prev = input.x_base.clone();
    let mut z_prev = input.x_base.clone();

    // 量化系数
    let lambda_1_fp = quantize_fp(input.lambda_1, SCALE);
    let lambda_2_fp = quantize_fp(input.lambda_2, SCALE);
    let gamma_1_fp = quantize_fp(input.gamma_1, SCALE);
    let gamma_2_fp = quantize_fp(input.gamma_2, SCALE);
    let gamma_3_fp = quantize_fp(input.gamma_3, SCALE);

    for _round in 0..k {
        // ============ Stage 2.1: 扩散更新 ============
        // V₂ = Â·S^{(k-1)}
        let v2 = sparse_matmul(&input.a_hat, &s_prev, n_max, n_max, d_max);
        a_s_products.push(v2.clone());

        // S^{(k)} = λ₁·V₂ + λ₂·X_base
        let mut s_current = vec![Fp::zero(); n_max * d_max];
        for idx in 0..(n_max * d_max) {
            s_current[idx] = lambda_1_fp * v2[idx] + lambda_2_fp * input.x_base[idx];
        }
        s_states.push(s_current.clone());

        // ============ Stage 2.2: 聚合更新 ============
        // term = γ₂·S^{(k)} + γ₃·X_base
        let mut term = vec![Fp::zero(); n_max * d_max];
        for idx in 0..(n_max * d_max) {
            term[idx] = gamma_2_fp * s_current[idx] + gamma_3_fp * input.x_base[idx];
        }

        // γ₁·Z^{(k-1)}
        let mut z_prev_scaled = vec![Fp::zero(); n_max * d_max];
        for idx in 0..(n_max * d_max) {
            z_prev_scaled[idx] = gamma_1_fp * z_prev[idx];
        }

        // Z^{(k)} = γ₁·Z^{(k-1)} ⊕ term
        let z_current = apply_aggregation_operator(
            &z_prev_scaled,
            &term,
            n_max,
            d_max,
            input.architecture,
            container,
        );
        z_states.push(z_current.clone());
        z_prev = z_current;

        // 更新 s_prev 为下一轮准备
        s_prev = s_current;
    }

    // 提取最终 Z^{(K)}
    let z_final = z_states
        .last()
        .cloned()
        .unwrap_or_else(|| input.x_base.clone());

    DiffusionOutput {
        s_states,
        z_states,
        z_final,
        a_s_product: a_s_products,
    }
}

// ============================================================================
// 聚合算子 ⊕ 的实现
// ============================================================================

/// 应用聚合算子 ⊕
///
/// - SGC / SSGC / APPNP: 加法 Z = a + b
/// - SIGN: 拼接 (b 写入 Z 的指定偏移列)
fn apply_aggregation_operator(
    a: &[Fp], // γ₁·Z^{(k-1)}, N_max × D_max
    b: &[Fp], // γ₂·S^{(k)} + γ₃·X_base, N_max × D_max
    n_max: usize,
    d_max: usize,
    architecture: ModelArchitecture,
    container: &crate::model_registry::SuperContainer,
) -> Vec<Fp> {
    match architecture {
        ModelArchitecture::SIGN => {
            // SIGN 拼接: a 占据前半部分, b 的数据写入后半部分
            // 超级容器 D_max 为 2d, 前半为前 d 列, 后半为后 d 列
            sign_concatenate(a, b, n_max, d_max, container)
        }
        _ => {
            // SGC / SSGC / APPNP: 逐元素加法
            elementwise_add(a, b, n_max * d_max)
        }
    }
}

/// SIGN 模型的拼接操作
///
/// Z[i, 0..d-1] = a[i, 0..d-1]
/// Z[i, d..2d-1] = b[i, 0..d-1]
fn sign_concatenate(
    a: &[Fp],
    b: &[Fp],
    n_max: usize,
    d_max: usize,
    container: &crate::model_registry::SuperContainer,
) -> Vec<Fp> {
    let d = container.d_real;
    let mut z = vec![Fp::zero(); n_max * d_max];
    for i in 0..n_max {
        let row_start = i * d_max;
        // 前半部分: 保留 a 的前 d 列
        z[row_start..row_start + d].copy_from_slice(&a[row_start..row_start + d]);
        // 后半部分: 写入 b 的前 d 列
        for j in 0..d {
            z[row_start + d + j] = b[row_start + j];
        }
    }
    z
}

/// 逐元素加法
fn elementwise_add(a: &[Fp], b: &[Fp], len: usize) -> Vec<Fp> {
    let mut result = vec![Fp::zero(); len];
    for idx in 0..len {
        result[idx] = a[idx] + b[idx];
    }
    result
}

// ============================================================================
// 稀疏矩阵乘法: C = A × B (A 稀疏, B 稠密)
// ============================================================================

/// 稀疏矩阵乘法 (CSR 格式或稠密退化)
///
/// Â: N_max × N_max (row-major, 但仅需非零元素)
/// B: N_max × D_max (row-major)
/// 结果 C: N_max × D_max (row-major)
///
/// 注意: 如果 Â 以稠密形式存储，使用稠密 matmul；
/// 若为 CSR，使用稀疏加速路径。
/// 当前实现使用稠密 matmul (后续可优化为 CSR)。
pub fn sparse_matmul(
    a_hat: &[Fp], // N_max × N_max
    b: &[Fp],     // N_max × D_max
    m: usize,     // N_max
    k: usize,     // N_max
    n: usize,     // D_max
) -> Vec<Fp> {
    use crate::witness::pre_transform::matmul_in_fp;
    matmul_in_fp(a_hat, b, m, k, n)
}

// ============================================================================
// ZENO 剪枝: 掩码抹除 Padding 区
// ============================================================================

/// 利用静态掩码 M 对每轮状态执行 ZENO 剪枝
///
/// 对于 i >= N_real 的行，全部元素置零。
/// 对于 i < N_real 但 j >= D_effective 的列，置零。
///
/// # 参数
/// - state: 状态矩阵 (N_max × D_max, row-major)
/// - n_max: 物理行数
/// - d_max: 物理列数
/// - n_real: 有效行边界
/// - d_eff: 有效列边界
pub fn zeno_prune_state(
    state: &[Fp],
    n_max: usize,
    d_max: usize,
    n_real: usize,
    d_eff: usize,
) -> Vec<Fp> {
    let mut pruned = state.to_vec();
    for i in 0..n_max {
        let row_start = i * d_max;
        if i >= n_real {
            // 整个虚拟节点行归零
            for j in 0..d_max {
                pruned[row_start + j] = Fp::zero();
            }
        } else {
            // 有效节点但超过 d_eff 的列归零
            for j in d_eff..d_max {
                pruned[row_start + j] = Fp::zero();
            }
        }
    }
    pruned
}

/// 对所有扩散与聚合状态执行剪枝
pub fn zeno_prune_all_states(
    outputs: &DiffusionOutput,
    n_max: usize,
    d_max: usize,
    n_real: usize,
    d_eff: usize,
) -> (Vec<Vec<Fp>>, Vec<Vec<Fp>>) {
    let pruned_s: Vec<Vec<Fp>> = outputs
        .s_states
        .iter()
        .map(|s| zeno_prune_state(s, n_max, d_max, n_real, d_eff))
        .collect();
    let pruned_z: Vec<Vec<Fp>> = outputs
        .z_states
        .iter()
        .map(|z| zeno_prune_state(z, n_max, d_max, n_real, d_eff))
        .collect();
    (pruned_s, pruned_z)
}

// ============================================================================
// SPARK 内存轨迹提取 (从扩散循环的稀疏访存中)
// ============================================================================

/// 为 SPARK 协议准备内存读取轨迹
///
/// 每轮扩散中，Â·S^{(k-1)} 涉及对 S^{(k-1)} 的稀疏读取。
/// 本函数提取 2 × |E| 次内存查询的操作轨迹。
///
/// # 返回
/// - e_row: 读取的行操作轨迹 (N_max → 查询行索引)
/// - e_col: 读取的列操作轨迹 (N_max → 查询列索引)
pub fn extract_spark_trajectory(a_hat: &[Fp], n_max: usize, n_real: usize) -> (Vec<Fp>, Vec<Fp>) {
    let mut e_row = Vec::new();
    let mut e_col = Vec::new();

    // 遍历邻接矩阵的非零元素
    for i in 0..n_real {
        let row_start = i * n_max;
        for j in 0..n_real {
            let val = a_hat[row_start + j];
            if !val.is_zero() {
                e_row.push(Fp::from(i as u64));
                e_col.push(Fp::from(j as u64));
            }
        }
    }

    // SPARK 读写轨迹按 2|E| 规模建模，再补齐至 2 的幂。
    let target_len = next_power_of_two((2 * e_row.len()).max(1));
    while e_row.len() < target_len {
        e_row.push(Fp::zero());
        e_col.push(Fp::zero());
    }

    (e_row, e_col)
}

fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

// ============================================================================
// 多轮状态的全局多线性扩展求值
// ============================================================================

/// 对 K 轮扩散状态进行全局 MLE 求值
///
/// 对应文档中:
/// ```text
/// S̃_global(z, x, y) = Σ_{k=1}^{K} Σ_{i=0}^{N_real-1} Σ_{j=0}^{d-1}
///     S^{(k)}[i,j] · eq(z,bin(k)) · eq(x,i) · eq(y,j)
/// ```
pub fn eval_s_global(
    s_states: &[Vec<Fp>],
    _n_max: usize,
    d_max: usize,
    n_real: usize,
    d_eff: usize,
    r_z: &[Fp],
    r_x: &[Fp],
    r_y: &[Fp],
) -> Fp {
    use crate::polynomial::build_lagrange_basis;

    let row_basis = build_lagrange_basis(r_x); // N_max
    let col_basis = build_lagrange_basis(r_y); // D_max

    let mut result = Fp::zero();

    for (k, s_k) in s_states.iter().enumerate() {
        // 计算 eq(z, bin(k))
        let z_weight = eval_eq(r_z, k);
        if z_weight.is_zero() {
            continue;
        }

        for (i, rw) in row_basis.iter().enumerate().take(n_real) {
            let rw = *rw;
            if rw.is_zero() {
                continue;
            }
            let row_start = i * d_max;
            for j in 0..d_eff {
                result += z_weight * rw * col_basis[j] * s_k[row_start + j];
            }
        }
    }
    result
}

/// 同理对聚合状态 Z 进行全局 MLE 求值
pub fn eval_z_global(
    z_states: &[Vec<Fp>],
    _n_max: usize,
    d_max: usize,
    n_real: usize,
    d_eff: usize,
    r_z: &[Fp],
    r_x: &[Fp],
    r_y: &[Fp],
) -> Fp {
    use crate::polynomial::build_lagrange_basis;

    let row_basis = build_lagrange_basis(r_x);
    let col_basis = build_lagrange_basis(r_y);

    let mut result = Fp::zero();

    for (k, z_k) in z_states.iter().enumerate() {
        let z_weight = eval_eq(r_z, k);
        if z_weight.is_zero() {
            continue;
        }

        for (i, rw) in row_basis.iter().enumerate().take(n_real) {
            let rw = *rw;
            if rw.is_zero() {
                continue;
            }
            let row_start = i * d_max;
            for j in 0..d_eff {
                result += z_weight * rw * col_basis[j] * z_k[row_start + j];
            }
        }
    }
    result
}

/// 计算拉格朗日基函数 eq(x, e)
///
/// eq(x, e) = Π_{m=1}^{|x|} (x_m · e_m + (1 - x_m)(1 - e_m))
pub fn eval_eq(r: &[Fp], e: usize) -> Fp {
    let mut result = Fp::one();
    for (m, &r_m) in r.iter().enumerate() {
        let e_m = ((e >> m) & 1) as u64;
        let e_fp = Fp::from(e_m);
        result *= r_m * e_fp + (Fp::one() - r_m) * (Fp::one() - e_fp);
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
    fn test_elementwise_add() {
        let a: Vec<Fp> = vec![Fp::from(1u64), Fp::from(2u64)];
        let b: Vec<Fp> = vec![Fp::from(3u64), Fp::from(4u64)];
        let c = elementwise_add(&a, &b, 2);
        assert_eq!(c[0], Fp::from(4u64));
        assert_eq!(c[1], Fp::from(6u64));
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(8), 8);
        assert_eq!(next_power_of_two(0), 1);
    }

    #[test]
    fn test_zeno_prune_state() {
        // 2×3 state: [1,2,3; 4,5,6]
        let state: Vec<Fp> = (1..=6).map(|i| Fp::from(i as u64)).collect();
        // n_real=1, d_eff=2 → 第一行保留前2列, 第二行全零
        let pruned = zeno_prune_state(&state, 2, 3, 1, 2);
        assert_eq!(pruned[0], Fp::from(1u64));
        assert_eq!(pruned[1], Fp::from(2u64));
        assert_eq!(pruned[2], Fp::zero()); // j=2 >= d_eff → 0
        assert_eq!(pruned[3], Fp::zero()); // i=1 >= n_real → 0
        assert_eq!(pruned[4], Fp::zero());
        assert_eq!(pruned[5], Fp::zero());
    }

    #[test]
    fn test_eval_eq_identity() {
        // r = (0, 0, 0), e = 0
        let r: Vec<Fp> = vec![Fp::zero(); 3];
        let v = eval_eq(&r, 0);
        assert_eq!(v, Fp::one());
    }

    #[test]
    fn test_eval_eq_mismatch() {
        // r = (1, 1, 1), e = 0
        let r: Vec<Fp> = vec![Fp::one(); 3];
        let v = eval_eq(&r, 0);
        assert_eq!(v, Fp::zero());
    }

    #[test]
    fn test_extract_spark_trajectory() {
        // 2×2 邻接矩阵: [[0, 1], [1, 0]]
        let a_hat: Vec<Fp> = vec![
            Fp::from(0u64),
            Fp::from(1u64),
            Fp::from(1u64),
            Fp::from(0u64),
        ];
        let (e_row, e_col) = extract_spark_trajectory(&a_hat, 2, 2);
        // 有两个非零元素: (0,1) 和 (1,0)
        // 补齐至 2^2 = 4
        assert_eq!(e_row.len(), 4);
        assert_eq!(e_col.len(), 4);
        assert_eq!(e_row[0], Fp::from(0u64));
        assert_eq!(e_col[0], Fp::from(1u64));
        assert_eq!(e_row[1], Fp::from(1u64));
        assert_eq!(e_col[1], Fp::from(0u64));
        // padding 为 0
        assert_eq!(e_row[2], Fp::zero());
        assert_eq!(e_col[2], Fp::zero());
    }
}
