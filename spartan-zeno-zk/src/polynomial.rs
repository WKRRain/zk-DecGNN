//! 多线性扩展 (MLE) 与拉格朗日基函数（三维多线性扩展工具）
//!
//!
//! ## 核心功能
//! - `eq(x, e)`: 拉格朗日基函数，当 x=e 时等于 1，否则等于 0
//! - 张量积 (Kronecker Product) 动态规划生成完整拉格朗日基向量
//! - 布尔超立方体上的多线性多项式求值



use crate::curve_config::Fp;
use ark_ff::{One, Zero};

/// 通用一维多线性扩展多项式容器。
#[derive(Clone, Debug)]
pub struct MLEPolynomial {
    num_vars: usize,
    evaluations: Vec<Fp>,
}

impl MLEPolynomial {
    pub fn from_evaluations(num_vars: usize, evaluations: &[Fp]) -> Self {
        let physical_len = 1usize << num_vars;
        let mut padded = vec![Fp::zero(); physical_len];
        let copy_len = evaluations.len().min(physical_len);
        padded[..copy_len].copy_from_slice(&evaluations[..copy_len]);
        Self {
            num_vars,
            evaluations: padded,
        }
    }

    pub fn evaluate(&self, point: &[Fp]) -> Fp {
        assert_eq!(point.len(), self.num_vars);
        let basis = build_lagrange_basis(point);
        self.evaluations
            .iter()
            .zip(basis.iter())
            .map(|(v, b)| *v * *b)
            .sum()
    }

    pub fn evaluations(&self) -> &[Fp] {
        &self.evaluations
    }

    pub fn evaluate_direct<const D: usize>(
        evaluations: &[Fp],
        num_vars: &[usize; D],
        points: &[&[Fp]],
        effective_bounds: &[usize; D],
    ) -> Fp {
        assert_eq!(points.len(), D);

        let physical_dims: Vec<usize> = num_vars.iter().map(|v| 1usize << v).collect();
        let bases: Vec<Vec<Fp>> = points
            .iter()
            .zip(num_vars.iter())
            .map(|(point, vars)| {
                assert_eq!(point.len(), *vars);
                build_lagrange_basis(point)
            })
            .collect();

        let mut strides = vec![1usize; D];
        for d in (0..D).rev() {
            strides[d] = if d + 1 == D {
                1
            } else {
                strides[d + 1] * physical_dims[d + 1]
            };
        }

        let mut idx = vec![0usize; D];
        let mut result = Fp::zero();
        loop {
            let mut flat_idx = 0usize;
            let mut weight = Fp::one();
            for d in 0..D {
                flat_idx += idx[d] * strides[d];
                weight *= bases[d][idx[d]];
            }
            if flat_idx < evaluations.len() {
                result += evaluations[flat_idx] * weight;
            }

            let mut carry_dim = D;
            while carry_dim > 0 {
                carry_dim -= 1;
                idx[carry_dim] += 1;
                if idx[carry_dim] < effective_bounds[carry_dim] {
                    break;
                }
                idx[carry_dim] = 0;
                if carry_dim == 0 {
                    return result;
                }
            }
        }
    }
}

// ============================================================================
// 拉格朗日基函数 eq(x, e)
// ============================================================================

/// 计算拉格朗日基函数 eq(x, e)
///
/// 定义:
/// ```text
/// eq(x, e) = ∏_{m=1}^{|x|} (x_m · e_m + (1 - x_m)(1 - e_m))
/// ```
///
/// 其中 x_m 和 e_m 分别是向量 x 和 e 的第 m 个分量（标量域元素）。
///
/// # 性质
/// - 当 x = e（逐分量相等）时，eq = 1
/// - 否则 eq = 0
///
/// # 参数
/// - x: 连续变量向量 (Fp 标量)
/// - e_bits: 布尔向量 (true=1, false=0)，表示 e 的二进制展开
///
/// # 位运算优化
/// 使用代数恒等式避免分支:
///   e_m ∈ {0, 1}
///   term = x_m * e_m + (1 - x_m)(1 - e_m)
///        = x_m * e_m + 1 - x_m - e_m + x_m * e_m
///        = 1 - x_m - e_m + 2 * x_m * e_m
/// 由于 e_m ∈ {0, 1}:
///   term = 1 - x_m - e_m + 2·x_m·e_m
pub fn eq_scalar(x: &[Fp], e_bits: &[bool]) -> Fp {
    assert_eq!(x.len(), e_bits.len(), "x and e_bits must have same length");

    let mut result = Fp::one();
    for (x_m, &e_m_bit) in x.iter().zip(e_bits.iter()) {
        // 使用代数展开避免分支: e_m 作为 Fp(0) 或 Fp(1)
        // term = 1 - x_m - e_m + 2·x_m·e_m
        let e_m = if e_m_bit { Fp::one() } else { Fp::zero() };
        let one = Fp::one();
        let two = Fp::from(2u64);
        let term = one - *x_m - e_m + two * (*x_m) * e_m;
        result *= term;
    }
    result
}

/// 给定整数索引 i，生成其二进制展开的布尔向量
///
/// # 位序约定 (LSB-first, 与张量积动态规划一致)
/// bit 0 = LSB (最低位), bit (num_bits-1) = MSB (最高位)
/// 即: bits[m] = 第 m 位 (0-indexed, LSB first)
///
/// 索引 i 的二进制展开: i = Σ_{m=0}^{num_bits-1} bit_m(i) · 2^m
///
/// # 参数
/// - i: 整数索引
/// - num_bits: 二进制向量的长度
pub fn int_to_bits(i: usize, num_bits: usize) -> Vec<bool> {
    let mut bits = Vec::with_capacity(num_bits);
    for b in 0..num_bits {
        bits.push(((i >> b) & 1) == 1);
    }
    bits
}

/// 从布尔位数组重建整数索引 (LSB-first)
///
/// i = Σ_{m=0}^{len-1} bit_m · 2^m
pub fn bits_to_int(bits: &[bool]) -> usize {
    bits.iter()
        .enumerate()
        .fold(0usize, |acc, (m, &bit)| acc | ((bit as usize) << m))
}

/// 将三维索引 (k, i, j) 展平为字典序一维索引
///
/// ## 字典序 (Lexicographical Order) 定义
/// 布尔变量拼接顺序: (z_{n_k-1} ... z_0, x_{n_v-1} ... x_0, y_{n_d-1} ... y_0)
/// 即 z 为最高位段，y 为最低位段：
///   idx = k * 2^{n_v + n_d} + i * 2^{n_d} + j
///
/// 此函数是 squarify 前的一维平铺核心，必须与超立方体上的
/// 布尔变量顺序保持严格一致，否则 PCS 打开验证将失败。
///
/// # 约束
/// - k ∈ [0, K_max)  (实际轮次索引，非经过补零对齐后的索引)
/// - i ∈ [0, N_max)   (行索引)
/// - j ∈ [0, D_max)   (列索引)
/// - n_v: 行布尔维度 (N_max = 2^{n_v})
/// - n_d: 列布尔维度 (D_max = 2^{n_d})
pub fn dict_lexicographic_3d_to_1d(k: usize, i: usize, j: usize, n_v: usize, n_d: usize) -> usize {
    debug_assert!(i < (1 << n_v), "row index out of bounds");
    debug_assert!(j < (1 << n_d), "col index out of bounds");
    let stride_2d = 1 << (n_v + n_d); // 每一层 (i,j) 平面的元素数
    let stride_col = 1 << n_d; // 每一行的元素数
    k * stride_2d + i * stride_col + j
}

/// 将二维索引 (i, j) 展平为字典序一维索引
///
/// idx = i * 2^{n_d} + j
pub fn dict_lexicographic_2d_to_1d(i: usize, j: usize, n_d: usize) -> usize {
    debug_assert!(j < (1 << n_d), "col index out of bounds");
    i * (1 << n_d) + j
}

// ============================================================================
// 张量积动态规划：O(N) 生成完整拉格朗日基向量
// ============================================================================

/// 利用张量积 (Kronecker Product) 动态规划，在 O(2^n) 时间内生成
/// 完整的拉格朗日基向量 L。
///
/// 底层算法:
/// 1. 初始化 L = [1]（长度为 1）
/// 2. 遍历 m 从 1 到 n_v:
///    - 将当前 L 复制一份
///    - 前半部分乘以 (1 - r_m)
///    - 后半部分乘以 r_m
///    - 拼接两部分，L 长度翻倍
///
/// 最终 L[i] = ∏_{m=1}^{n_v} (r_m · bit_m(i) + (1 - r_m)(1 - bit_m(i)))
///
/// # 参数
/// - r: 随机评估点向量 r ∈ Fp^{n}
///
/// # 返回
/// - 长度为 2^n 的拉格朗日基向量
pub fn build_lagrange_basis(r: &[Fp]) -> Vec<Fp> {
    let mut basis = vec![Fp::one()];

    // 坐标按 MSB-first 展开：每吸收一个新坐标，当前整段基底整体分裂为
    // bit=0 与 bit=1 两个分支。这与字典序索引 idx = i*2^n + j 对齐。
    for &r_m in r.iter() {
        let one_minus_r = Fp::one() - r_m;
        let mut next = Vec::with_capacity(basis.len() * 2);
        for old in basis {
            next.push(old * one_minus_r);
            next.push(old * r_m);
        }
        basis = next;
    }

    basis
}

/// 生成二维拉格朗日基向量对：行基 L_row ∈ Fp^{2^{n_v}} 和列基 L_col ∈ Fp^{2^{n_d}}
pub fn build_2d_lagrange_basis(r_x: &[Fp], r_y: &[Fp]) -> (Vec<Fp>, Vec<Fp>) {
    let row_basis = build_lagrange_basis(r_x);
    let col_basis = build_lagrange_basis(r_y);
    (row_basis, col_basis)
}

// ============================================================================
// 多线性扩展 (MLE) 求值
// ============================================================================

/// 对二维张量 (N × D) 进行多线性扩展求值
///
/// 对应公式:
/// ```text
/// T̃(x, y) = Σ_{i=0}^{N-1} Σ_{j=0}^{D-1} T[i,j] · eq(x, i) · eq(y, j)
/// ```
///
/// 使用拉格朗日基底向量实现高效求值:
/// - 先行求和: w_j = Σ_i L_row(i) · T[i, j]
/// - 再列求和: result = Σ_j L_col(j) · w_j
///
/// # 参数
/// - tensor: 矩阵数据 (row-major)，大小为 n_rows × n_cols
/// - n_rows: 物理行数
/// - n_cols: 物理列数
/// - r_x: 行评估点 ∈ Fp^{n_v}
/// - r_y: 列评估点 ∈ Fp^{n_d}
/// - nv: 布尔行维度 (n_rows = 2^nv)
/// - nd: 布尔列维度 (n_cols = 2^nd)
/// - n_real: ZENO 有效行边界 (可选 None = 全扫描)
/// - d_eff: ZENO 有效列边界 (可选 None = 全扫描)
pub fn eval_mle_2d(
    tensor: &[Fp],
    n_rows: usize,
    n_cols: usize,
    r_x: &[Fp],
    r_y: &[Fp],
    nv: usize,
    nd: usize,
    n_real: Option<usize>,
    d_eff: Option<usize>,
) -> Fp {
    assert_eq!(n_rows, 1 << nv, "n_rows must be 2^nv");
    assert_eq!(n_cols, 1 << nd, "n_cols must be 2^nd");
    let row_basis = build_lagrange_basis(r_x);
    let col_basis = build_lagrange_basis(r_y);

    let effective_rows = n_real.unwrap_or(n_rows);
    let effective_cols = d_eff.unwrap_or(n_cols);
    let compact_cols = if tensor.len() == n_rows * n_cols {
        n_cols
    } else if let Some(cols) = d_eff {
        cols
    } else {
        let side = (tensor.len() as f64).sqrt() as usize;
        if side * side == tensor.len() {
            side
        } else {
            n_cols
        }
    };
    let compact_rows = if tensor.len() == n_rows * n_cols {
        n_rows
    } else {
        tensor.len().checked_div(compact_cols).unwrap_or(0)
    };

    // 先行方向压缩: w_col[j] = Σ_i L_row[i] * T[i, j]
    let mut w_col = vec![Fp::zero(); n_cols];
    for j in 0..effective_cols {
        let mut sum = Fp::zero();
        for i in 0..effective_rows {
            let value = if i < compact_rows && j < compact_cols {
                tensor[i * compact_cols + j]
            } else {
                Fp::zero()
            };
            sum += row_basis[i] * value;
        }
        w_col[j] = sum;
    }

    // 再列方向压缩: result = Σ_j L_col[j] * w_col[j]
    let mut result = Fp::zero();
    for j in 0..effective_cols {
        result += col_basis[j] * w_col[j];
    }

    result
}

/// 对三维张量 (K × N × D) 进行多线性扩展求值
///
/// 对应公式:
/// ```text
/// T̃(z, x, y) = Σ_{k=1}^{K} Σ_{i=0}^{N-1} Σ_{j=0}^{D-1} T^{(k)}[i,j] · eq(z, bin(k)) · eq(x, i) · eq(y, j)
/// ```
///
/// # 参数
/// - slices: K 个二维矩阵，每层大小 n_rows × n_cols (row-major)
/// - k_start: 起始轮次编号 (通常为 1)
/// - n_rows: 物理行数
/// - n_cols: 物理列数
/// - r_z: 时间轴评估点 ∈ Fp^{n_k}
/// - r_x: 空间轴评估点 ∈ Fp^{n_v}
/// - r_y: 特征轴评估点 ∈ Fp^{n_d}
/// - nk: 时间布尔维度 (K = 2^nk 或向上取整)
/// - nv: 布尔行维度
/// - nd: 布尔列维度
/// - n_real: 有效行边界
/// - d_eff: 有效列边界
pub fn eval_mle_3d(
    slices: &[Vec<Fp>],
    k_start: usize,
    n_rows: usize,
    n_cols: usize,
    r_z: &[Fp],
    r_x: &[Fp],
    r_y: &[Fp],
    nk: usize,
    nv: usize,
    nd: usize,
    n_real: Option<usize>,
    d_eff: Option<usize>,
) -> Fp {
    assert_eq!(n_rows, 1 << nv, "n_rows must be 2^nv");
    assert_eq!(n_cols, 1 << nd, "n_cols must be 2^nd");
    assert_eq!(r_z.len(), nk, "r_z length must match nk");
    assert_eq!(r_x.len(), nv, "r_x length must match nv");
    assert_eq!(r_y.len(), nd, "r_y length must match nd");

    let time_basis = build_lagrange_basis(r_z);
    let row_basis = build_lagrange_basis(r_x);
    let col_basis = build_lagrange_basis(r_y);

    let effective_rows = n_real.unwrap_or(n_rows);
    let effective_cols = d_eff.unwrap_or(n_cols);

    let mut result = Fp::zero();

    for (k_idx, slice) in slices.iter().enumerate() {
        let k_val = k_start + k_idx;
        // eq(z, bin(k)) = time_basis[k_val] (仅在 k_val < 2^nk 时有定义)
        let t_weight = if k_val < (1 << nk) {
            time_basis[k_val]
        } else {
            Fp::zero()
        };

        if t_weight.is_zero() {
            continue;
        }

        // 对每一层执行 2D 求值
        let mut w_col = vec![Fp::zero(); n_cols];
        for j in 0..effective_cols {
            let mut sum = Fp::zero();
            for i in 0..effective_rows {
                sum += row_basis[i] * slice[i * n_cols + j];
            }
            w_col[j] = sum;
        }

        let mut slice_eval = Fp::zero();
        for j in 0..effective_cols {
            slice_eval += col_basis[j] * w_col[j];
        }

        result += t_weight * slice_eval;
    }

    result
}

// ============================================================================
// 一维多线性扩展 (用于内存轨迹)
// ============================================================================

/// 对一维数组进行多线性扩展求值
///
/// ```text
/// T̃(x) = Σ_{m=0}^{2^n-1} T[m] · eq(x, bin(m))
/// ```
pub fn eval_mle_1d(data: &[Fp], r: &[Fp], n: usize) -> Fp {
    assert_eq!(data.len(), 1 << n, "data length must be 2^n");
    let basis = build_lagrange_basis(r);
    let mut result = Fp::zero();
    for (i, &val) in data.iter().enumerate() {
        result += basis[i] * val;
    }
    result
}

// ============================================================================
// 矩阵方阵化 (Squarification) 辅助函数
// ============================================================================

/// 将高维张量一维平铺后重新折叠为方阵（用于 Hyrax 承诺）
///
/// 对应设计文档:
/// > 证明者在底层物理内存中，将其一维平铺后，重新折叠为一个
/// > 极其接近正方形的二维方阵
///
/// # 参数
/// - flat_data: 一维平铺数据
/// - total_bits: 总布尔变量数 μ_total
/// - nr: 行布尔维度 = ceil(μ_total / 2)
/// - nc: 列布尔维度 = floor(μ_total / 2)
///
/// # 返回
/// - 方阵 (row-major, 2^nr × 2^nc)，不足部分填零
pub fn squarify(flat_data: &[Fp], total_bits: usize) -> (usize, usize, Vec<Fp>) {
    let nr = total_bits.div_ceil(2); // ceil(μ/2)
    let nc = total_bits / 2; // floor(μ/2)
    let rows = 1 << nr;
    let cols = 1 << nc;
    let total_size = rows * cols;

    let mut matrix = vec![Fp::zero(); total_size];
    let copy_len = flat_data.len().min(total_size);
    matrix[..copy_len].copy_from_slice(&flat_data[..copy_len]);

    (nr, nc, matrix)
}

/// 从三维组合坐标截取方阵行/列坐标
///
/// 对应文档:
/// > 提取原始张量折叠前的总布尔变量数 μ_total = n_k + n_v + n_d
/// > 方阵行路由坐标: 截取前 nr = ceil(μ_total / 2) 个标量
/// > 方阵列路由坐标: 截取剩余 nc = floor(μ_total / 2) 个标量
pub fn split_squarified_coords(r_z: &[Fp], r_x: &[Fp], r_y: &[Fp]) -> (Vec<Fp>, Vec<Fp>) {
    let mu_total = r_z.len() + r_x.len() + r_y.len();
    let nr = mu_total.div_ceil(2);

    // 拼接所有坐标
    let full: Vec<Fp> = r_z
        .iter()
        .chain(r_x.iter())
        .chain(r_y.iter())
        .copied()
        .collect();

    let r_row = full[..nr.min(full.len())].to_vec();
    let r_col = full[nr.min(full.len())..].to_vec();

    (r_row, r_col)
}

// ============================================================================
// ZENO 剪枝：零行检测
// ============================================================================

/// 检测方阵中的纯零行，返回有效行索引列表
///
/// ZENO 静态掩码在内存映射时精准定位纯零行，触发短路承诺。
pub fn find_nonzero_rows(matrix: &[Fp], rows: usize, cols: usize) -> Vec<usize> {
    let mut nonzero = Vec::new();
    for i in 0..rows {
        let start = i * cols;
        let is_zero = matrix[start..start + cols].iter().all(|&v| v.is_zero());
        if !is_zero {
            nonzero.push(i);
        }
    }
    nonzero
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::Field;

    #[test]
    fn test_eq_scalar() {
        let x = vec![Fp::from(1u64), Fp::from(0u64), Fp::from(1u64)];
        let e_bits = vec![true, false, true];
        assert_eq!(eq_scalar(&x, &e_bits), Fp::one());

        let x_wrong = vec![Fp::from(0u64), Fp::from(1u64), Fp::from(1u64)];
        assert_eq!(eq_scalar(&x_wrong, &e_bits), Fp::zero());
    }

    #[test]
    fn test_int_to_bits() {
        // 5 = 0b101，低位在前
        let bits = int_to_bits(5, 4);
        assert_eq!(bits, vec![true, false, true, false]);
    }

    #[test]
    fn test_build_lagrange_basis_simple() {
        // n=1: r = [0.5]
        let r = vec![Fp::from(5u64) * Fp::from(10u64).inverse().unwrap()];
        let basis = build_lagrange_basis(&r);

        assert_eq!(basis.len(), 2);
        // L[0] = 1 - r₀ = 0.5
        assert_eq!(basis[0], Fp::one() - r[0]);
        // L[1] = r₀ = 0.5
        assert_eq!(basis[1], r[0]);
    }

    #[test]
    fn test_build_lagrange_basis_identity() {
        // 如果 r = [1, 1, ..., 1]，则 L[全部 1] = 1，其余 = 0
        let r = vec![Fp::one(); 3];
        let basis = build_lagrange_basis(&r);
        // 只有索引 7 (0b111) 处 = 1
        assert_eq!(basis[7], Fp::one());
        for value in basis.iter().take(7) {
            assert_eq!(*value, Fp::zero());
        }
    }

    #[test]
    fn test_eval_mle_2d() {
        // 2×2 矩阵: [[1, 2], [3, 4]]
        let tensor: Vec<Fp> = vec![
            Fp::from(1u64),
            Fp::from(2u64),
            Fp::from(3u64),
            Fp::from(4u64),
        ];

        // 在点 (1,0) 处求值应该得到 3 (第 1 行第 0 列)
        let r_x = vec![Fp::one(), Fp::zero()]; // i=2 (0b10) -> L[2]=1
        let r_y = vec![Fp::zero(), Fp::one()]; // j=1 (0b01) -> L[1]=1

        let val = eval_mle_2d(&tensor, 4, 4, &r_x, &r_y, 2, 2, None, None);

        // 期望值 = tensor[2*4 + 1] = tensor[9]
        // 但行 2 索引越界 (矩阵只有 2 行)，r_x = [1,0] 对应 i=2 在 4 行空间内
        // T[2,1] = 0 (超出行索引，padding zero)
        assert_eq!(val, Fp::zero());
    }

    #[test]
    fn test_eval_mle_2d_at_corner() {
        // 2×2 矩阵，r_x=[0,0], r_y=[0,0] → (0,0) = 1
        let tensor: Vec<Fp> = vec![
            Fp::from(1u64),
            Fp::from(2u64),
            Fp::from(3u64),
            Fp::from(4u64),
        ];
        let r_x = vec![Fp::zero(); 2];
        let r_y = vec![Fp::zero(); 2];

        let val = eval_mle_2d(&tensor, 4, 4, &r_x, &r_y, 2, 2, Some(2), Some(2));
        // L[0,0] = 1, T[0,0] = 1
        assert_eq!(val, Fp::one());
    }

    #[test]
    fn test_eval_mle_1d() {
        let data: Vec<Fp> = vec![
            Fp::from(5u64),  // bin(0)
            Fp::from(7u64),  // bin(1)
            Fp::from(9u64),  // bin(2)
            Fp::from(11u64), // bin(3)
        ];

        // r = [1, 0] → 求值 bin(2) 即数据 9
        let r = vec![Fp::one(), Fp::zero()];
        let val = eval_mle_1d(&data, &r, 2);
        assert_eq!(val, Fp::from(9u64));
    }

    #[test]
    fn test_squarify() {
        // μ = 5: nr=3 (8行), nc=2 (4列)
        let data: Vec<Fp> = (0..10).map(|i| Fp::from(i as u64)).collect();
        let (nr, nc, matrix) = squarify(&data, 5);

        assert_eq!(nr, 3);
        assert_eq!(nc, 2);
        assert_eq!(matrix.len(), 32); // 8×4

        // 前 10 个元素应保留
        for i in 0..10 {
            assert_eq!(matrix[i], data[i]);
        }
        // 其余为 0
        for value in matrix.iter().take(32).skip(10) {
            assert_eq!(*value, Fp::zero());
        }
    }

    #[test]
    fn test_split_squarified_coords() {
        let r_z = vec![Fp::from(1u64)];
        let r_x = vec![Fp::from(2u64), Fp::from(3u64)];
        let r_y = vec![Fp::from(4u64), Fp::from(5u64)];
        // μ = 5, nr=3, nc=2

        let (r_row, r_col) = split_squarified_coords(&r_z, &r_x, &r_y);

        assert_eq!(r_row.len(), 3);
        assert_eq!(r_col.len(), 2);

        assert_eq!(r_row[0], Fp::from(1u64));
        assert_eq!(r_row[1], Fp::from(2u64));
        assert_eq!(r_row[2], Fp::from(3u64));
        assert_eq!(r_col[0], Fp::from(4u64));
        assert_eq!(r_col[1], Fp::from(5u64));
    }

    #[test]
    fn test_find_nonzero_rows() {
        let matrix: Vec<Fp> = vec![
            Fp::from(1u64),
            Fp::from(2u64), // row 0: nonzero
            Fp::zero(),
            Fp::zero(), // row 1: zero
            Fp::zero(),
            Fp::zero(), // row 2: zero
            Fp::from(3u64),
            Fp::zero(), // row 3: nonzero
        ];

        let nonzero = find_nonzero_rows(&matrix, 4, 2);
        assert_eq!(nonzero, vec![0, 3]);
    }
}
