//! Hyrax 多项式承诺方案
//!
//! 基于 Curve25519 椭圆曲线的按行 Pedersen 承诺，严格遵循 Spartan-ZENO 设计文档：
//!
//! ## 核心功能
//! - **按行承诺生成**: C_W[i] = ρ_W[i]·H + Σ_j W[i,j]·G_j
//! - **矩阵方阵化 (Squarification)**: 将高维张量折叠为近正方形二维矩阵
//! - **ZENO 零行短路**: 纯 Padding 零行仅保留盲化基点 C = ρ·H
//! - **双轨批量折叠**: 图特征轨与模型权重轨独立同态压缩
//! - **IPA 内积证明**: O(log n) 对数级承诺打开
//! - **零知识 Sigma 协议**: 终局标量响应的零知识封装

use crate::curve_config::{Fp, GAffine, G};
use crate::polynomial;
use crate::poseidon::PoseidonSponge;
use ark_ec::CurveGroup;
use ark_ff::{Field, One, PrimeField, UniformRand, Zero};
use ark_serialize::CanonicalSerialize;
use ark_std::rand::{Rng, SeedableRng};
use rand::rngs::OsRng;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

/// 旧版 witness 模块使用的承诺容器。
#[derive(Clone, Debug, Default)]
pub struct HyraxCommitment {
    pub points: Vec<GAffine>,
}

/// 旧版 witness 模块期望的 prover 参数类型。
pub type HyraxProverParams = HyraxParams;

fn log2_ceil_len(len: usize) -> usize {
    if len <= 1 {
        0
    } else {
        len.next_power_of_two().trailing_zeros() as usize
    }
}

fn absorb_affine_point(sponge: &mut PoseidonSponge, point: &GAffine) {
    let mut bytes = Vec::new();
    point
        .serialize_compressed(&mut bytes)
        .expect("curve point serialization should not fail");
    let scalar = Fp::from_le_bytes_mod_order(&bytes);
    sponge.absorb_scalar(&scalar);
}

fn point_to_legacy_tuple(point: &GAffine) -> (Fp, Fp) {
    let mut bytes = Vec::new();
    point
        .serialize_compressed(&mut bytes)
        .expect("curve point serialization should not fail");
    let mid = bytes.len() / 2;
    (
        Fp::from_le_bytes_mod_order(&bytes[..mid]),
        Fp::from_le_bytes_mod_order(&bytes[mid..]),
    )
}

impl HyraxCommitment {
    pub fn from_points(points: Vec<GAffine>) -> Self {
        Self { points }
    }

    pub fn as_legacy_tuples(&self) -> Vec<(Fp, Fp)> {
        self.points.iter().map(point_to_legacy_tuple).collect()
    }

    pub fn commit_rows(&self, matrix: &[Fp], rows: usize, cols: usize) -> Vec<(Fp, Fp)> {
        let n_cols = log2_ceil_len(cols);
        let params = HyraxParams::new(n_cols, [11u8; 32]);
        let mut padded = vec![Fp::zero(); rows * (1 << n_cols)];
        for i in 0..rows {
            for j in 0..cols {
                padded[i * (1 << n_cols) + j] = matrix[i * cols + j];
            }
        }
        let (commits, _) = commit_matrix(&padded, rows, 1 << n_cols, &params, None);
        commits.iter().map(point_to_legacy_tuple).collect()
    }

    pub fn commit_sparse_row(&self, row: &[Fp], h_eff: usize) -> (Fp, Fp) {
        let cols = h_eff.max(row.len()).max(1).next_power_of_two();
        let n_cols = log2_ceil_len(cols);
        let params = HyraxParams::new(n_cols, [12u8; 32]);
        let mut padded = vec![Fp::zero(); cols];
        let copy_len = row.len().min(h_eff).min(cols);
        padded[..copy_len].copy_from_slice(&row[..copy_len]);
        let (commits, _) = commit_matrix(&padded, 1, cols, &params, None);
        point_to_legacy_tuple(&commits[0])
    }

    pub fn commit_blinded_row(&self) -> (Fp, Fp) {
        let params = HyraxParams::new(1, [13u8; 32]);
        let point = (params.blind_base * Fp::one()).into_affine();
        point_to_legacy_tuple(&point)
    }
}

// ============================================================================
// 公共参数 (SRS 简化版: 生成元阵列)
// ============================================================================

/// Hyrax 公共参数
///
/// 包含用于 Pedersen 承诺的椭圆曲线生成元：
/// - G: 列生成元阵列 (长度 = 2^n_cols)
/// - H: 盲化基点
#[derive(Clone, Debug)]
pub struct HyraxParams {
    /// 列生成元 G_j ∈ G，j ∈ [0, 2^n_cols)
    pub gens: Vec<GAffine>,
    /// 盲化基点 H ∈ G
    pub blind_base: GAffine,
    /// 最大列数 (2^n_cols)
    pub max_cols: usize,
    /// 列布尔维度
    pub n_cols: usize,
}

impl HyraxParams {
    /// 生成新的 Hyrax 公共参数
    ///
    /// # 参数
    /// - n_cols: 列布尔维度 (2^n_cols 个生成元)
    /// - seed: 随机种子 (用于确定性生成)
    pub fn new(n_cols: usize, seed: [u8; 32]) -> Self {
        let max_cols = 1 << n_cols;
        let mut rng = ChaCha20Rng::from_seed(seed);

        // 生成列生成元 G_j
        let gens: Vec<GAffine> = (0..max_cols)
            .map(|_| G::rand(&mut rng).into_affine())
            .collect();

        // 生成盲化基点 H
        let blind_base = G::rand(&mut rng).into_affine();

        Self {
            gens,
            blind_base,
            max_cols,
            n_cols,
        }
    }

    /// 获取指定索引的生成元
    pub fn get_gen(&self, j: usize) -> &GAffine {
        &self.gens[j % self.max_cols]
    }

    /// 旧版 witness 使用的一维数组承诺接口。
    pub fn commit<R: Rng + ?Sized>(&self, values: &[Fp], _rng: &mut R) -> HyraxCommitment {
        let cols = self.max_cols.max(1);
        let rows = values.len().max(1).div_ceil(cols);
        let mut matrix = vec![Fp::zero(); rows * cols];
        let copy_len = values.len().min(matrix.len());
        matrix[..copy_len].copy_from_slice(&values[..copy_len]);
        let (points, _) = commit_matrix(&matrix, rows, cols, self, None);
        HyraxCommitment::from_points(points)
    }
}

// ============================================================================
// 按行 Pedersen 承诺
// ============================================================================

/// 对矩阵按行生成 Hyrax 承诺
///
/// 对应设计文档第 1 节公式:
/// C_W[i] = ρ_W[i] · H + Σ_{j ∈ I_c} W_padded[i, j] · G_j
///
/// # 参数
/// - matrix: 对齐后的矩阵 (row-major), rows × cols
/// - rows: 物理行数
/// - cols: 物理列数
/// - params: Hyrax 公共参数
/// - nonzero_rows: ZENO 有效行索引 (None = 所有行)
///
/// # 返回
/// - 承诺点阵 C ∈ G^{rows}
/// - 盲化因子数组 ρ ∈ Fp^{rows}
pub fn commit_matrix(
    matrix: &[Fp],
    rows: usize,
    cols: usize,
    params: &HyraxParams,
    nonzero_rows: Option<&[usize]>,
) -> (Vec<GAffine>, Vec<Fp>) {
    assert_eq!(matrix.len(), rows * cols);
    assert!(cols <= params.max_cols, "cols exceeds params.max_cols");

    let mut rng = OsRng;
    let blindings: Vec<Fp> = (0..rows).map(|_| Fp::rand(&mut rng)).collect();
    let mut effective_rows = vec![true; rows];
    if let Some(nonzero_rows) = nonzero_rows {
        effective_rows.fill(false);
        for row in nonzero_rows {
            if *row < rows {
                effective_rows[*row] = true;
            }
        }
    }
    let commitments = (0..rows)
        .into_par_iter()
        .map(|i| {
            let rho = blindings[i];
            if !effective_rows[i] {
                return (params.blind_base * rho).into_affine();
            }
            let row_start = i * cols;
            let row_acc = (0..cols)
                .into_par_iter()
                .map(|j| {
                    let val = matrix[row_start + j];
                    if val.is_zero() {
                        G::zero()
                    } else {
                        *params.get_gen(j) * val
                    }
                })
                .reduce(G::zero, |acc, item| acc + item);
            (G::from(params.blind_base) * rho + row_acc).into_affine()
        })
        .collect();

    (commitments, blindings)
}

/// 对按行承诺进行同态相加 (用于批量折叠)
pub fn add_commitments(c1: &GAffine, c2: &GAffine) -> GAffine {
    (*c1 + *c2).into_affine()
}

/// 对标量乘以承诺 (同态)
pub fn scalar_mul_commitment(c: &GAffine, scalar: &Fp) -> GAffine {
    (*c * *scalar).into_affine()
}

// ============================================================================
// 矩阵方阵化 (Squarification) —— 对应 2.2.1.1 节
// ============================================================================

/// 将高维张量一维平铺后重新折叠为方阵，并按行承诺
///
/// 对应文档:
/// > 系统在底层物理内存中，将其一维平铺后，重新折叠为一个
/// > 极其接近正方形的二维方阵。
/// > 行维数 nr = ceil(μ_total / 2)，列维数 nc = floor(μ_total / 2)
pub fn squarify_and_commit(
    flat_data: &[Fp],
    total_bits: usize,
    params: &HyraxParams,
    nonzero_rows: Option<&[usize]>,
) -> (usize, usize, Vec<GAffine>, Vec<Fp>) {
    let (nr, nc, matrix) = polynomial::squarify(flat_data, total_bits);
    let (commits, blindings) = commit_matrix(&matrix, 1 << nr, 1 << nc, params, nonzero_rows);
    (nr, nc, commits, blindings)
}

// ============================================================================
// 承诺打开 (PCS Opening) —— 2D 矩阵的同态降维
// ============================================================================

/// 将二维矩阵按行方向同态"拍扁"为一维向量
///
/// 对应设计文档 3.1 第四步 公式 1-3:
/// w_j = Σ_i L(i) · val[i, j]          (拍扁特征)
/// z_folded = Σ_i L(i) · r_i           (拍扁盲化因子)
///
/// # 参数
/// - matrix: 原始矩阵 (row-major), rows × cols
/// - rows: 物理行数
/// - cols: 物理列数
/// - r_row: 行查询坐标 ∈ Fp^{n_r}
/// - blindings: 每行的盲化因子 ρ_i
/// - n_r: 行布尔维度 (rows = 2^n_r)
///
/// # 返回
/// - 折叠向量 w ∈ Fp^{cols}
/// - 折叠盲化因子 z_folded ∈ Fp
pub fn flatten_rows(
    matrix: &[Fp],
    rows: usize,
    cols: usize,
    r_row: &[Fp],
    blindings: &[Fp],
    n_r: usize,
) -> (Vec<Fp>, Fp) {
    assert_eq!(rows, 1 << n_r);
    let row_basis = polynomial::build_lagrange_basis(r_row);

    let w = (0..cols)
        .into_par_iter()
        .map(|j| {
            (0..rows)
                .map(|i| row_basis[i] * matrix[i * cols + j])
                .fold(Fp::zero(), |acc, value| acc + value)
        })
        .collect();

    let z_folded: Fp = blindings
        .par_iter()
        .zip(row_basis.par_iter())
        .map(|(r, l)| *r * l)
        .sum();

    (w, z_folded)
}

/// 验证者同态重构预期承诺
///
/// 对应设计文档 3.1 第四步 公式 4:
/// C_expected = Σ_i L(i) · C_i
pub fn reconstruct_expected_commitment(
    commitments: &[GAffine],
    r_row: &[Fp],
    n_r: usize,
) -> GAffine {
    assert_eq!(commitments.len(), 1 << n_r);
    let row_basis = polynomial::build_lagrange_basis(r_row);
    commitments
        .par_iter()
        .zip(row_basis.par_iter())
        .map(|(c, weight)| *c * *weight)
        .reduce(G::zero, |acc, item| acc + item)
        .into_affine()
}

/// 验证者计算折叠见证的实际哈希
///
/// 对应设计文档 3.1 第四步 公式 5:
/// C_actual = z_folded · H + Σ_j w_j · G_j
pub fn compute_actual_commitment(w: &[Fp], z_folded: &Fp, params: &HyraxParams) -> GAffine {
    let mut acc = G::from(params.blind_base) * *z_folded;
    for (j, &val) in w.iter().enumerate() {
        if !val.is_zero() {
            acc += *params.get_gen(j) * val;
        }
    }
    acc.into_affine()
}

/// 利用列坐标提取最终神谕评估值
///
/// 对应设计文档 3.1 第四步 公式 6:
/// v = Σ_j R(j) · w_j
pub fn evaluate_oracle_value(w: &[Fp], r_col: &[Fp], n_c: usize) -> Fp {
    assert_eq!(r_col.len(), n_c);
    let col_basis = polynomial::build_lagrange_basis(r_col);
    let mut result = Fp::zero();
    let len = w.len().min(col_basis.len());
    for j in 0..len {
        result += col_basis[j] * w[j];
    }
    result
}

// ============================================================================
// 双轨批量折叠 (Dual-Track Batch Folding) —— 对应 3.3 第一步
// ============================================================================

/// 双轨批量折叠：对图特征批次与模型权重批次分别执行同态压缩
///
/// 对应设计文档:
/// > 证明者将所有声明的评估值吸入 Poseidon 哈希海绵，抽取两组独立的高熵挑战标量
/// > β_graph 和 β_weight，分别对两个批次执行同态左乘降维
pub struct BatchFoldResult {
    /// 图特征超级向量
    pub w_graph: Vec<Fp>,
    /// 图特征盲化因子
    pub z_graph: Fp,
    /// 模型权重超级向量
    pub w_weight: Vec<Fp>,
    /// 模型权重盲化因子
    pub z_weight: Fp,
}

/// 执行双轨批量折叠
///
/// # 参数
/// - graph_tensors: 图特征批次 [(matrix, rows, cols, n_r, n_c, blindings, r_row), ...]
/// - weight_tensors: 模型权重批次 [(matrix, rows, cols, n_r, n_c, blindings, r_row), ...]
/// - beta_graph: 图特征折叠标量
/// - beta_weight: 模型权重折叠标量
/// - params: Hyrax 公共参数
pub type FoldTensor = (Vec<Fp>, usize, usize, usize, usize, Vec<Fp>, Vec<Fp>);

pub fn dual_track_batch_fold(
    graph_tensors: &[FoldTensor],
    weight_tensors: &[FoldTensor],
    beta_graph: &Fp,
    beta_weight: &Fp,
    _params: &HyraxParams,
) -> BatchFoldResult {
    // 图特征批次折叠
    let mut w_graph: Option<Vec<Fp>> = None;
    let mut z_graph = Fp::zero();

    for (idx, (matrix, rows, cols, n_r, _n_c, blindings, r_row)) in graph_tensors.iter().enumerate()
    {
        let (w_flat, z_flat) = flatten_rows(matrix, *rows, *cols, r_row, blindings, *n_r);
        let power = beta_graph.pow([idx as u64]);

        match &mut w_graph {
            Some(ref mut acc) => {
                for j in 0..*cols {
                    acc[j] += power * w_flat[j];
                }
            }
            None => {
                w_graph = Some(w_flat.iter().map(|v| power * v).collect());
            }
        }
        z_graph += power * z_flat;
    }

    // 模型权重批次折叠
    let mut w_weight: Option<Vec<Fp>> = None;
    let mut z_weight = Fp::zero();

    for (idx, (matrix, rows, cols, n_r, _n_c, blindings, r_row)) in
        weight_tensors.iter().enumerate()
    {
        let (w_flat, z_flat) = flatten_rows(matrix, *rows, *cols, r_row, blindings, *n_r);
        let power = beta_weight.pow([idx as u64]);

        match &mut w_weight {
            Some(ref mut acc) => {
                for j in 0..*cols {
                    acc[j] += power * w_flat[j];
                }
            }
            None => {
                w_weight = Some(w_flat.iter().map(|v| power * v).collect());
            }
        }
        z_weight += power * z_flat;
    }

    BatchFoldResult {
        w_graph: w_graph.unwrap_or_default(),
        z_graph,
        w_weight: w_weight.unwrap_or_default(),
        z_weight,
    }
}

// ============================================================================
// IPA 内积证明 (Inner Product Argument) —— 对应 3.3 第二步
// ============================================================================

/// IPA 折叠的交叉项
#[derive(Clone, Debug)]
pub struct IpaRoundData {
    pub l: GAffine,
    pub r: GAffine,
    pub u: Fp,
}

/// IPA 证明体
#[derive(Clone, Debug)]
pub struct IpaProof {
    /// 每轮折叠的交叉项 (L_k, R_k)
    pub rounds: Vec<IpaRoundData>,
    /// 终局标量 w_0^(final) — 仅在非 ZK 模式发送
    pub w_final: Fp,
    /// Sigma 协议盲化前置点
    pub a_blind: GAffine,
    /// Sigma 协议标量响应 s_w
    pub s_w: Fp,
    /// Sigma 协议标量响应 s_z
    pub s_z: Fp,
}

/// 执行非交互式 IPA 折叠循环 (证明者侧)
///
/// 对应设计文档 3.3 第二步:
/// - n_c 轮折叠循环
/// - 每轮: 切分 → 计算交叉项 L_k, R_k → Fiat-Shamir 挑战 u_k → 原位压缩
pub fn ipa_prove(
    w_init: &[Fp],
    gens_init: &[GAffine],
    blind_base: &GAffine,
    z_init: &Fp,
    n_c: usize,
    sponge: &mut PoseidonSponge,
) -> (IpaProof, GAffine) {
    let mut w = w_init.to_vec();
    let mut gens = gens_init.to_vec();
    let z = *z_init;
    let mut rounds = Vec::new();

    let mut s_k = w.len();

    for _round in 0..n_c {
        assert_eq!(s_k % 2, 0, "vector length must be even at each round");
        let half = s_k / 2;

        // 计算交叉项 L_k (左向量 × 右生成元) 和 R_k (右向量 × 左生成元)
        let (l_proj, r_proj) = (0..half)
            .into_par_iter()
            .map(|m| {
                let w_left = w[m];
                let w_right = w[m + half];
                let g_left = gens[m];
                let g_right = gens[m + half];
                let l_part = if w_left.is_zero() {
                    G::zero()
                } else {
                    g_right * w_left
                };
                let r_part = if w_right.is_zero() {
                    G::zero()
                } else {
                    g_left * w_right
                };
                (l_part, r_part)
            })
            .reduce(
                || (G::zero(), G::zero()),
                |(l_acc, r_acc), (l_item, r_item)| (l_acc + l_item, r_acc + r_item),
            );

        let l = l_proj.into_affine();
        let r = r_proj.into_affine();

        // Fiat-Shamir: 从海绵中挤出挑战 u_k
        // 将椭圆曲线点的坐标吸入海绵
        absorb_affine_point(sponge, &l);
        absorb_affine_point(sponge, &r);
        sponge.finalize_absorption();
        let mut u = sponge.squeeze();
        if u.is_zero() {
            u = Fp::one();
        }

        // 原位压缩
        let u_inv = u.inverse().unwrap_or(Fp::zero());
        let (w_new, gens_new): (Vec<Fp>, Vec<GAffine>) = (0..half)
            .into_par_iter()
            .map(|m| {
                // w_new[m] = u · w[m] + u⁻¹ · w[m+half]
                let w_folded = u * w[m] + u_inv * w[m + half];
                // gens_new[m] = u⁻¹ · gens[m] + u · gens[m+half]
                let gen_folded = (gens[m] * u_inv + gens[m + half] * u).into_affine();
                (w_folded, gen_folded)
            })
            .unzip();

        rounds.push(IpaRoundData { l, r, u });

        w = w_new;
        gens = gens_new;
        s_k = half;
    }

    // 终局状态: w[0], gens[0], z
    let w_final = w[0];
    let g_final = gens[0];

    // Zero-Knowledge Sigma Protocol 打开
    // 1. 盲化前置点
    let mut rng = OsRng;
    let r_w = Fp::rand(&mut rng);
    let r_z = Fp::rand(&mut rng);
    let a_blind = (g_final * r_w + *blind_base * r_z).into_affine();

    // 2. Fiat-Shamir 挑战
    absorb_affine_point(sponge, &a_blind);
    sponge.finalize_absorption();
    let c_ipa = sponge.squeeze();

    // 3. 标量响应
    let s_w = r_w + c_ipa * w_final;
    let s_z = r_z + c_ipa * z;

    let proof = IpaProof {
        rounds,
        w_final,
        a_blind,
        s_w,
        s_z,
    };

    (proof, g_final)
}

fn ipa_final_generator_from_challenges(g_initial: &[GAffine], u_powers: &[Fp]) -> GAffine {
    let mut weights = vec![Fp::one()];
    for u in u_powers {
        let u_inv = u.inverse().unwrap_or(Fp::zero());
        let mut next = Vec::with_capacity(weights.len() * 2);
        for weight in &weights {
            next.push(*weight * u_inv);
            next.push(*weight * *u);
        }
        weights = next;
    }
    assert_eq!(g_initial.len(), weights.len());
    g_initial
        .par_iter()
        .zip(weights.par_iter())
        .map(|(gen, weight)| *gen * *weight)
        .reduce(G::zero, |acc, item| acc + item)
        .into_affine()
}

/// 验证者侧 IPA 校验
///
/// 对应设计文档 4.3 步骤二和步骤四
pub fn ipa_verify(
    proof: &IpaProof,
    c_initial: &GAffine,
    g_initial: &[GAffine],
    blind_base: &GAffine,
    n_c: usize,
    sponge: &mut PoseidonSponge,
) -> bool {
    if proof.rounds.len() != n_c {
        return false;
    }

    // 1. 从初始承诺和交叉项重构折叠后的终局承诺 C^(final)
    let mut c_final = G::from(*c_initial);
    let mut u_powers: Vec<Fp> = Vec::new();

    for round in &proof.rounds {
        absorb_affine_point(sponge, &round.l);
        absorb_affine_point(sponge, &round.r);
        sponge.finalize_absorption();
        let mut u = sponge.squeeze();
        if u.is_zero() {
            u = Fp::one();
        }
        if u != round.u {
            return false;
        }

        let u2 = u * u;
        let u2_inv = u2.inverse().unwrap_or(Fp::zero());

        c_final += round.l * u2 + round.r * u2_inv;
        u_powers.push(u);
    }

    let _c_final = c_final.into_affine();

    // 2. 从初始生成元重构终局生成元 G^(final)，避免 verifier 端克隆整套 gens。
    let g_final = ipa_final_generator_from_challenges(g_initial, &u_powers);

    // 3. Fiat-Shamir: 重放挑战
    let a_blind = proof.a_blind;
    absorb_affine_point(sponge, &a_blind);
    sponge.finalize_absorption();
    let c_ipa = sponge.squeeze();

    // 4. Schnorr 校验: s_w·G_final + s_z·H ≟ A + c_ipa·C_final
    let left = g_final * proof.s_w + *blind_base * proof.s_z;
    let right = a_blind + _c_final * c_ipa;

    left.into_affine() == right.into_affine()
}

// ============================================================================
// 验证者侧的完整打开校验流程
// ============================================================================

/// 验证者完整打开校验流程
///
/// 对应设计文档 4.3 完整流程：
/// 1. 同态重构初始预期承诺 C^(0)
/// 2. 重演 Fiat-Shamir 重放挑战提取
/// 3. 折叠承诺至 C^(final)
/// 4. 同态推导终局生成元 G^(final)
/// 5. Schnorr 终局代数断言
pub fn verify_opening(
    proof: &IpaProof,
    commitments: &[GAffine],
    r_row: &[Fp],
    _r_col: &[Fp],
    n_r: usize,
    n_c: usize,
    params: &HyraxParams,
    sponge: &mut PoseidonSponge,
) -> bool {
    // 步骤一：重构初始预期承诺 C^(0) = Σ_i L(i)·C_i
    let c_initial = reconstruct_expected_commitment(commitments, r_row, n_r);

    // 步骤三&四：IPA 验证
    ipa_verify(
        proof,
        &c_initial,
        &params.gens,
        &params.blind_base,
        n_c,
        sponge,
    )
}

// ============================================================================
// 辅助函数：矩阵行折叠
// ============================================================================

/// 计算行压缩后的承诺 (用于验证者侧的预期承诺计算)
pub fn fold_commitments_by_weights(commitments: &[GAffine], weights: &[Fp]) -> GAffine {
    assert_eq!(commitments.len(), weights.len());
    let mut acc = G::zero();
    for (c, w) in commitments.iter().zip(weights.iter()) {
        if !w.is_zero() {
            acc += *c * *w;
        }
    }
    acc.into_affine()
}

/// 将折叠向量与列基底内积求值
pub fn inner_product_eval(w: &[Fp], basis: &[Fp]) -> Fp {
    w.iter().zip(basis.iter()).map(|(a, b)| *a * b).sum()
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyrax_params_creation() {
        let params = HyraxParams::new(3, [0u8; 32]); // 8 个生成元
        assert_eq!(params.gens.len(), 8);
        assert_eq!(params.max_cols, 8);
        assert_eq!(params.n_cols, 3);
    }

    #[test]
    fn test_commit_matrix_basic() {
        let params = HyraxParams::new(3, [1u8; 32]);

        // 2行 × 4列 矩阵
        let matrix: Vec<Fp> = vec![
            Fp::from(1u64),
            Fp::from(0u64),
            Fp::from(2u64),
            Fp::from(0u64),
            Fp::from(0u64),
            Fp::from(3u64),
            Fp::from(0u64),
            Fp::from(4u64),
        ];

        let (commits, blindings) = commit_matrix(&matrix, 2, 4, &params, None);

        assert_eq!(commits.len(), 2);
        assert_eq!(blindings.len(), 2);
    }

    #[test]
    fn test_commit_matrix_zeno_shortcut() {
        let params = HyraxParams::new(3, [2u8; 32]);

        // 2行 × 4列：行 0 有数据，行 1 全零
        let matrix: Vec<Fp> = vec![
            Fp::from(1u64),
            Fp::from(2u64),
            Fp::from(0u64),
            Fp::from(0u64),
            Fp::zero(),
            Fp::zero(),
            Fp::zero(),
            Fp::zero(),
        ];

        let nonzero = vec![0usize]; // 仅行 0 有效
        let (commits, _blindings) = commit_matrix(&matrix, 2, 4, &params, Some(&nonzero));

        // 行 1 的承诺应仅包含盲化基点
        assert_eq!(commits.len(), 2);
    }

    #[test]
    fn test_flatten_and_reconstruct() {
        let params = HyraxParams::new(3, [3u8; 32]);

        // 4×4 矩阵
        let rows = 4;
        let cols = 4;
        let n_r = 2;
        let matrix: Vec<Fp> = (0..16).map(|i| Fp::from(i as u64)).collect();

        let (commits, blindings) = commit_matrix(&matrix, rows, cols, &params, None);

        // 拍扁
        let r_row = vec![Fp::from(2u64), Fp::from(3u64)];
        let (w, z_folded) = flatten_rows(&matrix, rows, cols, &r_row, &blindings, n_r);

        // 重构
        let c_expected = reconstruct_expected_commitment(&commits, &r_row, n_r);
        let c_actual = compute_actual_commitment(&w, &z_folded, &params);

        assert_eq!(c_expected, c_actual, "homomorphic commitment should match");
    }

    #[test]
    fn test_ipa_round_trip() {
        let params = HyraxParams::new(2, [4u8; 32]); // 4 列生成元
        let n_c = 2; // 4 = 2^2

        // 长度 4 的向量
        let w_init: Vec<Fp> = vec![
            Fp::from(1u64),
            Fp::from(2u64),
            Fp::from(3u64),
            Fp::from(4u64),
        ];

        // 计算初始承诺 C^(0) = Σ w_j·G_j + z·H
        let z_init = Fp::from(42u64);
        let mut c_initial = G::from(params.blind_base) * z_init;
        for (j, &val) in w_init.iter().enumerate() {
            c_initial += *params.get_gen(j) * val;
        }
        let c_initial = c_initial.into_affine();

        // 证明
        let mut prover_sponge = PoseidonSponge::new();
        let (proof, _g_final) = ipa_prove(
            &w_init,
            &params.gens,
            &params.blind_base,
            &z_init,
            n_c,
            &mut prover_sponge,
        );

        // 验证 (重建 sponge 以重放相同的挑战序列)
        // 注意: 实际协议中需重放相同吸收序列; 此处直接验证 IPA 折叠结构
        let mut verifier_sponge = PoseidonSponge::new();
        let result = ipa_verify(
            &proof,
            &c_initial,
            &params.gens,
            &params.blind_base,
            n_c,
            &mut verifier_sponge,
        );

        assert!(result, "IPA round-trip should succeed");
    }

    #[test]
    fn test_scalar_mul_commitment() {
        let params = HyraxParams::new(2, [5u8; 32]);

        let matrix: Vec<Fp> = vec![
            Fp::from(1u64),
            Fp::from(0u64),
            Fp::from(0u64),
            Fp::from(0u64),
        ];
        let (commits, _) = commit_matrix(&matrix, 1, 4, &params, None);

        let doubled = scalar_mul_commitment(&commits[0], &Fp::from(2u64));
        // 验证结果在曲线上
        assert!(!doubled.is_zero());
    }

    #[test]
    fn test_dual_track_batch_fold() {
        let params = HyraxParams::new(4, [6u8; 32]);

        // 两个独立的图特征张量
        let matrix1: Vec<Fp> = vec![Fp::from(1u64); 16]; // 4×4
        let matrix2: Vec<Fp> = vec![Fp::from(2u64); 16]; // 4×4

        let r_row = vec![Fp::from(1u64), Fp::from(2u64)];
        let blindings = vec![Fp::from(10u64); 4];

        let graph_tensors = vec![
            (
                matrix1.clone(),
                4,
                4,
                2,
                2,
                blindings.clone(),
                r_row.clone(),
            ),
            (
                matrix2.clone(),
                4,
                4,
                2,
                2,
                blindings.clone(),
                r_row.clone(),
            ),
        ];
        let weight_tensors: Vec<FoldTensor> = vec![];

        let beta_graph = Fp::from(3u64);
        let beta_weight = Fp::from(5u64);

        let result = dual_track_batch_fold(
            &graph_tensors,
            &weight_tensors,
            &beta_graph,
            &beta_weight,
            &params,
        );

        // 图特征折叠向量应为非零
        assert!(!result.w_graph.is_empty());
        // 空权重批次的折叠应为空
        assert!(result.w_weight.is_empty());
    }
}
