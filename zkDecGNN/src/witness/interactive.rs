//! 交互式见证生成 (Interactive Witness Generation)
//!
//! 严格遵循 Spartan-ZENO 设计文档第 2.2 节。
//!
//! ## 核心流程
//!
//! ### 2.2.1 提交承诺与获取随机挑战 (Fiat-Shamir Heuristic)
//! - 将确定性见证转化为多项式承诺
//! - 通过 Poseidon 海绵生成所有随机挑战标量
//!
//! ### 2.2.2 内存元组的代数压缩与时序排序校验
//! - RLC 特征坍缩
//! - 双轨排序轨迹生成
//! - 全局连乘排列校验
//!
//! ### 2.2.3 构建连乘二叉树与底层轨迹承诺
//! - 四组二叉树构建
//! - 底层轨迹的多线性扩展与 Hyrax 承诺
//!
//! ### 2.2.4 非线性辅助见证的扩展与承诺
//! - ReLU 符号位矩阵的 MLE 与承诺

use crate::curve_config::Fp;
use crate::hyrax::HyraxCommitment;
use crate::poseidon::PoseidonHash;
use ark_ff::{One, Zero};
use std::collections::HashMap;

use super::transcript::Transcript;
use super::ContainerDimensions;

// ============================================================================
// 交互式见证生成器
// ============================================================================

/// 交互式见证生成器
///
/// 将确定性见证 (Stage 1-3) 转化为承诺，
/// 并通过 Fiat-Shamir 变换获取全部随机挑战。
#[derive(Debug)]
pub struct InteractiveWitnessGenerator {
    /// 容器维度
    pub dims: ContainerDimensions,
    /// 哈希海绵 (用于 Fiat-Shamir)
    pub poseidon: PoseidonHash,
    /// 公共参数哈希输入
    pub pp_serialized: Vec<Fp>,
    /// Hyrax 承诺参数
    pub hyrax: HyraxCommitment,
    /// 确定性见证数据
    pub deterministic_witness: Option<DeterministicWitness>,
}

/// 确定性见证数据包
#[derive(Debug, Clone)]
pub struct DeterministicWitness {
    /// X_base (N_max × D_max, row-major)
    pub x_base: Vec<Fp>,
    /// 每轮扩散状态 S^{(k)}
    pub s_states: Vec<Vec<Fp>>,
    /// 每轮聚合状态 Z^{(k)}
    pub z_states: Vec<Vec<Fp>>,
    /// 前置 ReLU 符号位
    pub s_sign_pre: Vec<Fp>,
    /// 后置 ReLU 符号位
    pub s_sign: Vec<Fp>,
    /// 深度对齐的张量 (已方阵化)
    pub squarified_tensors: HashMap<String, Vec<Fp>>,
    /// SPARK 内存操作流
    pub spark_init: Vec<(u64, Fp, u64)>, // (addr, val, ts)
    pub spark_rs: Vec<(u64, Fp, u64)>,    // (addr, val, ts)
    pub spark_ws: Vec<(u64, Fp, u64)>,    // (addr, val, ts)
    pub spark_audit: Vec<(u64, Fp, u64)>, // (addr, val, ts)
    /// 最终 Logits \tilde{Y}
    pub y_tilde: Vec<Fp>,
}

impl InteractiveWitnessGenerator {
    /// 创建新的交互式见证生成器
    pub fn new(dims: &ContainerDimensions, hyrax: HyraxCommitment) -> Self {
        let poseidon = PoseidonHash::new();

        // 序列化公共参数 (对应文档 V_pp 的 16 个标量)
        let pp_serialized = vec![
            Fp::from(dims.n_max as u64),
            Fp::from(dims.d_max as u64),
            Fp::from(dims.h_max as u64),
            Fp::from(dims.edges as u64),
            Fp::from(dims.n_real as u64),
            Fp::from(dims.d as u64),
            Fp::from(dims.h as u64),
            Fp::from(dims.k as u64),
            // alpha_pre, beta, model_id 将在后续填入
            Fp::zero(), // alpha_pre placeholder
            Fp::zero(), // beta placeholder
            Fp::zero(), // ID_model placeholder
            // gamma_1, gamma_2, gamma_3
            Fp::zero(),
            Fp::zero(),
            Fp::zero(),
            // lambda_1, lambda_2
            Fp::zero(),
            Fp::zero(),
        ];

        Self {
            dims: dims.clone(),
            poseidon,
            pp_serialized,
            hyrax,
            deterministic_witness: None,
        }
    }

    /// 设置模型参数 (超参数 + 架构指纹)
    pub fn set_model_params(
        &mut self,
        alpha_pre: f64,
        beta: f64,
        model_id: Fp,
        hyperparams: &HashMap<String, f64>,
    ) {
        self.pp_serialized[8] = Fp::from(alpha_pre as u64);
        self.pp_serialized[9] = Fp::from(beta as u64);
        self.pp_serialized[10] = model_id;

        // 提取超参数 (gamma_1, gamma_2, gamma_3, lambda_1, lambda_2)
        for (key, val) in hyperparams {
            let fp_val = Fp::from(*val as u64);
            match key.as_str() {
                "gamma_1" => self.pp_serialized[11] = fp_val,
                "gamma_2" => self.pp_serialized[12] = fp_val,
                "gamma_3" => self.pp_serialized[13] = fp_val,
                "lambda_1" => self.pp_serialized[14] = fp_val,
                "lambda_2" => self.pp_serialized[15] = fp_val,
                _ => {}
            }
        }
    }

    /// 摄入确定性见证数据
    pub fn set_deterministic_witness(&mut self, witness: DeterministicWitness) {
        self.deterministic_witness = Some(witness);
    }

    /// 为张量生成 Hyrax 方阵化承诺
    ///
    /// 对应文档 2.2.1.1 节：
    /// - 将数据一维平铺后重新折叠为方阵
    /// - 按行生成 Pedersen 承诺
    pub fn commit_squarified_tensor(
        &mut self,
        _name: &str,
        tensor: &[Fp],
        total_bool_vars: usize,
    ) -> Vec<(Fp, Fp)> {
        let n_c = total_bool_vars.div_ceil(2);
        let n_r = total_bool_vars - n_c;
        let num_cols = 1 << n_c;
        let num_rows = 1 << n_r;

        // 一维平铺 → 方阵重塑
        let mut squarified = vec![Fp::zero(); num_rows * num_cols];
        for (idx, &val) in tensor.iter().enumerate() {
            if idx < squarified.len() {
                squarified[idx] = val;
            }
        }

        // 按行承诺
        self.hyrax.commit_rows(&squarified, num_rows, num_cols)
    }

    // ========================================================================
    // Phase 2.2.1.2: 生成转录本并抽取全局挑战
    // ========================================================================

    /// 执行 Fiat-Shamir 变换：吸入全部承诺，挤出全局挑战
    ///
    /// 对应文档第 2.2.1.2 节第二步和第三步。
    pub fn generate_transcript(
        &mut self,
        commitment_map: &HashMap<String, Vec<(Fp, Fp)>>,
        trace_commitments: &HashMap<String, Vec<(Fp, Fp)>>,
        y_tilde: &[Fp],
    ) -> Transcript {
        let mut sponge = PoseidonHash::new();

        // ==== 第一步：吸入公共参数 V_pp ====
        for &val in &self.pp_serialized {
            sponge.absorb(&[val]);
        }

        // ==== 第二步：吸入张量承诺坐标 ====
        // 按固定顺序吸入以确保确定性和可复现
        let commit_order = ["X_base", "S_global", "Z_global", "ReLU_pre", "ReLU_post"];

        for name in &commit_order {
            if let Some(commits) = commitment_map.get(*name) {
                for &(x, y) in commits {
                    sponge.absorb(&[x, y]);
                }
            }
        }

        // ==== 第三步：吸入内存轨迹承诺 ====
        let trace_order = ["V_read", "T_read", "V_write", "T_write"];
        for name in &trace_order {
            if let Some(commits) = trace_commitments.get(*name) {
                for &(x, y) in commits {
                    sponge.absorb(&[x, y]);
                }
            }
        }

        // ==== 第四步：吸入公开输出 Ÿ ====
        for &val in y_tilde {
            sponge.absorb(&[val]);
        }

        // ==== 第五步：挤出全部挑战标量 ====
        let mut transcript = Transcript::new();

        // SPARK 内存压缩标量
        // δ₁, δ₂, τ: 元组代数压缩
        transcript.delta_1 = sponge.squeeze();
        transcript.delta_2 = sponge.squeeze();
        transcript.tau = sponge.squeeze();

        // η 向量: 特征维度 RLC 压缩 (d 个标量)
        transcript.eta = (0..self.dims.d).map(|_| sponge.squeeze()).collect();

        // R1CS 约束折叠标量 α
        transcript.alpha = sponge.squeeze();

        // 多维随机评估点
        // 时间维度
        transcript.r_z = (0..self.dims.n_k).map(|_| sponge.squeeze()).collect();

        // 空间维度
        transcript.r_x = (0..self.dims.n_v).map(|_| sponge.squeeze()).collect();

        // 特征维度
        transcript.r_y = (0..self.dims.n_d).map(|_| sponge.squeeze()).collect();

        // 权重维度 (隐藏层)
        transcript.r_w = (0..self.dims.n_h).map(|_| sponge.squeeze()).collect();

        // 双轨批量折叠标量
        transcript.beta_graph = sponge.squeeze();
        transcript.beta_weight = sponge.squeeze();

        // R1CS 盲化折叠标量
        transcript.r_blind = sponge.squeeze();

        // IPA 正交混淆基点 (横坐标 + 曲线映射)
        transcript.q_ipa_x = sponge.squeeze();

        transcript
    }
}

// ============================================================================
// SPARK 内存元组的代数压缩与排列校验 (Phase 2.2.2)
// ============================================================================

/// 对高维特征向量执行 RLC 标量化坍缩
///
/// val_compressed = Σ_{j=0}^{d-1} η_j · val[j]
///
/// # 参数
/// - val: 高维特征向量 (长度为 d)
/// - eta: 随机 RLC 向量 (长度为 d)
pub fn compress_feature_rlc(val: &[Fp], eta: &[Fp]) -> Fp {
    assert_eq!(
        val.len(),
        eta.len(),
        "Feature and eta dimensions must match"
    );
    let mut result = Fp::zero();
    for (v, e) in val.iter().zip(eta.iter()) {
        result += *v * *e;
    }
    result
}

/// 生成 E(t) 代数压缩多项式
///
/// E(t) = addr + δ₁ · val_compressed + δ₂ · ts − τ
pub fn compute_compression_poly(
    addr: u64,
    val_compressed: Fp,
    ts: u64,
    delta_1: Fp,
    delta_2: Fp,
    tau: Fp,
) -> Fp {
    Fp::from(addr) + delta_1 * val_compressed + delta_2 * Fp::from(ts) - tau
}

/// 执行全局连乘排列校验
///
/// Π_{t ∈ T_chron} E(t) == Π_{t ∈ T_sorted} E(t)
pub fn grand_product_permutation_check(e_chronological: &[Fp], e_sorted: &[Fp]) -> bool {
    if e_chronological.len() != e_sorted.len() {
        return false;
    }

    let mut prod_chron = Fp::one();
    let mut prod_sorted = Fp::one();

    for &e in e_chronological {
        prod_chron *= e;
    }
    for &e in e_sorted {
        prod_sorted *= e;
    }

    prod_chron == prod_sorted
}

/// 生成时序排序轨迹
///
/// 按照"地址优先，时间戳其次"的规则对操作元组排序。
///
/// # 参数
/// - ops: 操作元组列表 (addr, val_compressed, ts)
///
/// # 返回
/// - 排序后的操作元组列表
pub fn sort_by_addr_and_ts(ops: &[(u64, Fp, u64)]) -> Vec<(u64, Fp, u64)> {
    let mut sorted = ops.to_vec();
    sorted.sort_by(|a, b| {
        a.0.cmp(&b.0) // 首先按地址排序
            .then_with(|| a.2.cmp(&b.2)) // 其次按时间戳排序
    });
    sorted
}

/// 验证排序轨迹的时序因果律
///
/// - 若 addr_{m+1} == addr_m:
///   - Read-to-Read: ts_{m+1} >= ts_m && val_{m+1} == val_m
///   - Write-to-Read/Write-to-Write: ts_{m+1} > ts_m
pub fn verify_causal_monotonicity(sorted_ops: &[(u64, Fp, u64)]) -> bool {
    for window in sorted_ops.windows(2) {
        let (a1, v1, t1) = window[0];
        let (a2, v2, t2) = window[1];

        // 地址连续性
        if a2 < a1 {
            return false;
        }

        // 同一地址的时序因果律
        if a2 == a1 {
            if t2 < t1 {
                return false;
            }
            // 若时间戳不变，值必须一致 (静态读取)
            if t2 == t1 && v1 != v2 {
                return false;
            }
        }
    }
    true
}

// ============================================================================
// 非线性辅助见证承诺 (Phase 2.2.4)
// ============================================================================

/// 为符号位矩阵生成 Hyrax 行承诺
///
/// 对应文档 2.2.4 节第三步和第四步。
///
/// # 参数
/// - s_sign: 符号位矩阵 (N_max × H_max, row-major)
/// - n_max: 物理行数
/// - h_max: 物理列数
/// - n_real: 有效行边界
/// - h_eff: 有效列边界 (h 或 h_pre)
/// - gens: 公共生成元阵列 (长度 H_max)
/// - h_blind: 盲化基点
///
/// # 返回
/// - 按行承诺向量 C_i
pub fn commit_sign_matrix(
    hyrax: &HyraxCommitment,
    s_sign: &[Fp],
    n_max: usize,
    h_max: usize,
    n_real: usize,
    h_eff: usize,
    _gens: &[Fp], // 公共生成元 (在椭圆曲线群 G 上)
    _h_blind: &Fp,
) -> Vec<(Fp, Fp)> {
    let mut commits = Vec::with_capacity(n_max);

    for i in 0..n_max {
        let row_start = i * h_max;

        if i < n_real {
            // 分支 1: 真实节点的符号位承诺
            // C_i = r_i · H + Σ_{j=0}^{h_eff-1} S_sign[i,j] · G_j
            let point = hyrax.commit_sparse_row(&s_sign[row_start..row_start + h_eff], h_eff);
            commits.push(point);
        } else {
            // 分支 2: 虚拟节点的符号位 (仅盲化)
            // C_i = r_i · H
            let point = hyrax.commit_blinded_row();
            commits.push(point);
        }
    }

    commits
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_feature_rlc() {
        let val: Vec<Fp> = vec![Fp::from(1u64), Fp::from(2u64), Fp::from(3u64)];
        let eta: Vec<Fp> = vec![Fp::from(2u64), Fp::from(3u64), Fp::from(5u64)];
        let compressed = compress_feature_rlc(&val, &eta);
        // 1*2 + 2*3 + 3*5 = 2 + 6 + 15 = 23
        assert_eq!(compressed, Fp::from(23u64));
    }

    #[test]
    fn test_compute_compression_poly() {
        let addr = 5u64;
        let val = Fp::from(10u64);
        let ts = 42u64;
        let d1 = Fp::from(2u64);
        let d2 = Fp::from(3u64);
        let tau = Fp::from(100u64);

        let e = compute_compression_poly(addr, val, ts, d1, d2, tau);
        // E = 5 + 2*10 + 3*42 - 100 = 5 + 20 + 126 - 100 = 51
        assert_eq!(e, Fp::from(51u64));
    }

    #[test]
    fn test_grand_product_permutation_check() {
        let e_chron: Vec<Fp> = vec![Fp::from(2u64), Fp::from(3u64), Fp::from(5u64)];
        let e_sorted: Vec<Fp> = vec![Fp::from(5u64), Fp::from(2u64), Fp::from(3u64)];
        // 乘积: 2*3*5 = 30 ; 5*2*3 = 30
        assert!(grand_product_permutation_check(&e_chron, &e_sorted));

        // 错误情况：乘积不匹配
        let e_bad: Vec<Fp> = vec![
            Fp::from(5u64),
            Fp::from(2u64),
            Fp::from(4u64), // 40 ≠ 30
        ];
        assert!(!grand_product_permutation_check(&e_chron, &e_bad));
    }

    #[test]
    fn test_sort_by_addr_and_ts() {
        let ops = vec![
            (2u64, Fp::from(10u64), 100u64),
            (1u64, Fp::from(20u64), 50u64),
            (1u64, Fp::from(30u64), 10u64),
            (2u64, Fp::from(40u64), 90u64),
        ];
        let sorted = sort_by_addr_and_ts(&ops);
        // 期望顺序: (1,10), (1,50), (2,90), (2,100)
        assert_eq!(sorted[0], (1u64, Fp::from(30u64), 10u64));
        assert_eq!(sorted[1], (1u64, Fp::from(20u64), 50u64));
        assert_eq!(sorted[2], (2u64, Fp::from(40u64), 90u64));
        assert_eq!(sorted[3], (2u64, Fp::from(10u64), 100u64));
    }

    #[test]
    fn test_verify_causal_monotonicity_valid() {
        let ops = vec![
            (1u64, Fp::from(10u64), 100u64),
            (1u64, Fp::from(10u64), 100u64), // 同地址同值同时间戳 → 合法 (并发读)
            (2u64, Fp::from(20u64), 200u64),
        ];
        assert!(verify_causal_monotonicity(&ops));
    }

    #[test]
    fn test_verify_causal_monotonicity_invalid_ts() {
        // 同一地址，时间戳递减 → 非法
        let ops = vec![
            (1u64, Fp::from(10u64), 200u64),
            (1u64, Fp::from(10u64), 100u64), // 时间戳回退
        ];
        assert!(!verify_causal_monotonicity(&ops));
    }

    #[test]
    fn test_verify_causal_monotonicity_invalid_addr() {
        // 地址递减 → 非法
        let ops = vec![
            (2u64, Fp::from(10u64), 100u64),
            (1u64, Fp::from(10u64), 200u64),
        ];
        assert!(!verify_causal_monotonicity(&ops));
    }
}
