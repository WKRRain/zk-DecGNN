//! 全局随机转录本 (Transcript Data Structure)
//!
//! 严格遵循 Spartan-ZENO 设计文档第 2.3 节第三大模块。
//!
//! ## 核心定义
//!
//! 非交互式零知识证明 (NIZK) 中，证明者必须将通过 Fiat-Shamir 启发式
//! (从 Poseidon 海绵中挤出) 的所有"天意随机数"冻结在一个不可变的
//! 常量字典 Γ 中。
//!
//! ## 转录本字典 Γ 的三个子集
//!
//! 1. Γ_SPARK: 离线内存校验压缩常量 (δ₁, δ₂, τ)
//! 2. Γ_RLC: R1CS 约束折叠标量 (α)
//! 3. Γ_batch: 双轨批量折叠标量 (β_graph, β_weight)
//! 4. Γ_blind: 零知识盲化折叠标量 (r_blind)
//! 5. Γ_SC: Sum-check 多维随机评估点向量
//!    - r_z: 时间轴 (迭代轮次)
//!    - r_x: 空间轴 (行/物理节点)
//!    - r_y: 特征轴 (列/通道)
//!    - r_w: 模型权重维度 (隐藏层列轴)
//!
//! ## 最终类型
//!
//! Γ = ({δ₁, δ₂, τ}, α, β_graph, β_weight, r_blind, r_z, r_x, r_y, r_w)

use crate::curve_config::Fp;
use ark_ff::Zero;
use std::fmt;

// ============================================================================
// 全局转录本数据结构
// ============================================================================

/// 全局随机转录本 Γ
///
/// 包含所有通过 Fiat-Shamir 变换从 Poseidon 海绵中挤出的挑战标量。
/// 该结构一旦构建完成即不可修改，确保整个 NIZK 协议的确定性。
#[derive(Debug, Clone)]
pub struct Transcript {
    // === Γ_SPARK: SPARK 离线内存校验压缩常量 ===
    /// δ₁: 绑定特征值维度的随机压缩标量
    pub delta_1: Fp,

    /// δ₂: 绑定时间戳防重放的随机压缩标量
    pub delta_2: Fp,

    /// τ: 离散元组转化为多项式求根验证的全局位移标量
    pub tau: Fp,

    /// η: 特征维度 RLC 压缩向量 (长度为 d)
    pub eta: Vec<Fp>,

    // === Γ_RLC: R1CS 约束折叠标量 ===
    /// α: 触发 Schwartz-Zippel 引理的 R1CS 折叠标量
    pub alpha: Fp,

    // === Γ_SC: Sum-check 多维随机评估点向量 ===
    /// r_z: 时间轴评估点 (长度 n_k = ceil(log2(K)))
    pub r_z: Vec<Fp>,

    /// r_x: 空间轴评估点 (长度 n_v = log2(N_max))
    pub r_x: Vec<Fp>,

    /// r_y: 特征轴评估点 (长度 n_d = log2(D_max))
    pub r_y: Vec<Fp>,

    /// r_w: 模型权重维度评估点 (长度 n_h = log2(H_max))
    pub r_w: Vec<Fp>,

    // === Γ_batch: 双轨批量折叠标量 ===
    /// β_graph: 图特征轨批量折叠权重
    pub beta_graph: Fp,

    /// β_weight: 模型权重轨批量折叠权重
    pub beta_weight: Fp,

    // === Γ_blind: 零知识盲化折叠标量 ===
    /// r_blind: 零知识盲化多项式 R(x,y) 的注入权重
    pub r_blind: Fp,

    // === IPA 正交混淆基点 ===
    /// Q_IPA 的横坐标 (用于构造复合生成元)
    pub q_ipa_x: Fp,
}

impl Transcript {
    /// 创建空的转录本 (所有字段初始化为 0)
    pub fn new() -> Self {
        Self {
            delta_1: Fp::zero(),
            delta_2: Fp::zero(),
            tau: Fp::zero(),
            eta: Vec::new(),
            alpha: Fp::zero(),
            r_z: Vec::new(),
            r_x: Vec::new(),
            r_y: Vec::new(),
            r_w: Vec::new(),
            beta_graph: Fp::zero(),
            beta_weight: Fp::zero(),
            r_blind: Fp::zero(),
            q_ipa_x: Fp::zero(),
        }
    }

    /// 获取 SPARK 子集
    pub fn spark_subset(&self) -> (Fp, Fp, Fp, &[Fp]) {
        (self.delta_1, self.delta_2, self.tau, &self.eta)
    }

    /// 获取 R1CS 折叠子集
    pub fn r1cs_subset(&self) -> Fp {
        self.alpha
    }

    /// 获取双轨批量子集
    pub fn batch_subset(&self) -> (Fp, Fp) {
        (self.beta_graph, self.beta_weight)
    }

    /// 获取盲化子集
    pub fn blind_subset(&self) -> Fp {
        self.r_blind
    }

    /// 获取 Sum-check 评估点子集
    pub fn sumcheck_subset(&self) -> (&[Fp], &[Fp], &[Fp], &[Fp]) {
        (&self.r_z, &self.r_x, &self.r_y, &self.r_w)
    }

    /// 序列化转录本的标量指纹 (用于调试/日志)
    pub fn fingerprint(&self) -> Vec<Fp> {
        let mut fp = Vec::new();
        fp.push(self.delta_1);
        fp.push(self.delta_2);
        fp.push(self.tau);
        fp.extend(&self.eta);
        fp.push(self.alpha);
        fp.extend(&self.r_z);
        fp.extend(&self.r_x);
        fp.extend(&self.r_y);
        fp.extend(&self.r_w);
        fp.push(self.beta_graph);
        fp.push(self.beta_weight);
        fp.push(self.r_blind);
        fp.push(self.q_ipa_x);
        fp
    }
}

impl Default for Transcript {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Transcript {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Transcript(delta_1={:?}, delta_2={:?}, tau={:?}, alpha={:?}, \
             |r_z|={}, |r_x|={}, |r_y|={}, |r_w|={}, \
             beta_graph={:?}, beta_weight={:?}, r_blind={:?})",
            self.delta_1,
            self.delta_2,
            self.tau,
            self.alpha,
            self.r_z.len(),
            self.r_x.len(),
            self.r_y.len(),
            self.r_w.len(),
            self.beta_graph,
            self.beta_weight,
            self.r_blind,
        )
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcript_new_zeros() {
        let t = Transcript::new();
        assert!(t.delta_1.is_zero());
        assert!(t.delta_2.is_zero());
        assert!(t.tau.is_zero());
        assert!(t.alpha.is_zero());
    }

    #[test]
    fn test_transcript_subsets() {
        let mut t = Transcript::new();
        t.delta_1 = Fp::from(1u64);
        t.delta_2 = Fp::from(2u64);
        t.tau = Fp::from(3u64);
        t.beta_graph = Fp::from(10u64);
        t.beta_weight = Fp::from(20u64);
        t.r_blind = Fp::from(30u64);

        let (d1, d2, tau, _eta) = t.spark_subset();
        assert_eq!(d1, Fp::from(1u64));
        assert_eq!(d2, Fp::from(2u64));
        assert_eq!(tau, Fp::from(3u64));

        let (bg, bw) = t.batch_subset();
        assert_eq!(bg, Fp::from(10u64));
        assert_eq!(bw, Fp::from(20u64));

        assert_eq!(t.blind_subset(), Fp::from(30u64));
    }

    #[test]
    fn test_transcript_fingerprint() {
        let mut t = Transcript::new();
        t.delta_1 = Fp::from(1u64);
        t.delta_2 = Fp::from(2u64);
        t.tau = Fp::from(3u64);
        // 至少一个元素以避免空指纹
        t.eta = vec![Fp::from(4u64)];

        let fp = t.fingerprint();
        assert!(!fp.is_empty());
    }

    #[test]
    fn test_transcript_display() {
        let t = Transcript::new();
        let s = format!("{}", t);
        assert!(s.contains("Transcript"));
    }
}
