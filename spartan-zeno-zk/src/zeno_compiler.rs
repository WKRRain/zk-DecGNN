//! ZENO 静态类型分析与编译器
//!
//! 对计算图进行静态扫描，标注变量类型（Public/Private），
//! 并生成静态多轨验证密钥（Static Multi-Track VK）所需的架构指纹。

use crate::curve_config::Fp;
pub use crate::model_registry::ModelArchitecture;
use crate::poseidon::PoseidonSponge;
use ark_std::{One, Zero};

/// 变量可见性类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarType {
    /// 公开变量（如 X, A, 掩码 M, 超参数等）
    Public,
    /// 私密变量（如权重矩阵 W, 偏置 b 等）
    Private,
}

/// ZENO 编译器静态分析结果
#[derive(Debug, Clone)]
pub struct ZenoAnalysis {
    /// 模型架构指纹
    pub model_id: Fp,
    /// 真实节点数
    pub n_real: usize,
    /// 真实特征维度
    pub d: usize,
    /// 真实隐藏层维度
    pub h: usize,
    /// 分类数
    pub c: usize,
    /// 图扩散层数 K
    pub k_layers: usize,
    /// 前置变换开关 α_pre（0 为直通，1 为激活）
    pub alpha_pre: Fp,
    /// 后置变换开关 β
    pub beta: Fp,
    /// 聚合权重系数
    pub gamma_1: Fp,
    pub gamma_2: Fp,
    pub gamma_3: Fp,
    /// 扩散权重系数
    pub lambda_1: Fp,
    pub lambda_2: Fp,
    /// 聚合操作符类型（仅用于元数据记录，电路连线由 VK 决定）
    pub op_type: AggregationOp,
}

/// 聚合操作符类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationOp {
    /// 加法聚合（SGC / SSGC / APPNP）
    Add,
    /// 拼接聚合（SIGN）
    Concat,
}

/// 旧版 witness 上下文保存的静态类型信息。
pub type StaticTypeInfo = ZenoAnalysis;

impl ZenoAnalysis {
    /// 创建新的 ZENO 分析结果
    pub fn new(
        n_real: usize,
        d: usize,
        h: usize,
        c: usize,
        k_layers: usize,
        alpha_pre: Fp,
        beta: Fp,
        gamma_1: Fp,
        gamma_2: Fp,
        gamma_3: Fp,
        lambda_1: Fp,
        lambda_2: Fp,
        op_type: AggregationOp,
    ) -> Self {
        ZenoAnalysis {
            model_id: Fp::zero(),
            n_real,
            d,
            h,
            c,
            k_layers,
            alpha_pre,
            beta,
            gamma_1,
            gamma_2,
            gamma_3,
            lambda_1,
            lambda_2,
            op_type,
        }
    }

    /// 生成模型架构指纹
    ///
    /// 将 ZENO 分析结果中的关键参数吸入 Poseidon 海绵，
    /// 输出唯一的架构指纹标量 ID_model。
    pub fn compute_model_id(&mut self, sponge: &mut PoseidonSponge) -> Fp {
        // 吸入所有架构相关参数
        sponge.absorb_scalar(&Fp::from(self.n_real as u64));
        sponge.absorb_scalar(&Fp::from(self.d as u64));
        sponge.absorb_scalar(&Fp::from(self.h as u64));
        sponge.absorb_scalar(&Fp::from(self.c as u64));
        sponge.absorb_scalar(&Fp::from(self.k_layers as u64));
        sponge.absorb_scalar(&self.alpha_pre);
        sponge.absorb_scalar(&self.beta);
        sponge.absorb_scalar(&self.gamma_1);
        sponge.absorb_scalar(&self.gamma_2);
        sponge.absorb_scalar(&self.gamma_3);
        sponge.absorb_scalar(&self.lambda_1);
        sponge.absorb_scalar(&self.lambda_2);

        // 吸入聚合操作符类型
        let op_val = match self.op_type {
            AggregationOp::Add => Fp::zero(),
            AggregationOp::Concat => Fp::one(),
        };
        sponge.absorb_scalar(&op_val);

        self.model_id = sponge.squeeze();
        self.model_id
    }

    /// 获取模型 ID
    pub fn model_id(&self) -> Fp {
        self.model_id
    }

    /// 判断是否为 APPNP 模型（有前置变换）
    pub fn has_pre_transform(&self) -> bool {
        !self.alpha_pre.is_zero()
    }

    /// 判断是否为 SIGN 模型（拼接聚合）
    pub fn is_sign_model(&self) -> bool {
        matches!(self.op_type, AggregationOp::Concat)
    }

    /// 获取有效列边界 D_effective
    ///
    /// 根据模型类型动态确定有效列维度：
    /// - APPNP: D_effective = c（分类数，因为前置 MLP 改变了特征维度）
    /// - SGC / SSGC / SIGN: D_effective = d（原始特征维度）
    pub fn effective_dim(&self) -> usize {
        if self.has_pre_transform() {
            self.c
        } else {
            self.d
        }
    }
}

/// 变量类型标注器
///
/// 对计算图中的变量进行静态类型分析，
/// 区分 Public 和 Private 变量。
pub struct TypeAnnotator;

impl TypeAnnotator {
    /// 判断权重矩阵是否为私密变量
    pub fn is_private_weight(_name: &str) -> bool {
        // 所有权重矩阵和偏置均为私密
        true
    }

    /// 判断变量是否为公开变量
    pub fn is_public(name: &str) -> bool {
        matches!(
            name,
            "X" | "A"
                | "M"
                | "I"
                | "C"
                | "gamma_1"
                | "gamma_2"
                | "gamma_3"
                | "lambda_1"
                | "lambda_2"
                | "alpha_pre"
                | "beta"
        )
    }

    /// 标注变量类型
    pub fn annotate(name: &str) -> VarType {
        if Self::is_public(name) {
            VarType::Public
        } else {
            VarType::Private
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::Zero;

    #[test]
    fn test_zeno_analysis_creation() {
        let analysis = ZenoAnalysis::new(
            2708,           // n_real (Cora)
            1433,           // d
            64,             // h
            7,              // c
            2,              // K
            Fp::zero(),     // alpha_pre = 0 (无前置变换)
            Fp::zero(),     // beta = 0 (无 ReLU)
            Fp::from(1u64), // gamma_1
            Fp::from(1u64), // gamma_2
            Fp::from(0u64), // gamma_3
            Fp::from(1u64), // lambda_1
            Fp::from(0u64), // lambda_2
            AggregationOp::Add,
        );

        assert_eq!(analysis.n_real, 2708);
        assert_eq!(analysis.d, 1433);
        assert!(!analysis.has_pre_transform());
        assert!(!analysis.is_sign_model());
    }

    #[test]
    fn test_effective_dim() {
        // SGC 模型：有效维度 = d
        let sgc = ZenoAnalysis::new(
            100,
            64,
            32,
            7,
            2,
            Fp::zero(),
            Fp::zero(),
            Fp::one(),
            Fp::one(),
            Fp::zero(),
            Fp::one(),
            Fp::zero(),
            AggregationOp::Add,
        );
        assert_eq!(sgc.effective_dim(), 64);

        // APPNP 模型：有效维度 = c
        let appnp = ZenoAnalysis::new(
            100,
            64,
            32,
            7,
            2,
            Fp::one(), // alpha_pre = 1
            Fp::zero(),
            Fp::one(),
            Fp::one(),
            Fp::zero(),
            Fp::one(),
            Fp::one(),
            AggregationOp::Add,
        );
        assert_eq!(appnp.effective_dim(), 7);
    }

    #[test]
    fn test_type_annotation() {
        assert_eq!(TypeAnnotator::annotate("X"), VarType::Public);
        assert_eq!(TypeAnnotator::annotate("A"), VarType::Public);
        assert_eq!(TypeAnnotator::annotate("M"), VarType::Public);
        assert_eq!(TypeAnnotator::annotate("W1"), VarType::Private);
        assert_eq!(TypeAnnotator::annotate("W2"), VarType::Private);
        assert_eq!(TypeAnnotator::annotate("b1"), VarType::Private);
    }
}
