//! Phase 2: 见证生成 (Witness Generation)
//!
//! 本模块实现了 Spartan-ZENO 方案中的确定性见证生成，
//! 包括容器初始化、Stage 1 前置变换、Stage 2 扩散循环、
//! Stage 3 后置变换与量化截断，以及见证终态组装。
//!
//! # 子模块
//!
//! - `container`: 物理容器初始化与静态掩码对齐
//! - `pre_transform`: Stage 1 前置特征变换 (含 ReLU 辅助见证)
//! - `diffusion`: Stage 2 扩散与聚合追踪循环
//! - `post_transform`: Stage 3 后置变换、量化截断与非线性激活
//! - `assembly`: 见证终态组装 (张量引擎、内存引擎、转录本固化)
//! - `interactive`: 交互式见证生成 (Fiat-Shamir)
//! - `transcript`: 全局随机转录本
//! - `memory_check`: SPARK 内存排序与排列校验
//! - `binary_tree`: 连乘二叉树与底层轨迹承诺

pub mod assembly;
pub mod binary_tree;
pub mod container;
pub mod diffusion;
pub mod interactive;
pub mod memory_check;
pub mod post_transform;
pub mod pre_transform;
pub mod transcript;

use crate::quantization::ScaleFactor;
use crate::zeno_compiler::{ModelArchitecture, StaticTypeInfo};

/// 物理容器维度配置
#[derive(Debug, Clone)]
pub struct ContainerDimensions {
    /// 超级容器节点物理上限 (2^{n_v})
    pub n_max: usize,
    /// 超级容器特征维度物理上限 (2^{n_d})
    pub d_max: usize,
    /// 隐藏层物理维度上限 (2^{n_h})
    pub h_max: usize,
    /// 分类数
    pub c: usize,
    /// 真实节点数
    pub n_real: usize,
    /// 真实特征维度
    pub d: usize,
    /// 真实隐藏层维度
    pub h: usize,
    /// 真实边数
    pub edges: usize,
    /// 迭代轮次 K
    pub k: usize,
    /// 布尔维度: n_v = log2(N_max)
    pub n_v: usize,
    /// 布尔维度: n_d = log2(D_max)
    pub n_d: usize,
    /// 布尔维度: n_h = log2(H_max)
    pub n_h: usize,
    /// 布尔维度: n_k = ceil(log2(K))
    pub n_k: usize,
}

impl ContainerDimensions {
    /// 根据实际图数据计算容器维度
    pub fn from_real_dimensions(
        n_real: usize,
        d: usize,
        h: usize,
        c: usize,
        edges: usize,
        k: usize,
    ) -> Self {
        let n_max = n_real.next_power_of_two();
        let d_max = d.next_power_of_two();
        let h_max = h.next_power_of_two();
        let n_v = n_max.trailing_zeros() as usize;
        let n_d = d_max.trailing_zeros() as usize;
        let n_h = h_max.trailing_zeros() as usize;
        let n_k = if k <= 1 { 1 } else { k.next_power_of_two() }.trailing_zeros() as usize;

        Self {
            n_max,
            d_max,
            h_max,
            c,
            n_real,
            d,
            h,
            edges,
            k,
            n_v,
            n_d,
            n_h,
            n_k,
        }
    }
}

/// 见证生成器的全局状态
#[derive(Debug, Clone)]
pub struct WitnessContext {
    /// 容器维度配置
    pub dims: ContainerDimensions,
    /// 模型架构类型
    pub architecture: ModelArchitecture,
    /// 静态类型信息
    pub type_info: StaticTypeInfo,
    /// 全局缩放因子
    pub scale_factor: ScaleFactor,
}
