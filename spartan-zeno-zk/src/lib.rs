//! Spartan-ZENO 通用图模型安全推理方案
//!
//! 基于 Curve25519 椭圆曲线与 Arkworks 密码学原语构建的零知识证明底层库。
//! 严格遵循对称均匀定点量化标准，所有非交互式证明通过 Poseidon 哈希海绵的 Fiat-Shamir 变换生成。
//!
//! # 模块结构
//!
//! - `curve_config`: Curve25519 域与椭圆曲线群配置
//! - `quantization`: 对称均匀定点量化 (INT8/INT16)
//! - `poseidon`: Poseidon 哈希海绵 (t=9, R=8, C=1)
//! - `polynomial`: 多线性扩展 (MLE) 与拉格朗日基函数
//! - `hyrax`: Hyrax 多项式承诺方案 (按行承诺 + 矩阵方阵化)
//! - `spark`: SPARK 拓扑结构承诺
//! - `model_registry`: 模型指纹固化与注册
//! - `zeno_compiler`: ZENO 静态类型分析

#![allow(clippy::too_many_arguments)]

pub mod curve_config;
pub mod graph_classification;
pub mod hyrax;
pub mod ipa;
pub mod model_registry;
pub mod polynomial;
pub mod poseidon;
pub mod quantization;
pub mod spark;
pub mod streaming;
pub mod verifier;
pub mod witness;
pub mod zeno_compiler;

// 外部 crate 复导出 (为方便上层调用)
pub use ark_ec;
pub use ark_ff;
pub use ark_poly;
pub use ark_std;

/// 全局缩放因子类型定义
pub type ScaleFactor = u16;

/// 默认全局缩放因子 (S_scale = 256.00)
pub const DEFAULT_SCALE_FACTOR: ScaleFactor = 256;

use pyo3::prelude::*;

/// Python extension entry point used by benchmark scripts.
///
/// The heavy Spartan-ZENO benchmark path stays in Python for protocol object
/// orchestration, but large MLE/streaming kernels are registered here so the
/// Reddit-scale experiments do not fall back to Python scalar loops.
#[pymodule]
fn spartan_zeno_zk(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    streaming::register_py_module(m)?;
    graph_classification::register_py_module(m)?;
    Ok(())
}
