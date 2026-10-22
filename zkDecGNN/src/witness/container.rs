//! 物理容器初始化与静态掩码对齐 (Physical Container Padding & Mask Alignment)
//!
//! 实现了《Spartan-ZENO》2.1.1 节：
//! - 将真实节点数据嵌入超级容器左上角，剩余区域填 0
//! - 生成静态对角掩码矩阵 M，区分有效数据区与 Padding 区
//! - 对私密权重矩阵执行向上取整到 2 的幂次方 (Power-of-2 Padding)
//! - 构建全 1 向量（用于偏置广播）和极大负数矩阵 C（用于 Logits 掩码）
//!
//! 设计方案使用了兼容四大模型(SGC, SSGC, SIGN, APPNP)的
//! 尺寸锁死的超级容器，大小为 N_max × D_max。

use super::ContainerDimensions;
use crate::curve_config::Fr;
use ark_std::{One, Zero};
use std::collections::HashMap;

// ============================================================
// 结构体定义
// ============================================================

/// 物理容器，存储所有补齐后的矩阵
#[derive(Debug, Clone)]
pub struct PhysicalContainer {
    /// 补齐后的初始节点特征矩阵 X_padded (N_max × D_max)
    pub x_padded: Vec<Vec<Fr>>,

    /// 补齐后的混合特征矩阵 X_base (N_max × D_max)
    /// 初始为 None，由 Stage 1 前置变换填充
    pub x_base: Option<Vec<Vec<Fr>>>,

    /// 静态对角掩码矩阵 M (N_max × N_max)，1 表示有效节点，0 表示虚拟节点
    pub mask_diag: Vec<Fr>,

    /// 补齐后的私密权重矩阵集合
    /// key: 权重名称 (如 "W1", "W2", "W_pre_0", "W_pre_1", "Theta")
    /// value: 对应的补齐矩阵 (rows × cols)
    pub weights: HashMap<String, WeightMatrix>,

    /// 全 1 向量 (N_max × 1)，用于偏置项广播
    pub ones_vector: Vec<Fr>,

    /// 极大负数矩阵 C (N_max × D_max)，用于 Logits 掩码替换
    pub large_negatives: Vec<Vec<Fr>>,

    /// 容器维度配置
    pub dims: ContainerDimensions,
}

/// 补齐后的私密权重矩阵
#[derive(Debug, Clone)]
pub struct WeightMatrix {
    /// 权重数据 (rows × cols)，已补齐至 2 的幂次方
    pub data: Vec<Vec<Fr>>,
    /// 补齐前的原始行数
    pub original_rows: usize,
    /// 补齐前的原始列数
    pub original_cols: usize,
    /// 补齐后的行数 (2^{ceil(log2(original_rows))})
    pub padded_rows: usize,
    /// 补齐后的列数 (2^{ceil(log2(original_cols))})
    pub padded_cols: usize,
    /// 布尔行维度 n_r
    pub n_r: usize,
    /// 布尔列维度 n_c
    pub n_c: usize,
}

/// 静态掩码配置
#[derive(Debug, Clone)]
pub struct MaskConfig {
    /// 对角掩码向量 M[i] = 1 if i < N_real else 0
    pub diag: Vec<Fr>,
    /// 掩码补集 (I - M) 的对角线元素
    pub complement_diag: Vec<Fr>,
}

// ============================================================
// 核心函数实现
// ============================================================

/// 初始化物理容器：将真实特征数据嵌入超级容器并生成掩码
///
/// # 参数
/// - `x_real`: 真实节点特征矩阵 (N_real × d)，元素为 Fr 标量
/// - `dims`: 容器维度配置
///
/// # 返回
/// - `PhysicalContainer`: 包含所有补齐后数据的容器
pub fn initialize_container(x_real: &[Vec<Fr>], dims: &ContainerDimensions) -> PhysicalContainer {
    let n_real = dims.n_real;
    let d = dims.d;
    let n_max = dims.n_max;
    let d_max = dims.d_max;

    // ---- 第 1 步：初始化 X_padded 超级特征矩阵 ----
    // 文档规范：将真实数据放置到超级容器的左上角，剩余填充 0
    let mut x_padded = vec![vec![Fr::zero(); d_max]; n_max];
    for i in 0..n_real {
        for j in 0..d {
            if i < x_real.len() && j < x_real[i].len() {
                x_padded[i][j] = x_real[i][j];
            }
        }
    }

    // ---- 第 2 步：生成静态对角掩码矩阵 M ----
    // 文档规范：M 是 N_max × N_max 的对角矩阵，
    // 前 N_real 个对角线元素为 1，后续为 0
    // 存储为长度为 N_max 的向量（仅对角线非零）
    let mut mask_diag = vec![Fr::zero(); n_max];
    for entry in mask_diag.iter_mut().take(n_real) {
        *entry = Fr::one();
    }

    // ---- 第 3 步：初始化全 1 向量 ----
    // 用于偏置项 b_1, b_2 的广播加法
    let ones_vector: Vec<Fr> = (0..n_max)
        .map(|i| if i < n_real { Fr::one() } else { Fr::zero() })
        .collect();

    // ---- 第 4 步：生成极大负数矩阵 C ----
    // 文档规范：C 的所有元素设为域内允许的极大负数
    // 在有限域中，我们用 -10^4 映射为 p - 10000
    let large_neg_val = Fr::from(-10000i64);
    let large_negatives = vec![vec![large_neg_val; d_max]; n_max];

    PhysicalContainer {
        x_padded,
        x_base: None,
        mask_diag,
        weights: HashMap::new(),
        ones_vector,
        large_negatives,
        dims: dims.clone(),
    }
}

/// 补齐私密权重矩阵至 2 的幂次方并注册到容器
///
/// # 参数
/// - `container`: 可变引用到物理容器
/// - `name`: 权重名称
/// - `w_raw`: 原始权重矩阵 (rows × cols)
pub fn pad_and_register_weight(container: &mut PhysicalContainer, name: &str, w_raw: &[Vec<Fr>]) {
    let original_rows = w_raw.len();
    let original_cols = if original_rows > 0 { w_raw[0].len() } else { 0 };

    // 向上取整到 2 的幂次方
    let padded_rows = original_rows.next_power_of_two();
    let padded_cols = original_cols.next_power_of_two();
    let n_r = padded_rows.trailing_zeros() as usize;
    let n_c = padded_cols.trailing_zeros() as usize;

    // 创建补齐后的矩阵
    let mut data = vec![vec![Fr::zero(); padded_cols]; padded_rows];
    for i in 0..original_rows {
        for j in 0..original_cols {
            data[i][j] = w_raw[i][j];
        }
    }

    let weight_matrix = WeightMatrix {
        data,
        original_rows,
        original_cols,
        padded_rows,
        padded_cols,
        n_r,
        n_c,
    };

    container.weights.insert(name.to_string(), weight_matrix);
}

/// 获取 X_padded 中真实的节点特征 (去除 Padding)
pub fn extract_real_features(container: &PhysicalContainer) -> Vec<Vec<Fr>> {
    let n_real = container.dims.n_real;
    let d = container.dims.d;
    let mut real = vec![vec![Fr::zero(); d]; n_real];
    for (i, row) in real.iter_mut().enumerate().take(n_real) {
        for (j, value) in row.iter_mut().enumerate().take(d) {
            *value = container.x_padded[i][j];
        }
    }
    real
}

/// 应用掩码提取有效区域（用于 Logits 掩码替换）
///
/// 文档公式：Y_tilde = M · Y + (I - M) · C
/// 其中 M 是对角掩码，I 是单位矩阵，C 是极大负数矩阵
pub fn apply_logit_mask(
    logits: &[Vec<Fr>],
    mask_diag: &[Fr],
    large_negatives: &[Vec<Fr>],
) -> Vec<Vec<Fr>> {
    let n_max = logits.len();
    if n_max == 0 {
        return vec![];
    }
    let d_max = logits[0].len();

    let mut masked = vec![vec![Fr::zero(); d_max]; n_max];
    for i in 0..n_max {
        let m_i = mask_diag[i]; // M[i,i]
        let c_m_i = Fr::one() - m_i; // (I-M)[i,i]

        for j in 0..d_max {
            // Y_tilde[i,j] = M[i,i] * Y[i,j] + (1-M[i,i]) * C[i,j]
            masked[i][j] = m_i * logits[i][j] + c_m_i * large_negatives[i][j];
        }
    }
    masked
}

/// 提取掩码配置（用于传输到客户端）
pub fn build_mask_config(dims: &ContainerDimensions) -> MaskConfig {
    let mut diag = vec![Fr::zero(); dims.n_max];
    let mut complement_diag = vec![Fr::one(); dims.n_max];

    for i in 0..dims.n_real {
        diag[i] = Fr::one();
        complement_diag[i] = Fr::zero();
    }

    MaskConfig {
        diag,
        complement_diag,
    }
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dims() -> ContainerDimensions {
        ContainerDimensions::from_real_dimensions(
            10, // n_real
            8,  // d
            16, // h
            7,  // c
            50, // edges
            2,  // k
        )
    }

    #[test]
    fn test_container_initialization() {
        let dims = make_dims();
        // 创建 10×8 的特征矩阵
        let x_real: Vec<Vec<Fr>> = (0..10)
            .map(|i| (0..8).map(|j| Fr::from((i * 8 + j) as u64)).collect())
            .collect();

        let container = initialize_container(&x_real, &dims);

        // 验证 X_padded 维度
        assert_eq!(container.x_padded.len(), dims.n_max); // 16 (2^4)
        assert_eq!(container.x_padded[0].len(), dims.d_max); // 8 (2^3)

        // 验证左上角数据正确嵌入
        for i in 0..10 {
            for j in 0..8 {
                assert_eq!(
                    container.x_padded[i][j],
                    Fr::from((i * 8 + j) as u64),
                    "特征 X_padded[{},{}] 不匹配",
                    i,
                    j
                );
            }
        }

        // 验证 Padding 区全为 0
        for i in 10..16 {
            for j in 0..8 {
                assert_eq!(container.x_padded[i][j], Fr::zero());
            }
        }

        // 验证掩码
        for i in 0..10 {
            assert_eq!(container.mask_diag[i], Fr::one());
        }
        for i in 10..16 {
            assert_eq!(container.mask_diag[i], Fr::zero());
        }
    }

    #[test]
    fn test_weight_padding() {
        let dims = make_dims();
        let x_real: Vec<Vec<Fr>> = (0..10)
            .map(|_| (0..8).map(|_| Fr::one()).collect())
            .collect();
        let mut container = initialize_container(&x_real, &dims);

        // 测试权重补齐：原始 10×8 的矩阵
        let w_raw: Vec<Vec<Fr>> = (0..10)
            .map(|i| (0..8).map(|j| Fr::from((i * 8 + j) as u64)).collect())
            .collect();

        pad_and_register_weight(&mut container, "W1", &w_raw);

        let w = container.weights.get("W1").expect("权重注册失败");
        assert_eq!(w.original_rows, 10);
        assert_eq!(w.original_cols, 8);
        // 10 向上取整到 16, 8 向上取整到 8
        assert_eq!(w.padded_rows, 16);
        assert_eq!(w.padded_cols, 8);

        // 验证原始数据正确复制
        for i in 0..10 {
            for j in 0..8 {
                assert_eq!(w.data[i][j], Fr::from((i * 8 + j) as u64));
            }
        }
        // 验证 padding 行全为 0
        for i in 10..16 {
            for j in 0..8 {
                assert_eq!(w.data[i][j], Fr::zero());
            }
        }
    }

    #[test]
    fn test_logit_masking() {
        let dims = make_dims();
        let n_max = dims.n_max;
        let d_max = dims.d_max;

        // 创建 logits（假设 c=7, D_max=8）
        let logits: Vec<Vec<Fr>> = (0..n_max)
            .map(|i| {
                (0..d_max)
                    .map(|j| Fr::from((i * d_max + j) as u64))
                    .collect()
            })
            .collect();

        let mask_config = build_mask_config(&dims);
        let large_neg_val = Fr::from(-10000i64);
        let large_negatives = vec![vec![large_neg_val; d_max]; n_max];

        let masked = apply_logit_mask(&logits, &mask_config.diag, &large_negatives);

        // 前 N_real 个节点应保持原值
        for (i, row) in masked.iter().enumerate().take(10) {
            for (j, value) in row.iter().enumerate().take(d_max) {
                assert_eq!(*value, Fr::from((i * d_max + j) as u64));
            }
        }

        // 虚拟节点应被替换为极大负数
        for row in masked.iter().take(n_max).skip(10) {
            for value in row.iter().take(d_max) {
                assert_eq!(*value, large_neg_val);
            }
        }
    }

    #[test]
    fn test_ones_vector() {
        let dims = make_dims();
        let x_real: Vec<Vec<Fr>> = (0..10)
            .map(|_| (0..8).map(|_| Fr::one()).collect())
            .collect();
        let container = initialize_container(&x_real, &dims);

        for i in 0..10 {
            assert_eq!(container.ones_vector[i], Fr::one());
        }
        for i in 10..16 {
            assert_eq!(container.ones_vector[i], Fr::zero());
        }
    }

    #[test]
    fn test_dimensions_power_of_two() {
        let d1 = ContainerDimensions::from_real_dimensions(7, 5, 3, 3, 20, 2);
        assert_eq!(d1.n_max, 8);
        assert_eq!(d1.d_max, 8);
        assert_eq!(d1.h_max, 4);

        let d2 = ContainerDimensions::from_real_dimensions(8, 8, 8, 3, 50, 1);
        assert_eq!(d2.n_max, 8);
        assert_eq!(d2.d_max, 8);
        assert_eq!(d2.h_max, 8);

        // k=2: 需要 ceil(log2(2)) = 1
        let d3 = ContainerDimensions::from_real_dimensions(10, 8, 16, 3, 50, 3);
        // k=3 -> next_power_of_two=4 -> trailing_zeros=2
        assert_eq!(d3.n_k, 2);
    }
}
