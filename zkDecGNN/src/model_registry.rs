//! 模型注册中心与超级容器 (Model Registry & SuperContainer)
//!
//! 严格遵循 Spartan-ZENO 设计文档第 1 节。
//!
//! ## 核心组件
//! - `SuperContainer`: 硬编码固定物理维度 N_max × D_max 的超级容器
//! - 静态对角掩码 M: 精准过滤 N_real 之外区域，防止电路信息泄漏
//! - 多轨验证密钥 VK: 四套架构 (SGC, SSGC, SIGN, APPNP) 的 ID_model 指纹生成
//!
//! ## 设计约束
//! - 所有物理维度向上取整至 2 的幂次方 (Power-of-2 Padding)
//! - ID_model 由 Poseidon 海绵对架构参数序列化后挤出
//! - 四套 VK 共享完全相同的底层物理变量池 (N_max × D_max)

use crate::curve_config::{serialize_scalar, Fp};
use crate::poseidon::PoseidonSponge;
use ark_std::{One, Zero};
use std::collections::HashMap;
use std::str::FromStr;

// ============================================================================
// 模型架构枚举
// ============================================================================

/// ZENO 编译器支持的四类图模型架构
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelArchitecture {
    SGC,
    SSGC,
    SIGN,
    APPNP,
}

impl ModelArchitecture {
    /// 返回架构的字符串标识
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SGC => "SGC",
            Self::SSGC => "SSGC",
            Self::SIGN => "SIGN",
            Self::APPNP => "APPNP",
        }
    }

    /// 从字符串解析
    pub fn parse_architecture(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "SGC" => Some(Self::SGC),
            "SSGC" => Some(Self::SSGC),
            "SIGN" => Some(Self::SIGN),
            "APPNP" => Some(Self::APPNP),
            _ => None,
        }
    }
}

impl FromStr for ModelArchitecture {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse_architecture(s).ok_or(())
    }
}

// ============================================================================
// Stage 2 聚合算子类型
// ============================================================================

/// 聚合更新的物理连线方式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationOperator {
    /// 加法聚合（SGC / SSGC / APPNP 使用）
    Add,
    /// 拼接聚合（SIGN 使用，特征填入超级容器后半部）
    Concat,
}

impl AggregationOperator {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Add => "Add",
            Self::Concat => "Concat",
        }
    }
}

// ============================================================================
// SuperContainer: 硬编码固定物理维度的超级容器
// ============================================================================

/// 超级容器结构体
///
/// 硬编码固定物理维度 N_max × D_max，兼容四大模型（SGC, SSGC, SIGN, APPNP）。
/// 内部存储对齐至 2 的幂次方，并携带静态掩码 M。
///
/// # 设计文档引用
/// > 通用推理框架利用 ZENO 编译器的能力，在预处理阶段针对四个模型生成四套公开验证密钥
/// > 这四套 VK 共享完全相同的底层物理变量池 (N_max × D_max)
#[derive(Debug, Clone)]
pub struct SuperContainer {
    /// 物理节点上限 N_max（严格对齐至 2 的幂次方）
    pub n_max: usize,
    /// 物理节点上限的布尔维度 n_v = log2(N_max)
    pub n_v: usize,

    /// 物理特征维度上限 D_max（严格对齐至 2 的幂次方）
    pub d_max: usize,
    /// 物理特征维度的布尔维度 n_d = log2(D_max)
    pub n_d: usize,

    /// 物理隐藏层维度上限 H_max
    pub h_max: usize,
    /// 隐藏层布尔维度 n_h = log2(H_max)
    pub n_h: usize,

    /// 真实节点数 N_real
    pub n_real: usize,
    /// 真实特征维度 d
    pub d_real: usize,
    /// 真实隐藏层维度 h
    pub h_real: usize,
    /// 分类数 c
    pub num_classes: usize,
    /// 真实边数
    pub num_edges: usize,

    /// 静态对角掩码矩阵 M: N_max × N_max
    /// 前 N_real 个对角线元素 = 1，其余 = 0
    /// 作为 Public Instance 在运行时动态阻断无效 Padding 区的约束传播
    pub mask: Vec<Fp>,
}

impl SuperContainer {
    /// 创建超级容器，自动将维度对齐至 2 的幂次方
    ///
    /// # 参数
    /// - n_real: 真实节点数
    /// - d_real: 真实特征维度
    /// - h_real: 真实隐藏层维度
    /// - num_classes: 分类数 c
    /// - num_edges: 真实边数
    /// - n_max_hint: 期望的 N_max（将自动向上取整至 2 的幂次方）
    /// - d_max_hint: 期望的 D_max（同上）
    /// - h_max_hint: 期望的 H_max（同上）
    pub fn new(
        n_real: usize,
        d_real: usize,
        h_real: usize,
        num_classes: usize,
        num_edges: usize,
        n_max_hint: usize,
        d_max_hint: usize,
        h_max_hint: usize,
    ) -> Self {
        let n_max = next_power_of_two(n_max_hint.max(n_real));
        let d_max = next_power_of_two(d_max_hint.max(d_real).max(num_classes));
        let h_max = next_power_of_two(h_max_hint.max(h_real));

        let n_v = log2_exact(n_max);
        let n_d = log2_exact(d_max);
        let n_h = log2_exact(h_max);

        // 生成静态对角掩码 M (N_max × N_max)
        let mask = generate_static_mask(n_real, n_max);

        Self {
            n_max,
            n_v,
            d_max,
            n_d,
            h_max,
            n_h,
            n_real,
            d_real,
            h_real,
            num_classes,
            num_edges,
            mask,
        }
    }

    /// 获取容器总元素数 (N_max × D_max)
    pub fn total_capacity(&self) -> usize {
        self.n_max * self.d_max
    }

    /// 获取有效元素数 (N_real × D_effective)
    pub fn effective_capacity(&self, arch: ModelArchitecture) -> usize {
        let d_eff = self.get_effective_dimension(arch);
        self.n_real * d_eff
    }

    /// 根据模型架构确定有效列边界 D_effective
    ///
    /// - APPNP: D_effective = c (分类数)
    /// - SGC / SSGC / SIGN: D_effective = d (原始特征维度)
    pub fn get_effective_dimension(&self, arch: ModelArchitecture) -> usize {
        match arch {
            ModelArchitecture::APPNP => self.num_classes,
            _ => self.d_real,
        }
    }

    /// 验证给定模型架构的物理边界是否合法
    pub fn validate_bounds(&self, arch: ModelArchitecture) -> bool {
        let d_eff = self.get_effective_dimension(arch);
        self.n_max >= self.n_real
            && self.d_max >= d_eff
            && self.h_max >= self.h_real
            && self.n_max.is_power_of_two()
            && self.d_max.is_power_of_two()
            && self.h_max.is_power_of_two()
    }

    /// 将真实数据嵌入超级容器左上角
    ///
    /// # 参数
    /// - data: 真实数据 (row-major)，尺寸 n_real × d_in
    /// - d_in: 输入列数
    ///
    /// # 返回
    /// - 尺寸为 N_max × D_max 的补齐矩阵（row-major），Padding 区为 0
    pub fn embed_in_container(&self, data: &[Fp], d_in: usize) -> Vec<Fp> {
        assert!(
            d_in <= self.d_max,
            "d_in={} exceeds D_max={}",
            d_in,
            self.d_max
        );
        assert!(
            data.len() == self.n_real * d_in,
            "data length mismatch: expected {}, got {}",
            self.n_real * d_in,
            data.len()
        );

        let mut container = vec![Fp::zero(); self.n_max * self.d_max];
        for i in 0..self.n_real {
            let src_start = i * d_in;
            let dst_start = i * self.d_max;
            container[dst_start..dst_start + d_in]
                .copy_from_slice(&data[src_start..src_start + d_in]);
        }
        container
    }

    /// 检查给定索引 (row, col) 是否为有效节点区域
    #[inline]
    pub fn is_valid_node(&self, row: usize) -> bool {
        row < self.n_real
    }

    /// 获取掩码矩阵中位置 (row, col) 的值
    #[inline]
    pub fn mask_at(&self, row: usize, col: usize) -> Fp {
        self.mask[row * self.n_max + col]
    }
}

// ============================================================================
// 静态对角掩码生成
// ============================================================================

/// 生成静态对角掩码矩阵 M (N_max × N_max)
///
/// 严格遵循设计文档:
/// > M 是 N_max × N_max 的对角矩阵，前 N_real 个真实节点对角线元素为 1，
/// > 后续虚拟节点对角线元素为 0。
/// > ZENO 静态掩码 M 被剥离出物理布线，降维为一个由客户端在每次请求时
/// > 动态提供的公开常数向量 (Public Instance)。
///
/// # 参数
/// - n_real: 真实节点数
/// - n_max: 超级容器物理节点上限
///
/// # 返回
/// - 行优先排列的掩码矩阵 (n_max × n_max)
pub fn generate_static_mask(n_real: usize, n_max: usize) -> Vec<Fp> {
    assert!(n_max >= n_real, "N_max must be >= N_real");
    let mut mask = vec![Fp::zero(); n_max * n_max];
    for i in 0..n_real {
        mask[i * n_max + i] = Fp::one();
    }
    mask
}

/// 生成极小值替换矩阵 C (N_max × D_max)
///
/// 所有元素设为域内允许的极大负数（-10^4 经量化后），
/// 用于在 Softmax 前将 Padding 区 Logits 替换为近似 -∞。
///
/// # 参数
/// - n_max: 物理节点上限
/// - d_max: 物理特征维度上限
/// - scale: 量化缩放因子
pub fn generate_negative_infinity_matrix(n_max: usize, d_max: usize, scale: u16) -> Vec<Fp> {
    let neg_val = -Fp::from(10_000u64 * scale as u64); // -10^4 * S_scale
    vec![neg_val; n_max * d_max]
}

/// 执行对角掩码极小值替换: Y_tilde = M · Y + (I - M) · C
///
/// 此过程在 R1CS 中只是简单的线性组合，不需要二次乘法中间值，ZK 开销为零。
///
/// # 参数
/// - y: 原始 Logits 矩阵 (row-major, n_max × d_out)
/// - mask: 对角掩码矩阵 (n_max × n_max)
/// - c_mat: 极小值替换矩阵 (n_max × d_out)
/// - n_max: 物理节点上限
/// - d_out: 输出维度
pub fn apply_mask_min_value_replacement(
    y: &[Fp],
    mask: &[Fp],
    c_mat: &[Fp],
    n_max: usize,
    d_out: usize,
) -> Vec<Fp> {
    assert_eq!(y.len(), n_max * d_out);
    assert_eq!(mask.len(), n_max * n_max);
    assert_eq!(c_mat.len(), n_max * d_out);

    let mut result = vec![Fp::zero(); n_max * d_out];

    // Y_tilde = M @ Y + (I - M) @ C
    // 由于 M 是对角矩阵，可以简化为逐元素操作
    for i in 0..n_max {
        let m_ii = mask[i * n_max + i]; // M[i, i]
        let one_minus_m = Fp::one() - m_ii;
        for j in 0..d_out {
            let idx = i * d_out + j;
            // 对于对角掩码：M @ Y 的结果 = M[i,i] * Y[i,j]
            // (I-M) @ C 的结果 = (1 - M[i,i]) * C[i,j]
            result[idx] = m_ii * y[idx] + one_minus_m * c_mat[idx];
        }
    }
    result
}

// ============================================================================
// 多轨验证密钥与架构指纹
// ============================================================================

/// 验证密钥 (Verification Key) 结构体
///
/// 四套静态多轨 VK: VK_SGC, VK_SSGC, VK_SIGN, VK_APPNP。
/// 每套 VK 包含完整的架构参数与唯一指纹 ID。
#[derive(Debug, Clone)]
pub struct VerificationKey {
    /// 架构类型
    pub architecture: ModelArchitecture,
    /// 关联的超级容器引用（共享底层物理变量池）
    pub container: SuperContainer,
    /// 图扩散迭代层数 K
    pub k: usize,

    // === Stage 1 前置变换控制 ===
    /// 前置 MLP 控制开关 α_pre (0 = 直通, 1 = 激活)
    pub alpha_pre: f64,
    /// 前置隐藏层维度 h_pre (仅 APPNP)
    pub h_pre: usize,

    // === Stage 2.1 扩散更新参数 ===
    /// 邻居信息接收权重 λ₁
    pub lambda_1: f64,
    /// 基底特征接收权重 λ₂
    pub lambda_2: f64,

    // === Stage 2.2 聚合更新参数 ===
    /// 历史累加值继承权重 γ₁
    pub gamma_1: f64,
    /// 当前扩散特征衰减权重 γ₂
    pub gamma_2: f64,
    /// 基底特征比例拼接权重 γ₃
    pub gamma_3: f64,
    /// 物理连线方式
    pub operator: AggregationOperator,

    // === Stage 3 后置变换控制 ===
    /// 非线性激活开关 β (0 = 直通, 1 = ReLU)
    pub beta: f64,

    /// 唯一的架构指纹标量 ID_model ∈ Fp
    pub model_id: Fp,
}

impl VerificationKey {
    /// 创建 SGC 模型的验证密钥
    pub fn new_sgc(
        n_real: usize,
        d: usize,
        c: usize,
        k: usize,
        num_edges: usize,
        n_max_hint: usize,
    ) -> Self {
        let container = SuperContainer::new(n_real, d, c, c, num_edges, n_max_hint, d.max(c), c);
        Self::build(
            ModelArchitecture::SGC,
            container,
            k,
            0.0, // alpha_pre = 0 (短路前置变换)
            d,   // h_pre 占位
            1.0, // lambda_1
            0.0, // lambda_2
            1.0, // gamma_1
            1.0, // gamma_2
            0.0, // gamma_3
            AggregationOperator::Add,
            0.0, // beta (无后置激活)
        )
    }

    /// 创建 SSGC 模型的验证密钥
    ///
    /// 严格遵循 SSGC 论文公式:
    /// γ₂ = (1-α)/K, γ₃ = α/K
    pub fn new_ssgc(
        n_real: usize,
        d: usize,
        c: usize,
        k: usize,
        alpha: f64,
        num_edges: usize,
        n_max_hint: usize,
    ) -> Self {
        let container = SuperContainer::new(n_real, d, c, c, num_edges, n_max_hint, d.max(c), c);
        Self::build(
            ModelArchitecture::SSGC,
            container,
            k,
            0.0, // alpha_pre = 0
            d,
            1.0,                      // lambda_1
            0.0,                      // lambda_2
            1.0,                      // gamma_1
            (1.0 - alpha) / k as f64, // gamma_2 (衰减接收)
            alpha / k as f64,         // gamma_3 (基底注入)
            AggregationOperator::Add,
            0.0, // beta = 0
        )
    }

    /// 创建 SIGN 模型的验证密钥
    ///
    /// SIGN 需要 D_max = 4d 的总线带宽以支持多阶特征拼接。
    /// 聚合方式为 Concat。
    pub fn new_sign(
        n_real: usize,
        d: usize,
        c: usize,
        k: usize,
        num_edges: usize,
        n_max_hint: usize,
    ) -> Self {
        let d_max_required = 4 * d;
        let container = SuperContainer::new(
            n_real,
            d_max_required, // D_max 扩展为 4d
            c,
            c,
            num_edges,
            n_max_hint,
            d_max_required,
            c,
        );
        Self::build(
            ModelArchitecture::SIGN,
            container,
            k,
            0.0, // alpha_pre = 0
            d,
            1.0, // lambda_1
            0.0, // lambda_2
            1.0, // gamma_1
            1.0, // gamma_2
            0.0, // gamma_3
            AggregationOperator::Concat,
            0.0, // beta
        )
    }

    /// 创建 APPNP 模型的验证密钥
    ///
    /// APPNP 包含前置 MLP 变换和 Teleport 传送概率。
    pub fn new_appnp(
        n_real: usize,
        d: usize,
        c: usize,
        k: usize,
        alpha: f64,
        h_pre: usize,
        num_edges: usize,
        n_max_hint: usize,
    ) -> Self {
        let container = SuperContainer::new(
            n_real, d, h_pre, c, num_edges, n_max_hint, c, // D_effective = c (分类数)
            h_pre,
        );
        Self::build(
            ModelArchitecture::APPNP,
            container,
            k,
            1.0, // alpha_pre = 1 (激活前置 MLP)
            h_pre,
            1.0 - alpha, // lambda_1 (邻居信息)
            alpha,       // lambda_2 (Teleport 概率)
            0.0,         // gamma_1 (不继承历史)
            1.0,         // gamma_2
            0.0,         // gamma_3
            AggregationOperator::Add,
            1.0, // beta = 1 (激活后置 ReLU)
        )
    }

    /// 通用构建器：组装验证密钥并计算架构指纹
    #[allow(clippy::too_many_arguments)]
    fn build(
        architecture: ModelArchitecture,
        container: SuperContainer,
        k: usize,
        alpha_pre: f64,
        h_pre: usize,
        lambda_1: f64,
        lambda_2: f64,
        gamma_1: f64,
        gamma_2: f64,
        gamma_3: f64,
        operator: AggregationOperator,
        beta: f64,
    ) -> Self {
        let mut vk = Self {
            architecture,
            container,
            k,
            alpha_pre,
            h_pre,
            lambda_1,
            lambda_2,
            gamma_1,
            gamma_2,
            gamma_3,
            operator,
            beta,
            model_id: Fp::zero(), // 占位，下面立即计算
        };
        // 注入 Poseidon 海绵计算指纹
        vk.model_id = vk.compute_fingerprint();
        vk
    }

    /// 计算架构指纹 ID_model
    ///
    /// 严格遵循设计文档:
    /// > 证明者初始化一个 Poseidon 哈希海绵，将上述庞大的承诺点阵依次吸入
    /// > 并执行挤出操作，坍缩为一个单一的标量指纹
    ///
    /// 对于 VK 本身，我们将所有架构参数序列化为标量吸入海绵，
    /// 挤出一个唯一的 ID_model ∈ Fp。
    pub fn compute_fingerprint(&self) -> Fp {
        let mut sponge = PoseidonSponge::new();

        // 吸入架构标识
        let arch_byte = match self.architecture {
            ModelArchitecture::SGC => 0u8,
            ModelArchitecture::SSGC => 1u8,
            ModelArchitecture::SIGN => 2u8,
            ModelArchitecture::APPNP => 3u8,
        };
        sponge.absorb_scalar(&Fp::from(arch_byte as u64));

        // 吸入物理容器维度
        sponge.absorb_scalar(&Fp::from(self.container.n_max as u64));
        sponge.absorb_scalar(&Fp::from(self.container.d_max as u64));
        sponge.absorb_scalar(&Fp::from(self.container.h_max as u64));
        sponge.absorb_scalar(&Fp::from(self.container.n_real as u64));
        sponge.absorb_scalar(&Fp::from(self.container.d_real as u64));
        sponge.absorb_scalar(&Fp::from(self.container.h_real as u64));
        sponge.absorb_scalar(&Fp::from(self.container.num_classes as u64));
        sponge.absorb_scalar(&Fp::from(self.container.num_edges as u64));

        // 吸入迭代层数 K
        sponge.absorb_scalar(&Fp::from(self.k as u64));

        // 吸入 Stage 1 控制参数
        sponge.absorb_scalar(&float_to_fp(self.alpha_pre));
        sponge.absorb_scalar(&Fp::from(self.h_pre as u64));

        // 吸入 Stage 2 参数
        sponge.absorb_scalar(&float_to_fp(self.lambda_1));
        sponge.absorb_scalar(&float_to_fp(self.lambda_2));
        sponge.absorb_scalar(&float_to_fp(self.gamma_1));
        sponge.absorb_scalar(&float_to_fp(self.gamma_2));
        sponge.absorb_scalar(&float_to_fp(self.gamma_3));

        // 吸入聚合算子类型
        let op_byte = match self.operator {
            AggregationOperator::Add => 0u8,
            AggregationOperator::Concat => 1u8,
        };
        sponge.absorb_scalar(&Fp::from(op_byte as u64));

        // 吸入 Stage 3 控制参数
        sponge.absorb_scalar(&float_to_fp(self.beta));

        // 强制完成吸收，挤出指纹
        sponge.finalize_absorption();
        sponge.squeeze()
    }

    /// 返回公共参数数组 V_pp (16 个 Fp 标量)
    ///
    /// 对应文档中的:
    /// V_pp = [N_max, D_max, H_max, Edges, N_real, d, h, K,
    ///         α_pre, β, ID_model, γ₁, γ₂, γ₃, λ₁, λ₂]
    pub fn to_public_params(&self) -> [Fp; 16] {
        [
            Fp::from(self.container.n_max as u64),     // 0: N_max
            Fp::from(self.container.d_max as u64),     // 1: D_max
            Fp::from(self.container.h_max as u64),     // 2: H_max
            Fp::from(self.container.num_edges as u64), // 3: Edges
            Fp::from(self.container.n_real as u64),    // 4: N_real
            Fp::from(self.container.d_real as u64),    // 5: d
            Fp::from(self.container.h_real as u64),    // 6: h
            Fp::from(self.k as u64),                   // 7: K
            float_to_fp(self.alpha_pre),               // 8: α_pre
            float_to_fp(self.beta),                    // 9: β
            self.model_id,                             // 10: ID_model
            float_to_fp(self.gamma_1),                 // 11: γ₁
            float_to_fp(self.gamma_2),                 // 12: γ₂
            float_to_fp(self.gamma_3),                 // 13: γ₃
            float_to_fp(self.lambda_1),                // 14: λ₁
            float_to_fp(self.lambda_2),                // 15: λ₂
        ]
    }
}

// ============================================================================
// 模型注册中心
// ============================================================================

/// 模型注册中心
///
/// 管理四套验证密钥与对应的架构指纹。
/// 客户端只需核对该 ID_model 即可确认模型身份。
#[derive(Debug, Default)]
pub struct ModelRegistry {
    /// 指纹 → 验证密钥映射
    registry: HashMap<String, VerificationKey>,
    /// 架构名称 → 指纹映射
    arch_to_id: HashMap<ModelArchitecture, String>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            registry: HashMap::new(),
            arch_to_id: HashMap::new(),
        }
    }

    /// 注册验证密钥
    pub fn register(&mut self, vk: VerificationKey) -> String {
        let id_hex = hex::encode(serialize_scalar(&vk.model_id));
        self.arch_to_id.insert(vk.architecture, id_hex.clone());
        self.registry.insert(id_hex.clone(), vk);
        id_hex
    }

    /// 根据指纹 ID 查找 VK
    pub fn lookup(&self, model_id: &str) -> Option<&VerificationKey> {
        self.registry.get(model_id)
    }

    /// 验证模型指纹与预期架构是否匹配
    pub fn verify_match(&self, model_id: &str, expected: ModelArchitecture) -> bool {
        if let Some(vk) = self.registry.get(model_id) {
            vk.architecture == expected
        } else {
            false
        }
    }

    /// 一次性注册全部四套模型
    ///
    /// 返回架构名称到指纹的映射。
    pub fn register_all_four(
        &mut self,
        n_real: usize,
        d: usize,
        c: usize,
        k: usize,
        alpha: f64,
        num_edges: usize,
        n_max_hint: usize,
    ) -> HashMap<ModelArchitecture, String> {
        let mut map = HashMap::new();

        // SGC
        let vk_sgc = VerificationKey::new_sgc(n_real, d, c, 2, num_edges, n_max_hint);
        map.insert(ModelArchitecture::SGC, self.register(vk_sgc));

        // SSGC
        let vk_ssgc = VerificationKey::new_ssgc(n_real, d, c, k, alpha, num_edges, n_max_hint);
        map.insert(ModelArchitecture::SSGC, self.register(vk_ssgc));

        // SIGN
        let vk_sign = VerificationKey::new_sign(n_real, d, c, 3, num_edges, n_max_hint);
        map.insert(ModelArchitecture::SIGN, self.register(vk_sign));

        // APPNP
        let h_pre = 64; // 默认前置隐藏层
        let vk_appnp =
            VerificationKey::new_appnp(n_real, d, c, 10, 0.1, h_pre, num_edges, n_max_hint);
        map.insert(ModelArchitecture::APPNP, self.register(vk_appnp));

        map
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 将 usize 向上取整至 2 的幂次方
pub fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    if n.is_power_of_two() {
        return n;
    }
    n.next_power_of_two()
}

/// 计算精确的 log2 (假定 n 是 2 的幂次方)
pub fn log2_exact(n: usize) -> usize {
    assert!(n.is_power_of_two(), "n must be a power of two, got {}", n);
    n.trailing_zeros() as usize
}

/// 将 f64 转换为 Fp 标量 (精度: 小数点后 6 位)
///
/// 注意: 此函数仅用于将公开参数序列化至 Poseidon 海绵，
/// 不用于重量化电路内的模型参数。
pub fn float_to_fp(value: f64) -> Fp {
    let scaled = (value * 1_000_000.0).round() as i64;
    if scaled >= 0 {
        Fp::from(scaled as u64)
    } else {
        -Fp::from((-scaled) as u64)
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_container_creation() {
        let container = SuperContainer::new(2708, 1433, 64, 7, 10556, 4096, 2048, 128);
        assert_eq!(container.n_max, 4096);
        assert_eq!(container.d_max, 2048);
        assert_eq!(container.h_max, 128);
        assert_eq!(container.n_v, 12); // log2(4096) = 12
        assert_eq!(container.n_d, 11); // log2(2048) = 11
        assert_eq!(container.n_h, 7); // log2(128) = 7
    }

    #[test]
    fn test_generate_static_mask() {
        let mask = generate_static_mask(3, 8);
        assert_eq!(mask.len(), 64); // 8 × 8
                                    // 前 3 个对角线 = 1
        assert_eq!(mask[0], Fp::one()); // row 0, col 0
        assert_eq!(mask[9], Fp::one()); // row 1, col 1 (1*8+1)
        assert_eq!(mask[18], Fp::one()); // row 2, col 2 (2*8+2)
                                         // 第 4 个对角线 = 0
        assert_eq!(mask[27], Fp::zero()); // row 3, col 3 (3*8+3)
    }

    #[test]
    fn test_apply_mask_min_value_replacement() {
        let n = 4;
        let d = 2;

        let y: Vec<Fp> = (0..8).map(|i| Fp::from(i as u64)).collect();
        let mask = generate_static_mask(2, n); // N_real = 2
        let c_mat = vec![-Fp::from(10_000u64 * 256); n * d];

        let result = apply_mask_min_value_replacement(&y, &mask, &c_mat, n, d);

        // 前 2 行不变
        assert_eq!(result[0], Fp::from(0u64));
        assert_eq!(result[1], Fp::from(1u64));
        assert_eq!(result[2], Fp::from(2u64));
        assert_eq!(result[3], Fp::from(3u64));
        // 后 2 行被替换为极小值
        assert_eq!(result[4], c_mat[4]);
        assert_eq!(result[5], c_mat[5]);
        assert_eq!(result[6], c_mat[6]);
        assert_eq!(result[7], c_mat[7]);
    }

    #[test]
    fn test_embed_in_container() {
        let container = SuperContainer::new(2, 2, 4, 2, 0, 4, 4, 4);
        let data: Vec<Fp> = (1..=4).map(|i| Fp::from(i as u64)).collect(); // 2×2
        let embedded = container.embed_in_container(&data, 2);

        assert_eq!(embedded.len(), 16); // 4×4
                                        // 左上角 2×2 = [1, 2; 3, 4]
        assert_eq!(embedded[0], Fp::from(1u64));
        assert_eq!(embedded[1], Fp::from(2u64));
        assert_eq!(embedded[4], Fp::from(3u64)); // row 1, col 0
        assert_eq!(embedded[5], Fp::from(4u64)); // row 1, col 1
                                                 // Padding = 0
        assert_eq!(embedded[2], Fp::zero());
        assert_eq!(embedded[15], Fp::zero());
    }

    #[test]
    fn test_fingerprint_determinism() {
        let vk1 = VerificationKey::new_sgc(100, 50, 10, 2, 500, 256);
        let vk2 = VerificationKey::new_sgc(100, 50, 10, 2, 500, 256);
        assert_eq!(vk1.model_id, vk2.model_id);
    }

    #[test]
    fn test_fingerprint_uniqueness() {
        let vk_sgc = VerificationKey::new_sgc(100, 50, 10, 2, 500, 256);
        let vk_ssgc = VerificationKey::new_ssgc(100, 50, 10, 16, 0.05, 500, 256);
        let vk_sign = VerificationKey::new_sign(100, 50, 10, 3, 500, 256);
        let vk_appnp = VerificationKey::new_appnp(100, 50, 10, 10, 0.1, 64, 500, 256);

        // 所有指纹应互不相同
        assert_ne!(vk_sgc.model_id, vk_ssgc.model_id);
        assert_ne!(vk_sgc.model_id, vk_sign.model_id);
        assert_ne!(vk_sgc.model_id, vk_appnp.model_id);
        assert_ne!(vk_ssgc.model_id, vk_sign.model_id);
        assert_ne!(vk_ssgc.model_id, vk_appnp.model_id);
        assert_ne!(vk_sign.model_id, vk_appnp.model_id);
    }

    #[test]
    fn test_public_params_roundtrip() {
        let vk = VerificationKey::new_ssgc(100, 50, 10, 16, 0.05, 500, 256);
        let params = vk.to_public_params();

        assert_eq!(params[0], Fp::from(vk.container.n_max as u64));
        assert_eq!(params[4], Fp::from(vk.container.n_real as u64));
        assert_eq!(params[7], Fp::from(vk.k as u64));
        assert_eq!(params[10], vk.model_id);
    }

    #[test]
    fn test_registry_register_all() {
        let mut registry = ModelRegistry::new();
        let map = registry.register_all_four(100, 50, 10, 16, 0.05, 500, 256);

        assert_eq!(map.len(), 4);
        assert!(map.contains_key(&ModelArchitecture::SGC));
        assert!(map.contains_key(&ModelArchitecture::SSGC));
        assert!(map.contains_key(&ModelArchitecture::SIGN));
        assert!(map.contains_key(&ModelArchitecture::APPNP));

        // 验证匹配
        let sgc_id = map.get(&ModelArchitecture::SGC).unwrap();
        assert!(registry.verify_match(sgc_id, ModelArchitecture::SGC));
        assert!(!registry.verify_match(sgc_id, ModelArchitecture::SSGC));
    }

    #[test]
    fn test_get_effective_dimension() {
        let container = SuperContainer::new(100, 50, 64, 10, 0, 256, 128, 128);
        assert_eq!(
            container.get_effective_dimension(ModelArchitecture::SGC),
            50
        );
        assert_eq!(
            container.get_effective_dimension(ModelArchitecture::SSGC),
            50
        );
        assert_eq!(
            container.get_effective_dimension(ModelArchitecture::SIGN),
            50
        );
        assert_eq!(
            container.get_effective_dimension(ModelArchitecture::APPNP),
            10
        );
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(1433), 2048);
        assert_eq!(next_power_of_two(2708), 4096);
    }
}
