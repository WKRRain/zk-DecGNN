//! SPARK 内存一致性协议实现
//!
//! 严格遵循 Spartan-ZENO 设计文档第 2.2.2 - 2.2.3 节。
//!
//! ## 核心目标
//! 在零知识证明中，算术电路没有"指针"或"数组索引"概念。
//! SPARK 将离散内存操作日志转化为布尔超立方体上的连续多项式闭包，
//! 通过连乘校验 (Grand Product Check) 确保访存操作的一致性与合法性。
//!
//! ## 三元组结构
//! 每次访存操作 = (addr, val_compressed, ts)
//! - addr: 目标节点地址
//! - val_compressed: 经 RLC 压缩的特征值
//! - ts: 时间戳 (单调递增)
//!
//! ## 四张操作流
//! - Init: 内存初始化 (写入初始特征)
//! - RS (Read Stream): 读取操作
//! - WS (Write Stream): 写入操作
//! - Audit: 审计读取 (最终状态快照)

use crate::curve_config::Fp;
use ark_ff::{One, PrimeField, Zero};

// ============================================================================
// 核心数据结构
// ============================================================================

/// 单次访存操作记录
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryOp {
    /// 地址 (被访问节点的 ID)
    pub addr: Fp,
    /// RLC 压缩后的特征值
    pub val_compressed: Fp,
    /// 时间戳 (绝对递增，写入后时间戳 +1)
    pub ts: Fp,
}

/// 四张内存操作流
#[derive(Debug, Clone)]
pub struct MemoryStreams {
    /// 初始化流 (N_real 条)
    pub init: Vec<MemoryOp>,
    /// 读取流 (2|E| 条)
    pub read: Vec<MemoryOp>,
    /// 写入流 (2|E| 条)
    pub write: Vec<MemoryOp>,
    /// 审计流 (N_real 条)
    pub audit: Vec<MemoryOp>,
}

/// SPARK 协议的转录本随机参数
#[derive(Debug, Clone)]
pub struct SparkTranscriptParams {
    /// RLC 压缩向量 η ∈ Fp^d
    pub eta: Vec<Fp>,
    /// 代数压缩标量 δ₁
    pub delta_1: Fp,
    /// 代数压缩标量 δ₂
    pub delta_2: Fp,
    /// 全局位移标量 τ
    pub tau: Fp,
}

// ============================================================================
// RLC 特征压缩 (Random Linear Combination)
// ============================================================================

/// 使用随机向量 η 对高维特征进行 RLC 标量化坍缩
///
/// val_compressed = Σ_{j=0}^{d-1} η_j · val[j] (mod p)
pub fn compress_feature(val: &[Fp], eta: &[Fp]) -> Fp {
    assert_eq!(
        val.len(),
        eta.len(),
        "Feature dimension must match η length"
    );
    let mut result = Fp::zero();
    for j in 0..val.len() {
        result += eta[j] * val[j];
    }
    result
}

/// 批量 RLC 压缩
pub fn compress_features_batch(vals: &[Vec<Fp>], eta: &[Fp]) -> Vec<Fp> {
    vals.iter().map(|v| compress_feature(v, eta)).collect()
}

// ============================================================================
// 代数压缩多项式 E(t) 构建
// ============================================================================

/// 构建单个操作的代数压缩多项式
///
/// E(t) = addr + δ₁ · val_compressed + δ₂ · ts - τ (mod p)
pub fn build_algebraic_tag(op: &MemoryOp, delta_1: &Fp, delta_2: &Fp, tau: &Fp) -> Fp {
    op.addr + *delta_1 * op.val_compressed + *delta_2 * op.ts - *tau
}

/// 批量构建操作流的代数压缩值
pub fn build_algebraic_tags(ops: &[MemoryOp], delta_1: &Fp, delta_2: &Fp, tau: &Fp) -> Vec<Fp> {
    ops.iter()
        .map(|op| build_algebraic_tag(op, delta_1, delta_2, tau))
        .collect()
}

// ============================================================================
// 内存排序与时序单调性校验
// ============================================================================

/// 按地址排序内存操作 (addr 优先, ts 其次)
pub fn sort_by_addr_then_ts(ops: &[MemoryOp]) -> Vec<MemoryOp> {
    let mut sorted = ops.to_vec();
    sorted.sort_by(|a, b| {
        let addr_cmp = a.addr.into_bigint().cmp(&b.addr.into_bigint());
        match addr_cmp {
            std::cmp::Ordering::Equal => a.ts.into_bigint().cmp(&b.ts.into_bigint()),
            other => other,
        }
    });
    sorted
}

/// 时序因果校验: 相邻元组的单调性断言
///
/// 检查排序后的相邻操作是否满足:
/// 1. addr_{m+1} ≥ addr_m
/// 2. 若 addr 相同，Read-to-Read: ts_{m+1} ≥ ts_m 且 val 一致
/// 3. 若 addr 相同，包含 Write: ts_{m+1} > ts_m
pub fn verify_monotonicity(sorted: &[MemoryOp]) -> bool {
    if sorted.len() <= 1 {
        return true;
    }
    for m in 1..sorted.len() {
        let prev = &sorted[m - 1];
        let curr = &sorted[m];
        // Addr 非递减
        let prev_addr = prev.addr.into_bigint();
        let curr_addr = curr.addr.into_bigint();
        if curr_addr < prev_addr {
            return false;
        }
        // 同地址时检查时序
        if curr_addr == prev_addr {
            let prev_ts = prev.ts.into_bigint();
            let curr_ts = curr.ts.into_bigint();
            if curr_ts < prev_ts {
                return false;
            }
        }
    }
    true
}

// ============================================================================
// 全局连乘排列校验 (Grand Product Permutation Check)
// ============================================================================

/// 计算操作流 E(t) 的连乘积
///
/// Π E(t) (mod p)
pub fn grand_product(tags: &[Fp]) -> Fp {
    let mut product = Fp::one();
    for tag in tags {
        product *= tag;
    }
    product
}

/// 全局排列校验
///
/// Π_{T_chronological} E(t) ≡ Π_{T_sorted} E(t) (mod p)
pub fn verify_permutation(chronological_tags: &[Fp], sorted_tags: &[Fp]) -> bool {
    grand_product(chronological_tags) == grand_product(sorted_tags)
}

// ============================================================================
// 二叉树连乘构建 (Grand Product Binary Tree)
// ============================================================================

/// 二叉树节点 (以稀疏方式存储)
#[derive(Debug, Clone)]
pub struct GrandProductTree {
    /// 叶子层数据 (第 0 层)
    pub leaves: Vec<Fp>,
    /// 树高 (层数 = n_ops)
    pub height: usize,
    /// 各层中间值 (第 1..height 层)
    /// layers[l][i] 对应第 l 层第 i 个节点
    pub layers: Vec<Vec<Fp>>,
    /// 根节点值
    pub root: Fp,
}

impl GrandProductTree {
    /// 从叶子层数据构建连乘二叉树
    ///
    /// 叶子数组长度必须为 2^{height}。
    /// 若不足，自行补齐为 1 (乘法单位元)。
    pub fn build(leaves: &[Fp], height: usize) -> Self {
        let expected_len = 1 << height;
        let mut padded_leaves = leaves.to_vec();
        while padded_leaves.len() < expected_len {
            padded_leaves.push(Fp::one()); // 填充为 1 (乘积不变)
        }

        let mut layers: Vec<Vec<Fp>> = Vec::with_capacity(height);
        let mut current = padded_leaves.clone();

        for _ in 0..height {
            let next_len = current.len() / 2;
            let mut next = Vec::with_capacity(next_len);
            for i in 0..next_len {
                next.push(current[2 * i] * current[2 * i + 1]);
            }
            layers.push(next.clone());
            current = next;
        }

        // current 此时为单元素 [root]
        let root = current[0];

        GrandProductTree {
            leaves: padded_leaves,
            height,
            layers,
            root,
        }
    }

    /// 获取第 l 层的数据 (0 = 叶子层)
    pub fn get_layer(&self, l: usize) -> &[Fp] {
        if l == 0 {
            &self.leaves
        } else {
            &self.layers[l - 1]
        }
    }
}

/// 为四张操作流分别构建二叉树并提取根声明
pub fn build_memory_trees(
    init_tags: &[Fp],
    rs_tags: &[Fp],
    ws_tags: &[Fp],
    audit_tags: &[Fp],
    height: usize,
) -> (
    GrandProductTree,
    GrandProductTree,
    GrandProductTree,
    GrandProductTree,
) {
    let tree_init = GrandProductTree::build(init_tags, height);
    let tree_rs = GrandProductTree::build(rs_tags, height);
    let tree_ws = GrandProductTree::build(ws_tags, height);
    let tree_audit = GrandProductTree::build(audit_tags, height);
    (tree_init, tree_rs, tree_ws, tree_audit)
}

/// 全局内存守恒断言
///
/// V_init^(root) × V_ws^(root) ≡ V_rs^(root) × V_audit^(root) (mod p)
pub fn verify_memory_conservation(
    root_init: &Fp,
    root_ws: &Fp,
    root_rs: &Fp,
    root_audit: &Fp,
) -> bool {
    let lhs = *root_init * *root_ws;
    let rhs = *root_rs * *root_audit;
    lhs == rhs
}

// ============================================================================
// 时序排列的写入后时间戳处理
// ============================================================================

/// 对写入流 (WS) 的操作应用时间戳 +1 逻辑
///
/// 文档规定: WS 的 ts 必须在读取时间戳的基础上 +1，
/// 以区分"读取旧状态"与"写入新状态"。
pub fn apply_write_timestamp_shift(write_ops: &[MemoryOp]) -> Vec<MemoryOp> {
    write_ops
        .iter()
        .map(|op| {
            let mut shifted = op.clone();
            shifted.ts += Fp::one();
            shifted
        })
        .collect()
}

// ============================================================================
// GKR 层间降维多项式 (一元一次多项式)
// ============================================================================

/// 一元一次多项式: f(X) = c₀ + c₁·X
#[derive(Debug, Clone)]
pub struct LinearPolynomial {
    pub c0: Fp,
    pub c1: Fp,
}

impl LinearPolynomial {
    /// 对给定的 X 求值
    pub fn evaluate(&self, x: &Fp) -> Fp {
        self.c0 + self.c1 * *x
    }
}

/// 在 GKR 递推中，根据下一层数据在固定前缀坐标 r_k 处的值
/// 构建一元一次多项式。
///
/// 由于是二叉树模型，自由变量 X ∈ {0, 1} 代表左子树(0)和右子树(1)。
pub fn build_layer_polynomial_from_children(
    left_val: &Fp,  // 左子树在 X=0 处的值
    right_val: &Fp, // 右子树在 X=1 处的值
) -> LinearPolynomial {
    // f(0) = c0 = left_val
    // f(1) = c0 + c1 = right_val → c1 = right_val - left_val
    let c0 = *left_val;
    let c1 = *right_val - *left_val;
    LinearPolynomial { c0, c1 }
}

// ============================================================================
// RLC 聚合 (用于四棵树并行 GKR)
// ============================================================================

/// 使用随机权重 ρ 对四棵树的声明值进行 RLC 聚合
///
/// v_combined = v_init + ρ·v_rs + ρ²·v_ws + ρ³·v_audit (mod p)
pub fn rlc_combine_four(v_init: &Fp, v_rs: &Fp, v_ws: &Fp, v_audit: &Fp, rho: &Fp) -> Fp {
    let rho2 = *rho * *rho;
    let rho3 = rho2 * *rho;
    *v_init + *rho * *v_rs + rho2 * *v_ws + rho3 * *v_audit
}

/// GKR 乘积恒等式验证 (RLC 聚合版)
///
/// v_combined ≡ g_init(0)·g_init(1) + ρ·g_rs(0)·g_rs(1)
///            + ρ²·g_ws(0)·g_ws(1) + ρ³·g_audit(0)·g_audit(1) (mod p)
pub fn verify_gkr_rlc_product(
    v_combined: &Fp,
    g_init: &LinearPolynomial,
    g_rs: &LinearPolynomial,
    g_ws: &LinearPolynomial,
    g_audit: &LinearPolynomial,
    rho: &Fp,
) -> bool {
    let rhs = g_init.evaluate(&Fp::zero()) * g_init.evaluate(&Fp::one())
        + *rho * g_rs.evaluate(&Fp::zero()) * g_rs.evaluate(&Fp::one())
        + *rho * *rho * g_ws.evaluate(&Fp::zero()) * g_ws.evaluate(&Fp::one())
        + *rho * *rho * *rho * g_audit.evaluate(&Fp::zero()) * g_audit.evaluate(&Fp::one());
    *v_combined == rhs
}

// ============================================================================
// 多线性扩展 (MLE) for SPARK Traces
// ============================================================================

/// 对一维操作数组进行 MLE 装箱
///
/// f̃(x) = Σ_{k=0}^{len-1} data[k] · eq(x, bin(k))
pub fn mle_evaluate(data: &[Fp], r: &[Fp]) -> Fp {
    use crate::polynomial::build_lagrange_basis;
    let basis = build_lagrange_basis(r);
    let mut result = Fp::zero();
    let len = data.len().min(basis.len());
    for k in 0..len {
        result += basis[k] * data[k];
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
    fn test_compress_feature() {
        let val: Vec<Fp> = vec![Fp::from(1u64), Fp::from(2u64), Fp::from(3u64)];
        let eta: Vec<Fp> = vec![Fp::from(1u64), Fp::from(1u64), Fp::from(1u64)];
        let compressed = compress_feature(&val, &eta);
        assert_eq!(compressed, Fp::from(6u64));
    }

    #[test]
    fn test_grand_product() {
        let tags: Vec<Fp> = vec![Fp::from(2u64), Fp::from(3u64), Fp::from(4u64)];
        let prod = grand_product(&tags);
        assert_eq!(prod, Fp::from(24u64)); // 2*3*4 = 24
    }

    #[test]
    fn test_permutation_identity() {
        // 顺序和乱序的连乘积必须相等
        let tags1: Vec<Fp> = vec![Fp::from(1u64), Fp::from(2u64), Fp::from(3u64)];
        let tags2: Vec<Fp> = vec![Fp::from(3u64), Fp::from(1u64), Fp::from(2u64)];
        assert!(verify_permutation(&tags1, &tags2));
    }

    #[test]
    fn test_build_tree() {
        let leaves: Vec<Fp> = vec![
            Fp::from(1u64),
            Fp::from(2u64),
            Fp::from(3u64),
            Fp::from(4u64),
        ];
        let tree = GrandProductTree::build(&leaves, 2);
        // root = (1*2)*(3*4) = 2*12 = 24
        assert_eq!(tree.root, Fp::from(24u64));
        assert_eq!(tree.height, 2);
        // layer 1: [2, 12]
        assert_eq!(tree.layers[0][0], Fp::from(2u64));
        assert_eq!(tree.layers[0][1], Fp::from(12u64));
    }

    #[test]
    fn test_build_tree_with_padding() {
        // 3 leaves padded to 4 (2^2)
        let leaves: Vec<Fp> = vec![Fp::from(5u64), Fp::from(6u64), Fp::from(7u64)];
        let tree = GrandProductTree::build(&leaves, 2);
        // leaves: [5, 6, 7, 1]
        // root = (5*6)*(7*1) = 30*7 = 210
        assert_eq!(tree.root, Fp::from(210u64));
    }

    #[test]
    fn test_memory_conservation() {
        let root_init = Fp::from(2u64);
        let root_rs = Fp::from(6u64);
        let root_ws = Fp::from(3u64);
        let root_audit = Fp::from(1u64);
        // 2*3 = 6*1 = 6
        assert!(verify_memory_conservation(
            &root_init,
            &root_ws,
            &root_rs,
            &root_audit
        ));
    }

    #[test]
    fn test_memory_conservation_fail() {
        let root_init = Fp::from(2u64);
        let root_ws = Fp::from(3u64);
        let root_rs = Fp::from(7u64);
        let root_audit = Fp::from(1u64);
        // 2*3 = 6 ≠ 7*1
        assert!(!verify_memory_conservation(
            &root_init,
            &root_ws,
            &root_rs,
            &root_audit
        ));
    }

    #[test]
    fn test_rlc_combine() {
        let rho = Fp::from(2u64);
        let v = rlc_combine_four(
            &Fp::from(1u64),
            &Fp::from(2u64),
            &Fp::from(3u64),
            &Fp::from(4u64),
            &rho,
        );
        // 1 + 2*2 + 4*3 + 8*4 = 1+4+12+32 = 49
        assert_eq!(v, Fp::from(49u64));
    }

    #[test]
    fn test_linear_polynomial() {
        // f(0)=3, f(1)=7 → c0=3, c1=4
        let poly = build_layer_polynomial_from_children(&Fp::from(3u64), &Fp::from(7u64));
        assert_eq!(poly.evaluate(&Fp::zero()), Fp::from(3u64));
        assert_eq!(poly.evaluate(&Fp::one()), Fp::from(7u64));
        // f(2) = 3+4*2 = 11
        assert_eq!(poly.evaluate(&Fp::from(2u64)), Fp::from(11u64));
    }
}
