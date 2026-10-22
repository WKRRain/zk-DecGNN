//! 连乘二叉树与底层轨迹承诺 (Binary Tree & Trace Commitments)
//!
//! 严格遵循 Spartan-ZENO 设计文档第 2.2.3 节。
//!
//! ## 核心架构
//!
//! ### 第一步：构建四组二叉树并提取根节点声明
//! - Init, RS, WS, Audit 各自构建深度为 n_ops 的连乘二叉树
//! - 提取树根标量: V_Init, V_RS, V_WS, V_Audit
//! - 全局内存守恒断言: V_Init × V_WS == V_RS × V_Audit
//!
//! ### 第二步：底层动态访存轨迹的多线性扩展 (Trace MLEs)
//! - 读操作流: addr_read, val_read, ts_read
//! - 写操作流: addr_write, val_write, ts_write
//!
//! ### 第三步：生成底层轨迹的 Hyrax 椭圆曲线承诺
//! - 仅对动态特征值与时间戳数组承诺
//! - 地址轨迹由静态图拓扑决定，无需动态承诺
//!
//! ### 架构优化体现
//! 通过引入二叉树规约，并仅对底层叶子节点数据进行 Hyrax 承诺，
//! 彻底免去了对中间连乘状态的 O(M_ops) 级别庞大椭圆曲线运算。

use crate::curve_config::Fp;
use crate::hyrax::{HyraxCommitment, HyraxProverParams};
use crate::polynomial::MLEPolynomial;
use ark_ff::One;
use ark_std::rand::Rng;

use super::memory_check::MemoryTuple;

// ============================================================================
// 连乘二叉树节点
// ============================================================================

/// 二叉树节点
///
/// 存储连乘二叉树某一层的节点值。
/// 叶子层 (第 0 层) 存储的是经过指示器旁路处理后的有效压缩元组 E(t)。
/// 上层节点是下层相邻两个子节点的乘积。
#[derive(Debug, Clone)]
pub struct BinaryTreeLayer {
    /// 当前层的节点值数组
    /// 长度为 2^(depth_of_layer)
    pub values: Vec<Fp>,
    /// 当前层的高度 (0 = 叶子层, n_ops = 根节点)
    pub depth: usize,
}

impl BinaryTreeLayer {
    /// 创建叶子层
    pub fn new_leaf(values: Vec<Fp>) -> Self {
        Self { values, depth: 0 }
    }

    /// 从叶子层值构建完整二叉树并返回根节点
    ///
    /// # 参数
    /// - leaf_values: 叶子节点 E(t) 值数组
    ///
    /// # 返回
    /// - 根节点标量
    pub fn build_tree(leaf_values: &[Fp]) -> Fp {
        if leaf_values.is_empty() {
            return Fp::one();
        }
        if leaf_values.len() == 1 {
            return leaf_values[0];
        }

        let mut current = leaf_values.to_vec();

        // 自底向上构建
        while current.len() > 1 {
            let mut next = Vec::with_capacity(current.len().div_ceil(2));

            for chunk in current.chunks(2) {
                let left = chunk[0];
                let right = if chunk.len() > 1 { chunk[1] } else { Fp::one() };
                next.push(left * right);
            }

            current = next;
        }

        current[0]
    }

    /// 构建完整二叉树 (保留所有层，用于 GKR 递推)
    ///
    /// # 返回
    /// - Vec<BinaryTreeLayer>: 从第 0 层 (叶子) 到根节点的所有层
    pub fn build_full_tree(leaf_values: &[Fp]) -> Vec<Self> {
        if leaf_values.is_empty() {
            return vec![Self {
                values: vec![Fp::one()],
                depth: 0,
            }];
        }

        let mut layers = Vec::new();
        layers.push(Self::new_leaf(leaf_values.to_vec()));

        let mut current = leaf_values.to_vec();
        let mut depth = 0;

        while current.len() > 1 {
            let mut next = Vec::with_capacity(current.len().div_ceil(2));
            for chunk in current.chunks(2) {
                let left = chunk[0];
                let right = if chunk.len() > 1 { chunk[1] } else { Fp::one() };
                next.push(left * right);
            }
            depth += 1;
            layers.push(Self {
                values: next.clone(),
                depth,
            });
            current = next;
        }

        layers
    }

    /// 获取当前层的节点数量
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// 判断是否为空
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

// ============================================================================
// 四组二叉树的全局内存守恒断言
// ============================================================================

/// 四棵二叉树的根节点
#[derive(Debug, Clone)]
pub struct TreeRoots {
    /// V_Init (root): 初始写入操作树的根
    pub v_init: Fp,
    /// V_RS (root): 读取操作树的根
    pub v_rs: Fp,
    /// V_WS (root): 写入操作树的根
    pub v_ws: Fp,
    /// V_Audit (root): 终态审计树的根
    pub v_audit: Fp,
}

impl TreeRoots {
    /// 验证全局内存守恒等式
    ///
    /// V_Init × V_WS == V_RS × V_Audit
    pub fn verify_global_conservation(&self) -> bool {
        self.v_init * self.v_ws == self.v_rs * self.v_audit
    }
}

/// 从四组 E(t) 压缩值构建四棵二叉树并提取根节点
pub fn build_four_trees_and_extract_roots(
    e_init: &[Fp],
    e_rs: &[Fp],
    e_ws: &[Fp],
    e_audit: &[Fp],
) -> TreeRoots {
    TreeRoots {
        v_init: BinaryTreeLayer::build_tree(e_init),
        v_rs: BinaryTreeLayer::build_tree(e_rs),
        v_ws: BinaryTreeLayer::build_tree(e_ws),
        v_audit: BinaryTreeLayer::build_tree(e_audit),
    }
}

// ============================================================================
// 指示器旁路处理 (Indicator Bypass)
// ============================================================================

/// 活动的操作指示器
///
/// 对于有效操作 (active = true)，叶子节点值为 E(t)
/// 对于无效操作 (active = false，超出真实边数的 Padding 区)，叶子节点值为 1
///   (因为 1 是乘法的单位元，不影响连乘积)
#[derive(Debug, Clone)]
pub struct IndicatorBypass {
    /// 活动指示器: i ∈ {0, 1} 表示操作 m 是否有效
    pub active_flags: Vec<bool>,
}

impl IndicatorBypass {
    /// 创建指示器
    pub fn new(active_flags: Vec<bool>) -> Self {
        Self { active_flags }
    }

    /// 应用指示器，生成有效的叶子节点值
    ///
    /// leaf[m] = active[m] * E[m] + (1 - active[m]) * 1
    ///         = active ? E[m] : 1
    pub fn apply(&self, e_values: &[Fp]) -> Vec<Fp> {
        assert_eq!(
            self.active_flags.len(),
            e_values.len(),
            "IndicatorBypass: length mismatch"
        );

        self.active_flags
            .iter()
            .zip(e_values.iter())
            .map(|(&active, &e)| {
                if active {
                    e
                } else {
                    Fp::one() // 单位元，不影响连乘积
                }
            })
            .collect()
    }
}

// ============================================================================
// 底层访存轨迹的多线性扩展 (Trace MLEs)
// ============================================================================

/// 访存轨迹的 MLE 装箱
///
/// 将离散的访存操作数组转化为定义在布尔超立方体上的连续多项式。
#[derive(Debug, Clone)]
pub struct TraceMLEs {
    /// 读操作地址 MLE
    pub addr_read: MLEPolynomial,
    /// 读操作特征值 MLE
    pub val_read: MLEPolynomial,
    /// 读操作时间戳 MLE
    pub ts_read: MLEPolynomial,
    /// 写操作地址 MLE
    pub addr_write: MLEPolynomial,
    /// 写操作特征值 MLE
    pub val_write: MLEPolynomial,
    /// 写操作时间戳 MLE
    pub ts_write: MLEPolynomial,
}

impl TraceMLEs {
    /// 从离散数组构建所有轨迹 MLEs
    ///
    /// # 参数
    /// - n_ops: 布尔变量数 (ceil(log2(M_ops)))
    /// - addr_read_arr: 读地址数组
    /// - val_read_arr: 读特征值数组
    /// - ts_read_arr: 读时间戳数组
    /// - addr_write_arr: 写地址数组
    /// - val_write_arr: 写特征值数组
    /// - ts_write_arr: 写时间戳数组
    pub fn from_arrays(
        n_ops: usize,
        addr_read_arr: &[Fp],
        val_read_arr: &[Fp],
        ts_read_arr: &[Fp],
        addr_write_arr: &[Fp],
        val_write_arr: &[Fp],
        ts_write_arr: &[Fp],
    ) -> Self {
        Self {
            addr_read: MLEPolynomial::from_evaluations(n_ops, addr_read_arr),
            val_read: MLEPolynomial::from_evaluations(n_ops, val_read_arr),
            ts_read: MLEPolynomial::from_evaluations(n_ops, ts_read_arr),
            addr_write: MLEPolynomial::from_evaluations(n_ops, addr_write_arr),
            val_write: MLEPolynomial::from_evaluations(n_ops, val_write_arr),
            ts_write: MLEPolynomial::from_evaluations(n_ops, ts_write_arr),
        }
    }

    /// 从 MemoryTuple 数组构建读/写轨迹 MLEs
    pub fn from_memory_tuples(
        n_ops: usize,
        rs_tuples: &[MemoryTuple],
        ws_tuples: &[MemoryTuple],
    ) -> Self {
        let addr_read_arr: Vec<Fp> = rs_tuples.iter().map(|t| Fp::from(t.addr)).collect();
        let val_read_arr: Vec<Fp> = rs_tuples.iter().map(|t| t.val_compressed).collect();
        let ts_read_arr: Vec<Fp> = rs_tuples.iter().map(|t| Fp::from(t.ts)).collect();

        let addr_write_arr: Vec<Fp> = ws_tuples.iter().map(|t| Fp::from(t.addr)).collect();
        let val_write_arr: Vec<Fp> = ws_tuples.iter().map(|t| t.val_compressed).collect();
        let ts_write_arr: Vec<Fp> = ws_tuples.iter().map(|t| Fp::from(t.ts)).collect();

        Self::from_arrays(
            n_ops,
            &addr_read_arr,
            &val_read_arr,
            &ts_read_arr,
            &addr_write_arr,
            &val_write_arr,
            &ts_write_arr,
        )
    }
}

// ============================================================================
// 底层轨迹的 Hyrax 椭圆曲线承诺
// ============================================================================

/// 四组底层轨迹的椭圆曲线承诺
///
/// 仅对动态特征值与时间戳数组进行承诺。
/// 地址轨迹由静态图拓扑完全决定，无需动态承诺。
#[derive(Debug, Clone)]
pub struct TraceCommitments {
    /// C_V_read: 读取特征值承诺
    pub cv_read: HyraxCommitment,
    /// C_T_read: 读取时间戳承诺
    pub ct_read: HyraxCommitment,
    /// C_V_write: 写入特征值承诺
    pub cv_write: HyraxCommitment,
    /// C_T_write: 写入时间戳承诺
    pub ct_write: HyraxCommitment,
}

impl TraceCommitments {
    /// 生成四组轨迹承诺
    ///
    /// # 参数
    /// - pp: Hyrax 公共参数
    /// - val_read_arr: 读取特征值数组
    /// - ts_read_arr: 读取时间戳数组
    /// - val_write_arr: 写入特征值数组
    /// - ts_write_arr: 写入时间戳数组
    pub fn commit<R: Rng>(
        pp: &HyraxProverParams,
        rng: &mut R,
        val_read_arr: &[Fp],
        ts_read_arr: &[Fp],
        val_write_arr: &[Fp],
        ts_write_arr: &[Fp],
    ) -> Self {
        Self {
            cv_read: pp.commit(val_read_arr, rng),
            ct_read: pp.commit(ts_read_arr, rng),
            cv_write: pp.commit(val_write_arr, rng),
            ct_write: pp.commit(ts_write_arr, rng),
        }
    }

    /// 从 TraceMLEs 的底层数组生成承诺
    pub fn from_trace_mles<R: Rng>(
        pp: &HyraxProverParams,
        rng: &mut R,
        trace_mles: &TraceMLEs,
    ) -> Self {
        Self::commit(
            pp,
            rng,
            trace_mles.val_read.evaluations(),
            trace_mles.ts_read.evaluations(),
            trace_mles.val_write.evaluations(),
            trace_mles.ts_write.evaluations(),
        )
    }
}

// ============================================================================
// 叶子节点代数压缩求值
// ============================================================================

/// 计算读操作流在随机求值点 r_m 处的代数压缩值
///
/// E_RS(r_m) = A_read(r_m) + δ₁ · V_read(r_m) + δ₂ · T_read(r_m) − τ
pub fn eval_read_compression(
    trace: &TraceMLEs,
    r_m: &[Fp],
    delta_1: Fp,
    delta_2: Fp,
    tau: Fp,
) -> Fp {
    let addr = trace.addr_read.evaluate(r_m);
    let val = trace.val_read.evaluate(r_m);
    let ts = trace.ts_read.evaluate(r_m);
    addr + delta_1 * val + delta_2 * ts - tau
}

/// 计算写操作流在随机求值点 r_m 处的代数压缩值
///
/// E_WS(r_m) = A_write(r_m) + δ₁ · V_write(r_m) + δ₂ · T_write(r_m) − τ
pub fn eval_write_compression(
    trace: &TraceMLEs,
    r_m: &[Fp],
    delta_1: Fp,
    delta_2: Fp,
    tau: Fp,
) -> Fp {
    let addr = trace.addr_write.evaluate(r_m);
    let val = trace.val_write.evaluate(r_m);
    let ts = trace.ts_write.evaluate(r_m);
    addr + delta_1 * val + delta_2 * ts - tau
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{One, Zero};

    // ---- 二叉树构建测试 ----

    #[test]
    fn test_build_tree_empty() {
        let root = BinaryTreeLayer::build_tree(&[]);
        assert_eq!(root, Fp::one());
    }

    #[test]
    fn test_build_tree_single() {
        let root = BinaryTreeLayer::build_tree(&[Fp::from(42u64)]);
        assert_eq!(root, Fp::from(42u64));
    }

    #[test]
    fn test_build_tree_power_of_two() {
        // 4 个叶子: [2, 3, 5, 7]
        // Layer 1: [2*3=6, 5*7=35]
        // Layer 2 (root): 6*35=210
        let leaves = vec![
            Fp::from(2u64),
            Fp::from(3u64),
            Fp::from(5u64),
            Fp::from(7u64),
        ];
        let root = BinaryTreeLayer::build_tree(&leaves);
        assert_eq!(root, Fp::from(210u64));
    }

    #[test]
    fn test_build_tree_non_power_of_two() {
        // 3 个叶子: [2, 3, 5]
        // Pad to 4: [2, 3, 5, 1]
        // Layer 1: [2*3=6, 5*1=5]
        // Layer 2: 6*5=30
        let leaves = vec![Fp::from(2u64), Fp::from(3u64), Fp::from(5u64)];
        let root = BinaryTreeLayer::build_tree(&leaves);
        assert_eq!(root, Fp::from(30u64));
    }

    #[test]
    fn test_build_full_tree() {
        let leaves = vec![
            Fp::from(2u64),
            Fp::from(3u64),
            Fp::from(5u64),
            Fp::from(7u64),
        ];
        let layers = BinaryTreeLayer::build_full_tree(&leaves);
        // Layer 0: [2, 3, 5, 7] (depth=0)
        // Layer 1: [6, 35]       (depth=1)
        // Layer 2: [210]         (depth=2)
        assert_eq!(layers.len(), 3);
        assert_eq!(
            layers[0].values,
            vec![
                Fp::from(2u64),
                Fp::from(3u64),
                Fp::from(5u64),
                Fp::from(7u64)
            ]
        );
        assert_eq!(layers[0].depth, 0);
        assert_eq!(layers[1].values, vec![Fp::from(6u64), Fp::from(35u64)]);
        assert_eq!(layers[1].depth, 1);
        assert_eq!(layers[2].values, vec![Fp::from(210u64)]);
        assert_eq!(layers[2].depth, 2);
    }

    // ---- 指示器旁路测试 ----

    #[test]
    fn test_indicator_bypass_all_active() {
        let bypass = IndicatorBypass::new(vec![true, true, true]);
        let e_values = vec![Fp::from(2u64), Fp::from(3u64), Fp::from(5u64)];
        let result = bypass.apply(&e_values);
        assert_eq!(result, vec![Fp::from(2u64), Fp::from(3u64), Fp::from(5u64)]);
    }

    #[test]
    fn test_indicator_bypass_some_inactive() {
        let bypass = IndicatorBypass::new(vec![true, false, true]);
        let e_values = vec![Fp::from(2u64), Fp::from(3u64), Fp::from(5u64)];
        let result = bypass.apply(&e_values);
        // active: 2, inactive: 1, active: 5
        assert_eq!(result, vec![Fp::from(2u64), Fp::one(), Fp::from(5u64)]);
    }

    #[test]
    fn test_indicator_bypass_all_inactive() {
        let bypass = IndicatorBypass::new(vec![false, false, false]);
        let e_values = vec![Fp::from(2u64), Fp::from(3u64), Fp::from(5u64)];
        let result = bypass.apply(&e_values);
        assert_eq!(result, vec![Fp::one(), Fp::one(), Fp::one()]);
    }

    // ---- 树根与全局守恒测试 ----

    #[test]
    fn test_tree_roots_global_conservation() {
        // 构建合法数据
        let e_init = vec![Fp::from(2u64), Fp::from(3u64)];
        let e_rs = vec![Fp::from(2u64), Fp::from(5u64)];
        let e_ws = vec![Fp::from(7u64), Fp::from(11u64)];
        let e_audit = vec![Fp::from(3u64), Fp::from(5u64)];

        let roots = build_four_trees_and_extract_roots(&e_init, &e_rs, &e_ws, &e_audit);

        // V_Init = 2*3 = 6
        // V_RS = 2*5 = 10
        // V_WS = 7*11 = 77
        // V_Audit = 3*5 = 15
        // 6*77 = 462, 10*15 = 150 (不守恒!)
        // 这个测试不期望守恒，只是验证构建正确
        assert_eq!(roots.v_init, Fp::from(6u64));
        assert_eq!(roots.v_rs, Fp::from(10u64));
        assert_eq!(roots.v_ws, Fp::from(77u64));
        assert_eq!(roots.v_audit, Fp::from(15u64));
    }

    #[test]
    fn test_tree_roots_global_conservation_holds() {
        // 构建守恒数据: Init × WS == RS × Audit
        // 设 Init=[2,3], WS=[5,7], RS=[2,5], Audit=[3,7]
        // Init 根: 2*3=6
        // WS 根: 5*7=35
        // RS 根: 2*5=10
        // Audit 根: 3*7=21
        // 6*35=210, 10*21=210 ✓
        let e_init = vec![Fp::from(2u64), Fp::from(3u64)];
        let e_ws = vec![Fp::from(5u64), Fp::from(7u64)];
        let e_rs = vec![Fp::from(2u64), Fp::from(5u64)];
        let e_audit = vec![Fp::from(3u64), Fp::from(7u64)];

        let roots = build_four_trees_and_extract_roots(&e_init, &e_rs, &e_ws, &e_audit);
        assert!(roots.verify_global_conservation());
    }

    #[test]
    fn test_tree_roots_global_conservation_fails() {
        let e_init = vec![Fp::from(2u64), Fp::from(3u64)];
        let e_ws = vec![Fp::from(5u64), Fp::from(7u64)];
        let e_rs = vec![Fp::from(2u64), Fp::from(5u64)];
        let e_audit = vec![Fp::from(3u64), Fp::from(99u64)]; // 篡改

        let roots = build_four_trees_and_extract_roots(&e_init, &e_rs, &e_ws, &e_audit);
        assert!(!roots.verify_global_conservation());
    }

    // ---- Trace MLEs 测试 ----

    #[test]
    fn test_trace_mles_from_arrays() {
        let n_ops = 2; // M=4
        let addr_read = vec![
            Fp::from(0u64),
            Fp::from(1u64),
            Fp::from(2u64),
            Fp::from(3u64),
        ];
        let val_read = vec![
            Fp::from(10u64),
            Fp::from(11u64),
            Fp::from(12u64),
            Fp::from(13u64),
        ];
        let ts_read = vec![
            Fp::from(100u64),
            Fp::from(101u64),
            Fp::from(102u64),
            Fp::from(103u64),
        ];
        let addr_write = addr_read.clone();
        let val_write = val_read.clone();
        let ts_write = ts_read.clone();

        let trace = TraceMLEs::from_arrays(
            n_ops,
            &addr_read,
            &val_read,
            &ts_read,
            &addr_write,
            &val_write,
            &ts_write,
        );

        // 在坐标 [0, 0] 处应该求值到第 0 个元素
        let point = vec![Fp::zero(), Fp::zero()];
        assert_eq!(trace.addr_read.evaluate(&point), Fp::from(0u64));
        assert_eq!(trace.val_read.evaluate(&point), Fp::from(10u64));
        assert_eq!(trace.ts_read.evaluate(&point), Fp::from(100u64));

        // 在坐标 [1, 1] 处应该求值到第 3 个元素
        let point = vec![Fp::one(), Fp::one()];
        assert_eq!(trace.addr_read.evaluate(&point), Fp::from(3u64));
        assert_eq!(trace.val_read.evaluate(&point), Fp::from(13u64));
        assert_eq!(trace.ts_read.evaluate(&point), Fp::from(103u64));
    }

    // ---- Leaf Compression 求值测试 ----

    #[test]
    fn test_eval_read_compression() {
        let n_ops = 1; // M=2
        let addr_read = vec![Fp::from(5u64), Fp::from(6u64)];
        let val_read = vec![Fp::from(10u64), Fp::from(20u64)];
        let ts_read = vec![Fp::from(100u64), Fp::from(200u64)];
        let addr_write = addr_read.clone();
        let val_write = val_read.clone();
        let ts_write = ts_read.clone();

        let trace = TraceMLEs::from_arrays(
            n_ops,
            &addr_read,
            &val_read,
            &ts_read,
            &addr_write,
            &val_write,
            &ts_write,
        );

        let delta_1 = Fp::from(2u64);
        let delta_2 = Fp::from(3u64);
        let tau = Fp::from(100u64);

        // 在坐标 [0] 处: addr=5, val=10, ts=100
        // E = 5 + 2*10 + 3*100 - 100 = 5 + 20 + 300 - 100 = 225
        let r_m = vec![Fp::zero()];
        let e = eval_read_compression(&trace, &r_m, delta_1, delta_2, tau);
        assert_eq!(e, Fp::from(225u64));

        // 在坐标 [1] 处: addr=6, val=20, ts=200
        // E = 6 + 2*20 + 3*200 - 100 = 6 + 40 + 600 - 100 = 546
        let r_m = vec![Fp::one()];
        let e = eval_read_compression(&trace, &r_m, delta_1, delta_2, tau);
        assert_eq!(e, Fp::from(546u64));
    }

    #[test]
    fn test_eval_write_compression() {
        let n_ops = 1;
        let addr_write = vec![Fp::from(1u64), Fp::from(2u64)];
        let val_write = vec![Fp::from(5u64), Fp::from(7u64)];
        let ts_write = vec![Fp::from(50u64), Fp::from(51u64)];
        let addr = addr_write.clone();
        let val = val_write.clone();
        let ts = ts_write.clone();

        let trace =
            TraceMLEs::from_arrays(n_ops, &addr, &val, &ts, &addr_write, &val_write, &ts_write);

        let delta_1 = Fp::from(1u64);
        let delta_2 = Fp::from(2u64);
        let tau = Fp::from(50u64);

        // 在坐标 [0] 处: addr=1, val=5, ts=50
        // E = 1 + 1*5 + 2*50 - 50 = 1 + 5 + 100 - 50 = 56
        let r_m = vec![Fp::zero()];
        let e = eval_write_compression(&trace, &r_m, delta_1, delta_2, tau);
        assert_eq!(e, Fp::from(56u64));
    }
}
