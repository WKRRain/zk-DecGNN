//! SPARK 内存排序排列校验 (Memory Sorting & Permutation Check)
//!
//! 严格遵循 Spartan-ZENO 设计文档第 2.2.2 节。
//!
//! ## 核心流程
//!
//! ### 第一步：高维特征的 RLC 标量化坍缩
//! - 利用转录本中的随机向量 η 进行内积坍缩
//! - val_compressed = Σ η_j · val[j]
//!
//! ### 第二步：基于地址和时间戳的双重排序轨迹
//! - 地址连续性: addr_{m+1} >= addr_m
//! - 内存时序因果律: 读写/写读强制 ts 单调
//!
//! ### 第三步：全局连乘排列校验
//! - E(t) = addr + δ₁·val + δ₂·ts − τ
//! - Π E(chron) == Π E(sorted)

use crate::curve_config::Fp;
use ark_ff::{One, Zero};

use super::transcript::Transcript;

// ============================================================================
// 内存操作元组
// ============================================================================

/// 内存操作元组
///
/// 三元组 (addr, val_compressed, ts)
/// 对应 SPARK 协议中的核心访存操作。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryTuple {
    /// 地址 (图节点 ID 或位置索引)
    pub addr: u64,
    /// 压缩后的特征值标量
    pub val_compressed: Fp,
    /// 时间戳 (单调递增的物理时钟)
    pub ts: u64,
}

impl MemoryTuple {
    /// 创建新的内存元组
    pub fn new(addr: u64, val_compressed: Fp, ts: u64) -> Self {
        Self {
            addr,
            val_compressed,
            ts,
        }
    }

    /// 计算代数压缩多项式 E(t)
    ///
    /// E(t) = addr + δ₁ · val_compressed + δ₂ · ts − τ
    pub fn to_compression_poly(&self, delta_1: Fp, delta_2: Fp, tau: Fp) -> Fp {
        Fp::from(self.addr) + delta_1 * self.val_compressed + delta_2 * Fp::from(self.ts) - tau
    }
}

// ============================================================================
// RLC 特征坍缩
// ============================================================================

/// 对高维特征向量执行 RLC 标量化坍缩
///
/// val_compressed = Σ_{j=0}^{d-1} η_j · val[j]
///
/// # 参数
/// - val: 高维特征向量 (长度为 d)
/// - eta: 随机 RLC 向量 (长度为 d, 来自转录本)
///
/// # 注意
/// 此函数与 interactive.rs 中的 compress_feature_rlc 等价，
/// 此处提供完整的模块化实现以便独立测试。
pub fn feature_rlc_compress(val: &[Fp], eta: &[Fp]) -> Fp {
    assert_eq!(
        val.len(),
        eta.len(),
        "feature_rlc_compress: dimension mismatch val={} eta={}",
        val.len(),
        eta.len()
    );
    let mut result = Fp::zero();
    for (v, e) in val.iter().zip(eta.iter()) {
        result += *v * *e;
    }
    result
}

/// 批量 RLC 特征坍缩
///
/// 对一批高维特征逐个执行 RLC 压缩，返回标量数组。
pub fn batch_feature_rlc_compress(features: &[Vec<Fp>], eta: &[Fp]) -> Vec<Fp> {
    features
        .iter()
        .map(|f| feature_rlc_compress(f, eta))
        .collect()
}

// ============================================================================
// 双重排序轨迹 (Dual-Sorted Trajectory)
// ============================================================================

/// 排序规则：地址优先，时间戳其次
///
/// 将按时间顺序发生的操作轨迹，按照"地址 (addr) 优先，时间戳 (ts) 其次"
/// 的规则重新排序。
///
/// # 参数
/// - ops: 按时间顺序的操作元组列表 (addr, val_compressed, ts)
///
/// # 返回
/// - 排序后的操作元组列表 T_sorted
pub fn dual_sort_trajectory(ops: &[MemoryTuple]) -> Vec<MemoryTuple> {
    let mut sorted = ops.to_vec();
    sorted.sort_by(|a, b| a.addr.cmp(&b.addr).then_with(|| a.ts.cmp(&b.ts)));
    sorted
}

// ============================================================================
// 时序因果律验证 (Memory Causal Monotonicity Check)
// ============================================================================

/// 验证排序轨迹的时序因果律
///
/// 在 T_sorted 中，强制断言相邻元组的物理因果逻辑：
///
/// 1. **地址连续性**：addr_{m+1} >= addr_m
/// 2. **同地址时序因果律**：
///    - 若 ts_{m+1} == ts_m 且 val_{m+1} == val_m:
///      允许 (同一节点被多个邻居并发安全读取)
///    - 若 ts_{m+1} > ts_m: 允许 (读写、写读、写写)
///    - 若 ts_{m+1} < ts_m: 非法 (时间戳回退)
///    - 若 ts_{m+1} == ts_m 且 val_{m+1} != val_m: 非法
///
/// # 返回
/// - true: 时序因果律成立
/// - false: 检测到违规的时间戳回退或状态不一致
pub fn verify_causal_monotonicity(sorted_ops: &[MemoryTuple]) -> bool {
    if sorted_ops.len() <= 1 {
        return true;
    }

    for window in sorted_ops.windows(2) {
        let prev = &window[0];
        let curr = &window[1];

        // 地址必须非递减
        if curr.addr < prev.addr {
            return false;
        }

        // 同地址：检查时序约束
        if curr.addr == prev.addr {
            // 时间戳必须非递减
            if curr.ts < prev.ts {
                return false;
            }

            // 若时间戳相同，值也必须相同 (并发读一致性)
            if curr.ts == prev.ts && curr.val_compressed != prev.val_compressed {
                return false;
            }
        }
        // 不同地址：无额外时序约束
    }

    true
}

// ============================================================================
// 全局连乘排列校验 (Grand Product Permutation Check)
// ============================================================================

/// 计算所有操作元组的 E(t) 压缩值
pub fn compute_compression_polys(
    ops: &[MemoryTuple],
    delta_1: Fp,
    delta_2: Fp,
    tau: Fp,
) -> Vec<Fp> {
    ops.iter()
        .map(|t| t.to_compression_poly(delta_1, delta_2, tau))
        .collect()
}

/// 全局连乘排列校验
///
/// Π_{t ∈ T_chronological} E(t) == Π_{t ∈ T_sorted} E(t)
///
/// # 参数
/// - e_chronological: 按时序轨迹计算的 E(t) 数组
/// - e_sorted: 按地址+时间戳排序轨迹计算的 E(t) 数组
///
/// # 返回
/// - true: 连乘等价，排列校验通过
/// - false: 数据被篡改
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

/// 批量执行全局连乘排列校验
///
/// 对四组操作流 (Init, RS, WS, Audit) 同时执行排列校验。
///
/// # 返回
/// - (bool, bool, bool, bool): 四个流的校验结果
pub fn batch_grand_product_check(
    init_chron: &[Fp],
    init_sorted: &[Fp],
    rs_chron: &[Fp],
    rs_sorted: &[Fp],
    ws_chron: &[Fp],
    ws_sorted: &[Fp],
    audit_chron: &[Fp],
    audit_sorted: &[Fp],
) -> (bool, bool, bool, bool) {
    (
        grand_product_permutation_check(init_chron, init_sorted),
        grand_product_permutation_check(rs_chron, rs_sorted),
        grand_product_permutation_check(ws_chron, ws_sorted),
        grand_product_permutation_check(audit_chron, audit_sorted),
    )
}

// ============================================================================
// SPARK 内存校验器
// ============================================================================

/// SPARK 内存校验器
///
/// 封装完整的内存排序与排列校验流程。
#[derive(Debug)]
pub struct SparkMemoryChecker {
    /// 压缩常量 (来自转录本 Γ)
    pub delta_1: Fp,
    pub delta_2: Fp,
    pub tau: Fp,
    /// RLC 降维向量
    pub eta: Vec<Fp>,
}

impl SparkMemoryChecker {
    /// 从转录本创建内存校验器
    pub fn from_transcript(transcript: &Transcript) -> Self {
        Self {
            delta_1: transcript.delta_1,
            delta_2: transcript.delta_2,
            tau: transcript.tau,
            eta: transcript.eta.clone(),
        }
    }

    /// 执行完整的内存校验流程
    ///
    /// 1. 对四组操作流分别执行排序
    /// 2. 验证排序后的时序因果律
    /// 3. 计算 E(t) 压缩多项式
    /// 4. 执行全局连乘排列校验
    ///
    /// # 返回
    /// - Ok(()): 所有校验通过
    /// - Err(String): 校验失败 (含详细错误信息)
    pub fn full_verify(
        &self,
        init_chron: &[MemoryTuple],
        rs_chron: &[MemoryTuple],
        ws_chron: &[MemoryTuple],
        audit_chron: &[MemoryTuple],
    ) -> Result<(), String> {
        // 1. 排序
        let init_sorted = dual_sort_trajectory(init_chron);
        let rs_sorted = dual_sort_trajectory(rs_chron);
        let ws_sorted = dual_sort_trajectory(ws_chron);
        let audit_sorted = dual_sort_trajectory(audit_chron);

        // 2. 时序因果律
        if !verify_causal_monotonicity(&init_sorted) {
            return Err("Init 轨迹时序因果律校验失败".to_string());
        }
        if !verify_causal_monotonicity(&rs_sorted) {
            return Err("RS 轨迹时序因果律校验失败".to_string());
        }
        if !verify_causal_monotonicity(&ws_sorted) {
            return Err("WS 轨迹时序因果律校验失败".to_string());
        }
        if !verify_causal_monotonicity(&audit_sorted) {
            return Err("Audit 轨迹时序因果律校验失败".to_string());
        }

        // 3. 计算 E(t)
        let e_init_chron =
            compute_compression_polys(init_chron, self.delta_1, self.delta_2, self.tau);
        let e_init_sorted =
            compute_compression_polys(&init_sorted, self.delta_1, self.delta_2, self.tau);
        let e_rs_chron = compute_compression_polys(rs_chron, self.delta_1, self.delta_2, self.tau);
        let e_rs_sorted =
            compute_compression_polys(&rs_sorted, self.delta_1, self.delta_2, self.tau);
        let e_ws_chron = compute_compression_polys(ws_chron, self.delta_1, self.delta_2, self.tau);
        let e_ws_sorted =
            compute_compression_polys(&ws_sorted, self.delta_1, self.delta_2, self.tau);
        let e_audit_chron =
            compute_compression_polys(audit_chron, self.delta_1, self.delta_2, self.tau);
        let e_audit_sorted =
            compute_compression_polys(&audit_sorted, self.delta_1, self.delta_2, self.tau);

        // 4. 连乘排列校验
        let (ok_init, ok_rs, ok_ws, ok_audit) = batch_grand_product_check(
            &e_init_chron,
            &e_init_sorted,
            &e_rs_chron,
            &e_rs_sorted,
            &e_ws_chron,
            &e_ws_sorted,
            &e_audit_chron,
            &e_audit_sorted,
        );

        if !ok_init {
            return Err("Init 排列校验失败".to_string());
        }
        if !ok_rs {
            return Err("RS 排列校验失败".to_string());
        }
        if !ok_ws {
            return Err("WS 排列校验失败".to_string());
        }
        if !ok_audit {
            return Err("Audit 排列校验失败".to_string());
        }

        Ok(())
    }

    /// 验证 WS 时间戳单调递增 (+1 逻辑)
    ///
    /// 对应文档 3.1 节第三步的架构修正：
    /// 写操作的时间戳必须是读取时间戳 + 1
    pub fn verify_ws_timestamp_monotonic(
        &self,
        rs_tuples: &[MemoryTuple],
        ws_tuples: &[MemoryTuple],
    ) -> bool {
        if rs_tuples.len() != ws_tuples.len() {
            return false;
        }

        for (rs, ws) in rs_tuples.iter().zip(ws_tuples.iter()) {
            // 写操作的时间戳必须 > 对应读操作的时间戳
            if ws.ts <= rs.ts {
                return false;
            }
        }
        true
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- RLC 压缩测试 ----

    #[test]
    fn test_feature_rlc_compress_basic() {
        let val: Vec<Fp> = vec![Fp::from(1u64), Fp::from(2u64), Fp::from(3u64)];
        let eta: Vec<Fp> = vec![Fp::from(2u64), Fp::from(3u64), Fp::from(5u64)];
        let compressed = feature_rlc_compress(&val, &eta);
        // 1*2 + 2*3 + 3*5 = 2 + 6 + 15 = 23
        assert_eq!(compressed, Fp::from(23u64));
    }

    #[test]
    fn test_batch_feature_rlc_compress() {
        let features = vec![
            vec![Fp::from(1u64), Fp::from(2u64)],
            vec![Fp::from(3u64), Fp::from(4u64)],
        ];
        let eta = vec![Fp::from(1u64), Fp::from(1u64)];
        let compressed = batch_feature_rlc_compress(&features, &eta);
        // [1+2, 3+4] = [3, 7]
        assert_eq!(compressed, vec![Fp::from(3u64), Fp::from(7u64)]);
    }

    #[test]
    #[should_panic]
    fn test_feature_rlc_compress_dimension_mismatch() {
        let val = vec![Fp::from(1u64)];
        let eta = vec![Fp::from(1u64), Fp::from(2u64)];
        feature_rlc_compress(&val, &eta);
    }

    // ---- 排序测试 ----

    #[test]
    fn test_dual_sort_trajectory() {
        let ops = vec![
            MemoryTuple::new(2, Fp::from(10u64), 100),
            MemoryTuple::new(1, Fp::from(20u64), 50),
            MemoryTuple::new(1, Fp::from(30u64), 10),
            MemoryTuple::new(2, Fp::from(40u64), 90),
        ];
        let sorted = dual_sort_trajectory(&ops);

        assert_eq!(sorted[0], MemoryTuple::new(1, Fp::from(30u64), 10));
        assert_eq!(sorted[1], MemoryTuple::new(1, Fp::from(20u64), 50));
        assert_eq!(sorted[2], MemoryTuple::new(2, Fp::from(40u64), 90));
        assert_eq!(sorted[3], MemoryTuple::new(2, Fp::from(10u64), 100));
    }

    // ---- 时序因果律测试 ----

    #[test]
    fn test_causal_monotonicity_valid_concurrent_reads() {
        let ops = vec![
            MemoryTuple::new(1, Fp::from(10u64), 100),
            MemoryTuple::new(1, Fp::from(10u64), 100), // 并发读
        ];
        assert!(verify_causal_monotonicity(&ops));
    }

    #[test]
    fn test_causal_monotonicity_valid_different_addr() {
        let ops = vec![
            MemoryTuple::new(1, Fp::from(10u64), 100),
            MemoryTuple::new(2, Fp::from(20u64), 50), // 不同地址，ts 可任意
        ];
        assert!(verify_causal_monotonicity(&ops));
    }

    #[test]
    fn test_causal_monotonicity_invalid_ts_rollback() {
        let ops = vec![
            MemoryTuple::new(1, Fp::from(10u64), 200),
            MemoryTuple::new(1, Fp::from(10u64), 100), // 时间戳回退
        ];
        assert!(!verify_causal_monotonicity(&ops));
    }

    #[test]
    fn test_causal_monotonicity_invalid_addr_decrease() {
        let ops = vec![
            MemoryTuple::new(2, Fp::from(10u64), 100),
            MemoryTuple::new(1, Fp::from(10u64), 200),
        ];
        assert!(!verify_causal_monotonicity(&ops));
    }

    #[test]
    fn test_causal_monotonicity_invalid_same_ts_diff_val() {
        let ops = vec![
            MemoryTuple::new(1, Fp::from(10u64), 100),
            MemoryTuple::new(1, Fp::from(99u64), 100), // 同地址同时刻不同值
        ];
        assert!(!verify_causal_monotonicity(&ops));
    }

    #[test]
    fn test_causal_monotonicity_empty_and_single() {
        assert!(verify_causal_monotonicity(&[]));
        assert!(verify_causal_monotonicity(&[MemoryTuple::new(
            1,
            Fp::from(10u64),
            100
        ),]));
    }

    // ---- 压缩多项式测试 ----

    #[test]
    fn test_memory_tuple_to_compression_poly() {
        let t = MemoryTuple::new(5, Fp::from(10u64), 42);
        let d1 = Fp::from(2u64);
        let d2 = Fp::from(3u64);
        let tau = Fp::from(100u64);
        let e = t.to_compression_poly(d1, d2, tau);
        // E = 5 + 2*10 + 3*42 - 100 = 5 + 20 + 126 - 100 = 51
        assert_eq!(e, Fp::from(51u64));
    }

    // ---- 连乘排列校验测试 ----

    #[test]
    fn test_grand_product_check_pass() {
        let e_chron: Vec<Fp> = vec![Fp::from(2u64), Fp::from(3u64), Fp::from(5u64)];
        let e_sorted: Vec<Fp> = vec![Fp::from(5u64), Fp::from(2u64), Fp::from(3u64)];
        assert!(grand_product_permutation_check(&e_chron, &e_sorted));
    }

    #[test]
    fn test_grand_product_check_fail() {
        let e_chron: Vec<Fp> = vec![Fp::from(2u64), Fp::from(3u64), Fp::from(5u64)];
        let e_bad: Vec<Fp> = vec![
            Fp::from(5u64),
            Fp::from(2u64),
            Fp::from(999u64), // 乘积不匹配
        ];
        assert!(!grand_product_permutation_check(&e_chron, &e_bad));
    }

    #[test]
    fn test_grand_product_check_length_mismatch() {
        let e_chron = vec![Fp::from(1u64), Fp::from(2u64)];
        let e_sorted = vec![Fp::from(1u64)];
        assert!(!grand_product_permutation_check(&e_chron, &e_sorted));
    }

    #[test]
    fn test_batch_grand_product_check() {
        let e1 = vec![Fp::from(2u64), Fp::from(3u64)];
        let e2 = vec![Fp::from(3u64), Fp::from(2u64)];
        let e3 = vec![Fp::from(5u64), Fp::from(7u64)];
        let e4 = vec![Fp::from(7u64), Fp::from(5u64)];
        let e5 = vec![Fp::from(11u64), Fp::from(13u64)];
        let e6 = vec![Fp::from(13u64), Fp::from(11u64)];
        let e7 = vec![Fp::from(17u64), Fp::from(19u64)];
        let e8 = vec![Fp::from(19u64), Fp::from(17u64)];

        let (ok1, ok2, ok3, ok4) =
            batch_grand_product_check(&e1, &e2, &e3, &e4, &e5, &e6, &e7, &e8);
        assert!(ok1);
        assert!(ok2);
        assert!(ok3);
        assert!(ok4);
    }

    // ---- WS 时间戳单调性测试 ----

    #[test]
    fn test_ws_timestamp_monotonic_valid() {
        let checker = SparkMemoryChecker {
            delta_1: Fp::zero(),
            delta_2: Fp::zero(),
            tau: Fp::zero(),
            eta: vec![],
        };
        let rs = vec![MemoryTuple::new(1, Fp::zero(), 10)];
        let ws = vec![
            MemoryTuple::new(1, Fp::zero(), 11), // ts > rs.ts
        ];
        assert!(checker.verify_ws_timestamp_monotonic(&rs, &ws));
    }

    #[test]
    fn test_ws_timestamp_monotonic_invalid() {
        let checker = SparkMemoryChecker {
            delta_1: Fp::zero(),
            delta_2: Fp::zero(),
            tau: Fp::zero(),
            eta: vec![],
        };
        let rs = vec![MemoryTuple::new(1, Fp::zero(), 10)];
        let ws = vec![
            MemoryTuple::new(1, Fp::zero(), 10), // ts == rs.ts (非法!)
        ];
        assert!(!checker.verify_ws_timestamp_monotonic(&rs, &ws));
    }
}
