//! 见证生成终态组装 (Prover State Finalization)
//!
//! 严格遵循 Spartan-ZENO 设计文档第 2.3 节。
//!
//! ## 封箱动作
//!
//! 系统执行"封箱"动作，组装全局 `ProverState` 核心对象，
//! 正式移交给协议阶段 3。
//!
//! ## ProverState 三大核心模块
//!
//! ### 第一大模块：张量多项式引擎 (Tensor Polynomial Engine)
//! 将阶段 2.1 产生的所有图特征张量，封装为定义在布尔超立方体上的
//! 拉格朗日多项式闭包，供阶段 3 随时代入点进行计算。
//!
//! ### 第二大模块：SPARK 内存轨迹引擎 (SPARK Memory Engine)
//! 将阶段 2 中记录的离散内存操作日志，彻底转化为定义在
//! 布尔超立方体上的连续多项式闭包。
//!
//! ### 第三大模块：全局随机转录本 (Transcript Data Structure)
//! 冻结所有通过 Fiat-Shamir 启发式抽取的"天意随机数"。

use crate::curve_config::Fp;
use crate::polynomial::MLEPolynomial;
use ark_ff::Zero;

use super::binary_tree::TraceMLEs;
use super::transcript::Transcript;

// ============================================================================
// 张量多项式引擎 (Tensor Polynomial Engine)
// ============================================================================

/// 张量多项式引擎
///
/// 包装阶段 2.1 产生的所有图特征张量，以拉格朗日多项式闭包形式存储。
/// 闭包在阶段 3 的 Sum-check 和 PCS 中被反复调用求值。
#[derive(Clone)]
pub struct TensorPolyEngine {
    /// X_base 张量的多项式闭包
    pub x_base_closure: TensorClosure<2>,

    /// 扩散状态 S^(k) 的全局三维多项式闭包 (z, x, y)
    pub s_global_closure: GlobalStateClosure,

    /// 聚合状态 Z^(k) 的全局三维多项式闭包 (z, x, y)
    pub z_global_closure: GlobalStateClosure,

    /// 后置 ReLU 符号位多项式闭包 (x, w) — 隐藏层维度
    pub b_sign_closure: Option<TensorClosure<2>>,

    /// 前置 ReLU 符号位多项式闭包 (x, w_pre)
    pub b_sign_pre_closure: Option<TensorClosure<2>>,

    // ---- 张量维度元数据 ----
    /// 迭代层数 K
    pub n_k: usize,
    /// 有效节点数 N_real
    pub n_real: usize,
    /// 有效输入特征维度 d
    pub d_effective: usize,
    /// 有效隐藏层维度 h
    pub h_effective: usize,
    /// 有效前置隐藏层维度 h_pre
    pub h_pre_effective: usize,
}

/// 通用张量多项式闭包
///
/// 泛型参数 D: 张量的维度 (2 表示矩阵, 3 表示带时间轴的张量)
/// 存储评估值数组和布尔变量维度，提供闭包式求值。
#[derive(Clone)]
pub struct TensorClosure<const D: usize> {
    /// 离散评估值数组 (一维平铺)
    pub evaluations: Vec<Fp>,
    /// 各维度的布尔变量数 [n_var_0, n_var_1, ...]
    pub num_vars: [usize; D],
    /// 各维度的有效边界 (用于 ZENO 剪枝)
    pub effective_bounds: [usize; D],
}

impl<const D: usize> TensorClosure<D> {
    /// 创建张量闭包
    pub fn new(evaluations: Vec<Fp>, num_vars: [usize; D], effective_bounds: [usize; D]) -> Self {
        Self {
            evaluations,
            num_vars,
            effective_bounds,
        }
    }

    /// 在随机坐标点求值
    ///
    /// 计算 f(r_0, r_1, ..., r_{D-1})
    /// ZENO 剪枝仅遍历有效边界内的项。
    pub fn evaluate(&self, points: &[&[Fp]]) -> Fp {
        assert_eq!(points.len(), D, "TensorClosure: dimension mismatch");
        for (d, point) in points.iter().enumerate().take(D) {
            assert_eq!(
                point.len(),
                self.num_vars[d],
                "TensorClosure: var count mismatch at dim {}",
                d
            );
        }

        MLEPolynomial::evaluate_direct(
            &self.evaluations,
            &self.num_vars,
            points,
            &self.effective_bounds,
        )
    }

    /// 获取有效边界内元素总数
    pub fn effective_size(&self) -> usize {
        self.effective_bounds.iter().product()
    }

    /// 获取物理总大小 (含 Padding)
    pub fn physical_size(&self) -> usize {
        self.num_vars.iter().map(|v| 1usize << v).product()
    }
}

/// 全局状态多项式闭包 (三维：时间 × 行 × 列)
#[derive(Clone)]
pub struct GlobalStateClosure {
    /// 离散评估值数组 (一维平铺)
    pub evaluations: Vec<Fp>,
    /// 三维布尔变量数: [n_k, n_v, n_d]
    pub num_vars: [usize; 3],
    /// 有效边界: [K, N_real, d]
    pub effective_bounds: [usize; 3],
}

impl GlobalStateClosure {
    /// 创建全局状态闭包
    pub fn new(evaluations: Vec<Fp>, num_vars: [usize; 3], effective_bounds: [usize; 3]) -> Self {
        Self {
            evaluations,
            num_vars,
            effective_bounds,
        }
    }

    /// 在三维坐标点求值: (r_z, r_x, r_y)
    pub fn evaluate(&self, r_z: &[Fp], r_x: &[Fp], r_y: &[Fp]) -> Fp {
        let points: [&[Fp]; 3] = [r_z, r_x, r_y];
        MLEPolynomial::evaluate_direct(
            &self.evaluations,
            &self.num_vars,
            &points,
            &self.effective_bounds,
        )
    }

    /// 获取有效边界内元素总数
    pub fn effective_size(&self) -> usize {
        self.effective_bounds.iter().product()
    }
}

impl TensorPolyEngine {
    /// 创建空的张量多项式引擎
    pub fn new(n_k: usize, n_real: usize, d_effective: usize, h_effective: usize) -> Self {
        Self {
            x_base_closure: TensorClosure {
                evaluations: Vec::new(),
                num_vars: [0, 0],
                effective_bounds: [0, 0],
            },
            s_global_closure: GlobalStateClosure {
                evaluations: Vec::new(),
                num_vars: [0, 0, 0],
                effective_bounds: [0, 0, 0],
            },
            z_global_closure: GlobalStateClosure {
                evaluations: Vec::new(),
                num_vars: [0, 0, 0],
                effective_bounds: [0, 0, 0],
            },
            b_sign_closure: None,
            b_sign_pre_closure: None,
            n_k,
            n_real,
            d_effective,
            h_effective,
            h_pre_effective: h_effective,
        }
    }

    /// 注册 X_base 张量闭包
    ///
    /// # 参数
    /// - x_base: 形状为 N_max × D_max 的一维平铺数组
    /// - n_v: 行布尔变量数 (log2(N_max))
    /// - n_d: 列布尔变量数 (log2(D_max))
    /// - d_eff: 有效列边界
    pub fn register_x_base(&mut self, x_base: &[Fp], n_v: usize, n_d: usize, d_eff: usize) {
        self.x_base_closure = TensorClosure {
            evaluations: x_base.to_vec(),
            num_vars: [n_v, n_d],
            effective_bounds: [self.n_real, d_eff],
        };
    }

    /// 注册扩散状态 S 的全局张量闭包
    ///
    /// # 参数
    /// - s_states: K × N_max × D_max 一维平铺数组 (字典序: z, x, y)
    /// - n_k: 时间布尔变量数
    /// - n_v: 行布尔变量数
    /// - n_d: 列布尔变量数
    pub fn register_s_global(&mut self, s_states: &[Fp], n_k: usize, n_v: usize, n_d: usize) {
        self.s_global_closure = GlobalStateClosure {
            evaluations: s_states.to_vec(),
            num_vars: [n_k, n_v, n_d],
            effective_bounds: [self.n_k, self.n_real, self.d_effective],
        };
    }

    /// 注册聚合状态 Z 的全局张量闭包
    pub fn register_z_global(&mut self, z_states: &[Fp], n_k: usize, n_v: usize, n_d: usize) {
        self.z_global_closure = GlobalStateClosure {
            evaluations: z_states.to_vec(),
            num_vars: [n_k, n_v, n_d],
            effective_bounds: [self.n_k, self.n_real, self.d_effective],
        };
    }

    /// 注册后置 ReLU 符号位张量闭包
    ///
    /// # 参数
    /// - sign_matrix: N_max × H_max 符号位矩阵 (一维平铺)
    /// - n_v: 行布尔变量数
    /// - n_h_max: 隐藏层列布尔变量数 (log2(H_max))
    pub fn register_b_sign(&mut self, sign_matrix: &[Fp], n_v: usize, n_h_max: usize) {
        self.b_sign_closure = Some(TensorClosure {
            evaluations: sign_matrix.to_vec(),
            num_vars: [n_v, n_h_max],
            effective_bounds: [self.n_real, self.h_effective],
        });
    }

    /// 注册前置 ReLU 符号位张量闭包
    pub fn register_b_sign_pre(&mut self, sign_matrix: &[Fp], n_v: usize, n_h_pre: usize) {
        self.b_sign_pre_closure = Some(TensorClosure {
            evaluations: sign_matrix.to_vec(),
            num_vars: [n_v, n_h_pre],
            effective_bounds: [self.n_real, self.h_pre_effective],
        });
    }

    /// 求值 X_base 多项式
    pub fn eval_x_base(&self, r_x: &[Fp], r_y: &[Fp]) -> Fp {
        self.x_base_closure.evaluate(&[r_x, r_y])
    }

    /// 求值扩散全局多项式 S_global
    pub fn eval_s_global(&self, r_z: &[Fp], r_x: &[Fp], r_y: &[Fp]) -> Fp {
        self.s_global_closure.evaluate(r_z, r_x, r_y)
    }

    /// 求值聚合全局多项式 Z_global
    pub fn eval_z_global(&self, r_z: &[Fp], r_x: &[Fp], r_y: &[Fp]) -> Fp {
        self.z_global_closure.evaluate(r_z, r_x, r_y)
    }
}

// ============================================================================
// SPARK 内存轨迹引擎 (SPARK Memory Engine)
// ============================================================================

/// SPARK 内存轨迹引擎
///
/// 封装 GKR 二叉树架构下的全部访存轨迹元组闭包，
/// 支持叶子和中间层的动态重构求值。
#[derive(Clone)]
pub struct SparkMemoryEngine {
    /// 读/写轨迹的多线性扩展
    pub trace_mles: TraceMLEs,

    /// 四棵二叉树的根节点声明
    pub v_init_root: Fp,
    pub v_rs_root: Fp,
    pub v_ws_root: Fp,
    pub v_audit_root: Fp,

    /// 总访存操作数对应的布尔维度 n_ops
    pub n_ops: usize,
}

impl SparkMemoryEngine {
    /// 创建内存轨迹引擎
    pub fn new(
        trace_mles: TraceMLEs,
        v_init_root: Fp,
        v_rs_root: Fp,
        v_ws_root: Fp,
        v_audit_root: Fp,
        n_ops: usize,
    ) -> Self {
        Self {
            trace_mles,
            v_init_root,
            v_rs_root,
            v_ws_root,
            v_audit_root,
            n_ops,
        }
    }

    /// 验证全局内存守恒
    pub fn verify_global_conservation(&self) -> bool {
        self.v_init_root * self.v_ws_root == self.v_rs_root * self.v_audit_root
    }

    /// 求值读取操作流的压缩多项式 E_RS
    pub fn eval_rs_compression(&self, r_m: &[Fp], delta_1: Fp, delta_2: Fp, tau: Fp) -> Fp {
        super::binary_tree::eval_read_compression(&self.trace_mles, r_m, delta_1, delta_2, tau)
    }

    /// 求值写入操作流的压缩多项式 E_WS
    pub fn eval_ws_compression(&self, r_m: &[Fp], delta_1: Fp, delta_2: Fp, tau: Fp) -> Fp {
        super::binary_tree::eval_write_compression(&self.trace_mles, r_m, delta_1, delta_2, tau)
    }
}

// ============================================================================
// ProverState 终态组装
// ============================================================================

/// 全局证明者终态 (ProverState)
///
/// 封装阶段 2 完整见证生成的全部核心对象，
/// 作为阶段 3 证明生成的单一入参。
#[derive(Clone)]
pub struct ProverState {
    /// 第一大模块：张量多项式引擎
    pub tensor_engine: TensorPolyEngine,

    /// 第二大模块：SPARK 内存轨迹引擎
    pub spark_engine: SparkMemoryEngine,

    /// 第三大模块：全局随机转录本 (Fiat-Shamir 冻结)
    pub transcript: Transcript,

    // ---- 容器维度元数据 ----
    /// 超级容器行上限 N_max
    pub n_max: usize,
    /// 超级容器列上限 D_max
    pub d_max: usize,
    /// 隐藏层维度上限 H_max
    pub h_max: usize,
    /// 分类数 c
    pub num_classes: usize,

    // ---- 最终输出 ----
    /// 掩码过滤后的最终 Logits 预测明文 Y_tilde
    pub y_tilde: Vec<Fp>,
}

impl ProverState {
    /// 从三大模块组装全局 ProverState
    pub fn new(
        tensor_engine: TensorPolyEngine,
        spark_engine: SparkMemoryEngine,
        transcript: Transcript,
        n_max: usize,
        d_max: usize,
        h_max: usize,
        num_classes: usize,
        y_tilde: Vec<Fp>,
    ) -> Self {
        Self {
            tensor_engine,
            spark_engine,
            transcript,
            n_max,
            d_max,
            h_max,
            num_classes,
            y_tilde,
        }
    }

    /// 验证 ProverState 内部一致性
    ///
    /// 检查各模块间的维度对齐、内存守恒等。
    pub fn verify_internal_consistency(&self) -> Result<(), String> {
        // 1. 内存守恒检查
        if !self.spark_engine.verify_global_conservation() {
            return Err("SPARK 全局内存守恒校验失败".to_string());
        }

        // 2. 张量闭包非空检查
        if self.tensor_engine.x_base_closure.evaluations.is_empty() {
            return Err("X_base 张量闭包未注册".to_string());
        }
        if self.tensor_engine.s_global_closure.evaluations.is_empty() {
            return Err("S_global 张量闭包未注册".to_string());
        }
        if self.tensor_engine.z_global_closure.evaluations.is_empty() {
            return Err("Z_global 张量闭包未注册".to_string());
        }

        // 3. 输出维度检查
        let expected_y_len = self.n_max * self.num_classes;
        if self.y_tilde.len() != expected_y_len {
            return Err(format!(
                "Y_tilde 长度 ({}) 与 N_max * c ({} * {} = {}) 不匹配",
                self.y_tilde.len(),
                self.n_max,
                self.num_classes,
                expected_y_len
            ));
        }

        Ok(())
    }

    /// 提取公开展示 (Public Instance) 的 Logits 明文
    pub fn get_public_logits(&self) -> &[Fp] {
        &self.y_tilde
    }
}

// ============================================================================
// 辅助函数：从 2D 矩阵平铺构建张量闭包
// ============================================================================

/// 将 2D 矩阵 (行主序) 平铺为 MLE 评估值数组
///
/// 行维度向上取整到 2^{n_v}, 列维度向上取整到 2^{n_d}
pub fn flatten_matrix_for_mle(matrix: &[Vec<Fp>], n_v: usize, n_d: usize) -> Vec<Fp> {
    let rows = 1usize << n_v;
    let cols = 1usize << n_d;

    let mut flat = vec![Fp::zero(); rows * cols];

    for i in 0..matrix.len() {
        for j in 0..matrix[i].len() {
            flat[i * cols + j] = matrix[i][j];
        }
    }

    flat
}

/// 将三维张量 (K × N × D) 平铺为 MLE 评估值数组
///
/// 字典序: z, x, y (时间、空间、特征)
pub fn flatten_3d_tensor_for_mle(
    tensor: &[Vec<Vec<Fp>>],
    n_k: usize,
    n_v: usize,
    n_d: usize,
) -> Vec<Fp> {
    let k_size = 1usize << n_k;
    let rows = 1usize << n_v;
    let cols = 1usize << n_d;

    let mut flat = vec![Fp::zero(); k_size * rows * cols];

    for (k, slice) in tensor.iter().enumerate() {
        for (i, row) in slice.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                let idx = k * (rows * cols) + i * cols + j;
                flat[idx] = *value;
            }
        }
    }

    flat
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::One;

    // ---- TensorClosure 测试 ----

    #[test]
    fn test_tensor_closure_2d() {
        // 2×2 矩阵: [[1, 2], [3, 4]]
        // n_v = 1, n_d = 1
        let evals = vec![
            Fp::from(1u64),
            Fp::from(2u64),
            Fp::from(3u64),
            Fp::from(4u64),
        ];
        let closure = TensorClosure::<2>::new(evals.clone(), [1, 1], [2, 2]);

        // 在 (0, 0) 处求值 -> 第 1 行第 1 列 = 1
        let r_x = vec![Fp::zero()];
        let r_y = vec![Fp::zero()];
        let val = closure.evaluate(&[&r_x, &r_y]);
        assert_eq!(val, Fp::from(1u64));

        // 在 (1, 1) 处求值 -> 第 2 行第 2 列 = 4
        let r_x = vec![Fp::one()];
        let r_y = vec![Fp::one()];
        let val = closure.evaluate(&[&r_x, &r_y]);
        assert_eq!(val, Fp::from(4u64));
    }

    #[test]
    fn test_tensor_closure_physical_size() {
        let closure = TensorClosure::<2>::new(vec![Fp::zero(); 8], [2, 1], [3, 2]);
        assert_eq!(closure.physical_size(), 8);
        assert_eq!(closure.effective_size(), 6);
    }

    // ---- GlobalStateClosure 测试 ----

    #[test]
    fn test_global_state_closure() {
        // K=2, N=1, D=1 -> 2×1×1 = 2 个元素
        let evals = vec![Fp::from(5u64), Fp::from(7u64)];
        let closure = GlobalStateClosure::new(
            evals,
            [1, 0, 0], // n_k=1 (log2(2)), n_v=0 (log2(1)), n_d=0
            [2, 1, 1], // 有效边界
        );

        // 在 k=0 处求值 -> 5
        let r_z = vec![Fp::zero()];
        let r_x: Vec<Fp> = vec![];
        let r_y: Vec<Fp> = vec![];
        let val = closure.evaluate(&r_z, &r_x, &r_y);
        assert_eq!(val, Fp::from(5u64));

        // 在 k=1 处求值 -> 7
        let r_z = vec![Fp::one()];
        let val = closure.evaluate(&r_z, &r_x, &r_y);
        assert_eq!(val, Fp::from(7u64));
    }

    // ---- flatten 函数测试 ----

    #[test]
    fn test_flatten_matrix_for_mle() {
        // 2 行 2 列
        let matrix = vec![
            vec![Fp::from(1u64), Fp::from(2u64)],
            vec![Fp::from(3u64), Fp::from(4u64)],
        ];
        let flat = flatten_matrix_for_mle(&matrix, 1, 1); // n_v=1, n_d=1

        assert_eq!(flat.len(), 4);
        assert_eq!(
            flat,
            vec![
                Fp::from(1u64),
                Fp::from(2u64),
                Fp::from(3u64),
                Fp::from(4u64),
            ]
        );
    }

    #[test]
    fn test_flatten_matrix_padding() {
        // 1 行 1 列, 但 n_v=1, n_d=1 -> 物理 2×2
        let matrix = vec![vec![Fp::from(42u64)]];
        let flat = flatten_matrix_for_mle(&matrix, 1, 1);

        assert_eq!(flat.len(), 4);
        assert_eq!(flat[0], Fp::from(42u64));
        assert_eq!(flat[1], Fp::zero()); // padding
        assert_eq!(flat[2], Fp::zero());
        assert_eq!(flat[3], Fp::zero());
    }

    #[test]
    fn test_flatten_3d_tensor() {
        // K=2, N=1, D=2
        let tensor = vec![
            vec![vec![Fp::from(1u64), Fp::from(2u64)]],
            vec![vec![Fp::from(3u64), Fp::from(4u64)]],
        ];
        let flat = flatten_3d_tensor_for_mle(&tensor, 1, 0, 1); // n_v=0 -> rows=1

        // 字典序: k*rows*cols + i*cols + j
        // k=0: [1, 2]
        // k=1: [3, 4]
        assert_eq!(flat.len(), 4);
        assert_eq!(
            flat,
            vec![
                Fp::from(1u64),
                Fp::from(2u64),
                Fp::from(3u64),
                Fp::from(4u64),
            ]
        );
    }

    // ---- ProverState 一致性测试 ----

    #[test]
    fn test_prover_state_consistency() {
        // 构建最小化 ProverState 并验证内部一致性
        let n_k = 2;
        let n_real = 3;
        let d_eff = 4;
        let h_eff = 8;
        let n_v = 2; // N_max = 4
        let n_d = 2; // D_max = 4
        let n_max = 4;
        let d_max = 4;
        let h_max = 8;
        let num_classes = 3;

        // X_base: 4×4 = 16 个元素
        let x_base: Vec<Fp> = (0..16).map(|i| Fp::from(i as u64)).collect();
        // S: K × N_max × D_max = 2×4×4 = 32
        let s_states: Vec<Fp> = (0..32).map(|i| Fp::from(i as u64)).collect();
        // Z: 2×4×4 = 32
        let z_states: Vec<Fp> = (0..32).map(|i| Fp::from(i as u64)).collect();

        let mut tensor_engine = TensorPolyEngine::new(n_k, n_real, d_eff, h_eff);
        tensor_engine.register_x_base(&x_base, n_v, n_d, d_eff);
        tensor_engine.register_s_global(&s_states, 1, n_v, n_d);
        tensor_engine.register_z_global(&z_states, 1, n_v, n_d);

        // SPARK engine (占位)
        let spark_engine = SparkMemoryEngine::new(
            TraceMLEs::from_arrays(0, &[], &[], &[], &[], &[], &[]),
            Fp::one(),
            Fp::one(),
            Fp::one(),
            Fp::one(),
            0,
        );

        let transcript = Transcript::default();
        let y_tilde = vec![Fp::zero(); n_max * num_classes];

        let state = ProverState::new(
            tensor_engine,
            spark_engine,
            transcript,
            n_max,
            d_max,
            h_max,
            num_classes,
            y_tilde,
        );

        assert!(state.verify_internal_consistency().is_ok());
    }
}
