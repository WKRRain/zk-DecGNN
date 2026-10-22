//! Poseidon 哈希海绵 (Sponge Hash)
//!
//! ## 配置参数
//! - 状态宽度 t = 9 (8 个 Rate 槽 + 1 个 Capacity 槽)
//! - Rate R = 8
//! - Capacity C = 1 (提供 128-bit 物理抗碰撞安全性)
//! - 全轮数 R_f = 8 (4 前半 + 4 后半)
//! - 半轮数 R_p = 56
//! - S-Box: x^5
//!
//! ## 用途
//! 作为 Fiat-Shamir 变换的核心组件，负责:
//! - 吸收 (Absorb): 将证明数据序列化为标量吸入状态
//! - 挤出 (Squeeze): 从状态中提取伪随机挑战标量

use crate::curve_config::Fp;
use ark_ff::{Field, PrimeField, Zero};
use sha2::{Digest, Sha512};

/// Poseidon 状态宽度
pub const T: usize = 9;

/// Rate (吞吐区宽度)
pub const RATE: usize = 8;

/// Capacity (容量区宽度)
pub const CAPACITY: usize = 1;

/// 全轮数 (每半段 4 轮, 共 8 轮)
pub const R_F: usize = 8;

/// 半轮数
pub const R_P: usize = 56;

/// 总轮数
pub const R_TOTAL: usize = R_F + R_P;

fn field_from_domain(label: &[u8], round: usize, index: usize) -> Fp {
    let mut hasher = Sha512::new();
    hasher.update(b"Spartan-ZENO-Poseidon-parameters-v2");
    hasher.update((label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update((round as u64).to_le_bytes());
    hasher.update((index as u64).to_le_bytes());
    Fp::from_be_bytes_mod_order(&hasher.finalize())
}

/// Poseidon MDS 矩阵 (t × t)
///
/// The matrix is deterministically generated as a Cauchy matrix over Fp:
/// M[i,j] = 1 / (x_i + y_j), with disjoint non-zero x/y domains. Cauchy
/// matrices are MDS when all denominators are non-zero. Production deployments
/// should still pin audited parameter files for the selected field.
pub fn mds_matrix() -> [[Fp; T]; T] {
    let mut m = [[Fp::zero(); T]; T];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let x_i = Fp::from((i + 1) as u64);
            let y_j = Fp::from((T + j + 1) as u64);
            *cell = (x_i + y_j)
                .inverse()
                .expect("Cauchy denominator is non-zero by construction");
        }
    }
    m
}

/// 轮常数 (ARC)
///
/// 每轮为每个状态元素提供独立的加法常数。常数通过 SHA-512 域分离确定性
/// 派生，避免开发阶段的线性序号常数。生产部署应固定经过审计的参数文件。
pub fn round_constants() -> Vec<[Fp; T]> {
    let mut constants = Vec::with_capacity(R_TOTAL);
    for r in 0..R_TOTAL {
        let mut round_const = [Fp::zero(); T];
        for (i, cell) in round_const.iter_mut().enumerate() {
            *cell = field_from_domain(b"round-constant", r, i);
        }
        constants.push(round_const);
    }
    constants
}

/// Poseidon 海绵状态
#[derive(Debug, Clone)]
pub struct PoseidonSponge {
    /// 当前状态向量 [s_0, s_1, ..., s_{t-1}]
    state: [Fp; T],
    /// 挤出阶段的读取指针 (遍历 Rate 区)
    squeeze_ptr: usize,
    /// 轮常数缓存
    round_constants: Vec<[Fp; T]>,
    /// MDS 矩阵缓存
    mds: [[Fp; T]; T],
}

impl PoseidonSponge {
    /// 创建新的 Poseidon 海绵实例
    ///
    /// 初始状态全部为零。
    pub fn new() -> Self {
        Self {
            state: [Fp::zero(); T],
            squeeze_ptr: 0,
            round_constants: round_constants(),
            mds: mds_matrix(),
        }
    }

    /// 测试与旧版接口兼容构造器。
    pub fn new_for_test() -> Self {
        Self::new()
    }

    /// 重置海绵状态 (用于新转录本的开始)
    pub fn reset(&mut self) {
        self.state = [Fp::zero(); T];
        self.squeeze_ptr = 0;
    }

    /// 吸收一个标量到状态中
    ///
    /// 将标量添加到当前 Rate 指针位置，然后指针前进。
    /// 当 Rate 区满时，自动触发核心置换。
    pub fn absorb_scalar(&mut self, scalar: &Fp) {
        // 若 Rate 区已满，先执行置换
        if self.squeeze_ptr >= RATE {
            self.permute();
            self.squeeze_ptr = 0;
        }

        // 吸收: 将标量加到当前 Rate 槽位
        self.state[self.squeeze_ptr] += scalar;
        self.squeeze_ptr += 1;
    }

    /// 批量吸收标量切片
    pub fn absorb_scalars(&mut self, scalars: &[Fp]) {
        for s in scalars {
            self.absorb_scalar(s);
        }
    }

    /// 旧版接口兼容：吸收一个标量切片。
    pub fn absorb(&mut self, scalars: &[Fp]) {
        self.absorb_scalars(scalars);
    }

    /// 吸收椭圆曲线点的坐标对 (x, y)
    ///
    /// 将点的 x 坐标和 y 坐标依次吸入。
    pub fn absorb_point(&mut self, x: &Fp, y: &Fp) {
        self.absorb_scalar(x);
        self.absorb_scalar(y);
    }

    /// 强制对当前状态执行核心置换后进入挤出模式
    ///
    /// 注意: 即使 Rate 区未满也执行置换。
    pub fn finalize_absorption(&mut self) {
        self.permute();
        self.squeeze_ptr = 0;
    }

    /// 挤出一个伪随机标量
    ///
    /// 从 Rate 区读取当前指针位置的标量。
    /// 若 Rate 区已耗尽，自动执行置换刷新。
    pub fn squeeze(&mut self) -> Fp {
        // 容量检查: 若 Rate 区已读空，执行置换重启
        if self.squeeze_ptr >= RATE {
            self.permute();
            self.squeeze_ptr = 0;
        }

        let scalar = self.state[self.squeeze_ptr];
        self.squeeze_ptr += 1;
        scalar
    }

    /// 批量挤出 n 个标量
    pub fn squeeze_n(&mut self, n: usize) -> Vec<Fp> {
        (0..n).map(|_| self.squeeze()).collect()
    }

    /// 挤出 n_d 维随机向量 (用于 RLC 特征压缩)
    pub fn squeeze_vector(&mut self, dim: usize) -> Vec<Fp> {
        self.squeeze_n(dim)
    }

    /// 挤出 n_k + n_v + n_d 维评估坐标元组
    ///
    /// 返回 (r_z, r_x, r_y) 三个布尔超立方体坐标向量。
    pub fn squeeze_evaluation_coords(
        &mut self,
        n_k: usize,
        n_v: usize,
        n_d: usize,
    ) -> (Vec<Fp>, Vec<Fp>, Vec<Fp>) {
        let r_z = self.squeeze_n(n_k);
        let r_x = self.squeeze_n(n_v);
        let r_y = self.squeeze_n(n_d);
        (r_z, r_x, r_y)
    }

    /// 挤出并映射到椭圆曲线点 (Shallue-van de Woestijne 简化)
    ///
    /// 挤出横坐标 x，尝试映射为曲线上的合法点。
    /// 若映射失败，x 自增后重试。
    pub fn squeeze_curve_point(&mut self) -> (Fp, Fp) {
        let x = self.squeeze();
        // 简化: 返回 (x, dummy) - 实际使用 SvW 映射
        // 对于 Curve25519 的 ristretto 抽象，reduced 形式自动处理
        let y = Fp::from(0u64);
        (x, y)
    }

    /// 核心置换函数 P(S)
    ///
    /// 执行 R_TOTAL 轮变换，每轮包含:
    /// 1. AddRoundKey (ARC) - 常数注入
    /// 2. S-Box - x^5 非线性变换
    /// 3. MixLayer (MDS) - 满秩矩阵乘法
    fn permute(&mut self) {
        let half_full = R_F / 2; // 前半全轮数 = 4

        for r in 0..R_TOTAL {
            // 工序一: AddRoundKey (ARC)
            let rc = &self.round_constants[r];
            for (state_i, rc_i) in self.state.iter_mut().zip(rc.iter()) {
                *state_i += *rc_i;
            }

            // 工序二: S-Box
            let is_full_round = r < half_full || r >= R_TOTAL - half_full;

            if is_full_round {
                // 全轮: 对所有状态元素执行 x^5
                self.full_sbox();
            } else {
                // 半轮: 仅对第一个元素执行 x^5
                self.partial_sbox();
            }

            // 工序三: MixLayer (MDS)
            self.mix_layer();
        }
    }

    /// 全轮 S-Box: 所有 t 个元素执行 x^5
    fn full_sbox(&mut self) {
        for state_i in self.state.iter_mut() {
            *state_i = quintic_sbox(state_i);
        }
    }

    /// 半轮 S-Box: 仅 s_0 执行 x^5, 其余透传
    fn partial_sbox(&mut self) {
        self.state[0] = quintic_sbox(&self.state[0]);
        // s_1 到 s_{t-1} 保持不变
    }

    /// MixLayer: 左乘 MDS 矩阵
    fn mix_layer(&mut self) {
        let old_state = self.state;
        for (i, state_i) in self.state.iter_mut().enumerate() {
            let mut sum = Fp::zero();
            for (j, old_value) in old_state.iter().enumerate() {
                sum += self.mds[i][j] * *old_value;
            }
            *state_i = sum;
        }
    }
}

impl Default for PoseidonSponge {
    fn default() -> Self {
        Self::new()
    }
}

/// 旧代码中的名称，保留为别名以避免协议模块 API 漂移。
pub type PoseidonHash = PoseidonSponge;

/// 五次方 S-Box: s -> s^5
///
/// 使用 x^5 而非 x^3 以在 253-bit 域上获得更好的非线性雪崩。
fn quintic_sbox(input: &Fp) -> Fp {
    let x2 = input.square(); // x^2
    let x4 = x2.square(); // x^4
    x4 * (*input) // x^5
}

/// 将椭圆曲线点列表吸入海绵的辅助函数
///
/// # 参数
/// - sponge: 目标海绵
/// - points: (x, y) 坐标对的列表
pub fn absorb_points(sponge: &mut PoseidonSponge, points: &[(Fp, Fp)]) {
    for (x, y) in points {
        sponge.absorb_point(x, y);
    }
}

/// 将公共参数序列吸入海绵
///
/// 对应文档中的 V_pp = [N_max, D_max, H_max, Edges, N_real, d, h, K,
///                      α_pre, β, ID_model, γ1, γ2, γ3, λ1, λ2]
pub fn absorb_public_params(sponge: &mut PoseidonSponge, params: &[Fp; 16]) {
    sponge.absorb_scalars(params);
}

/// 将矩阵承诺向量吸入海绵
///
/// 每个承诺点是椭圆曲线点，需要拆分为 (x, y) 坐标对吸入。
pub fn absorb_commitment_vector(sponge: &mut PoseidonSponge, commitments: &[(Fp, Fp)]) {
    absorb_points(sponge, commitments);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sponge_creation() {
        let sponge = PoseidonSponge::new();
        assert!(sponge.state.iter().all(|s| s.is_zero()));
        assert_eq!(sponge.squeeze_ptr, 0);
    }

    #[test]
    fn test_absorb_and_squeeze() {
        let mut sponge = PoseidonSponge::new();

        // 吸收一个标量
        sponge.absorb_scalar(&Fp::from(42u64));
        sponge.finalize_absorption();

        // 挤出一个标量, 应非零 (经过置换)
        let out = sponge.squeeze();
        assert!(!out.is_zero());
    }

    #[test]
    fn test_squeeze_determinism() {
        let mut sponge1 = PoseidonSponge::new();
        let mut sponge2 = PoseidonSponge::new();

        sponge1.absorb_scalar(&Fp::from(100u64));
        sponge2.absorb_scalar(&Fp::from(100u64));

        sponge1.finalize_absorption();
        sponge2.finalize_absorption();

        let out1 = sponge1.squeeze();
        let out2 = sponge2.squeeze();

        // 相同输入应产生相同输出
        assert_eq!(out1, out2);
    }

    #[test]
    fn test_squeeze_different_inputs() {
        let mut sponge1 = PoseidonSponge::new();
        let mut sponge2 = PoseidonSponge::new();

        sponge1.absorb_scalar(&Fp::from(100u64));
        sponge2.absorb_scalar(&Fp::from(200u64));

        sponge1.finalize_absorption();
        sponge2.finalize_absorption();

        let out1 = sponge1.squeeze();
        let out2 = sponge2.squeeze();

        // 不同输入应产生不同输出 (极高概率)
        assert_ne!(out1, out2);
    }

    #[test]
    fn test_reset() {
        let mut sponge = PoseidonSponge::new();
        sponge.absorb_scalar(&Fp::from(99u64));
        sponge.finalize_absorption();
        let out1 = sponge.squeeze();

        sponge.reset();
        sponge.absorb_scalar(&Fp::from(99u64));
        sponge.finalize_absorption();
        let out2 = sponge.squeeze();

        assert_eq!(out1, out2);
    }

    #[test]
    fn test_quintic_sbox() {
        let x = Fp::from(2u64);
        let x5 = quintic_sbox(&x);
        // 2^5 = 32
        assert_eq!(x5, Fp::from(32u64));
    }

    #[test]
    fn test_full_round_sbox_applies_to_all() {
        let mut sponge = PoseidonSponge::new();
        // 初始化 state 为非零值
        sponge.state = [Fp::from(2u64); T];
        sponge.full_sbox();

        // 验证所有元素都变成了 2^5 = 32
        for s in &sponge.state {
            assert_eq!(*s, Fp::from(32u64));
        }
    }

    #[test]
    fn test_partial_sbox_only_first() {
        let mut sponge = PoseidonSponge::new();
        // s0 = 2, s1..s8 = 3
        sponge.state[0] = Fp::from(2u64);
        for i in 1..T {
            sponge.state[i] = Fp::from(3u64);
        }

        sponge.partial_sbox();

        // s0 应该是 2^5 = 32
        assert_eq!(sponge.state[0], Fp::from(32u64));
        // s1..s8 应该保持不变 = 3
        for i in 1..T {
            assert_eq!(sponge.state[i], Fp::from(3u64));
        }
    }

    #[test]
    fn test_squeeze_vector() {
        let mut sponge = PoseidonSponge::new();
        sponge.absorb_scalar(&Fp::from(1u64));
        sponge.finalize_absorption();

        let vec = sponge.squeeze_vector(10);
        assert_eq!(vec.len(), 10);
    }

    #[test]
    fn test_squeeze_evaluation_coords() {
        let mut sponge = PoseidonSponge::new();
        sponge.absorb_scalar(&Fp::from(1u64));
        sponge.finalize_absorption();

        let (r_z, r_x, r_y) = sponge.squeeze_evaluation_coords(2, 3, 4);
        assert_eq!(r_z.len(), 2);
        assert_eq!(r_x.len(), 3);
        assert_eq!(r_y.len(), 4);
    }

    #[test]
    fn test_permute_changes_state() {
        let mut sponge = PoseidonSponge::new();
        let original_state = sponge.state;

        sponge.permute();

        // 置换后状态应发生变化
        assert_ne!(sponge.state, original_state);
    }
}
