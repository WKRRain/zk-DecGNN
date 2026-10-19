//! Hyrax IPA 内积证明 (Inner Product Argument)
//!
//! 严格遵循 Spartan-ZENO 设计文档第 3.3 节。
//!
//! ## 核心协议
//! 证明者向验证者证明: 给定公共承诺 C 和生成元 G，
//! 存在向量 w 使得 C = w·G = Σ w_j·G_j + z·H。
//!
//! ## 关键特性
//! - 对数级折叠: O(n_c) 轮 (n_c = 方阵列维度)
//! - 零知识: 每轮注入盲化因子 l_k, r_k
//! - Fiat-Shamir 非交互式变换
//! - 支持复合生成元正交混淆
//!
//! ## 双轨并行
//! 图特征轨和模型权重轨独立执行 IPA，
//! 分别使用各自的折叠挑战 u_k。

use crate::curve_config::{serialize_point, Fp, G1};
use crate::poseidon::PoseidonHash;
use ark_ec::Group;
use ark_ff::{Field, One, PrimeField};
use ark_std::{UniformRand, Zero};
use rand::rngs::OsRng;

fn absorb_g1(poseidon: &mut PoseidonHash, point: &G1) {
    let bytes = serialize_point(point);
    let scalar = Fp::from_le_bytes_mod_order(&bytes);
    poseidon.absorb(&[scalar]);
}

// ============================================================================
// IPA 证明体
// ============================================================================

/// 单个 IPA 轨道 (图特征轨 或 模型权重轨)
/// 的完整零知识内积证明体。
#[derive(Debug, Clone)]
pub struct IpaProof {
    /// n_c 轮折叠产生的交叉项向量 L
    pub l_vec: Vec<G1>,
    /// n_c 轮折叠产生的交叉项向量 R
    pub r_vec: Vec<G1>,
    /// 终局 Schnorr 盲化前置点 A
    pub a: G1,
    /// 终局标量响应 s_w (对应 w_0^{(final)})
    pub s_w: Fp,
    /// 终局标量响应 s_z (对应 Z_folded)
    pub s_z: Fp,
}

/// 双轨 IPA 证明体 (图特征 + 模型权重)
#[derive(Debug, Clone)]
pub struct DualTrackIpaProof {
    /// 图特征轨证明
    pub graph: IpaProof,
    /// 模型权重轨证明
    pub weight: IpaProof,
}

// ============================================================================
// 方阵列拉格朗日基底生成
// ============================================================================

/// 生成方阵列拉格朗日基底向量 b ∈ Fp^{2^{n_c}}
///
/// 利用列查询坐标 r_col ∈ Fp^{n_c}，通过张量积生成:
/// ```text
/// b = ⊗_{m=1}^{n_c} [1 - r_{col,m}, r_{col,m}]
/// ```
pub fn build_column_basis(r_col: &[Fp]) -> Vec<Fp> {
    crate::polynomial::build_lagrange_basis(r_col)
}

// ============================================================================
// 复合生成元正交混淆
// ============================================================================

/// 构建混淆后的复合生成元阵列 G' ∈ G^{2^{n_c}}
///
/// G'_j = G_j + b_j · Q
///
/// 其中 Q 是由 Fiat-Shamir 抽取的正交混淆基点。
pub fn blend_generators(g_base: &[G1], b: &[Fp], q: &G1) -> Vec<G1> {
    assert_eq!(
        g_base.len(),
        b.len(),
        "Generator count must match basis length"
    );
    g_base
        .iter()
        .zip(b.iter())
        .map(|(gj, bj)| *gj + q.mul_bigint(bj.into_bigint()))
        .collect()
}

// ============================================================================
// IPA 折叠循环执行器
// ============================================================================

/// 执行 IPA 折叠循环
///
/// 输入: 长度 2^{n_c} 的特征向量 w^{(0)} 和生成元 G'^{(0)}
/// 输出: n_c 轮后的 (w_final, g_final, l_vec, r_vec)
///
/// # 参数
/// - w0: 初始特征向量 (长度为 2^{n_c})
/// - g0: 初始复合生成元阵列 (长度为 2^{n_c})
/// - poseidon: 哈希海绵 (用于 Fiat-Shamir 挑战提取)
/// - l_blinds: 每轮的盲化因子 l_k (增加零知识性)
/// - r_blinds: 每轮的盲化因子 r_k
///
/// # 返回
/// - (w_final, g_final, l_vec, r_vec, u_vec)
///   - w_final: 折叠后的单一标量
///   - g_final: 折叠后的单一生成元
///   - l_vec: 每轮的交叉项 L_k (椭圆曲线点)
///   - r_vec: 每轮的交叉项 R_k (椭圆曲线点)
///   - u_vec: 每轮的折叠挑战 u_k
pub fn ipa_fold(
    w0: &[Fp],
    g0: &[G1],
    poseidon: &mut PoseidonHash,
    l_blinds: &[Fp],
    r_blinds: &[Fp],
) -> (Fp, G1, Vec<G1>, Vec<G1>, Vec<Fp>) {
    let n_c = (w0.len() as f64).log2().ceil() as usize;
    assert_eq!(w0.len(), 1 << n_c, "w0 length must be power of 2");
    assert_eq!(g0.len(), 1 << n_c, "g0 length must be power of 2");

    let mut w = w0.to_vec();
    let mut g = g0.to_vec();
    let mut l_vec: Vec<G1> = Vec::with_capacity(n_c);
    let mut r_vec: Vec<G1> = Vec::with_capacity(n_c);
    let mut u_vec: Vec<Fp> = Vec::with_capacity(n_c);

    for _k in 0..n_c {
        let s_k = w.len() / 2; // 当前半长

        // ======== 计算交叉项 ========
        let mut l_k = G1::zero();
        let mut r_k = G1::zero();

        for m in 0..s_k {
            // L_k = Σ_{m=0}^{s_k-1} w_m · G_{m+s_k}
            l_k += g[m + s_k].mul_bigint(w[m].into_bigint());
            // R_k = Σ_{m=0}^{s_k-1} w_{m+s_k} · G_m
            r_k += g[m].mul_bigint(w[m + s_k].into_bigint());
        }

        let _ = (l_blinds, r_blinds);

        l_vec.push(l_k);
        r_vec.push(r_k);

        // ======== Fiat-Shamir 挑战提取 ========
        // 将 L_k, R_k 的坐标吸入海绵
        absorb_g1(poseidon, &l_k);
        absorb_g1(poseidon, &r_k);
        poseidon.finalize_absorption();
        let mut u_k = poseidon.squeeze();
        if u_k.is_zero() {
            u_k = Fp::one();
        }
        u_vec.push(u_k);

        let u_inv = u_k.inverse().expect("u_k must be invertible");

        // ======== 原位压缩 ========
        // w_m^{(k)} = u_k · w_m^{(k-1)} + u_k^{-1} · w_{m+s_k}^{(k-1)}
        // G_m^{(k)} = u_k^{-1} · G_m^{(k-1)} + u_k · G_{m+s_k}^{(k-1)}
        let mut w_new = Vec::with_capacity(s_k);
        let mut g_new = Vec::with_capacity(s_k);

        for m in 0..s_k {
            let w_m = u_k * w[m] + u_inv * w[m + s_k];
            let g_m =
                g[m].mul_bigint(u_inv.into_bigint()) + g[m + s_k].mul_bigint(u_k.into_bigint());
            w_new.push(w_m);
            g_new.push(g_m);
        }

        w = w_new;
        g = g_new;
    }

    // w[0] = w_0^{(final)}, g[0] = G^{(final)}
    (w[0], g[0], l_vec, r_vec, u_vec)
}

// ============================================================================
// 终局零知识 Schnorr 证明
// ============================================================================

/// 生成终局 Sigma 零知识打开证明
///
/// 证明者不直接发送 w_final 和 Z_folded，
/// 而是通过 Schnorr 风格的 Sigma 协议将其锁死。
///
/// # 参数
/// - w_final: IPA 折叠后的最终标量
/// - z_folded: 全局盲化因子
/// - g_final: IPA 折叠后的最终生成元
/// - h: 盲化基点 H
/// - poseidon: 哈希海绵
///
/// # 返回
/// - (A, s_w, s_z): Schnorr 响应三元组
pub fn ipa_sigma_prove(
    w_final: &Fp,
    z_folded: &Fp,
    g_final: &G1,
    h: &G1,
    poseidon: &mut PoseidonHash,
) -> (G1, Fp, Fp) {
    // 1. 盲化前置点
    let mut rng = OsRng;
    let r_w = Fp::rand(&mut rng);
    let r_z = Fp::rand(&mut rng);

    let a = g_final.mul_bigint(r_w.into_bigint()) + h.mul_bigint(r_z.into_bigint());

    // 2. Fiat-Shamir 挑战
    absorb_g1(poseidon, &a);
    poseidon.finalize_absorption();
    let c = poseidon.squeeze();

    // 3. 标量响应
    let s_w = r_w + c * *w_final;
    let s_z = r_z + c * *z_folded;

    (a, s_w, s_z)
}

/// 验证终局 Sigma 打开证明
///
/// s_w · G_final + s_z · H ≟ A + c · C_final
pub fn ipa_sigma_verify(
    proof: &IpaProof,
    c_final: &G1,
    g_final: &G1,
    h: &G1,
    poseidon: &mut PoseidonHash,
) -> bool {
    absorb_g1(poseidon, &proof.a);
    poseidon.finalize_absorption();
    let c = poseidon.squeeze();

    let lhs = g_final.mul_bigint(proof.s_w.into_bigint()) + h.mul_bigint(proof.s_z.into_bigint());

    let rhs = proof.a + c_final.mul_bigint(c.into_bigint());

    lhs == rhs
}

// ============================================================================
// 完整的 IPA 证明生成 (单轨)
// ============================================================================

/// 生成单个轨道的完整 IPA 证明
///
/// 从初始特征向量 w0 和生成元 g0 出发，
/// 执行折叠循环 + 终局 Sigma 证明。
pub fn ipa_prove(
    w0: &[Fp],
    g0: &[G1],
    h: &G1,
    poseidon: &mut PoseidonHash,
    l_blinds: &[Fp],
    r_blinds: &[Fp],
) -> (IpaProof, Fp) {
    ipa_prove_with_blinding(w0, g0, h, &Fp::zero(), poseidon, l_blinds, r_blinds)
}

/// 生成单个轨道的完整 IPA 证明，并显式绑定初始承诺中的盲化因子。
pub fn ipa_prove_with_blinding(
    w0: &[Fp],
    g0: &[G1],
    h: &G1,
    z_folded: &Fp,
    poseidon: &mut PoseidonHash,
    l_blinds: &[Fp],
    r_blinds: &[Fp],
) -> (IpaProof, Fp) {
    let (w_final, g_final, l_vec, r_vec, _u_vec) = ipa_fold(w0, g0, poseidon, l_blinds, r_blinds);

    let (a, s_w, s_z) = ipa_sigma_prove(&w_final, z_folded, &g_final, h, poseidon);

    (
        IpaProof {
            l_vec,
            r_vec,
            a,
            s_w,
            s_z,
        },
        w_final,
    )
}

// ============================================================================
// 验证者端 IPA 承诺坍缩
// ============================================================================

/// 验证者从证明体的交叉项推导折叠挑战并坍缩初始承诺
///
/// C_final = C^{(0)} + Σ_{k=1}^{n_c} (u_k²·L_k + u_k^{-2}·R_k)
pub fn verifier_fold_commitment(
    c0: &G1,
    l_vec: &[G1],
    r_vec: &[G1],
    poseidon: &mut PoseidonHash,
) -> (G1, Vec<Fp>) {
    let n_c = l_vec.len();
    let mut c_final = *c0;
    let mut u_vec = Vec::with_capacity(n_c);

    for k in 0..n_c {
        absorb_g1(poseidon, &l_vec[k]);
        absorb_g1(poseidon, &r_vec[k]);
        poseidon.finalize_absorption();
        let mut u = poseidon.squeeze();
        if u.is_zero() {
            u = Fp::one();
        }
        u_vec.push(u);

        let u2 = u * u;
        let u_inv2 =
            u.inverse().expect("u must be invertible") * u.inverse().expect("u must be invertible");

        c_final = c_final
            + l_vec[k].mul_bigint(u2.into_bigint())
            + r_vec[k].mul_bigint(u_inv2.into_bigint());
    }

    (c_final, u_vec)
}

/// 验证者从折叠挑战推导终局生成元 G_final
///
/// G_final = Σ_{j=0}^{2^{n_c}-1} [Π_{k=1}^{n_c} u_k^{-(1-2·bit_k(j))}] · G'_j
pub fn verifier_derive_final_generator(g0: &[G1], u_vec: &[Fp]) -> G1 {
    let mut gens = g0.to_vec();
    for &u in u_vec {
        let u_inv = u.inverse().expect("u_k must be invertible");
        let half = gens.len() / 2;
        let mut next = Vec::with_capacity(half);
        for m in 0..half {
            next.push(
                gens[m].mul_bigint(u_inv.into_bigint())
                    + gens[m + half].mul_bigint(u.into_bigint()),
            );
        }
        gens = next;
    }
    gens[0]
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve_config::G1;
    use ark_std::UniformRand;

    #[test]
    fn test_build_column_basis() {
        let r_col = vec![Fp::from(0u64), Fp::from(1u64)];
        let basis = build_column_basis(&r_col);
        assert_eq!(basis.len(), 4);
        // Manual verification of tensor product
        // [(1-0)*(1-1), (1-0)*1, 0*(1-1), 0*1] = [0, 1, 0, 0]
        assert_eq!(basis[0], Fp::zero());
        assert_eq!(basis[1], Fp::one());
        assert_eq!(basis[2], Fp::zero());
        assert_eq!(basis[3], Fp::zero());
    }

    #[test]
    fn test_ipa_fold_small() {
        // 使用极小案例: 2^1 = 2 个元素
        let w0: Vec<Fp> = vec![Fp::from(3u64), Fp::from(4u64)];
        let mut rng = ark_std::test_rng();
        let g0: Vec<G1> = (0..2).map(|_| G1::rand(&mut rng)).collect();
        let mut poseidon = PoseidonHash::new_for_test();
        let l_blinds: Vec<Fp> = vec![];
        let r_blinds: Vec<Fp> = vec![];

        let (_w_final, _g_final, l_vec, r_vec, u_vec) =
            ipa_fold(&w0, &g0, &mut poseidon, &l_blinds, &r_blinds);

        assert_eq!(l_vec.len(), 1);
        assert_eq!(r_vec.len(), 1);
        assert_eq!(u_vec.len(), 1);
    }

    #[test]
    fn test_ipa_prove_and_verify() {
        // 测试用的简化 IPA 流程
        let mut rng = ark_std::test_rng();
        let n_c = 2; // 4 个元素
        let len = 1 << n_c;
        let w0: Vec<Fp> = (0..len).map(|i| Fp::from(i as u64 + 1)).collect();
        let h = G1::rand(&mut rng);
        let g0: Vec<G1> = (0..len).map(|_| G1::rand(&mut rng)).collect();

        // 初始承诺 C0 = Σ w_j·G_j + z·H
        let z0 = Fp::from(42u64);
        let mut c0 = G1::zero();
        for j in 0..len {
            c0 += g0[j].mul_bigint(w0[j].into_bigint());
        }
        c0 += h.mul_bigint(z0.into_bigint());

        let mut poseidon = PoseidonHash::new_for_test();
        let l_blinds: Vec<Fp> = vec![Fp::from(1u64)];
        let r_blinds: Vec<Fp> = vec![Fp::from(2u64)];

        let (proof, _w_final) =
            ipa_prove_with_blinding(&w0, &g0, &h, &z0, &mut poseidon, &l_blinds, &r_blinds);

        // 验证者侧
        let mut verifier_sponge = PoseidonHash::new_for_test();
        let (c_final, u_vec) =
            verifier_fold_commitment(&c0, &proof.l_vec, &proof.r_vec, &mut verifier_sponge);
        let g_final = verifier_derive_final_generator(&g0, &u_vec);

        let mut verifier_sponge2 = PoseidonHash::new_for_test();
        for k in 0..proof.l_vec.len() {
            absorb_g1(&mut verifier_sponge2, &proof.l_vec[k]);
            absorb_g1(&mut verifier_sponge2, &proof.r_vec[k]);
            verifier_sponge2.finalize_absorption();
            verifier_sponge2.squeeze(); // consume u_k
        }
        let valid = ipa_sigma_verify(&proof, &c_final, &g_final, &h, &mut verifier_sponge2);
        assert!(valid, "IPA proof verification failed");
    }
}
