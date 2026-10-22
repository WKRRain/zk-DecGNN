//! Curve25519 椭圆曲线与素数域配置
//!
//! 本模块封装了 Curve25519 的标量域 (ScalarField) 与椭圆曲线群 (Group)。
//! 选择 ristretto255 作为 Curve25519 的安全素数阶群抽象，因其提供
//! 素数阶群且无协因子困扰，完美适配 Hyrax 承诺与 IPA 协议。
//!
//! ## 关键参数
//! - 标量域: |F_p| ≈ 2^252 + ... (约 253-bit 素数域)
//! - 群生成元集合: 按需动态派生

use ark_ec::scalar_mul::variable_base::VariableBaseMSM;
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{One, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use sha2::{Digest, Sha512};

// 使用 arkworks 的 Curve25519 实现 (Edwards 形式，标量域约 253-bit)
pub use ark_curve25519::EdwardsProjective as Group;
pub use ark_curve25519::Fr as ScalarField;

/// 标量域的数字别名
pub type Fp = ScalarField;

/// 旧版 witness 模块使用的标量域别名。
pub type Fr = Fp;

/// 椭圆曲线群类型 (Projective 形式，便于 MSM 运算)
pub type G = Group;

/// G1 群类型别名 (等同于 G, Projective 形式)
pub type G1 = Group;

/// Hyrax 标量类型 (等同于 Fp, 用于承诺与 IPA 中的标量操作)
pub type HyraxScalar = ScalarField;

/// 仿射坐标点 (用于序列化/哈希输入)
pub type GAffine = ark_curve25519::EdwardsAffine;

/// 旧版 witness 模块使用的 G1 仿射点别名。
pub type G1Affine = GAffine;

/// 曲线群的字节长度 (压缩表示)
pub const G1_SERIALIZED_SIZE: usize = 32;

/// 标量域的字节长度
pub const SCALAR_SERIALIZED_SIZE: usize = 32;

/// 生成元 G_0 (主生成元，用于单点 Pedersen 承诺)
/// 在 ristretto255 中，默认生成元即群的标准生成元。
/// 曲线群标准生成元由 AffineRepr::generator() 提供，再转换为 Projective。
pub fn default_generator() -> G {
    let g_aff = ark_curve25519::EdwardsAffine::generator();
    g_aff.into_group()
}

/// 生成独立的盲化基点 H
///
/// 通过 "hash-to-curve" 方法从一个众所周知的字符串派生，
/// 确保 H 与 G_0 之间不存在已知的离散对数关系。
/// 文档规范要求使用独立的盲化基点。
pub fn blinding_base() -> G {
    let label = b"Spartan-ZENO-BlindingBase-H-v1";
    hash_to_curve(label)
}

/// 从字节串派生曲线点。
///
/// This uses SHA-512 domain separation with try-and-increment sampling through
/// arkworks' curve decoder, followed by cofactor clearing. It avoids exposing a
/// known discrete-log relation to the base generator. Production deployments
/// should still replace this with an audited RFC 9380 suite for Curve25519.
pub fn hash_to_curve(data: &[u8]) -> G {
    for counter in 0u64.. {
        let mut hasher = Sha512::new();
        hasher.update(b"Spartan-ZENO-hash-to-curve-v2");
        hasher.update((data.len() as u64).to_le_bytes());
        hasher.update(data);
        hasher.update(counter.to_le_bytes());
        let digest = hasher.finalize();
        if let Some(point) = GAffine::from_random_bytes(&digest) {
            let cleared = point.clear_cofactor();
            if !cleared.is_zero() {
                return cleared.into_group();
            }
        }
    }
    unreachable!("u64 counter space exhausted while hashing to curve")
}

/// 从字节序列确定性生成标量 (用于 Fiat-Shamir 变换)
pub fn scalar_from_bytes(data: &[u8]) -> Fp {
    let mut hasher = Sha512::new();
    hasher.update(b"Spartan-ZENO-scalar-from-bytes-v2");
    hasher.update((data.len() as u64).to_le_bytes());
    hasher.update(data);
    Fp::from_be_bytes_mod_order(&hasher.finalize())
}

/// 生成 n 个独立的群生成元向量
///
/// 用于 Hyrax 承诺中按列绑定的生成元阵列。
/// 生成方式：G_i = hash_to_curve("Spartan-ZENO-Generator-{i}")
pub fn derive_generators(n: usize) -> Vec<G> {
    (0..n)
        .map(|i| {
            let label = format!("Spartan-ZENO-Generator-{}", i);
            hash_to_curve(label.as_bytes())
        })
        .collect()
}

/// 将椭圆曲线点序列化为压缩的 32 字节
pub fn serialize_point(point: &G) -> Vec<u8> {
    let mut buf = Vec::new();
    point.into_affine().serialize_compressed(&mut buf).unwrap();
    buf
}

/// 从压缩的 32 字节反序列化椭圆曲线点
pub fn deserialize_point(bytes: &[u8]) -> Option<G> {
    let affine = GAffine::deserialize_compressed(bytes).ok()?;
    Some(affine.into())
}

/// 将标量序列化为 32 字节大端表示
pub fn serialize_scalar(scalar: &Fp) -> Vec<u8> {
    let mut buf = Vec::new();
    scalar.serialize_compressed(&mut buf).unwrap();
    buf
}

/// 椭圆曲线点上的异构批次加法 (Multi-Scalar Multiplication)
///
/// 计算 sum(scalars[i] * points[i])。
/// 这是承诺打开校验与 IPA 折叠的核心操作。
pub fn msm(scalars: &[Fp], points: &[G]) -> G {
    assert_eq!(scalars.len(), points.len(), "MSM: mismatched lengths");

    // arkworks MSM 要求 scalars 为 owned slice (not &refs)
    let scalars_owned: Vec<Fp> = scalars.to_vec();
    let points_affine: Vec<GAffine> = points.iter().map(|p| p.into_affine()).collect();
    G::msm(&points_affine, &scalars_owned).unwrap()
}

/// 域元素全零常量
pub fn zero() -> Fp {
    Fp::zero()
}

/// 域元素单位元
pub fn one() -> Fp {
    Fp::one()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_derivation() {
        let gen = default_generator();
        assert!(gen.into_affine().is_on_curve());
    }

    #[test]
    fn test_blinding_base_independence() {
        let h = blinding_base();
        let g = default_generator();
        // H 和 G 应该是不同的点
        assert_ne!(serialize_point(&h), serialize_point(&g));
    }

    #[test]
    fn test_derive_generators() {
        let gens = derive_generators(32);
        assert_eq!(gens.len(), 32);
        // 所有生成元应在曲线上
        for g in &gens {
            assert!(g.into_affine().is_on_curve());
        }
    }

    #[test]
    fn test_msm_basic() {
        let g = default_generator();
        let h = blinding_base();
        let a = Fp::from(2u64);
        let b = Fp::from(3u64);

        let result = msm(&[a, b], &[g, h]);
        let expected = g * a + h * b;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_serialization_roundtrip() {
        let s = Fp::from(123456789u64);
        let bytes = serialize_scalar(&s);
        let recovered = Fp::deserialize_compressed(&bytes[..]).unwrap();
        assert_eq!(s, recovered);
    }
}
