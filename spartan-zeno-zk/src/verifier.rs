//! Stage 4 verifier for Spartan-ZENO.
//!
//! This module stitches together the verifier-side checks described in the
//! design document:
//! - transcript replay and challenge reconstruction,
//! - public-output anchoring,
//! - SPARK, Sum-check, and ReLU terminal boundary checks,
//! - O(1) Pedersen product Sigma verification for R1CS terminals,
//! - Hyrax/IPA final opening checks for graph and weight tracks.

use crate::curve_config::{default_generator, Fp, GAffine, G};
use crate::hyrax::{self, HyraxParams};
use crate::polynomial;
use crate::poseidon::PoseidonSponge;
use crate::spark;
use ark_ec::CurveGroup;
use ark_ff::{One, PrimeField, Zero};
use ark_serialize::CanonicalSerialize;

fn absorb_affine_point(sponge: &mut PoseidonSponge, point: &GAffine) {
    let mut bytes = Vec::new();
    point
        .serialize_compressed(&mut bytes)
        .expect("curve point serialization should not fail");
    let scalar = Fp::from_le_bytes_mod_order(&bytes);
    sponge.absorb_scalar(&scalar);
}

fn squeeze_nonzero(sponge: &mut PoseidonSponge) -> Fp {
    let mut challenge = sponge.squeeze();
    if challenge.is_zero() {
        challenge = Fp::one();
    }
    challenge
}

// ============================================================================
// Stage 4.1: transcript replay and public-output anchoring
// ============================================================================

/// Fiat-Shamir challenge vector sizes used while replaying the transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranscriptShape {
    pub eta_len: usize,
    pub n_k: usize,
    pub n_v: usize,
    pub n_d: usize,
    pub n_h: usize,
}

/// Public transcript material absorbed before challenges are squeezed.
#[derive(Debug, Clone, Default)]
pub struct TranscriptReplayInput {
    pub public_params: Vec<Fp>,
    pub commitment_scalars: Vec<Fp>,
    pub commitment_points: Vec<GAffine>,
    pub public_output: Vec<Fp>,
}

/// Verifier-side reconstruction of the global transcript Gamma.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierTranscript {
    pub delta_1: Fp,
    pub delta_2: Fp,
    pub tau: Fp,
    pub eta: Vec<Fp>,
    pub alpha: Fp,
    pub r_z: Vec<Fp>,
    pub r_x: Vec<Fp>,
    pub r_y: Vec<Fp>,
    pub r_w: Vec<Fp>,
    pub beta_graph: Fp,
    pub beta_weight: Fp,
    pub r_blind: Fp,
    pub q_ipa_x: Fp,
}

/// Replay the transcript absorption order and reconstruct all verifier
/// challenges.
pub fn replay_transcript(
    input: &TranscriptReplayInput,
    shape: TranscriptShape,
) -> VerifierTranscript {
    let mut sponge = PoseidonSponge::new();
    sponge.absorb_scalars(&input.public_params);
    sponge.absorb_scalars(&input.commitment_scalars);
    for point in &input.commitment_points {
        absorb_affine_point(&mut sponge, point);
    }
    sponge.absorb_scalars(&input.public_output);
    sponge.finalize_absorption();

    VerifierTranscript {
        delta_1: sponge.squeeze(),
        delta_2: sponge.squeeze(),
        tau: sponge.squeeze(),
        eta: sponge.squeeze_n(shape.eta_len),
        alpha: sponge.squeeze(),
        r_z: sponge.squeeze_n(shape.n_k),
        r_x: sponge.squeeze_n(shape.n_v),
        r_y: sponge.squeeze_n(shape.n_d),
        r_w: sponge.squeeze_n(shape.n_h),
        beta_graph: sponge.squeeze(),
        beta_weight: sponge.squeeze(),
        r_blind: sponge.squeeze(),
        q_ipa_x: sponge.squeeze(),
    }
}

/// Claim tying a public logits matrix to the circuit's hidden output evaluation.
#[derive(Debug, Clone)]
pub struct PublicOutputClaim {
    pub y_tilde: Vec<Fp>,
    pub n_rows: usize,
    pub n_cols: usize,
    pub n_v: usize,
    pub n_d: usize,
    pub n_real: usize,
    pub c: usize,
    pub r_x: Vec<Fp>,
    pub r_y: Vec<Fp>,
    pub y_circuit_eval: Fp,
}

/// Recompute v_public_Y from the clear logits and compare it with the circuit
/// oracle value.
pub fn verify_public_output_anchor(claim: &PublicOutputClaim) -> bool {
    let public_eval = polynomial::eval_mle_2d(
        &claim.y_tilde,
        claim.n_rows,
        claim.n_cols,
        &claim.r_x,
        &claim.r_y,
        claim.n_v,
        claim.n_d,
        Some(claim.n_real),
        Some(claim.c),
    );
    public_eval == claim.y_circuit_eval
}

// ============================================================================
// Stage 4.2: O(1) terminal boundary checks
// ============================================================================

/// SPARK root claims for the four memory streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparkRootClaims {
    pub init: Fp,
    pub read: Fp,
    pub write: Fp,
    pub audit: Fp,
}

/// Optional public address binding for the SPARK read stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparkAddressBinding {
    pub read_addr_eval: Fp,
    pub public_adjacency_eval: Fp,
}

/// SPARK terminal boundary claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparkBoundaryClaim {
    pub roots: SparkRootClaims,
    pub address_binding: Option<SparkAddressBinding>,
}

/// Verify SPARK root conservation and, when provided, the public adjacency
/// address binding.
pub fn verify_spark_boundary(claim: &SparkBoundaryClaim) -> bool {
    let roots_ok = spark::verify_memory_conservation(
        &claim.roots.init,
        &claim.roots.write,
        &claim.roots.read,
        &claim.roots.audit,
    );
    let addr_ok = claim
        .address_binding
        .map(|b| b.read_addr_eval == b.public_adjacency_eval)
        .unwrap_or(true);
    roots_ok && addr_ok
}

/// Sum-check terminal claim for F_check = F_R1CS + r_blind * R.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SumcheckTerminalClaim {
    pub v_r1cs: Fp,
    pub v_blind: Fp,
    pub r_blind: Fp,
    pub terminal_eval: Fp,
}

pub fn expected_sumcheck_terminal(claim: &SumcheckTerminalClaim) -> Fp {
    claim.v_r1cs + claim.r_blind * claim.v_blind
}

/// Verify the final univariate Sum-check value against the locally assembled
/// R1CS plus blinding evaluation.
pub fn verify_sumcheck_terminal(claim: &SumcheckTerminalClaim) -> bool {
    expected_sumcheck_terminal(claim) == claim.terminal_eval
}

/// O(1) residuals emitted by the ReLU/bit-decomposition Sum-check track.
///
/// Each field is expected to be the terminal evaluation of an already-summed
/// constraint polynomial. A valid proof drives all residuals to zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReluBoundaryClaim {
    pub bit_boolean_residual: Fp,
    pub sign_boolean_residual: Fp,
    pub reconstruction_residual: Fp,
    pub activation_residual: Fp,
}

pub fn verify_relu_boundary(claim: &ReluBoundaryClaim) -> bool {
    claim.bit_boolean_residual.is_zero()
        && claim.sign_boolean_residual.is_zero()
        && claim.reconstruction_residual.is_zero()
        && claim.activation_residual.is_zero()
}

pub fn verify_relu_boundaries(claims: &[ReluBoundaryClaim]) -> bool {
    claims.iter().all(verify_relu_boundary)
}

// ============================================================================
// R1CS terminal Sigma protocol
// ============================================================================

/// Pedersen commitment to one scalar.
pub fn pedersen_commit(
    value: &Fp,
    blind: &Fp,
    value_base: &GAffine,
    blind_base: &GAffine,
) -> GAffine {
    (*value_base * *value + *blind_base * *blind).into_affine()
}

/// Randomness used by the Pedersen product Sigma protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductProofRandomness {
    pub rho_1: Fp,
    pub rho_2: Fp,
    pub rho_3: Fp,
    pub rho_4: Fp,
    pub rho_5: Fp,
}

/// O(1) Sigma proof that committed values satisfy v_A * v_B = v_C.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PedersenProductProof {
    pub r_a_point: GAffine,
    pub r_b_point: GAffine,
    pub r_c_point: GAffine,
    pub z_1: Fp,
    pub z_2: Fp,
    pub z_3: Fp,
    pub z_4: Fp,
    pub z_5: Fp,
}

fn product_challenge(
    c_a: &GAffine,
    c_b: &GAffine,
    c_c: &GAffine,
    proof_prefix: (&GAffine, &GAffine, &GAffine),
    sponge: &mut PoseidonSponge,
) -> Fp {
    absorb_affine_point(sponge, c_a);
    absorb_affine_point(sponge, c_b);
    absorb_affine_point(sponge, c_c);
    absorb_affine_point(sponge, proof_prefix.0);
    absorb_affine_point(sponge, proof_prefix.1);
    absorb_affine_point(sponge, proof_prefix.2);
    sponge.finalize_absorption();
    squeeze_nonzero(sponge)
}

/// Build a Pedersen product proof from explicit randomness.
pub fn prove_pedersen_product(
    v_a: &Fp,
    r_a: &Fp,
    v_b: &Fp,
    r_b: &Fp,
    r_c: &Fp,
    randomness: ProductProofRandomness,
    value_base: &GAffine,
    blind_base: &GAffine,
    sponge: &mut PoseidonSponge,
) -> (PedersenProductProof, GAffine, GAffine, GAffine) {
    let v_c = *v_a * *v_b;
    let c_a = pedersen_commit(v_a, r_a, value_base, blind_base);
    let c_b = pedersen_commit(v_b, r_b, value_base, blind_base);
    let c_c = pedersen_commit(&v_c, r_c, value_base, blind_base);

    let r_a_point = pedersen_commit(&randomness.rho_1, &randomness.rho_2, value_base, blind_base);
    let r_b_point = pedersen_commit(&randomness.rho_3, &randomness.rho_4, value_base, blind_base);
    let r_c_point = (c_b * randomness.rho_1 + *blind_base * randomness.rho_5).into_affine();

    let e = product_challenge(
        &c_a,
        &c_b,
        &c_c,
        (&r_a_point, &r_b_point, &r_c_point),
        sponge,
    );

    let proof = PedersenProductProof {
        r_a_point,
        r_b_point,
        r_c_point,
        z_1: randomness.rho_1 + e * *v_a,
        z_2: randomness.rho_2 + e * *r_a,
        z_3: randomness.rho_3 + e * *v_b,
        z_4: randomness.rho_4 + e * *r_b,
        z_5: randomness.rho_5 + e * (*r_c - *v_a * *r_b),
    };

    (proof, c_a, c_b, c_c)
}

/// Verify the O(1) product relation used by the R1CS terminal check.
pub fn verify_pedersen_product(
    proof: &PedersenProductProof,
    c_a: &GAffine,
    c_b: &GAffine,
    c_c: &GAffine,
    value_base: &GAffine,
    blind_base: &GAffine,
    sponge: &mut PoseidonSponge,
) -> bool {
    let e = product_challenge(
        c_a,
        c_b,
        c_c,
        (&proof.r_a_point, &proof.r_b_point, &proof.r_c_point),
        sponge,
    );

    let eq_a_left = *value_base * proof.z_1 + *blind_base * proof.z_2;
    let eq_a_right = G::from(proof.r_a_point) + *c_a * e;

    let eq_b_left = *value_base * proof.z_3 + *blind_base * proof.z_4;
    let eq_b_right = G::from(proof.r_b_point) + *c_b * e;

    let eq_c_left = *c_b * proof.z_1 + *blind_base * proof.z_5;
    let eq_c_right = G::from(proof.r_c_point) + *c_c * e;

    eq_a_left.into_affine() == eq_a_right.into_affine()
        && eq_b_left.into_affine() == eq_b_right.into_affine()
        && eq_c_left.into_affine() == eq_c_right.into_affine()
}

pub fn default_value_base() -> GAffine {
    default_generator().into_affine()
}

// ============================================================================
// Stage 4.3: Hyrax/IPA opening verification
// ============================================================================

/// One tensor commitment batch in a graph or weight track.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitmentBatch {
    pub commitments: Vec<GAffine>,
    pub r_row: Vec<Fp>,
    pub n_r: usize,
}

/// A batched Hyrax opening claim for one logical track.
#[derive(Debug, Clone)]
pub struct BatchedOpeningClaim {
    pub proof: hyrax::IpaProof,
    pub batches: Vec<CommitmentBatch>,
    pub beta: Fp,
    pub n_c: usize,
}

/// Reconstruct C^(0) = sum_idx beta^idx * sum_i L_i C_idx,i.
pub fn reconstruct_batched_commitment(claim: &BatchedOpeningClaim) -> GAffine {
    let mut acc = G::zero();
    let mut beta_power = Fp::one();

    for batch in &claim.batches {
        let folded =
            hyrax::reconstruct_expected_commitment(&batch.commitments, &batch.r_row, batch.n_r);
        acc += folded * beta_power;
        beta_power *= claim.beta;
    }

    acc.into_affine()
}

/// Verify one batched Hyrax/IPA opening track.
pub fn verify_batched_opening(
    claim: &BatchedOpeningClaim,
    params: &HyraxParams,
    sponge: &mut PoseidonSponge,
) -> bool {
    if claim.batches.is_empty() {
        return false;
    }
    let c_initial = reconstruct_batched_commitment(claim);
    hyrax::ipa_verify(
        &claim.proof,
        &c_initial,
        &params.gens,
        &params.blind_base,
        claim.n_c,
        sponge,
    )
}

/// Verify one non-batched Hyrax opening using the existing commitment vector.
#[derive(Debug, Clone)]
pub struct OpeningClaim {
    pub proof: hyrax::IpaProof,
    pub commitments: Vec<GAffine>,
    pub r_row: Vec<Fp>,
    pub r_col: Vec<Fp>,
    pub n_r: usize,
    pub n_c: usize,
}

pub fn verify_opening(
    claim: &OpeningClaim,
    params: &HyraxParams,
    sponge: &mut PoseidonSponge,
) -> bool {
    hyrax::verify_opening(
        &claim.proof,
        &claim.commitments,
        &claim.r_row,
        &claim.r_col,
        claim.n_r,
        claim.n_c,
        params,
        sponge,
    )
}

/// Verify graph and weight tracks independently.
pub fn verify_dual_track_openings(
    graph: &BatchedOpeningClaim,
    weight: &BatchedOpeningClaim,
    params: &HyraxParams,
) -> bool {
    let mut graph_sponge = PoseidonSponge::new();
    let mut weight_sponge = PoseidonSponge::new();
    verify_batched_opening(graph, params, &mut graph_sponge)
        && verify_batched_opening(weight, params, &mut weight_sponge)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyrax::{commit_matrix, flatten_rows, ipa_prove};

    #[test]
    fn test_replay_transcript_is_deterministic() {
        let input = TranscriptReplayInput {
            public_params: vec![Fp::from(1u64), Fp::from(2u64)],
            commitment_scalars: vec![Fp::from(3u64)],
            commitment_points: vec![],
            public_output: vec![Fp::from(4u64), Fp::from(5u64)],
        };
        let shape = TranscriptShape {
            eta_len: 2,
            n_k: 1,
            n_v: 1,
            n_d: 1,
            n_h: 1,
        };

        assert_eq!(
            replay_transcript(&input, shape),
            replay_transcript(&input, shape)
        );
    }

    #[test]
    fn test_public_output_anchor() {
        let y = vec![
            Fp::from(1u64),
            Fp::from(2u64),
            Fp::from(3u64),
            Fp::from(4u64),
        ];
        let claim = PublicOutputClaim {
            y_tilde: y,
            n_rows: 2,
            n_cols: 2,
            n_v: 1,
            n_d: 1,
            n_real: 2,
            c: 2,
            r_x: vec![Fp::zero()],
            r_y: vec![Fp::one()],
            y_circuit_eval: Fp::from(2u64),
        };
        assert!(verify_public_output_anchor(&claim));

        let mut bad = claim.clone();
        bad.y_circuit_eval = Fp::from(3u64);
        assert!(!verify_public_output_anchor(&bad));
    }

    #[test]
    fn test_spark_and_sumcheck_boundaries() {
        let spark_claim = SparkBoundaryClaim {
            roots: SparkRootClaims {
                init: Fp::from(2u64),
                read: Fp::from(6u64),
                write: Fp::from(3u64),
                audit: Fp::one(),
            },
            address_binding: Some(SparkAddressBinding {
                read_addr_eval: Fp::from(9u64),
                public_adjacency_eval: Fp::from(9u64),
            }),
        };
        assert!(verify_spark_boundary(&spark_claim));

        let sumcheck_claim = SumcheckTerminalClaim {
            v_r1cs: Fp::from(11u64),
            v_blind: Fp::from(7u64),
            r_blind: Fp::from(5u64),
            terminal_eval: Fp::from(46u64),
        };
        assert!(verify_sumcheck_terminal(&sumcheck_claim));
    }

    #[test]
    fn test_relu_boundary_residuals() {
        let ok = ReluBoundaryClaim {
            bit_boolean_residual: Fp::zero(),
            sign_boolean_residual: Fp::zero(),
            reconstruction_residual: Fp::zero(),
            activation_residual: Fp::zero(),
        };
        assert!(verify_relu_boundary(&ok));

        let bad = ReluBoundaryClaim {
            activation_residual: Fp::one(),
            ..ok
        };
        assert!(!verify_relu_boundary(&bad));
    }

    #[test]
    fn test_pedersen_product_sigma_round_trip() {
        let value_base = default_value_base();
        let blind_base = (default_generator() * Fp::from(17u64)).into_affine();
        let randomness = ProductProofRandomness {
            rho_1: Fp::from(3u64),
            rho_2: Fp::from(4u64),
            rho_3: Fp::from(5u64),
            rho_4: Fp::from(6u64),
            rho_5: Fp::from(7u64),
        };

        let mut prover_sponge = PoseidonSponge::new();
        let (proof, c_a, c_b, c_c) = prove_pedersen_product(
            &Fp::from(8u64),
            &Fp::from(10u64),
            &Fp::from(9u64),
            &Fp::from(11u64),
            &Fp::from(12u64),
            randomness,
            &value_base,
            &blind_base,
            &mut prover_sponge,
        );

        let mut verifier_sponge = PoseidonSponge::new();
        assert!(verify_pedersen_product(
            &proof,
            &c_a,
            &c_b,
            &c_c,
            &value_base,
            &blind_base,
            &mut verifier_sponge,
        ));

        let mut bad_proof = proof.clone();
        bad_proof.z_1 += Fp::one();
        let mut bad_sponge = PoseidonSponge::new();
        assert!(!verify_pedersen_product(
            &bad_proof,
            &c_a,
            &c_b,
            &c_c,
            &value_base,
            &blind_base,
            &mut bad_sponge,
        ));
    }

    fn make_opening_claim(seed: [u8; 32]) -> (BatchedOpeningClaim, HyraxParams) {
        let params = HyraxParams::new(2, seed);
        let rows = 2;
        let cols = 4;
        let n_r = 1;
        let n_c = 2;
        let matrix = vec![
            Fp::from(1u64),
            Fp::from(2u64),
            Fp::from(3u64),
            Fp::from(4u64),
            Fp::from(5u64),
            Fp::from(6u64),
            Fp::from(7u64),
            Fp::from(8u64),
        ];
        let (commitments, blindings) = commit_matrix(&matrix, rows, cols, &params, None);
        let r_row = vec![Fp::from(13u64)];
        let (w, z) = flatten_rows(&matrix, rows, cols, &r_row, &blindings, n_r);
        let mut sponge = PoseidonSponge::new();
        let (proof, _) = ipa_prove(&w, &params.gens, &params.blind_base, &z, n_c, &mut sponge);

        let claim = BatchedOpeningClaim {
            proof,
            batches: vec![CommitmentBatch {
                commitments,
                r_row,
                n_r,
            }],
            beta: Fp::from(19u64),
            n_c,
        };
        (claim, params)
    }

    #[test]
    fn test_batched_opening_round_trip() {
        let (claim, params) = make_opening_claim([31u8; 32]);
        let mut sponge = PoseidonSponge::new();
        assert!(verify_batched_opening(&claim, &params, &mut sponge));
    }

    #[test]
    fn test_dual_track_openings() {
        let (claim, params) = make_opening_claim([41u8; 32]);
        assert!(verify_dual_track_openings(&claim, &claim, &params));
    }
}
