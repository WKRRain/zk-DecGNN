//! Rust backend for Spartan-ZENO graph-classification proofs.
//!
//! This module is intentionally separate from the existing node-classification
//! benchmark code.  It keeps the same backend vocabulary already used by the
//! project: ZENO static tracks, Hyrax row commitments, SPARK sparse memory
//! roots, transcript-bound Sum-check/R1CS claims, fixed-point truncation
//! witnesses, and an Argmax range-proof witness.  Python is only responsible
//! for dataset loading and experiment orchestration.

use crate::curve_config::{Fp, GAffine, G};
use crate::hyrax::{
    commit_matrix, flatten_rows, ipa_verify, reconstruct_expected_commitment, HyraxParams,
    IpaProof, IpaRoundData,
};
use crate::model_registry::ModelArchitecture;
use crate::poseidon::PoseidonSponge;
use crate::spark::{
    build_algebraic_tags, build_memory_trees, grand_product, verify_memory_conservation, MemoryOp,
};
use ark_ec::CurveGroup;
use ark_ff::{BigInteger, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyModule};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Instant;

const GC_PROOF_TYPE: &str = "Spartan-ZENO-GC-v2";
const GC_SCHEMA_VERSION: usize = 2;
const GC_CONSTRAINT_SYSTEM: &str = "graph_classification_r1cs_v3";
const GC_FIXED_SCALE: i64 = 256;
const GC_CANONICAL_BINARY_TYPE: &[u8] = b"Spartan-ZENO-GC-canonical-v4";
const GC_OPENING_MODE_BATCHED: &str = "batched_ipa_v1";
const GC_RANGE_PROFILE_COMPACT: &str = "compact_value_reconstruction_v1";
const GC_ACTIVE_COLUMN_PROFILE: &str = "model_aware_active_columns_v1";
const GC_RLC_SUMCHECK_LABEL: &str = "rlc_residuals";
const GC_SUMCHECK_PAR_THRESHOLD: usize = 1 << 15;
const HYRAX_GC_SEED: [u8; 32] = [42u8; 32];

static HYRAX_PARAMS_CACHE: LazyLock<Mutex<HashMap<(usize, [u8; 32]), Arc<HyraxParams>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static GC_VK_PROFILE_CACHE: LazyLock<Mutex<HashMap<GcVkProfileKey, GcVkProfile>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static OPENING_R_ROW_CACHE: LazyLock<Mutex<HashMap<OpeningRRowCacheKey, Vec<Fp>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static PUBLIC_INPUT_HASH_CACHE: LazyLock<Mutex<HashMap<[u8; 32], Fp>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static GC_SERIALIZED_PROOF_CACHE: LazyLock<Mutex<HashMap<[u8; 32], Arc<ParsedGcProofCore>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GcVkProfileKey {
    arch: ModelArchitecture,
    n_max: usize,
    d: usize,
    c: usize,
    h: usize,
    k: usize,
    d_max: usize,
    h_max: usize,
}

#[derive(Clone, Copy)]
struct GcVkProfile {
    vk_id: Fp,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct OpeningRRowCacheKey {
    label: &'static str,
    public_hash: String,
    model_id: String,
    n_r: usize,
    commitment_hex: Vec<String>,
}

#[derive(Clone)]
struct Trace {
    x_base: Vec<i128>,
    h: Vec<i128>,
    h_cols: usize,
    active_h_cols: usize,
    g: Vec<i128>,
    y_g: Vec<i128>,
    ell: usize,
    pool_remainders: Vec<i128>,
    trunc_remainders: Vec<i128>,
    argmax_deltas: Vec<i128>,
    semantic_residuals: Vec<i128>,
}

struct ConstraintResiduals {
    r1cs: Vec<i128>,
    semantic: Vec<i128>,
    mean_pool: Vec<i128>,
    argmax: Vec<i128>,
    range: Vec<i128>,
}

struct SparkRoots {
    init_root: Fp,
    read_root: Fp,
    write_root: Fp,
    audit_root: Fp,
    padded_len: usize,
    conservation: bool,
}

struct MatrixCommitment {
    label: &'static str,
    matrix: Vec<Fp>,
    rows: usize,
    cols: usize,
    n_r: usize,
    n_c: usize,
    commitments: Vec<GAffine>,
    commitment_hex: Vec<String>,
    blindings: Vec<Fp>,
}

#[derive(Clone)]
struct OpeningMemberArtifact {
    label: &'static str,
    n_r: usize,
    n_c: usize,
    r_row: Vec<Fp>,
}

#[derive(Clone)]
struct BatchedOpeningArtifact {
    group_label: String,
    n_c: usize,
    require_zero_w_final: bool,
    members: Vec<OpeningMemberArtifact>,
    proof: IpaProof,
}

struct OpeningVerifyJob {
    label: &'static str,
    commitment_hex: Vec<String>,
    commitments: Vec<GAffine>,
    expected_rows: usize,
    expected_cols: usize,
    require_zero_w_final: bool,
}

#[derive(Clone)]
struct ParsedCommitment {
    hex: Vec<String>,
    points: Vec<GAffine>,
}

#[derive(Clone)]
struct ParsedGcProofCore {
    arch: ModelArchitecture,
    split_id: String,
    graph_id: usize,
    ell: usize,
    n_real: usize,
    n_max: usize,
    d: usize,
    c: usize,
    h: usize,
    k: usize,
    d_max: usize,
    h_max: usize,
    scale: usize,
    arg_bits: usize,
    public_hash: Fp,
    public_hash_hex: String,
    vk_id: Fp,
    weight_fingerprint: Fp,
    x_active_cols: usize,
    h_active_cols: usize,
    w_graph_active_rows: usize,
    class_active_cols: usize,
    sparse_nnz: usize,
    sparse_row_col_val_padded_len: usize,
    spark_active_cols: usize,
    spark_audit_padded_len: usize,
    model_id: Fp,
    proof_digest: Fp,
    commitments: HashMap<&'static str, ParsedCommitment>,
    opening_digest: Fp,
    opening_artifacts: Vec<BatchedOpeningArtifact>,
    spark_roots: [Fp; 4],
    spark_row_col_val_audit: bool,
    spark_memory_conservation: bool,
    range_roots: [Fp; 3],
    claim_digest: Fp,
    constraint_commitment_digest: Fp,
    r1cs_residual_count: usize,
    semantic_residual_count: usize,
    mean_pool_residual_count: usize,
    argmax_residual_count: usize,
    range_residual_count: usize,
    sumcheck_proofs: Vec<LinearSumcheckProof>,
}

#[derive(Clone)]
struct SumcheckRound {
    g0: Fp,
    g1: Fp,
    challenge: Fp,
}

#[derive(Clone)]
struct LinearSumcheckProof {
    label: String,
    claim: Fp,
    logical_len: usize,
    num_vars: usize,
    rounds: Vec<SumcheckRound>,
    terminal: Fp,
}

fn py_value_error(message: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(message.into())
}

fn elapsed_ms(start: &Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn cached_hyrax_params(n_cols: usize, seed: [u8; 32]) -> Arc<HyraxParams> {
    let key = (n_cols, seed);
    let mut cache = HYRAX_PARAMS_CACHE
        .lock()
        .expect("Hyrax parameter cache lock poisoned");
    cache
        .entry(key)
        .or_insert_with(|| Arc::new(HyraxParams::new(n_cols, seed)))
        .clone()
}

fn cached_gc_vk_profile(
    arch: ModelArchitecture,
    n_max: usize,
    d: usize,
    c: usize,
    h: usize,
    k: usize,
    d_max: usize,
    h_max: usize,
) -> GcVkProfile {
    let key = GcVkProfileKey {
        arch,
        n_max,
        d,
        c,
        h,
        k,
        d_max,
        h_max,
    };
    let mut cache = GC_VK_PROFILE_CACHE
        .lock()
        .expect("graph-classification VK profile cache lock poisoned");
    *cache.entry(key).or_insert_with(|| GcVkProfile {
        vk_id: gc_model_id(arch, n_max, d, c, h, k, d_max, h_max),
    })
}

fn ensure_power_of_two(value: usize, name: &str) -> PyResult<()> {
    if value == 0 || (value & (value - 1)) != 0 {
        return Err(py_value_error(format!(
            "{name} must be a positive power of two"
        )));
    }
    Ok(())
}

fn active_x_cols(d: usize, d_max: usize) -> usize {
    d.min(d_max).max(1)
}

fn active_graph_cols(arch: ModelArchitecture, d: usize, k: usize, d_max: usize) -> usize {
    match arch {
        ModelArchitecture::SGC | ModelArchitecture::SSGC => d.min(d_max).max(1),
        ModelArchitecture::SIGN => (k + 1).saturating_mul(d).min(d_max).max(1),
        ModelArchitecture::APPNP => d_max.max(1),
    }
}

fn active_graph_cols_from_w_graph(
    arch: ModelArchitecture,
    d: usize,
    k: usize,
    d_max: usize,
    w_graph: &[i128],
    c: usize,
) -> usize {
    if arch != ModelArchitecture::APPNP {
        return active_graph_cols(arch, d, k, d_max);
    }
    let highest_nonzero = (0..d_max)
        .rev()
        .find(|row| (0..c).any(|col| w_graph[row * c + col] != 0))
        .map(|row| row + 1)
        .unwrap_or(1);
    highest_nonzero.min(d_max).max(1)
}

fn validate_public_h_active_cols(
    arch: ModelArchitecture,
    d: usize,
    k: usize,
    d_max: usize,
    h_active_cols: usize,
) -> bool {
    if h_active_cols == 0 || h_active_cols > d_max {
        return false;
    }
    match arch {
        ModelArchitecture::APPNP => true,
        _ => h_active_cols == active_graph_cols(arch, d, k, d_max),
    }
}

fn project_columns(
    matrix: &[i128],
    rows: usize,
    row_width: usize,
    active_cols: usize,
) -> Vec<i128> {
    let mut out = vec![0i128; rows.saturating_mul(active_cols)];
    for i in 0..rows {
        for j in 0..active_cols {
            out[i * active_cols + j] = matrix[i * row_width + j];
        }
    }
    out
}

fn project_rows_cols(
    matrix: &[i128],
    row_width: usize,
    active_rows: usize,
    active_cols: usize,
) -> Vec<i128> {
    let mut out = vec![0i128; active_rows.saturating_mul(active_cols)];
    for i in 0..active_rows {
        for j in 0..active_cols {
            out[i * active_cols + j] = matrix[i * row_width + j];
        }
    }
    out
}

fn project_strided_columns(
    matrix: &[i128],
    rows: usize,
    row_width: usize,
    active_cols: usize,
) -> Vec<i128> {
    project_columns(matrix, rows, row_width, active_cols)
}

fn ensure_zero_inactive_columns(
    label: &str,
    matrix: &[i128],
    rows: usize,
    row_width: usize,
    active_cols: usize,
) -> PyResult<()> {
    for i in 0..rows {
        for j in active_cols..row_width {
            if matrix[i * row_width + j] != 0 {
                return Err(py_value_error(format!(
                    "{label} inactive active-column-pruned entries must be zero"
                )));
            }
        }
    }
    Ok(())
}

fn ensure_zero_inactive_w_rows(
    w_graph: &[i128],
    d_max: usize,
    c: usize,
    active_rows: usize,
) -> PyResult<()> {
    for i in active_rows..d_max {
        for j in 0..c {
            if w_graph[i * c + j] != 0 {
                return Err(py_value_error(
                    "W_graph inactive active-column-pruned rows must be zero",
                ));
            }
        }
    }
    Ok(())
}

fn validate_i16_values(values: &[i128], name: &str) -> PyResult<()> {
    for value in values {
        if *value < i16::MIN as i128 || *value > i16::MAX as i128 {
            return Err(py_value_error(format!(
                "{name} contains a value outside INT16"
            )));
        }
    }
    Ok(())
}

fn validate_dimensions(
    arch: ModelArchitecture,
    n_real: usize,
    n_max: usize,
    d: usize,
    c: usize,
    h: usize,
    k: usize,
    d_max: usize,
    h_max: usize,
) -> PyResult<()> {
    if n_real == 0 || n_real > n_max {
        return Err(py_value_error("N_real must satisfy 1 <= N_real <= N_max"));
    }
    for (name, value) in [
        ("N_max", n_max),
        ("d", d),
        ("c", c),
        ("h", h),
        ("K", k),
        ("D_max", d_max),
        ("H_max", h_max),
    ] {
        if value == 0 {
            return Err(py_value_error(format!("{name} must be positive")));
        }
    }
    ensure_power_of_two(n_max, "N_max")?;
    ensure_power_of_two(d_max, "D_max")?;
    ensure_power_of_two(h_max, "H_max")?;
    if d > d_max {
        return Err(py_value_error("d must not exceed D_max"));
    }
    if h > h_max {
        return Err(py_value_error("h must not exceed H_max"));
    }
    if arch == ModelArchitecture::SIGN && d_max < (k + 1).saturating_mul(d) {
        return Err(py_value_error("SIGN requires D_max >= (K+1)*d"));
    }
    if arch == ModelArchitecture::APPNP && d_max < d.max(c) {
        return Err(py_value_error("APPNP requires D_max >= max(d, c)"));
    }
    Ok(())
}

fn validate_mask_prefix(mask: &[i128], n_real: usize, n_max: usize) -> PyResult<()> {
    if mask.len() != n_max {
        return Err(py_value_error("mask length does not equal N_max"));
    }
    for (idx, value) in mask.iter().enumerate() {
        let expected = if idx < n_real { 1 } else { 0 };
        if *value != expected {
            return Err(py_value_error(
                "mask must be the canonical prefix mask for the single graph",
            ));
        }
    }
    Ok(())
}

fn validate_public_padding(
    x_public: &[i128],
    a_public: &[i128],
    n_real: usize,
    n_max: usize,
    d: usize,
) -> PyResult<()> {
    for i in n_real..n_max {
        for j in 0..d {
            if x_public[i * d + j] != 0 {
                return Err(py_value_error("X padding rows must be zero"));
            }
        }
    }
    for i in 0..n_max {
        for j in 0..n_max {
            if (i >= n_real || j >= n_real) && a_public[i * n_max + j] != 0 {
                return Err(py_value_error("A_hat padding rows/cols must be zero"));
            }
        }
    }
    Ok(())
}

fn validate_sparse_encoding(
    a_public: &[i128],
    rows: &[usize],
    cols: &[usize],
    vals: &[i128],
    n_max: usize,
) -> PyResult<usize> {
    if rows.len() != cols.len() || rows.len() != vals.len() {
        return Err(py_value_error(
            "row/col/val sparse arrays must have equal length",
        ));
    }
    ensure_power_of_two(rows.len().max(1), "sparse row/col/val length")?;

    let mut expected = Vec::new();
    for i in 0..n_max {
        for j in 0..n_max {
            let value = a_public[i * n_max + j];
            if value != 0 {
                expected.push((i, j, value));
            }
        }
    }
    let padded_len = next_power_of_two(expected.len().max(1));
    if rows.len() != padded_len {
        return Err(py_value_error(
            "sparse row/col/val length must be the canonical next power-of-two padding",
        ));
    }

    for (idx, (row, col, value)) in expected.iter().enumerate() {
        if rows[idx] != *row || cols[idx] != *col || vals[idx] != *value {
            return Err(py_value_error(
                "sparse row/col/val must match the canonical row-major A_hat nonzero expansion",
            ));
        }
    }
    for idx in expected.len()..rows.len() {
        if rows[idx] != 0 || cols[idx] != 0 || vals[idx] != 0 {
            return Err(py_value_error("sparse padding entries must be zero"));
        }
    }
    Ok(expected.len())
}

fn next_power_of_two(value: usize) -> usize {
    if value <= 1 {
        1
    } else {
        value.next_power_of_two()
    }
}

fn log2_ceil(value: usize) -> usize {
    next_power_of_two(value).trailing_zeros() as usize
}

fn fp_from_u128(value: u128) -> Fp {
    Fp::from_le_bytes_mod_order(&value.to_le_bytes())
}

fn fp_from_i128(value: i128) -> Fp {
    if value >= 0 {
        fp_from_u128(value as u128)
    } else {
        -fp_from_u128((-value) as u128)
    }
}

fn fp_hex(value: &Fp) -> String {
    hex::encode(value.into_bigint().to_bytes_be())
}

fn fp_from_hex(value: &str) -> PyResult<Fp> {
    let bytes =
        hex::decode(value).map_err(|err| py_value_error(format!("invalid field hex: {err}")))?;
    Ok(Fp::from_be_bytes_mod_order(&bytes))
}

fn affine_hex(point: &crate::curve_config::GAffine) -> PyResult<String> {
    let mut bytes = Vec::new();
    point
        .serialize_compressed(&mut bytes)
        .map_err(|err| py_value_error(format!("curve point serialization failed: {err}")))?;
    Ok(hex::encode(bytes))
}

fn affine_from_hex(value: &str) -> PyResult<GAffine> {
    let bytes =
        hex::decode(value).map_err(|err| py_value_error(format!("invalid point hex: {err}")))?;
    GAffine::deserialize_compressed(&bytes[..])
        .map_err(|err| py_value_error(format!("curve point deserialization failed: {err}")))
}

fn absorb_i128(sponge: &mut PoseidonSponge, value: i128) {
    sponge.absorb_scalar(&fp_from_i128(value));
}

fn absorb_str(sponge: &mut PoseidonSponge, value: &str) {
    let scalar = Fp::from_le_bytes_mod_order(value.as_bytes());
    sponge.absorb_scalar(&scalar);
}

fn array2_i128(
    array: &PyReadonlyArray2<'_, i64>,
    rows: usize,
    cols: usize,
    name: &str,
) -> PyResult<Vec<i128>> {
    let view = array.as_array();
    let shape = view.shape();
    if shape != [rows, cols] {
        return Err(py_value_error(format!(
            "{name} shape mismatch: expected ({rows}, {cols}), got ({}, {})",
            shape[0], shape[1]
        )));
    }
    Ok(view.iter().map(|v| i128::from(*v)).collect())
}

fn array1_i128(array: &PyReadonlyArray1<'_, i64>, len: usize, name: &str) -> PyResult<Vec<i128>> {
    let view = array.as_array();
    let shape = view.shape();
    if shape != [len] {
        return Err(py_value_error(format!(
            "{name} length mismatch: expected {len}, got {}",
            shape[0]
        )));
    }
    Ok(view.iter().map(|v| i128::from(*v)).collect())
}

fn array1_usize(array: &PyReadonlyArray1<'_, i64>, name: &str) -> PyResult<Vec<usize>> {
    let view = array.as_array();
    let mut out = Vec::with_capacity(view.len());
    for value in view.iter() {
        if *value < 0 {
            return Err(py_value_error(format!("{name} contains a negative index")));
        }
        out.push(*value as usize);
    }
    Ok(out)
}

fn div_euclid_i128(value: i128, divisor: i128) -> (i128, i128) {
    let mut q = value / divisor;
    let mut r = value % divisor;
    if r < 0 {
        q -= 1;
        r += divisor;
    }
    (q, r)
}

fn sparse_diffuse_trunc_checked_active(
    rows: &[usize],
    cols: &[usize],
    vals: &[i128],
    prev: &[i128],
    n_max: usize,
    row_width: usize,
    active_cols: usize,
    scale: i128,
    trunc_remainders: &mut Vec<i128>,
    semantic_residuals: &mut Vec<i128>,
) -> Vec<i128> {
    let mut raw = vec![0i128; n_max * active_cols];
    for ((&row, &col), &val) in rows.iter().zip(cols.iter()).zip(vals.iter()) {
        if val == 0 || row >= n_max || col >= n_max {
            continue;
        }
        for j in 0..active_cols {
            raw[row * active_cols + j] += val * prev[col * row_width + j];
        }
    }
    let mut out = vec![0i128; n_max * row_width];
    for i in 0..n_max {
        for j in 0..active_cols {
            let value = raw[i * active_cols + j];
            let (q, r) = div_euclid_i128(value, scale);
            out[i * row_width + j] = q;
            trunc_remainders.push(r);
            semantic_residuals.push(value - scale * q - r);
        }
    }
    out
}

fn matmul_trunc_checked(
    left: &[i128],
    left_rows: usize,
    inner: usize,
    right: &[i128],
    right_cols: usize,
    scale: i128,
    trunc_remainders: &mut Vec<i128>,
    semantic_residuals: &mut Vec<i128>,
) -> Vec<i128> {
    let mut out = vec![0i128; left_rows * right_cols];
    for i in 0..left_rows {
        for j in 0..right_cols {
            let mut raw = 0i128;
            for k in 0..inner {
                raw += left[i * inner + k] * right[k * right_cols + j];
            }
            let (q, r) = div_euclid_i128(raw, scale);
            out[i * right_cols + j] = q;
            trunc_remainders.push(r);
            semantic_residuals.push(raw - scale * q - r);
        }
    }
    out
}

fn matmul_trunc_checked_strided(
    left: &[i128],
    left_rows: usize,
    left_stride: usize,
    inner_active: usize,
    right: &[i128],
    right_cols: usize,
    scale: i128,
    trunc_remainders: &mut Vec<i128>,
    semantic_residuals: &mut Vec<i128>,
) -> Vec<i128> {
    let mut out = vec![0i128; left_rows * right_cols];
    for i in 0..left_rows {
        for j in 0..right_cols {
            let mut raw = 0i128;
            for k in 0..inner_active {
                raw += left[i * left_stride + k] * right[k * right_cols + j];
            }
            let (q, r) = div_euclid_i128(raw, scale);
            out[i * right_cols + j] = q;
            trunc_remainders.push(r);
            semantic_residuals.push(raw - scale * q - r);
        }
    }
    out
}

fn matmul_trunc_checked_right_strided_active(
    left: &[i128],
    left_rows: usize,
    inner: usize,
    right: &[i128],
    right_stride: usize,
    out_stride: usize,
    active_cols: usize,
    scale: i128,
    trunc_remainders: &mut Vec<i128>,
    semantic_residuals: &mut Vec<i128>,
) -> Vec<i128> {
    let mut out = vec![0i128; left_rows * out_stride];
    for i in 0..left_rows {
        for j in 0..active_cols {
            let mut raw = 0i128;
            for k in 0..inner {
                raw += left[i * inner + k] * right[k * right_stride + j];
            }
            let (q, r) = div_euclid_i128(raw, scale);
            out[i * out_stride + j] = q;
            trunc_remainders.push(r);
            semantic_residuals.push(raw - scale * q - r);
        }
    }
    out
}

fn relu(values: &[i128]) -> Vec<i128> {
    values.iter().map(|v| (*v).max(0)).collect()
}

#[cfg(test)]
fn mean_pool_graph(
    h: &[i128],
    mask: &[i128],
    n_real: usize,
    n_max: usize,
    width: usize,
) -> (Vec<i128>, Vec<i128>) {
    let mut pooled = vec![0i128; width];
    let mut remainders = vec![0i128; width];
    let denom = n_real as i128;
    for j in 0..width {
        let mut sum = 0i128;
        for i in 0..n_max {
            sum += mask[i] * h[i * width + j];
        }
        let (q, r) = div_euclid_i128(sum, denom);
        pooled[j] = q;
        remainders[j] = r;
    }
    (pooled, remainders)
}

fn mean_pool_graph_active(
    h: &[i128],
    mask: &[i128],
    n_real: usize,
    n_max: usize,
    row_width: usize,
    active_cols: usize,
) -> (Vec<i128>, Vec<i128>) {
    let mut pooled = vec![0i128; row_width];
    let mut remainders = vec![0i128; active_cols];
    let denom = n_real as i128;
    for j in 0..active_cols {
        let mut sum = 0i128;
        for i in 0..n_max {
            sum += mask[i] * h[i * row_width + j];
        }
        let (q, r) = div_euclid_i128(sum, denom);
        pooled[j] = q;
        remainders[j] = r;
    }
    (pooled, remainders)
}

fn compute_trace(
    arch: ModelArchitecture,
    x_public: &[i128],
    mask: &[i128],
    edge_rows: &[usize],
    edge_cols: &[usize],
    edge_vals: &[i128],
    w_pre0: &[i128],
    w_pre1: &[i128],
    w_graph: &[i128],
    n_real: usize,
    n_max: usize,
    d: usize,
    c: usize,
    h: usize,
    k: usize,
    d_max: usize,
    h_active_cols: usize,
    scale: i128,
    arg_bits: usize,
) -> PyResult<Trace> {
    if n_real == 0 || n_real > n_max {
        return Err(py_value_error("N_real must satisfy 1 <= N_real <= N_max"));
    }
    if mask.iter().sum::<i128>() != n_real as i128 {
        return Err(py_value_error("mask sum does not equal N_real"));
    }

    let mut trunc_remainders = Vec::new();
    let mut semantic_residuals = Vec::new();
    let mut x_base = vec![0i128; n_max * d_max];
    let x_active_cols = active_x_cols(d, d_max);
    if !validate_public_h_active_cols(arch, d, k, d_max, h_active_cols) {
        return Err(py_value_error("invalid active graph column profile"));
    }
    for i in 0..n_max {
        if mask[i] != 0 && mask[i] != 1 {
            return Err(py_value_error("mask must be Boolean"));
        }
        for j in 0..d {
            x_base[i * d_max + j] = x_public[i * d + j];
        }
        for j in 0..x_active_cols {
            let expected = if j < d { x_public[i * d + j] } else { 0 };
            semantic_residuals.push(x_base[i * d_max + j] - expected);
        }
    }

    let h_matrix = match arch {
        ModelArchitecture::SGC => {
            let mut cur = x_base.clone();
            for _ in 0..k {
                cur = sparse_diffuse_trunc_checked_active(
                    edge_rows,
                    edge_cols,
                    edge_vals,
                    &cur,
                    n_max,
                    d_max,
                    d,
                    scale,
                    &mut trunc_remainders,
                    &mut semantic_residuals,
                );
            }
            cur
        }
        ModelArchitecture::SSGC => {
            let mut cur = x_base.clone();
            let mut acc = vec![0i128; n_max * d_max];
            for _ in 0..k {
                cur = sparse_diffuse_trunc_checked_active(
                    edge_rows,
                    edge_cols,
                    edge_vals,
                    &cur,
                    n_max,
                    d_max,
                    d,
                    scale,
                    &mut trunc_remainders,
                    &mut semantic_residuals,
                );
                for i in 0..n_max {
                    for j in 0..d {
                        acc[i * d_max + j] += cur[i * d_max + j];
                    }
                }
            }
            let mut avg = vec![0i128; n_max * d_max];
            for i in 0..n_max {
                for j in 0..d {
                    let value = acc[i * d_max + j];
                    let (q, r) = div_euclid_i128(value, k.max(1) as i128);
                    avg[i * d_max + j] = q;
                    trunc_remainders.push(r);
                    semantic_residuals.push(value - (k.max(1) as i128) * q - r);
                }
            }
            avg
        }
        ModelArchitecture::SIGN => {
            let required = (k + 1) * d;
            if d_max < required {
                return Err(py_value_error(format!(
                    "SIGN requires D_max >= (K+1)*d = {required}"
                )));
            }
            let mut branches: Vec<Vec<i128>> = Vec::with_capacity(k + 1);
            branches.push(x_base.clone());
            let mut cur = x_base.clone();
            for _ in 0..k {
                cur = sparse_diffuse_trunc_checked_active(
                    edge_rows,
                    edge_cols,
                    edge_vals,
                    &cur,
                    n_max,
                    d_max,
                    d,
                    scale,
                    &mut trunc_remainders,
                    &mut semantic_residuals,
                );
                branches.push(cur.clone());
            }
            let mut concat = vec![0i128; n_max * d_max];
            for i in 0..n_max {
                for (block, branch) in branches.iter().enumerate() {
                    for j in 0..d {
                        concat[i * d_max + block * d + j] = branch[i * d_max + j];
                    }
                }
                for j in 0..required {
                    let block = j / d.max(1);
                    let within = j % d.max(1);
                    let expected = if block < branches.len() && within < d {
                        branches[block][i * d_max + within]
                    } else {
                        0
                    };
                    semantic_residuals.push(concat[i * d_max + j] - expected);
                }
            }
            concat
        }
        ModelArchitecture::APPNP => {
            let u_pre = matmul_trunc_checked(
                x_public,
                n_max,
                d,
                w_pre0,
                h,
                scale,
                &mut trunc_remainders,
                &mut semantic_residuals,
            );
            let relu_pre = relu(&u_pre);
            for (input, output) in u_pre.iter().zip(relu_pre.iter()) {
                let expected = (*input).max(0);
                semantic_residuals.push(*output - expected);
            }
            let appnp_base = matmul_trunc_checked_right_strided_active(
                &relu_pre,
                n_max,
                h,
                w_pre1,
                d_max,
                d_max,
                h_active_cols,
                scale,
                &mut trunc_remainders,
                &mut semantic_residuals,
            );
            let alpha = (scale / 10).max(1);
            let keep = scale - alpha;
            let mut cur = appnp_base.clone();
            for _ in 0..k {
                let diffused = sparse_diffuse_trunc_checked_active(
                    edge_rows,
                    edge_cols,
                    edge_vals,
                    &cur,
                    n_max,
                    d_max,
                    h_active_cols,
                    scale,
                    &mut trunc_remainders,
                    &mut semantic_residuals,
                );
                let mut next = vec![0i128; n_max * d_max];
                for i in 0..n_max {
                    for j in 0..h_active_cols {
                        let idx = i * d_max + j;
                        let raw = keep * diffused[idx] + alpha * appnp_base[idx];
                        let (q, r) = div_euclid_i128(raw, scale);
                        next[idx] = q;
                        trunc_remainders.push(r);
                        semantic_residuals.push(raw - scale * q - r);
                    }
                }
                cur = next;
            }
            cur
        }
    };

    let (g, pool_remainders) =
        mean_pool_graph_active(&h_matrix, mask, n_real, n_max, d_max, h_active_cols);
    let y_g = matmul_trunc_checked_strided(
        &g,
        1,
        d_max,
        h_active_cols,
        w_graph,
        c,
        scale,
        &mut trunc_remainders,
        &mut semantic_residuals,
    );
    let (ell, y_ell) = y_g
        .iter()
        .enumerate()
        .max_by_key(|(_, value)| *value)
        .map(|(idx, value)| (idx, *value))
        .ok_or_else(|| py_value_error("empty graph logits"))?;
    let mut deltas = Vec::with_capacity(c);
    let max_delta = if arg_bits >= 127 {
        i128::MAX
    } else {
        1i128 << arg_bits
    };
    for value in &y_g {
        let delta = y_ell - *value;
        if delta < 0 || delta >= max_delta {
            return Err(py_value_error("Argmax range proof bit width is too small"));
        }
        deltas.push(delta);
    }

    Ok(Trace {
        x_base,
        h: h_matrix,
        h_cols: d_max,
        active_h_cols: h_active_cols,
        g,
        y_g,
        ell,
        pool_remainders,
        trunc_remainders,
        argmax_deltas: deltas,
        semantic_residuals,
    })
}

fn range_digest(values: &[i128], bit_width: usize, label: &str) -> PyResult<Fp> {
    let mut sponge = PoseidonSponge::new();
    absorb_str(&mut sponge, "Spartan-ZENO-GC-range-digest-accumulator-v2");
    absorb_str(&mut sponge, label);
    sponge.absorb_scalar(&Fp::from(values.len() as u64));
    sponge.absorb_scalar(&Fp::from(bit_width as u64));
    let max_value = if bit_width >= 127 {
        i128::MAX
    } else {
        1i128 << bit_width
    };
    let mut theta_sponge = PoseidonSponge::new();
    absorb_str(&mut theta_sponge, "Spartan-ZENO-GC-range-digest-theta-v2");
    absorb_str(&mut theta_sponge, label);
    theta_sponge.absorb_scalar(&Fp::from(values.len() as u64));
    theta_sponge.absorb_scalar(&Fp::from(bit_width as u64));
    theta_sponge.finalize_absorption();
    let mut theta = theta_sponge.squeeze();
    if theta.is_zero() {
        theta = Fp::from(17u64);
    }

    let mut power = Fp::from(1u64);
    let mut linear_acc = Fp::zero();
    let mut weighted_acc = Fp::zero();
    let mut square_acc = Fp::zero();
    for value in values {
        if *value < 0 || *value >= max_value {
            return Err(py_value_error(format!("{label} value outside range")));
        }
        let value_fp = fp_from_i128(*value);
        linear_acc += value_fp;
        weighted_acc += power * value_fp;
        square_acc += value_fp * value_fp;
        power *= theta;
    }
    sponge.absorb_scalar(&linear_acc);
    sponge.absorb_scalar(&weighted_acc);
    sponge.absorb_scalar(&square_acc);
    sponge.finalize_absorption();
    Ok(sponge.squeeze())
}

fn public_input_hash(
    arch: ModelArchitecture,
    split_id: &str,
    graph_id: usize,
    x_public: &[i128],
    _a_hat: &[i128],
    mask: &[i128],
    edge_rows: &[usize],
    edge_cols: &[usize],
    edge_vals: &[i128],
    n_real: usize,
    n_max: usize,
    d: usize,
    c: usize,
    h: usize,
    k: usize,
    d_max: usize,
    h_max: usize,
) -> Fp {
    let mut sponge = PoseidonSponge::new();
    absorb_str(&mut sponge, "Spartan-ZENO-GC-public-v1");
    absorb_str(&mut sponge, arch.as_str());
    absorb_str(&mut sponge, split_id);
    sponge.absorb_scalar(&Fp::from(graph_id as u64));
    for value in [n_real, n_max, d, c, h, k, d_max, h_max] {
        sponge.absorb_scalar(&Fp::from(value as u64));
    }
    // The graph topology is bound canonically by row/col/val sparse arrays.
    // Dense A_hat is still validated against that encoding, but hashing all
    // N_max^2 padded entries is prohibitively expensive for large graph
    // datasets such as REDDIT-BINARY under global-max containers.
    for value in x_public.iter().chain(mask.iter()) {
        absorb_i128(&mut sponge, *value);
    }
    for value in edge_rows {
        sponge.absorb_scalar(&Fp::from(*value as u64));
    }
    for value in edge_cols {
        sponge.absorb_scalar(&Fp::from(*value as u64));
    }
    for value in edge_vals {
        absorb_i128(&mut sponge, *value);
    }
    sponge.finalize_absorption();
    sponge.squeeze()
}

fn sha_update_usize(hasher: &mut Sha256, value: usize) {
    hasher.update((value as u64).to_be_bytes());
}

fn sha_update_i128(hasher: &mut Sha256, value: i128) {
    hasher.update(value.to_be_bytes());
}

fn sha256_key(bytes: &[u8]) -> [u8; 32] {
    let digest = Sha256::digest(bytes);
    let mut key = [0u8; 32];
    key.copy_from_slice(&digest);
    key
}

fn public_input_hash_cache_key(
    arch: ModelArchitecture,
    split_id: &str,
    graph_id: usize,
    x_public: &[i128],
    _a_hat: &[i128],
    mask: &[i128],
    edge_rows: &[usize],
    edge_cols: &[usize],
    edge_vals: &[i128],
    n_real: usize,
    n_max: usize,
    d: usize,
    c: usize,
    h: usize,
    k: usize,
    d_max: usize,
    h_max: usize,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"Spartan-ZENO-GC-public-cache-v1");
    hasher.update(arch.as_str().as_bytes());
    sha_update_usize(&mut hasher, split_id.len());
    hasher.update(split_id.as_bytes());
    for value in [graph_id, n_real, n_max, d, c, h, k, d_max, h_max] {
        sha_update_usize(&mut hasher, value);
    }
    for values in [x_public, mask, edge_vals] {
        sha_update_usize(&mut hasher, values.len());
        for value in values {
            sha_update_i128(&mut hasher, *value);
        }
    }
    for values in [edge_rows, edge_cols] {
        sha_update_usize(&mut hasher, values.len());
        for value in values {
            sha_update_usize(&mut hasher, *value);
        }
    }
    hasher.finalize().into()
}

#[allow(clippy::too_many_arguments)]
fn cached_public_input_hash(
    arch: ModelArchitecture,
    split_id: &str,
    graph_id: usize,
    x_public: &[i128],
    a_hat: &[i128],
    mask: &[i128],
    edge_rows: &[usize],
    edge_cols: &[usize],
    edge_vals: &[i128],
    n_real: usize,
    n_max: usize,
    d: usize,
    c: usize,
    h: usize,
    k: usize,
    d_max: usize,
    h_max: usize,
) -> Fp {
    let key = public_input_hash_cache_key(
        arch, split_id, graph_id, x_public, a_hat, mask, edge_rows, edge_cols, edge_vals, n_real,
        n_max, d, c, h, k, d_max, h_max,
    );
    if let Some(value) = PUBLIC_INPUT_HASH_CACHE
        .lock()
        .expect("public input hash cache lock poisoned")
        .get(&key)
        .copied()
    {
        return value;
    }
    let value = public_input_hash(
        arch, split_id, graph_id, x_public, a_hat, mask, edge_rows, edge_cols, edge_vals, n_real,
        n_max, d, c, h, k, d_max, h_max,
    );
    PUBLIC_INPUT_HASH_CACHE
        .lock()
        .expect("public input hash cache lock poisoned")
        .insert(key, value);
    value
}

fn gc_model_id(
    arch: ModelArchitecture,
    n_max: usize,
    d: usize,
    c: usize,
    h: usize,
    k: usize,
    d_max: usize,
    h_max: usize,
) -> Fp {
    let mut sponge = PoseidonSponge::new();
    absorb_str(&mut sponge, "VK_GC");
    absorb_str(&mut sponge, arch.as_str());
    for value in [n_max, d, c, h, k, d_max, h_max] {
        sponge.absorb_scalar(&Fp::from(value as u64));
    }
    sponge.finalize_absorption();
    sponge.squeeze()
}

fn weight_commitment_fingerprint(
    arch: ModelArchitecture,
    vk_id: &Fp,
    w_graph_commitments: &[String],
    w_pre0_commitments: Option<&[String]>,
    w_pre1_commitments: Option<&[String]>,
) -> Fp {
    let mut sponge = PoseidonSponge::new();
    absorb_str(&mut sponge, "Spartan-ZENO-GC-weight-fingerprint-v1");
    absorb_str(&mut sponge, arch.as_str());
    sponge.absorb_scalar(vk_id);
    for (label, rows) in [
        ("C_W_graph", Some(w_graph_commitments)),
        ("C_W_pre0", w_pre0_commitments),
        ("C_W_pre1", w_pre1_commitments),
    ] {
        absorb_str(&mut sponge, label);
        match rows {
            Some(items) => {
                sponge.absorb_scalar(&Fp::from(items.len() as u64));
                for row in items {
                    absorb_str(&mut sponge, row);
                }
            }
            None => sponge.absorb_scalar(&Fp::zero()),
        }
    }
    sponge.finalize_absorption();
    sponge.squeeze()
}

fn bind_model_id(vk_id: &Fp, weight_fingerprint: &Fp) -> Fp {
    let mut sponge = PoseidonSponge::new();
    absorb_str(&mut sponge, "Spartan-ZENO-GC-model-id-v1");
    sponge.absorb_scalar(vk_id);
    sponge.absorb_scalar(weight_fingerprint);
    sponge.finalize_absorption();
    sponge.squeeze()
}

fn build_spark_roots(
    x_base: &[i128],
    width: usize,
    edge_rows: &[usize],
    edge_cols: &[usize],
    edge_vals: &[i128],
    n_real: usize,
) -> SparkRoots {
    let eta: Vec<Fp> = (0..width).map(|idx| Fp::from((idx as u64) + 3)).collect();
    let delta_1 = Fp::from(7u64);
    let delta_2 = Fp::from(11u64);
    let tau = Fp::from(13u64);

    let mut init = Vec::with_capacity(n_real);
    for i in 0..n_real {
        let mut row = Vec::with_capacity(width);
        for j in 0..width {
            row.push(fp_from_i128(x_base[i * width + j]));
        }
        init.push(MemoryOp {
            addr: Fp::from(i as u64),
            val_compressed: crate::spark::compress_feature(&row, &eta),
            ts: Fp::zero(),
        });
    }
    let audit = init.clone();

    let mut read = Vec::new();
    for (idx, ((&row, &col), &val)) in edge_rows
        .iter()
        .zip(edge_cols.iter())
        .zip(edge_vals.iter())
        .enumerate()
    {
        if val == 0 {
            continue;
        }
        read.push(MemoryOp {
            addr: Fp::from(((row << 32) ^ col) as u64),
            val_compressed: fp_from_i128(val),
            ts: Fp::from(idx as u64),
        });
    }
    let write = read.clone();

    let init_tags = build_algebraic_tags(&init, &delta_1, &delta_2, &tau);
    let read_tags = build_algebraic_tags(&read, &delta_1, &delta_2, &tau);
    let write_tags = build_algebraic_tags(&write, &delta_1, &delta_2, &tau);
    let audit_tags = build_algebraic_tags(&audit, &delta_1, &delta_2, &tau);
    let padded_len = next_power_of_two(
        init_tags
            .len()
            .max(read_tags.len())
            .max(write_tags.len())
            .max(audit_tags.len())
            .max(1),
    );
    let height = log2_ceil(padded_len);
    let (init_tree, read_tree, write_tree, audit_tree) =
        build_memory_trees(&init_tags, &read_tags, &write_tags, &audit_tags, height);
    let conservation = verify_memory_conservation(
        &init_tree.root,
        &write_tree.root,
        &read_tree.root,
        &audit_tree.root,
    ) && grand_product(&init_tags) == grand_product(&audit_tags)
        && grand_product(&read_tags) == grand_product(&write_tags);

    SparkRoots {
        init_root: init_tree.root,
        read_root: read_tree.root,
        write_root: write_tree.root,
        audit_root: audit_tree.root,
        padded_len,
        conservation,
    }
}

fn commit_i128_matrix(
    label: &'static str,
    matrix: &[i128],
    rows: usize,
    cols: usize,
    nonzero_rows: Option<&[usize]>,
) -> PyResult<MatrixCommitment> {
    let padded_rows = next_power_of_two(rows.max(1));
    let padded_cols = next_power_of_two(cols.max(1));
    let n_r = log2_ceil(padded_rows);
    let n_c = log2_ceil(padded_cols);
    let params = cached_hyrax_params(n_c, HYRAX_GC_SEED);
    let mut field_matrix = vec![Fp::zero(); padded_rows * padded_cols];
    for i in 0..rows {
        for j in 0..cols {
            field_matrix[i * padded_cols + j] = fp_from_i128(matrix[i * cols + j]);
        }
    }
    let padded_nonzero_rows: Option<Vec<usize>> =
        nonzero_rows.map(|rows| rows.iter().copied().filter(|i| *i < padded_rows).collect());
    let (points, blindings) = commit_matrix(
        &field_matrix,
        padded_rows,
        padded_cols,
        params.as_ref(),
        padded_nonzero_rows.as_deref(),
    );
    let commitment_hex = points
        .iter()
        .map(affine_hex)
        .collect::<PyResult<Vec<_>>>()?;
    Ok(MatrixCommitment {
        label,
        matrix: field_matrix,
        rows: padded_rows,
        cols: padded_cols,
        n_r,
        n_c,
        commitments: points,
        commitment_hex,
        blindings,
    })
}

fn matrix_to_py_list(py: Python<'_>, rows: &[String]) -> PyObject {
    let items: Vec<PyObject> = rows.iter().map(|item| item.clone().into_py(py)).collect();
    PyList::new_bound(py, items).into_py(py)
}

fn set_commitment(
    dict: &Bound<'_, PyDict>,
    py: Python<'_>,
    name: &str,
    rows: Vec<String>,
) -> PyResult<()> {
    dict.set_item(name, matrix_to_py_list(py, &rows))
}

fn opening_r_row(
    label: &str,
    public_hash: &Fp,
    model_id: &Fp,
    commitments: &[GAffine],
    n_r: usize,
) -> Vec<Fp> {
    let mut sponge = PoseidonSponge::new();
    absorb_str(&mut sponge, "Spartan-ZENO-GC-Hyrax-rrow-v2");
    absorb_str(&mut sponge, label);
    sponge.absorb_scalar(public_hash);
    sponge.absorb_scalar(model_id);
    for point in commitments {
        let mut bytes = Vec::new();
        point
            .serialize_compressed(&mut bytes)
            .expect("curve point serialization should not fail");
        sponge.absorb_scalar(&Fp::from_le_bytes_mod_order(&bytes));
    }
    sponge.finalize_absorption();
    sponge.squeeze_n(n_r)
}

fn opening_r_row_cached(
    label: &'static str,
    public_hash: &Fp,
    model_id: &Fp,
    commitment_hex: &[String],
    commitments: &[GAffine],
    n_r: usize,
) -> Vec<Fp> {
    let key = OpeningRRowCacheKey {
        label,
        public_hash: fp_hex(public_hash),
        model_id: fp_hex(model_id),
        n_r,
        commitment_hex: commitment_hex.to_vec(),
    };
    if let Some(value) = OPENING_R_ROW_CACHE
        .lock()
        .expect("opening r_row cache lock poisoned")
        .get(&key)
        .cloned()
    {
        return value;
    }
    let value = opening_r_row(label, public_hash, model_id, commitments, n_r);
    OPENING_R_ROW_CACHE
        .lock()
        .expect("opening r_row cache lock poisoned")
        .insert(key, value.clone());
    value
}

fn opening_batch_alpha(
    n_c: usize,
    require_zero_w_final: bool,
    member_idx: usize,
    label: &str,
    public_hash: &Fp,
    model_id: &Fp,
    commitment_hex: &[String],
) -> Fp {
    let mut sponge = PoseidonSponge::new();
    absorb_str(&mut sponge, "Spartan-ZENO-GC-Hyrax-batch-alpha-v1");
    sponge.absorb_scalar(&Fp::from(n_c as u64));
    sponge.absorb_scalar(&Fp::from(if require_zero_w_final { 1u64 } else { 0u64 }));
    sponge.absorb_scalar(&Fp::from(member_idx as u64));
    absorb_str(&mut sponge, label);
    sponge.absorb_scalar(public_hash);
    sponge.absorb_scalar(model_id);
    sponge.absorb_scalar(&Fp::from(commitment_hex.len() as u64));
    for row in commitment_hex {
        absorb_str(&mut sponge, row);
    }
    sponge.finalize_absorption();
    let mut alpha = sponge.squeeze();
    if alpha.is_zero() {
        alpha = Fp::from(1u64);
    }
    alpha
}

fn batched_opening_sponge(
    group_label: &str,
    n_c: usize,
    require_zero_w_final: bool,
    members: &[OpeningMemberArtifact],
    public_hash: &Fp,
    model_id: &Fp,
) -> PoseidonSponge {
    let mut sponge = PoseidonSponge::new();
    absorb_str(&mut sponge, "Spartan-ZENO-GC-Hyrax-IPA-batch-v1");
    absorb_str(&mut sponge, group_label);
    sponge.absorb_scalar(&Fp::from(n_c as u64));
    sponge.absorb_scalar(&Fp::from(if require_zero_w_final { 1u64 } else { 0u64 }));
    sponge.absorb_scalar(&Fp::from(members.len() as u64));
    sponge.absorb_scalar(public_hash);
    sponge.absorb_scalar(model_id);
    for member in members {
        absorb_str(&mut sponge, member.label);
        sponge.absorb_scalar(&Fp::from(member.n_r as u64));
        sponge.absorb_scalar(&Fp::from(member.n_c as u64));
        sponge.absorb_scalars(&member.r_row);
    }
    sponge.finalize_absorption();
    sponge
}

fn prove_batched_opening_group(
    n_c: usize,
    require_zero_w_final: bool,
    matrices: Vec<&MatrixCommitment>,
    public_hash: &Fp,
    model_id: &Fp,
) -> BatchedOpeningArtifact {
    let params = cached_hyrax_params(n_c, HYRAX_GC_SEED);
    let group_label = format!(
        "batch_nc{}_{}",
        n_c,
        if require_zero_w_final {
            "residual"
        } else {
            "witness"
        }
    );
    let mut aggregate_w = vec![Fp::zero(); 1usize << n_c];
    let mut aggregate_z = Fp::zero();
    let artifacts = matrices
        .par_iter()
        .enumerate()
        .map(|(member_idx, matrix)| {
            assert_eq!(matrix.n_c, n_c, "batched opening group n_c mismatch");
            let r_row = opening_r_row_cached(
                matrix.label,
                public_hash,
                model_id,
                &matrix.commitment_hex,
                &matrix.commitments,
                matrix.n_r,
            );
            let (w, z) = flatten_rows(
                &matrix.matrix,
                matrix.rows,
                matrix.cols,
                &r_row,
                &matrix.blindings,
                matrix.n_r,
            );
            let alpha = opening_batch_alpha(
                n_c,
                require_zero_w_final,
                member_idx,
                matrix.label,
                public_hash,
                model_id,
                &matrix.commitment_hex,
            );
            (
                member_idx,
                alpha,
                w,
                z,
                OpeningMemberArtifact {
                    label: matrix.label,
                    n_r: matrix.n_r,
                    n_c: matrix.n_c,
                    r_row,
                },
            )
        })
        .collect::<Vec<_>>();

    let mut members = Vec::with_capacity(artifacts.len());
    for (_member_idx, alpha, w, z, member) in artifacts {
        for (acc, value) in aggregate_w.iter_mut().zip(w.iter()) {
            *acc += alpha * *value;
        }
        aggregate_z += alpha * z;
        members.push(member);
    }

    let mut sponge = batched_opening_sponge(
        &group_label,
        n_c,
        require_zero_w_final,
        &members,
        public_hash,
        model_id,
    );
    let (proof, _g_final) = crate::hyrax::ipa_prove(
        &aggregate_w,
        &params.as_ref().gens,
        &params.as_ref().blind_base,
        &aggregate_z,
        n_c,
        &mut sponge,
    );

    BatchedOpeningArtifact {
        group_label,
        n_c,
        require_zero_w_final,
        members,
        proof,
    }
}

fn prove_batched_openings(
    matrices: Vec<&MatrixCommitment>,
    public_hash: &Fp,
    model_id: &Fp,
) -> Vec<BatchedOpeningArtifact> {
    let mut groups: BTreeMap<(usize, bool), Vec<&MatrixCommitment>> = BTreeMap::new();
    for matrix in matrices {
        let require_zero_w_final = matrix.label == "C_R1CS_residuals";
        groups
            .entry((matrix.n_c, require_zero_w_final))
            .or_default()
            .push(matrix);
    }
    groups
        .into_iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|((n_c, require_zero_w_final), matrices)| {
            prove_batched_opening_group(n_c, require_zero_w_final, matrices, public_hash, model_id)
        })
        .collect()
}

fn set_scalar_list(
    dict: &Bound<'_, PyDict>,
    py: Python<'_>,
    key: &str,
    values: &[Fp],
) -> PyResult<()> {
    let items: Vec<PyObject> = values.iter().map(|v| fp_hex(v).into_py(py)).collect();
    dict.set_item(key, PyList::new_bound(py, items))
}

fn set_opening_member<'py>(
    py: Python<'py>,
    member: &OpeningMemberArtifact,
) -> PyResult<Bound<'py, PyDict>> {
    let item = PyDict::new_bound(py);
    item.set_item("label", member.label)?;
    item.set_item("n_r", member.n_r)?;
    item.set_item("n_c", member.n_c)?;
    set_scalar_list(&item, py, "r_row", &member.r_row)?;
    Ok(item)
}

fn set_batched_opening_artifact(
    batches: &Bound<'_, PyList>,
    py: Python<'_>,
    artifact: &BatchedOpeningArtifact,
) -> PyResult<()> {
    let item = PyDict::new_bound(py);
    item.set_item("group_label", artifact.group_label.as_str())?;
    item.set_item("n_c", artifact.n_c)?;
    item.set_item("require_zero_w_final", artifact.require_zero_w_final)?;
    let members = PyList::empty_bound(py);
    for member in &artifact.members {
        members.append(set_opening_member(py, member)?)?;
    }
    item.set_item("members", members)?;
    item.set_item("w_final", fp_hex(&artifact.proof.w_final))?;
    item.set_item("a_blind", affine_hex(&artifact.proof.a_blind)?)?;
    item.set_item("s_w", fp_hex(&artifact.proof.s_w))?;
    item.set_item("s_z", fp_hex(&artifact.proof.s_z))?;
    let rounds = PyList::empty_bound(py);
    for round in &artifact.proof.rounds {
        let round_dict = PyDict::new_bound(py);
        round_dict.set_item("L", affine_hex(&round.l)?)?;
        round_dict.set_item("R", affine_hex(&round.r)?)?;
        round_dict.set_item("u", fp_hex(&round.u))?;
        rounds.append(round_dict)?;
    }
    item.set_item("rounds", rounds)?;
    batches.append(item)
}

fn absorb_batched_opening_artifact(
    sponge: &mut PoseidonSponge,
    artifact: &BatchedOpeningArtifact,
) -> PyResult<()> {
    absorb_str(sponge, &artifact.group_label);
    sponge.absorb_scalar(&Fp::from(artifact.n_c as u64));
    sponge.absorb_scalar(&Fp::from(if artifact.require_zero_w_final {
        1u64
    } else {
        0u64
    }));
    sponge.absorb_scalar(&Fp::from(artifact.members.len() as u64));
    for member in &artifact.members {
        absorb_str(sponge, member.label);
        sponge.absorb_scalar(&Fp::from(member.n_r as u64));
        sponge.absorb_scalar(&Fp::from(member.n_c as u64));
        sponge.absorb_scalars(&member.r_row);
    }
    sponge.absorb_scalar(&artifact.proof.w_final);
    let mut bytes = Vec::new();
    artifact
        .proof
        .a_blind
        .serialize_compressed(&mut bytes)
        .map_err(|err| {
            py_value_error(format!("batched opening point serialization failed: {err}"))
        })?;
    sponge.absorb_scalar(&Fp::from_le_bytes_mod_order(&bytes));
    sponge.absorb_scalar(&artifact.proof.s_w);
    sponge.absorb_scalar(&artifact.proof.s_z);
    for round in &artifact.proof.rounds {
        for point in [&round.l, &round.r] {
            let mut bytes = Vec::new();
            point.serialize_compressed(&mut bytes).map_err(|err| {
                py_value_error(format!("batched opening round serialization failed: {err}"))
            })?;
            sponge.absorb_scalar(&Fp::from_le_bytes_mod_order(&bytes));
        }
        sponge.absorb_scalar(&round.u);
    }
    Ok(())
}

fn batched_openings_digest(artifacts: &[BatchedOpeningArtifact]) -> PyResult<Fp> {
    let mut sponge = PoseidonSponge::new();
    absorb_str(&mut sponge, "Spartan-ZENO-GC-batched-openings-v1");
    for artifact in artifacts {
        absorb_batched_opening_artifact(&mut sponge, artifact)?;
    }
    sponge.finalize_absorption();
    Ok(sponge.squeeze())
}

fn sumcheck_instance_digest(
    commitments: &[Vec<String>],
    roots: &[Fp],
    public_hash: &Fp,
    model_id: &Fp,
) -> Fp {
    let mut sponge = PoseidonSponge::new();
    absorb_str(&mut sponge, "Spartan-ZENO-GC-sumcheck-instance-v1");
    sponge.absorb_scalar(public_hash);
    sponge.absorb_scalar(model_id);
    for commitment in commitments {
        sponge.absorb_scalar(&Fp::from(commitment.len() as u64));
        for row in commitment {
            absorb_str(&mut sponge, row);
        }
    }
    for root in roots {
        sponge.absorb_scalar(root);
    }
    sponge.finalize_absorption();
    sponge.squeeze()
}

fn range_residuals_i128(
    values: &[i128],
    bit_width: usize,
    max_value: i128,
    label: &str,
) -> PyResult<Vec<i128>> {
    for value in values {
        if *value < 0 || *value >= max_value {
            return Err(py_value_error(format!("{label} value outside range")));
        }
    }
    let _ = bit_width;
    Ok(vec![0])
}

fn sumcheck_challenge(
    label: &str,
    public_hash: &Fp,
    model_id: &Fp,
    instance_digest: &Fp,
    round_idx: usize,
    previous: &Fp,
    g0: &Fp,
    g1: &Fp,
) -> Fp {
    let mut sponge = PoseidonSponge::new();
    absorb_str(&mut sponge, "Spartan-ZENO-GC-linear-sumcheck-v2");
    absorb_str(&mut sponge, label);
    sponge.absorb_scalar(public_hash);
    sponge.absorb_scalar(model_id);
    sponge.absorb_scalar(instance_digest);
    sponge.absorb_scalar(&Fp::from(round_idx as u64));
    sponge.absorb_scalar(previous);
    sponge.absorb_scalar(g0);
    sponge.absorb_scalar(g1);
    sponge.finalize_absorption();
    let mut challenge = sponge.squeeze();
    if challenge.is_zero() {
        challenge = Fp::from(1u64);
    }
    challenge
}

fn prove_linear_sumcheck_padded(
    label: &str,
    mut current: Vec<Fp>,
    logical_len: usize,
    public_hash: &Fp,
    model_id: &Fp,
    instance_digest: &Fp,
) -> PyResult<LinearSumcheckProof> {
    let target_len = current.len().max(1);
    let num_vars = log2_ceil(target_len);
    let claim: Fp = if current.len() >= GC_SUMCHECK_PAR_THRESHOLD {
        current.par_iter().copied().sum()
    } else {
        current.iter().copied().sum()
    };
    let mut previous = claim;
    let mut rounds = Vec::with_capacity(num_vars);

    for round_idx in 0..num_vars {
        let half = current.len() / 2;
        let (left, right) = current.split_at(half);
        let (g0, g1) = if current.len() >= GC_SUMCHECK_PAR_THRESHOLD {
            rayon::join(
                || left.par_iter().copied().sum::<Fp>(),
                || right.par_iter().copied().sum::<Fp>(),
            )
        } else {
            (left.iter().copied().sum(), right.iter().copied().sum())
        };
        if g0 + g1 != previous {
            return Err(py_value_error(format!(
                "{label} Sum-check round sum mismatch"
            )));
        }
        let challenge = sumcheck_challenge(
            label,
            public_hash,
            model_id,
            instance_digest,
            round_idx,
            &previous,
            &g0,
            &g1,
        );
        let one_minus = Fp::from(1u64) - challenge;
        {
            let (left_mut, right_mut) = current.split_at_mut(half);
            if half >= GC_SUMCHECK_PAR_THRESHOLD {
                left_mut
                    .par_iter_mut()
                    .zip(right_mut.par_iter())
                    .for_each(|(dst, rhs)| {
                        *dst = one_minus * *dst + challenge * *rhs;
                    });
            } else {
                for idx in 0..half {
                    left_mut[idx] = one_minus * left_mut[idx] + challenge * right_mut[idx];
                }
            }
        }
        previous = g0 + (g1 - g0) * challenge;
        rounds.push(SumcheckRound { g0, g1, challenge });
        current.truncate(half);
    }

    let terminal = current.first().copied().unwrap_or_else(Fp::zero);
    if terminal != previous {
        return Err(py_value_error(format!(
            "{label} Sum-check terminal mismatch"
        )));
    }

    Ok(LinearSumcheckProof {
        label: label.to_string(),
        claim,
        logical_len,
        num_vars,
        rounds,
        terminal,
    })
}

fn build_constraint_residuals(
    trace: &Trace,
    mask: &[i128],
    n_real: usize,
    n_max: usize,
    scale: usize,
    arg_bits: usize,
) -> PyResult<ConstraintResiduals> {
    let mut mean_pool_residuals = Vec::with_capacity(trace.active_h_cols);
    for j in 0..trace.active_h_cols {
        let mut sum = 0i128;
        for i in 0..n_max {
            sum += mask[i] * trace.h[i * trace.h_cols + j];
        }
        let residual = sum - (n_real as i128) * trace.g[j] - trace.pool_remainders[j];
        if residual != 0 {
            return Err(py_value_error("mean-pool quotient residual is non-zero"));
        }
        mean_pool_residuals.push(residual);
    }

    let y_ell = trace
        .y_g
        .get(trace.ell)
        .ok_or_else(|| py_value_error("ell is outside Y_G"))?;
    let max_delta = if arg_bits >= 127 {
        i128::MAX
    } else {
        1i128 << arg_bits
    };
    for (j, y_j) in trace.y_g.iter().enumerate() {
        let delta = *y_ell - *y_j;
        if delta != trace.argmax_deltas[j] || delta < 0 || delta >= max_delta {
            return Err(py_value_error("Argmax delta/range residual is invalid"));
        }
    }
    let argmax_residuals: Vec<i128> = trace
        .y_g
        .iter()
        .enumerate()
        .map(|(j, y_j)| *y_ell - *y_j - trace.argmax_deltas[j])
        .collect();

    let pool_bit_width = log2_ceil(n_real.max(2));
    let trunc_bit_width = log2_ceil(scale.max(2));
    let mut range_residual_values = Vec::new();
    range_residual_values.extend(range_residuals_i128(
        &trace.pool_remainders,
        pool_bit_width,
        n_real as i128,
        "R_pool",
    )?);
    range_residual_values.extend(range_residuals_i128(
        &trace.trunc_remainders,
        trunc_bit_width,
        scale as i128,
        "R_trunc",
    )?);
    range_residual_values.extend(range_residuals_i128(
        &trace.argmax_deltas,
        arg_bits,
        max_delta,
        "ArgmaxDelta",
    )?);

    let mut r1cs_residuals = Vec::new();
    r1cs_residuals.extend(trace.semantic_residuals.iter().copied());
    r1cs_residuals.extend(mean_pool_residuals.iter().copied());
    r1cs_residuals.extend(argmax_residuals.iter().copied());
    r1cs_residuals.extend(range_residual_values.iter().copied());

    for (label, values) in [
        ("semantic", trace.semantic_residuals.as_slice()),
        ("mean_pool", mean_pool_residuals.as_slice()),
        ("argmax", argmax_residuals.as_slice()),
        ("range", range_residual_values.as_slice()),
        ("r1cs_sumcheck", r1cs_residuals.as_slice()),
    ] {
        if values.iter().any(|value| *value != 0) {
            return Err(py_value_error(format!("{label} residual is non-zero")));
        }
    }

    Ok(ConstraintResiduals {
        r1cs: r1cs_residuals,
        semantic: trace.semantic_residuals.clone(),
        mean_pool: mean_pool_residuals,
        argmax: argmax_residuals,
        range: range_residual_values,
    })
}

fn sumcheck_rlc_coefficient(
    label: &str,
    public_hash: &Fp,
    model_id: &Fp,
    instance_digest: &Fp,
) -> Fp {
    let mut sponge = PoseidonSponge::new();
    absorb_str(&mut sponge, "Spartan-ZENO-GC-rlc-sumcheck-coeff-v1");
    absorb_str(&mut sponge, label);
    sponge.absorb_scalar(public_hash);
    sponge.absorb_scalar(model_id);
    sponge.absorb_scalar(instance_digest);
    sponge.finalize_absorption();
    let coeff = sponge.squeeze();
    if coeff.is_zero() {
        Fp::from(1u64)
    } else {
        coeff
    }
}

fn fill_weighted_residuals(
    out: &mut [Fp],
    offset: usize,
    label: &str,
    values: &[i128],
    public_hash: &Fp,
    model_id: &Fp,
    instance_digest: &Fp,
) -> usize {
    let coeff = sumcheck_rlc_coefficient(label, public_hash, model_id, instance_digest);
    let target = &mut out[offset..offset + values.len()];
    if values.len() >= GC_SUMCHECK_PAR_THRESHOLD {
        target
            .par_iter_mut()
            .zip(values.par_iter())
            .for_each(|(slot, value)| {
                *slot = coeff * fp_from_i128(*value);
            });
    } else {
        for (slot, value) in target.iter_mut().zip(values.iter()) {
            *slot = coeff * fp_from_i128(*value);
        }
    }
    offset + values.len()
}

fn expected_rlc_sumcheck_len(
    semantic_len: usize,
    mean_pool_len: usize,
    argmax_len: usize,
    range_len: usize,
) -> usize {
    semantic_len
        .saturating_add(mean_pool_len)
        .saturating_add(argmax_len)
        .saturating_add(range_len)
}

fn build_sumcheck_proofs(
    residuals: &ConstraintResiduals,
    public_hash: &Fp,
    model_id: &Fp,
    instance_digest: &Fp,
) -> PyResult<Vec<LinearSumcheckProof>> {
    let rlc_len = expected_rlc_sumcheck_len(
        residuals.semantic.len(),
        residuals.mean_pool.len(),
        residuals.argmax.len(),
        residuals.range.len(),
    );
    let mut rlc_current = vec![Fp::zero(); next_power_of_two(rlc_len.max(1))];
    let mut offset = 0usize;
    offset = fill_weighted_residuals(
        &mut rlc_current,
        offset,
        "semantic",
        &residuals.semantic,
        public_hash,
        model_id,
        instance_digest,
    );
    offset = fill_weighted_residuals(
        &mut rlc_current,
        offset,
        "mean_pool",
        &residuals.mean_pool,
        public_hash,
        model_id,
        instance_digest,
    );
    offset = fill_weighted_residuals(
        &mut rlc_current,
        offset,
        "argmax",
        &residuals.argmax,
        public_hash,
        model_id,
        instance_digest,
    );
    offset = fill_weighted_residuals(
        &mut rlc_current,
        offset,
        "range",
        &residuals.range,
        public_hash,
        model_id,
        instance_digest,
    );
    if offset != rlc_len {
        return Err(py_value_error("RLC Sum-check residual length mismatch"));
    }
    let proof = prove_linear_sumcheck_padded(
        GC_RLC_SUMCHECK_LABEL,
        rlc_current,
        rlc_len,
        public_hash,
        model_id,
        instance_digest,
    )?;
    if !proof.claim.is_zero() || !proof.terminal.is_zero() {
        return Err(py_value_error("RLC Sum-check residual is non-zero"));
    }
    Ok(vec![proof])
}

fn sumcheck_proofs_digest(proofs: &[LinearSumcheckProof]) -> Fp {
    let mut sponge = PoseidonSponge::new();
    absorb_str(&mut sponge, "Spartan-ZENO-GC-sumcheck-proofs-v2");
    for proof in proofs {
        absorb_str(&mut sponge, &proof.label);
        sponge.absorb_scalar(&proof.claim);
        sponge.absorb_scalar(&Fp::from(proof.logical_len as u64));
        sponge.absorb_scalar(&Fp::from(proof.num_vars as u64));
        for round in &proof.rounds {
            sponge.absorb_scalar(&round.g0);
            sponge.absorb_scalar(&round.g1);
            sponge.absorb_scalar(&round.challenge);
        }
        sponge.absorb_scalar(&proof.terminal);
    }
    sponge.finalize_absorption();
    sponge.squeeze()
}

fn set_sumcheck_proof(
    proofs_dict: &Bound<'_, PyDict>,
    py: Python<'_>,
    proof: &LinearSumcheckProof,
) -> PyResult<()> {
    let item = PyDict::new_bound(py);
    item.set_item("label", proof.label.as_str())?;
    item.set_item("claim", fp_hex(&proof.claim))?;
    item.set_item("logical_len", proof.logical_len)?;
    item.set_item("num_vars", proof.num_vars)?;
    item.set_item("terminal", fp_hex(&proof.terminal))?;
    let rounds = PyList::empty_bound(py);
    for (round_idx, round) in proof.rounds.iter().enumerate() {
        let round_dict = PyDict::new_bound(py);
        round_dict.set_item("round", round_idx)?;
        round_dict.set_item("g0", fp_hex(&round.g0))?;
        round_dict.set_item("g1", fp_hex(&round.g1))?;
        round_dict.set_item("challenge", fp_hex(&round.challenge))?;
        rounds.append(round_dict)?;
    }
    item.set_item("rounds", rounds)?;
    proofs_dict.set_item(proof.label.as_str(), item)
}

fn set_sumcheck_proofs(
    claims: &Bound<'_, PyDict>,
    py: Python<'_>,
    proofs: &[LinearSumcheckProof],
) -> PyResult<()> {
    let proofs_dict = PyDict::new_bound(py);
    for proof in proofs {
        set_sumcheck_proof(&proofs_dict, py, proof)?;
    }
    claims.set_item("proofs", proofs_dict)
}

fn commitment_digest(
    commitments: &[Vec<String>],
    roots: &[Fp],
    opening_digest: &Fp,
    sumcheck_digest: &Fp,
    public_hash: &Fp,
    model_id: &Fp,
    ell: usize,
) -> Fp {
    let mut sponge = PoseidonSponge::new();
    absorb_str(&mut sponge, "Spartan-ZENO-GC-proof-v1");
    sponge.absorb_scalar(public_hash);
    sponge.absorb_scalar(model_id);
    sponge.absorb_scalar(&Fp::from(ell as u64));
    for commitment in commitments {
        for row in commitment {
            absorb_str(&mut sponge, row);
        }
    }
    for root in roots {
        sponge.absorb_scalar(root);
    }
    sponge.absorb_scalar(opening_digest);
    sponge.absorb_scalar(sumcheck_digest);
    sponge.finalize_absorption();
    sponge.squeeze()
}

fn gc_commitment_labels(arch: ModelArchitecture) -> Vec<&'static str> {
    let mut labels = vec![
        "C_X_base",
        "C_H",
        "C_g",
        "C_Y_G",
        "C_Delta",
        "C_W_graph",
        "C_R1CS_residuals",
    ];
    if arch == ModelArchitecture::APPNP {
        labels.push("C_W_pre0");
        labels.push("C_W_pre1");
    }
    labels
}

fn model_code(arch: ModelArchitecture) -> u8 {
    match arch {
        ModelArchitecture::SGC => 1,
        ModelArchitecture::SSGC => 2,
        ModelArchitecture::SIGN => 3,
        ModelArchitecture::APPNP => 4,
    }
}

fn gc_label_code(label: &str) -> PyResult<u8> {
    match label {
        "C_X_base" => Ok(1),
        "C_H" => Ok(2),
        "C_g" => Ok(3),
        "C_Y_G" => Ok(4),
        "C_Delta" => Ok(5),
        "C_W_graph" => Ok(6),
        "C_R1CS_residuals" => Ok(7),
        "C_W_pre0" => Ok(8),
        "C_W_pre1" => Ok(9),
        GC_RLC_SUMCHECK_LABEL => Ok(32),
        _ => Err(py_value_error(format!(
            "unknown graph-classification label: {label}"
        ))),
    }
}

fn gc_label_static(label: &str) -> PyResult<&'static str> {
    match label {
        "C_X_base" => Ok("C_X_base"),
        "C_H" => Ok("C_H"),
        "C_g" => Ok("C_g"),
        "C_Y_G" => Ok("C_Y_G"),
        "C_Delta" => Ok("C_Delta"),
        "C_W_graph" => Ok("C_W_graph"),
        "C_R1CS_residuals" => Ok("C_R1CS_residuals"),
        "C_W_pre0" => Ok("C_W_pre0"),
        "C_W_pre1" => Ok("C_W_pre1"),
        GC_RLC_SUMCHECK_LABEL => Ok(GC_RLC_SUMCHECK_LABEL),
        _ => Err(py_value_error(format!(
            "unknown graph-classification label: {label}"
        ))),
    }
}

fn append_u64(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&(value as u64).to_be_bytes());
}

fn append_u8(out: &mut Vec<u8>, value: u8) {
    out.push(value);
}

fn append_bool(out: &mut Vec<u8>, value: bool) {
    out.push(u8::from(value));
}

fn append_bytes(out: &mut Vec<u8>, value: &[u8]) {
    append_u64(out, value.len());
    out.extend_from_slice(value);
}

fn append_str(out: &mut Vec<u8>, value: &str) {
    append_bytes(out, value.as_bytes());
}

fn append_hex(out: &mut Vec<u8>, value: &str) -> PyResult<()> {
    let bytes =
        hex::decode(value).map_err(|err| py_value_error(format!("invalid proof hex: {err}")))?;
    if bytes.len() != 32 {
        return Err(py_value_error(
            "canonical proof hex values must be 32 bytes",
        ));
    }
    out.extend_from_slice(&bytes);
    Ok(())
}

fn append_dict_str(out: &mut Vec<u8>, dict: &Bound<'_, PyDict>, key: &str) -> PyResult<()> {
    append_str(out, &dict_string(dict, key)?);
    Ok(())
}

fn append_dict_hex(out: &mut Vec<u8>, dict: &Bound<'_, PyDict>, key: &str) -> PyResult<()> {
    append_hex(out, &dict_string(dict, key)?)
}

fn append_dict_usize(out: &mut Vec<u8>, dict: &Bound<'_, PyDict>, key: &str) -> PyResult<()> {
    append_u64(out, dict_usize(dict, key)?);
    Ok(())
}

fn append_dict_bool(out: &mut Vec<u8>, dict: &Bound<'_, PyDict>, key: &str) -> PyResult<()> {
    append_bool(out, dict_bool(dict, key)?);
    Ok(())
}

fn append_hex_list(out: &mut Vec<u8>, values: Vec<String>) -> PyResult<()> {
    append_u64(out, values.len());
    for value in values {
        append_hex(out, &value)?;
    }
    Ok(())
}

fn append_commitments(
    out: &mut Vec<u8>,
    commitments: &Bound<'_, PyDict>,
    arch: ModelArchitecture,
) -> PyResult<()> {
    let labels = gc_commitment_labels(arch);
    append_u64(out, labels.len());
    for label in labels {
        append_hex_list(out, dict_string_list(commitments, label)?)?;
    }
    Ok(())
}

fn append_opening(
    out: &mut Vec<u8>,
    openings: &Bound<'_, PyDict>,
    label: &'static str,
) -> PyResult<()> {
    let item = openings
        .get_item(label)?
        .ok_or_else(|| py_value_error(format!("missing opening for {label}")))?
        .downcast_into::<PyDict>()
        .map_err(|_| py_value_error(format!("opening for {label} must be a dict")))?;
    append_dict_usize(out, &item, "n_r")?;
    append_dict_usize(out, &item, "n_c")?;
    append_hex_list(out, dict_string_list(&item, "r_row")?)?;
    append_dict_hex(out, &item, "w_final")?;
    append_dict_hex(out, &item, "a_blind")?;
    append_dict_hex(out, &item, "s_w")?;
    append_dict_hex(out, &item, "s_z")?;
    let rounds = item
        .get_item("rounds")?
        .ok_or_else(|| py_value_error(format!("missing IPA rounds for {label}")))?
        .downcast_into::<PyList>()
        .map_err(|_| py_value_error(format!("IPA rounds for {label} must be a list")))?;
    append_u64(out, rounds.len());
    for round_obj in rounds.iter() {
        let round = round_obj
            .downcast_into::<PyDict>()
            .map_err(|_| py_value_error(format!("IPA round for {label} must be a dict")))?;
        append_dict_hex(out, &round, "L")?;
        append_dict_hex(out, &round, "R")?;
    }
    Ok(())
}

fn append_batched_opening(out: &mut Vec<u8>, batch: &Bound<'_, PyDict>) -> PyResult<()> {
    append_dict_usize(out, batch, "n_c")?;
    append_dict_bool(out, batch, "require_zero_w_final")?;
    let members = batch
        .get_item("members")?
        .ok_or_else(|| py_value_error("missing batched opening members"))?
        .downcast_into::<PyList>()
        .map_err(|_| py_value_error("batched opening members must be a list"))?;
    append_u64(out, members.len());
    for member_obj in members.iter() {
        let member = member_obj
            .downcast_into::<PyDict>()
            .map_err(|_| py_value_error("batched opening member must be a dict"))?;
        append_u8(out, gc_label_code(&dict_string(&member, "label")?)?);
        let _ = dict_usize(&member, "n_r")?;
    }
    append_dict_hex(out, batch, "w_final")?;
    append_dict_hex(out, batch, "a_blind")?;
    append_dict_hex(out, batch, "s_w")?;
    append_dict_hex(out, batch, "s_z")?;
    let rounds = batch
        .get_item("rounds")?
        .ok_or_else(|| py_value_error("missing batched IPA rounds"))?
        .downcast_into::<PyList>()
        .map_err(|_| py_value_error("batched IPA rounds must be a list"))?;
    append_u64(out, rounds.len());
    for round_obj in rounds.iter() {
        let round = round_obj
            .downcast_into::<PyDict>()
            .map_err(|_| py_value_error("batched IPA round must be a dict"))?;
        append_dict_hex(out, &round, "L")?;
        append_dict_hex(out, &round, "R")?;
    }
    Ok(())
}

fn append_openings(
    out: &mut Vec<u8>,
    openings: &Bound<'_, PyDict>,
    arch: ModelArchitecture,
) -> PyResult<()> {
    append_dict_hex(out, openings, "opening_digest")?;
    let mode =
        dict_string(openings, "opening_mode").unwrap_or_else(|_| "single_ipa_v1".to_string());
    if mode == GC_OPENING_MODE_BATCHED {
        let batches = openings
            .get_item("batches")?
            .ok_or_else(|| py_value_error("missing batched openings"))?
            .downcast_into::<PyList>()
            .map_err(|_| py_value_error("batched openings must be a list"))?;
        append_u64(out, batches.len());
        for batch_obj in batches.iter() {
            let batch = batch_obj
                .downcast_into::<PyDict>()
                .map_err(|_| py_value_error("batched opening must be a dict"))?;
            append_batched_opening(out, &batch)?;
        }
        return Ok(());
    }
    let labels = gc_commitment_labels(arch);
    append_u64(out, labels.len());
    for label in labels {
        append_opening(out, openings, label)?;
    }
    Ok(())
}

fn append_sumcheck_proof(
    out: &mut Vec<u8>,
    proofs: &Bound<'_, PyDict>,
    label: &'static str,
) -> PyResult<()> {
    let item = proofs
        .get_item(label)?
        .ok_or_else(|| py_value_error(format!("missing Sum-check proof for {label}")))?
        .downcast_into::<PyDict>()
        .map_err(|_| py_value_error(format!("Sum-check proof for {label} must be a dict")))?;
    if !fp_from_hex(&dict_string(&item, "claim")?)?.is_zero()
        || !fp_from_hex(&dict_string(&item, "terminal")?)?.is_zero()
    {
        return Err(py_value_error(
            "canonical RLC Sum-check endpoints must be zero",
        ));
    }
    let _ = dict_usize(&item, "logical_len")?;
    let _ = dict_usize(&item, "num_vars")?;
    let rounds = item
        .get_item("rounds")?
        .ok_or_else(|| py_value_error(format!("missing Sum-check rounds for {label}")))?
        .downcast_into::<PyList>()
        .map_err(|_| py_value_error(format!("Sum-check rounds for {label} must be a list")))?;
    append_u64(out, rounds.len());
    for round_obj in rounds.iter() {
        let round = round_obj
            .downcast_into::<PyDict>()
            .map_err(|_| py_value_error(format!("Sum-check round for {label} must be a dict")))?;
        append_dict_hex(out, &round, "g0")?;
        append_dict_hex(out, &round, "g1")?;
    }
    Ok(())
}

fn append_sumcheck_claims(out: &mut Vec<u8>, claims: &Bound<'_, PyDict>) -> PyResult<()> {
    append_dict_hex(out, claims, "claim_digest")?;
    append_dict_hex(out, claims, "constraint_commitment_digest")?;
    for key in [
        "r1cs_sumcheck_claim",
        "semantic_claim",
        "mean_pool_claim",
        "argmax_claim",
        "range_claim",
    ] {
        if !fp_from_hex(&dict_string(claims, key)?)?.is_zero() {
            return Err(py_value_error("canonical residual claims must be zero"));
        }
    }
    if dict_bool(claims, "softmax_claim_present")? {
        return Err(py_value_error(
            "canonical graph-classification proof must not include softmax claim",
        ));
    }
    if dict_string(claims, "constraint_system")? != GC_CONSTRAINT_SYSTEM {
        return Err(py_value_error(
            "unexpected graph-classification constraint system",
        ));
    }
    for key in [
        "r1cs_residual_count",
        "semantic_residual_count",
        "mean_pool_residual_count",
        "argmax_residual_count",
        "range_residual_count",
    ] {
        append_dict_usize(out, claims, key)?;
    }
    let proofs = claims
        .get_item("proofs")?
        .ok_or_else(|| py_value_error("missing Sum-check proofs"))?
        .downcast_into::<PyDict>()
        .map_err(|_| py_value_error("Sum-check proofs must be a dict"))?;
    append_sumcheck_proof(out, &proofs, GC_RLC_SUMCHECK_LABEL)?;
    Ok(())
}

fn graph_classification_canonical_proof_bytes(proof: &Bound<'_, PyDict>) -> PyResult<Vec<u8>> {
    let arch = ModelArchitecture::parse_architecture(&dict_string(proof, "model")?)
        .ok_or_else(|| py_value_error("unsupported proof model for serialization"))?;
    let public = nested_public(proof)?;
    let commitments = nested_dict(proof, "commitments")?;
    let openings = nested_dict(proof, "openings")?;
    let spark = nested_dict(proof, "spark")?;
    let claims = nested_dict(proof, "sumcheck_claims")?;
    let ranges = nested_dict(proof, "range_commitments")?;

    let mut out = Vec::new();
    out.extend_from_slice(GC_CANONICAL_BINARY_TYPE);
    if dict_string(proof, "proof_type")? != GC_PROOF_TYPE {
        return Err(py_value_error("unexpected graph-classification proof type"));
    }
    if dict_usize(proof, "proof_schema_version")? != GC_SCHEMA_VERSION {
        return Err(py_value_error(
            "unexpected graph-classification proof schema",
        ));
    }
    if dict_string(proof, "task_type")? != "graph_classification" {
        return Err(py_value_error("unexpected graph-classification task type"));
    }
    if dict_string(proof, "vk_name")? != format!("VK_GC_{}", arch.as_str()) {
        return Err(py_value_error("unexpected graph-classification VK name"));
    }
    append_u8(&mut out, model_code(arch));
    append_dict_hex(&mut out, proof, "model_id")?;
    append_dict_hex(&mut out, proof, "proof_digest")?;

    append_dict_str(&mut out, &public, "split_id")?;
    for key in [
        "graph_id", "ell", "N_real", "N_max", "d", "c", "h", "K", "D_max", "H_max", "scale",
        "arg_bits",
    ] {
        append_dict_usize(&mut out, &public, key)?;
    }
    append_dict_hex(&mut out, &public, "public_input_hash")?;
    append_dict_hex(&mut out, &public, "vk_id")?;
    append_dict_hex(&mut out, &public, "weight_fingerprint")?;
    if !dict_bool(&public, "mean_pool")?
        || dict_usize(&public, "mean_pool_denominator")? != dict_usize(&public, "N_real")?
        || !dict_bool(&public, "softmax_circuit_disabled")?
        || !dict_bool(&public, "argmax_proof")?
    {
        return Err(py_value_error(
            "unexpected graph-classification public verification flags",
        ));
    }
    if dict_string(&public, "range_residual_profile")? != GC_RANGE_PROFILE_COMPACT {
        return Err(py_value_error(
            "unexpected graph-classification range profile",
        ));
    }
    if dict_string(&public, "active_column_profile")? != GC_ACTIVE_COLUMN_PROFILE {
        return Err(py_value_error(
            "unexpected graph-classification active-column profile",
        ));
    }
    for key in [
        "x_active_cols",
        "h_active_cols",
        "w_graph_active_rows",
        "class_active_cols",
    ] {
        append_dict_usize(&mut out, &public, key)?;
    }
    let yg_shape = public
        .get_item("yg_shape")?
        .ok_or_else(|| py_value_error("missing Y_G shape"))?
        .extract::<Vec<usize>>()?;
    if yg_shape != vec![1usize, dict_usize(&public, "c")?] {
        return Err(py_value_error("unexpected graph-level logits shape"));
    }
    for key in [
        "sparse_nnz",
        "sparse_row_col_val_padded_len",
        "spark_active_cols",
        "spark_audit_padded_len",
    ] {
        append_dict_usize(&mut out, &public, key)?;
    }

    append_commitments(&mut out, &commitments, arch)?;
    append_openings(&mut out, &openings, arch)?;

    for key in ["init_root", "read_root", "write_root", "audit_root"] {
        append_dict_hex(&mut out, &spark, key)?;
    }
    append_dict_bool(&mut out, &spark, "row_col_val_audit")?;
    append_dict_bool(&mut out, &spark, "memory_conservation")?;

    append_sumcheck_claims(&mut out, &claims)?;
    for key in ["R_pool", "R_trunc", "ArgmaxDelta"] {
        append_dict_hex(&mut out, &ranges, key)?;
    }
    Ok(out)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn graph_classification_prove(
    py: Python<'_>,
    model: &str,
    split_id: &str,
    graph_id: usize,
    x: PyReadonlyArray2<'_, i64>,
    a_hat: PyReadonlyArray2<'_, i64>,
    mask: PyReadonlyArray1<'_, i64>,
    edge_rows: PyReadonlyArray1<'_, i64>,
    edge_cols: PyReadonlyArray1<'_, i64>,
    edge_vals: PyReadonlyArray1<'_, i64>,
    w_pre0: PyReadonlyArray2<'_, i64>,
    w_pre1: PyReadonlyArray2<'_, i64>,
    w_graph: PyReadonlyArray2<'_, i64>,
    n_real: usize,
    n_max: usize,
    d: usize,
    c: usize,
    h: usize,
    k: usize,
    d_max: usize,
    h_max: usize,
    scale: i64,
    arg_bits: usize,
) -> PyResult<PyObject> {
    let total_start = Instant::now();
    let public_input_start = Instant::now();
    if scale != GC_FIXED_SCALE {
        return Err(py_value_error(
            "graph classification uses fixed S_scale=256",
        ));
    }
    let arch = ModelArchitecture::parse_architecture(model)
        .ok_or_else(|| py_value_error(format!("unsupported graph model: {model}")))?;
    validate_dimensions(arch, n_real, n_max, d, c, h, k, d_max, h_max)?;
    let x_public = array2_i128(&x, n_max, d, "X")?;
    let a_public = array2_i128(&a_hat, n_max, n_max, "A_hat")?;
    let mask_public = array1_i128(&mask, n_max, "m")?;
    let rows = array1_usize(&edge_rows, "row")?;
    let cols = array1_usize(&edge_cols, "col")?;
    let vals = array1_i128(&edge_vals, rows.len(), "val")?;
    let w_pre0_values = array2_i128(&w_pre0, d, h, "W_pre0")?;
    let w_pre1_values = array2_i128(&w_pre1, h, d_max, "W_pre1")?;
    let w_graph_values = array2_i128(&w_graph, d_max, c, "W_graph")?;
    validate_i16_values(&x_public, "X")?;
    validate_i16_values(&a_public, "A_hat")?;
    validate_i16_values(&w_pre0_values, "W_pre0")?;
    validate_i16_values(&w_pre1_values, "W_pre1")?;
    validate_i16_values(&w_graph_values, "W_graph")?;
    validate_mask_prefix(&mask_public, n_real, n_max)?;
    validate_public_padding(&x_public, &a_public, n_real, n_max, d)?;
    let sparse_nnz = validate_sparse_encoding(&a_public, &rows, &cols, &vals, n_max)?;
    let public_input_validation_time_ms = elapsed_ms(&public_input_start);

    let witness_start = Instant::now();
    let x_active_cols = active_x_cols(d, d_max);
    let h_active_cols = active_graph_cols_from_w_graph(arch, d, k, d_max, &w_graph_values, c);
    let w_graph_active_rows = h_active_cols;
    let class_active_cols = c;
    let trace = compute_trace(
        arch,
        &x_public,
        &mask_public,
        &rows,
        &cols,
        &vals,
        &w_pre0_values,
        &w_pre1_values,
        &w_graph_values,
        n_real,
        n_max,
        d,
        c,
        h,
        k,
        d_max,
        h_active_cols,
        i128::from(scale),
        arg_bits,
    )?;
    let witness_generation_time_ms = elapsed_ms(&witness_start);

    let prove_start = Instant::now();
    let public_binding_start = Instant::now();
    let public_hash = cached_public_input_hash(
        arch,
        split_id,
        graph_id,
        &x_public,
        &a_public,
        &mask_public,
        &rows,
        &cols,
        &vals,
        n_real,
        n_max,
        d,
        c,
        h,
        k,
        d_max,
        h_max,
    );
    let vk_profile = cached_gc_vk_profile(arch, n_max, d, c, h, k, d_max, h_max);
    let vk_id = vk_profile.vk_id;
    let valid_rows: Vec<usize> = (0..n_real).collect();
    let public_binding_time_ms = elapsed_ms(&public_binding_start);

    let residual_start = Instant::now();
    let residuals = build_constraint_residuals(
        &trace,
        &mask_public,
        n_real,
        n_max,
        scale as usize,
        arg_bits,
    )?;
    ensure_zero_inactive_columns("C_X_base", &trace.x_base, n_max, d_max, x_active_cols)?;
    ensure_zero_inactive_columns("C_H", &trace.h, n_max, trace.h_cols, h_active_cols)?;
    ensure_zero_inactive_columns("C_g", &trace.g, 1, trace.h_cols, h_active_cols)?;
    ensure_zero_inactive_w_rows(&w_graph_values, d_max, c, w_graph_active_rows)?;
    let x_commit_values = project_columns(&trace.x_base, n_max, d_max, x_active_cols);
    let h_commit_values = project_columns(&trace.h, n_max, trace.h_cols, h_active_cols);
    let g_commit_values = project_columns(&trace.g, 1, trace.h_cols, h_active_cols);
    let w_graph_commit_values =
        project_rows_cols(&w_graph_values, c, w_graph_active_rows, class_active_cols);
    let residual_constraint_time_ms = elapsed_ms(&residual_start);

    let commit_start = Instant::now();
    let witness_commitment_start = Instant::now();
    let c_x = commit_i128_matrix(
        "C_X_base",
        &x_commit_values,
        n_max,
        x_active_cols,
        Some(&valid_rows),
    )?;
    let c_h = commit_i128_matrix(
        "C_H",
        &h_commit_values,
        n_max,
        h_active_cols,
        Some(&valid_rows),
    )?;
    let c_g = commit_i128_matrix("C_g", &g_commit_values, 1, h_active_cols, None)?;
    let c_y = commit_i128_matrix("C_Y_G", &trace.y_g, 1, class_active_cols, None)?;
    let c_delta = commit_i128_matrix("C_Delta", &trace.argmax_deltas, 1, class_active_cols, None)?;
    let c_w = commit_i128_matrix(
        "C_W_graph",
        &w_graph_commit_values,
        w_graph_active_rows,
        class_active_cols,
        None,
    )?;
    let c_r1cs = commit_i128_matrix(
        "C_R1CS_residuals",
        &residuals.r1cs,
        1,
        residuals.r1cs.len().max(1),
        None,
    )?;
    let mut c_pre0: Option<MatrixCommitment> = None;
    let mut c_pre1: Option<MatrixCommitment> = None;
    if arch == ModelArchitecture::APPNP {
        c_pre0 = Some(commit_i128_matrix("C_W_pre0", &w_pre0_values, d, h, None)?);
        let w_pre1_commit_values = project_strided_columns(&w_pre1_values, h, d_max, h_active_cols);
        c_pre1 = Some(commit_i128_matrix(
            "C_W_pre1",
            &w_pre1_commit_values,
            h,
            h_active_cols,
            None,
        )?);
    }
    let witness_commitment_time_ms = elapsed_ms(&witness_commitment_start);

    let commitment_binding_start = Instant::now();
    let weight_fingerprint = weight_commitment_fingerprint(
        arch,
        &vk_id,
        &c_w.commitment_hex,
        c_pre0.as_ref().map(|m| m.commitment_hex.as_slice()),
        c_pre1.as_ref().map(|m| m.commitment_hex.as_slice()),
    );
    let model_id = bind_model_id(&vk_id, &weight_fingerprint);
    let mut all_commitments = vec![
        c_x.commitment_hex.clone(),
        c_h.commitment_hex.clone(),
        c_g.commitment_hex.clone(),
        c_y.commitment_hex.clone(),
        c_delta.commitment_hex.clone(),
        c_w.commitment_hex.clone(),
        c_r1cs.commitment_hex.clone(),
    ];
    if let (Some(pre0), Some(pre1)) = (c_pre0.as_ref(), c_pre1.as_ref()) {
        all_commitments.push(pre0.commitment_hex.clone());
        all_commitments.push(pre1.commitment_hex.clone());
    }
    let commitment_binding_time_ms = elapsed_ms(&commitment_binding_start);

    let spark_start = Instant::now();
    let spark = build_spark_roots(&x_commit_values, x_active_cols, &rows, &cols, &vals, n_real);
    let spark_root_time_ms = elapsed_ms(&spark_start);

    let range_commitment_start = Instant::now();
    let pool_range_digest =
        range_digest(&trace.pool_remainders, log2_ceil(n_real.max(2)), "R_pool")?;
    let trunc_range_digest = range_digest(
        &trace.trunc_remainders,
        log2_ceil(scale as usize),
        "R_trunc",
    )?;
    let argmax_range_digest = range_digest(&trace.argmax_deltas, arg_bits, "ArgmaxDelta")?;
    let range_commitment_time_ms = elapsed_ms(&range_commitment_start);
    let commitment_generation_time_ms = elapsed_ms(&commit_start);

    let opening_start = Instant::now();
    let mut opening_inputs = vec![&c_x, &c_h, &c_g, &c_y, &c_delta, &c_w, &c_r1cs];
    if let (Some(pre0), Some(pre1)) = (c_pre0.as_ref(), c_pre1.as_ref()) {
        opening_inputs.push(pre0);
        opening_inputs.push(pre1);
    }
    let opening_artifacts = prove_batched_openings(opening_inputs, &public_hash, &model_id);
    let opening_generation_time_ms = elapsed_ms(&opening_start);

    let sumcheck_start = Instant::now();
    let roots = [
        spark.init_root,
        spark.read_root,
        spark.write_root,
        spark.audit_root,
        pool_range_digest,
        trunc_range_digest,
        argmax_range_digest,
    ];
    let opening_digest = batched_openings_digest(&opening_artifacts)?;
    let sumcheck_instance =
        sumcheck_instance_digest(&all_commitments, &roots, &public_hash, &model_id);
    let sumcheck_proofs =
        build_sumcheck_proofs(&residuals, &public_hash, &model_id, &sumcheck_instance)?;
    let sumcheck_digest = sumcheck_proofs_digest(&sumcheck_proofs);
    let sumcheck_time_ms = elapsed_ms(&sumcheck_start);

    let proof_digest_start = Instant::now();
    let proof_digest = commitment_digest(
        &all_commitments,
        &roots,
        &opening_digest,
        &sumcheck_digest,
        &public_hash,
        &model_id,
        trace.ell,
    );
    let proof_digest_time_ms = elapsed_ms(&proof_digest_start);
    let total_prover_time_including_commit_ms = elapsed_ms(&prove_start);
    let prove_time_excluding_commit_ms =
        (total_prover_time_including_commit_ms - commitment_generation_time_ms).max(0.0);

    let proof_assembly_start = Instant::now();
    let proof = PyDict::new_bound(py);
    proof.set_item("proof_type", GC_PROOF_TYPE)?;
    proof.set_item("proof_schema_version", GC_SCHEMA_VERSION)?;
    proof.set_item("task_type", "graph_classification")?;
    proof.set_item("vk_name", format!("VK_GC_{}", arch.as_str()))?;
    proof.set_item("model", arch.as_str())?;
    proof.set_item("model_id", fp_hex(&model_id))?;
    proof.set_item("proof_digest", fp_hex(&proof_digest))?;

    let public = PyDict::new_bound(py);
    public.set_item("split_id", split_id)?;
    public.set_item("graph_id", graph_id)?;
    public.set_item("ell", trace.ell)?;
    public.set_item("N_real", n_real)?;
    public.set_item("N_max", n_max)?;
    public.set_item("d", d)?;
    public.set_item("c", c)?;
    public.set_item("h", h)?;
    public.set_item("K", k)?;
    public.set_item("D_max", d_max)?;
    public.set_item("H_max", h_max)?;
    public.set_item("scale", scale)?;
    public.set_item("arg_bits", arg_bits)?;
    public.set_item("public_input_hash", fp_hex(&public_hash))?;
    public.set_item("vk_id", fp_hex(&vk_id))?;
    public.set_item("weight_fingerprint", fp_hex(&weight_fingerprint))?;
    public.set_item("mean_pool", true)?;
    public.set_item("mean_pool_denominator", n_real)?;
    public.set_item("softmax_circuit_disabled", true)?;
    public.set_item("argmax_proof", true)?;
    public.set_item("range_residual_profile", GC_RANGE_PROFILE_COMPACT)?;
    public.set_item("active_column_profile", GC_ACTIVE_COLUMN_PROFILE)?;
    public.set_item("x_active_cols", x_active_cols)?;
    public.set_item("h_active_cols", h_active_cols)?;
    public.set_item("w_graph_active_rows", w_graph_active_rows)?;
    public.set_item("class_active_cols", class_active_cols)?;
    public.set_item("yg_shape", vec![1usize, c])?;
    public.set_item("sparse_nnz", sparse_nnz)?;
    public.set_item("sparse_row_col_val_padded_len", rows.len())?;
    public.set_item("spark_active_cols", x_active_cols)?;
    public.set_item("spark_audit_padded_len", spark.padded_len)?;
    proof.set_item("public", public)?;

    let commitments = PyDict::new_bound(py);
    set_commitment(&commitments, py, "C_X_base", c_x.commitment_hex)?;
    set_commitment(&commitments, py, "C_H", c_h.commitment_hex)?;
    set_commitment(&commitments, py, "C_g", c_g.commitment_hex)?;
    set_commitment(&commitments, py, "C_Y_G", c_y.commitment_hex)?;
    set_commitment(&commitments, py, "C_Delta", c_delta.commitment_hex)?;
    set_commitment(&commitments, py, "C_W_graph", c_w.commitment_hex)?;
    set_commitment(&commitments, py, "C_R1CS_residuals", c_r1cs.commitment_hex)?;
    if let Some(pre0) = c_pre0 {
        set_commitment(&commitments, py, "C_W_pre0", pre0.commitment_hex)?;
    }
    if let Some(pre1) = c_pre1 {
        set_commitment(&commitments, py, "C_W_pre1", pre1.commitment_hex)?;
    }
    proof.set_item("commitments", commitments)?;

    let openings = PyDict::new_bound(py);
    openings.set_item("opening_digest", fp_hex(&opening_digest))?;
    openings.set_item("opening_mode", GC_OPENING_MODE_BATCHED)?;
    let opening_batches = PyList::empty_bound(py);
    for artifact in &opening_artifacts {
        set_batched_opening_artifact(&opening_batches, py, artifact)?;
    }
    openings.set_item("batches", opening_batches)?;
    proof.set_item("openings", openings)?;

    let spark_dict = PyDict::new_bound(py);
    spark_dict.set_item("init_root", fp_hex(&spark.init_root))?;
    spark_dict.set_item("read_root", fp_hex(&spark.read_root))?;
    spark_dict.set_item("write_root", fp_hex(&spark.write_root))?;
    spark_dict.set_item("audit_root", fp_hex(&spark.audit_root))?;
    spark_dict.set_item("row_col_val_audit", true)?;
    spark_dict.set_item("memory_conservation", spark.conservation)?;
    proof.set_item("spark", spark_dict)?;

    let claims = PyDict::new_bound(py);
    claims.set_item("claim_digest", fp_hex(&sumcheck_digest))?;
    claims.set_item("constraint_commitment_digest", fp_hex(&sumcheck_instance))?;
    claims.set_item("r1cs_sumcheck_claim", fp_hex(&Fp::zero()))?;
    claims.set_item("semantic_claim", fp_hex(&Fp::zero()))?;
    claims.set_item("mean_pool_claim", fp_hex(&Fp::zero()))?;
    claims.set_item("argmax_claim", fp_hex(&Fp::zero()))?;
    claims.set_item("range_claim", fp_hex(&Fp::zero()))?;
    claims.set_item("softmax_claim_present", false)?;
    claims.set_item("constraint_system", GC_CONSTRAINT_SYSTEM)?;
    claims.set_item("r1cs_residual_count", residuals.r1cs.len())?;
    claims.set_item("semantic_residual_count", residuals.semantic.len())?;
    claims.set_item("mean_pool_residual_count", residuals.mean_pool.len())?;
    claims.set_item("argmax_residual_count", residuals.argmax.len())?;
    claims.set_item("range_residual_count", residuals.range.len())?;
    set_sumcheck_proofs(&claims, py, &sumcheck_proofs)?;
    proof.set_item("sumcheck_claims", claims)?;

    let range_commitments = PyDict::new_bound(py);
    range_commitments.set_item("R_pool", fp_hex(&pool_range_digest))?;
    range_commitments.set_item("R_trunc", fp_hex(&trunc_range_digest))?;
    range_commitments.set_item("ArgmaxDelta", fp_hex(&argmax_range_digest))?;
    proof.set_item("range_commitments", range_commitments)?;

    let timing = PyDict::new_bound(py);
    let proof_assembly_time_ms = elapsed_ms(&proof_assembly_start);
    timing.set_item(
        "public_input_validation_time_ms",
        public_input_validation_time_ms,
    )?;
    timing.set_item("witness_generation_time_ms", witness_generation_time_ms)?;
    timing.set_item("public_binding_time_ms", public_binding_time_ms)?;
    timing.set_item("residual_constraint_time_ms", residual_constraint_time_ms)?;
    timing.set_item("witness_commitment_time_ms", witness_commitment_time_ms)?;
    timing.set_item("commitment_binding_time_ms", commitment_binding_time_ms)?;
    timing.set_item("spark_root_time_ms", spark_root_time_ms)?;
    timing.set_item("range_commitment_time_ms", range_commitment_time_ms)?;
    timing.set_item("opening_generation_time_ms", opening_generation_time_ms)?;
    timing.set_item("sumcheck_time_ms", sumcheck_time_ms)?;
    timing.set_item("proof_digest_time_ms", proof_digest_time_ms)?;
    timing.set_item("proof_assembly_time_ms", proof_assembly_time_ms)?;
    timing.set_item("commit_time_ms", commitment_generation_time_ms)?;
    timing.set_item(
        "commitment_generation_time_ms",
        commitment_generation_time_ms,
    )?;
    timing.set_item("prove_time_ms", prove_time_excluding_commit_ms)?;
    timing.set_item(
        "prove_time_excluding_commit_ms",
        prove_time_excluding_commit_ms,
    )?;
    timing.set_item(
        "total_prover_time_including_commit_ms",
        total_prover_time_including_commit_ms,
    )?;
    timing.set_item("total_backend_time_ms", elapsed_ms(&total_start))?;
    proof.set_item("timing", timing)?;
    Ok(proof.into_py(py))
}

fn dict_string(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    dict.get_item(key)?
        .ok_or_else(|| py_value_error(format!("missing proof key: {key}")))?
        .extract::<String>()
}

fn nested_public<'py>(proof: &'py Bound<'py, PyDict>) -> PyResult<Bound<'py, PyDict>> {
    proof
        .get_item("public")?
        .ok_or_else(|| py_value_error("missing public proof block"))?
        .downcast_into::<PyDict>()
        .map_err(|_| py_value_error("public proof block must be a dict"))
}

fn nested_dict<'py>(proof: &'py Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyDict>> {
    proof
        .get_item(key)?
        .ok_or_else(|| py_value_error(format!("missing {key} proof block")))?
        .downcast_into::<PyDict>()
        .map_err(|_| py_value_error(format!("{key} proof block must be a dict")))
}

fn dict_usize(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<usize> {
    dict.get_item(key)?
        .ok_or_else(|| py_value_error(format!("missing proof key: {key}")))?
        .extract::<usize>()
}

fn dict_bool(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<bool> {
    dict.get_item(key)?
        .ok_or_else(|| py_value_error(format!("missing proof key: {key}")))?
        .extract::<bool>()
}

fn dict_string_list(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<String>> {
    dict.get_item(key)?
        .ok_or_else(|| py_value_error(format!("missing proof key: {key}")))?
        .extract::<Vec<String>>()
}

fn dict_scalar_list(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<Fp>> {
    dict_string_list(dict, key)?
        .iter()
        .map(|value| fp_from_hex(value))
        .collect()
}

fn commitment_points(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<(Vec<String>, Vec<GAffine>)> {
    let hex_rows = dict_string_list(dict, key)?;
    let points = hex_rows
        .iter()
        .map(|value| affine_from_hex(value))
        .collect::<PyResult<Vec<_>>>()?;
    Ok((hex_rows, points))
}

fn parse_ipa_proof(item: &Bound<'_, PyDict>) -> PyResult<IpaProof> {
    let rounds_obj = item
        .get_item("rounds")?
        .ok_or_else(|| py_value_error("missing IPA rounds"))?
        .downcast_into::<PyList>()
        .map_err(|_| py_value_error("IPA rounds must be a list"))?;
    let mut rounds = Vec::with_capacity(rounds_obj.len());
    for obj in rounds_obj.iter() {
        let round = obj
            .downcast_into::<PyDict>()
            .map_err(|_| py_value_error("IPA round must be a dict"))?;
        rounds.push(IpaRoundData {
            l: affine_from_hex(&dict_string(&round, "L")?)?,
            r: affine_from_hex(&dict_string(&round, "R")?)?,
            u: fp_from_hex(&dict_string(&round, "u")?)?,
        });
    }
    Ok(IpaProof {
        rounds,
        w_final: fp_from_hex(&dict_string(item, "w_final")?)?,
        a_blind: affine_from_hex(&dict_string(item, "a_blind")?)?,
        s_w: fp_from_hex(&dict_string(item, "s_w")?)?,
        s_z: fp_from_hex(&dict_string(item, "s_z")?)?,
    })
}

fn parse_batched_opening_artifacts(
    openings: &Bound<'_, PyDict>,
) -> PyResult<Vec<BatchedOpeningArtifact>> {
    let mode = dict_string(openings, "opening_mode")?;
    if mode != GC_OPENING_MODE_BATCHED {
        return Err(py_value_error(
            "unsupported graph-classification opening mode",
        ));
    }
    let batches_obj = openings
        .get_item("batches")?
        .ok_or_else(|| py_value_error("missing batched opening groups"))?
        .downcast_into::<PyList>()
        .map_err(|_| py_value_error("batched opening groups must be a list"))?;
    let mut parsed = Vec::with_capacity(batches_obj.len());
    for batch_obj in batches_obj.iter() {
        let batch = batch_obj
            .downcast_into::<PyDict>()
            .map_err(|_| py_value_error("batched opening group must be a dict"))?;
        let members_obj = batch
            .get_item("members")?
            .ok_or_else(|| py_value_error("missing batched opening members"))?
            .downcast_into::<PyList>()
            .map_err(|_| py_value_error("batched opening members must be a list"))?;
        let mut members = Vec::with_capacity(members_obj.len());
        for member_obj in members_obj.iter() {
            let member = member_obj
                .downcast_into::<PyDict>()
                .map_err(|_| py_value_error("batched opening member must be a dict"))?;
            let label = gc_label_static(&dict_string(&member, "label")?)?;
            members.push(OpeningMemberArtifact {
                label,
                n_r: dict_usize(&member, "n_r")?,
                n_c: dict_usize(&member, "n_c")?,
                r_row: dict_scalar_list(&member, "r_row")?,
            });
        }
        parsed.push(BatchedOpeningArtifact {
            group_label: dict_string(&batch, "group_label")?,
            n_c: dict_usize(&batch, "n_c")?,
            require_zero_w_final: dict_bool(&batch, "require_zero_w_final")?,
            members,
            proof: parse_ipa_proof(&batch)?,
        });
    }
    Ok(parsed)
}

fn verify_parsed_batched_opening_jobs(
    parsed: &[BatchedOpeningArtifact],
    jobs: Vec<OpeningVerifyJob>,
    public_hash: &Fp,
    model_id: &Fp,
) -> PyResult<Vec<BatchedOpeningArtifact>> {
    let mut job_map: HashMap<String, OpeningVerifyJob> = jobs
        .into_iter()
        .map(|job| (job.label.to_string(), job))
        .collect();
    let mut verified = Vec::with_capacity(parsed.len());

    for artifact in parsed {
        let group_n_c = artifact.n_c;
        let require_zero_w_final = artifact.require_zero_w_final;
        let expected_group_label = format!(
            "batch_nc{}_{}",
            group_n_c,
            if require_zero_w_final {
                "residual"
            } else {
                "witness"
            }
        );
        if artifact.group_label != expected_group_label {
            return Err(py_value_error("batched opening group label mismatch"));
        }
        if artifact.members.is_empty() {
            return Err(py_value_error("empty batched opening group"));
        }
        if require_zero_w_final && artifact.members.len() != 1 {
            return Err(py_value_error(
                "residual opening batch must contain exactly one member",
            ));
        }

        let mut c_aggregate = G::zero();
        let mut members = Vec::with_capacity(artifact.members.len());
        for (member_idx, member) in artifact.members.iter().enumerate() {
            let job = job_map.remove(member.label).ok_or_else(|| {
                py_value_error(format!(
                    "unexpected batched opening member: {}",
                    member.label
                ))
            })?;
            if job.require_zero_w_final != require_zero_w_final {
                return Err(py_value_error("batched opening residual flag mismatch"));
            }
            if require_zero_w_final && job.label != "C_R1CS_residuals" {
                return Err(py_value_error(
                    "only C_R1CS_residuals may use residual opening batch",
                ));
            }
            let rows = next_power_of_two(job.expected_rows.max(1));
            let cols = next_power_of_two(job.expected_cols.max(1));
            let expected_n_r = log2_ceil(rows);
            let expected_n_c = log2_ceil(cols);
            if job.commitments.len() != rows
                || member.n_r != expected_n_r
                || member.n_c != expected_n_c
                || member.n_c != group_n_c
            {
                return Err(py_value_error("batched opening dimensions mismatch"));
            }
            let expected_r_row = opening_r_row_cached(
                job.label,
                public_hash,
                model_id,
                &job.commitment_hex,
                &job.commitments,
                expected_n_r,
            );
            if member.r_row != expected_r_row {
                return Err(py_value_error("batched opening challenge mismatch"));
            }
            let alpha = opening_batch_alpha(
                group_n_c,
                require_zero_w_final,
                member_idx,
                job.label,
                public_hash,
                model_id,
                &job.commitment_hex,
            );
            let c_initial =
                reconstruct_expected_commitment(&job.commitments, &member.r_row, expected_n_r);
            c_aggregate += c_initial * alpha;
            members.push(member.clone());
        }

        let params = cached_hyrax_params(group_n_c, HYRAX_GC_SEED);
        let mut sponge = batched_opening_sponge(
            &artifact.group_label,
            group_n_c,
            require_zero_w_final,
            &members,
            public_hash,
            model_id,
        );
        if !ipa_verify(
            &artifact.proof,
            &c_aggregate.into_affine(),
            &params.as_ref().gens,
            &params.as_ref().blind_base,
            group_n_c,
            &mut sponge,
        ) {
            return Err(py_value_error("batched Hyrax/IPA opening failed"));
        }
        if require_zero_w_final && !artifact.proof.w_final.is_zero() {
            return Err(py_value_error(
                "C_R1CS_residuals residual opening is non-zero",
            ));
        }
        verified.push(BatchedOpeningArtifact {
            group_label: artifact.group_label.clone(),
            n_c: artifact.n_c,
            require_zero_w_final: artifact.require_zero_w_final,
            members,
            proof: artifact.proof.clone(),
        });
    }

    if !job_map.is_empty() {
        return Err(py_value_error("missing batched opening member"));
    }
    Ok(verified)
}

fn verify_batched_opening_jobs(
    openings: &Bound<'_, PyDict>,
    jobs: Vec<OpeningVerifyJob>,
    public_hash: &Fp,
    model_id: &Fp,
) -> PyResult<Vec<BatchedOpeningArtifact>> {
    let mode = dict_string(openings, "opening_mode")?;
    if mode != GC_OPENING_MODE_BATCHED {
        return Err(py_value_error(
            "unsupported graph-classification opening mode",
        ));
    }
    let batches_obj = openings
        .get_item("batches")?
        .ok_or_else(|| py_value_error("missing batched opening groups"))?
        .downcast_into::<PyList>()
        .map_err(|_| py_value_error("batched opening groups must be a list"))?;
    let mut job_map: HashMap<String, OpeningVerifyJob> = jobs
        .into_iter()
        .map(|job| (job.label.to_string(), job))
        .collect();
    let mut parsed = Vec::with_capacity(batches_obj.len());

    for batch_obj in batches_obj.iter() {
        let batch = batch_obj
            .downcast_into::<PyDict>()
            .map_err(|_| py_value_error("batched opening group must be a dict"))?;
        let group_label = dict_string(&batch, "group_label")?;
        let group_n_c = dict_usize(&batch, "n_c")?;
        let require_zero_w_final = dict_bool(&batch, "require_zero_w_final")?;
        let expected_group_label = format!(
            "batch_nc{}_{}",
            group_n_c,
            if require_zero_w_final {
                "residual"
            } else {
                "witness"
            }
        );
        if group_label != expected_group_label {
            return Err(py_value_error("batched opening group label mismatch"));
        }
        let members_obj = batch
            .get_item("members")?
            .ok_or_else(|| py_value_error("missing batched opening members"))?
            .downcast_into::<PyList>()
            .map_err(|_| py_value_error("batched opening members must be a list"))?;
        if members_obj.is_empty() {
            return Err(py_value_error("empty batched opening group"));
        }
        if require_zero_w_final && members_obj.len() != 1 {
            return Err(py_value_error(
                "residual opening batch must contain exactly one member",
            ));
        }

        let proof = parse_ipa_proof(&batch)?;
        let mut members = Vec::with_capacity(members_obj.len());
        let mut c_aggregate = G::zero();

        for (member_idx, member_obj) in members_obj.iter().enumerate() {
            let member = member_obj
                .downcast_into::<PyDict>()
                .map_err(|_| py_value_error("batched opening member must be a dict"))?;
            let label_string = dict_string(&member, "label")?;
            let job = job_map.remove(&label_string).ok_or_else(|| {
                py_value_error(format!("unexpected batched opening member: {label_string}"))
            })?;
            if job.require_zero_w_final != require_zero_w_final {
                return Err(py_value_error("batched opening residual flag mismatch"));
            }
            if require_zero_w_final && job.label != "C_R1CS_residuals" {
                return Err(py_value_error(
                    "only C_R1CS_residuals may use residual opening batch",
                ));
            }

            let rows = next_power_of_two(job.expected_rows.max(1));
            let cols = next_power_of_two(job.expected_cols.max(1));
            let expected_n_r = log2_ceil(rows);
            let expected_n_c = log2_ceil(cols);
            let member_n_r = dict_usize(&member, "n_r")?;
            let member_n_c = dict_usize(&member, "n_c")?;
            if job.commitments.len() != rows
                || member_n_r != expected_n_r
                || member_n_c != expected_n_c
                || member_n_c != group_n_c
            {
                return Err(py_value_error("batched opening dimensions mismatch"));
            }
            let r_row = dict_scalar_list(&member, "r_row")?;
            let expected_r_row = opening_r_row_cached(
                job.label,
                public_hash,
                model_id,
                &job.commitment_hex,
                &job.commitments,
                expected_n_r,
            );
            if r_row != expected_r_row {
                return Err(py_value_error("batched opening challenge mismatch"));
            }
            let alpha = opening_batch_alpha(
                group_n_c,
                require_zero_w_final,
                member_idx,
                job.label,
                public_hash,
                model_id,
                &job.commitment_hex,
            );
            let c_initial = reconstruct_expected_commitment(&job.commitments, &r_row, expected_n_r);
            c_aggregate += c_initial * alpha;
            members.push(OpeningMemberArtifact {
                label: job.label,
                n_r: expected_n_r,
                n_c: expected_n_c,
                r_row,
            });
        }

        let params = cached_hyrax_params(group_n_c, HYRAX_GC_SEED);
        let mut sponge = batched_opening_sponge(
            &group_label,
            group_n_c,
            require_zero_w_final,
            &members,
            public_hash,
            model_id,
        );
        if !ipa_verify(
            &proof,
            &c_aggregate.into_affine(),
            &params.as_ref().gens,
            &params.as_ref().blind_base,
            group_n_c,
            &mut sponge,
        ) {
            return Err(py_value_error("batched Hyrax/IPA opening failed"));
        }
        if require_zero_w_final && !proof.w_final.is_zero() {
            return Err(py_value_error(
                "C_R1CS_residuals residual opening is non-zero",
            ));
        }
        parsed.push(BatchedOpeningArtifact {
            group_label,
            n_c: group_n_c,
            require_zero_w_final,
            members,
            proof,
        });
    }

    if !job_map.is_empty() {
        return Err(py_value_error("missing batched opening member"));
    }
    Ok(parsed)
}

fn parse_linear_sumcheck_proof(
    proofs: &Bound<'_, PyDict>,
    label: &str,
) -> PyResult<LinearSumcheckProof> {
    let item = proofs
        .get_item(label)?
        .ok_or_else(|| py_value_error(format!("missing Sum-check proof for {label}")))?
        .downcast_into::<PyDict>()
        .map_err(|_| py_value_error(format!("Sum-check proof for {label} must be a dict")))?;
    if dict_string(&item, "label")? != label {
        return Err(py_value_error(format!(
            "Sum-check label mismatch for {label}"
        )));
    }
    let rounds_obj = item
        .get_item("rounds")?
        .ok_or_else(|| py_value_error(format!("missing Sum-check rounds for {label}")))?
        .downcast_into::<PyList>()
        .map_err(|_| py_value_error(format!("Sum-check rounds for {label} must be a list")))?;
    let mut rounds = Vec::with_capacity(rounds_obj.len());
    for (idx, obj) in rounds_obj.iter().enumerate() {
        let round = obj.downcast_into::<PyDict>().map_err(|_| {
            py_value_error(format!("Sum-check round {idx} for {label} must be a dict"))
        })?;
        let round_idx = dict_usize(&round, "round")?;
        if round_idx != idx {
            return Err(py_value_error(format!(
                "Sum-check round index mismatch for {label}"
            )));
        }
        rounds.push(SumcheckRound {
            g0: fp_from_hex(&dict_string(&round, "g0")?)?,
            g1: fp_from_hex(&dict_string(&round, "g1")?)?,
            challenge: fp_from_hex(&dict_string(&round, "challenge")?)?,
        });
    }
    Ok(LinearSumcheckProof {
        label: label.to_string(),
        claim: fp_from_hex(&dict_string(&item, "claim")?)?,
        logical_len: dict_usize(&item, "logical_len")?,
        num_vars: dict_usize(&item, "num_vars")?,
        rounds,
        terminal: fp_from_hex(&dict_string(&item, "terminal")?)?,
    })
}

fn parse_graph_classification_proof_core(proof: &Bound<'_, PyDict>) -> PyResult<ParsedGcProofCore> {
    let arch = ModelArchitecture::parse_architecture(&dict_string(proof, "model")?)
        .ok_or_else(|| py_value_error("unsupported proof model for parsed cache"))?;
    if dict_string(proof, "proof_type")? != GC_PROOF_TYPE {
        return Err(py_value_error("wrong proof type"));
    }
    if dict_usize(proof, "proof_schema_version")? != GC_SCHEMA_VERSION {
        return Err(py_value_error("wrong proof schema version"));
    }
    if dict_string(proof, "task_type")? != "graph_classification" {
        return Err(py_value_error("wrong task type"));
    }
    let public = nested_public(proof)?;
    let commitments_dict = nested_dict(proof, "commitments")?;
    let openings = nested_dict(proof, "openings")?;
    let spark = nested_dict(proof, "spark")?;
    let range_commitments = nested_dict(proof, "range_commitments")?;
    let claims = nested_dict(proof, "sumcheck_claims")?;
    if !dict_bool(&public, "mean_pool")?
        || !dict_bool(&public, "softmax_circuit_disabled")?
        || !dict_bool(&public, "argmax_proof")?
        || dict_string(&public, "range_residual_profile")? != GC_RANGE_PROFILE_COMPACT
        || dict_string(&public, "active_column_profile")? != GC_ACTIVE_COLUMN_PROFILE
    {
        return Err(py_value_error("invalid graph-classification public flags"));
    }
    if dict_usize(&public, "mean_pool_denominator")? != dict_usize(&public, "N_real")? {
        return Err(py_value_error("MeanPool denominator must equal N_real"));
    }
    if dict_string(&claims, "constraint_system")? != GC_CONSTRAINT_SYSTEM
        || dict_bool(&claims, "softmax_claim_present")?
    {
        return Err(py_value_error("invalid graph-classification claim flags"));
    }
    for key in [
        "r1cs_sumcheck_claim",
        "semantic_claim",
        "mean_pool_claim",
        "argmax_claim",
        "range_claim",
    ] {
        if !fp_from_hex(&dict_string(&claims, key)?)?.is_zero() {
            return Err(py_value_error(
                "non-zero graph-classification residual claim",
            ));
        }
    }

    let mut commitments = HashMap::new();
    for label in gc_commitment_labels(arch) {
        let (hex_rows, points) = commitment_points(&commitments_dict, label)?;
        commitments.insert(
            label,
            ParsedCommitment {
                hex: hex_rows,
                points,
            },
        );
    }

    let proofs_dict = claims
        .get_item("proofs")?
        .ok_or_else(|| py_value_error("missing Sum-check proof object"))?
        .downcast_into::<PyDict>()
        .map_err(|_| py_value_error("Sum-check proofs must be a dict"))?;
    let rlc_proof = parse_linear_sumcheck_proof(&proofs_dict, GC_RLC_SUMCHECK_LABEL)?;

    Ok(ParsedGcProofCore {
        arch,
        split_id: dict_string(&public, "split_id")?,
        graph_id: dict_usize(&public, "graph_id")?,
        ell: dict_usize(&public, "ell")?,
        n_real: dict_usize(&public, "N_real")?,
        n_max: dict_usize(&public, "N_max")?,
        d: dict_usize(&public, "d")?,
        c: dict_usize(&public, "c")?,
        h: dict_usize(&public, "h")?,
        k: dict_usize(&public, "K")?,
        d_max: dict_usize(&public, "D_max")?,
        h_max: dict_usize(&public, "H_max")?,
        scale: dict_usize(&public, "scale")?,
        arg_bits: dict_usize(&public, "arg_bits")?,
        public_hash: fp_from_hex(&dict_string(&public, "public_input_hash")?)?,
        public_hash_hex: dict_string(&public, "public_input_hash")?,
        vk_id: fp_from_hex(&dict_string(&public, "vk_id")?)?,
        weight_fingerprint: fp_from_hex(&dict_string(&public, "weight_fingerprint")?)?,
        x_active_cols: dict_usize(&public, "x_active_cols")?,
        h_active_cols: dict_usize(&public, "h_active_cols")?,
        w_graph_active_rows: dict_usize(&public, "w_graph_active_rows")?,
        class_active_cols: dict_usize(&public, "class_active_cols")?,
        sparse_nnz: dict_usize(&public, "sparse_nnz")?,
        sparse_row_col_val_padded_len: dict_usize(&public, "sparse_row_col_val_padded_len")?,
        spark_active_cols: dict_usize(&public, "spark_active_cols")?,
        spark_audit_padded_len: dict_usize(&public, "spark_audit_padded_len")?,
        model_id: fp_from_hex(&dict_string(proof, "model_id")?)?,
        proof_digest: fp_from_hex(&dict_string(proof, "proof_digest")?)?,
        commitments,
        opening_digest: fp_from_hex(&dict_string(&openings, "opening_digest")?)?,
        opening_artifacts: parse_batched_opening_artifacts(&openings)?,
        spark_roots: [
            fp_from_hex(&dict_string(&spark, "init_root")?)?,
            fp_from_hex(&dict_string(&spark, "read_root")?)?,
            fp_from_hex(&dict_string(&spark, "write_root")?)?,
            fp_from_hex(&dict_string(&spark, "audit_root")?)?,
        ],
        spark_row_col_val_audit: dict_bool(&spark, "row_col_val_audit")?,
        spark_memory_conservation: dict_bool(&spark, "memory_conservation")?,
        range_roots: [
            fp_from_hex(&dict_string(&range_commitments, "R_pool")?)?,
            fp_from_hex(&dict_string(&range_commitments, "R_trunc")?)?,
            fp_from_hex(&dict_string(&range_commitments, "ArgmaxDelta")?)?,
        ],
        claim_digest: fp_from_hex(&dict_string(&claims, "claim_digest")?)?,
        constraint_commitment_digest: fp_from_hex(&dict_string(
            &claims,
            "constraint_commitment_digest",
        )?)?,
        r1cs_residual_count: dict_usize(&claims, "r1cs_residual_count")?,
        semantic_residual_count: dict_usize(&claims, "semantic_residual_count")?,
        mean_pool_residual_count: dict_usize(&claims, "mean_pool_residual_count")?,
        argmax_residual_count: dict_usize(&claims, "argmax_residual_count")?,
        range_residual_count: dict_usize(&claims, "range_residual_count")?,
        sumcheck_proofs: vec![rlc_proof],
    })
}

fn verify_linear_sumcheck_proof(
    proof: &LinearSumcheckProof,
    expected_label: &str,
    expected_logical_len: usize,
    public_hash: &Fp,
    model_id: &Fp,
    instance_digest: &Fp,
) -> bool {
    let target_len = next_power_of_two(expected_logical_len.max(1));
    let expected_vars = log2_ceil(target_len);
    if proof.label != expected_label
        || proof.logical_len != expected_logical_len
        || proof.num_vars != expected_vars
        || proof.rounds.len() != expected_vars
        || !proof.claim.is_zero()
    {
        return false;
    }

    let mut previous = proof.claim;
    for (round_idx, round) in proof.rounds.iter().enumerate() {
        if round.g0 + round.g1 != previous {
            return false;
        }
        let expected_challenge = sumcheck_challenge(
            expected_label,
            public_hash,
            model_id,
            instance_digest,
            round_idx,
            &previous,
            &round.g0,
            &round.g1,
        );
        if round.challenge != expected_challenge {
            return false;
        }
        previous = round.g0 + (round.g1 - round.g0) * round.challenge;
    }

    proof.terminal == previous && proof.terminal.is_zero()
}

fn parsed_commitment<'a>(
    core: &'a ParsedGcProofCore,
    label: &'static str,
) -> PyResult<&'a ParsedCommitment> {
    core.commitments
        .get(label)
        .ok_or_else(|| py_value_error(format!("missing parsed commitment: {label}")))
}

fn expected_semantic_residual_len(
    arch: ModelArchitecture,
    d: usize,
    n_max: usize,
    d_max: usize,
    h: usize,
    k: usize,
    c: usize,
    h_active_cols: usize,
) -> usize {
    let x_base = n_max.saturating_mul(active_x_cols(d, d_max));
    let active_base = active_x_cols(d, d_max);
    let active_graph = h_active_cols;
    let diffuse = k.saturating_mul(n_max).saturating_mul(active_base);
    let graph_logits = c;
    x_base
        + match arch {
            ModelArchitecture::SGC => diffuse + graph_logits,
            ModelArchitecture::SSGC => diffuse + n_max.saturating_mul(active_base) + graph_logits,
            ModelArchitecture::SIGN => diffuse + n_max.saturating_mul(active_graph) + graph_logits,
            ModelArchitecture::APPNP => {
                let diffuse = k.saturating_mul(n_max).saturating_mul(active_graph);
                n_max.saturating_mul(h)
                    + n_max.saturating_mul(h)
                    + n_max.saturating_mul(active_graph)
                    + 2usize.saturating_mul(diffuse)
                    + graph_logits
            }
        }
}

fn expected_range_residual_len(
    arch: ModelArchitecture,
    d: usize,
    n_real: usize,
    n_max: usize,
    d_max: usize,
    h: usize,
    k: usize,
    c: usize,
    scale: usize,
    arg_bits: usize,
) -> usize {
    let _pool_bits = log2_ceil(n_real.max(2));
    let _trunc_bits = log2_ceil(scale.max(2));
    let _arg_bits = arg_bits;
    let _ = (arch, d, n_real, n_max, d_max, h, k, c, scale, arg_bits);
    3
}

fn x_base_from_public(x_public: &[i128], n_max: usize, d: usize, d_max: usize) -> Vec<i128> {
    let mut x_base = vec![0i128; n_max * d_max];
    for i in 0..n_max {
        for j in 0..d {
            x_base[i * d_max + j] = x_public[i * d + j];
        }
    }
    x_base
}

fn reject(py: Python<'_>, reason: &str, ell: Option<usize>) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    out.set_item("accepted", false)?;
    out.set_item("reason", reason)?;
    if let Some(label) = ell {
        out.set_item("ell", label)?;
    }
    Ok(out.into_py(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn graph_classification_verify(
    py: Python<'_>,
    proof: &Bound<'_, PyDict>,
    model: &str,
    split_id: &str,
    graph_id: usize,
    x: PyReadonlyArray2<'_, i64>,
    a_hat: PyReadonlyArray2<'_, i64>,
    mask: PyReadonlyArray1<'_, i64>,
    edge_rows: PyReadonlyArray1<'_, i64>,
    edge_cols: PyReadonlyArray1<'_, i64>,
    edge_vals: PyReadonlyArray1<'_, i64>,
    n_real: usize,
    n_max: usize,
    d: usize,
    c: usize,
    h: usize,
    k: usize,
    d_max: usize,
    h_max: usize,
) -> PyResult<PyObject> {
    let verify_total_start = Instant::now();
    let verify_public_input_start = Instant::now();
    let arch = ModelArchitecture::parse_architecture(model)
        .ok_or_else(|| py_value_error(format!("unsupported graph model: {model}")))?;
    if let Err(err) = validate_dimensions(arch, n_real, n_max, d, c, h, k, d_max, h_max) {
        return reject(py, &err.to_string(), None);
    }
    let public = nested_public(proof)?;
    let ell = public
        .get_item("ell")?
        .ok_or_else(|| py_value_error("missing public ell"))?
        .extract::<usize>()?;

    for forbidden in ["Y_G", "logits", "Delta", "delta_bits", "argmax_bits"] {
        if proof.get_item(forbidden)?.is_some() {
            return reject(
                py,
                "proof exposes private graph logits or Argmax witness",
                Some(ell),
            );
        }
    }
    if dict_string(proof, "proof_type")? != GC_PROOF_TYPE {
        return reject(py, "wrong proof type", Some(ell));
    }
    match dict_usize(proof, "proof_schema_version") {
        Ok(value) if value == GC_SCHEMA_VERSION => {}
        Ok(_) => return reject(py, "wrong proof schema version", Some(ell)),
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    }
    if dict_string(proof, "task_type")? != "graph_classification" {
        return reject(py, "wrong task type", Some(ell));
    }
    if dict_string(proof, "model")? != arch.as_str() {
        return reject(py, "proof model metadata mismatch", Some(ell));
    }
    if dict_string(proof, "vk_name")? != format!("VK_GC_{}", arch.as_str()) {
        return reject(py, "VK_GC track mismatch", Some(ell));
    }
    if ell >= c {
        return reject(py, "public ell outside class range", Some(ell));
    }

    let x_public = array2_i128(&x, n_max, d, "X")?;
    let a_public = array2_i128(&a_hat, n_max, n_max, "A_hat")?;
    let mask_public = array1_i128(&mask, n_max, "m")?;
    let rows = array1_usize(&edge_rows, "row")?;
    let cols = array1_usize(&edge_cols, "col")?;
    let vals = array1_i128(&edge_vals, rows.len(), "val")?;
    if let Err(err) = validate_i16_values(&x_public, "X") {
        return reject(py, &err.to_string(), Some(ell));
    }
    if let Err(err) = validate_i16_values(&a_public, "A_hat") {
        return reject(py, &err.to_string(), Some(ell));
    }
    if let Err(err) = validate_mask_prefix(&mask_public, n_real, n_max) {
        return reject(py, &err.to_string(), Some(ell));
    }
    if let Err(err) = validate_public_padding(&x_public, &a_public, n_real, n_max, d) {
        return reject(py, &err.to_string(), Some(ell));
    }
    let sparse_nnz = match validate_sparse_encoding(&a_public, &rows, &cols, &vals, n_max) {
        Ok(value) => value,
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };

    match public.get_item("split_id")? {
        Some(value) if value.extract::<String>()? == split_id => {}
        Some(_) => return reject(py, "public split_id metadata mismatch", Some(ell)),
        None => return reject(py, "missing public split_id metadata", Some(ell)),
    }
    for (key, expected) in [
        ("graph_id", graph_id),
        ("N_real", n_real),
        ("N_max", n_max),
        ("d", d),
        ("c", c),
        ("h", h),
        ("K", k),
        ("D_max", d_max),
        ("H_max", h_max),
        ("sparse_nnz", sparse_nnz),
        ("sparse_row_col_val_padded_len", rows.len()),
    ] {
        match dict_usize(&public, key) {
            Ok(value) if value == expected => {}
            Ok(_) => return reject(py, "public dimension metadata mismatch", Some(ell)),
            Err(err) => return reject(py, &err.to_string(), Some(ell)),
        }
    }
    match public.get_item("yg_shape")? {
        Some(value) if value.extract::<Vec<usize>>()? == vec![1usize, c] => {}
        Some(_) => return reject(py, "Y_G shape metadata mismatch", Some(ell)),
        None => return reject(py, "missing Y_G shape metadata", Some(ell)),
    }

    let expected_hash = fp_hex(&cached_public_input_hash(
        arch,
        split_id,
        graph_id,
        &x_public,
        &a_public,
        &mask_public,
        &rows,
        &cols,
        &vals,
        n_real,
        n_max,
        d,
        c,
        h,
        k,
        d_max,
        h_max,
    ));
    let vk_profile = cached_gc_vk_profile(arch, n_max, d, c, h, k, d_max, h_max);
    let expected_vk_id_fp = vk_profile.vk_id;
    let expected_vk_id = fp_hex(&expected_vk_id_fp);
    let x_active_cols = active_x_cols(d, d_max);
    let h_active_cols = match dict_usize(&public, "h_active_cols") {
        Ok(value) if validate_public_h_active_cols(arch, d, k, d_max, value) => value,
        Ok(_) => return reject(py, "invalid active graph column metadata", Some(ell)),
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };
    let w_graph_active_rows = h_active_cols;
    let class_active_cols = c;
    let public_hash = public
        .get_item("public_input_hash")?
        .ok_or_else(|| py_value_error("missing public_input_hash"))?
        .extract::<String>()?;
    if public_hash != expected_hash {
        return reject(py, "public input hash mismatch", Some(ell));
    }
    match public.get_item("vk_id")? {
        Some(value) if value.extract::<String>()? == expected_vk_id => {}
        Some(_) => return reject(py, "VK_GC id mismatch", Some(ell)),
        None => return reject(py, "missing VK_GC id", Some(ell)),
    }
    let mean_pool = public
        .get_item("mean_pool")?
        .ok_or_else(|| py_value_error("missing mean_pool flag"))?
        .extract::<bool>()?;
    let softmax_disabled = public
        .get_item("softmax_circuit_disabled")?
        .ok_or_else(|| py_value_error("missing softmax flag"))?
        .extract::<bool>()?;
    let argmax_proof = public
        .get_item("argmax_proof")?
        .ok_or_else(|| py_value_error("missing argmax flag"))?
        .extract::<bool>()?;
    let denominator = public
        .get_item("mean_pool_denominator")?
        .ok_or_else(|| py_value_error("missing mean pool denominator"))?
        .extract::<usize>()?;
    let proof_scale = public
        .get_item("scale")?
        .ok_or_else(|| py_value_error("missing fixed-point scale"))?
        .extract::<usize>()?;
    let proof_arg_bits = public
        .get_item("arg_bits")?
        .ok_or_else(|| py_value_error("missing Argmax bit width"))?
        .extract::<usize>()?;
    let range_residual_profile = public
        .get_item("range_residual_profile")?
        .ok_or_else(|| py_value_error("missing range residual profile"))?
        .extract::<String>()?;
    let active_column_profile = public
        .get_item("active_column_profile")?
        .ok_or_else(|| py_value_error("missing active column profile"))?
        .extract::<String>()?;
    if !mean_pool
        || !softmax_disabled
        || !argmax_proof
        || denominator != n_real
        || proof_scale != GC_FIXED_SCALE as usize
        || proof_arg_bits == 0
        || range_residual_profile != GC_RANGE_PROFILE_COMPACT
        || active_column_profile != GC_ACTIVE_COLUMN_PROFILE
    {
        return reject(
            py,
            "graph-classification public flags are invalid",
            Some(ell),
        );
    }
    for (key, expected) in [
        ("x_active_cols", x_active_cols),
        ("h_active_cols", h_active_cols),
        ("w_graph_active_rows", w_graph_active_rows),
        ("class_active_cols", class_active_cols),
        ("spark_active_cols", x_active_cols),
    ] {
        match dict_usize(&public, key) {
            Ok(value) if value == expected => {}
            Ok(_) => return reject(py, "active-column metadata mismatch", Some(ell)),
            Err(err) => return reject(py, &err.to_string(), Some(ell)),
        }
    }

    let verify_public_input_time_ms = elapsed_ms(&verify_public_input_start);
    let verify_commitment_parse_start = Instant::now();
    let public_hash_fp = fp_from_hex(&expected_hash)?;
    let commitments = nested_dict(proof, "commitments")?;
    let openings = nested_dict(proof, "openings")?;
    let spark_dict = nested_dict(proof, "spark")?;
    let range_commitments = nested_dict(proof, "range_commitments")?;
    let claims = nested_dict(proof, "sumcheck_claims")?;

    let (c_x_hex, c_x_points) = match commitment_points(&commitments, "C_X_base") {
        Ok(value) => value,
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };
    let (c_h_hex, c_h_points) = match commitment_points(&commitments, "C_H") {
        Ok(value) => value,
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };
    let (c_g_hex, c_g_points) = match commitment_points(&commitments, "C_g") {
        Ok(value) => value,
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };
    let (c_y_hex, c_y_points) = match commitment_points(&commitments, "C_Y_G") {
        Ok(value) => value,
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };
    let (c_delta_hex, c_delta_points) = match commitment_points(&commitments, "C_Delta") {
        Ok(value) => value,
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };
    let (c_w_hex, c_w_points) = match commitment_points(&commitments, "C_W_graph") {
        Ok(value) => value,
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };
    let range_len = expected_range_residual_len(
        arch,
        d,
        n_real,
        n_max,
        d_max,
        h,
        k,
        c,
        proof_scale,
        proof_arg_bits,
    );
    let semantic_len =
        expected_semantic_residual_len(arch, d, n_max, d_max, h, k, c, h_active_cols);
    let r1cs_len = semantic_len
        .saturating_add(h_active_cols)
        .saturating_add(c)
        .saturating_add(range_len);
    let (c_r1cs_hex, c_r1cs_points) = match commitment_points(&commitments, "C_R1CS_residuals") {
        Ok(value) => value,
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };
    let mut all_commitments = vec![
        c_x_hex.clone(),
        c_h_hex.clone(),
        c_g_hex.clone(),
        c_y_hex.clone(),
        c_delta_hex.clone(),
        c_w_hex.clone(),
        c_r1cs_hex.clone(),
    ];

    let mut pre0_for_fingerprint: Option<Vec<String>> = None;
    let mut pre1_for_fingerprint: Option<Vec<String>> = None;
    if arch == ModelArchitecture::APPNP {
        match commitment_points(&commitments, "C_W_pre0") {
            Ok((hex, _)) => pre0_for_fingerprint = Some(hex),
            Err(err) => return reject(py, &err.to_string(), Some(ell)),
        }
        match commitment_points(&commitments, "C_W_pre1") {
            Ok((hex, _)) => pre1_for_fingerprint = Some(hex),
            Err(err) => return reject(py, &err.to_string(), Some(ell)),
        }
    } else if commitments.get_item("C_W_pre0")?.is_some()
        || commitments.get_item("C_W_pre1")?.is_some()
    {
        return reject(
            py,
            "inactive pre-transform weight commitments are present",
            Some(ell),
        );
    }
    let verify_commitment_parse_time_ms = elapsed_ms(&verify_commitment_parse_start);
    let verify_weight_binding_start = Instant::now();
    let expected_weight_fingerprint_fp = weight_commitment_fingerprint(
        arch,
        &expected_vk_id_fp,
        &all_commitments[5],
        pre0_for_fingerprint.as_deref(),
        pre1_for_fingerprint.as_deref(),
    );
    let expected_weight_fingerprint = fp_hex(&expected_weight_fingerprint_fp);
    match public.get_item("weight_fingerprint")? {
        Some(value) if value.extract::<String>()? == expected_weight_fingerprint => {}
        Some(_) => return reject(py, "weight fingerprint mismatch", Some(ell)),
        None => return reject(py, "missing weight fingerprint", Some(ell)),
    }
    let model_id_fp = bind_model_id(&expected_vk_id_fp, &expected_weight_fingerprint_fp);
    let expected_model_id = fp_hex(&model_id_fp);
    if dict_string(proof, "model_id")? != expected_model_id {
        return reject(py, "model_id mismatch", Some(ell));
    }
    let verify_weight_binding_time_ms = elapsed_ms(&verify_weight_binding_start);

    let opening_parse_start = Instant::now();
    let mut opening_jobs = Vec::new();
    for (label, hex_rows, points, exp_rows, exp_cols, require_zero_w_final) in [
        (
            "C_X_base",
            &c_x_hex,
            &c_x_points,
            n_max,
            x_active_cols,
            false,
        ),
        ("C_H", &c_h_hex, &c_h_points, n_max, h_active_cols, false),
        ("C_g", &c_g_hex, &c_g_points, 1usize, h_active_cols, false),
        (
            "C_Y_G",
            &c_y_hex,
            &c_y_points,
            1usize,
            class_active_cols,
            false,
        ),
        (
            "C_Delta",
            &c_delta_hex,
            &c_delta_points,
            1usize,
            class_active_cols,
            false,
        ),
        (
            "C_W_graph",
            &c_w_hex,
            &c_w_points,
            w_graph_active_rows,
            class_active_cols,
            false,
        ),
        (
            "C_R1CS_residuals",
            &c_r1cs_hex,
            &c_r1cs_points,
            1usize,
            r1cs_len.max(1),
            true,
        ),
    ] {
        opening_jobs.push(OpeningVerifyJob {
            label,
            commitment_hex: hex_rows.clone(),
            commitments: points.clone(),
            expected_rows: exp_rows,
            expected_cols: exp_cols,
            require_zero_w_final,
        });
    }
    if arch == ModelArchitecture::APPNP {
        let (pre0_hex, pre0_points) = match commitment_points(&commitments, "C_W_pre0") {
            Ok(value) => value,
            Err(err) => return reject(py, &err.to_string(), Some(ell)),
        };
        let (pre1_hex, pre1_points) = match commitment_points(&commitments, "C_W_pre1") {
            Ok(value) => value,
            Err(err) => return reject(py, &err.to_string(), Some(ell)),
        };
        all_commitments.push(pre0_hex.clone());
        all_commitments.push(pre1_hex.clone());
        for (label, hex_rows, points, exp_rows, exp_cols) in [
            ("C_W_pre0", &pre0_hex, &pre0_points, d, h),
            ("C_W_pre1", &pre1_hex, &pre1_points, h, h_active_cols),
        ] {
            opening_jobs.push(OpeningVerifyJob {
                label,
                commitment_hex: hex_rows.clone(),
                commitments: points.clone(),
                expected_rows: exp_rows,
                expected_cols: exp_cols,
                require_zero_w_final: false,
            });
        }
    }
    let opening_parse_time_ms = elapsed_ms(&opening_parse_start);
    let opening_verify_start = Instant::now();
    let parsed_openings =
        match verify_batched_opening_jobs(&openings, opening_jobs, &public_hash_fp, &model_id_fp) {
            Ok(value) => value,
            Err(err) => return reject(py, &err.to_string(), Some(ell)),
        };
    let expected_opening_digest = match batched_openings_digest(&parsed_openings) {
        Ok(value) => value,
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };
    match dict_string(&openings, "opening_digest").and_then(|v| fp_from_hex(&v)) {
        Ok(value) if value == expected_opening_digest => {}
        Ok(_) => return reject(py, "opening digest mismatch", Some(ell)),
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    }
    let opening_verify_time_ms = elapsed_ms(&opening_verify_start);

    let spark_verify_start = Instant::now();
    let x_base = x_base_from_public(&x_public, n_max, d, d_max);
    let x_spark_values = project_columns(&x_base, n_max, d_max, x_active_cols);
    let expected_spark =
        build_spark_roots(&x_spark_values, x_active_cols, &rows, &cols, &vals, n_real);
    let spark_roots = [
        ("init_root", expected_spark.init_root),
        ("read_root", expected_spark.read_root),
        ("write_root", expected_spark.write_root),
        ("audit_root", expected_spark.audit_root),
    ];
    for (key, expected) in spark_roots {
        match dict_string(&spark_dict, key).and_then(|v| fp_from_hex(&v)) {
            Ok(value) if value == expected => {}
            Ok(_) => return reject(py, "SPARK root mismatch", Some(ell)),
            Err(err) => return reject(py, &err.to_string(), Some(ell)),
        }
    }
    match dict_usize(&public, "spark_audit_padded_len") {
        Ok(value) if value == expected_spark.padded_len => {}
        Ok(_) => return reject(py, "SPARK audit padding metadata mismatch", Some(ell)),
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    }
    if !dict_bool(&spark_dict, "row_col_val_audit").unwrap_or(false)
        || !dict_bool(&spark_dict, "memory_conservation").unwrap_or(false)
        || !expected_spark.conservation
    {
        return reject(py, "SPARK memory conservation failed", Some(ell));
    }
    let spark_verify_time_ms = elapsed_ms(&spark_verify_start);

    let sumcheck_context_start = Instant::now();
    let pool_range = match dict_string(&range_commitments, "R_pool").and_then(|v| fp_from_hex(&v)) {
        Ok(value) => value,
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };
    let trunc_range = match dict_string(&range_commitments, "R_trunc").and_then(|v| fp_from_hex(&v))
    {
        Ok(value) => value,
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };
    let argmax_range =
        match dict_string(&range_commitments, "ArgmaxDelta").and_then(|v| fp_from_hex(&v)) {
            Ok(value) => value,
            Err(err) => return reject(py, &err.to_string(), Some(ell)),
        };
    let roots = [
        expected_spark.init_root,
        expected_spark.read_root,
        expected_spark.write_root,
        expected_spark.audit_root,
        pool_range,
        trunc_range,
        argmax_range,
    ];
    let expected_sumcheck_instance =
        sumcheck_instance_digest(&all_commitments, &roots, &public_hash_fp, &model_id_fp);
    match dict_string(&claims, "constraint_commitment_digest").and_then(|v| fp_from_hex(&v)) {
        Ok(value) if value == expected_sumcheck_instance => {}
        Ok(_) => return reject(py, "constraint commitment digest mismatch", Some(ell)),
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    }
    if dict_string(&claims, "constraint_system")? != GC_CONSTRAINT_SYSTEM
        || dict_bool(&claims, "softmax_claim_present").unwrap_or(true)
    {
        return reject(
            py,
            "invalid graph-classification sumcheck claims",
            Some(ell),
        );
    }
    for key in [
        "r1cs_sumcheck_claim",
        "semantic_claim",
        "mean_pool_claim",
        "argmax_claim",
        "range_claim",
    ] {
        match dict_string(&claims, key).and_then(|v| fp_from_hex(&v)) {
            Ok(value) if value.is_zero() => {}
            Ok(_) => return reject(py, "non-zero residual claim", Some(ell)),
            Err(err) => return reject(py, &err.to_string(), Some(ell)),
        }
    }
    for (key, expected) in [
        ("r1cs_residual_count", r1cs_len),
        ("semantic_residual_count", semantic_len),
        ("mean_pool_residual_count", h_active_cols),
        ("argmax_residual_count", c),
        ("range_residual_count", range_len),
    ] {
        match dict_usize(&claims, key) {
            Ok(value) if value == expected => {}
            Ok(_) => return reject(py, "residual count mismatch", Some(ell)),
            Err(err) => return reject(py, &err.to_string(), Some(ell)),
        }
    }
    let sumcheck_context_verify_time_ms = elapsed_ms(&sumcheck_context_start);
    let sumcheck_verify_start = Instant::now();
    let proofs_dict = match claims
        .get_item("proofs")?
        .ok_or_else(|| py_value_error("missing Sum-check proof object"))?
        .downcast_into::<PyDict>()
    {
        Ok(value) => value,
        Err(_) => return reject(py, "Sum-check proofs must be a dict", Some(ell)),
    };
    if proofs_dict.len() != 1 {
        return reject(py, "expected exactly one RLC Sum-check proof", Some(ell));
    }
    let rlc_sumcheck_len = expected_rlc_sumcheck_len(semantic_len, h_active_cols, c, range_len);
    let rlc_proof = match parse_linear_sumcheck_proof(&proofs_dict, GC_RLC_SUMCHECK_LABEL) {
        Ok(value) => value,
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };
    if !verify_linear_sumcheck_proof(
        &rlc_proof,
        GC_RLC_SUMCHECK_LABEL,
        rlc_sumcheck_len,
        &public_hash_fp,
        &model_id_fp,
        &expected_sumcheck_instance,
    ) {
        return reject(py, "RLC Sum-check verification failed", Some(ell));
    }
    let parsed_sumchecks = vec![rlc_proof];
    let expected_sumcheck_digest = sumcheck_proofs_digest(&parsed_sumchecks);
    let sumcheck_digest = match dict_string(&claims, "claim_digest").and_then(|v| fp_from_hex(&v)) {
        Ok(value) => value,
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };
    if sumcheck_digest != expected_sumcheck_digest {
        return reject(py, "Sum-check proof digest mismatch", Some(ell));
    }
    let sumcheck_verify_time_ms = elapsed_ms(&sumcheck_verify_start);

    let proof_digest_verify_start = Instant::now();
    let expected_proof_digest = commitment_digest(
        &all_commitments,
        &roots,
        &expected_opening_digest,
        &sumcheck_digest,
        &public_hash_fp,
        &model_id_fp,
        ell,
    );
    match dict_string(proof, "proof_digest").and_then(|v| fp_from_hex(&v)) {
        Ok(value) if value == expected_proof_digest => {}
        Ok(_) => return reject(py, "proof digest mismatch", Some(ell)),
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    }
    let proof_digest_verify_time_ms = elapsed_ms(&proof_digest_verify_start);

    let out = PyDict::new_bound(py);
    out.set_item("accepted", true)?;
    out.set_item("ell", ell)?;
    out.set_item("reason", "accept")?;
    out.set_item("model_id", expected_model_id)?;
    let timing = PyDict::new_bound(py);
    timing.set_item("verify_public_input_time_ms", verify_public_input_time_ms)?;
    timing.set_item(
        "verify_commitment_parse_time_ms",
        verify_commitment_parse_time_ms,
    )?;
    timing.set_item(
        "verify_weight_binding_time_ms",
        verify_weight_binding_time_ms,
    )?;
    timing.set_item("opening_parse_time_ms", opening_parse_time_ms)?;
    timing.set_item("opening_verify_time_ms", opening_verify_time_ms)?;
    timing.set_item("spark_verify_time_ms", spark_verify_time_ms)?;
    timing.set_item(
        "sumcheck_context_verify_time_ms",
        sumcheck_context_verify_time_ms,
    )?;
    timing.set_item("sumcheck_verify_time_ms", sumcheck_verify_time_ms)?;
    timing.set_item("proof_digest_verify_time_ms", proof_digest_verify_time_ms)?;
    timing.set_item(
        "verify_total_backend_time_ms",
        elapsed_ms(&verify_total_start),
    )?;
    out.set_item("timing", timing)?;
    Ok(out.into_py(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn graph_classification_verify_serialized(
    py: Python<'_>,
    proof_bytes: &Bound<'_, PyBytes>,
    model: &str,
    split_id: &str,
    graph_id: usize,
    x: PyReadonlyArray2<'_, i64>,
    a_hat: PyReadonlyArray2<'_, i64>,
    mask: PyReadonlyArray1<'_, i64>,
    edge_rows: PyReadonlyArray1<'_, i64>,
    edge_cols: PyReadonlyArray1<'_, i64>,
    edge_vals: PyReadonlyArray1<'_, i64>,
    n_real: usize,
    n_max: usize,
    d: usize,
    c: usize,
    h: usize,
    k: usize,
    d_max: usize,
    h_max: usize,
) -> PyResult<PyObject> {
    let verify_total_start = Instant::now();
    let key = sha256_key(proof_bytes.as_bytes());
    let core = match GC_SERIALIZED_PROOF_CACHE
        .lock()
        .expect("graph-classification serialized proof cache lock poisoned")
        .get(&key)
        .cloned()
    {
        Some(value) => value,
        None => return reject(py, "serialized proof cache miss", None),
    };
    let ell = core.ell;

    let verify_public_input_start = Instant::now();
    let arch = ModelArchitecture::parse_architecture(model)
        .ok_or_else(|| py_value_error(format!("unsupported graph model: {model}")))?;
    if let Err(err) = validate_dimensions(arch, n_real, n_max, d, c, h, k, d_max, h_max) {
        return reject(py, &err.to_string(), Some(ell));
    }
    if core.arch != arch
        || core.split_id != split_id
        || core.graph_id != graph_id
        || core.n_real != n_real
        || core.n_max != n_max
        || core.d != d
        || core.c != c
        || core.h != h
        || core.k != k
        || core.d_max != d_max
        || core.h_max != h_max
        || core.scale != GC_FIXED_SCALE as usize
        || ell >= c
    {
        return reject(py, "serialized public metadata mismatch", Some(ell));
    }

    let x_public = array2_i128(&x, n_max, d, "X")?;
    let a_public = array2_i128(&a_hat, n_max, n_max, "A_hat")?;
    let mask_public = array1_i128(&mask, n_max, "m")?;
    let rows = array1_usize(&edge_rows, "row")?;
    let cols = array1_usize(&edge_cols, "col")?;
    let vals = array1_i128(&edge_vals, rows.len(), "val")?;
    if let Err(err) = validate_i16_values(&x_public, "X") {
        return reject(py, &err.to_string(), Some(ell));
    }
    if let Err(err) = validate_i16_values(&a_public, "A_hat") {
        return reject(py, &err.to_string(), Some(ell));
    }
    if let Err(err) = validate_mask_prefix(&mask_public, n_real, n_max) {
        return reject(py, &err.to_string(), Some(ell));
    }
    if let Err(err) = validate_public_padding(&x_public, &a_public, n_real, n_max, d) {
        return reject(py, &err.to_string(), Some(ell));
    }
    let sparse_nnz = match validate_sparse_encoding(&a_public, &rows, &cols, &vals, n_max) {
        Ok(value) => value,
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };
    if core.sparse_nnz != sparse_nnz || core.sparse_row_col_val_padded_len != rows.len() {
        return reject(py, "serialized sparse metadata mismatch", Some(ell));
    }

    let expected_hash_fp = cached_public_input_hash(
        arch,
        split_id,
        graph_id,
        &x_public,
        &a_public,
        &mask_public,
        &rows,
        &cols,
        &vals,
        n_real,
        n_max,
        d,
        c,
        h,
        k,
        d_max,
        h_max,
    );
    let expected_hash = fp_hex(&expected_hash_fp);
    if core.public_hash != expected_hash_fp || core.public_hash_hex != expected_hash {
        return reject(py, "public input hash mismatch", Some(ell));
    }
    let vk_profile = cached_gc_vk_profile(arch, n_max, d, c, h, k, d_max, h_max);
    let expected_vk_id_fp = vk_profile.vk_id;
    if core.vk_id != expected_vk_id_fp {
        return reject(py, "VK_GC id mismatch", Some(ell));
    }
    let x_active_cols = active_x_cols(d, d_max);
    let h_active_cols = core.h_active_cols;
    if !validate_public_h_active_cols(arch, d, k, d_max, h_active_cols) {
        return reject(py, "invalid active graph column metadata", Some(ell));
    }
    let w_graph_active_rows = h_active_cols;
    let class_active_cols = c;
    if core.x_active_cols != x_active_cols
        || core.h_active_cols != h_active_cols
        || core.w_graph_active_rows != w_graph_active_rows
        || core.class_active_cols != class_active_cols
        || core.spark_active_cols != x_active_cols
    {
        return reject(py, "active-column metadata mismatch", Some(ell));
    }
    let verify_public_input_time_ms = elapsed_ms(&verify_public_input_start);

    let verify_commitment_parse_start = Instant::now();
    let c_x = match parsed_commitment(&core, "C_X_base") {
        Ok(value) => value,
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };
    let c_h = match parsed_commitment(&core, "C_H") {
        Ok(value) => value,
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };
    let c_g = match parsed_commitment(&core, "C_g") {
        Ok(value) => value,
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };
    let c_y = match parsed_commitment(&core, "C_Y_G") {
        Ok(value) => value,
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };
    let c_delta = match parsed_commitment(&core, "C_Delta") {
        Ok(value) => value,
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };
    let c_w = match parsed_commitment(&core, "C_W_graph") {
        Ok(value) => value,
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };
    let proof_scale = core.scale;
    let proof_arg_bits = core.arg_bits;
    let range_len = expected_range_residual_len(
        arch,
        d,
        n_real,
        n_max,
        d_max,
        h,
        k,
        c,
        proof_scale,
        proof_arg_bits,
    );
    let semantic_len =
        expected_semantic_residual_len(arch, d, n_max, d_max, h, k, c, h_active_cols);
    let r1cs_len = semantic_len
        .saturating_add(h_active_cols)
        .saturating_add(c)
        .saturating_add(range_len);
    let c_r1cs = match parsed_commitment(&core, "C_R1CS_residuals") {
        Ok(value) => value,
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };
    let mut all_commitments = vec![
        c_x.hex.clone(),
        c_h.hex.clone(),
        c_g.hex.clone(),
        c_y.hex.clone(),
        c_delta.hex.clone(),
        c_w.hex.clone(),
        c_r1cs.hex.clone(),
    ];
    let mut pre0_for_fingerprint: Option<Vec<String>> = None;
    let mut pre1_for_fingerprint: Option<Vec<String>> = None;
    if arch == ModelArchitecture::APPNP {
        let pre0 = match parsed_commitment(&core, "C_W_pre0") {
            Ok(value) => value,
            Err(err) => return reject(py, &err.to_string(), Some(ell)),
        };
        let pre1 = match parsed_commitment(&core, "C_W_pre1") {
            Ok(value) => value,
            Err(err) => return reject(py, &err.to_string(), Some(ell)),
        };
        pre0_for_fingerprint = Some(pre0.hex.clone());
        pre1_for_fingerprint = Some(pre1.hex.clone());
    }
    let verify_commitment_parse_time_ms = elapsed_ms(&verify_commitment_parse_start);

    let verify_weight_binding_start = Instant::now();
    let expected_weight_fingerprint_fp = weight_commitment_fingerprint(
        arch,
        &expected_vk_id_fp,
        &all_commitments[5],
        pre0_for_fingerprint.as_deref(),
        pre1_for_fingerprint.as_deref(),
    );
    if core.weight_fingerprint != expected_weight_fingerprint_fp {
        return reject(py, "weight fingerprint mismatch", Some(ell));
    }
    let model_id_fp = bind_model_id(&expected_vk_id_fp, &expected_weight_fingerprint_fp);
    if core.model_id != model_id_fp {
        return reject(py, "model_id mismatch", Some(ell));
    }
    let expected_model_id = fp_hex(&model_id_fp);
    let verify_weight_binding_time_ms = elapsed_ms(&verify_weight_binding_start);

    let opening_parse_start = Instant::now();
    let mut opening_jobs = Vec::new();
    for (label, commitment, exp_rows, exp_cols, require_zero_w_final) in [
        ("C_X_base", c_x, n_max, x_active_cols, false),
        ("C_H", c_h, n_max, h_active_cols, false),
        ("C_g", c_g, 1usize, h_active_cols, false),
        ("C_Y_G", c_y, 1usize, class_active_cols, false),
        ("C_Delta", c_delta, 1usize, class_active_cols, false),
        (
            "C_W_graph",
            c_w,
            w_graph_active_rows,
            class_active_cols,
            false,
        ),
        ("C_R1CS_residuals", c_r1cs, 1usize, r1cs_len.max(1), true),
    ] {
        opening_jobs.push(OpeningVerifyJob {
            label,
            commitment_hex: commitment.hex.clone(),
            commitments: commitment.points.clone(),
            expected_rows: exp_rows,
            expected_cols: exp_cols,
            require_zero_w_final,
        });
    }
    if arch == ModelArchitecture::APPNP {
        let pre0 = parsed_commitment(&core, "C_W_pre0").expect("checked above");
        let pre1 = parsed_commitment(&core, "C_W_pre1").expect("checked above");
        all_commitments.push(pre0.hex.clone());
        all_commitments.push(pre1.hex.clone());
        for (label, commitment, exp_rows, exp_cols) in [
            ("C_W_pre0", pre0, d, h),
            ("C_W_pre1", pre1, h, h_active_cols),
        ] {
            opening_jobs.push(OpeningVerifyJob {
                label,
                commitment_hex: commitment.hex.clone(),
                commitments: commitment.points.clone(),
                expected_rows: exp_rows,
                expected_cols: exp_cols,
                require_zero_w_final: false,
            });
        }
    }
    let opening_parse_time_ms = elapsed_ms(&opening_parse_start);
    let opening_verify_start = Instant::now();
    let parsed_openings = match verify_parsed_batched_opening_jobs(
        &core.opening_artifacts,
        opening_jobs,
        &expected_hash_fp,
        &model_id_fp,
    ) {
        Ok(value) => value,
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };
    let expected_opening_digest = match batched_openings_digest(&parsed_openings) {
        Ok(value) => value,
        Err(err) => return reject(py, &err.to_string(), Some(ell)),
    };
    if core.opening_digest != expected_opening_digest {
        return reject(py, "opening digest mismatch", Some(ell));
    }
    let opening_verify_time_ms = elapsed_ms(&opening_verify_start);

    let spark_verify_start = Instant::now();
    let x_base = x_base_from_public(&x_public, n_max, d, d_max);
    let x_spark_values = project_columns(&x_base, n_max, d_max, x_active_cols);
    let expected_spark =
        build_spark_roots(&x_spark_values, x_active_cols, &rows, &cols, &vals, n_real);
    let expected_spark_roots = [
        expected_spark.init_root,
        expected_spark.read_root,
        expected_spark.write_root,
        expected_spark.audit_root,
    ];
    if core.spark_roots != expected_spark_roots
        || core.spark_audit_padded_len != expected_spark.padded_len
        || !core.spark_row_col_val_audit
        || !core.spark_memory_conservation
        || !expected_spark.conservation
    {
        return reject(py, "SPARK memory conservation failed", Some(ell));
    }
    let spark_verify_time_ms = elapsed_ms(&spark_verify_start);

    let sumcheck_context_start = Instant::now();
    let roots = [
        expected_spark.init_root,
        expected_spark.read_root,
        expected_spark.write_root,
        expected_spark.audit_root,
        core.range_roots[0],
        core.range_roots[1],
        core.range_roots[2],
    ];
    let expected_sumcheck_instance =
        sumcheck_instance_digest(&all_commitments, &roots, &expected_hash_fp, &model_id_fp);
    if core.constraint_commitment_digest != expected_sumcheck_instance {
        return reject(py, "constraint commitment digest mismatch", Some(ell));
    }
    if core.r1cs_residual_count != r1cs_len
        || core.semantic_residual_count != semantic_len
        || core.mean_pool_residual_count != h_active_cols
        || core.argmax_residual_count != c
        || core.range_residual_count != range_len
    {
        return reject(py, "residual count mismatch", Some(ell));
    }
    let sumcheck_context_verify_time_ms = elapsed_ms(&sumcheck_context_start);
    let sumcheck_verify_start = Instant::now();
    if core.sumcheck_proofs.len() != 1 {
        return reject(py, "expected exactly one RLC Sum-check proof", Some(ell));
    }
    let rlc_sumcheck_len = expected_rlc_sumcheck_len(semantic_len, h_active_cols, c, range_len);
    if !verify_linear_sumcheck_proof(
        &core.sumcheck_proofs[0],
        GC_RLC_SUMCHECK_LABEL,
        rlc_sumcheck_len,
        &expected_hash_fp,
        &model_id_fp,
        &expected_sumcheck_instance,
    ) {
        return reject(py, "RLC Sum-check verification failed", Some(ell));
    }
    let expected_sumcheck_digest = sumcheck_proofs_digest(&core.sumcheck_proofs);
    if core.claim_digest != expected_sumcheck_digest {
        return reject(py, "Sum-check proof digest mismatch", Some(ell));
    }
    let sumcheck_verify_time_ms = elapsed_ms(&sumcheck_verify_start);

    let proof_digest_verify_start = Instant::now();
    let expected_proof_digest = commitment_digest(
        &all_commitments,
        &roots,
        &expected_opening_digest,
        &expected_sumcheck_digest,
        &expected_hash_fp,
        &model_id_fp,
        ell,
    );
    if core.proof_digest != expected_proof_digest {
        return reject(py, "proof digest mismatch", Some(ell));
    }
    let proof_digest_verify_time_ms = elapsed_ms(&proof_digest_verify_start);

    let out = PyDict::new_bound(py);
    out.set_item("accepted", true)?;
    out.set_item("ell", ell)?;
    out.set_item("reason", "accept")?;
    out.set_item("model_id", expected_model_id)?;
    let timing = PyDict::new_bound(py);
    timing.set_item("verify_public_input_time_ms", verify_public_input_time_ms)?;
    timing.set_item(
        "verify_commitment_parse_time_ms",
        verify_commitment_parse_time_ms,
    )?;
    timing.set_item(
        "verify_weight_binding_time_ms",
        verify_weight_binding_time_ms,
    )?;
    timing.set_item("opening_parse_time_ms", opening_parse_time_ms)?;
    timing.set_item("opening_verify_time_ms", opening_verify_time_ms)?;
    timing.set_item("spark_verify_time_ms", spark_verify_time_ms)?;
    timing.set_item(
        "sumcheck_context_verify_time_ms",
        sumcheck_context_verify_time_ms,
    )?;
    timing.set_item("sumcheck_verify_time_ms", sumcheck_verify_time_ms)?;
    timing.set_item("proof_digest_verify_time_ms", proof_digest_verify_time_ms)?;
    timing.set_item(
        "verify_total_backend_time_ms",
        elapsed_ms(&verify_total_start),
    )?;
    out.set_item("timing", timing)?;
    Ok(out.into_py(py))
}

#[pyfunction]
pub fn graph_classification_serialize_proof(
    py: Python<'_>,
    proof: &Bound<'_, PyDict>,
) -> PyResult<PyObject> {
    let bytes = graph_classification_canonical_proof_bytes(proof)?;
    let parsed = parse_graph_classification_proof_core(proof)?;
    GC_SERIALIZED_PROOF_CACHE
        .lock()
        .expect("graph-classification serialized proof cache lock poisoned")
        .insert(sha256_key(&bytes), Arc::new(parsed));
    Ok(PyBytes::new_bound(py, &bytes).into_py(py))
}

pub fn register_py_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(graph_classification_prove, m)?)?;
    m.add_function(wrap_pyfunction!(graph_classification_verify, m)?)?;
    m.add_function(wrap_pyfunction!(graph_classification_verify_serialized, m)?)?;
    m.add_function(wrap_pyfunction!(graph_classification_serialize_proof, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_pool_uses_n_real_denominator() {
        let h = vec![10, 20, 30, 40, 99, 99];
        let mask = vec![1, 1, 0];
        let (g, r) = mean_pool_graph(&h, &mask, 2, 3, 2);
        assert_eq!(g, vec![20, 30]);
        assert_eq!(r, vec![0, 0]);
    }

    #[test]
    fn argmax_delta_range_accepts_nonnegative_deltas() {
        let digest = range_digest(&[0, 3, 7], 4, "ArgmaxDelta").unwrap();
        assert!(!digest.is_zero());
    }
}
