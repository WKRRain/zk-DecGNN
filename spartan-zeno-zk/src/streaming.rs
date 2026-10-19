//! Streaming kernels for large Spartan-ZENO benchmark instances.
//!
//! The Python benchmark implementation is useful for protocol orchestration,
//! but Reddit-scale traces cannot be handled by Python scalar loops.  This
//! module exposes Rust/PyO3 kernels for the hot MLE, sparse adjacency, and row
//! commitment operations while preserving the Python protocol object layout.

use crate::curve_config::Fp;
use ark_ff::{BigInteger, PrimeField, Zero};
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyModule, PySequence};
use rand::RngCore;
use rayon::prelude::*;
use std::str::FromStr;

fn fp_from_i64(value: i64) -> Fp {
    if value >= 0 {
        Fp::from(value as u64)
    } else {
        -Fp::from(value.unsigned_abs())
    }
}

fn random_fp() -> Fp {
    let mut bytes = [0u8; 64];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    Fp::from_le_bytes_mod_order(&bytes)
}

fn py_int_to_fp(obj: &Bound<'_, PyAny>) -> PyResult<Fp> {
    let text = obj.str()?.to_str()?.to_owned();
    Fp::from_str(&text).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!("cannot parse field scalar: {text}"))
    })
}

fn py_sequence_to_fp_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Fp>> {
    let seq = obj.downcast::<PySequence>()?;
    let len = seq.len()?;
    let mut out = Vec::with_capacity(len);
    for idx in 0..len {
        out.push(py_int_to_fp(&seq.get_item(idx)?)?);
    }
    Ok(out)
}

fn python_tensor_basis(point: &[Fp]) -> Vec<Fp> {
    let mut basis = vec![Fp::from(1u64)];
    for &coord in point {
        let one_minus = Fp::from(1u64) - coord;
        let mut next = Vec::with_capacity(basis.len() * 2);
        for old in basis.iter() {
            next.push(*old * one_minus);
        }
        for old in basis.iter() {
            next.push(*old * coord);
        }
        basis = next;
    }
    basis
}

fn fp_to_py_object(py: Python<'_>, value: &Fp) -> PyResult<PyObject> {
    let bytes = value.into_bigint().to_bytes_be();
    let builtins = py.import_bound("builtins")?;
    let int_type = builtins.getattr("int")?;
    let py_bytes = PyBytes::new_bound(py, &bytes);
    Ok(int_type
        .call_method1("from_bytes", (py_bytes, "big"))?
        .into_py(py))
}

fn fp_vec_to_py_list(py: Python<'_>, values: &[Fp]) -> PyResult<PyObject> {
    let mut items = Vec::with_capacity(values.len());
    for value in values {
        items.push(fp_to_py_object(py, value)?);
    }
    Ok(PyList::new_bound(py, items).into_py(py))
}

fn opening_tuple_to_py(
    py: Python<'_>,
    folded: &[Fp],
    folded_blinding: &Fp,
    folded_commitment: &Fp,
    value: &Fp,
) -> PyResult<PyObject> {
    Ok((
        fp_vec_to_py_list(py, folded)?,
        fp_to_py_object(py, folded_blinding)?,
        fp_to_py_object(py, folded_commitment)?,
        fp_to_py_object(py, value)?,
    )
        .into_py(py))
}

fn matrix_shape_i64(matrix: &PyReadonlyArray2<'_, i64>) -> PyResult<(usize, usize)> {
    let view = matrix.as_array();
    let shape = view.shape();
    if shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "expected a 2D contiguous numpy array",
        ));
    }
    Ok((shape[0], shape[1]))
}

fn next_power_of_two(value: usize) -> usize {
    if value <= 1 {
        1
    } else {
        value.next_power_of_two()
    }
}

fn transcript_absorb_scalar(
    py: Python<'_>,
    transcript: &Bound<'_, PyAny>,
    label: String,
    value: &Fp,
) -> PyResult<()> {
    transcript.call_method1("absorb_scalar", (label, fp_to_py_object(py, value)?))?;
    Ok(())
}

fn transcript_challenge(transcript: &Bound<'_, PyAny>, label: String) -> PyResult<Fp> {
    let value = transcript.call_method1("challenge", (label,))?;
    py_int_to_fp(&value)
}

fn interpolate_at_two(left: Fp, right: Fp) -> Fp {
    -left + Fp::from(2u64) * right
}

#[pyfunction]
fn row_mle_values_i64(
    py: Python<'_>,
    matrix: PyReadonlyArray2<'_, i64>,
    row_point: &Bound<'_, PyAny>,
    virtual_cols: usize,
) -> PyResult<PyObject> {
    let (rows, cols) = matrix_shape_i64(&matrix)?;
    let values = matrix.as_slice()?;
    let row_basis = python_tensor_basis(&py_sequence_to_fp_vec(row_point)?);
    let effective_rows = rows.min(row_basis.len());
    let out: Vec<Fp> = (0..virtual_cols)
        .into_par_iter()
        .map(|j| {
            if j >= cols {
                return Fp::zero();
            }
            let mut acc = Fp::zero();
            for i in 0..effective_rows {
                let v = values[i * cols + j];
                if v != 0 {
                    acc += row_basis[i] * fp_from_i64(v);
                }
            }
            acc
        })
        .collect();
    fp_vec_to_py_list(py, &out)
}

#[pyfunction]
fn col_mle_values_i64(
    py: Python<'_>,
    matrix: PyReadonlyArray2<'_, i64>,
    col_point: &Bound<'_, PyAny>,
    virtual_rows: usize,
) -> PyResult<PyObject> {
    let (rows, cols) = matrix_shape_i64(&matrix)?;
    let values = matrix.as_slice()?;
    let col_basis = python_tensor_basis(&py_sequence_to_fp_vec(col_point)?);
    let effective_cols = cols.min(col_basis.len());
    let out: Vec<Fp> = (0..virtual_rows)
        .into_par_iter()
        .map(|i| {
            if i >= rows {
                return Fp::zero();
            }
            let base = i * cols;
            let mut acc = Fp::zero();
            for j in 0..effective_cols {
                let v = values[base + j];
                if v != 0 {
                    acc += fp_from_i64(v) * col_basis[j];
                }
            }
            acc
        })
        .collect();
    fp_vec_to_py_list(py, &out)
}

#[pyfunction]
fn sparse_a_row_vector_i64(
    py: Python<'_>,
    rows: PyReadonlyArray1<'_, i64>,
    cols: PyReadonlyArray1<'_, i64>,
    weights: PyReadonlyArray1<'_, i64>,
    row_point: &Bound<'_, PyAny>,
    n_max: usize,
) -> PyResult<PyObject> {
    let row_idx = rows.as_slice()?;
    let col_idx = cols.as_slice()?;
    let weight_values = weights.as_slice()?;
    if row_idx.len() != col_idx.len() || row_idx.len() != weight_values.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rows, cols, and weights must have equal length",
        ));
    }
    let row_basis = python_tensor_basis(&py_sequence_to_fp_vec(row_point)?);
    let out = (0..row_idx.len())
        .into_par_iter()
        .with_min_len(250_000)
        .fold(
            || vec![Fp::zero(); n_max],
            |mut acc, idx| {
                let row = row_idx[idx];
                let col = col_idx[idx];
                let weight = weight_values[idx];
                if row >= 0 && col >= 0 && weight != 0 {
                    let r = row as usize;
                    let c = col as usize;
                    if r < row_basis.len() && c < n_max {
                        acc[c] += row_basis[r] * fp_from_i64(weight);
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![Fp::zero(); n_max],
            |mut left, right| {
                for (l, r) in left.iter_mut().zip(right.iter()) {
                    *l += *r;
                }
                left
            },
        );
    fp_vec_to_py_list(py, &out)
}

#[pyfunction]
fn sparse_a_terminal_eval_i64(
    py: Python<'_>,
    rows: PyReadonlyArray1<'_, i64>,
    cols: PyReadonlyArray1<'_, i64>,
    weights: PyReadonlyArray1<'_, i64>,
    outer_row_point: &Bound<'_, PyAny>,
    terminal_source_point: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let row_idx = rows.as_slice()?;
    let col_idx = cols.as_slice()?;
    let weight_values = weights.as_slice()?;
    if row_idx.len() != col_idx.len() || row_idx.len() != weight_values.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rows, cols, and weights must have equal length",
        ));
    }
    let outer = python_tensor_basis(&py_sequence_to_fp_vec(outer_row_point)?);
    let source = python_tensor_basis(&py_sequence_to_fp_vec(terminal_source_point)?);
    let total = (0..row_idx.len())
        .into_par_iter()
        .with_min_len(250_000)
        .map(|idx| {
            let row = row_idx[idx];
            let col = col_idx[idx];
            let weight = weight_values[idx];
            if row < 0 || col < 0 || weight == 0 {
                return Fp::zero();
            }
            let r = row as usize;
            let c = col as usize;
            if r >= outer.len() || c >= source.len() {
                return Fp::zero();
            }
            outer[r] * source[c] * fp_from_i64(weight)
        })
        .reduce(Fp::zero, |a, b| a + b);
    fp_to_py_object(py, &total)
}

#[pyfunction]
fn sign2_concat_row_mle_values_i64(
    py: Python<'_>,
    x_base: PyReadonlyArray2<'_, i64>,
    s1: PyReadonlyArray2<'_, i64>,
    s2: PyReadonlyArray2<'_, i64>,
    row_point: &Bound<'_, PyAny>,
    d_real: usize,
    d_block: usize,
    d_max: usize,
) -> PyResult<PyObject> {
    let matrices = [x_base, s1, s2];
    let row_basis = python_tensor_basis(&py_sequence_to_fp_vec(row_point)?);
    let mut slices = Vec::with_capacity(3);
    let mut shapes = Vec::with_capacity(3);
    for matrix in &matrices {
        let (rows, cols) = matrix_shape_i64(matrix)?;
        if d_real > cols {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "d_real exceeds one SIGN branch matrix width",
            ));
        }
        slices.push(matrix.as_slice()?);
        shapes.push((rows, cols));
    }
    let out: Vec<Fp> = (0..d_max)
        .into_par_iter()
        .map(|global_col| {
            let block = global_col / d_block;
            let local_col = global_col % d_block;
            if block >= 3 || local_col >= d_real {
                return Fp::zero();
            }
            let (rows, cols) = shapes[block];
            let effective_rows = rows.min(row_basis.len());
            let data = slices[block];
            let mut acc = Fp::zero();
            for i in 0..effective_rows {
                let v = data[i * cols + local_col];
                if v != 0 {
                    acc += row_basis[i] * fp_from_i64(v);
                }
            }
            acc
        })
        .collect();
    fp_vec_to_py_list(py, &out)
}

#[pyfunction]
fn commit_matrix_i64(
    py: Python<'_>,
    matrix: PyReadonlyArray2<'_, i64>,
    virtual_rows: usize,
    virtual_cols: usize,
    generators: &Bound<'_, PyAny>,
    h_value: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let (rows, cols) = matrix_shape_i64(&matrix)?;
    let values = matrix.as_slice()?;
    let gens = py_sequence_to_fp_vec(generators)?;
    if gens.len() < virtual_cols {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "not enough Hyrax generators for virtual_cols",
        ));
    }
    let h = py_int_to_fp(h_value)?;
    let blindings: Vec<Fp> = (0..virtual_rows).map(|_| random_fp()).collect();
    let commitments: Vec<Fp> = (0..virtual_rows)
        .into_par_iter()
        .map(|row| {
            let mut acc = h * blindings[row];
            if row < rows {
                let base = row * cols;
                let effective_cols = cols.min(virtual_cols);
                for col in 0..effective_cols {
                    let v = values[base + col];
                    if v != 0 {
                        acc += gens[col] * fp_from_i64(v);
                    }
                }
            }
            acc
        })
        .collect();
    let py_commitments = fp_vec_to_py_list(py, &commitments)?;
    let py_blindings = fp_vec_to_py_list(py, &blindings)?;
    Ok((py_commitments, py_blindings).into_py(py))
}

#[pyfunction]
fn commit_sign2_z_i64(
    py: Python<'_>,
    x_base: PyReadonlyArray2<'_, i64>,
    s1: PyReadonlyArray2<'_, i64>,
    s2: PyReadonlyArray2<'_, i64>,
    virtual_rows: usize,
    virtual_cols: usize,
    d_real: usize,
    d_block: usize,
    generators: &Bound<'_, PyAny>,
    h_value: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let matrices = [x_base, s1, s2];
    let mut slices = Vec::with_capacity(3);
    let mut shapes = Vec::with_capacity(3);
    for matrix in &matrices {
        let (rows, cols) = matrix_shape_i64(matrix)?;
        if d_real > cols {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "d_real exceeds one SIGN branch matrix width",
            ));
        }
        slices.push(matrix.as_slice()?);
        shapes.push((rows, cols));
    }
    let gens = py_sequence_to_fp_vec(generators)?;
    if gens.len() < virtual_cols {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "not enough Hyrax generators for virtual_cols",
        ));
    }
    let h = py_int_to_fp(h_value)?;
    let blindings: Vec<Fp> = (0..virtual_rows).map(|_| random_fp()).collect();
    let commitments: Vec<Fp> = (0..virtual_rows)
        .into_par_iter()
        .map(|row| {
            let mut acc = h * blindings[row];
            for block in 0..3 {
                let (rows, cols) = shapes[block];
                if row >= rows {
                    continue;
                }
                let base = row * cols;
                let offset = block * d_block;
                for local_col in 0..d_real {
                    let global_col = offset + local_col;
                    if global_col >= virtual_cols {
                        break;
                    }
                    let v = slices[block][base + local_col];
                    if v != 0 {
                        acc += gens[global_col] * fp_from_i64(v);
                    }
                }
            }
            acc
        })
        .collect();
    let py_commitments = fp_vec_to_py_list(py, &commitments)?;
    let py_blindings = fp_vec_to_py_list(py, &blindings)?;
    Ok((py_commitments, py_blindings).into_py(py))
}

#[pyfunction]
fn fold_commitments(
    py: Python<'_>,
    commitments: &Bound<'_, PyAny>,
    row_point: &Bound<'_, PyAny>,
    virtual_rows: usize,
) -> PyResult<PyObject> {
    let commitment_values = py_sequence_to_fp_vec(commitments)?;
    if commitment_values.len() < virtual_rows {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "not enough row commitments for virtual_rows",
        ));
    }
    let row_basis = python_tensor_basis(&py_sequence_to_fp_vec(row_point)?);
    if row_basis.len() != virtual_rows {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "row point dimension does not match virtual_rows",
        ));
    }
    let total = (0..virtual_rows)
        .into_par_iter()
        .with_min_len(16_384)
        .map(|row| commitment_values[row] * row_basis[row])
        .reduce(Fp::zero, |a, b| a + b);
    fp_to_py_object(py, &total)
}

#[pyfunction]
fn fold_matrix_opening_i64(
    py: Python<'_>,
    matrix: PyReadonlyArray2<'_, i64>,
    blindings: &Bound<'_, PyAny>,
    commitments: &Bound<'_, PyAny>,
    row_point: &Bound<'_, PyAny>,
    col_point: &Bound<'_, PyAny>,
    virtual_rows: usize,
    virtual_cols: usize,
) -> PyResult<PyObject> {
    let (rows, cols) = matrix_shape_i64(&matrix)?;
    let values = matrix.as_slice()?;
    let blinding_values = py_sequence_to_fp_vec(blindings)?;
    let commitment_values = py_sequence_to_fp_vec(commitments)?;
    if blinding_values.len() < virtual_rows || commitment_values.len() < virtual_rows {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "not enough blindings/commitments for virtual_rows",
        ));
    }
    let row_basis = python_tensor_basis(&py_sequence_to_fp_vec(row_point)?);
    let col_basis = python_tensor_basis(&py_sequence_to_fp_vec(col_point)?);
    if row_basis.len() != virtual_rows || col_basis.len() != virtual_cols {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "opening point dimensions do not match virtual matrix shape",
        ));
    }
    let effective_cols = cols.min(virtual_cols);
    let (folded, folded_blinding, folded_commitment) = (0..virtual_rows)
        .into_par_iter()
        .with_min_len(1024)
        .fold(
            || (vec![Fp::zero(); virtual_cols], Fp::zero(), Fp::zero()),
            |mut state, row| {
                let b = row_basis[row];
                state.1 += b * blinding_values[row];
                state.2 += b * commitment_values[row];
                if row < rows {
                    let base = row * cols;
                    for col in 0..effective_cols {
                        let v = values[base + col];
                        if v != 0 {
                            state.0[col] += b * fp_from_i64(v);
                        }
                    }
                }
                state
            },
        )
        .reduce(
            || (vec![Fp::zero(); virtual_cols], Fp::zero(), Fp::zero()),
            |mut left, right| {
                for (l, r) in left.0.iter_mut().zip(right.0.iter()) {
                    *l += *r;
                }
                left.1 += right.1;
                left.2 += right.2;
                left
            },
        );
    let value = folded
        .iter()
        .zip(col_basis.iter())
        .fold(Fp::zero(), |acc, (v, b)| acc + (*v * *b));
    opening_tuple_to_py(py, &folded, &folded_blinding, &folded_commitment, &value)
}

#[pyfunction]
fn fold_sign2_z_opening_i64(
    py: Python<'_>,
    x_base: PyReadonlyArray2<'_, i64>,
    s1: PyReadonlyArray2<'_, i64>,
    s2: PyReadonlyArray2<'_, i64>,
    blindings: &Bound<'_, PyAny>,
    commitments: &Bound<'_, PyAny>,
    row_point: &Bound<'_, PyAny>,
    col_point: &Bound<'_, PyAny>,
    virtual_rows: usize,
    virtual_cols: usize,
    d_real: usize,
    d_block: usize,
) -> PyResult<PyObject> {
    let matrices = [x_base, s1, s2];
    let mut slices = Vec::with_capacity(3);
    let mut shapes = Vec::with_capacity(3);
    for matrix in &matrices {
        let (rows, cols) = matrix_shape_i64(matrix)?;
        if d_real > cols {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "d_real exceeds one SIGN branch matrix width",
            ));
        }
        slices.push(matrix.as_slice()?);
        shapes.push((rows, cols));
    }
    let blinding_values = py_sequence_to_fp_vec(blindings)?;
    let commitment_values = py_sequence_to_fp_vec(commitments)?;
    if blinding_values.len() < virtual_rows || commitment_values.len() < virtual_rows {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "not enough blindings/commitments for virtual_rows",
        ));
    }
    let row_basis = python_tensor_basis(&py_sequence_to_fp_vec(row_point)?);
    let col_basis = python_tensor_basis(&py_sequence_to_fp_vec(col_point)?);
    if row_basis.len() != virtual_rows || col_basis.len() != virtual_cols {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "opening point dimensions do not match virtual matrix shape",
        ));
    }
    let (folded, folded_blinding, folded_commitment) = (0..virtual_rows)
        .into_par_iter()
        .with_min_len(1024)
        .fold(
            || (vec![Fp::zero(); virtual_cols], Fp::zero(), Fp::zero()),
            |mut state, row| {
                let b = row_basis[row];
                state.1 += b * blinding_values[row];
                state.2 += b * commitment_values[row];
                for block in 0..3 {
                    let (rows, cols) = shapes[block];
                    if row >= rows {
                        continue;
                    }
                    let base = row * cols;
                    let offset = block * d_block;
                    for local_col in 0..d_real {
                        let global_col = offset + local_col;
                        if global_col >= virtual_cols {
                            break;
                        }
                        let v = slices[block][base + local_col];
                        if v != 0 {
                            state.0[global_col] += b * fp_from_i64(v);
                        }
                    }
                }
                state
            },
        )
        .reduce(
            || (vec![Fp::zero(); virtual_cols], Fp::zero(), Fp::zero()),
            |mut left, right| {
                for (l, r) in left.0.iter_mut().zip(right.0.iter()) {
                    *l += *r;
                }
                left.1 += right.1;
                left.2 += right.2;
                left
            },
        );
    let value = folded
        .iter()
        .zip(col_basis.iter())
        .fold(Fp::zero(), |acc, (v, b)| acc + (*v * *b));
    opening_tuple_to_py(py, &folded, &folded_blinding, &folded_commitment, &value)
}

#[pyfunction]
fn product_sumcheck_flat_matrices_i64(
    py: Python<'_>,
    left_matrix: PyReadonlyArray2<'_, i64>,
    right_matrix: PyReadonlyArray2<'_, i64>,
    transcript: &Bound<'_, PyAny>,
    label: String,
) -> PyResult<PyObject> {
    let (left_rows, left_cols) = matrix_shape_i64(&left_matrix)?;
    let (right_rows, right_cols) = matrix_shape_i64(&right_matrix)?;
    if left_rows != right_rows || left_cols != right_cols {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "left/right matrices must have equal shape",
        ));
    }
    let left_values = left_matrix.as_slice()?;
    let right_values = right_matrix.as_slice()?;
    let logical_len = left_rows.saturating_mul(left_cols);
    let target_len = next_power_of_two(logical_len.max(1));
    let mut left = vec![Fp::zero(); target_len];
    let mut right = vec![Fp::zero(); target_len];
    left.par_iter_mut()
        .zip(right.par_iter_mut())
        .enumerate()
        .for_each(|(idx, (l, r))| {
            if idx < logical_len {
                *l = fp_from_i64(left_values[idx]);
                *r = fp_from_i64(right_values[idx]);
            }
        });

    let claim = left
        .par_iter()
        .zip(right.par_iter())
        .map(|(a, b)| *a * *b)
        .reduce(Fp::zero, |a, b| a + b);
    transcript_absorb_scalar(py, transcript, format!("{label}.claim"), &claim)?;

    let mut rounds: Vec<(Fp, Fp, Fp)> = Vec::new();
    let mut challenges: Vec<Fp> = Vec::new();
    let mut round_idx = 0usize;
    while left.len() > 1 {
        let half = left.len() / 2;
        let (left_l, left_r) = left.split_at(half);
        let (right_l, right_r) = right.split_at(half);

        let g0 = left_l
            .par_iter()
            .zip(right_l.par_iter())
            .map(|(a, b)| *a * *b)
            .reduce(Fp::zero, |a, b| a + b);
        let g1 = left_r
            .par_iter()
            .zip(right_r.par_iter())
            .map(|(a, b)| *a * *b)
            .reduce(Fp::zero, |a, b| a + b);
        let g2 = (0..half)
            .into_par_iter()
            .map(|idx| {
                let l2 = interpolate_at_two(left_l[idx], left_r[idx]);
                let r2 = interpolate_at_two(right_l[idx], right_r[idx]);
                l2 * r2
            })
            .reduce(Fp::zero, |a, b| a + b);

        rounds.push((g0, g1, g2));
        transcript_absorb_scalar(py, transcript, format!("{label}.g0.{round_idx}"), &g0)?;
        transcript_absorb_scalar(py, transcript, format!("{label}.g1.{round_idx}"), &g1)?;
        transcript_absorb_scalar(py, transcript, format!("{label}.g2.{round_idx}"), &g2)?;
        let r = transcript_challenge(transcript, format!("{label}.r.{round_idx}"))?;
        challenges.push(r);

        let one_minus_r = Fp::from(1u64) - r;
        let next_left: Vec<Fp> = (0..half)
            .into_par_iter()
            .map(|idx| one_minus_r * left_l[idx] + r * left_r[idx])
            .collect();
        let next_right: Vec<Fp> = (0..half)
            .into_par_iter()
            .map(|idx| one_minus_r * right_l[idx] + r * right_r[idx])
            .collect();
        left = next_left;
        right = next_right;
        round_idx += 1;
    }

    let terminal_left = left.first().copied().unwrap_or_else(Fp::zero);
    let terminal_right = right.first().copied().unwrap_or_else(Fp::zero);
    transcript_absorb_scalar(
        py,
        transcript,
        format!("{label}.terminal_left"),
        &terminal_left,
    )?;
    transcript_absorb_scalar(
        py,
        transcript,
        format!("{label}.terminal_right"),
        &terminal_right,
    )?;

    let py_rounds: Vec<PyObject> = rounds
        .iter()
        .map(|(g0, g1, g2)| {
            Ok((
                fp_to_py_object(py, g0)?,
                fp_to_py_object(py, g1)?,
                fp_to_py_object(py, g2)?,
            )
                .into_py(py))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok((
        fp_to_py_object(py, &claim)?,
        PyList::new_bound(py, py_rounds).into_py(py),
        fp_to_py_object(py, &terminal_left)?,
        fp_to_py_object(py, &terminal_right)?,
        fp_vec_to_py_list(py, &challenges)?,
    )
        .into_py(py))
}

pub fn register_py_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(row_mle_values_i64, m)?)?;
    m.add_function(wrap_pyfunction!(col_mle_values_i64, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_a_row_vector_i64, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_a_terminal_eval_i64, m)?)?;
    m.add_function(wrap_pyfunction!(sign2_concat_row_mle_values_i64, m)?)?;
    m.add_function(wrap_pyfunction!(commit_matrix_i64, m)?)?;
    m.add_function(wrap_pyfunction!(commit_sign2_z_i64, m)?)?;
    m.add_function(wrap_pyfunction!(fold_commitments, m)?)?;
    m.add_function(wrap_pyfunction!(fold_matrix_opening_i64, m)?)?;
    m.add_function(wrap_pyfunction!(fold_sign2_z_opening_i64, m)?)?;
    m.add_function(wrap_pyfunction!(product_sumcheck_flat_matrices_i64, m)?)?;
    Ok(())
}
