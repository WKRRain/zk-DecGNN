"""Rust streaming backend loader for Spartan-ZENO benchmarks.

The strict benchmark scripts keep the protocol transcript and proof objects in
Python, but Reddit-scale MLE and commitment kernels are loaded from the Rust
extension built in ``target/release``.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = REPO_ROOT / "target" / "release" / "libspartan_zeno_zk.so"


def _rust_sources_newer_than_library() -> bool:
    if not LIB_PATH.exists():
        return True
    lib_mtime = LIB_PATH.stat().st_mtime
    candidates = [REPO_ROOT / "Cargo.toml", REPO_ROOT / "Cargo.lock"]
    candidates.extend((REPO_ROOT / "src").rglob("*.rs"))
    return any(path.exists() and path.stat().st_mtime > lib_mtime for path in candidates)


def load(build: bool = True):
    if build and _rust_sources_newer_than_library():
        subprocess.run(["cargo", "build", "--release", "--lib"], cwd=REPO_ROOT, check=True)
    if not LIB_PATH.exists():
        raise RuntimeError(f"Rust streaming backend not built: {LIB_PATH}")
    spec = importlib.util.spec_from_file_location("spartan_zeno_zk", LIB_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Rust streaming backend from {LIB_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contiguous_i64(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


def curve_point_values(points: Iterable[object]) -> list[int]:
    return [int(point.value) for point in points]


def curve_points(point_cls: type, values: Sequence[int]) -> list[object]:
    return [point_cls(int(value)) for value in values]


def rust_commit_matrix(
    rust,
    matrix: np.ndarray,
    virtual_rows: int,
    virtual_cols: int,
    generators: Sequence[int],
    h_value: int,
) -> Tuple[list[int], list[int]]:
    commitments, blindings = rust.commit_matrix_i64(
        contiguous_i64(matrix),
        int(virtual_rows),
        int(virtual_cols),
        list(generators),
        int(h_value),
    )
    return [int(v) for v in commitments], [int(v) for v in blindings]


def rust_commit_sign2_z(
    rust,
    x_base: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    virtual_rows: int,
    virtual_cols: int,
    d_real: int,
    d_block: int,
    generators: Sequence[int],
    h_value: int,
) -> Tuple[list[int], list[int]]:
    commitments, blindings = rust.commit_sign2_z_i64(
        contiguous_i64(x_base),
        contiguous_i64(s1),
        contiguous_i64(s2),
        int(virtual_rows),
        int(virtual_cols),
        int(d_real),
        int(d_block),
        list(generators),
        int(h_value),
    )
    return [int(v) for v in commitments], [int(v) for v in blindings]


def rust_fold_matrix_opening(
    rust,
    matrix: np.ndarray,
    blindings: Sequence[int],
    commitments: Sequence[int],
    row_point: Sequence[int],
    col_point: Sequence[int],
    virtual_rows: int,
    virtual_cols: int,
) -> tuple[list[int], int, int, int]:
    folded, folded_blinding, folded_commitment, value = rust.fold_matrix_opening_i64(
        contiguous_i64(matrix),
        list(map(int, blindings)),
        list(map(int, commitments)),
        list(map(int, row_point)),
        list(map(int, col_point)),
        int(virtual_rows),
        int(virtual_cols),
    )
    return [int(v) for v in folded], int(folded_blinding), int(folded_commitment), int(value)


def rust_fold_sign2_z_opening(
    rust,
    x_base: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    blindings: Sequence[int],
    commitments: Sequence[int],
    row_point: Sequence[int],
    col_point: Sequence[int],
    virtual_rows: int,
    virtual_cols: int,
    d_real: int,
    d_block: int,
) -> tuple[list[int], int, int, int]:
    folded, folded_blinding, folded_commitment, value = rust.fold_sign2_z_opening_i64(
        contiguous_i64(x_base),
        contiguous_i64(s1),
        contiguous_i64(s2),
        list(map(int, blindings)),
        list(map(int, commitments)),
        list(map(int, row_point)),
        list(map(int, col_point)),
        int(virtual_rows),
        int(virtual_cols),
        int(d_real),
        int(d_block),
    )
    return [int(v) for v in folded], int(folded_blinding), int(folded_commitment), int(value)


def rust_product_sumcheck_flat_matrices(
    rust,
    left_matrix: np.ndarray,
    right_matrix: np.ndarray,
    transcript: object,
    label: str,
) -> tuple[int, list[tuple[int, int, int]], int, int, list[int]]:
    claim, rounds, terminal_left, terminal_right, challenges = rust.product_sumcheck_flat_matrices_i64(
        contiguous_i64(left_matrix),
        contiguous_i64(right_matrix),
        transcript,
        str(label),
    )
    return (
        int(claim),
        [(int(g0), int(g1), int(g2)) for g0, g1, g2 in rounds],
        int(terminal_left),
        int(terminal_right),
        [int(challenge) for challenge in challenges],
    )
