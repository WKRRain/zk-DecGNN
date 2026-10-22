#!/usr/bin/env python3
"""Full-dataset MUTAG SIGN(3,0,0) Spartan-ZENO ZK benchmark launcher.

Purpose:
  Run a zero-knowledge correctness and integrity benchmark for MUTAG graph
  classification. The actual GNN is SIGN(3,0,0), while the physical proving
  container, VK capacity, D_max, and H_max are SIGN(3,0,0)_MAX in global-max
  mode. This measures SIGN inside its matching global maximum container, not a
  compact MUTAG-specific VK shape.

Dataset-run policy:
  One benchmark run means one complete MUTAG test-split experiment. Since the
  graph-classification proof has batchsize=1, each graph gets its own
  proof/verify flow. The launcher defaults to six complete dataset runs:
  run_id=0 is a full-test-split warm-up and is excluded from aggregate
  statistics; run_id=1..5 are five measured full-test-split experiments.

Relation semantics:
  - dataset=MUTAG, task=graph_classification, split_ratio=8:1:1
  - actual_model=SIGN, actual_K=3, actual architecture=SIGN(3,0,0)
  - shape_profile=SIGN(3,0,0)_MAX, container_mode=global_max
  - use_per_vk_compact_shape=false
  - readout=MeanPool with denominator N_real
  - padding nodes are excluded by node mask m_i
  - softmax is not checked in-circuit; the proof verifies argmax(Y_G)
  - Y_G, H, g, and model weights are private witnesses
  - graph topology, mask, padding, sparse row/col/val, commitments, and SPARK
    memory roots are bound into the verification transcript
  - optimized Rust route: graph_classification_prove, canonical binary proof
    serialization, and graph_classification_verify_serialized

SIGN(3,0,0) formula profile:
  alpha_pre=0, lambda_1=1, lambda_2=0, gamma_1=1, gamma_2=1,
  gamma_3=0, aggregation=Concat, beta=1, W1=Theta_block, W2=Omega,
  readout=MeanPool, graph_logits=Y_G.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import run_mutag_sgc_k2_sign300max_full_dataset_optimized_rust as base


def sign_print(*args: Any, **kwargs: Any) -> None:
    if args == ("actual_model=SIGN, actual_K=3 (not SIGN)",):
        print("actual_model=SIGN, actual_K=3, architecture=SIGN(3,0,0)")
        return
    print(*args, **kwargs)


def sign_fixed_point_embedding(pre: Any, d: int, d_max: int, k: int, scale: int) -> np.ndarray:
    x_base = np.zeros((int(pre.n_max), d_max), dtype=np.int64)
    x_base[:, :d] = np.asarray(pre.x, dtype=np.int64)
    rows = np.asarray(pre.edge_rows, dtype=np.int64)
    cols = np.asarray(pre.edge_cols, dtype=np.int64)
    vals = np.asarray(pre.edge_vals, dtype=np.int64)

    branches = [x_base.copy()]
    cur = x_base.copy()
    for _ in range(k):
        raw = np.zeros_like(cur)
        for row, col, val in zip(rows, cols, vals):
            if int(val) == 0:
                continue
            raw[int(row), :d] += int(val) * cur[int(col), :d]
        nxt = np.zeros_like(cur)
        for i in range(int(pre.n_max)):
            for j in range(d):
                nxt[i, j] = base.div_euclid_i128(int(raw[i, j]), int(scale))[0]
        cur = nxt
        branches.append(cur.copy())

    concat = np.zeros((int(pre.n_max), d_max), dtype=np.int64)
    for block, branch in enumerate(branches):
        start = block * d
        stop = min(start + d, d_max)
        if start >= d_max:
            break
        concat[:, start:stop] = branch[:, : stop - start]

    mask = np.asarray(pre.mask, dtype=np.int64)
    pooled = np.zeros((d_max,), dtype=np.int64)
    active_cols = min((k + 1) * d, d_max)
    for j in range(active_cols):
        total = int(np.dot(mask, concat[:, j]))
        pooled[j] = base.div_euclid_i128(total, int(pre.n_real))[0]
    return pooled


def apply_sign_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    row["actual_model"] = "SIGN"
    row["actual_K"] = 3
    row["actual_architecture"] = "SIGN(3,0,0)"
    row["alpha_pre"] = 0
    row["lambda_1"] = 1
    row["lambda_2"] = 0
    row["gamma_1"] = 1
    row["gamma_2"] = 1
    row["gamma_3"] = 0
    row["aggregation_operator"] = "Concat"
    row["beta"] = 1
    row["W1"] = "Theta_block"
    row["W2"] = "Omega"
    row["graph_logits"] = "Y_G"
    row["sign_formula_profile"] = "SIGN(3,0,0): Concat(X, AX, A^2X, A^3X)"
    return row


def apply_sign_purpose(purpose: Dict[str, Any]) -> Dict[str, Any]:
    purpose["purpose"] = (
        "Six-run complete-test-split Spartan-ZENO ZK correctness and integrity "
        "benchmark for MUTAG graph classification with actual model SIGN(3,0,0)."
    )
    purpose["actual_model"] = "SIGN"
    purpose["actual_K"] = 3
    purpose["actual_architecture"] = "SIGN(3,0,0)"
    purpose["sign_formula_profile"] = {
        "alpha_pre": 0,
        "lambda_1": 1,
        "lambda_2": 0,
        "gamma_1": 1,
        "gamma_2": 1,
        "gamma_3": 0,
        "aggregation_operator": "Concat",
        "beta": 1,
        "W1": "Theta_block",
        "W2": "Omega",
        "readout": "MeanPool",
        "graph_logits": "Y_G",
    }
    return purpose


_original_write_progress_marker = base.write_progress_marker
_original_save_results = base.save_results


def sign_write_progress_marker(path: Path, row: Dict[str, Any]) -> None:
    _original_write_progress_marker(path, apply_sign_metadata(row))


def sign_save_results(
    per_graph_rows: List[Dict[str, Any]],
    dataset_rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    purpose: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Path]:
    for row in per_graph_rows:
        apply_sign_metadata(row)
    for row in dataset_rows:
        row["actual_model"] = "SIGN"
        row["actual_K"] = 3
        row["actual_architecture"] = "SIGN(3,0,0)"
    apply_sign_purpose(purpose)
    apply_sign_purpose(summary)
    return _original_save_results(per_graph_rows, dataset_rows, summary, purpose, output_dir)


base.ACTUAL_MODEL = "SIGN"
base.ACTUAL_K = 3
base.RESULT_PREFIX = "mutag_sign300_under_sign300max_graph_classification"
base.DEFAULT_CHECKPOINT = (
    base.REPO_ROOT
    / "benchmarks"
    / "artifacts"
    / "mutag_sign300_under_sign300max_graph_weights.npz"
)
base.fixed_point_embedding = sign_fixed_point_embedding
base.write_progress_marker = sign_write_progress_marker
base.save_results = sign_save_results
base.print = sign_print


if __name__ == "__main__":
    base.main()
