#!/usr/bin/env python3
"""Full-dataset MUTAG APPNP(K=10) Spartan-ZENO ZK benchmark launcher.

Purpose:
  Run a zero-knowledge correctness and integrity benchmark for MUTAG graph
  classification. The actual GNN is APPNP(K=10), while the physical proving
  container, VK capacity, D_max, and H_max are SIGN(3,0,0)_MAX in global-max
  mode. This measures APPNP inside the larger SIGN container, not a compact
  MUTAG-specific VK shape.

Dataset-run policy:
  One benchmark run means one complete MUTAG test-split experiment. Since the
  graph-classification proof has batchsize=1, each graph gets its own
  proof/verify flow. The launcher defaults to six complete dataset runs:
  run_id=0 is a full-test-split warm-up and is excluded from aggregate
  statistics; run_id=1..5 are five measured full-test-split experiments.

Relation semantics:
  - dataset=MUTAG, task=graph_classification, split_ratio=8:1:1
  - actual_model=APPNP, actual_K=10, not SIGN
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

APPNP(K=10) formula profile used by this Rust graph-classification relation:
  alpha_pre=1, lambda_1=1-alpha, lambda_2=alpha, gamma_1=0, gamma_2=1,
  gamma_3=0, aggregation=+, beta=0, W1=I, W2=I, readout=MeanPool,
  graph_logits=Y_G. The optimized Rust relation fixes alpha to floor(scale/10)
  in fixed-point arithmetic, so with scale=256 the effective alpha is 25/256.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import run_mutag_sgc_k2_sign300max_full_dataset_optimized_rust as base


APPNP_K = 10
APPNP_ALPHA_NUMERATOR = base.S_SCALE // 10
APPNP_ALPHA = APPNP_ALPHA_NUMERATOR / float(base.S_SCALE)


def appnp_print(*args: Any, **kwargs: Any) -> None:
    if args == ("actual_model=APPNP, actual_K=10 (not SIGN)",):
        print(
            "actual_model=APPNP, actual_K=10, "
            f"alpha={APPNP_ALPHA:.8f} ({APPNP_ALPHA_NUMERATOR}/{base.S_SCALE}), not SIGN"
        )
        return
    print(*args, **kwargs)


def appnp_identity_aux_weights(d: int, h: int, d_max: int, scale: int):
    w_pre0 = np.zeros((d, h), dtype=np.int64)
    w_pre1 = np.zeros((h, d_max), dtype=np.int64)
    for idx in range(min(d, h)):
        w_pre0[idx, idx] = scale
    for idx in range(min(h, d_max, d)):
        w_pre1[idx, idx] = scale
    return w_pre0, w_pre1


def appnp_fixed_point_embedding(pre: Any, d: int, d_max: int, k: int, scale: int) -> np.ndarray:
    base_h = np.zeros((int(pre.n_max), d_max), dtype=np.int64)
    base_h[:, :d] = np.maximum(np.asarray(pre.x, dtype=np.int64), 0)
    rows = np.asarray(pre.edge_rows, dtype=np.int64)
    cols = np.asarray(pre.edge_cols, dtype=np.int64)
    vals = np.asarray(pre.edge_vals, dtype=np.int64)
    alpha = max(scale // 10, 1)
    keep = scale - alpha
    cur = base_h.copy()

    for _ in range(k):
        raw = np.zeros_like(cur)
        for row, col, val in zip(rows, cols, vals):
            if int(val) == 0:
                continue
            raw[int(row), :d] += int(val) * cur[int(col), :d]
        diffused = np.zeros_like(cur)
        for i in range(int(pre.n_max)):
            for j in range(d):
                diffused[i, j] = base.div_euclid_i128(int(raw[i, j]), int(scale))[0]
        nxt = np.zeros_like(cur)
        for i in range(int(pre.n_max)):
            for j in range(d):
                mixed = keep * int(diffused[i, j]) + alpha * int(base_h[i, j])
                nxt[i, j] = base.div_euclid_i128(mixed, int(scale))[0]
        cur = nxt

    mask = np.asarray(pre.mask, dtype=np.int64)
    pooled = np.zeros((d_max,), dtype=np.int64)
    for j in range(d):
        total = int(np.dot(mask, cur[:, j]))
        pooled[j] = base.div_euclid_i128(total, int(pre.n_real))[0]
    return pooled


def apply_appnp_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    row["actual_model"] = "APPNP"
    row["actual_K"] = APPNP_K
    row["alpha"] = APPNP_ALPHA
    row["alpha_fixed_point_numerator"] = APPNP_ALPHA_NUMERATOR
    row["alpha_pre"] = 1
    row["lambda_1"] = 1.0 - APPNP_ALPHA
    row["lambda_2"] = APPNP_ALPHA
    row["gamma_1"] = 0
    row["gamma_2"] = 1
    row["gamma_3"] = 0
    row["aggregation_operator"] = "+"
    row["beta"] = 0
    row["W1"] = "I"
    row["W2"] = "I"
    row["graph_logits"] = "Y_G"
    row["appnp_formula_profile"] = (
        f"APPNP(K=10), alpha={APPNP_ALPHA_NUMERATOR}/{base.S_SCALE}, "
        "alpha_pre=1, lambda_1=1-alpha, lambda_2=alpha"
    )
    return row


def apply_appnp_purpose(purpose: Dict[str, Any]) -> Dict[str, Any]:
    purpose["purpose"] = (
        "Six-run complete-test-split Spartan-ZENO ZK correctness and integrity "
        "benchmark for MUTAG graph classification with actual model APPNP(K=10)."
    )
    purpose["actual_model"] = "APPNP"
    purpose["actual_K"] = APPNP_K
    purpose["appnp_alpha"] = APPNP_ALPHA
    purpose["appnp_alpha_fixed_point_numerator"] = APPNP_ALPHA_NUMERATOR
    purpose["appnp_formula_profile"] = {
        "alpha_pre": 1,
        "lambda_1": 1.0 - APPNP_ALPHA,
        "lambda_2": APPNP_ALPHA,
        "gamma_1": 0,
        "gamma_2": 1,
        "gamma_3": 0,
        "aggregation_operator": "+",
        "beta": 0,
        "W1": "I",
        "W2": "I",
        "readout": "MeanPool",
        "graph_logits": "Y_G",
        "note": "Current optimized Rust graph-classification APPNP relation fixes alpha=floor(scale/10)/scale.",
    }
    return purpose


_original_write_progress_marker = base.write_progress_marker
_original_save_results = base.save_results


def appnp_write_progress_marker(path: Path, row: Dict[str, Any]) -> None:
    _original_write_progress_marker(path, apply_appnp_metadata(row))


def appnp_save_results(
    per_graph_rows: List[Dict[str, Any]],
    dataset_rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    purpose: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Path]:
    for row in per_graph_rows:
        apply_appnp_metadata(row)
    for row in dataset_rows:
        row["actual_model"] = "APPNP"
        row["actual_K"] = APPNP_K
    apply_appnp_purpose(purpose)
    apply_appnp_purpose(summary)
    return _original_save_results(per_graph_rows, dataset_rows, summary, purpose, output_dir)


base.ACTUAL_MODEL = "APPNP"
base.ACTUAL_K = APPNP_K
base.RESULT_PREFIX = "mutag_appnp_k10_under_sign300max_graph_classification"
base.DEFAULT_CHECKPOINT = (
    base.REPO_ROOT
    / "benchmarks"
    / "artifacts"
    / "mutag_appnp_k10_under_sign300max_graph_weights.npz"
)
base.identity_aux_weights = appnp_identity_aux_weights
base.fixed_point_embedding = appnp_fixed_point_embedding
base.write_progress_marker = appnp_write_progress_marker
base.save_results = appnp_save_results
base.print = appnp_print


if __name__ == "__main__":
    base.main()
