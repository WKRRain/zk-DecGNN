#!/usr/bin/env python3
"""Full-dataset NCI1 SGC(K=2) Spartan-ZENO ZK benchmark launcher.

Purpose:
  Run a zero-knowledge correctness and integrity benchmark for NCI1 graph
  classification. The actual GNN is SGC(K=2), while the physical proving
  container, VK capacity, D_max, and H_max are SIGN(3,0,0)_MAX in global-max
  mode. This measures SGC inside the larger SIGN container, not a compact
  NCI1-specific VK shape.

Dataset-run policy:
  One benchmark run means one complete NCI1 test-split experiment. Since the
  graph-classification proof has batchsize=1, each graph gets its own
  proof/verify flow. The launcher defaults to six complete dataset runs:
  run_id=0 is a full-test-split warm-up and is excluded from aggregate
  statistics; run_id=1..5 are five measured full-test-split experiments.

Relation semantics:
  - dataset=NCI1, task=graph_classification, split_ratio=8:1:1
  - actual_model=SGC, actual_K=2, not SIGN
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

SGC formula profile:
  alpha_pre=0, lambda_1=1, lambda_2=0, gamma_1=0, gamma_2=1,
  gamma_3=0, aggregation=+, beta=0, W1=Theta, W2=I,
  readout=MeanPool, graph_logits=Y_G.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import run_mutag_sgc_k2_sign300max_full_dataset_optimized_rust as base


NCI1_SCOPE = "complete_NCI1_test_split"


def nci1_print(*args: Any, **kwargs: Any) -> None:
    if args and isinstance(args[0], str):
        args = (args[0].replace("MUTAG", "NCI1").replace("complete_MUTAG_test_split", NCI1_SCOPE),) + args[1:]
    print(*args, **kwargs)


def apply_nci1_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    row["dataset"] = "NCI1"
    row["task"] = "graph_classification"
    row["complete_dataset_scope"] = NCI1_SCOPE
    row["actual_model"] = "SGC"
    row["actual_K"] = 2
    row["alpha_pre"] = 0
    row["lambda_1"] = 1
    row["lambda_2"] = 0
    row["gamma_1"] = 0
    row["gamma_2"] = 1
    row["gamma_3"] = 0
    row["aggregation_operator"] = "+"
    row["beta"] = 0
    row["W1"] = "Theta"
    row["W2"] = "I"
    row["graph_logits"] = "Y_G"
    row["sgc_formula_profile"] = "SGC(K=2): H=A_hat^2 X, readout=MeanPool, logits=Y_G"
    return row


def apply_nci1_purpose(purpose: Dict[str, Any]) -> Dict[str, Any]:
    purpose["purpose"] = (
        "Six-run complete-test-split Spartan-ZENO ZK correctness and integrity "
        "benchmark for NCI1 graph classification with actual model SGC(K=2)."
    )
    purpose["dataset"] = "NCI1"
    purpose["task"] = "graph_classification"
    purpose["complete_dataset_scope"] = NCI1_SCOPE
    purpose["actual_model"] = "SGC"
    purpose["actual_K"] = 2
    purpose["sgc_formula_profile"] = {
        "alpha_pre": 0,
        "lambda_1": 1,
        "lambda_2": 0,
        "gamma_1": 0,
        "gamma_2": 1,
        "gamma_3": 0,
        "aggregation_operator": "+",
        "beta": 0,
        "W1": "Theta",
        "W2": "I",
        "readout": "MeanPool",
        "graph_logits": "Y_G",
    }
    return purpose


_original_write_progress_marker = base.write_progress_marker
_original_save_results = base.save_results


def nci1_write_progress_marker(path: Path, row: Dict[str, Any]) -> None:
    _original_write_progress_marker(path, apply_nci1_metadata(row))


def nci1_save_results(
    per_graph_rows: List[Dict[str, Any]],
    dataset_rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    purpose: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Path]:
    for row in per_graph_rows:
        apply_nci1_metadata(row)
    for row in dataset_rows:
        row["dataset"] = "NCI1"
        row["task"] = "graph_classification"
        row["complete_dataset_scope"] = NCI1_SCOPE
        row["actual_model"] = "SGC"
        row["actual_K"] = 2
    apply_nci1_purpose(purpose)
    apply_nci1_purpose(summary)
    return _original_save_results(per_graph_rows, dataset_rows, summary, purpose, output_dir)


base.DATASET = "NCI1"
base.ACTUAL_MODEL = "SGC"
base.ACTUAL_K = 2
base.RESULT_PREFIX = "nci1_sgc_k2_under_sign300max_graph_classification"
base.DEFAULT_CHECKPOINT = (
    base.REPO_ROOT
    / "benchmarks"
    / "artifacts"
    / "nci1_sgc_k2_under_sign300max_graph_weights.npz"
)
base.write_progress_marker = nci1_write_progress_marker
base.save_results = nci1_save_results
base.print = nci1_print


if __name__ == "__main__":
    base.main()
