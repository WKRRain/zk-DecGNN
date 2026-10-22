#!/usr/bin/env python3
"""Full-dataset MUTAG SGC(K=2) Spartan-ZENO benchmark launcher.

Purpose:
  Run a zero-knowledge correctness and integrity benchmark for MUTAG graph
  classification.  The actual GNN is SGC(K=2), while the physical proving
  container, VK capacity, D_max, and H_max are SIGN(3,0,0)_MAX in global-max
  mode.  This intentionally measures SGC inside the larger SIGN container, not
  a compact MUTAG-specific VK shape.

Dataset-run policy:
  One benchmark run means one complete MUTAG test-split experiment.  Since the
  graph-classification proof has batchsize=1, each graph still gets its own
  proof/verify flow.  The launcher defaults to six complete dataset runs:
  run_id=0 is a full-test-split warm-up and is excluded from aggregate
  statistics; run_id=1..5 are five measured full-test-split experiments.

Relation semantics:
  - dataset=MUTAG, task=graph_classification, split_ratio=8:1:1
  - actual_model=SGC, actual_K=2, not SIGN
  - shape_profile=SIGN(3,0,0)_MAX, container_mode=global_max
  - use_per_vk_compact_shape=false
  - readout=MeanPool with denominator N_real
  - padding nodes are excluded by node mask m_i
  - softmax is not checked in-circuit; the proof verifies argmax(Y_G)
  - Y_G, H, g, and model weights are private witnesses
  - graph topology, mask, padding, sparse row/col/val, commitments, and SPARK
    memory roots are bound into the verification transcript
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = Path(__file__).resolve().parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

import rust_streaming_backend as rust_backend  # noqa: E402
from graph_dataset_loader import (  # noqa: E402
    S_SCALE,
    SPLIT_RATIO,
    load_graph_dataset,
    next_power_of_two,
    preprocess_graph,
    split_dataset,
)


DATASET = "MUTAG"
TASK = "graph_classification"
ACTUAL_MODEL = "SGC"
ACTUAL_K = 2
PHYSICAL_SIGN_K = 3
SHAPE_PROFILE = "SIGN(3,0,0)_MAX"
CONTAINER_MODE = "global_max"
USE_PER_VK_COMPACT_SHAPE = False
BATCHSIZE = 1
READOUT = "MeanPool"
POOLING_DENOMINATOR = "N_real"
RESULT_PREFIX = "mutag_sgc_k2_under_sign300max_graph_classification"

DEFAULT_DATA_ROOT = REPO_ROOT / "benchmarks" / "data" / "graph_classification"
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "benchmarks"
    / "artifacts"
    / "mutag_sgc_k2_under_sign300max_graph_weights.npz"
)

PER_GRAPH_FIELDS = [
    "run_id",
    "warmup",
    "proof_id",
    "graph_position",
    "dataset",
    "task",
    "split_ratio",
    "split_id",
    "complete_dataset_scope",
    "actual_model",
    "actual_K",
    "shape_profile",
    "container_mode",
    "use_per_vk_compact_shape",
    "graph_id",
    "batchsize",
    "N_real",
    "N_max",
    "d",
    "c",
    "D_max",
    "H_max",
    "num_edges",
    "num_edges_with_self_loops",
    "readout",
    "pooling_denominator",
    "exclude_padding_nodes",
    "node_mask",
    "do_not_verify_softmax",
    "verify_argmax_relation",
    "public_output",
    "private_output",
    "commit_time_ms",
    "commit_time_s",
    "prove_time_excluding_commit_ms",
    "prove_time_excluding_commit_s",
    "total_prover_time_including_commit_ms",
    "total_prover_time_including_commit_s",
    "proof_size_bytes",
    "proof_size_kb",
    "proof_size_mb",
    "verify_time_ms",
    "verify_time_s",
    "predicted_label",
    "true_label",
    "verification_passed",
    "checkpoint_loaded",
    "checkpoint_path",
    "training_time_ms_excluded",
    "weight_source",
    "alpha_pre",
    "lambda_1",
    "lambda_2",
    "gamma_1",
    "gamma_2",
    "gamma_3",
    "aggregation_operator",
    "beta",
    "W1",
    "W2",
    "graph_logits",
    "zk_correctness_verification",
    "memory_integrity_verification",
    "weights_private",
    "optimized_rust_proof_route",
]

DATASET_RUN_FIELDS = [
    "run_id",
    "warmup",
    "dataset",
    "task",
    "split_ratio",
    "split_id",
    "complete_dataset_scope",
    "actual_model",
    "actual_K",
    "shape_profile",
    "container_mode",
    "use_per_vk_compact_shape",
    "batchsize",
    "test_graph_count",
    "unique_graph_count",
    "graph_ids",
    "N_max",
    "d",
    "c",
    "D_max",
    "H_max",
    "readout",
    "pooling_denominator",
    "commit_time_ms",
    "commit_time_s",
    "prove_time_excluding_commit_ms",
    "prove_time_excluding_commit_s",
    "total_prover_time_including_commit_ms",
    "total_prover_time_including_commit_s",
    "proof_size_bytes",
    "proof_size_kb",
    "proof_size_mb",
    "verify_time_ms",
    "verify_time_s",
    "verification_passed",
    "correct_predictions",
    "accuracy",
]

AGGREGATE_METRICS = [
    "commit_time_ms",
    "prove_time_excluding_commit_ms",
    "total_prover_time_including_commit_ms",
    "proof_size_bytes",
    "proof_size_kb",
    "proof_size_mb",
    "verify_time_ms",
]

PROVER_PHASE_FIELDS = [
    "public_input_validation_time_ms",
    "witness_generation_time_ms",
    "public_binding_time_ms",
    "residual_constraint_time_ms",
    "witness_commitment_time_ms",
    "commitment_binding_time_ms",
    "spark_root_time_ms",
    "range_commitment_time_ms",
    "opening_generation_time_ms",
    "sumcheck_time_ms",
    "proof_digest_time_ms",
    "proof_assembly_time_ms",
    "total_backend_time_ms",
]

VERIFIER_PHASE_FIELDS = [
    "verify_public_input_time_ms",
    "verify_commitment_parse_time_ms",
    "verify_weight_binding_time_ms",
    "opening_parse_time_ms",
    "opening_verify_time_ms",
    "spark_verify_time_ms",
    "sumcheck_context_verify_time_ms",
    "sumcheck_verify_time_ms",
    "proof_digest_verify_time_ms",
    "verify_total_backend_time_ms",
]

PER_GRAPH_FIELDS.extend(
    [
        "public_preprocess_time_ms",
        "public_preprocess_cache_hit",
        "progress_marker_path",
        "resumed_from_marker",
        *PROVER_PHASE_FIELDS,
        *VERIFIER_PHASE_FIELDS,
    ]
)

DATASET_RUN_FIELDS.extend(
    [
        "public_preprocess_time_ms",
        "public_preprocess_cache_hits",
        "progress_markers_written",
        "resumed_graphs",
        "wall_clock_dataset_run_time_ms",
        "wall_clock_dataset_run_time_s",
        *PROVER_PHASE_FIELDS,
        *VERIFIER_PHASE_FIELDS,
    ]
)

AGGREGATE_METRICS.extend(
    [
        "public_preprocess_time_ms",
        "wall_clock_dataset_run_time_ms",
        *PROVER_PHASE_FIELDS,
        *VERIFIER_PHASE_FIELDS,
    ]
)

PER_GRAPH_AGGREGATE_METRICS = [
    metric for metric in AGGREGATE_METRICS if metric != "wall_clock_dataset_run_time_ms"
]


def now_ms() -> float:
    return time.perf_counter_ns() / 1_000_000.0


def div_euclid_i128(value: int, divisor: int) -> Tuple[int, int]:
    q = int(value) // int(divisor)
    r = int(value) % int(divisor)
    if r < 0:
        q -= 1
        r += int(divisor)
    return q, r


def format_cell(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def print_markdown_table(rows: Iterable[Dict[str, Any]], columns: List[str], title: str) -> None:
    print(f"\n{title}")
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        print("| " + " | ".join(format_cell(row.get(col, "")) for col in columns) + " |")


def json_safe(row: Dict[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, np.generic):
            safe[key] = value.item()
        else:
            safe[key] = value
    return safe


def timing_float(timing: Dict[str, Any], key: str) -> float:
    return float(timing.get(key, 0.0))


def phase_values(timing: Dict[str, Any], fields: List[str]) -> Dict[str, float]:
    return {field: timing_float(timing, field) for field in fields}


def progress_marker_path(progress_dir: Path, run_id: int, graph_position: int, graph_id: int) -> Path:
    return progress_dir / f"run_{run_id:03d}" / f"graph_{graph_position:03d}_id_{graph_id}.json"


def write_progress_marker(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(json_safe(row), indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def load_progress_marker(
    path: Path,
    run_id: int,
    graph_position: int,
    graph_id: int,
    split_id: str,
    dims: Dict[str, int],
) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "run_id": run_id,
        "graph_position": graph_position,
        "graph_id": graph_id,
        "split_id": split_id,
        "actual_model": ACTUAL_MODEL,
        "actual_K": ACTUAL_K,
        "shape_profile": SHAPE_PROFILE,
        "container_mode": CONTAINER_MODE,
        "task": TASK,
        "N_max": dims["N_max"],
        "d": dims["d"],
        "c": dims["c"],
        "D_max": dims["D_max"],
        "H_max": dims["H_max"],
    }
    for key, value in expected.items():
        if row.get(key) != value:
            return None
    if not bool(row.get("verification_passed", False)):
        return None
    row["resumed_from_marker"] = True
    return row


def build_preprocessed_test_cache(
    graphs: List[Any],
    test_indices: List[int],
    dims: Dict[str, int],
    scale: int,
) -> Tuple[List[Dict[str, Any]], float]:
    cache: List[Dict[str, Any]] = []
    cache_start = now_ms()
    for graph_position, dataset_index in enumerate(test_indices):
        graph = graphs[dataset_index]
        graph_start = now_ms()
        pre = preprocess_graph(graph, dims["N_max"], dims["d"], scale=scale)
        cache.append(
            {
                "graph_position": graph_position,
                "dataset_index": dataset_index,
                "graph": graph,
                "pre": pre,
                "public_preprocess_time_ms": now_ms() - graph_start,
            }
        )
    return cache, now_ms() - cache_start


def sign300max_dims(graphs: List[Any], n_max_arg: int | None, hidden_dim: int) -> Dict[str, int]:
    n_max = next_power_of_two(n_max_arg or max(int(graph.x.shape[0]) for graph in graphs))
    d = max(int(graph.x.shape[1]) for graph in graphs)
    c = max(int(graph.y) for graph in graphs) + 1
    h = next_power_of_two(max(1, int(hidden_dim)))
    d_max = next_power_of_two((PHYSICAL_SIGN_K + 1) * d)
    h_max = next_power_of_two(max(c, (PHYSICAL_SIGN_K + 1) * h))
    return {"N_max": n_max, "d": d, "c": c, "h": h, "D_max": d_max, "H_max": h_max}


def identity_aux_weights(d: int, h: int, d_max: int, scale: int) -> Tuple[np.ndarray, np.ndarray]:
    w_pre0 = np.zeros((d, h), dtype=np.int64)
    w_pre1 = np.zeros((h, d_max), dtype=np.int64)
    for idx in range(min(d, h)):
        w_pre0[idx, idx] = scale
    for idx in range(min(h, d_max, d)):
        w_pre1[idx, idx] = scale
    return w_pre0, w_pre1


def fixed_point_embedding(pre: Any, d: int, d_max: int, k: int, scale: int) -> np.ndarray:
    cur = np.zeros((int(pre.n_max), d_max), dtype=np.int64)
    cur[:, :d] = np.asarray(pre.x, dtype=np.int64)
    rows = np.asarray(pre.edge_rows, dtype=np.int64)
    cols = np.asarray(pre.edge_cols, dtype=np.int64)
    vals = np.asarray(pre.edge_vals, dtype=np.int64)
    for _ in range(k):
        raw = np.zeros_like(cur)
        for row, col, val in zip(rows, cols, vals):
            if int(val) == 0:
                continue
            raw[int(row), :] += int(val) * cur[int(col), :]
        nxt = np.zeros_like(cur)
        for idx, value in np.ndenumerate(raw):
            nxt[idx] = div_euclid_i128(int(value), int(scale))[0]
        cur = nxt

    mask = np.asarray(pre.mask, dtype=np.int64)
    pooled = np.zeros((d_max,), dtype=np.int64)
    for j in range(d_max):
        total = int(np.dot(mask, cur[:, j]))
        pooled[j] = div_euclid_i128(total, int(pre.n_real))[0]
    return pooled


def load_weight_checkpoint(path: Path, dims: Dict[str, int]) -> Dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        required = {"W_pre0", "W_pre1", "W_graph"}
        if not required.issubset(set(data.files)):
            return None
        weights = {key: np.ascontiguousarray(data[key], dtype=np.int64) for key in required}
    expected = {
        "W_pre0": (dims["d"], dims["h"]),
        "W_pre1": (dims["h"], dims["D_max"]),
        "W_graph": (dims["D_max"], dims["c"]),
    }
    for key, shape in expected.items():
        if weights[key].shape != shape:
            return None
        if np.any(weights[key] < -32768) or np.any(weights[key] > 32767):
            return None
    return weights


def train_and_export_checkpoint(
    path: Path,
    graphs: List[Any],
    split: Dict[str, Any],
    dims: Dict[str, int],
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    scale: int,
) -> Tuple[Dict[str, np.ndarray], float]:
    try:
        import torch
        from torch.nn import functional as F
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required to train the MUTAG SGC checkpoint") from exc

    start = now_ms()
    torch.manual_seed(seed)
    features = []
    labels = []
    for dataset_index in [int(idx) for idx in split["train"]]:
        pre = preprocess_graph(graphs[dataset_index], dims["N_max"], dims["d"], scale=scale)
        features.append(fixed_point_embedding(pre, dims["d"], dims["D_max"], ACTUAL_K, scale))
        labels.append(int(pre.y_true))

    x_train = torch.tensor(np.stack(features).astype(np.float32) / float(scale), dtype=torch.float32)
    y_train = torch.tensor(labels, dtype=torch.long)
    w = torch.zeros((dims["D_max"], dims["c"]), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=float(lr), weight_decay=float(weight_decay))
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        loss = F.cross_entropy(x_train @ w, y_train)
        loss.backward()
        optimizer.step()

    w_graph = np.rint(w.detach().cpu().numpy() * float(scale)).astype(np.int64)
    w_graph = np.clip(w_graph, -32768, 32767).astype(np.int64)
    w_pre0, w_pre1 = identity_aux_weights(dims["d"], dims["h"], dims["D_max"], scale)
    weights = {"W_pre0": w_pre0, "W_pre1": w_pre1, "W_graph": w_graph}
    training_ms = now_ms() - start

    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset": DATASET,
        "task": TASK,
        "actual_model": ACTUAL_MODEL,
        "actual_K": ACTUAL_K,
        "shape_profile": SHAPE_PROFILE,
        "container_mode": CONTAINER_MODE,
        "split_id": split["split_id"],
        "training_time_ms_excluded": training_ms,
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        **dims,
    }
    np.savez_compressed(path, **weights, meta_json=np.array(json.dumps(meta, sort_keys=True)))
    return weights, training_ms


def ensure_weights(
    checkpoint: Path,
    graphs: List[Any],
    split: Dict[str, Any],
    dims: Dict[str, int],
    args: argparse.Namespace,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if not args.force_retrain:
        loaded = load_weight_checkpoint(checkpoint, dims)
        if loaded is not None:
            return loaded, {
                "checkpoint_loaded": True,
                "checkpoint_path": str(checkpoint),
                "training_time_ms_excluded": 0.0,
                "weight_source": "trained_npz_checkpoint",
            }

    weights, training_ms = train_and_export_checkpoint(
        checkpoint,
        graphs,
        split,
        dims,
        args.train_epochs,
        args.train_lr,
        args.train_weight_decay,
        args.seed,
        args.scale,
    )
    return weights, {
        "checkpoint_loaded": False,
        "checkpoint_path": str(checkpoint),
        "training_time_ms_excluded": training_ms,
        "weight_source": "trained_npz_checkpoint_created_for_benchmark",
    }


def aggregate(rows: List[Dict[str, Any]], metrics: List[str]) -> Dict[str, Dict[str, float]]:
    measured = [row for row in rows if not row["warmup"]]
    if len(measured) < 5:
        raise ValueError("at least five measured full-dataset runs are required")
    return {
        metric: {
            "mean": statistics.fmean(float(row[metric]) for row in measured),
            "std": statistics.stdev(float(row[metric]) for row in measured),
            "min": min(float(row[metric]) for row in measured),
            "max": max(float(row[metric]) for row in measured),
        }
        for metric in metrics
    }


def build_dataset_run_row(
    run_id: int,
    warmup: bool,
    proof_rows: List[Dict[str, Any]],
    split: Dict[str, Any],
    dims: Dict[str, int],
    wall_clock_ms: float,
) -> Dict[str, Any]:
    correct = sum(int(row["predicted_label"] == row["true_label"]) for row in proof_rows)
    graph_ids = [str(row["graph_id"]) for row in proof_rows]
    row = {
        "run_id": run_id,
        "warmup": warmup,
        "dataset": DATASET,
        "task": TASK,
        "split_ratio": SPLIT_RATIO,
        "split_id": split["split_id"],
        "complete_dataset_scope": "complete_MUTAG_test_split",
        "actual_model": ACTUAL_MODEL,
        "actual_K": ACTUAL_K,
        "shape_profile": SHAPE_PROFILE,
        "container_mode": CONTAINER_MODE,
        "use_per_vk_compact_shape": USE_PER_VK_COMPACT_SHAPE,
        "batchsize": BATCHSIZE,
        "test_graph_count": len(proof_rows),
        "unique_graph_count": len(set(graph_ids)),
        "graph_ids": ",".join(graph_ids),
        "N_max": dims["N_max"],
        "d": dims["d"],
        "c": dims["c"],
        "D_max": dims["D_max"],
        "H_max": dims["H_max"],
        "readout": READOUT,
        "pooling_denominator": POOLING_DENOMINATOR,
        "commit_time_ms": sum(float(row["commit_time_ms"]) for row in proof_rows),
        "prove_time_excluding_commit_ms": sum(float(row["prove_time_excluding_commit_ms"]) for row in proof_rows),
        "total_prover_time_including_commit_ms": sum(
            float(row["total_prover_time_including_commit_ms"]) for row in proof_rows
        ),
        "proof_size_bytes": sum(int(row["proof_size_bytes"]) for row in proof_rows),
        "verify_time_ms": sum(float(row["verify_time_ms"]) for row in proof_rows),
        "public_preprocess_time_ms": sum(float(row.get("public_preprocess_time_ms", 0.0)) for row in proof_rows),
        "public_preprocess_cache_hits": sum(int(bool(row.get("public_preprocess_cache_hit", False))) for row in proof_rows),
        "progress_markers_written": sum(int(bool(row.get("progress_marker_path", ""))) for row in proof_rows),
        "resumed_graphs": sum(int(bool(row.get("resumed_from_marker", False))) for row in proof_rows),
        "wall_clock_dataset_run_time_ms": float(wall_clock_ms),
        "wall_clock_dataset_run_time_s": float(wall_clock_ms) / 1000.0,
        "verification_passed": all(bool(row["verification_passed"]) for row in proof_rows),
        "correct_predictions": correct,
        "accuracy": correct / len(proof_rows),
    }
    for field in PROVER_PHASE_FIELDS + VERIFIER_PHASE_FIELDS:
        row[field] = sum(float(proof_row.get(field, 0.0)) for proof_row in proof_rows)
    row["commit_time_s"] = row["commit_time_ms"] / 1000.0
    row["prove_time_excluding_commit_s"] = row["prove_time_excluding_commit_ms"] / 1000.0
    row["total_prover_time_including_commit_s"] = row["total_prover_time_including_commit_ms"] / 1000.0
    row["proof_size_kb"] = row["proof_size_bytes"] / 1024.0
    row["proof_size_mb"] = row["proof_size_bytes"] / (1024.0 * 1024.0)
    row["verify_time_s"] = row["verify_time_ms"] / 1000.0
    return row


def save_results(
    per_graph_rows: List[Dict[str, Any]],
    dataset_rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    purpose: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_graph_json = output_dir / f"{RESULT_PREFIX}_per_graph_raw.json"
    per_graph_csv = output_dir / f"{RESULT_PREFIX}_per_graph_raw.csv"
    dataset_json = output_dir / f"{RESULT_PREFIX}_dataset_runs_raw.json"
    dataset_csv = output_dir / f"{RESULT_PREFIX}_dataset_runs_raw.csv"
    summary_json = output_dir / f"{RESULT_PREFIX}_summary.json"
    purpose_json = output_dir / f"{RESULT_PREFIX}_purpose.json"

    per_graph_json.write_text(
        json.dumps([json_safe(row) for row in per_graph_rows], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    dataset_json.write_text(
        json.dumps([json_safe(row) for row in dataset_rows], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    purpose_json.write_text(json.dumps(purpose, indent=2, sort_keys=True), encoding="utf-8")

    with per_graph_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_GRAPH_FIELDS)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in PER_GRAPH_FIELDS} for row in per_graph_rows])
    with dataset_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_RUN_FIELDS)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in DATASET_RUN_FIELDS} for row in dataset_rows])

    return {
        "per_graph_json": per_graph_json,
        "per_graph_csv": per_graph_csv,
        "dataset_runs_json": dataset_json,
        "dataset_runs_csv": dataset_csv,
        "summary_json": summary_json,
        "purpose_json": purpose_json,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-runs", type=int, default=6, help="6 means 1 full-dataset warm-up + 5 measured full-dataset runs")
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--n-max", type=int, default=None)
    parser.add_argument("--scale", type=int, default=S_SCALE)
    parser.add_argument("--arg-bits", type=int, default=64)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train-epochs", type=int, default=300)
    parser.add_argument("--train-lr", type=float, default=0.05)
    parser.add_argument("--train-weight-decay", type=float, default=5e-4)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--resume", action="store_true", help="reuse verified per-graph progress markers when present")
    parser.add_argument("--force-rerun", action="store_true", help="ignore existing progress markers even with --resume")
    parser.add_argument("--progress-dir", type=Path, default=None, help="directory for per-graph resume markers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.dataset_runs) < 6:
        raise ValueError("--dataset-runs must be at least 6: one warm-up plus five measured full-dataset runs")
    if int(args.scale) != S_SCALE:
        raise ValueError("this graph-classification Rust relation uses fixed S_scale=256")

    output_dir = Path(args.output_dir).resolve()
    progress_dir = (
        Path(args.progress_dir).resolve()
        if args.progress_dir is not None
        else output_dir / f"{RESULT_PREFIX}_progress"
    )
    graphs = load_graph_dataset(DATASET, Path(args.data_root).resolve())
    split = split_dataset(graphs, DATASET, int(args.seed), output_dir)
    test_indices = [int(idx) for idx in split["test"]]
    if len(test_indices) < 5:
        raise RuntimeError("complete MUTAG test split must contain at least five graphs")

    dims = sign300max_dims(graphs, args.n_max, args.hidden_dim)
    weights, weight_meta = ensure_weights(Path(args.checkpoint).resolve(), graphs, split, dims, args)
    preprocessed_test_cache, preprocess_cache_build_time_ms = build_preprocessed_test_cache(
        graphs,
        test_indices,
        dims,
        int(args.scale),
    )
    rust = rust_backend.load(build=not bool(args.skip_build))
    split_counts = split["counts"]

    purpose = {
        "purpose": "Six-run complete-test-split Spartan-ZENO ZK correctness and integrity benchmark for MUTAG graph classification.",
        "dataset": DATASET,
        "task": TASK,
        "split_ratio": SPLIT_RATIO,
        "split_id": split["split_id"],
        "split_counts": split_counts,
        "complete_dataset_scope": "complete_MUTAG_test_split",
        "dataset_runs_total": int(args.dataset_runs),
        "dataset_runs_measured": int(args.dataset_runs) - 1,
        "warmup_policy": "run_id=0 is one complete MUTAG test-split experiment and is excluded from aggregates",
        "batchsize": BATCHSIZE,
        "actual_model": ACTUAL_MODEL,
        "actual_K": ACTUAL_K,
        "shape_profile": SHAPE_PROFILE,
        "container_mode": CONTAINER_MODE,
        "use_per_vk_compact_shape": USE_PER_VK_COMPACT_SHAPE,
        "readout": READOUT,
        "pooling_denominator": POOLING_DENOMINATOR,
        "exclude_padding_nodes": True,
        "do_not_verify_softmax": True,
        "verify_argmax_relation": True,
        "public_output": "predicted_label ell",
        "private_output": "graph_level_logits Y_G",
        "optimized_rust_proof_route": True,
        "public_preprocess_cached": True,
        "preprocess_cache_build_time_ms": preprocess_cache_build_time_ms,
        "progress_dir": str(progress_dir),
        "resume_enabled": bool(args.resume),
        "dims": dims,
        **weight_meta,
    }

    print("Spartan-ZENO MUTAG graph-classification full-dataset benchmark")
    print(f"dataset={DATASET}, task={TASK}, split_ratio={SPLIT_RATIO}, batchsize={BATCHSIZE}")
    print(
        "split_counts="
        f"total={split_counts['total']}, train={split_counts['train']}, "
        f"validation={split_counts['val']}, test={split_counts['test']}"
    )
    print(f"complete_dataset_scope=complete_MUTAG_test_split ({len(test_indices)} graphs per dataset run)")
    print(f"dataset_runs={int(args.dataset_runs)} (run_id=0 warm-up, run_id=1..{int(args.dataset_runs)-1} measured)")
    print(f"actual_model={ACTUAL_MODEL}, actual_K={ACTUAL_K} (not SIGN)")
    print(f"shape_profile={SHAPE_PROFILE}, container_mode={CONTAINER_MODE}, global max shape")
    print(f"use_per_vk_compact_shape={USE_PER_VK_COMPACT_SHAPE}")
    print(
        "dims="
        f"N_max={dims['N_max']}, d={dims['d']}, c={dims['c']}, "
        f"D_max={dims['D_max']}, H_max={dims['H_max']}"
    )
    print(f"checkpoint_loaded={weight_meta['checkpoint_loaded']}, checkpoint_path={weight_meta['checkpoint_path']}")
    print(f"training_time_ms_excluded={weight_meta['training_time_ms_excluded']:.3f}")
    print(f"public_preprocess_cache_build_time_ms={preprocess_cache_build_time_ms:.3f}")
    print(f"progress_dir={progress_dir}")
    print(
        "proof_backend=optimized Rust release "
        "graph_classification_prove + canonical binary serialization + "
        "graph_classification_verify_serialized"
    )

    expected_graph_ids = {int(graphs[idx].graph_id) for idx in test_indices}
    per_graph_rows: List[Dict[str, Any]] = []
    dataset_rows: List[Dict[str, Any]] = []

    for dataset_run_id in range(int(args.dataset_runs)):
        warmup = dataset_run_id == 0
        dataset_run_start = now_ms()
        print(f"\nFull-dataset run {dataset_run_id}/{int(args.dataset_runs) - 1}" + (" (warm-up)" if warmup else ""))
        run_rows: List[Dict[str, Any]] = []
        for cached in preprocessed_test_cache:
            graph_position = int(cached["graph_position"])
            graph = cached["graph"]
            pre = cached["pre"]
            proof_id = dataset_run_id * len(preprocessed_test_cache) + graph_position
            marker_path = progress_marker_path(progress_dir, dataset_run_id, graph_position, int(pre.graph_id))

            if bool(args.resume) and not bool(args.force_rerun):
                resumed = load_progress_marker(
                    marker_path,
                    dataset_run_id,
                    graph_position,
                    int(pre.graph_id),
                    split["split_id"],
                    dims,
                )
                if resumed is not None:
                    print(
                        f"  graph {graph_position + 1}/{len(preprocessed_test_cache)} "
                        f"graph_id={pre.graph_id} resumed marker={marker_path}",
                        flush=True,
                    )
                    per_graph_rows.append(resumed)
                    run_rows.append(resumed)
                    continue

            print(
                f"  graph {graph_position + 1}/{len(preprocessed_test_cache)} "
                f"graph_id={pre.graph_id} proof_id={proof_id}",
                flush=True,
            )

            proof = rust.graph_classification_prove(
                ACTUAL_MODEL,
                split["split_id"],
                int(pre.graph_id),
                pre.x,
                pre.a_hat,
                pre.mask,
                pre.edge_rows,
                pre.edge_cols,
                pre.edge_vals,
                weights["W_pre0"],
                weights["W_pre1"],
                weights["W_graph"],
                int(pre.n_real),
                int(dims["N_max"]),
                int(dims["d"]),
                int(dims["c"]),
                int(dims["h"]),
                int(ACTUAL_K),
                int(dims["D_max"]),
                int(dims["H_max"]),
                int(args.scale),
                int(args.arg_bits),
            )
            proof_bytes = rust.graph_classification_serialize_proof(proof)
            proof_size_bytes = len(proof_bytes)

            verify_start = now_ms()
            verify_fn = getattr(rust, "graph_classification_verify_serialized", None)
            if verify_fn is None:
                verdict = rust.graph_classification_verify(
                    proof,
                    ACTUAL_MODEL,
                    split["split_id"],
                    int(pre.graph_id),
                    pre.x,
                    pre.a_hat,
                    pre.mask,
                    pre.edge_rows,
                    pre.edge_cols,
                    pre.edge_vals,
                    int(pre.n_real),
                    int(dims["N_max"]),
                    int(dims["d"]),
                    int(dims["c"]),
                    int(dims["h"]),
                    int(ACTUAL_K),
                    int(dims["D_max"]),
                    int(dims["H_max"]),
                )
            else:
                verdict = verify_fn(
                    proof_bytes,
                    ACTUAL_MODEL,
                    split["split_id"],
                    int(pre.graph_id),
                    pre.x,
                    pre.a_hat,
                    pre.mask,
                    pre.edge_rows,
                    pre.edge_cols,
                    pre.edge_vals,
                    int(pre.n_real),
                    int(dims["N_max"]),
                    int(dims["d"]),
                    int(dims["c"]),
                    int(dims["h"]),
                    int(ACTUAL_K),
                    int(dims["D_max"]),
                    int(dims["H_max"]),
                )
            verify_time_ms = now_ms() - verify_start
            if not bool(verdict["accepted"]):
                raise RuntimeError(f"verification failed for graph_id={pre.graph_id}: {verdict['reason']}")

            timing = proof["timing"]
            verifier_timing = verdict.get("timing", {})
            commit_ms = float(timing["commit_time_ms"])
            prove_ex_commit_ms = float(timing["prove_time_excluding_commit_ms"])
            total_ms = commit_ms + prove_ex_commit_ms
            row = {
                "run_id": dataset_run_id,
                "warmup": warmup,
                "proof_id": proof_id,
                "graph_position": graph_position,
                "dataset": DATASET,
                "task": TASK,
                "split_ratio": SPLIT_RATIO,
                "split_id": split["split_id"],
                "complete_dataset_scope": "complete_MUTAG_test_split",
                "actual_model": ACTUAL_MODEL,
                "actual_K": ACTUAL_K,
                "shape_profile": SHAPE_PROFILE,
                "container_mode": CONTAINER_MODE,
                "use_per_vk_compact_shape": USE_PER_VK_COMPACT_SHAPE,
                "graph_id": int(pre.graph_id),
                "batchsize": BATCHSIZE,
                "N_real": int(pre.n_real),
                "N_max": int(dims["N_max"]),
                "d": int(dims["d"]),
                "c": int(dims["c"]),
                "D_max": int(dims["D_max"]),
                "H_max": int(dims["H_max"]),
                "num_edges": int(graph.edge_index.shape[1]),
                "num_edges_with_self_loops": int(pre.sparse_nnz),
                "readout": READOUT,
                "pooling_denominator": POOLING_DENOMINATOR,
                "exclude_padding_nodes": True,
                "node_mask": "m_i",
                "do_not_verify_softmax": True,
                "verify_argmax_relation": True,
                "public_output": "predicted_label ell",
                "private_output": "graph_level_logits Y_G",
                "commit_time_ms": commit_ms,
                "commit_time_s": commit_ms / 1000.0,
                "prove_time_excluding_commit_ms": prove_ex_commit_ms,
                "prove_time_excluding_commit_s": prove_ex_commit_ms / 1000.0,
                "total_prover_time_including_commit_ms": total_ms,
                "total_prover_time_including_commit_s": total_ms / 1000.0,
                "proof_size_bytes": proof_size_bytes,
                "proof_size_kb": proof_size_bytes / 1024.0,
                "proof_size_mb": proof_size_bytes / (1024.0 * 1024.0),
                "verify_time_ms": verify_time_ms,
                "verify_time_s": verify_time_ms / 1000.0,
                "predicted_label": int(verdict["ell"]),
                "true_label": int(pre.y_true),
                "verification_passed": bool(verdict["accepted"]),
                "checkpoint_loaded": bool(weight_meta["checkpoint_loaded"]),
                "checkpoint_path": str(weight_meta["checkpoint_path"]),
                "training_time_ms_excluded": float(weight_meta["training_time_ms_excluded"]),
                "weight_source": str(weight_meta["weight_source"]),
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
                "graph_logits": "Y_G",
                "zk_correctness_verification": True,
                "memory_integrity_verification": True,
                "weights_private": True,
                "optimized_rust_proof_route": True,
                "public_preprocess_time_ms": float(cached["public_preprocess_time_ms"]),
                "public_preprocess_cache_hit": True,
                "progress_marker_path": str(marker_path),
                "resumed_from_marker": False,
                **phase_values(timing, PROVER_PHASE_FIELDS),
                **phase_values(verifier_timing, VERIFIER_PHASE_FIELDS),
            }
            write_progress_marker(marker_path, row)
            per_graph_rows.append(row)
            run_rows.append(row)

        measured_graph_ids = {int(row["graph_id"]) for row in run_rows}
        if measured_graph_ids != expected_graph_ids:
            raise RuntimeError(f"run_id={dataset_run_id} does not cover the complete MUTAG test split")
        dataset_row = build_dataset_run_row(
            dataset_run_id,
            warmup,
            run_rows,
            split,
            dims,
            now_ms() - dataset_run_start,
        )
        dataset_rows.append(dataset_row)
        print(
            f"  graphs={dataset_row['test_graph_count']}, "
            f"commit_total={dataset_row['commit_time_ms']:.3f} ms, "
            f"prove_ex_commit_total={dataset_row['prove_time_excluding_commit_ms']:.3f} ms, "
            f"verify_total={dataset_row['verify_time_ms']:.3f} ms, "
            f"proof_total={dataset_row['proof_size_bytes']} bytes, "
            f"accuracy={dataset_row['accuracy']:.3f}, passed=True"
        )

    if not all(bool(row["verification_passed"]) for row in per_graph_rows):
        raise RuntimeError("at least one proof verification failed")
    if not all(bool(row["verification_passed"]) for row in dataset_rows):
        raise RuntimeError("at least one complete dataset run failed")

    dataset_aggregate = aggregate(dataset_rows, AGGREGATE_METRICS)
    per_graph_aggregate = aggregate(per_graph_rows, PER_GRAPH_AGGREGATE_METRICS)
    summary_payload: Dict[str, Any] = {
        "dataset_run_aggregate": dataset_aggregate,
        "per_graph_aggregate": per_graph_aggregate,
        "aggregate_primary_unit": "complete_MUTAG_test_split_run",
        "warmup_policy": "run_id=0 full test split excluded from aggregate statistics",
        "dataset_runs_total": len(dataset_rows),
        "dataset_runs_measured": len([row for row in dataset_rows if not row["warmup"]]),
        "per_graph_proofs_total": len(per_graph_rows),
        "per_graph_proofs_measured": len([row for row in per_graph_rows if not row["warmup"]]),
        "complete_test_split_graphs": len(test_indices),
        "complete_test_split_covered_every_run": True,
        "all_verification_passed": True,
        **purpose,
    }
    paths = save_results(per_graph_rows, dataset_rows, summary_payload, purpose, output_dir)

    print_markdown_table(
        dataset_rows,
        [
            "run_id",
            "warmup",
            "test_graph_count",
            "commit_time_ms",
            "prove_time_excluding_commit_ms",
            "total_prover_time_including_commit_ms",
            "proof_size_bytes",
            "proof_size_kb",
            "verify_time_ms",
            "wall_clock_dataset_run_time_ms",
            "resumed_graphs",
            "accuracy",
            "verification_passed",
        ],
        "Per full-dataset benchmark run raw results",
    )
    aggregate_rows = [{"metric": metric, **stats} for metric, stats in dataset_aggregate.items()]
    print_markdown_table(
        aggregate_rows,
        ["metric", "mean", "std", "min", "max"],
        "Aggregated full-dataset results (warm-up excluded)",
    )
    print("\nSaved artifacts")
    for key, path in paths.items():
        print(f"{key}={path}")


if __name__ == "__main__":
    main()
