#!/usr/bin/env python3
"""Run Spartan-ZENO graph-classification secure inference.

Each test graph is processed as one independent proof instance:
preprocess -> Rust witness/proof generation -> Rust verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = Path(__file__).resolve().parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

import rust_streaming_backend as rust_backend  # noqa: E402
from graph_dataset_loader import (  # noqa: E402
    SPLIT_RATIO,
    S_SCALE,
    load_graph_dataset,
    next_power_of_two,
    preprocess_graph,
    split_dataset,
)


DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
DEFAULT_DATA_ROOT = REPO_ROOT / "benchmarks" / "data" / "graph_classification"
MODEL_DEFAULT_K = {"SGC": 2, "SSGC": 4, "SIGN": 3, "APPNP": 2}


def now_ms() -> float:
    return time.perf_counter_ns() / 1_000_000.0


def _weights_for_model(model: str, d: int, c: int, h: int, d_max: int, scale: int) -> Dict[str, np.ndarray]:
    w_pre0 = np.zeros((d, h), dtype=np.int64)
    w_pre1 = np.zeros((h, d_max), dtype=np.int64)
    for idx in range(min(d, h)):
        w_pre0[idx, idx] = scale
    for idx in range(min(h, d_max, d)):
        w_pre1[idx, idx] = scale

    w_graph = np.zeros((d_max, c), dtype=np.int64)
    if c == 2 and d_max >= 2:
        w_graph[1, 0] = scale
        w_graph[0, 1] = scale
    else:
        for cls in range(c):
            w_graph[cls % max(1, d_max), cls] = scale
    return {"W_pre0": w_pre0, "W_pre1": w_pre1, "W_graph": w_graph}


def _weights_digest(weights: Dict[str, np.ndarray]) -> str:
    hasher = hashlib.sha256()
    for key in ["W_pre0", "W_pre1", "W_graph"]:
        arr = np.ascontiguousarray(weights[key], dtype=np.int64)
        hasher.update(key.encode("utf-8"))
        hasher.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        hasher.update(arr.tobytes())
    return hasher.hexdigest()[:16]


def _validate_weight_shapes(
    weights: Dict[str, np.ndarray],
    d: int,
    c: int,
    h: int,
    d_max: int,
) -> Dict[str, np.ndarray]:
    expected = {"W_pre0": (d, h), "W_pre1": (h, d_max), "W_graph": (d_max, c)}
    out: Dict[str, np.ndarray] = {}
    for key, shape in expected.items():
        if key not in weights:
            raise ValueError(f"weights file is missing {key}")
        arr = np.ascontiguousarray(weights[key], dtype=np.int64)
        if arr.shape != shape:
            raise ValueError(f"{key} shape mismatch: expected {shape}, got {arr.shape}")
        if np.any(arr < -32768) or np.any(arr > 32767):
            raise ValueError(f"{key} contains values outside INT16")
        out[key] = arr
    return out


def load_weights(
    weights_path: Path | None,
    model: str,
    d: int,
    c: int,
    h: int,
    d_max: int,
    scale: int,
    require_trained: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if weights_path is None:
        if require_trained:
            raise ValueError("--require-trained-weights was set but --weights was not provided")
        weights = _weights_for_model(model, d, c, h, d_max, scale)
        return weights, {
            "weight_source": "deterministic_smoke_weights",
            "weights_path": None,
            "smoke_weights": True,
            "weights_digest": _weights_digest(weights),
        }

    path = Path(weights_path).resolve()
    if path.suffix.lower() != ".npz":
        raise ValueError("graph classification weights must be provided as a .npz file")
    with np.load(path, allow_pickle=False) as data:
        raw = {key: data[key] for key in data.files}
    weights = _validate_weight_shapes(raw, d, c, h, d_max)
    return weights, {
        "weight_source": "external_npz",
        "weights_path": str(path),
        "smoke_weights": False,
        "weights_digest": _weights_digest(weights),
    }


def physical_dims(model: str, graphs: List[Any], k: int, n_max_arg: int | None, hidden_dim: int) -> Dict[str, int]:
    n_max = next_power_of_two(n_max_arg or max(int(graph.x.shape[0]) for graph in graphs))
    d = max(int(graph.x.shape[1]) for graph in graphs)
    c = max(int(graph.y) for graph in graphs) + 1
    if model == "SIGN":
        d_max = next_power_of_two((k + 1) * d)
    elif model == "APPNP":
        d_max = next_power_of_two(max(c, d))
    else:
        d_max = next_power_of_two(d)
    h = next_power_of_two(max(1, hidden_dim))
    h_max = h
    return {"N_max": n_max, "d": d, "c": c, "D_max": d_max, "h": h, "H_max": h_max}


def _json_safe(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in row.items():
        if isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def run_experiment(args: argparse.Namespace | SimpleNamespace) -> Dict[str, Any]:
    model = str(args.model).upper()
    if model not in MODEL_DEFAULT_K:
        raise ValueError(f"unsupported model: {args.model}")
    if int(args.batch_size) != 1:
        raise ValueError("graph classification requires batch_size=1")
    if str(args.split) != SPLIT_RATIO:
        raise ValueError("graph classification split must be 8:1:1")

    results_dir = Path(args.results_dir).resolve()
    data_root = Path(args.data_root).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    graphs = load_graph_dataset(str(args.dataset), data_root, max_graphs=args.max_graphs)
    split = split_dataset(graphs, str(args.dataset), int(args.seed), results_dir)
    k = int(args.K or MODEL_DEFAULT_K[model])
    dims = physical_dims(model, graphs, k, args.n_max, int(args.hidden_dim))
    weights_path = getattr(args, "weights", None)
    require_trained = bool(getattr(args, "require_trained_weights", False))
    weights, weight_meta = load_weights(
        Path(weights_path) if weights_path else None,
        model,
        dims["d"],
        dims["c"],
        dims["h"],
        dims["D_max"],
        int(args.scale),
        require_trained=require_trained,
    )
    rust = rust_backend.load(build=not bool(args.skip_build))

    test_indices = list(split["test"])
    if args.max_test_graphs:
        test_indices = test_indices[: int(args.max_test_graphs)]
    rows: List[Dict[str, Any]] = []
    correct = 0

    for run_id, dataset_index in enumerate(test_indices):
        graph = graphs[int(dataset_index)]
        pre = preprocess_graph(graph, dims["N_max"], dims["d"], scale=int(args.scale))
        proof = rust.graph_classification_prove(
            model,
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
            int(k),
            int(dims["D_max"]),
            int(dims["H_max"]),
            int(args.scale),
            int(args.arg_bits),
        )
        proof_bytes = rust.graph_classification_serialize_proof(proof)
        verify_start = now_ms()
        verify_fn = getattr(rust, "graph_classification_verify_serialized", None)
        if verify_fn is None:
            verdict = rust.graph_classification_verify(
                proof,
                model,
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
                int(k),
                int(dims["D_max"]),
                int(dims["H_max"]),
            )
        else:
            verdict = verify_fn(
                proof_bytes,
                model,
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
                int(k),
                int(dims["D_max"]),
                int(dims["H_max"]),
            )
        verify_time_ms = now_ms() - verify_start
        if not verdict["accepted"]:
            raise RuntimeError(f"verification failed for graph {pre.graph_id}: {verdict['reason']}")
        ell = int(verdict["ell"])
        correct += int(ell == int(pre.y_true))
        timing = proof["timing"]
        proof_size_bytes = len(proof_bytes)
        row = {
            "run_id": run_id,
            "dataset": args.dataset,
            "model": model,
            "task_type": "graph_classification",
            "split_ratio": SPLIT_RATIO,
            "split_id": split["split_id"],
            "batchsize": 1,
            "graph_id": pre.graph_id,
            "N_real": pre.n_real,
            "N_max": dims["N_max"],
            "d": dims["d"],
            "c": dims["c"],
            "h": dims["h"],
            "K": k,
            "D_max": dims["D_max"],
            "H_max": dims["H_max"],
            "sparse_nnz": pre.sparse_nnz,
            "sparse_padded_len": pre.sparse_padded_len,
            "ell": ell,
            "y_true_for_accuracy_only": int(pre.y_true),
            "correct": bool(ell == int(pre.y_true)),
            "witness_generation_time_ms": float(timing["witness_generation_time_ms"]),
            "prove_time_ms": float(timing["prove_time_ms"]),
            "verify_time_ms": verify_time_ms,
            "proof_size_bytes": proof_size_bytes,
            "verification_passed": bool(verdict["accepted"]),
            "mean_pool": bool(proof["public"]["mean_pool"]),
            "mean_pool_denominator": int(proof["public"]["mean_pool_denominator"]),
            "softmax_circuit_disabled": bool(proof["public"]["softmax_circuit_disabled"]),
            "argmax_proof": bool(proof["public"]["argmax_proof"]),
            "Y_G_public": False,
            "weight_source": weight_meta["weight_source"],
            "weights_digest": weight_meta["weights_digest"],
            "smoke_weights": weight_meta["smoke_weights"],
        }
        rows.append(row)

    test_count = len(rows)
    accuracy = float(correct / test_count) if test_count else 0.0
    avg = lambda field: statistics.fmean(float(row[field]) for row in rows) if rows else 0.0
    summary = {
        "dataset": args.dataset,
        "model": model,
        "split_ratio": SPLIT_RATIO,
        "split_id": split["split_id"],
        "split_file": split["split_file"],
        "batchsize": 1,
        "N_max": dims["N_max"],
        "test_graph_count": test_count,
        "accuracy": accuracy,
        "average_prove_time_ms": avg("prove_time_ms"),
        "average_verify_time_ms": avg("verify_time_ms"),
        "average_witness_generation_time_ms": avg("witness_generation_time_ms"),
        "mean_pool": True,
        "softmax_circuit_disabled": True,
        "argmax_proof": True,
        "proof_backend": "Rust Spartan-ZENO graph_classification",
        **weight_meta,
    }

    prefix = f"graph_classification_{str(args.dataset).lower()}_{model.lower()}_{split['split_id']}"
    raw_path = results_dir / f"{prefix}_raw.json"
    summary_path = results_dir / f"{prefix}_summary.json"
    raw_path.write_text(json.dumps([_json_safe(row) for row in rows], indent=2, sort_keys=True), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["raw_json"] = str(raw_path)
    summary["summary_json"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_DEFAULT_K), required=True)
    parser.add_argument("--dataset", default="MUTAG")
    parser.add_argument("--split", default=SPLIT_RATIO)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--n-max", type=int, default=None)
    parser.add_argument("--scale", type=int, default=S_SCALE)
    parser.add_argument("--arg-bits", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--max-test-graphs", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional .npz file with INT16 W_pre0, W_pre1, and W_graph arrays.",
    )
    parser.add_argument(
        "--require-trained-weights",
        action="store_true",
        help="Reject deterministic smoke weights and require --weights.",
    )
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def print_summary(summary: Dict[str, Any]) -> None:
    print(f"dataset={summary['dataset']}")
    print(f"model={summary['model']}")
    print(f"split_ratio={summary['split_ratio']}")
    print(f"split_id={summary['split_id']}")
    print(f"batchsize={summary['batchsize']}")
    print(f"N_max={summary['N_max']}")
    print(f"test_graph_count={summary['test_graph_count']}")
    print(f"accuracy={summary['accuracy']:.6f}")
    print(f"average_prove_time_ms={summary['average_prove_time_ms']:.3f}")
    print(f"average_verify_time_ms={summary['average_verify_time_ms']:.3f}")
    print(f"average_witness_generation_time_ms={summary['average_witness_generation_time_ms']:.3f}")
    print(f"MeanPool={summary['mean_pool']}")
    print(f"softmax_circuit_disabled={summary['softmax_circuit_disabled']}")
    print(f"argmax_proof={summary['argmax_proof']}")
    print(f"weight_source={summary['weight_source']}")
    print(f"weights_digest={summary['weights_digest']}")
    print(f"smoke_weights={summary['smoke_weights']}")
    print(f"raw_json={summary['raw_json']}")
    print(f"summary_json={summary['summary_json']}")


def main() -> None:
    try:
        summary = run_experiment(parse_args())
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print_summary(summary)


if __name__ == "__main__":
    main()
