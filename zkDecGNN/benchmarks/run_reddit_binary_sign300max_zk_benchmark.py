#!/usr/bin/env python3
"""Full-dataset REDDIT-BINARY SIGN(3,0,0) Spartan-ZENO ZK benchmark launcher.

Purpose:
  Run a zero-knowledge correctness and integrity benchmark for REDDIT-BINARY
  graph classification. The actual GNN is SIGN(3,0,0), and the physical
  proving container, VK capacity, D_max, and H_max are SIGN(3,0,0)_MAX in
  global-max mode. This measures SIGN inside its matching global maximum
  container, not a compact REDDIT-BINARY-specific VK shape.

Dataset-run policy:
  One benchmark run means one complete REDDIT-BINARY test-split experiment.
  Since the graph-classification proof has batchsize=1, each graph gets its
  own proof/verify flow. The launcher defaults to six complete dataset runs:
  run_id=0 is a full-test-split warm-up and is excluded from aggregate
  statistics; run_id=1..5 are five measured full-test-split experiments.

Relation semantics:
  - dataset=REDDIT-BINARY, task=graph_classification, split_ratio=8:1:1
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

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np

import run_mutag_sgc_k2_sign300max_full_dataset_optimized_rust as base
from graph_dataset_loader import quantize_int16


REDDIT_BINARY_SCOPE = "complete_REDDIT_BINARY_test_split"
SIGN_ARCHITECTURE = "SIGN(3,0,0)"
SIGN_K = 3


def reddit_sign_print(*args: Any, **kwargs: Any) -> None:
    if args and isinstance(args[0], str):
        text = (
            args[0]
            .replace("MUTAG", "REDDIT-BINARY")
            .replace("complete_MUTAG_test_split", REDDIT_BINARY_SCOPE)
        )
        if text == "actual_model=SIGN, actual_K=3 (not SIGN)":
            text = f"actual_model=SIGN, actual_K=3, architecture={SIGN_ARCHITECTURE}"
        args = (text,) + args[1:]
    print(*args, **kwargs)


class LazyPreprocessedTestCache:
    """Stream large REDDIT-BINARY test graphs without retaining dense A_hat."""

    def __init__(self, graphs: Sequence[Any], test_indices: Sequence[int], dims: Dict[str, int], scale: int) -> None:
        self.graphs = graphs
        self.test_indices = [int(idx) for idx in test_indices]
        self.dims = dims
        self.scale = int(scale)

    def __len__(self) -> int:
        return len(self.test_indices)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for graph_position, dataset_index in enumerate(self.test_indices):
            graph = self.graphs[dataset_index]
            graph_start = base.now_ms()
            pre = base.preprocess_graph(graph, self.dims["N_max"], self.dims["d"], scale=self.scale)
            yield {
                "graph_position": graph_position,
                "dataset_index": dataset_index,
                "graph": graph,
                "pre": pre,
                "public_preprocess_time_ms": base.now_ms() - graph_start,
            }


def reddit_sign_build_preprocessed_test_cache(
    graphs: List[Any],
    test_indices: List[int],
    dims: Dict[str, int],
    scale: int,
) -> Tuple[LazyPreprocessedTestCache, float]:
    return LazyPreprocessedTestCache(graphs, test_indices, dims, scale), 0.0


def div_euclid_i128(value: int, divisor: int) -> int:
    q = int(value) // int(divisor)
    r = int(value) % int(divisor)
    if r < 0:
        q -= 1
    return q


def normalized_sparse_entries(num_nodes: int, edge_index: np.ndarray, scale: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    present = set()
    if edge_index.size:
        for row, col in np.asarray(edge_index, dtype=np.int64).T:
            if 0 <= int(row) < num_nodes and 0 <= int(col) < num_nodes:
                present.add((int(row), int(col)))
    for idx in range(num_nodes):
        present.add((idx, idx))

    rows = np.fromiter((row for row, _ in present), dtype=np.int64, count=len(present))
    cols = np.fromiter((col for _, col in present), dtype=np.int64, count=len(present))
    degree = np.bincount(rows, minlength=num_nodes).astype(np.float64)
    inv_sqrt = np.zeros_like(degree)
    nonzero = degree > 0
    inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    values = inv_sqrt[rows] * inv_sqrt[cols]
    order = np.lexsort((cols, rows))
    rows = rows[order]
    cols = cols[order]
    vals = quantize_int16(values[order], scale=scale)
    return rows, cols, vals.astype(np.int64)


def sparse_sign_fixed_point_embedding(graph: Any, d: int, d_max: int, k: int, scale: int) -> np.ndarray:
    n_real = int(graph.x.shape[0])
    x_quant = quantize_int16(graph.x, scale=scale)
    x_base = np.zeros((n_real, d_max), dtype=np.int64)
    x_base[:, : int(graph.x.shape[1])] = x_quant[:, :d]
    rows, cols, vals = normalized_sparse_entries(n_real, graph.edge_index, scale)

    branches = [x_base.copy()]
    cur = x_base.copy()
    for _ in range(int(k)):
        raw = np.zeros_like(cur)
        for row, col, val in zip(rows, cols, vals):
            if int(val) != 0:
                raw[int(row), :d] += int(val) * cur[int(col), :d]
        nxt = np.zeros_like(cur)
        for idx, value in np.ndenumerate(raw[:, :d]):
            nxt[idx[0], idx[1]] = div_euclid_i128(int(value), int(scale))
        cur = nxt
        branches.append(cur.copy())

    concat = np.zeros((n_real, d_max), dtype=np.int64)
    for block, branch in enumerate(branches):
        start = block * d
        stop = min(start + d, d_max)
        if start >= d_max:
            break
        concat[:, start:stop] = branch[:, : stop - start]

    active_cols = min((int(k) + 1) * d, d_max)
    pooled = np.zeros((d_max,), dtype=np.int64)
    for col in range(active_cols):
        pooled[col] = div_euclid_i128(int(concat[:, col].sum()), n_real)
    return pooled


def reddit_sign_train_and_export_checkpoint(
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
        raise RuntimeError("PyTorch is required to train the REDDIT-BINARY SIGN checkpoint") from exc

    start = base.now_ms()
    torch.manual_seed(seed)
    features = []
    labels = []
    for dataset_index in [int(idx) for idx in split["train"]]:
        features.append(
            sparse_sign_fixed_point_embedding(
                graphs[dataset_index],
                dims["d"],
                dims["D_max"],
                SIGN_K,
                scale,
            )
        )
        labels.append(int(graphs[dataset_index].y))

    x_train = torch.tensor(np.stack(features).astype(np.float32) / float(scale), dtype=torch.float32)
    y_train = torch.tensor(labels, dtype=torch.long)
    w = torch.zeros((dims["D_max"], dims["c"]), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=float(lr), weight_decay=float(weight_decay))
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        loss = F.cross_entropy(x_train @ w, y_train)
        loss.backward()
        optimizer.step()

    active_cols = min((SIGN_K + 1) * dims["d"], dims["D_max"])
    w_graph = np.rint(w.detach().cpu().numpy() * float(scale)).astype(np.int64)
    w_graph[active_cols:, :] = 0
    w_graph = np.clip(w_graph, -32768, 32767).astype(np.int64)
    w_pre0, w_pre1 = base.identity_aux_weights(dims["d"], dims["h"], dims["D_max"], scale)
    weights = {"W_pre0": w_pre0, "W_pre1": w_pre1, "W_graph": w_graph}
    training_ms = base.now_ms() - start

    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset": "REDDIT-BINARY",
        "task": "graph_classification",
        "actual_model": "SIGN",
        "actual_K": SIGN_K,
        "actual_architecture": SIGN_ARCHITECTURE,
        "shape_profile": base.SHAPE_PROFILE,
        "container_mode": base.CONTAINER_MODE,
        "split_id": split["split_id"],
        "training_time_ms_excluded": training_ms,
        "training_feature_builder": "sparse_sign300_equivalent_to_rust_relation",
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        **dims,
    }
    np.savez_compressed(path, **weights, meta_json=np.array(json.dumps(meta, sort_keys=True)))
    return weights, training_ms


def reddit_sign_ensure_weights(
    checkpoint: Path,
    graphs: List[Any],
    split: Dict[str, Any],
    dims: Dict[str, int],
    args: Any,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if not args.force_retrain:
        loaded = base.load_weight_checkpoint(checkpoint, dims)
        if loaded is not None:
            return loaded, {
                "checkpoint_loaded": True,
                "checkpoint_path": str(checkpoint),
                "training_time_ms_excluded": 0.0,
                "weight_source": "trained_npz_checkpoint",
            }

    weights, training_ms = reddit_sign_train_and_export_checkpoint(
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


def apply_reddit_sign_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    row["dataset"] = "REDDIT-BINARY"
    row["task"] = "graph_classification"
    row["complete_dataset_scope"] = REDDIT_BINARY_SCOPE
    row["actual_model"] = "SIGN"
    row["actual_K"] = SIGN_K
    row["actual_architecture"] = SIGN_ARCHITECTURE
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


def apply_reddit_sign_purpose(purpose: Dict[str, Any]) -> Dict[str, Any]:
    purpose["purpose"] = (
        "Six-run complete-test-split Spartan-ZENO ZK correctness and integrity "
        "benchmark for REDDIT-BINARY graph classification with actual model SIGN(3,0,0)."
    )
    purpose["dataset"] = "REDDIT-BINARY"
    purpose["task"] = "graph_classification"
    purpose["complete_dataset_scope"] = REDDIT_BINARY_SCOPE
    purpose["aggregate_primary_unit"] = REDDIT_BINARY_SCOPE + "_run"
    purpose["warmup_policy"] = "run_id=0 full REDDIT-BINARY test split excluded from aggregate statistics"
    purpose["actual_model"] = "SIGN"
    purpose["actual_K"] = SIGN_K
    purpose["actual_architecture"] = SIGN_ARCHITECTURE
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


def reddit_sign_write_progress_marker(path: Path, row: Dict[str, Any]) -> None:
    _original_write_progress_marker(path, apply_reddit_sign_metadata(row))


def reddit_sign_save_results(
    per_graph_rows: List[Dict[str, Any]],
    dataset_rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    purpose: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Path]:
    for row in per_graph_rows:
        apply_reddit_sign_metadata(row)
    for row in dataset_rows:
        row["dataset"] = "REDDIT-BINARY"
        row["task"] = "graph_classification"
        row["complete_dataset_scope"] = REDDIT_BINARY_SCOPE
        row["actual_model"] = "SIGN"
        row["actual_K"] = SIGN_K
        row["actual_architecture"] = SIGN_ARCHITECTURE
    apply_reddit_sign_purpose(purpose)
    apply_reddit_sign_purpose(summary)
    return _original_save_results(per_graph_rows, dataset_rows, summary, purpose, output_dir)


base.DATASET = "REDDIT-BINARY"
base.ACTUAL_MODEL = "SIGN"
base.ACTUAL_K = SIGN_K
base.RESULT_PREFIX = "reddit_binary_sign300_under_sign300max_graph_classification"
base.DEFAULT_CHECKPOINT = (
    base.REPO_ROOT
    / "benchmarks"
    / "artifacts"
    / "reddit_binary_sign300_under_sign300max_graph_weights.npz"
)
base.build_preprocessed_test_cache = reddit_sign_build_preprocessed_test_cache
base.fixed_point_embedding = sparse_sign_fixed_point_embedding
base.ensure_weights = reddit_sign_ensure_weights
base.write_progress_marker = reddit_sign_write_progress_marker
base.save_results = reddit_sign_save_results
base.print = reddit_sign_print


if __name__ == "__main__":
    base.main()
