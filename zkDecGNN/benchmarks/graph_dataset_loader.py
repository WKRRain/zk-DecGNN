"""Graph-classification data loading and preprocessing for Spartan-ZENO.

The Rust backend owns witness/proof/verification.  This file only prepares the
public graph instance required by the graph-classification relation:
X, A_hat, mask m, N_real, and row/col/val sparse audit arrays.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


S_SCALE = 256
SPLIT_RATIO = "8:1:1"


@dataclass(frozen=True)
class GraphExample:
    graph_id: int
    x: np.ndarray
    edge_index: np.ndarray
    y: int


@dataclass(frozen=True)
class PreprocessedGraph:
    graph_id: int
    x: np.ndarray
    a_hat: np.ndarray
    mask: np.ndarray
    edge_rows: np.ndarray
    edge_cols: np.ndarray
    edge_vals: np.ndarray
    n_real: int
    n_max: int
    d: int
    y_true: int
    sparse_nnz: int
    sparse_padded_len: int


def next_power_of_two(value: int) -> int:
    value = int(value)
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def quantize_int16(values: np.ndarray, scale: int = S_SCALE) -> np.ndarray:
    quantized = np.rint(np.asarray(values, dtype=np.float64) * float(scale))
    return np.clip(quantized, -32768, 32767).astype(np.int64)


def _degree_features(edge_index: np.ndarray, num_nodes: int, width: int = 11) -> np.ndarray:
    degrees = np.bincount(edge_index[0], minlength=num_nodes)
    clipped = np.minimum(degrees, width - 1)
    x = np.zeros((num_nodes, width), dtype=np.float64)
    x[np.arange(num_nodes), clipped] = 1.0
    return x


def _synthetic_mutag(num_graphs: int = 20) -> List[GraphExample]:
    graphs: List[GraphExample] = []
    for graph_id in range(num_graphs):
        label = graph_id % 2
        n = 4 + (graph_id % 5)
        x = np.zeros((n, 4), dtype=np.float64)
        x[:, 0] = float(label)
        x[:, 1] = float(1 - label)
        x[:, 2] = np.linspace(0.0, 1.0, n)
        x[:, 3] = 1.0
        edges = []
        for i in range(n - 1):
            edges.append((i, i + 1))
            edges.append((i + 1, i))
        if label == 1 and n > 4:
            edges.append((0, n - 1))
            edges.append((n - 1, 0))
        edge_index = np.asarray(edges, dtype=np.int64).T
        graphs.append(GraphExample(graph_id=graph_id, x=x, edge_index=edge_index, y=label))
    return graphs


def load_graph_dataset(name: str, root: Path, max_graphs: int | None = None) -> List[GraphExample]:
    normalized = name.upper()
    if normalized in {"SYNTHETIC_MUTAG", "SYNTHETIC"}:
        graphs = _synthetic_mutag(max_graphs or 20)
        return graphs

    try:
        from torch_geometric.datasets import TUDataset
    except Exception as exc:  # pragma: no cover - exercised only without PyG
        raise RuntimeError("PyTorch Geometric is required for real graph datasets") from exc

    dataset = TUDataset(root=str(root), name=name)
    graphs: List[GraphExample] = []
    for graph_id, data in enumerate(dataset):
        num_nodes = int(data.num_nodes)
        edge_index = data.edge_index.detach().cpu().numpy().astype(np.int64)
        if data.x is None:
            x = _degree_features(edge_index, num_nodes)
        else:
            x = data.x.detach().cpu().numpy().astype(np.float64)
        y_tensor = data.y.detach().cpu().view(-1)
        graphs.append(GraphExample(graph_id=graph_id, x=x, edge_index=edge_index, y=int(y_tensor[0])))
        if max_graphs is not None and len(graphs) >= max_graphs:
            break
    if not graphs:
        raise RuntimeError(f"dataset {name} produced no graphs")
    return graphs


def split_dataset(
    graphs: Sequence[GraphExample],
    dataset_name: str,
    seed: int,
    results_dir: Path,
) -> Dict[str, object]:
    count = len(graphs)
    rng = np.random.default_rng(seed)
    indices = np.arange(count, dtype=np.int64)
    rng.shuffle(indices)
    m_train = int(np.floor(0.8 * count))
    m_val = int(np.floor(0.1 * count))
    train = indices[:m_train].tolist()
    val = indices[m_train : m_train + m_val].tolist()
    test = indices[m_train + m_val :].tolist()
    split_material = json.dumps(
        {"dataset": dataset_name, "seed": seed, "train": train, "val": val, "test": test},
        sort_keys=True,
    ).encode("utf-8")
    split_id = hashlib.sha256(split_material).hexdigest()[:16]
    split_dir = results_dir / "graph_classification_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_file = split_dir / f"{dataset_name.lower()}_{split_id}.json"
    payload = {
        "dataset": dataset_name,
        "split_ratio": SPLIT_RATIO,
        "seed": seed,
        "split_id": split_id,
        "train": train,
        "val": val,
        "test": test,
        "counts": {"total": count, "train": len(train), "val": len(val), "test": len(test)},
    }
    split_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    payload["split_file"] = str(split_file)
    return payload


def _normalized_adjacency(num_nodes: int, edge_index: np.ndarray) -> np.ndarray:
    a = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    if edge_index.size:
        for row, col in edge_index.T:
            if 0 <= row < num_nodes and 0 <= col < num_nodes:
                a[int(row), int(col)] = 1.0
    a[np.arange(num_nodes), np.arange(num_nodes)] = 1.0
    degree = a.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    nonzero = degree > 0
    inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    return inv_sqrt[:, None] * a * inv_sqrt[None, :]


def preprocess_graph(graph: GraphExample, n_max: int, d: int, scale: int = S_SCALE) -> PreprocessedGraph:
    n_real = int(graph.x.shape[0])
    if n_real > n_max:
        raise ValueError(
            f"graph {graph.graph_id} has {n_real} nodes, exceeding N_max={n_max}; regenerate a larger VK"
        )
    if int(graph.x.shape[1]) > d:
        raise ValueError(f"graph {graph.graph_id} feature dimension exceeds configured d={d}")

    x_padded = np.zeros((n_max, d), dtype=np.int64)
    x_quant = quantize_int16(graph.x, scale=scale)
    x_padded[:n_real, : graph.x.shape[1]] = x_quant

    a_real = _normalized_adjacency(n_real, graph.edge_index)
    a_quant = quantize_int16(a_real, scale=scale)
    a_padded = np.zeros((n_max, n_max), dtype=np.int64)
    a_padded[:n_real, :n_real] = a_quant

    mask = np.zeros(n_max, dtype=np.int64)
    mask[:n_real] = 1
    rows, cols = np.nonzero(a_padded)
    vals = a_padded[rows, cols].astype(np.int64)
    sparse_nnz = int(vals.size)
    padded_len = next_power_of_two(max(1, sparse_nnz))
    row_pad = np.zeros(padded_len, dtype=np.int64)
    col_pad = np.zeros(padded_len, dtype=np.int64)
    val_pad = np.zeros(padded_len, dtype=np.int64)
    row_pad[:sparse_nnz] = rows.astype(np.int64)
    col_pad[:sparse_nnz] = cols.astype(np.int64)
    val_pad[:sparse_nnz] = vals

    return PreprocessedGraph(
        graph_id=graph.graph_id,
        x=x_padded,
        a_hat=a_padded,
        mask=mask,
        edge_rows=row_pad,
        edge_cols=col_pad,
        edge_vals=val_pad,
        n_real=n_real,
        n_max=n_max,
        d=d,
        y_true=graph.y,
        sparse_nnz=sparse_nnz,
        sparse_padded_len=padded_len,
    )
