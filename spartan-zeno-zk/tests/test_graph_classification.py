import sys
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import rust_streaming_backend as rust_backend
from graph_classification_runner import run_experiment
from graph_dataset_loader import load_graph_dataset, preprocess_graph
from graph_classification_runner import _weights_for_model, physical_dims


def _one_graph_contract(model: str):
    graphs = load_graph_dataset("SYNTHETIC_MUTAG", ROOT / "benchmarks" / "data", max_graphs=10)
    k = {"SGC": 2, "SSGC": 2, "SIGN": 2, "APPNP": 2}[model]
    dims = physical_dims(model, graphs, k, None, 8)
    graph = graphs[0]
    pre = preprocess_graph(graph, dims["N_max"], dims["d"])
    weights = _weights_for_model(model, dims["d"], dims["c"], dims["h"], dims["D_max"], 256)
    rust = rust_backend.load(build=True)
    proof = rust.graph_classification_prove(
        model,
        "testsplit",
        pre.graph_id,
        pre.x,
        pre.a_hat,
        pre.mask,
        pre.edge_rows,
        pre.edge_cols,
        pre.edge_vals,
        weights["W_pre0"],
        weights["W_pre1"],
        weights["W_graph"],
        pre.n_real,
        dims["N_max"],
        dims["d"],
        dims["c"],
        dims["h"],
        k,
        dims["D_max"],
        dims["H_max"],
        256,
        64,
    )
    verdict = rust.graph_classification_verify(
        proof,
        model,
        "testsplit",
        pre.graph_id,
        pre.x,
        pre.a_hat,
        pre.mask,
        pre.edge_rows,
        pre.edge_cols,
        pre.edge_vals,
        pre.n_real,
        dims["N_max"],
        dims["d"],
        dims["c"],
        dims["h"],
        k,
        dims["D_max"],
        dims["H_max"],
    )
    return pre, proof, verdict, dims


def _verify_contract(model: str, pre, proof, dims):
    k = {"SGC": 2, "SSGC": 2, "SIGN": 2, "APPNP": 2}[model]
    rust = rust_backend.load(build=False)
    return rust.graph_classification_verify(
        proof,
        model,
        "testsplit",
        pre.graph_id,
        pre.x,
        pre.a_hat,
        pre.mask,
        pre.edge_rows,
        pre.edge_cols,
        pre.edge_vals,
        pre.n_real,
        dims["N_max"],
        dims["d"],
        dims["c"],
        dims["h"],
        k,
        dims["D_max"],
        dims["H_max"],
    )


def test_single_graph_preprocess_contract():
    graphs = load_graph_dataset("SYNTHETIC_MUTAG", ROOT / "benchmarks" / "data", max_graphs=10)
    dims = physical_dims("SGC", graphs, 2, None, 8)
    pre = preprocess_graph(graphs[0], dims["N_max"], dims["d"])

    assert pre.x.shape == (dims["N_max"], dims["d"])
    assert pre.a_hat.shape == (dims["N_max"], dims["N_max"])
    assert int(pre.mask.sum()) == pre.n_real
    assert pre.sparse_padded_len & (pre.sparse_padded_len - 1) == 0


def test_rust_proof_verify_contract_all_models():
    rust = rust_backend.load(build=False)
    for model in ["SGC", "SSGC", "SIGN", "APPNP"]:
        pre, proof, verdict, dims = _one_graph_contract(model)
        assert verdict["accepted"]
        proof_bytes = rust.graph_classification_serialize_proof(proof)
        assert isinstance(proof_bytes, bytes)
        assert len(proof_bytes) > 0
        assert proof_bytes == rust.graph_classification_serialize_proof(proof)
        fast_verdict = rust.graph_classification_verify_serialized(
            proof_bytes,
            model,
            "testsplit",
            pre.graph_id,
            pre.x,
            pre.a_hat,
            pre.mask,
            pre.edge_rows,
            pre.edge_cols,
            pre.edge_vals,
            pre.n_real,
            dims["N_max"],
            dims["d"],
            dims["c"],
            dims["h"],
            {"SGC": 2, "SSGC": 2, "SIGN": 2, "APPNP": 2}[model],
            dims["D_max"],
            dims["H_max"],
        )
        assert fast_verdict["accepted"]
        tampered_bytes = bytearray(proof_bytes)
        tampered_bytes[-1] ^= 1
        assert not rust.graph_classification_verify_serialized(
            bytes(tampered_bytes),
            model,
            "testsplit",
            pre.graph_id,
            pre.x,
            pre.a_hat,
            pre.mask,
            pre.edge_rows,
            pre.edge_cols,
            pre.edge_vals,
            pre.n_real,
            dims["N_max"],
            dims["d"],
            dims["c"],
            dims["h"],
            {"SGC": 2, "SSGC": 2, "SIGN": 2, "APPNP": 2}[model],
            dims["D_max"],
            dims["H_max"],
        )["accepted"]
        assert proof["proof_type"] == "Spartan-ZENO-GC-v2"
        assert proof["proof_schema_version"] == 2
        assert proof["public"]["mean_pool_denominator"] == pre.n_real
        assert proof["public"]["mean_pool"]
        assert proof["public"]["softmax_circuit_disabled"]
        assert proof["public"]["argmax_proof"]
        assert proof["public"]["range_residual_profile"] == "compact_value_reconstruction_v1"
        assert proof["public"]["active_column_profile"] == "model_aware_active_columns_v1"
        assert proof["public"]["x_active_cols"] <= dims["D_max"]
        assert proof["public"]["h_active_cols"] <= dims["D_max"]
        assert proof["public"]["w_graph_active_rows"] == proof["public"]["h_active_cols"]
        assert proof["public"]["class_active_cols"] == dims["c"]
        assert proof["public"]["spark_active_cols"] == proof["public"]["x_active_cols"]
        assert proof["public"]["yg_shape"] == [1, dims["c"]]
        assert proof["public"]["sparse_nnz"] == pre.sparse_nnz
        assert proof["public"]["sparse_row_col_val_padded_len"] == pre.sparse_padded_len
        assert proof["public"]["vk_id"]
        assert proof["public"]["weight_fingerprint"]
        assert "C_Delta" in proof["commitments"]
        assert "C_R1CS_residuals" in proof["commitments"]
        assert proof["openings"]["opening_mode"] == "batched_ipa_v1"
        assert len(proof["openings"]["batches"]) > 0
        assert proof["sumcheck_claims"]["constraint_commitment_digest"]
        assert set(proof["sumcheck_claims"]["proofs"]) == {"rlc_residuals"}
        assert proof["sumcheck_claims"]["semantic_residual_count"] > 0
        assert proof["sumcheck_claims"]["r1cs_residual_count"] > proof["sumcheck_claims"]["range_residual_count"]
        assert len(proof["sumcheck_claims"]["proofs"]["rlc_residuals"]["rounds"]) > 0
        assert "Y_G" not in proof
        assert "logits" not in proof
        assert "Delta" not in proof
        assert "delta_bits" not in proof


def test_tampered_graph_classification_proofs_are_rejected():
    pre, proof, verdict, dims = _one_graph_contract("SGC")
    assert verdict["accepted"]

    tampered = copy.deepcopy(proof)
    tampered["proof_digest"] = "00"
    assert not _verify_contract("SGC", pre, tampered, dims)["accepted"]

    tampered = copy.deepcopy(proof)
    tampered["proof_schema_version"] = 1
    assert not _verify_contract("SGC", pre, tampered, dims)["accepted"]

    tampered = copy.deepcopy(proof)
    tampered["model"] = "SSGC"
    assert not _verify_contract("SGC", pre, tampered, dims)["accepted"]

    tampered = copy.deepcopy(proof)
    tampered["public"]["N_real"] = pre.n_real + 1
    assert not _verify_contract("SGC", pre, tampered, dims)["accepted"]

    tampered = copy.deepcopy(proof)
    tampered["spark"]["init_root"] = "00"
    assert not _verify_contract("SGC", pre, tampered, dims)["accepted"]

    tampered = copy.deepcopy(proof)
    tampered["commitments"]["C_Y_G"][0] = "00"
    assert not _verify_contract("SGC", pre, tampered, dims)["accepted"]

    tampered = copy.deepcopy(proof)
    tampered["commitments"]["C_Delta"][0] = "00"
    assert not _verify_contract("SGC", pre, tampered, dims)["accepted"]

    tampered = copy.deepcopy(proof)
    tampered["commitments"]["C_R1CS_residuals"][0] = "00"
    assert not _verify_contract("SGC", pre, tampered, dims)["accepted"]

    tampered = copy.deepcopy(proof)
    tampered["openings"]["batches"][0]["rounds"][0]["u"] = "00"
    assert not _verify_contract("SGC", pre, tampered, dims)["accepted"]

    tampered = copy.deepcopy(proof)
    tampered["public"]["weight_fingerprint"] = "00"
    assert not _verify_contract("SGC", pre, tampered, dims)["accepted"]

    tampered = copy.deepcopy(proof)
    tampered["range_commitments"]["ArgmaxDelta"] = "00"
    assert not _verify_contract("SGC", pre, tampered, dims)["accepted"]

    tampered = copy.deepcopy(proof)
    tampered["sumcheck_claims"]["constraint_commitment_digest"] = "00"
    assert not _verify_contract("SGC", pre, tampered, dims)["accepted"]

    tampered = copy.deepcopy(proof)
    tampered["sumcheck_claims"]["proofs"]["rlc_residuals"]["rounds"][0]["g0"] = "01"
    assert not _verify_contract("SGC", pre, tampered, dims)["accepted"]


def test_public_instance_canonicalization_is_enforced():
    pre, _proof, _verdict, dims = _one_graph_contract("SGC")
    weights = _weights_for_model("SGC", dims["d"], dims["c"], dims["h"], dims["D_max"], 256)
    rust = rust_backend.load(build=False)

    bad_mask = pre.mask.copy()
    if pre.n_real < dims["N_max"]:
        bad_mask[1] = 0
        bad_mask[pre.n_real] = 1
        with pytest.raises(ValueError, match="canonical prefix mask"):
            rust.graph_classification_prove(
                "SGC",
                "testsplit",
                pre.graph_id,
                pre.x,
                pre.a_hat,
                bad_mask,
                pre.edge_rows,
                pre.edge_cols,
                pre.edge_vals,
                weights["W_pre0"],
                weights["W_pre1"],
                weights["W_graph"],
                pre.n_real,
                dims["N_max"],
                dims["d"],
                dims["c"],
                dims["h"],
                2,
                dims["D_max"],
                dims["H_max"],
                256,
                64,
            )

    bad_vals = pre.edge_vals.copy()
    bad_vals[0] = bad_vals[0] + 1
    with pytest.raises(ValueError, match="A_hat nonzero expansion"):
        rust.graph_classification_prove(
            "SGC",
            "testsplit",
            pre.graph_id,
            pre.x,
            pre.a_hat,
            pre.mask,
            pre.edge_rows,
            pre.edge_cols,
            bad_vals,
            weights["W_pre0"],
            weights["W_pre1"],
            weights["W_graph"],
            pre.n_real,
            dims["N_max"],
            dims["d"],
            dims["c"],
            dims["h"],
            2,
            dims["D_max"],
            dims["H_max"],
            256,
            64,
        )


def test_runner_outputs_accuracy_summary():
    args = SimpleNamespace(
        model="SGC",
        dataset="SYNTHETIC_MUTAG",
        split="8:1:1",
        batch_size=1,
        K=2,
        hidden_dim=8,
        n_max=None,
        scale=256,
        arg_bits=64,
        seed=20260705,
        max_graphs=10,
        max_test_graphs=1,
        data_root=ROOT / "benchmarks" / "data",
        results_dir=ROOT / "benchmarks" / "results" / "pytest_graph_classification",
        skip_build=True,
    )
    summary = run_experiment(args)

    assert summary["dataset"] == "SYNTHETIC_MUTAG"
    assert summary["model"] == "SGC"
    assert summary["split_ratio"] == "8:1:1"
    assert summary["batchsize"] == 1
    assert summary["test_graph_count"] == 1
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert summary["mean_pool"]
    assert summary["softmax_circuit_disabled"]
    assert summary["argmax_proof"]
    assert summary["weight_source"] == "deterministic_smoke_weights"
    assert summary["smoke_weights"]


def test_runner_accepts_external_npz_weights(tmp_path):
    graphs = load_graph_dataset("SYNTHETIC_MUTAG", ROOT / "benchmarks" / "data", max_graphs=10)
    dims = physical_dims("SGC", graphs, 2, None, 8)
    weights = _weights_for_model("SGC", dims["d"], dims["c"], dims["h"], dims["D_max"], 256)
    weights_path = tmp_path / "sgc_weights.npz"
    np.savez(weights_path, **weights)

    args = SimpleNamespace(
        model="SGC",
        dataset="SYNTHETIC_MUTAG",
        split="8:1:1",
        batch_size=1,
        K=2,
        hidden_dim=8,
        n_max=None,
        scale=256,
        arg_bits=64,
        seed=20260705,
        max_graphs=10,
        max_test_graphs=1,
        data_root=ROOT / "benchmarks" / "data",
        results_dir=tmp_path / "results",
        weights=weights_path,
        require_trained_weights=True,
        skip_build=True,
    )
    summary = run_experiment(args)

    assert summary["weight_source"] == "external_npz"
    assert not summary["smoke_weights"]
    assert summary["weights_path"] == str(weights_path.resolve())
