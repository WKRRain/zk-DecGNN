# zk-DecGNN

This repository provides the implementation and experimental scripts for privacy-preserving graph classification with zero-knowledge proofs. The prover evaluates a quantized graph neural network and produces a Spartan-ZENO proof. The verifier checks the proof and learns only the public predicted label, while graph-level logits, intermediate representations, and model weights remain private.

The implementation supports SGC, SSGC, SIGN, and APPNP. Graph readout uses mean pooling over real nodes only; padding nodes are excluded by a public mask. The proof relation verifies the argmax of the graph-level logits, and does not include softmax in the circuit.

## Environment

The code has been tested with a Linux environment, Rust stable, and Python 3.9 or newer. Install the Rust toolchain and the Python dependencies:

```bash
python -m pip install numpy pytest torch torch-geometric
```

Build the Rust backend:

```bash
cd zkDecGNN
cargo build --release --lib
```

The Python runner will also build the release library automatically when needed.

## Sanity check

From `zkDecGNN`, run:

```bash
python -m pytest -q tests/test_graph_classification.py
```

This checks proof generation and verification for SGC, SSGC, SIGN, and APPNP, checks canonical proof serialization, and confirms that a modified proof is rejected.

## Running graph classification

The following example runs SSGC with two propagation steps on MUTAG:

```bash
cd zkDecGNN
python run_graph_classification.py \
  --model SSGC \
  --dataset MUTAG \
  --K 2
```

Supported values of `--model` are `SGC`, `SSGC`, `SIGN`, and `APPNP`. The runner downloads or reuses datasets through `torch-geometric`. By default it uses deterministic weights for an end-to-end smoke test. To use a trained checkpoint, provide an `.npz` file containing `W_pre0`, `W_pre1`, and `W_graph`:

```bash
python run_graph_classification.py \
  --model SSGC \
  --dataset MUTAG \
  --K 2 \
  --weights /path/to/checkpoint.npz \
  --require-trained-weights
```

Run `python run_graph_classification.py --help` for all configuration options.

## Reproducing benchmarks

Benchmark launchers are provided for MUTAG, NCI1, and REDDIT-BINARY under `zkDecGNN/benchmarks/`. For example, the MUTAG SSGC experiment can be run with:

```bash
cd zkDecGNN
python benchmarks/run_mutag_ssgc_k2_sign300max_zk_benchmark.py
```

Each launcher runs a complete test split repeatedly, uses the first complete split as warm-up, and aggregates the remaining runs. Per-graph measurements, per-dataset-run measurements, and summary statistics are saved in `benchmarks/results/`.

The trained checkpoints required by strict benchmark launchers must be supplied separately in `benchmarks/artifacts/`. The benchmark records identify the exact dataset split, physical proof shape, model configuration, proof size, prover time, and verifier time.

## Notes

- Every test graph is proved and verified as an independent proof instance.
- Public inputs bind the graph topology, padding mask, sparse edge representation, model configuration, and predicted label.
- The model owner keeps weights, private activations, graph-level logits, and witness commitments private.
- In the provided SSGC graph-classification relation, `alpha` is fixed to zero.

## License

This project is distributed under the MIT License.
