#!/usr/bin/env python3
"""Convenience entrypoint for Spartan-ZENO graph classification."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from graph_classification_runner import main  # noqa: E402


if __name__ == "__main__":
    main()
