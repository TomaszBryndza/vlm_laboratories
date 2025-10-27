#!/usr/bin/env python3
"""Backward-compatible wrapper for Phi VLM manual control.

Refactored: core simulator + model runners now live in `simulator.py` and `vlm_runners.py`.
This script simply forwards to the generic simulator with `--vlm-model=phi` unless
the user already supplied a different `--vlm-model` argument.
"""
from simulator import main as simulator_main
import sys

def _inject_default_phi():
    if not any(a.startswith("--vlm-model") for a in sys.argv[1:]):
        sys.argv.append("--vlm-model=phi")

def main():
    _inject_default_phi()
    simulator_main()

if __name__ == "__main__":
    main()
