#!/usr/bin/env python3
"""Backward-compatible wrapper for Qwen VLM manual control.

Refactored: logic lives in `simulator.py` / `vlm_runners.py`. This forwards to the
generic simulator ensuring `--vlm-model=qwen` is set if the caller didn't specify it.
"""
from simulator import main as simulator_main
import sys

def _inject_default_qwen():
    if not any(a.startswith("--vlm-model") for a in sys.argv[1:]):
        sys.argv.append("--vlm-model=qwen")

def main():
    _inject_default_qwen()
    simulator_main()

if __name__ == "__main__":
    main()
