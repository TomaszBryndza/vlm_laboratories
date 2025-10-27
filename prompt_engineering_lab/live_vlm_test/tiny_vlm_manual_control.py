#!/usr/bin/env python3
"""Backward-compatible wrapper for TinyVLM manual control.

Refactored: implementation moved to `simulator.py` and model classes to `vlm_runners.py`.
This wrapper enforces `--vlm-model=tiny` if not already supplied.
"""
from simulator import main as simulator_main
import sys

def _inject_default_tiny():
    if not any(a.startswith("--vlm-model") for a in sys.argv[1:]):
        sys.argv.append("--vlm-model=tiny")

def main():
    _inject_default_tiny()
    simulator_main()

if __name__ == "__main__":
    main()
