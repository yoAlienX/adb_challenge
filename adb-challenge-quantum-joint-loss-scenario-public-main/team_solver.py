"""Team solver entry point for the qjlb evaluation harness."""

from __future__ import annotations

import os
import sys

from qjlb.worker import run_solver
from qjlb_qae.solver import solve as qae_solve


def _extract_mode() -> str:
    mode = os.environ.get("QAE_SOLVER_MODE", "tilted_iqae")
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        if idx + 1 < len(sys.argv):
            mode = sys.argv[idx + 1]
            del sys.argv[idx : idx + 2]
    return mode


def solve(model, samples, seed):
    return qae_solve(model, samples, seed, mode=_extract_mode())


if __name__ == "__main__":
    mode = _extract_mode()
    run_solver(lambda model, samples, seed: qae_solve(model, samples, seed, mode=mode))
