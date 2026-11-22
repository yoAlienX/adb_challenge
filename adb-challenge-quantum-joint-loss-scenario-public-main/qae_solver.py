"""CLI worker for the QAE, hybrid, and MPS solvers."""

from __future__ import annotations

import os
import sys

from qjlb.worker import run_solver
from qjlb_qae.solver import solve


def _extract_mode() -> str:
    mode = os.environ.get("QAE_SOLVER_MODE", "tilted_iqae")
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        if idx + 1 < len(sys.argv):
            mode = sys.argv[idx + 1]
            del sys.argv[idx : idx + 2]
    return mode


if __name__ == "__main__":
    selected_mode = _extract_mode()
    run_solver(lambda model, samples, seed: solve(model, samples, seed, mode=selected_mode))
