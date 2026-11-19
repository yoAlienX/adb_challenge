"""Runnable team example. Replace solve with your method."""

from qjlb.cli import methods_for
from qjlb.worker import run_solver


def solve(model, samples, seed):
    # This example is classical importance sampling.
    # Return estimates and resources in the same format as this method.
    # Quantum methods must also supply execution evidence; see EVALUATION.md.
    return methods_for(model)["importance"](model, samples, seed)


if __name__ == "__main__":
    run_solver(solve)
