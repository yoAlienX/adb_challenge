"""Gate-level resources of the A operators and of the deepest Grover circuit used.

Transpiles each target's A circuit to {cx, u} (all-to-all connectivity) and
reports qubits, CX count and depth for A, for one Grover iteration Q, and for
the deepest circuit IQAE actually ran (Q^k_max A), using k_max from a solve.

    python benchmarks/qae_resources.py --config scenarios/adb-base.json
"""

import argparse
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import grover_operator

from qjlb.cli import load_model
from qjlb_qae.solver import build_target_circuit, payoffs, solve

SHORT = ["E[L]", "P(both pay)", "P(L>T)", "E[(L-T)+]", "P(both exhausted)"]


def counts(circuit):
    qc = transpile(circuit, basis_gates=["cx", "u"], optimization_level=1)
    return qc.count_ops().get("cx", 0), qc.depth()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scenarios/adb-base.json")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    model, _ = load_model(Path(args.config))
    f_grids, scales = payoffs(model)
    run = solve(model, 8192, args.seed)["resources"]["per_target"]
    print(f"{'target':18s} {'tilt':>5s} {'qubits':>6s} {'CX(A)':>7s} {'depth(A)':>8s} {'CX(Q)':>7s} "
          f"{'k_max':>5s} {'CX(Q^k A)':>10s} {'depth(Q^k A)':>12s} {'oracle calls':>12s}")
    for i, label in enumerate(SHORT):
        if run[i].get("structural_zero"):
            print(f"{label:18s} structural zero (no circuit)")
            continue
        a_circ, _, gammas = build_target_circuit(model, f_grids[i])
        cx_a, depth_a = counts(a_circ)
        oracle = QuantumCircuit(a_circ.num_qubits)
        oracle.z(a_circ.num_qubits - 1)
        cx_q, depth_q = counts(grover_operator(oracle, state_preparation=a_circ))
        k = run[i]["max_k"]
        print(f"{label:18s} {'yes' if gammas else 'no':>5s} {a_circ.num_qubits:>6d} {cx_a:>7,d} {depth_a:>8,d} {cx_q:>7,d} "
              f"{k:>5d} {cx_a + k * cx_q:>10,d} {depth_a + k * depth_q:>12,d} {run[i]['oracle_calls']:>12,d}")


if __name__ == "__main__":
    main()
