"""Qiskit circuits: Bayesian-network state preparation and payoff rotations.

State preparation uses uniformly controlled Ry rotations (UCRY, Mottonen et al.
2004): each register is loaded bit by bit, most significant bit first, with the
rotation angle multiplexed on the parent registers and the bits already loaded.
The payoff is one more UCRY that rotates an objective qubit by
2*arcsin(sqrt(f(x))), so P(objective = 1) = sum_x p(x) f(x) = the target.
"""

from __future__ import annotations

import hashlib
import io

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.circuit.library import UCRYGate

from .loader import ORDER, Node, layout


def _ucry(circuit: QuantumCircuit, angles: np.ndarray, target: int, controls: list[int]):
    angles = [float(x) for x in angles]
    if not controls:
        if angles[0]:
            circuit.ry(angles[0], target)
        return
    if not any(angles):
        return
    circuit.append(UCRYGate(angles), [target, *controls])


def _load_register(circuit, node: Node, parent_qubits: list[int], target_qubits: list[int]):
    m = node.num_qubits
    n_parent = len(parent_qubits)
    for level in range(m):
        pos = m - 1 - level  # load MSB first
        higher = list(range(m - 1, pos, -1))  # already-loaded bit positions
        controls = parent_qubits + [target_qubits[h] for h in higher]
        angles = np.zeros(2 ** len(controls))
        for c_index in range(2**n_parent):
            row = node.table[c_index]
            for prefix in range(2 ** len(higher)):
                # prefix bit j corresponds to bit position higher[j]
                mask_val = sum(((prefix >> j) & 1) << h for j, h in enumerate(higher))
                values = np.arange(2**m)
                hi_mask = sum(1 << h for h in higher)
                match = (values & hi_mask) == mask_val
                total = row[match].sum()
                if total <= 0:
                    continue
                one = row[match & (((values >> pos) & 1) == 1)].sum()
                angles[c_index + (prefix << n_parent)] = 2 * np.arcsin(np.sqrt(min(1.0, one / total)))
        _ucry(circuit, angles, target_qubits[pos], controls)


def state_preparation(nodes: dict[str, Node]) -> QuantumCircuit:
    regs = layout(nodes)
    n = sum(len(r) for r in regs.values())
    circuit = QuantumCircuit(n, name="P_load")
    for name in ORDER:
        node = nodes[name]
        parents = [q for par in node.parents for q in regs[par]]
        _load_register(circuit, node, parents, regs[name])
    return circuit


def a_operator(nodes: dict[str, Node], payoff: np.ndarray, payoff_qubits: list[int]) -> QuantumCircuit:
    """A = payoff rotation o state preparation. The last qubit is the objective.

    payoff[i] in [0, 1] is f for control index i over `payoff_qubits`
    (little-endian in the order given).
    """
    load = state_preparation(nodes)
    n = load.num_qubits
    circuit = QuantumCircuit(n + 1, name="A")
    circuit.compose(load, range(n), inplace=True)
    payoff = np.clip(payoff, 0.0, 1.0)
    _ucry(circuit, 2 * np.arcsin(np.sqrt(payoff)), n, list(payoff_qubits))
    return circuit


def circuit_sha256(circuit: QuantumCircuit) -> str:
    buffer = io.BytesIO()
    qpy.dump(circuit, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def statevector(circuit: QuantumCircuit) -> np.ndarray:
    """Simulate A on Aer's statevector simulator and return the full 2^n statevector."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    sim = AerSimulator(method="statevector")
    qc = circuit.copy()
    qc.save_statevector()
    result = sim.run(transpile(qc, sim, optimization_level=0)).result()
    return np.asarray(result.get_statevector(qc), dtype=np.complex128)


def objective_mask(num_qubits: int) -> np.ndarray:
    """Boolean mask selecting basis states where the objective (MSB) qubit is 1."""
    idx = np.arange(1 << num_qubits)
    return ((idx >> (num_qubits - 1)) & 1) == 1


def objective_probability(circuit: QuantumCircuit) -> float:
    """Simulate A on Aer's statevector simulator and return P(objective = 1)."""
    psi = statevector(circuit)
    mask = objective_mask(circuit.num_qubits)
    return float(np.sum(np.abs(psi[mask]) ** 2))


def factorized_a_operator(
    nodes: dict[str, Node], f_grid: np.ndarray, gammas: dict[str, float] | None
) -> tuple[QuantumCircuit, float]:
    """Factorised tilted A operator using per-node likelihood-ratio ancillas."""
    load = state_preparation(nodes)
    regs = layout(nodes)
    n_state = load.num_qubits

    active_tilts = [name for name in ORDER if gammas and gammas.get(name, 0.0) != 0.0]
    n_anc = 1 + len(active_tilts)
    n_total = n_state + n_anc + 1
    obj_q = n_total - 1
    f_anc = n_state
    tilt_ancs = {name: n_state + 1 + idx for idx, name in enumerate(active_tilts)}

    circuit = QuantumCircuit(n_total, name="A_fact")
    circuit.compose(load, range(n_state), inplace=True)

    max_f = float(f_grid.max())
    if max_f <= 0.0:
        return circuit, 0.0
    payoff_qubits = regs["a"] + regs["b"] + regs["r1"] + regs["r2"]
    idx = np.arange(256)
    av, bv, r1, r2 = idx & 7, (idx >> 3) & 7, (idx >> 6) & 1, (idx >> 7) & 1
    ok = (av < 5) & (bv < 5)
    f_norm = np.where(ok, f_grid[np.minimum(av, 4), np.minimum(bv, 4), r1, r2], 0.0) / max_f
    _ucry(circuit, 2 * np.arcsin(np.sqrt(np.clip(f_norm, 0.0, 1.0))), f_anc, payoff_qubits)

    W_fact = max_f
    for name in active_tilts:
        node = nodes[name]
        p = node.meta["p_table"]
        q = node.table
        ratio = np.zeros_like(p)
        valid = q > 0
        ratio[valid] = p[valid] / q[valid]
        max_r = float(ratio[: (5 if node.parents == ("s",) else ratio.shape[0])].max())
        if max_r <= 0.0:
            max_r = 1.0
        W_fact *= max_r
        r_norm = np.clip(ratio / max_r, 0.0, 1.0)
        parent_qs = [q_idx for par in node.parents for q_idx in regs[par]]
        controls = regs[name] + parent_qs
        n_v = 1 << node.num_qubits
        n_p = 1 << len(parent_qs)
        angles = np.zeros(n_v * n_p)
        for p_idx in range(n_p):
            for v_idx in range(n_v):
                angles[v_idx + (p_idx << node.num_qubits)] = 2.0 * np.arcsin(
                    np.sqrt(r_norm[p_idx, v_idx])
                )
        _ucry(circuit, angles, tilt_ancs[name], controls)

    and_controls = [f_anc] + [tilt_ancs[name] for name in active_tilts]
    circuit.mcx(and_controls, obj_q)
    return circuit, float(W_fact)
