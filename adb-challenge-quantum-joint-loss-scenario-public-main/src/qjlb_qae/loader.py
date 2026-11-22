"""Latent-shock Bayesian-network loader for the ADB two-country bond model.

The joint term distribution p(a, b, r1, r2) is never computed classically.
Instead we use the model's conditional-independence structure:

    s  ~ number of shared earthquakes in the term (Poisson(c*T), >=4 lumped)
    a | s, b | s       capped quarter-steps per country, independent given s
    z  ~ any shared precipitation trigger in the term (Bernoulli)
    r1 | z, r2 | z     precipitation trigger per country, independent given z

Each node only needs a small conditional table (at most 5 x 5 numbers), built in
O(1) classical work per country. The circuit entangles the registers so that the
joint distribution appears in the amplitudes.

Two exact facts make this work for any term length T (both verified in tests):
  * Payout caps are absorbing and years are i.i.d. Poisson, so a T-year term is
    the annual process run for time T: every rate is simply multiplied by T.
  * Given s shared events, each country receives s independent severity marks on
    top of its own compound-Poisson events. Four or more shared events exhaust
    both countries' earthquake cover, so truncating s at 4 is exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm
from scipy.stats import poisson

EQ_LEVELS = 5  # 0..4 quarter steps
EQ_QUBITS = 3  # ceil(log2(5))


@dataclass
class Node:
    """One variable of the Bayesian network, stored in its own qubit register."""

    name: str
    num_qubits: int
    parents: tuple[str, ...]
    # table[parent_index, value] = P(value | parents). Parent index is the
    # little-endian concatenation of the parent registers in `parents` order.
    table: np.ndarray
    meta: dict = field(default_factory=dict)


def _own_steps(rate: float, marks: np.ndarray) -> np.ndarray:
    """Capped compound-Poisson quarter steps over a period with total `rate`."""
    generator = np.zeros((EQ_LEVELS, EQ_LEVELS))
    for state in range(EQ_LEVELS):
        for k, w in enumerate(marks, start=1):
            generator[state, min(state + k, 4)] += rate * w
        generator[state, state] -= rate
    out = expm(generator)[0]
    out = np.maximum(out, 0)
    return out / out.sum()


def _add_marks(dist: np.ndarray, marks: np.ndarray, count: int) -> np.ndarray:
    for _ in range(count):
        nxt = np.zeros(EQ_LEVELS)
        for state, p in enumerate(dist):
            for k, w in enumerate(marks, start=1):
                nxt[min(state + k, 4)] += p * w
        dist = nxt
    return dist


def _pad(vec: np.ndarray, num_qubits: int) -> np.ndarray:
    out = np.zeros(2**num_qubits)
    out[: len(vec)] = vec
    return out


def network(model, tilt: dict | None = None) -> dict[str, Node]:
    """Conditional tables of the term distribution, optionally importance-tilted.

    `tilt` maps node name -> gamma for exponential tilting (see apply_tilt).
    The untilted tables stay available in node.meta["p_table"]; the solver folds
    the likelihood ratios p/q into the payoff rotation.
    """
    T = model.dimension
    rates = model.earthquake_rates  # (2 countries, 4 step sizes), annual
    totals = rates.sum(axis=1)
    c = model.earthquake_common_rate
    marks = rates / totals[:, None]

    shock = np.array([poisson.pmf(n, c * T) for n in range(4)] + [poisson.sf(3, c * T)])
    nodes = {
        "s": Node("s", EQ_QUBITS, (), _pad(shock, EQ_QUBITS)[None, :], {"tail": 4}),
    }
    for name, i in (("a", 0), ("b", 1)):
        own = _own_steps((totals[i] - c) * T, marks[i])
        table = np.zeros((2**EQ_QUBITS, 2**EQ_QUBITS))
        for n in range(EQ_LEVELS):
            table[n] = _pad(_add_marks(own, marks[i], n), EQ_QUBITS)
        table[EQ_LEVELS:, 0] = 1.0  # unreachable parent states: any valid row
        nodes[name] = Node(name, EQ_QUBITS, ("s",), table, {"tail": 4})

    cfg = model.config
    q = np.asarray(cfg["precipitation_annual_trigger_probability"], dtype=float)
    rain = -np.log1p(-q) * float(cfg["precipitation_rate_multiplier"])
    common = float(cfg["precipitation_common_rate_share"]) * rain.min()
    p_z = 1.0 - np.exp(-common * T)
    nodes["z"] = Node("z", 1, (), np.array([[1 - p_z, p_z]]), {"tail": 1})
    for name, i in (("r1", 0), ("r2", 1)):
        p_own = 1.0 - np.exp(-(rain[i] - common) * T)
        table = np.array([[1 - p_own, p_own], [0.0, 1.0]])
        nodes[name] = Node(name, 1, ("z",), table, {"tail": 1})

    if tilt:
        apply_tilt(nodes, tilt)
    return nodes


def apply_tilt(nodes: dict[str, Node], gammas: dict[str, float]) -> dict[str, Node]:
    """Exponential tilting q(v | c) proportional to p(v | c) * exp(gamma * v).

    v is the node's value (quarter steps, shared-event count or trigger bit), so
    a positive gamma pushes probability toward bigger losses. The likelihood
    ratio of a full path is prod_nodes Z_node(c) * exp(-gamma_node * v_node).
    """
    for name, gamma in gammas.items():
        node = nodes[name]
        node.meta.setdefault("p_table", node.table.copy())
        if not gamma:
            continue
        p = node.meta["p_table"]
        weights = p * np.exp(gamma * np.arange(p.shape[1]))[None, :]
        node.table = weights / weights.sum(axis=1, keepdims=True)
    return nodes


ORDER = ("s", "a", "b", "z", "r1", "r2")


def layout(nodes: dict[str, Node]) -> dict[str, list[int]]:
    """Qubit indices of each register (little-endian within a register)."""
    out, start = {}, 0
    for name in ORDER:
        out[name] = list(range(start, start + nodes[name].num_qubits))
        start += nodes[name].num_qubits
    return out


def path_probabilities(nodes: dict[str, Node], use_p: bool = False) -> np.ndarray:
    """Probability of every basis state of the loader register (2**12 entries).

    Used only by tests and diagnostics to check the circuit; the solver never
    calls it. With use_p=True on a tilted network it returns the untilted p.
    """
    regs = layout(nodes)
    n = sum(len(r) for r in regs.values())
    idx = np.arange(2**n)
    values = {
        name: sum(((idx >> q) & 1) << j for j, q in enumerate(qs)) for name, qs in regs.items()
    }
    prob = np.ones(2**n)
    for name in ORDER:
        node = nodes[name]
        table = node.meta.get("p_table", node.table) if use_p else node.table
        parent = np.zeros(2**n, dtype=int)
        shift = 0
        for par in node.parents:
            parent += values[par] << shift
            shift += nodes[par].num_qubits
        prob *= table[parent, values[name]]
    return prob


def register_values(nodes: dict[str, Node]) -> dict[str, np.ndarray]:
    regs = layout(nodes)
    n = sum(len(r) for r in regs.values())
    idx = np.arange(2**n)
    return {
        name: sum(((idx >> q) & 1) << j for j, q in enumerate(qs)) for name, qs in regs.items()
    }
