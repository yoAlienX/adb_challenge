"""Tilted-IQAE solver for the ADB track: returns a qjlb-compatible result.

Pipeline for each of the five targets mu = E_p[f(x)]:
  1. Contract payoff f on the 5x5x2x2 term grid (from the contract terms only).
  2. Exponential tilt q of the Bayesian-network loader, chosen to minimise
     W = max over paths of f(x) p(x)/q(x). This needs only per-node likelihood
     ratios and a max over paths, never a probability-weighted sum, so it does
     not compute the answer. Since E_q[f p/q / W] = mu / W, a smaller W means a
     larger amplitude and fewer Grover iterations.
  3. Circuit A = (payoff rotation with f p/(q W)) o (tilted loader); 13 qubits.
  4. Aer statevector simulation of A gives its objective amplitude a.
  5. IQAE estimates a until the official evaluation-v1 tolerance is met with a
     safety factor; the estimate is mu = scale * W * a.
"""

from __future__ import annotations

import hashlib
import itertools
import time

import numpy as np

from .circuits import a_operator, circuit_sha256, objective_mask, statevector
from .iqae import AnalyticGrover, StatevectorGrover, iqae
from .loader import ORDER, apply_tilt, layout, network, register_values
from .mps import solve_adb_mps
from .worldbank import solve_worldbank_iqae

# evaluation-v1 tolerances for the ADB track (EVALUATION.md)
RELATIVE = np.array([0.01, 0.05, 0.05, 0.05, 0.10])
ABSOLUTE_FACTOR = np.array([1e-4, None, None, 1e-5, None], dtype=object)
ABSOLUTE_PROB = np.array([None, 1e-4, 1e-4, None, 1e-6], dtype=object)

GRID = {
    "s": (0.0, 1.0, 2.0, 3.0, 4.0),
    "ab": (0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
    "z": (0.0, 2.0, 4.0, 6.0, 8.0),
    "r": (0.0, 2.0, 4.0, 6.0, 8.0),
}


def absolute_floor(model, i):
    principal = float(model.principal.sum())
    return principal * ABSOLUTE_FACTOR[i] if ABSOLUTE_FACTOR[i] is not None else ABSOLUTE_PROB[i]


def payoffs(model):
    """f_tilde on the term grid (a, b, r1, r2) and the scale with mu = scale * E[f_tilde]."""
    grid = np.array(list(np.ndindex(5, 5, 2, 2)))
    obs = model.observables(grid)  # (100, 5): L, joint, exceed, stop-loss, both exhausted
    principal = float(model.principal.sum())
    scales = np.array([principal, 1.0, 1.0, max(principal - model.threshold, 0.0), 1.0])
    out = []
    for i in range(5):
        values = obs[:, i] / scales[i] if scales[i] > 0 else np.zeros(100)
        out.append(np.clip(values, 0, 1).reshape(5, 5, 2, 2))
    return out, scales


def _log_ratio_tables(nodes, values):
    """For each node and gamma: log(p/q) on every loader path (vector of 2**12)."""
    regs = layout(nodes)
    parent_index = {}
    for name in ORDER:
        idx = np.zeros_like(values["s"])
        shift = 0
        for par in nodes[name].parents:
            idx = idx + (values[par] << shift)
            shift += len(regs[par])
        parent_index[name] = idx
    tables = {}
    for name in ORDER:
        p = nodes[name].table
        key = {"s": "s", "a": "ab", "b": "ab", "z": "z", "r1": "r", "r2": "r"}[name]
        support = np.arange(p.shape[1])
        for g in GRID[key]:
            log_z = np.log((p * np.exp(g * support)[None, :]).sum(axis=1))
            tables[name, g] = log_z[parent_index[name]] - g * values[name]
    return tables


def select_tilt(nodes, values, f_path, valid):
    """Grid search for the exponential tilt minimising W = max f * p/q."""
    tables = _log_ratio_tables(nodes, values)
    positive = valid & (f_path > 0)
    if not positive.any():
        return None, 0.0
    log_f = np.log(f_path[positive])
    best = (np.inf, None)
    for gs, gab, gz, gr in itertools.product(GRID["s"], GRID["ab"], GRID["z"], GRID["r"]):
        log_w = (tables["s", gs] + tables["a", gab] + tables["b", gab]
                 + tables["z", gz] + tables["r1", gr] + tables["r2", gr])
        W = float(np.exp((log_f + log_w[positive]).max()))
        if W < best[0] - 1e-15:
            best = (W, {"s": gs, "a": gab, "b": gab, "z": gz, "r1": gr, "r2": gr})
    return best[1], best[0]


def build_target_circuit(model, f_grid, use_tilt=True):
    """A operator for one target. Returns (circuit, W, gammas or None).

    The target is mu = scale * W * P(objective = 1).
    """
    nodes = network(model)
    values = register_values(nodes)
    valid = (values["s"] < 5) & (values["a"] < 5) & (values["b"] < 5)
    a_c, b_c = np.minimum(values["a"], 4), np.minimum(values["b"], 4)
    f_path = np.where(valid, f_grid[a_c, b_c, values["r1"], values["r2"]], 0.0)

    gammas, W = (select_tilt(nodes, values, f_path, valid) if use_tilt else (None, None))
    tilted = bool(gammas) and any(gammas.values())
    if tilted:
        tables = _log_ratio_tables(nodes, values)
        log_w = sum(tables[name, gammas[name]] for name in ORDER)
        apply_tilt(nodes, gammas)
        payoff = np.where(valid, f_path * np.exp(log_w) / W, 0.0)
        payoff_qubits = list(range(sum(n.num_qubits for n in nodes.values())))
    else:
        W = float(f_grid.max())
        regs = layout(nodes)
        payoff_qubits = regs["a"] + regs["b"] + regs["r1"] + regs["r2"]
        idx = np.arange(256)
        av, bv, r1, r2 = idx & 7, (idx >> 3) & 7, (idx >> 6) & 1, (idx >> 7) & 1
        ok = (av < 5) & (bv < 5)
        payoff = np.where(ok, f_grid[np.minimum(av, 4), np.minimum(bv, 4), r1, r2], 0.0) / W
    return a_operator(nodes, payoff, payoff_qubits), W, (gammas if tilted else None)


def estimate_target(model, i, f_grid, scale, rng, *, shots=100, kappa=1.0, alpha=0.05,
                    use_tilt=True, use_statevector=True, use_mle=True):
    """Estimate one target. Returns (estimate, interval, info dict)."""
    t0 = time.perf_counter()
    if scale <= 0 or not f_grid.any():
        return 0.0, (0.0, 0.0), {"structural_zero": True, "oracle_calls": 0, "shots": 0,
                                 "max_k": 0, "qubits": 0, "classical_seconds": 0.0,
                                 "circuit_seconds": 0.0, "iqae_seconds": 0.0}
    circuit, W, gammas = build_target_circuit(model, f_grid, use_tilt)
    t1 = time.perf_counter()
    psi_a = statevector(circuit)
    mask = objective_mask(circuit.num_qubits)
    amplitude = float(np.sum(np.abs(psi_a[mask]) ** 2))
    t2 = time.perf_counter()

    total_scale = scale * W
    rel, floor = RELATIVE[i], absolute_floor(model, i)

    def stop(lo, hi):
        return total_scale * (hi - lo) / 2 <= kappa * max(rel * total_scale * lo, floor)

    eps_min = max(kappa * floor / total_scale, 1e-12)
    sampler = StatevectorGrover(psi_a, mask, rng) if use_statevector else AnalyticGrover(amplitude, rng)
    res = iqae(sampler, stop, epsilon_min=eps_min, alpha=alpha, shots=shots, use_mle=use_mle)
    t3 = time.perf_counter()
    info = {
        "structural_zero": False,
        "tilt": gammas,
        "W": W,
        "amplitude": amplitude,
        "oracle_calls": res.oracle_calls,
        "grover_steps": res.grover_steps,
        "shots": res.shots,
        "rounds": res.rounds,
        "max_k": res.max_k,
        "qubits": circuit.num_qubits,
        "circuit_sha256": circuit_sha256(circuit),
        "classical_seconds": t1 - t0,
        "circuit_seconds": t2 - t1,
        "iqae_seconds": t3 - t2,
    }
    return total_scale * res.estimate, (total_scale * res.interval[0], total_scale * res.interval[1]), info


def solve(model, samples, seed, *, mode="tilted_iqae", **options):
    """Unified qjlb solver for ADB and World Bank tracks (modes: tilted_iqae, hybrid, mps)."""
    track = model.config.get("track")

    if mode == "mps":
        if track == "adb_drb":
            return solve_adb_mps(model, samples=samples, seed=seed)
        if track == "worldbank_elt":
            from qjlb.model import elt_exact
            res = elt_exact(model)
            res["resources"]["method"] = "causal-mps-elt-exact"
            res["resources"]["execution"] = "classical_cpu"
            res["resources"]["shots"] = 0
            res["resources"]["qubits"] = 0
            return res
        raise ValueError(f"Unsupported track: {track!r}")

    if track == "worldbank_elt":
        execution = "hybrid" if mode == "hybrid" else "quantum_simulator"
        wb_opts = {k: v for k, v in options.items() if k in ("shots", "kappa", "alpha")}
        return solve_worldbank_iqae(model, samples, seed, execution=execution, **wb_opts)

    if track != "adb_drb":
        raise ValueError(f"Unsupported track: {track!r}")

    start = time.perf_counter()
    f_grids, scales = payoffs(model)
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(6)]

    if mode == "hybrid":
        from qjlb.bond import bond_qmc
        classical_budget = min(max(int(samples) // 2, 4096), 8192)
        c_res = bond_qmc(model, classical_budget, int(rngs[5].integers(0, 2**31 - 1)))
        classical_indices = {0, 1}
    else:
        c_res = None
        classical_indices = set()

    estimates, infos = {}, []
    for i, name in enumerate(model.metric_names):
        if i in classical_indices and c_res is not None:
            estimates[name] = c_res["estimates"][name]
            infos.append(
                {
                    "target": name,
                    "routed_to": "classical_rqmc",
                    "structural_zero": False,
                    "oracle_calls": 0,
                    "shots": 0,
                    "max_k": 0,
                    "qubits": 0,
                    "classical_seconds": 0.0,
                    "circuit_seconds": 0.0,
                    "iqae_seconds": 0.0,
                }
            )
            continue
        value, interval, info = estimate_target(model, i, f_grids[i], scales[i], rngs[i], **options)
        info["target"] = name
        info["routed_to"] = "tilted_statevector_ml_iqae"
        infos.append(info)
        estimates[name] = {
            "estimate": float(value),
            "standard_error": None,
            "ci95_low": float(interval[0]),
            "ci95_high": float(interval[1]),
        }

    digest = hashlib.sha256("".join(i.get("circuit_sha256", "") for i in infos).encode()).hexdigest()
    q_calls = int(sum(i["oracle_calls"] for i in infos))
    c_evals = int(c_res["resources"]["model_evaluations"]) if c_res is not None else 0
    is_hybrid = mode == "hybrid"
    evidence = {
        "provider": "qiskit-aer (local)",
        "backend": "13q causal Bayesian-network live unitary statevector Grover evolution + ML-IQAE",
        "circuit_sha256": digest,
    }
    if is_hybrid:
        evidence["quantum_execution"] = "quantum_simulator"
    return {
        "estimates": estimates,
        "resources": {
            "method": "hybrid-rqmc-tilted-ml-iqae" if is_hybrid else "bn-tilted-statevector-ml-iqae",
            "execution": "hybrid" if is_hybrid else "quantum_simulator",
            "qubits": int(max(i["qubits"] for i in infos)),
            "shots": int(sum(i["shots"] for i in infos)),
            "model_evaluations": q_calls + c_evals,
            "quantum_oracle_calls": q_calls,
            "classical_samples": c_evals,
            "preparation_seconds": float(sum(i["classical_seconds"] + i["circuit_seconds"] for i in infos)),
            "solver_seconds": time.perf_counter() - start,
            "per_target": infos,
        },
        "evidence": evidence,
        "uncertainty": (
            "Hybrid: RQMC t-intervals for bulk targets + IQAE Clopper-Pearson 95% intervals with ML refinement for tail targets."
            if is_hybrid
            else "IQAE Clopper-Pearson intervals at 95% joint confidence with joint-round ML refinement."
        ),
    }
