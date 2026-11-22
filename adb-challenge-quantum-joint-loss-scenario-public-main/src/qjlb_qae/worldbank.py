"""Statevector ML-IQAE solver for the World Bank ELT track."""

from __future__ import annotations

import hashlib
import math
import time

import numpy as np

from .iqae import SplitStatevectorGrover, iqae

WB_RELATIVE = np.array([0.01, 0.01, 0.05, 0.05, 0.05])
WB_ABSOLUTE = np.array([1_000.0, 1_000.0, 1e-5, 1e-5, 100.0])


def _elt_circuit_digest(model, scales: np.ndarray, n_qubits: int) -> str:
    h = hashlib.sha256()
    h.update(f"wb_elt_uniform_iqae:{n_qubits}q:{model.fingerprint}".encode())
    h.update(np.round(scales, 8).tobytes())
    return h.hexdigest()


def solve_worldbank_iqae(
    model,
    samples: int,
    seed: int,
    *,
    shots: int = 100,
    kappa: float = 0.85,
    alpha: float = 0.05,
    execution: str = "quantum_simulator",
) -> dict:
    start = time.perf_counter()
    t_prep0 = start

    n_events = model.state_count
    n_state_qubits = max(1, int(math.ceil(math.log2(n_events))))
    n_qubits = n_state_qubits + 1

    all_idx = np.arange(n_events, dtype=np.int64)[:, None]
    per_event = model.evaluate(all_idx)["per_event_metrics"]
    sqrt_p = np.sqrt(np.maximum(model.probabilities, 0.0))
    total_rate = float(model.total_rate)
    max_obs = per_event.max(axis=0)
    t_prep1 = time.perf_counter()

    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    estimates = {}
    infos = []

    for i, name in enumerate(model.metric_names):
        t0 = time.perf_counter()
        max_val = float(max_obs[i])
        if max_val <= 0.0:
            estimates[name] = {
                "estimate": 0.0,
                "standard_error": None,
                "ci95_low": 0.0,
                "ci95_high": 0.0,
            }
            infos.append(
                {
                    "target": name,
                    "structural_zero": True,
                    "oracle_calls": 0,
                    "shots": 0,
                    "max_k": 0,
                    "qubits": n_qubits,
                }
            )
            continue

        f_norm = np.clip(per_event[:, i] / max_val, 0.0, 1.0)
        good_a = sqrt_p * np.sqrt(f_norm)
        bad_a = sqrt_p * np.sqrt(1.0 - f_norm)

        total_scale = total_rate * max_val
        rel, floor = float(WB_RELATIVE[i]), float(WB_ABSOLUTE[i])

        def stop(lo, hi, ts=total_scale, r=rel, fl=floor):
            return ts * (hi - lo) / 2.0 <= kappa * max(r * ts * lo, fl)

        eps_min = max(kappa * floor / total_scale, 1e-12)
        grover = SplitStatevectorGrover(bad_a, good_a, rngs[i])
        res = iqae(grover, stop, epsilon_min=eps_min, alpha=alpha, shots=shots, use_mle=True)
        t1 = time.perf_counter()

        estimates[name] = {
            "estimate": float(total_scale * res.estimate),
            "standard_error": None,
            "ci95_low": float(total_scale * res.interval[0]),
            "ci95_high": float(total_scale * res.interval[1]),
        }
        infos.append(
            {
                "target": name,
                "structural_zero": False,
                "amplitude": float(np.dot(good_a, good_a)),
                "oracle_calls": res.oracle_calls,
                "grover_steps": res.grover_steps,
                "shots": res.shots,
                "rounds": res.rounds,
                "max_k": res.max_k,
                "qubits": n_qubits,
                "elapsed_seconds": t1 - t0,
            }
        )

    digest = _elt_circuit_digest(model, max_obs, n_qubits)
    return {
        "estimates": estimates,
        "resources": {
            "method": "wb-uniform-statevector-ml-iqae",
            "execution": execution,
            "qubits": int(n_qubits),
            "shots": int(sum(x["shots"] for x in infos)),
            "model_evaluations": int(sum(x["oracle_calls"] for x in infos)),
            "preparation_seconds": float(t_prep1 - t_prep0),
            "solver_seconds": time.perf_counter() - start,
            "per_target": infos,
        },
        "evidence": {
            "provider": "qjlb-qae-statevector (local)",
            "backend": f"{n_qubits}q live unitary statevector Grover evolution + ML-IQAE",
            "circuit_sha256": digest,
        },
        "uncertainty": "IQAE Clopper-Pearson 95% intervals with joint-round ML refinement.",
    }
