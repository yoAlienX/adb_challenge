"""Depolarizing + readout noise model and Richardson ZNE for Braket QPU profiles."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class QPUProfile:
    name: str
    two_qubit_error: float
    readout_error_0to1: float
    readout_error_1to0: float


BRAKET_PROFILES = {
    "iqm_emerald": QPUProfile("IQM Emerald", 4.0e-3, 1.5e-2, 2.5e-2),
    "ionq_forte": QPUProfile("IonQ Forte", 1.5e-3, 4.0e-3, 6.0e-3),
    "quera_libra_2028": QPUProfile("QuEra Libra (2028 FT)", 1.0e-5, 1.0e-4, 1.0e-4),
}


def noisy_objective_probability(
    ideal_prob: float,
    cx_count: int,
    profile: QPUProfile,
    noise_scale: float = 1.0,
) -> float:
    p2 = min(profile.two_qubit_error * noise_scale, 0.75)
    fidelity = (1.0 - p2) ** max(cx_count, 0)
    p_dep = fidelity * ideal_prob + (1.0 - fidelity) * 0.5
    e01 = profile.readout_error_0to1
    e10 = profile.readout_error_1to0
    return float((1.0 - e10) * p_dep + e01 * (1.0 - p_dep))


def mitigate_readout(raw_prob: float, profile: QPUProfile) -> float:
    e01 = profile.readout_error_0to1
    e10 = profile.readout_error_1to0
    denom = 1.0 - e01 - e10
    if abs(denom) < 1e-12:
        return raw_prob
    return float(min(max((raw_prob - e01) / denom, 0.0), 1.0))


def zne_richardson(
    values: tuple[float, float, float],
    scales: tuple[float, float, float] = (1.0, 3.0, 5.0),
) -> float:
    c1, c2, c3 = scales
    v1, v2, v3 = values
    w1 = (c2 * c3) / ((c2 - c1) * (c3 - c1))
    w2 = (c1 * c3) / ((c1 - c2) * (c3 - c2))
    w3 = (c1 * c2) / ((c1 - c3) * (c2 - c3))
    return float(min(max(w1 * v1 + w2 * v2 + w3 * v3, 0.0), 1.0))


def evaluate_qpu_coherence(
    amplitude: float,
    cx_a: int,
    cx_q: int,
    k_values: tuple[int, ...] = (0, 1, 2, 4, 8, 16),
) -> list[dict]:
    theta = math.asin(math.sqrt(min(max(amplitude, 0.0), 1.0)))
    rows = []
    for prof in BRAKET_PROFILES.values():
        for k in k_values:
            ideal = math.sin((2 * k + 1) * theta) ** 2
            cx_total = cx_a + k * cx_q
            raw = noisy_objective_probability(ideal, cx_total, prof, noise_scale=1.0)
            ro_only = mitigate_readout(raw, prof)
            p1 = mitigate_readout(noisy_objective_probability(ideal, cx_total, prof, 1.0), prof)
            p3 = mitigate_readout(noisy_objective_probability(ideal, cx_total, prof, 3.0), prof)
            p5 = mitigate_readout(noisy_objective_probability(ideal, cx_total, prof, 5.0), prof)
            zne = zne_richardson((p1, p3, p5))
            rows.append(
                {
                    "profile": prof.name,
                    "k": k,
                    "cx_total": cx_total,
                    "ideal": ideal,
                    "raw_err": abs(raw - ideal),
                    "readout_err": abs(ro_only - ideal),
                    "zne_err": abs(zne - ideal),
                }
            )
    return rows
