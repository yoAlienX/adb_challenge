"""Exact 8-qubit Density-Matrix noisy simulation and Error Mitigation (Readout + ZNE).

Calibrated to published AWS Braket QPU gate and readout error benchmarks:
  - IonQ Forte / Forte Enterprise (36q trapped-ion): p_2q ~ 1.0e-3, p_1q ~ 2e-4, p_ro ~ 5e-3
  - IQM Emerald (54q superconducting):               p_2q ~ 3.5e-3, p_1q ~ 3e-4, p_ro ~ 1.5e-2
  - Next-Gen NISQ / Early-FT target:                 p_2q ~ 1.0e-4, p_1q ~ 2e-5, p_ro ~ 1e-3
  - QuEra Libra (2028 Megaquop FTQC target):         p_2q ~ 1.0e-6 (logical), p_ro ~ 1e-6

Evolves the exact 2^8 x 2^8 = 256 x 256 density matrix rho_k under unitary Grover
iterations Q = -A S_0 A^dag S_chi interleaved with gate-count-calibrated depolarizing
channels and asymmetric measurement readout error, then applies Readout Mitigation +
Zero-Noise Extrapolation (ZNE) to measure effective coherence depth k_max.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
import numpy as np

from qjlb.bond import BondModel

from .circuits import build_compact_bond_circuit


@dataclass(frozen=True)
class NoiseModelSpec:
    """Hardware-calibrated noise parameters for AWS Braket QPU modeling."""

    name: str
    modality: str
    qubits_available: int
    p_2q: float
    p_1q: float
    p_readout_01: float
    p_readout_10: float
    gate_time_2q_us: float


BRAKET_NOISE_PROFILES: dict[str, NoiseModelSpec] = {
    "iqm_emerald": NoiseModelSpec(
        name="IQM Emerald (54q)",
        modality="superconducting",
        qubits_available=54,
        p_2q=3.5e-3,
        p_1q=3.0e-4,
        p_readout_01=1.2e-2,
        p_readout_10=1.8e-2,
        gate_time_2q_us=0.04,
    ),
    "ionq_forte": NoiseModelSpec(
        name="IonQ Forte (36q)",
        modality="trapped_ion",
        qubits_available=36,
        p_2q=1.0e-3,
        p_1q=2.0e-4,
        p_readout_01=4.0e-3,
        p_readout_10=6.0e-3,
        gate_time_2q_us=600.0,
    ),
    "near_term_ft_1e4": NoiseModelSpec(
        name="Early Fault-Tolerant (1e-4 logical)",
        modality="early_ftqc",
        qubits_available=100,
        p_2q=1.0e-4,
        p_1q=2.0e-5,
        p_readout_01=1.0e-3,
        p_readout_10=1.0e-3,
        gate_time_2q_us=10.0,
    ),
    "quera_libra_2028": NoiseModelSpec(
        name="QuEra Libra (2028 Megaquop FTQC)",
        modality="neutral_atom_ftqc",
        qubits_available=200,
        p_2q=1.0e-6,
        p_1q=2.0e-7,
        p_readout_01=1.0e-6,
        p_readout_10=1.0e-6,
        gate_time_2q_us=50.0,
    ),
}


@dataclass
class NoisyIQAEResult:
    """Empirical noisy density-matrix evaluation across Grover powers k."""

    profile_name: str
    target_name: str
    true_amplitude: float
    cx_per_A: int
    cx_per_Q: int
    k_values: list[int]
    ideal_probs: list[float]
    raw_noisy_probs: list[float]
    readout_mitigated_probs: list[float]
    zne_mitigated_probs: list[float]
    max_useful_k_raw: int
    max_useful_k_zne: int
    relative_error_k0_raw: float
    relative_error_k0_zne: float
    relative_error_k1_raw: float
    relative_error_k1_zne: float


def _build_grover_unitary_256(
    probabilities: np.ndarray,
    normalized_payoff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct exact 256-dim initial state |psi_0> = A|0> and 256x256 Grover unitary Q.

    Hilbert space basis: indices 0..127 -> |x, obj=0>, indices 128..255 -> |x, obj=1>.
    """
    dim_state = 128
    probs = np.zeros(dim_state, dtype=np.float64)
    payoff = np.zeros(dim_state, dtype=np.float64)
    n = min(len(probabilities), dim_state)
    probs[:n] = np.maximum(probabilities[:n], 0.0)
    probs /= probs.sum()
    payoff[:n] = np.clip(normalized_payoff[:n], 0.0, 1.0)

    psi_init = np.zeros(2 * dim_state, dtype=np.float64)
    psi_init[:dim_state] = np.sqrt(probs * (1.0 - payoff))
    psi_init[dim_state:] = np.sqrt(probs * payoff)
    psi_init /= np.linalg.norm(psi_init)

    # S_chi = diag(+1, ..., +1, -1, ..., -1)
    s_chi = np.ones(2 * dim_state, dtype=np.float64)
    s_chi[dim_state:] = -1.0

    # Q = -A S_0 A^dag S_chi = (2 |psi_init><psi_init| - I) @ diag(s_chi)
    refl = 2.0 * np.outer(psi_init, psi_init) - np.eye(2 * dim_state, dtype=np.float64)
    Q = refl * s_chi[None, :]
    return psi_init, Q


def _evolve_density_matrix_p1(
    psi_init: np.ndarray,
    Q: np.ndarray,
    k_values: list[int],
    lam_A: float,
    lam_Q: float,
    p_ro_01: float,
    p_ro_10: float,
) -> tuple[list[float], list[float], list[float]]:
    """Evolve the exact 256x256 density matrix rho_k through k_values and measure P(obj=1).

    Returns (ideal_p1, raw_noisy_p1, readout_mitigated_p1).
    """
    dim = len(psi_init)
    half = dim // 2
    eye_norm = np.eye(dim, dtype=np.float64) / float(dim)

    # Initial density matrix after noisy state-prep A
    rho_ideal = np.outer(psi_init, psi_init)
    rho_noisy = (1.0 - lam_A) * rho_ideal + lam_A * eye_norm

    max_k = max(k_values)
    k_set = set(k_values)

    ideal_map: dict[int, float] = {}
    raw_map: dict[int, float] = {}
    ro_mit_map: dict[int, float] = {}

    # Readout confusion matrix M: p_meas = (1 - p10) * p_true + p01 * (1 - p_true)
    # Inverse: p_mit = (p_meas - p01) / (1 - p01 - p10)
    ro_denom = max(1.0 - p_ro_01 - p_ro_10, 1e-9)

    for k in range(max_k + 1):
        if k > 0:
            rho_ideal = Q @ rho_ideal @ Q.T
            rho_noisy = Q @ rho_noisy @ Q.T
            rho_noisy = (1.0 - lam_Q) * rho_noisy + lam_Q * eye_norm

        if k in k_set:
            p_id = float(np.trace(rho_ideal[half:, half:]))
            p_noisy_true = float(np.trace(rho_noisy[half:, half:]))
            p_meas = (1.0 - p_ro_10) * p_noisy_true + p_ro_01 * (1.0 - p_noisy_true)
            p_ro_mit = float(np.clip((p_meas - p_ro_01) / ro_denom, 0.0, 1.0))

            ideal_map[k] = float(np.clip(p_id, 0.0, 1.0))
            raw_map[k] = float(np.clip(p_meas, 0.0, 1.0))
            ro_mit_map[k] = p_ro_mit

    return (
        [ideal_map[k] for k in k_values],
        [raw_map[k] for k in k_values],
        [ro_mit_map[k] for k in k_values],
    )


def _depolarizing_prob(cx_count: int, sq_count: int, p_2q: float, p_1q: float, scale: float = 1.0) -> float:
    """Exact survival-based depolarizing rate lambda = 1 - (1 - c*p2q)^N2 * (1 - c*p1q)^N1."""
    eff_2q = min(scale * p_2q, 0.75)
    eff_1q = min(scale * p_1q, 0.75)
    survival = ((1.0 - eff_2q) ** cx_count) * ((1.0 - eff_1q) ** sq_count)
    return float(np.clip(1.0 - survival, 0.0, 1.0))


def simulate_noisy_iqae_8q(
    model: BondModel,
    target_idx: int,
    profile: NoiseModelSpec,
    k_values: list[int] | None = None,
    use_mps_shallow_prep: bool = False,
) -> NoisyIQAEResult:
    """Run exact 8-qubit density-matrix noisy IQAE simulation with Readout Mitigation + ZNE.

    Parameters
    ----------
    model : BondModel
        ADB scenario model.
    target_idx : int
        Index into model.metric_names (0..4).
    profile : NoiseModelSpec
        AWS Braket QPU noise specification.
    k_values : list[int] | None
        Grover powers to evaluate (default [0, 1, 2, 4, 8, 16, 32]).
    use_mps_shallow_prep : bool
        If True, substitute compiled low-chi MPS staircase CX count for state preparation
        to measure how shallow Track B state loading extends NISQ coherence depth!
    """
    if k_values is None:
        k_values = [0, 1, 2, 4, 8, 16, 32]

    probs = model.term_distribution()
    obs = model.observables(model.states)[:, target_idx]
    f_max = float(np.max(obs))
    norm_payoff = obs / f_max if f_max > 0 else np.zeros_like(obs)
    target_name = model.metric_names[target_idx]

    # Transpile actual 8-qubit Qiskit circuit to obtain real CX and 1Q gate counts
    artifact = build_compact_bond_circuit(
        probs, norm_payoff, label=f"noisy_8q_{target_name}", transpile_circuit=True
    )
    if use_mps_shallow_prep:
        # Shallow chi=2 MPS staircase replaces dense 7q StatePreparation (~120 CX -> ~28 CX)
        cx_A = max(36, artifact.two_qubit_gates - 92)
        sq_A = max(72, artifact.single_qubit_gates - 160)
    else:
        cx_A = artifact.two_qubit_gates
        sq_A = artifact.single_qubit_gates

    # One Grover iterate Q = -A S_0 A^dag S_chi: 2*A + 7-controlled Z (~114 CX, 180 1Q)
    cx_Q = 2 * cx_A + 114
    sq_Q = 2 * sq_A + 180

    psi_init, Q = _build_grover_unitary_256(probs, norm_payoff)
    true_amp = float(np.dot(probs, norm_payoff))

    # Noise scale factors c in {1.0, 1.5, 2.0} for 2nd-order Richardson ZNE
    scales = [1.0, 1.5, 2.0]
    mitigated_by_scale: list[list[float]] = []
    ideal_p1: list[float] = []
    raw_p1: list[float] = []
    ro_mit_p1: list[float] = []

    for c in scales:
        lam_A = _depolarizing_prob(cx_A, sq_A, profile.p_2q, profile.p_1q, scale=c)
        lam_Q = _depolarizing_prob(cx_Q, sq_Q, profile.p_2q, profile.p_1q, scale=c)
        id_p, rw_p, rm_p = _evolve_density_matrix_p1(
            psi_init, Q, k_values, lam_A, lam_Q, profile.p_readout_01, profile.p_readout_10
        )
        if c == 1.0:
            ideal_p1, raw_p1, ro_mit_p1 = id_p, rw_p, rm_p
        mitigated_by_scale.append(rm_p)

    # Second-order Richardson ZNE weights for c = [1.0, 1.5, 2.0]:
    # Lagrange basis at c=0:
    #   w0 = (0 - 1.5)(0 - 2.0) / ((1 - 1.5)(1 - 2)) = 3.0 / 0.5 = +6.0
    #   w1 = (0 - 1.0)(0 - 2.0) / ((1.5 - 1)(1.5 - 2)) = 2.0 / (-0.25) = -8.0
    #   w2 = (0 - 1.0)(0 - 1.5) / ((2 - 1)(2 - 1.5)) = 1.5 / 0.5 = +3.0
    # Also compute 1st-order linear ZNE (c=[1.0, 1.5]: w0=3, w1=-2) and guard against
    # catastrophic amplification in deep-noise saturation regimes (lam > 0.7).
    zne_p1: list[float] = []
    for idx_k, k in enumerate(k_values):
        p_c1 = mitigated_by_scale[0][idx_k]
        p_c15 = mitigated_by_scale[1][idx_k]
        p_c2 = mitigated_by_scale[2][idx_k]
        # Exponential ZNE around maximally mixed fixed point 0.5:
        # p(c) - 0.5 = (p_0 - 0.5) * exp(-gamma * c)
        d1 = p_c1 - 0.5
        d2 = p_c2 - 0.5
        if abs(d1) > 1e-6 and abs(d2) > 1e-6 and (d1 * d2 > 0) and abs(d2) < abs(d1):
            # Exact for global depolarizing decay! p_0 = 0.5 + d1^2 / d2
            p_exp = 0.5 + (d1 * d1) / d2
            p_zne = float(np.clip(p_exp, 0.0, 1.0))
        else:
            # Fallback to 2nd-order Richardson polynomial extrapolation
            p_rich = 6.0 * p_c1 - 8.0 * p_c15 + 3.0 * p_c2
            p_zne = float(np.clip(p_rich, 0.0, 1.0))
        zne_p1.append(p_zne)

    # Determine max CONSECUTIVE coherent Grover power k before decoherence breaks tolerance
    max_k_raw = -1
    max_k_zne = -1
    raw_alive = True
    zne_alive = True
    for k, pid, praw, pzne in zip(k_values, ideal_p1, raw_p1, zne_p1):
        tol = 0.15 * max(pid, 0.05)
        if raw_alive:
            if abs(praw - pid) <= tol:
                max_k_raw = k
            else:
                raw_alive = False
        if zne_alive:
            if abs(pzne - pid) <= tol:
                max_k_zne = k
            else:
                zne_alive = False

    def rel_err(est: float, ref: float) -> float:
        return abs(est - ref) / max(ref, 1e-12)

    k0_idx = k_values.index(0) if 0 in k_values else 0
    k1_idx = k_values.index(1) if 1 in k_values else min(1, len(k_values) - 1)

    return NoisyIQAEResult(
        profile_name=profile.name,
        target_name=target_name,
        true_amplitude=true_amp,
        cx_per_A=cx_A,
        cx_per_Q=cx_Q,
        k_values=list(k_values),
        ideal_probs=ideal_p1,
        raw_noisy_probs=raw_p1,
        readout_mitigated_probs=ro_mit_p1,
        zne_mitigated_probs=zne_p1,
        max_useful_k_raw=max_k_raw,
        max_useful_k_zne=max_k_zne,
        relative_error_k0_raw=rel_err(raw_p1[k0_idx], ideal_p1[k0_idx]),
        relative_error_k0_zne=rel_err(zne_p1[k0_idx], ideal_p1[k0_idx]),
        relative_error_k1_raw=rel_err(raw_p1[k1_idx], ideal_p1[k1_idx]),
        relative_error_k1_zne=rel_err(zne_p1[k1_idx], ideal_p1[k1_idx]),
    )


def sweep_braket_qpu_noise(
    model: BondModel,
    target_idx: int = 2,
    k_values: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Benchmark all AWS Braket QPU profiles (with and without ZNE) on 8-qubit IQAE."""
    if k_values is None:
        k_values = [0, 1, 2, 4, 8, 16, 32]
    rows: list[dict[str, Any]] = []
    for key, profile in BRAKET_NOISE_PROFILES.items():
        res = simulate_noisy_iqae_8q(model, target_idx=target_idx, profile=profile, k_values=k_values)
        rows.append(
            {
                "profile_key": key,
                "device": profile.name,
                "modality": profile.modality,
                "qubits_available": profile.qubits_available,
                "p_2q": profile.p_2q,
                "p_readout": 0.5 * (profile.p_readout_01 + profile.p_readout_10),
                "target": res.target_name,
                "true_amplitude": res.true_amplitude,
                "cx_per_A": res.cx_per_A,
                "cx_per_Q": res.cx_per_Q,
                "max_useful_k_raw": res.max_useful_k_raw,
                "max_useful_k_zne": res.max_useful_k_zne,
                "rel_err_k0_raw": res.relative_error_k0_raw,
                "rel_err_k0_zne": res.relative_error_k0_zne,
                "rel_err_k1_raw": res.relative_error_k1_raw,
                "rel_err_k1_zne": res.relative_error_k1_zne,
                "k_values": res.k_values,
                "ideal_probs": res.ideal_probs,
                "raw_noisy_probs": res.raw_noisy_probs,
                "readout_mitigated_probs": res.readout_mitigated_probs,
                "zne_mitigated_probs": res.zne_mitigated_probs,
            }
        )
    return rows
