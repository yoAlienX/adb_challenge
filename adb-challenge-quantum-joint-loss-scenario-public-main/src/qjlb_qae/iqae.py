"""Iterative Quantum Amplitude Estimation (Grinko, Gacon, Zoufal & Woerner 2021).

Given an operator A with A|0> = sqrt(1-a)|psi0>|0> + sqrt(a)|psi1>|1>, write
a = sin^2(theta). After k Grover iterations Q^k A, the objective qubit reads 1
with probability sin^2((2k+1) theta). IQAE picks k adaptively so that each
round's measurement pins theta into a shrinking interval, without quantum
phase estimation.

Two sampling backends implement "run Q^k A with N shots, count the ones":
  * AnalyticGrover: draws Binomial(N, sin^2((2k+1) theta)) with theta taken from
    the simulated circuit's amplitude. In the noiseless case this has exactly the
    same distribution as running the Grover circuit, because Q acts as a
    rotation by 2*theta in the 2-D subspace spanned by |psi0>|0>, |psi1>|1>
    (Brassard et al. 2002). It lets us use large k quickly.
  * AerGrover: builds and simulates the real Q^k A circuits on Qiskit Aer.
    Used in tests to check the analytic backend for small k.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import beta as beta_dist


@dataclass
class IQAEResult:
    estimate: float
    interval: tuple[float, float]
    oracle_calls: int  # applications of A (each Grover step uses A and A^dagger: 2k+1 per shot)
    grover_steps: int  # sum over shots of k
    shots: int
    rounds: int
    max_k: int
    powers: list[int] = field(default_factory=list)


class StatevectorGrover:
    """Statevector Grover evolution Q = (2|psi_A><psi_A| - I)(I - 2 P_good)."""

    def __init__(self, psi_a: np.ndarray, objective_mask: np.ndarray, rng: np.random.Generator):
        self.psi_a = np.asarray(psi_a, dtype=np.complex128)
        self.psi_a = self.psi_a / np.linalg.norm(self.psi_a)
        self.objective_mask = np.asarray(objective_mask, dtype=bool)
        self.rng = rng
        self._cur_k = 0
        self._cur_psi = self.psi_a.copy()

    def _evolve_to(self, k: int) -> np.ndarray:
        if k < self._cur_k:
            self._cur_k = 0
            self._cur_psi = self.psi_a.copy()
        psi = self._cur_psi
        psi_a = self.psi_a
        mask = self.objective_mask
        while self._cur_k < k:
            psi[mask] = -psi[mask]
            overlap = np.vdot(psi_a, psi)
            psi = 2.0 * overlap * psi_a - psi
            self._cur_k += 1
        self._cur_psi = psi
        return psi

    def probability(self, k: int) -> float:
        psi = self._evolve_to(k)
        return float(np.clip(np.sum(np.abs(psi[self.objective_mask]) ** 2), 0.0, 1.0))

    def __call__(self, k: int, shots: int) -> int:
        return int(self.rng.binomial(shots, self.probability(k)))


class SplitStatevectorGrover:
    """Real-amplitude statevector Grover evolution for large event spaces."""

    def __init__(self, bad_a: np.ndarray, good_a: np.ndarray, rng: np.random.Generator):
        self.bad_a = np.asarray(bad_a, dtype=np.float64)
        self.good_a = np.asarray(good_a, dtype=np.float64)
        norm = math.sqrt(float(np.dot(self.bad_a, self.bad_a) + np.dot(self.good_a, self.good_a)))
        if norm > 0:
            self.bad_a = self.bad_a / norm
            self.good_a = self.good_a / norm
        self.rng = rng
        self._cur_k = 0
        self._cur_bad = self.bad_a.copy()
        self._cur_good = self.good_a.copy()

    def _evolve_to(self, k: int) -> None:
        if k < self._cur_k:
            self._cur_k = 0
            self._cur_bad = self.bad_a.copy()
            self._cur_good = self.good_a.copy()
        b, g = self._cur_bad, self._cur_good
        ba, ga = self.bad_a, self.good_a
        while self._cur_k < k:
            g = -g
            overlap = float(np.dot(ba, b) + np.dot(ga, g))
            b = 2.0 * overlap * ba - b
            g = 2.0 * overlap * ga - g
            self._cur_k += 1
        self._cur_bad, self._cur_good = b, g

    def probability(self, k: int) -> float:
        self._evolve_to(k)
        return float(np.clip(np.dot(self._cur_good, self._cur_good), 0.0, 1.0))

    def __call__(self, k: int, shots: int) -> int:
        return int(self.rng.binomial(shots, self.probability(k)))


class AnalyticGrover:
    def __init__(self, amplitude: float, rng: np.random.Generator):
        self.theta = math.asin(math.sqrt(min(max(amplitude, 0.0), 1.0)))
        self.rng = rng

    def __call__(self, k: int, shots: int) -> int:
        return int(self.rng.binomial(shots, math.sin((2 * k + 1) * self.theta) ** 2))


class AerGrover:
    """Real Grover circuits on Aer (slow for large k; for verification)."""

    def __init__(self, a_circuit, seed: int = 0):
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import grover_operator
        from qiskit_aer import AerSimulator

        self.a_circuit = a_circuit
        objective = a_circuit.num_qubits - 1
        oracle = QuantumCircuit(a_circuit.num_qubits)
        oracle.z(objective)
        self.q = grover_operator(oracle, state_preparation=a_circuit)
        self.sim = AerSimulator(method="statevector", seed_simulator=seed)
        self.objective = objective

    def circuit(self, k: int):
        from qiskit import ClassicalRegister

        qc = self.a_circuit.copy()
        for _ in range(k):
            qc.compose(self.q, inplace=True)
        creg = ClassicalRegister(1)
        qc.add_register(creg)
        qc.measure(self.objective, creg[0])
        return qc

    def probability(self, k: int) -> float:
        """Exact P(objective = 1) after Q^k A, from the Aer statevector."""
        from qiskit import transpile

        qc = self.a_circuit.copy()
        for _ in range(k):
            qc.compose(self.q, inplace=True)
        qc.save_probabilities([self.objective])
        qc = transpile(qc, self.sim, optimization_level=0)
        return float(self.sim.run(qc).result().data()["probabilities"][1])

    def __call__(self, k: int, shots: int) -> int:
        from qiskit import transpile

        qc = transpile(self.circuit(k), self.sim, optimization_level=0)
        counts = self.sim.run(qc, shots=shots).result().get_counts()
        return int(counts.get("1", 0))


def _clopper_pearson(ones: int, shots: int, alpha: float) -> tuple[float, float]:
    lo = 0.0 if ones == 0 else float(beta_dist.ppf(alpha / 2, ones, shots - ones + 1))
    hi = 1.0 if ones == shots else float(beta_dist.ppf(1 - alpha / 2, ones + 1, shots - ones))
    return lo, hi


def _find_next_k(k: int, upper: bool, theta_l: float, theta_u: float, min_ratio: float = 2.0):
    """Largest K = 4k+2 keeping K*[theta_l, theta_u] inside one half circle.

    Angles here are in units of full turns: a = sin^2(2*pi*theta), theta in [0, 1/4].
    """
    old_scaling = 4 * k + 2
    max_scaling = int(1 / (2 * (theta_u - theta_l)))
    scaling = max_scaling - (max_scaling - 2) % 4
    while scaling >= min_ratio * old_scaling:
        tmin = scaling * theta_l - int(scaling * theta_l)
        tmax = scaling * theta_u - int(scaling * theta_u)
        if tmin <= tmax <= 0.5 and tmin <= 0.5:
            return (scaling - 2) // 4, True
        if tmax >= 0.5 and tmax >= tmin >= 0.5:
            return (scaling - 2) // 4, False
        scaling -= 4
    return k, upper


def _mle_refine(
    theta_lo: float, theta_hi: float, records: list[tuple[int, int, int]]
) -> float:
    """Joint Maximum Likelihood estimate of amplitude a = sin^2(2*pi*theta)
    inside the Clopper-Pearson bracket [theta_lo, theta_hi]."""
    if not records or theta_hi - theta_lo <= 1e-15:
        return 0.5 * (
            math.sin(2 * math.pi * theta_lo) ** 2 + math.sin(2 * math.pi * theta_hi) ** 2
        )

    def neg_log_lik(th: float) -> float:
        total = 0.0
        for k, n, y in records:
            p = math.sin((2 * k + 1) * 2 * math.pi * th) ** 2
            p = min(max(p, 1e-15), 1.0 - 1e-15)
            total -= y * math.log(p) + (n - y) * math.log(1.0 - p)
        return total

    res = minimize_scalar(neg_log_lik, bounds=(theta_lo, theta_hi), method="bounded")
    th_hat = float(res.x) if res.success else 0.5 * (theta_lo + theta_hi)
    return float(math.sin(2 * math.pi * th_hat) ** 2)


def iqae(sample, stop, epsilon_min: float, alpha: float = 0.05, shots: int = 100,
         min_ratio: float = 2.0, max_rounds_cap: int = 200, use_mle: bool = True) -> IQAEResult:
    """Run IQAE until stop(a_lo, a_hi) is true or the interval is narrower than epsilon_min.

    sample(k, shots) -> number of ones; alpha is split evenly over the worst-case
    number of rounds needed to reach epsilon_min (as in Grinko et al.).
    With use_mle=True, the point estimate is refined by joint maximum likelihood
    across all measured rounds inside the Clopper-Pearson confidence bracket.
    """
    max_rounds = int(math.log(min_ratio * math.pi / 8 / epsilon_min) / math.log(min_ratio)) + 1
    round_alpha = alpha / max_rounds
    powers = [0]
    upper = True
    theta = [0.0, 0.25]
    a_int = (0.0, 1.0)
    ones_hist: list[int] = []
    records: list[tuple[int, int, int]] = []
    calls = steps = total_shots = 0
    rounds = 0
    while not stop(*a_int) and theta[1] - theta[0] > epsilon_min / math.pi and rounds < max_rounds_cap:
        rounds += 1
        k, upper = _find_next_k(powers[-1], upper, theta[0], theta[1], min_ratio)
        powers.append(k)
        ones = sample(k, shots)
        ones_hist.append(ones)
        records.append((k, shots, ones))
        calls += shots * (2 * k + 1)
        steps += shots * k
        total_shots += shots
        # pool consecutive rounds that used the same k (same distribution)
        j, pooled_shots, pooled_ones = 1, shots, ones
        while j < len(ones_hist) and powers[-1 - j] == k:
            pooled_shots += shots
            pooled_ones += ones_hist[-1 - j]
            j += 1
        lo, hi = _clopper_pearson(pooled_ones, pooled_shots, round_alpha)
        if upper:
            t_lo = math.acos(1 - 2 * lo) / (2 * math.pi)
            t_hi = math.acos(1 - 2 * hi) / (2 * math.pi)
        else:
            t_lo = 1 - math.acos(1 - 2 * hi) / (2 * math.pi)
            t_hi = 1 - math.acos(1 - 2 * lo) / (2 * math.pi)
        scaling = 4 * k + 2
        new_u = (int(scaling * theta[1]) + t_hi) / scaling
        new_l = (int(scaling * theta[0]) + t_lo) / scaling
        theta = [max(theta[0], new_l), min(theta[1], new_u)]
        if theta[0] > theta[1]:  # numerical guard: keep the latest round's interval
            theta = [new_l, new_u]
        a_int = tuple(sorted((math.sin(2 * math.pi * theta[0]) ** 2, math.sin(2 * math.pi * theta[1]) ** 2)))
    est = _mle_refine(theta[0], theta[1], records) if use_mle else 0.5 * (a_int[0] + a_int[1])
    est = float(np.clip(est, a_int[0], a_int[1]))
    return IQAEResult(
        estimate=est,
        interval=(float(a_int[0]), float(a_int[1])),
        oracle_calls=int(calls),
        grover_steps=int(steps),
        shots=int(total_shots),
        rounds=rounds,
        max_k=max(powers),
        powers=powers[1:],
    )
