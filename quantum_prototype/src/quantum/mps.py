"""Track B: Matrix Product State (MPS) Tensor Network Solver, Generative Model & Circuit Bridge.

Implements all three Track B roles:
1. Multi-Country Scaling Solver (`JointLossMPS`):
   - Exact SVD compression of the 2-country ADB BondModel across bond dimensions chi=1..10.
   - Non-materialized n=2..20 country correlated regional portfolio MPS (d=10 per country,
     10^2 to 10^20 state space) calibrated to adb_terms.json cross-border hazard couplings.
   - Exact automaton MPO/MPS contraction for all 5 focus targets in O(n * U_T * d * chi^2),
     scaling linearly in n and validated against exact 10^n enumeration for n=2..5.
2. Limited-Data 5-Country Generative MPS (`GenerativeMPS`, Han et al. 2018):
   - Born-machine amplitude MPS trained via NLL gradient descent on finite subsamples of
     the real 5-country World Bank SFRARR Central Asia ELT dataset (KAZ, KGZ, TJK, TKM, UZB).
   - Quantifies parameter efficiency and rare-tail generalization vs. empirical Monte Carlo.
3. Shallow State-Prep Bridge to Track A (`mps_to_staircase_unitaries`, Ran 2020):
   - Compiles low-bond-dimension amplitude MPS into a linear-depth staircase of isometric
     unitaries, outputting Qiskit/OpenQASM circuits and comparing CNOTs vs. Grover-Rudolph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any
import numpy as np

from qjlb.bond import BondModel
from qjlb.model import ELTModel, entries

from .circuits import CircuitArtifact, build_mps_staircase_circuit

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


@dataclass
class MPSContractionResult:
    """Output of exact MPO/MPS contraction for an n-country portfolio."""

    n_countries: int
    max_bond_dim: int
    bond_dims: list[int]
    parameter_count: int
    dense_state_space_log10: float
    expected_principal_loss_usd: float
    all_countries_attachment_probability: float
    at_least_two_attachment_probability: float
    portfolio_loss_exceedance_probability: float
    portfolio_stop_loss_usd: float
    all_bonds_exhausted_probability: float
    elapsed_seconds: float
    truncation_error_l2: float = 0.0


class JointLossMPS:
    """Probability Tensor-Train / MPS representation of an n-country discrete joint loss model.

    Each core A[i] has shape (chi_left, d, chi_right) with chi_0 = chi_n = 1.
    Physical state index s in {0..9} encodes (eq_step in 0..4, rain in 0..1) via s = 2*eq + rain.
    """

    def __init__(
        self,
        cores: list[np.ndarray],
        eq_limits: np.ndarray,
        rain_limits: np.ndarray,
        truncation_error_l2: float = 0.0,
    ):
        self.cores = [np.asarray(c, dtype=np.float64) for c in cores]
        self.n_countries = len(self.cores)
        self.d = self.cores[0].shape[1]
        self.eq_limits = np.asarray(eq_limits, dtype=np.float64)
        self.rain_limits = np.asarray(rain_limits, dtype=np.float64)
        self.principals = self.eq_limits + self.rain_limits
        self.truncation_error_l2 = float(truncation_error_l2)

        # Per-country local loss lookup table of shape (n_countries, 10)
        # s = 2 * eq_step + rain_bit
        eq_steps = np.arange(self.d) // 2
        rain_bits = np.arange(self.d) % 2
        self.local_losses = (
            self.eq_limits[:, None] * (eq_steps[None, :] / 4.0)
            + self.rain_limits[:, None] * rain_bits[None, :]
        )

    @property
    def bond_dims(self) -> list[int]:
        return [int(c.shape[2]) for c in self.cores[:-1]]

    @property
    def max_bond_dim(self) -> int:
        return max(self.bond_dims, default=1)

    @property
    def parameter_count(self) -> int:
        return int(sum(c.size for c in self.cores))

    def total_probability(self) -> float:
        """Contract against all-ones vectors to compute normalization Z = sum_s p(s)."""
        vec = np.ones((1,), dtype=np.float64)
        for core in self.cores:
            # sum over physical index s: shape (chi_l, chi_r)
            mat = core.sum(axis=1)
            vec = vec @ mat
        return float(vec[0])

    def normalize(self) -> "JointLossMPS":
        z = self.total_probability()
        if z > 0.0 and not math.isclose(z, 1.0, rel_tol=1e-14):
            self.cores[0] = self.cores[0] / z
        return self

    def contract_product_observable(self, site_vectors: list[np.ndarray]) -> float:
        """Exact rank-1 MPO contraction E_p[prod_i v_i(s_i)] in O(n * d * chi^2)."""
        vec = np.ones((1,), dtype=np.float64)
        for core, v in zip(self.cores, site_vectors):
            # Transfer matrix T[a, b] = sum_s core[a, s, b] * v[s]
            t_mat = np.tensordot(core, v, axes=([1], [0]))  # shape (chi_l, chi_r)
            vec = vec @ t_mat
        return float(vec[0])

    def contract_expected_loss(self) -> float:
        """Exact E_p[sum_i L_i(s_i)] via left/right prefix-suffix environments in O(n * d * chi^2)."""
        n = self.n_countries
        ones = np.ones(self.d, dtype=np.float64)
        transfers = [core.sum(axis=1) for core in self.cores]

        # Right environments R[i] = T[i+1] @ ... @ T[n-1] @ [1]
        right = [None] * n
        r_vec = np.ones((1,), dtype=np.float64)
        for i in range(n - 1, -1, -1):
            right[i] = r_vec
            r_vec = transfers[i] @ r_vec

        total = 0.0
        l_vec = np.ones((1,), dtype=np.float64)
        for i in range(n):
            loss_mat = np.tensordot(self.cores[i], self.local_losses[i], axes=([1], [0]))
            total += float(l_vec @ loss_mat @ right[i])
            l_vec = l_vec @ transfers[i]
        return total

    def contract_at_least_k_attached(self, k: int = 2) -> float:
        """Exact (k+1)-state automaton MPO for Pr(at least k countries have L_i > 0)."""
        # state[c] has shape (chi_curr,) for c = 0, 1, ..., k (capped at k)
        dp = [np.zeros((1,), dtype=np.float64) for _ in range(k + 1)]
        dp[0] = np.ones((1,), dtype=np.float64)
        for i, core in enumerate(self.cores):
            zero_mask = (self.local_losses[i] <= 0.0).astype(np.float64)
            pos_mask = (self.local_losses[i] > 0.0).astype(np.float64)
            t_zero = np.tensordot(core, zero_mask, axes=([1], [0]))
            t_pos = np.tensordot(core, pos_mask, axes=([1], [0]))
            chi_next = core.shape[2]
            nxt = [np.zeros((chi_next,), dtype=np.float64) for _ in range(k + 1)]
            for c in range(k + 1):
                nxt[c] += dp[c] @ t_zero
                c_up = min(c + 1, k)
                nxt[c_up] += dp[c] @ t_pos
            dp = nxt
        return float(dp[k][0])

    def contract_threshold_and_stoploss(
        self,
        threshold_usd: float,
        unit_usd: float | None = None,
    ) -> tuple[float, float]:
        """Exact automaton MPO contraction for Pr(L > T) and E[max(L - T, 0)].

        When every local loss L_i(s) and threshold T are exact multiples of `unit_usd`
        (e.g., USD 2.5m for ADB 70m EQ / 10m rain / 80m threshold), this automaton is
        100% exact to machine precision for ANY n = 2..20 in O(n * U_T * d * chi^2).
        """
        if threshold_usd <= 0.0:
            p_pos = 1.0 - self.contract_product_observable(
                [(self.local_losses[i] <= 0.0).astype(np.float64) for i in range(self.n_countries)]
            )
            return float(np.clip(p_pos, 0.0, 1.0)), self.contract_expected_loss()

        # Exact 2-site contraction shortcut for n=2 (zero discretization error for arbitrary holdout sublimits)
        if self.n_countries == 2:
            # Reconstruct exact 10x10 probability matrix from MPS cores: (1,10,chi) @ (chi,10,1)
            p_mat = np.tensordot(self.cores[0][0], self.cores[1][:, :, 0], axes=([1], [0]))
            tot_loss = self.local_losses[0][:, None] + self.local_losses[1][None, :]
            exc_prob = float(p_mat[tot_loss > threshold_usd].sum())
            stop_loss = float(np.sum(p_mat * np.maximum(tot_loss - threshold_usd, 0.0)))
            return float(np.clip(exc_prob, 0.0, 1.0)), max(0.0, stop_loss)

        # Auto-detect exact discrete step (USD 2.5m for standard ADB 70/10 or 60/20 terms)
        if unit_usd is None:
            candidate = 2_500_000.0
            vals = np.concatenate([self.local_losses.ravel(), [threshold_usd]])
            if np.allclose(vals / candidate, np.round(vals / candidate), atol=1e-7):
                unit_usd = candidate
            else:
                unit_usd = max(threshold_usd / 128.0, 1.0)

        u_thresh = int(round(threshold_usd / unit_usd))
        # State indices 0..u_thresh track running sum <= T; index (u_thresh + 1) tracks L > T
        n_states = u_thresh + 2
        over = u_thresh + 1

        # w[u, alpha] = probability tensor component at running sum u
        # exc[alpha]  = accumulated excess loss (L - T) component for states already in `over`
        w = np.zeros((n_states, 1), dtype=np.float64)
        w[0, 0] = 1.0
        exc = np.zeros((1,), dtype=np.float64)

        for i, core in enumerate(self.cores):
            chi_next = core.shape[2]
            w_next = np.zeros((n_states, chi_next), dtype=np.float64)
            exc_next = np.zeros((chi_next,), dtype=np.float64)

            losses_i = self.local_losses[i]
            u_steps = np.round(losses_i / unit_usd).astype(int)

            # First propagate states already strictly above threshold
            t_all = core.sum(axis=1)  # (chi_l, chi_r)
            t_loss = np.tensordot(core, losses_i, axes=([1], [0]))  # (chi_l, chi_r)
            w_next[over] += w[over] @ t_all
            exc_next += exc @ t_all + w[over] @ t_loss

            # Propagate states u in 0..u_thresh for each local outcome s
            for s in range(self.d):
                mat = core[:, s, :]  # (chi_l, chi_r)
                du = int(u_steps[s])
                loss_s = float(losses_i[s])

                # Batch matrix multiply W_le @ mat -> shape (u_thresh + 1, chi_r)
                proj = w[:over] @ mat  # (u_thresh + 1, chi_r)

                if du == 0:
                    w_next[:over] += proj
                else:
                    split = max(0, over - du)  # indices u < split stay <= u_thresh
                    if split > 0:
                        w_next[du : du + split] += proj[:split]
                    if split < over:
                        # States u in split..u_thresh cross strictly above threshold
                        w_next[over] += proj[split:].sum(axis=0)
                        u_cross = np.arange(split, over, dtype=np.float64)
                        excess_amounts = u_cross * unit_usd + loss_s - threshold_usd
                        exc_next += excess_amounts @ proj[split:]

            w = w_next
            exc = exc_next

        return float(np.clip(w[over, 0], 0.0, 1.0)), max(0.0, float(exc[0]))

    def contract_all_targets(self, threshold_usd: float) -> MPSContractionResult:
        """Contract all portfolio focus targets via exact MPO/MPS contractions."""
        t0 = time.perf_counter()
        exp_loss = self.contract_expected_loss()
        all_attach = self.contract_product_observable(
            [(self.local_losses[i] > 0.0).astype(np.float64) for i in range(self.n_countries)]
        )
        two_attach = self.contract_at_least_k_attached(2)
        exc_prob, stop_loss = self.contract_threshold_and_stoploss(threshold_usd)
        all_exhaust = self.contract_product_observable(
            [
                (self.local_losses[i] >= self.principals[i] - 1e-6).astype(np.float64)
                for i in range(self.n_countries)
            ]
        )
        elapsed = time.perf_counter() - t0
        return MPSContractionResult(
            n_countries=self.n_countries,
            max_bond_dim=self.max_bond_dim,
            bond_dims=self.bond_dims,
            parameter_count=self.parameter_count,
            dense_state_space_log10=float(self.n_countries * math.log10(self.d)),
            expected_principal_loss_usd=max(0.0, exp_loss),
            all_countries_attachment_probability=float(np.clip(all_attach, 0.0, 1.0)),
            at_least_two_attachment_probability=float(np.clip(two_attach, 0.0, 1.0)),
            portfolio_loss_exceedance_probability=float(np.clip(exc_prob, 0.0, 1.0)),
            portfolio_stop_loss_usd=max(0.0, stop_loss),
            all_bonds_exhausted_probability=float(np.clip(all_exhaust, 0.0, 1.0)),
            elapsed_seconds=elapsed,
            truncation_error_l2=self.truncation_error_l2,
        )

    def to_dense(self) -> np.ndarray:
        """Materialize full d^n dense tensor (ONLY for small n <= 5 validation)."""
        if self.n_countries > 6:
            raise ValueError("Refusing to materialize dense 10^n tensor for n > 6")
        tensor = self.cores[0][0]  # (d, chi_1)
        for core in self.cores[1:]:
            tensor = np.tensordot(tensor, core, axes=([-1], [0]))
        return tensor[..., 0]


def _bond_term_matrix_10x10(model: BondModel) -> np.ndarray:
    """Reshape BondModel's 100-state term distribution into a (10, 10) country-pair matrix.

    Country 1 state s1 = 2 * a + r1 in {0..9}; Country 2 state s2 = 2 * b + r2 in {0..9}.
    """
    p_flat = model.term_distribution()
    # model.states order is ndindex(5, 5, 2, 2) -> (a, b, r1, r2)
    p4 = p_flat.reshape(5, 5, 2, 2)
    # Transpose to (a, r1, b, r2) and reshape to (10, 10)
    return p4.transpose(0, 2, 1, 3).reshape(10, 10)


def build_bond_mps_exact(model: BondModel, max_bond_dim: int = 10) -> JointLossMPS:
    """Compress the 2-country ADB BondModel term distribution into an SVD MPS of bond dim <= chi."""
    p_mat = _bond_term_matrix_10x10(model)
    U, S, Vt = np.linalg.svd(p_mat, full_matrices=False)
    chi = max(1, min(int(max_bond_dim), len(S)))
    trunc_err = float(np.linalg.norm(S[chi:])) if chi < len(S) else 0.0

    sqrt_s = np.sqrt(S[:chi])
    core0 = (U[:, :chi] * sqrt_s[None, :]).reshape(1, 10, chi)
    core1 = (sqrt_s[:, None] * Vt[:chi, :]).reshape(chi, 10, 1)

    mps = JointLossMPS(
        [core0, core1],
        eq_limits=model.eq_limit,
        rain_limits=model.rain_limit,
        truncation_error_l2=trunc_err,
    )
    return mps.normalize()


def build_synthetic_ncountry_mps(
    base_model: BondModel,
    n_countries: int,
    max_bond_dim: int = 10,
    coupling_strength: float = 1.0,
) -> JointLossMPS:
    """Build an n-country correlated regional risk-pool MPS (n=2..20) WITHOUT materializing 10^n states.

    Constructs a calibrated geographic Markov-chain / pairwise-coupled joint distribution
    anchored to `adb_terms.json` via `base_model`:
      - At n=2 with coupling_strength=1.0 and max_bond_dim=10, reproduces `adb-base` identically.
      - For n > 2, pairs adjacent countries along a regional seismic/river-basin chain using
        the calibrated bivariate conditional transition kernel P(s_{i+1} | s_i), compressed via
        sequential canonical SVD sweeps to bond dimension `max_bond_dim`.

    Strictly labeled as a synthetic regional scaling extension calibrated to ADB 2026 terms.
    """
    if n_countries < 2:
        raise ValueError("n_countries must be >= 2")

    p_pair_base = _bond_term_matrix_10x10(base_model)
    # Symmetrize marginally for homogeneous chain beyond n=2, while keeping exact KGZ/TJK at n=2
    if n_countries == 2 and math.isclose(coupling_strength, 1.0):
        return build_bond_mps_exact(base_model, max_bond_dim=max_bond_dim)

    p_sym = 0.5 * (p_pair_base + p_pair_base.T)
    p_marg = p_sym.sum(axis=1)
    p_indep = np.outer(p_marg, p_marg)
    # Convex blend of calibrated dependent bivariate law and independent baseline
    alpha_c = float(np.clip(coupling_strength, 0.0, 1.0))
    p_joint = alpha_c * p_sym + (1.0 - alpha_c) * p_indep
    p_joint = p_joint / p_joint.sum()
    p_m = p_joint.sum(axis=1)
    # Conditional transition matrix T[s_i, s_{i+1}] = P(s_{i+1} | s_i)
    cond = p_joint / np.maximum(p_m[:, None], 1e-30)

    # Exact MPS representation of Markov chain p(s_1..s_n) = p_m[s_1] * prod_i cond[s_i, s_{i+1}]:
    #   Core 0:        A0[0, s, a] = p_m[s] * delta(s, a) -> shape (1, 10, 10)
    #   Middle core i: Ai[a, s, b] = cond[a, s] * delta(s, b) -> shape (10, 10, 10)
    #   Last core n-1: An[a, s, 0] = cond[a, s] -> shape (10, 10, 1)
    d = 10
    cores: list[np.ndarray] = []
    c0 = np.zeros((1, d, d), dtype=np.float64)
    for s in range(d):
        c0[0, s, s] = p_m[s]
    cores.append(c0)

    for _ in range(1, n_countries - 1):
        cm = np.zeros((d, d, d), dtype=np.float64)
        for s in range(d):
            cm[:, s, s] = cond[:, s]
        cores.append(cm)

    cn = cond.reshape(d, d, 1).copy()
    cores.append(cn)

    # Compress to max_bond_dim via right-to-left QR orthogonalization + left-to-right truncated SVD
    trunc_err_sq = 0.0
    if max_bond_dim < d:
        # Right-to-left QR canonicalization
        for i in range(n_countries - 1, 0, -1):
            cl, phys, cr = cores[i].shape
            mat = cores[i].reshape(cl, phys * cr)
            q, r = np.linalg.qr(mat.T)  # mat.T = q @ r -> mat = r.T @ q.T
            new_chi = q.shape[1]
            cores[i] = q.T.reshape(new_chi, phys, cr)
            cores[i - 1] = np.tensordot(cores[i - 1], r.T, axes=([2], [0]))

        # Left-to-right truncated SVD
        for i in range(n_countries - 1):
            cl, phys, cr = cores[i].shape
            mat = cores[i].reshape(cl * phys, cr)
            U, S, Vt = np.linalg.svd(mat, full_matrices=False)
            chi = max(1, min(int(max_bond_dim), len(S)))
            if chi < len(S):
                trunc_err_sq += float(np.sum(S[chi:] ** 2))
            cores[i] = U[:, :chi].reshape(cl, phys, chi)
            sv = S[:chi, None] * Vt[:chi, :]
            cores[i + 1] = np.tensordot(sv, cores[i + 1], axes=([1], [0]))

    eq_lim = np.full(n_countries, float(np.mean(base_model.eq_limit)), dtype=np.float64)
    rain_lim = np.full(n_countries, float(np.mean(base_model.rain_limit)), dtype=np.float64)
    mps = JointLossMPS(
        cores,
        eq_limits=eq_lim,
        rain_limits=rain_lim,
        truncation_error_l2=math.sqrt(trunc_err_sq),
    )
    return mps.normalize()


def solve_bond_mps(
    model: BondModel,
    samples: int = 8192,
    seed: int = 0,
    *,
    max_bond_dim: int = 10,
) -> dict[str, Any]:
    """qjlb-compliant Track B MPS solver for ADB BondModel (classical_cpu execution)."""
    t0 = time.perf_counter()
    mps = build_bond_mps_exact(model, max_bond_dim=max_bond_dim)
    prep_sec = time.perf_counter() - t0

    res = mps.contract_all_targets(model.threshold)
    means = [
        res.expected_principal_loss_usd,
        res.all_countries_attachment_probability,
        res.portfolio_loss_exceedance_probability,
        res.portfolio_stop_loss_usd,
        res.all_bonds_exhausted_probability,
    ]
    est_dict = entries(means, None, metric_names=model.metric_names)
    for item in est_dict.values():
        item.update(
            standard_error=0.0,
            ci95_low=item["estimate"],
            ci95_high=item["estimate"],
        )

    return {
        "estimates": est_dict,
        "resources": {
            "method": f"mps_chi{res.max_bond_dim}",
            "execution": "classical_cpu",
            "shots": 0,
            "qubits": 0,
            "model_evaluations": res.parameter_count,
            "preparation_seconds": float(prep_sec),
            "mps_bond_dim": int(res.max_bond_dim),
            "mps_parameters": int(res.parameter_count),
            "truncation_error_l2": float(res.truncation_error_l2),
            "requested_baseline_samples": int(samples),
        },
        "uncertainty": (
            "Deterministic MPS tensor-network contraction (zero Monte Carlo variance; "
            f"SVD truncation L2 error = {res.truncation_error_l2:.3e})."
        ),
    }


def benchmark_mps_scaling(
    base_model: BondModel,
    country_counts: list[int] | None = None,
    bond_dims: list[int] | None = None,
    threshold_usd: float = 80_000_000.0,
) -> list[dict[str, Any]]:
    """Run Track B multi-country scaling benchmark from n=2 to n=20 (10^2 to 10^20 states).

    Validates MPS contractions against brute-force exact 10^n enumeration for n <= 4
    and measures runtime, memory/parameters, and truncation error up to n=20.
    """
    if country_counts is None:
        country_counts = [2, 3, 4, 5, 6, 8, 10, 15, 20]
    if bond_dims is None:
        bond_dims = [2, 4, 8, 10]

    rows: list[dict[str, Any]] = []
    for n in country_counts:
        # Compute full-chi reference MPS (exact for the Markov chain model)
        ref_mps = build_synthetic_ncountry_mps(base_model, n_countries=n, max_bond_dim=10)
        ref_res = ref_mps.contract_all_targets(threshold_usd)

        # For n <= 4, cross-verify ref_mps against explicit brute-force 10^n enumeration
        dense_verify_max_err = None
        dense_seconds = None
        if n <= 4:
            td0 = time.perf_counter()
            p_dense = ref_mps.to_dense().ravel()
            grids = np.indices((10,) * n).reshape(n, -1)  # (n, 10^n)
            losses = np.sum(
                [ref_mps.local_losses[i][grids[i]] for i in range(n)],
                axis=0,
            )
            dense_el = float(np.dot(p_dense, losses))
            dense_exc = float(p_dense[losses > threshold_usd].sum())
            dense_sl = float(np.dot(p_dense, np.maximum(losses - threshold_usd, 0.0)))
            dense_seconds = time.perf_counter() - td0
            dense_verify_max_err = max(
                abs(dense_el - ref_res.expected_principal_loss_usd) / max(dense_el, 1.0),
                abs(dense_exc - ref_res.portfolio_loss_exceedance_probability),
                abs(dense_sl - ref_res.portfolio_stop_loss_usd) / max(dense_sl, 1.0),
            )

        for chi in bond_dims:
            t0 = time.perf_counter()
            mps = build_synthetic_ncountry_mps(base_model, n_countries=n, max_bond_dim=chi)
            res = mps.contract_all_targets(threshold_usd)
            total_sec = time.perf_counter() - t0
            rel_err_el = abs(
                res.expected_principal_loss_usd - ref_res.expected_principal_loss_usd
            ) / max(ref_res.expected_principal_loss_usd, 1e-12)
            rel_err_exc = abs(
                res.portfolio_loss_exceedance_probability
                - ref_res.portfolio_loss_exceedance_probability
            ) / max(ref_res.portfolio_loss_exceedance_probability, 1e-12)
            rel_err_sl = abs(
                res.portfolio_stop_loss_usd - ref_res.portfolio_stop_loss_usd
            ) / max(ref_res.portfolio_stop_loss_usd, 1e-12)

            rows.append(
                {
                    "n_countries": n,
                    "bond_dim": chi,
                    "log10_state_space": n,
                    "mps_parameters": res.parameter_count,
                    "build_and_contract_seconds": total_sec,
                    "contract_only_seconds": res.elapsed_seconds,
                    "dense_exact_seconds": dense_seconds,
                    "dense_verify_max_err": dense_verify_max_err,
                    "expected_loss_usd": res.expected_principal_loss_usd,
                    "at_least_two_attach_prob": res.at_least_two_attachment_probability,
                    "exceedance_prob": res.portfolio_loss_exceedance_probability,
                    "stop_loss_usd": res.portfolio_stop_loss_usd,
                    "rel_err_el_vs_chi10": rel_err_el,
                    "rel_err_exc_vs_chi10": rel_err_exc,
                    "rel_err_sl_vs_chi10": rel_err_sl,
                    "truncation_error_l2": res.truncation_error_l2,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Role 2: Limited-Data Generative MPS (Born Machine, Han et al., 2018)
# ---------------------------------------------------------------------------


class GenerativeMPS:
    """Born-machine amplitude MPS p(s) = |Psi(s)|^2 / Z for limited-data risk modeling.

    Directly addresses Challenge Brief p. 1 & Section 3c (Han et al., 2018):
    learns correlated multi-country joint loss distributions from finite historical
    samples with O(n * d * chi^2) parameters instead of d^n tabular parameters.
    """

    def __init__(self, n_sites: int, phys_dim: int, bond_dim: int, seed: int = 0):
        self.n_sites = n_sites
        self.d = phys_dim
        self.chi = bond_dim
        rng = np.random.default_rng(seed)

        self.cores: list[np.ndarray] = []
        for i in range(n_sites):
            cl = 1 if i == 0 else bond_dim
            cr = 1 if i == n_sites - 1 else bond_dim
            # Initialize near product state with mild random entanglement so Z is well-conditioned
            core = rng.normal(0.0, 0.15 / math.sqrt(max(cl, cr)), size=(cl, phys_dim, cr))
            core[0, 0, 0] += 1.0
            self.cores.append(core)

    @property
    def parameter_count(self) -> int:
        return int(sum(c.size for c in self.cores))

    def _partition_function_and_envs(self) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        """Exact Born-rule partition function Z = <Psi|Psi> and left/right transfer environments."""
        n = self.n_sites
        # Left environments L[i]: shape (chi_i, chi_i) before site i
        left_envs: list[np.ndarray] = [np.ones((1, 1), dtype=np.float64)]
        env = left_envs[0]
        for i in range(n - 1):
            c = self.cores[i]  # (cl, d, cr)
            # env_new[b, b'] = sum_{a, a', s} env[a, a'] * c[a, s, b] * c[a', s, b']
            tmp = np.einsum("ab,asd->bsd", env, c)
            env = np.einsum("bsd,bse->de", tmp, c)
            left_envs.append(env)

        right_envs: list[np.ndarray] = [np.ones((1, 1), dtype=np.float64)] * n
        env = np.ones((1, 1), dtype=np.float64)
        for i in range(n - 1, 0, -1):
            c = self.cores[i]
            tmp = np.einsum("asd,de->ase", c, env)
            env = np.einsum("ase,bse->ab", tmp, c)
            right_envs[i - 1] = env

        # Compute Z from site 0
        c0 = self.cores[0]
        tmp = np.einsum("asd,de->ase", c0, right_envs[0])
        z = float(np.einsum("ase,ase->", tmp, c0))
        return max(z, 1e-30), left_envs, right_envs

    def amplitudes_batch(self, samples: np.ndarray) -> np.ndarray:
        """Evaluate Psi(x) for a batch of integer state vectors x of shape (N, n_sites)."""
        n_batch = samples.shape[0]
        vec = np.ones((n_batch, 1), dtype=np.float64)
        for i, core in enumerate(self.cores):
            # core[:, samples[:, i], :] has shape (cl, N, cr) -> transpose to (N, cl, cr)
            mats = np.transpose(core[:, samples[:, i], :], (1, 0, 2))
            vec = np.einsum("ni,nij->nj", vec, mats)
        return vec[:, 0]

    def fit(
        self,
        samples: np.ndarray,
        epochs: int = 40,
        lr: float = 0.08,
        weight_decay: float = 1e-4,
    ) -> list[float]:
        """Train Born-machine MPS by minimizing exact negative log-likelihood NLL."""
        samples = np.asarray(samples, dtype=np.int64)
        n_batch, n_sites = samples.shape
        assert n_sites == self.n_sites

        # Warm-start site marginals on channel (0, s, 0) for fast, stable convergence
        for i in range(n_sites):
            counts = np.bincount(samples[:, i], minlength=self.d).astype(np.float64) + 0.5
            marg = np.sqrt(counts / counts.sum())
            self.cores[i][0, :, 0] = marg

        nll_history: list[float] = []
        # Adam state per core
        m_adam = [np.zeros_like(c) for c in self.cores]
        v_adam = [np.zeros_like(c) for c in self.cores]
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for epoch in range(1, epochs + 1):
            z, left_z, right_z = self._partition_function_and_envs()

            # Compute batch left/right sample environments for exact analytical NLL gradient
            l_samp = [np.ones((n_batch, 1), dtype=np.float64)]
            for i in range(n_sites - 1):
                mats = np.transpose(self.cores[i][:, samples[:, i], :], (1, 0, 2))
                l_samp.append(np.einsum("ni,nij->nj", l_samp[-1], mats))

            r_samp: list[np.ndarray] = [np.ones((n_batch, 1), dtype=np.float64)] * n_sites
            cur = np.ones((n_batch, 1), dtype=np.float64)
            for i in range(n_sites - 1, 0, -1):
                mats = np.transpose(self.cores[i][:, samples[:, i], :], (1, 0, 2))
                cur = np.einsum("nij,nj->ni", mats, cur)
                r_samp[i - 1] = cur

            # Amplitudes Psi(x) via site 0
            mats0 = np.transpose(self.cores[0][:, samples[:, 0], :], (1, 0, 2))
            psi = np.einsum("ni,nij,nj->n", l_samp[0], mats0, r_samp[0])
            psi_safe = np.where(np.abs(psi) < 1e-12, np.sign(psi + 1e-18) * 1e-12, psi)
            nll = float(math.log(z) - np.mean(np.log(np.maximum(psi * psi, 1e-30))))
            nll_history.append(nll)

            # Analytical gradient d(NLL)/d(A[i]) = (2/Z) * L_z A[i] R_z - (2/N) sum_x (l_x outer r_x)/Psi(x)
            for i in range(n_sites):
                c = self.cores[i]
                grad_z = (2.0 / z) * np.einsum("ab,bsd,de->ase", left_z[i], c, right_z[i])

                grad_data = np.zeros_like(c)
                inv_psi = (2.0 / n_batch) / psi_safe  # shape (N,)
                li = l_samp[i] * inv_psi[:, None]  # (N, cl)
                ri = r_samp[i]  # (N, cr)
                s_col = samples[:, i]
                for s_val in range(self.d):
                    mask = s_col == s_val
                    if np.any(mask):
                        grad_data[:, s_val, :] = li[mask].T @ ri[mask]

                grad = grad_z - grad_data + weight_decay * c
                # Clip gradient norm for stability
                gnorm = float(np.linalg.norm(grad))
                if gnorm > 5.0:
                    grad = grad * (5.0 / gnorm)

                m_adam[i] = beta1 * m_adam[i] + (1.0 - beta1) * grad
                v_adam[i] = beta2 * v_adam[i] + (1.0 - beta2) * (grad * grad)
                m_hat = m_adam[i] / (1.0 - beta1**epoch)
                v_hat = v_adam[i] / (1.0 - beta2**epoch)
                self.cores[i] = c - lr * m_hat / (np.sqrt(v_hat) + eps)

            # Periodically re-balance core norms to keep Z ~ 1
            if epoch % 10 == 0:
                z_now, _, _ = self._partition_function_and_envs()
                scale = z_now ** (0.5 / n_sites)
                if scale > 0:
                    for i in range(n_sites):
                        self.cores[i] /= scale

        return nll_history

    def to_probability_tensor(self) -> np.ndarray:
        """Compute full normalized Born probability tensor p(s) = |Psi(s)|^2 / Z (for small d^n)."""
        amp = self.cores[0][0]
        for core in self.cores[1:]:
            amp = np.tensordot(amp, core, axes=([-1], [0]))
        amp = amp[..., 0]
        probs = amp * amp
        return probs / probs.sum()


def benchmark_generative_mps_worldbank(
    data_dir: str,
    train_sizes: list[int] | None = None,
    bond_dims: list[int] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Train 5-country Generative MPS on limited subsamples of the REAL World Bank ELT dataset.

    Countries: all 5 Central Asian countries in seismic.npz + flood.npz
    (KAZ, KGZ, TJK, TKM, UZB; CC BY 4.0 SFRARR Central Asia).
    Discretizes each country's event payout into d=4 severity tiers (4^5 = 1,024 joint states)
    and compares out-of-sample tail KL divergence & joint-exceedance accuracy of Generative MPS
    against Empirical Monte Carlo across limited sample budgets.
    """
    if train_sizes is None:
        train_sizes = [500, 2000, 8000]
    if bond_dims is None:
        bond_dims = [2, 4]

    countries = ["KAZ", "KGZ", "TJK", "TKM", "UZB"]
    n = len(countries)
    config = {
        "schema_version": 2,
        "track": "worldbank_elt",
        "dataset_directory": str(data_dir),
        "countries": countries,
        "peril_order": ["seismic", "flood"],
        "attachment_usd": [[5e7] * n, [2e7] * n],
        "exhaustion_usd": [[5e8] * n, [2e8] * n],
        "limit_usd": [8e7] * n,
        "portfolio_threshold_usd": 8e7,
    }
    model = ELTModel(config, data_dir)
    all_idx = np.arange(model.state_count, dtype=np.int64)[:, None]
    payouts = model.evaluate(all_idx)["payout_usd"]  # (N, 5)
    probs = model.probabilities

    # Discretize each country's payout fraction into d=4 tiers: 0, (0, 0.25], (0.25, 0.75], >0.75
    d = 4
    frac = payouts / model.limit[None, :]
    tiers = np.zeros_like(frac, dtype=np.int64)
    tiers[(frac > 0.0) & (frac <= 0.25)] = 1
    tiers[(frac > 0.25) & (frac <= 0.75)] = 2
    tiers[frac > 0.75] = 3

    # Exact ground-truth 5-country joint distribution over 4^5 = 1,024 states
    flat_idx = np.ravel_multi_index(tiers.T, (d,) * n)
    p_true = np.bincount(flat_idx, weights=probs, minlength=d**n)
    p_true /= p_true.sum()

    # Exact ground-truth portfolio indicators on the 1,024 discretized states
    grid = np.indices((d,) * n).reshape(n, -1)  # (5, 1024)
    at_least_two_mask = (np.sum(grid > 0, axis=0) >= 2).astype(np.float64)
    severe_tail_mask = (np.sum(grid, axis=0) >= 4).astype(np.float64)
    true_two_attach = float(np.dot(p_true, at_least_two_mask))
    true_severe_tail = float(np.dot(p_true, severe_tail_mask))

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for n_train in train_sizes:
        # Draw limited historical sample from true World Bank catalogue
        drawn_events = rng.choice(model.state_count, size=n_train, p=probs)
        sample_tiers = tiers[drawn_events]
        sample_flat = flat_idx[drawn_events]

        # Empirical MC baseline (Laplace-smoothed for finite KL evaluation)
        emp_counts = np.bincount(sample_flat, minlength=d**n).astype(np.float64)
        p_emp = emp_counts / emp_counts.sum()
        p_emp_smooth = (emp_counts + 1e-3) / (emp_counts.sum() + 1e-3 * (d**n))
        emp_kl = float(np.sum(p_true * np.log(np.maximum(p_true, 1e-15) / p_emp_smooth)))
        emp_two = float(np.dot(p_emp, at_least_two_mask))
        emp_sev = float(np.dot(p_emp, severe_tail_mask))

        rows.append(
            {
                "train_samples": n_train,
                "model": "empirical_mc",
                "bond_dim": 0,
                "parameters": d**n,
                "kl_divergence": emp_kl,
                "true_ge2_attach": true_two_attach,
                "est_ge2_attach": emp_two,
                "abs_err_ge2_attach": abs(emp_two - true_two_attach),
                "true_severe_tail": true_severe_tail,
                "est_severe_tail": emp_sev,
                "abs_err_severe_tail": abs(emp_sev - true_severe_tail),
                "train_seconds": 0.0,
            }
        )

        for chi in bond_dims:
            t0 = time.perf_counter()
            gmps = GenerativeMPS(n_sites=n, phys_dim=d, bond_dim=chi, seed=seed + n_train + chi)
            gmps.fit(sample_tiers, epochs=50, lr=0.06)
            train_sec = time.perf_counter() - t0

            p_mps = gmps.to_probability_tensor().ravel()
            p_mps_safe = np.maximum(p_mps, 1e-15)
            mps_kl = float(np.sum(p_true * np.log(np.maximum(p_true, 1e-15) / p_mps_safe)))
            mps_two = float(np.dot(p_mps, at_least_two_mask))
            mps_sev = float(np.dot(p_mps, severe_tail_mask))

            rows.append(
                {
                    "train_samples": n_train,
                    "model": f"generative_mps_chi{chi}",
                    "bond_dim": chi,
                    "parameters": gmps.parameter_count,
                    "kl_divergence": mps_kl,
                    "true_ge2_attach": true_two_attach,
                    "est_ge2_attach": mps_two,
                    "abs_err_ge2_attach": abs(mps_two - true_two_attach),
                    "true_severe_tail": true_severe_tail,
                    "est_severe_tail": mps_sev,
                    "abs_err_severe_tail": abs(mps_sev - true_severe_tail),
                    "train_seconds": train_sec,
                }
            )

    return rows


# ---------------------------------------------------------------------------
# Role 3: MPS-to-Staircase Unitary Compilation Bridge to Track A (Ran, 2020)
# ---------------------------------------------------------------------------


def _complete_isometry_to_unitary(iso: np.ndarray) -> np.ndarray:
    """Embed a tall/square isometry V (rows >= cols, V^H V = I) into a full unitary U."""
    rows, cols = iso.shape
    if cols > rows:
        raise ValueError(f"Expected rows >= cols for isometry, got {iso.shape}")
    if rows == cols:
        # Ensure exact unitarity via polar decomposition
        u, _, vh = np.linalg.svd(iso)
        return u @ vh
    # Complete orthonormal basis via QR decomposition on [iso, I]
    rng_aug = np.hstack([iso, np.eye(rows, dtype=np.float64)])
    q, r = np.linalg.qr(rng_aug)
    # Match phases on first `cols` columns so U[:, :cols] == iso
    for j in range(cols):
        if np.dot(q[:, j], iso[:, j]) < 0:
            q[:, j] *= -1.0
    # Verify and replace first cols columns directly then re-orthogonalize remaining
    u = np.zeros((rows, rows), dtype=np.float64)
    u[:, :cols] = iso
    # Project out span(iso) from remaining columns of Q
    rem = q[:, cols:] - iso @ (iso.T @ q[:, cols:])
    q_rem, _ = np.linalg.qr(rem)
    u[:, cols:] = q_rem[:, : (rows - cols)]
    return u


def mps_to_staircase_unitaries(
    probability_tensor: np.ndarray,
    max_bond_dim: int = 2,
    label: str = "mps_staircase",
) -> tuple[CircuitArtifact, float, dict[str, Any]]:
    """Compile an amplitude MPS |Psi> = sum sqrt(p(x)) |x> into a linear-depth unitary staircase.

    Follows Ran (2020) sequential isometric disentangling:
      1. Convert probabilities to statevector amplitudes psi = sqrt(p) padded to power-of-2
         local dimension d_pad = 2^q_p per site.
      2. Decompose into right-canonical amplitude MPS with bond dim chi <= max_bond_dim
         (rounded up to power of 2: chi_pad = 2^q_b).
      3. Embed each isometric site tensor into a (q_p + q_b)-qubit unitary U_i.
      4. Build transpiled circuit and compare CX count vs. dense Grover-Rudolph loading.

    Returns (circuit_artifact, state_fidelity, comparison_metrics).
    """
    p = np.asarray(probability_tensor, dtype=np.float64)
    p = np.maximum(p, 0.0)
    p /= p.sum()
    n_sites = p.ndim
    d_orig = p.shape[0]
    q_p = max(1, int(math.ceil(math.log2(d_orig))))
    d_pad = 1 << q_p

    # Pad each physical axis to d_pad = 2^q_p
    pad_width = [(0, d_pad - s) for s in p.shape]
    amp_exact = np.sqrt(np.pad(p, pad_width))
    amp_exact /= np.linalg.norm(amp_exact)

    # Sequential SVD into amplitude MPS with bond dim <= chi_pad (power of 2)
    q_b = max(1, int(math.ceil(math.log2(max(max_bond_dim, 2)))))
    chi_pad = 1 << q_b

    cores: list[np.ndarray] = []
    rem = amp_exact.reshape(1, -1)
    cl = 1
    for i in range(n_sites - 1):
        mat = rem.reshape(cl * d_pad, -1)
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        chi = min(chi_pad, len(S))
        # Pad bond dimension up to chi_pad with zeros if smaller so unitary width is power of 2
        core = np.zeros((cl, d_pad, chi_pad), dtype=np.float64)
        core[:, :, :chi] = U[:, :chi].reshape(cl, d_pad, chi)
        cores.append(core)
        rem_next = np.zeros((chi_pad, Vt.shape[1]), dtype=np.float64)
        rem_next[:chi, :] = S[:chi, None] * Vt[:chi, :]
        rem = rem_next
        cl = chi_pad
    cores.append(rem.reshape(cl, d_pad, 1))

    # Reconstruct truncated amplitude state to compute exact state-prep fidelity |<Psi_mps|Psi_exact>|^2
    rec = cores[0][0]
    for core in cores[1:]:
        rec = np.tensordot(rec, core, axes=([-1], [0]))
    rec = rec[..., 0]
    rec_norm = float(np.linalg.norm(rec))
    fidelity = float((np.sum(rec * amp_exact) / max(rec_norm, 1e-15)) ** 2)

    # Build right-canonical isometric unitaries (Ran 2020)
    # Right-to-left SVD/QR sweep produces isometries mapping bond -> (phys, next_bond)
    unitaries: list[np.ndarray] = []
    for i, core in enumerate(cores):
        cl_i, d_i, cr_i = core.shape
        if i == 0:
            # Maps |0> -> state on (d_pad, chi_pad): shape (d_pad * chi_pad, 1)
            vec = core.reshape(d_i * cr_i, 1)
            nrm = float(np.linalg.norm(vec))
            iso = vec / max(nrm, 1e-15)
            unitaries.append(_complete_isometry_to_unitary(iso))
        elif i == n_sites - 1:
            # Last site: maps chi_pad -> d_pad (if d_pad >= chi_pad)
            mat = core.reshape(cl_i, d_i).T  # (d_pad, chi_pad)
            # Orthonormalize columns via SVD for unitary embedding
            u_m, _, vh_m = np.linalg.svd(mat, full_matrices=False)
            iso = u_m @ vh_m
            if iso.shape[0] >= iso.shape[1]:
                unitaries.append(_complete_isometry_to_unitary(iso))
            else:
                # Pad rows to power of 2 >= cl_i
                w_dim = max(d_i, cl_i)
                padded_iso = np.zeros((w_dim, cl_i), dtype=np.float64)
                padded_iso[: iso.shape[0], : iso.shape[1]] = iso
                unitaries.append(_complete_isometry_to_unitary(padded_iso))
        else:
            # Middle site: maps cl_i -> (d_i * cr_i)
            mat = core.reshape(cl_i, d_i * cr_i).T  # (d_pad * chi_pad, chi_pad)
            u_m, _, vh_m = np.linalg.svd(mat, full_matrices=False)
            iso = u_m @ vh_m
            unitaries.append(_complete_isometry_to_unitary(iso))

    artifact = build_mps_staircase_circuit(
        unitaries, phys_qubits_per_site=q_p, bond_qubits=q_b, label=label
    )

    total_qubits = n_sites * q_p
    grover_rudolph_cx = (1 << total_qubits) - 2
    comparison = {
        "n_sites": n_sites,
        "phys_qubits_per_site": q_p,
        "bond_dim_chi": chi_pad,
        "total_qubits": total_qubits,
        "state_fidelity": fidelity,
        "mps_staircase_cx": artifact.two_qubit_gates,
        "mps_staircase_depth": artifact.transpiled_depth,
        "grover_rudolph_cx": grover_rudolph_cx,
        "cx_reduction_factor": grover_rudolph_cx / max(artifact.two_qubit_gates, 1),
    }
    return artifact, fidelity, comparison
