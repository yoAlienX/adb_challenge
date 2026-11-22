"""Exact chi=10 causal MPS contraction for multi-country latent-shock portfolios."""

from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from scipy.stats import poisson

from .loader import EQ_LEVELS, _add_marks, _own_steps


@dataclass
class CausalMPSResult:
    n_countries: int
    bond_dim: int
    parameter_count: int
    state_space_log10: float
    expected_principal_loss_usd: float
    joint_attachment_probability: float
    portfolio_exceedance_probability: float
    portfolio_stop_loss_usd: float
    all_bonds_exhausted_probability: float
    elapsed_seconds: float
    truncation_error: float = 0.0


class CausalLatentMPS:
    """Exact chi=10 MPS where countries are independent given the (s, z) shock."""

    def __init__(
        self,
        shock_probs: np.ndarray,
        country_conditionals: list[np.ndarray],
        eq_limits: np.ndarray,
        rain_limits: np.ndarray,
        threshold: float,
    ):
        self.shock_probs = np.asarray(shock_probs, dtype=np.float64)
        self.conditionals = [np.asarray(c, dtype=np.float64) for c in country_conditionals]
        self.n_countries = len(self.conditionals)
        self.eq_limits = np.asarray(eq_limits, dtype=np.float64)
        self.rain_limits = np.asarray(rain_limits, dtype=np.float64)
        self.principals = self.eq_limits + self.rain_limits
        self.threshold = float(threshold)
        self.bond_dim = len(self.shock_probs)

        eq_steps = np.arange(10) // 2
        rain_bits = np.arange(10) % 2
        self.local_losses = (
            self.eq_limits[:, None] * (eq_steps[None, :] / 4.0)
            + self.rain_limits[:, None] * rain_bits[None, :]
        )

    @property
    def parameter_count(self) -> int:
        return int(self.bond_dim + self.n_countries * (25 + 4))

    def contract_all(self) -> CausalMPSResult:
        t0 = time.perf_counter()
        p_shock = self.shock_probs
        n = self.n_countries

        el_per_shock = np.zeros(self.bond_dim, dtype=np.float64)
        p_all_attach = np.ones(self.bond_dim, dtype=np.float64)
        p_all_exhaust = np.ones(self.bond_dim, dtype=np.float64)
        for i in range(n):
            el_per_shock += self.conditionals[i] @ self.local_losses[i]
            pos_mask = (self.local_losses[i] > 0.0).astype(np.float64)
            exh_mask = (np.isclose(self.local_losses[i], self.principals[i])).astype(np.float64)
            p_all_attach *= self.conditionals[i] @ pos_mask
            p_all_exhaust *= self.conditionals[i] @ exh_mask

        expected_loss = float(p_shock @ el_per_shock)
        joint_attach = float(p_shock @ p_all_attach)
        all_exhaust = float(p_shock @ p_all_exhaust)

        exceed = 0.0
        stop_loss = 0.0
        for b in range(self.bond_dim):
            pb = float(p_shock[b])
            if pb <= 0.0:
                continue
            dist: dict[float, float] = {0.0: 1.0}
            for i in range(n):
                probs_i = self.conditionals[i][b]
                losses_i = self.local_losses[i]
                nxt: dict[float, float] = {}
                for x_idx in range(10):
                    px = float(probs_i[x_idx])
                    if px <= 0.0:
                        continue
                    lx = float(losses_i[x_idx])
                    for cur_l, cur_p in dist.items():
                        key = round(cur_l + lx, 4)
                        nxt[key] = nxt.get(key, 0.0) + cur_p * px
                dist = nxt
            for total_l, prob_l in dist.items():
                if total_l > self.threshold:
                    exceed += pb * prob_l
                    stop_loss += pb * prob_l * (total_l - self.threshold)

        return CausalMPSResult(
            n_countries=n,
            bond_dim=self.bond_dim,
            parameter_count=self.parameter_count,
            state_space_log10=float(n),
            expected_principal_loss_usd=expected_loss,
            joint_attachment_probability=joint_attach,
            portfolio_exceedance_probability=exceed,
            portfolio_stop_loss_usd=stop_loss,
            all_bonds_exhausted_probability=all_exhaust,
            elapsed_seconds=time.perf_counter() - t0,
            truncation_error=0.0,
        )


def from_adb_model(model) -> CausalLatentMPS:
    T = model.dimension
    rates = model.earthquake_rates
    totals = rates.sum(axis=1)
    c = model.earthquake_common_rate
    marks = rates / totals[:, None]

    p_s = np.array([poisson.pmf(k, c * T) for k in range(4)] + [poisson.sf(3, c * T)])

    cfg = model.config
    q = np.asarray(cfg["precipitation_annual_trigger_probability"], dtype=float)
    rain = -np.log1p(-q) * float(cfg["precipitation_rate_multiplier"])
    common_r = float(cfg["precipitation_common_rate_share"]) * rain.min()
    p_z = np.array([np.exp(-common_r * T), 1.0 - np.exp(-common_r * T)])
    shock_probs = np.outer(p_s, p_z).reshape(10)

    conditionals = []
    for i in range(2):
        own_eq = _own_steps((totals[i] - c) * T, marks[i])
        eq_given_s = np.array([_add_marks(own_eq, marks[i], s) for s in range(EQ_LEVELS)])
        p_own_r = 1.0 - np.exp(-(rain[i] - common_r) * T)
        r_given_z = np.array([[1.0 - p_own_r, p_own_r], [0.0, 1.0]])
        cond = np.zeros((10, 10), dtype=np.float64)
        for s in range(5):
            for z in range(2):
                cond[2 * s + z] = np.outer(eq_given_s[s], r_given_z[z]).reshape(10)
        conditionals.append(cond)

    return CausalLatentMPS(
        shock_probs=shock_probs,
        country_conditionals=conditionals,
        eq_limits=model.eq_limit,
        rain_limits=model.rain_limit,
        threshold=model.threshold,
    )


def build_ncountry_portfolio(base_model, n_countries: int) -> CausalLatentMPS:
    base_mps = from_adb_model(base_model)
    conditionals = [base_mps.conditionals[i % 2].copy() for i in range(n_countries)]
    eq_limits = np.array([base_model.eq_limit[i % 2] for i in range(n_countries)])
    rain_limits = np.array([base_model.rain_limit[i % 2] for i in range(n_countries)])
    threshold = 0.5 * float((eq_limits + rain_limits).sum())
    return CausalLatentMPS(
        shock_probs=base_mps.shock_probs,
        country_conditionals=conditionals,
        eq_limits=eq_limits,
        rain_limits=rain_limits,
        threshold=threshold,
    )


def solve_adb_mps(model, samples: int = 0, seed: int = 0) -> dict:
    t0 = time.perf_counter()
    mps = from_adb_model(model)
    t_prep = time.perf_counter() - t0
    res = mps.contract_all()
    values = [
        res.expected_principal_loss_usd,
        res.joint_attachment_probability,
        res.portfolio_exceedance_probability,
        res.portfolio_stop_loss_usd,
        res.all_bonds_exhausted_probability,
    ]
    estimates = {
        name: {
            "estimate": float(val),
            "standard_error": 0.0,
            "ci95_low": float(val),
            "ci95_high": float(val),
        }
        for name, val in zip(model.metric_names, values)
    }
    return {
        "estimates": estimates,
        "resources": {
            "method": "causal-mps-exact",
            "execution": "classical_cpu",
            "qubits": 0,
            "shots": 0,
            "model_evaluations": int(res.parameter_count),
            "preparation_seconds": float(t_prep),
            "solver_seconds": float(time.perf_counter() - t0),
            "bond_dim": int(res.bond_dim),
            "truncation_error": 0.0,
        },
        "uncertainty": "Exact latent-shock MPS contraction.",
    }
