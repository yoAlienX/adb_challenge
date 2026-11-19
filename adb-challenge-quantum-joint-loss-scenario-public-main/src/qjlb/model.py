"""World Bank catalogue integration, with explicitly synthetic loss-based layers.

All target quantities are sums of annual rates times a per-row observable.
The source's conditional mean loss is held fixed. Secondary uncertainty and
actual parametric trigger measurements are outside this model.
"""

from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import qmc, t

ELT_METRICS = (
    "annual_economic_loss_usd",
    "annual_layer_payment_usd",
    "joint_layer_exceedance_rate_per_year",
    "portfolio_layer_exceedance_rate_per_year",
    "annual_layer_stop_loss_usd",
)


class ELTModel:
    metric_names = ELT_METRICS
    reference_status = "exact_conditional_mean_catalogue"
    dimension = 1

    def __init__(self, config, dataset_directory):
        self.config = copy.deepcopy(config)
        c = self.config
        if c.get("schema_version") != 2 or c.get("track") != "worldbank_elt":
            raise ValueError("requires a schema 2 worldbank_elt configuration")
        self.dataset_directory = Path(dataset_directory)
        self.countries = tuple(c["countries"])
        if len(self.countries) < 2 or len(set(self.countries)) != len(self.countries):
            raise ValueError("at least two distinct countries required")
        self.n_countries = len(self.countries)
        self.definition_version = (
            "elt-mean-loss-v1" if self.n_countries == 2 else "elt-all-country-v1"
        )
        self.attachment = self._parameter("attachment_usd", (2, self.n_countries))
        self.exhaustion = self._parameter("exhaustion_usd", (2, self.n_countries))
        self.limit = self._parameter("limit_usd", (self.n_countries,))
        if np.any(self.exhaustion <= self.attachment) or np.any(self.limit <= 0):
            raise ValueError("require positive limits and exhaustion > attachment >= 0")
        self.threshold = float(c["portfolio_threshold_usd"])
        if not np.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("invalid portfolio threshold")
        audit = json.loads((self.dataset_directory / "audit.json").read_text())
        blocks, hashes = [], {}
        for h, peril in enumerate(("seismic", "flood")):
            path = self.dataset_directory / f"{peril}.npz"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != audit["perils"][peril]["sha256"]:
                raise ValueError(f"{peril}: dataset checksum mismatch")
            hashes[peril] = digest
            with np.load(path, allow_pickle=False) as data:
                order = data["countries"].tolist()
                columns = [order.index(country) for country in self.countries]
                rates = data["annual_rate"].copy()
                ids = data["event_id"].copy()
                loss = data["loss_usd"][:, columns].copy()
                sd = data["sd_usd"][:, columns].copy()
            if (
                rates.ndim != 1
                or ids.shape != rates.shape
                or loss.shape != (len(rates), self.n_countries)
                or sd.shape != loss.shape
            ):
                raise ValueError("invalid dataset shapes")
            if len(np.unique(ids)) != len(ids) or np.any(ids <= 0):
                raise ValueError(
                    "event IDs must be distinct positive integers within each peril"
                )
            if (
                not all(np.all(np.isfinite(a)) for a in (rates, loss, sd))
                or np.any(rates <= 0)
                or np.any(loss < 0)
                or np.any(sd < 0)
            ):
                raise ValueError("invalid rates or losses")
            blocks.append((ids, rates, loss, sd, np.full(len(ids), h, dtype=np.int8)))
        self.event_id, self.rates, self.loss, self.sd, self.peril = [
            np.concatenate([b[i] for b in blocks]) for i in range(5)
        ]
        self.total_rate = float(self.rates.sum())
        self.probabilities = self.rates / self.total_rate
        self.cdf = np.cumsum(self.probabilities)
        self.cdf[-1] = 1.0
        self.state_count = len(self.rates)
        raw = json.dumps(
            {
                "definition_version": self.definition_version,
                "config": c,
                "data_sha256": hashes,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self.fingerprint = hashlib.sha256(raw.encode()).hexdigest()
        self.data_hashes = hashes

    def _parameter(self, name, shape):
        value = np.asarray(self.config[name], dtype=float)
        if value.shape != shape or not np.all(np.isfinite(value)) or np.any(value < 0):
            raise ValueError(
                f"{name} must have shape {shape} and finite nonnegative values"
            )
        return value

    @classmethod
    def load(cls, path):
        path = Path(path)
        config = json.loads(path.read_text())
        return cls(config, path.parent / config["dataset_directory"])

    def from_uniforms(self, uniforms):
        u = np.asarray(uniforms, dtype=float)
        if (
            u.ndim != 2
            or u.shape[1] != 1
            or not np.all(np.isfinite(u))
            or np.any(u < 0)
            or np.any(u >= 1)
        ):
            raise ValueError("uniforms must have shape (samples, 1), in [0,1)")
        return np.searchsorted(self.cdf, u[:, 0], side="right")[:, None]

    def evaluate(self, states):
        states = np.asarray(states)
        if (
            states.ndim != 2
            or states.shape[1] != 1
            or not np.issubdtype(states.dtype, np.integer)
            or np.any(states < 0)
            or np.any(states >= self.state_count)
        ):
            raise ValueError(
                "states must contain valid integer row indices with shape (samples,1)"
            )
        idx = states[:, 0]
        h = self.peril[idx]
        economic = self.loss[idx]
        payout = self.limit * np.clip(
            (economic - self.attachment[h]) / (self.exhaustion[h] - self.attachment[h]),
            0,
            1,
        )
        total = payout.sum(axis=1)
        values = np.column_stack(
            [
                economic.sum(axis=1),
                total,
                np.all(payout > 0, axis=1),
                total > self.threshold,
                np.maximum(total - self.threshold, 0),
            ]
        )
        return {
            "economic_loss_usd": economic,
            "payout_usd": payout,
            "per_event_metrics": values,
            "metrics": self.total_rate * values,
        }


def entries(means, errors=None, critical=1.959963984540054, metric_names=ELT_METRICS):
    out = {}
    for i, name in enumerate(metric_names):
        se = None if errors is None else float(errors[i])
        valid = se is not None and se > 0
        out[name] = {
            "estimate": float(means[i]),
            "standard_error": se,
            "ci95_low": max(0.0, float(means[i] - critical * se)) if valid else None,
            "ci95_high": float(means[i] + critical * se) if valid else None,
        }
    return out


def sample_result(values, n, method, extra=None):
    return {
        "estimates": entries(
            values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(n)
        ),
        "resources": {
            "method": method,
            "model_evaluations": n,
            "samples": n,
            "execution": "classical_cpu",
            **(extra or {}),
        },
        "uncertainty": "approximate sampling interval; null for zero observed variance",
    }


def elt_mc(model, budget, seed):
    if budget < 2:
        raise ValueError("budget must be >= 2")
    u = np.random.default_rng(seed).random((budget, 1))
    return sample_result(
        model.evaluate(model.from_uniforms(u))["metrics"], budget, "elt_mc"
    )


def elt_rqmc(model, budget, seed):
    if budget < 16:
        raise ValueError("elt_rqmc budget must be >= 16")
    power = int(np.floor(np.log2(budget // 8)))
    means = []
    for child in np.random.SeedSequence(seed).spawn(8):
        u = qmc.Sobol(
            1, scramble=True, seed=int(child.generate_state(1)[0])
        ).random_base2(power)
        means.append(model.evaluate(model.from_uniforms(u))["metrics"].mean(axis=0))
    means = np.asarray(means)
    return {
        "estimates": entries(
            means.mean(axis=0), means.std(axis=0, ddof=1) / np.sqrt(8), t.ppf(0.975, 7)
        ),
        "resources": {
            "method": "elt_rqmc",
            "model_evaluations": 8 * 2**power,
            "samples": 8 * 2**power,
            "scrambles": 8,
            "execution": "classical_cpu",
        },
        "uncertainty": "approximate t interval across eight independent scrambles; null for zero replicate variance",
    }


def elt_stratified(model, budget, seed):
    """Allocate half the budget to each peril; retain annual-rate scaling."""
    if budget < 4:
        raise ValueError("stratified budget must be >= 4")
    rng = np.random.default_rng(seed)
    means = np.zeros(5)
    variances = np.zeros(5)
    for h, n in enumerate((budget // 2, budget - budget // 2)):
        idx = np.flatnonzero(model.peril == h)
        rate = model.rates[idx].sum()
        cdf = np.cumsum(model.rates[idx] / rate)
        cdf[-1] = 1.0
        selected = idx[np.searchsorted(cdf, rng.random(n), side="right")]
        v = model.evaluate(selected[:, None])["per_event_metrics"] * rate
        means += v.mean(axis=0)
        variances += v.var(axis=0, ddof=1) / n
    return {
        "estimates": entries(means, np.sqrt(variances)),
        "resources": {
            "method": "elt_stratified",
            "model_evaluations": budget,
            "samples": budget,
            "catalogue_rows_scanned": model.state_count,
            "execution": "classical_cpu",
        },
        "uncertainty": "approximate independent-stratum normal interval; null for zero observed variance",
    }


def elt_is(model, budget, seed):
    if budget < 2:
        raise ValueError("budget must be >= 2")
    # Defensive loss-weighted proposal. The full loss scan and CDF build are timed.
    severity = model.loss.sum(axis=1)
    tilted = model.probabilities * severity
    tilted = tilted / tilted.sum() if tilted.sum() > 0 else model.probabilities
    proposal = 0.5 * model.probabilities + 0.5 * tilted
    cdf = np.cumsum(proposal)
    cdf[-1] = 1.0
    idx = np.searchsorted(cdf, np.random.default_rng(seed).random(budget), side="right")
    weights = model.probabilities[idx] / proposal[idx]
    values = model.evaluate(idx[:, None])["metrics"] * weights[:, None]
    return sample_result(
        values,
        budget,
        "elt_is",
        {
            "catalogue_rows_scanned": model.state_count,
            "effective_sample_size": float(
                weights.sum() ** 2 / np.square(weights).sum()
            ),
            "mean_weight": float(weights.mean()),
            "defensive_fraction": 0.5,
        },
    )


def elt_exact(model, budget=0, seed=0):
    # One vectorised pass. Count its full work even when used as a reference.
    data = model.evaluate(np.arange(model.state_count)[:, None])
    values = model.rates @ data["per_event_metrics"]
    estimates = entries(values)
    for item in estimates.values():
        item.update(
            standard_error=0.0, ci95_low=item["estimate"], ci95_high=item["estimate"]
        )
    return {
        "estimates": estimates,
        "resources": {
            "method": "elt_exact",
            "model_evaluations": model.state_count,
            "execution": "classical_cpu",
        },
        "uncertainty": "Exact for the fixed conditional-mean catalogue and configured layers, up to floating-point error; excludes source model uncertainty",
    }
