"""Two-country principal-loss model calibrated to ADB's public 2026 disclosures."""

from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.stats import qmc, t
from .model import entries

BOND_METRICS = (
    "term_expected_principal_loss_usd",
    "term_joint_attachment_probability",
    "term_portfolio_loss_exceedance_probability",
    "term_portfolio_stop_loss_usd",
    "term_both_bonds_exhausted_probability",
)


def positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} is outside its permitted range")
    return value


def earthquake_jump_rates(tail):
    """Recover quarter-step Poisson jump rates from four annual tail probabilities."""
    tail = np.asarray(tail, dtype=float)
    if tail.shape != (4,) or not np.all(np.isfinite(tail)):
        raise ValueError("earthquake tails must contain four finite probabilities")
    if np.any(tail <= 0) or np.any(tail >= 1) or np.any(np.diff(tail) > 0):
        raise ValueError("earthquake tails must decrease within (0, 1)")
    p0, p1, p2, p3, _ = -np.diff(np.r_[1, tail, 0])
    total = -np.log(p0)
    r1 = p1 / p0
    r2 = p2 / p0 - r1**2 / 2
    r3 = p3 / p0 - r1 * r2 - r1**3 / 6
    rates = np.array([r1, r2, r3, total - r1 - r2 - r3])
    if np.any(rates < 0):
        raise ValueError("annual tails do not support nonnegative Poisson jump rates")
    return rates


def earthquake_year(rates_a, rates_b, common_rate):
    """Joint annual payouts, with shared events and independent severity marks."""
    totals = np.array([sum(rates_a), sum(rates_b)])
    if common_rate < 0 or common_rate > min(totals):
        raise ValueError("shared earthquake rate exceeds a country's trigger rate")
    severity_a, severity_b = rates_a / totals[0], rates_b / totals[1]
    jumps = np.zeros((5, 5))
    jumps[1:, 0] = (totals[0] - common_rate) * severity_a
    jumps[0, 1:] = (totals[1] - common_rate) * severity_b
    jumps[1:, 1:] = common_rate * np.outer(severity_a, severity_b)
    states = np.array(list(np.ndindex(5, 5)))
    generator = np.zeros((25, 25))
    rows = np.arange(25)
    for a, b in np.argwhere(jumps > 0):
        target = np.minimum(states + [a, b], 4)
        columns = target[:, 0] * 5 + target[:, 1]
        generator[rows, columns] += jumps[a, b]
        generator[rows, rows] -= jumps[a, b]
    probabilities = expm(generator)[0].reshape(5, 5)
    if np.min(probabilities) < -1e-12:
        raise ValueError("invalid earthquake probability calculation")
    probabilities = np.maximum(probabilities, 0)
    return probabilities / probabilities.sum()


def precipitation_year(probabilities, common_share, rate_multiplier):
    """Illustrative annual trigger process. Each country's payment is binary."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (2,) or not np.all(np.isfinite(probabilities)):
        raise ValueError("precipitation probabilities must contain two finite values")
    if np.any(probabilities <= 0) or np.any(probabilities >= 1):
        raise ValueError("precipitation probabilities must be in (0, 1)")
    if not 0 <= common_share <= 1:
        raise ValueError("precipitation common rate share must be in [0, 1]")
    rates = -np.log1p(-probabilities) * rate_multiplier
    common = common_share * min(rates)
    neither = np.exp(-(rates.sum() - common))
    a_only = np.exp(-rates[1]) - neither
    b_only = np.exp(-rates[0]) - neither
    both = 1 - neither - a_only - b_only
    return np.array([[neither, b_only], [a_only, both]])


class BondModel:
    metric_names = BOND_METRICS
    definition_version = "adb-quarter-step-principal-v1"
    reference_status = "exact_for_discrete_principal_loss_model"

    def __init__(self, config, terms_path):
        self.config = copy.deepcopy(config)
        c = self.config
        if c.get("schema_version") != 1 or c.get("track") != "adb_drb":
            raise ValueError("requires a schema 1 adb_drb configuration")
        if c.get("countries") != ["KGZ", "TJK"]:
            raise ValueError("public ADB calibration covers KGZ and TJK in that order")
        years = c.get("term_years")
        if (
            isinstance(years, bool)
            or not isinstance(years, int)
            or not 1 <= years <= 10
        ):
            raise ValueError("term_years must be an integer from 1 to 10")
        self.dimension = years
        self.countries = tuple(c["countries"])
        terms_path = Path(terms_path)
        terms = json.loads(terms_path.read_text())
        self.data_hashes = {
            "adb_terms": hashlib.sha256(terms_path.read_bytes()).hexdigest()
        }
        self.eq_limit = self._limits("earthquake_sublimit_usd")
        self.rain_limit = self._limits("precipitation_sublimit_usd")
        self.principal = self.eq_limit + self.rain_limit
        self.threshold = positive(
            c["portfolio_threshold_usd"], "portfolio threshold", True
        )
        rates = np.array(
            [
                earthquake_jump_rates(
                    terms["annual_earthquake_exceedance_probabilities"][country]
                )
                for country in self.countries
            ]
        )
        totals = rates.sum(axis=1)
        conditionals = terms["same_event_conditional_trigger_probabilities"]
        # Rounded public event conditionals imply slightly different common rates.
        # Average them; retain each country's calibrated marginal rate.
        common = np.mean(
            [
                totals[0] * conditionals["TJK_given_KGZ"],
                totals[1] * conditionals["KGZ_given_TJK"],
            ]
        ) * positive(
            c["earthquake_common_rate_multiplier"],
            "earthquake common rate multiplier",
            True,
        )
        self.earthquake_rates = rates
        self.earthquake_common_rate = float(common)
        self.earthquake_probabilities = earthquake_year(*rates, common)
        rain_share = positive(
            c["precipitation_common_rate_share"],
            "precipitation common rate share",
            True,
        )
        rain_multiplier = positive(
            c["precipitation_rate_multiplier"], "precipitation rate multiplier"
        )
        self.precipitation_probabilities = precipitation_year(
            c["precipitation_annual_trigger_probability"], rain_share, rain_multiplier
        )
        self.states = np.array(list(np.ndindex(5, 5, 2, 2)))
        self.probabilities = np.array(
            [
                self.earthquake_probabilities[a, b]
                * self.precipitation_probabilities[x, y]
                for a, b, x, y in self.states
            ]
        )
        self.probabilities /= self.probabilities.sum()
        self.cdf = np.cumsum(self.probabilities)
        self.cdf[-1] = 1.0
        self.state_count = len(self.states)
        self.caps = np.array([4, 4, 1, 1])
        combined = np.minimum(
            self.states[:, None, :] + self.states[None, :, :], self.caps
        )
        self.destinations = np.ravel_multi_index(
            combined.reshape(-1, 4).T, (5, 5, 2, 2)
        )
        raw = json.dumps(
            {
                "definition_version": self.definition_version,
                "config": c,
                "data_sha256": self.data_hashes,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self.fingerprint = hashlib.sha256(raw.encode()).hexdigest()

    def _limits(self, name):
        values = self.config[name]
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError(f"{name} must contain two positive limits")
        return np.array([positive(value, name) for value in values])

    @classmethod
    def load(cls, path):
        path = Path(path)
        config = json.loads(path.read_text())
        return cls(config, path.parent / config["terms_file"])

    def from_uniforms(self, uniforms):
        u = np.asarray(uniforms, dtype=float)
        if u.ndim != 2 or u.shape[1] != self.dimension or not np.all(np.isfinite(u)):
            raise ValueError("uniforms must have shape (samples, term_years)")
        if np.any(u < 0) or np.any(u >= 1):
            raise ValueError("uniforms must be in [0, 1)")
        return np.searchsorted(self.cdf, u, side="right")

    def principal_loss(self, states):
        return states[:, :2] * self.eq_limit / 4 + states[:, 2:] * self.rain_limit

    def observables(self, states):
        country_loss = self.principal_loss(states)
        total = country_loss.sum(axis=1)
        return np.column_stack(
            [
                total,
                np.all(country_loss > 0, axis=1),
                total > self.threshold,
                np.maximum(total - self.threshold, 0),
                np.all(country_loss >= self.principal, axis=1),
            ]
        )

    def evaluate(self, paths):
        paths = np.asarray(paths)
        if paths.ndim != 2 or paths.shape[1] != self.dimension:
            raise ValueError("paths must have shape (samples, term_years)")
        if (
            not np.issubdtype(paths.dtype, np.integer)
            or np.any(paths < 0)
            or np.any(paths >= self.state_count)
        ):
            raise ValueError("paths must contain valid integer annual-state indices")
        term_states = np.minimum(self.states[paths].sum(axis=1), self.caps)
        return {
            "principal_loss_usd": self.principal_loss(term_states),
            "metrics": self.observables(term_states),
        }

    def term_distribution(self):
        distribution = np.zeros(self.state_count)
        distribution[0] = 1.0
        for _ in range(self.dimension):
            distribution = np.bincount(
                self.destinations,
                weights=np.outer(distribution, self.probabilities).ravel(),
                minlength=self.state_count,
            )
        return distribution


def result(
    model, means, errors, method, budget, critical=1.959963984540054, **resources
):
    return {
        "estimates": entries(means, errors, critical, model.metric_names),
        "resources": {
            "method": method,
            "model_evaluations": budget,
            "samples": budget,
            "execution": "classical_cpu",
            **resources,
        },
        "uncertainty": "Approximate sampling intervals; null for zero observed variance.",
    }


def bond_exact(model, budget=0, seed=0):
    values = model.term_distribution() @ model.observables(model.states)
    output = result(
        model,
        values,
        None,
        "exact",
        model.state_count,
        transition_products=model.dimension * model.state_count**2,
    )
    for item in output["estimates"].values():
        item.update(
            standard_error=0.0, ci95_low=item["estimate"], ci95_high=item["estimate"]
        )
    output["uncertainty"] = (
        "Exact finite-state reference up to floating-point error; calibration and structural uncertainty excluded."
    )
    return output


def bond_mc(model, budget, seed):
    if budget < 2:
        raise ValueError("budget must be at least 2")
    u = np.random.default_rng(seed).random((budget, model.dimension))
    values = model.evaluate(model.from_uniforms(u))["metrics"]
    return result(
        model,
        values.mean(axis=0),
        values.std(axis=0, ddof=1) / np.sqrt(budget),
        "mc",
        budget,
    )


def bond_qmc(model, budget, seed):
    if budget < 16:
        raise ValueError("budget must be at least 16")
    power = int(np.floor(np.log2(budget // 8)))
    means = []
    for child in np.random.SeedSequence(seed).spawn(8):
        u = qmc.Sobol(
            model.dimension, scramble=True, seed=int(child.generate_state(1)[0])
        ).random_base2(power)
        means.append(model.evaluate(model.from_uniforms(u))["metrics"].mean(axis=0))
    means = np.array(means)
    return result(
        model,
        means.mean(axis=0),
        means.std(axis=0, ddof=1) / np.sqrt(8),
        "qmc",
        8 * 2**power,
        critical=t.ppf(0.975, 7),
        scrambles=8,
    )


def bond_stratified(model, budget, seed):
    """Stratify by first year with any payment; the no-payment path is exactly zero."""
    if budget < 2 * model.dimension:
        raise ValueError("budget must allow two samples per term year")
    rng = np.random.default_rng(seed)
    p0 = model.probabilities[0]
    active = model.probabilities[1:] / (1 - p0)
    means, variances = np.zeros(5), np.zeros(5)
    for year in range(model.dimension):
        n = budget // model.dimension + (year < budget % model.dimension)
        paths = model.from_uniforms(rng.random((n, model.dimension)))
        paths[:, :year] = 0
        paths[:, year] = rng.choice(np.arange(1, model.state_count), size=n, p=active)
        weight = p0**year * (1 - p0)
        values = model.evaluate(paths)["metrics"] * weight
        means += values.mean(axis=0)
        variances += values.var(axis=0, ddof=1) / n
    return result(
        model,
        means,
        np.sqrt(variances),
        "stratified",
        budget,
        annual_states_scanned=model.state_count,
    )


def bond_importance(model, budget, seed):
    if budget < 2:
        raise ValueError("budget must be at least 2")
    severity = model.principal_loss(model.states).sum(axis=1)
    tilted = model.probabilities * severity
    proposal = 0.5 * model.probabilities + 0.5 * tilted / tilted.sum()
    rng = np.random.default_rng(seed)
    paths = rng.choice(model.state_count, size=(budget, model.dimension), p=proposal)
    weights = np.prod(model.probabilities[paths] / proposal[paths], axis=1)
    values = model.evaluate(paths)["metrics"] * weights[:, None]
    return result(
        model,
        values.mean(axis=0),
        values.std(axis=0, ddof=1) / np.sqrt(budget),
        "importance",
        budget,
        annual_states_scanned=model.state_count,
        effective_sample_size=float(weights.sum() ** 2 / np.square(weights).sum()),
    )


BOND_METHODS = {
    "mc": bond_mc,
    "qmc": bond_qmc,
    "stratified": bond_stratified,
    "importance": bond_importance,
    "exact": bond_exact,
}
