"""Ablation: how much comes from the tilt, and how much from amplitude estimation?

For every target, cost to reach the official evaluation-v1 tolerance:
  mc_samples      plain Monte Carlo on the term distribution, N = Var_p / tol^2
  is_samples      classical importance sampling from the SAME tilted network
                  proposal used by the quantum circuit, N = Var_q(f p/q) / tol^2
  iqae_calls      IQAE without the tilt (median oracle calls over repeats)
  tilted_calls    IQAE with the tilt (the prototype)

Sample counts are exact expectations from the variance (RMSE = sqrt(Var / N));
oracle calls are medians of real IQAE runs at kappa = 1. The variances use the
exact distribution, which is fine for evaluation but never used by the solver.

    python benchmarks/qae_ablation.py --holdout-seed 2026
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from qjlb.bond import BOND_METHODS
from qjlb.cli import load_model
from qjlb.evaluation import tolerances
from qjlb.holdout import generate
from qjlb_qae.loader import ORDER, apply_tilt, network, path_probabilities, register_values
from qjlb_qae.solver import _log_ratio_tables, estimate_target, payoffs, select_tilt

ROOT = Path(__file__).resolve().parents[1]
SHORT = ["E[L]", "P(both pay)", "P(L>T)", "E[(L-T)+]", "P(both exh.)"]


def analytic_samples(model, i, f_grid, scale, tol):
    nodes = network(model)
    v = register_values(nodes)
    valid = (v["s"] < 5) & (v["a"] < 5) & (v["b"] < 5)
    f = np.where(valid, f_grid[np.minimum(v["a"], 4), np.minimum(v["b"], 4), v["r1"], v["r2"]], 0.0) * scale
    p = path_probabilities(nodes)
    mu = p @ f
    mc = (p @ f**2 - mu**2) / tol**2
    gammas, _ = select_tilt(nodes, v, f / max(scale, 1e-300), valid)
    if gammas and any(gammas.values()):
        tables = _log_ratio_tables(nodes, v)
        w = np.exp(sum(tables[name, gammas[name]] for name in ORDER))
        apply_tilt(nodes, gammas)
        q = path_probabilities(nodes)
        var_q = q @ (f * w) ** 2 - mu**2
        is_n = max(var_q, 0.0) / tol**2
    else:
        is_n = mc
    return max(mc, 1.0), max(is_n, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    configs = [(n, ROOT / "scenarios" / f"{n}.json") for n in ("adb-base", "adb-independent")]
    suite = Path(tempfile.mkdtemp()) / "suite"
    manifest = generate(suite, ROOT / "data", seed=args.holdout_seed)
    for name in manifest["cases"]:
        cfg = json.loads((suite / name).read_text())
        if cfg["track"] == "adb_drb":
            configs.append((f"holdout-{Path(name).stem}-T{cfg['term_years']}", suite / name))

    print(f"{'case':26s} {'target':13s} {'MC samples':>11s} {'IS(same q)':>11s} {'IQAE calls':>11s} {'tilted IQAE':>11s}")
    for case, path in configs:
        model, _ = load_model(path)
        reference = BOND_METHODS["exact"](model)
        bounds = tolerances(model, reference)
        f_grids, scales = payoffs(model)
        for i, name in enumerate(model.metric_names):
            if scales[i] <= 0 or not f_grids[i].any():
                continue
            mc, is_n = analytic_samples(model, i, f_grids[i], scales[i], bounds[name])
            calls = {}
            for tilt in (False, True):
                c = [estimate_target(model, i, f_grids[i], scales[i], np.random.default_rng(seed),
                                     use_tilt=tilt)[2]["oracle_calls"] for seed in range(args.repeats)]
                calls[tilt] = int(np.median(c))
            print(f"{case:26s} {SHORT[i]:13s} {mc:>11,.0f} {is_n:>11,.0f} {calls[False]:>11,d} {calls[True]:>11,d}", flush=True)


if __name__ == "__main__":
    main()
