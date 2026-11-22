"""Evaluate the tilted-IQAE prototype against the classical baselines.

Mirrors evaluation-v1 scoring (EVALUATION.md): RMSE over repeats against the
exact reference, per-target tolerances, every target must pass. Adds the brief's
"accuracy per query" view: how many model evaluations (classical samples) or
oracle calls (applications of A) each method needs before it passes.

Runs in-process, so it also works on Windows (the official process evaluator
needs Linux/macOS). Process start-up time is therefore not included.

    python benchmarks/qae_eval.py --holdout-seed 2026 --out results/qae-eval
    python benchmarks/qae_eval.py --qae-only --kappa 0.5 1 2      # tuning
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from qjlb.bond import BOND_METHODS
from qjlb.cli import load_model
from qjlb.evaluation import tolerances
from qjlb.holdout import generate
from qjlb_qae.solver import solve

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ["adb-base", "adb-independent", "adb-common-shock", "adb-rain-stress", "adb-rain-cover"]
CLASSICAL = ("mc", "qmc", "stratified", "importance")
SHORT = {
    "term_expected_principal_loss_usd": "E[L]",
    "term_joint_attachment_probability": "P(both pay)",
    "term_portfolio_loss_exceedance_probability": "P(L>T)",
    "term_portfolio_stop_loss_usd": "E[(L-T)+]",
    "term_both_bonds_exhausted_probability": "P(both exhausted)",
}


def cases(holdout_seed, include_public=True):
    out = []
    if include_public:
        out += [(name, ROOT / "scenarios" / f"{name}.json") for name in PUBLIC]
    if holdout_seed is not None:
        suite = Path(tempfile.mkdtemp()) / "suite"
        manifest = generate(suite, ROOT / "data", seed=holdout_seed)
        for name in manifest["cases"]:
            config = json.loads((suite / name).read_text())
            if config["track"] == "adb_drb":
                out.append((f"holdout{holdout_seed}-{Path(name).stem}-T{config['term_years']}", suite / name))
    return out


def normalised_rmse(estimates, reference, bounds, names):
    est = np.array([[e[n] for n in names] for e in estimates])
    ref = np.array([reference["estimates"][n]["estimate"] for n in names])
    tol = np.array([bounds[n] for n in names])
    return np.sqrt(((est - ref) ** 2).mean(axis=0)) / tol


def run_qae(model, reference, bounds, repeats, **options):
    names = model.metric_names
    estimates, calls, seconds = [], [], []
    for seed in range(repeats):
        start = time.perf_counter()
        result = solve(model, 8192, 1000 + seed, **options)
        seconds.append(time.perf_counter() - start)
        estimates.append({n: result["estimates"][n]["estimate"] for n in names})
        calls.append([i["oracle_calls"] for i in result["resources"]["per_target"]])
        info = result["resources"]["per_target"]
    nr = normalised_rmse(estimates, reference, bounds, names)
    return {
        "normalised_rmse": nr,
        "calls": np.median(np.array(calls), axis=0),
        "max_k": [i["max_k"] for i in info],
        "W": [i.get("W", 0.0) for i in info],
        "tilt": [i.get("tilt") for i in info],
        "seconds": float(np.median(seconds)),
        "qubits": result["resources"]["qubits"],
    }


def samples_to_pass(model, method, reference, bounds, repeats, budgets):
    """Smallest budget (from the doubling ladder) at which each target passes."""
    names = model.metric_names
    first = [None] * len(names)
    for n in budgets:
        estimates = []
        for seed in range(repeats):
            r = BOND_METHODS[method](model, n, 5000 + seed)
            estimates.append({k: r["estimates"][k]["estimate"] for k in names})
        nr = normalised_rmse(estimates, reference, bounds, names)
        for i, value in enumerate(nr):
            if first[i] is None and value <= 1:
                first[i] = n
        if all(f is not None for f in first):
            break
    return first


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--holdout-seed", type=int, default=2026)
    parser.add_argument("--no-public", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--kappa", type=float, nargs="+", default=[1.0])
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--qae-only", action="store_true")
    parser.add_argument("--max-log2-budget", type=int, default=20)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    budgets = [2**p for p in range(10, args.max_log2_budget + 1)]
    rows = []
    for case, path in cases(args.holdout_seed, not args.no_public):
        model, _ = load_model(path)
        reference = BOND_METHODS["exact"](model)
        bounds = tolerances(model, reference)
        names = model.metric_names
        print(f"\n== {case} (T={model.dimension}) ==", flush=True)
        qae = {k: run_qae(model, reference, bounds, args.repeats, kappa=k, shots=args.shots) for k in args.kappa}
        classical = {} if args.qae_only else {
            m: samples_to_pass(model, m, reference, bounds, args.repeats, budgets) for m in CLASSICAL
        }
        for i, name in enumerate(names):
            row = {"case": case, "term_years": model.dimension, "target": SHORT[name],
                   "reference": reference["estimates"][name]["estimate"], "tolerance": bounds[name]}
            for k, q in qae.items():
                row[f"qae_k{k}_norm_rmse"] = round(float(q["normalised_rmse"][i]), 4)
                row[f"qae_k{k}_oracle_calls"] = int(q["calls"][i])
                row[f"qae_k{k}_max_k"] = q["max_k"][i]
            for m, first in classical.items():
                row[f"{m}_samples_to_pass"] = first[i] if first[i] is not None else f">{budgets[-1]}"
            if classical:
                passing = [f[i] for f in classical.values() if f[i] is not None]
                row["best_classical_samples"] = min(passing) if passing else f">{budgets[-1]}"
            rows.append(row)
            q = qae[args.kappa[0]]
            best = row.get("best_classical_samples", "-")
            print(f"  {SHORT[name]:18s} QAE calls {int(q['calls'][i]):>8,}  normRMSE {q['normalised_rmse'][i]:.3f}"
                  f"  | best classical samples to pass: {best}", flush=True)
        for k, q in qae.items():
            print(f"  kappa={k}: all pass={bool((q['normalised_rmse'] <= 1).all())}, worst normRMSE "
                  f"{q['normalised_rmse'].max():.3f}, total calls {int(q['calls'].sum()):,}, median solve {q['seconds']:.2f}s")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / "qae_vs_classical.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {out / 'qae_vs_classical.csv'}")


if __name__ == "__main__":
    main()
