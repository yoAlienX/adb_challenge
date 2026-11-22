"""Comprehensive benchmark for the upgraded qjlb_qae suite.

Evaluates:
  1. All 7 public scenarios (5 ADB + 2 World Bank) in both `tilted_iqae` and `hybrid` modes
  2. All 6 unseen holdout cases (3 ADB + 3 World Bank, seed 2026)
  3. Factorized linear-depth vs. monolithic tilted payoff CX gate counts
  4. Exact chi=10 Causal MPS scaling from n = 2 to n = 50 countries (10^50 states)
  5. AWS Braket QPU noise + Readout/ZNE mitigation coherence table
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qiskit import transpile
from qjlb.bond import BondModel, bond_exact
from qjlb.cli import load_model
from qjlb.evaluation import tolerances
from qjlb.holdout import generate as generate_suite
from qjlb.model import elt_exact
from qjlb_qae.circuits import factorized_a_operator
from qjlb_qae.loader import network
from qjlb_qae.mps import build_ncountry_portfolio
from qjlb_qae.noise import evaluate_qpu_coherence
from qjlb_qae.solver import build_target_circuit, payoffs, solve


PUBLIC_SCENARIOS = [
    "adb-base",
    "adb-independent",
    "adb-common-shock",
    "adb-rain-stress",
    "adb-rain-cover",
    "kaz-kgz-base",
    "tjk-uzb-base",
]


def eval_case(label: str, model, repeats: int, mode: str) -> dict:
    ref = bond_exact(model) if isinstance(model, BondModel) else elt_exact(model)
    bounds = tolerances(model, ref)
    runs = [solve(model, 8192, seed, mode=mode) for seed in range(repeats)]
    norm_rmses = {}
    for name in model.metric_names:
        errs = [r["estimates"][name]["estimate"] - ref["estimates"][name]["estimate"] for r in runs]
        rmse = math.sqrt(float(np.mean(np.square(errs))))
        norm_rmses[name] = rmse / bounds[name] if bounds[name] > 0 else 0.0
    worst = max(norm_rmses.values())
    passed = worst <= 1.0
    med_evals = int(np.median([r["resources"]["model_evaluations"] for r in runs]))
    med_qcalls = int(np.median([r["resources"].get("quantum_oracle_calls", r["resources"]["model_evaluations"]) for r in runs]))
    med_time = float(np.median([r["resources"]["solver_seconds"] for r in runs]))
    qubits = runs[0]["resources"]["qubits"]
    return {
        "label": label,
        "track": model.config["track"],
        "mode": mode,
        "qubits": qubits,
        "pass": passed,
        "worst_norm_rmse": worst,
        "med_evals": med_evals,
        "med_qcalls": med_qcalls,
        "med_time": med_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--holdout-seed", type=int, default=2026)
    args = parser.parse_args()

    print("=== 1. ALL 7 PUBLIC SCENARIOS (5 ADB + 2 World Bank) ===")
    print(f"{'scenario':24s} {'mode':12s} {'q':>3s} {'pass':>5s} {'worstRMSE':>10s} {'totalCalls':>11s} {'qCalls':>9s} {'time(s)':>8s}")
    for name in PUBLIC_SCENARIOS:
        model, _ = load_model(ROOT / "scenarios" / f"{name}.json")
        for mode in ("tilted_iqae", "hybrid"):
            if model.config["track"] == "worldbank_elt" and mode == "hybrid":
                continue
            r = eval_case(name, model, args.repeats, mode)
            print(
                f"{r['label']:24s} {r['mode']:12s} {r['qubits']:3d} {str(r['pass']):>5s} "
                f"{r['worst_norm_rmse']:10.3f} {r['med_evals']:11,d} {r['med_qcalls']:9,d} {r['med_time']:8.2f}"
            )

    print(f"\n=== 2. ALL 6 UNSEEN HOLDOUT CASES (seed={args.holdout_seed}: 3 ADB + 3 World Bank) ===")
    print(f"{'case':24s} {'mode':12s} {'q':>3s} {'pass':>5s} {'worstRMSE':>10s} {'totalCalls':>11s} {'qCalls':>9s} {'time(s)':>8s}")
    with tempfile.TemporaryDirectory() as tmp:
        suite_dir = Path(tmp) / "suite"
        suite = generate_suite(suite_dir, ROOT / "data", seed=args.holdout_seed)
        for rel_cfg in suite["cases"]:
            case_id = Path(rel_cfg).stem
            model, _ = load_model(suite_dir / rel_cfg)
            r = eval_case(case_id, model, args.repeats, "tilted_iqae")
            print(
                f"{r['label']:24s} {r['mode']:12s} {r['qubits']:3d} {str(r['pass']):>5s} "
                f"{r['worst_norm_rmse']:10.3f} {r['med_evals']:11,d} {r['med_qcalls']:9,d} {r['med_time']:8.2f}"
            )

    print("\n=== 3. FACTORIZED LINEAR-DEPTH TILTED CIRCUIT vs MONOLITHIC (adb-base) ===")
    base_model, _ = load_model(ROOT / "scenarios/adb-base.json")
    f_grids, _ = payoffs(base_model)
    for idx, label in ((2, "P(L>T)"), (3, "E[(L-T)+]"), (4, "P(both exh.)")):
        mono, W_m, gammas = build_target_circuit(base_model, f_grids[idx], use_tilt=True)
        nodes = network(base_model, tilt=gammas)
        fact, W_f = factorized_a_operator(nodes, f_grids[idx], gammas)
        cx_m = transpile(mono, basis_gates=["cx", "u"], optimization_level=1).count_ops().get("cx", 0)
        cx_f = transpile(fact, basis_gates=["cx", "u"], optimization_level=1).count_ops().get("cx", 0)
        print(
            f"  {label:14s}: monolithic {cx_m:5,d} CX ({mono.num_qubits}q)  ->  "
            f"factorized {cx_f:4,d} CX ({fact.num_qubits}q)  [{cx_m / cx_f:.1f}x reduction]"
        )

    print("\n=== 4. EXACT CAUSAL MPS SCALING (chi=10, zero truncation error, n=2..50) ===")
    print(f"{'n':>4s} {'states':>10s} {'params':>8s} {'E[L] ($M)':>11s} {'P(L>T)':>10s} {'P(all exh)':>12s} {'time(ms)':>10s}")
    for n in (2, 5, 10, 20, 50):
        res = build_ncountry_portfolio(base_model, n).contract_all()
        print(
            f"{n:4d} {'10^' + str(n):>10s} {res.parameter_count:8,d} "
            f"{res.expected_principal_loss_usd / 1e6:11.2f} "
            f"{res.portfolio_exceedance_probability:10.5f} "
            f"{res.all_bonds_exhausted_probability:12.3e} "
            f"{res.elapsed_seconds * 1e3:10.2f}"
        )

    print("\n=== 5. AWS BRAKET QPU NOISE & ZNE COHERENCE (Factorized P(both exh.) circuit) ===")
    rows = evaluate_qpu_coherence(amplitude=0.61, cx_a=450, cx_q=1000, k_values=(0, 1, 2, 4))
    print(f"{'QPU Profile':24s} {'k':>3s} {'CX':>7s} {'raw_err':>10s} {'readout_err':>12s} {'zne_err':>10s}")
    for r in rows:
        print(
            f"{r['profile']:24s} {r['k']:3d} {r['cx_total']:7,d} "
            f"{r['raw_err']:10.5f} {r['readout_err']:12.5f} {r['zne_err']:10.5f}"
        )


if __name__ == "__main__":
    main()
