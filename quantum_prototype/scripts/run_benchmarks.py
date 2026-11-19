"""Reproducible benchmark & figure generation suite for ADB Quantum Innovation Challenge.

Generates all empirical tables (CSV/JSON) and publication figures (PNG) inside
`quantum_prototype/results/`:
  1. All 7 public scenarios (5 ADB + 2 World Bank): Exact reference, Classical baselines
     (MC, QMC, Stratified, Importance), Track A IQAE (8q compact + 22q non-circular path +
     20q uniform-rate ELT), and Track B MPS.
  2. Equal-query scaling sweep (M in [256..32768]) demonstrating IQAE O(1/M) vs classical
     O(1/sqrt(M)) on rare joint-tail targets (adb-base).
  3. Track B n=2..20 multi-country MPS scaling (10^2 to 10^20 states), verified against
     brute-force 10^n enumeration for n=2..4.
  4. Track B Role 2: 5-country Generative MPS (Han et al., 2018) trained on limited
     subsamples of the REAL World Bank SFRARR Central Asia ELT data (CC BY 4.0).
  5. Track B Role 3: Compiled Ran (2020) MPS staircase circuit vs. Grover-Rudolph CX counts.
  6. Exact 8-qubit Density-Matrix noisy simulation + Readout Mitigation + ZNE calibrated to
     AWS Braket QPU error profiles (IQM Emerald, IonQ Forte, Early-FT, QuEra Libra 2028).
  7. Chakrabarti-style end-to-end classical & quantum resource breakdown (JSON).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROTOTYPE_ROOT.parent
OFFICIAL_ROOT = REPO_ROOT / "adb-challenge-quantum-joint-loss-scenario-public-main"

sys.path.insert(0, str(PROTOTYPE_ROOT / "src"))

from qjlb.bond import BOND_METHODS, BondModel, bond_exact
from qjlb.cli import load_model, methods_for
from qjlb.evaluation import tolerances
from qjlb.model import ELTModel, elt_exact

from quantum.iqae import (
    iqae_statevector_estimate,
    solve_bond_iqae_compact,
    solve_bond_iqae_path,
    solve_elt_iqae,
)
from quantum.mps import (
    _bond_term_matrix_10x10,
    benchmark_generative_mps_worldbank,
    benchmark_mps_scaling,
    build_bond_mps_exact,
    mps_to_staircase_unitaries,
    solve_bond_mps,
)
from quantum.noise import sweep_braket_qpu_noise
from quantum.resources import estimate_all_resources


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_all_scenarios_benchmark(
    results_dir: Path,
    repeats: int = 5,
    samples: int = 8192,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Benchmark Track A (IQAE) + Track B (MPS) vs Classical Baselines on all 7 public scenarios."""
    scenario_files = sorted((OFFICIAL_ROOT / "scenarios").glob("*.json"))
    rows: list[dict[str, Any]] = []

    for sc_path in scenario_files:
        sc_name = sc_path.stem
        model, _ = load_model(sc_path)
        classical = methods_for(model)
        ref = classical["exact"](model)
        bounds = tolerances(model, ref)
        print(f"[Scenarios] Benchmarking {sc_name} ({model.config['track']}) ...", flush=True)

        # Classical sampling methods
        eval_methods: list[tuple[str, Any]] = [
            ("mc", lambda m, s, sd: classical["mc"](m, s, sd)),
            ("qmc", lambda m, s, sd: classical["qmc"](m, s, sd)),
            ("stratified", lambda m, s, sd: classical["stratified"](m, s, sd)),
            ("importance", lambda m, s, sd: classical["importance"](m, s, sd)),
        ]
        if isinstance(model, BondModel):
            eval_methods.extend(
                [
                    ("iqae_compact_8q", lambda m, s, sd: solve_bond_iqae_compact(m, s, sd, device=device)),
                    ("iqae_path_22q", lambda m, s, sd: solve_bond_iqae_path(m, s, sd, device=device)),
                    ("mps_chi10", lambda m, s, sd: solve_bond_mps(m, s, sd, max_bond_dim=10)),
                    ("mps_chi4", lambda m, s, sd: solve_bond_mps(m, s, sd, max_bond_dim=4)),
                ]
            )
        else:
            eval_methods.append(
                ("iqae_elt_20q", lambda m, s, sd: solve_elt_iqae(m, s, sd, device=device))
            )

        for method_name, fn in eval_methods:
            # Deterministic MPS needs only 1 run; stochastic methods run `repeats` seeds
            n_rep = 1 if method_name.startswith("mps_") else (
                min(repeats, 3) if method_name in ("iqae_path_22q", "iqae_elt_20q") and device == "cpu" else repeats
            )
            run_outputs = []
            times = []
            for r in range(n_rep):
                t0 = time.perf_counter()
                out = fn(model, samples, 2026 + r * 37)
                times.append(time.perf_counter() - t0)
                run_outputs.append(out)

            med_sec = float(np.median(times))
            med_evals = float(np.median([o["resources"].get("model_evaluations", 0) for o in run_outputs]))
            med_shots = float(np.median([o["resources"].get("shots", 0) for o in run_outputs]))
            qubits = int(run_outputs[0]["resources"].get("qubits", 0))

            for target in model.metric_names:
                exact_val = ref["estimates"][target]["estimate"]
                vals = np.array([o["estimates"][target]["estimate"] for o in run_outputs])
                rmse = float(np.sqrt(np.mean((vals - exact_val) ** 2)))
                rel_rmse = rmse / abs(exact_val) if exact_val != 0 else 0.0
                tol = bounds[target]
                rows.append(
                    {
                        "scenario": sc_name,
                        "track": model.config["track"],
                        "method": method_name,
                        "target": target,
                        "repeats": n_rep,
                        "reference": exact_val,
                        "mean_estimate": float(vals.mean()),
                        "rmse": rmse,
                        "relative_rmse": rel_rmse,
                        "tolerance": tol,
                        "normalised_rmse": rmse / tol,
                        "pass_tolerance": rmse <= tol,
                        "qubits": qubits,
                        "median_shots": med_shots,
                        "median_model_evaluations": med_evals,
                        "median_seconds": med_sec,
                    }
                )

    write_csv(results_dir / "public_scenarios_benchmark.csv", rows)
    return rows


def run_equal_query_scaling_sweep(
    adb_base: BondModel,
    results_dir: Path,
    budgets: list[int],
    repeats: int = 8,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Matched query-budget sweep comparing IQAE O(1/M) vs Classical O(1/sqrt(M)) on adb-base."""
    print(f"[Query Sweep] Running equal-query sweep over budgets={budgets} ...", flush=True)
    ref = bond_exact(adb_base)
    bounds = tolerances(adb_base, ref)
    probs = adb_base.term_distribution()
    obs = adb_base.observables(adb_base.states)

    # Focus on the two hardest rare-tail targets: portfolio exceedance (1.205%) & both exhausted (0.0032%)
    target_indices = [2, 4]
    rows: list[dict[str, Any]] = []

    for budget in budgets:
        # Classical baselines at exact sample count `budget`
        for cls_name in ("mc", "qmc", "stratified", "importance"):
            fn = BOND_METHODS[cls_name]
            cls_runs = [fn(adb_base, budget, 1000 + r * 19 + budget) for r in range(repeats)]
            actual_q = float(np.mean([o["resources"]["model_evaluations"] for o in cls_runs]))
            for tidx in target_indices:
                tname = adb_base.metric_names[tidx]
                exact_val = ref["estimates"][tname]["estimate"]
                ests = np.array([o["estimates"][tname]["estimate"] for o in cls_runs])
                rmse = float(np.sqrt(np.mean((ests - exact_val) ** 2)))
                rows.append(
                    {
                        "budget_cap": budget,
                        "actual_queries": actual_q,
                        "method": cls_name,
                        "target": tname,
                        "reference": exact_val,
                        "tolerance": bounds[tname],
                        "rmse": rmse,
                        "relative_rmse": rmse / exact_val,
                        "repeats": repeats,
                    }
                )

        # IQAE strictly capped at `max_oracle_calls = budget` (apples-to-apples!)
        for tidx in target_indices:
            tname = adb_base.metric_names[tidx]
            exact_val = ref["estimates"][tname]["estimate"]
            payoff = obs[:, tidx]
            iqae_runs = [
                iqae_statevector_estimate(
                    probs,
                    payoff,
                    seed=5000 + r * 29 + budget + tidx * 7,
                    shots_per_round=12,
                    max_oracle_calls=budget,
                    max_k=4096,
                    qubits_override=8,
                    device=device,
                )
                for r in range(repeats)
            ]
            actual_q = float(np.mean([o.oracle_calls for o in iqae_runs]))
            ests = np.array([o.estimate for o in iqae_runs])
            rmse = float(np.sqrt(np.mean((ests - exact_val) ** 2)))
            rows.append(
                {
                    "budget_cap": budget,
                    "actual_queries": actual_q,
                    "method": "iqae",
                    "target": tname,
                    "reference": exact_val,
                    "tolerance": bounds[tname],
                    "rmse": rmse,
                    "relative_rmse": rmse / exact_val,
                    "repeats": repeats,
                }
            )

    write_csv(results_dir / "query_scaling_sweep.csv", rows)

    # Plot Figure 1: Log-log Relative RMSE vs Queries
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=160)
    styles = {
        "mc": ("#7f8c8d", "o", "--", "Monte Carlo"),
        "qmc": ("#3498db", "s", "--", "Scrambled Sobol (QMC)"),
        "stratified": ("#e67e22", "^", "--", "Stratified Sampling"),
        "importance": ("#9b59b6", "D", "-.", "Importance Sampling"),
        "iqae": ("#c0392b", "*", "-", "Track A: IQAE (Statevector Grover)"),
    }
    titles = {
        "term_portfolio_loss_exceedance_probability": "Portfolio Exceedance Pr(L > $80m) = 1.205%",
        "term_both_bonds_exhausted_probability": "Both Bonds Exhausted Pr(L1=L2=$80m) = 0.0032%",
    }

    for ax, tidx in zip(axes, target_indices):
        tname = adb_base.metric_names[tidx]
        sub = [r for r in rows if r["target"] == tname]
        for mkey, (col, mk, ls, label) in styles.items():
            mrows = sorted([r for r in sub if r["method"] == mkey], key=lambda x: x["actual_queries"])
            xs = [r["actual_queries"] for r in mrows]
            ys = [max(r["relative_rmse"], 1e-4) for r in mrows]
            lw = 2.6 if mkey == "iqae" else 1.6
            ms = 9 if mkey == "iqae" else 5.5
            ax.loglog(xs, ys, marker=mk, linestyle=ls, color=col, linewidth=lw, markersize=ms, label=label)

        # Reference slopes O(1/sqrt(M)) and O(1/M)
        x_ref = np.array([min(budgets), max(budgets)], dtype=float)
        mc_start = [r["relative_rmse"] for r in sub if r["method"] == "mc" and r["budget_cap"] == budgets[0]][0]
        iq_start = [r["relative_rmse"] for r in sub if r["method"] == "iqae" and r["budget_cap"] == budgets[1]][0]
        ax.loglog(
            x_ref,
            mc_start * (x_ref / budgets[0]) ** (-0.5),
            ":",
            color="#555555",
            alpha=0.65,
            label=r"Ref slope $O(M^{-1/2})$",
        )
        ax.loglog(
            x_ref,
            iq_start * (x_ref / budgets[1]) ** (-1.0),
            ":",
            color="#c0392b",
            alpha=0.55,
            label=r"Ref slope $O(M^{-1})$",
        )
        # Tolerance line
        tol_rel = sub[0]["tolerance"] / sub[0]["reference"]
        ax.axhline(tol_rel, color="#27ae60", linestyle=":", linewidth=1.6, label=f"Eval-v1 Tolerance ({tol_rel:.0%})")
        ax.set_title(titles[tname], fontsize=11, fontweight="bold")
        ax.set_xlabel("Queries (Classical Samples vs. IQAE Grover Oracle Calls)", fontsize=10)
        ax.set_ylabel("Relative RMSE (across repeats)", fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(
        "Track A (IQAE) vs. Classical Baselines: Matched Query-Budget Scaling on adb-base",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(results_dir / "fig1_iqae_vs_classical_scaling.png")
    plt.close(fig)
    return rows


def run_mps_experiments(adb_base: BondModel, results_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run all three Track B MPS roles (n=2..20 scaling, 5-country generative MPS, staircase compilation)."""
    print("[Track B] Running n=2..20 multi-country MPS scaling ...", flush=True)
    scaling_rows = benchmark_mps_scaling(
        adb_base,
        country_counts=[2, 3, 4, 5, 6, 8, 10, 15, 20],
        bond_dims=[2, 4, 8, 10],
    )
    write_csv(results_dir / "mps_ncountry_scaling.csv", scaling_rows)

    # Also record n=2 SVD truncation profile on exact adb-base
    ref = bond_exact(adb_base)
    trunc_rows = []
    for chi in (1, 2, 3, 4, 6, 8, 10):
        mps = build_bond_mps_exact(adb_base, max_bond_dim=chi)
        res = mps.contract_all_targets(adb_base.threshold)
        trunc_rows.append(
            {
                "bond_dim": chi,
                "parameters": res.parameter_count,
                "truncation_error_l2": res.truncation_error_l2,
                "rel_err_el": abs(res.expected_principal_loss_usd - ref["estimates"]["term_expected_principal_loss_usd"]["estimate"])
                / ref["estimates"]["term_expected_principal_loss_usd"]["estimate"],
                "rel_err_exc": abs(res.portfolio_loss_exceedance_probability - ref["estimates"]["term_portfolio_loss_exceedance_probability"]["estimate"])
                / ref["estimates"]["term_portfolio_loss_exceedance_probability"]["estimate"],
                "rel_err_exh": abs(res.all_bonds_exhausted_probability - ref["estimates"]["term_both_bonds_exhausted_probability"]["estimate"])
                / ref["estimates"]["term_both_bonds_exhausted_probability"]["estimate"],
            }
        )
    write_csv(results_dir / "mps_adb_base_truncation.csv", trunc_rows)

    # Plot Figure 2: n=2..20 MPS Scaling + chi Truncation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0), dpi=160)
    for chi, col, mk in [(2, "#2ecc71", "o"), (4, "#3498db", "s"), (8, "#9b59b6", "^"), (10, "#e74c3c", "D")]:
        sub = [r for r in scaling_rows if r["bond_dim"] == chi]
        ax1.semilogy(
            [r["n_countries"] for r in sub],
            [r["contract_only_seconds"] * 1000.0 for r in sub],
            marker=mk,
            color=col,
            linewidth=2.0,
            label=f"MPS contraction ($\\chi={chi}$)",
        )
    dense_sub = [r for r in scaling_rows if r["bond_dim"] == 10 and r["dense_exact_seconds"] is not None]
    ax1.semilogy(
        [r["n_countries"] for r in dense_sub],
        [r["dense_exact_seconds"] * 1000.0 for r in dense_sub],
        "k--x",
        linewidth=2.2,
        markersize=8,
        label=r"Exact $10^n$ enumeration (wall at $n \geq 6$)",
    )
    ax1.set_title("Multi-Country Runtime Scaling ($10^2 \\to 10^{20}$ States)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Number of Countries $n$ (State Space $= 10^n$)", fontsize=10)
    ax1.set_ylabel("Contraction Runtime (ms, log scale)", fontsize=10)
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=8.5)

    chis = [r["bond_dim"] for r in trunc_rows]
    ax2.semilogy(chis, [max(r["rel_err_el"], 1e-15) for r in trunc_rows], "o-", label="Expected Loss rel. err")
    ax2.semilogy(chis, [max(r["rel_err_exc"], 1e-15) for r in trunc_rows], "s-", label="Exceedance Pr(L > $80m) rel. err")
    ax2.semilogy(chis, [max(r["rel_err_exh"], 1e-15) for r in trunc_rows], "^-", label="Both Exhausted (0.0032%) rel. err")
    ax2.set_title("ADB Base SVD Compression Error vs. Bond Dimension $\\chi$", fontsize=11, fontweight="bold")
    ax2.set_xlabel("MPS Bond Dimension $\\chi$", fontsize=10)
    ax2.set_ylabel("Relative Error vs. Exact (log scale)", fontsize=10)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=8.5)

    fig.suptitle(
        "Track B (MPS Tensor Network): Breaking the $10^n$ Multi-Country State-Space Wall",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(results_dir / "fig2_mps_ncountry_scaling.png")
    plt.close(fig)

    # Role 2: Limited-Data 5-Country Generative MPS on REAL World Bank SFRARR ELT Data
    print("[Track B] Training 5-country Generative MPS on real World Bank SFRARR data ...", flush=True)
    gen_rows = benchmark_generative_mps_worldbank(
        str(OFFICIAL_ROOT / "data"), train_sizes=[500, 2000, 8000], bond_dims=[2, 4], seed=42
    )
    write_csv(results_dir / "generative_mps_worldbank_5c.csv", gen_rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=160)
    for model_key, label, col, mk in [
        ("empirical_mc", "Empirical MC (1,024 full-table states)", "#7f8c8d", "o"),
        ("generative_mps_chi2", "Generative MPS ($\\chi=2$, ~64 params)", "#2980b9", "s"),
        ("generative_mps_chi4", "Generative MPS ($\\chi=4$, ~224 params)", "#c0392b", "^"),
    ]:
        sub = sorted([r for r in gen_rows if r["model"] == model_key], key=lambda x: x["train_samples"])
        xs = [r["train_samples"] for r in sub]
        ax1.semilogx(xs, [r["kl_divergence"] for r in sub], marker=mk, color=col, linewidth=2.2, label=label)
        ax2.loglog(xs, [max(r["abs_err_severe_tail"], 1e-5) for r in sub], marker=mk, color=col, linewidth=2.2, label=label)

    ax1.set_title("5-Country World Bank Distribution KL Divergence", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Historical Training Sample Size $N_{\\mathrm{train}}$", fontsize=10)
    ax1.set_ylabel("KL Divergence vs. Full 421,968-Event Catalogue", fontsize=10)
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=8.5)

    ax2.set_title("Severe Multi-Country Tail Error vs. Sample Size", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Historical Training Sample Size $N_{\\mathrm{train}}$", fontsize=10)
    ax2.set_ylabel("Absolute Error on Severe Joint Tail", fontsize=10)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=8.5)

    fig.suptitle(
        "Track B Role 2: 5-Country Generative MPS (Han et al., 2018) on Real World Bank SFRARR Data (KAZ, KGZ, TJK, TKM, UZB)",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(results_dir / "fig3_generative_mps_worldbank.png")
    plt.close(fig)

    # Role 3: Compiled MPS-to-Staircase Circuit Comparison
    print("[Track B] Compiling Ran (2020) MPS staircase circuit bridge ...", flush=True)
    p_mat = _bond_term_matrix_10x10(adb_base)
    staircase_rows = []
    for chi in (2, 4):
        _, fid, comp = mps_to_staircase_unitaries(p_mat, max_bond_dim=chi, label=f"adb_mps_stair_chi{chi}")
        staircase_rows.append(comp)
    write_csv(results_dir / "mps_staircase_compilation.csv", staircase_rows)

    return scaling_rows, gen_rows, staircase_rows


def run_noisy_braket_experiments(adb_base: BondModel, results_dir: Path) -> list[dict[str, Any]]:
    """Run exact 8q density-matrix noisy IQAE + ZNE across AWS Braket QPU profiles."""
    print("[Noise + ZNE] Running exact 8-qubit density-matrix Braket QPU noise sweep ...", flush=True)
    k_vals = [0, 1, 2, 4, 8, 16, 32]
    noise_rows = sweep_braket_qpu_noise(adb_base, target_idx=2, k_values=k_vals)

    # Export clean flat CSV summary
    summary_rows = [
        {k: v for k, v in r.items() if not isinstance(v, list)}
        for r in noise_rows
    ]
    write_csv(results_dir / "noisy_braket_zne_summary.csv", summary_rows)

    # Plot Figure 4: Noisy Born Probability vs Ideal across Grover Powers k
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), dpi=160)
    for ax, r in zip(axes.ravel(), noise_rows):
        ks = r["k_values"]
        ax.plot(ks, r["ideal_probs"], "k-o", linewidth=2.2, label="Ideal Statevector")
        ax.plot(ks, r["raw_noisy_probs"], "r--s", linewidth=1.8, alpha=0.85, label="Raw Noisy QPU")
        ax.plot(ks, r["readout_mitigated_probs"], "b-.^", linewidth=1.6, alpha=0.85, label="Readout Mitigated")
        ax.plot(ks, r["zne_mitigated_probs"], "g-D", linewidth=2.0, label="Readout + Exp-ZNE")
        ax.set_title(f"{r['device']} ($p_{{2q}}={r['p_2q']:.1e}$)", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Grover Power $k$", fontsize=9.5)
        ax.set_ylabel("Born Probability $P(\\mathrm{obj}=1 \\mid k)$", fontsize=9.5)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Exact 8-Qubit Density-Matrix Simulation: AWS Braket NISQ Noise Ceiling vs. QuEra Libra (2028) FTQC",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(results_dir / "fig4_noisy_braket_zne.png")
    plt.close(fig)
    return noise_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROTOTYPE_ROOT / "results",
    )
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    adb_base = BondModel.load(OFFICIAL_ROOT / "scenarios/adb-base.json")
    kaz_kgz = ELTModel.load(OFFICIAL_ROOT / "scenarios/kaz-kgz-base.json")

    run_all_scenarios_benchmark(
        args.results_dir, repeats=args.repeats, samples=args.samples, device=args.device
    )
    run_equal_query_scaling_sweep(
        adb_base,
        args.results_dir,
        budgets=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        repeats=max(args.repeats, 6),
        device=args.device,
    )
    run_mps_experiments(adb_base, args.results_dir)
    run_noisy_braket_experiments(adb_base, args.results_dir)

    print("[Resources] Computing Chakrabarti-style classical & quantum resource breakdown ...", flush=True)
    res_report = estimate_all_resources(adb_base, kaz_kgz)
    (args.results_dir / "resource_estimation.json").write_text(
        json.dumps(res_report, indent=2) + "\n"
    )

    elapsed = time.perf_counter() - t_start
    print(f"Done! All benchmarks, CSVs, JSONs, and 4 PNG figures saved to {args.results_dir} in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
