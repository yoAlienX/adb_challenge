"""Comprehensive test suite for the dual-track IQAE + MPS quantum prototype."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROTOTYPE_ROOT.parent
OFFICIAL_ROOT = REPO_ROOT / "adb-challenge-quantum-joint-loss-scenario-public-main"

sys.path.insert(0, str(PROTOTYPE_ROOT / "src"))

from qjlb.bond import BondModel, bond_exact
from qjlb.evaluation import strict_submission, tolerances
from qjlb.model import ELTModel, elt_exact

from quantum.circuits import (
    HAS_QISKIT,
    build_compact_bond_circuit,
    verify_qiskit_statevector_amplitude,
)
from quantum.iqae import (
    StatevectorGroverEngine,
    solve_bond_iqae_compact,
    solve_bond_iqae_path,
    solve_elt_iqae,
)
from quantum.mps import (
    benchmark_generative_mps_worldbank,
    benchmark_mps_scaling,
    build_bond_mps_exact,
    mps_to_staircase_unitaries,
    solve_bond_mps,
)
from quantum.noise import BRAKET_NOISE_PROFILES, simulate_noisy_iqae_8q
from quantum.resources import estimate_all_resources


@pytest.fixture(scope="module")
def adb_base() -> BondModel:
    return BondModel.load(OFFICIAL_ROOT / "scenarios/adb-base.json")


@pytest.fixture(scope="module")
def kaz_kgz() -> ELTModel:
    return ELTModel.load(OFFICIAL_ROOT / "scenarios/kaz-kgz-base.json")


def test_statevector_grover_matches_exact_born_law(adb_base: BondModel):
    """Verify StatevectorGroverEngine produces exact sin^2((2k+1)*theta) without computing theta."""
    probs = adb_base.term_distribution()
    payoff = adb_base.observables(adb_base.states)[:, 2]  # exceedance probability (~1.205%)
    true_a = float(np.dot(probs, payoff))
    theta = math.asin(math.sqrt(true_a))

    engine = StatevectorGroverEngine(probs, payoff, device="cpu")
    for k in (0, 1, 2, 5, 13, 37, 100):
        engine.advance_to(k)
        p_sv = engine.born_probability_one()
        p_exact = math.sin((2 * k + 1) * theta) ** 2
        assert p_sv == pytest.approx(p_exact, abs=1e-12)


@pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
def test_qiskit_8q_circuit_matches_exact_amplitude(adb_base: BondModel):
    """Verify real Qiskit 8-qubit StatePreparation + UCRYGate circuit matches exact reference."""
    probs = adb_base.term_distribution()
    obs = adb_base.observables(adb_base.states)
    # Test both continuous (expected principal loss) and indicator (both bonds exhausted)
    for col in (0, 2, 4):
        payoff = obs[:, col]
        f_max = float(payoff.max())
        norm = payoff / f_max
        p_qiskit, p_exact = verify_qiskit_statevector_amplitude(probs, norm)
        assert p_qiskit == pytest.approx(p_exact, abs=1e-12)

    artifact = build_compact_bond_circuit(
        probs, obs[:, 2] / obs[:, 2].max(), transpile_circuit=True
    )
    assert artifact.num_qubits == 8
    assert len(artifact.circuit_sha256) == 64
    assert artifact.two_qubit_gates > 0
    assert "OPENQASM 2.0;" in artifact.qasm2_str


def test_iqae_compact_8q_accuracy_within_evaluation_tolerances(adb_base: BondModel):
    """Regime A1 8q IQAE meets evaluation-v1 RMSE tolerances across 5 seeds on adb-base."""
    ref = bond_exact(adb_base)
    bounds = tolerances(adb_base, ref)

    runs = [solve_bond_iqae_compact(adb_base, samples=8192, seed=100 + i) for i in range(5)]
    for name in adb_base.metric_names:
        target = ref["estimates"][name]["estimate"]
        ests = np.array([r["estimates"][name]["estimate"] for r in runs])
        rmse = float(np.sqrt(np.mean((ests - target) ** 2)))
        assert rmse <= bounds[name], f"{name}: rmse={rmse} > tol={bounds[name]}"

    assert runs[0]["resources"]["qubits"] == 8
    assert runs[0]["resources"]["shots"] > 0
    assert runs[0]["resources"]["model_evaluations"] > runs[0]["resources"]["shots"]
    assert len(runs[0]["evidence"]["circuit_sha256"]) == 64


def test_iqae_noncircular_22q_path_matches_exact(adb_base: BondModel):
    """Regime A2 non-circular 22-qubit 3-year path IQAE passes all 5 tolerances on adb-base."""
    ref = bond_exact(adb_base)
    bounds = tolerances(adb_base, ref)

    res = solve_bond_iqae_path(adb_base, samples=8192, seed=42)
    assert res["resources"]["qubits"] == 22
    assert res["resources"]["regime"] == "A2_noncircular_path_22q"

    for name in adb_base.metric_names:
        target = ref["estimates"][name]["estimate"]
        est = res["estimates"][name]["estimate"]
        # Single-run error within 1.5x RMSE tolerance
        assert abs(est - target) <= 1.5 * bounds[name], (
            f"{name}: est={est}, ref={target}, tol={bounds[name]}"
        )


def test_iqae_elt_20q_passes_worldbank_tolerances(kaz_kgz: ELTModel):
    """Regime A2 20-qubit uniform-rate World Bank ELT IQAE passes tolerances on kaz-kgz-base."""
    ref = elt_exact(kaz_kgz)
    bounds = tolerances(kaz_kgz, ref)

    res = solve_elt_iqae(kaz_kgz, samples=8192, seed=7)
    assert res["resources"]["qubits"] == 20
    assert res["resources"]["uniform_rates_per_peril"] is True

    for name in kaz_kgz.metric_names:
        target = ref["estimates"][name]["estimate"]
        est = res["estimates"][name]["estimate"]
        assert abs(est - target) <= 1.5 * bounds[name], (
            f"{name}: est={est}, ref={target}, tol={bounds[name]}"
        )


def test_mps_exact_and_truncation_monotonicity(adb_base: BondModel):
    """Track B MPS reproduces bond_exact at chi=10 to machine precision and decays monotonically."""
    ref = bond_exact(adb_base)
    res_full = solve_bond_mps(adb_base, max_bond_dim=10)
    for name in adb_base.metric_names:
        assert res_full["estimates"][name]["estimate"] == pytest.approx(
            ref["estimates"][name]["estimate"], rel=1e-12, abs=1e-15
        )

    errs = [build_bond_mps_exact(adb_base, max_bond_dim=chi).truncation_error_l2 for chi in (1, 2, 4, 8, 10)]
    for i in range(len(errs) - 1):
        assert errs[i] >= errs[i + 1] - 1e-15
    assert errs[-1] == pytest.approx(0.0, abs=1e-14)


def test_mps_ncountry_scaling_matches_dense_and_reaches_n20(adb_base: BondModel):
    """Exact automaton MPO matches 10^n brute-force for n=2,3,4 and contracts n=20 (10^20 states) fast."""
    rows = benchmark_mps_scaling(
        adb_base, country_counts=[2, 3, 4, 10, 20], bond_dims=[4, 10]
    )
    for row in rows:
        if row["dense_verify_max_err"] is not None:
            assert row["dense_verify_max_err"] < 1e-12
    n20_rows = [r for r in rows if r["n_countries"] == 20 and r["bond_dim"] == 10]
    assert len(n20_rows) == 1
    assert n20_rows[0]["contract_only_seconds"] < 1.0
    assert n20_rows[0]["expected_loss_usd"] > 0.0


def test_generative_mps_worldbank_5country():
    """Role 2 Generative MPS trains cleanly on real 5-country World Bank SFRARR data."""
    rows = benchmark_generative_mps_worldbank(
        str(OFFICIAL_ROOT / "data"), train_sizes=[500], bond_dims=[2], seed=12
    )
    assert len(rows) == 2
    gmps_row = [r for r in rows if r["model"] == "generative_mps_chi2"][0]
    assert gmps_row["parameters"] < 1024  # far fewer parameters than 4^5 = 1,024 full table
    assert math.isfinite(gmps_row["kl_divergence"])
    assert 0.0 <= gmps_row["est_ge2_attach"] <= 1.0


def test_mps_staircase_compilation(adb_base: BondModel):
    """Role 3 Ran (2020) MPS staircase compiles valid circuit with high fidelity."""
    from quantum.mps import _bond_term_matrix_10x10

    p_mat = _bond_term_matrix_10x10(adb_base)
    artifact, fidelity, comp = mps_to_staircase_unitaries(p_mat, max_bond_dim=2, label="test_staircase")
    assert fidelity > 0.99
    assert len(artifact.circuit_sha256) == 64
    assert comp["total_qubits"] == 8


def test_noisy_density_matrix_and_zne(adb_base: BondModel):
    """Exact 8q density-matrix noise + ZNE reduces raw error on IonQ Forte and keeps Libra exact."""
    forte = simulate_noisy_iqae_8q(
        adb_base, target_idx=2, profile=BRAKET_NOISE_PROFILES["ionq_forte"], k_values=[0, 1, 2]
    )
    libra = simulate_noisy_iqae_8q(
        adb_base, target_idx=2, profile=BRAKET_NOISE_PROFILES["quera_libra_2028"], k_values=[0, 1, 2, 8]
    )
    # ZNE should substantially improve k=0 relative error vs raw noisy on IonQ Forte
    assert forte.relative_error_k0_zne < forte.relative_error_k0_raw
    # Fault-tolerant QuEra Libra should maintain high coherence through k=8
    assert libra.max_useful_k_raw == 8


def test_resource_estimator_complete(adb_base: BondModel, kaz_kgz: ELTModel):
    report = estimate_all_resources(adb_base, kaz_kgz)
    assert len(report["track_a_adb_drb"]) == 2
    assert len(report["track_a_worldbank_elt"]) == 1
    assert len(report["track_b_mps_scaling"]) >= 5


@pytest.mark.parametrize(
    "solver_mode,execution",
    [
        ("iqae", "quantum_simulator"),
        ("iqae_compact", "quantum_simulator"),
        ("mps", "classical_cpu"),
        ("hybrid", "hybrid"),
    ],
)
def test_team_solver_cli_passes_strict_submission(
    tmp_path: Path, adb_base: BondModel, solver_mode: str, execution: str
):
    """Verify team_solver.py CLI passes qjlb.evaluation.strict_submission for all modes."""
    cfg_src = OFFICIAL_ROOT / "scenarios/adb-base.json"
    terms_src = OFFICIAL_ROOT / "data/adb_terms.json"
    (tmp_path / "cases").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "cases/case.json").write_text(cfg_src.read_text())
    (tmp_path / "data/adb_terms.json").write_text(terms_src.read_text())

    request = {
        "schema_version": 1,
        "run_id": f"test-{solver_mode}",
        "problem_id": adb_base.fingerprint,
        "track": "adb_drb",
        "config_file": "cases/case.json",
        "seed": 42,
        "samples": 8192,
        "targets": list(adb_base.metric_names),
    }
    req_path = tmp_path / "request.json"
    out_path = tmp_path / "result.json"
    req_path.write_text(json.dumps(request))

    cmd = [
        sys.executable,
        str(PROTOTYPE_ROOT / "team_solver.py"),
        "--solver-mode",
        solver_mode,
        "--device",
        "cpu",
        "--request",
        str(req_path),
        "--output",
        str(out_path),
    ]
    proc = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"stderr: {proc.stderr}"

    submission = json.loads(out_path.read_text())
    review = strict_submission(submission, request, adb_base, execution, wall=60.0)
    expected_review = (
        "code_review_required" if execution == "classical_cpu" else "evidence_review_required"
    )
    assert review == expected_review


def test_time_complexity_environment_inspection_and_fit():
    """Verify platform-aware environment inspection and power-law complexity fitting."""
    scripts_dir = str(PROTOTYPE_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from benchmark_time_complexity import fit_power_law, inspect_environment

    for req in ("auto", "cpu", "cuda"):
        env = inspect_environment(req)
        assert env["active_device"] in ("cpu", "cuda")
        assert env["cpu_logical_cores"] >= 1
        assert isinstance(env["hardware_label"], str) and env["hardware_label"]

    fit = fit_power_law([10, 100, 1000, 10000], [0.01, 0.1, 1.0, 10.0])
    assert fit["alpha"] == pytest.approx(1.0, abs=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-6)
    empty_fit = fit_power_law([10], [0.01])
    assert empty_fit["alpha"] is None
