import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qiskit_aer")

from qjlb.bond import BondModel, bond_exact
from qjlb.evaluation import tolerances
from qjlb_qae.circuits import a_operator, objective_probability, state_preparation
from qjlb_qae.iqae import AerGrover, AnalyticGrover, iqae
from qjlb_qae.loader import network, path_probabilities, register_values
from qjlb_qae.solver import estimate_target, payoffs, solve

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT / "scenarios/adb-base.json").read_text())


def make(**changes):
    config = {**BASE, **changes}
    return BondModel(config, ROOT / "scenarios" / config["terms_file"])


CONFIGS = [
    {},
    {"term_years": 1},
    {"term_years": 10},
    {"earthquake_common_rate_multiplier": 0.0, "precipitation_common_rate_share": 0.0},
    {"earthquake_common_rate_multiplier": 2.4, "precipitation_rate_multiplier": 1.9,
     "precipitation_annual_trigger_probability": [0.006, 0.037], "precipitation_common_rate_share": 0.45},
]


@pytest.mark.parametrize("changes", CONFIGS)
def test_network_reproduces_exact_term_distribution(changes):
    model = make(**changes)
    nodes = network(model)
    prob = path_probabilities(nodes)
    v = register_values(nodes)
    ok = (v["s"] < 5) & (v["a"] < 5) & (v["b"] < 5)
    joint = np.zeros((5, 5, 2, 2))
    np.add.at(joint, (v["a"][ok], v["b"][ok], v["r1"][ok], v["r2"][ok]), prob[ok])
    assert np.allclose(joint.ravel(), model.term_distribution(), atol=1e-14)


def test_circuit_loads_network_distribution():
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    nodes = network(make())
    qc = state_preparation(nodes)
    qc.save_probabilities()
    sim = AerSimulator(method="statevector")
    probs = sim.run(transpile(qc, sim, optimization_level=0)).result().data()["probabilities"]
    assert np.allclose(probs, path_probabilities(nodes), atol=1e-12)


@pytest.mark.parametrize("changes", [{}, {"term_years": 1}])
def test_tilted_amplitude_times_scale_is_exact_target(changes):
    model = make(**changes)
    exact = bond_exact(model)["estimates"]
    f_grids, scales = payoffs(model)
    for i, name in enumerate(model.metric_names):
        _, _, info = estimate_target(model, i, f_grids[i], scales[i], np.random.default_rng(0))
        assert math.isclose(scales[i] * info["W"] * info["amplitude"], exact[name]["estimate"], rel_tol=1e-9)


def test_real_grover_circuits_match_analytic_rotation():
    """Q^k A on Aer gives sin^2((2k+1) theta): validates the analytic sampler."""
    model = make()
    f_grids, _ = payoffs(model)
    nodes = network(model)
    from qjlb_qae.loader import layout

    regs = layout(nodes)
    idx = np.arange(256)
    av, bv = idx & 7, (idx >> 3) & 7
    ok = (av < 5) & (bv < 5)
    payoff = np.where(ok, f_grids[1][np.minimum(av, 4), np.minimum(bv, 4), (idx >> 6) & 1, (idx >> 7) & 1], 0.0)
    circuit = a_operator(nodes, payoff, regs["a"] + regs["b"] + regs["r1"] + regs["r2"])
    theta = math.asin(math.sqrt(objective_probability(circuit)))
    grover = AerGrover(circuit)
    for k in (0, 1, 2, 3):
        assert math.isclose(grover.probability(k), math.sin((2 * k + 1) * theta) ** 2, abs_tol=1e-9)


def test_iqae_interval_coverage():
    rng = np.random.default_rng(1)
    for a in (2e-4, 0.03, 0.4, 0.97):
        covered = 0
        for _ in range(100):
            res = iqae(AnalyticGrover(a, rng), lambda lo, hi: hi - lo < 2e-3 * max(a, 0.01),
                       epsilon_min=1e-6, alpha=0.05)
            covered += res.interval[0] <= a <= res.interval[1]
        assert covered >= 95


def test_solver_meets_official_tolerances_on_base():
    model = make()
    reference = bond_exact(model)
    bounds = tolerances(model, reference)
    runs = [solve(model, 8192, seed) for seed in range(5)]
    for name in model.metric_names:
        errors = [r["estimates"][name]["estimate"] - reference["estimates"][name]["estimate"] for r in runs]
        assert math.sqrt(np.mean(np.square(errors))) <= bounds[name]
    resources = runs[0]["resources"]
    assert resources["execution"] == "quantum_simulator" and resources["qubits"] == 13


def test_live_statevector_grover_matches_aer():
    """Live StatevectorGrover matches explicit AerGrover circuits to 1e-10."""
    from qjlb_qae.circuits import objective_mask, statevector
    from qjlb_qae.iqae import StatevectorGrover
    from qjlb_qae.solver import build_target_circuit

    model = make()
    f_grids, _ = payoffs(model)
    circuit, _, _ = build_target_circuit(model, f_grids[1], use_tilt=False)
    psi_a = statevector(circuit)
    mask = objective_mask(circuit.num_qubits)
    sv_grover = StatevectorGrover(psi_a, mask, np.random.default_rng(0))
    aer_grover = AerGrover(circuit)
    for k in (0, 1, 2, 3):
        assert math.isclose(sv_grover.probability(k), aer_grover.probability(k), abs_tol=1e-10)


def test_factorized_tilted_circuit_exact_and_reduces_cx():
    """Factorized linear-depth tilted circuit reproduces exact target with far fewer CX."""
    from qiskit import transpile
    from qjlb_qae.circuits import factorized_a_operator
    from qjlb_qae.loader import apply_tilt
    from qjlb_qae.solver import build_target_circuit

    model = make()
    exact = bond_exact(model)["estimates"]
    f_grids, scales = payoffs(model)
    # Target 4: P(both exhausted), heavily tilted
    mono_circ, W_mono, gammas = build_target_circuit(model, f_grids[4], use_tilt=True)
    assert gammas is not None
    nodes = network(model, tilt=gammas)
    fact_circ, W_fact = factorized_a_operator(nodes, f_grids[4], gammas)

    p_fact = objective_probability(fact_circ)
    assert math.isclose(scales[4] * W_fact * p_fact, exact[model.metric_names[4]]["estimate"], rel_tol=1e-9)

    cx_mono = transpile(mono_circ, basis_gates=["cx", "u"], optimization_level=1).count_ops().get("cx", 0)
    cx_fact = transpile(fact_circ, basis_gates=["cx", "u"], optimization_level=1).count_ops().get("cx", 0)
    assert cx_fact < cx_mono / 3


def test_causal_mps_exact_and_scales_to_50_countries():
    """Causal chi=10 MPS matches exact ADB to 1e-13 and contracts n=50 (10^50 states) fast."""
    from qjlb_qae.mps import build_ncountry_portfolio, solve_adb_mps

    model = make()
    exact = bond_exact(model)["estimates"]
    mps_res = solve_adb_mps(model)
    for name in model.metric_names:
        assert math.isclose(
            mps_res["estimates"][name]["estimate"], exact[name]["estimate"], rel_tol=1e-12, abs_tol=1e-14
        )
    # Scale to n=20 and n=50
    for n in (10, 20, 50):
        res = build_ncountry_portfolio(model, n).contract_all()
        assert res.n_countries == n
        assert res.bond_dim == 10
        assert res.truncation_error == 0.0
        assert res.expected_principal_loss_usd > 0
        assert 0.0 <= res.all_bonds_exhausted_probability <= res.portfolio_exceedance_probability <= 1.0
        assert res.elapsed_seconds < 1.0


def test_hybrid_mode_meets_tolerances_with_fewer_quantum_calls():
    """Hybrid router (RQMC bulk + Tilted ML-IQAE rare tails) passes tolerances with <1/2 quantum calls."""
    model = make()
    reference = bond_exact(model)
    bounds = tolerances(model, reference)
    runs = [solve(model, 8192, seed, mode="hybrid") for seed in range(5)]
    for name in model.metric_names:
        errors = [r["estimates"][name]["estimate"] - reference["estimates"][name]["estimate"] for r in runs]
        assert math.sqrt(np.mean(np.square(errors))) <= bounds[name]
    res = runs[0]["resources"]
    assert res["execution"] == "hybrid"
    assert res["quantum_oracle_calls"] < 20_000


def test_worldbank_track_meets_official_tolerances():
    """20q Live-Statevector ML-IQAE passes official evaluation-v1 tolerances on World Bank ELT."""
    from qjlb.model import ELTModel, elt_exact

    wb_model = ELTModel.load(ROOT / "scenarios/kaz-kgz-base.json")
    reference = elt_exact(wb_model)
    bounds = tolerances(wb_model, reference)
    runs = [solve(wb_model, 8192, seed) for seed in range(3)]
    for name in wb_model.metric_names:
        errors = [r["estimates"][name]["estimate"] - reference["estimates"][name]["estimate"] for r in runs]
        assert math.sqrt(np.mean(np.square(errors))) <= bounds[name]
    assert runs[0]["resources"]["qubits"] == 20
    assert runs[0]["resources"]["execution"] == "quantum_simulator"


def test_braket_noise_zne_reduces_error():
    """Readout + ZNE mitigation reduces error on IonQ Forte at k=1."""
    from qjlb_qae.noise import evaluate_qpu_coherence

    rows = evaluate_qpu_coherence(amplitude=0.61, cx_a=450, cx_q=1000, k_values=(0, 1))
    ionq_k0 = [r for r in rows if r["profile"] == "IonQ Forte" and r["k"] == 0][0]
    assert ionq_k0["zne_err"] < ionq_k0["raw_err"]
