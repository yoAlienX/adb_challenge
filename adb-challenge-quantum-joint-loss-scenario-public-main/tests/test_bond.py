import json
import csv
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from qjlb.bond import BondModel, BOND_METHODS, bond_exact, earthquake_year
from qjlb.cli import load_model, methods_for
from qjlb.validation import check_submission, compare, make_submission

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bond():
    return BondModel.load(ROOT / "scenarios/adb-base.json")


def test_closed_form_shared_full_loss_events():
    probabilities = earthquake_year(
        np.array([0, 0, 0, 0.1]), np.array([0, 0, 0, 0.2]), 0.04
    )
    neither = np.exp(-0.26)
    assert probabilities[0, 0] == pytest.approx(neither)
    assert probabilities[4, 0] == pytest.approx(np.exp(-0.2) - neither)
    assert probabilities[0, 4] == pytest.approx(np.exp(-0.1) - neither)
    assert probabilities[4, 4] == pytest.approx(
        1 - np.exp(-0.1) - np.exp(-0.2) + neither
    )


def test_public_annual_calibration(bond):
    terms = json.loads((ROOT / "data/adb_terms.json").read_text())
    for axis, country in ((1, "KGZ"), (0, "TJK")):
        marginal = bond.earthquake_probabilities.sum(axis=axis)
        assert [marginal[i:].sum() for i in range(1, 5)] == pytest.approx(
            terms["annual_earthquake_exceedance_probabilities"][country], abs=1e-12
        )
        # Public percentages are rounded; the model groups payouts into quarters.
        assert marginal @ (np.arange(5) / 4) == pytest.approx(
            terms["published_annual_earthquake_expected_loss_fraction"][country],
            abs=1.1e-5,
        )
    ratios = bond.earthquake_common_rate / bond.earthquake_rates.sum(axis=1)
    assert ratios == pytest.approx([0.075, 0.074], abs=0.0001)


def test_multi_event_caps_and_one_time_rain_payment(bond):
    def index(state):
        return np.ravel_multi_index(state, (5, 5, 2, 2))

    paths = np.array(
        [
            [index((3, 1, 1, 0))] * 3,
            [index((0, 0, 1, 1))] * 3,
            [index((4, 4, 1, 1))] * 3,
        ]
    )
    output = bond.evaluate(paths)
    np.testing.assert_allclose(
        output["principal_loss_usd"], [[80e6, 52.5e6], [10e6, 10e6], [80e6, 80e6]]
    )
    np.testing.assert_allclose(output["metrics"][0], [132.5e6, 1, 1, 52.5e6, 0])
    np.testing.assert_allclose(output["metrics"][2], [160e6, 1, 1, 80e6, 1])


def test_exact_dynamic_program_against_two_year_path_enumeration(bond):
    model = BondModel({**bond.config, "term_years": 2}, ROOT / "data/adb_terms.json")
    paths = np.array(list(np.ndindex(100, 100)))
    weights = np.prod(model.probabilities[paths], axis=1)
    enumerated = weights @ model.evaluate(paths)["metrics"]
    exact = bond_exact(model)
    assert [
        exact["estimates"][name]["estimate"] for name in model.metric_names
    ] == pytest.approx(enumerated)
    assert model.term_distribution().sum() == pytest.approx(1)


def test_independence_and_common_shock_stress_keep_marginals(bond):
    independent = BondModel.load(ROOT / "scenarios/adb-independent.json")
    stressed = BondModel.load(ROOT / "scenarios/adb-common-shock.json")
    for model in (independent, stressed):
        for axis in (0, 1):
            np.testing.assert_allclose(
                model.earthquake_probabilities.sum(axis=axis),
                bond.earthquake_probabilities.sum(axis=axis),
                atol=1e-14,
            )
    np.testing.assert_allclose(
        independent.earthquake_probabilities,
        np.outer(
            independent.earthquake_probabilities.sum(axis=1),
            independent.earthquake_probabilities.sum(axis=0),
        ),
        atol=1e-14,
    )
    joint = "term_joint_attachment_probability"
    assert (
        bond_exact(independent)["estimates"][joint]["estimate"]
        < bond_exact(bond)["estimates"][joint]["estimate"]
    )
    assert (
        bond_exact(bond)["estimates"][joint]["estimate"]
        < bond_exact(stressed)["estimates"][joint]["estimate"]
    )


def test_rain_frequency_stress_and_contract_allocation(bond):
    stressed = BondModel.load(ROOT / "scenarios/adb-rain-stress.json")
    assert stressed.precipitation_probabilities.sum(axis=1)[1] == pytest.approx(
        1 - 0.98**2
    )
    changed = BondModel.load(ROOT / "scenarios/adb-rain-cover.json")
    np.testing.assert_array_equal(changed.principal, bond.principal)
    np.testing.assert_allclose(changed.probabilities, bond.probabilities)
    np.testing.assert_array_equal(changed.eq_limit, [60e6, 60e6])
    np.testing.assert_array_equal(changed.rain_limit, [20e6, 20e6])


@pytest.mark.parametrize("method", ["mc", "qmc", "stratified", "importance"])
def test_samplers_against_exact(bond, method):
    budget = 131072
    output = BOND_METHODS[method](bond, budget, 1234)
    reference = bond_exact(bond)
    bounds = [160e6, 1, 1, 80e6, 1]
    for name, bound in zip(bond.metric_names, bounds):
        estimate = output["estimates"][name]
        error = abs(estimate["estimate"] - reference["estimates"][name]["estimate"])
        assert error <= 6 * estimate["standard_error"] + 6 * bound / budget
    assert output == BOND_METHODS[method](bond, budget, 1234)


def test_scenario_results_cannot_be_interchanged():
    paths = sorted((ROOT / "scenarios").glob("*.json"))
    assert len(paths) == 7
    models = [load_model(path)[0] for path in paths]
    assert len({model.fingerprint for model in models}) == 7
    for model in models:
        reference = methods_for(model)["exact"](model)
        submission = make_submission(model, reference, "exact_check", 1, 0.1)
        assert compare(submission, reference, model.fingerprint, 0, model.metric_names)[
            "within_tolerance"
        ]
        other = next(item for item in models if item.fingerprint != model.fingerprint)
        with pytest.raises(ValueError, match="problem_id"):
            check_submission(submission, other.fingerprint, other.metric_names)


@pytest.mark.parametrize(
    "change",
    [
        {"term_years": True},
        {"term_years": 0},
        {"term_years": 3.5},
        {"earthquake_common_rate_multiplier": 100},
        {"precipitation_common_rate_share": 1.1},
        {"precipitation_annual_trigger_probability": [0.02, 1]},
        {"earthquake_sublimit_usd": [0, 70e6]},
    ],
)
def test_invalid_bond_configuration(bond, change):
    with pytest.raises(ValueError):
        BondModel({**bond.config, **change}, ROOT / "data/adb_terms.json")


def test_cli_exports_annual_states_and_validates_bond_results(tmp_path, bond):
    config = ROOT / "scenarios/adb-base.json"
    output = tmp_path / "baseline"

    def invoke(*args):
        return subprocess.run(
            [sys.executable, "-m", "qjlb.cli", *map(str, args)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    run = invoke(
        "run", "--config", config, "--samples", 16, "--repeats", 1, "--output", output
    )
    assert run.returncode == 0, run.stderr
    problem = json.loads((output / "problem.json").read_text())
    assert problem["uniform_dimension"] == 3
    assert problem["state_count"] == 100
    assert problem["targets"] == list(bond.metric_names)
    with (output / "annual-states.csv").open() as stream:
        states = list(csv.DictReader(stream))
    np.testing.assert_allclose(
        [float(row["probability"]) for row in states], bond.probabilities
    )
    assert [int(row["state_id"]) for row in states] == list(range(100))
    reference = json.loads((output / "reference.json").read_text())
    path = tmp_path / "submission.json"
    path.write_text(json.dumps(make_submission(bond, reference, "exact_test", 1, 0.1)))
    checked = invoke(
        "validate",
        path,
        "--config",
        config,
        "--rtol",
        0,
        "--output",
        tmp_path / "validation",
    )
    assert checked.returncode == 0, checked.stderr
