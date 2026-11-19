import copy
import sys
from pathlib import Path

import pytest

from qjlb.bond import BondModel, bond_exact
from qjlb.cli import load_model
from qjlb.evaluation import invoke, score, strict_submission
from qjlb.holdout import generate, load_suite
from qjlb.validation import make_submission

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    path = tmp_path_factory.mktemp("evaluation") / "suite"
    generate(path, ROOT / "data", count=6, seed=17)
    return path


def test_generated_suite_covers_workload_sizes(suite):
    manifest = load_suite(suite)
    models = [load_model(suite / name)[0] for name in manifest["cases"]]
    assert len({m.fingerprint for m in models}) == 6
    assert [m.config["term_years"] for m in models[::2]] == [1, 3, 10]
    assert [len(m.countries) for m in models[1::2]] == [2, 3, 5]


def test_tampering_is_rejected(tmp_path):
    generate(tmp_path / "suite", ROOT / "data", count=2, seed=17)
    case = tmp_path / "suite/cases/case-001.json"
    case.write_text(case.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        load_suite(tmp_path / "suite")


@pytest.fixture
def entry():
    model = BondModel.load(ROOT / "scenarios/adb-base.json")
    submission = make_submission(model, bond_exact(model), "exact", 0.5, 0.1)
    submission.update(run_id="test-run", seed=3)
    submission["resources"]["model_evaluations"] = 100
    return model, submission, {"run_id": "test-run", "seed": 3}


@pytest.mark.parametrize(
    "field,value", [("run_id", "old-run"), ("seed", True), ("seed", 4)]
)
def test_replay_and_wrong_seed_rejected(entry, field, value):
    model, submission, request = entry
    submission[field] = value
    with pytest.raises(ValueError):
        strict_submission(submission, request, model, "classical_cpu", 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("shots", None),
        ("qubits", -1),
        ("model_evaluations", 1.5),
        ("elapsed_seconds", 5),
    ],
)
def test_invalid_costs_rejected(entry, field, value):
    model, submission, request = entry
    submission["resources"][field] = value
    with pytest.raises(ValueError):
        strict_submission(submission, request, model, "classical_cpu", 1)


def test_hardware_metadata_requires_evidence_and_review(entry):
    model, submission, request = entry
    submission["execution"] = "quantum_hardware"
    submission["resources"].update(shots=100, qubits=4)
    with pytest.raises(ValueError, match="evidence"):
        strict_submission(submission, request, model, "quantum_hardware", 1)
    submission["evidence"] = dict(
        provider="test", backend="test", circuit_sha256="a" * 64, job_ids=["job-1"]
    )
    assert (
        strict_submission(submission, request, model, "quantum_hardware", 1)
        == "evidence_review_required"
    )


def test_score_uses_rmse_and_keeps_failures():
    runs = [
        dict(status="valid", measured_seconds=t, submission={"estimates": {"x": x}})
        for t, x in [(1, 12), (3, 8)]
    ]
    result = score(runs, {"estimates": {"x": {"estimate": 10}}}, {"x": 1}, ["x"])
    assert result["metrics"][0]["rmse"] == 2
    assert result["metrics"][0]["bias"] == 0
    assert not result["accuracy_pass"]
    assert result["median_seconds"] == 2
    failed = copy.deepcopy(runs)
    failed[0]["status"] = "timeout"
    assert not score(failed, {}, {}, [])["accuracy_pass"]


@pytest.mark.parametrize(
    "command,status",
    [
        (["-m", "qjlb.worker", "--method", "exact"], "valid"),
        (["-c", "raise SystemExit(7)"], "crash"),
        (["-c", "import time; time.sleep(5)"], "timeout"),
        (["-c", "pass"], "invalid"),
    ],
)
def test_process_results(suite, command, status):
    config = suite / "cases/case-001.json"
    model, _ = load_model(config)
    run = invoke(
        [sys.executable, *command],
        suite,
        config,
        model,
        3,
        64,
        0.2 if status == "timeout" else 15,
        "classical_cpu",
    )
    assert run["status"] == status
