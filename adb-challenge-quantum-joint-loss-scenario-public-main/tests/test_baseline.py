import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from qjlb.model import (
    ELTModel,
    ELT_METRICS,
    elt_exact,
    elt_mc,
    elt_rqmc,
    elt_stratified,
    elt_is,
)
from qjlb.validation import compare, check_submission, make_submission, read_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def model(tmp_path):
    audit = {"perils": {}}
    for peril, rate, loss in [
        ("seismic", [0.2, 0.3, 0.4], [[10, 0], [0, 20], [20, 20]]),
        ("flood", [0.5, 0.5], [[0, 0], [10, 10]]),
    ]:
        path = tmp_path / f"{peril}.npz"
        np.savez_compressed(
            path,
            event_id=np.arange(1, len(rate) + 1),
            annual_rate=rate,
            loss_usd=np.array(loss),
            sd_usd=np.zeros_like(loss),
            countries=["KGZ", "TJK"],
        )
        audit["perils"][peril] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        }
    (tmp_path / "audit.json").write_text(json.dumps(audit))
    config = {
        "schema_version": 2,
        "track": "worldbank_elt",
        "dataset_directory": ".",
        "countries": ["KGZ", "TJK"],
        "attachment_usd": [[5, 5], [5, 5]],
        "exhaustion_usd": [[15, 15], [15, 15]],
        "limit_usd": [10, 10],
        "portfolio_threshold_usd": 10,
    }
    return ELTModel(config, tmp_path)


@pytest.fixture
def submission(model):
    return make_submission(model, elt_exact(model), "test_exact", 1.0, 0.1)


def test_exact_annual_weights_and_strict_threshold(model):
    reference = elt_exact(model)
    assert [
        reference["estimates"][name]["estimate"] for name in ELT_METRICS
    ] == pytest.approx([34, 17, 0.9, 0.4, 4])
    assert model.total_rate == pytest.approx(1.9)
    rows = np.arange(model.state_count)[:, None]
    assert model.probabilities @ model.evaluate(rows)["metrics"] == pytest.approx(
        [34, 17, 0.9, 0.4, 4]
    )


@pytest.mark.parametrize("method", [elt_mc, elt_rqmc, elt_stratified, elt_is])
def test_samplers_against_hand_calculation(model, method):
    result = method(model, 32768, 72)
    assert [
        result["estimates"][name]["estimate"] for name in ELT_METRICS
    ] == pytest.approx([34, 17, 0.9, 0.4, 4], rel=0.025)
    assert result == method(model, 32768, 72)


def test_validation_threshold_and_intervals(model, submission):
    reference = elt_exact(model)
    assert compare(submission, reference, model.fingerprint)["within_tolerance"] is None
    assert (
        compare(submission, reference, model.fingerprint, 0)["within_tolerance"] is True
    )
    submission["estimates"][ELT_METRICS[0]] *= 1.1
    submission["intervals"][ELT_METRICS[0]] = [40.0, 50.0]
    report = compare(submission, reference, model.fingerprint, 0.01)
    assert report["within_tolerance"] is False
    assert report["comparisons"][0]["relative_error"] == pytest.approx(0.1)
    assert report["comparisons"][0]["ci95_contains_reference"] is False


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.update(problem_id="wrong"),
        lambda s: s.update(execution=[]),
        lambda s: s["estimates"].pop(ELT_METRICS[0]),
        lambda s: s["estimates"].update({ELT_METRICS[0]: float("nan")}),
        lambda s: s["estimates"].update({ELT_METRICS[0]: True}),
        lambda s: s["resources"].update(shots=1.5),
        lambda s: s["resources"].update(preparation_seconds=2.0),
        lambda s: s["intervals"].update({ELT_METRICS[0]: [3.0, 2.0]}),
    ],
)
def test_rejects_invalid_submission(model, submission, change):
    change(submission)
    with pytest.raises(ValueError):
        check_submission(submission, model.fingerprint)


def test_nullable_costs_and_missing_intervals(model, submission):
    submission["execution"] = "quantum_simulator"  # Test metadata; no quantum run.
    submission["resources"] = dict.fromkeys(
        ["elapsed_seconds", "preparation_seconds", "shots", "qubits"]
    )
    submission.pop("intervals")
    report = compare(submission, elt_exact(model), model.fingerprint, 0.01)
    assert report["within_tolerance"]
    assert report["resource_status"] == "self_reported"


def test_zero_reference_is_not_divided_by_zero(model, submission):
    reference = elt_exact(model)
    reference["estimates"][ELT_METRICS[0]]["estimate"] = 0.0
    submission["estimates"][ELT_METRICS[0]] = 1.0
    report = compare(submission, reference, model.fingerprint, 0.01)
    assert report["comparisons"][0]["relative_error"] is None
    assert report["within_tolerance"] is False


def test_duplicate_json_keys_are_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"method":"a","method":"b"}')
    with pytest.raises(ValueError, match="duplicate"):
        read_json(path)


def test_dataset_tampering_fails(model):
    (model.dataset_directory / "seismic.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        ELTModel(model.config, model.dataset_directory)


def test_public_reference():
    model = ELTModel.load(ROOT / "config.json")
    reference = elt_exact(model)
    assert model.state_count == 421968
    assert reference["estimates"]["annual_economic_loss_usd"][
        "estimate"
    ] == pytest.approx(701433183.6722711)


def test_cli_roundtrip_and_exit_codes(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))

    def invoke(*args):
        return subprocess.run(
            [sys.executable, "-m", "qjlb.cli", *args],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

    output = tmp_path / "baseline"
    result = invoke("run", "--samples", "16", "--repeats", "1", "--output", str(output))
    assert result.returncode == 0, result.stderr
    reference = json.loads((output / "reference.json").read_text())
    model = ELTModel.load(ROOT / "config.json")
    payload = make_submission(model, reference, "test_export", 1.0, 0.1)
    path = tmp_path / "submission.json"
    path.write_text(json.dumps(payload))
    report = tmp_path / "validation"
    assert (
        invoke("validate", str(path), "--rtol", "0", "--output", str(report)).returncode
        == 0
    )
    assert json.loads((report / "report.json").read_text())["within_tolerance"] is True
    payload["estimates"][ELT_METRICS[0]] *= 2
    path.write_text(json.dumps(payload))
    assert (
        invoke(
            "validate", str(path), "--rtol", ".01", "--output", str(report)
        ).returncode
        == 1
    )
    assert invoke("validate", str(path), "--output", str(report)).returncode == 0
    assert json.loads((report / "report.json").read_text())["within_tolerance"] is None
    payload["execution"] = []
    path.write_text(json.dumps(payload))
    invalid = invoke("validate", str(path), "--output", str(report))
    assert invalid.returncode == 2
    assert "Traceback" not in invalid.stderr
