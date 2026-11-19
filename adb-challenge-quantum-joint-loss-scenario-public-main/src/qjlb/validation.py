"""Validate exported estimates against the exact classical reference."""

from __future__ import annotations
import json
import math
from pathlib import Path
from .model import ELT_METRICS

EXECUTIONS = {"classical_cpu", "quantum_simulator", "quantum_hardware", "hybrid"}


def read_json(path):
    def distinct_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(), object_pairs_hook=distinct_keys)


def finite_number(value, name, nonnegative=False):
    try:
        valid = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
        )
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite number")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def check_submission(submission, problem_id, metric_names=ELT_METRICS):
    if not isinstance(submission, dict):
        raise ValueError("results must be a JSON object")
    if submission.get("problem_id") != problem_id:
        raise ValueError(
            "problem_id does not match the selected configuration and data"
        )
    if (
        not isinstance(submission.get("method"), str)
        or not submission["method"].strip()
    ):
        raise ValueError("method must be a nonempty name")
    if (
        not isinstance(submission.get("execution"), str)
        or submission["execution"] not in EXECUTIONS
    ):
        raise ValueError(
            "execution must be classical_cpu, quantum_simulator, quantum_hardware or hybrid"
        )
    estimates = submission.get("estimates")
    if not isinstance(estimates, dict) or set(estimates) != set(metric_names):
        raise ValueError(
            "estimates must contain exactly the target names for this scenario"
        )
    for name, value in estimates.items():
        finite_number(value, name)
    intervals = submission.get("intervals", {})
    if not isinstance(intervals, dict) or set(intervals) - set(metric_names):
        raise ValueError("intervals must map known target names to [lower, upper]")
    for name, interval in intervals.items():
        if interval is None:
            continue
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"{name}: interval must be [lower, upper] or null")
        lo, hi = [finite_number(x, f"{name} interval") for x in interval]
        if lo > hi:
            raise ValueError(f"{name}: interval lower bound exceeds upper bound")
    resources = submission.get("resources")
    required = {"elapsed_seconds", "preparation_seconds", "shots", "qubits"}
    if not isinstance(resources, dict) or not required <= set(resources):
        raise ValueError(
            "resources must include elapsed_seconds, preparation_seconds, shots and qubits"
        )
    for name in required:
        value = resources[name]
        if value is None:
            continue
        finite_number(value, name, nonnegative=True)
        if name in ("shots", "qubits") and not isinstance(value, int):
            raise ValueError(f"{name} must be an integer or null")
    elapsed, preparation = (
        resources["elapsed_seconds"],
        resources["preparation_seconds"],
    )
    if elapsed is not None and preparation is not None and preparation > elapsed:
        raise ValueError("preparation_seconds cannot exceed elapsed_seconds")
    # Reject non-finite numbers anywhere, including optional device metadata.
    try:
        json.dumps(submission, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "results must contain only valid finite JSON values"
        ) from error


def compare(submission, reference, problem_id, rtol=None, metric_names=ELT_METRICS):
    check_submission(submission, problem_id, metric_names)
    if rtol is not None:
        finite_number(rtol, "rtol", nonnegative=True)
    rows = []
    for name in metric_names:
        target = reference["estimates"][name]["estimate"]
        value = submission["estimates"][name]
        error = value - target
        interval = submission.get("intervals", {}).get(name)
        rows.append(
            {
                "target": name,
                "estimate": value,
                "reference": target,
                "error": error,
                "absolute_error": abs(error),
                "relative_error": abs(error) / abs(target) if target != 0 else None,
                "ci95_contains_reference": None
                if interval is None
                else interval[0] <= target <= interval[1],
                "within_tolerance": None
                if rtol is None
                else abs(error) <= rtol * abs(target),
            }
        )
    return {
        "problem_id": problem_id,
        "method": submission["method"],
        "execution": submission["execution"],
        "relative_tolerance": rtol,
        "within_tolerance": None
        if rtol is None
        else all(r["within_tolerance"] for r in rows),
        "resources": submission["resources"],
        "resource_status": "self_reported",
        "comparisons": rows,
    }


def make_submission(model, result, method, elapsed_seconds, preparation_seconds):
    """Export a built-in classical result in the participant JSON format."""
    return {
        "problem_id": model.fingerprint,
        "method": method,
        "execution": "classical_cpu",
        "estimates": {
            name: item["estimate"] for name, item in result["estimates"].items()
        },
        "intervals": {
            name: [item["ci95_low"], item["ci95_high"]]
            if item["ci95_low"] is not None
            else None
            for name, item in result["estimates"].items()
        },
        "resources": {
            "elapsed_seconds": elapsed_seconds,
            "preparation_seconds": preparation_seconds,
            "shots": 0,
            "qubits": 0,
            "model_evaluations": result["resources"]["model_evaluations"],
        },
    }
