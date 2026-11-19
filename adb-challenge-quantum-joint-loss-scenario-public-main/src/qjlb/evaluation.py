"""Run reviewed team code and classical methods on evaluator-owned inputs."""

import math
import os
from pathlib import Path
import random
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import numpy as np
from .cli import load_model, methods_for, environment, write_json, write_csv
from .holdout import digest, load_suite
from .validation import check_submission, finite_number, read_json

PROTOCOL = "evaluation-v1"
BASELINES = ("exact", "mc", "qmc", "stratified", "importance")


def tolerances(model, reference):
    if model.config["track"] == "adb_drb":
        principal = float(model.principal.sum())
        relative = [0.01, 0.05, 0.05, 0.05, 0.10]
        absolute = [principal * 1e-4, 1e-4, 1e-4, principal * 1e-5, 1e-6]
    else:
        relative = [0.01, 0.01, 0.05, 0.05, 0.05]
        absolute = [1000, 1000, 1e-5, 1e-5, 100]
    return {
        name: max(a, r * abs(reference["estimates"][name]["estimate"]))
        for name, r, a in zip(model.metric_names, relative, absolute)
    }


def strict_submission(submission, request, model, execution, wall):
    check_submission(submission, model.fingerprint, model.metric_names)
    if submission.get("run_id") != request["run_id"]:
        raise ValueError("run_id does not match this invocation")
    if type(submission.get("seed")) is not int or submission["seed"] != request["seed"]:
        raise ValueError("seed does not match this invocation")
    if submission["execution"] != execution:
        raise ValueError("execution differs from the declared entry")
    resources = submission["resources"]
    for name in (
        "elapsed_seconds",
        "preparation_seconds",
        "shots",
        "qubits",
        "model_evaluations",
    ):
        finite_number(resources.get(name), name, nonnegative=True)
        if (
            name in ("shots", "qubits", "model_evaluations")
            and type(resources[name]) is not int
        ):
            raise ValueError(f"{name} must be an integer")
    if resources["elapsed_seconds"] > wall * 1.1 + 0.05:
        raise ValueError("reported elapsed time exceeds measured process time")
    values = np.array([submission["estimates"][n] for n in model.metric_names])
    if model.config["track"] == "adb_drb":
        total = float(model.principal.sum())
        upper = np.array([total, 1, 1, max(total - model.threshold, 0), 1])
    else:
        total = float(model.limit.sum()) * model.total_rate
        upper = np.array(
            [
                float(model.loss.sum(axis=1).max()) * model.total_rate,
                total,
                model.total_rate,
                model.total_rate,
                max(float(model.limit.sum()) - model.threshold, 0) * model.total_rate,
            ]
        )
    if np.any(values < 0) or np.any(values > upper + np.maximum(1, upper) * 1e-12):
        raise ValueError("estimate is outside the physical range of its target")
    if execution == "classical_cpu":
        if resources["shots"] != 0 or resources["qubits"] != 0:
            raise ValueError("classical_cpu requires zero shots and qubits")
        return "code_review_required"
    if resources["qubits"] < 1:
        raise ValueError("quantum runs require positive qubits")
    evidence = submission.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("quantum execution evidence is required")
    for key in ("provider", "backend"):
        if not isinstance(evidence.get(key), str) or not evidence[key].strip():
            raise ValueError(f"evidence.{key} is required")
    if not isinstance(evidence.get("circuit_sha256"), str) or not re.fullmatch(
        "[0-9a-f]{64}", evidence["circuit_sha256"]
    ):
        raise ValueError("evidence.circuit_sha256 must be a SHA-256 digest")
    kind = execution
    if execution == "hybrid":
        kind = evidence.get("quantum_execution")
        if kind not in ("quantum_hardware", "quantum_simulator"):
            raise ValueError("hybrid evidence must name quantum_execution")
    jobs = evidence.get("job_ids", [])
    if (
        not isinstance(jobs, list)
        or any(not isinstance(j, str) or not j.strip() for j in jobs)
        or len(jobs) != len(set(jobs))
    ):
        raise ValueError("evidence.job_ids must contain distinct nonempty strings")
    if kind == "quantum_hardware" and (not jobs or resources["shots"] < 1):
        raise ValueError(
            "hardware evidence requires provider job_ids and positive shots"
        )
    return "evidence_review_required"


def invoke(
    command, suite, config, model, seed, samples, timeout, execution, pass_env=()
):
    """Fresh working files and a deadline. This function is not a security sandbox."""
    with tempfile.TemporaryDirectory(prefix="qjlb-run-") as directory:
        work = Path(directory)
        (work / "cases").mkdir()
        (work / "data").mkdir()
        shutil.copyfile(config, work / "cases/case.json")
        names = (
            ["adb_terms.json"]
            if model.config["track"] == "adb_drb"
            else ["audit.json", "seismic.npz", "flood.npz"]
        )
        for name in names:
            shutil.copyfile(suite / "data" / name, work / "data" / name)
        request = {
            "schema_version": 1,
            "run_id": secrets.token_hex(16),
            "problem_id": model.fingerprint,
            "track": model.config["track"],
            "config_file": "cases/case.json",
            "seed": seed,
            "samples": samples,
            "targets": list(model.metric_names),
        }
        write_json(work / "request.json", request)
        env = {
            k: os.environ[k] for k in ("PATH", "LANG", "SYSTEMROOT") if k in os.environ
        }
        env.update(
            HOME=str(work),
            TMPDIR=str(work),
            PYTHONNOUSERSITE="1",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        for name in pass_env:
            if name not in os.environ:
                raise ValueError(f"allowed environment variable is unset: {name}")
            env[name] = os.environ[name]
        argv = [*command, "--request", "request.json", "--output", "result.json"]
        start = time.perf_counter()
        try:
            process = subprocess.Popen(
                argv,
                cwd=work,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as error:
            return {
                "seed": seed,
                "run_id": request["run_id"],
                "measured_seconds": time.perf_counter() - start,
                "status": "crash",
                "detail": str(error),
            }
        status, detail = "valid", ""
        try:
            process.wait(timeout=timeout)
            if process.returncode:
                status, detail = "crash", f"exit code {process.returncode}"
        except subprocess.TimeoutExpired:
            status, detail = "timeout", "process exceeded the per-run deadline"
        finally:
            # Also reap descendants left behind by a wrapper that has exited.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        wall = time.perf_counter() - start
        run = {
            "seed": seed,
            "run_id": request["run_id"],
            "measured_seconds": wall,
            "status": status,
            "detail": detail,
        }
        if status != "valid":
            return run
        try:
            path = work / "result.json"
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size > 1_000_000
            ):
                raise ValueError(
                    "result.json is missing, a symlink or larger than 1 MB"
                )
            submission = read_json(path)
            review = strict_submission(submission, request, model, execution, wall)
            run.update(
                submission=submission, verification=review, result_sha256=digest(path)
            )
        except (ValueError, KeyError, TypeError, OSError) as error:
            run.update(status="invalid", detail=str(error))
        return run


def score(runs, reference, bounds, names):
    valid = [run for run in runs if run["status"] == "valid"]
    if not runs or len(valid) != len(runs):
        return {"accuracy_pass": False, "metrics": [], "median_seconds": None}
    rows = []
    for name in names:
        errors = (
            np.array([r["submission"]["estimates"][name] for r in valid])
            - reference["estimates"][name]["estimate"]
        )
        rmse = float(np.sqrt(np.mean(errors * errors)))
        rows.append(
            {
                "target": name,
                "reference": reference["estimates"][name]["estimate"],
                "rmse": rmse,
                "bias": float(errors.mean()),
                "tolerance": bounds[name],
                "normalised_rmse": rmse / bounds[name],
            }
        )
    return {
        "accuracy_pass": all(row["normalised_rmse"] <= 1 for row in rows),
        "metrics": rows,
        "median_seconds": float(np.median([r["measured_seconds"] for r in valid])),
    }


def command_hashes(command):
    paths = []
    for i, arg in enumerate(command):
        path = Path(shutil.which(arg) or arg) if i == 0 else Path(arg)
        if path.is_file():
            paths.append(path.resolve())
    return {str(p): digest(p) for p in paths}


def evaluate(args):
    if os.name != "posix":
        raise ValueError("the process evaluator requires Linux or macOS")
    if not args.trusted_code:
        raise ValueError(
            "review the command first, then use --trusted-code; this runner is not a sandbox"
        )
    if args.repeats < 2 or args.samples < 16:
        raise ValueError("use at least two repeats and 16 baseline samples")
    finite_number(args.timeout, "timeout", nonnegative=True)
    if args.timeout == 0:
        raise ValueError("timeout must be positive")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("provide a team command after --")
    command = list(command)
    command[0] = os.path.abspath(shutil.which(command[0]) or command[0])
    suite = Path(args.suite).resolve()
    manifest = load_suite(suite)
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("choose a new output folder; reports are never overwritten")
    output.mkdir(parents=True)
    original_hash = digest(suite / "suite.json")
    hashes = command_hashes(command)
    rng = random.Random(manifest["seed"])
    report = {
        "protocol": PROTOCOL,
        "ranking_status": "provisional_pending_execution_review",
        "suite_sha256": original_hash,
        "environment": environment(),
        "command": command,
        "command_file_sha256": hashes,
        "declared_execution": args.execution,
        "repeats": args.repeats,
        "baseline_samples": args.samples,
        "timeout_seconds": args.timeout,
        "official_repeat_count_met": args.repeats >= 10,
        "verification": "code_review_required"
        if args.execution == "classical_cpu"
        else "evidence_review_required",
        "timing_scope": "Fresh process: startup, loading, preparation, execution and result writing. Input staging and evaluator scoring excluded for every method.",
        "cases": [],
    }
    rows = []
    seen_jobs = set()
    for name in manifest["cases"]:
        config = suite / name
        model, _ = load_model(config)
        reference = methods_for(model)["exact"](model)
        bounds = tolerances(model, reference)
        seeds = [rng.randrange(2**32) for _ in range(args.repeats)]
        case = {
            "case": Path(name).stem,
            "track": model.config["track"],
            "problem_id": model.fingerprint,
            "countries": list(model.countries),
            "input_states": model.state_count,
            "path_dimensions": model.dimension,
            "tolerances": bounds,
            "methods": {},
        }
        for method in (*BASELINES, "team"):
            argv = (
                command
                if method == "team"
                else [sys.executable, "-m", "qjlb.worker", "--method", method]
            )
            execution = args.execution if method == "team" else "classical_cpu"
            runs = [
                invoke(
                    argv,
                    suite,
                    config,
                    model,
                    seed,
                    args.samples,
                    args.timeout,
                    execution,
                    args.pass_env if method == "team" else (),
                )
                for seed in seeds
            ]
            if method == "team":
                for run in runs:
                    if run["status"] != "valid":
                        continue
                    if run["submission"]["execution"] == "classical_cpu":
                        continue
                    evidence = run["submission"].get("evidence", {})
                    jobs = {
                        (evidence.get("provider"), evidence.get("backend"), job)
                        for job in evidence.get("job_ids", [])
                    }
                    if jobs & seen_jobs:
                        run.update(
                            status="invalid",
                            detail="provider job ID reused across invocations",
                        )
                    seen_jobs.update(jobs)
            result = score(runs, reference, bounds, model.metric_names)
            case["methods"][method] = {**result, "runs": runs}
            rows.append(
                {
                    "case": case["case"],
                    "track": case["track"],
                    "method": method,
                    "accuracy_pass": result["accuracy_pass"],
                    "median_seconds": result["median_seconds"],
                    "invalid_runs": sum(r["status"] != "valid" for r in runs),
                    "worst_normalised_rmse": max(
                        (m["normalised_rmse"] for m in result["metrics"]), default=None
                    ),
                }
            )
        passing = [
            case["methods"][method]["median_seconds"]
            for method in BASELINES
            if case["methods"][method]["accuracy_pass"]
        ]
        team = case["methods"]["team"]
        case["team_time_ratio"] = (
            team["median_seconds"] / min(passing)
            if passing and team["accuracy_pass"]
            else None
        )
        report["cases"].append(case)
        print(
            f"{case['case']}: team accuracy {'pass' if team['accuracy_pass'] else 'fail'}",
            flush=True,
        )
    if (
        digest(suite / "suite.json") != original_hash
        or load_suite(suite) != manifest
        or command_hashes(command) != hashes
    ):
        raise ValueError(
            "suite or command files changed during evaluation; results rejected"
        )
    configurations = [load_model(suite / name)[0].config for name in manifest["cases"]]
    horizons = {c["term_years"] for c in configurations if c["track"] == "adb_drb"}
    country_counts = {
        len(c["countries"]) for c in configurations if c["track"] == "worldbank_elt"
    }
    report["workload_coverage_met"] = {1, 3, 10} <= horizons and {
        2,
        3,
        5,
    } <= country_counts
    report["protocol_eligible"] = (
        report["official_repeat_count_met"] and report["workload_coverage_met"]
    )
    report["tracks"] = {}
    for track in sorted({case["track"] for case in report["cases"]}):
        cases = [case for case in report["cases"] if case["track"] == track]
        ratios = [case["team_time_ratio"] for case in cases]
        report["tracks"][track] = {
            "all_cases_pass": all(
                case["methods"]["team"]["accuracy_pass"] for case in cases
            ),
            "geometric_time_ratio": math.exp(
                sum(math.log(r) for r in ratios) / len(ratios)
            )
            if report["protocol_eligible"] and all(r is not None for r in ratios)
            else None,
        }
    write_json(output / "report.json", report)
    write_csv(output / "summary.csv", rows)
    print(f"Results: {output}. Execution claims require independent review.")
    return 0 if all(t["all_cases_pass"] for t in report["tracks"].values()) else 1
