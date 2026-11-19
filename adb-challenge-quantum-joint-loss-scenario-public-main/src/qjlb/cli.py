"""Run the classical baseline or validate a participant's results file."""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from . import __version__
from .model import (
    ELTModel,
    elt_exact,
    elt_mc,
    elt_rqmc,
    elt_stratified,
    elt_is,
)
from .bond import BondModel, BOND_METHODS
from .validation import read_json, check_submission, compare, make_submission

METHODS = {
    "mc": elt_mc,
    "qmc": elt_rqmc,
    "stratified": elt_stratified,
    "importance": elt_is,
    "exact": elt_exact,
}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def environment():
    return {
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }


def load_model(config):
    start = time.perf_counter()
    track = read_json(config).get("track")
    model = BondModel.load(config) if track == "adb_drb" else ELTModel.load(config)
    return model, time.perf_counter() - start


def methods_for(model):
    return BOND_METHODS if isinstance(model, BondModel) else METHODS


def run(args):
    if args.samples < 16 or args.repeats < 1 or args.seed < 0:
        raise ValueError(
            "samples must be at least 16; repeats positive; seed nonnegative"
        )
    model, preparation = load_model(args.config)
    methods = methods_for(model)
    start = time.perf_counter()
    reference = methods["exact"](model)
    reference_seconds = time.perf_counter() - start
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "problem.json",
        {
            "problem_id": model.fingerprint,
            "definition_version": getattr(
                model, "definition_version", "elt-mean-loss-v1"
            ),
            "config": model.config,
            "data_sha256": model.data_hashes,
            "targets": list(model.metric_names),
            "uniform_dimension": model.dimension,
            "state_count": model.state_count,
            "environment": environment(),
            "model_preparation_seconds": preparation,
            "timing_scope": "Estimator time includes method-specific preparation; one-off time also includes common data loading, verification and CDF construction. Interpreter startup and output writing are excluded.",
        },
    )
    if isinstance(model, BondModel):
        write_csv(
            output / "annual-states.csv",
            [
                {
                    "state_id": i,
                    "probability": float(probability),
                    "kgz_earthquake_fraction": float(state[0] / 4),
                    "tjk_earthquake_fraction": float(state[1] / 4),
                    "kgz_precipitation_triggered": int(state[2]),
                    "tjk_precipitation_triggered": int(state[3]),
                }
                for i, (state, probability) in enumerate(
                    zip(model.states, model.probabilities)
                )
            ],
        )
    write_json(
        output / "reference.json",
        {
            "problem_id": model.fingerprint,
            "calculation_seconds": reference_seconds,
            **reference,
        },
    )
    runs = []
    for method, function in methods.items():
        for repeat in range(1 if method == "exact" else args.repeats):
            if method == "exact":
                result, seconds = reference, reference_seconds
            else:
                start = time.perf_counter()
                result = function(model, args.samples, args.seed + repeat)
                seconds = time.perf_counter() - start
            submission = make_submission(
                model, result, method, preparation + seconds, preparation
            )
            comparison = compare(
                submission,
                reference,
                model.fingerprint,
                metric_names=model.metric_names,
            )
            runs.append(
                {
                    "method": method,
                    "seed": args.seed + repeat,
                    "samples": 0
                    if method == "exact"
                    else result["resources"]["samples"],
                    "estimator_seconds": seconds,
                    "one_off_seconds": preparation + seconds,
                    "result": result,
                    "comparison": comparison,
                }
            )
    summary = []
    for method in methods:
        group = [r for r in runs if r["method"] == method]
        for i, name in enumerate(model.metric_names):
            errors = [r["comparison"]["comparisons"][i]["error"] for r in group]
            coverage = [
                r["comparison"]["comparisons"][i]["ci95_contains_reference"]
                for r in group
            ]
            coverage = [x for x in coverage if x is not None]
            target = reference["estimates"][name]["estimate"]
            rmse = float(np.sqrt(np.mean(np.square(errors))))
            summary.append(
                {
                    "method": method,
                    "target": name,
                    "runs": len(group),
                    "samples_per_run": group[0]["samples"],
                    "reference": target,
                    "rmse": rmse,
                    "relative_rmse": rmse / abs(target) if target else None,
                    "bias": float(np.mean(errors)),
                    "median_estimator_seconds": float(
                        np.median([r["estimator_seconds"] for r in group])
                    ),
                    "median_one_off_seconds": float(
                        np.median([r["one_off_seconds"] for r in group])
                    ),
                    "interval_fraction": len(coverage) / len(group),
                    "ci95_coverage": float(np.mean(coverage)) if coverage else None,
                }
            )
    write_json(output / "runs.json", runs)
    write_csv(output / "summary.csv", summary)
    print(f"Baseline saved to {output}. Problem ID: {model.fingerprint}")
    return 0


def validate(args):
    model, preparation = load_model(args.config)
    submission = read_json(args.results)
    check_submission(submission, model.fingerprint, model.metric_names)
    start = time.perf_counter()
    reference = methods_for(model)["exact"](model)
    reference_seconds = time.perf_counter() - start
    report = compare(
        submission, reference, model.fingerprint, args.rtol, model.metric_names
    )
    report.update(
        environment=environment(),
        model_preparation_seconds=preparation,
        reference_seconds=reference_seconds,
        submission_sha256=hashlib.sha256(Path(args.results).read_bytes()).hexdigest(),
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", report)
    write_csv(output / "comparison.csv", report["comparisons"])
    if args.rtol is None:
        status = "Compared with exact reference; no tolerance was specified."
    elif report["within_tolerance"]:
        status = "All five estimates meet the specified tolerance."
    else:
        status = "One or more estimates exceed the specified tolerance."
    print(f"{status} Results: {output}")
    return 1 if report["within_tolerance"] is False else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser(
        "run", help="run classical methods and save exact reference values"
    )
    baseline.add_argument(
        "--config",
        default="config.json",
        help="model configuration (default: config.json)",
    )
    baseline.add_argument(
        "--samples",
        type=int,
        default=8192,
        help="sample budget per method (default: 8192)",
    )
    baseline.add_argument(
        "--repeats", type=int, default=10, help="sampling runs per method (default: 10)"
    )
    baseline.add_argument(
        "--seed", type=int, default=1000, help="first random seed (default: 1000)"
    )
    baseline.add_argument(
        "--output",
        default="results/baseline",
        help="output folder (existing files are replaced)",
    )
    baseline.set_defaults(function=run)
    validation = commands.add_parser(
        "validate", help="compare a results JSON file with the exact reference"
    )
    validation.add_argument("results", help="participant results JSON file")
    validation.add_argument(
        "--config",
        default="config.json",
        help="configuration used for the submitted estimates",
    )
    validation.add_argument(
        "--output",
        default="results/validation",
        help="output folder (existing files are replaced)",
    )
    validation.add_argument(
        "--rtol", type=float, help="optional relative error limit, e.g. 0.01 for 1%%"
    )
    validation.set_defaults(function=validate)
    holdout = commands.add_parser("holdout", help="generate private evaluation cases")
    holdout.add_argument("--output", default="private-evaluation/suite")
    holdout.add_argument("--data", default="data")
    holdout.add_argument("--cases", type=int, default=6)
    holdout.add_argument(
        "--seed", type=int, help="private reproducibility seed; random if omitted"
    )

    def generate_holdout(args):
        from .holdout import generate

        generate(args.output, args.data, args.cases, args.seed)
        print(
            f"Private suite saved to {args.output}. Keep the suite and seed out of Git."
        )
        return 0

    holdout.set_defaults(function=generate_holdout)
    evaluation = commands.add_parser(
        "evaluate", help="run reviewed team code and classical baselines"
    )
    evaluation.add_argument("suite")
    evaluation.add_argument("--output", default="private-evaluation/report")
    evaluation.add_argument("--repeats", type=int, default=10)
    evaluation.add_argument("--samples", type=int, default=8192)
    evaluation.add_argument("--timeout", type=float, default=60)
    evaluation.add_argument(
        "--execution",
        choices=["classical_cpu", "quantum_simulator", "quantum_hardware", "hybrid"],
        required=True,
    )
    evaluation.add_argument(
        "--trusted-code",
        action="store_true",
        help="confirm code review; the runner is not a security sandbox",
    )
    evaluation.add_argument(
        "--pass-env",
        action="append",
        default=[],
        help="explicitly allow a provider credential variable",
    )
    evaluation.add_argument(
        "command", nargs="+", help="team command after -- (use absolute script paths)"
    )

    def evaluate_team(args):
        from .evaluation import evaluate

        return evaluate(args)

    evaluation.set_defaults(function=evaluate_team)
    args = parser.parse_args()
    try:
        return args.function(args)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
