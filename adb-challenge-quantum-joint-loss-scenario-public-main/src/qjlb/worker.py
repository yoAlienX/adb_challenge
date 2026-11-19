"""JSON command interface for built-in methods and team solvers."""

import argparse
import time
from .cli import load_model, methods_for, write_json
from .validation import read_json, make_submission


def run_solver(solver=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--method",
        choices=["exact", "mc", "qmc", "stratified", "importance"],
        default="importance",
    )
    args = parser.parse_args()
    start = time.perf_counter()
    request = read_json(args.request)
    model, preparation = load_model(request["config_file"])
    if model.fingerprint != request["problem_id"]:
        raise ValueError("input problem ID does not match")
    function = solver or methods_for(model)[args.method]
    result = function(model, request["samples"], request["seed"])
    elapsed = time.perf_counter() - start
    resources = result["resources"]
    submission = make_submission(
        model,
        result,
        resources["method"],
        elapsed,
        preparation + resources.get("preparation_seconds", 0),
    )
    submission["execution"] = resources.get("execution", "classical_cpu")
    for name in ("shots", "qubits", "model_evaluations"):
        submission["resources"][name] = resources.get(name, 0)
    submission.update(run_id=request["run_id"], seed=request["seed"])
    if "evidence" in result:
        submission["evidence"] = result["evidence"]
    write_json(args.output, submission)


if __name__ == "__main__":
    run_solver()
