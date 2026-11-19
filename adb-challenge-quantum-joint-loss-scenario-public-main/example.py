"""Export a results file. Replace the Monte Carlo call with your method."""

import argparse
import time
from qjlb.cli import load_model, methods_for, write_json
from qjlb.validation import make_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default="results/my_results.json")
    args = parser.parse_args()

    start = time.perf_counter()
    model, _ = load_model(args.config)
    preparation_seconds = time.perf_counter() - start

    # Replace this call with your algorithm. The supplied example is classical.
    # Draw uniforms with shape (n, model.dimension), then call model.from_uniforms.
    # Average model.evaluate(states)['metrics'] to estimate model.metric_names.
    # ADB states are annual payout outcomes; ELT states are annual-rate-weighted rows.
    result = methods_for(model)["mc"](model, budget=65536, seed=42)
    elapsed_seconds = time.perf_counter() - start
    submission = make_submission(
        model, result, "example_mc", elapsed_seconds, preparation_seconds
    )

    # For quantum/hybrid work, set execution, timing, shots and qubits to the
    # values from your run. Add device and circuit details as needed.
    write_json(args.output, submission)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
