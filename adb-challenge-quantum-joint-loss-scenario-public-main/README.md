# ADB joint-loss baseline

Classical baselines and results validation for a quantum joint-loss challenge. The notebook runs seven scenarios: five based on a simplified model of ADB's 2026 Kyrgyz Republic and Tajikistan disaster relief bonds, and two World Bank cases with other country pairs and hypothetical contracts.

[View an executed example](examples/baseline_scenarios.ipynb) or [view a sample submission](examples/adb-base.json).

## Run

Use Python 3.10+ from the repository folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebook]"
jupyter lab baseline_scenarios.ipynb
```

On Windows, use `python` in the first command and activate with `.venv\Scripts\activate`.

Run [baseline_scenarios.ipynb](baseline_scenarios.ipynb) from top to bottom. It checks calibration, compares methods and validates example submissions. Results are saved in `results/scenarios/`.

The sampling methods are Monte Carlo, scrambled Sobol, stratified sampling and importance sampling. Exact references score numerical accuracy. Compare methods at matched computational cost, including preparation; resource measurements are self-reported.

## Evaluate team solutions

Use the [short evaluation protocol](EVALUATION.md) to generate private cases, run team code and compare measured costs. The notebook and `qjlb validate` provide practice checks.

## Submit your results

Keep the scenario configurations fixed. Adapt [example.py](example.py) and save one JSON file per scenario, such as `adb-base.json`. Set `SUBMISSION_DIR` in the notebook to your results folder.

Each file needs:

| Field | Value |
|---|---|
| `problem_id` | ID from that scenario's `problem.json` |
| `method` | Your method name |
| `execution` | `classical_cpu`, `quantum_simulator`, `quantum_hardware` or `hybrid` |
| `estimates` | All five target names in `problem.json`, each mapped to a finite number |
| `resources` | `elapsed_seconds`, `preparation_seconds`, `shots`, `qubits`; use `null` for unavailable values |

Elapsed time includes preparation. Shots and qubits must be nonnegative integers. Optional `intervals` maps targets to 95% intervals `[lower, upper]`. Include your code, software environment and device or simulator details.

`RTOL = None` reports errors. `RTOL = 0.01` requires all five estimates to be within 1%; a zero reference requires a zero estimate. The notebook lists missing or invalid files. Rare outcomes can have large sampling errors at the default budget.

For one scenario from the command line:

```sh
qjlb run --config scenarios/adb-base.json
python example.py --config scenarios/adb-base.json --output results/my_results.json
qjlb validate results/my_results.json --config scenarios/adb-base.json
```

CLI outputs go to `results/baseline/` and `results/validation/`. Validation exit codes: `0` valid comparison or tolerance pass, `1` tolerance failure, `2` invalid input.

## Read the results

Each scenario's `baseline/` folder contains `problem.json`, exact values in `reference.json`, method errors and timings in `summary.csv`, and individual runs in `runs.json`. The `validation/` folder contains `comparison.csv` and `report.json`.

ADB runs also export `annual-states.csv` for quantum encoding. Draw one state per year using its probability; apply each country's earthquake cap and one precipitation payment over the term.

For ADB targets, `L` is combined principal loss over three years and `T` is USD 80 million:

| Target | Quantity |
|---|---|
| `term_expected_principal_loss_usd` | Expected `L` |
| `term_joint_attachment_probability` | Probability both countries incur principal loss |
| `term_portfolio_loss_exceedance_probability` | Probability `L > T` |
| `term_portfolio_stop_loss_usd` | Expected `max(L - T, 0)` |
| `term_both_bonds_exhausted_probability` | Probability both bonds lose all principal |

## Model and sources

The ADB base case allocates USD 70m to earthquakes and USD 10m to precipitation per country. Earthquake payments accumulate to their limit; precipitation pays once. [Public inputs and offering documents](data/adb_terms.json).

Earthquakes use Poisson events and 25% payout steps fitted to published annual probabilities. Shared-event rates approximate public conditional trigger probabilities. Perils and years are independent; severity marks within shared earthquakes are independent.

Precipitation assumes a 2% annual trigger probability per country and a shared rate of 25% of the smaller country rate. Operational trigger reconstruction and coupon valuation are outside scope.

World Bank cases use annual rates and fixed conditional mean losses, with synthetic payments per row and no term cap. Their exceedance targets are rates per year. Event IDs match across countries within each peril. Absent national rows are zero-filled and losses reconcile within the [audit tolerance](data/audit.json). [Source manifest](data/source-manifest.json).

Code: [Apache 2.0](LICENSE). World Bank data: [CC BY 4.0](data/LICENSE.txt).
