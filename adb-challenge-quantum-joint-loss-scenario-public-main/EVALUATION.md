# Evaluation protocol

This challenge tests numerical estimation under supplied loss models. The public notebooks are practice cases. Score ADB term principal losses and World Bank annual quantities separately. Keep simulator, hardware and classical entries labelled separately.

## 1. Freeze and generate

Freeze each team's code, dependency lock file and execution type before generating cases. Record the code or container digest. Use the same private suite, seeds and budgets for every team.

```sh
qjlb holdout --output private-evaluation/suite
```

The default six cases vary contracts and dependence. ADB horizons are 1, 3 and 10 years; World Bank portfolios contain 2, 3 and 5 countries. Joint exceedance means every selected country incurs a payment. These are synthetic sensitivity and scaling cases within the existing model family.

The generator uses a fresh random seed and saves it with input hashes. Keep the suite, seed and reports private until judging ends. `private-evaluation/` is ignored by Git. Use `--seed` only to reproduce a saved evaluation. Committed test seeds are public practice inputs.

## 2. Run submitted code

Use Linux or macOS. Review code before running it. This local runner enforces deadlines and validates outputs; it provides no security sandbox. Run untrusted submissions in a separately managed isolated worker with no access to evaluator files or credentials.

```sh
qjlb evaluate private-evaluation/suite \
  --output private-evaluation/team-report \
  --execution classical_cpu --trusted-code \
  -- python "$PWD/team_solver.py"
```

Adapt `team_solver.py` for your method. Any executable can implement the same interface: it receives `--request request.json --output result.json` in a fresh directory. The request supplies one configuration, inputs, target names, seed and unique run ID. It supplies no reference answers.

Return the README's submission JSON, plus `run_id` and `seed` from the request. All resource fields must contain numbers; `model_evaluations` is also required. Counts must be nonnegative integers. `classical_cpu` requires zero shots and qubits. Wrong IDs, missing costs, invalid ranges, crashes and timeouts fail the run. Failed runs cannot be discarded.

## 3. Score accuracy and time

Defaults: 10 repeats, 8,192 baseline samples and a 60-second deadline per run. The sample count sets the classical sampling workload; the measured deadline is the common runtime budget. Freeze any changes before evaluation. Runs with fewer than 10 repeats or incomplete workload coverage are practice only and receive no track score.

Each target passes when RMSE across repeats is at most `max(relative × |reference|, absolute)`. These pilot tolerances are fixed by `evaluation-v1`; `P` is combined bond principal.

| Target | Relative | Absolute floor |
|---|---:|---:|
| ADB expected principal loss | 1% | 0.0001 × P USD |
| ADB joint attachment / portfolio exceedance | 5% | 0.0001 probability |
| ADB stop-loss | 5% | 0.00001 × P USD |
| ADB both exhausted | 10% | 0.000001 probability |
| World Bank economic loss / payment | 1% | USD 1,000 per year |
| World Bank joint / portfolio exceedance | 5% | 0.00001 per year |
| World Bank stop-loss | 5% | USD 100 per year |

Every target and case must pass. Compare median measured process time with the fastest passing classical method, including the exact solver. The track score is the geometric mean of these time ratios; lower is better. No score is assigned when a case fails or no classical comparator passes.

All methods use fresh processes. Timing includes startup, input loading, preparation and output writing. Input staging and scoring are excluded equally. `summary.csv` lists case results; `report.json` retains every run, hashes and environment details. Existing reports are never overwritten.

## 4. Verify execution claims

Quantum entries require positive qubits and `evidence` containing `provider`, `backend` and `circuit_sha256`. Hardware also requires positive shots and distinct provider `job_ids` for each invocation. Hybrid entries specify `evidence.quantum_execution` as `quantum_hardware` or `quantum_simulator`. Allow provider credentials explicitly with `--pass-env NAME` only in an appropriate worker.

Reports remain provisional. Independently match circuits, shots, job IDs and timings to provider records or simulator logs, and inspect the frozen code. JSON metadata alone cannot establish quantum execution. A speed-up claim must include this evidence and comparisons with the exact solver across workload sizes. Passing the numerical checks establishes accuracy under the supplied models.
