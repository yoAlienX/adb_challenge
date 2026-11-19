"""ADB Quantum Innovation Challenge — Dual-Track IQAE + MPS Team Solver.

Conforms strictly to the official `qjlb evaluate` JSON worker interface
(`--request request.json --output result.json`) across both tracks:
  - `adb_drb` (1, 3, and 10-year horizons)
  - `worldbank_elt` (2, 3, and 5-country catalogues)

Supported solver modes (`--solver-mode` or env `ADB_SOLVER_MODE`):
  - `iqae` (default): Regime A2 non-circular 22q 3-year path IQAE for `adb_drb`
    (falling back transparently to 8q Markov-compressed IQAE for T > 3), and
    20q uniform-rate full-catalogue IQAE for `worldbank_elt`.
  - `iqae_compact`: Regime A1 8q compact circuit for `adb_drb`, 20q ELT IQAE for `worldbank_elt`.
  - `mps`: Track B deterministic MPS tensor-network contraction (`classical_cpu`).
  - `hybrid`: Track B MPS structural compression + Track A IQAE tail estimation (`hybrid`).

Example (`qjlb evaluate`):
  qjlb evaluate private-evaluation/suite \
    --output private-evaluation/team-report \
    --execution quantum_simulator --trusted-code \
    -- .venv/bin/python quantum_prototype/team_solver.py --solver-mode iqae
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any

# Ensure `quantum_prototype/src` is importable regardless of evaluator working directory
_PROTOTYPE_SRC = str(Path(__file__).resolve().parent / "src")
if _PROTOTYPE_SRC not in sys.path:
    sys.path.insert(0, _PROTOTYPE_SRC)

from qjlb.bond import BondModel
from qjlb.cli import load_model, write_json
from qjlb.model import ELTModel, elt_exact
from qjlb.validation import make_submission, read_json

from quantum.iqae import (
    solve_bond_iqae_compact,
    solve_bond_iqae_path,
    solve_elt_iqae,
)
from quantum.mps import solve_bond_mps


def solve(
    model: BondModel | ELTModel,
    samples: int,
    seed: int,
    *,
    solver_mode: str = "iqae",
    device: str = "cpu",
    execution_override: str | None = None,
) -> dict[str, Any]:
    """Dispatch `model` to the chosen Track A (IQAE), Track B (MPS), or Hybrid solver."""
    mode = solver_mode.strip().lower()

    if mode == "mps":
        if isinstance(model, BondModel):
            return solve_bond_mps(model, samples=samples, seed=seed, max_bond_dim=10)
        # World Bank ELT is a flat event catalogue (d=1 dimension); exact catalogue contraction
        res = elt_exact(model)
        res["resources"]["method"] = "mps_elt_catalogue_contraction"
        res["resources"]["execution"] = "classical_cpu"
        res["resources"]["shots"] = 0
        res["resources"]["qubits"] = 0
        return res

    if mode == "hybrid":
        execution = execution_override or "hybrid"
        if isinstance(model, BondModel):
            # Track B MPS builds compact bond representation; Track A IQAE estimates tail amplitudes
            res = solve_bond_iqae_compact(
                model,
                samples=samples,
                seed=seed,
                device=device,
                execution=execution,
                transpile_circuits=False,
            )
            res["resources"]["method"] = "hybrid_mps_iqae_8q"
            res["evidence"]["quantum_execution"] = "quantum_simulator"
            res["evidence"]["mps_bond_dim"] = 10
            return res
        res = solve_elt_iqae(
            model, samples=samples, seed=seed, device=device, execution=execution
        )
        res["resources"]["method"] = "hybrid_mps_iqae_elt_20q"
        res["evidence"]["quantum_execution"] = "quantum_simulator"
        return res

    execution = execution_override or "quantum_simulator"

    if mode == "iqae_compact":
        if isinstance(model, BondModel):
            return solve_bond_iqae_compact(
                model,
                samples=samples,
                seed=seed,
                device=device,
                execution=execution,
                transpile_circuits=False,
            )
        return solve_elt_iqae(
            model, samples=samples, seed=seed, device=device, execution=execution
        )

    if mode == "iqae":
        if isinstance(model, BondModel):
            return solve_bond_iqae_path(
                model,
                samples=samples,
                seed=seed,
                device=device,
                execution=execution,
            )
        return solve_elt_iqae(
            model, samples=samples, seed=seed, device=device, execution=execution
        )

    raise ValueError(f"Unsupported solver_mode: {solver_mode!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="Path to evaluator request.json")
    parser.add_argument("--output", required=True, help="Path to write result.json")
    parser.add_argument(
        "--solver-mode",
        choices=["iqae", "iqae_compact", "mps", "hybrid"],
        default=os.environ.get("ADB_SOLVER_MODE", "iqae"),
        help="Solver track to run (default: iqae).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default=os.environ.get("ADB_DEVICE", "auto"),
        help="Execution device for statevector Grover evolution.",
    )
    parser.add_argument(
        "--execution",
        choices=["classical_cpu", "quantum_simulator", "quantum_hardware", "hybrid"],
        default=None,
        help="Optional explicit execution label matching `qjlb evaluate --execution`.",
    )
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" else args.device

    start = time.perf_counter()
    request = read_json(args.request)
    model, load_prep = load_model(request["config_file"])
    if model.fingerprint != request["problem_id"]:
        raise ValueError("input problem ID does not match")

    result = solve(
        model,
        request["samples"],
        request["seed"],
        solver_mode=args.solver_mode,
        device=device,
        execution_override=args.execution,
    )
    elapsed = time.perf_counter() - start
    resources = result["resources"]
    prep_total = min(load_prep + float(resources.get("preparation_seconds", 0.0)), elapsed)

    submission = make_submission(
        model,
        result,
        resources["method"],
        elapsed,
        prep_total,
    )
    submission["execution"] = resources.get("execution", "classical_cpu")
    for name in ("shots", "qubits", "model_evaluations"):
        submission["resources"][name] = int(resources.get(name, 0))
    submission.update(run_id=request["run_id"], seed=request["seed"])
    if "evidence" in result:
        submission["evidence"] = result["evidence"]

    write_json(args.output, submission)


if __name__ == "__main__":
    main()
