"""One-command NVIDIA A100 GPU experiment runner for ADB Quantum Innovation Challenge.

Runs the full high-resolution Phase 1 experimental suite using CUDA acceleration
(PyTorch CUDA statevector Grover operators for 22-qubit non-circular 3-year path IQAE
and 20-qubit full-catalogue World Bank ELT IQAE, plus 10-repeat holdout evaluation).

Usage on A100 node:
  python quantum_prototype/scripts/run_a100_experiments.py --run-holdout
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROTOTYPE_ROOT.parent
OFFICIAL_ROOT = REPO_ROOT / "adb-challenge-quantum-joint-loss-scenario-public-main"


def detect_gpu() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[GPU Detected] {name} ({vram_gb:.1f} GB VRAM, CUDA {torch.version.cuda})")
            return "cuda"
    except ImportError:
        pass
    print("[GPU Not Detected] Falling back to CPU backend.")
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument(
        "--run-holdout",
        action="store_true",
        help="Also generate a fresh holdout suite and run official `qjlb evaluate`.",
    )
    parser.add_argument("--holdout-seed", type=int, default=2026)
    args = parser.parse_args()

    device = detect_gpu()
    results_dir = PROTOTYPE_ROOT / ("results_a100" if device == "cuda" else "results")

    cmd = [
        sys.executable,
        str(PROTOTYPE_ROOT / "scripts/run_benchmarks.py"),
        "--device",
        device,
        "--repeats",
        str(args.repeats),
        "--samples",
        str(args.samples),
        "--results-dir",
        str(results_dir),
    ]
    print(f"[Run] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    if args.run_holdout:
        eval_root = PROTOTYPE_ROOT / "private-evaluation"
        suite_dir = eval_root / f"suite-seed-{args.holdout_seed}"
        report_dir = eval_root / f"report-{device}-seed-{args.holdout_seed}"
        if suite_dir.exists():
            shutil.rmtree(suite_dir)
        if report_dir.exists():
            shutil.rmtree(report_dir)

        qjlb_bin = str(Path(sys.executable).parent / "qjlb")
        subprocess.run(
            [
                qjlb_bin,
                "holdout",
                "--data",
                str(OFFICIAL_ROOT / "data"),
                "--output",
                str(suite_dir),
                "--seed",
                str(args.holdout_seed),
            ],
            check=True,
        )
        eval_cmd = [
            qjlb_bin,
            "evaluate",
            str(suite_dir),
            "--output",
            str(report_dir),
            "--execution",
            "quantum_simulator",
            "--trusted-code",
            "--repeats",
            str(args.repeats),
            "--samples",
            str(args.samples),
        ]
        if device == "cuda":
            import os

            if "CUDA_VISIBLE_DEVICES" in os.environ:
                eval_cmd.extend(["--pass-env", "CUDA_VISIBLE_DEVICES"])
        eval_cmd.extend(
            [
                "--",
                sys.executable,
                str(PROTOTYPE_ROOT / "team_solver.py"),
                "--solver-mode",
                "iqae",
                "--device",
                device,
            ]
        )
        print(f"[Evaluate] {' '.join(eval_cmd)}", flush=True)
        subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    main()
