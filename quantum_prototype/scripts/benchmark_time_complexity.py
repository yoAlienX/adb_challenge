"""Platform-aware empirical CPU/GPU time-complexity profiler for quantum_prototype.

Automatically inspects the runtime environment (OS, distro, CPU model/cores, RAM,
Python/NumPy/SciPy/Qiskit versions, PyTorch CUDA/MPS & nvidia-smi GPU detection),
selects the best available float64 compute backend (`cuda` or `cpu`), and empirically
measures and fits power-law time-complexity exponents T(x) = c * x^alpha across:

  1. Track A (IQAE) Statevector Grover Step Time vs. Hilbert Space Size N = 2^q
     (theoretical: O(2^q) = O(N) per direct Grover step; compares Direct vs. Verified Krylov).
  2. Track A (IQAE) Runtime vs. Query / Grover Budget M
     (theoretical: O(M) direct vs. O(log M) verified Krylov).
  3. Track B (MPS) Contraction Time vs. Country Count n (n=2..20) & Bond Dimension chi
     (theoretical: O(n) linear in n, O(chi^2) in bond dimension vs. O(10^n) dense exact).
  4. Classical Samplers (MC, QMC, Stratified, Importance) vs. Sample Budget M
     (theoretical: O(M) linear in samples).

Outputs saved to `quantum_prototype/results/`:
  - `time_complexity_report.json`
  - `time_complexity_measurements.csv`
  - `fig5_time_complexity_<device>.png`
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any
import numpy as np
import scipy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROTOTYPE_ROOT.parent
OFFICIAL_ROOT = REPO_ROOT / "adb-challenge-quantum-joint-loss-scenario-public-main"

if str(PROTOTYPE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_ROOT / "src"))

from qjlb.bond import BOND_METHODS, BondModel

from quantum.iqae import StatevectorGroverEngine, iqae_statevector_estimate
from quantum.mps import build_synthetic_ncountry_mps


def _read_linux_distro() -> str:
    os_release = Path("/etc/os-release")
    if os_release.is_file():
        for line in os_release.read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
    return f"{platform.system()} {platform.release()}"


def _read_cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _read_ram_gb() -> tuple[float | None, float | None]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None, None
    total_kb = avail_kb = None
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemTotal:"):
            total_kb = int(line.split()[1])
        elif line.startswith("MemAvailable:"):
            avail_kb = int(line.split()[1])
    return (
        round(total_kb / (1024**2), 2) if total_kb else None,
        round(avail_kb / (1024**2), 2) if avail_kb else None,
    )


def _probe_nvidia_smi() -> dict[str, Any]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"present": False, "status": "nvidia-smi not found"}
    try:
        proc = subprocess.run(
            [
                smi,
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout).strip().splitlines()
            return {
                "present": True,
                "status": "driver_unavailable",
                "detail": err[0] if err else f"exit code {proc.returncode}",
            }
        gpus = []
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append(
                    {
                        "name": parts[0],
                        "driver_version": parts[1],
                        "vram_mb": float(parts[2]),
                        "compute_cap": parts[3],
                    }
                )
        return {"present": True, "status": "active", "gpus": gpus}
    except Exception as exc:  # pragma: no cover
        return {"present": True, "status": "error", "detail": str(exc)}


def inspect_environment(requested_device: str = "auto") -> dict[str, Any]:
    """Inspect OS, CPU, RAM, GPU/CUDA/MPS, and quantum/scientific library stack."""
    total_ram_gb, avail_ram_gb = _read_ram_gb()
    nvidia_info = _probe_nvidia_smi()

    # Probe PyTorch / CUDA / Apple MPS
    torch_info: dict[str, Any] = {"installed": False, "cuda_available": False, "mps_available": False}
    try:
        import torch  # type: ignore[import-not-found]

        torch_info["installed"] = True
        torch_info["version"] = torch.__version__
        torch_info["cuda_version"] = getattr(torch.version, "cuda", None)
        torch_info["cuda_available"] = bool(torch.cuda.is_available())
        torch_info["mps_available"] = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if torch_info["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            torch_info["gpu_name"] = torch.cuda.get_device_name(0)
            torch_info["gpu_vram_gb"] = round(props.total_memory / (1024**3), 2)
            torch_info["compute_capability"] = f"{props.major}.{props.minor}"
            torch_info["multiprocessor_count"] = int(props.multi_processor_count)
    except ImportError:
        pass

    # Probe Qiskit & Aer
    qiskit_ver = None
    aer_ver = None
    try:
        import qiskit

        qiskit_ver = qiskit.__version__
    except ImportError:
        pass
    try:
        import qiskit_aer  # type: ignore[import-not-found]

        aer_ver = qiskit_aer.__version__
    except ImportError:
        pass

    # Decide active compute device
    fallback_reason: str | None = None
    if requested_device == "cuda":
        if torch_info["cuda_available"]:
            active_device = "cuda"
        else:
            active_device = "cpu"
            fallback_reason = "CUDA requested but torch.cuda.is_available() is False; falling back to CPU."
    elif requested_device == "cpu":
        active_device = "cpu"
    else:  # auto
        if torch_info["cuda_available"]:
            active_device = "cuda"
        else:
            active_device = "cpu"
            if not torch_info["installed"]:
                fallback_reason = "PyTorch not installed; running on CPU."
            elif nvidia_info.get("status") != "active":
                fallback_reason = "No active NVIDIA CUDA driver detected; running on CPU."
            else:
                fallback_reason = "PyTorch installed without active CUDA support; running on CPU."

    cpu_cores = os.cpu_count() or 1
    if active_device == "cuda":
        hardware_label = f"GPU: {torch_info['gpu_name']} ({torch_info['gpu_vram_gb']} GB)"
    else:
        hardware_label = f"CPU: {_read_cpu_model()} ({cpu_cores} threads)"

    return {
        "os": platform.system(),
        "distro": _read_linux_distro(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_model": _read_cpu_model(),
        "cpu_logical_cores": cpu_cores,
        "ram_total_gb": total_ram_gb,
        "ram_available_gb": avail_ram_gb,
        "blas_threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "unset"),
        },
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "qiskit": qiskit_ver,
            "qiskit_aer": aer_ver,
        },
        "nvidia_smi": nvidia_info,
        "torch": torch_info,
        "requested_device": requested_device,
        "active_device": active_device,
        "fallback_reason": fallback_reason,
        "hardware_label": hardware_label,
    }


def fit_power_law(
    xs: Sequence[float] | np.ndarray,
    ts: Sequence[float] | np.ndarray,
) -> dict[str, float | None]:
    """Fit log(T) = log(c) + alpha * log(x) to estimate empirical complexity exponent alpha."""
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ts, dtype=np.float64)
    mask = (x > 0) & (y > 0)
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    if lx.size < 2:
        return {"alpha": None, "coeff": None, "r2": None}
    slope, intercept = np.polyfit(lx, ly, 1)
    y_pred = slope * lx + intercept
    ss_res = float(np.sum((ly - y_pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "alpha": float(slope),
        "coeff": float(math.exp(intercept)),
        "r2": float(r2),
    }


def _sync_device(device: str) -> None:
    if device == "cuda":
        import torch  # type: ignore[import-not-found]

        torch.cuda.synchronize()


def benchmark_grover_hilbert_scaling(
    device: str,
    qubit_counts: list[int],
    steps_per_trial: int = 8,
    repeats: int = 3,
) -> list[dict[str, Any]]:
    """Measure per-step direct statevector Grover time & Krylov setup time vs. N = 2^q."""
    rng = np.random.default_rng(123)
    rows: list[dict[str, Any]] = []

    for q in qubit_counts:
        # State register has q-1 qubits -> S = 2^(q-1) states (+ 1 objective qubit = 2^q amps)
        s_states = 1 << (q - 1)
        probs = rng.dirichlet(np.ones(min(s_states, 64)))
        if s_states > 64:
            full_p = np.full(s_states, 1.0 / s_states, dtype=np.float64)
            full_p[:64] += probs
            full_p /= full_p.sum()
            probs = full_p
        payoff = (rng.random(s_states) < 0.02).astype(np.float64)
        payoff[0] = 1.0

        # 1. Direct statevector step timing (O(2^q) per Grover iterate)
        eng_direct = StatevectorGroverEngine(probs, payoff, device=device, force_direct=True)
        # Warmup
        eng_direct.advance_to(1)
        _sync_device(device)
        eng_direct.reset()

        step_times = []
        for _ in range(repeats):
            eng_direct.reset()
            _sync_device(device)
            t0 = time.perf_counter()
            eng_direct.advance_to(steps_per_trial)
            _sync_device(device)
            step_times.append((time.perf_counter() - t0) / steps_per_trial)

        rows.append(
            {
                "suite": "iqae_grover_step_vs_hilbert",
                "method": f"direct_statevector_{device}",
                "param_name": "hilbert_dim_2^q",
                "param_value": float(1 << q),
                "qubits": q,
                "median_seconds": float(np.median(step_times)),
                "min_seconds": float(np.min(step_times)),
                "theoretical_big_o": "O(2^q)",
            }
        )

        # 2. Verified Krylov setup + k=500 evolution timing (CPU speedup benchmark)
        if device == "cpu" and s_states > 16_384:
            krylov_times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                eng_k = StatevectorGroverEngine(probs, payoff, device="cpu", force_direct=False)
                eng_k.advance_to(500)
                _ = eng_k.born_probability_one()
                krylov_times.append(time.perf_counter() - t0)
            rows.append(
                {
                    "suite": "iqae_krylov_500steps_vs_hilbert",
                    "method": "verified_krylov_cpu",
                    "param_name": "hilbert_dim_2^q",
                    "param_value": float(1 << q),
                    "qubits": q,
                    "median_seconds": float(np.median(krylov_times)),
                    "min_seconds": float(np.min(krylov_times)),
                    "theoretical_big_o": "O(2^q) setup + O(log k)",
                }
            )

    return rows


def benchmark_iqae_vs_query_budget(
    adb_base: BondModel,
    device: str,
    budgets: list[int],
    repeats: int = 3,
) -> list[dict[str, Any]]:
    """Measure Track A IQAE runtime vs. oracle-call budget M on adb-base (8q)."""
    probs = adb_base.term_distribution()
    payoff = adb_base.observables(adb_base.states)[:, 2]  # portfolio exceedance
    rows: list[dict[str, Any]] = []

    for m in budgets:
        ts = []
        for r in range(repeats):
            _sync_device(device)
            t0 = time.perf_counter()
            iqae_statevector_estimate(
                probs,
                payoff,
                seed=700 + r + m,
                shots_per_round=16,
                max_oracle_calls=m,
                max_k=4096,
                qubits_override=8,
                device=device,
            )
            _sync_device(device)
            ts.append(time.perf_counter() - t0)
        rows.append(
            {
                "suite": "iqae_time_vs_queries",
                "method": f"iqae_8q_{device}",
                "param_name": "oracle_budget_M",
                "param_value": float(m),
                "qubits": 8,
                "median_seconds": float(np.median(ts)),
                "min_seconds": float(np.min(ts)),
                "theoretical_big_o": "O(M)",
            }
        )
    return rows


def benchmark_mps_complexity(
    adb_base: BondModel,
    country_counts: list[int],
    bond_dims: list[int],
    repeats: int = 3,
) -> list[dict[str, Any]]:
    """Measure Track B MPS contraction time vs. n (O(n)) and vs. chi (O(chi^2))."""
    rows: list[dict[str, Any]] = []

    # 1. Scaling vs. number of countries n at fixed chi=8
    for n in country_counts:
        mps = build_synthetic_ncountry_mps(adb_base, n_countries=n, max_bond_dim=8)
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            mps.contract_all_targets(adb_base.threshold)
            ts.append(time.perf_counter() - t0)
        rows.append(
            {
                "suite": "mps_time_vs_countries",
                "method": "mps_chi8_cpu",
                "param_name": "n_countries",
                "param_value": float(n),
                "qubits": 0,
                "median_seconds": float(np.median(ts)),
                "min_seconds": float(np.min(ts)),
                "theoretical_big_o": "O(n)",
            }
        )

    # 2. Scaling vs. bond dimension chi at fixed n=10
    for chi in bond_dims:
        mps = build_synthetic_ncountry_mps(adb_base, n_countries=10, max_bond_dim=chi)
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            mps.contract_all_targets(adb_base.threshold)
            ts.append(time.perf_counter() - t0)
        rows.append(
            {
                "suite": "mps_time_vs_bond_dim",
                "method": "mps_n10_cpu",
                "param_name": "bond_dim_chi",
                "param_value": float(chi),
                "qubits": 0,
                "median_seconds": float(np.median(ts)),
                "min_seconds": float(np.min(ts)),
                "theoretical_big_o": "O(chi^2)",
            }
        )

    return rows


def benchmark_classical_complexity(
    adb_base: BondModel,
    budgets: list[int],
    repeats: int = 3,
) -> list[dict[str, Any]]:
    """Measure classical sampler runtime vs. sample count M (theoretical O(M))."""
    rows: list[dict[str, Any]] = []
    for cls_name in ("mc", "qmc", "stratified", "importance"):
        fn = BOND_METHODS[cls_name]
        for m in budgets:
            ts = []
            for r in range(repeats):
                t0 = time.perf_counter()
                fn(adb_base, m, 900 + r + m)
                ts.append(time.perf_counter() - t0)
            rows.append(
                {
                    "suite": "classical_time_vs_samples",
                    "method": f"classical_{cls_name}",
                    "param_name": "samples_M",
                    "param_value": float(m),
                    "qubits": 0,
                    "median_seconds": float(np.median(ts)),
                    "min_seconds": float(np.min(ts)),
                    "theoretical_big_o": "O(M)",
                }
            )
    return rows


def plot_time_complexity(
    env: dict[str, Any],
    all_rows: list[dict[str, Any]],
    fits: dict[str, dict[str, Any]],
    out_path: Path,
) -> None:
    """Generate 2x2 log-log empirical time-complexity figure tagged with detected platform."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8), dpi=160)

    # Panel 1: Direct Statevector Grover Step Time vs Hilbert Dim 2^q
    ax = axes[0, 0]
    sub = [r for r in all_rows if r["suite"] == "iqae_grover_step_vs_hilbert"]
    xs = [r["param_value"] for r in sub]
    ys = [r["median_seconds"] * 1000.0 for r in sub]
    f1 = fits["iqae_grover_step_vs_hilbert"]
    ax.loglog(xs, ys, "o-", color="#c0392b", linewidth=2.2, markersize=6.5, label=f"Measured ({env['active_device'].upper()})")
    if f1["alpha"] is not None and f1["coeff"] is not None and f1["r2"] is not None:
        x_fit = np.array([min(xs), max(xs)])
        ax.loglog(
            x_fit,
            f1["coeff"] * (x_fit ** f1["alpha"]) * 1000.0,
            "--",
            color="#555555",
            label=f"Power-law fit: $O(N^{{{f1['alpha']:.2f}}})$ ($R^2={f1['r2']:.3f}$)",
        )
    ax.set_title("Track A: Single Grover Step vs. Hilbert Dim $N = 2^q$", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Statevector Dimension $N = 2^q$", fontsize=9.5)
    ax.set_ylabel("Time per Grover Step (ms)", fontsize=9.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8.5)

    # Panel 2: Track A IQAE vs Classical Samplers vs Query/Sample Budget M
    ax = axes[0, 1]
    iq_sub = [r for r in all_rows if r["suite"] == "iqae_time_vs_queries"]
    f2 = fits["iqae_time_vs_queries"]
    f2_lbl = f"$\\alpha={f2['alpha']:.2f}$" if f2["alpha"] is not None else "n/a"
    ax.loglog(
        [r["param_value"] for r in iq_sub],
        [r["median_seconds"] * 1000.0 for r in iq_sub],
        "*-",
        color="#c0392b",
        linewidth=2.3,
        markersize=8,
        label=f"IQAE 8q ({f2_lbl})",
    )
    cls_styles = {
        "classical_mc": ("#7f8c8d", "o", "MC"),
        "classical_qmc": ("#3498db", "s", "QMC"),
        "classical_stratified": ("#e67e22", "^", "Stratified"),
        "classical_importance": ("#9b59b6", "D", "Importance"),
    }
    for mkey, (col, mk, lbl) in cls_styles.items():
        csub = [r for r in all_rows if r["method"] == mkey]
        fc = fits[mkey]
        fc_lbl = f"$\\alpha={fc['alpha']:.2f}$" if fc["alpha"] is not None else "n/a"
        ax.loglog(
            [r["param_value"] for r in csub],
            [r["median_seconds"] * 1000.0 for r in csub],
            f"--{mk}",
            color=col,
            linewidth=1.6,
            markersize=5,
            label=f"{lbl} ({fc_lbl})",
        )
    ax.set_title("Runtime vs. Query / Sample Budget $M$ (adb-base)", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Budget $M$ (Oracle Calls / Classical Samples)", fontsize=9.5)
    ax.set_ylabel("Wall-Clock Runtime (ms)", fontsize=9.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    # Panel 3: Track B MPS Contraction Time vs Number of Countries n
    ax = axes[1, 0]
    m_n = [r for r in all_rows if r["suite"] == "mps_time_vs_countries"]
    xs_n = [r["param_value"] for r in m_n]
    ys_n = [r["median_seconds"] * 1000.0 for r in m_n]
    f3 = fits["mps_time_vs_countries"]
    ax.loglog(xs_n, ys_n, "s-", color="#27ae60", linewidth=2.2, markersize=6.5, label="MPS contraction ($\\chi=8$)")
    if f3["alpha"] is not None and f3["coeff"] is not None and f3["r2"] is not None:
        x_fit = np.array([min(xs_n), max(xs_n)])
        ax.loglog(
            x_fit,
            f3["coeff"] * (x_fit ** f3["alpha"]) * 1000.0,
            "--",
            color="#555555",
            label=f"Fit: $O(n^{{{f3['alpha']:.2f}}})$ ($R^2={f3['r2']:.3f}$, theory $O(n)$)",
        )
    ax.set_title("Track B MPS: Contraction Time vs. Countries $n$ ($10^2 \\to 10^{20}$)", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Number of Countries $n$", fontsize=9.5)
    ax.set_ylabel("Contraction Time (ms)", fontsize=9.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8.5)

    # Panel 4: Track B MPS Contraction Time vs Bond Dimension chi
    ax = axes[1, 1]
    m_c = [r for r in all_rows if r["suite"] == "mps_time_vs_bond_dim"]
    xs_c = [r["param_value"] for r in m_c]
    ys_c = [r["median_seconds"] * 1000.0 for r in m_c]
    f4 = fits["mps_time_vs_bond_dim"]
    ax.loglog(xs_c, ys_c, "D-", color="#2980b9", linewidth=2.2, markersize=6.5, label="MPS contraction ($n=10$)")
    if f4["alpha"] is not None and f4["coeff"] is not None and f4["r2"] is not None:
        x_fit = np.array([min(xs_c), max(xs_c)])
        ax.loglog(
            x_fit,
            f4["coeff"] * (x_fit ** f4["alpha"]) * 1000.0,
            "--",
            color="#555555",
            label=f"Fit: $O(\\chi^{{{f4['alpha']:.2f}}})$ ($R^2={f4['r2']:.3f}$, theory $\\leq O(\\chi^2)$)",
        )
    ax.set_title("Track B MPS: Contraction Time vs. Bond Dimension $\\chi$ ($n=10$)", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("MPS Bond Dimension $\\chi$", fontsize=9.5)
    ax.set_ylabel("Contraction Time (ms)", fontsize=9.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8.5)

    fig.suptitle(
        f"Empirical Time-Complexity Analysis — {env['hardware_label']} ({env['distro']})",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_time_complexity_suite(
    requested_device: str = "auto",
    repeats: int = 3,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """Run full platform-aware time-complexity suite and persist JSON, CSV, and PNG."""
    if results_dir is None:
        results_dir = PROTOTYPE_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    env = inspect_environment(requested_device=requested_device)
    device = env["active_device"]
    print(f"[Platform] {env['hardware_label']} | OS: {env['distro']} | Active: {device.upper()}", flush=True)
    if env.get("fallback_reason"):
        print(f"[Platform Note] {env['fallback_reason']}", flush=True)

    adb_base = BondModel.load(OFFICIAL_ROOT / "scenarios/adb-base.json")

    # Adjust upper qubit range automatically based on detected hardware
    if device == "cuda":
        qubit_counts = [8, 10, 12, 14, 16, 18, 20, 22, 24]
    else:
        qubit_counts = [8, 10, 12, 14, 16, 18, 20, 22]

    all_rows: list[dict[str, Any]] = []
    print(f"[1/4] Benchmarking Grover statevector scaling across q={qubit_counts} ...", flush=True)
    all_rows.extend(benchmark_grover_hilbert_scaling(device, qubit_counts, steps_per_trial=8, repeats=repeats))

    print("[2/4] Benchmarking Track A IQAE time vs. query budget M ...", flush=True)
    all_rows.extend(benchmark_iqae_vs_query_budget(adb_base, device, [256, 1024, 4096, 16384, 65536], repeats=repeats))

    print("[3/4] Benchmarking Track B MPS time vs. n=2..20 and chi=2..10 ...", flush=True)
    all_rows.extend(benchmark_mps_complexity(adb_base, [2, 4, 6, 8, 10, 14, 18, 20], [2, 4, 6, 8, 10], repeats=repeats))

    print("[4/4] Benchmarking Classical Baselines vs. sample budget M ...", flush=True)
    all_rows.extend(benchmark_classical_complexity(adb_base, [1024, 4096, 16384, 65536], repeats=repeats))

    # Tag every row with detected platform metadata
    for row in all_rows:
        row["active_device"] = device
        row["os_distro"] = env["distro"]
        row["hardware_label"] = env["hardware_label"]

    # Fit empirical power-law exponents T(x) = c * x^alpha
    fits: dict[str, dict[str, Any]] = {}
    for suite_key in ("iqae_grover_step_vs_hilbert", "iqae_time_vs_queries", "mps_time_vs_countries", "mps_time_vs_bond_dim"):
        sub = [r for r in all_rows if r["suite"] == suite_key]
        # Fit upper asymptotic tail for Hilbert dim (q >= 12) where Python overhead vanishes
        if suite_key == "iqae_grover_step_vs_hilbert":
            sub_fit = [r for r in sub if r["qubits"] >= 12]
        else:
            sub_fit = sub
        fits[suite_key] = {
            **fit_power_law([r["param_value"] for r in sub_fit], [r["median_seconds"] for r in sub_fit]),
            "theoretical_big_o": sub[0]["theoretical_big_o"],
        }

    for cls_key in ("classical_mc", "classical_qmc", "classical_stratified", "classical_importance"):
        sub = [r for r in all_rows if r["method"] == cls_key]
        fits[cls_key] = {
            **fit_power_law([r["param_value"] for r in sub], [r["median_seconds"] for r in sub]),
            "theoretical_big_o": "O(M)",
        }

    # Save CSV
    csv_path = results_dir / "time_complexity_measurements.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    # Save JSON report
    report = {
        "environment": env,
        "empirical_complexity_fits": fits,
        "measurements": all_rows,
    }
    json_path = results_dir / "time_complexity_report.json"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    # Save Figure 5
    fig_path = results_dir / f"fig5_time_complexity_{device}.png"
    plot_time_complexity(env, all_rows, fits, fig_path)

    print("\n=== Empirical Time-Complexity Summary ===", flush=True)
    for k, v in fits.items():
        if v["alpha"] is not None and v["r2"] is not None:
            print(f"  {k:32s}: T(x) ~ x^{v['alpha']:5.2f}  (R^2={v['r2']:.3f}, theory: {v['theoretical_big_o']})")
        else:
            print(f"  {k:32s}: insufficient data (theory: {v['theoretical_big_o']})")
    print(f"Saved: {json_path}, {csv_path}, {fig_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute backend (default: auto-detect GPU/CPU).",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--results-dir", type=Path, default=PROTOTYPE_ROOT / "results")
    args = parser.parse_args()
    run_time_complexity_suite(
        requested_device=args.device,
        repeats=args.repeats,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
