# `quantum_prototype` — Dual-Track IQAE + MPS Solver & Benchmarking Suite

Standalone Phase 1 prototype for the **ADB Quantum Innovation Challenge: Joint-Loss Modelling for Disaster Risk Financing**, designed to sit alongside the untouched official benchmark package (`../adb-challenge-quantum-joint-loss-scenario-public-main`) while complying strictly with the official `qjlb evaluate` (`evaluation-v1`) protocol across both the **ADB Disaster Relief Bond (`adb_drb`)** and **World Bank SFRARR ELT (`worldbank_elt`)** tracks.

---

## 1. Directory Structure

```text
quantum_prototype/
├── README.md                          # This guide (architecture, usage, outputs)
├── requirements.txt                   # Local CPU dependencies (Python 3.10–3.14)
├── requirements-gpu.txt               # NVIDIA A100 (CUDA 12) dependencies
├── team_solver.py                     # Official qjlb worker entrypoint (--request / --output)
├── src/
│   └── quantum/
│       ├── __init__.py                # Public package exports
│       ├── circuits.py                # Qiskit 2.6 + OpenQASM 2.0 synthesis & SHA-256 digests
│       ├── iqae.py                    # Track A: Statevector Grover IQAE (8q, 22q path, 20q ELT)
│       ├── mps.py                     # Track B: n=2..20 MPS contraction, 5c Generative MPS, staircase bridge
│       ├── noise.py                   # Exact 8q Density-Matrix AWS Braket noise + Readout/ZNE mitigation
│       └── resources.py               # Chakrabarti-style classical & quantum resource estimator
├── scripts/
│   ├── run_benchmarks.py              # Reproducible benchmark CSV, JSON, and 4 PNG figure generator
│   └── run_a100_experiments.py        # One-command NVIDIA A100 CUDA runner (+ optional holdout eval)
├── tests/
│   └── test_quantum_solver.py         # 15 unit & integration tests (Qiskit SV, IQAE, MPS, ZNE, CLI)
├── results/                           # Generated benchmark CSVs, resource JSON, and 4 PNG figures
└── private-evaluation/                # Local qjlb holdout suites & reports (git-ignored)
```

---

## 2. Dual-Track Architecture

### Track A: Iterative Quantum Amplitude Estimation (IQAE) — `src/quantum/iqae.py`
Addresses **rare joint-tail sampling error** ($O(1/M)$ vs. classical $O(1/\sqrt{M})$) using Grinko et al. (2021) Clopper-Pearson scheduling with joint-round Maximum Likelihood refinement.
- **Real Unitary Statevector Grover Evolution**: Executes $\mathcal{Q} = -\mathcal{A} S_0 \mathcal{A}^\dagger S_\chi$ directly on the live statevector (step-by-step for $S \le 16{,}384$ and PyTorch CUDA GPU; verified Arnoldi/Krylov unitary reduction with residual $< 10^{-10}$ for 20q/22q CPU runs) — **no analytical $\sin^2((2k+1)\theta)$ shortcut**.
- **Regime A1 (`iqae_compact`, 8 qubits)**: 7 state qubits (100 capped term states) + 1 objective qubit, transpiled via Qiskit 2.6 to `{cx, rz, sx, x}` (`248 CX` per $\mathcal{A}$, `610 CX` per $\mathcal{Q}$) for exact gate counts, `circuit_sha256`, and noisy density-matrix mitigation tests.
- **Regime A2 (`iqae`, 22-qubit non-circular 3-year ADB path)**: Constructed strictly from the 100 annual states (`model.probabilities`) across $3 \times 7 = 21$ state qubits + 1 objective qubit ($100^3 = 10^6$ coherent paths) with **zero classical 3-year convolution precomputation** (transparently falls back to 8-qubit Markov-compressed IQAE for `term_years > 3`, such as 10-year holdout cases).
- **Regime A2 (`iqae`, 20-qubit World Bank ELT)**: Covers all `421,968` seismic + flood events (`19` state + `1` objective qubits), exploiting the verified uniform peril-rate structure (`audit.json`) for **$O(\log N)$ distribution loading (`122 CX`)** while honestly separating the raw tabular QROM lookup bottleneck (`2.53M CX`).

### Track B: Matrix Product State (MPS) Tensor Network — `src/quantum/mps.py`
Addresses the **$10^n$ multi-country state-space wall** and **limited-historical-data dependence learning**:
1. **Multi-Country Scaling Solver ($n = 2 \to 20$)**: Exact automaton MPO/MPS contraction for all 5 focus targets in $O(n \, U_T \, d \, \chi^2)$, validated to machine precision (`1.85e-16`) against brute-force $10^n$ enumeration for $n \le 4$ and contracting $n = 20$ ($10^{20}$ states) in `12.1 ms`.
2. **5-Country Generative MPS (Han et al., 2018)**: Born-machine amplitude MPS trained via exact analytical NLL gradient descent on finite subsamples of the **real 5-country World Bank SFRARR Central Asia ELT catalogue (`KAZ, KGZ, TJK, TKM, UZB`)**.
3. **Compiled MPS-to-Staircase Bridge (Ran, 2020)**: Compiles low-$\chi$ MPS into linear-depth isometric unitary staircases (`src/quantum/circuits.py`) to eliminate exponential Grover–Rudolph state-loading overhead for $n \ge 4$ countries.

### Noisy Hardware Simulation & Resource Estimation — `src/quantum/noise.py`, `resources.py`
- Exact $256 \times 256$ (8-qubit) density-matrix evolution under gate-calibrated depolarizing + asymmetric readout noise for **AWS Braket** QPUs (`IQM Emerald`, `IonQ Forte`, `Early-FT 1e-4`, `QuEra Libra 2028`) with **Readout Mitigation + Exponential/Richardson ZNE**.
- Chakrabarti-style end-to-end resource tables covering logical/ancilla/physical qubits, surface-code distance $d$, Megaquop budgets, and Braket simulator/QPU mapping.

---

## 3. Setup & Quickstart

### Local CPU Environment (Python 3.10–3.14)
Run from the repository root (`adb_quantum_challenge/`):

```bash
python3 -m venv .venv
.venv/bin/pip install -e "./adb-challenge-quantum-joint-loss-scenario-public-main[test]"
.venv/bin/pip install -r quantum_prototype/requirements.txt

# Verify official qjlb tests (54 tests) and quantum prototype tests (15 tests)
.venv/bin/pytest adb-challenge-quantum-joint-loss-scenario-public-main/tests/ -v
.venv/bin/pytest quantum_prototype/tests/ -v
```

### NVIDIA A100 GPU Environment (CUDA 12, Python 3.11/3.12 recommended)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e "./adb-challenge-quantum-joint-loss-scenario-public-main[test]"
pip install -r quantum_prototype/requirements-gpu.txt

# One-command A100 benchmark + 10-repeat holdout evaluation
python quantum_prototype/scripts/run_a100_experiments.py --repeats 10 --run-holdout
```

---

## 4. Running the Solver & Official `qjlb evaluate`

`team_solver.py` supports four `--solver-mode` options:
- `iqae` *(default)*: 22q non-circular 3-yr path IQAE (`adb_drb`) + 20q uniform-rate ELT IQAE (`worldbank_elt`) (`--execution quantum_simulator`)
- `iqae_compact`: 8q compact ADB circuit + 20q ELT IQAE (`--execution quantum_simulator`)
- `mps`: Deterministic Track B MPS contraction (`--execution classical_cpu`)
- `hybrid`: Track B MPS structural compression + Track A IQAE tail estimation (`--execution hybrid`)

### Official `evaluation-v1` Holdout Command
> **Note:** Pass absolute paths (`"$PWD/..."`) after `--` because `qjlb evaluate` spawns each invocation inside a fresh temporary directory.

```bash
.venv/bin/qjlb holdout \
  --data adb-challenge-quantum-joint-loss-scenario-public-main/data \
  --output quantum_prototype/private-evaluation/dev-suite \
  --seed 2026

.venv/bin/qjlb evaluate quantum_prototype/private-evaluation/dev-suite \
  --output quantum_prototype/private-evaluation/dev-report-iqae \
  --execution quantum_simulator --trusted-code --repeats 10 --samples 8192 \
  -- "$PWD/.venv/bin/python" "$PWD/quantum_prototype/team_solver.py" --solver-mode iqae --device cpu
```

### Regenerating All Benchmark Tables & Figures
```bash
.venv/bin/python quantum_prototype/scripts/run_benchmarks.py --device cpu --repeats 5 --samples 8192
```

---

## 5. Summary of Generated Outputs (`results/`)

| Output Artifact | Contents & Headline Findings |
| :--- | :--- |
| [public_scenarios_benchmark.csv](file:///home/alienx/DUK/other/adb_challenge/adb_quantum_challenge/quantum_prototype/results/public_scenarios_benchmark.csv) | Full comparison across all 7 public scenarios (`5 ADB` + `2 World Bank`). `iqae_compact_8q`, `iqae_path_22q`, `iqae_elt_20q`, `mps_chi10`, and `mps_chi4` pass **100% of targets on all 7 scenarios** (worst norm. RMSE `≤ 0.12`), while all 4 classical samplers fail all 7 at `8,192` samples. |
| [query_scaling_sweep.csv](file:///home/alienx/DUK/other/adb_challenge/adb_quantum_challenge/quantum_prototype/results/query_scaling_sweep.csv) & [fig1_iqae_vs_classical_scaling.png](file:///home/alienx/DUK/other/adb_challenge/adb_quantum_challenge/quantum_prototype/results/fig1_iqae_vs_classical_scaling.png) | Matched query-budget sweeps ($M \in [256, 32768]$) on `adb-base`. At $M=8{,}192$, IQAE beats Importance Sampling by **5.6×** (`0.63%` vs. `3.53%`) on exceedance (`1.205%`). At $M=16{,}384$, IQAE is the **only** estimator under the 10% tolerance on `both_bonds_exhausted` (`5.90%` vs. Importance `27.7%`, MC `96.5%`). |
| [mps_ncountry_scaling.csv](file:///home/alienx/DUK/other/adb_challenge/adb_quantum_challenge/quantum_prototype/results/mps_ncountry_scaling.csv), [mps_adb_base_truncation.csv](file:///home/alienx/DUK/other/adb_challenge/adb_quantum_challenge/quantum_prototype/results/mps_adb_base_truncation.csv), & [fig2_mps_ncountry_scaling.png](file:///home/alienx/DUK/other/adb_challenge/adb_quantum_challenge/quantum_prototype/results/fig2_mps_ncountry_scaling.png) | Multi-country scaling from $n=2$ to $n=20$ ($10^{20}$ states contracted in **`12.1 ms`**, `11,680` params at $\chi=8$, verified to `1.85e-16` vs. dense enumeration at $n \le 4$). On `adb-base`, $\chi=4$ (`80` params) achieves `0.003%` EL and `0.35%` exhaustion error. |
| [generative_mps_worldbank_5c.csv](file:///home/alienx/DUK/other/adb_challenge/adb_quantum_challenge/quantum_prototype/results/generative_mps_worldbank_5c.csv) & [fig3_generative_mps_worldbank.png](file:///home/alienx/DUK/other/adb_challenge/adb_quantum_challenge/quantum_prototype/results/fig3_generative_mps_worldbank.png) | 5-country Generative MPS ($\chi=4$, `224` params vs. `1,024` states) trained on real World Bank SFRARR data achieves **44% lower KL** at $N=2{,}000$ (`0.0295` vs. `0.0527`) and **62% lower KL** at $N=8{,}000$ (`0.0078` vs. `0.0206`) than Empirical MC. |
| [mps_staircase_compilation.csv](file:///home/alienx/DUK/other/adb_challenge/adb_quantum_challenge/quantum_prototype/results/mps_staircase_compilation.csv) | Compiled Ran (2020) MPS staircase circuits (`99.895%` fidelity at $\chi=2$, `99.9999985%` at $\chi=4$) with linear-vs-exponential CX crossover at $n \ge 4$ (`38,300 CX` vs. `1.05M CX` at $n=5$). |
| [noisy_braket_zne_summary.csv](file:///home/alienx/DUK/other/adb_challenge/adb_quantum_challenge/quantum_prototype/results/noisy_braket_zne_summary.csv) & [fig4_noisy_braket_zne.png](file:///home/alienx/DUK/other/adb_challenge/adb_quantum_challenge/quantum_prototype/results/fig4_noisy_braket_zne.png) | Exact 8q density-matrix Braket QPU noise + ZNE sweep: ZNE extends coherence to $k=1$ on IQM Emerald (`4.1%` err) and $k=4$ on IonQ Forte (`0.37%` err at $k=1$), while QuEra Libra (2028) remains coherent raw through $k=32$. |
| [resource_estimation.json](file:///home/alienx/DUK/other/adb_challenge/adb_quantum_challenge/quantum_prototype/results/resource_estimation.json) | Complete classical/quantum resource breakdown: `0.275 Megaquops` (8q ADB), `0.894 Megaquops` (22q non-circular 3-yr path ADB, `36` logical qubits with ancillas, `d=11` surface code), and `122 CX` $O(\log N)$ ELT state loading vs. `2.53M CX` raw QROM. |

### Official Holdout Verification Snapshot (`private-evaluation/dev-report-iqae/`)
- **Protocol Status**: `evaluation-v1`, `protocol_eligible: True` (`10` repeats, `6/6` cases passed, `0` invalid runs).
- **Track Scores**: `adb_drb` geometric time ratio `1.70×` (`all_cases_pass: True`, worst norm. RMSE `0.0933`); `worldbank_elt` geometric time ratio `1.43×` (`all_cases_pass: True`, worst norm. RMSE `0.0991`).

---

## 6. Data Attribution & AI Disclosure
- **World Bank SFRARR Data (CC BY 4.0)**: Seismic and flood Event Loss Tables (`seismic.npz`, `flood.npz`) are sourced unmodified from the official ADB challenge package under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license (World Bank / GFDRR SFRARR Central Asia Programme; see `../adb-challenge-quantum-joint-loss-scenario-public-main/data/LICENSE.txt`).
- **AI Assistance Disclosure**: Code scaffolding, documentation drafting, and verification automation were assisted by Antigravity IDE (Google DeepMind); all mathematical derivations, circuit encodings, benchmarks, and `qjlb evaluate` results were empirically verified against the official test and evaluation harness.
