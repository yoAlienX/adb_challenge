# ADB Quantum Innovation Challenge — Dual-Track IQAE + MPS Prototype

Self-contained team prototype for the **ADB Quantum Innovation Challenge: Joint-Loss Modelling for Disaster Risk Financing**, designed to keep the official benchmark repository (`../adb-challenge-quantum-joint-loss-scenario-public-main`) untouched while complying strictly with the `qjlb evaluate` (`evaluation-v1`) protocol.

## Architecture

### Track A: Iterative Quantum Amplitude Estimation (IQAE) (`src/quantum/iqae.py`)
- Executes **real unitary statevector Grover operators** $\mathcal{Q} = -\mathcal{A} S_0 \mathcal{A}^\dagger S_\chi$ (no analytical $\sin^2((2k+1)\theta)$ shortcut) with Grinko et al. (2021) Clopper-Pearson scheduling and joint-round Maximum Likelihood refinement.
- **Regime A1 (8-qubit compact ADB circuit)**: 7 state qubits + 1 objective qubit transpiled via Qiskit to `{cx, rz, sx, x}` for exact 2Q gate counts, depth, OpenQASM 2.0 `circuit_sha256`, and noisy density-matrix error mitigation.
- **Regime A2 (22-qubit non-circular 3-year ADB path IQAE)**: Constructed strictly from `annual-states` ($3 \times 7 + 1 = 22$ qubits, $100^3 = 10^6$ coherent paths) with **zero classical 3-year convolution precomputation** (automatically accelerated on NVIDIA A100 via PyTorch CUDA).
- **Regime A2 (20-qubit World Bank ELT IQAE)**: Covers all `421,968` seismic + flood events (`19` state + `1` objective qubits), exploiting the verified uniform peril-rate structure (`audit.json`) for $O(\log N)$ state preparation while honestly separating tabular QROM oracle overhead.

### Track B: Matrix Product State (MPS) Tensor Network (`src/quantum/mps.py`)
1. **Multi-Country Scaling Solver ($n = 2 \to 20$)**: Exact automaton MPO/MPS contraction for all 5 focus targets in $O(n \, U_T \, d \, \chi^2)$, validated to machine precision against brute-force $10^n$ enumeration for $n \le 4$ and scaling cleanly to $n = 20$ ($10^{20}$ states).
2. **5-Country Generative MPS (Han et al., 2018)**: Born-machine amplitude MPS trained via exact analytical NLL gradient descent on finite subsamples of the real 5-country World Bank SFRARR Central Asia dataset (`KAZ, KGZ, TJK, TKM, UZB`).
3. **Compiled MPS-to-Staircase Bridge (Ran, 2020)**: Compiles low-$\chi$ MPS into linear-depth isometric unitary staircase circuits (`src/quantum/circuits.py`) to avoid exponential Grover–Rudolph state-loading depth.

### Noisy Density-Matrix Simulation & Resource Estimation (`src/quantum/noise.py`, `src/quantum/resources.py`)
- Exact $256 \times 256$ (8-qubit) density-matrix simulation with depolarizing + readout noise calibrated to **AWS Braket** QPUs (`IQM Emerald`, `IonQ Forte`, `Early-FT 1e-4`, `QuEra Libra 2028`) and **Readout Mitigation + Exponential/Richardson ZNE**.
- Chakrabarti-style end-to-end resource estimator covering logical/physical qubits, QEC code distance, Megaquop budgets, and Braket simulator/QPU mapping.

## Quickstart

```bash
# From repository root (using shared .venv)
python3 -m venv .venv
.venv/bin/pip install -e ./adb-challenge-quantum-joint-loss-scenario-public-main[test]
.venv/bin/pip install -r quantum_prototype/requirements.txt

# 1. Verify official qjlb tests and team quantum tests
.venv/bin/pytest adb-challenge-quantum-joint-loss-scenario-public-main/tests/ -v
.venv/bin/pytest quantum_prototype/tests/ -v

# 2. Generate all benchmark CSVs, JSON resource tables, and 4 PNG figures
.venv/bin/python quantum_prototype/scripts/run_benchmarks.py --device cpu --repeats 6

# 3. On an NVIDIA A100 GPU node (install requirements-gpu.txt first)
python quantum_prototype/scripts/run_a100_experiments.py --repeats 10 --run-holdout
```

## Official Evaluation Harness Usage

```bash
.venv/bin/qjlb holdout --data adb-challenge-quantum-joint-loss-scenario-public-main/data \
  --output quantum_prototype/private-evaluation/suite --seed 2026

.venv/bin/qjlb evaluate quantum_prototype/private-evaluation/suite \
  --output quantum_prototype/private-evaluation/report-iqae \
  --execution quantum_simulator --trusted-code --repeats 10 --samples 8192 \
  -- "$PWD/.venv/bin/python" "$PWD/quantum_prototype/team_solver.py" --solver-mode iqae
```

## Data Attribution
World Bank SFRARR Central Asia seismic and flood Event Loss Tables (`seismic.npz`, `flood.npz`) are used unmodified from the official challenge data package under the **CC BY 4.0** license (World Bank / GFDRR SFRARR Central Asia Programme; see `adb-challenge-quantum-joint-loss-scenario-public-main/data/LICENSE.txt`).
