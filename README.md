# ADB Quantum Innovation Challenge: Joint-Loss Modelling for Disaster Risk Financing

[![Phase 1 Submission](https://img.shields.io/badge/ADB%20Quantum%20Challenge-Phase%201-blue)](#)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](#)
[![Approach](https://img.shields.io/badge/Approach-IQAE%20%2B%20MPS-purple)](#)

## 1. Overview

This repository contains our team's Phase 1 prototype, benchmarking suite, and technical documentation for the **ADB Quantum Innovation Challenge: Joint-Loss Modelling for Disaster Risk Financing**, hosted by the Asian Development Bank (ADB), QAI Ventures, and AWS.

Parametric catastrophe bonds—such as ADB's US\$160 million Disaster Relief Bond covering the **Kyrgyz Republic and Tajikistan**—rely on accurate estimation of rare, cross-border **joint tail losses**. Classical approaches face two fundamental bottlenecks:
1. **Rare-Tail Sampling Error:** Standard Monte Carlo error decays as $O(1/\sqrt{M})$, making ultra-rare joint events (such as both bonds exhausting at $0.0032\%$) expensive to resolve accurately.
2. **Exponential State-Space Growth:** Exact enumeration scales as $10^n$ for $n$ countries ($100$ states for $n=2$, $100{,}000$ for $n=5$), quickly becoming intractable for regional multi-country risk pools.

---

## 2. Our Dual-Track Architecture: IQAE + MPS

To address both bottlenecks and provide rigorous **quantum-vs-classical** and **quantum-vs-quantum** benchmarks, we implement a complementary two-track framework:

### Track A: Iterative Quantum Amplitude Estimation (IQAE)
* **Target Bottleneck:** Rare joint-tail sampling error.
* **Method:** Encodes the capped 3-year joint distribution $p(x)$ into $\lceil \log_2 100 \rceil = 7$ state qubits plus objective/ancilla qubits, applies reversible threshold oracles and payoff rotations $R_y(2\arcsin(\sqrt{\tilde{f}(x)}))$, and estimates all five focus targets via **IQAE** (eliminating QPE overhead) for a theoretical $O(1/M)$ query scaling.

### Track B: Matrix Product State (MPS) Tensor Network
* **Target Bottleneck:** Exponential $10^n$ multi-country state-space scaling.
* **Method:** Exploits localized geographic correlations (shared fault lines and river/glacier basins) to represent $n$-country joint distributions as a low-bond-dimension MPS.
* **Dual Role:**
  1. **Quantum-Inspired Solver:** Direct, sampling-noise-free contraction of tail metrics for $n \ge 5$ countries (scalable on Phase 2 QAI Ventures GPUs and AWS Braket `TN1`).
  2. **Shallow State-Prep Front-End:** Compiles the MPS into a linear-depth unitary staircase for fault-tolerant IQAE state loading.

---

## 3. Benchmark Targets (Kyrgyz Republic & Tajikistan, 3-Year Term)

Every target takes the form $\mu = \mathbb{E}_p[f(c_1(X), \dots, c_n(X))]$ with threshold $T = \text{USD } 80\text{m}$:

| Focus Target | Definition | Exact Reference (`adb-base`) |
| :--- | :--- | :--- |
| `term_expected_principal_loss_usd` | $\mathbb{E}[L]$ | USD 10.06m |
| `term_joint_attachment_probability` | $\Pr(L_1 > 0 \text{ and } L_2 > 0)$ | $4.414\%$ |
| `term_portfolio_loss_exceedance_probability` | $\Pr(L > T)$ | $1.205\%$ |
| `term_portfolio_stop_loss_usd` | $\mathbb{E}[\max(L - T, 0)]$ | USD 282,355 |
| `term_both_bonds_exhausted_probability` | $\Pr(L_1 = 80\text{m} \text{ and } L_2 = 80\text{m})$ | $0.0032\%$ |

All quantum and quantum-inspired estimators are benchmarked against the four classical baselines (**Monte Carlo**, **Scrambled Sobol**, **Stratified Sampling**, and **Importance Sampling**) across all five ADB scenarios (`adb-base`, `adb-independent`, `adb-common-shock`, `adb-rain-stress`, `adb-rain-cover`) as well as the regional scenarios (`kaz-kgz-base`, `tjk-uzb-base`).

---

## 4. Repository Structure

```text
.
├── README.md                    # Project overview and quickstart
├── CHALLENGE_CONTEXT.md         # Full math spec, rubric, and architecture notes
├── EVALUATION.md                # Official ADB benchmark evaluation rules
├── pyproject.toml               # Package configuration for qjlb
├── team_solver.py               # Main team solver entrypoint for the benchmark harness
├── src/
│   ├── qjlb/                    # Official ADB benchmark library (bond, model, evaluation)
│   └── quantum/                 # [Team] IQAE circuits, MPS contraction, and resource estimator
├── scenarios/                   # JSON scenario definitions (ADB + regional pairs)
├── data/                        # Hazard datasets (under Fair Data Use Agreement)
├── notebooks/                   # Prototype experiments and accuracy-vs-query plots
├── docs/                        # Scientific proposal, resource tables, and Phase 2 plan
└── tests/                       # Unit tests for baseline, bond, evaluation, and team solver
```

---

## 5. Quickstart & Environment Setup

### Prerequisites
* Python 3.10+
* Recommended quantum/tensor packages: `qiskit` (or `pennylane` / `amazon-braket-sdk`), `numpy`, `scipy`, `pytest`

### Installation
```bash
# 1. Clone the repository
git clone <YOUR_GITHUB_REPO_URL>
cd <YOUR_REPO_FOLDER>

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install the official benchmark package in editable mode
pip install -e .

# 4. Install team prototype dependencies (once requirements.txt is added)
# pip install -r requirements.txt
```

### Verify the Benchmark Harness
```bash
# Run the official test suite
pytest tests/

# Run the example benchmark script
python example.py
```

---

## 6. Resource Estimation & Hardware Mapping

Our resource analysis evaluates both near-term and fault-tolerant execution on **Amazon Braket**:
* **Simulators:** `SV1` (exact state-vector), `TN1` (tensor-network validation), and `DM1` (noisy density-matrix + error mitigation).
* **Near-Term Gate-Based QPUs:** IonQ Forte / Forte Enterprise (36q), IQM Emerald (54q).
* **Fault-Tolerant Target (2028):** QuEra **Libra** (Megaquop-scale neutral-atom architecture targeting $\sim 10^6$ operations across hundreds of logical qubits).

---

## 7. Team

| Member | Role | Focus Area |
| :--- | :--- | :--- |
| **Member 1** (`@handle`) | Classical & Benchmark Lead | `qjlb` harness validation, classical baselines, regional scenarios |
| **Member 2** (`@handle`) | Quantum Algorithms Lead | Track A: IQAE circuits, oracles, and `team_solver.py` |
| **Member 3** (`@handle`) | Scaling & Resource Lead | Track B: $n$-country MPS scaling, gate/depth/QEC resource estimation |
| **Member 4** (`@handle`) | Scientific Proposal Lead | Mathematical formulation, literature review, technical report |
| **Member 5** (`@handle`) | Systems, QA & Phase 2 Lead | Phase 2 roadmap, risk/fallback matrix, packaging & reproducibility |

---

## 8. AI Use Disclosure

*(Required by ADB Phase 1 Submission Guidelines — update before final submission)*

* **Tools Used:** `[e.g., Gemini / Antigravity IDE]`
* **Scope of Assistance:** Literature organization, code scaffolding, documentation drafting, and debugging assistance. All mathematical formulations, quantum circuit designs, benchmark runs, and resource estimates were verified by the team against the official `qjlb` evaluation suite.

---

## 9. License & Data Usage

* Source code in this repository is governed by the repository's `LICENSE` file.
* Benchmark hazard datasets inside `data/` (`seismic.npz`, `flood.npz`, etc.) are subject to the ADB Challenge **Fair Data Use Agreement** (see `data/LICENSE.txt`). Do not redistribute restricted challenge data outside permitted terms.