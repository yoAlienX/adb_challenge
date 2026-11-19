"""Dual-track quantum (IQAE) and tensor-network (MPS) framework for ADB joint-loss risk modelling."""

from .circuits import (
    CircuitArtifact,
    build_compact_bond_circuit,
    build_elt_uniform_circuit_spec,
    build_mps_staircase_circuit,
    build_path_bond_circuit_spec,
)
from .iqae import (
    IQAEResult,
    iqae_statevector_estimate,
    solve_bond_iqae_compact,
    solve_bond_iqae_path,
    solve_elt_iqae,
)
from .mps import (
    GenerativeMPS,
    JointLossMPS,
    MPSContractionResult,
    benchmark_generative_mps_worldbank,
    benchmark_mps_scaling,
    build_bond_mps_exact,
    build_synthetic_ncountry_mps,
    mps_to_staircase_unitaries,
    solve_bond_mps,
)
from .noise import (
    NoiseModelSpec,
    NoisyIQAEResult,
    simulate_noisy_iqae_8q,
    sweep_braket_qpu_noise,
)
from .resources import (
    ResourceBreakdown,
    estimate_all_resources,
    estimate_bond_resources,
    estimate_elt_resources,
    estimate_mps_scaling_resources,
)

__all__ = [
    "CircuitArtifact",
    "build_compact_bond_circuit",
    "build_elt_uniform_circuit_spec",
    "build_mps_staircase_circuit",
    "build_path_bond_circuit_spec",
    "IQAEResult",
    "iqae_statevector_estimate",
    "solve_bond_iqae_compact",
    "solve_bond_iqae_path",
    "solve_elt_iqae",
    "GenerativeMPS",
    "JointLossMPS",
    "MPSContractionResult",
    "benchmark_generative_mps_worldbank",
    "benchmark_mps_scaling",
    "build_bond_mps_exact",
    "build_synthetic_ncountry_mps",
    "mps_to_staircase_unitaries",
    "solve_bond_mps",
    "NoiseModelSpec",
    "NoisyIQAEResult",
    "simulate_noisy_iqae_8q",
    "sweep_braket_qpu_noise",
    "ResourceBreakdown",
    "estimate_all_resources",
    "estimate_bond_resources",
    "estimate_elt_resources",
    "estimate_mps_scaling_resources",
]
