//! fp32 presets used for small-field integration and profiling.

use super::*;

/// Base field for the fp32 scaffold presets.
pub type Field = Prime32Offset99;
/// Akita's degree-4 extension for fp32 public claims and Fiat-Shamir challenges.
pub type ExtensionField = FpExt4<Field>;

/// Dense `D=64` preset for fp32 crossover profiling.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64Dense;

/// Onehot `D=64` preset for fp32 crossover profiling.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64OneHot;

/// Dense `D=128` preset for planner-backed fp32 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D128Dense;

/// Onehot `D=128` preset for planner-backed fp32 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D128OneHot;

/// Dense `D=256` preset for planner-backed fp32 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D256Dense;

/// Onehot `D=256` preset for planner-backed fp32 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D256OneHot;

impl_proof_optimized_preset!(
    D64Dense,
    Field,
    ExtensionField,
    akita_types::SisModulusProfileId::Q32Offset99,
    64,
    32,
    32
);
impl_proof_optimized_preset!(
    D64OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusProfileId::Q32Offset99,
    64,
    32,
    1
);
impl_proof_optimized_preset!(
    D128Dense,
    Field,
    ExtensionField,
    akita_types::SisModulusProfileId::Q32Offset99,
    128,
    32,
    32
);
impl_proof_optimized_preset!(
    D128OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusProfileId::Q32Offset99,
    128,
    32,
    1,
    schedules = (
        "schedules-fp32-d128-onehot",
        "fp32_d128_onehot",
        fp32_d128_onehot_table
    )
);
impl_proof_optimized_preset!(
    D256Dense,
    Field,
    ExtensionField,
    akita_types::SisModulusProfileId::Q32Offset99,
    256,
    32,
    32
);
impl_proof_optimized_preset!(
    D256OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusProfileId::Q32Offset99,
    256,
    32,
    1,
    schedules = (
        "schedules-fp32-d256-onehot",
        "fp32_d256_onehot",
        fp32_d256_onehot_table
    )
);
