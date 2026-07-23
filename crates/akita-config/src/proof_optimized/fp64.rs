//! fp64 presets used for small-field integration and profiling.

use super::*;

/// Base field for the fp64 scaffold presets.
pub type Field = Prime64Offset59;
/// ring-subfield used for fp64 public claims and Fiat-Shamir challenges.
pub type ExtensionField = Ext2<Field>;

/// Dense `D=64` preset.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64Dense;

/// Onehot `D=64` preset.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64OneHot;

/// Dense `D=128` preset for planner-backed fp64 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D128Dense;

/// Onehot `D=128` preset for planner-backed fp64 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D128OneHot;

/// Dense `D=256` preset for planner-backed fp64 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D256Dense;

/// Onehot `D=256` preset for planner-backed fp64 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D256OneHot;

impl_proof_optimized_preset!(
    D64Dense,
    Field,
    ExtensionField,
    akita_types::SisModulusProfileId::Q64Offset59,
    64,
    64,
    64
);
impl_proof_optimized_preset!(
    D64OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusProfileId::Q64Offset59,
    64,
    64,
    1
);
impl_proof_optimized_preset!(
    D128Dense,
    Field,
    ExtensionField,
    akita_types::SisModulusProfileId::Q64Offset59,
    128,
    64,
    64,
    schedules = (
        "schedules-fp64-d128-dense",
        "fp64_d128_dense",
        fp64_d128_dense_table
    )
);
impl_proof_optimized_preset!(
    D128OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusProfileId::Q64Offset59,
    128,
    64,
    1,
    schedules = (
        "schedules-fp64-d128-onehot",
        "fp64_d128_onehot",
        fp64_d128_onehot_table
    )
);
impl_proof_optimized_preset!(
    D256Dense,
    Field,
    ExtensionField,
    akita_types::SisModulusProfileId::Q64Offset59,
    256,
    64,
    64
);
impl_proof_optimized_preset!(
    D256OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusProfileId::Q64Offset59,
    256,
    64,
    1,
    schedules = (
        "schedules-fp64-d256-onehot",
        "fp64_d256_onehot",
        fp64_d256_onehot_table
    )
);
