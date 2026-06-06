//! fp32 presets used for small-field integration and profiling.

use super::*;

/// Base field for the fp32 scaffold presets.
pub type Field = Prime32Offset99;
/// ring-subfield used for fp32 public claims and Fiat-Shamir challenges.
pub type ExtensionField = RingSubfieldFpExt4<Field>;

/// Full-field `D=64` preset for fp32 crossover profiling.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64Full;

/// Onehot `D=64` preset for fp32 crossover profiling.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64OneHot;

/// Full-field `D=128` preset for planner-backed fp32 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D128Full;

/// Onehot `D=128` preset for planner-backed fp32 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D128OneHot;

/// Full-field `D=256` preset for planner-backed fp32 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D256Full;

/// Onehot `D=256` preset for planner-backed fp32 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D256OneHot;

impl_proof_optimized_preset!(
    D64Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    64,
    32,
    32
);
impl_proof_optimized_preset!(
    D64OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    64,
    32,
    1
);
impl_proof_optimized_preset!(
    D128Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    128,
    32,
    32
);
impl_proof_optimized_preset!(
    D128OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    128,
    32,
    1
);
impl_proof_optimized_preset!(
    D256Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    256,
    32,
    32
);
impl_proof_optimized_preset!(
    D256OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    256,
    32,
    1
);
