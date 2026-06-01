//! fp32 presets used for small-field integration and profiling.

use super::*;

/// Base field for the fp32 scaffold presets.
pub type Field = Prime32Offset99;
/// ring-subfield used for fp32 public claims and Fiat-Shamir challenges.
pub type ExtensionField = RingSubfieldFp4<Field>;

/// Full-field `D=32` preset for the default fp32 schedule path.
#[derive(Clone, Copy, Debug, Default)]
pub struct D32Full;

/// Onehot `D=32` preset for the default fp32 schedule path.
#[derive(Clone, Copy, Debug, Default)]
pub struct D32OneHot;

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

/// Full-field `D=512` preset for planner-backed fp32 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D512Full;

/// Onehot `D=512` preset for planner-backed fp32 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D512OneHot;

impl_small_field_preset!(
    D32Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    32,
    32,
    32,
    3,
    8,
    vec![-1, 1]
);
impl_small_field_preset!(
    D32OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    32,
    32,
    1,
    3,
    8,
    vec![-1, 1]
);
impl_small_field_preset!(
    D64Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    64,
    32,
    32,
    3,
    8,
    vec![-1, 1]
);
impl_small_field_preset!(
    D64OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    64,
    32,
    1,
    3,
    8,
    vec![-1, 1]
);
impl_small_field_preset!(
    D128Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    128,
    32,
    32,
    3,
    8,
    vec![-1, 1]
);
impl_small_field_preset!(
    D128OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    128,
    32,
    1,
    3,
    8,
    vec![-1, 1]
);
impl_small_field_preset!(
    D256Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    256,
    32,
    32,
    3,
    8,
    vec![-1, 1]
);
impl_small_field_preset!(
    D256OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    256,
    32,
    1,
    3,
    8,
    vec![-1, 1]
);
impl_small_field_preset!(
    D512Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    512,
    32,
    32,
    3,
    8,
    vec![-1, 1]
);
impl_small_field_preset!(
    D512OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    512,
    32,
    1,
    3,
    8,
    vec![-1, 1]
);
