use super::*;

/// Base field for the fp64 scaffold presets.
pub type Field = Prime64Offset59;
/// ring-subfield used for fp64 public claims and Fiat-Shamir challenges.
pub type ExtensionField = Ext2<Field>;

/// Full-field `D=32` preset for fp64 crossover profiling.
#[derive(Clone, Copy, Debug, Default)]
pub struct D32Full;

/// Onehot `D=32` preset for fp64 crossover profiling.
#[derive(Clone, Copy, Debug, Default)]
pub struct D32OneHot;

/// Full-field `D=64` preset.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64Full;

/// Onehot `D=64` preset.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64OneHot;

/// Full-field `D=128` preset for planner-backed fp64 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D128Full;

/// Onehot `D=128` preset for planner-backed fp64 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D128OneHot;

/// Full-field `D=256` preset for planner-backed fp64 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D256Full;

/// Onehot `D=256` preset for planner-backed fp64 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D256OneHot;

impl_small_field_preset!(
    D32Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
    32,
    64,
    64,
    3,
    8,
    vec![-1, 1],
    Some(akita_types::generated::fp64_d32_table())
);
impl_small_field_preset!(
    D32OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
    32,
    64,
    1,
    3,
    8,
    vec![-1, 1],
    Some(akita_types::generated::fp64_d32_onehot_table())
);
impl_small_field_preset!(
    D64Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
    64,
    64,
    64,
    3,
    8,
    vec![-1, 1],
    Some(akita_types::generated::fp64_d64_table())
);
impl_small_field_preset!(
    D64OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
    64,
    64,
    1,
    3,
    8,
    vec![-1, 1],
    Some(akita_types::generated::fp64_d64_onehot_table())
);
impl_small_field_preset!(
    D128Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
    128,
    64,
    64,
    3,
    8,
    vec![-1, 1],
    None
);
impl_small_field_preset!(
    D128OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
    128,
    64,
    1,
    3,
    8,
    vec![-1, 1],
    None
);
impl_small_field_preset!(
    D256Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
    256,
    64,
    64,
    3,
    8,
    vec![-1, 1],
    None
);
impl_small_field_preset!(
    D256OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
    256,
    64,
    1,
    3,
    8,
    vec![-1, 1],
    None
);
