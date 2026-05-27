use super::*;

/// Base field for the fp16 presets.
pub type Field = Prime16Offset99;
/// Degree-8 ring-subfield used for fp16 public claims and Fiat-Shamir challenges.
pub type ExtensionField = RingSubfieldFp8<Field>;

/// Full-field `D=32` preset for fp16 production profiling.
#[derive(Clone, Copy, Debug, Default)]
pub struct D32Full;

/// Onehot `D=32` preset for fp16 production profiling.
#[derive(Clone, Copy, Debug, Default)]
pub struct D32OneHot;

/// Full-field `D=64` preset for fp16 comparison profiling.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64Full;

/// Onehot `D=64` preset for fp16 comparison profiling.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64OneHot;

/// Full-field `D=128` preset for planner-backed fp16 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D128Full;

/// Onehot `D=128` preset for planner-backed fp16 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D128OneHot;

/// Full-field `D=256` preset for planner-backed fp16 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D256Full;

/// Onehot `D=256` preset for planner-backed fp16 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D256OneHot;

/// Full-field `D=512` preset for planner-backed fp16 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D512Full;

/// Onehot `D=512` preset for planner-backed fp16 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D512OneHot;

impl_small_field_preset!(
    D32Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q16,
    32,
    16,
    16,
    3,
    8,
    vec![-1, 1],
    Some(akita_types::generated::fp16_d32_full_table())
);
impl_small_field_preset!(
    D32OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q16,
    32,
    16,
    1,
    3,
    8,
    vec![-1, 1],
    Some(akita_types::generated::fp16_d32_onehot_table())
);
impl_small_field_preset!(
    D64Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q16,
    64,
    16,
    16,
    3,
    8,
    vec![-1, 1],
    Some(akita_types::generated::fp16_d64_full_table())
);
impl_small_field_preset!(
    D64OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q16,
    64,
    16,
    1,
    3,
    8,
    vec![-1, 1],
    Some(akita_types::generated::fp16_d64_onehot_table())
);
impl_small_field_preset!(
    D128Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q16,
    128,
    16,
    16,
    3,
    8,
    vec![-1, 1],
    None
);
impl_small_field_preset!(
    D128OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q16,
    128,
    16,
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
    akita_types::SisModulusFamily::Q16,
    256,
    16,
    16,
    3,
    8,
    vec![-1, 1],
    None
);
impl_small_field_preset!(
    D256OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q16,
    256,
    16,
    1,
    3,
    8,
    vec![-1, 1],
    None
);
impl_small_field_preset!(
    D512Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q16,
    512,
    16,
    16,
    3,
    8,
    vec![-1, 1],
    None
);
impl_small_field_preset!(
    D512OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q16,
    512,
    16,
    1,
    3,
    8,
    vec![-1, 1],
    None
);
