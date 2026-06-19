//! fp64 presets used for small-field integration and profiling.

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

impl_proof_optimized_preset!(
    D32Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
    32,
    64,
    64
);
impl_proof_optimized_preset!(
    D32OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
    32,
    64,
    1
);
impl_proof_optimized_preset!(
    D64Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
    64,
    64,
    64
);
impl_proof_optimized_preset!(
    D64OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
    64,
    64,
    1
);
impl_proof_optimized_preset!(
    D128Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
    128,
    64,
    64,
    schedules = ("schedules-fp64-d128", "fp64_d128", fp64_d128_table)
);
impl_proof_optimized_preset!(
    D128OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
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
    D256Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
    256,
    64,
    64
);
impl_proof_optimized_preset!(
    D256OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
    256,
    64,
    1,
    schedules = (
        "schedules-fp64-d256-onehot",
        "fp64_d256_onehot",
        fp64_d256_onehot_table
    )
);

/// Concrete fp64 onehot preset selected by a schedule-family query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp64OneHotPreset {
    D32OneHot,
    D64OneHot,
    D128OneHot,
    D256OneHot,
}

impl Fp64OneHotPreset {
    pub const fn ring_dimension(self) -> usize {
        match self {
            Self::D32OneHot => 32,
            Self::D64OneHot => 64,
            Self::D128OneHot => 128,
            Self::D256OneHot => 256,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::D32OneHot => "D32OneHot",
            Self::D64OneHot => "D64OneHot",
            Self::D128OneHot => "D128OneHot",
            Self::D256OneHot => "D256OneHot",
        }
    }
}

/// Concrete fp64 full-field preset selected by a schedule-family query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp64FullPreset {
    D32Full,
    D64Full,
    D128Full,
    D256Full,
}

impl Fp64FullPreset {
    pub const fn ring_dimension(self) -> usize {
        match self {
            Self::D32Full => 32,
            Self::D64Full => 64,
            Self::D128Full => 128,
            Self::D256Full => 256,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::D32Full => "D32Full",
            Self::D64Full => "D64Full",
            Self::D128Full => "D128Full",
            Self::D256Full => "D256Full",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Fp64OneHotScheduleSelection {
    pub preset: Fp64OneHotPreset,
    pub schedule: akita_types::Schedule,
}

#[derive(Clone, Debug)]
pub struct Fp64FullScheduleSelection {
    pub preset: Fp64FullPreset,
    pub schedule: akita_types::Schedule,
}

fn fp64_onehot_candidate<Cfg: CommitmentConfig>(
    preset: Fp64OneHotPreset,
    key: akita_types::AkitaScheduleLookupKey,
) -> Option<Fp64OneHotScheduleSelection> {
    let schedule = Cfg::runtime_schedule(key).ok()?;
    Some(Fp64OneHotScheduleSelection { preset, schedule })
}

fn fp64_full_candidate<Cfg: CommitmentConfig>(
    preset: Fp64FullPreset,
    key: akita_types::AkitaScheduleLookupKey,
) -> Option<Fp64FullScheduleSelection> {
    let schedule = Cfg::runtime_schedule(key).ok()?;
    Some(Fp64FullScheduleSelection { preset, schedule })
}

/// Select the best fp64 onehot preset for a schedule lookup key.
pub fn best_onehot_schedule(
    key: akita_types::AkitaScheduleLookupKey,
) -> Result<Option<Fp64OneHotScheduleSelection>, akita_field::AkitaError> {
    Ok([
        fp64_onehot_candidate::<D32OneHot>(Fp64OneHotPreset::D32OneHot, key),
        fp64_onehot_candidate::<D64OneHot>(Fp64OneHotPreset::D64OneHot, key),
        fp64_onehot_candidate::<D128OneHot>(Fp64OneHotPreset::D128OneHot, key),
        fp64_onehot_candidate::<D256OneHot>(Fp64OneHotPreset::D256OneHot, key),
    ]
    .into_iter()
    .flatten()
    .min_by_key(|selection| {
        (
            selection.schedule.total_bytes,
            selection.preset.ring_dimension(),
        )
    }))
}

/// Select the best fp64 full-field preset for a schedule lookup key.
pub fn best_full_schedule(
    key: akita_types::AkitaScheduleLookupKey,
) -> Result<Option<Fp64FullScheduleSelection>, akita_field::AkitaError> {
    Ok([
        fp64_full_candidate::<D32Full>(Fp64FullPreset::D32Full, key),
        fp64_full_candidate::<D64Full>(Fp64FullPreset::D64Full, key),
        fp64_full_candidate::<D128Full>(Fp64FullPreset::D128Full, key),
        fp64_full_candidate::<D256Full>(Fp64FullPreset::D256Full, key),
    ]
    .into_iter()
    .flatten()
    .min_by_key(|selection| {
        (
            selection.schedule.total_bytes,
            selection.preset.ring_dimension(),
        )
    }))
}
