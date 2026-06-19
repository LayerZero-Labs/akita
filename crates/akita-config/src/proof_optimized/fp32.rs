//! fp32 presets used for small-field integration and profiling.

use super::*;

/// Base field for the fp32 scaffold presets.
pub type Field = Prime32Offset99;
/// Akita's degree-4 extension for fp32 public claims and Fiat-Shamir challenges.
pub type ExtensionField = FpExt4<Field>;

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
    1,
    schedules = (
        "schedules-fp32-d128-onehot",
        "fp32_d128_onehot",
        fp32_d128_onehot_table
    )
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
    1,
    schedules = (
        "schedules-fp32-d256-onehot",
        "fp32_d256_onehot",
        fp32_d256_onehot_table
    )
);

/// Default ladder ring degrees for fp32 onehot (high → low).
pub const FP32_ONEHOT_LADDER_DIMS: &[usize] = &[256, 128, 64];

/// Concrete fp32 onehot preset selected by a schedule-family query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp32OneHotPreset {
    D64OneHot,
    D128OneHot,
    D256OneHot,
}

impl Fp32OneHotPreset {
    pub const fn ring_dimension(self) -> usize {
        match self {
            Self::D64OneHot => 64,
            Self::D128OneHot => 128,
            Self::D256OneHot => 256,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::D64OneHot => "D64OneHot",
            Self::D128OneHot => "D128OneHot",
            Self::D256OneHot => "D256OneHot",
        }
    }
}

/// Best generated schedule across fp32 onehot presets.
#[derive(Clone, Debug)]
pub struct Fp32OneHotScheduleSelection {
    pub preset: Fp32OneHotPreset,
    pub schedule: akita_types::Schedule,
}

fn fp32_onehot_candidate<Cfg: CommitmentConfig>(
    preset: Fp32OneHotPreset,
    key: akita_types::AkitaScheduleLookupKey,
) -> Option<Fp32OneHotScheduleSelection> {
    let schedule = Cfg::runtime_schedule(key).ok()?;
    Some(Fp32OneHotScheduleSelection { preset, schedule })
}

/// Select the best fp32 onehot preset for a schedule lookup key.
///
/// Presets that fail SIS-floor / planner validation for the key are skipped.
pub fn best_onehot_schedule(
    key: akita_types::AkitaScheduleLookupKey,
) -> Result<Option<Fp32OneHotScheduleSelection>, akita_field::AkitaError> {
    Ok([
        fp32_onehot_candidate::<D64OneHot>(Fp32OneHotPreset::D64OneHot, key),
        fp32_onehot_candidate::<D128OneHot>(Fp32OneHotPreset::D128OneHot, key),
        fp32_onehot_candidate::<D256OneHot>(Fp32OneHotPreset::D256OneHot, key),
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
