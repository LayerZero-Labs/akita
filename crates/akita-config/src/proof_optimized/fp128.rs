//! Default fp128 protocol presets on `p = 2^128 - 2^32 + 22537`
//! (`Prime128OffsetA7F7`).

use super::*;

/// Base field for the default fp128 presets.
pub type Field = Prime128OffsetA7F7;

/// Full-field `D=128` preset for planner-backed experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D128Full;

/// Full-field adaptive `D=64` preset.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64Full;

/// Binary onehot generated `D=64` preset.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64OneHot;

/// Binary onehot `D=64` preset with the tiered commitment matrix `F`.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64OneHotTiered;

/// Full-field adaptive `D=32` preset.
#[derive(Clone, Copy, Debug, Default)]
pub struct D32Full;

/// Onehot adaptive `D=32` preset.
#[derive(Clone, Copy, Debug, Default)]
pub struct D32OneHot;

/// Binary onehot `D=128` preset for planner-backed experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D128OneHot;

/// Multi-chunk (distributed-prover) companion of [`D64OneHot`]. Shares every
/// layout parameter with its sibling but prices the chunked witness layout.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64OneHotMultiChunk;

/// Multi-chunk (distributed-prover) companion of [`D64Full`].
#[derive(Clone, Copy, Debug, Default)]
pub struct D64FullMultiChunk;

impl_proof_optimized_preset!(
    D128Full,
    Field,
    Field,
    akita_types::SisModulusFamily::Q128,
    128,
    128,
    128,
    schedules = (
        "schedules-fp128-d128-full",
        "fp128_d128_full",
        fp128_d128_full_table
    )
);
impl_proof_optimized_preset!(
    D128OneHot,
    Field,
    Field,
    akita_types::SisModulusFamily::Q128,
    128,
    128,
    1,
    schedules = (
        "schedules-fp128-d128-onehot",
        "fp128_d128_onehot",
        fp128_d128_onehot_table
    )
);
impl_proof_optimized_preset!(
    D64Full,
    Field,
    Field,
    akita_types::SisModulusFamily::Q128,
    64,
    128,
    128,
    schedules = (
        "schedules-fp128-d64-full",
        "fp128_d64_full",
        fp128_d64_full_table
    )
);
impl_proof_optimized_preset!(
    D64OneHot,
    Field,
    Field,
    akita_types::SisModulusFamily::Q128,
    64,
    128,
    1,
    256,
    schedules = (
        "schedules-fp128-d64-onehot",
        "fp128_d64_onehot",
        fp128_d64_onehot_table
    )
);
impl_proof_optimized_preset!(
    D64OneHotTiered,
    Field,
    Field,
    akita_types::SisModulusFamily::Q128,
    64,
    128,
    1,
    256,
    true,
    schedules = (
        "schedules-fp128-d64-onehot-tiered",
        "fp128_d64_onehot_tiered",
        fp128_d64_onehot_tiered_table
    )
);
impl_multi_chunk_companion!(
    D64OneHotMultiChunk,
    D64OneHot,
    "schedules-fp128-d64-onehot-multi-chunk",
    fp128_d64_onehot_multi_chunk_table
);
impl_multi_chunk_companion!(
    D64FullMultiChunk,
    D64Full,
    "schedules-fp128-d64-full-multi-chunk",
    fp128_d64_full_multi_chunk_table
);
impl_proof_optimized_preset!(
    D32Full,
    Field,
    Field,
    akita_types::SisModulusFamily::Q128,
    32,
    128,
    128
);
impl_proof_optimized_preset!(
    D32OneHot,
    Field,
    Field,
    akita_types::SisModulusFamily::Q128,
    32,
    128,
    1
);

/// Concrete fp128 preset selected by a schedule-family query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp128Preset {
    /// Full-field adaptive `D=32` preset.
    D32Full,
    /// Full-field adaptive `D=64` preset.
    D64Full,
    /// Full-field `D=128` preset (comparison / legacy; D32/D64 are smaller under
    /// committed-fold A-role pricing).
    D128Full,
    /// Onehot adaptive `D=32` preset.
    D32OneHot,
    /// Binary onehot generated `D=64` preset.
    D64OneHot,
    /// Binary onehot `D=128` preset (comparison / legacy; D32/D64 are smaller under
    /// committed-fold A-role pricing).
    D128OneHot,
}

impl Fp128Preset {
    /// Ring dimension used by this preset.
    pub const fn ring_dimension(self) -> usize {
        match self {
            Self::D32Full | Self::D32OneHot => 32,
            Self::D64Full | Self::D64OneHot => 64,
            Self::D128Full | Self::D128OneHot => 128,
        }
    }

    /// Whether this preset is onehot-oriented.
    pub const fn is_onehot(self) -> bool {
        matches!(self, Self::D32OneHot | Self::D64OneHot | Self::D128OneHot)
    }

    /// Stable human-readable preset name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::D32Full => "D32Full",
            Self::D64Full => "D64Full",
            Self::D128Full => "D128Full",
            Self::D32OneHot => "D32OneHot",
            Self::D64OneHot => "D64OneHot",
            Self::D128OneHot => "D128OneHot",
        }
    }
}

/// Best generated schedule for one fp128 preset family.
#[derive(Clone, Debug)]
pub struct Fp128ScheduleSelection {
    /// Selected concrete preset.
    pub preset: Fp128Preset,
    /// Runtime schedule selected for the supplied lookup key.
    pub schedule: Schedule,
}

fn candidate<Cfg: CommitmentConfig>(
    preset: Fp128Preset,
    key: AkitaScheduleLookupKey,
) -> Result<Option<Fp128ScheduleSelection>, AkitaError> {
    // A genuine planner failure (invalid key shape, witness overflow,
    // SIS-floor gap) propagates rather than being swallowed into a missing
    // candidate. For any valid key the DP always yields a schedule (it falls
    // back to a root-direct cleartext schedule), so no preset is silently
    // dropped — the caller only ever sees `Err` on a real error.
    let schedule = Cfg::runtime_schedule(key)?;
    Ok(Some(Fp128ScheduleSelection { preset, schedule }))
}

fn best_by_exact_bytes<I>(candidates: I) -> Option<Fp128ScheduleSelection>
where
    I: IntoIterator<Item = Option<Fp128ScheduleSelection>>,
{
    candidates.into_iter().flatten().min_by_key(|selection| {
        (
            selection.schedule.total_bytes,
            selection.preset.ring_dimension(),
        )
    })
}

/// Select the best full-field fp128 preset for a schedule lookup key.
///
/// The key carries singleton and grouped batch shape data, so
/// this helper can be used by profile tooling without manually comparing
/// typed preset schedule tables. A genuine planner failure propagates as an
/// error; for any valid key every preset yields a schedule (the DP falls back
/// to a root-direct cleartext schedule), so the best one is always returned.
///
/// # Errors
///
/// Propagates a planner / runtime-schedule failure (invalid key shape,
/// witness overflow, or an uncovered SIS-floor width).
pub fn best_full_schedule(
    key: AkitaScheduleLookupKey,
) -> Result<Option<Fp128ScheduleSelection>, AkitaError> {
    Ok(best_by_exact_bytes([
        candidate::<D32Full>(Fp128Preset::D32Full, key)?,
        candidate::<D64Full>(Fp128Preset::D64Full, key)?,
        candidate::<D128Full>(Fp128Preset::D128Full, key)?,
    ]))
}

/// Select the best onehot fp128 preset for a schedule lookup key.
///
/// A genuine planner failure propagates as an error; for any valid key every
/// preset yields a schedule, so the best one is always returned.
///
/// # Errors
///
/// Propagates a planner / runtime-schedule failure (invalid key shape,
/// witness overflow, or an uncovered SIS-floor width).
pub fn best_onehot_schedule(
    key: AkitaScheduleLookupKey,
) -> Result<Option<Fp128ScheduleSelection>, AkitaError> {
    Ok(best_by_exact_bytes([
        candidate::<D32OneHot>(Fp128Preset::D32OneHot, key)?,
        candidate::<D64OneHot>(Fp128Preset::D64OneHot, key)?,
        candidate::<D128OneHot>(Fp128Preset::D128OneHot, key)?,
    ]))
}
