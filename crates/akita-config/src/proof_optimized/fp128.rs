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

/// Full-field adaptive `D=32` preset.
#[derive(Clone, Copy, Debug, Default)]
pub struct D32Full;

/// Onehot adaptive `D=32` preset.
#[derive(Clone, Copy, Debug, Default)]
pub struct D32OneHot;

/// Binary onehot `D=128` preset for planner-backed experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D128OneHot;

impl_fp128_preset!(D128Full, 128, 128, None);
impl_fp128_preset!(D128OneHot, 128, 1, None);
impl_fp128_preset!(
    D64Full,
    64,
    128,
    Some(akita_types::generated::fp128_d64_full_table())
);
impl_fp128_preset!(
    D64OneHot,
    64,
    1,
    Some(akita_types::generated::fp128_d64_onehot_table())
);
impl_fp128_preset!(
    D32Full,
    32,
    128,
    Some(akita_types::generated::fp128_d32_full_table())
);
impl_fp128_preset!(
    D32OneHot,
    32,
    1,
    Some(akita_types::generated::fp128_d32_onehot_table())
);

/// Concrete fp128 preset selected by a schedule-family query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp128Preset {
    /// Full-field adaptive `D=32` preset.
    D32Full,
    /// Full-field adaptive `D=64` preset.
    D64Full,
    /// Onehot adaptive `D=32` preset.
    D32OneHot,
    /// Binary onehot generated `D=64` preset.
    D64OneHot,
}

impl Fp128Preset {
    /// Ring dimension used by this preset.
    pub const fn ring_dimension(self) -> usize {
        match self {
            Self::D32Full | Self::D32OneHot => 32,
            Self::D64Full | Self::D64OneHot => 64,
        }
    }

    /// Whether this preset is onehot-oriented.
    pub const fn is_onehot(self) -> bool {
        matches!(self, Self::D32OneHot | Self::D64OneHot)
    }

    /// Stable human-readable preset name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::D32Full => "D32Full",
            Self::D64Full => "D64Full",
            Self::D32OneHot => "D32OneHot",
            Self::D64OneHot => "D64OneHot",
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
    Ok(Cfg::runtime_schedule(key)?.map(|schedule| Fp128ScheduleSelection { preset, schedule }))
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
/// The key carries singleton, grouped, and multipoint batch shape data, so
/// this helper can be used by profile tooling without manually comparing
/// typed preset schedule tables. Missing generated rows are ignored; the
/// returned value is `None` only when no full-field preset has a generated
/// entry for the key.
///
/// # Errors
///
/// Returns an error if a generated table entry is malformed.
pub fn best_full_schedule(
    key: AkitaScheduleLookupKey,
) -> Result<Option<Fp128ScheduleSelection>, AkitaError> {
    Ok(best_by_exact_bytes([
        candidate::<D32Full>(Fp128Preset::D32Full, key)?,
        candidate::<D64Full>(Fp128Preset::D64Full, key)?,
    ]))
}

/// Select the best onehot fp128 preset for a schedule lookup key.
///
/// Missing generated rows are ignored; the returned value is `None` only
/// when no onehot preset has a generated entry for the key.
///
/// # Errors
///
/// Returns an error if a generated table entry is malformed.
pub fn best_onehot_schedule(
    key: AkitaScheduleLookupKey,
) -> Result<Option<Fp128ScheduleSelection>, AkitaError> {
    Ok(best_by_exact_bytes([
        candidate::<D32OneHot>(Fp128Preset::D32OneHot, key)?,
        candidate::<D64OneHot>(Fp128Preset::D64OneHot, key)?,
    ]))
}
