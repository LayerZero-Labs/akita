//! fp64 presets used for small-field integration and profiling.

use super::*;

/// Base field for the fp64 scaffold presets.
pub type Field = Prime64Offset59;
/// ring-subfield used for fp64 public claims and Fiat-Shamir challenges.
pub type ExtensionField = Ext2<Field>;

const SUFFIX_RING_DIMENSIONS: &[usize] = &[64];
const A_RING_DIMENSIONS: &[usize] = &[64, 128, 256, 512, 1024, 2048];
const B_RING_DIMENSIONS: &[usize] = &[64, 128, 256];
const D_RING_DIMENSIONS: &[usize] = &[64, 128, 256];
const ADAPTIVE_RING_DIMENSION_MODE: akita_schedules::RingDimensionScheduleMode =
    akita_schedules::RingDimensionScheduleMode::AdaptiveDimension {
        num_search_levels: akita_schedules::ADAPTIVE_SEARCH_LEVELS,
        suffix_dimensions: SUFFIX_RING_DIMENSIONS,
        potential_a_dimensions: A_RING_DIMENSIONS,
        potential_b_dimensions: B_RING_DIMENSIONS,
        potential_d_dimensions: D_RING_DIMENSIONS,
    };

/// Default adaptive dense preset for fp64.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dense;

/// Default adaptive one-hot preset for fp64.
#[derive(Clone, Copy, Debug, Default)]
pub struct OneHot;

impl_proof_optimized_preset!(
    Dense,
    Field,
    ExtensionField,
    akita_types::SisModulusProfileId::Q64Offset59,
    64,
    64,
    source = balanced_digits,
    schedule_family = "fp64_dense",
    ring_dimension_schedule_mode = ADAPTIVE_RING_DIMENSION_MODE
);
impl_proof_optimized_preset!(
    OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusProfileId::Q64Offset59,
    64,
    1,
    source = unit_one_hot,
    schedule_family = "fp64_onehot",
    ring_dimension_schedule_mode = ADAPTIVE_RING_DIMENSION_MODE
);
