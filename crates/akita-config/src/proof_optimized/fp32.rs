//! fp32 presets used for small-field integration and profiling.

use super::*;

/// Base field for the fp32 scaffold presets.
pub type Field = Prime32Offset99;
/// Akita's degree-4 extension for fp32 public claims and Fiat-Shamir challenges.
pub type ExtensionField = FpExt4<Field>;

const SUFFIX_RING_DIMENSIONS: &[usize] = &[64, 128];
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

/// Default adaptive dense preset for fp32.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dense;

/// Default adaptive one-hot preset for fp32.
#[derive(Clone, Copy, Debug, Default)]
pub struct OneHot;

impl_proof_optimized_preset!(
    Dense,
    Field,
    ExtensionField,
    akita_types::SisModulusProfileId::Q32Offset99,
    32,
    32,
    source = balanced_digits,
    schedule_family = "fp32_dense",
    ring_dimension_schedule_mode = ADAPTIVE_RING_DIMENSION_MODE
);
impl_proof_optimized_preset!(
    OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusProfileId::Q32Offset99,
    32,
    1,
    source = unit_one_hot,
    schedule_family = "fp32_onehot",
    ring_dimension_schedule_mode = ADAPTIVE_RING_DIMENSION_MODE
);
