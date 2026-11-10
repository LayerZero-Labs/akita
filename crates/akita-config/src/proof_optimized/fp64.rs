//! fp64 presets used for small-field integration and profiling.

use super::*;

/// Base field for the fp64 scaffold presets.
pub type Field = Prime64Offset59;
/// ring-subfield used for fp64 public claims and Fiat-Shamir challenges.
pub type ExtensionField = Ext2<Field>;

const SUFFIX_RING_DIMENSIONS: &[usize] = &[64, 128];
const A_RING_DIMENSIONS: &[usize] = &[64, 128, 256, 512, 1024, 2048];
const B_RING_DIMENSIONS: &[usize] = &[64, 128, 256];
const D_RING_DIMENSIONS: &[usize] = &[64, 128, 256];
const ADAPTIVE_RING_DIMENSION_SCHEDULE: akita_schedules::RingDimensionSchedule =
    akita_schedules::RingDimensionSchedule {
        num_search_levels: akita_schedules::ADAPTIVE_SEARCH_LEVELS,
        suffix_dimensions: SUFFIX_RING_DIMENSIONS,
        potential_a_dimensions: A_RING_DIMENSIONS,
        potential_b_dimensions: B_RING_DIMENSIONS,
        potential_d_dimensions: D_RING_DIMENSIONS,
    };

/// Ring dimensions for an fp64 suffix below a P64 binary mixed root.
///
/// A stops at the largest fp64 dimension within the root's native degree. B is
/// the 64-dimensional ordinary image the root authenticates. The planner needs
/// the B domain to contain every suffix dimension, so later folds also use 64,
/// as the fp128 `DenseBounded` preset does.
#[cfg(feature = "binary")]
pub const BINARY_MIXED_ROOT_RING_DIMENSION_SCHEDULE: akita_schedules::RingDimensionSchedule = {
    use akita_types::binary::{MIXED_ROOT_MAX_NATIVE_DEGREE, MIXED_ROOT_ORDINARY_B_DIMENSION};
    assert!(512 <= MIXED_ROOT_MAX_NATIVE_DEGREE && 1024 > MIXED_ROOT_MAX_NATIVE_DEGREE);
    akita_schedules::RingDimensionSchedule {
        potential_a_dimensions: &[64, 128, 256, 512],
        potential_b_dimensions: &[MIXED_ROOT_ORDINARY_B_DIMENSION],
        suffix_dimensions: &[MIXED_ROOT_ORDINARY_B_DIMENSION],
        ..ADAPTIVE_RING_DIMENSION_SCHEDULE
    }
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
    akita_params::SisModulusProfileId::Q64Offset59,
    64,
    64,
    source = balanced_digits,
    schedule_family = "fp64_dense",
    ring_dimension_schedule = ADAPTIVE_RING_DIMENSION_SCHEDULE
);
impl_proof_optimized_preset!(
    OneHot,
    Field,
    ExtensionField,
    akita_params::SisModulusProfileId::Q64Offset59,
    64,
    1,
    source = unit_one_hot,
    schedule_family = "fp64_onehot",
    ring_dimension_schedule = ADAPTIVE_RING_DIMENSION_SCHEDULE
);

impl crate::recursive_commitment::RecursiveScheduleConfig for Dense {
    const RECURSIVE_SCHEDULE_FAMILY_NAME: &'static str = "fp64_dense_recursive";
}
