//! fp64 presets used for small-field integration and profiling.

use super::*;

/// Base field for the fp64 scaffold presets.
pub type Field = Prime64Offset59;
/// ring-subfield used for fp64 public claims and Fiat-Shamir challenges.
pub type ExtensionField = Ext2<Field>;

const SUFFIX_RING_DIMENSIONS: &[usize] = &[64];
const A_RING_DIMENSIONS: &[usize] = &[64, 128, 256, 512, 1024];
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
    fold_norms = akita_types::sis::FoldWitnessNorms::bounded(3, 64),
    schedules = ("schedules-fp64-dense", "fp64_dense", fp64_dense_table),
    ring_dimension_schedule_mode = ADAPTIVE_RING_DIMENSION_MODE
);
impl_proof_optimized_preset!(
    OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusProfileId::Q64Offset59,
    64,
    1,
    fold_norms = akita_types::sis::FoldWitnessNorms::new(1, 1),
    schedules = ("schedules-fp64-onehot", "fp64_onehot", fp64_onehot_table),
    ring_dimension_schedule_mode = ADAPTIVE_RING_DIMENSION_MODE
);
