//! fp64 presets used for small-field integration and profiling.

use super::*;

/// Base field for the fp64 scaffold presets.
pub type Field = Prime64Offset59;
/// ring-subfield used for fp64 public claims and Fiat-Shamir challenges.
pub type ExtensionField = Ext2<Field>;

/// Default adaptive dense preset for fp64.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dense;

impl Dense {
    pub const SUFFIX_RING_DIMENSIONS: [usize; 1] = [64];
    pub const A_RING_DIMENSIONS: [usize; 4] = [64, 128, 256, 512];
    pub const B_RING_DIMENSIONS: [usize; 3] = [64, 128, 256];
    pub const D_RING_DIMENSIONS: [usize; 3] = [64, 128, 256];
}

/// Default adaptive one-hot preset for fp64.
#[derive(Clone, Copy, Debug, Default)]
pub struct OneHot;

impl OneHot {
    pub const SUFFIX_RING_DIMENSIONS: [usize; 1] = [64];
    pub const A_RING_DIMENSIONS: [usize; 4] = [64, 128, 256, 512];
    pub const B_RING_DIMENSIONS: [usize; 3] = [64, 128, 256];
    pub const D_RING_DIMENSIONS: [usize; 3] = [64, 128, 256];
}

impl_proof_optimized_preset!(
    Dense,
    Field,
    ExtensionField,
    akita_types::SisModulusProfileId::Q64Offset59,
    256,
    64,
    64,
    fold_norms = akita_types::sis::FoldWitnessNorms::bounded(3, 64),
    schedules = ("schedules-fp64-dense", "fp64_dense", fp64_dense_table),
    ring_dimension_schedule_mode = akita_schedules::RingDimensionScheduleMode::AdaptiveDimension {
        num_search_levels: akita_schedules::ADAPTIVE_SEARCH_LEVELS,
        suffix_dimensions: &Dense::SUFFIX_RING_DIMENSIONS,
        potential_a_dimensions: &Dense::A_RING_DIMENSIONS,
        potential_b_dimensions: &Dense::B_RING_DIMENSIONS,
        potential_d_dimensions: &Dense::D_RING_DIMENSIONS,
    }
);
impl_proof_optimized_preset!(
    OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusProfileId::Q64Offset59,
    256,
    64,
    1,
    fold_norms = akita_types::sis::FoldWitnessNorms::new(1, 1),
    schedules = ("schedules-fp64-onehot", "fp64_onehot", fp64_onehot_table),
    ring_dimension_schedule_mode = akita_schedules::RingDimensionScheduleMode::AdaptiveDimension {
        num_search_levels: akita_schedules::ADAPTIVE_SEARCH_LEVELS,
        suffix_dimensions: &OneHot::SUFFIX_RING_DIMENSIONS,
        potential_a_dimensions: &OneHot::A_RING_DIMENSIONS,
        potential_b_dimensions: &OneHot::B_RING_DIMENSIONS,
        potential_d_dimensions: &OneHot::D_RING_DIMENSIONS,
    }
);
