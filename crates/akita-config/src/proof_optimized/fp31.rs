//! fp31 presets used for small-field integration and profiling.

use super::*;

/// Base field for the fp31 presets.
pub type Field = Prime31Offset19;
/// Akita's degree-4 extension for fp31 public claims and Fiat-Shamir challenges.
pub type ExtensionField = FpExt4<Field>;

const SUFFIX_RING_DIMENSIONS: &[usize] = &[64, 128];
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

/// Default adaptive dense preset for fp31.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dense;

/// Default adaptive one-hot preset for fp31.
#[derive(Clone, Copy, Debug, Default)]
pub struct OneHot;

impl_proof_optimized_preset!(
    Dense,
    Field,
    ExtensionField,
    akita_types::SisModulusProfileId::Q31Offset19,
    256,
    31,
    31,
    fold_norms = akita_types::sis::FoldWitnessNorms::bounded(3, 128),
    schedules = ("schedules-fp31-dense", "fp31_dense", fp31_dense_table),
    ring_dimension_schedule_mode = ADAPTIVE_RING_DIMENSION_MODE
);
impl_proof_optimized_preset!(
    OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusProfileId::Q31Offset19,
    256,
    31,
    1,
    fold_norms = akita_types::sis::FoldWitnessNorms::new(1, 1),
    schedules = ("schedules-fp31-onehot", "fp31_onehot", fp31_onehot_table),
    ring_dimension_schedule_mode = ADAPTIVE_RING_DIMENSION_MODE
);
