//! fp63 presets used for small-field integration and profiling.

use super::*;

/// Base field for the fp63 presets.
pub type Field = Prime63Offset259;
/// Quadratic extension used for fp63 public claims and Fiat-Shamir challenges.
pub type ExtensionField = Ext2<Field>;

const SUFFIX_RING_DIMENSIONS: &[usize] = &[64];
const A_RING_DIMENSIONS: &[usize] = &[64, 128, 256, 512];
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

/// Default adaptive dense preset for fp63.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dense;

/// Default adaptive one-hot preset for fp63.
#[derive(Clone, Copy, Debug, Default)]
pub struct OneHot;

impl_proof_optimized_preset!(
    Dense,
    Field,
    ExtensionField,
    akita_types::SisModulusProfileId::Q63Offset259,
    256,
    63,
    63,
    fold_norms = akita_types::sis::FoldWitnessNorms::bounded(3, 64),
    schedules = ("schedules-fp63-dense", "fp63_dense", fp63_dense_table),
    ring_dimension_schedule_mode = ADAPTIVE_RING_DIMENSION_MODE
);
impl_proof_optimized_preset!(
    OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusProfileId::Q63Offset259,
    256,
    63,
    1,
    fold_norms = akita_types::sis::FoldWitnessNorms::new(1, 1),
    schedules = ("schedules-fp63-onehot", "fp63_onehot", fp63_onehot_table),
    ring_dimension_schedule_mode = ADAPTIVE_RING_DIMENSION_MODE
);
