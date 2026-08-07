//! Default fp128 protocol presets on `p = 2^128 - 2^32 + 22537`
//! (`Prime128OffsetA7F7`).

use super::*;

/// Base field for the default fp128 presets.
pub type Field = Prime128OffsetA7F7;

/// Dense adaptive `D=64` preset.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64Dense;

/// Default dense preset with a dimension-free flat public matrix and
/// planner-selected per-level A/B/D commitment dimensions.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dense;

impl Dense {
    pub const A_RING_DIMENSIONS: [usize; 3] = [64, 128, 256];
    pub const B_RING_DIMENSIONS: [usize; 2] = [64, 128];
    pub const D_RING_DIMENSIONS: [usize; 2] = [64, 128];
}

/// Binary onehot generated `D=64` preset.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64OneHot;

/// Binary onehot `D=64`, `K=16` preset with planner-derived schedules.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64OneHotK16;

/// Binary onehot `D=256` tableless preset for mixed-ring experiments.
/// fp128 certifies all three commitment roles at this dimension.
#[derive(Clone, Copy, Debug, Default)]
pub struct D256OneHot;

/// Default binary onehot preset with a dimension-free flat public matrix and
/// planner-selected per-level A/B/D commitment dimensions.
///
/// Mixed-dimension planning is an offline generation step. Runtime proving
/// and verification resolve the exact generated catalog row.
#[derive(Clone, Copy, Debug, Default)]
pub struct OneHot;

impl OneHot {
    pub const A_RING_DIMENSIONS: [usize; 3] = [64, 128, 256];
    pub const B_RING_DIMENSIONS: [usize; 2] = [64, 128];
    pub const D_RING_DIMENSIONS: [usize; 2] = [64, 128];
}

/// Tableless policy marker for a `D = 512` inner (A-role) root.
///
/// fp128 certifies the A role at `D = 512` for the Q128 profile, but not B/D.
/// Consequently this preset is used only as the envelope type for the
/// three-band mixed-role builder, which keeps B/D within their audited
/// dimensions; it cannot produce a standalone uniform-D512 schedule.
#[derive(Clone, Copy, Debug, Default)]
pub struct D512OneHot;

/// Uniform-D64 multi-chunk companion retained as the base of the recursive
/// distributed-prover preset.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64OneHotMultiChunk;

/// Direct multi-chunk companion of [`OneHot`] using the W8R2 profile.
#[derive(Clone, Copy, Debug, Default)]
pub struct OneHotMultiChunk;

/// Direct multi-chunk companion of [`OneHot`] using the W2R2 profile.
#[derive(Clone, Copy, Debug, Default)]
pub struct OneHotMultiChunkW2R2;

/// Direct multi-chunk companion of [`OneHot`] using the W4R2 profile.
#[derive(Clone, Copy, Debug, Default)]
pub struct OneHotMultiChunkW4R2;

/// Direct multi-chunk companion of [`Dense`] using the W8R2 profile.
#[derive(Clone, Copy, Debug, Default)]
pub struct DenseMultiChunk;

impl_proof_optimized_preset!(
    D64Dense,
    Field,
    Field,
    akita_types::SisModulusProfileId::Q128OffsetA7F7,
    64,
    128,
    128,
    fold_norms = akita_types::sis::FoldWitnessNorms::bounded(3, 64)
);
impl_proof_optimized_preset!(
    Dense,
    Field,
    Field,
    akita_types::SisModulusProfileId::Q128OffsetA7F7,
    256,
    128,
    128,
    fold_norms = akita_types::sis::FoldWitnessNorms::bounded(3, 64),
    schedules = ("schedules-fp128-dense", "fp128_dense", fp128_dense_table),
    ring_dimension_schedule_mode = akita_schedules::RingDimensionScheduleMode::AdaptiveDimension {
        num_search_levels: akita_schedules::ADAPTIVE_SEARCH_LEVELS,
        uniform_suffix_dimension: 64,
        potential_a_dimensions: &Dense::A_RING_DIMENSIONS,
        potential_b_dimensions: &Dense::B_RING_DIMENSIONS,
        potential_d_dimensions: &Dense::D_RING_DIMENSIONS,
    }
);
impl_proof_optimized_preset!(
    D64OneHot,
    Field,
    Field,
    akita_types::SisModulusProfileId::Q128OffsetA7F7,
    64,
    128,
    1,
    fold_norms = akita_types::sis::FoldWitnessNorms::new(1, 1)
);
impl_proof_optimized_preset!(
    D64OneHotK16,
    Field,
    Field,
    akita_types::SisModulusProfileId::Q128OffsetA7F7,
    64,
    128,
    1,
    fold_norms = akita_types::sis::FoldWitnessNorms::new(1, 4)
);
impl_proof_optimized_preset!(
    D256OneHot,
    Field,
    Field,
    akita_types::SisModulusProfileId::Q128OffsetA7F7,
    256,
    128,
    1,
    fold_norms = akita_types::sis::FoldWitnessNorms::new(1, 1)
);
impl_proof_optimized_preset!(
    OneHot,
    Field,
    Field,
    akita_types::SisModulusProfileId::Q128OffsetA7F7,
    256,
    128,
    1,
    fold_norms = akita_types::sis::FoldWitnessNorms::new(1, 1),
    schedules = ("schedules-fp128-onehot", "fp128_onehot", fp128_onehot_table),
    ring_dimension_schedule_mode = akita_schedules::RingDimensionScheduleMode::AdaptiveDimension {
        num_search_levels: akita_schedules::ADAPTIVE_SEARCH_LEVELS,
        uniform_suffix_dimension: 64,
        potential_a_dimensions: &OneHot::A_RING_DIMENSIONS,
        potential_b_dimensions: &OneHot::B_RING_DIMENSIONS,
        potential_d_dimensions: &OneHot::D_RING_DIMENSIONS,
    }
);
impl_proof_optimized_preset!(
    D512OneHot,
    Field,
    Field,
    akita_types::SisModulusProfileId::Q128OffsetA7F7,
    512,
    128,
    1,
    fold_norms = akita_types::sis::FoldWitnessNorms::new(1, 2)
);
impl_multi_chunk_companion!(
    D64OneHotMultiChunk,
    D64OneHot,
    akita_types::MultiChunkProfileId::W8R2,
    "schedules-fp128-d64-onehot-multi-chunk",
    fp128_d64_onehot_multi_chunk_table
);
impl_multi_chunk_companion!(
    OneHotMultiChunk,
    OneHot,
    akita_types::MultiChunkProfileId::W8R2,
    "schedules-fp128-onehot-multi-chunk",
    fp128_onehot_multi_chunk_table
);
impl_multi_chunk_companion!(
    OneHotMultiChunkW2R2,
    OneHot,
    akita_types::MultiChunkProfileId::W2R2,
    "schedules-fp128-onehot-multi-chunk-w2r2",
    fp128_onehot_multi_chunk_w2r2_table
);
impl_multi_chunk_companion!(
    OneHotMultiChunkW4R2,
    OneHot,
    akita_types::MultiChunkProfileId::W4R2,
    "schedules-fp128-onehot-multi-chunk-w4r2",
    fp128_onehot_multi_chunk_w4r2_table
);
impl_multi_chunk_companion!(
    DenseMultiChunk,
    Dense,
    akita_types::MultiChunkProfileId::W8R2,
    "schedules-fp128-dense-multi-chunk",
    fp128_dense_multi_chunk_table
);
