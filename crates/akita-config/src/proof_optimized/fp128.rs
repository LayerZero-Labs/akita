//! Default fp128 protocol presets on `p = 2^128 - 2^32 + 22537`
//! (`Prime128OffsetA7F7`).

use super::*;

/// Base field for the default fp128 presets.
pub type Field = Prime128OffsetA7F7;

/// Default dense preset with a dimension-free flat public matrix and
/// planner-selected per-level A/B/D commitment dimensions.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dense;

impl Dense {
    pub const A_RING_DIMENSIONS: [usize; 3] = [64, 128, 256];
    pub const B_RING_DIMENSIONS: [usize; 2] = [64, 128];
    pub const D_RING_DIMENSIONS: [usize; 2] = [64, 128];
}

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

/// Dense preset for witnesses known to fit 64 signed bits of the 128-bit field.
///
/// Same field, same SIS modulus profile, and the same balanced signed-digit
/// source class as [`Dense`]; the only difference is the declared committed-source
/// bound (`log_commit_bound = 64` instead of `128`). That halves the A-role digit
/// depth, and with it the A input width, the shared setup matrix, and the
/// level-1 witness the whole recursion suffix inherits. Opening witnesses stay
/// full-width (`log_open_bound = Some(128)`), because `t̂` / `ŵ` carry genuine
/// field elements.
///
/// This is a **different commitment**, not a cheaper encoding of [`Dense`]: its
/// catalog identity differs, and it is binding and complete only for polynomials
/// whose centered coefficients fit the scheduled digit envelope. `commit` rejects
/// an out-of-range coefficient rather than committing a truncation, so a caller
/// that cannot guarantee the bound must use [`Dense`].
#[derive(Clone, Copy, Debug, Default)]
pub struct DenseBounded;

impl DenseBounded {
    /// Committed-source bound in signed bits.
    pub const LOG_COMMIT_BOUND: u32 = 64;
}

impl_proof_optimized_preset!(
    Dense,
    Field,
    Field,
    akita_types::SisModulusProfileId::Q128OffsetA7F7,
    256,
    128,
    128,
    source = balanced_digits,
    schedules = ("schedules-fp128-dense", "fp128_dense", fp128_dense_table),
    ring_dimension_schedule_mode = akita_schedules::RingDimensionScheduleMode::AdaptiveDimension {
        num_search_levels: akita_schedules::ADAPTIVE_SEARCH_LEVELS,
        suffix_dimensions: &[64],
        potential_a_dimensions: &Dense::A_RING_DIMENSIONS,
        potential_b_dimensions: &Dense::B_RING_DIMENSIONS,
        potential_d_dimensions: &Dense::D_RING_DIMENSIONS,
    }
);
impl_proof_optimized_preset!(
    DenseBounded,
    Field,
    Field,
    akita_types::SisModulusProfileId::Q128OffsetA7F7,
    256,
    128,
    64,
    source = balanced_digits,
    schedules = (
        "schedules-fp128-dense-bounded",
        "fp128_dense_bounded",
        fp128_dense_bounded_table
    ),
    ring_dimension_schedule_mode = akita_schedules::RingDimensionScheduleMode::AdaptiveDimension {
        num_search_levels: akita_schedules::ADAPTIVE_SEARCH_LEVELS,
        suffix_dimensions: &[64],
        potential_a_dimensions: &Dense::A_RING_DIMENSIONS,
        potential_b_dimensions: &Dense::B_RING_DIMENSIONS,
        potential_d_dimensions: &Dense::D_RING_DIMENSIONS,
    }
);
impl_proof_optimized_preset!(
    OneHot,
    Field,
    Field,
    akita_types::SisModulusProfileId::Q128OffsetA7F7,
    256,
    128,
    1,
    source = unit_one_hot,
    schedules = ("schedules-fp128-onehot", "fp128_onehot", fp128_onehot_table),
    ring_dimension_schedule_mode = akita_schedules::RingDimensionScheduleMode::AdaptiveDimension {
        num_search_levels: akita_schedules::ADAPTIVE_SEARCH_LEVELS,
        suffix_dimensions: &[64],
        potential_a_dimensions: &OneHot::A_RING_DIMENSIONS,
        potential_b_dimensions: &OneHot::B_RING_DIMENSIONS,
        potential_d_dimensions: &OneHot::D_RING_DIMENSIONS,
    }
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
