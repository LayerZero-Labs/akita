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
    pub const A_RING_DIMENSIONS: [usize; 5] = [64, 128, 256, 512, 1024];
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
    pub const A_RING_DIMENSIONS: [usize; 4] = [64, 128, 256, 512];
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

/// Dense preset for witnesses known to fit an **unsigned 64-bit** magnitude
/// inside the 128-bit field, i.e. `u64`-valued coefficients.
///
/// Same field, same SIS modulus profile, and the same balanced signed-digit
/// source class as [`Dense`]; the only difference is the declared committed-source
/// bound. That roughly halves the A-role digit depth, and with it the A input
/// width, the shared setup matrix, and the level-1 witness the whole recursion
/// suffix inherits. Opening witnesses stay full-width
/// (`log_open_bound = Some(128)`), because `t̂` / `ŵ` carry genuine field
/// elements.
///
/// # Why the bound is `65` and not `64`
///
/// [`akita_types::DecompositionParams::log_commit_bound`] is a **signed** bit
/// width: `k` denotes the centered range `[-2^(k-1), 2^(k-1) - 1]`, because the
/// gadget decomposition works on centered representatives and balanced digits are
/// themselves signed. A `u64` reaches `u64::MAX = 2^64 - 1`, so it needs one sign
/// bit plus 64 magnitude bits — `log_commit_bound = 65`, not `64`. Declaring `64`
/// would cover only `[-2^63, 2^63 - 1]` and miss half of a uniform `u64`
/// distribution.
///
/// [`Self::MAX_CENTERED_MAGNITUDE`] states the same fact without the
/// signed-bit-width indirection; prefer it when asserting your data fits.
///
/// # What you must guarantee
///
/// This is a **different commitment**, not a cheaper encoding of [`Dense`]: its
/// catalog identity differs, and it is binding and complete only for polynomials
/// whose centered coefficients lie inside the declared range. `commit` rejects an
/// out-of-range coefficient rather than committing a truncation, so a caller that
/// cannot guarantee the bound must use [`Dense`].
#[derive(Clone, Copy, Debug, Default)]
pub struct DenseBounded;

impl DenseBounded {
    /// Committed-source bound as a **signed** bit width.
    ///
    /// `65` means the centered range `[-2^64, 2^64 - 1]`, which contains every
    /// `u64`. See the type docs for why this is `65` rather than `64`.
    pub const LOG_COMMIT_BOUND: u32 = 65;

    /// Largest centered magnitude this preset commits, on the positive side.
    ///
    /// Exactly `u64::MAX`, so `u64`-valued coefficients sit on the endpoint and
    /// anything above is rejected at `commit`.
    pub const MAX_CENTERED_MAGNITUDE: u128 = (1u128 << (Self::LOG_COMMIT_BOUND - 1)) - 1;
}

// The preset's whole reason to exist, enforced at compile time rather than left
// to a test: the declared bound must contain every `u64`. `LOG_COMMIT_BOUND` is a
// *signed* bit width, so a value of 64 would silently cover only `[-2^63, 2^63-1]`
// and miss half of a uniform `u64` distribution. Asserted directly on the
// magnitude, so there is no derived boolean to drift from it — and because the
// macro is configured from `LOG_COMMIT_BOUND` itself, this covers the value that
// actually builds the preset.
const _: () = assert!(
    DenseBounded::MAX_CENTERED_MAGNITUDE >= u64::MAX as u128,
    "fp128::DenseBounded must declare a bound containing every u64"
);

impl_proof_optimized_preset!(
    Dense,
    Field,
    Field,
    akita_types::SisModulusProfileId::Q128OffsetA7F7,
    128,
    128,
    source = balanced_digits,
    schedule_family = "fp128_dense",
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
    128,
    // One declaration, not two: the macro takes an expression, so the preset is
    // configured from the same constant callers read. A signed bit width of 65 is
    // `[-2^64, 2^64 - 1]`, the smallest declaration containing every `u64`.
    DenseBounded::LOG_COMMIT_BOUND,
    source = balanced_digits,
    schedule_family = "fp128_dense_bounded",
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
    128,
    1,
    source = unit_one_hot,
    schedule_family = "fp128_onehot",
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
    "fp128_onehot_multi_chunk"
);

impl crate::recursive_commitment::RecursiveScheduleConfig for OneHot {
    const RECURSIVE_SCHEDULE_FAMILY_NAME: &'static str = "fp128_onehot_recursive";
}

impl crate::recursive_commitment::RecursiveScheduleConfig for OneHotMultiChunk {
    const RECURSIVE_SCHEDULE_FAMILY_NAME: &'static str = "fp128_onehot_recursive_multi_chunk_w8r2";
}
impl_multi_chunk_companion!(
    OneHotMultiChunkW2R2,
    OneHot,
    akita_types::MultiChunkProfileId::W2R2,
    "fp128_onehot_multi_chunk_w2r2"
);
impl_multi_chunk_companion!(
    OneHotMultiChunkW4R2,
    OneHot,
    akita_types::MultiChunkProfileId::W4R2,
    "fp128_onehot_multi_chunk_w4r2"
);
impl_multi_chunk_companion!(
    DenseMultiChunk,
    Dense,
    akita_types::MultiChunkProfileId::W8R2,
    "fp128_dense_multi_chunk"
);
