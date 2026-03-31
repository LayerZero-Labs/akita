//! Public preset bundles for commitment field/config pairings.

use super::config::{
    CommitmentPreset, Fp128AdaptiveBoundedPolicy, Fp128AdaptiveOneHotD64Policy,
    Fp128StaticBoundedPolicy,
};
use crate::algebra::{Prime128Offset275, Prime128Offset5823};

/// Default fp128 protocol presets on `p = 2^128 - 275`.
pub mod fp128 {
    use super::*;

    /// Base field for the default fp128 presets.
    pub type Field = Prime128Offset275;

    /// Static `D=128`, rank-1 schedule with explicit root/recursive bases.
    pub type StaticBounded<
        const LOG_COMMIT_BOUND: u32,
        const LOG_BASIS: u32,
        const W_LOG_BASIS: u32 = LOG_BASIS,
    > = CommitmentPreset<
        Field,
        Fp128StaticBoundedPolicy<128, LOG_COMMIT_BOUND, LOG_BASIS, W_LOG_BASIS, 1, 1, 1>,
    >;

    /// Static `D=64`, rank-1 schedule with explicit root/recursive bases.
    pub type D64StaticBounded<
        const LOG_COMMIT_BOUND: u32,
        const LOG_BASIS: u32,
        const W_LOG_BASIS: u32 = LOG_BASIS,
    > = CommitmentPreset<
        Field,
        Fp128StaticBoundedPolicy<64, LOG_COMMIT_BOUND, LOG_BASIS, W_LOG_BASIS, 1, 1, 1>,
    >;

    /// Adaptive `D=128`, rank-1 family with planner-selected bases.
    pub type AdaptiveBounded<const LOG_COMMIT_BOUND: u32> =
        CommitmentPreset<Field, Fp128AdaptiveBoundedPolicy<128, LOG_COMMIT_BOUND, 1, 1, 1>>;

    /// Adaptive `D=32`, rank-2 family with planner-selected bases.
    pub type D32AdaptiveBounded<const LOG_COMMIT_BOUND: u32> =
        CommitmentPreset<Field, Fp128AdaptiveBoundedPolicy<32, LOG_COMMIT_BOUND, 2, 2, 2>>;

    /// Adaptive `D=16`, rank-4 family with planner-selected bases.
    pub type D16AdaptiveBounded<const LOG_COMMIT_BOUND: u32> =
        CommitmentPreset<Field, Fp128AdaptiveBoundedPolicy<16, LOG_COMMIT_BOUND, 4, 4, 4>>;

    /// Full-field adaptive preset.
    pub type Full = AdaptiveBounded<128>;
    /// Log-bounded adaptive preset.
    pub type LogBasis = AdaptiveBounded<3>;
    /// Binary onehot adaptive preset.
    pub type OneHot = CommitmentPreset<Field, Fp128AdaptiveOneHotD64Policy>;

    /// Full-field adaptive `D=32` preset.
    pub type D32Full = D32AdaptiveBounded<128>;
    /// Log-bounded adaptive `D=32` preset.
    pub type D32LogBasis = D32AdaptiveBounded<3>;
    /// Onehot adaptive `D=32` preset.
    pub type D32OneHot = D32AdaptiveBounded<1>;

    /// Full-field adaptive `D=16` preset.
    pub type D16Full = D16AdaptiveBounded<128>;
    /// Log-bounded adaptive `D=16` preset.
    pub type D16LogBasis = D16AdaptiveBounded<3>;
    /// Onehot adaptive `D=16` preset.
    pub type D16OneHot = D16AdaptiveBounded<1>;
}

/// Explicit legacy fp128 protocol presets on `p = 2^128 - 5823`.
pub mod fp128_5823 {
    use super::*;

    /// Base field for the legacy fp128 presets.
    pub type Field = Prime128Offset5823;

    /// Static `D=128`, rank-1 schedule with explicit root/recursive bases.
    pub type StaticBounded<
        const LOG_COMMIT_BOUND: u32,
        const LOG_BASIS: u32,
        const W_LOG_BASIS: u32 = LOG_BASIS,
    > = CommitmentPreset<
        Field,
        Fp128StaticBoundedPolicy<128, LOG_COMMIT_BOUND, LOG_BASIS, W_LOG_BASIS, 1, 1, 1>,
    >;

    /// Static `D=64`, rank-1 schedule with explicit root/recursive bases.
    pub type D64StaticBounded<
        const LOG_COMMIT_BOUND: u32,
        const LOG_BASIS: u32,
        const W_LOG_BASIS: u32 = LOG_BASIS,
    > = CommitmentPreset<
        Field,
        Fp128StaticBoundedPolicy<64, LOG_COMMIT_BOUND, LOG_BASIS, W_LOG_BASIS, 1, 1, 1>,
    >;

    /// Adaptive `D=128`, rank-1 family with planner-selected bases.
    pub type AdaptiveBounded<const LOG_COMMIT_BOUND: u32> =
        CommitmentPreset<Field, Fp128AdaptiveBoundedPolicy<128, LOG_COMMIT_BOUND, 1, 1, 1>>;

    /// Full-field adaptive preset.
    pub type Full = AdaptiveBounded<128>;
    /// Log-bounded adaptive preset.
    pub type LogBasis = AdaptiveBounded<3>;
    /// Binary onehot adaptive preset.
    pub type OneHot = CommitmentPreset<Field, Fp128AdaptiveOneHotD64Policy>;
}
