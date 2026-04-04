//! Public preset bundles for commitment field/config pairings and default schemes.

use super::config::{
    CommitmentPreset, GeneratedAdaptivePolicy, PlannedAdaptiveBoundedPolicy, StaticBoundedPolicy,
};
use crate::protocol::commitment::profile::{CommitmentFieldProfile, Fp128PrimeProfile};
use crate::protocol::dynamic_commitment_scheme::{
    DynamicFullFamily, DynamicFullScheme, DynamicOneHotFamily, DynamicOneHotScheme,
};

/// Default fp128 protocol presets on `p = 2^128 - 275`.
pub mod fp128 {
    use super::*;

    /// Prime profile for the default fp128 presets.
    pub type Profile = Fp128PrimeProfile;
    /// Base field for the default fp128 presets.
    pub type Field = <Profile as CommitmentFieldProfile>::Field;

    /// Static `D=128`, rank-1 schedule with explicit root/recursive bases.
    pub type StaticBounded<
        const LOG_COMMIT_BOUND: u32,
        const LOG_BASIS: u32,
        const W_LOG_BASIS: u32 = LOG_BASIS,
    > = CommitmentPreset<
        Field,
        StaticBoundedPolicy<Profile, 128, LOG_COMMIT_BOUND, LOG_BASIS, W_LOG_BASIS, 1, 1, 1>,
    >;

    /// Static `D=64`, rank-1 schedule with explicit root/recursive bases.
    pub type D64StaticBounded<
        const LOG_COMMIT_BOUND: u32,
        const LOG_BASIS: u32,
        const W_LOG_BASIS: u32 = LOG_BASIS,
    > = CommitmentPreset<
        Field,
        StaticBoundedPolicy<Profile, 64, LOG_COMMIT_BOUND, LOG_BASIS, W_LOG_BASIS, 1, 1, 1>,
    >;

    /// Adaptive `D=128`, rank-1 family with planner-selected bases.
    pub type AdaptiveBounded<const LOG_COMMIT_BOUND: u32> = CommitmentPreset<
        Field,
        PlannedAdaptiveBoundedPolicy<Profile, 128, LOG_COMMIT_BOUND, 1, 1, 1>,
    >;

    /// Generated adaptive `D=32` family with pinned planner tables.
    pub type D32AdaptiveBounded<const LOG_COMMIT_BOUND: u32> =
        CommitmentPreset<Field, GeneratedAdaptivePolicy<Profile, 32, LOG_COMMIT_BOUND>>;

    /// Full-field adaptive `D=128` preset.
    pub type D128Full = AdaptiveBounded<128>;
    /// Log-bounded adaptive preset.
    pub type LogBasis = AdaptiveBounded<3>;
    /// Binary onehot generated `D=64` preset.
    pub type D64OneHot = CommitmentPreset<Field, GeneratedAdaptivePolicy<Profile, 64, 1>>;

    /// Full-field adaptive `D=32` preset.
    pub type D32Full = D32AdaptiveBounded<128>;
    /// Log-bounded adaptive `D=32` preset.
    pub type D32LogBasis = D32AdaptiveBounded<3>;
    /// Onehot adaptive `D=32` preset.
    pub type D32OneHot = D32AdaptiveBounded<1>;

    /// Family selector for the default dynamic full-field preset.
    pub type FullFamily = DynamicFullFamily<Profile>;
    /// Default full-field preset with runtime-selected root `D`.
    pub type Full = DynamicFullScheme<Profile>;
    /// Family selector for the default dynamic onehot preset.
    pub type OneHotFamily = DynamicOneHotFamily<Profile>;
    /// Default onehot preset with runtime-selected root `D`.
    pub type OneHot = DynamicOneHotScheme<Profile>;
}
