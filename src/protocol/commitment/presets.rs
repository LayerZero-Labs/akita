//! Public preset bundles for commitment field/config pairings and default schemes.

use super::config::{CommitmentPreset, GeneratedAdaptivePolicy, StaticBoundedPolicy};
use crate::protocol::commitment::profile::{CommitmentFieldProfile, Fp128PrimeProfile};

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

    /// Generated adaptive family with pinned planner tables.
    pub type AdaptiveBounded<const D: usize, const LOG_COMMIT_BOUND: u32> =
        CommitmentPreset<Field, GeneratedAdaptivePolicy<Profile, D, LOG_COMMIT_BOUND>>;

    /// Full-field adaptive `D=128` preset.
    pub type D128Full = AdaptiveBounded<128, 128>;
    /// Full-field adaptive `D=64` preset.
    pub type D64Full = AdaptiveBounded<64, 128>;
    /// Binary onehot generated `D=64` preset.
    pub type D64OneHot = AdaptiveBounded<64, 1>;

    /// Full-field adaptive `D=32` preset.
    pub type D32Full = AdaptiveBounded<32, 128>;
    /// Onehot adaptive `D=32` preset.
    pub type D32OneHot = AdaptiveBounded<32, 1>;
}
