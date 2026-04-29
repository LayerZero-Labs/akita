//! Public preset bundles for commitment field/config pairings and default schemes.

use super::config::{CommitmentPreset, GeneratedAdaptivePolicy};
use crate::protocol::commitment::profile::{CommitmentFieldProfile, Fp128PrimeProfile};

/// Default fp128 protocol presets on `p = 2^128 - 2355`.
pub mod fp128 {
    use super::*;

    /// Prime profile for the default fp128 presets.
    pub type Profile = Fp128PrimeProfile;
    /// Base field for the default fp128 presets.
    pub type Field = <Profile as CommitmentFieldProfile>::Field;

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
