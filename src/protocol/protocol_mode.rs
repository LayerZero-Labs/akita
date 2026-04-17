//! Selection between Split and Fused Hachi protocol shapes.
//!
//! [`HachiProtocolMode::Split`] is the original two-sumcheck layout: an
//! eq-factored Stage 1 proves the range check `S(z) = w(z)*(w(z)+1)`, and a
//! standard Stage 2 proves the relation `<w, alpha*M>` while batching in
//! `s_claim` via a sampled coefficient.
//!
//! [`HachiProtocolMode::Fused`] collapses range check and relation into a
//! single non-eq-factored Stage 1 (the "fused leaf"), absorbs the intermediate
//! claims `s_claim`, `w(r1)`, and (when delegation is enabled) `claimed_setup_val`,
//! and then runs a smaller claim-reduction Stage 2 that fuses s-virtualization
//! (proving `S(r1) = sum_z eq(r1, z) * w(z)*(w(z)+1)`) with w-adaptation
//! (proving `w(r1) = sum_z eq(r1, z) * w(z)`). The Stage 1 oracle check is
//! deferred until after Stage 2 so that the verifier can reuse the same
//! `r2 = r_stage2` to evaluate `alpha`, `M`, and `eq(tau0, r1)` once.
//!
//! The default ([`HachiProtocolMode::default`]) is [`HachiProtocolMode::Split`].
//! Fused mode is opt-in via [`HachiProverSetup::with_mode`] until its
//! schedule tables are regenerated.
//!
//! [`HachiProverSetup::with_mode`]: crate::protocol::commitment::commit::HachiProverSetup::with_mode

/// Which Stage 1 / Stage 2 sumcheck shape Hachi runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HachiProtocolMode {
    /// Eq-factored Stage 1 (range check) + Stage 2 (relation, batched with `s_claim`).
    Split,
    /// Non-eq-factored fused Stage 1 (range + relation) + claim-reduction Stage 2.
    Fused,
}

impl Default for HachiProtocolMode {
    fn default() -> Self {
        Self::Split
    }
}

impl HachiProtocolMode {
    /// Whether the fused Stage 1 leaf is present in the produced proof.
    #[inline]
    pub const fn has_fused_leaf(self) -> bool {
        matches!(self, Self::Fused)
    }
}
