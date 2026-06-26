//! Witness-layout configuration shared by the planner, prover, and verifier.
//!
//! [`ChunkedWitnessCfg`] describes the multi-chunk witness layout used by the
//! distributed prover: how many chunks the witness is split into and for how
//! many leading fold levels the chunked layout stays active before the schedule
//! reverts to single-chunk sizing.
//!
//! `num_chunks = 1` is the single-chunk (standard) case and is byte-identical to
//! the historical layout. The struct is the single source of truth for the chunk
//! layout — the planner prices schedules with it, the catalog identity embeds it,
//! and the per-level [`crate::LevelParams::witness_chunk`] carries the resolved
//! value the verifier consumes.

use akita_field::AkitaError;

/// Chunk-based witness layout parameters.
///
/// `num_chunks = 1` is the single-chunk (standard) case; `num_chunks` must be a
/// power of two. `num_activated_levels` is how many leading protocol levels the
/// multi-chunk layout is active; it is ignored when `num_chunks = 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkedWitnessCfg {
    /// Number of witness chunks / replicated ẑ segments while the multi-chunk
    /// layout is active. `1` means single-chunk (default).
    pub num_chunks: usize,
    /// Count of leading fold levels (absolute levels `0, 1, …, R−1`) priced
    /// under the chunked layout. `0` disables multi-chunk planning.
    pub num_activated_levels: usize,
}

impl Default for ChunkedWitnessCfg {
    fn default() -> Self {
        Self {
            num_chunks: 1,
            num_activated_levels: 0,
        }
    }
}

impl ChunkedWitnessCfg {
    /// Const equivalent of [`Default::default`], usable in const contexts such as
    /// generated catalog-identity literals.
    pub const fn default_non_chunked() -> Self {
        Self {
            num_chunks: 1,
            num_activated_levels: 0,
        }
    }

    /// True iff the planner should price the chunked layout for the leading
    /// levels. Both `num_chunks > 1` and `num_activated_levels > 0` are required;
    /// any other combination is either single-chunk or an invalid config caught
    /// by [`Self::validate`].
    pub const fn uses_multi_chunk(self) -> bool {
        self.num_chunks > 1 && self.num_activated_levels > 0
    }

    /// Preset helper for the initial D64 multi-chunk tables (book example: 8
    /// nodes, three leading chunked fold levels).
    pub const fn d64_production() -> Self {
        Self {
            num_chunks: 8,
            num_activated_levels: 3,
        }
    }

    /// Layout-only validation (no dependency on planner internals).
    ///
    /// The depth bound against the planner's recursion cap is enforced
    /// separately at the planner entry, where the constant is in scope.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] for `num_chunks == 0`, a
    /// non-power-of-two `num_chunks > 1`, or an inconsistent
    /// `(num_chunks, num_activated_levels)` pair.
    pub fn validate(self) -> Result<(), AkitaError> {
        if self.num_chunks == 0 {
            return Err(AkitaError::InvalidSetup(
                "ChunkedWitnessCfg: num_chunks must be >= 1".to_string(),
            ));
        }
        if self.num_chunks > 1 && !self.num_chunks.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "ChunkedWitnessCfg: num_chunks must be a power of two".to_string(),
            ));
        }
        if self.num_activated_levels > 0 && self.num_chunks == 1 {
            return Err(AkitaError::InvalidSetup(
                "ChunkedWitnessCfg: num_activated_levels > 0 requires num_chunks > 1".to_string(),
            ));
        }
        if self.num_chunks > 1 && self.num_activated_levels == 0 {
            return Err(AkitaError::InvalidSetup(
                "ChunkedWitnessCfg: num_chunks > 1 requires num_activated_levels > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Append canonical Fiat-Shamir descriptor bytes.
    ///
    /// Only invoked by [`crate::LevelParams`] descriptor binding when the level
    /// is chunked, so single-chunk levels stay byte-for-byte identical to the
    /// historical layout (the flag-off no-op invariant).
    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        crate::descriptor_bytes::push_usize(bytes, self.num_chunks);
        crate::descriptor_bytes::push_usize(bytes, self.num_activated_levels);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_single_chunk() {
        let cfg = ChunkedWitnessCfg::default();
        assert_eq!(cfg, ChunkedWitnessCfg::default_non_chunked());
        assert_eq!(cfg.num_chunks, 1);
        assert_eq!(cfg.num_activated_levels, 0);
        assert!(!cfg.uses_multi_chunk());
        cfg.validate().expect("default config is valid");
    }

    #[test]
    fn d64_production_uses_multi_chunk() {
        let cfg = ChunkedWitnessCfg::d64_production();
        assert_eq!(cfg.num_chunks, 8);
        assert_eq!(cfg.num_activated_levels, 3);
        assert!(cfg.uses_multi_chunk());
        cfg.validate().expect("d64_production is valid");
    }

    #[test]
    fn validate_rejects_invalid_configs() {
        assert!(ChunkedWitnessCfg {
            num_chunks: 0,
            num_activated_levels: 0,
        }
        .validate()
        .is_err());
        assert!(ChunkedWitnessCfg {
            num_chunks: 6,
            num_activated_levels: 2,
        }
        .validate()
        .is_err());
        assert!(ChunkedWitnessCfg {
            num_chunks: 1,
            num_activated_levels: 2,
        }
        .validate()
        .is_err());
        assert!(ChunkedWitnessCfg {
            num_chunks: 8,
            num_activated_levels: 0,
        }
        .validate()
        .is_err());
        for n in [2usize, 4, 8, 16] {
            ChunkedWitnessCfg {
                num_chunks: n,
                num_activated_levels: 1,
            }
            .validate()
            .expect("power-of-two chunk counts validate");
        }
    }
}
