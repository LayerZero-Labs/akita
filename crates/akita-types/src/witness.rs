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

use akita_field::{AkitaError, CanonicalField, FieldCore};

use crate::{
    w_ring_element_count_for_chunks, LevelParams, MRowLayout, RingRelationOpeningCounts,
    RingRelationSegmentLayout,
};

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

/// Per-chunk segment lengths.
/// Each chunk share the same lengths of ê/t̂/ẑ.
/// û is not present in any chunk since tiered commitment is not supported for multi-chunk layout.
/// r̂ is only present in the last chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WitnessChunkLengths {
    pub z_chunk_len: usize,
    pub e_chunk_len: usize,
    pub t_chunk_len: usize,
    pub u_chunk_len: usize,
    pub r_chunk_len: usize,
}

/// Per-chunk segment offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WitnessChunkLayout {
    pub global_block_base: usize, // chunk_idx * blocks_per_chunk
    pub offset_z: usize,
    pub offset_e: usize,
    pub offset_t: usize,
    pub offset_u: Option<usize>,
    pub offset_r: Option<usize>,
}

/// Full witness column layout for num_chunks chunks.
/// `chunks` and `chunk_lengths` are parallel Vecs of length `num_chunks`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessLayout {
    /// Number of blocks for witnes ê/t̂ per chunk.
    pub blocks_per_chunk: usize,
    pub chunks: Vec<WitnessChunkLayout>,
    /// Lengths for each chunk. Each chunk share the same lengths of ê/t̂/ẑ.
    /// û is not present in any chunk since tiered commitment is not supported for multi-chunk layout.
    /// r̂ is only present in the last chunk.
    pub chunk_lengths: WitnessChunkLengths,
}

impl WitnessLayout {
    /// Convert to the legacy segment layout for the single-chunk layout.
    ///
    /// TODO: remove this method after the legacy layout is deprecated.
    pub fn to_legacy_segment_layout(&self) -> RingRelationSegmentLayout {
        if self.chunks.len() != 1 {
            panic!("witness layout: multi-chunk layout is not supported");
        }
        let last = &self.chunks[0];
        RingRelationSegmentLayout {
            offset_e: last.offset_e,
            offset_t: last.offset_t,
            offset_u: last.offset_u.unwrap(),
            offset_z: last.offset_z,
            offset_r: last.offset_r.unwrap(),
        }
    }

    /// Convert from the legacy segment layout to the witness layout.
    ///
    /// TODO: remove this method after the legacy layout is deprecated.
    pub fn from_legacy_segment_layout(layout: RingRelationSegmentLayout) -> WitnessLayout {
        WitnessLayout {
            blocks_per_chunk: 0,
            chunks: vec![WitnessChunkLayout {
                global_block_base: 0,
                offset_e: layout.offset_e,
                offset_t: layout.offset_t,
                offset_u: Some(layout.offset_u),
                offset_z: layout.offset_z,
                offset_r: Some(layout.offset_r),
            }],
            chunk_lengths: WitnessChunkLengths {
                z_chunk_len: 0,
                e_chunk_len: 0,
                t_chunk_len: 0,
                u_chunk_len: 0,
                r_chunk_len: 0,
            },
        }
    }

    /// Layout span in ring columns: last chunk `offset_r + r_len`.
    pub fn witness_ring_len(&self) -> Result<usize, AkitaError> {
        let overflow = || AkitaError::InvalidSetup("witness layout: capacity overflow".to_string());
        let last = self.chunks.last().ok_or_else(|| {
            AkitaError::InvalidSetup("witness layout: missing chunk table".to_string())
        })?;
        let r_offset = last.offset_r.ok_or_else(|| {
            AkitaError::InvalidSetup("witness layout: last chunk missing r segment".to_string())
        })?;
        r_offset
            .checked_add(self.chunk_lengths.r_chunk_len)
            .ok_or_else(overflow)
    }

    /// Validate that the resolved layout span matches planner witness pricing.
    ///
    /// `required` comes from [`w_ring_element_count_for_chunks`]; `witness_ring_len`
    /// is derived from this layout (`last.offset_r + r_chunk_len`). Both must agree.
    pub fn check_capacity(
        &self,
        field_bits: u32,
        lp: &LevelParams,
        num_polynomials: usize,
        m_row_layout: MRowLayout,
    ) -> Result<(), AkitaError> {
        let required = w_ring_element_count_for_chunks(
            field_bits,
            lp,
            num_polynomials,
            m_row_layout,
            lp.witness_chunk.num_chunks,
        )?;
        let witness_ring_len = self.witness_ring_len()?;
        if witness_ring_len != required {
            return Err(AkitaError::InvalidSetup(
                "witness capacity mismatch".to_string(),
            ));
        }
        Ok(())
    }
}

pub fn witness_chunk_lengths<F: FieldCore + CanonicalField, const D: usize>(
    lp: &LevelParams,
    opening_counts: RingRelationOpeningCounts,
    m_row_layout: MRowLayout,
) -> Result<WitnessChunkLengths, AkitaError> {
    let num_blocks = lp.num_blocks;
    let num_chunks = lp.witness_chunk.num_chunks;

    if num_blocks == 0 || !num_blocks.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "witness_chunk_lengths: num_blocks must be a non-zero power of two".to_string(),
        ));
    }
    if num_chunks == 0 {
        return Err(AkitaError::InvalidSetup(
            "witness_chunk_lengths: num_chunks must be a non-zero power of two".to_string(),
        ));
    }
    if !num_blocks.is_multiple_of(num_chunks) {
        return Err(AkitaError::InvalidSetup(
            "witness_chunk_lengths: num_blocks must be divisible by num_chunks".to_string(),
        ));
    }

    // Multi-chunk + tiered is unsupported (the chunked closed form assumes a
    // non-tiered, empty û segment). Reject rather than silently mis-price.
    if lp.f_key.is_some() || lp.tier_split != 1 {
        return Err(AkitaError::InvalidSetup(
            "witness_chunk_lengths: multi-chunk layout does not support tiered commitments"
                .to_string(),
        ));
    }

    let overflow = || {
        AkitaError::InvalidSetup(
            "witness_chunk_lengths: chunked witness width overflow".to_string(),
        )
    };
    let num_blocks_per_chunk = num_blocks / num_chunks;
    let num_claims = opening_counts.num_claims;

    let depth_open = lp.num_digits_open;
    let depth_commit = lp.num_digits_commit;
    let depth_fold = lp.num_digits_fold(num_claims, F::modulus_bits())?;
    if depth_open == 0 || depth_commit == 0 || depth_fold == 0 {
        return Err(AkitaError::InvalidSetup(
            "witness_chunk_lengths: prepared ring-switch layout has zero width".to_string(),
        ));
    }

    let num_total_blocks_per_chunk = num_claims * num_blocks_per_chunk;
    // ê / t̂: partitioned over the per-chunk block window.
    let e_chunk_len = num_total_blocks_per_chunk
        .checked_mul(depth_open)
        .ok_or_else(overflow)?;
    let t_chunk_len = num_total_blocks_per_chunk
        .checked_mul(depth_open)
        .and_then(|n| n.checked_mul(lp.a_key.row_len()))
        .ok_or_else(overflow)?;
    // ẑ: replicated across all chunks.
    let z_chunk_len = lp
        .inner_width()
        .checked_mul(depth_fold)
        .ok_or_else(overflow)?;

    // r̂: summed quotient collected on the last chunk.
    let r_rows = lp.m_row_count_for(1, 0, m_row_layout)?;
    let r_chunk_len = r_rows
        .checked_mul(crate::sis::compute_num_digits_full_field(
            F::modulus_bits(),
            lp.log_basis,
        ))
        .ok_or_else(overflow)?;
    Ok(WitnessChunkLengths {
        z_chunk_len,
        e_chunk_len,
        t_chunk_len,
        u_chunk_len: 0,
        r_chunk_len,
    })
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
    fn witness_ring_len_uses_last_chunk_r_tail() {
        let layout = WitnessLayout {
            blocks_per_chunk: 4,
            chunks: vec![
                WitnessChunkLayout {
                    global_block_base: 0,
                    offset_z: 0,
                    offset_e: 10,
                    offset_t: 20,
                    offset_u: None,
                    offset_r: None,
                },
                WitnessChunkLayout {
                    global_block_base: 4,
                    offset_z: 30,
                    offset_e: 40,
                    offset_t: 50,
                    offset_u: None,
                    offset_r: Some(60),
                },
            ],
            chunk_lengths: WitnessChunkLengths {
                z_chunk_len: 10,
                e_chunk_len: 10,
                t_chunk_len: 10,
                u_chunk_len: 0,
                r_chunk_len: 5,
            },
        };
        assert_eq!(layout.witness_ring_len().unwrap(), 65);
    }

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
