//! Helpers for transcript-binding terminal cleartext witnesses.

use akita_field::{AkitaError, FieldCore};

use super::FlatDigitBlocks;
use crate::LevelParams;

/// Logical digit range occupied by terminal `e_hat` inside the final witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerminalWitnessSegmentLayout {
    /// Offset of the terminal `e_hat` digit range in final-witness order.
    pub e_hat_digit_offset: usize,
    /// Number of logical terminal `e_hat` digits.
    pub e_hat_digit_count: usize,
}

/// Transcript byte slices for terminal direct-witness replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalWitnessTranscriptParts {
    /// Logical terminal `e_hat` bytes, bound before sparse challenge sampling.
    pub e_hat: Vec<u8>,
    /// Remaining final-witness bytes, bound before ring-switch challenges.
    pub remainder: Vec<u8>,
}

/// Stage-2 inputs for terminal relation-only replay.
///
/// Terminal folds have no stage-1 norm-check claim. Setting
/// `batching_coeff = 0` removes the virtual norm contribution from every
/// stage-2 round, so `s_claim` and `stage1_point` are structural zeros.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationOnlyStage2Inputs<E: FieldCore> {
    /// Zero coefficient for the omitted virtual norm-check oracle.
    pub batching_coeff: E,
    /// Zero claim for the omitted stage-1 sumcheck.
    pub s_claim: E,
    /// Zero challenge vector with length `col_bits + ring_bits`.
    pub stage1_point: Vec<E>,
}

impl<E: FieldCore> RelationOnlyStage2Inputs<E> {
    /// Build the terminal relation-only stage-2 input bundle.
    #[must_use]
    pub fn new(num_stage1_vars: usize) -> Self {
        Self {
            batching_coeff: E::zero(),
            s_claim: E::zero(),
            stage1_point: vec![E::zero(); num_stage1_vars],
        }
    }
}

impl TerminalWitnessSegmentLayout {
    /// Exclusive end of the `e_hat` digit range.
    ///
    /// # Errors
    ///
    /// Returns an error on arithmetic overflow.
    pub fn e_hat_digit_end(&self) -> Result<usize, AkitaError> {
        self.e_hat_digit_offset
            .checked_add(self.e_hat_digit_count)
            .ok_or_else(|| AkitaError::InvalidSetup("terminal e_hat range overflow".to_string()))
    }
}

/// Convert signed terminal digits to their canonical transcript byte encoding.
#[must_use]
pub fn i8_digits_to_bytes(digits: &[i8]) -> Vec<u8> {
    digits.iter().copied().map(|digit| digit as u8).collect()
}

/// Split a terminal final-witness digit stream into transcript-bound slices.
///
/// # Errors
///
/// Returns an error when the descriptor-bound terminal segment is out of range
/// or either transcript-bound slice would be empty.
pub fn terminal_witness_transcript_parts(
    digits: &[i8],
    layout: TerminalWitnessSegmentLayout,
) -> Result<TerminalWitnessTranscriptParts, AkitaError> {
    let e_hat_start = layout.e_hat_digit_offset;
    let e_hat_end = layout.e_hat_digit_end()?;
    if e_hat_end > digits.len() {
        return Err(AkitaError::InvalidProof);
    }

    let e_hat = i8_digits_to_bytes(&digits[e_hat_start..e_hat_end]);
    let remainder_len = digits
        .len()
        .checked_sub(layout.e_hat_digit_count)
        .ok_or(AkitaError::InvalidProof)?;
    let mut remainder = Vec::with_capacity(remainder_len);
    remainder.extend(i8_digits_to_bytes(&digits[..e_hat_start]));
    remainder.extend(i8_digits_to_bytes(&digits[e_hat_end..]));
    if e_hat.is_empty() || remainder.is_empty() {
        return Err(AkitaError::InvalidProof);
    }
    Ok(TerminalWitnessTranscriptParts { e_hat, remainder })
}

/// Encode a prover-held terminal `e_hat` block collection in the same logical
/// order used by terminal witness replay.
///
/// `FlatDigitBlocks` stores each block's digit planes contiguously. The
/// terminal transcript binds digits by plane across all blocks so that this
/// prover-side encoding matches the packed final-witness segment replayed by
/// the verifier.
///
/// # Errors
///
/// Returns an error when the block layout is malformed or the logical
/// terminal `e_hat` byte stream would be empty.
pub fn terminal_e_hat_bytes_from_blocks<const D: usize>(
    e_hat: &FlatDigitBlocks<D>,
    planes_per_block: usize,
) -> Result<Vec<u8>, AkitaError> {
    let total_blocks = e_hat.block_count();
    let expected_planes = total_blocks
        .checked_mul(planes_per_block)
        .ok_or_else(|| AkitaError::InvalidSetup("terminal e_hat width overflow".to_string()))?;
    if planes_per_block == 0 || e_hat.flat_digits().len() != expected_planes {
        return Err(AkitaError::InvalidInput(
            "terminal e_hat block layout does not match open digit depth".to_string(),
        ));
    }

    let total_digits = expected_planes
        .checked_mul(D)
        .ok_or_else(|| AkitaError::InvalidSetup("terminal e_hat byte overflow".to_string()))?;
    let mut bytes = Vec::with_capacity(total_digits);
    for compound_dig in 0..planes_per_block {
        for block in 0..total_blocks {
            bytes.extend(
                e_hat.flat_digits()[block * planes_per_block + compound_dig]
                    .iter()
                    .copied()
                    .map(|digit| digit as u8),
            );
        }
    }
    if bytes.is_empty() {
        return Err(AkitaError::InvalidInput(
            "terminal e_hat absorb cannot be empty".to_string(),
        ));
    }
    Ok(bytes)
}

/// Derive the terminal `e_hat` byte range from the two prefix segments that can
/// precede it in the final witness.
///
/// This is the shared layout primitive used by both prover witness emission and
/// verifier/prover transcript slicing.
///
/// # Errors
///
/// Returns an error when counts overflow, the ring dimension is invalid, or the
/// logical `e_hat` range would be empty.
pub fn terminal_witness_segment_layout_from_counts(
    ring_dimension: usize,
    z_first: bool,
    z_folded_ring_count: usize,
    e_hat_ring_count: usize,
) -> Result<TerminalWitnessSegmentLayout, AkitaError> {
    if ring_dimension == 0 {
        return Err(AkitaError::InvalidSetup(
            "terminal witness layout has zero ring dimension".to_string(),
        ));
    }
    let e_hat_digit_count = e_hat_ring_count
        .checked_mul(ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup("terminal e_hat digit overflow".to_string()))?;
    if e_hat_digit_count == 0 {
        return Err(AkitaError::InvalidSetup(
            "terminal e_hat digit range is empty".to_string(),
        ));
    }
    let e_hat_digit_offset = if z_first {
        z_folded_ring_count
            .checked_mul(ring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("terminal e_hat offset overflow".to_string()))?
    } else {
        0
    };
    Ok(TerminalWitnessSegmentLayout {
        e_hat_digit_offset,
        e_hat_digit_count,
    })
}

/// Derive the terminal logical `e_hat` digit range from descriptor-bound layout
/// data.
///
/// The terminal proof carries one canonical packed final witness, but
/// transcript replay binds the logical `e_hat` segment before sparse-seed
/// sampling. This helper mirrors the segment ordering used when
/// `ring_switch_build_w` emits final-witness coefficients.
///
/// # Errors
///
/// Returns an error when counts overflow or the level has an invalid ring
/// dimension.
pub fn terminal_witness_segment_layout(
    lp: &LevelParams,
    num_w_vectors: usize,
    num_public_rows: usize,
    field_bits: u32,
) -> Result<TerminalWitnessSegmentLayout, AkitaError> {
    if lp.ring_dimension == 0 {
        return Err(AkitaError::InvalidSetup(
            "terminal witness layout has zero ring dimension".to_string(),
        ));
    }
    let e_hat_ring_count = num_w_vectors
        .checked_mul(lp.num_blocks)
        .and_then(|n| n.checked_mul(lp.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("terminal e_hat width overflow".to_string()))?;
    let num_digits_fold = lp.num_digits_fold(num_w_vectors, field_bits)?;
    let z_folded_ring_count = num_public_rows
        .checked_mul(lp.inner_width())
        .and_then(|n| n.checked_mul(num_digits_fold))
        .ok_or_else(|| AkitaError::InvalidSetup("terminal z-folded width overflow".to_string()))?;
    terminal_witness_segment_layout_from_counts(
        lp.ring_dimension,
        crate::proof::ring_relation::ring_column_z_first(lp),
        z_folded_ring_count,
        e_hat_ring_count,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SisModulusFamily;
    use akita_challenges::SparseChallengeConfig;

    fn segment_test_params(m_vars: usize, r_vars: usize) -> LevelParams {
        LevelParams::params_only(
            SisModulusFamily::Q128,
            8,
            3,
            2,
            3,
            2,
            SparseChallengeConfig::Uniform {
                weight: 3,
                nonzero_coeffs: vec![-1, 1],
            },
        )
        .with_decomp(m_vars, r_vars, 2, 3, 0)
        .expect("segment test params")
    }

    #[test]
    fn terminal_witness_segment_layout_places_e_hat_after_z_when_z_first() {
        let lp = segment_test_params(3, 2);
        let layout = terminal_witness_segment_layout(&lp, 5, 2, 128).unwrap();

        assert_eq!(
            layout.e_hat_digit_offset,
            2 * lp.inner_width() * lp.num_digits_fold(5, 128).unwrap() * lp.ring_dimension
        );
        assert_eq!(
            layout.e_hat_digit_count,
            5 * lp.num_blocks * lp.num_digits_open * lp.ring_dimension
        );
    }

    #[test]
    fn terminal_witness_segment_layout_places_e_hat_first_when_w_first() {
        let lp = segment_test_params(1, 3);
        let layout = terminal_witness_segment_layout(&lp, 5, 2, 128).unwrap();

        assert_eq!(layout.e_hat_digit_offset, 0);
        assert_eq!(
            layout.e_hat_digit_count,
            5 * lp.num_blocks * lp.num_digits_open * lp.ring_dimension
        );
    }

    #[test]
    fn terminal_witness_transcript_parts_split_e_hat_and_remainder() {
        let layout = TerminalWitnessSegmentLayout {
            e_hat_digit_offset: 2,
            e_hat_digit_count: 3,
        };
        let digits = [-2, -1, 0, 1, 2, 3];
        let parts = terminal_witness_transcript_parts(&digits, layout).unwrap();

        assert_eq!(parts.e_hat, vec![0, 1, 2]);
        assert_eq!(parts.remainder, vec![254, 255, 3]);
    }

    #[test]
    fn terminal_witness_transcript_parts_reuses_canonical_split() {
        let layout = TerminalWitnessSegmentLayout {
            e_hat_digit_offset: 1,
            e_hat_digit_count: 2,
        };
        let digits = [-1, 0, 1, 2];
        let parts = terminal_witness_transcript_parts(&digits, layout).unwrap();

        assert_eq!(parts.e_hat, vec![0, 1]);
        assert_eq!(parts.remainder, vec![255, 2]);
    }

    #[test]
    fn terminal_witness_transcript_parts_rejects_bad_ranges() {
        let layout = TerminalWitnessSegmentLayout {
            e_hat_digit_offset: 2,
            e_hat_digit_count: 4,
        };
        let digits = [0, 1, 2];

        assert!(terminal_witness_transcript_parts(&digits, layout).is_err());
    }

    #[test]
    fn terminal_e_hat_bytes_from_blocks_uses_plane_major_order() {
        let e_hat = FlatDigitBlocks::from_blocks(vec![vec![[1, 2], [3, 4]], vec![[5, 6], [7, 8]]]);

        assert_eq!(
            terminal_e_hat_bytes_from_blocks(&e_hat, 2).unwrap(),
            vec![1, 2, 5, 6, 3, 4, 7, 8]
        );
    }
}
