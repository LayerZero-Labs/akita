//! Shared setup-matrix envelope accumulation for certification paths.

use akita_field::AkitaError;
use akita_types::{LevelParams, SetupMatrixEnvelope};

/// Extend `max_setup_len` with the per-level A/B/D key footprint.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] on overflow.
pub(crate) fn accumulate_matrix_envelope_for_level(
    lp: &LevelParams,
    max_setup_len: &mut usize,
) -> Result<(), AkitaError> {
    let a_len = lp
        .a_key
        .row_len()
        .checked_mul(lp.inner_width())
        .ok_or_else(|| AkitaError::InvalidSetup("A setup envelope overflow".to_string()))?;
    let b_len = lp
        .b_key
        .row_len()
        .checked_mul(lp.outer_width())
        .ok_or_else(|| AkitaError::InvalidSetup("B setup envelope overflow".to_string()))?;
    let d_len = lp
        .d_key
        .row_len()
        .checked_mul(lp.d_matrix_width())
        .ok_or_else(|| AkitaError::InvalidSetup("D setup envelope overflow".to_string()))?;
    *max_setup_len = (*max_setup_len).max(a_len).max(b_len).max(d_len);
    Ok(())
}

/// Inflate a setup envelope for a compression flat-prefix requirement.
///
/// `max_flat_prefix_coeffs` is rounded up to whole generation rings at
/// `gen_ring_dim` via [`akita_types::compression_prefix_rings`], the same
/// ceil-to-rings helper used by compression setup compilation.
pub fn inflate_setup_envelope_for_compression_prefix(
    envelope: &mut SetupMatrixEnvelope,
    max_flat_prefix_coeffs: usize,
    gen_ring_dim: usize,
) -> Result<(), AkitaError> {
    let compression_prefix_rings =
        akita_types::compression_prefix_rings(max_flat_prefix_coeffs, gen_ring_dim)?;
    envelope.max_setup_len = envelope.max_setup_len.max(compression_prefix_rings);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inflate_rounds_flat_prefix_to_generation_rings() {
        let mut envelope = SetupMatrixEnvelope { max_setup_len: 1 };
        inflate_setup_envelope_for_compression_prefix(&mut envelope, 96, 64).expect("inflate");
        assert_eq!(envelope.max_setup_len, 2);
    }

    #[test]
    fn inflate_keeps_larger_existing_envelope() {
        let mut envelope = SetupMatrixEnvelope { max_setup_len: 4 };
        inflate_setup_envelope_for_compression_prefix(&mut envelope, 96, 64).expect("inflate");
        assert_eq!(envelope.max_setup_len, 4);
    }

    #[test]
    fn inflate_rejects_zero_generation_dimension() {
        let mut envelope = SetupMatrixEnvelope { max_setup_len: 1 };
        assert!(inflate_setup_envelope_for_compression_prefix(&mut envelope, 96, 0).is_err());
    }
}
