//! Commitment masking modes.

use akita_field::CanonicalField;

/// Statistical security target used by the LHL hiding mask.
pub const LHL_STATISTICAL_SECURITY_BITS: usize = 128;

/// Compile-time commitment masking behavior.
pub trait Mode: Clone + Copy + Send + Sync + 'static {
    /// Whether this mode applies fresh LHL hiding masks.
    const ZK_ENABLED: bool;

    /// Number of fresh blind ring elements needed for an output in
    /// `R_q^{output_ring_len}`.
    ///
    /// LHL requires source min-entropy at least output bits plus
    /// `2 * lambda - 2`; one uniformly sampled ring element contributes
    /// `D * log2(q)` bits up to rounding.
    fn blind_ring_count<F: CanonicalField>(output_ring_len: usize, ring_dimension: usize) -> usize;

    /// Number of B-matrix columns reserved for the fresh blinding vector.
    fn blind_column_count<F: CanonicalField>(
        output_ring_len: usize,
        ring_dimension: usize,
        num_digits_open: usize,
    ) -> usize {
        Self::blind_ring_count::<F>(output_ring_len, ring_dimension).saturating_mul(num_digits_open)
    }
}

/// Transparent, non-hiding commitments.
#[derive(Clone, Copy, Debug, Default)]
pub struct Transparent;

impl Mode for Transparent {
    const ZK_ENABLED: bool = false;

    fn blind_ring_count<F: CanonicalField>(
        _output_ring_len: usize,
        _ring_dimension: usize,
    ) -> usize {
        0
    }
}

/// ZK commitments with fresh LHL hiding masks.
#[cfg(feature = "zk")]
#[derive(Clone, Copy, Debug, Default)]
pub struct ZK;

#[cfg(feature = "zk")]
impl Mode for ZK {
    const ZK_ENABLED: bool = true;

    fn blind_ring_count<F: CanonicalField>(output_ring_len: usize, ring_dimension: usize) -> usize {
        if output_ring_len == 0 {
            return 0;
        }
        let entropy_per_ring = ring_dimension
            .saturating_mul(F::modulus_bits() as usize)
            .max(1);
        let lhl_slack = 2 * LHL_STATISTICAL_SECURITY_BITS - 2;
        output_ring_len + lhl_slack.div_ceil(entropy_per_ring)
    }
}
