//! Compile-time ZK commitment masking helpers.

use akita_field::CanonicalField;

/// Statistical security target used by the LHL hiding mask.
pub const LHL_STATISTICAL_SECURITY_BITS: usize = 128;

/// Number of fresh blind ring elements needed for an output in
/// `R_q^{output_ring_len}` when compiled with the `zk` feature.
///
/// LHL requires source min-entropy at least output bits plus
/// `2 * lambda - 2`; this helper treats `field_bits` as the modulus bit length
/// and uses `field_bits - 1` as a conservative lower bound for `log2(q)`.
pub fn blind_ring_count_from_bits(
    output_ring_len: usize,
    ring_dimension: usize,
    field_bits: usize,
) -> usize {
    if output_ring_len == 0 {
        return 0;
    }
    let log2_q_floor_bits = field_bits.saturating_sub(1).max(1);
    let entropy_per_ring = ring_dimension.saturating_mul(log2_q_floor_bits).max(1);
    let lhl_slack = 2 * LHL_STATISTICAL_SECURITY_BITS - 2;
    output_ring_len + lhl_slack.div_ceil(entropy_per_ring)
}

/// Number of fresh blind ring elements needed for an output in
/// `R_q^{output_ring_len}`.
pub fn blind_ring_count<F: CanonicalField>(output_ring_len: usize, ring_dimension: usize) -> usize {
    blind_ring_count_from_bits(output_ring_len, ring_dimension, F::modulus_bits() as usize)
}

/// Number of B-matrix columns reserved for the fresh blinding vector.
pub fn blind_column_count<F: CanonicalField>(
    output_ring_len: usize,
    ring_dimension: usize,
    num_digits_open: usize,
) -> usize {
    blind_ring_count::<F>(output_ring_len, ring_dimension).saturating_mul(num_digits_open)
}

/// Number of B-matrix columns reserved for the fresh blinding vector when only
/// the field bit length is available.
pub fn blind_column_count_from_bits(
    output_ring_len: usize,
    ring_dimension: usize,
    num_digits_open: usize,
    field_bits: u32,
) -> usize {
    blind_ring_count_from_bits(output_ring_len, ring_dimension, field_bits as usize)
        .saturating_mul(num_digits_open)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blind_ring_count_uses_conservative_entropy_floor() {
        assert_eq!(blind_ring_count_from_bits(3, 32, 128), 4);
        assert_eq!(blind_ring_count_from_bits(3, 64, 128), 4);
        assert_eq!(blind_ring_count_from_bits(3, 128, 128), 4);
        assert_eq!(blind_ring_count_from_bits(1, 1, 2), 255);
    }

    #[test]
    fn zero_output_needs_no_blinding_columns() {
        assert_eq!(blind_ring_count_from_bits(0, 32, 128), 0);
        assert_eq!(blind_column_count_from_bits(0, 32, 43, 128), 0);
    }
}
