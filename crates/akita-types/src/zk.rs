//! Compile-time ZK commitment masking helpers.

use akita_field::CanonicalField;

/// Statistical security target used by the LHL hiding mask.
pub const LHL_STATISTICAL_SECURITY_BITS: usize = 128;

/// Number of fresh blind ring elements needed for an output in
/// `R_q^{output_ring_len}` when compiled with the `zk` feature.
///
/// LHL requires source min-entropy at least output bits plus
/// `2 * lambda - 2`; one uniformly sampled ring element contributes
/// `D * log2(q)` bits up to rounding.
pub fn blind_ring_count_from_bits(
    output_ring_len: usize,
    ring_dimension: usize,
    field_bits: usize,
) -> usize {
    if output_ring_len == 0 {
        return 0;
    }
    let entropy_per_ring = ring_dimension.saturating_mul(field_bits).max(1);
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
