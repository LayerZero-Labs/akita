//! LHL hiding-mask capacity math for the optional hiding-commitment layer.
//!
//! This module is always compiled. The transparent protocol does not use it;
//! it documents the digit-plane sizing discipline and anchors post-audit ZK work.

use akita_field::CanonicalField;

/// Statistical security target used by the LHL hiding mask.
pub const LHL_STATISTICAL_SECURITY_BITS: usize = 128;

/// Number of fresh digit-ring planes needed for an output in
/// `R_q^{output_ring_len}`.
///
/// The digit-source LHL target is joint over the public hash seed and output,
/// `Delta((B, h_B(S)), (B, U))`.  For `kappa = output_ring_len`, each directly
/// sampled digit plane contributes `D * log_basis` bits, so the conservative
/// count is
/// `ceil((kappa * D * field_bits + 2 * lambda - 2) / (D * log_basis))`.
pub fn blinding_digit_plane_count_from_bits(
    output_ring_len: usize,
    ring_dimension: usize,
    log_basis: u32,
    field_bits: usize,
) -> usize {
    if output_ring_len == 0 {
        return 0;
    }
    let entropy_per_plane = ring_dimension.saturating_mul(log_basis as usize);
    if entropy_per_plane == 0 {
        return 0;
    }
    let lhl_slack = 2 * LHL_STATISTICAL_SECURITY_BITS - 2;
    output_ring_len
        .saturating_mul(ring_dimension)
        .saturating_mul(field_bits)
        .saturating_add(lhl_slack)
        .div_ceil(entropy_per_plane)
}

/// Number of fresh digit-ring planes needed for an output in
/// `R_q^{output_ring_len}`.
pub fn blinding_digit_plane_count<F: CanonicalField>(
    output_ring_len: usize,
    ring_dimension: usize,
    log_basis: u32,
) -> usize {
    blinding_digit_plane_count_from_bits(
        output_ring_len,
        ring_dimension,
        log_basis,
        F::modulus_bits() as usize,
    )
}

/// Number of B-matrix columns reserved for the fresh digit-source blinding.
pub fn blinding_column_count<F: CanonicalField>(
    output_ring_len: usize,
    ring_dimension: usize,
    log_basis: u32,
) -> usize {
    blinding_digit_plane_count::<F>(output_ring_len, ring_dimension, log_basis)
}

/// Number of B-matrix columns reserved for the fresh digit-source blinding when
/// only the field bit length is available.
pub fn blinding_column_count_from_bits(
    output_ring_len: usize,
    ring_dimension: usize,
    log_basis: u32,
    field_bits: usize,
) -> usize {
    blinding_digit_plane_count_from_bits(output_ring_len, ring_dimension, log_basis, field_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digit_plane_count_uses_direct_lhl_entropy() {
        // ceil((2 * 32 * 128 + 254) / (32 * 5)) = ceil(8446 / 160) = 53.
        assert_eq!(blinding_digit_plane_count_from_bits(2, 32, 5, 128), 53);
        // ceil((1 * 128 * 128 + 254) / (128 * 4)) = ceil(16638 / 512) = 33.
        assert_eq!(blinding_digit_plane_count_from_bits(1, 128, 4, 128), 33);
    }

    #[test]
    fn small_dimensions_can_need_many_digit_planes() {
        // ceil((3 * 8 * 8 + 254) / (8 * 2)) = ceil(446 / 16) = 28.
        assert_eq!(blinding_digit_plane_count_from_bits(3, 8, 2, 8), 28);
    }

    #[test]
    fn column_count_is_digit_plane_count() {
        assert_eq!(blinding_column_count_from_bits(3, 8, 2, 8), 28);
    }

    #[test]
    fn zero_output_needs_no_digit_planes() {
        assert_eq!(blinding_digit_plane_count_from_bits(0, 32, 4, 128), 0);
    }

    #[test]
    fn default_fp128_examples_match_spec() {
        assert_eq!(blinding_digit_plane_count_from_bits(1, 64, 5, 128), 27);
        assert_eq!(blinding_digit_plane_count_from_bits(1, 128, 5, 128), 26);
        assert_eq!(blinding_digit_plane_count_from_bits(1, 64, 4, 128), 33);
    }

    #[test]
    fn zero_output_needs_no_blinding_columns() {
        assert_eq!(blinding_column_count_from_bits(0, 32, 43, 128), 0);
    }
}
