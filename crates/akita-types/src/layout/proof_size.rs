//! Header-stripped proof-size and planned-witness sizing formulas.

use akita_field::{AkitaError, CanonicalField};

use crate::layout::digit_math::compute_num_digits_full_field;
use crate::{CleartextWitnessShape, LevelParams, EXTENSION_OPENING_REDUCTION_DEGREE};

/// Field element size in bytes for a field with `field_bits` bits.
pub fn field_bytes(field_bits: u32) -> usize {
    (field_bits as usize).div_ceil(8)
}

/// Ring vector bytes without a length prefix.
pub fn proof_ring_vec_bytes(ring_len: usize, ring_dim: usize, elem_bytes: usize) -> usize {
    ring_len.saturating_mul(ring_dim).saturating_mul(elem_bytes)
}

/// Number of root extension-opening reduction partials sent on the wire.
pub fn root_extension_opening_partials(
    claim_ext_degree: usize,
    num_reduced_opening_rows: usize,
) -> usize {
    claim_ext_degree.saturating_mul(num_reduced_opening_rows)
}

/// Packed digit bytes without a length/tag prefix.
pub fn packed_digits_bytes(num_elems: usize, bits_per_elem: u32) -> usize {
    num_elems.saturating_mul(bits_per_elem as usize).div_ceil(8)
}

/// Serialized byte size for a terminal direct witness shape.
pub fn direct_witness_bytes(field_bits: u32, shape: &CleartextWitnessShape) -> usize {
    match shape {
        CleartextWitnessShape::PackedDigits((num_elems, bits_per_elem)) => {
            packed_digits_bytes(*num_elems, *bits_per_elem)
        }
        CleartextWitnessShape::FieldElements(num_coeffs) => {
            num_coeffs.saturating_mul(field_bytes(field_bits))
        }
    }
}

fn compressed_unipoly_bytes(degree: usize, elem_bytes: usize) -> usize {
    degree * elem_bytes
}

fn sumcheck_bytes(rounds: usize, degree: usize, elem_bytes: usize) -> usize {
    rounds * compressed_unipoly_bytes(degree, elem_bytes)
}

/// Header-stripped byte size of an extension-opening reduction proof.
///
/// The reduction proof serializes `partials` challenge-field elements followed
/// by a fixed degree-two sumcheck over `opening_vars - log2(extension_width)`
/// rounds. `extension_width = 1` means the claim field is already the base
/// field and contributes zero bytes.
///
/// # Errors
///
/// Returns an error when `extension_width` is not a power of two or when the
/// tensor split is wider than the opened Boolean cube.
pub fn extension_opening_reduction_proof_bytes(
    challenge_field_bits: u32,
    partials: usize,
    opening_vars: usize,
    extension_width: usize,
) -> Result<usize, AkitaError> {
    if extension_width <= 1 {
        return Ok(0);
    }
    if !extension_width.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(format!(
            "extension opening width must be a power of two, got {extension_width}"
        )));
    }
    let split_bits = extension_width.trailing_zeros() as usize;
    if split_bits > opening_vars {
        return Err(AkitaError::InvalidSetup(format!(
            "extension opening split ({split_bits}) exceeds opening variables ({opening_vars})"
        )));
    }
    let elem_bytes = field_bytes(challenge_field_bits);
    let rounds = opening_vars - split_bits;
    Ok(partials.saturating_mul(elem_bytes).saturating_add({
        #[cfg(feature = "zk")]
        {
            sumcheck_bytes(rounds, EXTENSION_OPENING_REDUCTION_DEGREE, elem_bytes)
        }
        #[cfg(not(feature = "zk"))]
        {
            sumcheck_bytes(rounds, EXTENSION_OPENING_REDUCTION_DEGREE, elem_bytes)
        }
    }))
}

/// Planned recursive witness size in ring elements for a singleton fold.
pub fn planned_w_ring_element_count<F: CanonicalField>(
    field_bits: u32,
    lp: &LevelParams,
) -> Result<usize, AkitaError> {
    let _field_marker = core::marker::PhantomData::<F>;
    let w_hat_count = lp
        .num_blocks
        .checked_mul(lp.num_digits_open)
        .ok_or_else(|| AkitaError::InvalidSetup("planned W width overflow".to_string()))?;
    let t_hat_count = lp
        .num_blocks
        .checked_mul(lp.a_key.row_len())
        .and_then(|n| n.checked_mul(lp.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("planned T width overflow".to_string()))?;
    let z_pre_count = lp
        .inner_width()
        .checked_mul(lp.num_digits_fold(1, field_bits))
        .ok_or_else(|| AkitaError::InvalidSetup("planned Z width overflow".to_string()))?;
    let r_count = lp
        .m_row_count(1, 1)?
        .checked_mul(compute_num_digits_full_field(field_bits, lp.log_basis))
        .ok_or_else(|| AkitaError::InvalidSetup("planned r-tail width overflow".to_string()))?;

    #[cfg(feature = "zk")]
    {
        let d_blinding_count = crate::zk::blinding_column_count_from_bits(
            lp.d_key.row_len(),
            lp.ring_dimension,
            lp.log_basis,
            field_bits as usize,
        );
        let b_blinding_count = crate::zk::blinding_column_count_from_bits(
            lp.b_key.row_len(),
            lp.ring_dimension,
            lp.log_basis,
            field_bits as usize,
        );
        w_hat_count
            .checked_add(t_hat_count)
            .and_then(|n| n.checked_add(b_blinding_count))
            .and_then(|n| n.checked_add(d_blinding_count))
            .and_then(|n| n.checked_add(z_pre_count))
            .and_then(|n| n.checked_add(r_count))
            .ok_or_else(|| AkitaError::InvalidSetup("planned witness width overflow".to_string()))
    }
    #[cfg(not(feature = "zk"))]
    {
        w_hat_count
            .checked_add(t_hat_count)
            .and_then(|n| n.checked_add(z_pre_count))
            .and_then(|n| n.checked_add(r_count))
            .ok_or_else(|| AkitaError::InvalidSetup("planned witness width overflow".to_string()))
    }
}

/// Planned recursive witness size in field elements for a singleton fold.
pub fn planned_next_w_len<F: CanonicalField>(
    field_bits: u32,
    lp: &LevelParams,
) -> Result<usize, AkitaError> {
    planned_w_ring_element_count::<F>(field_bits, lp)?
        .checked_mul(lp.ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup("planned next witness length overflow".to_string()))
}

/// Total sumcheck rounds (`col_bits + ring_bits`) for one fold level.
pub fn sumcheck_rounds(level_d: usize, next_w_len: usize) -> usize {
    let ring_bits = level_d.trailing_zeros() as usize;
    let num_ring_elems = next_w_len / level_d;
    let col_bits = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    col_bits + ring_bits
}
