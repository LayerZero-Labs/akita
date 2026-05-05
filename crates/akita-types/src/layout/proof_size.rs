//! Header-stripped proof-size and planned-witness sizing formulas.

use crate::layout::digit_math::compute_num_digits_full_field;
use crate::stage1_tree_stage_shapes;
use crate::{DirectWitnessShape, LevelParams, Mode};
use akita_field::CanonicalField;

/// Field element size in bytes for a field with `field_bits` bits.
pub fn field_bytes(field_bits: u32) -> usize {
    (field_bits as usize).div_ceil(8)
}

/// Ring vector bytes without a length prefix.
pub fn proof_ring_vec_bytes(ring_len: usize, ring_dim: usize, elem_bytes: usize) -> usize {
    ring_len.saturating_mul(ring_dim).saturating_mul(elem_bytes)
}

/// Packed digit bytes without a length/tag prefix.
pub fn packed_digits_bytes(num_elems: usize, bits_per_elem: u32) -> usize {
    num_elems.saturating_mul(bits_per_elem as usize).div_ceil(8)
}

/// Serialized byte size for a terminal direct witness shape.
pub fn direct_witness_bytes(field_bits: u32, shape: &DirectWitnessShape) -> usize {
    match shape {
        DirectWitnessShape::PackedDigits((num_elems, bits_per_elem)) => {
            packed_digits_bytes(*num_elems, *bits_per_elem)
        }
        DirectWitnessShape::FieldElements(num_coeffs) => {
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

fn stage1_proof_bytes(rounds: usize, b: usize, elem_bytes: usize) -> usize {
    stage1_tree_stage_shapes(rounds, b)
        .into_iter()
        .map(|stage| {
            sumcheck_bytes(rounds, stage.sumcheck.1, elem_bytes) + stage.child_claims * elem_bytes
        })
        .sum::<usize>()
        + elem_bytes
}

/// Planned recursive witness size in ring elements for a singleton fold under
/// a concrete masking mode.
pub fn planned_w_ring_element_count<F, M>(field_bits: u32, lp: &LevelParams) -> usize
where
    F: CanonicalField,
    M: Mode,
{
    let w_hat_count = lp.num_blocks * lp.num_digits_open;
    let t_hat_count = lp.num_blocks * lp.a_key.row_len() * lp.num_digits_open;
    let blind_count =
        M::blind_column_count::<F>(lp.b_key.row_len(), lp.ring_dimension, lp.num_digits_open);
    let z_pre_count = lp.inner_width() * lp.num_digits_fold;
    let r_count = lp.m_row_count(1, 1) * compute_num_digits_full_field(field_bits, lp.log_basis);
    w_hat_count + t_hat_count + blind_count + z_pre_count + r_count
}

/// Planned recursive witness size in field elements for a singleton fold under
/// a concrete masking mode.
pub fn planned_next_w_len<F, M>(field_bits: u32, lp: &LevelParams) -> usize
where
    F: CanonicalField,
    M: Mode,
{
    planned_w_ring_element_count::<F, M>(field_bits, lp) * lp.ring_dimension
}

/// Total sumcheck rounds (`col_bits + ring_bits`) for one fold level.
pub fn sumcheck_rounds(level_d: usize, next_w_len: usize) -> usize {
    let ring_bits = level_d.trailing_zeros() as usize;
    let num_ring_elems = next_w_len / level_d;
    let col_bits = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    col_bits + ring_bits
}

/// Header-stripped byte size of one folded proof level.
pub fn level_proof_bytes(
    field_bits: u32,
    lp: &LevelParams,
    level_lp: &LevelParams,
    next_lp: &LevelParams,
    next_w_len: usize,
    num_claims: usize,
) -> usize {
    let elem_bytes = field_bytes(field_bits);
    let y_bytes = proof_ring_vec_bytes(num_claims, lp.ring_dimension, elem_bytes);
    let v_bytes = proof_ring_vec_bytes(lp.d_key.row_len(), lp.ring_dimension, elem_bytes);
    let next_commit_bytes =
        proof_ring_vec_bytes(next_lp.b_key.row_len(), next_lp.ring_dimension, elem_bytes);
    let next_eval_bytes = elem_bytes;
    let rounds = sumcheck_rounds(lp.ring_dimension, next_w_len);
    let b = 1usize << level_lp.log_basis;
    let stage1_bytes = stage1_proof_bytes(rounds, b, elem_bytes);

    y_bytes
        + v_bytes
        + stage1_bytes
        + sumcheck_bytes(rounds, 3, elem_bytes)
        + next_commit_bytes
        + next_eval_bytes
}
