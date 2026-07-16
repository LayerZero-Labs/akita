//! Shared stage-1 tree shape and polynomial helpers.

use crate::AkitaStage1StageShape;
use akita_error::AkitaError;
use akita_transcript::{labels, Transcript};
use jolt_field::{CanonicalField, FieldCore, FromPrimitiveInt};

/// Validate the stage-1 range basis.
///
/// # Errors
///
/// Returns an error if `b` is not a power-of-two basis at least 4.
pub fn validate_stage1_tree_basis(b: usize) -> Result<(), AkitaError> {
    if b < 4 || !b.is_power_of_two() {
        return Err(AkitaError::InvalidInput(format!(
            "stage1 tree requires a power-of-two basis >= 4, got {b}"
        )));
    }
    Ok(())
}

fn stage1_root_values<E: FieldCore + FromPrimitiveInt>(b: usize) -> Vec<E> {
    let half = b / 2;
    (0..half)
        .map(|k| {
            let k = k as i64;
            E::from_i64(k * (k + 1))
        })
        .collect()
}

fn poly_coeffs_from_roots<E: FieldCore>(roots: &[E]) -> Vec<E> {
    let mut coeffs = vec![E::one()];
    for &root in roots {
        let mut next = vec![E::zero(); coeffs.len() + 1];
        for (idx, &coeff) in coeffs.iter().enumerate() {
            next[idx] -= coeff * root;
            next[idx + 1] += coeff;
        }
        coeffs = next;
    }
    coeffs
}

/// Evaluate a small polynomial stored as coefficient slices.
pub fn eval_poly<E: FieldCore>(coeffs: &[E], x: E) -> E {
    coeffs
        .iter()
        .rev()
        .copied()
        .fold(E::zero(), |acc, coeff| acc * x + coeff)
}

/// Evaluate the full stage-1 range-check polynomial at `s`.
pub fn range_check_eval_from_s<E: FieldCore + FromPrimitiveInt>(s: E, b: usize) -> E {
    let half = (b / 2) as i64;
    let mut acc = E::one();
    for k in 0..half {
        acc *= s - E::from_i64(k * (k + 1));
    }
    acc
}

/// Reorder ring-switch coordinates into the stage-1 table coordinate order.
///
/// Ring-switch samples coordinates as columns followed by ring slots. Stage 1
/// stores the virtual table with ring-slot coordinates first, then columns.
///
/// # Panics
///
/// Panics if `coords.len() != col_bits + ring_bits`.
pub fn reorder_stage1_coords<F: FieldCore>(
    coords: &[F],
    col_bits: usize,
    ring_bits: usize,
) -> Vec<F> {
    assert_eq!(coords.len(), col_bits + ring_bits);
    let mut reordered = Vec::with_capacity(coords.len());
    reordered.extend_from_slice(&coords[col_bits..]);
    reordered.extend_from_slice(&coords[..col_bits]);
    reordered
}

fn stage1_leaf_groups<E: FieldCore + FromPrimitiveInt>(b: usize) -> Vec<Vec<E>> {
    stage1_root_values::<E>(b)
        .chunks(4)
        .map(|chunk| chunk.to_vec())
        .collect()
}

/// Return the quartic leaf polynomial coefficients for the stage-1 tree.
pub fn stage1_leaf_coeffs<E: FieldCore + FromPrimitiveInt>(b: usize) -> Vec<Vec<E>> {
    stage1_leaf_groups::<E>(b)
        .into_iter()
        .map(|roots| poly_coeffs_from_roots(&roots))
        .collect()
}

fn stage1_tree_binary_levels(b: usize) -> usize {
    debug_assert!(b >= 4 && b.is_power_of_two());
    b.trailing_zeros() as usize - 1
}

fn stage1_tree_stage_arities(b: usize) -> Vec<usize> {
    debug_assert!(b > 8 && b.is_power_of_two());
    let binary_levels = stage1_tree_binary_levels(b);
    let mut out = Vec::with_capacity(binary_levels.div_ceil(2));
    if binary_levels % 2 == 1 {
        out.push(2);
    }
    out.extend(std::iter::repeat_n(4, binary_levels / 2));
    out
}

/// Return product-stage arities before the leaf stage.
pub fn stage1_tree_product_stage_arities(b: usize) -> Vec<usize> {
    let mut out = stage1_tree_stage_arities(b);
    out.pop();
    out
}

fn stage1_leaf_factor_count(b: usize) -> usize {
    debug_assert!(b >= 8 && b.is_power_of_two());
    b / 8
}

/// Return the wire shapes for all stage-1 tree subproofs.
pub fn stage1_tree_stage_shapes(rounds: usize, b: usize) -> Vec<AkitaStage1StageShape> {
    debug_assert!(b >= 4 && b.is_power_of_two());
    if b <= 8 {
        return vec![AkitaStage1StageShape {
            sumcheck_proof: (rounds, b / 2),
            child_claims: 0,
        }];
    }

    let mut parent_count = 1usize;
    let mut out = Vec::new();
    for arity in stage1_tree_product_stage_arities(b) {
        let child_claims = parent_count * arity;
        out.push(AkitaStage1StageShape {
            sumcheck_proof: (rounds, arity),
            child_claims,
        });
        parent_count = child_claims;
    }
    debug_assert_eq!(parent_count, stage1_leaf_factor_count(b));
    out.push(AkitaStage1StageShape {
        sumcheck_proof: (rounds, 4),
        child_claims: 0,
    });
    out
}

/// Return the number of stage-1 tree subproofs for basis `b`.
pub fn stage1_stage_count(b: usize) -> usize {
    stage1_tree_stage_shapes(0, b).len()
}

/// Return powers of an interstage batching challenge.
pub fn stage1_interstage_batch_weights<E: FieldCore>(gamma: E, count: usize) -> Vec<E> {
    let mut out = Vec::with_capacity(count);
    let mut weight = E::one();
    for _ in 0..count {
        out.push(weight);
        weight *= gamma;
    }
    out
}

/// Form a weighted linear combination of polynomial coefficient vectors.
pub fn combine_polys<E: FieldCore>(weights: &[E], polys: &[Vec<E>]) -> Vec<E> {
    debug_assert_eq!(weights.len(), polys.len());
    let max_len = polys.iter().map(Vec::len).max().unwrap_or(0);
    let mut out = vec![E::zero(); max_len];
    for (weight, poly) in weights.iter().zip(polys.iter()) {
        for (idx, &coeff) in poly.iter().enumerate() {
            out[idx] += *weight * coeff;
        }
    }
    out
}

/// Form a weighted linear combination of scalar claims.
pub fn linear_combination<E: FieldCore>(weights: &[E], values: &[E]) -> E {
    debug_assert_eq!(weights.len(), values.len());
    weights
        .iter()
        .zip(values.iter())
        .fold(E::zero(), |acc, (&weight, &value)| acc + weight * value)
}

/// Absorb stage-1 interstage child claims into the transcript.
pub fn absorb_interstage_claims<F: FieldCore + CanonicalField, T: Transcript<F>>(
    claims: &[F],
    transcript: &mut T,
) {
    for claim in claims {
        transcript.append_field(labels::ABSORB_SUMCHECK_INTERSTAGE_CLAIM, claim);
    }
}
