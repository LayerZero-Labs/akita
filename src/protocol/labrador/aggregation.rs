//! Constraint aggregation for the Labrador protocol.
//!
//! Implements the "Aggregating" step from Section 5.2 of the LaBRADOR paper:
//! the 256 JL projection constraints and the existing statement constraints
//! are folded into a single aggregated constraint (φ_i, b) via random challenges.
//!
//! # Paper reference
//!
//! The JL constraints are first collapsed into ⌈128/log q⌉ functions using
//! 256 independent scalar collapse challenges per lift. Each collapsed
//! function produces a polynomial b''(k) whose constant term the verifier
//! can check. The prover sends b''(k) with the constant term zeroed out.
//! Note: If the field size is 128-bit or larger, instead of aggregating constraints
//! with random ring elements, we aggregate them with random field elements.
//! This diverges from the paper's protocol. For smaller fields, we precisely
//! follow the paper protocol.

use crate::algebra::ring::CyclotomicRing;
use crate::algebra::SparseChallenge;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::commitment::utils::linear::{
    mat_vec_mul_crt_ntt_i8_single, try_centered_i8_cache_from_ring_coeffs,
};
use crate::protocol::labrador::config::jl_lifts;
use crate::protocol::labrador::constraints::{pair_index, LabradorConstraint, NextWitnessLayout};
use crate::protocol::labrador::johnson_lindenstrauss::{
    restore_constant_term, zero_constant_term_for_proof, LabradorJlMatrix,
};
use crate::protocol::labrador::types::{
    LabradorReducedConstraintPlan, LabradorStatement, LabradorWitness,
};
use crate::protocol::labrador::utils::pow2_field;
use crate::protocol::transcript::labels;
use crate::protocol::transcript::{challenge_ring_element, Transcript};
use crate::{CanonicalField, FieldCore, FromSmallInt};

#[derive(Clone, Copy)]
enum AggregationRandomness<F: FieldCore, const D: usize> {
    /// Field is 128-bit or larger: aggregate with random field elements.
    Scalar(F),
    /// Field is smaller than 128-bit: aggregate with random ring elements.
    Ring(CyclotomicRing<F, D>),
}

#[inline]
/// Whether constraint aggregation may safely replace ring randomness with scalar randomness.
///
/// Security note: for prime moduli with bit-length greater than 128, we can
/// replace a ring-element challenge with a scalar field challenge and still keep
/// the claimed security level for the aggregation step.
pub(crate) fn safe_to_use_scalar_randomness<F: CanonicalField>() -> bool {
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let bits = u128::BITS - modulus.leading_zeros();
    bits == 128
}

#[inline]
fn sample_aggregation_randomness<F, T, const D: usize>(
    transcript: &mut T,
) -> AggregationRandomness<F, D>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    if safe_to_use_scalar_randomness::<F>() {
        AggregationRandomness::Scalar(
            transcript.challenge_scalar(labels::CHALLENGE_LABRADOR_AGGREGATION),
        )
    } else {
        AggregationRandomness::Ring(challenge_ring_element(
            transcript,
            labels::CHALLENGE_LABRADOR_AGGREGATION,
        ))
    }
}

type AggregatedConstraintSystem<F, const D: usize> =
    (Vec<Vec<CyclotomicRing<F, D>>>, CyclotomicRing<F, D>);

#[cfg(feature = "parallel")]
const STATEMENT_ROW_CHUNK_LEN: usize = 256;
const SPARSE_RING_MUL_MAX_WEIGHT: usize = 48;

/// Inner product of two ring-element slices.
pub(crate) fn dot_product<F: FieldCore, const D: usize>(
    lhs: &[CyclotomicRing<F, D>],
    rhs: &[CyclotomicRing<F, D>],
) -> CyclotomicRing<F, D> {
    let len = lhs.len().min(rhs.len());
    cfg_fold_reduce!(
        (0..len),
        || CyclotomicRing::<F, D>::zero(),
        |acc, i| acc + lhs[i] * rhs[i],
        |a, b| a + b
    )
}

/// Element-wise accumulate a flat `other` view into row-structured `acc`.
#[tracing::instrument(skip_all, name = "labrador::add_phi_flat_in_place")]
pub(crate) fn add_phi_flat_in_place<F: FieldCore, const D: usize>(
    acc: &mut [Vec<CyclotomicRing<F, D>>],
    other_flat: &[CyclotomicRing<F, D>],
) -> Result<(), HachiError> {
    let mut ranges = Vec::with_capacity(acc.len());
    let mut cursor = 0usize;
    for row in acc.iter() {
        let start = cursor;
        cursor += row.len();
        ranges.push((start, cursor));
    }
    if cursor != other_flat.len() {
        return Err(HachiError::InvalidInput(
            "flat phi length mismatch".to_string(),
        ));
    }

    cfg_iter_mut!(acc)
        .zip(cfg_iter!(ranges))
        .for_each(|(row_acc, &(start, end))| {
            for (dst, src) in row_acc.iter_mut().zip(other_flat[start..end].iter()) {
                *dst += *src;
            }
        });
    Ok(())
}

#[inline]
fn scalar_to_ring<F: FieldCore, const D: usize>(scalar: F) -> CyclotomicRing<F, D> {
    let mut coeffs = [F::zero(); D];
    coeffs[0] = scalar;
    CyclotomicRing::from_coefficients(coeffs)
}

#[inline]
fn mul_accumulate_with_alpha<F: FieldCore + CanonicalField, const D: usize>(
    alpha: &AggregationRandomness<F, D>,
    coeff: &CyclotomicRing<F, D>,
    dst: &mut CyclotomicRing<F, D>,
) {
    match alpha {
        AggregationRandomness::Scalar(alpha_scalar) => {
            if alpha_scalar.is_zero() {
                return;
            }
            for (dst_coeff, src_coeff) in
                dst.coefficients_mut().iter_mut().zip(coeff.coefficients())
            {
                *dst_coeff += *src_coeff * *alpha_scalar;
            }
        }
        AggregationRandomness::Ring(alpha_ring) => {
            mul_accumulate_term_coeff(alpha_ring, coeff, dst)
        }
    }
}

#[inline]
fn accumulate_rhs_with_alpha<F: FieldCore + CanonicalField, const D: usize>(
    alpha: &AggregationRandomness<F, D>,
    target: &CyclotomicRing<F, D>,
    acc: &mut CyclotomicRing<F, D>,
) {
    match alpha {
        AggregationRandomness::Scalar(alpha_scalar) => *acc += target.scale(alpha_scalar),
        AggregationRandomness::Ring(alpha_ring) => alpha_ring.mul_accumulate_into(target, acc),
    }
}

#[inline]
fn alpha_mul_sparse<F: FieldCore + CanonicalField + FromSmallInt, const D: usize>(
    alpha: &AggregationRandomness<F, D>,
    challenge: &SparseChallenge,
) -> CyclotomicRing<F, D> {
    match alpha {
        AggregationRandomness::Scalar(alpha_scalar) => {
            let mut coeffs = [F::zero(); D];
            for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
                coeffs[pos as usize] += *alpha_scalar * F::from_i64(coeff as i64);
            }
            CyclotomicRing::from_coefficients(coeffs)
        }
        AggregationRandomness::Ring(alpha_ring) => alpha_ring.mul_by_sparse(challenge),
    }
}

#[inline]
fn alpha_scale_to_ring<F: FieldCore + CanonicalField, const D: usize>(
    alpha: &AggregationRandomness<F, D>,
    scale: F,
) -> CyclotomicRing<F, D> {
    match alpha {
        AggregationRandomness::Scalar(alpha_scalar) => {
            scalar_to_ring::<F, D>(*alpha_scalar * scale)
        }
        AggregationRandomness::Ring(alpha_ring) => alpha_ring.scale(&scale),
    }
}

/// Sample 256 scalar collapse challenges (legacy schedule).
fn sample_jl_collapse_challenge<F, T>(transcript: &mut T) -> [F; 256]
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    std::array::from_fn(|_| transcript.challenge_scalar(labels::CHALLENGE_LABRADOR_JL_COLLAPSE))
}

/// Collapse JL projection coordinates with signed challenge weights directly in
/// the field, avoiding host-integer saturation.
fn collapse_to_field<F>(projection: &[i64; 256], alpha: &[F; 256]) -> F
where
    F: FieldCore + FromSmallInt,
{
    projection
        .iter()
        .zip(alpha.iter())
        .fold(F::zero(), |acc, (&p, &a)| acc + a * F::from_i64(p))
}

fn validate_matrix_cols(matrix: &LabradorJlMatrix, cols: usize) -> Result<(), HachiError> {
    if !matrix.is_well_formed() || matrix.cols() != cols {
        return Err(HachiError::InvalidInput(
            "JL matrix row length mismatch".to_string(),
        ));
    }
    Ok(())
}

#[inline]
fn build_four_russians_lookup_field<F: FieldCore>(
    alpha0: F,
    alpha1: F,
    alpha2: F,
    alpha3: F,
) -> [F; 256] {
    let mut lookup = [F::zero(); 256];
    for packed in 0u16..256 {
        let packed = packed as u8;
        let pair0 = packed & 0b11;
        let pair1 = (packed >> 2) & 0b11;
        let pair2 = (packed >> 4) & 0b11;
        let pair3 = (packed >> 6) & 0b11;
        let mut acc = F::zero();
        match pair0 {
            0b00 => acc -= alpha0,
            0b11 => acc += alpha0,
            _ => {}
        }
        match pair1 {
            0b00 => acc -= alpha1,
            0b11 => acc += alpha1,
            _ => {}
        }
        match pair2 {
            0b00 => acc -= alpha2,
            0b11 => acc += alpha2,
            _ => {}
        }
        match pair3 {
            0b00 => acc -= alpha3,
            0b11 => acc += alpha3,
            _ => {}
        }
        lookup[packed as usize] = acc;
    }
    lookup
}

#[inline]
fn accumulate_field_weight_contribution<F: FieldCore, const D: usize>(
    coeffs: &mut [F],
    local_idx: usize,
    contribution: F,
) {
    if contribution.is_zero() {
        return;
    }
    if local_idx == 0 {
        coeffs[0] += contribution;
    } else {
        coeffs[D - local_idx] -= contribution;
    }
}

type JlGroupRows<'a> = [&'a [u8]; 4];

#[inline]
fn apply_four_russians_group4_to_elem<F: FieldCore, const D: usize>(
    elem: &mut CyclotomicRing<F, D>,
    elem_idx: usize,
    rows: JlGroupRows<'_>,
    lookup: &[F; 256],
    bytes_per_ring: usize,
) {
    let start = elem_idx * bytes_per_ring;
    let end = start + bytes_per_ring;
    let sign_bytes0 = &rows[0][start..end];
    let sign_bytes1 = &rows[1][start..end];
    let sign_bytes2 = &rows[2][start..end];
    let sign_bytes3 = &rows[3][start..end];

    let coeffs = elem.coefficients_mut();
    let mut local_idx = 0usize;

    for (((&byte0, &byte1), &byte2), &byte3) in sign_bytes0
        .iter()
        .zip(sign_bytes1.iter())
        .zip(sign_bytes2.iter())
        .zip(sign_bytes3.iter())
    {
        let packed0 =
            (byte0 & 0b11) | ((byte1 & 0b11) << 2) | ((byte2 & 0b11) << 4) | ((byte3 & 0b11) << 6);
        let packed1 = ((byte0 >> 2) & 0b11)
            | (((byte1 >> 2) & 0b11) << 2)
            | (((byte2 >> 2) & 0b11) << 4)
            | (((byte3 >> 2) & 0b11) << 6);
        let packed2 = ((byte0 >> 4) & 0b11)
            | (((byte1 >> 4) & 0b11) << 2)
            | (((byte2 >> 4) & 0b11) << 4)
            | (((byte3 >> 4) & 0b11) << 6);
        let packed3 = ((byte0 >> 6) & 0b11)
            | (((byte1 >> 6) & 0b11) << 2)
            | (((byte2 >> 6) & 0b11) << 4)
            | (((byte3 >> 6) & 0b11) << 6);

        accumulate_field_weight_contribution::<F, D>(coeffs, local_idx, lookup[packed0 as usize]);
        accumulate_field_weight_contribution::<F, D>(
            coeffs,
            local_idx + 1,
            lookup[packed1 as usize],
        );
        accumulate_field_weight_contribution::<F, D>(
            coeffs,
            local_idx + 2,
            lookup[packed2 as usize],
        );
        accumulate_field_weight_contribution::<F, D>(
            coeffs,
            local_idx + 3,
            lookup[packed3 as usize],
        );
        local_idx += 4;
    }
}

/// Collapse 256 JL rows × omega into JL phi coefficients using
/// field arithmetic only.
///
/// # Errors
///
/// Returns [`HachiError::InvalidInput`] if the matrix dimensions are invalid.
///
/// # Panics
///
/// Panics if `D` is not divisible by 4, which is required by the Four Russians
/// JL collapse implementation.
#[tracing::instrument(skip_all, name = "labrador::aggregate_jl_contraints_one_lift")]
pub fn aggregate_jl_contraints_one_lift<F: CanonicalField, const D: usize>(
    matrix: &LabradorJlMatrix,
    omega: &[F; 256],
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    let cols = matrix.cols();
    validate_matrix_cols(matrix, cols)?;
    if D % 4 != 0 {
        panic!("Four Russians field collapse requires D divisible by 4, got D={D}");
    }
    let num_elems = cols / D;
    let bytes_per_ring = D / 4;
    debug_assert_eq!(omega.len() % 4, 0);
    let group_rows: Vec<JlGroupRows<'_>> = (0..omega.len())
        .step_by(4)
        .map(|group_start| {
            [
                matrix.packed_rows[group_start].as_slice(),
                matrix.packed_rows[group_start + 1].as_slice(),
                matrix.packed_rows[group_start + 2].as_slice(),
                matrix.packed_rows[group_start + 3].as_slice(),
            ]
        })
        .collect();
    let lookups: Vec<[F; 256]> = (0..omega.len())
        .step_by(4)
        .map(|group_start| {
            build_four_russians_lookup_field(
                omega[group_start],
                omega[group_start + 1],
                omega[group_start + 2],
                omega[group_start + 3],
            )
        })
        .collect();

    Ok(cfg_into_iter!(0..num_elems)
        .map(|elem_idx| {
            let mut elem = CyclotomicRing::<F, D>::zero();
            for (rows, lookup) in group_rows.iter().zip(lookups.iter()) {
                apply_four_russians_group4_to_elem::<F, D>(
                    &mut elem,
                    elem_idx,
                    *rows,
                    lookup,
                    bytes_per_ring,
                );
            }
            elem
        })
        .collect())
}

/// Pre-flattened witness layout used during JL aggregation.
struct FlatWitness<F: FieldCore, const D: usize> {
    rings: Vec<CyclotomicRing<F, D>>,
}

impl<F: FieldCore + CanonicalField, const D: usize> FlatWitness<F, D> {
    #[tracing::instrument(skip_all, name = "labrador::flat_witness_new")]
    fn new(witness: &LabradorWitness<F, D>) -> Self {
        let mut rings = Vec::new();
        for row in witness.rows() {
            rings.extend(row.iter().copied());
        }
        Self { rings }
    }
}

fn accumulate_phi_flat<F: FieldCore + CanonicalField, const D: usize>(
    phi_total_flat: &mut [CyclotomicRing<F, D>],
    phi_flat: &[CyclotomicRing<F, D>],
    beta: &AggregationRandomness<F, D>,
) {
    debug_assert_eq!(phi_total_flat.len(), phi_flat.len());
    match beta {
        AggregationRandomness::Scalar(s) => {
            if s.is_zero() {
                return;
            }
            cfg_iter_mut!(phi_total_flat)
                .zip(cfg_iter!(phi_flat))
                .for_each(|(dst, src)| {
                    for (dst_coeff, src_coeff) in
                        dst.coefficients_mut().iter_mut().zip(src.coefficients())
                    {
                        *dst_coeff += *src_coeff * *s;
                    }
                });
        }
        AggregationRandomness::Ring(r) => {
            cfg_iter_mut!(phi_total_flat)
                .zip(cfg_iter!(phi_flat))
                .for_each(|(dst, src)| mul_accumulate_term_coeff(r, src, dst));
        }
    }
}

/// Aggregate JL projection constraints on the prover side.
///
/// For each of the ⌈128/log q⌉ lifts:
///   1. Sample ω_j^(k) ∈ Z_q for `j = 1..256`.
///   2. Collapse the JL matrix rows → φ^''(k) ring-element vector.
///   3. Compute b^''(k) = ⟨φ^''(k), s⟩ and verify its constant term.
///   4. Transmit b^''(k) (constant term zeroed) and absorb into transcript.
///   5. If `q` is 128-bit, use scalar β_k; otherwise use ring β_k.
///
/// Returns `(phi_total_flat, jl_lift_residuals)` where
/// `phi_total_flat` is flattened in row-major order and
/// `jl_lift_residuals` holds the transmitted polynomials.
#[allow(clippy::type_complexity)]
#[tracing::instrument(skip_all, name = "labrador::aggregate_jl_constraints_prover")]
pub(crate) fn aggregate_jl_constraints_prover<F, T, const D: usize>(
    witness: &LabradorWitness<F, D>,
    matrix: &LabradorJlMatrix,
    transcript: &mut T,
) -> Result<(Vec<CyclotomicRing<F, D>>, Vec<CyclotomicRing<F, D>>), HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    let flat_witness = FlatWitness::new(witness);
    let flat_witness_i8 = try_centered_i8_cache_from_ring_coeffs(&flat_witness.rings);

    let mut phi_total_flat = vec![CyclotomicRing::<F, D>::zero(); flat_witness.rings.len()];
    let lifts = jl_lifts::<F>();
    let mut jl_lift_residuals = Vec::with_capacity(lifts);

    for _ in 0..lifts {
        let omega = sample_jl_collapse_challenge::<F, T>(transcript);
        let phi_flat = aggregate_jl_contraints_one_lift::<F, D>(matrix, &omega)?;
        let b_full = if let Some(witness_i8) = flat_witness_i8.as_ref() {
            mat_vec_mul_crt_ntt_i8_single(&phi_flat, witness_i8)
                .ok()
                .unwrap_or_else(|| dot_product(&phi_flat, &flat_witness.rings))
        } else {
            dot_product(&phi_flat, &flat_witness.rings)
        };

        let (b_tx, _c0) = zero_constant_term_for_proof(b_full);
        jl_lift_residuals.push(b_tx);
        transcript.append_serde(labels::ABSORB_LABRADOR_JL_LIFT_RESIDUALS, &b_tx);

        let beta = sample_aggregation_randomness::<F, _, D>(transcript);
        accumulate_phi_flat(&mut phi_total_flat, &phi_flat, &beta);
    }

    Ok((phi_total_flat, jl_lift_residuals))
}

/// Aggregate JL projection constraints on the verifier side.
///
/// Same transcript flow as the prover variant, but reconstructs the full
/// polynomial b^''(k) by restoring the constant term from the projection
/// and the transmitted `jl_lift_residuals[k]`. Returns a flattened `phi_total`.
#[allow(clippy::type_complexity)]
#[tracing::instrument(skip_all, name = "labrador::aggregate_jl_constraints_verifier")]
pub(crate) fn aggregate_jl_constraints_verifier<F, T, const D: usize>(
    row_lengths: &[usize],
    jl_projection: &[i64; 256],
    matrix: &LabradorJlMatrix,
    jl_lift_residuals: &[CyclotomicRing<F, D>],
    transcript: &mut T,
) -> Result<(Vec<CyclotomicRing<F, D>>, CyclotomicRing<F, D>), HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    let lifts = jl_lifts::<F>();
    if jl_lift_residuals.len() != lifts {
        return Err(HachiError::InvalidProof);
    }
    let total_phi_elems: usize = row_lengths.iter().sum();
    let mut phi_total_flat = vec![CyclotomicRing::<F, D>::zero(); total_phi_elems];
    let mut aggregated_rhs = CyclotomicRing::<F, D>::zero();

    for jl_lift_residual in jl_lift_residuals.iter() {
        let omega = sample_jl_collapse_challenge::<F, T>(transcript);
        let phi_flat = aggregate_jl_contraints_one_lift::<F, D>(matrix, &omega)?;
        let b_full = restore_constant_term(
            *jl_lift_residual,
            collapse_to_field::<F>(jl_projection, &omega),
        );
        transcript.append_serde(labels::ABSORB_LABRADOR_JL_LIFT_RESIDUALS, jl_lift_residual);
        let beta = sample_aggregation_randomness::<F, _, D>(transcript);
        accumulate_rhs_with_alpha(&beta, &b_full, &mut aggregated_rhs);
        accumulate_phi_flat(&mut phi_total_flat, &phi_flat, &beta);
    }

    Ok((phi_total_flat, aggregated_rhs))
}

#[inline]
fn mul_accumulate_term_coeff<F: FieldCore + CanonicalField, const D: usize>(
    alpha: &CyclotomicRing<F, D>,
    coeff: &CyclotomicRing<F, D>,
    dst: &mut CyclotomicRing<F, D>,
) {
    if coeff.hamming_weight() <= SPARSE_RING_MUL_MAX_WEIGHT {
        alpha.mul_accumulate_sparse_rhs_into(coeff, dst);
    } else {
        alpha.mul_accumulate_into(coeff, dst);
    }
}

fn accumulate_scaled_row<F: FieldCore + CanonicalField, const D: usize>(
    dst: &mut [CyclotomicRing<F, D>],
    src: &[CyclotomicRing<F, D>],
    alpha: &AggregationRandomness<F, D>,
    scale: F,
) {
    debug_assert_eq!(dst.len(), src.len());
    match alpha {
        AggregationRandomness::Scalar(alpha_scalar) => {
            let scaled_alpha = *alpha_scalar * scale;
            if scaled_alpha.is_zero() {
                return;
            }
            cfg_iter_mut!(dst)
                .zip(cfg_iter!(src))
                .for_each(|(dst_elem, src_elem)| {
                    for (dst_coeff, src_coeff) in dst_elem
                        .coefficients_mut()
                        .iter_mut()
                        .zip(src_elem.coefficients())
                    {
                        *dst_coeff += *src_coeff * scaled_alpha;
                    }
                });
        }
        AggregationRandomness::Ring(alpha_ring) => {
            let scaled_alpha = alpha_ring.scale(&scale);
            cfg_iter_mut!(dst)
                .zip(cfg_iter!(src))
                .for_each(|(dst_elem, src_elem)| {
                    mul_accumulate_term_coeff(&scaled_alpha, src_elem, dst_elem)
                });
        }
    }
}

fn accumulate_statement_row_work<F: FieldCore + CanonicalField, const D: usize>(
    row: &mut [CyclotomicRing<F, D>],
    work: &[(usize, usize)],
    constraints: &[LabradorConstraint<F, D>],
    alphas: &[AggregationRandomness<F, D>],
) {
    #[cfg(feature = "parallel")]
    row.par_chunks_mut(STATEMENT_ROW_CHUNK_LEN)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let chunk_start = chunk_idx * STATEMENT_ROW_CHUNK_LEN;
            let chunk_end = chunk_start + chunk.len();
            for &(ci, ti) in work {
                let term = &constraints[ci].terms[ti];
                let term_end = term.offset + term.coefficients.len();
                let start = chunk_start.max(term.offset);
                let end = chunk_end.min(term_end);
                if start >= end {
                    continue;
                }
                let alpha = &alphas[ci];
                let src = &term.coefficients[start - term.offset..end - term.offset];
                let dst = &mut chunk[start - chunk_start..end - chunk_start];
                for (dst_elem, src_elem) in dst.iter_mut().zip(src.iter()) {
                    mul_accumulate_with_alpha(alpha, src_elem, dst_elem);
                }
            }
        });

    #[cfg(not(feature = "parallel"))]
    for &(ci, ti) in work {
        let term = &constraints[ci].terms[ti];
        let alpha = &alphas[ci];
        for (dst_elem, src_elem) in row[term.offset..term.offset + term.coefficients.len()]
            .iter_mut()
            .zip(term.coefficients.iter())
        {
            mul_accumulate_with_alpha(alpha, src_elem, dst_elem);
        }
    }
}

#[tracing::instrument(skip_all, name = "labrador::aggregate_reduced_statement_constraints")]
fn aggregate_reduced_statement_constraints<F, T, const D: usize>(
    statement: &LabradorStatement<F, D>,
    plan: &LabradorReducedConstraintPlan<F, D>,
    row_lengths: &[usize],
    transcript: &mut T,
) -> Result<AggregatedConstraintSystem<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    let layout = NextWitnessLayout::new(plan.row_count, &plan.config);
    if row_lengths.len() != layout.num_rows() {
        return Err(HachiError::InvalidInput(
            "reduced statement row count mismatch".to_string(),
        ));
    }
    if row_lengths
        .iter()
        .take(plan.config.witness_digit_parts)
        .any(|&len| len != plan.max_len)
        || row_lengths[layout.aux_row] != layout.aux_row_len()
    {
        return Err(HachiError::InvalidInput(
            "reduced statement row layout mismatch".to_string(),
        ));
    }
    if statement.inner_opening_payload.len() != plan.config.outer_commit_rank
        || statement.linear_garbage_payload.len() != plan.config.outer_commit_rank
    {
        return Err(HachiError::InvalidInput(
            "reduced statement payload length mismatch".to_string(),
        ));
    }

    let pow_b: Vec<F> = (0..plan.config.witness_digit_parts)
        .map(|idx| pow2_field::<F>(plan.config.witness_digit_bits * idx))
        .collect();
    let pow_bu: Vec<F> = (0..plan.config.aux_digit_parts)
        .map(|idx| pow2_field::<F>(plan.config.aux_digit_bits * idx))
        .collect();

    let mut phi_total: Vec<Vec<CyclotomicRing<F, D>>> = row_lengths
        .iter()
        .map(|&len| vec![CyclotomicRing::zero(); len])
        .collect();
    let (z_rows, aux_rows) = phi_total.split_at_mut(plan.config.witness_digit_parts);
    let aux_row = aux_rows.first_mut().ok_or_else(|| {
        HachiError::InvalidInput("missing auxiliary row in reduced statement".to_string())
    })?;
    let inner_opening_start = layout.inner_opening_digits_range().start;
    let linear_garbage_start = layout.linear_garbage_digits_range().start;
    let mut aggregated_rhs = CyclotomicRing::<F, D>::zero();

    for (b_row, target) in plan
        .setup
        .b_mat
        .iter()
        .zip(statement.inner_opening_payload.iter())
    {
        let alpha = sample_aggregation_randomness::<F, T, D>(transcript);
        accumulate_rhs_with_alpha(&alpha, target, &mut aggregated_rhs);
        let dst = &mut aux_row[inner_opening_start..linear_garbage_start];
        for (dst, src) in dst.iter_mut().zip(b_row.iter()) {
            mul_accumulate_with_alpha(&alpha, src, dst);
        }
    }

    for (d_row, target) in plan
        .setup
        .d_mat
        .iter()
        .zip(statement.linear_garbage_payload.iter())
    {
        let alpha = sample_aggregation_randomness::<F, T, D>(transcript);
        accumulate_rhs_with_alpha(&alpha, target, &mut aggregated_rhs);
        let dst = &mut aux_row[linear_garbage_start..];
        for (dst, src) in dst.iter_mut().zip(d_row.iter()) {
            mul_accumulate_with_alpha(&alpha, src, dst);
        }
    }

    for output_idx in 0..plan.config.inner_commit_rank {
        let alpha = sample_aggregation_randomness::<F, T, D>(transcript);
        let a_row = &plan.setup.a_mat[output_idx];
        for (part_idx, &scale) in pow_b.iter().enumerate() {
            accumulate_scaled_row(&mut z_rows[part_idx], a_row, &alpha, scale);
        }

        for (row_idx, challenge) in plan.challenges.iter().enumerate() {
            let base = alpha_mul_sparse(&alpha, challenge);
            for (part_idx, &scale) in pow_bu.iter().enumerate() {
                let idx = inner_opening_start
                    + row_idx * plan.config.inner_commit_rank * plan.config.aux_digit_parts
                    + output_idx * plan.config.aux_digit_parts
                    + part_idx;
                aux_row[idx] -= base.scale(&scale);
            }
        }
    }

    let alpha_lg = sample_aggregation_randomness::<F, T, D>(transcript);
    for (part_idx, &scale) in pow_b.iter().enumerate() {
        accumulate_scaled_row(&mut z_rows[part_idx], &plan.amortized_phi, &alpha_lg, scale);
    }
    for i in 0..plan.challenges.len() {
        for j in i..plan.challenges.len() {
            let mut base = alpha_mul_sparse(&alpha_lg, &plan.challenges[i]);
            base = base.mul_by_sparse(&plan.challenges[j]);
            let pair = pair_index(i, j, plan.challenges.len());
            for (part_idx, &scale) in pow_bu.iter().enumerate() {
                let idx = linear_garbage_start + pair * plan.config.aux_digit_parts + part_idx;
                aux_row[idx] -= base.scale(&scale);
            }
        }
    }

    let alpha_diag = sample_aggregation_randomness::<F, T, D>(transcript);
    accumulate_rhs_with_alpha(&alpha_diag, &plan.aggregated_rhs, &mut aggregated_rhs);
    for i in 0..plan.row_count {
        let pair = pair_index(i, i, plan.row_count);
        for (part_idx, &scale) in pow_bu.iter().enumerate() {
            let idx = linear_garbage_start + pair * plan.config.aux_digit_parts + part_idx;
            aux_row[idx] += alpha_scale_to_ring(&alpha_diag, scale);
        }
    }

    Ok((phi_total, aggregated_rhs))
}

pub(crate) fn aggregate_statement<F, T, const D: usize>(
    statement: &LabradorStatement<F, D>,
    row_lengths: &[usize],
    transcript: &mut T,
) -> Result<AggregatedConstraintSystem<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    if let Some(plan) = statement.reduced_constraints.as_deref() {
        aggregate_reduced_statement_constraints(statement, plan, row_lengths, transcript)
    } else {
        aggregate_statement_constraints(&statement.constraints, row_lengths, transcript)
    }
}

/// Fold statement constraints into aggregated (φ, b) using transcript challenges.
///
/// Each scalar constraint is folded with one fresh dense challenge α: its
/// coefficient terms are fused-accumulated into `phi_total`, while `α · target`
/// is accumulated into `aggregated_rhs`.
#[allow(clippy::type_complexity)]
#[tracing::instrument(skip_all, name = "labrador::aggregate_statement_constraints")]
pub(crate) fn aggregate_statement_constraints<F, T, const D: usize>(
    constraints: &[LabradorConstraint<F, D>],
    row_lengths: &[usize],
    transcript: &mut T,
) -> Result<(Vec<Vec<CyclotomicRing<F, D>>>, CyclotomicRing<F, D>), HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    if constraints.is_empty() {
        let phi_total: Vec<Vec<CyclotomicRing<F, D>>> = row_lengths
            .iter()
            .map(|&len| vec![CyclotomicRing::zero(); len])
            .collect();
        return Ok((phi_total, CyclotomicRing::zero()));
    }

    let num_rows = row_lengths.len();

    // Phase 1: sample all challenges sequentially (Fiat-Shamir ordering).
    let alphas: Vec<AggregationRandomness<F, D>> = constraints
        .iter()
        .map(|_| sample_aggregation_randomness::<F, T, D>(transcript))
        .collect();

    // Phase 2: validate bounds (cheap, allows early `?` return).
    for cnst in constraints {
        for term in &cnst.terms {
            if term.row >= num_rows {
                return Err(HachiError::InvalidInput(
                    "constraint row index out of bounds".to_string(),
                ));
            }
            if term.offset + term.coefficients.len() > row_lengths[term.row] {
                return Err(HachiError::InvalidInput(
                    "constraint term exceeds row length".to_string(),
                ));
            }
        }
    }

    // Phase 3: aggregated_rhs — parallel fold-reduce over constraints.
    let aggregated_rhs = cfg_fold_reduce!(
        (0..constraints.len()),
        || CyclotomicRing::<F, D>::zero(),
        |mut acc, i| {
            accumulate_rhs_with_alpha(&alphas[i], &constraints[i].target, &mut acc);
            acc
        },
        |mut a, b| {
            a += b;
            a
        }
    );

    // Phase 4: phi_total — group work by target row, then parallel over rows.
    let mut row_work: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num_rows];
    for (ci, cnst) in constraints.iter().enumerate() {
        for (ti, term) in cnst.terms.iter().enumerate() {
            row_work[term.row].push((ci, ti));
        }
    }

    let phi_total: Vec<Vec<CyclotomicRing<F, D>>> = cfg_into_iter!(row_work)
        .zip(cfg_iter!(row_lengths).copied())
        .map(|(work, len)| {
            let mut row = vec![CyclotomicRing::<F, D>::zero(); len];
            accumulate_statement_row_work(&mut row, &work, constraints, &alphas);
            row
        })
        .collect();

    Ok((phi_total, aggregated_rhs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Prime128M13M4P0;
    use crate::algebra::{Pow2Offset32Field, Pow2Offset64Field};
    use crate::protocol::transcript::labels::DOMAIN_LABRADOR_RECURSION;
    use crate::protocol::transcript::Blake2bTranscript;

    const D: usize = 64;
    const TEST_RING_ELEMS: usize = 16;
    const TEST_COLS: usize = TEST_RING_ELEMS * D;

    #[test]
    fn safe_to_use_scalar_randomness_only_for_128_bit_fields() {
        assert!(
            !safe_to_use_scalar_randomness::<Pow2Offset32Field>(),
            "32-bit field must use legacy JL aggregation schedule"
        );
        assert!(
            !safe_to_use_scalar_randomness::<Pow2Offset64Field>(),
            "64-bit field must use legacy JL aggregation schedule"
        );
        assert!(
            safe_to_use_scalar_randomness::<Prime128M13M4P0>(),
            "128-bit field should use scalar JL aggregation schedule"
        );
    }

    fn assert_aggregate_jl_contraints_one_lift_matches_naive<
        F: FieldCore + CanonicalField + FromSmallInt,
    >(
        matrix: &LabradorJlMatrix,
        omega: [F; 256],
    ) {
        let cols = matrix.cols();
        assert_eq!(cols, TEST_COLS);

        let got = aggregate_jl_contraints_one_lift::<F, D>(matrix, &omega).unwrap();
        let mut expected = vec![CyclotomicRing::<F, D>::zero(); cols / D];
        // Naive paper-style reference:
        // φ''_i = Σ_j ω_j · σ_{-1}(π_i^(j))
        for (elem_idx, elem) in expected.iter_mut().enumerate() {
            for (row_idx, &alpha) in omega.iter().enumerate() {
                let row = &matrix.packed_rows[row_idx];
                let pi = std::array::from_fn(|local_idx| {
                    let col_idx = elem_idx * D + local_idx;
                    let shift = (col_idx & 0b11) << 1;
                    let pair = (row[col_idx >> 2] >> shift) & 0b11;
                    let sign = match pair {
                        0b00 => -1i64,
                        0b11 => 1i64,
                        _ => 0i64,
                    };
                    F::from_i64(sign)
                });
                *elem += CyclotomicRing::<F, D>::from_coefficients(pi)
                    .sigma_m1()
                    .scale(&alpha);
            }
        }

        assert!(got == expected);
    }

    #[test]
    fn aggregate_jl_contraints_one_lift_matches_naive_fp32() {
        let mut transcript = Blake2bTranscript::<Pow2Offset32Field>::new(DOMAIN_LABRADOR_RECURSION);
        let matrix =
            LabradorJlMatrix::generate::<Pow2Offset32Field, _>(&mut transcript, TEST_COLS).unwrap();
        let omega = sample_jl_collapse_challenge::<Pow2Offset32Field, _>(&mut transcript);
        assert_aggregate_jl_contraints_one_lift_matches_naive::<Pow2Offset32Field>(&matrix, omega);
    }

    #[test]
    fn aggregate_jl_contraints_one_lift_matches_naive_fp64() {
        type F64 = Pow2Offset64Field;
        let mut transcript = Blake2bTranscript::<F64>::new(DOMAIN_LABRADOR_RECURSION);
        let matrix = LabradorJlMatrix::generate::<F64, _>(&mut transcript, TEST_COLS).unwrap();
        let omega = sample_jl_collapse_challenge::<F64, _>(&mut transcript);
        assert_aggregate_jl_contraints_one_lift_matches_naive::<F64>(&matrix, omega);
    }

    #[test]
    fn aggregate_jl_contraints_one_lift_matches_naive_fp128() {
        type F128 = Prime128M13M4P0;
        let mut transcript = Blake2bTranscript::<F128>::new(DOMAIN_LABRADOR_RECURSION);
        let matrix = LabradorJlMatrix::generate::<F128, _>(&mut transcript, TEST_COLS).unwrap();
        let omega = sample_jl_collapse_challenge::<F128, _>(&mut transcript);
        assert_aggregate_jl_contraints_one_lift_matches_naive::<F128>(&matrix, omega);
    }
}
