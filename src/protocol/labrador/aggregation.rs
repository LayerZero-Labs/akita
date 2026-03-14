//! Constraint aggregation for the Labrador protocol.
//!
//! Implements the "Aggregating" step from Section 5.2 of the LaBRADOR paper:
//! the 256 JL projection constraints and the existing statement constraints
//! are folded into a single aggregated constraint (φ_i, b) via random challenges.
//!
//! # Paper reference
//!
//! The JL constraints are first collapsed into ⌈128/log q⌉ functions using
//! scalar challenges ω^(k) ∈ (Z_q)^256 (one per "lift"). Each collapsed
//! function produces a polynomial b''(k) whose constant term the verifier
//! can check. The prover sends b''(k) with the constant term zeroed out.
//! A ring-element challenge β_k then folds each lift into a running sum.
//! Statement constraints are folded separately with their own ring-element
//! challenges. Both contributions are combined by the caller.

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::labrador::config::jl_lifts;
use crate::protocol::labrador::constraints::{pair_index, LabradorConstraint, NextWitnessLayout};
use crate::protocol::labrador::johnson_lindenstrauss::{
    for_each_jl_group4_bytes, restore_constant_term, zero_constant_term_for_proof, LabradorJlMatrix,
};
use crate::protocol::labrador::types::{
    LabradorReducedConstraintPlan, LabradorStatement, LabradorWitness,
};
use crate::protocol::transcript::labels;
use crate::protocol::transcript::{challenge_ring_element, Transcript};
use crate::{CanonicalField, FieldCore, FromSmallInt};

type AggregatedConstraintSystem<F, const D: usize> =
    (Vec<Vec<CyclotomicRing<F, D>>>, CyclotomicRing<F, D>);

const STATEMENT_ROW_CHUNK_LEN: usize = 256;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

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

/// Sample 256 scalar challenges ω_j^(k) from the transcript.
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

// ---------------------------------------------------------------------------
// Collapse helpers
// ---------------------------------------------------------------------------

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

#[inline]
fn apply_four_russians_group4<F: FieldCore, const D: usize>(
    phi: &mut [CyclotomicRing<F, D>],
    row0: &[u8],
    row1: &[u8],
    row2: &[u8],
    row3: &[u8],
    lookup: &[F; 256],
    bytes_per_ring: usize,
) {
    debug_assert_eq!(row0.len(), phi.len() * bytes_per_ring);
    debug_assert_eq!(row1.len(), phi.len() * bytes_per_ring);
    debug_assert_eq!(row2.len(), phi.len() * bytes_per_ring);
    debug_assert_eq!(row3.len(), phi.len() * bytes_per_ring);

    cfg_iter_mut!(phi).enumerate().for_each(|(elem_idx, elem)| {
        let start = elem_idx * bytes_per_ring;
        let end = start + bytes_per_ring;
        let sign_bytes0 = &row0[start..end];
        let sign_bytes1 = &row1[start..end];
        let sign_bytes2 = &row2[start..end];
        let sign_bytes3 = &row3[start..end];

        let coeffs = elem.coefficients_mut();
        let mut local_idx = 0usize;

        for (((&byte0, &byte1), &byte2), &byte3) in sign_bytes0
            .iter()
            .zip(sign_bytes1.iter())
            .zip(sign_bytes2.iter())
            .zip(sign_bytes3.iter())
        {
            let packed0 = (byte0 & 0b11)
                | ((byte1 & 0b11) << 2)
                | ((byte2 & 0b11) << 4)
                | ((byte3 & 0b11) << 6);
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

            accumulate_field_weight_contribution::<F, D>(
                coeffs,
                local_idx,
                lookup[packed0 as usize],
            );
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
    });
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
    let mut phi = vec![CyclotomicRing::<F, D>::zero(); cols / D];
    let bytes_per_ring = D / 4;
    debug_assert_eq!(omega.len() % 4, 0);

    for group_start in (0..omega.len()).step_by(4) {
        let row0 = matrix.row_bytes(group_start);
        let row1 = matrix.row_bytes(group_start + 1);
        let row2 = matrix.row_bytes(group_start + 2);
        let row3 = matrix.row_bytes(group_start + 3);
        let lookup = build_four_russians_lookup_field(
            omega[group_start],
            omega[group_start + 1],
            omega[group_start + 2],
            omega[group_start + 3],
        );
        apply_four_russians_group4::<F, D>(
            &mut phi,
            row0,
            row1,
            row2,
            row3,
            &lookup,
            bytes_per_ring,
        );
    }

    Ok(phi)
}

#[tracing::instrument(skip_all, name = "labrador::aggregate_jl_contraints_one_lift_seeded")]
fn aggregate_jl_contraints_one_lift_seeded<F: CanonicalField, const D: usize>(
    cols: usize,
    row_bytes: usize,
    seed: &[u8; 32],
    omega: &[F; 256],
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    if D % 4 != 0 {
        panic!("Four Russians field collapse requires D divisible by 4, got D={D}");
    }
    let mut phi = vec![CyclotomicRing::<F, D>::zero(); cols / D];
    let bytes_per_ring = D / 4;

    for_each_jl_group4_bytes(seed, row_bytes, |group_start, row0, row1, row2, row3| {
        let lookup = build_four_russians_lookup_field(
            omega[group_start],
            omega[group_start + 1],
            omega[group_start + 2],
            omega[group_start + 3],
        );
        apply_four_russians_group4::<F, D>(
            &mut phi,
            row0,
            row1,
            row2,
            row3,
            &lookup,
            bytes_per_ring,
        );
        Ok::<(), HachiError>(())
    })?;

    Ok(phi)
}

// ---------------------------------------------------------------------------
// Witness flattening
// ---------------------------------------------------------------------------

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

/// Fold `phi_flat` into a flat accumulator scaled by `beta`.
fn accumulate_phi_flat<F: FieldCore, const D: usize>(
    phi_total_flat: &mut [CyclotomicRing<F, D>],
    phi_flat: &[CyclotomicRing<F, D>],
    beta: CyclotomicRing<F, D>,
) {
    debug_assert_eq!(phi_total_flat.len(), phi_flat.len());
    // In-place accumulation on destination:
    //   phi_total_flat[i] += beta * phi_flat[i]
    // Capture beta by reference so rayon workers don't copy the full ring value.
    let beta_ref = &beta;
    cfg_iter_mut!(phi_total_flat)
        .zip(cfg_iter!(phi_flat))
        .for_each(|(dst, src)| beta_ref.mul_accumulate_into(src, dst));
}

// ---------------------------------------------------------------------------
// Prover-side JL aggregation
// ---------------------------------------------------------------------------

/// Aggregate JL projection constraints on the prover side.
///
/// For each of the ⌈128/log q⌉ lifts:
///   1. Sample ω^(k) ∈ (Z_q)^256 from the transcript.
///   2. Collapse the JL matrix rows → φ^''(k) ring-element vector.
///   3. Compute b^''(k) = ⟨φ^''(k), s⟩ and verify its constant term.
///   4. Transmit b^''(k) (constant term zeroed) and absorb into transcript.
///   5. Sample ring-element β_k and accumulate into (φ_total, b_total).
///
/// Returns `(phi_total_flat, b_total, bb)` where `phi_total_flat` is flattened
/// in row-major order and `bb` holds the transmitted polynomials.
#[allow(clippy::type_complexity)]
#[tracing::instrument(skip_all, name = "labrador::aggregate_jl_constraints_prover")]
pub(crate) fn aggregate_jl_constraints_prover<F, T, const D: usize>(
    witness: &LabradorWitness<F, D>,
    matrix: &LabradorJlMatrix,
    transcript: &mut T,
) -> Result<
    (
        Vec<CyclotomicRing<F, D>>,
        CyclotomicRing<F, D>,
        Vec<CyclotomicRing<F, D>>,
    ),
    HachiError,
>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    let flat_witness = FlatWitness::new(witness);

    let mut phi_total_flat = vec![CyclotomicRing::<F, D>::zero(); flat_witness.rings.len()];
    let mut b_total = CyclotomicRing::<F, D>::zero();
    let lifts = jl_lifts::<F>();
    let mut bb = Vec::with_capacity(lifts);

    for _ in 0..lifts {
        let omega = sample_jl_collapse_challenge::<F, T>(transcript);
        let phi_flat = aggregate_jl_contraints_one_lift::<F, D>(matrix, &omega)?;
        let b_full = dot_product(&phi_flat, &flat_witness.rings);

        let (b_tx, _c0) = zero_constant_term_for_proof(b_full);
        bb.push(b_tx);
        transcript.append_serde(labels::ABSORB_LABRADOR_BB, &b_tx);

        let beta: CyclotomicRing<F, D> =
            challenge_ring_element(transcript, labels::CHALLENGE_LABRADOR_AGGREGATION);
        b_total += beta * b_full;
        accumulate_phi_flat(&mut phi_total_flat, &phi_flat, beta);
    }

    Ok((phi_total_flat, b_total, bb))
}

// ---------------------------------------------------------------------------
// Verifier-side JL aggregation
// ---------------------------------------------------------------------------

/// Aggregate JL projection constraints on the verifier side.
///
/// Same transcript flow as the prover variant, but reconstructs the full
/// polynomial b^''(k) by restoring the constant term from the projection
/// and the transmitted `bb[k]`. Returns a flattened `phi_total`.
#[cfg(test)]
#[allow(clippy::type_complexity)]
#[tracing::instrument(skip_all, name = "labrador::aggregate_jl_constraints_verifier")]
pub(crate) fn aggregate_jl_constraints_verifier<F, T, const D: usize>(
    row_lengths: &[usize],
    jl_projection: &[i64; 256],
    matrix: &LabradorJlMatrix,
    bb: &[CyclotomicRing<F, D>],
    transcript: &mut T,
) -> Result<(Vec<CyclotomicRing<F, D>>, CyclotomicRing<F, D>), HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    let lifts = jl_lifts::<F>();
    if bb.len() != lifts {
        return Err(HachiError::InvalidProof);
    }
    let total_phi_elems: usize = row_lengths.iter().sum();
    let mut phi_total_flat = vec![CyclotomicRing::<F, D>::zero(); total_phi_elems];
    let mut b_total = CyclotomicRing::<F, D>::zero();

    for bb_lift in bb.iter() {
        let omega = sample_jl_collapse_challenge::<F, T>(transcript);
        let phi_flat = aggregate_jl_contraints_one_lift::<F, D>(matrix, &omega)?;
        let b_full = restore_constant_term(*bb_lift, collapse_to_field::<F>(jl_projection, &omega));
        transcript.append_serde(labels::ABSORB_LABRADOR_BB, bb_lift);
        let beta: CyclotomicRing<F, D> =
            challenge_ring_element(transcript, labels::CHALLENGE_LABRADOR_AGGREGATION);
        b_total += beta * b_full;
        accumulate_phi_flat(&mut phi_total_flat, &phi_flat, beta);
    }

    Ok((phi_total_flat, b_total))
}

#[allow(clippy::type_complexity)]
#[tracing::instrument(skip_all, name = "labrador::aggregate_jl_constraints_verifier_seeded")]
pub(crate) fn aggregate_jl_constraints_verifier_seeded<F, T, const D: usize>(
    row_lengths: &[usize],
    jl_projection: &[i64; 256],
    cols: usize,
    row_bytes: usize,
    seed: &[u8; 32],
    bb: &[CyclotomicRing<F, D>],
    transcript: &mut T,
) -> Result<(Vec<CyclotomicRing<F, D>>, CyclotomicRing<F, D>), HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    let lifts = jl_lifts::<F>();
    if bb.len() != lifts {
        return Err(HachiError::InvalidProof);
    }
    let total_phi_elems: usize = row_lengths.iter().sum();
    if total_phi_elems * D != cols {
        return Err(HachiError::InvalidProof);
    }
    let mut phi_total_flat = vec![CyclotomicRing::<F, D>::zero(); total_phi_elems];
    let mut b_total = CyclotomicRing::<F, D>::zero();

    for bb_lift in bb.iter() {
        let omega = sample_jl_collapse_challenge::<F, T>(transcript);
        let phi_flat =
            aggregate_jl_contraints_one_lift_seeded::<F, D>(cols, row_bytes, seed, &omega)?;
        let b_full = restore_constant_term(*bb_lift, collapse_to_field::<F>(jl_projection, &omega));
        transcript.append_serde(labels::ABSORB_LABRADOR_BB, bb_lift);
        let beta: CyclotomicRing<F, D> =
            challenge_ring_element(transcript, labels::CHALLENGE_LABRADOR_AGGREGATION);
        b_total += beta * b_full;
        accumulate_phi_flat(&mut phi_total_flat, &phi_flat, beta);
    }

    Ok((phi_total_flat, b_total))
}

// ---------------------------------------------------------------------------
// Statement constraint aggregation (shared by prover and verifier)
// ---------------------------------------------------------------------------

fn pow2_field<F: FieldCore + FromSmallInt>(exp: usize) -> F {
    let two = F::from_u64(2);
    let mut acc = F::one();
    for _ in 0..exp {
        acc = acc * two;
    }
    acc
}

fn accumulate_scaled_row<F: FieldCore, const D: usize>(
    dst: &mut [CyclotomicRing<F, D>],
    src: &[CyclotomicRing<F, D>],
    alpha: &CyclotomicRing<F, D>,
    scale: F,
) {
    debug_assert_eq!(dst.len(), src.len());
    let scaled_alpha = alpha.scale(&scale);
    cfg_iter_mut!(dst)
        .zip(cfg_iter!(src))
        .for_each(|(dst_elem, src_elem)| scaled_alpha.mul_accumulate_into(src_elem, dst_elem));
}

fn accumulate_statement_row_work<F: FieldCore, const D: usize>(
    row: &mut [CyclotomicRing<F, D>],
    work: &[(usize, usize)],
    constraints: &[LabradorConstraint<F, D>],
    alphas: &[CyclotomicRing<F, D>],
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
                    alpha.mul_accumulate_into(src_elem, dst_elem);
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
            alpha.mul_accumulate_into(src_elem, dst_elem);
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
        .take(plan.config.f)
        .any(|&len| len != plan.max_len)
        || row_lengths[layout.aux_row] != layout.aux_row_len()
    {
        return Err(HachiError::InvalidInput(
            "reduced statement row layout mismatch".to_string(),
        ));
    }
    if statement.u1.len() != plan.config.kappa1 || statement.u2.len() != plan.config.kappa1 {
        return Err(HachiError::InvalidInput(
            "reduced statement u1/u2 length mismatch".to_string(),
        ));
    }

    let pow_b: Vec<F> = (0..plan.config.f)
        .map(|idx| pow2_field::<F>(plan.config.b * idx))
        .collect();
    let pow_bu: Vec<F> = (0..plan.config.fu)
        .map(|idx| pow2_field::<F>(plan.config.bu * idx))
        .collect();

    let mut phi_total: Vec<Vec<CyclotomicRing<F, D>>> = row_lengths
        .iter()
        .map(|&len| vec![CyclotomicRing::zero(); len])
        .collect();
    let (z_rows, aux_rows) = phi_total.split_at_mut(plan.config.f);
    let aux_row = aux_rows.first_mut().ok_or_else(|| {
        HachiError::InvalidInput("missing auxiliary row in reduced statement".to_string())
    })?;
    let t_hat_start = layout.t_hat_range().start;
    let h_hat_start = layout.h_hat_range().start;
    let mut b_total = CyclotomicRing::<F, D>::zero();

    for (b_row, target) in plan.setup.b_mat.iter().zip(statement.u1.iter()) {
        let alpha = challenge_ring_element(transcript, labels::CHALLENGE_LABRADOR_AGGREGATION);
        b_total += alpha * *target;
        for (dst, src) in aux_row[t_hat_start..h_hat_start]
            .iter_mut()
            .zip(b_row.iter())
        {
            alpha.mul_accumulate_into(src, dst);
        }
    }

    for (d_row, target) in plan.setup.d_mat.iter().zip(statement.u2.iter()) {
        let alpha = challenge_ring_element(transcript, labels::CHALLENGE_LABRADOR_AGGREGATION);
        b_total += alpha * *target;
        for (dst, src) in aux_row[h_hat_start..].iter_mut().zip(d_row.iter()) {
            alpha.mul_accumulate_into(src, dst);
        }
    }

    for output_idx in 0..plan.config.kappa {
        let alpha = challenge_ring_element(transcript, labels::CHALLENGE_LABRADOR_AGGREGATION);
        let a_row = &plan.setup.a_mat[output_idx];
        for (part_idx, &scale) in pow_b.iter().enumerate() {
            accumulate_scaled_row(&mut z_rows[part_idx], a_row, &alpha, scale);
        }

        for (row_idx, challenge) in plan.challenges.iter().enumerate() {
            let base = alpha * *challenge;
            for (part_idx, &scale) in pow_bu.iter().enumerate() {
                let idx = t_hat_start
                    + row_idx * plan.config.kappa * plan.config.fu
                    + output_idx * plan.config.fu
                    + part_idx;
                aux_row[idx] -= base.scale(&scale);
            }
        }
    }

    let alpha_lg = challenge_ring_element(transcript, labels::CHALLENGE_LABRADOR_AGGREGATION);
    for (part_idx, &scale) in pow_b.iter().enumerate() {
        accumulate_scaled_row(&mut z_rows[part_idx], &plan.combined_phi, &alpha_lg, scale);
    }
    for i in 0..plan.challenges.len() {
        for j in i..plan.challenges.len() {
            let base = alpha_lg * plan.challenges[i] * plan.challenges[j];
            let pair = pair_index(i, j, plan.challenges.len());
            for (part_idx, &scale) in pow_bu.iter().enumerate() {
                let idx = h_hat_start + pair * plan.config.fu + part_idx;
                aux_row[idx] -= base.scale(&scale);
            }
        }
    }

    let alpha_diag = challenge_ring_element(transcript, labels::CHALLENGE_LABRADOR_AGGREGATION);
    b_total += alpha_diag * plan.b_total;
    for i in 0..plan.row_count {
        let pair = pair_index(i, i, plan.row_count);
        for (part_idx, &scale) in pow_bu.iter().enumerate() {
            let idx = h_hat_start + pair * plan.config.fu + part_idx;
            aux_row[idx] += alpha_diag.scale(&scale);
        }
    }

    Ok((phi_total, b_total))
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

/// Fold statement constraints into aggregated (φ, b) using ring-element
/// challenges sampled from the transcript.
///
/// Each scalar constraint is folded with one fresh dense challenge α: its
/// coefficient terms are fused-accumulated into `phi_total`, while `α · target`
/// is accumulated into `b_total`.
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
    let alphas: Vec<CyclotomicRing<F, D>> = constraints
        .iter()
        .map(|_| challenge_ring_element(transcript, labels::CHALLENGE_LABRADOR_AGGREGATION))
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

    // Phase 3: b_total — parallel fold-reduce over constraints.
    let b_total = cfg_fold_reduce!(
        (0..constraints.len()),
        || CyclotomicRing::<F, D>::zero(),
        |mut acc, i| {
            alphas[i].mul_accumulate_into(&constraints[i].target, &mut acc);
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

    Ok((phi_total, b_total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Prime128M13M4P0;
    use crate::algebra::{Pow2Offset32Field, Pow2Offset64Field};
    use crate::protocol::transcript::labels::DOMAIN_LABRADOR_PROTOCOL;
    use crate::protocol::transcript::Blake2bTranscript;

    const D: usize = 64;
    const TEST_RING_ELEMS: usize = 16;
    const TEST_COLS: usize = TEST_RING_ELEMS * D;

    fn assert_aggregate_jl_contraints_one_lift_matches_naive<
        F: FieldCore + CanonicalField + FromSmallInt,
    >(
        matrix: &LabradorJlMatrix,
        omega: [F; 256],
    ) {
        let cols = matrix.cols();
        assert_eq!(cols, TEST_COLS);

        let got = aggregate_jl_contraints_one_lift::<F, D>(&matrix, &omega).unwrap();
        let mut expected = vec![CyclotomicRing::<F, D>::zero(); cols / D];
        // Naive paper-style reference:
        // φ''_i = Σ_j ω_j · σ_{-1}(π_i^(j))
        for (elem_idx, elem) in expected.iter_mut().enumerate() {
            for (row_idx, &alpha) in omega.iter().enumerate() {
                let row = matrix.row_bytes(row_idx);
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
        let mut transcript = Blake2bTranscript::<Pow2Offset32Field>::new(DOMAIN_LABRADOR_PROTOCOL);
        let matrix =
            LabradorJlMatrix::generate::<Pow2Offset32Field, _>(&mut transcript, TEST_COLS).unwrap();
        let omega = sample_jl_collapse_challenge::<Pow2Offset32Field, _>(&mut transcript);
        assert_aggregate_jl_contraints_one_lift_matches_naive::<Pow2Offset32Field>(&matrix, omega);
    }

    #[test]
    fn aggregate_jl_contraints_one_lift_matches_naive_fp64() {
        type F64 = Pow2Offset64Field;
        let mut transcript = Blake2bTranscript::<F64>::new(DOMAIN_LABRADOR_PROTOCOL);
        let matrix = LabradorJlMatrix::generate::<F64, _>(&mut transcript, TEST_COLS).unwrap();
        let omega = sample_jl_collapse_challenge::<F64, _>(&mut transcript);
        assert_aggregate_jl_contraints_one_lift_matches_naive::<F64>(&matrix, omega);
    }

    #[test]
    fn aggregate_jl_contraints_one_lift_matches_naive_fp128() {
        type F128 = Prime128M13M4P0;
        let mut transcript = Blake2bTranscript::<F128>::new(DOMAIN_LABRADOR_PROTOCOL);
        let matrix = LabradorJlMatrix::generate::<F128, _>(&mut transcript, TEST_COLS).unwrap();
        let omega = sample_jl_collapse_challenge::<F128, _>(&mut transcript);
        assert_aggregate_jl_contraints_one_lift_matches_naive::<F128>(&matrix, omega);
    }
}
