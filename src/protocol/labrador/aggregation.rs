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
use crate::protocol::labrador::constraints::LabradorConstraint;
use crate::protocol::labrador::johnson_lindenstrauss::{
    collapse, restore_constant_term, zero_constant_term_for_proof, LabradorJlMatrix,
};
use crate::protocol::labrador::types::LabradorWitness;
use crate::protocol::transcript::labels;
use crate::protocol::transcript::{challenge_ring_element_rejection_sampled, Transcript};
use crate::{CanonicalField, FieldCore, FromSmallInt};

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

/// Element-wise accumulate `other` into `acc`.
pub(crate) fn add_phi_in_place<F: FieldCore, const D: usize>(
    acc: &mut [Vec<CyclotomicRing<F, D>>],
    other: &[Vec<CyclotomicRing<F, D>>],
) -> Result<(), HachiError> {
    if acc.len() != other.len() {
        return Err(HachiError::InvalidInput(
            "phi row count mismatch".to_string(),
        ));
    }
    for (row_acc, row_other) in acc.iter().zip(other.iter()) {
        if row_acc.len() != row_other.len() {
            return Err(HachiError::InvalidInput(
                "phi row length mismatch".to_string(),
            ));
        }
    }
    cfg_iter_mut!(acc)
        .zip(cfg_iter!(other))
        .for_each(|(row_acc, row_other)| {
            for (a, b) in row_acc.iter_mut().zip(row_other.iter()) {
                *a += *b;
            }
        });
    Ok(())
}

/// Sample 256 centered-representative scalars from the transcript.
///
/// Each scalar is squeezed as a field element then converted to a signed
/// integer in [−(q−1)/2, (q−1)/2]. These are the ω_j^(k) challenges.
fn sample_jl_collapse_challenge<F, T>(transcript: &mut T) -> [i64; 256]
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let q = (-F::one()).to_canonical_u128() + 1;
    let half_q = q / 2;
    std::array::from_fn(|_| {
        let s = transcript.challenge_scalar(labels::CHALLENGE_LABRADOR_JL_COLLAPSE);
        let c = s.to_canonical_u128();
        if c > half_q {
            -((q - c) as i64)
        } else {
            c as i64
        }
    })
}

/// Collapse 256 JL matrix rows with challenge weights into φ ring-elements.
///
/// Computes `φ = Σ_j ω_j · σ_{-1}(Π_j)`, where Π_j is the j-th row of the
/// JL matrix interpreted as ring elements, and `σ_{-1}` is the conjugation
/// automorphism.
fn jl_collapse_phi_from_weights<F: FieldCore + CanonicalField + FromSmallInt, const D: usize>(
    matrix: &LabradorJlMatrix,
    omega: &[i64; 256],
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    let cols = matrix.cols();
    if cols % D != 0 {
        return Err(HachiError::InvalidInput(
            "JL matrix cols not divisible by ring degree".to_string(),
        ));
    }
    let mut weights = vec![0i64; cols];
    for (row_idx, row) in matrix.signs.iter().enumerate() {
        let alpha = omega[row_idx];
        for (col_idx, &sign) in row.iter().enumerate() {
            weights[col_idx] += alpha * (sign as i64);
        }
    }

    let ring_elems = cols / D;
    let phi: Vec<CyclotomicRing<F, D>> = cfg_into_iter!(0..ring_elems)
        .map(|idx| {
            let coeffs = std::array::from_fn(|k| {
                let w = weights[idx * D + k];
                F::from_i64(w)
            });
            CyclotomicRing::from_coefficients(coeffs).sigma_m1()
        })
        .collect();
    Ok(phi)
}

/// Flatten witness rows into a single ring-element vector, returning index
/// ranges that map each row back into the flat vector.
fn flatten_witness<F: FieldCore, const D: usize>(
    witness: &LabradorWitness<F, D>,
) -> (Vec<CyclotomicRing<F, D>>, Vec<(usize, usize)>) {
    let mut flat = Vec::new();
    let mut ranges = Vec::with_capacity(witness.rows().len());
    let mut cursor = 0usize;
    for row in witness.rows() {
        let start = cursor;
        flat.extend(row.iter().copied());
        cursor += row.len();
        ranges.push((start, cursor));
    }
    (flat, ranges)
}

/// Build index ranges from row lengths (verifier variant of [`flatten_witness`]).
fn ranges_from_row_lengths(row_lengths: &[usize]) -> Vec<(usize, usize)> {
    let mut ranges = Vec::with_capacity(row_lengths.len());
    let mut cursor = 0usize;
    for &len in row_lengths {
        let start = cursor;
        cursor += len;
        ranges.push((start, cursor));
    }
    ranges
}

/// Fold `phi_flat` into per-row accumulators scaled by `beta`.
fn accumulate_phi<F: FieldCore, const D: usize>(
    phi_total: &mut [Vec<CyclotomicRing<F, D>>],
    phi_flat: &[CyclotomicRing<F, D>],
    ranges: &[(usize, usize)],
    beta: CyclotomicRing<F, D>,
) {
    for (row_idx, &(start, end)) in ranges.iter().enumerate() {
        let row = &phi_flat[start..end];
        for (j, elem) in row.iter().enumerate() {
            phi_total[row_idx][j] += beta * *elem;
        }
    }
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
/// Returns `(phi_total, b_total, bb)` where `bb` holds the transmitted
/// polynomials.
#[allow(clippy::type_complexity)]
pub(crate) fn aggregate_jl_constraints_prover<F, T, const D: usize>(
    witness: &LabradorWitness<F, D>,
    jl_projection: &[i32; 256],
    matrix: &LabradorJlMatrix,
    transcript: &mut T,
) -> Result<
    (
        Vec<Vec<CyclotomicRing<F, D>>>,
        CyclotomicRing<F, D>,
        Vec<CyclotomicRing<F, D>>,
    ),
    HachiError,
>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    let (flat, ranges) = flatten_witness(witness);
    let mut phi_total: Vec<Vec<CyclotomicRing<F, D>>> = witness
        .rows()
        .iter()
        .map(|row| vec![CyclotomicRing::zero(); row.len()])
        .collect();
    let mut b_total = CyclotomicRing::<F, D>::zero();
    let lifts = jl_lifts::<F>();
    let mut bb = Vec::with_capacity(lifts);

    for _ in 0..lifts {
        let omega = sample_jl_collapse_challenge::<F, T>(transcript);
        let phi_flat = jl_collapse_phi_from_weights::<F, D>(matrix, &omega)?;
        let b_full = dot_product(&phi_flat, &flat);
        let target = collapse(jl_projection, &omega);
        let expected_c0 = F::from_i64(target);
        if b_full.coefficients()[0] != expected_c0 {
            return Err(HachiError::InvalidProof);
        }
        let (b_tx, _c0) = zero_constant_term_for_proof(b_full);
        bb.push(b_tx);
        transcript.append_serde(labels::ABSORB_LABRADOR_BB, &b_tx);

        let beta = challenge_ring_element_rejection_sampled(
            transcript,
            labels::CHALLENGE_LABRADOR_AGGREGATION,
        )?;
        b_total += beta * b_full;
        accumulate_phi(&mut phi_total, &phi_flat, &ranges, beta);
    }

    Ok((phi_total, b_total, bb))
}

// ---------------------------------------------------------------------------
// Verifier-side JL aggregation
// ---------------------------------------------------------------------------

/// Aggregate JL projection constraints on the verifier side.
///
/// Same transcript flow as the prover variant, but reconstructs the full
/// polynomial b^''(k) by restoring the constant term from the projection
/// and the transmitted `bb[k]`.
#[allow(clippy::type_complexity)]
pub(crate) fn aggregate_jl_constraints_verifier<F, T, const D: usize>(
    row_lengths: &[usize],
    jl_projection: &[i32; 256],
    matrix: &LabradorJlMatrix,
    bb: &[CyclotomicRing<F, D>],
    transcript: &mut T,
) -> Result<(Vec<Vec<CyclotomicRing<F, D>>>, CyclotomicRing<F, D>), HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    let lifts = jl_lifts::<F>();
    if bb.len() != lifts {
        return Err(HachiError::InvalidProof);
    }
    let ranges = ranges_from_row_lengths(row_lengths);

    let mut phi_total: Vec<Vec<CyclotomicRing<F, D>>> = row_lengths
        .iter()
        .map(|&len| vec![CyclotomicRing::zero(); len])
        .collect();
    let mut b_total = CyclotomicRing::<F, D>::zero();

    for bb_lift in bb.iter() {
        let omega = sample_jl_collapse_challenge::<F, T>(transcript);
        let phi_flat = jl_collapse_phi_from_weights::<F, D>(matrix, &omega)?;
        let target = collapse(jl_projection, &omega);
        let b_full = restore_constant_term(*bb_lift, F::from_i64(target));
        transcript.append_serde(labels::ABSORB_LABRADOR_BB, bb_lift);
        let beta = challenge_ring_element_rejection_sampled(
            transcript,
            labels::CHALLENGE_LABRADOR_AGGREGATION,
        )?;
        b_total += beta * b_full;
        accumulate_phi(&mut phi_total, &phi_flat, &ranges, beta);
    }

    Ok((phi_total, b_total))
}

// ---------------------------------------------------------------------------
// Statement constraint aggregation (shared by prover and verifier)
// ---------------------------------------------------------------------------

/// Fold statement constraints into aggregated (φ, b) using ring-element
/// challenges sampled from the transcript.
///
/// Each scalar constraint is folded with one fresh challenge α: its sparse
/// coefficient terms are added into `phi_total`, while `α · target` is
/// accumulated into `b_total`.
#[allow(clippy::type_complexity)]
pub(crate) fn aggregate_statement_constraints<F, T, const D: usize>(
    constraints: &[LabradorConstraint<F, D>],
    row_lengths: &[usize],
    transcript: &mut T,
) -> Result<(Vec<Vec<CyclotomicRing<F, D>>>, CyclotomicRing<F, D>), HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    let mut phi_total: Vec<Vec<CyclotomicRing<F, D>>> = row_lengths
        .iter()
        .map(|&len| vec![CyclotomicRing::zero(); len])
        .collect();
    let mut b_total = CyclotomicRing::<F, D>::zero();

    if constraints.is_empty() {
        return Ok((phi_total, b_total));
    }

    for cnst in constraints {
        let alpha = challenge_ring_element_rejection_sampled(
            transcript,
            labels::CHALLENGE_LABRADOR_AGGREGATION,
        )?;
        b_total += alpha * cnst.target;

        for term in &cnst.terms {
            if term.row >= phi_total.len() {
                return Err(HachiError::InvalidInput(
                    "constraint row index out of bounds".to_string(),
                ));
            }
            let row = &mut phi_total[term.row];
            if term.offset + term.coefficients.len() > row.len() {
                return Err(HachiError::InvalidInput(
                    "constraint term exceeds row length".to_string(),
                ));
            }
            for (j, coeff) in term.coefficients.iter().enumerate() {
                row[term.offset + j] += alpha * *coeff;
            }
        }
    }

    Ok((phi_total, b_total))
}
