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
    restore_constant_term, zero_constant_term_for_proof, LabradorJlMatrix,
};
use crate::protocol::labrador::types::{
    LabradorReducedConstraintPlan, LabradorStatement, LabradorWitness,
};
use crate::protocol::transcript::labels;
use crate::protocol::transcript::{challenge_ring_element, Transcript};
use crate::{CanonicalField, FieldCore, FromSmallInt};

type AggregatedConstraintSystem<F, const D: usize> =
    (Vec<Vec<CyclotomicRing<F, D>>>, CyclotomicRing<F, D>);

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
#[tracing::instrument(skip_all, name = "labrador::add_phi_in_place")]
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
// Width-dispatched JL weight collapse
// ---------------------------------------------------------------------------

/// Collapsed JL weights in the narrowest representation that avoids overflow.
enum CollapseWeights<F: FieldCore, const D: usize> {
    I64(Vec<i64>),
    I128(Vec<i128>),
    FieldPhi(Vec<CyclotomicRing<F, D>>),
}

impl<F: FieldCore + CanonicalField + FromSmallInt, const D: usize> CollapseWeights<F, D> {
    fn cols(&self) -> usize {
        match self {
            Self::I64(w) => w.len(),
            Self::I128(w) => w.len(),
            Self::FieldPhi(phi) => phi.len() * D,
        }
    }

    /// Build φ ring-elements: convert each D-coefficient block and apply σ_{-1}.
    fn into_phi(self) -> Vec<CyclotomicRing<F, D>> {
        let cols = self.cols();
        debug_assert!(cols % D == 0, "cols ({cols}) not divisible by D ({D})");
        let ring_elems = cols / D;
        match self {
            Self::I64(w) => cfg_into_iter!(0..ring_elems)
                .map(|idx| {
                    let coeffs = std::array::from_fn(|k| F::from_i64(w[idx * D + k]));
                    CyclotomicRing::from_coefficients(coeffs).sigma_m1()
                })
                .collect(),
            Self::I128(w) => cfg_into_iter!(0..ring_elems)
                .map(|idx| {
                    let coeffs = std::array::from_fn(|k| F::from_i128(w[idx * D + k]));
                    CyclotomicRing::from_coefficients(coeffs).sigma_m1()
                })
                .collect(),
            Self::FieldPhi(phi) => phi,
        }
    }

    /// Compute b = ⟨φ, s⟩, using the integer fast path when available.
    fn compute_b(&self, witness: &FlatWitness<F, D>) -> CyclotomicRing<F, D> {
        if let (Self::I64(weights), Some(wi64)) = (self, &witness.centered_i64) {
            integer_ring_dot_sigma_m1::<F, D>(weights, wi64, self.cols() / D)
        } else if let Self::FieldPhi(phi) = self {
            dot_product(phi, &witness.rings)
        } else {
            let phi = match self {
                Self::I128(weights) => cfg_into_iter!(0..self.cols() / D)
                    .map(|idx| {
                        let coeffs = std::array::from_fn(|k| F::from_i128(weights[idx * D + k]));
                        CyclotomicRing::from_coefficients(coeffs).sigma_m1()
                    })
                    .collect::<Vec<_>>(),
                Self::I64(_) | Self::FieldPhi(_) => unreachable!(),
            };
            dot_product(&phi, &witness.rings)
        }
    }
}

// ---------------------------------------------------------------------------
// Collapse helpers
// ---------------------------------------------------------------------------

/// Center 256 omega field elements to i128 and return the maximum magnitude.
fn center_omega_to_i128<F: CanonicalField>(omega: &[F; 256]) -> ([i128; 256], u128) {
    let q = (-F::one()).to_canonical_u128() + 1;
    let half_q = q / 2;
    let mut centered = [0i128; 256];
    let mut max_mag: u128 = 0;
    for (i, &val) in omega.iter().enumerate() {
        let canonical = val.to_canonical_u128();
        if canonical > half_q {
            let mag = q - canonical;
            centered[i] = -(mag as i128);
            if mag > max_mag {
                max_mag = mag;
            }
        } else {
            centered[i] = canonical as i128;
            if canonical > max_mag {
                max_mag = canonical;
            }
        }
    }
    (centered, max_mag)
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
fn jl_pair_unit_i64(pair: u8) -> i64 {
    ((pair == 0b11) as i64) - ((pair == 0b00) as i64)
}

#[inline]
fn jl_pair_unit_i128(pair: u8) -> i128 {
    ((pair == 0b11) as i128) - ((pair == 0b00) as i128)
}

#[inline]
fn row_pair_at(row: &[u8], col_idx: usize) -> u8 {
    let shift = (col_idx & 0b11) << 1;
    (row[col_idx >> 2] >> shift) & 0b11
}

#[inline]
fn accumulate_field_pair<F: FieldCore, const D: usize>(
    coeffs: &mut [F],
    local_idx: usize,
    pair: u8,
    alpha: F,
) {
    match pair {
        0b00 => {
            if local_idx == 0 {
                coeffs[0] -= alpha;
            } else {
                coeffs[D - local_idx] += alpha;
            }
        }
        0b11 => {
            if local_idx == 0 {
                coeffs[0] += alpha;
            } else {
                coeffs[D - local_idx] -= alpha;
            }
        }
        _ => {}
    }
}

/// Branchless i64 collapse. Caller guarantees `256 * max|omega|` fits in i64.
#[tracing::instrument(skip_all, name = "labrador::collapse_weights_i64")]
fn collapse_weights_i64(matrix: &LabradorJlMatrix, omega: &[i64; 256], cols: usize) -> Vec<i64> {
    let mut weights = vec![0i64; cols];
    let full_bytes = cols >> 2;
    let remainder = cols & 0b11;
    for (row_idx, &alpha) in omega.iter().enumerate() {
        let row = matrix.row_bytes(row_idx);
        let mut col_idx = 0usize;
        for &byte in row.iter().take(full_bytes) {
            weights[col_idx] += alpha * jl_pair_unit_i64(byte & 0b11);
            weights[col_idx + 1] += alpha * jl_pair_unit_i64((byte >> 2) & 0b11);
            weights[col_idx + 2] += alpha * jl_pair_unit_i64((byte >> 4) & 0b11);
            weights[col_idx + 3] += alpha * jl_pair_unit_i64((byte >> 6) & 0b11);
            col_idx += 4;
        }
        if remainder > 0 {
            let byte = row[full_bytes];
            for lane in 0..remainder {
                let pair = (byte >> (lane << 1)) & 0b11;
                weights[col_idx] += alpha * jl_pair_unit_i64(pair);
                col_idx += 1;
            }
        }
    }
    weights
}

/// Branchless i128 collapse. Caller guarantees `256 * max|omega|` fits in i128.
#[tracing::instrument(skip_all, name = "labrador::collapse_weights_i128")]
fn collapse_weights_i128_unchecked(
    matrix: &LabradorJlMatrix,
    omega: &[i128; 256],
    cols: usize,
) -> Vec<i128> {
    let mut weights = vec![0i128; cols];
    let full_bytes = cols >> 2;
    let remainder = cols & 0b11;
    for (row_idx, &alpha) in omega.iter().enumerate() {
        let row = matrix.row_bytes(row_idx);
        let mut col_idx = 0usize;
        for &byte in row.iter().take(full_bytes) {
            weights[col_idx] += alpha * jl_pair_unit_i128(byte & 0b11);
            weights[col_idx + 1] += alpha * jl_pair_unit_i128((byte >> 2) & 0b11);
            weights[col_idx + 2] += alpha * jl_pair_unit_i128((byte >> 4) & 0b11);
            weights[col_idx + 3] += alpha * jl_pair_unit_i128((byte >> 6) & 0b11);
            col_idx += 4;
        }
        if remainder > 0 {
            let byte = row[full_bytes];
            for lane in 0..remainder {
                let pair = (byte >> (lane << 1)) & 0b11;
                weights[col_idx] += alpha * jl_pair_unit_i128(pair);
                col_idx += 1;
            }
        }
    }
    weights
}

/// Field-arithmetic accumulation for fields too wide for integer collapse.
#[tracing::instrument(skip_all, name = "labrador::collapse_weights_field")]
fn collapse_weights_field<F: FieldCore, const D: usize>(
    matrix: &LabradorJlMatrix,
    omega: &[F; 256],
    cols: usize,
) -> Vec<CyclotomicRing<F, D>> {
    debug_assert!(cols % D == 0, "cols ({cols}) not divisible by D ({D})");
    let mut phi = vec![CyclotomicRing::<F, D>::zero(); cols / D];
    if D % 4 == 0 {
        let bytes_per_ring = D / 4;
        for (row_idx, &alpha) in omega.iter().enumerate() {
            let row = matrix.row_bytes(row_idx);
            for (elem, sign_bytes) in phi.iter_mut().zip(row.chunks_exact(bytes_per_ring)) {
                let coeffs = elem.coefficients_mut();
                let mut local_idx = 0usize;
                for &byte in sign_bytes {
                    accumulate_field_pair::<F, D>(coeffs, local_idx, byte & 0b11, alpha);
                    accumulate_field_pair::<F, D>(coeffs, local_idx + 1, (byte >> 2) & 0b11, alpha);
                    accumulate_field_pair::<F, D>(coeffs, local_idx + 2, (byte >> 4) & 0b11, alpha);
                    accumulate_field_pair::<F, D>(coeffs, local_idx + 3, (byte >> 6) & 0b11, alpha);
                    local_idx += 4;
                }
            }
        }
    } else {
        for (row_idx, &alpha) in omega.iter().enumerate() {
            let row = matrix.row_bytes(row_idx);
            let mut col_idx = 0usize;
            for elem in &mut phi {
                let coeffs = elem.coefficients_mut();
                for local_idx in 0..D {
                    accumulate_field_pair::<F, D>(
                        coeffs,
                        local_idx,
                        row_pair_at(row, col_idx),
                        alpha,
                    );
                    col_idx += 1;
                }
            }
        }
    }
    phi
}

/// Collapse 256 JL rows × omega into weights, dispatching to the narrowest
/// integer type that avoids overflow.
#[tracing::instrument(skip_all, name = "labrador::collapse_jl_weights")]
fn collapse_jl_weights<F: CanonicalField, const D: usize>(
    matrix: &LabradorJlMatrix,
    omega: &[F; 256],
) -> Result<CollapseWeights<F, D>, HachiError> {
    let cols = matrix.cols();
    validate_matrix_cols(matrix, cols)?;
    let (omega_centered, max_mag) = center_omega_to_i128(omega);

    const HEADROOM_I64: u128 = (i64::MAX as u128) / 256;
    const HEADROOM_I128: u128 = (i128::MAX as u128) / 256;

    if max_mag <= HEADROOM_I64 {
        let omega_i64: [i64; 256] = std::array::from_fn(|i| omega_centered[i] as i64);
        Ok(CollapseWeights::I64(collapse_weights_i64(
            matrix, &omega_i64, cols,
        )))
    } else if max_mag <= HEADROOM_I128 {
        Ok(CollapseWeights::I128(collapse_weights_i128_unchecked(
            matrix,
            &omega_centered,
            cols,
        )))
    } else {
        Ok(CollapseWeights::FieldPhi(collapse_weights_field::<F, D>(
            matrix, omega, cols,
        )))
    }
}

// ---------------------------------------------------------------------------
// Integer ring dot product for b computation
// ---------------------------------------------------------------------------

fn i128_to_field<F: CanonicalField>(val: i128) -> F {
    if val >= 0 {
        F::from_canonical_u128_reduced(val as u128)
    } else {
        -F::from_canonical_u128_reduced(val.unsigned_abs())
    }
}

/// Center witness coefficients to flat i64 values in the same order as
/// [`flatten_witness`]. Returns `None` if any coefficient exceeds i64 range.
fn center_flat_witness_i64<F: FieldCore + CanonicalField, const D: usize>(
    witness: &LabradorWitness<F, D>,
) -> Option<Vec<i64>> {
    let q = (-F::one()).to_canonical_u128() + 1;
    let half_q = q / 2;
    let total_coeffs: usize = witness.rows().iter().map(|r| r.len() * D).sum();
    let mut centered = Vec::with_capacity(total_coeffs);
    for row in witness.rows() {
        for ring in row {
            for &coeff in ring.coefficients() {
                let canonical = coeff.to_canonical_u128();
                if canonical > half_q {
                    let mag = q - canonical;
                    if mag > i64::MAX as u128 {
                        return None;
                    }
                    centered.push(-(mag as i64));
                } else {
                    if canonical > i64::MAX as u128 {
                        return None;
                    }
                    centered.push(canonical as i64);
                }
            }
        }
    }
    Some(centered)
}

/// Compute `b = Σ_c σ_{-1}(w_c) · s_c` in the negacyclic ring using integer
/// arithmetic (i64 × i64 → i128 accumulation, reduced mod q once at the end).
///
/// `weights` and `witness_flat` are both `ring_elems * D` long, laid out as
/// consecutive D-coefficient blocks per ring element.
#[tracing::instrument(skip_all, name = "labrador::integer_ring_dot_sigma_m1")]
fn integer_ring_dot_sigma_m1<F: CanonicalField, const D: usize>(
    weights: &[i64],
    witness_flat: &[i64],
    ring_elems: usize,
) -> CyclotomicRing<F, D> {
    debug_assert_eq!(weights.len(), ring_elems * D);
    debug_assert_eq!(witness_flat.len(), ring_elems * D);

    let mut result = [0i128; D];
    for c in 0..ring_elems {
        let w = &weights[c * D..(c + 1) * D];
        let s = &witness_flat[c * D..(c + 1) * D];

        // sigma_m1(w) = [w[0], -w[D-1], -w[D-2], ..., -w[1]]
        // Negacyclic convolution: X^D = -1

        // m = 0 contribution: sigma_m1(w)[0] = w[0], no wrap for any k
        let a0 = w[0] as i128;
        for k in 0..D {
            result[k] += a0 * (s[k] as i128);
        }

        for m in 1..D {
            let a_m = -(w[D - m] as i128); // sigma_m1(w)[m]
                                           // k = m..D-1: no wrap (positive sign)
            for k in m..D {
                result[k] += a_m * (s[k - m] as i128);
            }
            // k = 0..m-1: wrap (negacyclic negative sign)
            for k in 0..m {
                result[k] -= a_m * (s[k + D - m] as i128);
            }
        }
    }

    CyclotomicRing::from_coefficients(std::array::from_fn(|k| i128_to_field::<F>(result[k])))
}

// ---------------------------------------------------------------------------
// Witness flattening
// ---------------------------------------------------------------------------

/// Pre-flattened witness with optional i64 centering for the integer fast path.
struct FlatWitness<F: FieldCore, const D: usize> {
    rings: Vec<CyclotomicRing<F, D>>,
    centered_i64: Option<Vec<i64>>,
    ranges: Vec<(usize, usize)>,
}

impl<F: FieldCore + CanonicalField, const D: usize> FlatWitness<F, D> {
    #[tracing::instrument(skip_all, name = "labrador::flat_witness_new")]
    fn new(witness: &LabradorWitness<F, D>) -> Self {
        let mut rings = Vec::new();
        let mut ranges = Vec::with_capacity(witness.rows().len());
        let mut cursor = 0usize;
        for row in witness.rows() {
            let start = cursor;
            rings.extend(row.iter().copied());
            cursor += row.len();
            ranges.push((start, cursor));
        }
        let centered_i64 = center_flat_witness_i64::<F, D>(witness);
        Self {
            rings,
            centered_i64,
            ranges,
        }
    }
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
    cfg_iter_mut!(phi_total)
        .zip(cfg_iter!(ranges))
        .for_each(|(row, &(start, end))| {
            for (dst, src) in row.iter_mut().zip(phi_flat[start..end].iter()) {
                beta.mul_accumulate_into(src, dst);
            }
        });
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
#[tracing::instrument(skip_all, name = "labrador::aggregate_jl_constraints_prover")]
pub(crate) fn aggregate_jl_constraints_prover<F, T, const D: usize>(
    witness: &LabradorWitness<F, D>,
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
    let flat_witness = FlatWitness::new(witness);

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
        let cw = collapse_jl_weights::<F, D>(matrix, &omega)?;
        let b_full = cw.compute_b(&flat_witness);
        let phi_flat = cw.into_phi();

        let (b_tx, _c0) = zero_constant_term_for_proof(b_full);
        bb.push(b_tx);
        transcript.append_serde(labels::ABSORB_LABRADOR_BB, &b_tx);

        let beta: CyclotomicRing<F, D> =
            challenge_ring_element(transcript, labels::CHALLENGE_LABRADOR_AGGREGATION);
        b_total += beta * b_full;
        accumulate_phi(&mut phi_total, &phi_flat, &flat_witness.ranges, beta);
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
#[tracing::instrument(skip_all, name = "labrador::aggregate_jl_constraints_verifier")]
pub(crate) fn aggregate_jl_constraints_verifier<F, T, const D: usize>(
    row_lengths: &[usize],
    jl_projection: &[i64; 256],
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
        let cw = collapse_jl_weights::<F, D>(matrix, &omega)?;
        let phi_flat = cw.into_phi();
        let b_full = restore_constant_term(*bb_lift, collapse_to_field::<F>(jl_projection, &omega));
        transcript.append_serde(labels::ABSORB_LABRADOR_BB, bb_lift);
        let beta: CyclotomicRing<F, D> =
            challenge_ring_element(transcript, labels::CHALLENGE_LABRADOR_AGGREGATION);
        b_total += beta * b_full;
        accumulate_phi(&mut phi_total, &phi_flat, &ranges, beta);
    }

    Ok((phi_total, b_total))
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
    let scaled_alpha = alpha.scale(&scale);
    for (dst_elem, src_elem) in dst.iter_mut().zip(src.iter()) {
        scaled_alpha.mul_accumulate_into(src_elem, dst_elem);
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
            for &(ci, ti) in &work {
                let term = &constraints[ci].terms[ti];
                let alpha = &alphas[ci];
                for (j, coeff) in term.coefficients.iter().enumerate() {
                    alpha.mul_accumulate_into(coeff, &mut row[term.offset + j]);
                }
            }
            row
        })
        .collect();

    Ok((phi_total, b_total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Prime128M13M4P0;
    use crate::protocol::transcript::labels::DOMAIN_LABRADOR_PROTOCOL;
    use crate::protocol::transcript::Blake2bTranscript;

    type F = Prime128M13M4P0;
    const D: usize = 64;

    #[test]
    fn collapse_weights_field_matches_dense_reference() {
        let cols = 2 * D;
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let matrix = LabradorJlMatrix::generate::<F, _>(&mut transcript, cols).unwrap();
        let omega = std::array::from_fn(|i| F::from_canonical_u128_reduced((i as u128 + 17) << 72));

        let got = collapse_weights_field::<F, D>(&matrix, &omega, cols);
        let mut expected = vec![CyclotomicRing::<F, D>::zero(); cols / D];
        for (row_idx, &alpha) in omega.iter().enumerate() {
            let row = matrix.row_bytes(row_idx);
            for (elem_idx, elem) in expected.iter_mut().enumerate() {
                let coeffs = elem.coefficients_mut();
                for local_idx in 0..D {
                    let col_idx = elem_idx * D + local_idx;
                    let pair = row_pair_at(row, col_idx);
                    accumulate_field_pair::<F, D>(coeffs, local_idx, pair, alpha);
                }
            }
        }

        assert_eq!(got, expected);
    }
}
