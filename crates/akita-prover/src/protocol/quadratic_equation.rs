//! Quadratic equation builder for the Akita PCS (§4.2).
//!
//! This module encapsulates the stage-1 prover logic and the generation of
//! the quadratic equation components M, y, z, and v.

use crate::kernels::crt_ntt::NttSlotCache;
use crate::kernels::linear::{
    fused_split_eq_quotients, mat_vec_mul_ntt_single_i8, mat_vec_mul_ntt_single_i8_cyclic,
};
#[cfg(feature = "zk")]
use crate::protocol::masking::sample_blinding_digits;
use crate::{AkitaPolyOps, DecomposeFoldWitness, RecursiveWitnessView};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2I8Params;
use akita_algebra::CyclotomicRing;
use akita_challenges::sample_sparse_challenges;
use akita_challenges::SparseChallenge;
use akita_field::parallel::*;
use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore, HalvingField};
use akita_transcript::labels::{ABSORB_PROVER_V, CHALLENGE_STAGE1_FOLD};
use akita_transcript::Transcript;
use akita_types::RingOpeningPoint;
use akita_types::{AkitaCommitmentHint, FlatDigitBlocks, RingCommitment, RingSliceSerializer};
use akita_types::{AkitaExpandedSetup, ClaimIncidenceSummary, LevelParams};
use std::iter::repeat_n;
use std::time::Instant;

fn beta_linf_fold_bound(
    r: usize,
    challenge_l1_mass: usize,
    log_basis: u32,
) -> Result<u128, AkitaError> {
    if !(1..128).contains(&log_basis) {
        return Err(AkitaError::InvalidSetup("invalid LOG_BASIS".to_string()));
    }
    if r >= 128 {
        return Err(AkitaError::InvalidSetup("r_vars must be < 128".to_string()));
    }

    let blocks = 1u128 << r;
    let b = 1u128 << log_basis;
    let half_b = b / 2;

    let term = blocks
        .checked_mul(challenge_l1_mass as u128)
        .ok_or_else(|| AkitaError::InvalidSetup("beta bound overflow".to_string()))?;
    term.checked_mul(half_b)
        .ok_or_else(|| AkitaError::InvalidSetup("beta bound overflow".to_string()))
}

fn beta_linf_fold_bound_with_num_claims(
    r: usize,
    challenge_l1_mass: usize,
    log_basis: u32,
    num_claims: usize,
) -> Result<u128, AkitaError> {
    let beta = beta_linf_fold_bound(r, challenge_l1_mass, log_basis)?;
    beta.checked_mul(num_claims as u128)
        .ok_or_else(|| AkitaError::InvalidSetup("batched beta bound overflow".to_string()))
}

fn validate_decompose_fold<F: FieldCore + CanonicalField, const D: usize>(
    z: DecomposeFoldWitness<F, D>,
    lp: &LevelParams,
    num_claims: usize,
) -> Result<DecomposeFoldWitness<F, D>, AkitaError> {
    let norm = u128::from(z.centered_inf_norm);
    let beta = beta_linf_fold_bound_with_num_claims(
        lp.r_vars,
        lp.challenge_l1_mass(),
        lp.log_basis,
        num_claims,
    )?;
    if norm > beta {
        return Err(AkitaError::InvalidInput(format!(
            "prover abort: ||z||_inf = {norm} > beta = {beta}"
        )));
    }
    Ok(z)
}

fn aggregate_decompose_fold_witnesses<F: FieldCore, const D: usize>(
    witnesses: Vec<DecomposeFoldWitness<F, D>>,
) -> Result<DecomposeFoldWitness<F, D>, AkitaError> {
    let Some((first, rest)) = witnesses.split_first() else {
        return Err(AkitaError::InvalidInput(
            "batched decompose_fold requires at least one witness".to_string(),
        ));
    };
    let z_len = first.z_pre.len();
    let coeff_len = first.centered_coeffs.len();
    let mut z_pre = first.z_pre.clone();
    let mut centered_coeffs = first.centered_coeffs.clone();

    for witness in rest {
        if witness.z_pre.len() != z_len || witness.centered_coeffs.len() != coeff_len {
            return Err(AkitaError::InvalidInput(
                "batched decompose_fold witness length mismatch".to_string(),
            ));
        }
        for (dst, src) in z_pre.iter_mut().zip(witness.z_pre.iter()) {
            *dst += *src;
        }
        for (dst, src) in centered_coeffs
            .iter_mut()
            .zip(witness.centered_coeffs.iter())
        {
            for k in 0..D {
                dst[k] = dst[k].checked_add(src[k]).ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "batched decompose_fold centered coefficient overflow".to_string(),
                    )
                })?;
            }
        }
    }

    let centered_inf_norm = centered_coeffs
        .iter()
        .flat_map(|coeffs| coeffs.iter())
        .map(|coeff| coeff.unsigned_abs())
        .max()
        .unwrap_or(0);

    Ok(DecomposeFoldWitness {
        z_pre,
        centered_coeffs,
        centered_inf_norm,
    })
}

/// Stage-1 quadratic equation state for the Akita protocol.
///
/// Encapsulates the relation $M(x) \cdot z = y(x) + (X^D + 1) \cdot r(x)$
/// along with intermediate prover witness data (`w_hat`, `z_pre`, `hint`).
///
/// M and z are never materialized on the hot path — split-eq factoring computes
/// their products on-the-fly via `compute_r_split_eq`, while debug/test code
/// can reconstruct reference `M_a` rows when needed.
pub struct QuadraticEquation<F: FieldCore, const D: usize> {
    /// Stage-1 proof vector `v = D · ŵ`.
    pub v: Vec<CyclotomicRing<F, D>>,
    /// Stage-1 folding challenges (sparse representation).
    pub challenges: Vec<SparseChallenge>,
    /// Vector `y`.
    y: Vec<CyclotomicRing<F, D>>,
    /// Opening points `(a_j, b_j)` used by the root relation.
    opening_points: Vec<RingOpeningPoint<F>>,
    /// Map from flattened claim index to opening-point index.
    claim_to_point: Vec<usize>,
    /// Map from flattened claim index to committed-group index.
    claim_to_group: Vec<usize>,
    /// Polynomial index within its committed group for each flattened claim.
    claim_poly_indices: Vec<usize>,
    /// Pre-decomposition folded witness `z_pre = Σ c_i · s_i` (prover only).
    /// Replaces both `z_hat` and `z`: `z_hat = J^{-1}(z_pre)`.
    z_pre: Option<DecomposeFoldWitness<F, D>>,
    /// Decomposed `ŵ_i = G_1^{-1}(w_i)` in flat column-major order plus block
    /// boundaries (prover only).
    w_hat: Option<FlatDigitBlocks<D>>,
    /// Fresh D-side blinding digits for `v = D · ŵ` (prover only).
    #[cfg(feature = "zk")]
    d_blinding_digits: Option<FlatDigitBlocks<D>>,
    /// Pre-decomposition folded ring elements (prover only, avoids recompose roundtrip).
    w_folded: Option<Vec<CyclotomicRing<F, D>>>,
    /// Commitment hint (prover only).
    hint: Option<AkitaCommitmentHint<F, D>>,
    /// Number of committed polynomials in each commitment group.
    group_poly_counts: Vec<usize>,
    /// Per-claim γ coefficients for batched linear-relation evaluation.
    gamma: Vec<F>,
    /// Number of batched evaluation rows in the matrix equation.  Equals
    /// the number of distinct opening points (one public y-row per point).
    num_public_eval_rows: usize,
}

fn compute_v_rows<F: FieldCore + CanonicalField, const D: usize>(
    ntt_d: &NttSlotCache<D>,
    row_len: usize,
    stride: usize,
    w_hat: &FlatDigitBlocks<D>,
    #[cfg(feature = "zk")] d_blinding_digits: &FlatDigitBlocks<D>,
) -> Vec<CyclotomicRing<F, D>> {
    #[cfg(feature = "zk")]
    {
        let mut d_input_digits = w_hat.flat_digits().to_vec();
        d_input_digits.extend_from_slice(d_blinding_digits.flat_digits());
        mat_vec_mul_ntt_single_i8(ntt_d, row_len, stride, &d_input_digits)
    }
    #[cfg(not(feature = "zk"))]
    {
        mat_vec_mul_ntt_single_i8(ntt_d, row_len, stride, w_hat.flat_digits())
    }
}

impl<F, const D: usize> QuadraticEquation<F, D>
where
    F: FieldCore + CanonicalField,
{
    /// Unified prover constructor covering all root-level scenarios
    /// (single-claim, same-point batching, multi-point batching, or any mix).
    ///
    /// `opening_points` holds the distinct ring-level opening points used by
    /// the batch; `claim_to_point` maps each flattened claim to its
    /// opening-point index.  The batched CWSS protocol γ-combines all
    /// polynomials opened at the same point into one ring element, so
    /// `y_rings` carries one entry per opening point
    /// (i.e. `y_rings.len() == opening_points.len()`).  For the trivial
    /// single-claim case use `opening_points = vec![pt]`,
    /// `claim_to_point = vec![0]`, `polys = &[poly]`,
    /// `group_poly_counts = &[1]`, `gamma = vec![F::one()]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the batched hints, folded witnesses, or decomposed
    /// aggregate witness are malformed.
    ///
    /// # Panics
    ///
    /// Panics if the batched `w_hat` decomposition or flattened batched hints
    /// produced by the prover do not preserve the expected block sizes.  These
    /// invariants hold by construction for well-formed inputs accepted by the
    /// error checks above and are therefore treated as internal programming
    /// errors rather than recoverable failures.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "QuadraticEquation::new_prover")]
    #[inline(never)]
    pub fn new_prover<T: Transcript<F>, P: AkitaPolyOps<F, D>>(
        ntt_d: &NttSlotCache<D>,
        opening_points: Vec<RingOpeningPoint<F>>,
        claim_to_point: Vec<usize>,
        polys: &[&P],
        pre_folded_by_poly: Vec<Vec<CyclotomicRing<F, D>>>,
        incidence_summary: &ClaimIncidenceSummary,
        lp: LevelParams,
        hints: Vec<AkitaCommitmentHint<F, D>>,
        transcript: &mut T,
        commitments: &[RingCommitment<F, D>],
        y_rings: &[CyclotomicRing<F, D>],
        gamma: Vec<F>,
        stride: usize,
    ) -> Result<Self, AkitaError> {
        {
            let x: u8 = 0;
            tracing::trace!(
                stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
                "QuadraticEquation::new_prover"
            );
        }
        if opening_points.is_empty() {
            return Err(AkitaError::InvalidInput(
                "batched prover requires at least one opening point".to_string(),
            ));
        }
        for opening_point in &opening_points {
            if opening_point.a.len() != lp.block_len || opening_point.b.len() != lp.num_blocks {
                return Err(AkitaError::InvalidInput(
                    "batched prover opening-point layout mismatch".to_string(),
                ));
            }
        }
        let num_claims = incidence_summary.num_claims;
        let group_poly_counts = &incidence_summary.group_poly_counts;
        if polys.is_empty() || group_poly_counts.is_empty() {
            return Err(AkitaError::InvalidInput(
                "batched prover requires at least one polynomial".to_string(),
            ));
        }
        if group_poly_counts.contains(&0) {
            return Err(AkitaError::InvalidInput(
                "batched prover requires nonempty commitment groups".to_string(),
            ));
        }
        // The batched protocol emits one public y-row per distinct opening point,
        // so `y_rings.len()` must equal `opening_points.len()`.
        if polys.len() != pre_folded_by_poly.len()
            || polys.len() != num_claims
            || y_rings.len() != opening_points.len()
            || claim_to_point.len() != num_claims
            || incidence_summary.claim_to_group.len() != num_claims
            || incidence_summary.claim_poly_indices.len() != num_claims
            || hints.len() != incidence_summary.num_groups
            || commitments.len() != incidence_summary.num_groups
        {
            return Err(AkitaError::InvalidInput(
                "batched prover input lengths do not match".to_string(),
            ));
        }
        if claim_to_point
            .iter()
            .any(|&point_idx| point_idx >= opening_points.len())
        {
            return Err(AkitaError::InvalidInput(
                "batched prover claim-to-point index out of range".to_string(),
            ));
        }
        for claim_idx in 0..num_claims {
            let group_idx = incidence_summary.claim_to_group[claim_idx];
            if group_idx >= incidence_summary.num_groups
                || incidence_summary.claim_poly_indices[claim_idx] >= group_poly_counts[group_idx]
            {
                return Err(AkitaError::InvalidInput(
                    "batched prover claim incidence index out of range".to_string(),
                ));
            }
        }
        for commitment in commitments {
            if commitment.u.len() != lp.b_key.row_len() {
                return Err(AkitaError::InvalidInput(
                    "batched prover received a commitment with the wrong length".to_string(),
                ));
            }
        }
        if gamma.len() != num_claims {
            return Err(AkitaError::InvalidInput(
                "batched prover gamma length does not match claim count".to_string(),
            ));
        }
        let num_public_eval_rows = opening_points.len();

        let w_hat = {
            let _span = tracing::info_span!("decompose_batched_w_hat").entered();
            let depth_open = lp.num_digits_open;
            let log_basis = lp.log_basis;
            let q = (-F::one()).to_canonical_u128() + 1;
            let decompose_params = BalancedDecomposePow2I8Params::new(depth_open, log_basis, q);
            let total_rows: usize = pre_folded_by_poly.iter().map(Vec::len).sum();
            let block_sizes = vec![depth_open; total_rows];
            let mut w_hat = FlatDigitBlocks::zeroed(block_sizes)
                .expect("batched w_hat decomposition preserves block sizes");
            let mut offset = 0usize;
            for folded_rows in &pre_folded_by_poly {
                for w_i in folded_rows {
                    w_i.balanced_decompose_pow2_i8_into_with_params(
                        &mut w_hat.flat_digits_mut()[offset..offset + depth_open],
                        &decompose_params,
                    );
                    offset += depth_open;
                }
            }
            w_hat
        };
        let flattened_hint = {
            let mut decomposed_inner_rows = Vec::new();
            let mut t_rows_by_poly = Vec::new();
            #[cfg(feature = "zk")]
            let mut b_blinding_digits = Vec::new();
            for (mut hint, &group_size) in hints.into_iter().zip(group_poly_counts.iter()) {
                if hint.decomposed_inner_rows.len() != group_size {
                    return Err(AkitaError::InvalidInput(
                        "batched prover hint group sizes do not match polynomial groups"
                            .to_string(),
                    ));
                }
                hint.ensure_recomposed_inner_rows(lp.num_digits_open, lp.log_basis)?;
                #[cfg(feature = "zk")]
                let (digits_by_poly, rows_by_poly, mut blinding_by_group) = hint.into_parts();
                #[cfg(not(feature = "zk"))]
                let (digits_by_poly, rows_by_poly) = hint.into_parts();
                #[cfg(feature = "zk")]
                if blinding_by_group.len() != 1 {
                    return Err(AkitaError::InvalidInput(
                        "batched prover hint must carry exactly one blinding group per commitment"
                            .to_string(),
                    ));
                }
                decomposed_inner_rows.extend(digits_by_poly);
                let rows_by_poly = rows_by_poly.ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "missing recomposed inner rows in batched prover hint".to_string(),
                    )
                })?;
                t_rows_by_poly.extend(rows_by_poly);
                #[cfg(feature = "zk")]
                b_blinding_digits.append(&mut blinding_by_group);
            }
            #[cfg(feature = "zk")]
            {
                AkitaCommitmentHint::with_recomposed_inner_rows(
                    decomposed_inner_rows,
                    t_rows_by_poly,
                    b_blinding_digits,
                )
            }
            #[cfg(not(feature = "zk"))]
            {
                AkitaCommitmentHint::with_recomposed_inner_rows(
                    decomposed_inner_rows,
                    t_rows_by_poly,
                )
            }
        };

        #[cfg(feature = "zk")]
        let d_blinding_digits = sample_blinding_digits::<F, D>(lp.d_key.row_len(), lp.log_basis)?;

        let v = {
            let _span = tracing::info_span!(
                "compute_batched_v",
                w_hat_planes = w_hat.flat_digits().len()
            )
            .entered();
            compute_v_rows(
                ntt_d,
                lp.d_key.row_len(),
                stride,
                &w_hat,
                #[cfg(feature = "zk")]
                &d_blinding_digits,
            )
        };

        transcript.append_serde(ABSORB_PROVER_V, &RingSliceSerializer(&v));

        let total_blocks = lp.num_blocks.checked_mul(num_claims).ok_or_else(|| {
            AkitaError::InvalidSetup("batched challenge count overflow".to_string())
        })?;
        let challenges = sample_sparse_challenges::<F, T, D>(
            transcript,
            CHALLENGE_STAGE1_FOLD,
            total_blocks,
            &lp.stage1_config,
        )?;

        let z_pre = {
            let num_points = opening_points.len();
            let _span =
                tracing::info_span!("compute_batched_z_pre", num_points = num_points).entered();
            let mut polys_by_point: Vec<Vec<&P>> = vec![Vec::new(); num_points];
            let mut challenges_by_point: Vec<Vec<SparseChallenge>> = vec![Vec::new(); num_points];
            for (claim_idx, poly) in polys.iter().enumerate() {
                let point_idx = claim_to_point[claim_idx];
                polys_by_point[point_idx].push(*poly);
                let challenge_offset = claim_idx.checked_mul(lp.num_blocks).ok_or_else(|| {
                    AkitaError::InvalidSetup("batched challenge offset overflow".to_string())
                })?;
                let next_offset = challenge_offset.checked_add(lp.num_blocks).ok_or_else(|| {
                    AkitaError::InvalidSetup("batched challenge offset overflow".to_string())
                })?;
                challenges_by_point[point_idx]
                    .extend_from_slice(&challenges[challenge_offset..next_offset]);
            }

            let mut z_pre = Vec::new();
            let mut centered_coeffs = Vec::new();
            let mut centered_inf_norm = 0u32;
            for (point_idx, point_polys) in polys_by_point.iter().enumerate() {
                let point_challenges = &challenges_by_point[point_idx];
                let point_claim_count = point_polys.len();
                let witness = if let Some(z_point) = P::decompose_fold_batched(
                    point_polys,
                    point_challenges,
                    lp.block_len,
                    lp.num_digits_commit,
                    lp.log_basis,
                ) {
                    z_point
                } else {
                    let witnesses: Vec<DecomposeFoldWitness<F, D>> = point_polys
                        .iter()
                        .zip(point_challenges.chunks(lp.num_blocks))
                        .map(|(poly, poly_challenges)| {
                            poly.decompose_fold(
                                poly_challenges,
                                lp.block_len,
                                lp.num_digits_commit,
                                lp.log_basis,
                            )
                        })
                        .collect();
                    aggregate_decompose_fold_witnesses(witnesses)?
                };
                let witness = validate_decompose_fold(witness, &lp, point_claim_count)?;
                centered_inf_norm = centered_inf_norm.max(witness.centered_inf_norm);
                z_pre.extend(witness.z_pre);
                centered_coeffs.extend(witness.centered_coeffs);
            }

            DecomposeFoldWitness {
                z_pre,
                centered_coeffs,
                centered_inf_norm,
            }
        };

        let commitment_rows: Vec<CyclotomicRing<F, D>> = commitments
            .iter()
            .flat_map(|commitment| commitment.u.iter().copied())
            .collect();
        let y = generate_y::<F, D>(
            &v,
            &commitment_rows,
            y_rings,
            lp.d_key.row_len(),
            lp.b_key.row_len(),
            lp.a_key.row_len(),
        )?;
        let w_folded = pre_folded_by_poly.into_iter().flatten().collect();

        Ok(Self {
            v,
            challenges,
            y,
            opening_points,
            claim_to_point,
            claim_to_group: incidence_summary.claim_to_group.clone(),
            claim_poly_indices: incidence_summary.claim_poly_indices.clone(),
            z_pre: Some(z_pre),
            w_hat: Some(w_hat),
            #[cfg(feature = "zk")]
            d_blinding_digits: Some(d_blinding_digits),
            w_folded: Some(w_folded),
            hint: Some(flattened_hint),
            group_poly_counts: group_poly_counts.to_vec(),
            gamma,
            num_public_eval_rows,
        })
    }

    /// Recursive prover constructor: single-claim path driven by the dedicated
    /// recursive witness view instead of the root polynomial trait.
    ///
    /// # Errors
    ///
    /// Returns an error if the norm check, challenge sampling, or matrix
    /// generation fails.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "QuadraticEquation::new_recursive_prover")]
    #[inline(never)]
    pub fn new_recursive_prover<T: Transcript<F>>(
        ntt_d: &NttSlotCache<D>,
        ring_opening_point: RingOpeningPoint<F>,
        witness: &RecursiveWitnessView<'_, F, D>,
        pre_folded: Vec<CyclotomicRing<F, D>>,
        lp: LevelParams,
        mut hint: AkitaCommitmentHint<F, D>,
        transcript: &mut T,
        commitment: &[CyclotomicRing<F, D>],
        y_ring: &CyclotomicRing<F, D>,
        stride: usize,
    ) -> Result<Self, AkitaError> {
        {
            let x: u8 = 0;
            tracing::trace!(
                stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
                "QuadraticEquation::new_recursive_prover"
            );
        }
        let w_hat = {
            let _span = tracing::info_span!("decompose_w_hat").entered();
            let depth_open = lp.num_digits_open;
            let log_basis = lp.log_basis;
            let q = (-F::one()).to_canonical_u128() + 1;

            let decompose_params = BalancedDecomposePow2I8Params::new(depth_open, log_basis, q);
            let mut w_hat = FlatDigitBlocks::zeroed(vec![depth_open; pre_folded.len()])?;
            for (idx, w_i) in pre_folded.iter().enumerate() {
                let start = idx * depth_open;
                w_i.balanced_decompose_pow2_i8_into_with_params(
                    &mut w_hat.flat_digits_mut()[start..start + depth_open],
                    &decompose_params,
                );
            }
            w_hat
        };
        hint.ensure_recomposed_inner_rows(lp.num_digits_open, lp.log_basis)?;

        #[cfg(feature = "zk")]
        let d_blinding_digits = sample_blinding_digits::<F, D>(lp.d_key.row_len(), lp.log_basis)?;

        let v = {
            let _span = tracing::info_span!("compute_v", w_hat_planes = w_hat.flat_digits().len())
                .entered();
            compute_v_rows(
                ntt_d,
                lp.d_key.row_len(),
                stride,
                &w_hat,
                #[cfg(feature = "zk")]
                &d_blinding_digits,
            )
        };

        transcript.append_serde(ABSORB_PROVER_V, &RingSliceSerializer(&v));

        let challenges = sample_sparse_challenges::<F, T, D>(
            transcript,
            CHALLENGE_STAGE1_FOLD,
            lp.num_blocks,
            &lp.stage1_config,
        )?;

        let z_pre = {
            let _span = tracing::info_span!("compute_z_pre").entered();
            let z = witness.decompose_fold(
                &challenges,
                lp.block_len,
                lp.num_blocks,
                lp.num_digits_commit,
                lp.log_basis,
            );
            validate_decompose_fold(z, &lp, 1)?
        };

        let y = generate_y::<F, D>(
            &v,
            commitment,
            std::slice::from_ref(y_ring),
            lp.d_key.row_len(),
            lp.b_key.row_len(),
            lp.a_key.row_len(),
        )?;

        Ok(Self {
            v,
            challenges,
            y,
            opening_points: vec![ring_opening_point],
            claim_to_point: vec![0],
            claim_to_group: vec![0],
            claim_poly_indices: vec![0],
            z_pre: Some(z_pre),
            w_hat: Some(w_hat),
            #[cfg(feature = "zk")]
            d_blinding_digits: Some(d_blinding_digits),
            w_folded: Some(pre_folded),
            hint: Some(hint),
            group_poly_counts: vec![1],
            gamma: vec![F::one()],
            num_public_eval_rows: 1,
        })
    }

    /// Get the vector y.
    pub fn y(&self) -> &[CyclotomicRing<F, D>] {
        &self.y
    }

    /// Get the vector v.
    pub fn v(&self) -> &[CyclotomicRing<F, D>] {
        &self.v
    }

    /// Get the first opening point `(a, b)` used by this relation.
    ///
    /// # Panics
    ///
    /// Panics if the quadratic equation was constructed without any opening
    /// points.
    pub fn opening_point(&self) -> &RingOpeningPoint<F> {
        self.opening_points
            .first()
            .expect("quadratic equation must store at least one opening point")
    }

    /// Get all opening points `(a_j, b_j)` used by this relation.
    pub fn opening_points(&self) -> &[RingOpeningPoint<F>] {
        &self.opening_points
    }

    /// Map each flattened claim index to its opening-point index.
    pub fn claim_to_point(&self) -> &[usize] {
        &self.claim_to_point
    }

    /// Number of committed polynomials carried by each commitment group.
    pub fn group_poly_counts(&self) -> &[usize] {
        &self.group_poly_counts
    }

    /// Map each flattened claim index to its committed-group index.
    pub fn claim_to_group(&self) -> &[usize] {
        &self.claim_to_group
    }

    /// Polynomial index within the committed group for each flattened claim.
    pub fn claim_poly_indices(&self) -> &[usize] {
        &self.claim_poly_indices
    }

    /// Per-claim batching coefficients used by the relation rows.
    pub fn gamma(&self) -> &[F] {
        &self.gamma
    }

    /// Number of batched public y-rows in the matrix equation.  Equals
    /// the number of distinct opening points (one row per point).
    pub fn num_public_eval_rows(&self) -> usize {
        self.num_public_eval_rows
    }

    /// Get the pre-decomposition folded witness `z_pre` (prover only).
    pub fn z_pre(&self) -> Option<&[CyclotomicRing<F, D>]> {
        self.z_pre.as_ref().map(|witness| witness.z_pre.as_slice())
    }

    /// Get centered coefficients for each `z_pre` row (prover only).
    pub fn z_pre_centered(&self) -> Option<&[[i32; D]]> {
        self.z_pre
            .as_ref()
            .map(|witness| witness.centered_coeffs.as_slice())
    }

    /// Get `||z_pre||_inf` from the centered witness representation.
    pub fn z_pre_centered_inf_norm(&self) -> Option<u32> {
        self.z_pre.as_ref().map(|witness| witness.centered_inf_norm)
    }

    /// Take ownership of the `z_pre` witness, leaving `None` in its place.
    pub fn take_z_pre(&mut self) -> Option<DecomposeFoldWitness<F, D>> {
        self.z_pre.take()
    }

    /// Get the decomposed witness `ŵ` as i8 digit planes (prover only).
    pub fn w_hat(&self) -> Option<&FlatDigitBlocks<D>> {
        self.w_hat.as_ref()
    }

    /// Get the flat `w_hat` digit planes (prover only).
    pub fn w_hat_flat(&self) -> Option<&[[i8; D]]> {
        self.w_hat.as_ref().map(|digits| digits.flat_digits())
    }

    /// Take ownership of `w_hat`, leaving `None` in its place.
    pub fn take_w_hat(&mut self) -> Option<FlatDigitBlocks<D>> {
        self.w_hat.take()
    }

    /// Get D-side blinding digits for `v` (prover/debug only).
    #[cfg(feature = "zk")]
    pub fn d_blinding_digits(&self) -> Option<&FlatDigitBlocks<D>> {
        self.d_blinding_digits.as_ref()
    }

    /// Take ownership of D-side blinding digits, leaving `None` in their place.
    #[cfg(feature = "zk")]
    pub fn take_d_blinding_digits(&mut self) -> Option<FlatDigitBlocks<D>> {
        self.d_blinding_digits.take()
    }

    /// Get the pre-decomposition folded ring elements (prover only).
    pub fn w_folded(&self) -> Option<&[CyclotomicRing<F, D>]> {
        self.w_folded.as_deref()
    }

    /// Take ownership of the pre-decomposition folded witness, leaving `None`
    /// in its place.
    pub fn take_w_folded(&mut self) -> Option<Vec<CyclotomicRing<F, D>>> {
        self.w_folded.take()
    }

    /// Get the commitment hint (prover only).
    pub fn hint(&self) -> Option<&AkitaCommitmentHint<F, D>> {
        self.hint.as_ref()
    }

    /// Take ownership of the hint, leaving `None` in its place.
    pub fn take_hint(&mut self) -> Option<AkitaCommitmentHint<F, D>> {
        self.hint.take()
    }
}

/// Add only the high-half quotient contribution of `challenge * ring`.
///
/// Skips the first `D - pos` coefficients per challenge term that cannot
/// contribute (degree < D), cutting iteration count roughly in half.
#[inline(always)]
fn add_sparse_ring_product_high_half<F: FieldCore + CanonicalField, const D: usize>(
    quotient: &mut [F],
    challenge: &SparseChallenge,
    ring: &CyclotomicRing<F, D>,
) {
    let rc = ring.coefficients();
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let c = F::from_i64(coeff as i64);
        let p = pos as usize;
        for s in (D - p)..D {
            quotient[p + s - D] += c * rc[s];
        }
    }
}

/// Parallel high-half accumulation over blocks.
///
/// Replaces `for (i, ring) in rings { add_sparse_ring_product_high_half(...) }`
/// with a fold-reduce giving block-level parallelism.
fn parallel_high_half_accumulate<F: FieldCore + CanonicalField, const D: usize>(
    challenges: &[SparseChallenge],
    rings: &[CyclotomicRing<F, D>],
) -> Vec<F> {
    cfg_fold_reduce!(
        0..rings.len(),
        || vec![F::zero(); D],
        |mut acc: Vec<F>, i: usize| {
            add_sparse_ring_product_high_half::<F, D>(&mut acc, &challenges[i], &rings[i]);
            acc
        },
        |mut a: Vec<F>, b: Vec<F>| {
            for (ai, bi) in a.iter_mut().zip(b.iter()) {
                *ai += *bi;
            }
            a
        }
    )
}

fn quotient_from_cyclic_and_reduced<F: FieldCore + HalvingField, const D: usize>(
    cyclic: &CyclotomicRing<F, D>,
    reduced: &CyclotomicRing<F, D>,
) -> CyclotomicRing<F, D> {
    let cyc_c = cyclic.coefficients();
    let red_c = reduced.coefficients();
    let quotient = std::array::from_fn(|k| (cyc_c[k] - red_c[k]).half());
    CyclotomicRing::from_coefficients(quotient)
}

#[cfg(feature = "zk")]
fn add_blinding_cyclic_rows<F: FieldCore + CanonicalField, const D: usize>(
    ntt_shared: &NttSlotCache<D>,
    n_b: usize,
    stride: usize,
    message_planes: usize,
    blinding: &FlatDigitBlocks<D>,
    rows: &mut [CyclotomicRing<F, D>],
) -> Result<(), AkitaError> {
    if blinding.is_empty() {
        return Ok(());
    }
    if rows.len() != n_b {
        return Err(AkitaError::InvalidProof);
    }
    let total_planes = message_planes
        .checked_add(blinding.flat_digits().len())
        .ok_or(AkitaError::InvalidProof)?;
    if total_planes > stride {
        return Err(AkitaError::InvalidProof);
    }
    let mut padded = vec![[0i8; D]; message_planes];
    padded.extend_from_slice(blinding.flat_digits());
    let b_blinding_rows = mat_vec_mul_ntt_single_i8_cyclic(ntt_shared, n_b, stride, &padded);
    for (row, b_blinding_row) in rows.iter_mut().zip(b_blinding_rows) {
        *row += b_blinding_row;
    }
    Ok(())
}

fn repeated_b_commitment_rows<F: FieldCore + CanonicalField, const D: usize>(
    ntt_shared: &NttSlotCache<D>,
    n_b: usize,
    stride: usize,
    t_hat: &FlatDigitBlocks<D>,
    #[cfg(feature = "zk")] b_blinding_digits: &[FlatDigitBlocks<D>],
    group_poly_counts: &[usize],
    blocks_per_claim: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    if group_poly_counts.is_empty() || blocks_per_claim == 0 {
        return Err(AkitaError::InvalidProof);
    }
    let num_group_polys = group_poly_counts
        .iter()
        .try_fold(0usize, |acc, &group_poly_count| {
            if group_poly_count == 0 {
                return Err(AkitaError::InvalidProof);
            }
            acc.checked_add(group_poly_count)
                .ok_or(AkitaError::InvalidProof)
        })?;
    if t_hat.block_count() != num_group_polys * blocks_per_claim {
        return Err(AkitaError::InvalidProof);
    }
    #[cfg(not(feature = "zk"))]
    let b_blinding_digits = vec![FlatDigitBlocks::<D>::empty(); group_poly_counts.len()];
    if b_blinding_digits.len() != group_poly_counts.len() {
        return Err(AkitaError::InvalidProof);
    }
    let mut rows = Vec::with_capacity(group_poly_counts.len() * n_b);
    let mut block_offset = 0usize;
    let mut plane_offset = 0usize;
    for (&group_poly_count, blinding) in group_poly_counts.iter().zip(b_blinding_digits.iter()) {
        #[cfg(not(feature = "zk"))]
        let _ = blinding;
        let group_block_count = group_poly_count
            .checked_mul(blocks_per_claim)
            .ok_or(AkitaError::InvalidProof)?;
        let next_block_offset = block_offset
            .checked_add(group_block_count)
            .ok_or(AkitaError::InvalidProof)?;
        let group_block_sizes = t_hat
            .block_sizes()
            .get(block_offset..next_block_offset)
            .ok_or(AkitaError::InvalidProof)?;
        let group_planes: usize = group_block_sizes.iter().sum();
        let next_plane_offset = plane_offset
            .checked_add(group_planes)
            .ok_or(AkitaError::InvalidProof)?;
        let group_digits = t_hat
            .flat_digits()
            .get(plane_offset..next_plane_offset)
            .ok_or(AkitaError::InvalidProof)?;
        #[cfg(feature = "zk")]
        let row_start = rows.len();
        rows.extend(mat_vec_mul_ntt_single_i8_cyclic(
            ntt_shared,
            n_b,
            stride,
            group_digits,
        ));
        #[cfg(feature = "zk")]
        {
            add_blinding_cyclic_rows(
                ntt_shared,
                n_b,
                stride,
                group_planes,
                blinding,
                &mut rows[row_start..row_start + n_b],
            )?;
        }
        block_offset = next_block_offset;
        plane_offset = next_plane_offset;
    }
    if block_offset != t_hat.block_count() || plane_offset != t_hat.flat_digits().len() {
        return Err(AkitaError::InvalidProof);
    }
    Ok(rows)
}

/// Split-eq replacement for `generate_m` + `compute_r_via_poly_division`.
///
/// Computes `r` such that `M·z = y + (X^D+1)·r` without materializing M or z.
/// Uses split-eq factoring: `kron(left, gadget) · decomposed = left · pre_decomp`.
///
/// # Errors
///
/// Returns an error if the claim grouping, row layout, or split-eq witness
/// dimensions are inconsistent.
#[allow(clippy::too_many_arguments, clippy::needless_borrow)]
#[tracing::instrument(skip_all, name = "compute_r_split_eq")]
pub fn compute_r_split_eq<F, const D: usize>(
    lp: &LevelParams,
    _setup: &AkitaExpandedSetup<F>,
    challenges: &[SparseChallenge],
    w_hat_flat: &[[i8; D]],
    #[cfg(feature = "zk")] d_blinding_digits: &FlatDigitBlocks<D>,
    t_hat: &FlatDigitBlocks<D>,
    #[cfg(feature = "zk")] b_blinding_digits: &[FlatDigitBlocks<D>],
    recomposed_inner_rows: &[Vec<CyclotomicRing<F, D>>],
    w_folded: &[CyclotomicRing<F, D>],
    z_pre_centered: &[[i32; D]],
    z_pre_centered_inf_norm: u32,
    y: &[CyclotomicRing<F, D>],
    group_poly_counts: &[usize],
    num_public_outputs: usize,
    blocks_per_claim: usize,
    inner_width: usize,
    stride: usize,
    ntt_shared: &NttSlotCache<D>,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField,
{
    if group_poly_counts.is_empty() || group_poly_counts.contains(&0) {
        return Err(AkitaError::InvalidProof);
    }
    if num_public_outputs == 0 {
        return Err(AkitaError::InvalidProof);
    }
    let _num_group_polys =
        group_poly_counts
            .iter()
            .try_fold(0usize, |acc, &group_poly_count| {
                acc.checked_add(group_poly_count)
                    .ok_or(AkitaError::InvalidProof)
            })?;
    let num_commitment_groups = group_poly_counts.len();
    let n_b = lp.b_key.row_len();
    let n_d = lp.d_key.row_len();
    let n_a = lp.a_key.row_len();
    let commitment_row_count = n_b
        .checked_mul(num_commitment_groups)
        .ok_or(AkitaError::InvalidProof)?;
    let num_rows = lp.m_row_count(num_commitment_groups, num_public_outputs);
    if y.len() != num_rows {
        return Err(AkitaError::InvalidProof);
    }
    // Row layout: consistency (1) | public (num_public_outputs) | D (n_d) |
    //             B (commitment_row_count) | A (n_a)
    let d_start = 1 + num_public_outputs;
    let b_start = d_start + n_d;
    let a_start = b_start + commitment_row_count;

    if inner_width == 0 || !z_pre_centered.len().is_multiple_of(inner_width) {
        return Err(AkitaError::InvalidProof);
    }

    let mut z_segments = z_pre_centered.chunks(inner_width);
    let first_z_segment = z_segments.next().ok_or(AkitaError::InvalidProof)?;

    let (d_cyclic, b_cyclic, mut a_quotients) = fused_split_eq_quotients::<F, D>(
        ntt_shared,
        n_d,
        n_b,
        n_a,
        stride,
        w_hat_flat,
        t_hat.flat_digits(),
        first_z_segment,
        z_pre_centered_inf_norm,
    );
    #[cfg(feature = "zk")]
    let mut d_cyclic = d_cyclic;
    #[cfg(feature = "zk")]
    add_blinding_cyclic_rows(
        ntt_shared,
        n_d,
        stride,
        w_hat_flat.len(),
        d_blinding_digits,
        &mut d_cyclic,
    )?;
    for z_segment in z_segments {
        let (_, _, segment_a_quotients) = fused_split_eq_quotients::<F, D>(
            ntt_shared,
            0,
            0,
            n_a,
            stride,
            &[],
            &[],
            z_segment,
            z_pre_centered_inf_norm,
        );
        for (dst, src) in a_quotients.iter_mut().zip(segment_a_quotients.into_iter()) {
            *dst += src;
        }
    }
    let commitment_cyclic_rows = if commitment_row_count == n_b && num_commitment_groups == 1 {
        #[cfg(feature = "zk")]
        let mut rows = b_cyclic;
        #[cfg(not(feature = "zk"))]
        let rows = b_cyclic;
        #[cfg(feature = "zk")]
        {
            let blinding = b_blinding_digits.first().ok_or(AkitaError::InvalidProof)?;
            add_blinding_cyclic_rows(
                ntt_shared,
                n_b,
                stride,
                t_hat.flat_digits().len(),
                blinding,
                &mut rows,
            )?;
        }
        rows
    } else {
        repeated_b_commitment_rows(
            ntt_shared,
            n_b,
            stride,
            t_hat,
            #[cfg(feature = "zk")]
            b_blinding_digits,
            group_poly_counts,
            blocks_per_claim,
        )?
    };
    if commitment_cyclic_rows.len() != commitment_row_count {
        return Err(AkitaError::InvalidProof);
    }

    let mut result = Vec::with_capacity(num_rows);
    let mut other_time = 0.0f64;

    for row_idx in 0..num_rows {
        if row_idx == 0 {
            let t_row = Instant::now();
            let _span = tracing::info_span!("challenge_fold_row").entered();
            let quotient = parallel_high_half_accumulate::<F, D>(challenges, w_folded);
            result.push(CyclotomicRing::from_slice(&quotient));
            other_time += t_row.elapsed().as_secs_f64();
        } else if row_idx < d_start {
            let _span = tracing::info_span!("bTw_row").entered();
            result.push(CyclotomicRing::<F, D>::zero());
        } else if row_idx < b_start {
            result.push(quotient_from_cyclic_and_reduced(
                &d_cyclic[row_idx - d_start],
                &y[row_idx],
            ));
        } else if row_idx < a_start {
            result.push(quotient_from_cyclic_and_reduced(
                &commitment_cyclic_rows[row_idx - b_start],
                &y[row_idx],
            ));
        } else {
            let t_row = Instant::now();
            let _span = tracing::info_span!("A_row").entered();
            let a_idx = row_idx - a_start;

            let mut quotient = cfg_fold_reduce!(
                0..recomposed_inner_rows.len(),
                || vec![F::zero(); D],
                |mut acc: Vec<F>, i: usize| {
                    if let Some(inner_row_i) = recomposed_inner_rows[i].get(a_idx) {
                        add_sparse_ring_product_high_half::<F, D>(
                            &mut acc,
                            &challenges[i],
                            inner_row_i,
                        );
                    }
                    acc
                },
                |mut a: Vec<F>, b: Vec<F>| {
                    for (ai, bi) in a.iter_mut().zip(b.iter()) {
                        *ai += *bi;
                    }
                    a
                }
            );

            let a_q = a_quotients[a_idx].coefficients();
            for k in 0..D {
                quotient[k] -= a_q[k];
            }
            result.push(CyclotomicRing::from_slice(&quotient));
            other_time += t_row.elapsed().as_secs_f64();
        }
    }

    tracing::debug!(other_s = other_time, "compute_r breakdown");

    Ok(result)
}

/// Build the RHS vector `y` matching the M row layout:
/// consistency (zero) | public outputs | D (`v`) | B (`commitment_rows`) | A (zeros).
///
/// # Errors
///
/// Returns an error if the supplied row slices do not match the expected row
/// counts for the level layout.
pub fn generate_y<F, const D: usize>(
    v: &[CyclotomicRing<F, D>],
    commitment_rows: &[CyclotomicRing<F, D>],
    public_outputs: &[CyclotomicRing<F, D>],
    n_d: usize,
    n_b: usize,
    n_a: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore,
{
    if v.len() != n_d {
        return Err(AkitaError::InvalidSize {
            expected: n_d,
            actual: v.len(),
        });
    }
    if commitment_rows.is_empty() || !commitment_rows.len().is_multiple_of(n_b) {
        return Err(AkitaError::InvalidSize {
            expected: n_b,
            actual: commitment_rows.len(),
        });
    }
    if public_outputs.is_empty() {
        return Err(AkitaError::InvalidInput(
            "generate_y requires at least one public output".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(1 + public_outputs.len() + n_d + commitment_rows.len() + n_a);
    out.push(CyclotomicRing::<F, D>::zero());
    out.extend_from_slice(public_outputs);
    out.extend_from_slice(v);
    out.extend_from_slice(commitment_rows);
    out.extend(repeat_n(CyclotomicRing::<F, D>::zero(), n_a));
    Ok(out)
}
