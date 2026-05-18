//! Quadratic equation builder for the Akita PCS (§4.2).
//!
//! This module encapsulates the stage-1 prover logic and the generation of
//! the quadratic equation components M, y, z, and v.

use crate::kernels::crt_ntt::NttSlotCache;
use crate::kernels::linear::{
    fused_split_eq_quotients, mat_vec_mul_ntt_single_i8, mat_vec_mul_ntt_single_i8_cyclic,
    unreduced_quotient_rows_ntt_cached,
};
use crate::protocol::flow::RecursiveHandlePoly;
use crate::{AkitaPolyOps, CenteredCoeff, CenteredInfNorm, DecomposeFoldWitness};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2I8Params;
use akita_algebra::CyclotomicRing;
use akita_challenges::{
    sample_stage1_challenges, IntegerChallenge, SparseChallenge, Stage1Challenges,
    TensorStage1Challenges,
};
use akita_field::parallel::*;
use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore, HalvingField};
use akita_transcript::labels::ABSORB_PROVER_V;
use akita_transcript::Transcript;
use akita_types::RingOpeningPoint;
use akita_types::{
    validate_stage1_accumulator_headroom, AkitaExpandedSetup, GroupLayout, LevelParams, MRowLayout,
};
use akita_types::{AkitaCommitmentHint, FlatDigitBlocks, RingCommitment, RingSliceSerializer};
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

    let centered_inf_norm: CenteredInfNorm = centered_coeffs
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

fn stage1_challenges_for_claims(
    challenges: &Stage1Challenges,
    claim_indices: &[usize],
    num_blocks: usize,
) -> Result<Stage1Challenges, AkitaError> {
    match challenges {
        Stage1Challenges::Flat(flat) => {
            let mut out = Vec::with_capacity(claim_indices.len() * num_blocks);
            for &claim_idx in claim_indices {
                let start = claim_idx.checked_mul(num_blocks).ok_or_else(|| {
                    AkitaError::InvalidSetup("flat stage-1 claim offset overflow".to_string())
                })?;
                let end = start.checked_add(num_blocks).ok_or_else(|| {
                    AkitaError::InvalidSetup("flat stage-1 claim end overflow".to_string())
                })?;
                out.extend_from_slice(flat.get(start..end).ok_or(AkitaError::InvalidSize {
                    expected: end,
                    actual: flat.len(),
                })?);
            }
            Ok(Stage1Challenges::Flat(out))
        }
        Stage1Challenges::Tensor(tensor) => {
            if tensor.left_len.checked_mul(tensor.right_len) != Some(num_blocks) {
                return Err(AkitaError::InvalidSize {
                    expected: num_blocks,
                    actual: tensor.left_len.saturating_mul(tensor.right_len),
                });
            }
            let mut left = Vec::with_capacity(claim_indices.len() * tensor.left_len);
            let mut right = Vec::with_capacity(claim_indices.len() * tensor.right_len);
            for &claim_idx in claim_indices {
                let left_start = claim_idx.checked_mul(tensor.left_len).ok_or_else(|| {
                    AkitaError::InvalidSetup("tensor-left claim offset overflow".to_string())
                })?;
                let left_end = left_start.checked_add(tensor.left_len).ok_or_else(|| {
                    AkitaError::InvalidSetup("tensor-left claim end overflow".to_string())
                })?;
                left.extend_from_slice(tensor.left.get(left_start..left_end).ok_or(
                    AkitaError::InvalidSize {
                        expected: left_end,
                        actual: tensor.left.len(),
                    },
                )?);

                let right_start = claim_idx.checked_mul(tensor.right_len).ok_or_else(|| {
                    AkitaError::InvalidSetup("tensor-right claim offset overflow".to_string())
                })?;
                let right_end = right_start.checked_add(tensor.right_len).ok_or_else(|| {
                    AkitaError::InvalidSetup("tensor-right claim end overflow".to_string())
                })?;
                right.extend_from_slice(tensor.right.get(right_start..right_end).ok_or(
                    AkitaError::InvalidSize {
                        expected: right_end,
                        actual: tensor.right.len(),
                    },
                )?);
            }
            Ok(Stage1Challenges::Tensor(TensorStage1Challenges {
                left,
                right,
                left_len: tensor.left_len,
                right_len: tensor.right_len,
                num_claims: claim_indices.len(),
            }))
        }
    }
}

fn integer_challenges_for_claim_groups(
    challenges: &[IntegerChallenge],
    claim_indices_by_point: &[Vec<usize>],
    num_blocks: usize,
) -> Result<Vec<Vec<IntegerChallenge>>, AkitaError> {
    claim_indices_by_point
        .iter()
        .map(|claim_indices| {
            let mut out = Vec::with_capacity(claim_indices.len() * num_blocks);
            for &claim_idx in claim_indices {
                let start = claim_idx.checked_mul(num_blocks).ok_or_else(|| {
                    AkitaError::InvalidSetup("integer stage-1 claim offset overflow".to_string())
                })?;
                let end = start.checked_add(num_blocks).ok_or_else(|| {
                    AkitaError::InvalidSetup("integer stage-1 claim end overflow".to_string())
                })?;
                out.extend_from_slice(challenges.get(start..end).ok_or_else(|| {
                    AkitaError::InvalidSetup(format!(
                        "integer challenge grouping out of range: start={start}, end={end}, actual={}, claim_idx={claim_idx}, num_blocks={num_blocks}",
                        challenges.len()
                    ))
                })?);
            }
            Ok(out)
        })
        .collect()
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
    /// Stage-1 folding challenges.
    pub challenges: Stage1Challenges,
    /// Expanded integer challenges, cached only for prover-side fallback paths
    /// that cannot consume the compact stage-1 representation directly.
    integer_challenges: Option<Vec<IntegerChallenge>>,
    /// Vector `y`.
    y: Vec<CyclotomicRing<F, D>>,
    /// Opening points `(a_j, b_j)` used by the root relation.
    opening_points: Vec<RingOpeningPoint<F>>,
    /// Map from flattened claim index to opening-point index.
    claim_to_point: Vec<usize>,
    /// Pre-decomposition folded witness `z_pre = Σ c_i · s_i` (prover only).
    /// Replaces both `z_hat` and `z`: `z_hat = J^{-1}(z_pre)`.
    z_pre: Option<DecomposeFoldWitness<F, D>>,
    /// Decomposed `ŵ_i = G_1^{-1}(w_i)` in flat column-major order plus block
    /// boundaries (prover only).
    w_hat: Option<FlatDigitBlocks<D>>,
    /// Pre-decomposition folded ring elements (prover only, avoids recompose roundtrip).
    w_folded: Option<Vec<CyclotomicRing<F, D>>>,
    /// Commitment hint (prover only).
    hint: Option<AkitaCommitmentHint<F, D>>,
    /// Number of flattened public claims per commitment group.
    claim_group_sizes: Vec<usize>,
    /// Per-claim γ coefficients for batched linear-relation evaluation.
    gamma: Vec<F>,
    /// Number of batched evaluation rows in the matrix equation.  Equals
    /// the number of distinct opening points (one public y-row per point).
    num_eval_rows: usize,
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
    /// `claim_group_sizes = &[1]`, `gamma = vec![F::one()]`.
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
        claim_group_sizes: &[usize],
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
        if polys.is_empty() || claim_group_sizes.is_empty() {
            return Err(AkitaError::InvalidInput(
                "batched prover requires at least one polynomial".to_string(),
            ));
        }
        if claim_group_sizes.contains(&0) {
            return Err(AkitaError::InvalidInput(
                "batched prover requires nonempty commitment groups".to_string(),
            ));
        }
        let num_claims = claim_group_sizes
            .iter()
            .try_fold(0usize, |acc, &group_size| {
                acc.checked_add(group_size).ok_or_else(|| {
                    AkitaError::InvalidInput("batched prover claim count overflow".to_string())
                })
            })?;
        // The batched protocol emits one public y-row per distinct opening point,
        // so `y_rings.len()` must equal `opening_points.len()`.
        if polys.len() != pre_folded_by_poly.len()
            || polys.len() != num_claims
            || y_rings.len() != opening_points.len()
            || claim_to_point.len() != num_claims
            || hints.len() != claim_group_sizes.len()
            || commitments.len() != claim_group_sizes.len()
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
        validate_stage1_accumulator_headroom(&lp, num_claims)?;
        let num_eval_rows = opening_points.len();

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
            let mut inner_opening_digits = Vec::new();
            let mut t_rows_by_poly = Vec::new();
            for (mut hint, &group_size) in hints.into_iter().zip(claim_group_sizes.iter()) {
                if hint.inner_opening_digits.len() != group_size {
                    return Err(AkitaError::InvalidInput(
                        "batched prover hint group sizes do not match polynomial groups"
                            .to_string(),
                    ));
                }
                hint.ensure_t_recomposed(lp.num_digits_open, lp.log_basis)?;
                let (digits_by_poly, rows_by_poly) = hint.into_parts();
                inner_opening_digits.extend(digits_by_poly);
                let rows_by_poly = rows_by_poly.ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "missing recomposed t rows in batched prover hint".to_string(),
                    )
                })?;
                t_rows_by_poly.extend(rows_by_poly);
            }
            AkitaCommitmentHint::with_t(inner_opening_digits, t_rows_by_poly)
        };

        let group_layouts = lp.group_layouts(claim_group_sizes, opening_points.len())?;
        let v = {
            let _span = tracing::info_span!(
                "compute_batched_v",
                w_hat_planes = w_hat.flat_digits().len()
            )
            .entered();
            compute_v_tier_aware::<F, D>(
                ntt_d,
                lp.d_key.row_len(),
                stride,
                w_hat.flat_digits(),
                &group_layouts,
                &lp,
            )
        };

        transcript.append_serde(ABSORB_PROVER_V, &RingSliceSerializer(&v));
        let total_stage1_blocks = group_layouts
            .last()
            .map(|layout| layout.block_start + layout.claim_count * layout.spec.num_blocks)
            .unwrap_or(0);
        // Tensor stage-1 challenges require a power-of-two block count.
        // For heterogeneous groups (e.g. tiered W + k chunks + meta with
        // distinct per-group `num_blocks`), the natural sum of group
        // blocks is not a power of two; round up so the tensor split
        // evenly halves. Surplus challenge slots are zero-weighted at
        // evaluation time per `MRowLayout`'s row layout.
        let (challenge_blocks, challenge_claims) = if lp.groups_are_homogeneous() {
            (lp.num_blocks, num_claims)
        } else {
            (total_stage1_blocks.next_power_of_two().max(1), 1)
        };
        let challenges = sample_stage1_challenges::<F, T, D>(
            transcript,
            challenge_blocks,
            challenge_claims,
            &lp.stage1_config,
            &lp.stage1_challenge_shape,
        )?;
        let mut integer_challenges: Option<Vec<IntegerChallenge>> = None;

        let z_pre = {
            let num_points = opening_points.len();
            let _span =
                tracing::info_span!("compute_batched_z_pre", num_points = num_points).entered();
            let mut polys_by_point: Vec<Vec<&P>> = vec![Vec::new(); num_points];
            let mut claim_indices_by_point: Vec<Vec<usize>> = vec![Vec::new(); num_points];
            for (claim_idx, poly) in polys.iter().enumerate() {
                let point_idx = claim_to_point[claim_idx];
                polys_by_point[point_idx].push(*poly);
                claim_indices_by_point[point_idx].push(claim_idx);
            }
            let stage1_challenges_by_point: Vec<Stage1Challenges> = claim_indices_by_point
                .iter()
                .map(|claim_indices| {
                    stage1_challenges_for_claims(&challenges, claim_indices, lp.num_blocks)
                })
                .collect::<Result<_, _>>()?;
            let mut integer_challenges_by_point: Option<Vec<Vec<IntegerChallenge>>> = None;

            let mut z_pre = Vec::new();
            let mut centered_coeffs = Vec::new();
            let mut centered_inf_norm = 0 as CenteredInfNorm;
            for (point_idx, point_polys) in polys_by_point.iter().enumerate() {
                let point_stage1_challenges = &stage1_challenges_by_point[point_idx];
                let point_claim_count = point_polys.len();
                let witness = if let Some(z_point) = P::decompose_fold_stage1_batched(
                    point_polys,
                    point_stage1_challenges,
                    lp.block_len,
                    lp.num_digits_commit,
                    lp.log_basis,
                )? {
                    z_point
                } else {
                    if integer_challenges.is_none() {
                        integer_challenges = Some(challenges.expand_integer::<D>()?);
                    }
                    if integer_challenges_by_point.is_none() {
                        integer_challenges_by_point = Some(integer_challenges_for_claim_groups(
                            integer_challenges
                                .as_ref()
                                .expect("integer challenges cached"),
                            &claim_indices_by_point,
                            lp.num_blocks,
                        )?);
                    }
                    let point_challenges = &integer_challenges_by_point
                        .as_ref()
                        .expect("grouped challenges")[point_idx];
                    let witnesses: Vec<DecomposeFoldWitness<F, D>> = point_polys
                        .iter()
                        .zip(point_challenges.chunks(lp.num_blocks))
                        .map(|(poly, poly_challenges)| {
                            poly.decompose_fold_integer(
                                poly_challenges,
                                lp.block_len,
                                lp.num_digits_commit,
                                lp.log_basis,
                            )
                        })
                        .collect::<Result<_, _>>()?;
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
        let y = generate_y_for_layout::<F, D>(
            &v,
            &commitment_rows,
            y_rings,
            &lp.m_row_layout(claim_group_sizes.len(), y_rings.len()),
        )?;
        let w_folded = pre_folded_by_poly.into_iter().flatten().collect();

        Ok(Self {
            v,
            challenges,
            integer_challenges,
            y,
            opening_points,
            claim_to_point,
            z_pre: Some(z_pre),
            w_hat: Some(w_hat),
            w_folded: Some(w_folded),
            hint: Some(flattened_hint),
            claim_group_sizes: claim_group_sizes.to_vec(),
            gamma,
            num_eval_rows,
        })
    }

    /// Recursive prover constructor: multi-claim path driven by the dedicated
    /// recursive witness view instead of the root polynomial trait.
    ///
    /// Mirrors [`Self::new_prover`] for the recursive case where every
    /// polynomial is a [`RecursiveWitnessView`]. `opening_points` holds the
    /// distinct ring-level opening points, `claim_to_point` maps each
    /// flattened claim to its opening-point index, and
    /// `claim_group_sizes` describes the commitment grouping. The
    /// trivial single-claim case is `opening_points = vec![pt]`,
    /// `claim_to_point = vec![0]`, `witnesses = &[&w]`,
    /// `claim_group_sizes = &[1]`, `gamma = vec![F::one()]`,
    /// `num_eval_rows = 1`.
    ///
    /// # Errors
    ///
    /// Returns an error if input lengths do not agree, the norm check, the
    /// challenge sampling, or the matrix generation fails.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "QuadraticEquation::new_recursive_prover")]
    #[inline(never)]
    pub fn new_recursive_prover<T: Transcript<F>>(
        ntt_d: &NttSlotCache<D>,
        opening_points: Vec<RingOpeningPoint<F>>,
        claim_to_point: Vec<usize>,
        witnesses: &[RecursiveHandlePoly<'_, F, D>],
        pre_folded_by_claim: Vec<Vec<CyclotomicRing<F, D>>>,
        claim_group_sizes: &[usize],
        lp: LevelParams,
        hints: Vec<AkitaCommitmentHint<F, D>>,
        transcript: &mut T,
        commitments: &[&[CyclotomicRing<F, D>]],
        y_rings: &[CyclotomicRing<F, D>],
        gamma: Vec<F>,
        num_eval_rows: usize,
        stride: usize,
    ) -> Result<Self, AkitaError> {
        {
            let x: u8 = 0;
            tracing::trace!(
                stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
                "QuadraticEquation::new_recursive_prover"
            );
        }

        if opening_points.is_empty() {
            return Err(AkitaError::InvalidInput(
                "recursive prover requires at least one opening point".to_string(),
            ));
        }
        // Opening-point inner dimensions are not validated here: at recursive
        // levels `lp.block_len` need not equal `1 << lp.r_vars`, so the strict
        // shape check used in `new_prover` does not apply. Downstream stage-1
        // and ring-switch surface any inconsistency.
        if witnesses.is_empty() || claim_group_sizes.is_empty() {
            return Err(AkitaError::InvalidInput(
                "recursive prover requires at least one witness".to_string(),
            ));
        }
        if claim_group_sizes.contains(&0) {
            return Err(AkitaError::InvalidInput(
                "recursive prover requires nonempty commitment groups".to_string(),
            ));
        }
        let num_claims = claim_group_sizes
            .iter()
            .try_fold(0usize, |acc, &group_size| {
                acc.checked_add(group_size).ok_or_else(|| {
                    AkitaError::InvalidInput("recursive prover claim count overflow".to_string())
                })
            })?;
        if witnesses.len() != pre_folded_by_claim.len()
            || witnesses.len() != num_claims
            || y_rings.len() != opening_points.len()
            || claim_to_point.len() != num_claims
            || hints.len() != claim_group_sizes.len()
            || commitments.len() != claim_group_sizes.len()
            || num_eval_rows != opening_points.len()
        {
            return Err(AkitaError::InvalidInput(
                "recursive prover input lengths do not match".to_string(),
            ));
        }
        if claim_to_point
            .iter()
            .any(|&point_idx| point_idx >= opening_points.len())
        {
            return Err(AkitaError::InvalidInput(
                "recursive prover claim-to-point index out of range".to_string(),
            ));
        }
        if gamma.len() != num_claims {
            return Err(AkitaError::InvalidInput(
                "recursive prover gamma length mismatch".to_string(),
            ));
        }
        validate_stage1_accumulator_headroom(&lp, num_claims)?;
        let group_layouts = lp.group_layouts(claim_group_sizes, num_eval_rows)?;
        for (commitment, layout) in commitments.iter().zip(group_layouts.iter()) {
            // Phase 5: a tier-marked group with `claim_count = k`
            // aggregates `k` per-chunk B-side commitments into one
            // group-level u-vector of length `k * n_B_chunk`. Other
            // groups have one claim and one u-vector of length `n_B`.
            let expected = layout.claim_count * layout.spec.b_key.row_len();
            if commitment.len() != expected {
                return Err(AkitaError::InvalidInput(format!(
                    "recursive prover commitment width mismatch: \
                     group={}, claim_count={}, n_B={}, expected={expected}, actual={}",
                    layout.group_idx,
                    layout.claim_count,
                    layout.spec.b_key.row_len(),
                    commitment.len(),
                )));
            }
        }

        let w_hat = {
            let _span = tracing::info_span!("decompose_recursive_w_hat").entered();
            let log_basis = lp.log_basis;
            let q = (-F::one()).to_canonical_u128() + 1;
            let mut block_sizes = Vec::new();
            for layout in &group_layouts {
                let depth_open = layout.spec.num_digits_open;
                for folded in &pre_folded_by_claim
                    [layout.claim_start..layout.claim_start + layout.claim_count]
                {
                    block_sizes.extend(std::iter::repeat_n(depth_open, folded.len()));
                }
            }
            let mut w_hat = FlatDigitBlocks::zeroed(block_sizes)?;
            let mut offset = 0usize;
            for layout in &group_layouts {
                let depth_open = layout.spec.num_digits_open;
                let decompose_params = BalancedDecomposePow2I8Params::new(depth_open, log_basis, q);
                for folded in &pre_folded_by_claim
                    [layout.claim_start..layout.claim_start + layout.claim_count]
                {
                    for w_i in folded {
                        w_i.balanced_decompose_pow2_i8_into_with_params(
                            &mut w_hat.flat_digits_mut()[offset..offset + depth_open],
                            &decompose_params,
                        );
                        offset += depth_open;
                    }
                }
            }
            w_hat
        };

        let flattened_hint = {
            let mut inner_opening_digits = Vec::new();
            let mut t_rows_by_poly = Vec::new();
            for ((mut hint, &group_size), layout) in hints
                .into_iter()
                .zip(claim_group_sizes.iter())
                .zip(group_layouts.iter())
            {
                if hint.inner_opening_digits.len() != group_size {
                    return Err(AkitaError::InvalidInput(
                        "recursive prover hint group sizes do not match polynomial groups"
                            .to_string(),
                    ));
                }
                hint.ensure_t_recomposed(layout.spec.num_digits_open, lp.log_basis)?;
                let (digits_by_poly, rows_by_poly) = hint.into_parts();
                inner_opening_digits.extend(digits_by_poly);
                let rows_by_poly = rows_by_poly.ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "missing recomposed t rows in recursive prover hint".to_string(),
                    )
                })?;
                t_rows_by_poly.extend(rows_by_poly);
            }
            AkitaCommitmentHint::with_t(inner_opening_digits, t_rows_by_poly)
        };

        let v = {
            let _span = tracing::info_span!(
                "compute_recursive_v",
                w_hat_planes = w_hat.flat_digits().len()
            )
            .entered();
            compute_v_tier_aware::<F, D>(
                ntt_d,
                lp.d_key.row_len(),
                stride,
                w_hat.flat_digits(),
                &group_layouts,
                &lp,
            )
        };

        transcript.append_serde(ABSORB_PROVER_V, &RingSliceSerializer(&v));

        let total_stage1_blocks = group_layouts
            .last()
            .map(|layout| layout.block_start + layout.claim_count * layout.spec.num_blocks)
            .unwrap_or(0);
        let (challenge_blocks, challenge_claims) = if lp.groups_are_homogeneous() {
            (lp.num_blocks, num_claims)
        } else {
            (total_stage1_blocks.next_power_of_two().max(1), 1)
        };
        let challenges = sample_stage1_challenges::<F, T, D>(
            transcript,
            challenge_blocks,
            challenge_claims,
            &lp.stage1_config,
            &lp.stage1_challenge_shape,
        )?;
        // Recursive witnesses do not yet expose a `decompose_fold_stage1_batched`
        // shortcut, so the path always expands to the integer-challenge form.
        let integer_challenges = {
            let _span = tracing::info_span!("expand_recursive_stage1_challenges").entered();
            challenges.expand_integer::<D>()?
        };

        let z_pre = {
            let num_points = opening_points.len();
            let _span =
                tracing::info_span!("compute_recursive_z_pre", num_points = num_points).entered();
            if !lp.groups_are_homogeneous() {
                let mut z_pre = Vec::new();
                let mut centered_coeffs = Vec::new();
                let mut centered_inf_norm = 0 as CenteredInfNorm;
                for layout in &group_layouts {
                    let spec = &layout.spec;
                    let inner_width = spec.block_len * spec.num_digits_commit;
                    for point_idx in 0..num_points {
                        let mut witnesses_dfw = Vec::new();
                        for claim_idx in layout.claim_start..layout.claim_start + layout.claim_count
                        {
                            if claim_to_point[claim_idx] != point_idx {
                                continue;
                            }
                            let local_claim = claim_idx - layout.claim_start;
                            let start = layout.block_start + local_claim * spec.num_blocks;
                            let end = start + spec.num_blocks;
                            let claim_challenges =
                                integer_challenges.get(start..end).ok_or_else(|| {
                                    AkitaError::InvalidSetup(format!(
                                        "heterogeneous recursive challenge slice out of range: start={start}, end={end}, actual={}, group={}, block_start={}, num_blocks={}, claim_count={}, homogeneous={}",
                                        integer_challenges.len(),
                                        layout.group_idx,
                                        layout.block_start,
                                        spec.num_blocks,
                                        layout.claim_count,
                                        lp.groups_are_homogeneous()
                                    ))
                                })?;
                            witnesses_dfw.push(witnesses[claim_idx].decompose_fold_integer(
                                claim_challenges,
                                spec.block_len,
                                spec.num_blocks,
                                spec.num_digits_commit,
                                lp.log_basis,
                            )?);
                        }
                        let witness = if witnesses_dfw.is_empty() {
                            DecomposeFoldWitness {
                                z_pre: vec![CyclotomicRing::<F, D>::zero(); inner_width],
                                centered_coeffs: vec![[0 as CenteredCoeff; D]; inner_width],
                                centered_inf_norm: 0 as CenteredInfNorm,
                            }
                        } else {
                            aggregate_decompose_fold_witnesses(witnesses_dfw)?
                        };
                        if witness.z_pre.len() != inner_width
                            || witness.centered_coeffs.len() != inner_width
                        {
                            return Err(AkitaError::InvalidInput(
                                "heterogeneous recursive decompose-fold produced an invalid witness shape"
                                    .to_string(),
                            ));
                        }
                        centered_inf_norm = centered_inf_norm.max(witness.centered_inf_norm);
                        z_pre.extend(witness.z_pre);
                        centered_coeffs.extend(witness.centered_coeffs);
                    }
                }
                let commitment_rows: Vec<CyclotomicRing<F, D>> = commitments
                    .iter()
                    .flat_map(|commitment| commitment.iter().copied())
                    .collect();
                let y = generate_y_for_layout::<F, D>(
                    &v,
                    &commitment_rows,
                    y_rings,
                    &lp.m_row_layout(claim_group_sizes.len(), y_rings.len()),
                )?;
                return Ok(Self {
                    v,
                    challenges,
                    integer_challenges: Some(integer_challenges),
                    y,
                    opening_points,
                    claim_to_point,
                    z_pre: Some(DecomposeFoldWitness {
                        z_pre,
                        centered_coeffs,
                        centered_inf_norm,
                    }),
                    w_hat: Some(w_hat),
                    w_folded: Some(pre_folded_by_claim.into_iter().flatten().collect()),
                    hint: Some(flattened_hint),
                    claim_group_sizes: claim_group_sizes.to_vec(),
                    gamma,
                    num_eval_rows,
                });
            }
            let mut witnesses_by_point: Vec<Vec<&RecursiveHandlePoly<'_, F, D>>> =
                vec![Vec::new(); num_points];
            let mut claim_indices_by_point: Vec<Vec<usize>> = vec![Vec::new(); num_points];
            for (claim_idx, witness) in witnesses.iter().enumerate() {
                let point_idx = claim_to_point[claim_idx];
                witnesses_by_point[point_idx].push(witness);
                claim_indices_by_point[point_idx].push(claim_idx);
            }
            let integer_challenges_by_point = integer_challenges_for_claim_groups(
                &integer_challenges,
                &claim_indices_by_point,
                lp.num_blocks,
            )?;

            let mut z_pre = Vec::new();
            let mut centered_coeffs = Vec::new();
            let mut centered_inf_norm = 0 as CenteredInfNorm;
            for (point_idx, point_witnesses) in witnesses_by_point.iter().enumerate() {
                let point_claim_count = point_witnesses.len();
                let point_challenges = &integer_challenges_by_point[point_idx];
                let witnesses_dfw: Vec<DecomposeFoldWitness<F, D>> = point_witnesses
                    .iter()
                    .zip(point_challenges.chunks(lp.num_blocks))
                    .map(|(witness, witness_challenges)| {
                        witness.decompose_fold_integer(
                            witness_challenges,
                            lp.block_len,
                            lp.num_blocks,
                            lp.num_digits_commit,
                            lp.log_basis,
                        )
                    })
                    .collect::<Result<_, _>>()?;
                let witness = aggregate_decompose_fold_witnesses(witnesses_dfw)?;
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
            .flat_map(|commitment| commitment.iter().copied())
            .collect();
        let y = generate_y_for_layout::<F, D>(
            &v,
            &commitment_rows,
            y_rings,
            &lp.m_row_layout(claim_group_sizes.len(), y_rings.len()),
        )?;
        let w_folded: Vec<CyclotomicRing<F, D>> =
            pre_folded_by_claim.into_iter().flatten().collect();

        Ok(Self {
            v,
            challenges,
            integer_challenges: Some(integer_challenges),
            y,
            opening_points,
            claim_to_point,
            z_pre: Some(z_pre),
            w_hat: Some(w_hat),
            w_folded: Some(w_folded),
            hint: Some(flattened_hint),
            claim_group_sizes: claim_group_sizes.to_vec(),
            gamma,
            num_eval_rows,
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

    /// Get cached expanded integer stage-1 folding challenges when a fallback
    /// path needed to materialize them.
    pub fn integer_challenges(&self) -> Option<&[IntegerChallenge]> {
        self.integer_challenges.as_deref()
    }

    /// Expand stage-1 challenges into the integer representation for debug or
    /// reference code.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor expansion fails.
    pub fn expanded_integer_challenges(&self) -> Result<Vec<IntegerChallenge>, AkitaError> {
        match &self.integer_challenges {
            Some(challenges) => Ok(challenges.clone()),
            None => self.challenges.expand_integer::<D>(),
        }
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

    /// Number of flattened public claims carried by each commitment group.
    pub fn claim_group_sizes(&self) -> &[usize] {
        &self.claim_group_sizes
    }

    /// Per-claim batching coefficients used by the relation rows.
    pub fn gamma(&self) -> &[F] {
        &self.gamma
    }

    /// Number of batched public y-rows in the matrix equation.  Equals
    /// the number of distinct opening points (one row per point).
    pub fn num_eval_rows(&self) -> usize {
        self.num_eval_rows
    }

    /// Get the pre-decomposition folded witness `z_pre` (prover only).
    pub fn z_pre(&self) -> Option<&[CyclotomicRing<F, D>]> {
        self.z_pre.as_ref().map(|witness| witness.z_pre.as_slice())
    }

    /// Get centered coefficients for each `z_pre` row (prover only).
    pub fn z_pre_centered(&self) -> Option<&[[CenteredCoeff; D]]> {
        self.z_pre
            .as_ref()
            .map(|witness| witness.centered_coeffs.as_slice())
    }

    /// Get `||z_pre||_inf` from the centered witness representation.
    pub fn z_pre_centered_inf_norm(&self) -> Option<CenteredInfNorm> {
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

enum Stage1ChallengeProducts<'a, const D: usize> {
    Flat(&'a [SparseChallenge]),
    Tensor(&'a TensorStage1Challenges),
}

impl<'a, const D: usize> Stage1ChallengeProducts<'a, D> {
    fn new(
        challenges: &'a Stage1Challenges,
        blocks_per_claim: usize,
        num_claims: usize,
    ) -> Result<Self, AkitaError> {
        let expected = blocks_per_claim
            .checked_mul(num_claims)
            .ok_or(AkitaError::InvalidProof)?;
        if challenges.logical_len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: challenges.logical_len(),
            });
        }
        match challenges {
            Stage1Challenges::Flat(flat) => Ok(Self::Flat(flat)),
            Stage1Challenges::Tensor(tensor) => {
                if tensor.left_len.checked_mul(tensor.right_len) != Some(blocks_per_claim) {
                    return Err(AkitaError::InvalidSize {
                        expected: blocks_per_claim,
                        actual: tensor.left_len.saturating_mul(tensor.right_len),
                    });
                }
                if tensor.num_claims != num_claims {
                    return Err(AkitaError::InvalidSize {
                        expected: num_claims,
                        actual: tensor.num_claims,
                    });
                }
                Ok(Self::Tensor(tensor))
            }
        }
    }

    #[inline(always)]
    fn add_high_half<F: FieldCore + CanonicalField>(
        &self,
        quotient: &mut [F],
        block_idx: usize,
        ring: &CyclotomicRing<F, D>,
    ) {
        match self {
            Self::Flat(challenges) => {
                add_sparse_challenge_ring_product_high_half::<F, D>(
                    quotient,
                    &challenges[block_idx],
                    ring,
                );
            }
            Self::Tensor(tensor) => {
                let (left, right) = tensor_challenges_for_block(tensor, block_idx);
                add_tensor_challenge_ring_product_high_half::<F, D>(quotient, left, right, ring);
            }
        }
    }
}

#[inline(always)]
fn tensor_challenges_for_block(
    tensor: &TensorStage1Challenges,
    block_idx: usize,
) -> (&SparseChallenge, &SparseChallenge) {
    let blocks_per_claim = tensor.left_len * tensor.right_len;
    let claim_idx = block_idx / blocks_per_claim;
    let local_idx = block_idx % blocks_per_claim;
    let left_idx = claim_idx * tensor.left_len + (local_idx / tensor.right_len);
    let right_idx = claim_idx * tensor.right_len + (local_idx % tensor.right_len);
    (&tensor.left[left_idx], &tensor.right[right_idx])
}

fn add_sparse_challenge_ring_product_high_half<F: FieldCore + CanonicalField, const D: usize>(
    quotient: &mut [F],
    challenge: &SparseChallenge,
    ring: &CyclotomicRing<F, D>,
) {
    let rc = ring.coefficients();
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let c = F::from_i64(i64::from(coeff));
        let p = pos as usize;
        for s in (D - p)..D {
            quotient[p + s - D] += c * rc[s];
        }
    }
}

#[inline(always)]
fn add_tensor_challenge_ring_product_high_half<F: FieldCore + CanonicalField, const D: usize>(
    quotient: &mut [F],
    left: &SparseChallenge,
    right: &SparseChallenge,
    ring: &CyclotomicRing<F, D>,
) {
    let rc = ring.coefficients();
    for (&left_pos, &left_coeff) in left.positions.iter().zip(left.coeffs.iter()) {
        for (&right_pos, &right_coeff) in right.positions.iter().zip(right.coeffs.iter()) {
            let degree = left_pos as usize + right_pos as usize;
            let (p, sign) = if degree < D {
                (degree, 1i64)
            } else {
                (degree - D, -1i64)
            };
            let c = F::from_i64(i64::from(left_coeff) * i64::from(right_coeff) * sign);
            for s in (D - p)..D {
                quotient[p + s - D] += c * rc[s];
            }
        }
    }
}

/// Parallel high-half accumulation over blocks.
///
/// Replaces `for (i, ring) in rings { add_sparse_ring_product_high_half(...) }`
/// with a fold-reduce giving block-level parallelism.
fn parallel_high_half_accumulate<F: FieldCore + CanonicalField, const D: usize>(
    challenges: &Stage1ChallengeProducts<'_, D>,
    rings: &[CyclotomicRing<F, D>],
) -> Vec<F> {
    cfg_fold_reduce!(
        0..rings.len(),
        || vec![F::zero(); D],
        |mut acc: Vec<F>, i: usize| {
            challenges.add_high_half::<F>(&mut acc, i, &rings[i]);
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

fn decompose_centered_coeffs_i8<const D: usize>(
    centered: &[CenteredCoeff; D],
    out: &mut [[i8; D]],
    log_basis: u32,
) {
    let half_b = 1i128 << (log_basis - 1);
    let b = half_b << 1;
    let mask = b - 1;
    for coeff_idx in 0..D {
        let mut c = centered[coeff_idx] as i128;
        for plane in out.iter_mut() {
            let d = c & mask;
            let balanced = if d >= half_b { d - b } else { d };
            c = (c - balanced) >> log_basis;
            plane[coeff_idx] = balanced as i8;
        }
    }
}

fn repeated_b_commitment_rows<F: FieldCore + CanonicalField, const D: usize>(
    ntt_shared: &NttSlotCache<D>,
    n_b: usize,
    outer_width: usize,
    t_hat: &FlatDigitBlocks<D>,
    claim_group_sizes: &[usize],
    blocks_per_claim: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    if claim_group_sizes.is_empty() || blocks_per_claim == 0 {
        return Err(AkitaError::InvalidProof);
    }
    let num_claims = claim_group_sizes
        .iter()
        .try_fold(0usize, |acc, &group_size| {
            if group_size == 0 {
                return Err(AkitaError::InvalidProof);
            }
            acc.checked_add(group_size).ok_or(AkitaError::InvalidProof)
        })?;
    if t_hat.block_count() != num_claims * blocks_per_claim {
        return Err(AkitaError::InvalidProof);
    }
    let mut rows = Vec::with_capacity(claim_group_sizes.len() * n_b);
    let mut block_offset = 0usize;
    let mut plane_offset = 0usize;
    for &group_size in claim_group_sizes {
        let group_block_count = group_size
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
        rows.extend(mat_vec_mul_ntt_single_i8_cyclic(
            ntt_shared,
            n_b,
            outer_width,
            group_digits,
        ));
        block_offset = next_block_offset;
        plane_offset = next_plane_offset;
    }
    if block_offset != t_hat.block_count() || plane_offset != t_hat.flat_digits().len() {
        return Err(AkitaError::InvalidProof);
    }
    Ok(rows)
}

/// Compute the D-row RHS vector `v = D · W_hat mod (X^D+1)` with
/// tier-aware block-diagonal structure for tier-marked groups.
///
/// For homogeneous layouts (no tiered groups), this reduces to a single
/// fused mat-vec over the entire flat W_hat.
///
/// For heterogeneous layouts without a tier marker, this keeps the book §5.3
/// joint D check and returns `n_D` rows. Once a tier-marked group appears,
/// the book §5.4 per-chunk D checks are emitted explicitly: ordinary groups
/// contribute one `n_D` slice, and tiered groups contribute one `n_D` slice per
/// chunk, all using the shared low D columns for their per-group shape.
///
/// This must mirror the verifier's `eval_split_at_point_grouped` D-row
/// column selection (book §5.4 line 752 "shared `D_chunk` / `B_chunk`
/// MLE cost independent of `k`"). A mismatch here causes
/// `relation_claim_from_rows` to disagree with the sumcheck oracle
/// `sum_{y,x} w(y,x) · α(y) · m(x)` at recursive tiered levels.
fn compute_v_tier_aware<F, const D: usize>(
    ntt_d: &NttSlotCache<D>,
    n_d: usize,
    stride: usize,
    w_hat_flat: &[[i8; D]],
    group_layouts: &[GroupLayout],
    lp: &LevelParams,
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField,
{
    if lp.groups_are_homogeneous() {
        return mat_vec_mul_ntt_single_i8(ntt_d, n_d, stride, w_hat_flat);
    }
    let has_tiered_group = group_layouts
        .iter()
        .any(|layout| layout.spec.tier.is_some_and(|tier| tier.is_tiered()));
    let mut v: Vec<CyclotomicRing<F, D>> = if has_tiered_group {
        Vec::with_capacity(lp.total_d_row_count(group_layouts.len()))
    } else {
        vec![CyclotomicRing::<F, D>::zero(); n_d]
    };
    for layout in group_layouts {
        let per_claim_w_hat_len = layout.spec.num_blocks * layout.spec.num_digits_open;
        let is_tiered = layout.spec.tier.is_some_and(|tier| tier.is_tiered());
        if has_tiered_group && is_tiered {
            for claim_local in 0..layout.claim_count {
                let start = layout.w_hat_start + claim_local * per_claim_w_hat_len;
                let slice = &w_hat_flat[start..start + per_claim_w_hat_len];
                v.extend(mat_vec_mul_ntt_single_i8(ntt_d, n_d, stride, slice));
            }
        } else {
            let max_sum_magnitude =
                (layout.claim_count as i64).saturating_mul(((1i64 << lp.log_basis) + 1) / 2);
            debug_assert!(
                max_sum_magnitude <= i8::MAX as i64,
                "group W_hat sum may overflow i8 in compute_v_tier_aware"
            );
            let mut summed: Vec<[i8; D]> = vec![[0i8; D]; per_claim_w_hat_len];
            for claim_local in 0..layout.claim_count {
                let start = layout.w_hat_start + claim_local * per_claim_w_hat_len;
                let slice = &w_hat_flat[start..start + per_claim_w_hat_len];
                for (s, d) in slice.iter().zip(summed.iter_mut()) {
                    for k in 0..D {
                        d[k] = d[k].wrapping_add(s[k]);
                    }
                }
            }
            let group_v = mat_vec_mul_ntt_single_i8(ntt_d, n_d, stride, &summed);
            if has_tiered_group {
                v.extend(group_v);
            } else {
                for (acc, g) in v.iter_mut().zip(group_v.into_iter()) {
                    *acc += g;
                }
            }
        }
    }
    v
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
#[allow(
    clippy::too_many_arguments,
    clippy::needless_borrow,
    clippy::needless_range_loop
)]
#[tracing::instrument(skip_all, name = "compute_r_split_eq")]
pub fn compute_r_split_eq<F, const D: usize>(
    lp: &LevelParams,
    _setup: &AkitaExpandedSetup<F>,
    challenges: &Stage1Challenges,
    w_hat_flat: &[[i8; D]],
    t_hat: &FlatDigitBlocks<D>,
    t: &[Vec<CyclotomicRing<F, D>>],
    w_folded: &[CyclotomicRing<F, D>],
    z_pre_centered: &[[CenteredCoeff; D]],
    z_pre_centered_inf_norm: CenteredInfNorm,
    y: &[CyclotomicRing<F, D>],
    claim_group_sizes: &[usize],
    num_public_outputs: usize,
    blocks_per_claim: usize,
    inner_width: usize,
    stride: usize,
    ntt_shared: &NttSlotCache<D>,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField,
{
    if claim_group_sizes.is_empty() || claim_group_sizes.contains(&0) {
        return Err(AkitaError::InvalidProof);
    }
    if num_public_outputs == 0 {
        return Err(AkitaError::InvalidProof);
    }
    let num_claims = claim_group_sizes
        .iter()
        .try_fold(0usize, |acc, &group_size| {
            acc.checked_add(group_size).ok_or(AkitaError::InvalidProof)
        })?;
    let group_layouts = lp.group_layouts(claim_group_sizes, num_public_outputs)?;
    let total_group_blocks = group_layouts
        .last()
        .map(|layout| layout.block_start + layout.claim_count * layout.spec.num_blocks)
        .unwrap_or(0);
    let challenge_products = if lp.groups_are_homogeneous() {
        Stage1ChallengeProducts::new(challenges, blocks_per_claim, num_claims)?
    } else {
        // Tensor stage-1 challenges are sized to the next power of two
        // of `total_group_blocks`. The surplus slots are zero-weighted
        // at row evaluation time per the heterogeneous-group layout.
        Stage1ChallengeProducts::new(challenges, total_group_blocks.next_power_of_two().max(1), 1)?
    };
    let num_commitment_groups = claim_group_sizes.len();
    let n_b = lp.b_key.row_len();
    let n_d = lp.d_key.row_len();
    let n_a = lp.a_key.row_len();
    let a_row_count = lp.total_a_row_count(num_commitment_groups);
    let d_row_count = lp.total_d_row_count(num_commitment_groups);
    let commitment_row_count = lp.total_b_row_count(num_commitment_groups);
    let row_layout = lp.m_row_layout(num_commitment_groups, num_public_outputs);
    let num_rows = row_layout.rows;
    if y.len() != num_rows {
        return Err(AkitaError::InvalidSetup(format!(
            "compute_r_split_eq y row mismatch: y_len={}, expected={num_rows}, groups={claim_group_sizes:?}, total_b_rows={commitment_row_count}",
            y.len()
        )));
    }
    let expected_blocks = if lp.groups_are_homogeneous() {
        blocks_per_claim
            .checked_mul(num_claims)
            .ok_or(AkitaError::InvalidProof)?
    } else {
        // Heterogeneous groups: each group contributes its own
        // `claim_count * spec.num_blocks` to the folded witness, so
        // the total is the layout's `total_group_blocks`.
        total_group_blocks
    };
    if w_folded.len() != expected_blocks || t.len() != expected_blocks {
        return Err(AkitaError::InvalidSetup(format!(
            "compute_r_split_eq folded/t length mismatch: w_folded={}, t={}, expected_blocks={expected_blocks}",
            w_folded.len(),
            t.len()
        )));
    }
    let has_tiered_group = group_layouts
        .iter()
        .any(|layout| layout.spec.tier.is_some_and(|tier| tier.is_tiered()));

    let (d_cyclic, b_cyclic, a_quotients) = if lp.groups_are_homogeneous() {
        if inner_width == 0 || !z_pre_centered.len().is_multiple_of(inner_width) {
            return Err(AkitaError::InvalidSetup(format!(
                "compute_r_split_eq z_pre shape mismatch: z_pre={}, inner_width={inner_width}",
                z_pre_centered.len()
            )));
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
        (d_cyclic, b_cyclic, a_quotients)
    } else {
        // Heterogeneous-group D-row and A-row quotients per book §3.4:
        // D and A are applied per-group with each group's first
        // `inner_d_g` / `inner_width_g` cols (matching the per-group
        // commit's `[0, inner_width_g)` cols). This mirrors the
        // verifier's `d_col = local_blk * spec.num_digits_open + dig`
        // and `phys_k = blk * spec.num_digits_commit + dc` formulas.
        // Tier-marked groups sum the `claim_count` chunks' W_hat /
        // z_pre slices before the per-group mat-vec; un-tiered groups
        // (claim_count = 1) just copy their slice.
        let mut d_cyclic: Vec<CyclotomicRing<F, D>> = if has_tiered_group {
            Vec::with_capacity(d_row_count)
        } else {
            vec![CyclotomicRing::<F, D>::zero(); n_d]
        };
        let mut a_quotients: Vec<CyclotomicRing<F, D>> =
            vec![CyclotomicRing::<F, D>::zero(); a_row_count];
        for layout in &group_layouts {
            let spec = &layout.spec;
            let inner_width_g = spec.block_len * spec.num_digits_commit;
            let per_claim_w_hat_len = spec.num_blocks * spec.num_digits_open;
            // D-row: without tiering, retain the split-commitment joint D
            // row. With tiering, emit one D slice per ordinary group and
            // one D slice per chunk of each tiered group.
            let is_tiered = spec.tier.is_some_and(|tier| tier.is_tiered());
            if has_tiered_group && is_tiered {
                for claim_local in 0..layout.claim_count {
                    let start = layout.w_hat_start + claim_local * per_claim_w_hat_len;
                    let end = start + per_claim_w_hat_len;
                    let claim_digits =
                        w_hat_flat.get(start..end).ok_or(AkitaError::InvalidProof)?;
                    d_cyclic.extend(mat_vec_mul_ntt_single_i8_cyclic(
                        ntt_shared,
                        n_d,
                        stride,
                        claim_digits,
                    ));
                }
            } else {
                let max_sum_magnitude =
                    (layout.claim_count as i64).saturating_mul(((1i64 << lp.log_basis) + 1) / 2);
                if max_sum_magnitude > i8::MAX as i64 {
                    return Err(AkitaError::InvalidSetup(format!(
                        "group W_hat sum may overflow i8: claim_count={}, b={}, max=|{}| > {}",
                        layout.claim_count,
                        1usize << lp.log_basis,
                        max_sum_magnitude,
                        i8::MAX
                    )));
                }
                let mut summed_w: Vec<[i8; D]> = vec![[0i8; D]; per_claim_w_hat_len];
                for claim_local in 0..layout.claim_count {
                    let start = layout.w_hat_start + claim_local * per_claim_w_hat_len;
                    let slice = w_hat_flat
                        .get(start..start + per_claim_w_hat_len)
                        .ok_or(AkitaError::InvalidProof)?;
                    for (s, d) in slice.iter().zip(summed_w.iter_mut()) {
                        for k in 0..D {
                            d[k] = d[k].wrapping_add(s[k]);
                        }
                    }
                }
                let group_d_cyclic =
                    mat_vec_mul_ntt_single_i8_cyclic(ntt_shared, n_d, stride, &summed_w);
                if has_tiered_group {
                    d_cyclic.extend(group_d_cyclic);
                } else {
                    for (acc, g) in d_cyclic.iter_mut().zip(group_d_cyclic.into_iter()) {
                        *acc += g;
                    }
                }
            }

            // A-row Z-part: mat-vec A's first `inner_width_g` cols
            // against this group's z_pre slot at point_idx = group_idx.
            // For un-tiered groups, only the matching point_idx has
            // non-zero z_pre; for tier-marked, chunks aggregate via
            // unweighted sum at point_idx = group_idx.
            //
            for point_idx in 0..num_public_outputs {
                if point_idx != layout.group_idx {
                    continue;
                }
                let slot_start = layout.z_base_start + point_idx * inner_width_g;
                let slot_end = slot_start + inner_width_g;
                let Some(z_slice) = z_pre_centered.get(slot_start..slot_end) else {
                    continue;
                };
                let a_offset = if has_tiered_group {
                    if group_layouts.len() >= 3 {
                        if layout.group_idx == 0 {
                            0
                        } else if layout.group_idx + 1 == group_layouts.len() {
                            2 * n_a
                        } else {
                            n_a
                        }
                    } else if layout.group_idx + 1 == group_layouts.len() {
                        n_a
                    } else {
                        0
                    }
                } else {
                    0
                };
                if has_tiered_group && is_tiered {
                    // Tier-marked chunks: each per-cell polynomial product
                    // `A_row[c] · z_pre[c]` already has integer
                    // coefficient bound `D · |A| · |z|` that exceeds
                    // the `Q128` CRT product `P ≈ 2^150` once
                    // `|z| > 2^16`. The NTT-based kernel reconstructs
                    // mod `P` per tile, so even a single-cell tile
                    // wraps. The kernel-level tile-reduce-to-`F` fix
                    // only addresses cross-tile CRT overflow, not
                    // per-cell. Take the direct field-domain
                    // high-half path here for tier-marked chunks at
                    // the cost of `O(n_a · inner_width_g · D²)` field
                    // multiplies (book §5.4 chunks A-row, exercised
                    // once per recursive level per tiered group).
                    let a_view = _setup.shared_matrix.ring_view::<D>(n_a, stride);
                    #[allow(clippy::needless_range_loop)]
                    for r in 0..n_a {
                        let mut high_half = [F::zero(); D];
                        for (col_idx, z_cell) in z_slice.iter().enumerate() {
                            let a_cell = a_view.row(r)[col_idx];
                            let a_coeffs = a_cell.coefficients();
                            #[allow(clippy::needless_range_loop)]
                            for i in 0..D {
                                let a_i = a_coeffs[i];
                                if a_i.is_zero() {
                                    continue;
                                }
                                for (j, &z_j) in z_cell.iter().enumerate() {
                                    if z_j == 0 || i + j < D {
                                        continue;
                                    }
                                    let term = a_i * F::from_i64(z_j);
                                    high_half[i + j - D] += term;
                                }
                            }
                        }
                        a_quotients[a_offset + r] += CyclotomicRing::from_coefficients(high_half);
                    }
                } else {
                    let (_, _, group_a_q) = fused_split_eq_quotients::<F, D>(
                        ntt_shared,
                        0,
                        0,
                        n_a,
                        stride,
                        &[],
                        &[],
                        z_slice,
                        z_pre_centered_inf_norm,
                    );
                    for (dst, src) in a_quotients[a_offset..a_offset + n_a]
                        .iter_mut()
                        .zip(group_a_q.into_iter())
                    {
                        *dst += src;
                    }
                }
            }
        }
        (d_cyclic, Vec::new(), a_quotients)
    };
    if d_cyclic.len() != d_row_count {
        return Err(AkitaError::InvalidProof);
    }

    let commitment_cyclic_rows = if !lp.groups_are_homogeneous() {
        let mut rows = Vec::with_capacity(commitment_row_count);
        let flat = t_hat.flat_digits();
        for layout in &group_layouts {
            // Phase 5 tier-aware: iterate per-claim within the group so
            // each chunk contributes its own `n_B_chunk` B-rows. For
            // un-tiered groups (claim_count = 1), this is one mat-vec.
            let per_claim_len = layout.spec.num_blocks * n_a * layout.spec.num_digits_open;
            for claim_local in 0..layout.claim_count {
                let start = layout.t_hat_start + claim_local * per_claim_len;
                let end = start + per_claim_len;
                let claim_digits = flat.get(start..end).ok_or(AkitaError::InvalidProof)?;
                rows.extend(mat_vec_mul_ntt_single_i8_cyclic(
                    ntt_shared,
                    layout.spec.b_key.row_len(),
                    stride,
                    claim_digits,
                ));
            }
        }
        rows
    } else if commitment_row_count == n_b && num_commitment_groups == 1 {
        b_cyclic
    } else {
        repeated_b_commitment_rows(
            ntt_shared,
            n_b,
            stride,
            t_hat,
            claim_group_sizes,
            blocks_per_claim,
        )?
    };
    if commitment_cyclic_rows.len() != commitment_row_count {
        return Err(AkitaError::InvalidProof);
    }

    let mut result = Vec::with_capacity(num_rows);
    let mut other_time = 0.0f64;

    let range_index = |range: &std::ops::Range<usize>, row_idx: usize| {
        range.contains(&row_idx).then(|| row_idx - range.start)
    };

    for row_idx in 0..num_rows {
        if row_layout.w_fold == Some(row_idx)
            || row_idx == row_layout.original_fold
            || row_layout.meta_fold == Some(row_idx)
        {
            let t_row = Instant::now();
            let _span = tracing::info_span!("challenge_fold_row").entered();
            let row_role = if row_layout.w_fold == Some(row_idx) {
                0usize
            } else if row_layout.meta_fold == Some(row_idx) {
                2usize
            } else {
                1usize
            };
            let quotient = if !has_tiered_group {
                parallel_high_half_accumulate::<F, D>(&challenge_products, w_folded)
            } else {
                cfg_fold_reduce!(
                    0..group_layouts.len(),
                    || vec![F::zero(); D],
                    |mut acc: Vec<F>, layout_idx: usize| {
                        let layout = &group_layouts[layout_idx];
                        let layout_role = if has_tiered_group && group_layouts.len() >= 3 {
                            if layout.group_idx == 0 {
                                0usize
                            } else if layout.group_idx + 1 == group_layouts.len() {
                                2usize
                            } else {
                                1usize
                            }
                        } else if has_tiered_group && layout.group_idx + 1 == group_layouts.len() {
                            2usize
                        } else {
                            1usize
                        };
                        if layout_role == row_role {
                            let group_blocks = layout.claim_count * layout.spec.num_blocks;
                            for local_blk in 0..group_blocks {
                                let i = layout.block_start + local_blk;
                                if let Some(w_i) = w_folded.get(i) {
                                    challenge_products.add_high_half::<F>(&mut acc, i, w_i);
                                }
                            }
                        }
                        acc
                    },
                    |mut a: Vec<F>, b: Vec<F>| {
                        for (ai, bi) in a.iter_mut().zip(b.iter()) {
                            *ai += *bi;
                        }
                        a
                    }
                )
            };
            result.push(CyclotomicRing::from_slice(&quotient));
            other_time += t_row.elapsed().as_secs_f64();
        } else if row_layout.w_eval.contains(&row_idx)
            || row_layout.original_eval.contains(&row_idx)
            || row_layout.meta_eval.contains(&row_idx)
        {
            let _span = tracing::info_span!("bTw_row").entered();
            result.push(CyclotomicRing::<F, D>::zero());
        } else if let Some(d_idx) = range_index(&row_layout.w_d, row_idx)
            .or_else(|| {
                range_index(&row_layout.original_d, row_idx).map(|idx| row_layout.w_d.len() + idx)
            })
            .or_else(|| {
                range_index(&row_layout.meta_d, row_idx)
                    .map(|idx| row_layout.w_d.len() + row_layout.original_d.len() + idx)
            })
        {
            result.push(quotient_from_cyclic_and_reduced(
                &d_cyclic[d_idx],
                &y[row_idx],
            ));
        } else if let Some(b_idx) = range_index(&row_layout.w_b, row_idx)
            .or_else(|| {
                range_index(&row_layout.original_b, row_idx).map(|idx| row_layout.w_b.len() + idx)
            })
            .or_else(|| {
                range_index(&row_layout.meta_b, row_idx)
                    .map(|idx| row_layout.w_b.len() + row_layout.original_b.len() + idx)
            })
        {
            result.push(quotient_from_cyclic_and_reduced(
                &commitment_cyclic_rows[b_idx],
                &y[row_idx],
            ));
        } else if let Some(a_idx) = range_index(&row_layout.w_a, row_idx)
            .or_else(|| {
                range_index(&row_layout.original_a, row_idx).map(|idx| row_layout.w_a.len() + idx)
            })
            .or_else(|| {
                range_index(&row_layout.meta_a, row_idx)
                    .map(|idx| row_layout.w_a.len() + row_layout.original_a.len() + idx)
            })
        {
            let t_row = Instant::now();
            let _span = tracing::info_span!("A_row").entered();
            let a_base_idx = a_idx % n_a;
            let a_slice_idx = a_idx / n_a;

            let mut quotient = cfg_fold_reduce!(
                0..group_layouts.len(),
                || vec![F::zero(); D],
                |mut acc: Vec<F>, layout_idx: usize| {
                    let layout = &group_layouts[layout_idx];
                    let include = if has_tiered_group {
                        if group_layouts.len() >= 3 {
                            (a_slice_idx == 0 && layout.group_idx == 0)
                                || (a_slice_idx == 1
                                    && layout.group_idx != 0
                                    && layout.group_idx + 1 != group_layouts.len())
                                || (a_slice_idx == 2 && layout.group_idx + 1 == group_layouts.len())
                        } else {
                            (a_slice_idx == 0 && layout.group_idx + 1 != group_layouts.len())
                                || (a_slice_idx == 1 && layout.group_idx + 1 == group_layouts.len())
                        }
                    } else {
                        true
                    };
                    if include {
                        let group_blocks = layout.claim_count * layout.spec.num_blocks;
                        let open_gadget = akita_types::gadget_row_scalars::<F>(
                            layout.spec.num_digits_open,
                            lp.log_basis,
                        );
                        let planes_per_block = n_a * layout.spec.num_digits_open;
                        for local_blk in 0..group_blocks {
                            let i = layout.block_start + local_blk;
                            for (digit_idx, &gadget) in open_gadget.iter().enumerate() {
                                let plane_idx = layout.t_hat_start
                                    + local_blk * planes_per_block
                                    + a_base_idx * layout.spec.num_digits_open
                                    + digit_idx;
                                if let Some(plane) = t_hat.flat_digits().get(plane_idx) {
                                    let ring = CyclotomicRing::from_coefficients(
                                        std::array::from_fn(|coeff_idx| {
                                            gadget * F::from_i64(plane[coeff_idx] as i64)
                                        }),
                                    );
                                    challenge_products.add_high_half::<F>(&mut acc, i, &ring);
                                }
                            }
                        }
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

            let a_q = if lp.groups_are_homogeneous() || has_tiered_group {
                *a_quotients[a_idx].coefficients()
            } else {
                let mut acc = [F::zero(); D];
                for layout in &group_layouts {
                    let include = if has_tiered_group {
                        (a_slice_idx == 0 && layout.group_idx + 1 != group_layouts.len())
                            || (a_slice_idx == 1 && layout.group_idx + 1 == group_layouts.len())
                    } else {
                        true
                    };
                    if !include {
                        continue;
                    }
                    let spec = &layout.spec;
                    let inner_width_g = spec.block_len * spec.num_digits_commit;
                    let fold_gadget =
                        akita_types::gadget_row_scalars::<F>(spec.num_digits_fold, lp.log_basis);
                    for point_idx in 0..num_public_outputs {
                        let slot_start = layout.z_base_start + point_idx * inner_width_g;
                        let Some(z_slice) =
                            z_pre_centered.get(slot_start..slot_start + inner_width_g)
                        else {
                            continue;
                        };
                        let z_hat: Vec<CyclotomicRing<F, D>> = z_slice
                            .iter()
                            .map(|centered| {
                                let mut digits = vec![[0i8; D]; spec.num_digits_fold];
                                decompose_centered_coeffs_i8(centered, &mut digits, lp.log_basis);
                                CyclotomicRing::from_coefficients(std::array::from_fn(
                                    |coeff_idx| {
                                        digits.iter().zip(fold_gadget.iter()).fold(
                                            F::zero(),
                                            |sum, (plane, &gadget)| {
                                                sum + gadget * F::from_i64(plane[coeff_idx] as i64)
                                            },
                                        )
                                    },
                                ))
                            })
                            .collect();
                        let group_a_q = unreduced_quotient_rows_ntt_cached::<F, D>(
                            ntt_shared, n_a, stride, &z_hat,
                        );
                        for (dst, src) in acc
                            .iter_mut()
                            .zip(group_a_q[a_base_idx].coefficients().iter())
                        {
                            *dst += *src;
                        }
                    }
                }
                acc
            };
            for k in 0..D {
                quotient[k] -= a_q[k];
            }
            result.push(CyclotomicRing::from_slice(&quotient));
            other_time += t_row.elapsed().as_secs_f64();
        } else {
            result.push(CyclotomicRing::<F, D>::zero());
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

/// Build the RHS vector `y` using an explicit M-row layout.
///
/// # Errors
///
/// Returns an error if the supplied `v`, `commitment_rows`, or
/// `public_outputs` slices do not match the row counts implied by the
/// explicit layout (sum of W/original/meta sub-ranges per family).
pub fn generate_y_for_layout<F, const D: usize>(
    v: &[CyclotomicRing<F, D>],
    commitment_rows: &[CyclotomicRing<F, D>],
    public_outputs: &[CyclotomicRing<F, D>],
    layout: &MRowLayout,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore,
{
    let d_len = layout.w_d.len() + layout.original_d.len() + layout.meta_d.len();
    let b_len = layout.w_b.len() + layout.original_b.len() + layout.meta_b.len();
    if v.len() != d_len {
        return Err(AkitaError::InvalidSize {
            expected: d_len,
            actual: v.len(),
        });
    }
    if commitment_rows.len() != b_len {
        return Err(AkitaError::InvalidSize {
            expected: b_len,
            actual: commitment_rows.len(),
        });
    }
    if public_outputs.len()
        != layout.w_eval.len() + layout.original_eval.len() + layout.meta_eval.len()
    {
        return Err(AkitaError::InvalidSize {
            expected: layout.w_eval.len() + layout.original_eval.len() + layout.meta_eval.len(),
            actual: public_outputs.len(),
        });
    }
    let mut out = vec![CyclotomicRing::<F, D>::zero(); layout.rows];

    let mut y_idx = 0usize;
    for row_idx in layout.w_eval.clone() {
        out[row_idx] = public_outputs[y_idx];
        y_idx += 1;
    }
    for row_idx in layout.original_eval.clone() {
        out[row_idx] = public_outputs[y_idx];
        y_idx += 1;
    }
    for row_idx in layout.meta_eval.clone() {
        out[row_idx] = public_outputs[y_idx];
        y_idx += 1;
    }

    let mut v_idx = 0usize;
    for row_idx in layout.w_d.clone() {
        out[row_idx] = v[v_idx];
        v_idx += 1;
    }
    for row_idx in layout.original_d.clone() {
        out[row_idx] = v[v_idx];
        v_idx += 1;
    }
    for row_idx in layout.meta_d.clone() {
        out[row_idx] = v[v_idx];
        v_idx += 1;
    }

    let mut u_idx = 0usize;
    for row_idx in layout.w_b.clone() {
        out[row_idx] = commitment_rows[u_idx];
        u_idx += 1;
    }
    for row_idx in layout.original_b.clone() {
        out[row_idx] = commitment_rows[u_idx];
        u_idx += 1;
    }
    for row_idx in layout.meta_b.clone() {
        out[row_idx] = commitment_rows[u_idx];
        u_idx += 1;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::ring_switch::{build_w_coeffs, build_w_evals_compact, compute_m_evals_x};
    use crate::AkitaProverSetup;
    use akita_algebra::ring::{eval::scalar_powers, eval_ring_at};
    use akita_challenges::{IntegerChallenge, SparseChallengeConfig, Stage1ChallengeShape};
    use akita_field::Prime128OffsetA7F7;
    use akita_transcript::Blake2bTranscript;
    use akita_types::{
        r_decomp_levels, AjtaiKeyParams, GroupSpec, RingOpeningPoint, TieredSetupParams,
    };
    use std::array::from_fn;

    type TestField = Prime128OffsetA7F7;
    const D_TEST: usize = 32;

    fn outer_level_params() -> LevelParams {
        LevelParams {
            ring_dimension: D_TEST,
            log_basis: 2,
            a_key: AjtaiKeyParams::new_unchecked(1, 8, 0, D_TEST),
            b_key: AjtaiKeyParams::new_unchecked(2, 4, 0, D_TEST),
            d_key: AjtaiKeyParams::new_unchecked(1, 2, 0, D_TEST),
            num_blocks: 4,
            block_len: 2,
            m_vars: 1,
            r_vars: 2,
            stage1_config: SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![-1, 1],
            },
            stage1_challenge_shape: Stage1ChallengeShape::Flat,
            use_setup_claim_reduction: false,
            num_digits_commit: 1,
            num_digits_open: 1,
            num_digits_fold: 1,
            groups: None,
        }
    }

    fn ring_from_i8(coeffs: &[i8; D_TEST]) -> CyclotomicRing<TestField, D_TEST> {
        CyclotomicRing::from_coefficients(from_fn(|i| TestField::from_i64(coeffs[i] as i64)))
    }

    fn negacyclic_sparse_i32_mul_acc(
        digit_plane: &[i8; D_TEST],
        challenge: &IntegerChallenge,
        acc: &mut [CenteredCoeff; D_TEST],
    ) {
        for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
            let p = pos as usize;
            let split = D_TEST - p;
            let coeff = CenteredCoeff::from(coeff);
            for i in 0..split {
                acc[i + p] += coeff * CenteredCoeff::from(digit_plane[i]);
            }
            for i in split..D_TEST {
                acc[i - split] -= coeff * CenteredCoeff::from(digit_plane[i]);
            }
        }
    }

    fn cyclic_mul_accumulate_i8(
        lhs: &CyclotomicRing<TestField, D_TEST>,
        rhs: &[i8; D_TEST],
        dst: &mut CyclotomicRing<TestField, D_TEST>,
    ) {
        let lhs_coeffs = lhs.coefficients();
        let mut out = *dst.coefficients();
        for (i, &a) in lhs_coeffs.iter().enumerate() {
            if a.is_zero() {
                continue;
            }
            for (j, &b) in rhs.iter().enumerate() {
                if b == 0 {
                    continue;
                }
                out[(i + j) % D_TEST] += a * TestField::from_i64(b as i64);
            }
        }
        *dst = CyclotomicRing::from_coefficients(out);
    }

    #[test]
    fn tiered_compute_v_matches_direct_ring_matvec() {
        let setup = AkitaProverSetup::<TestField, D_TEST>::generate_with_capacity(8, 4, 3, 4, 64)
            .expect("test setup");
        let outer = outer_level_params();
        let tier = TieredSetupParams::new(2).expect("f=2 tier");
        let chunk_spec = GroupSpec {
            m_vars: outer.m_vars.saturating_sub(1),
            r_vars: outer.r_vars.saturating_sub(1),
            num_blocks: outer.num_blocks / 2,
            block_len: outer.block_len / 2,
            b_key: outer.b_key.clone(),
            num_digits_commit: outer.num_digits_commit,
            num_digits_open: outer.num_digits_open,
            num_digits_fold: outer.num_digits_fold,
            tier: Some(tier),
        };
        let tiered_lp = LevelParams {
            groups: Some(vec![
                GroupSpec::from_outer(&outer),
                chunk_spec,
                GroupSpec::from_outer(&outer),
            ]),
            ..outer
        };
        let claim_group_sizes = [1usize, tier.num_chunks, 1usize];
        let num_eval_rows = claim_group_sizes.len();
        let layouts = tiered_lp
            .group_layouts(&claim_group_sizes, num_eval_rows)
            .expect("tiered group layout");
        let w_hat_len = layouts
            .last()
            .map(|layout| {
                layout.w_hat_start
                    + layout.claim_count * layout.spec.num_blocks * layout.spec.num_digits_open
            })
            .unwrap_or(0);
        let w_hat: Vec<[i8; D_TEST]> = (0..w_hat_len)
            .map(|plane_idx| {
                from_fn(|coeff_idx| {
                    let raw = ((plane_idx * 17 + coeff_idx * 5 + 3) % 5) as i8;
                    raw - 2
                })
            })
            .collect();

        let actual = compute_v_tier_aware::<TestField, D_TEST>(
            &setup.ntt_shared,
            tiered_lp.d_key.row_len(),
            setup.expanded.seed.max_stride,
            &w_hat,
            &layouts,
            &tiered_lp,
        );

        let d_view = setup
            .expanded
            .shared_matrix
            .ring_view::<D_TEST>(tiered_lp.d_key.row_len(), setup.expanded.seed.max_stride);
        let mut expected = Vec::new();
        for layout in &layouts {
            let per_claim_w_hat_len = layout.spec.num_blocks * layout.spec.num_digits_open;
            let is_tiered = layout.spec.tier.is_some_and(|tier| tier.is_tiered());
            let emit_expected_rows = |cells: &[[i8; D_TEST]], expected: &mut Vec<_>| {
                let mut rows =
                    vec![CyclotomicRing::<TestField, D_TEST>::zero(); tiered_lp.d_key.row_len()];
                #[allow(clippy::needless_range_loop)]
                for row_idx in 0..tiered_lp.d_key.row_len() {
                    for (col_idx, cell) in cells.iter().enumerate() {
                        d_view.row(row_idx)[col_idx]
                            .mul_accumulate_into(&ring_from_i8(cell), &mut rows[row_idx]);
                    }
                }
                expected.extend(rows);
            };
            if is_tiered {
                for claim_local in 0..layout.claim_count {
                    let start = layout.w_hat_start + claim_local * per_claim_w_hat_len;
                    emit_expected_rows(&w_hat[start..start + per_claim_w_hat_len], &mut expected);
                }
            } else {
                let mut summed = vec![[0i8; D_TEST]; per_claim_w_hat_len];
                for claim_local in 0..layout.claim_count {
                    let start = layout.w_hat_start + claim_local * per_claim_w_hat_len;
                    for (src, dst) in w_hat[start..start + per_claim_w_hat_len]
                        .iter()
                        .zip(summed.iter_mut())
                    {
                        for (s, d) in src.iter().zip(dst.iter_mut()) {
                            *d += *s;
                        }
                    }
                }
                emit_expected_rows(&summed, &mut expected);
            }
        }

        assert_eq!(
            actual, expected,
            "tier-aware D-row RHS must match direct coefficient-level ring matvec"
        );
    }

    #[test]
    fn tiered_cyclic_d_rows_match_direct_ring_matvec() {
        let setup = AkitaProverSetup::<TestField, D_TEST>::generate_with_capacity(8, 4, 3, 4, 64)
            .expect("test setup");
        let outer = outer_level_params();
        let tier = TieredSetupParams::new(2).expect("f=2 tier");
        let chunk_spec = GroupSpec {
            m_vars: outer.m_vars.saturating_sub(1),
            r_vars: outer.r_vars.saturating_sub(1),
            num_blocks: outer.num_blocks / 2,
            block_len: outer.block_len / 2,
            b_key: outer.b_key.clone(),
            num_digits_commit: outer.num_digits_commit,
            num_digits_open: outer.num_digits_open,
            num_digits_fold: outer.num_digits_fold,
            tier: Some(tier),
        };
        let tiered_lp = LevelParams {
            groups: Some(vec![
                GroupSpec::from_outer(&outer),
                chunk_spec,
                GroupSpec::from_outer(&outer),
            ]),
            ..outer
        };
        let claim_group_sizes = [1usize, tier.num_chunks, 1usize];
        let num_eval_rows = claim_group_sizes.len();
        let layouts = tiered_lp
            .group_layouts(&claim_group_sizes, num_eval_rows)
            .expect("tiered group layout");
        let w_hat_len = layouts
            .last()
            .map(|layout| {
                layout.w_hat_start
                    + layout.claim_count * layout.spec.num_blocks * layout.spec.num_digits_open
            })
            .unwrap_or(0);
        let w_hat: Vec<[i8; D_TEST]> = (0..w_hat_len)
            .map(|plane_idx| {
                from_fn(|coeff_idx| {
                    let raw = ((plane_idx * 19 + coeff_idx * 7 + 1) % 5) as i8;
                    raw - 2
                })
            })
            .collect();

        let d_view = setup
            .expanded
            .shared_matrix
            .ring_view::<D_TEST>(tiered_lp.d_key.row_len(), setup.expanded.seed.max_stride);
        let mut actual = Vec::new();
        let mut expected = Vec::new();
        for layout in &layouts {
            let per_claim_w_hat_len = layout.spec.num_blocks * layout.spec.num_digits_open;
            let is_tiered = layout.spec.tier.is_some_and(|tier| tier.is_tiered());
            let emit_expected_rows = |cells: &[[i8; D_TEST]], expected: &mut Vec<_>| {
                let mut rows =
                    vec![CyclotomicRing::<TestField, D_TEST>::zero(); tiered_lp.d_key.row_len()];
                #[allow(clippy::needless_range_loop)]
                for row_idx in 0..tiered_lp.d_key.row_len() {
                    for (col_idx, cell) in cells.iter().enumerate() {
                        cyclic_mul_accumulate_i8(
                            &d_view.row(row_idx)[col_idx],
                            cell,
                            &mut rows[row_idx],
                        );
                    }
                }
                expected.extend(rows);
            };
            let emit_actual_rows = |cells: &[[i8; D_TEST]], actual: &mut Vec<_>| {
                actual.extend(mat_vec_mul_ntt_single_i8_cyclic::<TestField, D_TEST>(
                    &setup.ntt_shared,
                    tiered_lp.d_key.row_len(),
                    setup.expanded.seed.max_stride,
                    cells,
                ));
            };
            if is_tiered {
                for claim_local in 0..layout.claim_count {
                    let start = layout.w_hat_start + claim_local * per_claim_w_hat_len;
                    let cells = &w_hat[start..start + per_claim_w_hat_len];
                    emit_actual_rows(cells, &mut actual);
                    emit_expected_rows(cells, &mut expected);
                }
            } else {
                let mut summed = vec![[0i8; D_TEST]; per_claim_w_hat_len];
                for claim_local in 0..layout.claim_count {
                    let start = layout.w_hat_start + claim_local * per_claim_w_hat_len;
                    for (src, dst) in w_hat[start..start + per_claim_w_hat_len]
                        .iter()
                        .zip(summed.iter_mut())
                    {
                        for (s, d) in src.iter().zip(dst.iter_mut()) {
                            *d += *s;
                        }
                    }
                }
                emit_actual_rows(&summed, &mut actual);
                emit_expected_rows(&summed, &mut expected);
            }
        }

        assert_eq!(
            actual, expected,
            "tier-aware cyclic D-row matvec must match direct coefficient-level cyclic product"
        );
    }

    #[test]
    fn tiered_grouped_m_rows_match_committed_witness_locally() {
        let setup = AkitaProverSetup::<TestField, D_TEST>::generate_with_capacity(8, 4, 3, 4, 64)
            .expect("test setup");
        let mut outer = outer_level_params();
        outer.num_digits_open = 8;
        outer.num_digits_fold = 8;
        let tier = TieredSetupParams::new(2).expect("f=2 tier");
        let chunk_spec = GroupSpec {
            m_vars: outer.m_vars.saturating_sub(1),
            r_vars: outer.r_vars.saturating_sub(1),
            num_blocks: outer.num_blocks / 2,
            block_len: outer.block_len / 2,
            b_key: outer.b_key.clone(),
            num_digits_commit: outer.num_digits_commit,
            num_digits_open: outer.num_digits_open,
            num_digits_fold: outer.num_digits_fold,
            tier: Some(tier),
        };
        let tiered_lp = LevelParams {
            groups: Some(vec![
                GroupSpec::from_outer(&outer),
                chunk_spec,
                GroupSpec::from_outer(&outer),
            ]),
            ..outer
        };
        let claim_group_sizes = [1usize, tier.num_chunks, 1usize];
        let num_eval_rows = claim_group_sizes.len();
        let layouts = tiered_lp
            .group_layouts(&claim_group_sizes, num_eval_rows)
            .expect("tiered group layout");
        let total_group_blocks = layouts
            .last()
            .map(|layout| layout.block_start + layout.claim_count * layout.spec.num_blocks)
            .unwrap_or(0);
        let w_hat_len = layouts
            .last()
            .map(|layout| {
                layout.w_hat_start
                    + layout.claim_count * layout.spec.num_blocks * layout.spec.num_digits_open
            })
            .unwrap_or(0);
        let t_hat_len = layouts
            .last()
            .map(|layout| {
                layout.t_hat_start
                    + layout.claim_count
                        * layout.spec.num_blocks
                        * tiered_lp.a_key.row_len()
                        * layout.spec.num_digits_open
            })
            .unwrap_or(0);
        let z_len = layouts
            .last()
            .map(|layout| {
                layout.z_base_start
                    + num_eval_rows * layout.spec.block_len * layout.spec.num_digits_commit
            })
            .unwrap_or(0);

        let w_hat: Vec<[i8; D_TEST]> = (0..w_hat_len)
            .map(|plane_idx| {
                from_fn(|coeff_idx| {
                    let raw = ((plane_idx * 11 + coeff_idx * 3 + 5) % 5) as i8;
                    raw - 2
                })
            })
            .collect();
        let mut transcript = Blake2bTranscript::<TestField>::new(b"tiered-row-local");
        let challenges = sample_stage1_challenges::<TestField, _, D_TEST>(
            &mut transcript,
            total_group_blocks,
            1,
            &tiered_lp.stage1_config,
            &tiered_lp.stage1_challenge_shape,
        )
        .expect("stage1 challenges");
        let integer_challenges = challenges
            .expand_integer::<D_TEST>()
            .expect("integer stage1 challenges");

        let mut t_hat_flat = Vec::with_capacity(t_hat_len);
        let mut t_block_sizes = Vec::with_capacity(total_group_blocks);
        let mut t_rows = Vec::with_capacity(total_group_blocks);
        let mut z_pre_centered = vec![[0 as CenteredCoeff; D_TEST]; z_len];
        let q = (-TestField::one()).to_canonical_u128() + 1;
        let open_params =
            BalancedDecomposePow2I8Params::new(tiered_lp.num_digits_open, tiered_lp.log_basis, q);
        for layout in &layouts {
            let spec = &layout.spec;
            let inner_width_g = spec.block_len * spec.num_digits_commit;
            let group_blocks = layout.claim_count * spec.num_blocks;
            for local_blk in 0..group_blocks {
                let claim_within = local_blk / spec.num_blocks;
                let claim_idx = layout.claim_start + claim_within;
                let point_idx = layout.group_idx;
                let challenge_idx = layout.block_start + local_blk;
                let block_digits: Vec<[i8; D_TEST]> =
                    (0..inner_width_g).map(|_| [0i8; D_TEST]).collect();
                let z_slot = layout.z_base_start + point_idx * inner_width_g;
                for elem_idx in 0..spec.block_len {
                    for dc in 0..spec.num_digits_commit {
                        let plane_idx = elem_idx * spec.num_digits_commit + dc;
                        negacyclic_sparse_i32_mul_acc(
                            &block_digits[plane_idx],
                            &integer_challenges[challenge_idx],
                            &mut z_pre_centered[z_slot + plane_idx],
                        );
                    }
                }
                let rows = mat_vec_mul_ntt_single_i8::<TestField, D_TEST>(
                    &setup.ntt_shared,
                    tiered_lp.a_key.row_len(),
                    setup.expanded.seed.max_stride,
                    &block_digits,
                );
                let mut block_t_hat =
                    Vec::with_capacity(tiered_lp.a_key.row_len() * spec.num_digits_open);
                for row in &rows {
                    let mut digits = vec![[0i8; D_TEST]; spec.num_digits_open];
                    row.balanced_decompose_pow2_i8_into_with_params(&mut digits, &open_params);
                    block_t_hat.extend(digits);
                }
                let _ = claim_idx;
                t_block_sizes.push(block_t_hat.len());
                t_hat_flat.extend(block_t_hat);
                t_rows.push(rows);
            }
        }
        let t_hat = FlatDigitBlocks::new(t_hat_flat, t_block_sizes).expect("t-hat blocks");
        let z_pre_centered_inf_norm = z_pre_centered
            .iter()
            .flat_map(|coeffs| coeffs.iter())
            .map(|coeff| coeff.unsigned_abs())
            .max()
            .unwrap_or(0);

        let v = compute_v_tier_aware::<TestField, D_TEST>(
            &setup.ntt_shared,
            tiered_lp.d_key.row_len(),
            setup.expanded.seed.max_stride,
            &w_hat,
            &layouts,
            &tiered_lp,
        );
        let mut commitment_rows = Vec::new();
        for layout in &layouts {
            let per_claim_len =
                layout.spec.num_blocks * tiered_lp.a_key.row_len() * layout.spec.num_digits_open;
            for claim_local in 0..layout.claim_count {
                let start = layout.t_hat_start + claim_local * per_claim_len;
                let end = start + per_claim_len;
                commitment_rows.extend(mat_vec_mul_ntt_single_i8::<TestField, D_TEST>(
                    &setup.ntt_shared,
                    layout.spec.b_key.row_len(),
                    setup.expanded.seed.max_stride,
                    &t_hat.flat_digits()[start..end],
                ));
            }
        }

        let public_outputs: Vec<CyclotomicRing<TestField, D_TEST>> = (0..num_eval_rows)
            .map(|row_idx| {
                CyclotomicRing::from_coefficients(from_fn(|coeff_idx| {
                    TestField::from_u64(19 + (row_idx * 23 + coeff_idx) as u64)
                }))
            })
            .collect();
        let y = generate_y_for_layout::<TestField, D_TEST>(
            &v,
            &commitment_rows,
            &public_outputs,
            &tiered_lp.m_row_layout(claim_group_sizes.len(), num_eval_rows),
        )
        .expect("y rows");

        let alpha = TestField::from_u64(7);
        let alpha_pows = scalar_powers(alpha, D_TEST);
        let opening_points: Vec<RingOpeningPoint<TestField>> = (0..num_eval_rows)
            .map(|point_idx| RingOpeningPoint {
                a: (0..tiered_lp.block_len)
                    .map(|i| TestField::from_u64(3 + (point_idx * 5 + i) as u64))
                    .collect(),
                b: (0..tiered_lp.num_blocks)
                    .map(|i| TestField::from_u64(29 + (point_idx * 7 + i) as u64))
                    .collect(),
            })
            .collect();
        let mut claim_to_point = Vec::new();
        for (group_idx, &group_size) in claim_group_sizes.iter().enumerate() {
            claim_to_point.extend(repeat_n(group_idx, group_size));
        }
        let gamma: Vec<TestField> = (0..claim_to_point.len())
            .map(|i| TestField::from_u64(41 + i as u64))
            .collect();
        let w_folded = vec![CyclotomicRing::<TestField, D_TEST>::zero(); total_group_blocks];

        let r = compute_r_split_eq::<TestField, D_TEST>(
            &tiered_lp,
            &setup.expanded,
            &challenges,
            &w_hat,
            &t_hat,
            &t_rows,
            &w_folded,
            &z_pre_centered,
            z_pre_centered_inf_norm,
            &y,
            &claim_group_sizes,
            num_eval_rows,
            tiered_lp.num_blocks,
            tiered_lp.inner_width(),
            setup.expanded.seed.max_stride,
            &setup.ntt_shared,
        )
        .expect("r split equation");
        let w_hat_blocks = FlatDigitBlocks::new(
            w_hat.clone(),
            layouts
                .iter()
                .flat_map(|layout| {
                    repeat_n(
                        layout.spec.num_digits_open,
                        layout.claim_count * layout.spec.num_blocks,
                    )
                })
                .collect(),
        )
        .expect("w-hat blocks");
        let w = build_w_coeffs::<TestField, D_TEST>(
            &w_hat_blocks,
            &t_hat,
            &z_pre_centered,
            &r,
            &tiered_lp,
            &claim_group_sizes,
            num_eval_rows,
        );
        let (w_compact, _, _) =
            build_w_evals_compact(w.as_i8_digits(), D_TEST).expect("compact witness");
        let live_x_cols = w_compact.len() / D_TEST;
        let levels = r_decomp_levels::<TestField>(tiered_lp.log_basis);
        let rows = tiered_lp.m_row_count(claim_group_sizes.len(), num_eval_rows);
        let layout = tiered_lp.m_row_layout(claim_group_sizes.len(), num_eval_rows);
        let r_tail_start = live_x_cols - rows * levels;
        let denom = alpha_pows[D_TEST - 1] * alpha + TestField::one();

        let row_dot = |row_idx: usize| -> (TestField, TestField) {
            let tau1: Vec<TestField> = (0..rows.next_power_of_two().trailing_zeros() as usize)
                .map(|bit| {
                    if (row_idx >> bit) & 1 == 1 {
                        TestField::one()
                    } else {
                        TestField::zero()
                    }
                })
                .collect();
            let row_m = compute_m_evals_x::<TestField, D_TEST>(
                &setup.expanded,
                &opening_points,
                &claim_to_point,
                &challenges,
                alpha,
                &alpha_pows,
                &tiered_lp,
                &tau1,
                &claim_group_sizes,
                &gamma,
                num_eval_rows,
            )
            .expect("row M evals");
            let mut total = TestField::zero();
            let mut r_tail = TestField::zero();
            for x in 0..live_x_cols {
                let m = row_m[x];
                let w_row = &w_compact[x * D_TEST..(x + 1) * D_TEST];
                for (coeff_idx, &alpha_y) in alpha_pows.iter().enumerate() {
                    let term = TestField::from_i64(w_row[coeff_idx] as i64) * alpha_y * m;
                    total += term;
                    if x >= r_tail_start {
                        r_tail += term;
                    }
                }
            }
            (total, r_tail)
        };

        let d_row = layout.w_d.start;
        let b_row = layout.w_b.start;
        let original_a_row = layout.original_a.start;
        let meta_a_row = layout.meta_a.start;
        for (row_idx, expected) in [
            (d_row, eval_ring_at(&v[0], &alpha)),
            (b_row, eval_ring_at(&commitment_rows[0], &alpha)),
            (original_a_row, TestField::zero()),
            (meta_a_row, TestField::zero()),
        ] {
            let (actual, r_tail) = row_dot(row_idx);
            assert_eq!(
                r_tail,
                -denom * eval_ring_at(&r[row_idx], &alpha),
                "r-tail contribution must recompose row {row_idx}'s quotient"
            );
            assert_eq!(
                actual, expected,
                "tiered grouped M row {row_idx} must match its row RHS"
            );
        }
    }

    /// Production-shaped row-local invariant covering every D/B row of
    /// the book §5.4 `[W, chunks(k=4), meta]` layout with `n_a = n_b = n_d
    /// = 2`. The matrix relation rows (`A · z_pre = c` book §5.4 lines
    /// 728-729 / 747-749) are exercised end-to-end by the tiered E2E
    /// tests; replicating their consistency in a unit test would require
    /// constructing valid commitment material that satisfies the
    /// decomposition bound on `t = A·b`, which only holds for the
    /// bounded-norm witnesses produced by the prover.
    #[test]
    fn tiered_grouped_m_rows_match_committed_witness_multi_a() {
        let setup = AkitaProverSetup::<TestField, D_TEST>::generate_with_capacity(8, 4, 3, 4, 64)
            .expect("test setup");
        let mut outer = outer_level_params();
        outer.num_digits_open = 8;
        outer.num_digits_fold = 8;
        // Multi-row Ajtai / B / D widens the per-row identity to expose
        // any `a_base_idx > 0` mismatch that the single-row test masks.
        outer.a_key = AjtaiKeyParams::new_unchecked(2, outer.a_key.col_len(), 0, D_TEST);
        outer.b_key = AjtaiKeyParams::new_unchecked(2, outer.b_key.col_len(), 0, D_TEST);
        outer.d_key = AjtaiKeyParams::new_unchecked(2, outer.d_key.col_len(), 0, D_TEST);
        let tier = TieredSetupParams::new(2).expect("f=2 tier");
        let chunk_spec = GroupSpec {
            m_vars: outer.m_vars.saturating_sub(1),
            r_vars: outer.r_vars.saturating_sub(1),
            num_blocks: outer.num_blocks / 2,
            block_len: outer.block_len / 2,
            b_key: outer.b_key.clone(),
            num_digits_commit: outer.num_digits_commit,
            num_digits_open: outer.num_digits_open,
            num_digits_fold: outer.num_digits_fold,
            tier: Some(tier),
        };
        let tiered_lp = LevelParams {
            groups: Some(vec![
                GroupSpec::from_outer(&outer),
                chunk_spec,
                GroupSpec::from_outer(&outer),
            ]),
            ..outer
        };
        let claim_group_sizes = [1usize, tier.num_chunks, 1usize];
        let num_eval_rows = claim_group_sizes.len();
        let layouts = tiered_lp
            .group_layouts(&claim_group_sizes, num_eval_rows)
            .expect("tiered group layout");
        let total_group_blocks = layouts
            .last()
            .map(|layout| layout.block_start + layout.claim_count * layout.spec.num_blocks)
            .unwrap_or(0);
        let w_hat_len = layouts
            .last()
            .map(|layout| {
                layout.w_hat_start
                    + layout.claim_count * layout.spec.num_blocks * layout.spec.num_digits_open
            })
            .unwrap_or(0);
        let t_hat_len = layouts
            .last()
            .map(|layout| {
                layout.t_hat_start
                    + layout.claim_count
                        * layout.spec.num_blocks
                        * tiered_lp.a_key.row_len()
                        * layout.spec.num_digits_open
            })
            .unwrap_or(0);
        let z_len = layouts
            .last()
            .map(|layout| {
                layout.z_base_start
                    + num_eval_rows * layout.spec.block_len * layout.spec.num_digits_commit
            })
            .unwrap_or(0);

        let w_hat: Vec<[i8; D_TEST]> = (0..w_hat_len)
            .map(|plane_idx| {
                from_fn(|coeff_idx| {
                    let raw = ((plane_idx * 11 + coeff_idx * 3 + 5) % 5) as i8;
                    raw - 2
                })
            })
            .collect();
        let mut transcript = Blake2bTranscript::<TestField>::new(b"tiered-row-local-multi-a");
        let challenges = sample_stage1_challenges::<TestField, _, D_TEST>(
            &mut transcript,
            total_group_blocks,
            1,
            &tiered_lp.stage1_config,
            &tiered_lp.stage1_challenge_shape,
        )
        .expect("stage1 challenges");
        let integer_challenges = challenges
            .expand_integer::<D_TEST>()
            .expect("integer stage1 challenges");

        let mut t_hat_flat = Vec::with_capacity(t_hat_len);
        let mut t_block_sizes = Vec::with_capacity(total_group_blocks);
        let mut t_rows = Vec::with_capacity(total_group_blocks);
        let mut z_pre_centered = vec![[0 as CenteredCoeff; D_TEST]; z_len];
        let q = (-TestField::one()).to_canonical_u128() + 1;
        let open_params =
            BalancedDecomposePow2I8Params::new(tiered_lp.num_digits_open, tiered_lp.log_basis, q);
        for layout in &layouts {
            let spec = &layout.spec;
            let inner_width_g = spec.block_len * spec.num_digits_commit;
            let group_blocks = layout.claim_count * spec.num_blocks;
            for local_blk in 0..group_blocks {
                let claim_within = local_blk / spec.num_blocks;
                let claim_idx = layout.claim_start + claim_within;
                let point_idx = layout.group_idx;
                let challenge_idx = layout.block_start + local_blk;
                // Non-trivial block digits so the A-row Z-quotient picks
                // up coefficients distinct from the t-row contribution.
                let block_digits: Vec<[i8; D_TEST]> = (0..inner_width_g)
                    .map(|elem_idx| {
                        from_fn(|coeff_idx| {
                            let raw = ((local_blk * 13
                                + elem_idx * 7
                                + coeff_idx * 3
                                + layout.group_idx * 5
                                + 1) as i64)
                                .rem_euclid(5) as i8;
                            raw - 2
                        })
                    })
                    .collect();
                let z_slot = layout.z_base_start + point_idx * inner_width_g;
                for elem_idx in 0..spec.block_len {
                    for dc in 0..spec.num_digits_commit {
                        let plane_idx = elem_idx * spec.num_digits_commit + dc;
                        negacyclic_sparse_i32_mul_acc(
                            &block_digits[plane_idx],
                            &integer_challenges[challenge_idx],
                            &mut z_pre_centered[z_slot + plane_idx],
                        );
                    }
                }
                let rows = mat_vec_mul_ntt_single_i8::<TestField, D_TEST>(
                    &setup.ntt_shared,
                    tiered_lp.a_key.row_len(),
                    setup.expanded.seed.max_stride,
                    &block_digits,
                );
                let mut block_t_hat =
                    Vec::with_capacity(tiered_lp.a_key.row_len() * spec.num_digits_open);
                for row in &rows {
                    let mut digits = vec![[0i8; D_TEST]; spec.num_digits_open];
                    row.balanced_decompose_pow2_i8_into_with_params(&mut digits, &open_params);
                    block_t_hat.extend(digits);
                }
                let _ = claim_idx;
                t_block_sizes.push(block_t_hat.len());
                t_hat_flat.extend(block_t_hat);
                t_rows.push(rows);
            }
        }
        let t_hat = FlatDigitBlocks::new(t_hat_flat, t_block_sizes).expect("t-hat blocks");
        let z_pre_centered_inf_norm = z_pre_centered
            .iter()
            .flat_map(|coeffs| coeffs.iter())
            .map(|coeff| coeff.unsigned_abs())
            .max()
            .unwrap_or(0);

        let v = compute_v_tier_aware::<TestField, D_TEST>(
            &setup.ntt_shared,
            tiered_lp.d_key.row_len(),
            setup.expanded.seed.max_stride,
            &w_hat,
            &layouts,
            &tiered_lp,
        );
        let mut commitment_rows = Vec::new();
        for layout in &layouts {
            let per_claim_len =
                layout.spec.num_blocks * tiered_lp.a_key.row_len() * layout.spec.num_digits_open;
            for claim_local in 0..layout.claim_count {
                let start = layout.t_hat_start + claim_local * per_claim_len;
                let end = start + per_claim_len;
                commitment_rows.extend(mat_vec_mul_ntt_single_i8::<TestField, D_TEST>(
                    &setup.ntt_shared,
                    layout.spec.b_key.row_len(),
                    setup.expanded.seed.max_stride,
                    &t_hat.flat_digits()[start..end],
                ));
            }
        }

        let public_outputs: Vec<CyclotomicRing<TestField, D_TEST>> = (0..num_eval_rows)
            .map(|row_idx| {
                CyclotomicRing::from_coefficients(from_fn(|coeff_idx| {
                    TestField::from_u64(19 + (row_idx * 23 + coeff_idx) as u64)
                }))
            })
            .collect();
        let layout = tiered_lp.m_row_layout(claim_group_sizes.len(), num_eval_rows);
        let y = generate_y_for_layout::<TestField, D_TEST>(
            &v,
            &commitment_rows,
            &public_outputs,
            &layout,
        )
        .expect("y rows");

        let alpha = TestField::from_u64(7);
        let alpha_pows = scalar_powers(alpha, D_TEST);
        let opening_points: Vec<RingOpeningPoint<TestField>> = (0..num_eval_rows)
            .map(|point_idx| RingOpeningPoint {
                a: (0..tiered_lp.block_len)
                    .map(|i| TestField::from_u64(3 + (point_idx * 5 + i) as u64))
                    .collect(),
                b: (0..tiered_lp.num_blocks)
                    .map(|i| TestField::from_u64(29 + (point_idx * 7 + i) as u64))
                    .collect(),
            })
            .collect();
        let mut claim_to_point = Vec::new();
        for (group_idx, &group_size) in claim_group_sizes.iter().enumerate() {
            claim_to_point.extend(repeat_n(group_idx, group_size));
        }
        let gamma: Vec<TestField> = (0..claim_to_point.len())
            .map(|i| TestField::from_u64(41 + i as u64))
            .collect();
        let w_folded = vec![CyclotomicRing::<TestField, D_TEST>::zero(); total_group_blocks];

        let r = compute_r_split_eq::<TestField, D_TEST>(
            &tiered_lp,
            &setup.expanded,
            &challenges,
            &w_hat,
            &t_hat,
            &t_rows,
            &w_folded,
            &z_pre_centered,
            z_pre_centered_inf_norm,
            &y,
            &claim_group_sizes,
            num_eval_rows,
            tiered_lp.num_blocks,
            tiered_lp.inner_width(),
            setup.expanded.seed.max_stride,
            &setup.ntt_shared,
        )
        .expect("r split equation");
        let w_hat_blocks = FlatDigitBlocks::new(
            w_hat.clone(),
            layouts
                .iter()
                .flat_map(|layout| {
                    repeat_n(
                        layout.spec.num_digits_open,
                        layout.claim_count * layout.spec.num_blocks,
                    )
                })
                .collect(),
        )
        .expect("w-hat blocks");
        let w = build_w_coeffs::<TestField, D_TEST>(
            &w_hat_blocks,
            &t_hat,
            &z_pre_centered,
            &r,
            &tiered_lp,
            &claim_group_sizes,
            num_eval_rows,
        );
        let (w_compact, _, _) =
            build_w_evals_compact(w.as_i8_digits(), D_TEST).expect("compact witness");
        let live_x_cols = w_compact.len() / D_TEST;
        let levels = r_decomp_levels::<TestField>(tiered_lp.log_basis);
        let rows = tiered_lp.m_row_count(claim_group_sizes.len(), num_eval_rows);
        let r_tail_start = live_x_cols - rows * levels;
        let denom = alpha_pows[D_TEST - 1] * alpha + TestField::one();

        let row_dot = |row_idx: usize| -> (TestField, TestField) {
            let tau1: Vec<TestField> = (0..rows.next_power_of_two().trailing_zeros() as usize)
                .map(|bit| {
                    if (row_idx >> bit) & 1 == 1 {
                        TestField::one()
                    } else {
                        TestField::zero()
                    }
                })
                .collect();
            let row_m = compute_m_evals_x::<TestField, D_TEST>(
                &setup.expanded,
                &opening_points,
                &claim_to_point,
                &challenges,
                alpha,
                &alpha_pows,
                &tiered_lp,
                &tau1,
                &claim_group_sizes,
                &gamma,
                num_eval_rows,
            )
            .expect("row M evals");
            let mut total = TestField::zero();
            let mut r_tail = TestField::zero();
            for x in 0..live_x_cols {
                let m = row_m[x];
                let w_row = &w_compact[x * D_TEST..(x + 1) * D_TEST];
                for (coeff_idx, &alpha_y) in alpha_pows.iter().enumerate() {
                    let term = TestField::from_i64(w_row[coeff_idx] as i64) * alpha_y * m;
                    total += term;
                    if x >= r_tail_start {
                        r_tail += term;
                    }
                }
            }
            (total, r_tail)
        };

        // Enumerate every D / B / A row in the 10-group layout. Eval
        // and fold rows depend on the prover's per-claim y_ring values
        // (here we use synthetic `public_outputs` that aren't tied to
        // the random witness), so they're exercised end-to-end by the
        // E2E tests rather than here; the row-local invariant focuses
        // on the matrix relation (book §5.4 lines 717–729 / 735–749).
        let mut row_cases: Vec<(usize, TestField, &'static str)> = Vec::new();
        let mut v_idx = 0usize;
        for r_idx in layout.w_d.clone() {
            row_cases.push((r_idx, eval_ring_at(&v[v_idx], &alpha), "w_d"));
            v_idx += 1;
        }
        for r_idx in layout.original_d.clone() {
            row_cases.push((r_idx, eval_ring_at(&v[v_idx], &alpha), "original_d"));
            v_idx += 1;
        }
        for r_idx in layout.meta_d.clone() {
            row_cases.push((r_idx, eval_ring_at(&v[v_idx], &alpha), "meta_d"));
            v_idx += 1;
        }
        let mut u_idx = 0usize;
        for r_idx in layout.w_b.clone() {
            row_cases.push((r_idx, eval_ring_at(&commitment_rows[u_idx], &alpha), "w_b"));
            u_idx += 1;
        }
        for r_idx in layout.original_b.clone() {
            row_cases.push((
                r_idx,
                eval_ring_at(&commitment_rows[u_idx], &alpha),
                "original_b",
            ));
            u_idx += 1;
        }
        for r_idx in layout.meta_b.clone() {
            row_cases.push((
                r_idx,
                eval_ring_at(&commitment_rows[u_idx], &alpha),
                "meta_b",
            ));
            u_idx += 1;
        }

        for (row_idx, expected, label) in row_cases {
            let (actual, r_tail) = row_dot(row_idx);
            assert_eq!(
                r_tail,
                -denom * eval_ring_at(&r[row_idx], &alpha),
                "r-tail contribution must recompose row {row_idx} ({label}) quotient"
            );
            assert_eq!(
                actual, expected,
                "tiered grouped M row {row_idx} ({label}) must match its RHS"
            );
        }
    }
}
