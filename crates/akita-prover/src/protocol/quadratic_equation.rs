//! Quadratic equation builder for the Akita PCS (§4.2).
//!
//! This module encapsulates the stage-1 prover logic and the generation of
//! the quadratic equation components M, y, z, and v.

use crate::kernels::crt_ntt::NttSlotCache;
use crate::kernels::linear::{
    fused_split_eq_quotients, mat_vec_mul_ntt_single_i8, mat_vec_mul_ntt_single_i8_cyclic,
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
use akita_types::{validate_stage1_accumulator_headroom, AkitaExpandedSetup, LevelParams};
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

        let v = {
            let _span = tracing::info_span!(
                "compute_batched_v",
                w_hat_planes = w_hat.flat_digits().len()
            )
            .entered();
            mat_vec_mul_ntt_single_i8(ntt_d, lp.d_key.row_len(), stride, w_hat.flat_digits())
        };

        transcript.append_serde(ABSORB_PROVER_V, &RingSliceSerializer(&v));

        let group_layouts = lp.group_layouts(claim_group_sizes, opening_points.len())?;
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
            mat_vec_mul_ntt_single_i8(ntt_d, lp.d_key.row_len(), stride, w_hat.flat_digits())
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
                let y = generate_y::<F, D>(
                    &v,
                    &commitment_rows,
                    y_rings,
                    lp.d_key.row_len(),
                    lp.total_b_row_count(claim_group_sizes.len()),
                    lp.a_key.row_len(),
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
        let y = generate_y::<F, D>(
            &v,
            &commitment_rows,
            y_rings,
            lp.d_key.row_len(),
            lp.b_key.row_len(),
            lp.a_key.row_len(),
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
    let commitment_row_count = lp.total_b_row_count(num_commitment_groups);
    let num_rows = lp.m_row_count(num_commitment_groups, num_public_outputs);
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
    // Row layout: consistency (1) | public (num_public_outputs) | D (n_d) |
    //             B (commitment_row_count) | A (n_a)
    let d_start = 1 + num_public_outputs;
    let b_start = d_start + n_d;
    let a_start = b_start + commitment_row_count;

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
        fused_split_eq_quotients::<F, D>(
            ntt_shared,
            n_d,
            0,
            n_a,
            stride,
            w_hat_flat,
            &[],
            z_pre_centered,
            z_pre_centered_inf_norm,
        )
    };
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

    for row_idx in 0..num_rows {
        if row_idx == 0 {
            let t_row = Instant::now();
            let _span = tracing::info_span!("challenge_fold_row").entered();
            let quotient = parallel_high_half_accumulate::<F, D>(&challenge_products, w_folded);
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
                0..t.len(),
                || vec![F::zero(); D],
                |mut acc: Vec<F>, i: usize| {
                    if let Some(t_row_i) = t[i].get(a_idx) {
                        challenge_products.add_high_half::<F>(&mut acc, i, t_row_i);
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
    let mut out =
        Vec::with_capacity(1 + public_outputs.len() + n_d + commitment_rows.len() + n_a);
    out.push(CyclotomicRing::<F, D>::zero());
    out.extend_from_slice(public_outputs);
    out.extend_from_slice(v);
    out.extend_from_slice(commitment_rows);
    out.extend(repeat_n(CyclotomicRing::<F, D>::zero(), n_a));
    Ok(out)
}
