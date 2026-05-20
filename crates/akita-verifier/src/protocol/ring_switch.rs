//! Verifier-side ring-switch replay.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::summarize_pow2_block_carries;
use akita_algebra::ring::scalar_powers;
use akita_challenges::SparseChallenge;
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, MulBase, RandomSampling,
};
use akita_transcript::labels::{
    ABSORB_SUMCHECK_W, CHALLENGE_RING_SWITCH, CHALLENGE_TAU0, CHALLENGE_TAU1,
};
use akita_transcript::{sample_ext_challenge, Transcript};
#[cfg(feature = "zk")]
use akita_types::zk;
use akita_types::{
    embed_ring_subfield_scalar, gadget_row_scalars, r_decomp_levels,
    validate_opening_points_for_claims, AkitaExpandedSetup, FlatRingVec, LevelParams,
    RingMultiplierOpeningPoint, RingOpeningPoint, RingSubfieldEncoding,
};

#[cfg(feature = "zk")]
use super::slice_mle::{compute_b_blinding_part, compute_d_blinding_part};
use super::slice_mle::{
    compute_r_contribution, compute_setup_contribution, StructuredSliceMleEvaluator,
    TStructuredSlicesEvaluator, WStructuredSlicesEvaluator, ZDenseSlicesEvaluator,
    ZStructuredPow2SlicesEvaluator,
};

/// Verifier-side ring-switch output, carrying only the data needed to replay
/// the fused stage-1/stage-2 checks.
pub(crate) struct RingSwitchVerifyOutput<E: FieldCore> {
    /// Prepared data for deferred ring-switch row MLE evaluation.
    pub prepared_row_eval: RingSwitchDeferredRowEval<E>,
    /// Evaluation table of alpha powers over the ring-coordinate dimension.
    pub alpha_evals_y: Vec<E>,
    /// Number of upper variable bits.
    pub col_bits: usize,
    /// Number of lower variable bits.
    pub ring_bits: usize,
    /// Challenge tau0 for the stage-1 sumcheck.
    pub tau0: Vec<E>,
    /// Challenge tau1 for the stage-2 M-row combination.
    pub tau1: Vec<E>,
    /// Basis size `b = 2^log_basis`.
    pub b: usize,
    /// Ring-switch challenge alpha.
    pub alpha: E,
}

/// Precomputed challenge-derived data for deferred ring-switch row MLE evaluation.
///
/// Stores only data that cannot be derived from context at evaluation time:
/// alpha-evaluated folding challenges and the tau1 eq-polynomial expansion.
/// Everything else is passed by reference at evaluation time to avoid
/// duplicating setup matrix views, opening points, and gadget vectors.
pub struct RingSwitchDeferredRowEval<F: FieldCore> {
    pub(crate) c_alphas: Vec<F>,
    pub(crate) eq_tau1: Vec<F>,
    pub(crate) total_blocks: usize,
    pub(crate) num_t_vectors: usize,
    pub(crate) num_blocks: usize,
    pub(crate) num_claims: usize,
    pub(crate) depth_open: usize,
    pub(crate) depth_commit: usize,
    pub(crate) depth_fold: usize,
    #[cfg(feature = "zk")]
    pub(crate) d_blinding_segment_len: usize,
    #[cfg(feature = "zk")]
    pub(crate) b_blinding_digit_planes_per_point: usize,
    #[cfg(feature = "zk")]
    pub(crate) b_blinding_segment_len: usize,
    pub(crate) block_len: usize,
    pub(crate) inner_width: usize,
    pub(crate) log_basis: u32,
    pub(crate) n_a: usize,
    pub(crate) n_d: usize,
    pub(crate) n_b: usize,
    pub(crate) num_points: usize,
    pub(crate) rows: usize,
    pub(crate) z_first: bool,
    pub(crate) claim_to_point_poly: Vec<(usize, usize)>,
    pub(crate) num_polys_per_point: Vec<usize>,
    pub(crate) num_public_rows: usize,
    pub(crate) gamma: Vec<F>,
    pub(crate) claim_to_point: Vec<usize>,
    // Tiered root-commit fields (`specs/tiered_commit.md` §3 + §9).
    // For legacy LevelParams (`split_factor == 1`), `is_tiered` is
    // false and the rest of these fields are zero / empty.
    /// `true` iff `lp.is_tiered_root()` (i.e. `split_factor > 1`).
    pub(crate) is_tiered: bool,
    /// Tiering factor `f`. `1` for legacy.
    pub(crate) split_factor: usize,
    /// Outer gadget log-basis (`2..=6`). `0` for legacy.
    pub(crate) outer_log_basis: u32,
    /// `δ_outer`. `0` for legacy.
    pub(crate) num_digits_outer: usize,
    /// SIS rank of `F`. `0` for legacy.
    pub(crate) n_f: usize,
    /// Per-chunk `B'` width = `lp.b_prime_width()`. Equals
    /// `n_a · depth_open · num_blocks / split_factor` in the
    /// max-group-polys=1 case the planner currently emits.
    pub(crate) b_prime_width: usize,
}

/// Replay the verifier half of ring switching.
///
/// This handles multiple opening points, arbitrary claim-to-point mapping, and
/// arbitrary commitment grouping. The recursive/single-point path is the
/// `opening_points = [pt]`, `claim_to_point = [0]`,
/// `num_polys_per_point = [1]`, `num_public_rows = 1` specialization.
///
/// # Errors
///
/// Returns an error if the claim shape is invalid, opening-point routing is
/// inconsistent, transcript-bound challenge data has the wrong size, or ring-switch row-eval
/// preparation fails.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "ring_switch_verifier")]
#[inline(never)]
pub(crate) fn ring_switch_verifier<F, E, T, const D: usize>(
    opening_points: &[RingOpeningPoint<F>],
    ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
    claim_to_point: &[usize],
    challenges: &[SparseChallenge],
    w_len: usize,
    w_commitment: &FlatRingVec<F>,
    transcript: &mut T,
    lp: &LevelParams,
    num_polys_per_point: &[usize],
    claim_to_point_poly: &[usize],
    claim_poly_indices: &[usize],
    gamma: &[E],
    num_public_rows: usize,
) -> Result<RingSwitchVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    transcript.append_serde(ABSORB_SUMCHECK_W, w_commitment);

    let alpha: E = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_RING_SWITCH);

    let num_claims = claim_to_point.len();
    validate_opening_points_for_claims(opening_points, claim_to_point, lp, num_claims)?;
    if ring_multiplier_points.len() != opening_points.len()
        || ring_multiplier_points
            .iter()
            .any(|point| point.a_len() < lp.block_len || point.b_len() != lp.num_blocks)
    {
        return Err(AkitaError::InvalidProof);
    }
    if claim_to_point_poly.len() != num_claims || claim_poly_indices.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let num_points = num_polys_per_point.len();
    for claim_idx in 0..num_claims {
        let point_idx = claim_to_point_poly[claim_idx];
        if point_idx >= num_points
            || claim_poly_indices[claim_idx] >= num_polys_per_point[point_idx]
        {
            return Err(AkitaError::InvalidProof);
        }
    }

    let num_ring_elems = w_len / D;
    let col_bits = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    let ring_bits = D.trailing_zeros() as usize;
    let m_rows = lp.m_row_count(num_points, num_public_rows);
    let num_sc_vars = col_bits + ring_bits;
    let num_i = m_rows.next_power_of_two().trailing_zeros() as usize;

    let tau0: Vec<E> = (0..num_sc_vars)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU0))
        .collect();
    let tau1: Vec<E> = (0..num_i)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU1))
        .collect();
    let alpha_evals_y = scalar_powers(alpha, D);
    if gamma.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let prepared_row_eval = prepare_ring_switch_row_eval::<F, E, D>(
        challenges,
        alpha,
        lp,
        &tau1,
        num_polys_per_point,
        claim_to_point_poly,
        claim_poly_indices,
        gamma,
        num_public_rows,
        opening_points.len(),
        ring_multiplier_points,
        claim_to_point,
    )?;

    Ok(RingSwitchVerifyOutput {
        prepared_row_eval,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau0,
        tau1,
        b: 1usize << lp.log_basis,
        alpha,
    })
}

/// Prepare deferred verifier ring-switch row evaluation data.
///
/// # Errors
///
/// Returns an error if gamma/challenge lengths do not match the claim shape,
/// the expanded tau1 table is too short for the level layout, or sparse
/// challenge evaluation fails.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "prepare_ring_switch_row_eval")]
pub fn prepare_ring_switch_row_eval<F, E, const D: usize>(
    challenges: &[SparseChallenge],
    alpha: E,
    lp: &LevelParams,
    tau1: &[E],
    num_polys_per_point: &[usize],
    claim_to_point_poly: &[usize],
    claim_poly_indices: &[usize],
    gamma: &[E],
    num_public_rows: usize,
    opening_points_len: usize,
    ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
    claim_to_point: &[usize],
) -> Result<RingSwitchDeferredRowEval<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt + MulBase<F>,
{
    let alpha_pows = scalar_powers(alpha, D);
    let num_claims = claim_to_point.len();
    if claim_to_point_poly.len() != num_claims || claim_poly_indices.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let num_points = num_polys_per_point.len();
    for claim_idx in 0..num_claims {
        let point_idx = claim_to_point_poly[claim_idx];
        if point_idx >= num_points
            || claim_poly_indices[claim_idx] >= num_polys_per_point[point_idx]
        {
            return Err(AkitaError::InvalidProof);
        }
    }

    if gamma.len() != num_claims {
        return Err(AkitaError::InvalidSize {
            expected: num_claims,
            actual: gamma.len(),
        });
    }

    let depth_commit = lp.num_digits_commit;
    let depth_open = lp.num_digits_open;
    let depth_fold = lp.num_digits_fold;
    let log_basis = lp.log_basis;
    let num_blocks = lp.num_blocks;
    let n_b = lp.b_key.row_len();
    let n_d = lp.d_key.row_len();
    let num_t_vectors = num_polys_per_point
        .iter()
        .try_fold(0usize, |acc, &count| acc.checked_add(count))
        .ok_or_else(|| AkitaError::InvalidSetup("batched t-vector count overflow".to_string()))?;
    #[cfg(feature = "zk")]
    let d_blinding_segment_len = zk::blinding_digit_plane_count::<F>(n_d, D, log_basis);
    #[cfg(feature = "zk")]
    let b_blinding_digit_planes_per_point = zk::blinding_digit_plane_count::<F>(n_b, D, log_basis);
    #[cfg(feature = "zk")]
    let b_blinding_segment_len = num_points
        .checked_mul(b_blinding_digit_planes_per_point)
        .ok_or_else(|| AkitaError::InvalidSetup("ZK blinding width overflow".to_string()))?;
    let total_blocks = num_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("batched block count overflow".to_string()))?;
    if challenges.len() != total_blocks {
        return Err(AkitaError::InvalidSize {
            expected: total_blocks,
            actual: challenges.len(),
        });
    }
    let block_len = lp.block_len;
    let inner_width = block_len * depth_commit;
    let num_points = opening_points_len.max(1);
    if ring_multiplier_points.len() != opening_points_len {
        return Err(AkitaError::InvalidProof);
    }
    let rows = lp.m_row_count(num_points, num_public_rows);

    let eq_tau1 = EqPolynomial::evals(tau1);
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }

    let c_alphas: Vec<E> = challenges
        .iter()
        .map(|challenge| challenge.eval_at_pows::<F, E, D>(&alpha_pows))
        .collect::<Result<_, _>>()?;

    let z_first = lp.m_vars >= lp.r_vars;

    let claim_to_point_poly: Vec<(usize, usize)> = claim_to_point_poly
        .iter()
        .zip(claim_poly_indices.iter())
        .map(|(&point_idx, &poly_idx)| (point_idx, poly_idx))
        .collect();

    let is_tiered = lp.is_tiered_root();
    Ok(RingSwitchDeferredRowEval {
        c_alphas,
        eq_tau1,
        total_blocks,
        num_t_vectors,
        num_blocks,
        num_claims,
        depth_open,
        depth_commit,
        depth_fold,
        #[cfg(feature = "zk")]
        d_blinding_segment_len,
        #[cfg(feature = "zk")]
        b_blinding_digit_planes_per_point,
        #[cfg(feature = "zk")]
        b_blinding_segment_len,
        block_len,
        inner_width,
        log_basis,
        n_a: lp.a_key.row_len(),
        n_d,
        n_b,
        num_points,
        rows,
        z_first,
        claim_to_point_poly,
        num_polys_per_point: num_polys_per_point.to_vec(),
        num_public_rows,
        gamma: gamma.to_vec(),
        claim_to_point: claim_to_point.to_vec(),
        is_tiered,
        split_factor: if is_tiered { lp.split_factor } else { 1 },
        outer_log_basis: if is_tiered { lp.outer_log_basis } else { 0 },
        num_digits_outer: if is_tiered { lp.num_digits_outer } else { 0 },
        n_f: if is_tiered { lp.f_key.row_len() } else { 0 },
        b_prime_width: if is_tiered { lp.b_prime_width() } else { 0 },
    })
}

impl<E: FieldCore> RingSwitchDeferredRowEval<E> {
    /// Evaluate the prepared ring-switch row table at the supplied point.
    ///
    /// # Errors
    ///
    /// Returns an error if the setup matrix cannot be viewed at `D` or an
    /// internal offset-eq evaluation receives inconsistent dimensions.
    ///
    /// # Panics
    ///
    /// Panics if the prepared state was built for a layout inconsistent with
    /// the provided setup, opening points, or challenge vector. Callers should
    /// build values through [`prepare_ring_switch_row_eval`] or `ring_switch_verifier`.
    #[inline]
    pub fn eval_at_point<F, const D: usize>(
        &self,
        x_challenges: &[E],
        setup: &AkitaExpandedSetup<F>,
        opening_points: &[RingOpeningPoint<F>],
        ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
        alpha: E,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + CanonicalField + akita_field::RandomSampling,
        E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    {
        if ring_multiplier_points.len() != opening_points.len() {
            return Err(AkitaError::InvalidProof);
        }
        // ----- Witness-layout offsets ----------------------------------------
        let w_len = self.depth_open * self.total_blocks;
        let t_total_blocks = self.num_blocks * self.num_t_vectors;
        let t_len = self.depth_open * self.n_a * t_total_blocks;
        let z_len = self.depth_fold * self.depth_commit * self.num_points * self.block_len;
        // Tiered uhat segment: present only when `is_tiered`, sized
        // `num_points · n_b' · split_factor · num_digits_outer`. Placed
        // immediately after `t_hat` per spec §9, before any blinding
        // segments and before `z_hat` in the `z_first=false` ordering.
        let uhat_len = if self.is_tiered {
            self.num_points * self.n_b * self.split_factor * self.num_digits_outer
        } else {
            0usize
        };
        #[cfg(feature = "zk")]
        let b_blinding_segment_len = self.b_blinding_segment_len;
        #[cfg(not(feature = "zk"))]
        let b_blinding_segment_len = 0usize;
        #[cfg(feature = "zk")]
        let d_blinding_segment_len = self.d_blinding_segment_len;
        #[cfg(not(feature = "zk"))]
        let d_blinding_segment_len = 0usize;

        let offset_z = if self.z_first {
            0
        } else {
            w_len + t_len + uhat_len + b_blinding_segment_len + d_blinding_segment_len
        };
        let offset_w = if self.z_first { z_len } else { 0 };
        let offset_t = if self.z_first { z_len + w_len } else { w_len };
        let offset_uhat = offset_t + t_len;
        let offset_r =
            w_len + d_blinding_segment_len + t_len + uhat_len + b_blinding_segment_len + z_len;

        // ----- Shared precomputes --------------------------------------------
        let alpha_pows = scalar_powers(alpha, D);
        let g1_open = gadget_row_scalars::<F>(self.depth_open, self.log_basis);
        let fold_gadget = gadget_row_scalars::<F>(self.depth_fold, self.log_basis);

        // Eq table over the low `log₂(num_blocks)` bits, shared by W/T
        // peeled summaries and by `compute_setup_contribution`.
        let offset_low_bits = self.num_blocks.trailing_zeros() as usize;
        let eq_low = EqPolynomial::evals(&x_challenges[..offset_low_bits]);
        let block_offset_low = offset_w & (self.num_blocks - 1);
        debug_assert_eq!(block_offset_low, offset_t & (self.num_blocks - 1));

        // `z` peels `block_len` (not `num_blocks`) and uses its own
        // low-bit eq table.
        let z_offset_low_bits = self.block_len.trailing_zeros() as usize;
        let z_block_low_eq = EqPolynomial::evals(&x_challenges[..z_offset_low_bits]);

        let high_challenges = &x_challenges[offset_low_bits..];

        // Per-claim `c_alpha` carry summary — shared by W and T.
        let challenge_block_summaries: Vec<[E; 2]> = (0..self.num_claims)
            .map(|claim_idx| {
                let start = claim_idx * self.num_blocks;
                summarize_pow2_block_carries(
                    &eq_low,
                    block_offset_low,
                    &self.c_alphas[start..(start + self.num_blocks)],
                )
            })
            .collect();
        let mut challenge_block_summaries_by_t_vector =
            vec![[E::zero(), E::zero()]; self.num_t_vectors];
        // Per-point t-vector starting indices: `t_vector_offsets[p]` is the
        // running sum of bundle sizes for points `< p`. Lifted out of the
        // per-claim loop so the routing is O(num_points + num_claims) rather
        // than O(num_points * num_claims).
        let t_vector_offsets: Vec<usize> = self
            .num_polys_per_point
            .iter()
            .scan(0usize, |acc, &count| {
                let offset = *acc;
                *acc += count;
                Some(offset)
            })
            .collect();
        for (claim_idx, &(point_idx, poly_idx)) in self.claim_to_point_poly.iter().enumerate() {
            let t_vector_idx = t_vector_offsets[point_idx] + poly_idx;
            let [carry0, carry1] = challenge_block_summaries[claim_idx];
            challenge_block_summaries_by_t_vector[t_vector_idx][0] += carry0;
            challenge_block_summaries_by_t_vector[t_vector_idx][1] += carry1;
        }

        // ----- W -------------------------------------------------------------
        let w_structured_contribution = {
            let _span = tracing::info_span!("w_structured").entered();
            let uses_ring_multipliers = ring_multiplier_points
                .iter()
                .any(|point| point.as_base().is_none());
            let row_coefficient_rings = if uses_ring_multipliers {
                Some(
                    self.gamma
                        .iter()
                        .copied()
                        .map(|coefficient| {
                            embed_ring_subfield_scalar::<F, E, D>(
                                coefficient,
                                AkitaError::InvalidProof,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
            } else {
                None
            };
            let public_block_summaries: Vec<[E; 2]> = (0..self.num_claims)
                .map(|claim_idx| {
                    let point_idx = self.claim_to_point[claim_idx];
                    if point_idx >= ring_multiplier_points.len() {
                        return Err(AkitaError::InvalidProof);
                    }
                    let point = &ring_multiplier_points[point_idx];
                    let coefficient_ring = row_coefficient_rings
                        .as_ref()
                        .map(|rings| &rings[claim_idx]);
                    summarize_pow2_multiplier_block_carries(
                        &eq_low,
                        block_offset_low,
                        point.b_len(),
                        |idx| {
                            point.eval_b_with_coefficient(
                                idx,
                                self.gamma[claim_idx],
                                coefficient_ring,
                                &alpha_pows,
                            )
                        },
                    )
                })
                .collect::<Result<_, _>>()?;
            let public_row_weights_by_claim: Vec<E> = self
                .claim_to_point
                .iter()
                .map(|&point_idx| self.eq_tau1[1 + point_idx])
                .collect();
            WStructuredSlicesEvaluator {
                high_challenges,
                offset_high: offset_w >> offset_low_bits,
                gadget_vector: &g1_open,
                public_block_summaries: &public_block_summaries,
                challenge_block_summaries: &challenge_block_summaries,
                public_row_weights_by_claim: &public_row_weights_by_claim,
                challenge_weight: self.eq_tau1[0],
            }
            .evaluate()
        };

        // ----- T -------------------------------------------------------------
        let t_structured_contribution = {
            let _span = tracing::info_span!("t_structured").entered();
            // Tiered row layout (`specs/tiered_commit.md` §3) places
            // A-rows after the tier-1 + F row blocks instead of after
            // the legacy B rows. Per `compute_m_evals_x`'s tiered
            // dispatch in `crates/akita-prover/src/protocol/ring_switch.rs`.
            let a_start = if self.is_tiered {
                1 + self.num_public_rows
                    + self.n_d
                    + self.split_factor * self.n_b * self.num_points
                    + self.n_f * self.num_points
            } else {
                1 + self.num_public_rows + self.n_d + self.n_b * self.num_points
            };
            TStructuredSlicesEvaluator {
                high_challenges,
                offset_high: offset_t >> offset_low_bits,
                gadget_vector: &g1_open,
                challenge_block_summaries: &challenge_block_summaries_by_t_vector,
                a_row_weights: &self.eq_tau1[a_start..self.rows],
            }
            .evaluate()
        };

        // ----- Fused D·ŵ + B·t̂ + A·ẑ ---------------------------------------
        let setup_contribution = {
            let _span = tracing::info_span!("setup_contribution").entered();
            compute_setup_contribution::<F, E, D>(
                self,
                x_challenges,
                setup,
                &eq_low,
                &z_block_low_eq,
                &alpha_pows,
                &fold_gadget,
                offset_w,
                offset_t,
                offset_z,
            )
        };

        // ----- Z (consistency-row) ------------------------------------------
        let z_structured_contribution = {
            let _span = tracing::info_span!("z_structured").entered();
            let g1_commit = gadget_row_scalars::<F>(self.depth_commit, self.log_basis);
            if self.block_len.is_power_of_two() {
                let z_offset_low = offset_z & (self.block_len - 1);
                let a_block_summary: Vec<[E; 2]> = ring_multiplier_points
                    .iter()
                    .map(|ring_multiplier_point| {
                        summarize_pow2_multiplier_block_carries(
                            &z_block_low_eq,
                            z_offset_low,
                            self.block_len,
                            |idx| ring_multiplier_point.eval_a_at::<E>(idx, &alpha_pows),
                        )
                    })
                    .collect::<Result<_, _>>()?;
                ZStructuredPow2SlicesEvaluator {
                    high_challenges: &x_challenges[z_offset_low_bits..],
                    offset_high: offset_z >> z_offset_low_bits,
                    g1_commit: &g1_commit,
                    fold_gadget: &fold_gadget,
                    a_block_summary: &a_block_summary,
                    consistency_weight: self.eq_tau1[0],
                }
                .evaluate()
            } else {
                let a_evals_by_point: Vec<Vec<E>> = ring_multiplier_points
                    .iter()
                    .map(|ring_multiplier_point| {
                        (0..self.block_len)
                            .map(|idx| ring_multiplier_point.eval_a_at::<E>(idx, &alpha_pows))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<_, AkitaError>>()?;
                ZDenseSlicesEvaluator {
                    g1_commit: &g1_commit,
                    fold_gadget: &fold_gadget,
                    consistency_weight: self.eq_tau1[0],
                    a_evals_by_point: &a_evals_by_point,
                    full_vec_randomness: x_challenges,
                    offset_z,
                    block_len: self.block_len,
                }
                .evaluate()
            }
        };

        // ----- r-tail --------------------------------------------------------
        let r_contribution = {
            let r_gadget =
                gadget_row_scalars::<F>(r_decomp_levels::<F>(self.log_basis), self.log_basis);
            let denom = alpha_pows[D - 1] * alpha + E::one();
            compute_r_contribution(self, x_challenges, offset_r, denom, &r_gadget)
        };

        #[allow(unused_mut)]
        let mut total = w_structured_contribution
            + t_structured_contribution
            + z_structured_contribution
            + setup_contribution
            + r_contribution;

        // Tiered tier-1 + F contribution. `compute_setup_contribution`
        // above skips the legacy T-half when `prepared.is_tiered`, so
        // we add the replacement contribution here. Per spec §3, this
        // contribution covers the `(tier1 + F) × num_points` row block
        // of M.
        if self.is_tiered {
            let _span = tracing::info_span!("tier1_and_f_contribution").entered();
            use super::slice_mle::tier1_reference::{
                compute_tier1_and_f_contribution_optimized, BPhysicalLayout, Tier1AndFInputs,
            };
            use crate::protocol::tier1_f_matrix::derive_tier1_f_matrix_flat;

            let n_b_prime = self.n_b;
            let chunk_width = self.b_prime_width;
            let n_f = self.n_f;
            let f_width = n_b_prime * self.split_factor * self.num_digits_outer;

            // B' view: the leading `chunk_width` columns of the first
            // `n_b_prime` physical rows of the shared B matrix. We must
            // view with the full `max_stride` width so `row(r)` reads
            // physical row r (otherwise contiguous reading slips into
            // row 0's tail columns when `chunk_width < stride`). The
            // tier-1 reference below indexes only `[0..chunk_width)`
            // within each row, matching the prover's
            // `mat_vec_mul_ntt_single_i8_cyclic` slicing.
            let b_prime_view = setup
                .shared_matrix
                .ring_view::<D>(n_b_prime, setup.seed.max_stride);

            // F is derived deterministically from the same public
            // matrix seed using the domain-separated label
            // `b"tier1-f"`. Mirrors the prover-side derivation in
            // `crates/akita-prover/src/kernels/matrix.rs` and commit
            // `f39615bf`. We rebuild the F matrix here on every
            // evaluation; future revisions can cache it on the
            // verifier setup once `AkitaVerifierSetup` carries
            // tiering metadata.
            let f_flat = {
                let _f_span = tracing::info_span!("tier1_f_matrix_derive").entered();
                derive_tier1_f_matrix_flat::<F, D>(n_f * f_width, &setup.seed.public_matrix_seed)
            };
            let f_view = f_flat.ring_view::<D>(n_f, f_width);

            // Row weight slices from eq_tau1. Tiered row layout from
            // spec §3:
            //   consistency (1) | public | D (n_d) | tier1 (f·n_b'·num_points)
            //     | F (n_F·num_points) | A (n_a)
            let d_start = 1 + self.num_public_rows;
            let tier1_start = d_start + self.n_d;
            let tier1_end = tier1_start + self.split_factor * n_b_prime * self.num_points;
            let f_start = tier1_end;
            let f_end = f_start + n_f * self.num_points;
            let tier1_row_weights = self.eq_tau1[tier1_start..tier1_end].to_vec();
            let f_row_weights = self.eq_tau1[f_start..f_end].to_vec();

            // Outer gadget vector G = (1, 2^b, 2^{2b}, …). Computed in
            // the field so we avoid the `1u64 << k` overflow when
            // `outer_log_basis · num_digits_outer ≥ 64` (always the
            // case for full-field Q128 with `outer_log_basis ≤ 6`).
            let two_to_b = F::from_u64(1u64 << self.outer_log_basis);
            let mut outer_gadget = Vec::with_capacity(self.num_digits_outer);
            let mut step = F::one();
            for _ in 0..self.num_digits_outer {
                outer_gadget.push(step);
                step *= two_to_b;
            }

            let inputs = Tier1AndFInputs::<F, E, D> {
                b_prime_view,
                b_prime_chunk_width: chunk_width,
                f_view,
                tier1_row_weights: &tier1_row_weights,
                f_row_weights: &f_row_weights,
                alpha_pows: &alpha_pows,
                full_vec_randomness: x_challenges,
                outer_gadget: &outer_gadget,
                offset_t,
                offset_uhat,
                split_factor: self.split_factor,
                num_digits_outer: self.num_digits_outer,
                b_physical: BPhysicalLayout {
                    n_a: self.n_a,
                    num_blocks: self.num_blocks,
                    depth_open: self.depth_open,
                    num_t_vectors: self.num_t_vectors,
                },
                num_points: self.num_points,
            };
            let tier1_and_f = compute_tier1_and_f_contribution_optimized::<F, E, D>(
                &inputs,
                &self.num_polys_per_point,
            );
            total += tier1_and_f;
        }

        #[cfg(feature = "zk")]
        {
            let b_blinding = compute_b_blinding_part::<F, E, D>(self, x_challenges, setup, alpha);
            let d_blinding = compute_d_blinding_part::<F, E, D>(self, x_challenges, setup, alpha);
            total = total + b_blinding + d_blinding;
        }

        Ok(total)
    }
}

#[inline]
fn summarize_pow2_multiplier_block_carries<E, EvalAt>(
    eq_low: &[E],
    offset_low: usize,
    values_len: usize,
    mut eval_at: EvalAt,
) -> Result<[E; 2], AkitaError>
where
    E: FieldCore,
    EvalAt: FnMut(usize) -> Result<E, AkitaError>,
{
    assert!(
        values_len.is_power_of_two(),
        "peeled inner block length must be a power of two"
    );
    assert_eq!(
        eq_low.len(),
        values_len,
        "low eq table must match peeled inner block length"
    );
    assert!(
        offset_low < values_len,
        "low offset must lie inside the peeled block"
    );

    let inner_bits = values_len.trailing_zeros() as usize;
    let inner_mask = values_len - 1;
    let mut out = [E::zero(), E::zero()];

    for u in 0..values_len {
        let sum = offset_low + u;
        let carry = sum >> inner_bits;
        debug_assert!(
            carry < 2,
            "sum of two peeled indices must carry at most one bit"
        );
        let low_idx = sum & inner_mask;
        out[carry] += eq_low[low_idx] * eval_at(u)?;
    }

    Ok(out)
}

#[cfg(test)]
#[inline]
pub(crate) fn summarize_pow2_block_carries_base<F, E>(
    eq_low: &[E],
    offset_low: usize,
    values: &[F],
) -> [E; 2]
where
    F: FieldCore,
    E: akita_field::ExtField<F>,
{
    assert!(
        values.len().is_power_of_two(),
        "peeled inner block length must be a power of two"
    );
    assert_eq!(
        eq_low.len(),
        values.len(),
        "low eq table must match peeled inner block length"
    );
    assert!(
        offset_low < values.len(),
        "low offset must lie inside the peeled block"
    );

    let inner_bits = values.len().trailing_zeros() as usize;
    let inner_mask = values.len() - 1;
    let mut out = [E::zero(), E::zero()];

    for (u, &value) in values.iter().enumerate() {
        let sum = offset_low + u;
        let carry = sum >> inner_bits;
        debug_assert!(
            carry < 2,
            "sum of two peeled indices must carry at most one bit"
        );
        let low_idx = sum & inner_mask;
        out[carry] += eq_low[low_idx].mul_base(value);
    }

    out
}
