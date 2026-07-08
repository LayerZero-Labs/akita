//! Verifier-side ring-switch replay.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::summarize_pow2_block_carries;
use akita_algebra::ring::scalar_powers;
use akita_challenges::Challenges;
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, MulBase, RandomSampling,
};
use akita_transcript::labels::{
    ABSORB_NEXT_LEVEL_WITNESS_BINDING, ABSORB_TERMINAL_W_REMAINDER, CHALLENGE_RING_SWITCH,
    CHALLENGE_TAU0, CHALLENGE_TAU1,
};
use akita_transcript::{sample_ext_challenge, Transcript};
use akita_types::{
    gadget_row_scalars, r_decomp_levels, validate_role_dispatch, AkitaExpandedSetup,
    CommitmentRingDims, FpExtEncoding, LevelParams, MRowLayout, RingMultiplierOpeningPoint,
    RingOpeningPoint, RingRelationInstance, RingRole, RingVec, SetupContributionPlanInputs,
    TerminalWitnessTranscriptParts, WitnessChunkLayout, WitnessLayout,
};
use std::ops::Range;

use super::slice_mle::{
    compute_r_contribution, high_eq_window, EStructuredSlicesEvaluator, SetupEvaluation,
    SetupEvaluator, SetupEvaluatorMode, StructuredSliceMleEvaluator, TStructuredSlicesEvaluator,
    ZDenseSlicesEvaluator, ZStructuredPow2SlicesEvaluator,
};
use super::validate_log_basis;
use akita_types::validate_ring_dispatch;
pub(crate) use tensor_challenges::PreparedChallengeEvals;

mod tensor_challenges;
#[cfg(test)]
mod tests;

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

struct RingSwitchVerifyCoreOutput<E: FieldCore> {
    prepared_row_eval: RingSwitchDeferredRowEval<E>,
    alpha_evals_y: Vec<E>,
    col_bits: usize,
    ring_bits: usize,
    tau0: Option<Vec<E>>,
    tau1: Vec<E>,
    b: usize,
    alpha: E,
}

impl<E: FieldCore> RingSwitchVerifyCoreOutput<E> {
    fn into_intermediate(self) -> Result<RingSwitchVerifyOutput<E>, AkitaError> {
        let tau0 = self.tau0.ok_or(AkitaError::InvalidProof)?;
        Ok(RingSwitchVerifyOutput {
            prepared_row_eval: self.prepared_row_eval,
            alpha_evals_y: self.alpha_evals_y,
            col_bits: self.col_bits,
            ring_bits: self.ring_bits,
            tau0,
            tau1: self.tau1,
            b: self.b,
            alpha: self.alpha,
        })
    }

    fn into_terminal_as_output(self) -> Result<RingSwitchVerifyOutput<E>, AkitaError> {
        if self.tau0.is_some() {
            return Err(AkitaError::InvalidProof);
        }
        Ok(RingSwitchVerifyOutput {
            prepared_row_eval: self.prepared_row_eval,
            alpha_evals_y: self.alpha_evals_y,
            col_bits: self.col_bits,
            ring_bits: self.ring_bits,
            tau0: Vec::new(),
            tau1: self.tau1,
            b: self.b,
            alpha: self.alpha,
        })
    }
}

/// Precomputed challenge-derived data for deferred ring-switch row MLE evaluation.
///
/// Stores only data that cannot be derived from context at evaluation time:
/// alpha-evaluated folding challenges and the tau1 eq-polynomial expansion.
/// Everything else is passed by reference at evaluation time to avoid
/// duplicating setup matrix views, opening points, and gadget vectors.
#[derive(Clone)]
pub struct RingSwitchDeferredRowEval<F: FieldCore> {
    pub(crate) eq_tau1: Vec<F>,
    pub(crate) role_dims: CommitmentRingDims,
    pub(crate) groups: Vec<RingSwitchDeferredRowGroupEval<F>>,
    pub(crate) e_setup_cols: usize,
    pub(crate) n_d_active: usize,
    pub(crate) d_start: usize,
    /// Batch-wide fold depth used by setup-sumcheck planning.
    pub(crate) depth_fold: usize,
    /// Batch-wide basis used by the shared r-tail.
    pub(crate) log_basis: u32,
    /// Resolved witness column layout (one chunk for the single-chunk case,
    /// `W` chunks for the distributed-prover layout).
    pub(crate) chunk_layout: WitnessLayout,
    pub(crate) setup_contribution_inputs: SetupContributionPlanInputs<F>,
}

#[derive(Clone)]
pub(crate) struct RingSwitchDeferredRowGroupEval<F: FieldCore> {
    pub(crate) c_alphas: PreparedChallengeEvals<F>,
    pub(crate) a_evals: Vec<F>,
    pub(crate) chunk_range: Range<usize>,
    pub(crate) e_setup_offset: usize,
    pub(crate) num_claims: usize,
    pub(crate) num_blocks: usize,
    pub(crate) block_len: usize,
    pub(crate) depth_open: usize,
    pub(crate) depth_commit: usize,
    pub(crate) depth_fold: usize,
    pub(crate) log_basis: u32,
    pub(crate) n_a: usize,
    pub(crate) n_b: usize,
    pub(crate) t_cols_per_vector: usize,
    pub(crate) a_row_start: usize,
    pub(crate) b_row_start: usize,
}

/// Fixed public relation inputs for verifier ring-switch replay.
pub struct RingSwitchReplay<'a, F: FieldCore, E> {
    pub relation: &'a RingRelationInstance<F>,
    pub row_coefficients: &'a [E],
    pub lp: &'a LevelParams,
}

/// Replay the verifier half of ring switching.
///
/// This handles the single-point relation replay for one committed polynomial
/// bundle.
///
/// # Errors
///
/// Returns an error if the claim shape is invalid, opening-point routing is
/// inconsistent, transcript-bound challenge data has the wrong size, or ring-switch row-eval
/// preparation fails.
#[tracing::instrument(skip_all, name = "ring_switch_verifier")]
#[inline(never)]
pub(crate) fn ring_switch_verifier<F, E, T, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E>,
    w_len: usize,
    w_commitment: &RingVec<F>,
    next_ring_dim: usize,
    transcript: &mut T,
) -> Result<RingSwitchVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: FpExtEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    // `validate_ring_dispatch` is called inside `ring_switch_verifier_core`;
    // the outer wrapper just performs the witness absorb before delegating.
    // The next-witness commitment is shaped at the *next* level's schedule
    // ring dimension, which may differ from this level's dispatch `D` in
    // mixed-D schedules.
    if next_ring_dim == 0 || !w_commitment.can_decode_vec(next_ring_dim) {
        return Err(AkitaError::InvalidProof);
    }
    transcript.absorb_and_record_serde(ABSORB_NEXT_LEVEL_WITNESS_BINDING, w_commitment);
    ring_switch_verifier_core::<F, E, T, D>(replay, w_len, transcript, MRowLayout::WithDBlock)?
        .into_intermediate()
}

/// Terminal variant of [`ring_switch_verifier`].
///
/// This owns the required terminal final-witness remainder absorb before
/// sampling ring-switch challenges.
///
/// # Errors
///
/// Returns an error if the claim shape is invalid, opening-point routing is
/// inconsistent, transcript-bound challenge data has the wrong size, or
/// ring-switch row-eval preparation fails.
#[tracing::instrument(skip_all, name = "ring_switch_verifier_terminal")]
#[inline(never)]
pub(crate) fn ring_switch_verifier_terminal<F, E, T, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E>,
    w_len: usize,
    transcript: &mut T,
    terminal_parts: &TerminalWitnessTranscriptParts,
) -> Result<RingSwitchVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: FpExtEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    transcript.absorb_and_record_bytes(ABSORB_TERMINAL_W_REMAINDER, &terminal_parts.remainder);
    ring_switch_verifier_core::<F, E, T, D>(replay, w_len, transcript, MRowLayout::WithoutDBlock)?
        .into_terminal_as_output()
}

#[tracing::instrument(skip_all, name = "ring_switch_verifier_core")]
#[inline(never)]
fn ring_switch_verifier_core<F, E, T, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E>,
    w_len: usize,
    transcript: &mut T,
    m_row_layout: MRowLayout,
) -> Result<RingSwitchVerifyCoreOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: FpExtEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    let relation = replay.relation;
    let lp = replay.lp;
    let opening_batch = relation.opening_batch();
    let num_polys = opening_batch.num_total_polynomials();
    let gamma = replay.row_coefficients;

    let alpha: E = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_RING_SWITCH);

    let num_claims = relation.opening_batch().num_total_polynomials();
    // Validate each group's opening/multiplier point against that group's own
    // block geometry (final vs frozen-precommit). For a scalar batch this is the
    // single group at `lp`'s geometry, byte-identical to the historical check.
    for group_index in 0..opening_batch.num_groups() {
        let group_lp = lp.root_group_params(opening_batch, group_index)?;
        let opening_point = relation.group_opening_point(group_index)?;
        if opening_point.a.len() < group_lp.block_len()
            || opening_point.b.len() != group_lp.num_blocks()
        {
            return Err(AkitaError::InvalidProof);
        }
        let multiplier_point = relation.group_ring_multiplier_point(group_index)?;
        if multiplier_point.a_len() < group_lp.block_len()
            || multiplier_point.b_len() != group_lp.num_blocks()
        {
            return Err(AkitaError::InvalidProof);
        }
    }
    if num_polys != num_claims {
        return Err(AkitaError::InvalidProof);
    }

    if w_len == 0 || !w_len.is_multiple_of(D) {
        return Err(AkitaError::InvalidProof);
    }
    let num_ring_elems = w_len / D;
    let col_bits = num_ring_elems
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("ring-switch column count overflow".to_string()))?
        .trailing_zeros() as usize;
    let ring_bits = validate_ring_dispatch::<D>()?;
    let m_rows = lp.m_row_count_for(opening_batch.num_groups(), m_row_layout)?;
    let num_sc_vars = col_bits + ring_bits;
    let num_i = m_rows
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("ring-switch row count overflow".to_string()))?
        .trailing_zeros() as usize;

    let tau0 = match m_row_layout {
        MRowLayout::WithDBlock => Some(
            (0..num_sc_vars)
                .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU0))
                .collect(),
        ),
        MRowLayout::WithoutDBlock => None,
    };
    let tau1: Vec<E> = (0..num_i)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU1))
        .collect();
    let alpha_evals_y = scalar_powers(alpha, D);
    if gamma.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let prepared_row_eval =
        prepare_ring_switch_row_eval::<F, E, D>(replay, alpha, &tau1, Some(num_ring_elems))?;

    Ok(RingSwitchVerifyCoreOutput {
        prepared_row_eval,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau0,
        tau1,
        b: 1usize
            .checked_shl(lp.log_basis)
            .ok_or_else(|| AkitaError::InvalidSetup("basis size overflow".to_string()))?,
        alpha,
    })
}

/// Prepare deferred verifier ring-switch row evaluation data from a fixed
/// [`RingRelationInstance`] and transcript-sampled row coefficients.
///
/// # Errors
///
/// Returns an error if gamma/challenge lengths do not match the claim shape,
/// the expanded tau1 table is too short for the level layout, or sparse
/// challenge evaluation fails.
#[tracing::instrument(skip_all, name = "prepare_ring_switch_row_eval")]
pub fn prepare_ring_switch_row_eval<F, E, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E>,
    alpha: E,
    tau1: &[E],
    witness_ring_len: Option<usize>,
) -> Result<RingSwitchDeferredRowEval<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F>,
{
    let relation = replay.relation;
    let lp = replay.lp;
    let chunk_layout = relation.segment_layout(lp, witness_ring_len)?;
    let opening_batch = relation.opening_batch();
    let num_polys = opening_batch.num_total_polynomials();
    let depth_fold = lp.num_digits_fold(num_polys, F::modulus_bits())?;
    let rows = lp.m_row_count_for(opening_batch.num_groups(), relation.m_row_layout())?;
    if lp.has_precommitted_groups() {
        return prepare_grouped_ring_switch_row_eval::<F, E, D>(
            replay,
            alpha,
            tau1,
            chunk_layout,
            depth_fold,
            rows,
        );
    }
    let challenges = relation
        .group_challenges()
        .first()
        .ok_or(AkitaError::InvalidProof)?;
    let ring_multiplier_point = relation.group_ring_multiplier_point(0)?;
    prepare_ring_switch_row_eval_inner::<F, E, D>(
        challenges,
        ring_multiplier_point,
        alpha,
        lp,
        tau1,
        num_polys,
        replay.row_coefficients,
        relation.m_row_layout(),
        chunk_layout,
        depth_fold,
        rows,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_grouped_ring_switch_row_eval<F, E, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E>,
    alpha: E,
    tau1: &[E],
    chunk_layout: WitnessLayout,
    depth_fold: usize,
    rows: usize,
) -> Result<RingSwitchDeferredRowEval<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F>,
{
    let relation = replay.relation;
    let lp = replay.lp;
    let opening_batch = relation.opening_batch();
    lp.reject_grouped_multi_chunk("prepare_grouped_ring_switch_row_eval")?;
    lp.validate_root_opening_batch(opening_batch)?;
    validate_role_dispatch::<D>(relation.role_dims(), RingRole::Inner)?;
    if replay.row_coefficients.len() != opening_batch.num_total_polynomials() {
        return Err(AkitaError::InvalidProof);
    }

    let eq_tau1 = EqPolynomial::evals(tau1)?;
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }

    let order = opening_batch.root_group_order()?;
    if chunk_layout.chunks.len() != order.len() || chunk_layout.chunk_lengths.len() != order.len() {
        return Err(AkitaError::InvalidSetup(
            "grouped witness layout does not match root group order".to_string(),
        ));
    }

    let mut group_e_offsets = vec![0usize; opening_batch.num_groups()];
    let mut e_setup_cols = 0usize;
    for &group_index in &order {
        let group_lp = lp.root_group_params(opening_batch, group_index)?;
        let group_layout = opening_batch.group_layout(group_index)?;
        group_e_offsets[group_index] = e_setup_cols;
        let e_len = group_layout
            .num_polynomials()
            .checked_mul(group_lp.num_blocks())
            .and_then(|n| n.checked_mul(group_lp.num_digits_open()))
            .ok_or_else(|| AkitaError::InvalidSetup("grouped e width overflow".to_string()))?;
        e_setup_cols = e_setup_cols
            .checked_add(e_len)
            .ok_or_else(|| AkitaError::InvalidSetup("grouped e width overflow".to_string()))?;
    }

    let alpha_pows_a = scalar_powers(alpha, D);
    let mut groups = Vec::with_capacity(order.len());
    for (order_pos, &group_index) in order.iter().enumerate() {
        let group_lp = lp.root_group_params(opening_batch, group_index)?;
        let group_layout = opening_batch.group_layout(group_index)?;
        let k_g = group_layout.num_polynomials();
        let num_blocks = group_lp.num_blocks();
        let block_len = group_lp.block_len();
        let depth_open = group_lp.num_digits_open();
        let depth_commit = group_lp.num_digits_commit();
        let depth_fold = lp.num_digits_fold_for_params(group_lp, k_g, lp.field_bits_for_cache())?;
        let log_basis = group_lp.log_basis();
        validate_log_basis(log_basis)?;
        let n_a = group_lp.a_rows_len();
        let n_b = group_lp.b_rows_len();
        let inner_width = group_lp.a_col_len();
        let expected_inner_width = block_len
            .checked_mul(depth_commit)
            .ok_or_else(|| AkitaError::InvalidSetup("grouped inner width overflow".to_string()))?;
        if inner_width < expected_inner_width {
            return Err(AkitaError::InvalidSetup(
                "grouped A-key column width is too small".to_string(),
            ));
        }

        let opening_point = relation.group_opening_point(group_index)?;
        if opening_point.a.len() < block_len || opening_point.b.len() != num_blocks {
            return Err(AkitaError::InvalidProof);
        }
        let ring_multiplier_point = relation.group_ring_multiplier_point(group_index)?;
        if ring_multiplier_point.a_len() < block_len || ring_multiplier_point.b_len() != num_blocks
        {
            return Err(AkitaError::InvalidProof);
        }

        let total_blocks = k_g
            .checked_mul(num_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("grouped block count overflow".to_string()))?;
        let challenges = relation
            .group_challenges()
            .get(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        if challenges.logical_len() != total_blocks {
            return Err(AkitaError::InvalidSize {
                expected: total_blocks,
                actual: challenges.logical_len(),
            });
        }
        let c_alphas =
            prepare_challenge_evals::<F, E, D>(challenges, &alpha_pows_a, k_g, num_blocks)?;
        let a_evals = (0..block_len)
            .map(|idx| ring_multiplier_point.eval_a_at_dyn::<E>(idx, &alpha_pows_a))
            .collect::<Result<Vec<_>, _>>()?;

        let t_cols_per_vector = n_a
            .checked_mul(depth_open)
            .and_then(|len| len.checked_mul(num_blocks))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("grouped B vector width overflow".to_string())
            })?;
        let a_range = lp.root_a_row_range(opening_batch, group_index, relation.m_row_layout())?;
        let b_range =
            lp.root_commitment_row_range(opening_batch, group_index, relation.m_row_layout())?;
        if a_range.len() != n_a || b_range.len() != n_b {
            return Err(AkitaError::InvalidSetup(
                "grouped row ranges do not match group matrix heights".to_string(),
            ));
        }

        groups.push(RingSwitchDeferredRowGroupEval {
            c_alphas,
            a_evals,
            chunk_range: order_pos..order_pos + 1,
            e_setup_offset: group_e_offsets[group_index],
            num_claims: k_g,
            num_blocks,
            block_len,
            depth_open,
            depth_commit,
            depth_fold,
            log_basis,
            n_a,
            n_b,
            t_cols_per_vector,
            a_row_start: a_range.start,
            b_row_start: b_range.start,
        });
    }

    let n_d_active = lp.n_d_active_for(relation.m_row_layout());
    let d_start = rows
        .checked_sub(n_d_active)
        .ok_or(AkitaError::InvalidProof)?;
    let setup_contribution_inputs = SetupContributionPlanInputs {
        eq_tau1: eq_tau1.clone(),
        num_t_vectors: opening_batch.num_total_polynomials(),
        num_blocks: lp.num_blocks,
        num_claims: opening_batch.num_total_polynomials(),
        depth_open: lp.num_digits_open,
        depth_commit: lp.num_digits_commit,
        depth_fold,
        block_len: lp.block_len,
        inner_width: lp.a_key.col_len(),
        n_a: lp.a_key.row_len(),
        n_d: lp.d_key.row_len(),
        m_row_layout: relation.m_row_layout(),
        n_b: lp.b_key.row_len(),
        num_groups: opening_batch.num_groups(),
        rows,
        num_polys_per_group: opening_batch.group_sizes(),
    };

    Ok(RingSwitchDeferredRowEval {
        eq_tau1,
        role_dims: relation.role_dims(),
        groups,
        e_setup_cols,
        n_d_active,
        d_start,
        depth_fold,
        log_basis: lp.log_basis,
        chunk_layout,
        setup_contribution_inputs,
    })
}

fn prepare_challenge_evals<F, E, const D: usize>(
    challenges: &Challenges,
    alpha_pows: &[E],
    num_claims: usize,
    num_blocks: usize,
) -> Result<PreparedChallengeEvals<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F>,
{
    match challenges {
        Challenges::Sparse {
            challenges: sparse, ..
        } => Ok(PreparedChallengeEvals::Flat(
            sparse
                .iter()
                .map(|challenge| challenge.eval_at_pows::<F, E>(alpha_pows))
                .collect::<Result<_, _>>()?,
        )),
        Challenges::Tensor { factored } => {
            if D < 2 {
                return Err(AkitaError::InvalidInput(
                    "tensor challenge factored evaluation requires D >= 2".to_string(),
                ));
            }
            factored.validate::<D>()?;
            if factored.num_claims != num_claims {
                return Err(AkitaError::InvalidSize {
                    expected: num_claims,
                    actual: factored.num_claims,
                });
            }
            let blocks_per_claim = factored.blocks_per_claim()?;
            if blocks_per_claim != num_blocks {
                return Err(AkitaError::InvalidSize {
                    expected: num_blocks,
                    actual: blocks_per_claim,
                });
            }
            Ok(PreparedChallengeEvals::Tensor {
                challenges: factored.clone(),
                alpha_pows: alpha_pows.to_vec(),
            })
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_ring_switch_row_eval_inner<F, E, const D: usize>(
    challenges: &Challenges,
    ring_multiplier_point: &RingMultiplierOpeningPoint<F>,
    alpha: E,
    lp: &LevelParams,
    tau1: &[E],
    num_polys: usize,
    gamma: &[E],
    m_row_layout: MRowLayout,
    chunk_layout: WitnessLayout,
    depth_fold: usize,
    rows: usize,
) -> Result<RingSwitchDeferredRowEval<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F>,
{
    validate_role_dispatch::<D>(lp.role_dims, RingRole::Inner)?;
    let setup_contribution_inputs =
        SetupContributionPlanInputs::from_level_params(lp, &[num_polys], m_row_layout, depth_fold)?
            .with_eq_tau1_from_tau(tau1, rows)?;
    let eq_tau1 = setup_contribution_inputs.eq_tau1.clone();
    let alpha_pows = scalar_powers(alpha, D);
    let num_claims = gamma.len();
    if num_polys != num_claims {
        return Err(AkitaError::InvalidProof);
    }

    let log_basis = lp.log_basis;
    validate_log_basis(log_basis)?;
    let depth_commit = lp.num_digits_commit;
    let depth_open = lp.num_digits_open;
    let num_blocks = lp.num_blocks;
    let total_blocks = num_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("batched block count overflow".to_string()))?;
    if challenges.logical_len() != total_blocks {
        return Err(AkitaError::InvalidSize {
            expected: total_blocks,
            actual: challenges.logical_len(),
        });
    }
    let block_len = lp.block_len;
    let n_a = lp.a_key.row_len();

    let c_alphas =
        prepare_challenge_evals::<F, E, D>(challenges, &alpha_pows, num_claims, lp.num_blocks)?;
    let a_evals = (0..block_len)
        .map(|idx| ring_multiplier_point.eval_a_at::<D, E>(idx, &alpha_pows))
        .collect::<Result<Vec<_>, _>>()?;
    let n_b = lp.b_key.row_len();
    let t_cols_per_vector = n_a
        .checked_mul(depth_open)
        .and_then(|len| len.checked_mul(num_blocks))
        .ok_or_else(|| AkitaError::InvalidSetup("B vector width overflow".to_string()))?;
    let n_d_active = lp.n_d_active_for(m_row_layout);
    let d_start = rows
        .checked_sub(n_d_active)
        .ok_or(AkitaError::InvalidProof)?;
    let e_setup_cols = num_claims
        .checked_mul(num_blocks)
        .and_then(|n| n.checked_mul(depth_open))
        .ok_or_else(|| AkitaError::InvalidSetup("e width overflow".to_string()))?;
    let group = RingSwitchDeferredRowGroupEval {
        c_alphas,
        a_evals,
        chunk_range: 0..chunk_layout.chunks.len(),
        e_setup_offset: 0,
        num_claims,
        num_blocks,
        block_len,
        depth_open,
        depth_commit,
        depth_fold,
        log_basis,
        n_a,
        n_b,
        t_cols_per_vector,
        a_row_start: 1,
        b_row_start: 1 + n_a,
    };

    Ok(RingSwitchDeferredRowEval {
        eq_tau1,
        role_dims: lp.role_dims,
        groups: vec![group],
        e_setup_cols,
        n_d_active,
        d_start,
        depth_fold,
        log_basis,
        chunk_layout,
        setup_contribution_inputs,
    })
}

#[inline(always)]
fn checked_add(lhs: usize, rhs: usize, context: &'static str) -> Result<usize, AkitaError> {
    lhs.checked_add(rhs)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{context} overflow")))
}

#[inline(always)]
fn checked_mul(lhs: usize, rhs: usize, context: &'static str) -> Result<usize, AkitaError> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{context} overflow")))
}

#[inline(always)]
fn checked_slice<'a, T>(
    slice: &'a [T],
    start: usize,
    len: usize,
    context: &'static str,
) -> Result<&'a [T], AkitaError> {
    let end = checked_add(start, len, context)?;
    slice.get(start..end).ok_or(AkitaError::InvalidProof)
}

impl<E: FieldCore> RingSwitchDeferredRowEval<E> {
    pub(crate) fn chunk_layout(&self) -> &WitnessLayout {
        &self.chunk_layout
    }

    pub(crate) fn create_setup_contribution_inputs(&self) -> SetupContributionPlanInputs<E> {
        self.setup_contribution_inputs.clone()
    }

    fn single_group(&self) -> Result<&RingSwitchDeferredRowGroupEval<E>, AkitaError> {
        match self.groups.as_slice() {
            [group] => Ok(group),
            _ => Err(AkitaError::InvalidSetup(
                "flat row evaluation requires exactly one group".to_string(),
            )),
        }
    }

    fn group_chunks(
        &self,
        group: &RingSwitchDeferredRowGroupEval<E>,
    ) -> Result<&[WitnessChunkLayout], AkitaError> {
        self.chunk_layout
            .chunks
            .get(group.chunk_range.clone())
            .ok_or(AkitaError::InvalidProof)
    }

    #[inline(always)]
    fn uses_grouped_eval<const D: usize>(&self) -> bool {
        self.groups.len() != 1 || self.role_dims.d_b() != D || self.role_dims.d_d() != D
    }

    fn eval_grouped_at_point<F, const D: usize>(
        &self,
        x_challenges: &[E],
        setup: &AkitaExpandedSetup<F>,
        alpha: E,
        setup_claim: Option<E>,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: FpExtEncoding<F> + FromPrimitiveInt,
    {
        validate_role_dispatch::<D>(self.role_dims, RingRole::Inner)?;
        let d_b = self.role_dims.d_b();
        let d_d = self.role_dims.d_d();
        let alpha_pows_a = scalar_powers(alpha, D);
        let alpha_pows_b = scalar_powers(alpha, d_b);
        let alpha_pows_d = scalar_powers(alpha, d_d);
        let consistency_weight = *self.eq_tau1.first().ok_or(AkitaError::InvalidProof)?;

        let mut e_structured_contribution = E::zero();
        let mut t_structured_contribution = E::zero();
        let mut z_structured_contribution = E::zero();
        {
            let _span = tracing::info_span!("grouped_structured_chunks").entered();
            for group in &self.groups {
                if group.num_blocks == 0 || !group.num_blocks.is_power_of_two() {
                    return Err(AkitaError::InvalidSetup(
                        "grouped witness block count must be a non-zero power of two".to_string(),
                    ));
                }
                let total_blocks =
                    checked_mul(group.num_claims, group.num_blocks, "grouped block count")?;
                if let Some(c_alphas) = group.c_alphas.as_flat() {
                    if c_alphas.len() != total_blocks {
                        return Err(AkitaError::InvalidSize {
                            expected: total_blocks,
                            actual: c_alphas.len(),
                        });
                    }
                }

                let g_open = gadget_row_scalars::<F>(group.depth_open, group.log_basis);
                let g_commit = gadget_row_scalars::<F>(group.depth_commit, group.log_basis);
                let fold_gadget = gadget_row_scalars::<F>(group.depth_fold, group.log_basis);

                let block_bits = group.num_blocks.trailing_zeros() as usize;
                if block_bits > x_challenges.len() {
                    return Err(AkitaError::InvalidSize {
                        expected: block_bits,
                        actual: x_challenges.len(),
                    });
                }
                let eq_low = EqPolynomial::evals(&x_challenges[..block_bits])?;
                let high_challenges = &x_challenges[block_bits..];
                let x_low_challenges = &x_challenges[..block_bits];

                let chunks = self.group_chunks(group)?;
                let chunk = chunks.first().ok_or(AkitaError::InvalidProof)?;
                let block_offset_low = chunk.offset_e & (group.num_blocks - 1);
                let summaries = group.c_alphas.summarize_chunk_block_carries::<F, D>(
                    group.num_claims,
                    x_low_challenges,
                    &eq_low,
                    block_offset_low,
                    0,
                    group.num_blocks,
                    group.num_blocks,
                )?;

                let e_offset_high = chunk.offset_e >> block_bits;
                let eq_hi_e_table = high_eq_window(
                    high_challenges,
                    e_offset_high,
                    group.num_claims * group.depth_open,
                );
                e_structured_contribution += EStructuredSlicesEvaluator {
                    gadget_vector: &g_open,
                    challenge_block_summaries: &summaries,
                    challenge_weight: consistency_weight,
                    high_eq_table: &eq_hi_e_table,
                }
                .evaluate();

                let a_row_weights = checked_slice(
                    &self.eq_tau1,
                    group.a_row_start,
                    group.n_a,
                    "grouped A rows",
                )?;
                let t_offset_high = chunk.offset_t >> block_bits;
                let eq_hi_t_table = high_eq_window(
                    high_challenges,
                    t_offset_high,
                    group.num_claims * group.depth_open * group.n_a,
                );
                t_structured_contribution += TStructuredSlicesEvaluator {
                    gadget_vector: &g_open,
                    challenge_block_summaries: &summaries,
                    a_row_weights,
                    high_eq_table: &eq_hi_t_table,
                }
                .evaluate();

                if group.block_len.is_power_of_two() {
                    let z_offset_low_bits = group.block_len.trailing_zeros() as usize;
                    if z_offset_low_bits > x_challenges.len() {
                        return Err(AkitaError::InvalidSize {
                            expected: z_offset_low_bits,
                            actual: x_challenges.len(),
                        });
                    }
                    let z_block_low_eq = EqPolynomial::evals(&x_challenges[..z_offset_low_bits])?;
                    let z_offset_low = chunk.offset_z & (group.block_len - 1);
                    let a_block_summary = summarize_pow2_block_carries(
                        &z_block_low_eq,
                        z_offset_low,
                        &group.a_evals,
                    )?;
                    let z_offset_high = chunk.offset_z >> z_offset_low_bits;
                    let z_hi_len = fold_gadget.len() * g_commit.len();
                    let eq_hi_z_table =
                        high_eq_window(&x_challenges[z_offset_low_bits..], z_offset_high, z_hi_len);
                    z_structured_contribution += ZStructuredPow2SlicesEvaluator {
                        g1_commit: &g_commit,
                        fold_gadget: &fold_gadget,
                        a_block_summary,
                        consistency_weight,
                        high_eq_table: &eq_hi_z_table,
                    }
                    .evaluate();
                } else {
                    z_structured_contribution += ZDenseSlicesEvaluator {
                        g1_commit: &g_commit,
                        fold_gadget: &fold_gadget,
                        consistency_weight,
                        a_evals: &group.a_evals,
                        full_vec_randomness: x_challenges,
                        offset_z: chunk.offset_z,
                        block_len: group.block_len,
                    }
                    .evaluate()?;
                }
            }
        }

        let setup_contribution = if let Some(claim) = setup_claim {
            claim
        } else {
            let _span = tracing::info_span!("grouped_setup_contribution").entered();
            let setup_contribution_inputs = self.create_setup_contribution_inputs();
            let no_fold_gadget: &[F] = &[];
            let evaluator = SetupEvaluator::new(
                &setup_contribution_inputs,
                x_challenges,
                None,
                None,
                &alpha_pows_a,
                no_fold_gadget,
                self.chunk_layout(),
            );
            match evaluator.evaluate::<D>(SetupEvaluatorMode::GroupedDirect {
                setup,
                prepared: self,
                alpha_pows_b: &alpha_pows_b,
                alpha_pows_d: &alpha_pows_d,
            })? {
                SetupEvaluation::Direct(value) => value,
                #[cfg(test)]
                SetupEvaluation::Recursive(_) => {
                    return Err(AkitaError::InvalidSetup(
                        "setup evaluator returned recursive output for grouped direct mode".into(),
                    ))
                }
            }
        };

        let r_contribution = {
            let r_gadget =
                gadget_row_scalars::<F>(r_decomp_levels::<F>(self.log_basis), self.log_basis);
            let alpha_pow_d = *alpha_pows_d.get(d_d - 1).ok_or(AkitaError::InvalidProof)?;
            let denom = alpha_pow_d * alpha + E::one();
            let offset_r = self.chunk_layout.r_offset()?;
            compute_r_contribution(self, x_challenges, offset_r, denom, &r_gadget)?
        };

        Ok(e_structured_contribution
            + t_structured_contribution
            + z_structured_contribution
            + setup_contribution
            + r_contribution)
    }

    /// Evaluate the prepared ring-switch row table at the supplied point.
    ///
    /// # Errors
    ///
    /// Returns an error if the setup matrix cannot be viewed at `D` or an
    /// internal offset-eq evaluation receives inconsistent dimensions.
    #[inline]
    pub fn eval_at_point<F, const D: usize>(
        &self,
        x_challenges: &[E],
        setup: &AkitaExpandedSetup<F>,
        opening_point: &RingOpeningPoint<F>,
        ring_multiplier_point: &RingMultiplierOpeningPoint<F>,
        alpha: E,
        setup_claim: Option<E>,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: FpExtEncoding<F> + FromPrimitiveInt,
    {
        let _ring_bits = validate_ring_dispatch::<D>()?;
        if self.uses_grouped_eval::<D>() {
            return self.eval_grouped_at_point::<F, D>(x_challenges, setup, alpha, setup_claim);
        }
        let group = self.single_group()?;
        // ----- Witness layout (chunk list) -----------------------------------
        let layout = self.chunk_layout();
        let blocks_per_chunk = layout.blocks_per_chunk;
        if blocks_per_chunk == 0 || !blocks_per_chunk.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "witness chunk block window must be a power of two".to_string(),
            ));
        }
        validate_log_basis(group.log_basis)?;
        if opening_point.b.len() != group.num_blocks || opening_point.a.len() < group.block_len {
            return Err(AkitaError::InvalidProof);
        }
        if ring_multiplier_point.b_len() != group.num_blocks
            || ring_multiplier_point.a_len() < group.block_len
        {
            return Err(AkitaError::InvalidProof);
        }
        if self.setup_contribution_inputs.num_t_vectors != group.num_claims {
            return Err(AkitaError::InvalidProof);
        }

        // ----- Shared precomputes --------------------------------------------
        let alpha_pows = scalar_powers(alpha, D);
        let g1_open = gadget_row_scalars::<F>(group.depth_open, group.log_basis);
        let g1_commit = gadget_row_scalars::<F>(group.depth_commit, group.log_basis);
        let fold_gadget = gadget_row_scalars::<F>(group.depth_fold, group.log_basis);

        // e/t block peel is over `blocks_per_chunk` (`== num_blocks` single-chunk);
        // the `eq_low` table is shared across chunks. z peels `block_len`.
        let block_bits = blocks_per_chunk.trailing_zeros() as usize;
        if block_bits > x_challenges.len() {
            return Err(AkitaError::InvalidSize {
                expected: block_bits,
                actual: x_challenges.len(),
            });
        }
        let eq_low = EqPolynomial::evals(&x_challenges[..block_bits])?;
        let high_challenges = &x_challenges[block_bits..];
        let x_low_challenges = &x_challenges[..block_bits];

        let z_offset_low_bits = group.block_len.trailing_zeros() as usize;
        if z_offset_low_bits > x_challenges.len() {
            return Err(AkitaError::InvalidSize {
                expected: z_offset_low_bits,
                actual: x_challenges.len(),
            });
        }
        let z_block_low_eq = EqPolynomial::evals(&x_challenges[..z_offset_low_bits])?;

        let total_blocks = group.num_blocks * group.num_claims;
        if let Some(c_alphas) = group.c_alphas.as_flat() {
            if c_alphas.len() != total_blocks {
                return Err(AkitaError::InvalidSize {
                    expected: total_blocks,
                    actual: c_alphas.len(),
                });
            }
        }

        // Canonical A-block start: consistency (1) | A | B | D.
        let a_row_count = group.n_a;

        // ----- E-hat / T-hat / Z structured: fold over chunks ----------------
        // `e`/`t` are partitioned (each chunk covers a disjoint global block
        // window, so the contributions sum to the whole component); `z` is
        // replicated (each chunk carries a full `block_len` fold). The cost
        // asymmetry falls out of the chunk geometry, not control flow.
        let mut e_structured_contribution = E::zero();
        let mut t_structured_contribution = E::zero();
        let mut z_structured_contribution = E::zero();
        {
            let _span = tracing::info_span!("structured_chunks").entered();
            for chunk in &layout.chunks {
                // e and t share the in-window block residue: `|e^j|` is a
                // multiple of `blocks_per_chunk`, so `offset_t ≡ offset_e`.
                let block_offset_low = chunk.offset_e & (blocks_per_chunk - 1);
                let summaries = group.c_alphas.summarize_chunk_block_carries::<F, D>(
                    group.num_claims,
                    x_low_challenges,
                    &eq_low,
                    block_offset_low,
                    chunk.global_block_base,
                    blocks_per_chunk,
                    group.num_blocks,
                )?;

                let e_offset_high = chunk.offset_e >> block_bits;
                let eq_hi_e_table = high_eq_window(
                    high_challenges,
                    e_offset_high,
                    group.num_claims * group.depth_open,
                );
                e_structured_contribution += EStructuredSlicesEvaluator {
                    gadget_vector: &g1_open,
                    challenge_block_summaries: &summaries,
                    challenge_weight: self.eq_tau1[0],
                    high_eq_table: &eq_hi_e_table,
                }
                .evaluate();

                let t_offset_high = chunk.offset_t >> block_bits;
                let eq_hi_t_table = high_eq_window(
                    high_challenges,
                    t_offset_high,
                    group.num_claims * group.depth_open * a_row_count,
                );
                t_structured_contribution += TStructuredSlicesEvaluator {
                    gadget_vector: &g1_open,
                    challenge_block_summaries: &summaries,
                    a_row_weights: checked_slice(
                        &self.eq_tau1,
                        group.a_row_start,
                        a_row_count,
                        "flat A rows",
                    )?,
                    high_eq_table: &eq_hi_t_table,
                }
                .evaluate();
            }

            // z dispatches once on `block_len` (chunk-independent); the chunk
            // loop sits outside the case split. Chunk `j>0` exercises a nonzero
            // in-block shift `z_lo = offset_z mod block_len`. The `a` values are
            // global (chunk-independent), so materialize them once and reuse the
            // slice across every chunk and across the pow2/dense split.
            let a_evals = (0..group.block_len)
                .map(|idx| ring_multiplier_point.eval_a_at::<D, E>(idx, &alpha_pows))
                .collect::<Result<Vec<_>, _>>()?;
            if group.block_len.is_power_of_two() {
                for chunk in &layout.chunks {
                    let z_offset_low = chunk.offset_z & (group.block_len - 1);
                    let a_block_summary =
                        summarize_pow2_block_carries(&z_block_low_eq, z_offset_low, &a_evals)?;
                    let z_offset_high = chunk.offset_z >> z_offset_low_bits;
                    let z_hi_len = fold_gadget.len() * g1_commit.len();
                    let eq_hi_z_table =
                        high_eq_window(&x_challenges[z_offset_low_bits..], z_offset_high, z_hi_len);
                    z_structured_contribution += ZStructuredPow2SlicesEvaluator {
                        g1_commit: &g1_commit,
                        fold_gadget: &fold_gadget,
                        a_block_summary,
                        consistency_weight: self.eq_tau1[0],
                        high_eq_table: &eq_hi_z_table,
                    }
                    .evaluate();
                }
            } else {
                for chunk in &layout.chunks {
                    z_structured_contribution += ZDenseSlicesEvaluator {
                        g1_commit: &g1_commit,
                        fold_gadget: &fold_gadget,
                        consistency_weight: self.eq_tau1[0],
                        a_evals: &a_evals,
                        full_vec_randomness: x_challenges,
                        offset_z: chunk.offset_z,
                        block_len: group.block_len,
                    }
                    .evaluate()?;
                }
            }
        }

        // ----- Fused D·ŵ + B·t̂ + A·ẑ (one shared setup scan) ---------------
        let setup_contribution = {
            let _span = tracing::info_span!("setup_contribution").entered();
            if let Some(claim) = setup_claim {
                claim
            } else {
                let setup_contribution_inputs = self.create_setup_contribution_inputs();
                let evaluator = SetupEvaluator::new(
                    &setup_contribution_inputs,
                    x_challenges,
                    Some(&eq_low),
                    Some(&z_block_low_eq),
                    &alpha_pows,
                    &fold_gadget,
                    layout,
                );
                match evaluator.evaluate::<D>(SetupEvaluatorMode::GroupedDirect {
                    setup,
                    prepared: self,
                    alpha_pows_b: &alpha_pows,
                    alpha_pows_d: &alpha_pows,
                })? {
                    SetupEvaluation::Direct(value) => value,
                    #[cfg(test)]
                    SetupEvaluation::Recursive(_) => {
                        return Err(AkitaError::InvalidSetup(
                            "setup evaluator returned recursive output for grouped direct mode"
                                .into(),
                        ))
                    }
                }
            }
        };

        // ----- r-tail (single shared quotient on the last chunk) -------------
        let r_contribution = {
            let r_gadget =
                gadget_row_scalars::<F>(r_decomp_levels::<F>(self.log_basis), self.log_basis);
            let denom = alpha_pows[D - 1] * alpha + E::one();
            let offset_r = layout.r_offset()?;
            compute_r_contribution(self, x_challenges, offset_r, denom, &r_gadget)?
        };

        let total = e_structured_contribution
            + t_structured_contribution
            + z_structured_contribution
            + setup_contribution
            + r_contribution;

        Ok(total)
    }
}
