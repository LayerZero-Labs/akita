//! Verifier-side ring-switch replay.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::summarize_pow2_block_carries;
use akita_algebra::ring::scalar_powers;
use akita_challenges::Challenges;
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, MulBase, MulBaseUnreduced,
    RandomSampling,
};
use akita_transcript::labels::{
    ABSORB_NEXT_LEVEL_WITNESS_BINDING, ABSORB_TERMINAL_W_REMAINDER, CHALLENGE_RING_SWITCH,
    CHALLENGE_TAU0, CHALLENGE_TAU1,
};
use akita_transcript::{sample_ext_challenge, Transcript};
use akita_types::{
    gadget_row_scalars, r_decomp_levels, validate_role_dispatch, AkitaExpandedSetup,
    CommitmentRingDims, FpExtEncoding, LevelParams, RelationMatrixRowLayout,
    RingMultiplierOpeningPoint, RingRelationInstance, RingRole, RingVec,
    SetupContributionGroupInputs, SetupContributionPlan, SetupContributionPlanInputs,
    SetupContributionStatic, TerminalWitnessTranscriptParts, WitnessChunkLayout, WitnessLayout,
};
use std::ops::Range;

use super::slice_mle::{
    compute_r_contribution, evaluate_setup_contribution_direct, high_eq_window,
    EStructuredSlicesEvaluator, StructuredSliceMleEvaluator, TStructuredSlicesEvaluator,
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
    /// Prepared data for prepared relation-matrix MLE evaluation.
    pub relation_matrix_evaluator: RelationMatrixEvaluator<E>,
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
    relation_matrix_evaluator: RelationMatrixEvaluator<E>,
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
            relation_matrix_evaluator: self.relation_matrix_evaluator,
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
            relation_matrix_evaluator: self.relation_matrix_evaluator,
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

/// Precomputed challenge-derived data for prepared relation-matrix MLE evaluation.
///
/// Stores only data that cannot be derived from context at evaluation time:
/// alpha-evaluated folding challenges and the tau1 eq-polynomial expansion.
/// Everything else is passed by reference at evaluation time to avoid
/// duplicating setup matrix views, opening points, and gadget vectors.
#[derive(Clone)]
pub struct RelationMatrixEvaluator<F: FieldCore> {
    pub(crate) role_dims: CommitmentRingDims,
    pub(crate) groups: Vec<RelationMatrixGroupEvaluator<F>>,
    /// Batch-wide fold depth used by setup-sumcheck planning.
    pub(crate) depth_fold: usize,
    /// Batch-wide basis used by the shared r-tail.
    pub(crate) log_basis: u32,
    /// Resolved witness column layout (one chunk for the single-chunk case,
    /// `W` chunks for the distributed-prover layout).
    pub(crate) chunk_layout: WitnessLayout,
    pub(crate) setup_contribution_groups: Vec<SetupContributionGroupInputs>,
    pub(crate) setup_contribution_inputs: SetupContributionPlanInputs<F>,
    pub(crate) setup_contribution_static: SetupContributionStatic<F>,
}

#[derive(Clone)]
pub(crate) struct RelationMatrixGroupEvaluator<F: FieldCore> {
    pub(crate) c_alphas: PreparedChallengeEvals<F>,
    pub(crate) a_evals: Vec<F>,
    pub(crate) chunk_range: Range<usize>,
    pub(crate) e_col_offset: usize,
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
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBaseUnreduced<F>,
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
    ring_switch_verifier_core::<F, E, T, D>(
        replay,
        w_len,
        transcript,
        RelationMatrixRowLayout::WithDBlock,
    )?
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
/// relation-matrix evaluator preparation fails.
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
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    transcript.absorb_and_record_bytes(ABSORB_TERMINAL_W_REMAINDER, &terminal_parts.remainder);
    ring_switch_verifier_core::<F, E, T, D>(
        replay,
        w_len,
        transcript,
        RelationMatrixRowLayout::WithoutDBlock,
    )?
    .into_terminal_as_output()
}

#[tracing::instrument(skip_all, name = "ring_switch_verifier_core")]
#[inline(never)]
fn ring_switch_verifier_core<F, E, T, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E>,
    w_len: usize,
    transcript: &mut T,
    relation_matrix_row_layout: RelationMatrixRowLayout,
) -> Result<RingSwitchVerifyCoreOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBaseUnreduced<F>,
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
    let m_rows =
        lp.relation_matrix_row_count_for(opening_batch.num_groups(), relation_matrix_row_layout)?;
    let num_sc_vars = col_bits + ring_bits;
    let num_i = m_rows
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("ring-switch row count overflow".to_string()))?
        .trailing_zeros() as usize;

    let tau0 = match relation_matrix_row_layout {
        RelationMatrixRowLayout::WithDBlock => Some(
            (0..num_sc_vars)
                .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU0))
                .collect(),
        ),
        RelationMatrixRowLayout::WithoutDBlock => None,
    };
    let tau1: Vec<E> = (0..num_i)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU1))
        .collect();
    let alpha_evals_y = scalar_powers(alpha, D);
    if gamma.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let relation_matrix_evaluator =
        prepare_relation_matrix_evaluator::<F, E, D>(replay, alpha, &tau1, Some(num_ring_elems))?;

    Ok(RingSwitchVerifyCoreOutput {
        relation_matrix_evaluator,
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

/// Prepare relation-matrix evaluator state from a fixed
/// [`RingRelationInstance`] and transcript-sampled row coefficients.
///
/// # Errors
///
/// Returns an error if gamma/challenge lengths do not match the claim shape,
/// the expanded tau1 table is too short for the level layout, or sparse
/// challenge evaluation fails.
#[tracing::instrument(skip_all, name = "prepare_relation_matrix_evaluator")]
pub fn prepare_relation_matrix_evaluator<F, E, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E>,
    alpha: E,
    tau1: &[E],
    witness_ring_len: Option<usize>,
) -> Result<RelationMatrixEvaluator<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F> + MulBaseUnreduced<F>,
{
    let relation = replay.relation;
    let lp = replay.lp;
    let chunk_layout = relation.segment_layout(lp, witness_ring_len)?;
    reject_mixed_d_multi_chunk::<D>(
        lp.role_dims(),
        &chunk_layout,
        "prepare_relation_matrix_evaluator",
    )?;
    let opening_batch = relation.opening_batch();
    let num_polys = opening_batch.num_total_polynomials();
    let depth_fold = lp.num_digits_fold(num_polys, F::modulus_bits())?;
    let rows = lp.relation_matrix_row_count_for(
        opening_batch.num_groups(),
        relation.relation_matrix_row_layout(),
    )?;
    if lp.has_precommitted_groups() {
        return prepare_relation_matrix_evaluator_multi_group::<F, E, D>(
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
    prepare_relation_matrix_evaluator_inner::<F, E, D>(
        challenges,
        ring_multiplier_point,
        alpha,
        lp,
        tau1,
        num_polys,
        replay.row_coefficients,
        relation.relation_matrix_row_layout(),
        chunk_layout,
        depth_fold,
        rows,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_relation_matrix_evaluator_multi_group<F, E, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E>,
    alpha: E,
    tau1: &[E],
    chunk_layout: WitnessLayout,
    depth_fold: usize,
    rows: usize,
) -> Result<RelationMatrixEvaluator<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F> + MulBaseUnreduced<F>,
{
    let relation = replay.relation;
    let lp = replay.lp;
    let opening_batch = relation.opening_batch();
    lp.reject_multi_group_multi_chunk("prepare_relation_matrix_evaluator_multi_group")?;
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
            "multi-group witness layout does not match root group order".to_string(),
        ));
    }

    let mut group_e_offsets = vec![0usize; opening_batch.num_groups()];
    let mut d_physical_cols = 0usize;
    for &group_index in &order {
        let group_lp = lp.root_group_params(opening_batch, group_index)?;
        let group_layout = opening_batch.group_layout(group_index)?;
        group_e_offsets[group_index] = d_physical_cols;
        let e_len = group_layout
            .num_polynomials()
            .checked_mul(group_lp.num_blocks())
            .and_then(|n| n.checked_mul(group_lp.num_digits_open()))
            .ok_or_else(|| AkitaError::InvalidSetup("multi-group e width overflow".to_string()))?;
        d_physical_cols = d_physical_cols
            .checked_add(e_len)
            .ok_or_else(|| AkitaError::InvalidSetup("multi-group e width overflow".to_string()))?;
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
        let expected_inner_width = block_len.checked_mul(depth_commit).ok_or_else(|| {
            AkitaError::InvalidSetup("multi-group inner width overflow".to_string())
        })?;
        if inner_width < expected_inner_width {
            return Err(AkitaError::InvalidSetup(
                "multi-group A-key column width is too small".to_string(),
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

        let total_blocks = k_g.checked_mul(num_blocks).ok_or_else(|| {
            AkitaError::InvalidSetup("multi-group block count overflow".to_string())
        })?;
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
                AkitaError::InvalidSetup("multi-group B vector width overflow".to_string())
            })?;
        let a_range = lp.root_a_row_range(
            opening_batch,
            group_index,
            relation.relation_matrix_row_layout(),
        )?;
        let b_range = lp.root_commitment_row_range(
            opening_batch,
            group_index,
            relation.relation_matrix_row_layout(),
        )?;
        if a_range.len() != n_a || b_range.len() != n_b {
            return Err(AkitaError::InvalidSetup(
                "multi-group row ranges do not match group matrix heights".to_string(),
            ));
        }

        groups.push(RelationMatrixGroupEvaluator {
            c_alphas,
            a_evals,
            chunk_range: order_pos..order_pos + 1,
            e_col_offset: group_e_offsets[group_index],
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

    let n_d_active = lp.n_d_active_for(relation.relation_matrix_row_layout());
    let d_start = rows
        .checked_sub(n_d_active)
        .ok_or(AkitaError::InvalidProof)?;
    let setup_contribution_inputs = SetupContributionPlanInputs {
        relation_matrix_row_layout: relation.relation_matrix_row_layout(),
        rows,
        n_a: lp.a_key.row_len(),
        n_b: lp.b_key.row_len(),
        n_d: lp.d_key.row_len(),
        num_groups: opening_batch.num_groups(),
        num_polys_per_group: opening_batch.group_sizes(),
        num_t_vectors: opening_batch.num_total_polynomials(),
        num_claims: opening_batch.num_total_polynomials(),
        num_blocks: lp.num_blocks,
        block_len: lp.block_len,
        depth_open: lp.num_digits_open,
        depth_commit: lp.num_digits_commit,
        depth_fold,
        inner_width: lp.a_key.col_len(),
        eq_tau1: eq_tau1.clone(),
    };

    let setup_contribution_groups = build_setup_contribution_groups(&chunk_layout, &groups)?;
    let setup_contribution_static = SetupContributionPlan::prepare_static(
        &setup_contribution_inputs,
        &setup_contribution_groups,
        d_start,
        n_d_active,
        d_physical_cols,
    )?;

    Ok(RelationMatrixEvaluator {
        role_dims: relation.role_dims(),
        groups,
        depth_fold,
        log_basis: lp.log_basis,
        chunk_layout,
        setup_contribution_groups,
        setup_contribution_inputs,
        setup_contribution_static,
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
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F> + MulBaseUnreduced<F>,
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
fn prepare_relation_matrix_evaluator_inner<F, E, const D: usize>(
    challenges: &Challenges,
    ring_multiplier_point: &RingMultiplierOpeningPoint<F>,
    alpha: E,
    lp: &LevelParams,
    tau1: &[E],
    num_polys: usize,
    gamma: &[E],
    relation_matrix_row_layout: RelationMatrixRowLayout,
    chunk_layout: WitnessLayout,
    depth_fold: usize,
    rows: usize,
) -> Result<RelationMatrixEvaluator<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F> + MulBaseUnreduced<F>,
{
    validate_role_dispatch::<D>(lp.role_dims, RingRole::Inner)?;
    let setup_contribution_inputs = SetupContributionPlanInputs::from_level_params(
        lp,
        &[num_polys],
        relation_matrix_row_layout,
        depth_fold,
    )?
    .with_eq_tau1_from_tau(tau1, rows)?;
    reject_mixed_d_multi_chunk::<D>(
        lp.role_dims,
        &chunk_layout,
        "prepare_relation_matrix_evaluator",
    )?;
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
    let n_d_active = lp.n_d_active_for(relation_matrix_row_layout);
    let d_start = rows
        .checked_sub(n_d_active)
        .ok_or(AkitaError::InvalidProof)?;
    let d_physical_cols = num_claims
        .checked_mul(num_blocks)
        .and_then(|n| n.checked_mul(depth_open))
        .ok_or_else(|| AkitaError::InvalidSetup("e width overflow".to_string()))?;
    let group = RelationMatrixGroupEvaluator {
        c_alphas,
        a_evals,
        chunk_range: 0..chunk_layout.chunks.len(),
        e_col_offset: 0,
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

    let groups = vec![group];
    let setup_contribution_groups = build_setup_contribution_groups(&chunk_layout, &groups)?;
    let setup_contribution_static = SetupContributionPlan::prepare_static(
        &setup_contribution_inputs,
        &setup_contribution_groups,
        d_start,
        n_d_active,
        d_physical_cols,
    )?;

    Ok(RelationMatrixEvaluator {
        role_dims: lp.role_dims,
        groups,
        depth_fold,
        log_basis,
        chunk_layout,
        setup_contribution_groups,
        setup_contribution_inputs,
        setup_contribution_static,
    })
}

fn reject_mixed_d_multi_chunk<const D: usize>(
    role_dims: CommitmentRingDims,
    chunk_layout: &WitnessLayout,
    context: &str,
) -> Result<(), AkitaError> {
    if chunk_layout.chunks.len() > 1 && (role_dims.d_b() != D || role_dims.d_d() != D) {
        return Err(AkitaError::InvalidSetup(format!(
            "{context}: multi-chunk witness layout requires uniform ring dimensions across roles"
        )));
    }
    Ok(())
}

pub(crate) fn build_setup_contribution_groups<F: FieldCore>(
    chunk_layout: &WitnessLayout,
    groups: &[RelationMatrixGroupEvaluator<F>],
) -> Result<Vec<SetupContributionGroupInputs>, AkitaError> {
    groups
        .iter()
        .map(|group| {
            let chunks = chunk_layout
                .chunks
                .get(group.chunk_range.clone())
                .ok_or(AkitaError::InvalidProof)?
                .to_vec();
            let blocks_per_chunk = if chunks.len() == 1 {
                group.num_blocks
            } else {
                chunk_layout.blocks_per_chunk
            };
            Ok(SetupContributionGroupInputs {
                e_col_offset: group.e_col_offset,
                num_claims: group.num_claims,
                num_blocks: group.num_blocks,
                block_len: group.block_len,
                depth_open: group.depth_open,
                depth_commit: group.depth_commit,
                depth_fold: group.depth_fold,
                log_basis: group.log_basis,
                n_a: group.n_a,
                n_b: group.n_b,
                t_cols_per_vector: group.t_cols_per_vector,
                a_row_start: group.a_row_start,
                b_row_start: group.b_row_start,
                blocks_per_chunk,
                chunks,
            })
        })
        .collect()
}

struct SetupContributionEqCache<E, F> {
    eq_low: Option<Vec<E>>,
    z_block_low_eq: Option<Vec<E>>,
    fold_gadget: Option<Vec<F>>,
}

fn precompute_setup_contribution_eq_cache<E, F>(
    setup_groups: &[SetupContributionGroupInputs],
    x_challenges: &[E],
) -> Result<SetupContributionEqCache<E, F>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore,
{
    let mut cache = SetupContributionEqCache {
        eq_low: None,
        z_block_low_eq: None,
        fold_gadget: None,
    };
    let Some(first) = setup_groups.first() else {
        return Ok(cache);
    };

    if setup_groups
        .iter()
        .all(|group| group.blocks_per_chunk == first.blocks_per_chunk)
    {
        let block_bits = first.blocks_per_chunk.trailing_zeros() as usize;
        if block_bits > x_challenges.len() {
            return Err(AkitaError::InvalidSize {
                expected: block_bits,
                actual: x_challenges.len(),
            });
        }
        cache.eq_low = Some(EqPolynomial::evals(&x_challenges[..block_bits])?);
    }

    if setup_groups
        .iter()
        .all(|group| group.block_len == first.block_len)
    {
        let z_offset_low_bits = first.block_len.trailing_zeros() as usize;
        if z_offset_low_bits > x_challenges.len() {
            return Err(AkitaError::InvalidSize {
                expected: z_offset_low_bits,
                actual: x_challenges.len(),
            });
        }
        cache.z_block_low_eq = Some(EqPolynomial::evals(&x_challenges[..z_offset_low_bits])?);
    }

    if setup_groups
        .iter()
        .all(|group| group.depth_fold == first.depth_fold && group.log_basis == first.log_basis)
    {
        cache.fold_gadget = Some(gadget_row_scalars::<F>(first.depth_fold, first.log_basis));
    }

    Ok(cache)
}

impl<E: FieldCore> RelationMatrixEvaluator<E> {
    fn group_chunks(
        &self,
        group: &RelationMatrixGroupEvaluator<E>,
    ) -> Result<&[WitnessChunkLayout], AkitaError> {
        self.chunk_layout
            .chunks
            .get(group.chunk_range.clone())
            .ok_or(AkitaError::InvalidProof)
    }

    /// Evaluate the relation matrix at a point at the supplied point.
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
        alpha: E,
        setup_claim: Option<E>,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: FpExtEncoding<F> + FromPrimitiveInt + MulBaseUnreduced<F>,
    {
        let _ring_bits = validate_ring_dispatch::<D>()?;
        validate_role_dispatch::<D>(self.role_dims, RingRole::Inner)?;
        let layout = &self.chunk_layout;
        let d_b = self.role_dims.d_b();
        let d_d = self.role_dims.d_d();
        let alpha_pows_a = scalar_powers(alpha, D);
        let alpha_pows_b = scalar_powers(alpha, d_b);
        let alpha_pows_d = scalar_powers(alpha, d_d);

        let mut e_structured_contribution = E::zero();
        let mut t_structured_contribution = E::zero();
        let mut z_structured_contribution = E::zero();
        let setup_eq_cache = precompute_setup_contribution_eq_cache::<E, F>(
            &self.setup_contribution_groups,
            x_challenges,
        )?;
        let setup_eq_low = setup_eq_cache.eq_low;
        let setup_z_block_low_eq = setup_eq_cache.z_block_low_eq;
        let setup_fold_gadget = setup_eq_cache.fold_gadget;

        {
            let _span = tracing::info_span!("structured_chunks").entered();
            for group in &self.groups {
                let chunks = self.group_chunks(group)?;
                let blocks_per_chunk = if chunks.len() == 1 {
                    group.num_blocks
                } else {
                    if layout.blocks_per_chunk == 0 || !layout.blocks_per_chunk.is_power_of_two() {
                        return Err(AkitaError::InvalidSetup(
                            "witness chunk block window must be a power of two".to_string(),
                        ));
                    }
                    layout.blocks_per_chunk
                };
                if blocks_per_chunk == 0 || !blocks_per_chunk.is_power_of_two() {
                    return Err(AkitaError::InvalidSetup(
                        "witness block window must be a non-zero power of two".to_string(),
                    ));
                }
                validate_log_basis(group.log_basis)?;

                let total_blocks =
                    group
                        .num_claims
                        .checked_mul(group.num_blocks)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("witness block count overflow".into())
                        })?;
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
                let fold_gadget_storage;
                let fold_gadget = match setup_fold_gadget.as_deref() {
                    Some(fold_gadget) if fold_gadget.len() >= group.depth_fold => fold_gadget,
                    _ => {
                        fold_gadget_storage =
                            gadget_row_scalars::<F>(group.depth_fold, group.log_basis);
                        &fold_gadget_storage
                    }
                };

                let block_bits = blocks_per_chunk.trailing_zeros() as usize;
                if block_bits > x_challenges.len() {
                    return Err(AkitaError::InvalidSize {
                        expected: block_bits,
                        actual: x_challenges.len(),
                    });
                }
                let eq_low_storage;
                let eq_low = match setup_eq_low.as_deref() {
                    Some(eq_low) if eq_low.len() >= blocks_per_chunk => eq_low,
                    _ => {
                        eq_low_storage = EqPolynomial::evals(&x_challenges[..block_bits])?;
                        &eq_low_storage
                    }
                };
                let high_challenges = &x_challenges[block_bits..];
                let x_low_challenges = &x_challenges[..block_bits];

                let z_offset_low_bits = group.block_len.trailing_zeros() as usize;
                if z_offset_low_bits > x_challenges.len() {
                    return Err(AkitaError::InvalidSize {
                        expected: z_offset_low_bits,
                        actual: x_challenges.len(),
                    });
                }
                let z_block_low_eq_storage;
                let z_block_low_eq = match setup_z_block_low_eq.as_deref() {
                    Some(z_block_low_eq) if z_block_low_eq.len() >= group.block_len => {
                        z_block_low_eq
                    }
                    _ => {
                        z_block_low_eq_storage =
                            EqPolynomial::evals(&x_challenges[..z_offset_low_bits])?;
                        &z_block_low_eq_storage
                    }
                };

                for chunk in chunks {
                    let block_offset_low = chunk.offset_e & (blocks_per_chunk - 1);
                    let summaries = group.c_alphas.summarize_chunk_block_carries::<F, D>(
                        group.num_claims,
                        x_low_challenges,
                        eq_low,
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
                        gadget_vector: &g_open,
                        challenge_block_summaries: &summaries,
                        challenge_weight: self.setup_contribution_inputs.eq_tau1[0],
                        high_eq_table: &eq_hi_e_table,
                    }
                    .evaluate();

                    let t_offset_high = chunk.offset_t >> block_bits;
                    let eq_hi_t_table = high_eq_window(
                        high_challenges,
                        t_offset_high,
                        group.num_claims * group.depth_open * group.n_a,
                    );
                    let a_row_end = group
                        .a_row_start
                        .checked_add(group.n_a)
                        .ok_or_else(|| AkitaError::InvalidSetup("A rows overflow".into()))?;
                    t_structured_contribution += TStructuredSlicesEvaluator {
                        gadget_vector: &g_open,
                        challenge_block_summaries: &summaries,
                        a_row_weights: self
                            .setup_contribution_inputs
                            .eq_tau1
                            .get(group.a_row_start..a_row_end)
                            .ok_or(AkitaError::InvalidProof)?,
                        high_eq_table: &eq_hi_t_table,
                    }
                    .evaluate();
                }

                if group.block_len.is_power_of_two() {
                    for chunk in chunks {
                        let z_offset_low = chunk.offset_z & (group.block_len - 1);
                        let a_block_summary = summarize_pow2_block_carries(
                            z_block_low_eq,
                            z_offset_low,
                            &group.a_evals,
                        )?;
                        let z_offset_high = chunk.offset_z >> z_offset_low_bits;
                        let z_hi_len = fold_gadget.len() * g_commit.len();
                        let eq_hi_z_table = high_eq_window(
                            &x_challenges[z_offset_low_bits..],
                            z_offset_high,
                            z_hi_len,
                        );
                        z_structured_contribution += ZStructuredPow2SlicesEvaluator {
                            g1_commit: &g_commit,
                            fold_gadget,
                            a_block_summary,
                            consistency_weight: self.setup_contribution_inputs.eq_tau1[0],
                            high_eq_table: &eq_hi_z_table,
                        }
                        .evaluate();
                    }
                } else {
                    for chunk in chunks {
                        z_structured_contribution += ZDenseSlicesEvaluator {
                            g1_commit: &g_commit,
                            fold_gadget,
                            consistency_weight: self.setup_contribution_inputs.eq_tau1[0],
                            a_evals: &group.a_evals,
                            full_vec_randomness: x_challenges,
                            offset_z: chunk.offset_z,
                            block_len: group.block_len,
                        }
                        .evaluate()?;
                    }
                }
            }
        }

        let setup_contribution = if let Some(claim) = setup_claim {
            claim
        } else {
            let _span = tracing::info_span!("setup_contribution").entered();
            let fold_gadget = setup_fold_gadget.as_deref().unwrap_or(&[]);
            evaluate_setup_contribution_direct::<F, E, D>(
                self,
                x_challenges,
                setup_eq_low.as_deref(),
                setup_z_block_low_eq.as_deref(),
                &alpha_pows_a,
                &alpha_pows_b,
                &alpha_pows_d,
                fold_gadget,
                setup,
            )?
        };

        let r_contribution = {
            let r_gadget =
                gadget_row_scalars::<F>(r_decomp_levels::<F>(self.log_basis), self.log_basis);
            let alpha_pow_d = *alpha_pows_d.get(d_d - 1).ok_or(AkitaError::InvalidProof)?;
            let denom = alpha_pow_d * alpha + E::one();
            let offset_r = layout.r_offset()?;
            compute_r_contribution(self, x_challenges, offset_r, denom, &r_gadget)?
        };

        Ok(e_structured_contribution
            + t_structured_contribution
            + z_structured_contribution
            + setup_contribution
            + r_contribution)
    }
}
