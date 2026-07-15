//! Verifier-side ring-switch replay.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
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
    eval_relation_weight_at_point, gadget_row_scalars, r_decomp_levels, validate_role_dispatch,
    AkitaExpandedSetup, CommitmentRingDims, FpExtEncoding, LevelParams, RelationMatrixRowLayout,
    RingMultiplierOpeningPoint, RingRelationInstance, RingRole, RingVec,
    SetupContributionGroupInputs, SetupContributionPlan, SetupContributionPlanInputs,
    SetupContributionStatic, TerminalWitnessTranscriptParts, WitnessLayout, WitnessUnitLayout,
};

use super::slice_mle::{compute_r_contribution, evaluate_setup_contribution_direct};
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
    /// Batch-wide basis used by the shared r-tail.
    pub(crate) log_basis: u32,
    /// Canonical semantic and ownership layout for all witness columns.
    pub(crate) layout: WitnessLayout,
    /// Exact compact witness source length before Boolean suffix padding.
    pub(crate) opening_source_len: usize,
    pub(crate) setup_contribution_groups: Vec<SetupContributionGroupInputs>,
    pub(crate) setup_contribution_inputs: SetupContributionPlanInputs<F>,
    pub(crate) setup_contribution_static: SetupContributionStatic<F>,
    pub(crate) flat_context: Option<FlatRelationContext<F>>,
}

#[derive(Clone)]
pub(crate) struct FlatRelationContext<F: FieldCore> {
    level_params: LevelParams,
    row_coefficients: Vec<F>,
    tau1: Vec<F>,
    relation_matrix_row_layout: RelationMatrixRowLayout,
    opening_ring_dim: usize,
}

#[derive(Clone)]
pub(crate) struct RelationMatrixGroupEvaluator<F: FieldCore> {
    pub(crate) c_alphas: PreparedChallengeEvals<F>,
    pub(crate) opening_a_evals: Vec<F>,
    pub(crate) group_id: usize,
    pub(crate) e_col_offset: usize,
    pub(crate) num_claims: usize,
    pub(crate) live_fold_count: usize,
    pub(crate) fold_position_count: usize,
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
    pub setup: &'a AkitaExpandedSetup<F>,
    pub relation: &'a RingRelationInstance<F>,
    pub row_coefficients: &'a [E],
    pub lp: &'a LevelParams,
    pub opening_source_len: usize,
    pub opening_ring_dim: usize,
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
        let group_lp = lp.group_params(opening_batch, group_index)?;
        let opening_point = relation.group_opening_point(group_index)?;
        if opening_point.position_weights.len() != group_lp.fold_position_count()
            || opening_point.fold_weights.len() != group_lp.live_fold_count()
        {
            return Err(AkitaError::InvalidProof);
        }
        let multiplier_point = relation.group_ring_multiplier_point(group_index)?;
        if multiplier_point.position_len() != group_lp.fold_position_count()
            || multiplier_point.fold_len() != group_lp.live_fold_count()
        {
            return Err(AkitaError::InvalidProof);
        }
    }
    if num_polys != num_claims {
        return Err(AkitaError::InvalidProof);
    }

    let opening_capacity = replay
        .opening_source_len
        .checked_mul(replay.opening_ring_dim)
        .ok_or(AkitaError::InvalidProof)?;
    if w_len == 0
        || !w_len.is_multiple_of(D)
        || replay.opening_ring_dim == 0
        || !replay.opening_ring_dim.is_power_of_two()
        || !w_len.is_multiple_of(replay.opening_ring_dim)
        || w_len > opening_capacity
    {
        return Err(AkitaError::InvalidProof);
    }
    let num_ring_elems = w_len / D;
    let opening_ring_dim = replay.opening_ring_dim;
    let opening_field_len = akita_types::opening_domain_len(replay.opening_source_len)?
        .checked_mul(opening_ring_dim)
        .ok_or(AkitaError::InvalidProof)?;
    let col_bits = opening_field_len.trailing_zeros() as usize;
    let ring_bits = 0;
    let num_sc_vars = col_bits + ring_bits;
    let num_i =
        lp.relation_row_index_num_vars_for_layout(relation_matrix_row_layout, opening_batch)?;

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
    if gamma.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let relation_matrix_evaluator =
        prepare_relation_matrix_evaluator::<F, E, D>(replay, alpha, &tau1, Some(num_ring_elems))?;
    Ok(RingSwitchVerifyCoreOutput {
        relation_matrix_evaluator,
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
    let layout = relation.segment_layout(lp, witness_ring_len)?;
    if layout.total_len() > replay.opening_source_len {
        return Err(AkitaError::InvalidProof);
    }
    reject_mixed_d_multi_chunk::<D>(lp.role_dims(), &layout, "prepare_relation_matrix_evaluator")?;
    let opening_batch = relation.opening_batch();
    let num_polys = opening_batch.num_total_polynomials();
    let depth_fold = lp.num_digits_fold(num_polys, F::modulus_bits())?;
    let rows = lp.relation_matrix_row_count_for(
        opening_batch.num_groups(),
        relation.relation_matrix_row_layout(),
    )?;
    if lp.has_precommitted_groups() {
        return prepare_relation_matrix_evaluator_multi_group::<F, E, D>(
            replay, alpha, tau1, layout, depth_fold, rows,
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
        layout,
        replay.opening_source_len,
        replay.opening_ring_dim,
        depth_fold,
        rows,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_relation_matrix_evaluator_multi_group<F, E, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E>,
    alpha: E,
    tau1: &[E],
    layout: WitnessLayout,
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
    lp.validate_opening_batch(opening_batch)?;
    validate_role_dispatch::<D>(relation.role_dims(), RingRole::Inner)?;
    if replay.row_coefficients.len() != opening_batch.num_total_polynomials() {
        return Err(AkitaError::InvalidProof);
    }

    let eq_tau1: std::sync::Arc<[E]> = EqPolynomial::evals_prefix(tau1, rows)?.into();

    let order = opening_batch.root_group_order()?;
    if order
        .iter()
        .any(|&group_index| layout.num_shards_for_group(group_index) != lp.witness_chunk.num_chunks)
    {
        return Err(AkitaError::InvalidSetup(
            "multi-group witness layout does not match root group order".to_string(),
        ));
    }

    let mut group_e_offsets = vec![0usize; opening_batch.num_groups()];
    let mut d_physical_cols = 0usize;
    for &group_index in &order {
        let group_lp = lp.group_params(opening_batch, group_index)?;
        let group_layout = opening_batch.group_layout(group_index)?;
        group_e_offsets[group_index] = d_physical_cols;
        let e_len = group_layout
            .num_polynomials()
            .checked_mul(group_lp.live_fold_count())
            .and_then(|n| n.checked_mul(group_lp.num_digits_open()))
            .ok_or_else(|| AkitaError::InvalidSetup("multi-group e width overflow".to_string()))?;
        d_physical_cols = d_physical_cols
            .checked_add(e_len)
            .ok_or_else(|| AkitaError::InvalidSetup("multi-group e width overflow".to_string()))?;
    }

    let alpha_pows_a = scalar_powers(alpha, D);
    let mut groups = Vec::with_capacity(order.len());
    for &group_index in &order {
        let group_lp = lp.group_params(opening_batch, group_index)?;
        let group_layout = opening_batch.group_layout(group_index)?;
        let k_g = group_layout.num_polynomials();
        let live_fold_count = group_lp.live_fold_count();
        let fold_position_count = group_lp.fold_position_count();
        let depth_open = group_lp.num_digits_open();
        let depth_commit = group_lp.num_digits_commit();
        let depth_fold = lp.num_digits_fold_for_params(group_lp, k_g, lp.field_bits_for_cache())?;
        let log_basis = group_lp.log_basis();
        validate_log_basis(log_basis)?;
        let n_a = group_lp.a_rows_len();
        let n_b = group_lp.b_rows_len();
        let inner_width = group_lp.a_col_len();
        let expected_inner_width =
            fold_position_count
                .checked_mul(depth_commit)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("multi-group inner width overflow".to_string())
                })?;
        if inner_width < expected_inner_width {
            return Err(AkitaError::InvalidSetup(
                "multi-group A-key column width is too small".to_string(),
            ));
        }

        let opening_point = relation.group_opening_point(group_index)?;
        if opening_point.position_weights.len() != fold_position_count
            || opening_point.fold_weights.len() != live_fold_count
        {
            return Err(AkitaError::InvalidProof);
        }
        let ring_multiplier_point = relation.group_ring_multiplier_point(group_index)?;
        if ring_multiplier_point.position_len() != fold_position_count
            || ring_multiplier_point.fold_len() != live_fold_count
        {
            return Err(AkitaError::InvalidProof);
        }

        let total_blocks = k_g.checked_mul(live_fold_count).ok_or_else(|| {
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
            prepare_challenge_evals::<F, E, D>(challenges, &alpha_pows_a, k_g, live_fold_count)?;
        let opening_a_evals = (0..fold_position_count)
            .map(|idx| ring_multiplier_point.eval_position_at_dyn::<E>(idx, &alpha_pows_a))
            .collect::<Result<Vec<_>, _>>()?;

        let t_cols_per_vector = n_a
            .checked_mul(depth_open)
            .and_then(|len| len.checked_mul(live_fold_count))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("multi-group B vector width overflow".to_string())
            })?;
        let a_range = lp.a_row_range(
            opening_batch,
            group_index,
            relation.relation_matrix_row_layout(),
        )?;
        let b_range = lp.commitment_row_range(
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
            opening_a_evals,
            group_id: group_index,
            e_col_offset: group_e_offsets[group_index],
            num_claims: k_g,
            live_fold_count,
            fold_position_count,
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
        live_fold_count: lp.live_fold_count,
        fold_position_count: lp.fold_position_count,
        depth_open: lp.num_digits_open,
        depth_commit: lp.num_digits_commit,
        depth_fold,
        inner_width: lp.a_key.col_len(),
        eq_tau1: eq_tau1.clone(),
    };

    let setup_contribution_groups =
        build_setup_contribution_groups(&layout, replay.opening_source_len, &groups)?;
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
        log_basis: lp.log_basis,
        layout,
        opening_source_len: replay.opening_source_len,
        setup_contribution_groups,
        setup_contribution_inputs,
        setup_contribution_static,
        flat_context: Some(FlatRelationContext {
            level_params: lp.clone(),
            row_coefficients: replay.row_coefficients.to_vec(),
            tau1: tau1.to_vec(),
            relation_matrix_row_layout: relation.relation_matrix_row_layout(),
            opening_ring_dim: replay.opening_ring_dim,
        }),
    })
}

fn prepare_challenge_evals<F, E, const D: usize>(
    challenges: &Challenges,
    alpha_pows: &[E],
    num_claims: usize,
    live_fold_count: usize,
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
            if blocks_per_claim != live_fold_count {
                return Err(AkitaError::InvalidSize {
                    expected: live_fold_count,
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
    layout: WitnessLayout,
    opening_source_len: usize,
    opening_ring_dim: usize,
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
    reject_mixed_d_multi_chunk::<D>(lp.role_dims, &layout, "prepare_relation_matrix_evaluator")?;
    let alpha_pows = scalar_powers(alpha, D);
    let num_claims = gamma.len();
    if num_polys != num_claims {
        return Err(AkitaError::InvalidProof);
    }

    let log_basis = lp.log_basis;
    validate_log_basis(log_basis)?;
    let depth_commit = lp.num_digits_commit;
    let depth_open = lp.num_digits_open;
    let live_fold_count = lp.live_fold_count;
    let total_blocks = live_fold_count
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("batched block count overflow".to_string()))?;
    if challenges.logical_len() != total_blocks {
        return Err(AkitaError::InvalidSize {
            expected: total_blocks,
            actual: challenges.logical_len(),
        });
    }
    let fold_position_count = lp.fold_position_count;
    let n_a = lp.a_key.row_len();

    let c_alphas = prepare_challenge_evals::<F, E, D>(
        challenges,
        &alpha_pows,
        num_claims,
        lp.live_fold_count,
    )?;
    let opening_a_evals = (0..fold_position_count)
        .map(|idx| ring_multiplier_point.eval_position_at::<D, E>(idx, &alpha_pows))
        .collect::<Result<Vec<_>, _>>()?;
    let n_b = lp.b_key.row_len();
    let t_cols_per_vector = n_a
        .checked_mul(depth_open)
        .and_then(|len| len.checked_mul(live_fold_count))
        .ok_or_else(|| AkitaError::InvalidSetup("B vector width overflow".to_string()))?;
    let n_d_active = lp.n_d_active_for(relation_matrix_row_layout);
    let d_start = rows
        .checked_sub(n_d_active)
        .ok_or(AkitaError::InvalidProof)?;
    let d_physical_cols = num_claims
        .checked_mul(live_fold_count)
        .and_then(|n| n.checked_mul(depth_open))
        .ok_or_else(|| AkitaError::InvalidSetup("e width overflow".to_string()))?;
    let group = RelationMatrixGroupEvaluator {
        c_alphas,
        opening_a_evals,
        group_id: 0,
        e_col_offset: 0,
        num_claims,
        live_fold_count,
        fold_position_count,
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
    let setup_contribution_groups =
        build_setup_contribution_groups(&layout, opening_source_len, &groups)?;
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
        log_basis,
        layout,
        opening_source_len,
        setup_contribution_groups,
        setup_contribution_inputs,
        setup_contribution_static,
        flat_context: Some(FlatRelationContext {
            level_params: lp.clone(),
            row_coefficients: gamma.to_vec(),
            tau1: tau1.to_vec(),
            relation_matrix_row_layout,
            opening_ring_dim,
        }),
    })
}

fn reject_mixed_d_multi_chunk<const D: usize>(
    role_dims: CommitmentRingDims,
    layout: &WitnessLayout,
    context: &str,
) -> Result<(), AkitaError> {
    if layout.units().len() > layout.num_groups() && (role_dims.d_b() != D || role_dims.d_d() != D)
    {
        return Err(AkitaError::InvalidSetup(format!(
            "{context}: multi-chunk witness layout requires uniform ring dimensions across roles"
        )));
    }
    Ok(())
}

pub(crate) fn build_setup_contribution_groups<F: FieldCore>(
    layout: &WitnessLayout,
    opening_source_len: usize,
    groups: &[RelationMatrixGroupEvaluator<F>],
) -> Result<Vec<SetupContributionGroupInputs>, AkitaError> {
    groups
        .iter()
        .map(|group| {
            Ok(SetupContributionGroupInputs {
                group_id: group.group_id,
                e_col_offset: group.e_col_offset,
                num_claims: group.num_claims,
                live_fold_count: group.live_fold_count,
                fold_position_count: group.fold_position_count,
                depth_open: group.depth_open,
                depth_commit: group.depth_commit,
                depth_fold: group.depth_fold,
                log_basis: group.log_basis,
                n_a: group.n_a,
                n_b: group.n_b,
                t_cols_per_vector: group.t_cols_per_vector,
                a_row_start: group.a_row_start,
                b_row_start: group.b_row_start,
                layout: std::sync::Arc::new(layout.clone()),
                opening_source_len,
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
    _x_challenges: &[E],
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
        .all(|group| group.depth_fold == first.depth_fold && group.log_basis == first.log_basis)
    {
        cache.fold_gadget = Some(gadget_row_scalars::<F>(first.depth_fold, first.log_basis));
    }

    Ok(cache)
}

impl<E: FieldCore> RelationMatrixEvaluator<E> {
    /// Evaluate the canonical relation weights directly in the flattened
    /// opening domain, without materializing its padded Boolean suffix.
    pub fn eval_flat_at_point<F, const D: usize>(
        &self,
        point: &[E],
        setup: &AkitaExpandedSetup<F>,
        instance: &RingRelationInstance<F>,
        alpha: E,
        setup_claim: Option<E>,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F> + MulBaseUnreduced<F>,
    {
        let context = self.flat_context.as_ref().ok_or(AkitaError::InvalidProof)?;
        if context.opening_ring_dim == D && self.role_dims == CommitmentRingDims::uniform(D) {
            let coefficient_bits = D.trailing_zeros() as usize;
            if point.len() < coefficient_bits {
                return Err(AkitaError::InvalidProof);
            }
            let (coefficient_point, column_point) = point.split_at(coefficient_bits);
            let alpha_evals = scalar_powers(alpha, D);
            let coefficient_eval =
                akita_sumcheck::multilinear_eval(&alpha_evals, coefficient_point)?;
            return Ok(coefficient_eval
                * self.eval_at_point::<F, D>(column_point, setup, alpha, setup_claim)?);
        }
        if setup_claim.is_some() {
            return Err(AkitaError::InvalidProof);
        }
        eval_relation_weight_at_point::<F, E>(
            setup,
            instance,
            alpha,
            &scalar_powers(alpha, D),
            self.role_dims,
            &context.level_params,
            &context.tau1,
            &context.row_coefficients,
            context.relation_matrix_row_layout,
            self.opening_source_len,
            context.opening_ring_dim,
            point,
        )
    }

    fn group_units(
        &self,
        group: &RelationMatrixGroupEvaluator<E>,
    ) -> Result<Vec<&WitnessUnitLayout>, AkitaError> {
        self.layout.units_for_group(group.group_id)
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
        let d_b = self.role_dims.d_b();
        let d_d = self.role_dims.d_d();
        let alpha_pows_a = scalar_powers(alpha, D);
        let alpha_pows_b_storage;
        let alpha_pows_b: &[E] = if d_b == D {
            &alpha_pows_a
        } else {
            alpha_pows_b_storage = scalar_powers(alpha, d_b);
            &alpha_pows_b_storage
        };
        let alpha_pows_d_storage;
        let alpha_pows_d: &[E] = if d_d == D {
            &alpha_pows_a
        } else if d_d == d_b {
            alpha_pows_b
        } else {
            alpha_pows_d_storage = scalar_powers(alpha, d_d);
            &alpha_pows_d_storage
        };

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
                let units = self.group_units(group)?;
                validate_log_basis(group.log_basis)?;

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

                let consistency_weight = self.setup_contribution_inputs.eq_tau1[0];
                let a_row_end = group
                    .a_row_start
                    .checked_add(group.n_a)
                    .ok_or_else(|| AkitaError::InvalidSetup("A rows overflow".into()))?;
                let a_row_weights = self
                    .setup_contribution_inputs
                    .eq_tau1
                    .get(group.a_row_start..a_row_end)
                    .ok_or(AkitaError::InvalidProof)?;
                for unit in units {
                    for claim in 0..group.num_claims {
                        for global_block in unit.global_fold_range() {
                            let challenge = group.c_alphas.eval_at::<F>(
                                claim,
                                global_block,
                                group.live_fold_count,
                            )?;
                            for (digit, &gadget) in g_open.iter().enumerate() {
                                let e_index = self.layout.e_index(
                                    unit,
                                    group.num_claims,
                                    group.depth_open,
                                    claim,
                                    global_block,
                                    digit,
                                )?;
                                let e_opening_index = akita_types::checked_opening_source_index(
                                    self.opening_source_len,
                                    e_index,
                                )?;
                                e_structured_contribution +=
                                    eq_eval_at_index(x_challenges, e_opening_index)
                                        * consistency_weight
                                        * challenge
                                        * E::lift_base(gadget);
                                for (a_row, &row_weight) in a_row_weights.iter().enumerate() {
                                    let t_index = self.layout.t_index(
                                        unit,
                                        group.num_claims,
                                        group.n_a,
                                        group.depth_open,
                                        claim,
                                        global_block,
                                        a_row,
                                        digit,
                                    )?;
                                    let t_opening_index =
                                        akita_types::checked_opening_source_index(
                                            self.opening_source_len,
                                            t_index,
                                        )?;
                                    t_structured_contribution +=
                                        eq_eval_at_index(x_challenges, t_opening_index)
                                            * row_weight
                                            * challenge
                                            * E::lift_base(gadget);
                                }
                            }
                        }
                    }
                    for (position, &opening_a) in group.opening_a_evals.iter().enumerate() {
                        for (commit_digit, &commit) in g_commit.iter().enumerate() {
                            let mut z_weight = E::zero();
                            for (fold_digit, &fold) in fold_gadget.iter().enumerate() {
                                let z_index = self.layout.z_index(
                                    unit,
                                    group.opening_a_evals.len(),
                                    group.depth_commit,
                                    group.depth_fold,
                                    position,
                                    commit_digit,
                                    fold_digit,
                                )?;
                                let z_opening_index = akita_types::checked_opening_source_index(
                                    self.opening_source_len,
                                    z_index,
                                )?;
                                z_weight -= eq_eval_at_index(x_challenges, z_opening_index)
                                    * E::lift_base(fold);
                            }
                            z_structured_contribution +=
                                z_weight * consistency_weight * opening_a * E::lift_base(commit);
                        }
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
                alpha_pows_b,
                alpha_pows_d,
                fold_gadget,
                setup,
            )?
        };

        let r_contribution = {
            let r_gadget =
                gadget_row_scalars::<F>(r_decomp_levels::<F>(self.log_basis), self.log_basis);
            let alpha_pow_d = *alpha_pows_d.get(d_d - 1).ok_or(AkitaError::InvalidProof)?;
            let denom = alpha_pow_d * alpha + E::one();
            let offset_r = self.layout.r_offset();
            compute_r_contribution(self, x_challenges, offset_r, denom, &r_gadget)?
        };

        Ok(e_structured_contribution
            + t_structured_contribution
            + z_structured_contribution
            + setup_contribution
            + r_contribution)
    }
}
