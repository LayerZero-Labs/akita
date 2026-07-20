//! Verifier-side ring-switch replay.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::scalar_powers;
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, MulBase, MulBaseUnreduced,
    RandomSampling,
};
use akita_transcript::labels::{CHALLENGE_RING_SWITCH, CHALLENGE_TAU0, CHALLENGE_TAU1};
use akita_transcript::{sample_ext_challenge, Transcript};
use akita_types::{
    build_relation_weight_events, shared_setup_fold_gadget, validate_role_dispatch,
    AkitaExpandedSetup, CommitmentRingDims, FpExtEncoding, LevelParams, OpeningClaimsLayout,
    RelationMatrixRowLayout, RelationSetupSource, RelationWeightEventInputs, RingRelationInstance,
    RingRole, SetupContributionGroupInputs, SetupContributionPlan, WitnessLayout,
};
use std::sync::Arc;

use super::validate_log_basis;
#[cfg(test)]
mod tests;

/// Verifier-side ring-switch output, carrying only the data needed to replay
/// the fused stage-1/stage-2 checks.
pub(crate) struct RingSwitchVerifyOutput<E: FieldCore> {
    /// Checked geometry for semantic relation-weight evaluation.
    pub relation_weight_evaluator: RelationWeightEvaluator<E>,
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
    relation_weight_evaluator: RelationWeightEvaluator<E>,
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
            relation_weight_evaluator: self.relation_weight_evaluator,
            col_bits: self.col_bits,
            ring_bits: self.ring_bits,
            tau0,
            tau1: self.tau1,
            b: self.b,
            alpha: self.alpha,
        })
    }
}

/// Checked geometry for semantic relation-weight and setup evaluation.
///
/// The relation formula remains in `akita-types`; this verifier state retains
/// only the row equality table, setup-offload group geometry, and the checked
/// context needed to invoke that shared authority at the final point.
#[derive(Clone)]
pub struct RelationWeightEvaluator<F: FieldCore> {
    pub(crate) role_dims: CommitmentRingDims,
    pub(crate) setup_groups: Vec<SetupContributionGroupInputs>,
    pub(crate) eq_tau1: Arc<[F]>,
    pub(crate) context: FlatRelationContext<F>,
}

#[derive(Clone)]
pub(crate) struct FlatRelationContext<F: FieldCore> {
    pub(crate) level_params: LevelParams,
    pub(crate) opening_batch: OpeningClaimsLayout,
    pub(crate) witness_layout: Arc<WitnessLayout>,
    pub(crate) opening_source_len: usize,
    pub(crate) row_coefficients: Vec<F>,
    pub(crate) tau1: Vec<F>,
    pub(crate) relation_matrix_row_layout: RelationMatrixRowLayout,
    pub(crate) opening_ring_dim: usize,
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

/// Replay the verifier half of ring switching after the caller has absorbed
/// the schedule-selected outgoing witness binding.
#[tracing::instrument(skip_all, name = "ring_switch_verifier")]
#[inline(never)]
pub(crate) fn ring_switch_verifier<F, E, T, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E>,
    w_len: usize,
    transcript: &mut T,
    relation_matrix_row_layout: RelationMatrixRowLayout,
) -> Result<RingSwitchVerifyOutput<E>, AkitaError>
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

    let alpha: E = {
        let _span = tracing::info_span!("ring_switch_transcript_challenges").entered();
        sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_RING_SWITCH)
    };

    let num_claims = relation.opening_batch().num_total_polynomials();
    // Validate each group's opening/multiplier point against that group's own
    // block geometry (final vs frozen-precommit). For a scalar batch this is the
    // single group at `lp`'s geometry, byte-identical to the historical check.
    for group_index in 0..opening_batch.num_groups() {
        let group_lp = lp.group_params(opening_batch, group_index)?;
        let opening_point = relation.group_opening_point(group_index)?;
        if opening_point.position_weights.len() != group_lp.num_positions_per_block()
            || opening_point.live_block_weights.len() != group_lp.num_live_blocks()
        {
            return Err(AkitaError::InvalidProof);
        }
        let multiplier_point = relation.group_ring_multiplier_point(group_index)?;
        if multiplier_point.position_len() != group_lp.num_positions_per_block()
            || multiplier_point.fold_len() != group_lp.num_live_blocks()
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
    let x_capacity = akita_types::opening_domain_len(replay.opening_source_len)?;
    // Mirror the prover's ring-switch geometry (see `ring_switch_finalize`):
    // uniform role dimensions expose the separable (x, y) opening domain, while
    // non-uniform roles fall back to the flattened single domain.
    let uniform = relation.role_dims() == CommitmentRingDims::uniform(opening_ring_dim);
    let (col_bits, ring_bits) = if uniform {
        (
            x_capacity.trailing_zeros() as usize,
            opening_ring_dim.trailing_zeros() as usize,
        )
    } else {
        let opening_field_len = x_capacity
            .checked_mul(opening_ring_dim)
            .ok_or(AkitaError::InvalidProof)?;
        (opening_field_len.trailing_zeros() as usize, 0usize)
    };
    let num_sc_vars = col_bits + ring_bits;
    let num_i =
        lp.relation_row_index_num_vars_for_layout(relation_matrix_row_layout, opening_batch)?;

    let (tau0, tau1) = {
        let _span = tracing::info_span!(
            "ring_switch_transcript_challenges",
            tau0_len = num_sc_vars,
            tau1_len = num_i
        )
        .entered();
        let tau0 = match relation_matrix_row_layout {
            RelationMatrixRowLayout::WithDBlock => Some(
                (0..num_sc_vars)
                    .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU0))
                    .collect(),
            ),
            RelationMatrixRowLayout::WithoutCommitmentBlocks => None,
        };
        let tau1 = (0..num_i)
            .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU1))
            .collect::<Vec<_>>();
        (tau0, tau1)
    };
    if gamma.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let relation_weight_evaluator =
        prepare_relation_weight_evaluator::<F, E, D>(replay, &tau1, Some(num_ring_elems))?;
    RingSwitchVerifyCoreOutput {
        relation_weight_evaluator,
        col_bits,
        ring_bits,
        tau0,
        tau1,
        b: 1usize
            .checked_shl(lp.log_basis)
            .ok_or_else(|| AkitaError::InvalidSetup("basis size overflow".to_string()))?,
        alpha,
    }
    .into_intermediate()
}

/// Prepare relation-weight evaluator state from a fixed
/// [`RingRelationInstance`] and transcript-sampled row coefficients.
///
/// # Errors
///
/// Returns an error if claim, group, row, or witness geometry is inconsistent.
#[tracing::instrument(skip_all, name = "prepare_relation_weight_evaluator")]
pub fn prepare_relation_weight_evaluator<F, E, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E>,
    tau1: &[E],
    witness_ring_len: Option<usize>,
) -> Result<RelationWeightEvaluator<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F> + MulBaseUnreduced<F>,
{
    let relation = replay.relation;
    let level_params = replay.lp;
    let opening_batch = relation.opening_batch();
    let relation_matrix_row_layout = relation.relation_matrix_row_layout();
    let witness_layout = relation.segment_layout(level_params, witness_ring_len)?;
    if witness_layout.total_len() > replay.opening_source_len {
        return Err(AkitaError::InvalidProof);
    }
    reject_mixed_d_multi_chunk::<D>(
        level_params.role_dims(),
        &witness_layout,
        "prepare_relation_weight_evaluator",
    )?;
    level_params.validate_opening_batch(opening_batch)?;
    validate_role_dispatch::<D>(relation.role_dims(), RingRole::Inner)?;
    if replay.row_coefficients.len() != opening_batch.num_total_polynomials() {
        return Err(AkitaError::InvalidProof);
    }

    let rows = level_params
        .relation_matrix_row_count_for(opening_batch.num_groups(), relation_matrix_row_layout)?;
    let eq_tau1: Arc<[E]> = EqPolynomial::evals_prefix(tau1, rows)?.into();
    let group_order = opening_batch.root_group_order()?;
    if group_order.iter().any(|&group_index| {
        witness_layout.num_chunks_for_group(group_index) != level_params.witness_chunk.num_chunks
    }) {
        return Err(AkitaError::InvalidSetup(
            "multi-group witness layout does not match root group order".to_string(),
        ));
    }

    let mut setup_groups = Vec::with_capacity(group_order.len());
    for group_index in group_order {
        let group_params = level_params.group_params(opening_batch, group_index)?;
        let group_layout = opening_batch.group_layout(group_index)?;
        validate_log_basis(group_params.log_basis())?;
        let a_rows =
            level_params.a_row_range(opening_batch, group_index, relation_matrix_row_layout)?;
        let b_rows = level_params.commitment_row_range(
            opening_batch,
            group_index,
            relation_matrix_row_layout,
        )?;
        if a_rows.len() != group_params.a_rows_len() || b_rows.len() != group_params.b_rows_len() {
            return Err(AkitaError::InvalidSetup(
                "relation row ranges do not match group matrix heights".to_string(),
            ));
        }
        setup_groups.push(SetupContributionGroupInputs {
            group_id: group_index,
            num_claims: group_layout.num_polynomials(),
            depth_fold: level_params.num_digits_fold_for_params(
                group_params,
                group_layout.num_polynomials(),
                level_params.field_bits_for_cache(),
            )?,
            a_row_start: a_rows.start,
            b_row_start: b_rows.start,
        });
    }

    Ok(RelationWeightEvaluator {
        role_dims: relation.role_dims(),
        setup_groups,
        eq_tau1,
        context: FlatRelationContext {
            level_params: level_params.clone(),
            opening_batch: opening_batch.clone(),
            witness_layout: Arc::new(witness_layout),
            opening_source_len: replay.opening_source_len,
            row_coefficients: replay.row_coefficients.to_vec(),
            tau1: tau1.to_vec(),
            relation_matrix_row_layout,
            opening_ring_dim: replay.opening_ring_dim,
        },
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

impl<E: FieldCore> RelationWeightEvaluator<E> {
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
        let context = &self.context;
        validate_role_dispatch::<D>(self.role_dims, RingRole::Inner)?;
        let deferred_setup_evaluation = if let Some(claim) = setup_claim {
            if context.opening_ring_dim != D || self.role_dims != CommitmentRingDims::uniform(D) {
                return Err(AkitaError::InvalidProof);
            }
            let coefficient_bits = D.trailing_zeros() as usize;
            let coefficient_point = point
                .get(..coefficient_bits)
                .ok_or(AkitaError::InvalidProof)?;
            let coefficient_factor =
                akita_sumcheck::multilinear_eval(&scalar_powers(alpha, D), coefficient_point)?;
            Some(claim * coefficient_factor)
        } else {
            None
        };
        let setup_source = if deferred_setup_evaluation.is_some() {
            RelationSetupSource::DeferredClaim
        } else {
            RelationSetupSource::Matrix(setup)
        };
        build_relation_weight_events(RelationWeightEventInputs {
            setup: setup_source,
            instance,
            alpha,
            level_params: &context.level_params,
            relation_row_point: &context.tau1,
            claim_coefficients: &context.row_coefficients,
            relation_matrix_row_layout: context.relation_matrix_row_layout,
            opening_source_len: context.opening_source_len,
            opening_ring_dim: context.opening_ring_dim,
        })?
        .evaluate_at_point(point, deferred_setup_evaluation)
    }

    pub(crate) fn setup_contribution_fold_gadget<F>(&self) -> Result<Option<Vec<F>>, AkitaError>
    where
        F: FieldCore + CanonicalField,
    {
        let context = &self.context;
        Ok(shared_setup_fold_gadget(
            &context.level_params,
            &context.opening_batch,
            &self.setup_groups,
        ))
    }

    pub(crate) fn setup_contribution_plan<F>(
        &self,
        x_challenges: &[E],
        fold_gadget: Option<&[F]>,
    ) -> Result<SetupContributionPlan<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: MulBase<F>,
    {
        let context = &self.context;
        SetupContributionPlan::prepare::<F>(
            &context.level_params,
            &context.opening_batch,
            context.relation_matrix_row_layout,
            self.eq_tau1.clone(),
            &context.witness_layout,
            context.opening_source_len,
            &self.setup_groups,
            x_challenges,
            fold_gadget,
            self.role_dims,
        )
    }

    pub(crate) fn setup_index_weight_evaluator<F>(
        &self,
        plan: &SetupContributionPlan<E>,
        tau1: &[E],
        x_challenges: &[E],
        fold_gadget: &[F],
        alpha: E,
    ) -> Result<akita_types::SetupIndexWeightEvaluator<E>, AkitaError>
    where
        F: FieldCore,
        E: MulBase<F>,
    {
        let context = &self.context;
        akita_types::SetupIndexWeightEvaluator::new::<F>(
            plan,
            &context.level_params,
            &context.opening_batch,
            context.relation_matrix_row_layout,
            &context.witness_layout,
            context.opening_source_len,
            &self.setup_groups,
            tau1,
            x_challenges,
            fold_gadget,
            alpha,
        )
    }
}
