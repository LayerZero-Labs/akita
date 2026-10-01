//! Prepared ring-switch relation evaluator and its public replay inputs.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::scalar_powers;
use akita_challenges::Challenges;
use akita_error::AkitaError;
use akita_params::{
    dispatch_for_field, CommittedGroupParams, OpeningClaimsLayout, RelationAddressGeometry,
    RingRelationMode, WitnessLayout,
};
use akita_types::{
    shared_setup_fold_gadget, AkitaExpandedSetup, FpExtEncoding, OpeningFamily,
    PreparedRelationAddress, PreparedRingMultiplier, RingMultiplierOpeningPoint,
    RingRelationGroupOpeningView, RingRelationInstance, SetupContributionGroupInputs,
    SetupContributionPlan,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};
use std::sync::Arc;

pub(crate) mod evaluation;
pub(crate) mod evaluation_trace;
mod prepared_point;
#[cfg(test)]
mod tests;

#[inline]
pub(crate) fn validate_log_basis(log_basis: u32) -> Result<(), AkitaError> {
    if log_basis == 0 || log_basis >= 128 {
        return Err(AkitaError::InvalidSetup(
            "log_basis must be in 1..128 for verifier gadget evaluation".to_string(),
        ));
    }
    Ok(())
}

/// Precomputed challenge-derived data for prepared relation-matrix MLE evaluation.
///
/// Stores only data that cannot be derived from context at evaluation time:
/// alpha-evaluated folding challenges and the tau1 eq-polynomial expansion.
/// Everything else is passed by reference at evaluation time to avoid
/// duplicating setup matrix views, opening points, and gadget vectors.
#[derive(Clone)]
pub struct RelationMatrixEvaluator<F: Field> {
    pub(crate) relation_address_geometry: RelationAddressGeometry,
    pub(crate) groups: PreparedRelationGroups<F>,
    /// Batch-wide basis used by the shared r-tail.
    pub(crate) log_basis: u32,
    pub(crate) eq_tau1: Arc<[F]>,
    pub(crate) flat_context: FlatRelationContext,
}

#[derive(Clone)]
pub(crate) struct FlatRelationContext {
    pub(crate) level_params: CommittedGroupParams,
    pub(crate) opening_batch: OpeningClaimsLayout,
    pub(crate) witness_layout: Arc<WitnessLayout>,
    pub(crate) extension_degree: usize,
}

#[derive(Clone)]
pub(crate) struct RelationMatrixGroupEvaluator<M> {
    pub(crate) multipliers: M,
    pub(crate) group_id: usize,
    pub(crate) num_claims: usize,
    pub(crate) depth_fold: usize,
    pub(crate) a_row_start: usize,
    pub(crate) b_row_start: usize,
}

#[derive(Clone)]
pub(crate) struct QuotientRelationMultipliers<E: Field> {
    pub(crate) c_alphas: Vec<E>,
    pub(crate) opening_a_evals: Vec<E>,
}

#[derive(Clone)]
pub(crate) struct ReducedRelationMultipliers<E: Field> {
    pub(crate) challenges: Challenges,
    pub(crate) opening: PreparedRingMultiplier<E>,
}

#[derive(Clone)]
pub(crate) enum PreparedRelationGroups<E: Field> {
    QuotientLift(Vec<RelationMatrixGroupEvaluator<QuotientRelationMultipliers<E>>>),
    ReducedEvaluation(Vec<RelationMatrixGroupEvaluator<ReducedRelationMultipliers<E>>>),
}

/// Fixed public relation inputs for verifier ring-switch replay.
pub(crate) struct RingSwitchReplay<'a, F: Field, E> {
    pub setup: &'a AkitaExpandedSetup<F>,
    pub relation: &'a RingRelationInstance<F>,
    pub row_coefficients: &'a [E],
    pub lp: &'a CommittedGroupParams,
    pub opening_source_len: usize,
    pub opening_ring_dim: usize,
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
pub(crate) fn prepare_relation_matrix_evaluator<F, E>(
    replay: &RingSwitchReplay<'_, F, E>,
    alpha: E,
    tau1: &[E],
    witness_ring_len: Option<usize>,
) -> Result<RelationMatrixEvaluator<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + Ring + ExtField<F> + MulBaseUnreduced<F>,
{
    let relation = replay.relation;
    let lp = replay.lp;
    let opening_batch = relation.opening_batch();
    let layout = relation.segment_layout(lp, witness_ring_len)?;
    let relation_address_geometry = lp.relation_address_geometry(
        opening_batch,
        relation.extension_degree(),
        replay.opening_ring_dim,
        layout.live_coeff_len(),
    )?;
    let opening_capacity = replay
        .opening_source_len
        .checked_mul(replay.opening_ring_dim)
        .ok_or_else(|| AkitaError::InvalidSetup("opening capacity overflow".into()))?;
    if layout.live_coeff_len() > opening_capacity {
        return Err(AkitaError::InvalidProof);
    }
    let rows = lp.relation_matrix_row_count(opening_batch.num_groups())?;
    prepare_relation_matrix_evaluator_groups::<F, E>(
        replay,
        alpha,
        tau1,
        layout,
        rows,
        relation_address_geometry,
        relation.extension_degree(),
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_relation_matrix_evaluator_groups<F, E>(
    replay: &RingSwitchReplay<'_, F, E>,
    alpha: E,
    tau1: &[E],
    layout: WitnessLayout,
    rows: usize,
    relation_address_geometry: RelationAddressGeometry,
    extension_degree: usize,
) -> Result<RelationMatrixEvaluator<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + Ring + ExtField<F> + MulBaseUnreduced<F>,
{
    let relation = replay.relation;
    let lp = replay.lp;
    let opening_batch = relation.opening_batch();
    lp.validate_opening_batch(opening_batch)?;
    if replay.row_coefficients.len() != opening_batch.num_total_polynomials() {
        return Err(AkitaError::InvalidProof);
    }

    let eq_tau1: std::sync::Arc<[E]> = EqPolynomial::evals_prefix(tau1, rows)?.into();

    let order = opening_batch.root_group_order()?;
    if order
        .iter()
        .any(|&group_index| layout.num_chunks_for_group(group_index) != lp.witness_chunk.num_chunks)
    {
        return Err(AkitaError::InvalidSetup(
            "multi-group witness layout does not match root group order".to_string(),
        ));
    }

    struct GroupInputs<F: Field> {
        group_index: usize,
        d_a: usize,
        num_claims: usize,
        num_live_blocks: usize,
        num_positions: usize,
        depth_fold: usize,
        a_row_start: usize,
        b_row_start: usize,
        challenges: Challenges,
        opening: OpeningFamily<RingMultiplierOpeningPoint<F>, ()>,
    }

    let group_inputs = order
        .iter()
        .map(|&group_index| -> Result<GroupInputs<F>, AkitaError> {
            let group_lp = lp.group_params(opening_batch, group_index)?;
            let group_role_dims = lp.group_role_dims(opening_batch, group_index)?;
            let group_layout = opening_batch.group_layout(group_index)?;
            let k_g = group_layout.num_polynomials();
            let num_live_blocks = group_lp.num_live_blocks();
            let num_positions_per_block = group_lp.num_positions_per_block();
            let depth_witness = group_lp.num_digits_inner();
            let depth_fold = group_lp.num_digits_fold();
            let log_basis_inner = group_lp.log_basis_inner();
            let log_basis_outer = group_lp.log_basis_outer();
            let log_basis_open = group_lp.log_basis_open();
            validate_log_basis(log_basis_inner)?;
            validate_log_basis(log_basis_outer)?;
            validate_log_basis(log_basis_open)?;
            let n_a = group_lp.a_rows_len();
            let n_b = group_lp.logical_b_rows_len()?;
            let inner_width = group_lp.a_col_len();
            let expected_inner_width = num_positions_per_block
                .checked_mul(depth_witness)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("multi-group inner width overflow".to_string())
                })?;
            if inner_width < expected_inner_width {
                return Err(AkitaError::InvalidSetup(
                    "multi-group A-key column width is too small".to_string(),
                ));
            }

            let opening = match relation.group_opening_view(group_index)? {
                RingRelationGroupOpeningView::EvaluationTrace {
                    ring_multiplier_point,
                    ..
                } => OpeningFamily::EvaluationTrace(ring_multiplier_point.clone()),
                RingRelationGroupOpeningView::SubringCoefficientPacking { .. } => {
                    OpeningFamily::SubringCoefficientPacking(())
                }
            };
            if let OpeningFamily::EvaluationTrace(ring_multiplier_point) = &opening {
                if ring_multiplier_point.position_len() != num_positions_per_block
                    || ring_multiplier_point.fold_len() != num_live_blocks
                {
                    return Err(AkitaError::InvalidProof);
                }
            }

            let total_blocks = k_g.checked_mul(num_live_blocks).ok_or_else(|| {
                AkitaError::InvalidSetup("multi-group block count overflow".to_string())
            })?;
            let challenges = relation.group_ambient_a_challenges(group_index)?;
            if challenges.len() != total_blocks {
                return Err(AkitaError::InvalidSize {
                    expected: total_blocks,
                    actual: challenges.len(),
                });
            }
            let a_range = lp.a_row_range(opening_batch, group_index)?;
            let b_range = lp.commitment_row_range(opening_batch, group_index)?;
            if a_range.len() != n_a || b_range.len() != n_b {
                return Err(AkitaError::InvalidSetup(
                    "multi-group row ranges do not match group matrix heights".to_string(),
                ));
            }

            Ok(GroupInputs {
                group_index,
                d_a: group_role_dims.d_a(),
                num_claims: k_g,
                num_live_blocks,
                num_positions: num_positions_per_block,
                depth_fold,
                a_row_start: a_range.start,
                b_row_start: b_range.start,
                challenges: challenges.clone(),
                opening,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let groups = match lp.ring_relation_mode {
        RingRelationMode::QuotientLift => {
            let groups = group_inputs
                .into_iter()
                .map(|group| {
                    let multipliers = dispatch_for_field!(
                        akita_params::ProtocolDispatchSlot::Role(akita_params::RingRole::Inner),
                        F,
                        group.d_a,
                        |D_GROUP| {
                            let alpha_pows = scalar_powers(alpha, D_GROUP);
                            let c_alphas = prepare_challenge_evals::<F, E>(
                                &group.challenges,
                                &alpha_pows,
                                group.num_claims,
                                group.num_live_blocks,
                            )?;
                            let opening_a_evals = match &group.opening {
                                OpeningFamily::EvaluationTrace(point) => (0..group.num_positions)
                                    .map(|index| point.eval_position_at::<E>(index, &alpha_pows))
                                    .collect::<Result<Vec<_>, _>>()?,
                                OpeningFamily::SubringCoefficientPacking(()) => {
                                    vec![E::zero(); group.num_positions]
                                }
                            };
                            Ok::<_, AkitaError>(QuotientRelationMultipliers {
                                c_alphas,
                                opening_a_evals,
                            })
                        }
                    )?;
                    Ok(RelationMatrixGroupEvaluator {
                        multipliers,
                        group_id: group.group_index,
                        num_claims: group.num_claims,
                        depth_fold: group.depth_fold,
                        a_row_start: group.a_row_start,
                        b_row_start: group.b_row_start,
                    })
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            PreparedRelationGroups::QuotientLift(groups)
        }
        RingRelationMode::ReducedEvaluation => {
            let groups = group_inputs
                .into_iter()
                .map(|group| {
                    let opening = match group.opening {
                        OpeningFamily::EvaluationTrace(point) => {
                            point.prepare_functional_multiplier::<E>()
                        }
                        OpeningFamily::SubringCoefficientPacking(()) => {
                            return Err(AkitaError::InvalidSetup(
                                "reduced relation requires ring-multiplier openings".into(),
                            ));
                        }
                    };
                    Ok(RelationMatrixGroupEvaluator {
                        multipliers: ReducedRelationMultipliers {
                            challenges: group.challenges,
                            opening,
                        },
                        group_id: group.group_index,
                        num_claims: group.num_claims,
                        depth_fold: group.depth_fold,
                        a_row_start: group.a_row_start,
                        b_row_start: group.b_row_start,
                    })
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            PreparedRelationGroups::ReducedEvaluation(groups)
        }
    };

    let layout = Arc::new(layout);

    Ok(RelationMatrixEvaluator {
        relation_address_geometry,
        groups,
        log_basis: lp.open().digits.log_basis,
        eq_tau1,
        flat_context: FlatRelationContext {
            level_params: lp.clone(),
            opening_batch: opening_batch.clone(),
            witness_layout: layout,
            extension_degree,
        },
    })
}

fn prepare_challenge_evals<F, E>(
    challenges: &Challenges,
    alpha_pows: &[E],
    num_claims: usize,
    num_live_blocks: usize,
) -> Result<Vec<E>, AkitaError>
where
    F: Field + Ring,
    E: Field + ExtField<F>,
{
    if challenges.num_claims() != num_claims {
        return Err(AkitaError::InvalidSize {
            expected: num_claims,
            actual: challenges.num_claims(),
        });
    }
    if challenges.num_live_blocks_per_claim() != num_live_blocks {
        return Err(AkitaError::InvalidSize {
            expected: num_live_blocks,
            actual: challenges.num_live_blocks_per_claim(),
        });
    }
    challenges.evals_at_pows::<F, E>(alpha_pows)
}

pub(crate) fn setup_contribution_group_inputs<F: Field>(
    groups: &PreparedRelationGroups<F>,
) -> Vec<SetupContributionGroupInputs> {
    fn collect<M>(groups: &[RelationMatrixGroupEvaluator<M>]) -> Vec<SetupContributionGroupInputs> {
        groups
            .iter()
            .map(|group| SetupContributionGroupInputs {
                group_id: group.group_id,
                num_claims: group.num_claims,
                depth_fold: group.depth_fold,
                a_row_start: group.a_row_start,
                b_row_start: group.b_row_start,
            })
            .collect()
    }
    match groups {
        PreparedRelationGroups::QuotientLift(groups) => collect(groups),
        PreparedRelationGroups::ReducedEvaluation(groups) => collect(groups),
    }
}

impl<E: Field> RelationMatrixEvaluator<E> {
    pub(crate) fn setup_contribution_inputs(&self) -> Vec<SetupContributionGroupInputs> {
        setup_contribution_group_inputs(&self.groups)
    }

    pub(crate) fn setup_contribution_fold_gadget<F>(&self) -> Result<Option<Vec<F>>, AkitaError>
    where
        F: Field + CanonicalEncoding,
    {
        let context = &self.flat_context;
        let setup_groups = self.setup_contribution_inputs();
        Ok(shared_setup_fold_gadget(
            &context.level_params,
            &context.opening_batch,
            &setup_groups,
        ))
    }

    pub(crate) fn setup_contribution_plan<F>(
        &self,
        relation_address: PreparedRelationAddress<E>,
        fold_gadget: Option<&[F]>,
    ) -> Result<SetupContributionPlan<E>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        let context = &self.flat_context;
        let setup_groups = self.setup_contribution_inputs();
        SetupContributionPlan::prepare::<F>(
            &context.level_params,
            &context.opening_batch,
            context.extension_degree,
            self.eq_tau1.clone(),
            &context.witness_layout,
            &setup_groups,
            relation_address,
            fold_gadget,
            self.relation_address_geometry,
        )
    }

    pub(crate) fn witness_layout(&self) -> Result<&WitnessLayout, AkitaError> {
        let context = &self.flat_context;
        Ok(&context.witness_layout)
    }
}
