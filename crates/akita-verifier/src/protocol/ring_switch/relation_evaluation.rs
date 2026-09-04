//! Mode-bound prepared relation evaluation for every role geometry.
//!
//! Quotient lifting factors the common low alpha coordinates and evaluates its
//! explicit quotient tail. Reduced evaluation prepares exact terminal
//! coefficient functionals, performs the same structured/direct setup work,
//! and returns that already-complete flat MLE without either lifted-only step.

use super::{
    prepared_relation_point::{PreparedLiftedRelationPoint, PreparedReducedRelationPoint},
    PreparedRelationGroups, QuotientRelationMultipliers, ReducedRelationMultipliers,
    RelationMatrixEvaluator, RelationMatrixGroupEvaluator,
};
use akita_algebra::offset_eq::OffsetEqWindow;
use akita_error::AkitaError;
use akita_types::{
    gadget_row_scalars, r_decomp_levels, AkitaExpandedSetup, FpExtEncoding,
    PreparedRelationAddress, RelationAddressGeometry, RelationQuotientLayout, RelationRowFamily,
    RelationWitnessGeometry, SetupContributionPlan,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};

pub(super) fn evaluate_relation_at_point<F, E>(
    evaluator: &RelationMatrixEvaluator<E>,
    point: &[E],
    setup: &AkitaExpandedSetup<F>,
    alpha: E,
) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + Ring + ExtField<F> + MulBaseUnreduced<F>,
{
    let prepared = {
        let _span = tracing::info_span!("relation_coefficient_functional_preparation").entered();
        let mut prepared = PreparedDirectRelation::prepare::<F>(evaluator, point, alpha)?;
        prepared.materialize_setup()?;
        prepared
    };
    prepared.evaluate_materialized_direct::<F>(setup)
}

pub(super) fn evaluate_quotient_relation_with_deferred_setup<F, E>(
    evaluator: &RelationMatrixEvaluator<E>,
    point: &[E],
    _setup: &AkitaExpandedSetup<F>,
    alpha: E,
    setup_claim: E,
) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + ExtField<F> + MulBaseUnreduced<F>,
{
    let prepared = {
        let _span = tracing::info_span!("relation_coefficient_functional_preparation").entered();
        PreparedDirectRelation::prepare::<F>(evaluator, point, alpha)?
    };
    prepared.evaluate_deferred::<F>(setup_claim)
}

fn prepare_setup_plan<F, E>(
    evaluator: &RelationMatrixEvaluator<E>,
    address: &PreparedRelationAddress<E>,
) -> Result<SetupContributionPlan<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: Field + ExtField<F>,
{
    let fold_gadget = evaluator.setup_contribution_fold_gadget::<F>()?;
    let plan = {
        let _span = tracing::info_span!("relation_setup_plan").entered();
        let fold_gadget = fold_gadget.as_deref().unwrap_or(&[]);
        evaluator.setup_contribution_plan::<F>(
            address.clone(),
            (!fold_gadget.is_empty()).then_some(fold_gadget),
        )?
    };
    Ok(plan)
}

pub(super) enum PreparedDirectRelation<'a, E: Field> {
    Quotient {
        evaluator: &'a RelationMatrixEvaluator<E>,
        groups: &'a [RelationMatrixGroupEvaluator<QuotientRelationMultipliers<E>>],
        point: PreparedLiftedRelationPoint<E>,
        row_families: Vec<RelationRowFamily>,
        plan: SetupContributionPlan<E>,
    },
    Reduced {
        groups: &'a [RelationMatrixGroupEvaluator<ReducedRelationMultipliers<E>>],
        point: PreparedReducedRelationPoint<E>,
        plan: SetupContributionPlan<E>,
    },
}

impl<'a, E: Field> PreparedDirectRelation<'a, E> {
    pub(super) fn prepare<F>(
        evaluator: &'a RelationMatrixEvaluator<E>,
        point: &[E],
        alpha: E,
    ) -> Result<Self, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: FpExtEncoding<F> + Ring + ExtField<F> + MulBaseUnreduced<F>,
    {
        let context = &evaluator.flat_context;
        match &evaluator.groups {
            PreparedRelationGroups::QuotientLift(groups) => {
                if !matches!(
                    context.witness_layout.relation_quotient_layout(),
                    RelationQuotientLayout::QuotientLift { .. }
                ) || context
                    .level_params
                    .ring_relation_mode
                    .is_reduced_evaluation()
                {
                    return Err(AkitaError::InvalidSetup(
                        "quotient evaluator disagrees with the authenticated layout".into(),
                    ));
                }
                let row_families = RelationWitnessGeometry::for_level(
                    &context.level_params,
                    &context.opening_batch,
                    context.extension_degree,
                )?
                .rhs_layout()
                .row_families()?;
                let quotient_row_dims = row_families
                    .iter()
                    .filter(|family| {
                        !matches!(
                            family,
                            RelationRowFamily::CompressionF { .. }
                                | RelationRowFamily::CompressionH { .. }
                        )
                    })
                    .map(|family| family.geometry().polynomial_modulus_dimension())
                    .collect::<Vec<_>>();
                let point = PreparedLiftedRelationPoint::new(
                    point,
                    alpha,
                    evaluator.relation_address_geometry,
                    &quotient_row_dims,
                )?;
                let plan = prepare_setup_plan::<F, E>(evaluator, point.relation_address())?;
                Ok(Self::Quotient {
                    evaluator,
                    groups,
                    point,
                    row_families,
                    plan,
                })
            }
            PreparedRelationGroups::ReducedEvaluation(groups) => {
                if !matches!(
                    context.witness_layout.relation_quotient_layout(),
                    RelationQuotientLayout::ReducedEvaluation
                ) || !context
                    .level_params
                    .ring_relation_mode
                    .is_reduced_evaluation()
                {
                    return Err(AkitaError::InvalidSetup(
                        "reduced evaluator disagrees with the authenticated layout".into(),
                    ));
                }
                let point = PreparedReducedRelationPoint::new(
                    point,
                    alpha,
                    evaluator.relation_address_geometry,
                )?;
                let plan = prepare_setup_plan::<F, E>(evaluator, point.relation_address())?;
                Ok(Self::Reduced {
                    groups,
                    point,
                    plan,
                })
            }
        }
    }

    pub(super) fn materialize_setup(&mut self) -> Result<(), AkitaError> {
        let _span = tracing::info_span!("relation_setup_weights").entered();
        match self {
            Self::Quotient { point, plan, .. } => {
                plan.materialize_direct_scan(point.coefficient_functional())
            }
            Self::Reduced { point, plan, .. } => {
                plan.materialize_direct_scan(point.coefficient_functional())
            }
        }
    }

    #[cfg(any(test, feature = "benchmark-support"))]
    pub(super) fn setup_field_len(&self) -> usize {
        match self {
            Self::Quotient { plan, .. } | Self::Reduced { plan, .. } => {
                plan.projection_geometry().natural_field_len()
            }
        }
    }

    pub(super) fn evaluate_setup<F>(&self, setup: &AkitaExpandedSetup<F>) -> Result<E, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let _span = tracing::info_span!("relation_setup_scan").entered();
        match self {
            Self::Quotient { point, plan, .. } => {
                Ok(point.common_alpha_evaluation() * plan.evaluate_direct::<F>(setup)?)
            }
            Self::Reduced { plan, .. } => plan.evaluate_direct::<F>(setup),
        }
    }

    pub(super) fn evaluate_structured<F>(&self) -> Result<E, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: FpExtEncoding<F> + Ring + ExtField<F>,
    {
        let _span = tracing::info_span!("relation_structured_groups").entered();
        match self {
            Self::Quotient {
                groups,
                point,
                plan,
                ..
            } => {
                let structured = groups.iter().try_fold(E::zero(), |sum, group| {
                    Ok::<_, AkitaError>(
                        sum + plan.evaluate_structured_group::<F>(
                            group.group_id,
                            &group.multipliers.c_alphas,
                            &group.multipliers.opening_a_evals,
                            point.alpha(),
                        )?,
                    )
                })?;
                Ok(point.common_alpha_evaluation() * structured)
            }
            Self::Reduced { groups, plan, .. } => {
                groups.iter().try_fold(E::zero(), |sum, group| {
                    Ok(sum
                        + plan.evaluate_reduced_structured_group::<F>(
                            group.group_id,
                            &group.multipliers.challenges,
                            &group.multipliers.opening,
                        )?)
                })
            }
        }
    }

    pub(super) fn evaluate_quotient_tail<F>(&self) -> Result<E, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: FpExtEncoding<F> + Ring + ExtField<F>,
    {
        match self {
            Self::Quotient {
                evaluator,
                point,
                row_families,
                ..
            } => {
                let _span = tracing::info_span!("relation_quotient_tail").entered();
                Ok(point.common_alpha_evaluation()
                    * evaluate_quotient_tail::<F, E>(evaluator, point, row_families)?)
            }
            Self::Reduced { .. } => Ok(E::zero()),
        }
    }

    pub(super) fn evaluate_relation_weight<F>(&self) -> Result<E, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: FpExtEncoding<F> + Ring + ExtField<F>,
    {
        Ok(self.evaluate_structured::<F>()? + self.evaluate_quotient_tail::<F>()?)
    }

    fn evaluate_materialized_direct<F>(
        &self,
        setup: &AkitaExpandedSetup<F>,
    ) -> Result<E, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: FpExtEncoding<F> + Ring + ExtField<F> + MulBaseUnreduced<F>,
    {
        Ok(self.evaluate_relation_weight::<F>()? + self.evaluate_setup::<F>(setup)?)
    }

    fn evaluate_deferred<F>(self, setup_claim: E) -> Result<E, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: FpExtEncoding<F> + Ring + ExtField<F>,
    {
        let result = self.evaluate_relation_weight::<F>()?;
        let Self::Quotient {
            evaluator,
            point,
            plan,
            ..
        } = self
        else {
            return Err(AkitaError::InvalidProof);
        };
        let result = result + point.common_alpha_evaluation() * setup_claim;
        evaluator.cache_setup_contribution_plan(point.address_point(), plan)?;
        Ok(result)
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_quotient_tail<F, E>(
    evaluator: &RelationMatrixEvaluator<E>,
    prepared_point: &PreparedLiftedRelationPoint<E>,
    row_families: &[RelationRowFamily],
) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + Ring + ExtField<F>,
{
    let context = &evaluator.flat_context;
    let rows = row_families.len();
    if rows
        != context
            .level_params
            .relation_matrix_row_count(context.opening_batch.num_groups())?
    {
        return Err(AkitaError::InvalidSetup(
            "relation quotient row dimensions disagree with the matrix layout".into(),
        ));
    }
    let levels = r_decomp_levels::<F>(evaluator.log_basis);
    let quotient_gadget = gadget_row_scalars::<F>(levels, evaluator.log_basis);
    let mut evaluation = E::zero();
    for (row, family) in row_families.iter().enumerate() {
        if matches!(
            family,
            RelationRowFamily::CompressionF { .. }
                | RelationRowFamily::CompressionH { .. }
                | RelationRowFamily::Consistency {
                    opening_method: akita_types::OpeningMethod::SubringCoefficientPacking { .. },
                    ..
                }
        ) {
            continue;
        }
        let row_dimension = family.geometry().polynomial_modulus_dimension();
        let role_factors = prepared_point.for_dimension(row_dimension)?;
        let denominator = role_factors
            .powers
            .last()
            .copied()
            .ok_or(AkitaError::InvalidProof)?
            * prepared_point.alpha()
            + E::one();
        let row_weight = evaluator
            .eq_tau1
            .get(row)
            .copied()
            .ok_or(AkitaError::InvalidProof)?;
        let mut row_evaluation = E::zero();
        for (digit, &gadget) in quotient_gadget.iter().enumerate() {
            let physical_coefficient = context
                .witness_layout
                .r_coefficient_index(row, digit, 0, 0)?;
            let lane_start = canonical_relation_lane_index(
                evaluator.relation_address_geometry,
                physical_coefficient,
            )?;
            let lane_evaluation = evaluate_lane_segment(
                prepared_point.relation_address().equality_window(),
                lane_start,
                &role_factors.lane_powers,
            )?;
            row_evaluation += lane_evaluation.mul_base(gadget);
        }
        evaluation -= row_evaluation * row_weight * denominator;
    }
    Ok(evaluation)
}

fn evaluate_lane_segment<E: Field>(
    equality_window: &OffsetEqWindow<E>,
    lane_start: usize,
    lane_alpha_powers: &[E],
) -> Result<E, AkitaError> {
    lane_alpha_powers
        .iter()
        .enumerate()
        .try_fold(E::zero(), |sum, (lane, &alpha_power)| {
            let index = lane_start
                .checked_add(lane)
                .ok_or_else(|| AkitaError::InvalidSetup("relation lane address overflow".into()))?;
            Ok(sum + equality_window.eval(index) * alpha_power)
        })
}

fn canonical_relation_lane_index(
    geometry: RelationAddressGeometry,
    physical_coefficient: usize,
) -> Result<usize, AkitaError> {
    let coeff_count = geometry.relation_coefficient_block_len();
    if physical_coefficient >= geometry.digit_witness_domain().live_len()
        || !physical_coefficient.is_multiple_of(coeff_count)
    {
        return Err(AkitaError::InvalidProof);
    }
    Ok(physical_coefficient / coeff_count)
}
