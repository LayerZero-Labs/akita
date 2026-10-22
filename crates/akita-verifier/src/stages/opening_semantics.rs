//! Stage 2 opening-claim semantics: the authenticated coefficient-packing
//! openings, or the prepared evaluation-trace weight.

use crate::claims::ClaimReplay;
use crate::relation::evaluation_trace::prepare_evaluation_trace;
use crate::stages::opening_claims::{FoldPrefix, PreparedFoldOpeningPoint};
use crate::stages::relation_claim::RelationClaim;
use crate::stages::ring_switch::RingSwitchVerifyOutput;
use crate::stages::stage2::Stage2OpeningSemantics;
use akita_error::AkitaError;
use akita_params::{BasisMode, CommittedGroupParams, OpeningClaimsLayout};
use akita_types::{
    ensure_trace_stage2_supported, proof::relation::relation_row_weight, EvaluationTraceInputs,
    FpExtEncoding, OpeningFamily,
};
use jolt_algebra::{CanonicalDecode, CanonicalEncoding, ExtField, Field};

/// Fold-level inputs to the Stage 2 opening semantics.
pub(crate) struct OpeningSemanticsInput<'a, F: Field, E: Field, V = E> {
    pub(crate) lp: &'a CommittedGroupParams,
    pub(crate) opening_batch: &'a OpeningClaimsLayout,
    pub(crate) w_len: usize,
    pub(crate) evaluation_trace_basis: BasisMode,
    pub(crate) prefix: &'a FoldPrefix<F, E, V>,
}

/// Check the opening claims against the evaluation-batch target and prepare
/// the Stage 2 opening semantics.
pub(crate) fn prepare_opening_semantics<'a, F, E, G>(
    grinding: &mut G,
    input: &OpeningSemanticsInput<'_, F, E, G::Claim>,
    rs: &RingSwitchVerifyOutput<E>,
    relation: &'a RelationClaim<E>,
) -> Result<Stage2OpeningSemantics<'a, E, G::Claim>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + FpExtEncoding<F> + CanonicalDecode,
    G: ClaimReplay<F, E>,
{
    let trace_domain = rs.relation_address_geometry.digit_witness_domain();
    if trace_domain.live_len() != input.w_len {
        return Err(AkitaError::InvalidSize {
            expected: trace_domain.live_len(),
            actual: input.w_len,
        });
    }
    let prefix = input.prefix;
    let opening_semantics = if let Some(batch) = &relation.coefficient_packing_batch {
        if prefix
            .prepared_points
            .iter()
            .any(|point| matches!(point, PreparedFoldOpeningPoint::EvaluationTrace(_)))
        {
            return Err(AkitaError::InvalidProof);
        }
        let mut authenticated_total = G::constant(E::zero());
        let mut group_openings = Vec::with_capacity(batch.groups().len());
        for semantics in batch.groups() {
            let claim_range = semantics.group_claim_range();
            let openings = prefix
                .scalar_openings
                .get(claim_range.clone())
                .ok_or(AkitaError::InvalidProof)?;
            let coefficients = prefix
                .trace_claim_coefficients
                .get(claim_range)
                .ok_or(AkitaError::InvalidProof)?;
            let authenticated = openings
                .iter()
                .zip(coefficients)
                .fold(G::constant(E::zero()), |sum, (opening, &coefficient)| {
                    G::add(&sum, &G::scale(opening, coefficient))
                });
            authenticated_total = G::add(&authenticated_total, &authenticated);
            group_openings.push((semantics.group_index(), authenticated));
        }
        grinding.check_equal(&authenticated_total, &prefix.trace_eval_target)?;
        Stage2OpeningSemantics::packing::<F, G>(batch, &group_openings)?
    } else {
        let evaluation_trace_row = input.lp.evaluation_trace_row_index(input.opening_batch)?;
        let evaluation_trace_weight = relation_row_weight(evaluation_trace_row, &rs.tau1)?;
        ensure_trace_stage2_supported(<E as ExtField<F>>::DEGREE)?;
        let evaluation_trace_points = prefix
            .prepared_points
            .iter()
            .map(|point| match point {
                OpeningFamily::EvaluationTrace(point) => Ok(point.clone()),
                OpeningFamily::SubringCoefficientPacking(_) => Err(AkitaError::InvalidProof),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let evaluation_trace = prepare_evaluation_trace::<F, E>(&EvaluationTraceInputs {
            digit_witness_domain: trace_domain,
            relation_coefficient_block_len: rs
                .relation_address_geometry
                .relation_coefficient_block_len(),
            witness_layout: relation.range_image_plan.witness_layout(),
            level_params: input.lp,
            opening_batch: input.opening_batch,
            prepared_points: &evaluation_trace_points,
            claim_coefficients: &prefix.trace_claim_coefficients,
            basis: input.evaluation_trace_basis,
        })?;
        Stage2OpeningSemantics::evaluation_trace(
            evaluation_trace,
            evaluation_trace_weight,
            G::scale(&prefix.trace_eval_target, evaluation_trace_weight),
        )
    };
    Ok(opening_semantics)
}
