//! Setup-product sumcheck for a dense table against two disjoint factors.
//!
//! The table is laid out as `left * right_len + right`. The right factor is
//! bound first, then the left factor. This matches setup products of the form
//! `S(i, y) * setup_index_weight(i) * alpha(y)` without materializing the full
//! `setup_index_weight(i) * alpha(y)` table.

mod product_table;
#[cfg(test)]
mod session_tests;
mod utils;

use crate::opaque::OperationBinding;
use crate::opaque::{CpuBackend, OpaqueStage3Kernel, Stage3Request};
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::scalar_powers;
use jolt_poly::UnivariatePoly;

use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::{
    ensure_setup_envelope, setup_prefix_coverage_eval_len, shared_setup_fold_gadget,
    AkitaExpandedSetup, CommittedGroupParams, FpExtEncoding, PreparedRelationAddress,
    RelationAddressGeometry, RingRelationInstance, SetupContributionGroupInputs,
    SetupContributionPlan, SetupProjectionGeometry,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};
use product_table::{RectangularSetupProductTerm, SetupProductSource};
use std::sync::Arc;

/// Linear setup-product state. Its setup allocation survives cache trimming.
pub struct CpuStage3Session<F: Field, E: Field> {
    binding: OperationBinding,
    lease: crate::opaque::ScopeLease,
    setup: RectangularSetupProductTerm<F, E>,
    round: usize,
    pending: Option<UnivariatePoly<E>>,
    claim: E,
}

impl<F, E> OpaqueStage3Kernel<F, E> for CpuBackend<F, E>
where
    F: Field + CanonicalEncoding + AkitaSerialize + 'static,
    E: Field
        + Ring
        + MulBaseUnreduced<F>
        + FpExtEncoding<F>
        + ExtField<F>
        + AkitaSerialize
        + 'static,
{
    type Stage3SessionHandle = CpuStage3Session<F, E>;

    fn begin_stage3(
        &self,
        request: Stage3Request<'_, F, E, Self::ProofSessionHandle>,
    ) -> Result<(E, Self::Stage3SessionHandle), AkitaError> {
        let proof = request.session.validate_owner(self.owner())?;
        let scope = proof.scope_id();
        let context = crate::opaque::ProofContext::new(
            self.owner_id(),
            self.owner().setup_digest(),
            scope,
            request.level,
        );
        let binding = self.binding(request.session, &context)?;
        let lease = binding.scope_lease().clone();
        let (schedule, _) = proof.proof_plan()?;
        let parameters = if request.level == 0 {
            &schedule.root.params
        } else {
            &schedule
                .recursive_folds
                .get(request.level as usize - 1)
                .ok_or(AkitaError::InvalidProof)?
                .params
        };
        let next = &schedule
            .recursive_folds
            .get(request.level as usize)
            .ok_or(AkitaError::InvalidProof)?
            .params;
        if parameters != request.parameters || next != request.next_parameters {
            return Err(AkitaError::InvalidInput(
                "Stage 3 parameters differ from the admitted proof".into(),
            ));
        }
        let expanded = Arc::clone(&self.prepared()?.expanded);
        let setup_coefficient_bits = request
            .address_geometry
            .relation_coefficient_variable_count();
        let setup_x_challenges = request
            .stage2_challenges
            .get(setup_coefficient_bits..)
            .ok_or(AkitaError::InvalidProof)?;
        let setup = build_setup_product_term(
            expanded,
            request.prefix,
            request.parameters,
            request.next_parameters,
            request.relation,
            request.tau1,
            request.alpha,
            setup_x_challenges,
            request.address_geometry,
        )?;
        let claim = setup.input_claim();
        Ok((
            claim,
            CpuStage3Session {
                binding,
                lease,
                setup,
                round: 0,
                pending: None,
                claim,
            },
        ))
    }

    fn stage3_round_polynomial(
        &self,
        session: &mut Self::Stage3SessionHandle,
        round: usize,
        claim: E,
    ) -> Result<UnivariatePoly<E>, AkitaError> {
        self.validate_leased_binding(&session.binding, &session.lease)?;
        if round != session.round
            || round >= session.setup.num_rounds()
            || session.pending.is_some()
            || claim != session.claim
        {
            return Err(AkitaError::InvalidInput(
                "invalid Stage 3 round progression".into(),
            ));
        }
        let polynomial = session.setup.compute_round_univariate(round);
        session.pending = Some(polynomial.clone());
        Ok(polynomial)
    }

    fn bind_stage3_challenge(
        &self,
        session: &mut Self::Stage3SessionHandle,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError> {
        self.validate_leased_binding(&session.binding, &session.lease)?;
        if round != session.round || round >= session.setup.num_rounds() {
            return Err(AkitaError::InvalidInput(
                "invalid Stage 3 challenge progression".into(),
            ));
        }
        let polynomial = session.pending.take().ok_or_else(|| {
            AkitaError::InvalidInput("Stage 3 challenge has no pending round".into())
        })?;
        session.claim = polynomial.evaluate(challenge);
        session.setup.ingest_challenge(round, challenge);
        session.round += 1;
        Ok(())
    }

    fn finish_stage3(&self, session: Self::Stage3SessionHandle) -> Result<E, AkitaError> {
        self.validate_leased_binding(&session.binding, &session.lease)?;
        if session.round != session.setup.num_rounds() || session.pending.is_some() {
            return Err(AkitaError::InvalidInput(
                "incomplete Stage 3 session".into(),
            ));
        }
        session.setup.folded_table_value()
    }
}

#[allow(clippy::too_many_arguments)]
fn build_setup_product_term<F, E>(
    expanded: Arc<AkitaExpandedSetup<F>>,
    prefix: &akita_types::SetupPrefixSlotId,
    lp: &CommittedGroupParams,
    next_fold_level_params: &CommittedGroupParams,
    relation: &RingRelationInstance<F>,
    tau1: &[E],
    alpha: E,
    x_challenges: &[E],
    relation_address_geometry: RelationAddressGeometry,
) -> Result<RectangularSetupProductTerm<F, E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: FpExtEncoding<F> + Ring + ExtField<F> + MulBaseUnreduced<F> + AkitaSerialize,
{
    let (geometry, mut setup_index_weight, alpha_pows) = {
        let _span = tracing::info_span!("stage3_setup_weights_prepare").entered();
        prepare_setup_sumcheck_terms::<F, E>(
            lp,
            relation,
            tau1,
            alpha,
            x_challenges,
            relation_address_geometry,
        )?
    };

    let active_weight_rows = geometry.required();
    let ring_d = geometry.base_ring_dim();
    let _source_span = tracing::info_span!(
        "stage3_setup_source_select",
        active_weight_rows,
        ring_dim = ring_d,
    )
    .entered();
    ensure_setup_envelope(expanded.as_ref(), active_weight_rows, ring_d)?;
    let natural_field_len = geometry.natural_field_len();
    let selected_slot_id = next_fold_level_params.setup_prefix().ok_or_else(|| {
        AkitaError::InvalidSetup("Stage 3 requires a selected setup-prefix slot".to_string())
    })?;
    if selected_slot_id.slot_id().as_ref() != Some(prefix) {
        return Err(AkitaError::InvalidSetup(
            "Stage 3 setup prefix mismatch".into(),
        ));
    }
    let setup_eval_len = setup_prefix_coverage_eval_len(
        Some(expanded.shared_matrix().num_field_elements()),
        prefix,
        next_fold_level_params,
        natural_field_len,
        ring_d,
        "selected setup-prefix slot does not cover setup product",
    )?;
    // Ring elements at `ring_d` are `ring_d` consecutive field coefficients of
    // the flat shared matrix; read them directly instead of building a typed
    // ring view that would immediately be flattened back into the table. The
    // setup weight is zero after `active_weight_rows`, but the committed and
    // opened setup source is the actual full power-of-two prefix.
    let setup_field = expanded.shared_matrix().as_field_slice();
    let setup_idx_len = active_weight_rows
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("setup product index length overflow".into()))?;
    if setup_idx_len > setup_eval_len {
        return Err(AkitaError::InvalidSetup(
            "setup product exceeds selected setup view".to_string(),
        ));
    }

    setup_index_weight.resize(setup_idx_len, E::zero());
    let source_len = setup_idx_len
        .checked_mul(ring_d)
        .ok_or_else(|| AkitaError::InvalidSetup("setup product source length overflow".into()))?;
    setup_field.get(..source_len).ok_or_else(|| {
        AkitaError::InvalidSetup("setup source is shorter than product view".into())
    })?;
    drop(_source_span);

    RectangularSetupProductTerm::new(
        SetupProductSource::Expanded(expanded),
        active_weight_rows,
        setup_index_weight,
        alpha_pows.to_vec(),
    )
}

/// Derive the factored product-sumcheck terms `(required, setup_index_weight, alpha_pows)`
/// from the level parameters and ring relation via the ring-switch row
/// evaluation.
fn prepare_setup_sumcheck_terms<F, E>(
    lp: &CommittedGroupParams,
    relation: &RingRelationInstance<F>,
    tau1: &[E],
    alpha: E,
    x_challenges: &[E],
    relation_address_geometry: RelationAddressGeometry,
) -> Result<(SetupProjectionGeometry, Vec<E>, Vec<E>), AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: FpExtEncoding<F> + Ring + ExtField<F>,
{
    let plan = prepare_setup_contribution_plan::<F, E>(
        relation,
        lp,
        tau1,
        x_challenges,
        relation_address_geometry,
    )?;
    let geometry = plan.projection_geometry();
    let alpha_pows = scalar_powers(alpha, geometry.alpha_power_len());
    let setup_index_weight = plan.materialize_setup_index_weights(alpha)?;
    Ok((geometry, setup_index_weight, alpha_pows.to_vec()))
}

/// Build the stage-3 setup-contribution plan from local prover inputs.
fn prepare_setup_contribution_plan<F, E>(
    relation: &RingRelationInstance<F>,
    lp: &CommittedGroupParams,
    tau1: &[E],
    x_challenges: &[E],
    relation_address_geometry: RelationAddressGeometry,
) -> Result<SetupContributionPlan<E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: Field + ExtField<F>,
{
    let opening_batch = relation.opening_batch();
    let chunk_layout = relation.segment_layout(lp, None)?;
    let rows = lp.relation_matrix_row_count(opening_batch.num_groups())?;
    let eq_tau1: Arc<[E]> = EqPolynomial::evals_prefix(tau1, rows)?.into();

    lp.validate_opening_batch(opening_batch)?;
    let order = opening_batch.root_group_order()?;
    if order.iter().any(|&group_index| {
        chunk_layout.num_chunks_for_group(group_index) != lp.witness_chunk.num_chunks
    }) {
        return Err(AkitaError::InvalidSetup(
            "multi-group witness layout does not match root group order".to_string(),
        ));
    }

    let mut groups = Vec::with_capacity(order.len());
    for &group_index in &order {
        let group_lp = lp.group_params(opening_batch, group_index)?;
        let group_layout = opening_batch.group_layout(group_index)?;
        let num_claims = group_layout.num_polynomials();
        let n_a = group_lp.a_rows_len();
        let n_b = group_lp.logical_b_rows_len()?;
        let a_range = lp.a_row_range(opening_batch, group_index)?;
        let b_range = lp.commitment_row_range(opening_batch, group_index)?;
        if a_range.len() != n_a || b_range.len() != n_b {
            return Err(AkitaError::InvalidSetup(
                "multi-group row ranges do not match group matrix heights".to_string(),
            ));
        }
        groups.push(SetupContributionGroupInputs {
            group_id: group_index,
            num_claims,
            depth_fold: group_lp.num_digits_fold(),
            a_row_start: a_range.start,
            b_row_start: b_range.start,
        });
    }

    let fold_gadget = shared_setup_fold_gadget::<F>(lp, opening_batch, &groups);
    let plan = SetupContributionPlan::prepare::<F>(
        lp,
        opening_batch,
        relation.extension_degree(),
        eq_tau1,
        &chunk_layout,
        &groups,
        PreparedRelationAddress::new(x_challenges)?,
        fold_gadget.as_deref(),
        relation_address_geometry,
    )?;
    Ok(plan)
}
