//! Setup-product sumcheck over two canonical coefficient-first table phases.
//!
//! Coefficient variables are bound first, followed by setup-index variables.
//! This matches setup products of the form
//! `S(i, y) * setup_index_weight(i) * alpha(y)` without materializing the full
//! `setup_index_weight(i) * alpha(y)` table or a combined factor table.

mod product_table;
#[cfg(test)]
mod utils;

use crate::kernels::sumcheck::SumcheckTableOperations;
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::scalar_powers;
use akita_algebra::uni_poly::UniPoly;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, LiftBase, MulBase,
    MulBaseUnreduced,
};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::{SumcheckInstanceProver, SumcheckInstanceProverExt, SumcheckProof};
use akita_transcript::{labels::ABSORB_SETUP_PREFIX_SLOT, Transcript};
use akita_types::{
    ensure_setup_envelope, setup_prefix_coverage_eval_len, shared_setup_fold_gadget,
    AkitaExpandedSetup, CommittedGroupParams, FpExtEncoding, PreparedRelationAddress,
    RelationAddressGeometry, RingRelationInstance, SetupContributionGroupInputs,
    SetupContributionPlan, SetupPrefixProverRegistry, SetupProjectionGeometry,
    SETUP_SUMCHECK_DEGREE,
};
use product_table::RectangularSetupProductTerm;
use std::sync::Arc;

/// Output of the setup-only stage-3 prover.
pub struct AkitaStage3ProverOutput<E: FieldCore> {
    /// Setup-product claim carried in the serialized stage-3 proof.
    pub setup_product_claim: E,
    /// Setup-prefix MLE value at the stage-3 challenge.
    pub setup_prefix_eval: E,
    /// Setup-prefix opening point.
    pub setup_prefix_point: Vec<E>,
    /// Degree-two setup-product sumcheck.
    pub sumcheck: SumcheckProof<E>,
}

/// Stage-3 setup-product sumcheck prover.
pub struct AkitaStage3Prover<'a, F: FieldCore, E: FieldCore> {
    setup: RectangularSetupProductTerm<'a, F, E>,
    setup_product_claim: E,
}

impl<'a, F, E> AkitaStage3Prover<'a, F, E>
where
    F: FieldCore,
    E: ExtField<F> + FromPrimitiveInt + MulBaseUnreduced<F> + SumcheckTableOperations<F>,
{
    /// Construct a recursive setup-product sumcheck prover.
    #[allow(clippy::too_many_arguments)]
    pub fn new<T>(
        expanded: &'a AkitaExpandedSetup<F>,
        prefix_slots: &SetupPrefixProverRegistry<F>,
        lp: &CommittedGroupParams,
        next_fold_level_params: &CommittedGroupParams,
        relation: &RingRelationInstance<F>,
        tau1: &[E],
        alpha: E,
        stage2_challenges: &[E],
        relation_address_geometry: RelationAddressGeometry,
        transcript: &mut T,
    ) -> Result<Self, AkitaError>
    where
        F: CanonicalField,
        E: FpExtEncoding<F> + LiftBase<F> + AkitaSerialize,
        T: Transcript<F>,
    {
        let setup_coefficient_bits =
            relation_address_geometry.relation_coefficient_variable_count();
        let setup_x_challenges = stage2_challenges
            .get(setup_coefficient_bits..)
            .ok_or(AkitaError::InvalidProof)?;
        let setup_term = {
            let _span = tracing::info_span!("stage3_setup_term_prepare").entered();
            build_setup_product_term::<F, E, T>(
                expanded,
                prefix_slots,
                lp,
                next_fold_level_params,
                relation,
                tau1,
                alpha,
                setup_x_challenges,
                relation_address_geometry,
                transcript,
            )?
        };
        let setup_product_claim = setup_term.input_claim();
        Ok(Self {
            setup: setup_term,
            setup_product_claim,
        })
    }

    pub fn prove<T, SampleRound>(
        &mut self,
        transcript: &mut T,
        sample_round: SampleRound,
    ) -> Result<AkitaStage3ProverOutput<E>, AkitaError>
    where
        F: CanonicalField,
        E: AkitaSerialize,
        T: Transcript<F>,
        SampleRound: FnMut(&mut T) -> E,
    {
        let (sumcheck, setup_prefix_point, _final_claim) = <Self as SumcheckInstanceProverExt<
            E,
        >>::prove::<F, T, _>(
            self, transcript, sample_round
        )?;
        let setup_prefix_eval = self.setup.folded_table_value()?;
        Ok(AkitaStage3ProverOutput {
            setup_product_claim: self.setup_product_claim,
            setup_prefix_eval,
            setup_prefix_point,
            sumcheck,
        })
    }
}

impl<F, E> SumcheckInstanceProver<E> for AkitaStage3Prover<'_, F, E>
where
    F: FieldCore,
    E: ExtField<F> + FromPrimitiveInt + MulBaseUnreduced<F> + SumcheckTableOperations<F>,
{
    fn num_rounds(&self) -> usize {
        self.setup.num_rounds()
    }

    fn degree_bound(&self) -> usize {
        SETUP_SUMCHECK_DEGREE
    }

    fn input_claim(&self) -> E {
        self.setup.input_claim()
    }

    fn compute_round_univariate(&mut self, round: usize, previous_claim: E) -> UniPoly<E> {
        self.setup.compute_round_univariate(round, previous_claim)
    }

    fn ingest_challenge(&mut self, round: usize, r_round: E) {
        self.setup.ingest_challenge(round, r_round);
    }
}

#[allow(clippy::too_many_arguments)]
fn build_setup_product_term<'a, F, E, T>(
    expanded: &'a AkitaExpandedSetup<F>,
    prefix_slots: &SetupPrefixProverRegistry<F>,
    lp: &CommittedGroupParams,
    next_fold_level_params: &CommittedGroupParams,
    relation: &RingRelationInstance<F>,
    tau1: &[E],
    alpha: E,
    x_challenges: &[E],
    relation_address_geometry: RelationAddressGeometry,
    transcript: &mut T,
) -> Result<RectangularSetupProductTerm<'a, F, E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F>
        + FromPrimitiveInt
        + LiftBase<F>
        + MulBaseUnreduced<F>
        + AkitaSerialize
        + SumcheckTableOperations<F>,
    T: Transcript<F>,
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
    ensure_setup_envelope(expanded, active_weight_rows, ring_d)?;
    let natural_field_len = geometry.natural_field_len();
    let selected_slot_id = next_fold_level_params
        .setup_prefix
        .as_ref()
        .ok_or_else(|| {
            AkitaError::InvalidSetup("Stage 3 requires a selected setup-prefix slot".to_string())
        })?;
    let slot = prefix_slots.get(selected_slot_id).ok_or_else(|| {
        AkitaError::InvalidSetup(
            "planned setup-prefix slot is missing from prover setup".to_string(),
        )
    })?;
    let setup_eval_len = setup_prefix_coverage_eval_len(
        Some(expanded.shared_matrix().num_field_elements()),
        &slot.id,
        next_fold_level_params,
        natural_field_len,
        ring_d,
        "selected setup-prefix slot does not cover setup product",
    )?;
    transcript.append_serde(ABSORB_SETUP_PREFIX_SLOT, &slot.id);
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
    let setup_source = setup_field.get(..source_len).ok_or_else(|| {
        AkitaError::InvalidSetup("setup source is shorter than product view".into())
    })?;
    drop(_source_span);

    RectangularSetupProductTerm::new(
        setup_source,
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
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + MulBase<F>,
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
    F: FieldCore + CanonicalField,
    E: FieldCore + LiftBase<F> + MulBase<F>,
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
        let n_b = group_lp.b_rows_len();
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
        eq_tau1,
        &chunk_layout,
        &groups,
        PreparedRelationAddress::new(x_challenges)?,
        fold_gadget.as_deref(),
        relation_address_geometry,
    )?;
    Ok(plan)
}
