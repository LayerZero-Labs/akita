//! Verifier for the setup-product sumcheck — the verifier counterpart to the
//! prover-side `AkitaStage3Prover`.

use crate::protocol::ring_switch::RelationMatrixEvaluator;
use akita_algebra::eq_poly::{EqPolynomial, SplitEqEvals};
use akita_algebra::ring::{eval_ring_at_pows_fast, evaluate_power_sequence_mle};
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt};
use akita_serialization::AkitaSerialize;
use akita_transcript::labels::{
    ABSORB_SETUP_PREFIX_SLOT, ABSORB_SUMCHECK_CLAIM, CHALLENGE_SUMCHECK_ROUND,
};
use akita_transcript::{sample_ext_challenge, Transcript};
use akita_types::{
    dispatch_for_field, ensure_setup_envelope, select_setup_prefix_slot, AkitaExpandedSetup,
    AkitaVerifierSetup, CommittedGroupParams, PreparedRelationAddress, SetupContributionPlan,
    SetupSumcheckProof, SETUP_SUMCHECK_DEGREE,
};

/// Verifier counterpart to `AkitaStage3Prover`: replays the setup product
/// sumcheck for the setup contribution at `x_challenges`.
///
/// Construct with [`SetupSumcheckVerifier::new`], which derives the setup
/// evaluation plan and sumcheck round count from the ring-switch row
/// evaluation, then call [`verify_stage3`](Self::verify_stage3)
/// with the proof and transcript.
pub(crate) struct SetupSumcheckVerifier<E: FieldCore> {
    setup_contribution_plan: SetupContributionPlan<E>,
    alpha: E,
    ring_bits: usize,
    rounds: usize,
}

impl<E: FieldCore> SetupSumcheckVerifier<E> {
    /// Prepare the setup-product sumcheck verifier for the setup contribution
    /// at `x_challenges`.
    ///
    /// Derives the setup evaluation plan (and thus the per-round shape) from
    /// the relation-matrix evaluation; must be called before
    /// [`verify_stage3`](Self::verify_stage3).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new<F>(
        relation_matrix_evaluator: &RelationMatrixEvaluator<E>,
        x_challenges: &[E],
        alpha: E,
    ) -> Result<Self, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: ExtField<F>,
    {
        let fold_gadget = relation_matrix_evaluator.setup_contribution_fold_gadget::<F>()?;
        let plan = relation_matrix_evaluator
            .take_cached_setup_contribution_plan(x_challenges)?
            .map_or_else(
                || {
                    relation_matrix_evaluator.setup_contribution_plan::<F>(
                        PreparedRelationAddress::new(x_challenges)?,
                        fold_gadget.as_deref(),
                    )
                },
                Ok,
            )?;
        let geometry = plan.projection_geometry();
        Ok(Self {
            setup_contribution_plan: plan,
            alpha,
            ring_bits: geometry.ring_bits(),
            rounds: geometry.rounds(),
        })
    }

    /// Verify the setup-product stage-3 sumcheck.
    ///
    /// Returns the setup opening point for the next recursive suffix level.
    pub(crate) fn verify_stage3<F, T>(
        &self,
        setup: &AkitaVerifierSetup<F>,
        next_fold_level_params: &CommittedGroupParams,
        proof: &SetupSumcheckProof<E>,
        transcript: &mut T,
    ) -> Result<Vec<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: ExtField<F> + FromPrimitiveInt + AkitaSerialize + akita_field::MulBaseUnreduced<F>,
        T: Transcript<F>,
    {
        let ring_d = self
            .setup_contribution_plan
            .projection_geometry()
            .base_ring_dim();
        if ring_d == 0 {
            return Err(AkitaError::InvalidSetup(
                "Stage 3 setup ring dimension must be nonzero".into(),
            ));
        }
        let setup_field_len = setup.expanded.shared_matrix().num_field_elements();
        if !setup_field_len.is_multiple_of(ring_d) {
            return Err(AkitaError::InvalidSetup(
                "shared setup field length is not divisible by Stage 3 ring dimension".into(),
            ));
        }
        let setup_len = setup_field_len / ring_d;
        let setup_eval_len = self.setup_eval_len::<F, T>(
            setup,
            next_fold_level_params,
            ring_d,
            setup_len,
            transcript,
        )?;
        let setup_prefix_eval = next_fold_level_params
            .setup_prefix
            .as_ref()
            .map(|_| proof.setup_prefix_eval);
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Opening),
            F,
            ring_d,
            |D| {
                self.verify_stage3_kernel::<F, T, D>(
                    setup,
                    proof,
                    setup_eval_len,
                    setup_prefix_eval,
                    transcript,
                )
            }
        )
    }

    fn setup_eval_len<F, T>(
        &self,
        setup: &AkitaVerifierSetup<F>,
        next_fold_level_params: &CommittedGroupParams,
        ring_d: usize,
        setup_len: usize,
        transcript: &mut T,
    ) -> Result<usize, AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
    {
        if next_fold_level_params.setup_prefix.is_some() {
            let geometry = self.setup_contribution_plan.projection_geometry();
            let natural_field_len = geometry.natural_field_len();
            ensure_setup_envelope(&setup.expanded, geometry.required(), ring_d)?;
            let setup_prefix_selection = select_setup_prefix_slot(
                setup_len,
                |id| {
                    setup
                        .prefix_slots
                        .get(id)
                        .map(|slot| (slot, slot.natural_len, slot.padded_len))
                },
                next_fold_level_params,
                natural_field_len,
                ring_d,
                "verifier setup-prefix slot does not cover setup product",
            )?;
            if let Some((slot, setup_eval_len)) = setup_prefix_selection {
                transcript.append_serde(ABSORB_SETUP_PREFIX_SLOT, &slot.id);
                Ok(setup_eval_len)
            } else if next_fold_level_params.setup_prefix.is_some() {
                Err(AkitaError::InvalidSetup(
                    "planned setup-prefix slot is missing from verifier setup".to_string(),
                ))
            } else {
                Ok(setup_len)
            }
        } else {
            Ok(setup_len)
        }
    }

    fn verify_stage3_kernel<F, T, const D: usize>(
        &self,
        setup: &AkitaVerifierSetup<F>,
        proof: &SetupSumcheckProof<E>,
        setup_eval_len: usize,
        setup_prefix_eval: Option<E>,
        transcript: &mut T,
    ) -> Result<Vec<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: ExtField<F> + FromPrimitiveInt + AkitaSerialize + akita_field::MulBaseUnreduced<F>,
        T: Transcript<F>,
    {
        let required = self.setup_contribution_plan.required();
        transcript.append_serde(ABSORB_SUMCHECK_CLAIM, &proof.claim);
        let (final_claim, challenges) = proof.sumcheck.verify::<F, _, _>(
            proof.claim,
            self.rounds,
            SETUP_SUMCHECK_DEGREE,
            transcript,
            |tr| sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND),
        )?;
        let (rho_y, rho_setup_idx) = challenges.split_at(self.ring_bits);

        // The setup prefix itself is still evaluated by scanning the selected
        // prefix. The setup-index weight is structured, so evaluate its MLE
        // directly at `rho_setup_idx` instead of building a dense equality
        // table for that factor.
        let eq_y = ring_eq_table::<E, D>(rho_y)?;
        let setup_val = {
            let _span =
                tracing::info_span!("stage3_setup_prefix", cached = setup_prefix_eval.is_some())
                    .entered();
            match setup_prefix_eval {
                Some(value) => value,
                None => setup_mle_at_eq_tables::<F, E, D>(
                    &setup.expanded,
                    required,
                    setup_eval_len,
                    rho_setup_idx,
                    &eq_y,
                )?,
            }
        };
        let setup_index_weight = {
            let _span = tracing::info_span!("stage3_setup_index_weight_eval").entered();
            self.setup_contribution_plan
                .evaluate_setup_index_weight_mle(rho_setup_idx, self.alpha)?
        };
        let alpha_val = evaluate_power_sequence_mle(self.alpha, rho_y);
        let setup_term = setup_val * setup_index_weight * alpha_val;
        if final_claim != setup_term {
            return Err(AkitaError::InvalidProof);
        }
        Ok(challenges)
    }
}

fn ring_eq_table<E: FieldCore, const D: usize>(rho_y: &[E]) -> Result<Vec<E>, AkitaError> {
    if rho_y.len() != D.trailing_zeros() as usize {
        return Err(AkitaError::InvalidProof);
    }
    let eq_y = EqPolynomial::evals(rho_y)?;
    if eq_y.len() != D {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: eq_y.len(),
        });
    }
    Ok(eq_y)
}

fn setup_mle_at_eq_tables<F, E, const D: usize>(
    setup: &AkitaExpandedSetup<F>,
    required: usize,
    setup_eval_len: usize,
    rho_setup_idx: &[E],
    eq_y: &[E],
) -> Result<E, AkitaError>
where
    F: FieldCore,
    E: ExtField<F> + akita_field::MulBaseUnreduced<F>,
{
    if required > setup_eval_len {
        return Err(AkitaError::InvalidSetup(
            "setup prefix is too small for selected verifier layout".into(),
        ));
    }
    let setup_idx_len = required
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("setup MLE index length overflow".into()))?;
    let eq_setup_idx = SplitEqEvals::new(rho_setup_idx)?;
    if eq_setup_idx.len() != setup_idx_len {
        return Err(AkitaError::InvalidSize {
            expected: setup_idx_len,
            actual: eq_setup_idx.len(),
        });
    }
    if eq_y.len() != D {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: eq_y.len(),
        });
    }
    let setup_view = setup.shared_matrix().ring_view::<D>(1, setup_eval_len)?;
    let setup_entries = setup_view.as_slice();

    // Scan the selected setup prefix once. Each entry contracts the ring with
    // `eq_y` and the setup-index equality; the scan is `O(required · D)` and is
    // the dominant recursive-mode verifier cost, so evaluate it in parallel.
    let _span = tracing::info_span!("stage3_setup_mle_scan", required).entered();
    let terms = cfg_into_iter!(0..required)
        .map(|setup_idx| -> Result<E, AkitaError> {
            let entry = setup_entries
                .get(setup_idx)
                .ok_or(AkitaError::InvalidProof)?;
            let ring_eval = eval_ring_at_pows_fast(entry, eq_y);
            Ok(eq_setup_idx.eval_at(setup_idx)? * ring_eval)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(terms.into_iter().fold(E::zero(), |acc, value| acc + value))
}
