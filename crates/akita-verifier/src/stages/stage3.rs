//! Verifier for the setup-product sumcheck — the verifier counterpart to the
//! prover-side `AkitaStage3Prover`.

use crate::protocol::ring_switch::RelationMatrixEvaluator;
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::{eval_ring_at_pows_fast, scalar_powers};
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt};
use akita_serialization::AkitaSerialize;
use akita_transcript::labels::{
    ABSORB_SETUP_PREFIX_SLOT, ABSORB_SUMCHECK_CLAIM, CHALLENGE_SUMCHECK_ROUND,
};
use akita_transcript::{sample_ext_challenge, Transcript};
use akita_types::verifier_work::{record_verifier_work, VerifierWorkEvent};
use akita_types::{
    dispatch_for_field, ensure_setup_envelope, select_setup_prefix_slot, shared_setup_fold_gadget,
    stage3_offload_natural_field_len, AkitaExpandedSetup, AkitaVerifierSetup, LevelParams,
    SetupContributionPlan, SetupIndexWeightEvaluator, SetupSumcheckProof, SETUP_OFFLOAD_D_SETUP,
    SETUP_SUMCHECK_DEGREE,
};

/// Verifier counterpart to `AkitaStage3Prover`: replays the setup product
/// sumcheck for the setup contribution at `x_challenges`.
///
/// Construct with [`SetupSumcheckVerifier::new`], which derives the setup
/// evaluation plan and sumcheck round count from the ring-switch row
/// evaluation, then call [`verify_batched_stage3`](Self::verify_batched_stage3)
/// with the proof and transcript.
pub(crate) struct SetupSumcheckVerifier<E: FieldCore> {
    plan: SetupContributionPlan<E>,
    setup_index_weight_evaluator: Option<SetupIndexWeightEvaluator<E>>,
    alpha_pows: Vec<E>,
    ring_bits: usize,
    rounds: usize,
}

impl<E: FieldCore> SetupSumcheckVerifier<E> {
    /// Prepare the setup-product sumcheck verifier for the setup contribution
    /// at `x_challenges`.
    ///
    /// Derives the setup evaluation plan (and thus the per-round shape) from
    /// the relation-matrix evaluation; must be called before
    /// [`verify_batched_stage3`](Self::verify_batched_stage3).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new<F, const D: usize>(
        relation_matrix_evaluator: &RelationMatrixEvaluator<E>,
        x_challenges: &[E],
        tau1: &[E],
        alpha: E,
    ) -> Result<Self, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: ExtField<F>,
    {
        let alpha_pows = scalar_powers(alpha, D);
        let fold_gadget =
            shared_setup_fold_gadget::<F>(&relation_matrix_evaluator.setup_contribution_groups);
        let plan = SetupContributionPlan::finish_plan::<F>(
            &relation_matrix_evaluator.setup_contribution_static,
            x_challenges,
            None,
            None,
            fold_gadget.as_deref(),
            &relation_matrix_evaluator.setup_contribution_groups,
        )?;
        let role_dims = relation_matrix_evaluator.role_dims;
        let setup_index_weight_evaluator = if let Some(fold_gadget) = fold_gadget.as_deref() {
            if role_dims.d_a() == D && role_dims.d_b() == D && role_dims.d_d() == D {
                let evaluator = SetupIndexWeightEvaluator::new::<F>(
                    &relation_matrix_evaluator.setup_contribution_inputs,
                    &relation_matrix_evaluator.setup_contribution_static,
                    &relation_matrix_evaluator.setup_contribution_groups,
                    tau1,
                    x_challenges,
                    fold_gadget,
                    D,
                    role_dims,
                    alpha,
                )?;
                evaluator.prefers_succinct_path().then_some(evaluator)
            } else {
                None
            }
        } else {
            None
        };
        let setup_idx_len = plan
            .required()?
            .checked_next_power_of_two()
            .ok_or_else(|| {
                AkitaError::InvalidSetup("setup product index length overflow".into())
            })?;
        let setup_idx_bits = setup_idx_len.trailing_zeros() as usize;
        let ring_bits = D.trailing_zeros() as usize;
        let rounds = ring_bits
            .checked_add(setup_idx_bits)
            .ok_or_else(|| AkitaError::InvalidSetup("setup product round count overflow".into()))?;

        Ok(Self {
            plan,
            setup_index_weight_evaluator,
            alpha_pows,
            ring_bits,
            rounds,
        })
    }

    /// Verify the batched setup-product + carried-next-witness stage-3 sumcheck.
    ///
    /// Returns the projected next-witness opening point `rho_w` to be threaded
    /// into the next recursive suffix level.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn verify_batched_stage3<F, T>(
        &self,
        setup: &AkitaVerifierSetup<F>,
        next_fold_level_params: &LevelParams,
        ring_d: usize,
        proof: &SetupSumcheckProof<E>,
        stage2_next_w_eval: E,
        stage2_challenges: &[E],
        witness_rounds: usize,
        eta: E,
        transcript: &mut T,
    ) -> Result<Vec<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: ExtField<F> + FromPrimitiveInt + AkitaSerialize + akita_field::MulBaseUnreduced<F>,
        T: Transcript<F>,
    {
        if stage2_challenges.len() != witness_rounds {
            return Err(AkitaError::InvalidSize {
                expected: witness_rounds,
                actual: stage2_challenges.len(),
            });
        }
        let setup_len = setup
            .expanded
            .shared_matrix()
            .total_ring_elements_at_dyn(ring_d)?;
        let setup_eval_len = self.setup_eval_len::<F, T>(
            setup,
            next_fold_level_params,
            ring_d,
            setup_len,
            transcript,
        )?;
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            ring_d,
            |D| {
                self.verify_batched_stage3_kernel::<F, T, D>(
                    setup,
                    proof,
                    stage2_next_w_eval,
                    stage2_challenges,
                    witness_rounds,
                    setup_eval_len,
                    eta,
                    transcript,
                )
            }
        )
    }

    fn setup_eval_len<F, T>(
        &self,
        setup: &AkitaVerifierSetup<F>,
        next_fold_level_params: &LevelParams,
        ring_d: usize,
        setup_len: usize,
        transcript: &mut T,
    ) -> Result<usize, AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
    {
        if ring_d == SETUP_OFFLOAD_D_SETUP {
            let natural_field_len =
                stage3_offload_natural_field_len(self.plan.required()?, ring_d)?;
            ensure_setup_envelope(&setup.expanded, self.plan.required()?, ring_d)?;
            let setup_prefix_selection = select_setup_prefix_slot(
                setup.expanded.seed(),
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
            } else {
                Ok(setup_len)
            }
        } else {
            Ok(setup_len)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_batched_stage3_kernel<F, T, const D: usize>(
        &self,
        setup: &AkitaVerifierSetup<F>,
        proof: &SetupSumcheckProof<E>,
        stage2_next_w_eval: E,
        stage2_challenges: &[E],
        witness_rounds: usize,
        setup_eval_len: usize,
        eta: E,
        transcript: &mut T,
    ) -> Result<Vec<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: ExtField<F> + FromPrimitiveInt + AkitaSerialize + akita_field::MulBaseUnreduced<F>,
        T: Transcript<F>,
    {
        record_verifier_work(VerifierWorkEvent::Stage3Instance);
        let batched_rounds = self.rounds.max(witness_rounds);
        transcript.append_serde(
            ABSORB_SUMCHECK_CLAIM,
            &(proof.claim + eta * stage2_next_w_eval),
        );
        let (final_claim, challenges) = proof.sumcheck.verify::<F, _, _>(
            proof.claim + eta * stage2_next_w_eval,
            batched_rounds,
            SETUP_SUMCHECK_DEGREE,
            transcript,
            |tr| sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND),
        )?;
        let rho_w = challenges[..witness_rounds].to_vec();
        let rho_setup = &challenges[..self.rounds];
        let (rho_y, rho_setup_idx) = rho_setup.split_at(self.ring_bits);

        // The setup prefix itself is still evaluated by scanning the selected
        // prefix. The setup-index weight is structured, so evaluate its MLE
        // directly at `rho_setup_idx` instead of building a dense equality
        // table for that factor.
        let eq_setup_idx = setup_idx_eq_table(self.plan.required()?, rho_setup_idx)?;
        let eq_y = ring_eq_table::<E, D>(rho_y)?;
        let setup_val = setup_mle_at_eq_tables::<F, E, D>(
            &setup.expanded,
            self.plan.required()?,
            setup_eval_len,
            &eq_setup_idx,
            &eq_y,
        )?;
        let setup_index_weight = match &self.setup_index_weight_evaluator {
            Some(evaluator) => match evaluator.evaluate(rho_setup_idx)? {
                Some(value) => {
                    record_verifier_work(VerifierWorkEvent::SetupWeightSuccinctEval);
                    value
                }
                None => {
                    record_verifier_work(VerifierWorkEvent::SetupWeightPlanEval);
                    self.plan.evaluate_setup_index_weight_mle(rho_setup_idx)?
                }
            },
            None => {
                record_verifier_work(VerifierWorkEvent::SetupWeightPlanEval);
                self.plan.evaluate_setup_index_weight_mle(rho_setup_idx)?
            }
        };
        let alpha_val = eval_dense_table_with_eq(&self.alpha_pows, &eq_y)?;
        let witness_scale = lift_scale::<E>(batched_rounds - witness_rounds)?;
        let setup_scale = lift_scale::<E>(batched_rounds - self.rounds)?;
        let eq_w = EqPolynomial::mle(stage2_challenges, &rho_w)?;
        let expected = eta * witness_scale * eq_w * proof.next_w_eval
            + setup_scale * setup_val * setup_index_weight * alpha_val;
        if final_claim != expected {
            return Err(AkitaError::InvalidInput(
                "batched stage-3 final relation mismatch".to_string(),
            ));
        }
        Ok(rho_w)
    }
}

fn lift_scale<E: FieldCore + FromPrimitiveInt>(extra_rounds: usize) -> Result<E, AkitaError> {
    let inv_two = E::from_u64(2)
        .inverse()
        .ok_or_else(|| AkitaError::InvalidSetup("two is not invertible in Akita fields".into()))?;
    Ok((0..extra_rounds).fold(E::one(), |acc, _| acc * inv_two))
}

fn setup_idx_eq_table<E: FieldCore>(
    required: usize,
    rho_setup_idx: &[E],
) -> Result<Vec<E>, AkitaError> {
    let setup_idx_len = required
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("setup product index length overflow".into()))?;
    if rho_setup_idx.len() != setup_idx_len.trailing_zeros() as usize {
        return Err(AkitaError::InvalidProof);
    }
    let table = EqPolynomial::evals(rho_setup_idx)?;
    record_verifier_work(VerifierWorkEvent::SetupEqElements(table.len() as u64));
    Ok(table)
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
    record_verifier_work(VerifierWorkEvent::RingEqElements(eq_y.len() as u64));
    Ok(eq_y)
}

fn eval_dense_table_with_eq<E: FieldCore>(evals: &[E], eq: &[E]) -> Result<E, AkitaError> {
    if evals.len() != eq.len() {
        return Err(AkitaError::InvalidSize {
            expected: evals.len(),
            actual: eq.len(),
        });
    }
    Ok(cfg_fold_reduce!(
        0..evals.len(),
        E::zero,
        |mut acc, idx| {
            acc += evals[idx] * eq[idx];
            acc
        },
        |lhs, rhs| lhs + rhs
    ))
}

fn setup_mle_at_eq_tables<F, E, const D: usize>(
    setup: &AkitaExpandedSetup<F>,
    required: usize,
    setup_eval_len: usize,
    eq_setup_idx: &[E],
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
    record_verifier_work(VerifierWorkEvent::SetupRingsScanned(required as u64));

    Ok(cfg_fold_reduce!(
        0..required,
        E::zero,
        |mut acc, setup_idx| {
            let ring_eval = eval_ring_at_pows_fast(&setup_entries[setup_idx], eq_y);
            acc += eq_setup_idx[setup_idx] * ring_eval;
            acc
        },
        |lhs, rhs| lhs + rhs
    ))
}
