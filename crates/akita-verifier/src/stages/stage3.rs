//! Verifier for the setup-product sumcheck — the verifier counterpart to the
//! prover-side `SetupSumcheckProver`.

use crate::protocol::ring_switch::RingSwitchDeferredRowEval;
use crate::protocol::{SetupEvalPlan, SetupEvaluator};
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::scalar_powers;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_serialization::AkitaSerialize;
use akita_transcript::labels::{ABSORB_SUMCHECK_CLAIM, CHALLENGE_SUMCHECK_ROUND};
use akita_transcript::{sample_ext_challenge, Transcript};
use akita_types::{
    gadget_row_scalars, AkitaExpandedSetup, SetupSumcheckProof, SETUP_SUMCHECK_DEGREE,
};

/// Verifier counterpart to `SetupSumcheckProver`: replays the setup product
/// sumcheck for the setup contribution at `x_challenges`.
///
/// Construct with [`SetupSumcheckVerifier::new`], which derives the setup
/// evaluation plan and sumcheck round count from the ring-switch row
/// evaluation, then call [`verify`](Self::verify) with the proof and
/// transcript.
pub(crate) struct SetupSumcheckVerifier<E: FieldCore> {
    plan: SetupEvalPlan<E>,
    alpha_pows: Vec<E>,
    ring_bits: usize,
    rounds: usize,
}

impl<E: FieldCore> SetupSumcheckVerifier<E> {
    /// Prepare the setup-product sumcheck verifier for the setup contribution
    /// at `x_challenges`.
    ///
    /// Derives the setup evaluation plan (and thus the per-round shape) from
    /// the ring-switch row evaluation; must be called before [`verify`](Self::verify).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new<F, const D: usize>(
        prepared: &RingSwitchDeferredRowEval<E>,
        x_challenges: &[E],
        alpha: E,
    ) -> Result<Self, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: ExtField<F>,
    {
        let alpha_pows = scalar_powers(alpha, D);
        let fold_gadget = gadget_row_scalars::<F>(prepared.depth_fold, prepared.log_basis);
        let layout = prepared.segment_layout()?;
        let setup_contribution_inputs = prepared.create_setup_contribution_inputs();
        let evaluator = SetupEvaluator::new(
            &setup_contribution_inputs,
            x_challenges,
            None,
            None,
            &alpha_pows,
            &fold_gadget,
            layout.offset_e,
            layout.offset_t,
            layout.offset_z,
        );
        let plan = evaluator.prepare()?;
        let lambda_len = plan.required().checked_next_power_of_two().ok_or_else(|| {
            AkitaError::InvalidSetup("setup product lambda length overflow".into())
        })?;
        let lambda_bits = lambda_len.trailing_zeros() as usize;
        let ring_bits = D.trailing_zeros() as usize;
        let rounds = ring_bits
            .checked_add(lambda_bits)
            .ok_or_else(|| AkitaError::InvalidSetup("setup product round count overflow".into()))?;

        Ok(Self {
            plan,
            alpha_pows,
            ring_bits,
            rounds,
        })
    }

    /// Verify a setup product sumcheck for the setup contribution.
    ///
    /// The sumcheck variable order is ring-coordinate bits first (`y`), followed
    /// by setup-ring index bits (`lambda`).
    pub(crate) fn verify<F, T, const D: usize>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        proof: &SetupSumcheckProof<E>,
        transcript: &mut T,
    ) -> Result<(), AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: ExtField<F> + AkitaSerialize,
        T: Transcript<F>,
    {
        transcript.append_serde(ABSORB_SUMCHECK_CLAIM, &proof.claim);
        let (final_claim, challenges) = proof.sumcheck.verify::<F, _, _>(
            proof.claim,
            self.rounds,
            SETUP_SUMCHECK_DEGREE,
            transcript,
            |tr| sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND),
        )?;
        let (rho_y, rho_lambda) = challenges.split_at(self.ring_bits);

        let eq_lambda = lambda_eq_table(self.plan.required(), rho_lambda)?;
        let eq_y = ring_eq_table::<E, D>(rho_y)?;
        let setup_val =
            setup_mle_at_eq_tables::<F, E, D>(setup, self.plan.required(), &eq_lambda, &eq_y)?;
        let omega = self.plan.evaluate_bar_omega_with_eq(&eq_lambda)?;
        let alpha_val = eval_dense_table_with_eq(&self.alpha_pows, &eq_y)?;
        if final_claim != setup_val * omega * alpha_val {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }
}

fn lambda_eq_table<E: FieldCore>(required: usize, rho_lambda: &[E]) -> Result<Vec<E>, AkitaError> {
    let lambda_len = required
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("setup product lambda length overflow".into()))?;
    if rho_lambda.len() != lambda_len.trailing_zeros() as usize {
        return Err(AkitaError::InvalidProof);
    }
    EqPolynomial::evals(rho_lambda)
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
    eq_lambda: &[E],
    eq_y: &[E],
) -> Result<E, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let setup_len = setup.shared_matrix().total_ring_elements_at::<D>()?;
    if required > setup_len {
        return Err(AkitaError::InvalidSetup(
            "shared matrix is too small for selected verifier layout".into(),
        ));
    }
    let lambda_len = required
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("setup MLE lambda length overflow".into()))?;
    if eq_lambda.len() != lambda_len {
        return Err(AkitaError::InvalidSize {
            expected: lambda_len,
            actual: eq_lambda.len(),
        });
    }
    if eq_y.len() != D {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: eq_y.len(),
        });
    }
    let setup_view = setup.shared_matrix().ring_view::<D>(1, setup_len)?;
    let setup_entries = setup_view.as_slice();

    Ok(cfg_fold_reduce!(
        0..required,
        E::zero,
        |mut acc, lambda| {
            let ring = &setup_entries[lambda];
            let mut ring_eval = E::zero();
            for (weight, &coeff) in eq_y.iter().zip(ring.coefficients()) {
                ring_eval += weight.mul_base(coeff);
            }
            acc += eq_lambda[lambda] * ring_eval;
            acc
        },
        |lhs, rhs| lhs + rhs
    ))
}
