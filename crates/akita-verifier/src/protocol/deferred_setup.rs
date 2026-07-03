//! Verifier-local batching for deferred setup-matrix contributions.

use std::sync::Mutex;

use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt};
use akita_transcript::labels::CHALLENGE_SETUP_BATCH;
use akita_transcript::{sample_ext_challenge, Transcript};
use akita_types::{AkitaExpandedSetup, RingSubfieldEncoding};

use super::ring_switch::RingSwitchDeferredRowEval;

pub(crate) struct DeferredSetupCheck<E: FieldCore> {
    pub(crate) prepared: RingSwitchDeferredRowEval<E>,
    pub(crate) ring_dimension: usize,
    pub(crate) x_challenges: Vec<E>,
    pub(crate) alpha: E,
    pub(crate) scale: E,
    pub(crate) weighted_claim: E,
}

pub(crate) struct DeferredSetupBatch<E: FieldCore> {
    checks: Mutex<Vec<DeferredSetupCheck<E>>>,
}

impl<E: FieldCore> DeferredSetupBatch<E> {
    pub(crate) fn new() -> Self {
        Self {
            checks: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn push(&self, check: DeferredSetupCheck<E>) -> Result<(), AkitaError> {
        self.checks
            .lock()
            .map_err(|_| AkitaError::InvalidProof)?
            .push(check);
        Ok(())
    }

    #[tracing::instrument(skip_all, name = "deferred_setup_batch")]
    pub(crate) fn verify<F, T, const D: usize>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        transcript: &mut T,
    ) -> Result<(), AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: ExtField<F> + RingSubfieldEncoding<F> + FromPrimitiveInt,
        T: Transcript<F>,
    {
        let checks = self.checks.lock().map_err(|_| AkitaError::InvalidProof)?;
        if checks.is_empty() {
            return Ok(());
        }

        let mut lhs = E::zero();
        let mut rhs = E::zero();
        for check in checks.iter() {
            if check.ring_dimension != D {
                return Err(AkitaError::InvalidProof);
            }
            let lambda = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SETUP_BATCH);
            rhs += lambda * check.weighted_claim;

            if check.scale.is_zero() {
                continue;
            }
            let setup_eval = check.prepared.setup_contribution_at_point::<F, D>(
                &check.x_challenges,
                setup,
                check.alpha,
            )?;
            lhs += lambda * check.scale * setup_eval;
        }

        if lhs != rhs {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }
}
