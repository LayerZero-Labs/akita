//! Setup-product operations over backend-owned setup storage.

use akita_error::AkitaError;
use akita_params::{CommittedGroupParams, RelationAddressGeometry, SetupPrefixSlotId};
use akita_types::RingRelationInstance;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;

/// Public inputs fixing one setup-product computation.
pub struct Stage3Request<'a, F: Field, E: Field, ProofSession> {
    pub session: &'a ProofSession,
    pub level: u32,
    pub prefix: &'a SetupPrefixSlotId,
    pub parameters: &'a CommittedGroupParams,
    pub next_parameters: &'a CommittedGroupParams,
    pub relation: &'a RingRelationInstance<F>,
    pub tau1: &'a [E],
    pub alpha: E,
    pub stage2_challenges: &'a [E],
    pub address_geometry: RelationAddressGeometry,
}

/// Arithmetic for Stage 3. The prover owns prefix selection and transcript order.
pub trait OpaqueStage3Kernel<F: Field, E: Field>: super::ProofScopeConsumer {
    type Stage3SessionHandle: Send;

    /// Extension seam: prepare the outer contribution and all inherited setup claims.
    #[cfg(feature = "external-setup-accumulation")]
    fn prepare_setup_batch(
        &self,
        _request: super::SetupBatchRequest<'_, F, E, Self::ProofSessionHandle>,
    ) -> Result<
        (
            akita_types::setup_claims::SetupBatchStatement<E>,
            Self::Stage3SessionHandle,
        ),
        AkitaError,
    > {
        Err(AkitaError::InvalidInput(
            "backend does not support integrated setup accumulation".into(),
        ))
    }

    /// Fix batching coefficients after the complete statement has entered the transcript.
    #[cfg(feature = "external-setup-accumulation")]
    fn bind_setup_batch(
        &self,
        _session: &mut Self::Stage3SessionHandle,
        _statement: &akita_types::setup_claims::SetupBatchStatement<E>,
        _coefficients: &[E],
    ) -> Result<E, AkitaError> {
        Err(AkitaError::InvalidInput(
            "backend does not support integrated setup accumulation".into(),
        ))
    }

    fn begin_stage3(
        &self,
        request: Stage3Request<'_, F, E, Self::ProofSessionHandle>,
    ) -> Result<(E, Self::Stage3SessionHandle), AkitaError>;
    fn stage3_round_polynomial(
        &self,
        session: &mut Self::Stage3SessionHandle,
        round: usize,
        claim: E,
    ) -> Result<UnivariatePoly<E>, AkitaError>;
    fn bind_stage3_challenge(
        &self,
        session: &mut Self::Stage3SessionHandle,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError>;
    fn finish_stage3(&self, session: Self::Stage3SessionHandle) -> Result<E, AkitaError>;
}
