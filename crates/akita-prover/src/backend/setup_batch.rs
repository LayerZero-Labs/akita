//! Shared setup batch arithmetic and optional outer-fold preparation.
use akita_error::AkitaError;
use akita_sumcheck::SumcheckKernel;
use akita_types::setup_claims::{
    SetupBatchMemory, SetupBatchStatement, SetupPolynomialId, SetupQuery,
};
use jolt_field::{CanonicalEncoding, Field};

pub trait SetupBatchKernel<E: Field>: SumcheckKernel<E> {
    fn setup_evaluation(&self) -> Result<E, AkitaError>;
}

pub trait SetupBatchBackend<F: Field + CanonicalEncoding, E: Field> {
    type PreparedQuery;
    type BatchKernel: SetupBatchKernel<E>;

    /// Answer one query during native advice preparation and retain reusable contractions.
    fn prepare_setup_query(
        &self,
        polynomial: &SetupPolynomialId,
        query: &SetupQuery<E>,
        coefficient_len: usize,
        memory: SetupBatchMemory,
    ) -> Result<(E, Self::PreparedQuery), AkitaError>;

    /// Consume prepared state only after binding the complete statement and drawing its coefficients.
    fn begin_setup_batch(
        &self,
        statement: &SetupBatchStatement<E>,
        prepared: Vec<Self::PreparedQuery>,
        coefficients: &[E],
    ) -> Result<Self::BatchKernel, AkitaError>;
}

/// Public inputs to the outer fold's setup contribution. Its query is derived by the backend.
pub struct SetupBatchRequest<'a, F: Field, E: Field, ProofSession> {
    pub session: &'a ProofSession,
    pub level: u32,
    pub inherited: &'a SetupBatchStatement<E>,
    pub parameters: &'a akita_params::CommittedGroupParams,
    pub relation: &'a akita_types::RingRelationInstance<F>,
    pub tau1: &'a [E],
    pub alpha: E,
    pub stage2_challenges: &'a [E],
    pub address_geometry: akita_params::RelationAddressGeometry,
}
