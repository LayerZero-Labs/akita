//! Native openings and aggregate extension-opening reduction.
use super::{
    FoldProbeOutcome, OpeningSource, PreparedGroupOpening, ProofContext, ProverHandleFamily,
};
use jolt_poly::UnivariatePoly;

use akita_error::AkitaError;
use akita_types::OpeningClaimsLayout;
use jolt_field::{CanonicalEncoding, Field};

pub struct EorGroupRequest<'a, E: Field, C, W> {
    pub source: OpeningSource<'a, C, W>,
    pub point: &'a [E],
    pub ring_dimension: usize,
}
pub struct PreparedEor<E: Field, H> {
    pub openings: Vec<E>,
    pub proof_partials: Vec<E>,
    pub handle: H,
}
pub trait OpaqueOpeningKernel<F: Field + CanonicalEncoding, E: Field>:
    ProverHandleFamily<F, E> + super::ProofScopeConsumer
{
    fn prepare_opening(
        &self,
        session: &Self::ProofSessionHandle,
        context: &ProofContext,
        source: OpeningSource<'_, Self::CommitmentHandle, Self::WitnessHandle>,
        plan: &crate::backend::ValidatedRecursiveGroupOpeningPlan<'_, E>,
    ) -> Result<PreparedGroupOpening<E, Self::PreparedOpeningHandle>, AkitaError>;
    fn probe_opening_fold(
        &self,
        context: &ProofContext,
        opening: &Self::PreparedOpeningHandle,
        plan: &crate::backend::ValidatedFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<Self::AcceptedFoldHandle>, AkitaError>;
}
pub trait OpaqueEorKernel<F: Field + CanonicalEncoding, E: Field>:
    ProverHandleFamily<F, E> + super::ProofScopeConsumer
{
    fn prepare_eor(
        &self,
        session: &Self::ProofSessionHandle,
        context: &ProofContext,
        layout: &OpeningClaimsLayout,
        groups: &[EorGroupRequest<'_, E, Self::CommitmentHandle, Self::WitnessHandle>],
    ) -> Result<PreparedEor<E, Self::EorPreparationHandle>, AkitaError>;
    fn begin_eor(
        &self,
        preparation: Self::EorPreparationHandle,
        eta: &[E],
        coefficients: &[E],
    ) -> Result<(E, Self::EorSessionHandle), AkitaError>;
    fn eor_round(
        &self,
        session: &mut Self::EorSessionHandle,
        round: usize,
        claim: E,
    ) -> Result<UnivariatePoly<E>, AkitaError>;
    fn bind_eor_round(
        &self,
        session: &mut Self::EorSessionHandle,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError>;
    fn finish_eor(&self, session: Self::EorSessionHandle) -> Result<Vec<E>, AkitaError>;
}
