use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, Field};

/// CPU-side retention of materialized opening buffers.
///
/// These buffers are an implementation detail of the in-tree backends. The
/// protocol boundary transports only `PreparedOpeningHandle`.
pub(crate) trait PreparedGroupOpeningKernel<F, E>:
    akita_prover::ProverHandleFamily<F, E> + crate::opaque::ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    fn retain_evaluation_trace_opening(
        &self,
        proof_context: Option<&crate::opaque::ProofContext>,
        prepared: Option<&Self::PreparedSetup>,
        point: akita_types::PreparedOpeningPoint<F, E>,
        folded_by_claim: Vec<akita_types::RingVec<F>>,
        scalar_openings: Vec<E>,
    ) -> Result<crate::opaque::PreparedGroupOpening<E, Self::PreparedOpeningHandle>, AkitaError>;

    fn retain_coefficient_packing_opening(
        &self,
        proof_context: Option<&crate::opaque::ProofContext>,
        prepared: Option<&Self::PreparedSetup>,
        point: akita_types::PreparedSubringCoefficientPackingPoint<E>,
        partials_by_claim: Vec<crate::opaque::SubringCoefficientPackingPartials<F>>,
        scalar_openings: Vec<E>,
    ) -> Result<crate::opaque::PreparedGroupOpening<E, Self::PreparedOpeningHandle>, AkitaError>;

    fn terminal_evaluation_trace_opening(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        opening: Self::PreparedOpeningHandle,
    ) -> Result<Vec<akita_types::RingVec<F>>, AkitaError>;
}
