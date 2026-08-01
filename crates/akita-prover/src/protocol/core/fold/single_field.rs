// Explicit imports only: the compiler enforces that the single-field path has
// no extension-opening-reduction or root-tensor-projection symbols in scope.
use super::{finish_prepared_fold, prepare_non_eor_opening, FinishFoldArgs, PreparedFold};
use crate::compute::{ComputeBackendSetup, DigitRowsComputeBackend, ProverComputeStack};
use crate::protocol::core::RootProverGroupOpening;
use crate::{ProverOpeningData, ProverTranscriptGrind};
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps, HasWide, ReduceTo};
use akita_field::{
    AdditiveGroup, AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, HalvingField,
    MulBaseUnreduced, RandomSampling,
};
use akita_serialization::AkitaSerialize;
use akita_transcript::Transcript;
use akita_types::{BasisMode, CommittedGroupParams, FpExtEncoding};

/// Prepare a fold level when claim and coefficient fields coincide (`EXT_DEGREE == 1`).
///
/// This path never runs extension-opening reduction or root tensor projection.
#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn prepare_single_field_fold<'a, F, E, T, P, V, C, O, TS, R>(
    stack: &ProverComputeStack<'_, F, C, O, TS, R>,
    block_claims: ProverOpeningData<'a, E, P, F>,
    pad_base_evals: bool,
    transcript: &mut T,
    validate_non_eor: V,
    level_params: &CommittedGroupParams,
    basis: BasisMode,
) -> Result<PreparedFold<F, E>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + FromPrimitiveInt
        + HalvingField
        + HasWide
        + RandomSampling
        + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F> + AdditiveGroup,
    E: FpExtEncoding<F>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + MulBaseUnreduced<F>
        + AkitaSerialize,
    T: Transcript<F> + ProverTranscriptGrind<F>,
    P: RootProverGroupOpening<F, E, O>,
    V: FnOnce() -> Result<(), AkitaError>,
    C: ComputeBackendSetup<F>,
    O: DigitRowsComputeBackend<F>,
    TS: ComputeBackendSetup<F>,
    R: DigitRowsComputeBackend<F>,
{
    let opening_batch = block_claims
        .opening_claims()
        .layout()
        .map_err(|err| AkitaError::InvalidInput(format!("opening batch layout failed: {err:?}")))?;
    let (protocol_points, row_coefficients) = prepare_non_eor_opening(
        &block_claims,
        &opening_batch,
        pad_base_evals,
        validate_non_eor,
    )?;
    finish_prepared_fold::<F, E, T, P, C, O, TS, R>(FinishFoldArgs {
        stack,
        block_claims,
        protocol_points: &protocol_points,
        reduction: None,
        row_coefficients,
        trace_opening_batch: &opening_batch,
        level_params,
        basis,
        pad_base_evals,
        transcript,
    })
    .map_err(|err| AkitaError::InvalidInput(format!("finish prepared fold failed: {err:?}")))
}
