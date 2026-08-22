// Explicit imports only: the compiler enforces that the single-field path has
// no extension-opening-reduction or tensor-projection symbols in scope.
use super::{finish_prepared_fold, prepare_non_eor_opening, FinishFoldArgs, PreparedFold};
use crate::compute::{
    ComputeBackendSetup, DigitRowsComputeBackend, ProverComputeStack, RuntimeRingSwitchProveBackend,
};
use crate::protocol::core::RootProverGroupOpening;
use crate::{ProverOpeningData, ProverTranscriptGrind};
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_transcript::Transcript;
use akita_types::{BasisMode, CommittedGroupParams, FpExtEncoding};
use jolt_field::{AdditiveGroup, CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};
use jolt_field::{Fold, Unreduced};

/// Prepare a fold level when claim and coefficient fields coincide (`EXT_DEGREE == 1`).
///
/// This path never runs extension-opening reduction or tensor projection.
#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn prepare_single_field_fold<'a, F, E, T, P, V, C, O, TS, R>(
    stack: &ProverComputeStack<'_, F, C, O, TS, R>,
    block_claims: ProverOpeningData<'a, E, P, F>,
    pad_base_evals: bool,
    transcript: &mut T,
    level: u32,
    validate_non_eor: V,
    level_params: &CommittedGroupParams,
    basis: BasisMode,
) -> Result<PreparedFold<F, E>, AkitaError>
where
    F: Field
        + CanonicalEncoding
        + akita_serialization::AkitaSerialize
        + Ring
        + Field
        + Unreduced
        + Field
        + 'static,
    <F as Unreduced>::Wide: From<F> + AdditiveGroup,
    E: FpExtEncoding<F>
        + ExtField<F>
        + Unreduced
        + Fold
        + Ring
        + MulBaseUnreduced<F>
        + AkitaSerialize,
    T: Transcript<F> + ProverTranscriptGrind<F> + akita_types::ProverTranscriptGrinding<F>,
    P: RootProverGroupOpening<F, E, O>,
    V: FnOnce() -> Result<(), AkitaError>,
    C: ComputeBackendSetup<F>,
    O: DigitRowsComputeBackend<F>,
    TS: ComputeBackendSetup<F>,
    R: DigitRowsComputeBackend<F> + RuntimeRingSwitchProveBackend<F>,
{
    let opening_batch = block_claims
        .opening_layout()
        .map_err(|err| AkitaError::InvalidInput(format!("opening batch layout failed: {err:?}")))?;
    let protocol_points = prepare_non_eor_opening(&block_claims, &opening_batch, validate_non_eor)?;
    finish_prepared_fold::<F, E, T, P, C, O, TS, R>(FinishFoldArgs {
        stack,
        block_claims,
        protocol_points: &protocol_points,
        reduction: None,
        trace_opening_batch: &opening_batch,
        level,
        level_params,
        basis,
        pad_base_evals,
        transcript,
    })
    .map_err(|err| AkitaError::InvalidInput(format!("finish prepared fold failed: {err:?}")))
}
