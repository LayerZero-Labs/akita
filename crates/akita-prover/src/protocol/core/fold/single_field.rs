// Explicit imports only: the compiler enforces that the single-field path has
// no extension-opening-reduction or tensor-projection symbols in scope.
use super::{prepare_fold_relation_native, PreparedFold};
use crate::commitment::{CommitmentStatePolicy, InnerRelationState, OuterCompressionState};
use crate::compute::{
    ComputeBackendSetup, DigitRowsComputeBackend, ProverComputeStack, RuntimeRingSwitchProveBackend,
};
use crate::protocol::core::RootProverGroupOpening;
use crate::ProverOpeningData;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::{BasisMode, CommittedGroupParams, FpExtEncoding};
use jolt_field::{AdditiveGroup, CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};
use jolt_field::{Fold, Unreduced};

/// Prepare a degree-one fold directly against the native Spongefish stream.
#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn prepare_single_field_fold_native<'a, F, E, P, S, O, TS, R, SP>(
    stack: &ProverComputeStack<'_, F, O, TS, R, SP>,
    block_claims: ProverOpeningData<'a, E, P, F, S>,
    commitment_material: Vec<crate::types::PreparedCommitmentRelationMaterial<F>>,
    bind_protocol_points: bool,
    grinding: &mut akita_types::NativeProverGrinding<'_>,
    level: u32,
    level_params: &CommittedGroupParams,
    basis: BasisMode,
) -> Result<PreparedFold<F, E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring + Unreduced + 'static,
    <F as Unreduced>::Wide: From<F> + AdditiveGroup,
    E: FpExtEncoding<F>
        + ExtField<F>
        + Unreduced
        + Fold
        + Ring
        + MulBaseUnreduced<F>
        + AkitaSerialize,
    P: RootProverGroupOpening<F, E, O>,
    S: InnerRelationState<F> + OuterCompressionState<F>,
    O: DigitRowsComputeBackend<F>,
    TS: ComputeBackendSetup<F>,
    R: DigitRowsComputeBackend<F> + RuntimeRingSwitchProveBackend<F>,
    SP: CommitmentStatePolicy<F>,
{
    let opening_batch = block_claims.opening_layout().clone();
    let protocol_points = block_claims
        .opening_claims()
        .groups()
        .iter()
        .map(|group| group.point().to_vec())
        .collect::<Vec<_>>();
    prepare_fold_relation_native::<F, E, P, S, O, TS, R, SP>(
        stack,
        block_claims,
        commitment_material,
        &protocol_points,
        None,
        &opening_batch,
        level,
        level_params,
        basis,
        bind_protocol_points,
        grinding,
    )
    .map_err(|err| AkitaError::InvalidInput(format!("finish native fold failed: {err:?}")))
}
