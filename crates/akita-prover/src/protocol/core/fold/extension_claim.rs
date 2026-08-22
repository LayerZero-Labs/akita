use super::super::*;
use super::{finish_prepared_fold, prepare_non_eor_opening, FinishFoldArgs, PreparedFold};
use crate::compute::{
    ComputeBackendSetup, DigitRowsComputeBackend, ProverComputeStack, RuntimeRingSwitchProveBackend,
};
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::AdditiveGroup;

pub(in crate::protocol::core) enum ExtensionOpeningSource<'a, G> {
    Logical(&'a [G]),
}

/// Prepare a fold level when claims live in a proper extension of the coefficient field.
#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn prepare_extension_claim_fold<'a, F, E, T, P, V, C, O, TS, R>(
    stack: &ProverComputeStack<'_, F, C, O, TS, R>,
    run_eor: bool,
    block_claims: ProverOpeningData<'a, E, P, F>,
    eor_source: ExtensionOpeningSource<'_, P>,
    pad_base_evals: bool,
    transcript: &mut T,
    level: u32,
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
    T: akita_types::ProverTranscriptGrinding<F>,
    P: RootProverGroupOpening<F, E, O> + RootProverGroupTensor<F, E, TS>,
    V: FnOnce() -> Result<(), AkitaError>,
    TS: ComputeBackendSetup<F>,
    C: ComputeBackendSetup<F>,
    O: DigitRowsComputeBackend<F>,
    R: DigitRowsComputeBackend<F> + RuntimeRingSwitchProveBackend<F>,
{
    let opening_batch = block_claims
        .opening_layout()
        .map_err(|err| AkitaError::InvalidInput(format!("opening batch layout failed: {err:?}")))?;
    let tensor = stack.tensor();
    let (protocol_points, reduction) = if run_eor {
        let ExtensionOpeningSource::Logical(groups) = eor_source;
        let eor_groups: Vec<&P> = groups.iter().collect();
        if eor_groups.len() != opening_batch.num_groups() {
            return Err(AkitaError::InvalidInput(
                "extension-opening source group count mismatch".to_string(),
            ));
        }
        let eor_inputs = eor_groups
            .into_iter()
            .enumerate()
            .map(|(group_index, group)| {
                let group_layout = opening_batch.group_layout(group_index)?;
                if group.num_polynomials() != group_layout.num_polynomials() {
                    return Err(AkitaError::InvalidInput(
                        "extension-opening source polynomial count mismatch".to_string(),
                    ));
                }
                Ok(ExtensionOpeningGroupInput {
                    group,
                    point: block_claims.opening_claims().group_point(group_index)?,
                    ring_dimension: level_params
                        .group_role_dims(&opening_batch, group_index)?
                        .d_a(),
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let proved = prove_extension_opening_reduction::<F, E, T, P, TS>(
            tensor.backend(),
            Some(tensor.prepared()),
            &eor_inputs,
            transcript,
            level,
            if pad_base_evals { "recursive" } else { "root" },
        )
        .map_err(|err| {
            AkitaError::InvalidInput(format!("root opening preparation failed: {err:?}"))
        })?;
        (proved.protocol_points, Some(proved.reduction))
    } else {
        let protocol_points =
            prepare_non_eor_opening(&block_claims, &opening_batch, validate_non_eor)?;
        (protocol_points, None)
    };

    finish_prepared_fold::<F, E, T, P, C, O, TS, R>(FinishFoldArgs {
        stack,
        block_claims,
        protocol_points: &protocol_points,
        reduction,
        trace_opening_batch: &opening_batch,
        level,
        level_params,
        basis,
        pad_base_evals,
        transcript,
    })
    .map_err(|err| AkitaError::InvalidInput(format!("finish prepared fold failed: {err:?}")))
}
