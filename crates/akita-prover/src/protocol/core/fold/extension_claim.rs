use super::super::*;
use super::{finish_prepared_fold, prepare_non_eor_opening, FinishFoldArgs, PreparedFold};
use crate::compute::{
    tensor_root_projection, ComputeBackendSetup, DigitRowsComputeBackend, ProverComputeStack,
    RuntimeOpeningProveBackendFor, RuntimeRootProvePoly, RuntimeTensorBackendFor,
};
use crate::RootTensorProjectionPoly;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::AdditiveGroup;
use akita_types::dispatch_for_field;

pub(in crate::protocol::core) fn extension_opening_group_inputs<'a, E, P, F>(
    claims: &ProverOpeningData<'a, E, P, F>,
    level_params: &CommittedGroupParams,
) -> Result<Vec<ExtensionOpeningGroupInput<'a, E, P>>, AkitaError>
where
    E: FieldCore,
    F: FieldCore,
{
    let opening_batch = claims.opening_claims().layout()?;
    (0..opening_batch.num_groups())
        .map(|group_index| {
            let ring_dimension = level_params
                .group_role_dims(&opening_batch, group_index)?
                .d_a();
            Ok(ExtensionOpeningGroupInput {
                polynomials: claims.group_polys(group_index)?.to_vec(),
                point: claims.opening_claims().group_point(group_index)?.to_vec(),
                ring_dimension,
            })
        })
        .collect()
}

/// Prepare a fold level when claims live in a proper extension of the coefficient field.
#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn prepare_extension_claim_fold<'a, F, E, T, P, V, C, O, TS, R>(
    stack: &ProverComputeStack<'_, F, C, O, TS, R>,
    run_eor: bool,
    block_claims: ProverOpeningData<'a, E, P, F>,
    eor_inputs: Vec<ExtensionOpeningGroupInput<'a, E, P>>,
    pad_base_evals: bool,
    transcript: &mut T,
    validate_non_eor: V,
    level_params: &CommittedGroupParams,
    basis: BasisMode,
) -> Result<PreparedFold<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F> + AdditiveGroup,
    E: FpExtEncoding<F>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + MulBaseUnreduced<F>
        + AkitaSerialize,
    T: Transcript<F> + ProverTranscriptGrind<F>,
    P: RuntimeRootProvePoly<F>,
    V: FnOnce() -> Result<(), AkitaError>,
    TS: RuntimeTensorBackendFor<F, P, E>,
    C: ComputeBackendSetup<F>,
    O: DigitRowsComputeBackend<F>
        + RuntimeOpeningProveBackendFor<F, P>
        + RuntimeOpeningProveBackendFor<F, RootTensorProjectionPoly<F>>,
    R: DigitRowsComputeBackend<F>,
{
    let opening_batch = block_claims
        .opening_claims()
        .layout()
        .map_err(|err| AkitaError::InvalidInput(format!("opening batch layout failed: {err:?}")))?;
    let fold_polys = block_claims.flat_polys();
    let tensor = stack.tensor();
    let (protocol_points, row_coefficients, reduction) = if run_eor {
        let proved = prove_extension_opening_reduction::<F, E, T, P, TS>(
            tensor.backend(),
            Some(tensor.prepared()),
            &eor_inputs,
            pad_base_evals,
            transcript,
            if pad_base_evals { "recursive" } else { "root" },
        )
        .map_err(|err| {
            AkitaError::InvalidInput(format!("root opening preparation failed: {err:?}"))
        })?;
        (
            proved.protocol_points,
            Some(proved.row_coefficients),
            Some(proved.reduction),
        )
    } else {
        let (protocol_points, row_coefficients) = prepare_non_eor_opening(
            &block_claims,
            &opening_batch,
            pad_base_evals,
            validate_non_eor,
        )?;
        (protocol_points, row_coefficients, None)
    };

    // Tensor-project only when EOR ran without base-eval padding (root geometry).
    // All other arms share one finish path.
    if run_eor && !pad_base_evals {
        let transformed: Vec<RootTensorProjectionPoly<F>> = {
            let _span =
                tracing::info_span!("extension_transform_polys", num_claims = fold_polys.len())
                    .entered();
            let mut transformed = Vec::with_capacity(fold_polys.len());
            for group_index in 0..opening_batch.num_groups() {
                let group_dims = level_params.group_role_dims(&opening_batch, group_index)?;
                let group_polys = block_claims.group_polys(group_index)?;
                transformed.extend(dispatch_for_field!(
                    ProtocolDispatchSlot::Role(RingRole::Inner),
                    F,
                    group_dims.d_a(),
                    |D| {
                        cfg_iter!(group_polys)
                            .map(|poly| {
                                tensor_root_projection::<F, P, E, TS, D>(
                                    tensor.backend(),
                                    Some(tensor.prepared()),
                                    *poly,
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()
                    }
                )?);
            }
            transformed
        };
        let fold_refs = transformed.iter().collect::<Vec<_>>();
        let transformed_block_claims = block_claims.regroup_polynomial_refs(&fold_refs)?;
        finish_prepared_fold::<F, E, T, RootTensorProjectionPoly<F>, C, O, TS, R>(FinishFoldArgs {
            stack,
            block_claims: transformed_block_claims,
            protocol_points: &protocol_points,
            reduction,
            row_coefficients,
            trace_opening_batch: &opening_batch,
            level_params,
            basis,
            pad_base_evals,
            transcript,
        })
    } else {
        finish_prepared_fold::<F, E, T, P, C, O, TS, R>(FinishFoldArgs {
            stack,
            block_claims,
            protocol_points: &protocol_points,
            reduction,
            row_coefficients,
            trace_opening_batch: &opening_batch,
            level_params,
            basis,
            pad_base_evals,
            transcript,
        })
    }
    .map_err(|err| AkitaError::InvalidInput(format!("finish prepared fold failed: {err:?}")))
}
