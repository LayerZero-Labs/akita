use super::super::*;
use super::{finish_prepared_fold, prepare_non_eor_opening, FinishFoldArgs, PreparedFold};
use crate::compute::{
    ComputeBackendSetup, DigitRowsComputeBackend, ProverComputeStack, RuntimeOpeningProveBackendFor,
};
use crate::RootTensorProjectionPoly;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::AdditiveGroup;

pub(in crate::protocol::core) enum ExtensionOpeningSource<'a, G> {
    CurrentClaims,
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
    P: RootProverGroupOpening<F, E, O> + RootProverGroupTensor<F, E, TS>,
    V: FnOnce() -> Result<(), AkitaError>,
    TS: ComputeBackendSetup<F>,
    C: ComputeBackendSetup<F>,
    O: DigitRowsComputeBackend<F> + RuntimeOpeningProveBackendFor<F, RootTensorProjectionPoly<F>>,
    R: DigitRowsComputeBackend<F>,
{
    let opening_batch = block_claims
        .opening_claims()
        .layout()
        .map_err(|err| AkitaError::InvalidInput(format!("opening batch layout failed: {err:?}")))?;
    let tensor = stack.tensor();
    let (protocol_points, row_coefficients, reduction) = if run_eor {
        let eor_groups: Vec<&P> = match eor_source {
            ExtensionOpeningSource::CurrentClaims => block_claims.group_sources().collect(),
            ExtensionOpeningSource::Logical(groups) => groups.iter().collect(),
        };
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
            let _span = tracing::info_span!(
                "extension_transform_polys",
                num_claims = opening_batch.num_total_polynomials()
            )
            .entered();
            let mut transformed = Vec::with_capacity(opening_batch.num_total_polynomials());
            for group_index in 0..opening_batch.num_groups() {
                let group_dims = level_params.group_role_dims(&opening_batch, group_index)?;
                transformed.extend(block_claims.group_source(group_index)?.tensor_project(
                    tensor.backend(),
                    Some(tensor.prepared()),
                    group_dims.d_a(),
                )?);
            }
            transformed
        };
        let fold_refs = transformed.iter().collect::<Vec<_>>();
        let transformed_block_claims = block_claims.regroup_polynomial_refs(&fold_refs)?;
        finish_prepared_fold::<
            F,
            E,
            T,
            PreparedProverGroup<'_, RootTensorProjectionPoly<F>>,
            C,
            O,
            TS,
            R,
        >(FinishFoldArgs {
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
