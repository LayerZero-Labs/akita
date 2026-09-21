use super::super::*;
use super::{prepare_fold_relation_native, PreparedFold};
use crate::commitment::{CommitmentStatePolicy, InnerRelationState, OuterCompressionState};
use crate::compute::{
    ComputeBackendSetup, DigitRowsComputeBackend, ProverComputeStack, RuntimeRingSwitchProveBackend,
};
use jolt_field::AdditiveGroup;
use jolt_field::Unreduced;

pub(in crate::protocol::core) enum ExtensionOpeningSource<'a, G> {
    Logical(&'a [G]),
}

/// Prepare an extension-field fold directly against the native Spongefish
/// stream, including native EOR when scheduled.
#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn prepare_extension_claim_fold_native<'a, F, E, P, S, O, TS, R, SP>(
    stack: &ProverComputeStack<'_, F, O, TS, R, SP>,
    run_eor: bool,
    block_claims: ProverOpeningData<'a, E, P, F, S>,
    commitment_material: Vec<crate::types::PreparedCommitmentRelationMaterial<F>>,
    eor_source: ExtensionOpeningSource<'_, P>,
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
    P: RootProverGroupOpening<F, E, O> + RootProverGroupTensor<F, E, TS>,
    S: InnerRelationState<F> + OuterCompressionState<F>,
    TS: ComputeBackendSetup<F>,
    O: DigitRowsComputeBackend<F>,
    R: DigitRowsComputeBackend<F> + RuntimeRingSwitchProveBackend<F>,
    SP: CommitmentStatePolicy<F>,
{
    let opening_batch = block_claims.opening_layout().clone();
    let tensor = stack.tensor();
    let (protocol_points, reduction) = if run_eor {
        let ExtensionOpeningSource::Logical(groups) = eor_source;
        let eor_groups: Vec<&P> = groups.iter().collect();
        if eor_groups.len() != opening_batch.num_groups() {
            return Err(AkitaError::InvalidInput(
                "extension-opening source group count mismatch".into(),
            ));
        }
        let eor_inputs = eor_groups
            .into_iter()
            .enumerate()
            .map(|(group_index, group)| {
                let group_layout = opening_batch.group_layout(group_index)?;
                if group.num_polynomials() != group_layout.num_polynomials() {
                    return Err(AkitaError::InvalidInput(
                        "extension-opening source polynomial count mismatch".into(),
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
        let proved = prove_extension_opening_reduction_native::<F, E, P, TS>(
            tensor.backend(),
            Some(tensor.prepared()),
            &eor_inputs,
            grinding,
            level,
            if bind_protocol_points {
                "recursive"
            } else {
                "root"
            },
        )?;
        (proved.protocol_points, Some(proved.reduction))
    } else {
        let protocol_points = block_claims
            .opening_claims()
            .groups()
            .iter()
            .map(|group| group.point().to_vec())
            .collect();
        (protocol_points, None)
    };
    prepare_fold_relation_native::<F, E, P, S, O, TS, R, SP>(
        stack,
        block_claims,
        commitment_material,
        &protocol_points,
        reduction,
        &opening_batch,
        level,
        level_params,
        basis,
        bind_protocol_points,
        grinding,
    )
    .map_err(|err| AkitaError::InvalidInput(format!("finish native fold failed: {err:?}")))
}
