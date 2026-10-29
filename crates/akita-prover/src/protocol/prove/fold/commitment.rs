use super::FoldSuccessorParams;
use crate::backend::{ProverBackend, RecursiveWitnessHandle};
use crate::protocol::ring_switch::NextWitnessState;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::GrindingReplay;
use jolt_field::{CanonicalEncoding, Field};

pub(in crate::protocol::prove) struct CommittedNextWitness<F: Field, WitnessHandle, M> {
    pub(in crate::protocol::prove) committed_witness_len: usize,
    pub(in crate::protocol::prove) plan: crate::backend::ValidatedRecursiveWitnessCommitPlan,
    pub(in crate::protocol::prove) witness_handle: WitnessHandle,
    pub(in crate::protocol::prove) binding: NextWitnessState<F>,
    pub(in crate::protocol::prove) commitment_material_handle: M,
}

/// Extension seam: commit and transcript-bind the witness consumed by the successor.
/// Shared fold orchestration for <https://github.com/LayerZero-Labs/akita/issues/285>.
/// The logical witness is computed once and retained for ring switching.
#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::prove) fn commit_next_witness<F, E, B>(
    backend: &B,
    grinding: &mut akita_types::ProverGrinding<'_>,
    level: u32,
    next_params: FoldSuccessorParams<'_>,
    expected_output_witness_len: usize,
    next_witness_binding: akita_params::NextWitnessBindingPolicy,
    witness_handle: B::WitnessHandle,
) -> Result<CommittedNextWitness<F, B::WitnessHandle, B::CommitmentMaterialHandle>, AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize,
    E: Field,
    B: ProverBackend<F, E>,
{
    let next_opening_ring_dim = next_params.inner_ring_dimension();
    let committed_witness_len = akita_params::witness_commitment_domain_len(
        witness_handle.manifest().logical_len(),
        next_opening_ring_dim,
    )?;
    if witness_handle.manifest().logical_len() != expected_output_witness_len {
        return Err(AkitaError::Internal(format!(
            "scheduled fold level {level} produced unexpected next-w length: expected={expected_output_witness_len}, actual={}",
            witness_handle.manifest().logical_len()
        )));
    }
    let _span = tracing::info_span!("commit_w_level", level).entered();
    let (parameters, source_encoding) = match next_params {
        FoldSuccessorParams::Recursive(params) => (
            crate::backend::WitnessCommitmentParameters::Recursive(params.params.clone()),
            Some(params.params.source_encoding),
        ),
        FoldSuccessorParams::Terminal(params) => (
            crate::backend::WitnessCommitmentParameters::Terminal(params.clone()),
            None,
        ),
    };
    let commit_plan = crate::backend::ValidatedRecursiveWitnessCommitPlan::new(
        witness_handle.manifest().logical_len(),
        committed_witness_len,
        next_opening_ring_dim,
        parameters,
        source_encoding,
        next_witness_binding,
    );
    let commitment_output = backend.commit_witness(witness_handle, &commit_plan)?;
    let (message, witness_handle, commitment_material_handle) =
        commitment_output.into_commitment_parts();
    drop(_span);
    let binding = match (message, next_witness_binding) {
        (
            crate::backend::NextWitnessBindingMessage::OuterPayload(public_commitment),
            akita_params::NextWitnessBindingPolicy::OuterPayload,
        ) => {
            akita_transcript::send_field_group(
                grinding.state_mut(),
                akita_types::FoldSite::NextWitnessPayload { level }.id()?,
                public_commitment.coeffs(),
            )?;
            NextWitnessState::OuterPayload(public_commitment)
        }
        (
            crate::backend::NextWitnessBindingMessage::TerminalInnerState(message),
            akita_params::NextWitnessBindingPolicy::TerminalInnerState,
        ) => {
            let FoldSuccessorParams::Terminal(terminal) = next_params else {
                return Err(AkitaError::Internal(
                    "terminal inner state bound for a recursive successor".into(),
                ));
            };
            // Under `dev-protocol` the scheduled prefix stops before
            // the last block, which travels as a digest. Otherwise the prefix
            // is the whole state.
            let (sent, last_block) = message
                .fields()
                .split_at_checked(terminal.response_shape.layout.t_field_elems())
                .ok_or_else(|| {
                    AkitaError::Internal(
                        "terminal inner state is shorter than its scheduled prefix".into(),
                    )
                })?;
            akita_transcript::send_field_group(
                grinding.state_mut(),
                akita_types::FoldSite::NextWitnessInnerState { level }.id()?,
                sent,
            )?;
            if akita_params::RECOMPUTE_LAST_BLOCK {
                akita_transcript::send_field_digest(
                    grinding.state_mut(),
                    akita_types::FoldSite::NextWitnessInnerStateDigest { level }.id()?,
                    last_block,
                )?;
            } else if !last_block.is_empty() {
                return Err(AkitaError::Internal(
                    "terminal inner state is longer than its scheduled length".into(),
                ));
            }
            NextWitnessState::TerminalInnerState(message)
        }
        _ => {
            return Err(AkitaError::Internal(
                "next witness binding message differs from scheduled binding policy".into(),
            ))
        }
    };
    Ok(CommittedNextWitness {
        plan: commit_plan,
        committed_witness_len,
        witness_handle,
        binding,
        commitment_material_handle,
    })
}
