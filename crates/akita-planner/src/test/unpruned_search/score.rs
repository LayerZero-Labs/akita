use super::*;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum OracleObjective {
    Payload {
        proof_bytes: usize,
        setup_field_elements: usize,
    },
    SetupFirst {
        first_direct_setup_capacity: usize,
        proof_bytes: usize,
        setup_field_elements: usize,
    },
    PaddedSetupEnvelopeFirst {
        setup_envelope_capacity: usize,
        first_direct_setup_capacity: usize,
        proof_bytes: usize,
        first_direct_output_witness_len: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct OracleScore {
    objective: OracleObjective,
    legacy_root_output_witness_len: Option<usize>,
    descriptor: Vec<u8>,
}

pub(super) fn schedule_descriptor_bytes(
    candidate: &ScheduleCandidate,
) -> Result<Vec<u8>, AkitaError> {
    if candidate.folds.is_empty() {
        return Ok(candidate.terminal.params.canonical_descriptor_bytes());
    }
    let steps = candidate
        .folds
        .iter()
        .map(|fold| akita_types::FoldScheduleDescriptorStep {
            params: fold.params.as_ref(),
            payload_mode: fold.params.payload_mode,
            input_witness_len: fold.input_witness_len,
            output_witness_len: fold.output_witness_len,
        });
    let terminal = akita_types::TerminalFoldParams {
        fold_challenge_config: candidate.terminal.sparse_challenge_config,
        response_shape: candidate.terminal.response_shape.clone(),
        input_witness_len: candidate.terminal.input_witness_len,
        ..candidate.terminal.params.clone()
    };
    let mut descriptor = Vec::new();
    akita_types::FoldSchedule::append_descriptor_bytes_from_steps(
        &mut descriptor,
        steps,
        &terminal,
    )?;
    Ok(descriptor)
}

pub(super) fn score(
    policy: &PlannerPolicy,
    candidate: &ScheduleCandidate,
) -> Result<OracleScore, AkitaError> {
    let first_direct_setup_capacity = candidate
        .first_direct_setup_field_len
        .map(|natural_len| padded_setup_prefix_len(natural_len.get()));
    let objective = match policy.selection_policy {
        crate::SelectionPolicyId::MinEstimatedProofPayloadV2 => OracleObjective::Payload {
            proof_bytes: candidate.cost.proof_bytes(),
            setup_field_elements: candidate.setup_field_elements,
        },
        crate::SelectionPolicyId::MinFirstDirectSetupThenPayloadV2 => OracleObjective::SetupFirst {
            first_direct_setup_capacity: first_direct_setup_capacity.ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "unpruned setup-first candidate is missing direct setup size".into(),
                )
            })?,
            proof_bytes: candidate.cost.proof_bytes(),
            setup_field_elements: candidate.setup_field_elements,
        },
        crate::SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3 => {
            OracleObjective::PaddedSetupEnvelopeFirst {
                setup_envelope_capacity: padded_setup_prefix_len(candidate.setup_field_elements),
                first_direct_setup_capacity: first_direct_setup_capacity.ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "unpruned padded-envelope-first candidate is missing direct setup size"
                            .into(),
                    )
                })?,
                proof_bytes: candidate.cost.proof_bytes(),
                first_direct_output_witness_len: candidate.first_direct_output_witness_len,
            }
        }
    };
    Ok(OracleScore {
        objective,
        legacy_root_output_witness_len: (policy.selection_policy
            != crate::SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3)
            .then(|| {
                candidate
                    .folds
                    .first()
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "complete schedule is missing its root fold".into(),
                        )
                    })
                    .map(|fold| fold.output_witness_len)
            })
            .transpose()?,
        descriptor: schedule_descriptor_bytes(candidate)?,
    })
}
