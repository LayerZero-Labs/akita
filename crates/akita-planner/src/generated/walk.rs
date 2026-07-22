//! Canonical walker for compact generated schedule rows.
//!
//! [`walk_generated_schedule_entry`] is the single implementation shared by
//! runtime materialization ([`crate::schedule_from_entry`]) and admissibility
//! checks ([`super::validate::validate_generated_schedule_entry`]). Both paths
//! expand every typed fold once and recompute witness transitions and
//! proof-byte totals.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::{AkitaError, Prime128OffsetA7F7};
use akita_types::{
    extension_opening_reduction_level_bytes, level_proof_bytes, terminal_response_bytes,
    AkitaScheduleInputs, AkitaScheduleLookupKey, PlannedFoldSchedule, PolynomialGroupLayout,
    PrecommittedLevelParams, TerminalResponseShape,
};

use crate::generated::{
    validate_entry_key, GeneratedFoldScheduleEntry, GeneratedRootFinalChallenge,
};
use crate::group_batch::multi_group_root_precommitted_groups_for_open_basis;
use crate::schedule_params::{
    materialize_candidate_schedule, planned_next_witness_len, stage3_payload_bytes_for_successor,
    CandidateFoldStep, CandidateTerminalResponse,
};
use crate::PlannerPolicy;

pub(crate) struct GeneratedEntryWalkOutput {
    pub planned_schedule: PlannedFoldSchedule,
}

pub(crate) fn walk_generated_schedule_entry(
    entry: &GeneratedFoldScheduleEntry,
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: &impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<GeneratedEntryWalkOutput, AkitaError> {
    key.validate()?;
    validate_entry_key(entry, key)?;
    entry.validate()?;
    let is_multi_group = !key.precommitteds.is_empty();
    let expected_root_w_len = 1usize
        .checked_shl(key.final_group.num_vars() as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("root witness length overflow".to_string()))?;
    let field_bits = policy.decomposition.field_bits();
    let challenge_field_bits = field_bits
        .checked_mul(policy.chal_ext_degree as u32)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "generated schedule challenge field bit width overflow".to_string(),
            )
        })?;
    let root_eor_key =
        PolynomialGroupLayout::new(key.final_group.num_vars(), key.num_polynomials()?);
    let stored_root_shape = match entry.root.final_group.challenge {
        GeneratedRootFinalChallenge::Flat => TensorChallengeShape::Flat,
        GeneratedRootFinalChallenge::Tensor { fold_low_len } => TensorChallengeShape::Tensor {
            fold_low_len: fold_low_len as usize,
        },
    };
    let configured_root_shape = crate::schedule_params::optimize_fold_challenge_shape(
        fold_challenge_shape_at_level(AkitaScheduleInputs {
            num_vars: key.final_group.num_vars(),
            level: 0,
            input_witness_len: expected_root_w_len,
        }),
        usize::try_from(entry.root.final_group.commitment.geometry.live_blocks).map_err(|_| {
            AkitaError::InvalidSetup(
                "generated root live block count does not fit the target platform".to_string(),
            )
        })?,
    )?;
    if stored_root_shape != configured_root_shape {
        return Err(AkitaError::InvalidSetup(
            "generated root challenge does not match the catalog family".to_string(),
        ));
    }
    let mut root_params = if is_multi_group {
        let (precommitted_groups, precommitted_d_width) =
            multi_group_root_precommitted_groups_for_open_basis(
                key,
                policy,
                ring_challenge_config,
                entry.root.open_commit_matrix.log_basis,
            )?;
        validate_expanded_precommitted_groups(key, &precommitted_groups)?;
        entry
            .root
            .final_group
            .commitment
            .expand_to_multi_group_root_level_params_with_setup(
                policy,
                ring_challenge_config,
                stored_root_shape,
                key.final_group.num_polynomials(),
                precommitted_groups,
                precommitted_d_width,
                entry.root.open_commit_matrix,
            )?
    } else {
        entry
            .root
            .final_group
            .commitment
            .expand_to_level_params_with_setup(
                policy,
                ring_challenge_config,
                0,
                expected_root_w_len,
                stored_root_shape,
                key.final_group.num_polynomials(),
                entry.root.open_commit_matrix,
                None,
            )?
    };
    let distributed_levels = distributed_activation_depth(
        entry.root.witness_partition,
        entry
            .recursive_folds
            .iter()
            .map(|fold| fold.witness_partition),
    );
    root_params.witness_chunk =
        partition_to_chunk(entry.root.witness_partition, distributed_levels)?;
    let root_output_len = if is_multi_group {
        root_params.output_witness_len::<Prime128OffsetA7F7>(&key.opening_layout()?)?
    } else {
        planned_next_witness_len(
            field_bits,
            &root_params,
            key.final_group.num_polynomials(),
            root_params.witness_chunk.num_chunks,
        )?
    };

    let mut expanded = vec![(root_params, expected_root_w_len, root_output_len)];
    let mut input_witness_len = root_output_len;
    for (index, fold) in entry.recursive_folds.iter().enumerate() {
        let mut params = fold.witness.expand_to_level_params_with_setup(
            policy,
            ring_challenge_config,
            index + 1,
            input_witness_len,
            TensorChallengeShape::Flat,
            1,
            fold.open_commit_matrix,
            fold.incoming_setup_prefix,
        )?;
        params.witness_chunk = partition_to_chunk(fold.witness_partition, distributed_levels)?;
        let output_witness_len =
            planned_next_witness_len(field_bits, &params, 1, params.witness_chunk.num_chunks)?;
        expanded.push((params, input_witness_len, output_witness_len));
        input_witness_len = output_witness_len;
    }
    let terminal_level = entry.recursive_folds.len() + 1;
    let (terminal_params, admission_cap) = entry.terminal.expand_to_level_params(
        policy,
        ring_challenge_config,
        terminal_level,
        input_witness_len,
    )?;
    let witness_shape = TerminalResponseShape::derive(&terminal_params, admission_cap)?;
    let mut folds = Vec::with_capacity(expanded.len());
    let mut total_bytes = 0usize;
    for (fold_level, (lp, input_witness_len, output_witness_len)) in expanded.iter().enumerate() {
        let next_lp = expanded.get(fold_level + 1).map(|(params, _, _)| params);
        let binds_terminal = next_lp.is_none();
        let direct_level_bytes = level_proof_bytes(
            field_bits,
            challenge_field_bits,
            lp,
            next_lp,
            *output_witness_len,
            if binds_terminal {
                Some(akita_types::NextWitnessBindingPolicy::TerminalInnerState)
            } else {
                Some(akita_types::NextWitnessBindingPolicy::OuterCommitment)
            },
        )?
        .checked_add(extension_opening_reduction_level_bytes(
            challenge_field_bits,
            policy.claim_ext_degree,
            fold_level,
            root_eor_key,
            *input_witness_len,
        )?)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("generated level byte count overflow".to_string())
        })?;
        let stage3_bytes =
            stage3_payload_bytes_for_successor(policy, next_lp, *output_witness_len)?;
        total_bytes = total_bytes
            .checked_add(direct_level_bytes)
            .and_then(|value| value.checked_add(stage3_bytes))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("generated proof byte total overflow".to_string())
            })?;
        folds.push(CandidateFoldStep {
            params: lp.clone(),
            input_witness_len: *input_witness_len,
            output_witness_len: *output_witness_len,
            estimated_direct_payload_bytes: direct_level_bytes,
            estimated_stage3_payload_bytes: stage3_bytes,
        });
    }
    let terminal_direct_bytes = akita_types::FOLD_GRIND_NONCE_BYTES
        .checked_add(extension_opening_reduction_level_bytes(
            challenge_field_bits,
            policy.claim_ext_degree,
            terminal_level,
            root_eor_key,
            input_witness_len,
        )?)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("terminal direct byte count overflow".to_string())
        })?;
    let terminal_bytes = terminal_response_bytes(field_bits, &witness_shape);
    total_bytes = total_bytes
        .checked_add(terminal_direct_bytes)
        .and_then(|value| value.checked_add(terminal_bytes))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("generated proof byte total overflow".to_string())
        })?;
    if total_bytes == 0 {
        return Err(AkitaError::InvalidSetup(
            "generated schedule validates to zero proof bytes".to_string(),
        ));
    }
    let planned_schedule = materialize_candidate_schedule(
        total_bytes,
        folds,
        CandidateTerminalResponse {
            params: terminal_params,
            sparse_challenge_config: ring_challenge_config(
                entry.terminal.inner_commit_matrix.ring_dimension as usize,
            )?,
            input_witness_len,
            estimated_direct_payload_bytes: terminal_direct_bytes,
            response_shape: witness_shape,
            estimated_payload_bytes: terminal_bytes,
        },
    )?;

    Ok(GeneratedEntryWalkOutput { planned_schedule })
}

fn partition_to_chunk(
    partition: crate::generated::GeneratedWitnessPartition,
    activated_levels: usize,
) -> Result<akita_types::ChunkedWitnessCfg, AkitaError> {
    match partition {
        crate::generated::GeneratedWitnessPartition::Single => {
            Ok(akita_types::ChunkedWitnessCfg::default_non_chunked())
        }
        crate::generated::GeneratedWitnessPartition::Distributed { num_chunks } => {
            let cfg = akita_types::ChunkedWitnessCfg {
                num_chunks: num_chunks as usize,
                num_activated_levels: activated_levels,
            };
            cfg.validate()?;
            Ok(cfg)
        }
    }
}

fn distributed_activation_depth(
    current: crate::generated::GeneratedWitnessPartition,
    following: impl Iterator<Item = crate::generated::GeneratedWitnessPartition>,
) -> usize {
    if !matches!(
        current,
        crate::generated::GeneratedWitnessPartition::Distributed { .. }
    ) {
        return 0;
    }
    1 + following
        .take_while(|partition| {
            matches!(
                partition,
                crate::generated::GeneratedWitnessPartition::Distributed { .. }
            )
        })
        .count()
}

fn validate_expanded_precommitted_groups(
    key: &AkitaScheduleLookupKey,
    groups: &[PrecommittedLevelParams],
) -> Result<(), AkitaError> {
    if groups.len() != key.precommitteds.len() {
        return Err(AkitaError::InvalidSetup(format!(
            "multi-group root precommitted group count mismatch: expected {}, got {}",
            key.precommitteds.len(),
            groups.len()
        )));
    }
    for (expected, actual) in key.precommitteds.iter().zip(groups) {
        if &actual.layout != expected {
            return Err(AkitaError::InvalidSetup(
                "multi-group root expanded precommitted layout does not match frozen key"
                    .to_string(),
            ));
        }
    }
    Ok(())
}
