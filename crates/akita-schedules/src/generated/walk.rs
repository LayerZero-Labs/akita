//! Canonical walker for compact generated schedule rows.
//!
//! [`walk_generated_schedule_entry`] is the single implementation shared by
//! runtime materialization ([`crate::schedule_from_entry`]) and admissibility
//! checks ([`super::validate::validate_generated_schedule_entry`]). Both paths
//! expand every typed fold once and recompute witness transitions and
//! proof-byte totals.

use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_types::{
    extension_opening_reduction_level_bytes, AkitaScheduleLookupKey, GroupOpenPhaseParams,
    PlannedFoldSchedule, PolynomialGroupLayout, TailSegmentGroupLayout, TailSegmentLayout,
    TerminalResponseShape,
};

use crate::generated::{validate_entry_key, GeneratedFoldScheduleEntry};
use crate::group_batch::multi_group_root_precommitted_groups_for_open_basis;
use crate::runtime::{
    materialize_candidate_schedule, nonterminal_level_payload_bytes, planned_next_witness_len,
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
) -> Result<GeneratedEntryWalkOutput, AkitaError> {
    key.validate(policy.decomposition.field_bits())?;
    validate_entry_key(entry, key)?;
    entry.validate()?;
    let is_multi_group = !key.precommitteds.is_empty();
    let expected_root_w_len = 1usize
        .checked_shl(key.final_group.num_vars() as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("root witness length overflow".to_string()))?;
    let field_bits = policy.decomposition.field_bits();
    let challenge_field_bits = policy.challenge_field_bits()?;
    let mut root_params = if is_multi_group {
        let (precommitted_groups, precommitted_d_width) =
            multi_group_root_precommitted_groups_for_open_basis(
                key,
                entry.root.precommitted_groups,
                policy,
                ring_challenge_config,
                entry.root.open_commit_matrix.log_basis,
                entry.root.open_commit_matrix.ring_dimension as usize,
            )?;
        validate_expanded_precommitted_groups(key, &precommitted_groups)?;
        entry
            .root
            .group
            .expand_to_multi_group_root_level_params_with_setup(
                policy,
                ring_challenge_config,
                entry.root.group.opening_method,
                key.final_group.num_polynomials(),
                entry.root.group.num_digits_inner,
                entry.root.group.num_digits_fold,
                precommitted_groups,
                precommitted_d_width,
                entry.root.open_commit_matrix,
            )?
    } else {
        entry.root.group.expand_to_level_params_with_setup(
            policy,
            akita_types::CommitmentPayloadMode::Compressed,
            entry.root.group.opening_method,
            ring_challenge_config,
            0,
            entry.root.group.num_digits_inner,
            entry.root.group.num_digits_fold,
            None,
            expected_root_w_len,
            key.final_group.num_polynomials(),
            entry.root.open_commit_matrix,
            None,
        )?
    };
    let distributed_levels = distributed_activation_depth(
        entry.root.witness_chunks,
        entry.recursive_folds.iter().map(|fold| fold.witness_chunks),
    );
    root_params.witness_chunk = partition_to_chunk(entry.root.witness_chunks, distributed_levels)?;
    let root_output_len = if is_multi_group {
        root_params.output_witness_len_for_field_bits(
            field_bits,
            policy.claim_ext_degree,
            &key.opening_layout()?,
        )?
    } else {
        planned_next_witness_len(
            field_bits,
            policy.claim_ext_degree,
            &root_params,
            key.final_group.num_polynomials(),
            root_params.witness_chunk.num_chunks,
        )?
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "generated root uses unsupported compression-source geometry".to_string(),
            )
        })?
    };

    let mut expanded = vec![(root_params, expected_root_w_len, root_output_len)];
    let mut input_witness_len = root_output_len;
    for (index, fold) in entry.recursive_folds.iter().enumerate() {
        let mut params = fold.group.expand_to_level_params_with_setup(
            policy,
            fold.payload_mode,
            fold.group.opening_method,
            ring_challenge_config,
            index + 1,
            None,
            fold.group.num_digits_fold,
            fold.response_l2_sq_cap,
            input_witness_len,
            1,
            fold.open_commit_matrix,
            fold.setup_prefix,
        )?;
        params.witness_chunk = partition_to_chunk(fold.witness_chunks, distributed_levels)?;
        let output_witness_len = planned_next_witness_len(
            field_bits,
            policy.claim_ext_degree,
            &params,
            1,
            params.witness_chunk.num_chunks,
        )?
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "generated recursive fold {index} uses unsupported compression-source geometry"
            ))
        })?;
        expanded.push((params, input_witness_len, output_witness_len));
        input_witness_len = output_witness_len;
    }
    let terminal_level = entry.recursive_folds.len() + 1;
    let terminal_params = entry.terminal.expand_to_level_params(
        policy,
        ring_challenge_config,
        terminal_level,
        input_witness_len,
    )?;
    let z_coords = terminal_params
        .inner_width()
        .checked_mul(terminal_params.d_a())
        .ok_or_else(|| AkitaError::InvalidSetup("terminal z coordinates overflow".into()))?;
    let e_field_elems = terminal_params
        .blocks
        .live_blocks
        .checked_mul(terminal_params.d_a())
        .ok_or_else(|| AkitaError::InvalidSetup("terminal e coordinates overflow".into()))?;
    let t_field_elems = terminal_params
        .blocks
        .live_blocks
        .checked_mul(terminal_params.inner.matrix.output_rank())
        .and_then(|value| value.checked_mul(terminal_params.d_a()))
        .ok_or_else(|| AkitaError::InvalidSetup("terminal t coordinates overflow".into()))?;
    let logical_num_elems = z_coords
        .checked_add(e_field_elems)
        .and_then(|value| value.checked_add(t_field_elems))
        .ok_or_else(|| AkitaError::InvalidSetup("terminal response coordinates overflow".into()))?;
    let z_payload_bytes = usize::try_from(entry.terminal.z_payload_bytes).map_err(|_| {
        AkitaError::InvalidSetup(
            "generated terminal payload budget does not fit the target platform".into(),
        )
    })?;
    let witness_shape = TerminalResponseShape {
        layout: TailSegmentLayout {
            ring_dimension: terminal_params.d_a(),
            groups: vec![TailSegmentGroupLayout {
                z_coords,
                e_field_elems,
                t_field_elems,
                z_linf_cap: entry.terminal.z_linf_cap,
                z_rice_low_bits: entry.terminal.z_rice_low_bits,
                z_payload_bytes,
            }],
            logical_num_elems,
        },
    };
    let mut folds = Vec::with_capacity(expanded.len());
    let mut total_bytes = 0usize;
    for (fold_level, (lp, input_witness_len, output_witness_len)) in expanded.iter().enumerate() {
        let next_lp = expanded.get(fold_level + 1).map(|(params, _, _)| params);
        let (direct_level_bytes, stage3_bytes) = nonterminal_level_payload_bytes(
            policy,
            lp,
            next_lp,
            *input_witness_len,
            *output_witness_len,
        )?;
        total_bytes = total_bytes
            .checked_add(direct_level_bytes)
            .and_then(|value| value.checked_add(stage3_bytes))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("generated proof byte total overflow".to_string())
            })?;
        folds.push(CandidateFoldStep {
            params: std::sync::Arc::new(lp.clone()),
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
            PolynomialGroupLayout::singleton(akita_types::padded_boolean_opening_vars(
                input_witness_len,
            )?),
        )?)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("terminal direct byte count overflow".to_string())
        })?;
    let terminal_bytes = akita_types::terminal_response_planner_bytes(
        field_bits,
        &witness_shape,
        terminal_params.response_l2_sq_cap(),
    );
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
    let mut setup_field_elements = 1;
    for fold in &folds {
        akita_types::accumulate_matrix_field_elements_for_level(
            &fold.params,
            &mut setup_field_elements,
        )?;
    }
    akita_types::accumulate_terminal_matrix_field_elements(
        &terminal_params,
        &mut setup_field_elements,
    )?;
    let planned_schedule = materialize_candidate_schedule(
        total_bytes,
        setup_field_elements,
        None,
        policy.selection_policy,
        &key.opening_layout()?,
        folds,
        CandidateTerminalResponse {
            params: terminal_params,
            sparse_challenge_config: if entry.terminal.response_l2_sq_cap.is_some() {
                akita_challenges::selective_l2_challenge_config(
                    entry.terminal.inner_commit_matrix.ring_dimension as usize,
                )
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "generated terminal L2 route has no certified operator-norm challenge"
                            .into(),
                    )
                })?
            } else {
                ring_challenge_config(entry.terminal.inner_commit_matrix.ring_dimension as usize)?
            },
            input_witness_len,
            estimated_direct_payload_bytes: terminal_direct_bytes,
            response_shape: witness_shape,
            estimated_payload_bytes: terminal_bytes,
        },
    )?;
    Ok(GeneratedEntryWalkOutput { planned_schedule })
}

fn partition_to_chunk(
    witness_chunks: u32,
    activated_levels: usize,
) -> Result<akita_types::ChunkedWitnessCfg, AkitaError> {
    // A chunk count of 1 is the non-chunked layout; the enum that used to spell
    // that distinction carried no other information.
    if witness_chunks <= 1 {
        return Ok(akita_types::ChunkedWitnessCfg::default_non_chunked());
    }
    let cfg = akita_types::ChunkedWitnessCfg {
        num_chunks: witness_chunks as usize,
        num_activated_levels: activated_levels,
    };
    cfg.validate()?;
    Ok(cfg)
}

fn distributed_activation_depth(current: u32, following: impl Iterator<Item = u32>) -> usize {
    if current <= 1 {
        return 0;
    }
    1 + following.take_while(|chunks| *chunks > 1).count()
}

fn validate_expanded_precommitted_groups(
    key: &AkitaScheduleLookupKey,
    groups: &[GroupOpenPhaseParams],
) -> Result<(), AkitaError> {
    if groups.len() != key.precommitteds.len() {
        return Err(AkitaError::InvalidSetup(format!(
            "multi-group root precommitted group count mismatch: expected {}, got {}",
            key.precommitteds.len(),
            groups.len()
        )));
    }
    for (expected, actual) in key.precommitteds.iter().zip(groups) {
        if &actual.profile != expected {
            return Err(AkitaError::InvalidSetup(
                "multi-group root expanded precommitted layout does not match frozen key"
                    .to_string(),
            ));
        }
    }
    Ok(())
}
