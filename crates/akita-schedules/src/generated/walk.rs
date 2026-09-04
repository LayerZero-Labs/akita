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
    // One expansion, two length sources. A grouped root pins its live length
    // and carries the frozen precommitted D segments; a scalar root derives its
    // length from the witness it was planned for.
    let length_source = if is_multi_group {
        let precommitted_groups = multi_group_root_precommitted_groups_for_open_basis(
            key,
            entry.root.precommitted_groups,
            policy,
            ring_challenge_config,
            entry.root.core.open_commit_matrix.log_basis,
        )?;
        validate_expanded_precommitted_groups(key, &precommitted_groups)?;
        crate::generated::expand::GroupLengthSource::PinnedGrouped {
            num_claims: key.final_group.num_polynomials(),
            precommitted_groups,
        }
    } else {
        crate::generated::expand::GroupLengthSource::IncomingWitness {
            input_witness_len: expected_root_w_len,
            num_claims: key.final_group.num_polynomials(),
            setup_prefix: None,
        }
    };
    let mut root_params = entry.root.core.group.expand_group(
        policy,
        ring_challenge_config,
        crate::generated::expand::GeneratedGroupExpansion {
            role: crate::generated::expand::GeneratedFoldExpansionRole::Root {
                num_digits_inner: entry.root.num_digits_inner,
            },
            payload_mode: akita_types::CommitmentPayloadMode::Compressed,
            ring_relation_mode: entry.root.core.ring_relation_mode,
            open_commit_matrix: entry.root.core.open_commit_matrix,
            // The root's own group *is* the row's lookup key.
            group: key.final_group,
            source: length_source,
        },
    )?;
    let distributed_levels = distributed_activation_depth(
        entry.root.core.witness_chunks,
        entry
            .recursive_folds
            .iter()
            .map(|fold| fold.core.witness_chunks),
    );
    root_params.witness_chunk =
        partition_to_chunk(entry.root.core.witness_chunks, distributed_levels)?;
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
        let mut params = fold.core.group.expand_group(
            policy,
            ring_challenge_config,
            crate::generated::expand::GeneratedGroupExpansion {
                role: crate::generated::expand::GeneratedFoldExpansionRole::Recursive {
                    fold_level: index + 1,
                    response_l2_sq_cap: fold.response_l2_sq_cap,
                },
                payload_mode: fold.payload_mode,
                ring_relation_mode: fold.core.ring_relation_mode,
                open_commit_matrix: fold.core.open_commit_matrix,
                // A recursive fold commits one polynomial over the witness it
                // receives, so its layout follows from that length.
                group: akita_types::PolynomialGroupLayout::singleton(
                    akita_types::padded_boolean_opening_vars(input_witness_len)?,
                ),
                source: crate::generated::expand::GroupLengthSource::IncomingWitness {
                    input_witness_len,
                    num_claims: 1,
                    setup_prefix: fold.setup_prefix,
                },
            },
        )?;
        params.witness_chunk = partition_to_chunk(fold.core.witness_chunks, distributed_levels)?;
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
    let mut predecessor_rounds = None;
    for (fold_level, (lp, input_witness_len, output_witness_len)) in expanded.iter().enumerate() {
        let opening_layout = match predecessor_rounds {
            None => key.opening_layout()?,
            Some(rounds) => {
                lp.opening_layout_for_final_group(PolynomialGroupLayout::singleton(rounds))?
            }
        };
        let successor = expanded.get(fold_level + 1).map_or_else(
            || akita_types::FoldSuccessor::Terminal(&terminal_params),
            |(params, _, _)| akita_types::FoldSuccessor::Recursive(params),
        );
        let payload = nonterminal_level_payload_bytes(
            policy,
            lp,
            &opening_layout,
            successor,
            *output_witness_len,
        )?;
        let direct_level_bytes = payload.direct;
        let stage3_bytes = payload.stage3;
        predecessor_rounds = Some(payload.relation_geometry.relation_point_variable_count());
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
    let terminal_predecessor_rounds = predecessor_rounds.ok_or_else(|| {
        AkitaError::InvalidSetup("terminal proof is missing predecessor relation geometry".into())
    })?;
    let terminal_direct_bytes = extension_opening_reduction_level_bytes(
        challenge_field_bits,
        policy.claim_ext_degree,
        PolynomialGroupLayout::singleton(terminal_predecessor_rounds),
    )?;
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
    let terminal_response = CandidateTerminalResponse {
        params: terminal_params,
        sparse_challenge_config: if entry.terminal.response_l2_sq_cap.is_some() {
            akita_challenges::selective_l2_challenge_config(
                entry.terminal.inner_commit_matrix.ring_dimension as usize,
            )
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "generated terminal L2 route has no certified operator-norm challenge".into(),
                )
            })?
        } else {
            ring_challenge_config(entry.terminal.inner_commit_matrix.ring_dimension as usize)?
        },
        input_witness_len,
        estimated_direct_payload_bytes: terminal_direct_bytes,
        response_shape: witness_shape,
        estimated_payload_bytes: terminal_bytes,
    };
    let nonce_bits = crate::runtime::candidate_grinding_nonce_bits(
        policy,
        &key.opening_layout()?,
        &folds,
        &terminal_response,
    )?;
    let nonce_bytes = akita_error::checked::div_ceil(nonce_bits, 8)
        .ok_or_else(|| AkitaError::InvalidSetup("invalid nonce stream byte width".into()))?;
    total_bytes = total_bytes
        .checked_add(nonce_bytes)
        .ok_or_else(|| AkitaError::InvalidSetup("generated proof byte total overflow".into()))?;
    let planned_schedule = materialize_candidate_schedule(
        total_bytes,
        setup_field_elements,
        None,
        policy,
        &key.opening_layout()?,
        folds,
        terminal_response,
    )?;
    Ok(GeneratedEntryWalkOutput { planned_schedule })
}

fn partition_to_chunk(
    witness_chunks: u32,
    activated_levels: usize,
) -> Result<akita_types::ChunkedWitnessCfg, AkitaError> {
    // A chunk count of 1 is the non-chunked layout; the enum that used to spell
    // that distinction carried no other information.
    if witness_chunks == 0 {
        return Err(AkitaError::InvalidSetup(
            "generated witness chunk count must be nonzero".to_string(),
        ));
    }
    if witness_chunks == 1 {
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
