//! Canonical walker for compact generated schedule rows.
//!
//! [`walk_generated_schedule_entry`] is the single implementation shared by
//! runtime materialization ([`crate::schedule_from_entry`]) and admissibility
//! checks ([`super::validate::validate_generated_schedule_entry`]). Both paths
//! expand every fold step once, audit SIS ranks via
//! [`GeneratedFoldStep::expand_to_level_params`], and recompute witness
//! transitions and proof-byte totals.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::{AkitaError, Prime128OffsetA7F7};
use akita_types::{
    direct_witness_bytes, extension_opening_reduction_level_bytes, level_proof_bytes,
    segment_typed_witness_shape_from_groups, AkitaScheduleInputs, AkitaScheduleLookupKey,
    DirectStep, FoldStep, LevelParams, PolynomialGroupLayout, PrecommittedLevelParams,
    RelationMatrixRowLayout, Schedule, SetupContributionMode,
};

use crate::generated::{
    validate_entry_key, GeneratedFold, GeneratedFoldStep, GeneratedScheduleTableEntry,
};
use crate::group_batch::multi_group_root_precommitted_groups;
use crate::schedule_params::planned_next_witness_len;
use crate::PlannerPolicy;

pub(crate) struct GeneratedEntryWalkOutput {
    pub total_bytes: usize,
    pub schedule: Schedule,
}

pub(crate) fn walk_generated_schedule_entry(
    entry: &GeneratedScheduleTableEntry,
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: &impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<GeneratedEntryWalkOutput, AkitaError> {
    key.validate()?;
    validate_entry_key(entry, key)?;
    entry.validate()?;
    reject_scalar_recursive_catalog_row(entry, key)?;

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
    let mut folds = Vec::with_capacity(entry.folds.len());
    let mut current_w_len = expected_root_w_len;
    let mut terminal_witness_field_len = None;
    let mut last_fold_lp = None;
    let mut total_bytes = 0usize;

    for (fold_level, fold) in entry.folds.iter().enumerate() {
        let next = entry.folds.get(fold_level + 1);
        let is_terminal = next.is_none();
        let num_claims = if fold_level == 0 {
            key.final_group.num_polynomials()
        } else {
            1
        };
        let fold_inputs = AkitaScheduleInputs {
            num_vars: key.final_group.num_vars(),
            level: fold_level,
            current_w_len,
        };
        let fold_shape = fold_challenge_shape_at_level(fold_inputs);
        let mut lp = if is_multi_group && fold_level == 0 {
            validate_block_geometry(
                fold.fold_step(),
                key.final_group,
                policy,
                fold_level,
                current_w_len,
            )?;
            validate_log_basis(fold.fold_step().log_basis, policy)?;
            let (precommitted_groups, precommitted_d_width) =
                multi_group_root_precommitted_groups(key, policy, ring_challenge_config)?;
            validate_expanded_precommitted_groups(key, &precommitted_groups)?;
            let lp = expand_multi_group_root_fold_step(
                fold,
                policy,
                ring_challenge_config,
                fold_shape,
                num_claims,
                precommitted_groups,
                precommitted_d_width,
            )?;
            validate_expanded_level_params(&lp, fold.fold_step(), policy, fold_level, num_claims)?
        } else {
            expand_validated_fold_level(
                fold,
                key.final_group,
                policy,
                ring_challenge_config,
                fold_challenge_shape_at_level,
                fold_level,
                current_w_len,
                num_claims,
            )?
        };
        lp.witness_chunk = policy.witness_chunk_for_level(fold_level);
        if is_terminal && lp.has_precommitted_groups() {
            return Err(AkitaError::InvalidSetup(
                "grouped terminal fold must be followed by another fold".to_string(),
            ));
        }

        let (next_w_len, next_lp, layout) = if let Some(next) = next {
            let len = if is_multi_group && fold_level == 0 {
                lp.next_w_len::<Prime128OffsetA7F7>(
                    &key.opening_layout()?,
                    RelationMatrixRowLayout::WithDBlock,
                )?
            } else {
                planned_next_witness_len(field_bits, &lp, num_claims, lp.witness_chunk.num_chunks)?
            };
            let mut next_lp = expand_validated_fold_level(
                next,
                key.final_group,
                policy,
                ring_challenge_config,
                fold_challenge_shape_at_level,
                fold_level + 1,
                len,
                1,
            )?;
            next_lp.witness_chunk = policy.witness_chunk_for_level(fold_level + 1);
            (len, Some(next_lp), RelationMatrixRowLayout::WithDBlock)
        } else {
            let shape = segment_typed_witness_shape_from_groups(
                &lp,
                field_bits,
                [(&lp as &dyn akita_types::LevelParamsLike, 1, 1, 1)],
            )?;
            let len = shape.logical_num_elems();
            terminal_witness_field_len = Some(len);
            (len, None, RelationMatrixRowLayout::WithoutCommitmentBlocks)
        };

        let level_bytes = level_proof_bytes(
            field_bits,
            challenge_field_bits,
            &lp,
            next_lp.as_ref(),
            next_w_len,
            layout,
            if is_terminal {
                None
            } else if fold_level + 2 == entry.folds.len() {
                Some(akita_types::NextWitnessBindingPolicy::TerminalInnerState)
            } else {
                Some(akita_types::NextWitnessBindingPolicy::OuterCommitment)
            },
        )
        .checked_add(extension_opening_reduction_level_bytes(
            challenge_field_bits,
            policy.claim_ext_degree,
            fold_level,
            root_eor_key,
            current_w_len,
        )?)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("generated level byte count overflow".to_string())
        })?;
        total_bytes = total_bytes.checked_add(level_bytes).ok_or_else(|| {
            AkitaError::InvalidSetup("generated proof byte total overflow".to_string())
        })?;
        last_fold_lp = Some(lp.clone());
        folds.push(FoldStep {
            params: lp,
            current_w_len,
            next_w_len,
            level_bytes,
        });
        current_w_len = next_w_len;
    }

    let direct_current_w_len = terminal_witness_field_len.ok_or_else(|| {
        AkitaError::InvalidSetup("terminal direct step missing predecessor fold".to_string())
    })?;
    let terminal_lp = last_fold_lp.as_ref().ok_or_else(|| {
        AkitaError::InvalidSetup("terminal direct step missing predecessor fold params".to_string())
    })?;
    if terminal_lp.witness_chunk.num_chunks > 1 {
        return Err(AkitaError::InvalidSetup(
            "terminal-direct witness does not support a multi-chunk last fold level".to_string(),
        ));
    }
    let witness_shape = segment_typed_witness_shape_from_groups(
        terminal_lp,
        field_bits,
        [(terminal_lp as &dyn akita_types::LevelParamsLike, 1, 1, 1)],
    )?;
    if direct_current_w_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "generated direct step has zero witness length".to_string(),
        ));
    }
    let direct_bytes = direct_witness_bytes(field_bits, &witness_shape);
    total_bytes = total_bytes.checked_add(direct_bytes).ok_or_else(|| {
        AkitaError::InvalidSetup("generated proof byte total overflow".to_string())
    })?;
    if total_bytes == 0 {
        return Err(AkitaError::InvalidSetup(
            "generated schedule validates to zero proof bytes".to_string(),
        ));
    }
    let schedule = Schedule {
        folds,
        terminal: DirectStep {
            current_w_len: direct_current_w_len,
            witness_shape,
            direct_bytes,
        },
        total_bytes,
    };

    Ok(GeneratedEntryWalkOutput {
        total_bytes,
        schedule,
    })
}

fn reject_scalar_recursive_catalog_row(
    entry: &GeneratedScheduleTableEntry,
    key: &AkitaScheduleLookupKey,
) -> Result<(), AkitaError> {
    if !key.precommitteds.is_empty() {
        return Ok(());
    }
    for fold in entry.folds {
        if let GeneratedFold::FoldWithSetupMetadata(meta) = fold {
            if meta.setup_contribution_mode == SetupContributionMode::Recursive {
                return Err(AkitaError::InvalidSetup(
                    "scalar lookup keys (empty precommitteds) do not support recursive setup \
                     contribution; grouped-batch scheduling requires genuine precommits"
                        .to_string(),
                ));
            }
        }
    }
    Ok(())
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

#[allow(clippy::too_many_arguments)]
fn expand_validated_fold_level(
    step: &GeneratedFold,
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: &impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
    fold_level: usize,
    current_w_len: usize,
    num_claims: usize,
) -> Result<LevelParams, AkitaError> {
    let fold = step.fold_step();
    validate_block_geometry(fold, key, policy, fold_level, current_w_len)?;
    validate_log_basis(fold.log_basis, policy)?;
    let inputs = AkitaScheduleInputs {
        num_vars: key.num_vars(),
        level: fold_level,
        current_w_len,
    };
    let lp = expand_fold_step(
        step,
        policy,
        ring_challenge_config,
        fold_level,
        current_w_len,
        fold_challenge_shape_at_level(inputs),
        num_claims,
    )?;
    validate_expanded_level_params(&lp, fold, policy, fold_level, num_claims)
}

#[allow(clippy::too_many_arguments)]
fn expand_fold_step(
    step: &GeneratedFold,
    policy: &PlannerPolicy,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_level: usize,
    current_w_len: usize,
    fold_shape: TensorChallengeShape,
    num_claims: usize,
) -> Result<LevelParams, AkitaError> {
    match step {
        GeneratedFold::Fold(fold) => fold.expand_to_level_params(
            policy,
            ring_challenge_config,
            fold_level,
            current_w_len,
            fold_shape,
            num_claims,
        ),
        GeneratedFold::FoldWithSetupMetadata(fold) => fold.expand_to_level_params(
            policy,
            ring_challenge_config,
            fold_level,
            current_w_len,
            fold_shape,
            num_claims,
        ),
    }
}

fn expand_multi_group_root_fold_step(
    step: &GeneratedFold,
    policy: &PlannerPolicy,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_shape: TensorChallengeShape,
    main_num_polys: usize,
    precommitted_groups: Vec<PrecommittedLevelParams>,
    precommitted_d_width: usize,
) -> Result<LevelParams, AkitaError> {
    match step {
        GeneratedFold::Fold(fold) => fold.expand_to_multi_group_root_level_params(
            policy,
            ring_challenge_config,
            fold_shape,
            main_num_polys,
            precommitted_groups,
            precommitted_d_width,
        ),
        GeneratedFold::FoldWithSetupMetadata(fold) => fold.expand_to_multi_group_root_level_params(
            policy,
            ring_challenge_config,
            fold_shape,
            main_num_polys,
            precommitted_groups,
            precommitted_d_width,
        ),
    }
}

fn validate_log_basis(log_basis: u32, policy: &PlannerPolicy) -> Result<(), AkitaError> {
    let (min, max) = policy.basis_range;
    if log_basis < min || log_basis > max {
        return Err(AkitaError::InvalidSetup(format!(
            "generated fold step log_basis={log_basis} outside policy range [{min}, {max}]"
        )));
    }
    Ok(())
}

fn validate_block_geometry(
    step: &GeneratedFoldStep,
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    fold_level: usize,
    current_w_len: usize,
) -> Result<(), AkitaError> {
    if step.ring_d as usize != policy.ring_dimension || step.ring_d == 0 {
        return Err(AkitaError::InvalidSetup(format!(
            "generated fold step ring dimension {} does not match policy D={}",
            step.ring_d, policy.ring_dimension
        )));
    }
    if policy.ring_dimension == 0 || !policy.ring_dimension.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "generated schedule policy ring dimension must be a nonzero power of two".to_string(),
        ));
    }
    let block_index_bits = step.block_index_bits as usize;
    let position_index_bits = step.position_index_bits as usize;
    let block_index_domain_size = 1usize.checked_shl(step.block_index_bits).ok_or_else(|| {
        AkitaError::InvalidSetup(
            "generated schedule 2^block_index_bits overflows usize".to_string(),
        )
    })?;
    let num_live_blocks = step.num_live_blocks as usize;
    if num_live_blocks == 0
        || num_live_blocks > block_index_domain_size
        || num_live_blocks
            .checked_next_power_of_two()
            .is_none_or(|domain| domain != block_index_domain_size)
    {
        return Err(AkitaError::InvalidSetup(
            "generated schedule exact live block count disagrees with block-index domain"
                .to_string(),
        ));
    }
    let num_positions_per_block =
        1usize
            .checked_shl(step.position_index_bits)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "generated schedule 2^position_index_bits overflows usize".to_string(),
                )
            })?;
    if fold_level == 0 {
        // A small root polynomial may occupy only a prefix of its first ring.
        // Count that padded ring as live source storage; recursive witnesses
        // remain exactly ring-aligned below.
        let num_live_ring_elements_per_claim = current_w_len.div_ceil(policy.ring_dimension);
        let derived_num_live_blocks =
            num_live_ring_elements_per_claim.div_ceil(num_positions_per_block);
        if num_live_blocks != derived_num_live_blocks {
            return Err(AkitaError::InvalidSetup(format!(
                "generated root exact live block mismatch: stored={num_live_blocks}, derived={derived_num_live_blocks}"
            )));
        }
        let alpha = policy.ring_dimension.trailing_zeros() as usize;
        if position_index_bits
            .checked_add(block_index_bits)
            .and_then(|n| n.checked_add(alpha))
            != Some(key.num_vars().max(alpha))
        {
            return Err(AkitaError::InvalidSetup(
                "generated root geometry variable split disagrees with padded key domain"
                    .to_string(),
            ));
        }
        return Ok(());
    }

    if current_w_len == 0 || !current_w_len.is_multiple_of(policy.ring_dimension) {
        return Err(AkitaError::InvalidSetup(format!(
            "generated recursive fold level {fold_level} has invalid current_w_len={current_w_len}"
        )));
    }
    let num_ring_elems = current_w_len / policy.ring_dimension;
    let reduced_vars = num_ring_elems
        .checked_next_power_of_two()
        .ok_or_else(|| {
            AkitaError::InvalidSetup("generated recursive witness length overflow".to_string())
        })?
        .max(1)
        .trailing_zeros() as usize;
    if position_index_bits.checked_add(block_index_bits) != Some(reduced_vars) {
        return Err(AkitaError::InvalidSetup(format!(
            "generated recursive geometry mismatch at level {fold_level}: position_index_bits={position_index_bits}, block_index_bits={block_index_bits}, reduced_vars={reduced_vars}"
        )));
    }
    let derived_num_live_blocks = num_ring_elems.div_ceil(num_positions_per_block);
    if num_live_blocks != derived_num_live_blocks {
        return Err(AkitaError::InvalidSetup(format!(
            "generated recursive exact live block mismatch at level {fold_level}: stored={num_live_blocks}, derived={derived_num_live_blocks}"
        )));
    }
    Ok(())
}

fn validate_expanded_level_params(
    lp: &LevelParams,
    step: &GeneratedFoldStep,
    policy: &PlannerPolicy,
    fold_level: usize,
    num_claims: usize,
) -> Result<LevelParams, AkitaError> {
    if lp.position_index_bits() != step.position_index_bits as usize
        || lp.block_index_bits() != step.block_index_bits as usize
        || lp.num_live_blocks != step.num_live_blocks as usize
    {
        return Err(AkitaError::InvalidSetup(
            "expanded generated level has mismatched block geometry".to_string(),
        ));
    }
    if lp.log_basis != step.log_basis {
        return Err(AkitaError::InvalidSetup(
            "expanded generated level has mismatched log_basis".to_string(),
        ));
    }
    if fold_level > 0 && lp.onehot_chunk_size != 0 {
        return Err(AkitaError::InvalidSetup(
            "generated recursive level must not carry one-hot root metadata".to_string(),
        ));
    }
    if fold_level == 0
        && policy.decomposition.log_commit_bound == 1
        && policy.onehot_chunk_size == 0
    {
        return Err(AkitaError::InvalidSetup(
            "one-hot root requires onehot_chunk_size > 0".to_string(),
        ));
    }
    if fold_level == 0
        && policy.decomposition.log_commit_bound == 1
        && lp.onehot_chunk_size != policy.onehot_chunk_size
    {
        return Err(AkitaError::InvalidSetup(
            "generated one-hot root has mismatched chunk size".to_string(),
        ));
    }
    lp.num_digits_fold(num_claims, policy.decomposition.field_bits())?;
    Ok(lp.clone())
}
