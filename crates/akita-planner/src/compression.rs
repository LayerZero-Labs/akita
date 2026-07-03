//! Planner-side commitment compression plan emission.

use akita_field::AkitaError;
use akita_types::{
    plan_commitment_compression, CommitmentCompressionPlan, CompressionMapRole,
    CompressionPlanRequest, CompressionPolicy, FoldCompressionPlan, LevelParams, Step,
};

use crate::PlannerPolicy;

/// Mutable setup cursor while building compression plans for one schedule.
#[derive(Clone, Copy, Debug, Default)]
pub struct CompressionSetupCursor {
    pub offset: usize,
}

pub fn build_root_compression_plan(
    policy: &PlannerPolicy,
    root_lp: &LevelParams,
    cursor: &mut CompressionSetupCursor,
) -> Result<Option<CommitmentCompressionPlan>, AkitaError> {
    let raw_len = root_lp
        .b_key
        .row_len()
        .checked_mul(root_lp.ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup("root compression raw_len overflow".to_string()))?;
    let (plan, next_offset) = plan_commitment_compression(CompressionPlanRequest {
        policy: &policy.compression,
        role: CompressionMapRole::RootF,
        raw_len,
        log_basis: root_lp.log_basis,
        decomp: policy.decomposition,
        sis_family: policy.sis_family,
        ring_dimension: root_lp.ring_dimension,
        setup_offset: cursor.offset,
    })?;
    cursor.offset = next_offset;
    Ok(plan)
}

pub fn build_fold_compression_plans(
    policy: &PlannerPolicy,
    lp: &LevelParams,
    next_lp: &LevelParams,
    successor_is_direct: bool,
    cursor: &mut CompressionSetupCursor,
) -> Result<FoldCompressionPlan, AkitaError> {
    let v_raw_len = lp
        .d_key
        .row_len()
        .checked_mul(lp.ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup("v compression raw_len overflow".to_string()))?;
    let (v, mut setup_offset) = plan_commitment_compression(CompressionPlanRequest {
        policy: &policy.compression,
        role: CompressionMapRole::H,
        raw_len: v_raw_len,
        log_basis: lp.log_basis,
        decomp: policy.decomposition,
        sis_family: policy.sis_family,
        ring_dimension: lp.ring_dimension,
        setup_offset: cursor.offset,
    })?;

    let next_u = if successor_is_direct {
        None
    } else {
        let u_raw_len = next_lp
            .b_key
            .row_len()
            .checked_mul(next_lp.ring_dimension)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("next_u compression raw_len overflow".to_string())
            })?;
        let (plan, next_setup_offset) = plan_commitment_compression(CompressionPlanRequest {
            policy: &policy.compression,
            role: CompressionMapRole::F,
            raw_len: u_raw_len,
            log_basis: lp.log_basis,
            decomp: policy.decomposition,
            sis_family: policy.sis_family,
            ring_dimension: next_lp.ring_dimension,
            setup_offset,
        })?;
        setup_offset = next_setup_offset;
        plan
    };

    cursor.offset = setup_offset;
    Ok(FoldCompressionPlan { v, next_u })
}

/// i8 suffix digits appended to the next witness at this fold.
pub fn compression_suffix_for_fold(plan: &FoldCompressionPlan) -> usize {
    akita_types::compression_plan_suffix_digits(plan.v.as_ref()).saturating_add(
        akita_types::compression_plan_suffix_digits(plan.next_u.as_ref()),
    )
}

/// Public commitment bytes on the wire for one intermediate fold level.
pub fn fold_level_public_commit_bytes(
    policy: &CompressionPolicy,
    elem_bytes: usize,
    lp: &LevelParams,
    next_lp: &LevelParams,
    fold_compression: &FoldCompressionPlan,
    successor_is_direct: bool,
) -> (usize, usize) {
    if !policy.enabled {
        let v_bytes = lp
            .d_key
            .row_len()
            .saturating_mul(lp.ring_dimension)
            .saturating_mul(elem_bytes);
        let u_bytes = next_lp
            .b_key
            .row_len()
            .saturating_mul(next_lp.ring_dimension)
            .saturating_mul(elem_bytes);
        return (v_bytes, if successor_is_direct { 0 } else { u_bytes });
    }
    let v_bytes = fold_compression
        .v
        .as_ref()
        .map(|plan| plan.public_len.saturating_mul(elem_bytes))
        .unwrap_or_else(|| {
            lp.d_key
                .row_len()
                .saturating_mul(lp.ring_dimension)
                .saturating_mul(elem_bytes)
        });
    let u_bytes = if successor_is_direct {
        0
    } else {
        fold_compression
            .next_u
            .as_ref()
            .map(|plan| plan.public_len.saturating_mul(elem_bytes))
            .unwrap_or_else(|| {
                next_lp
                    .b_key
                    .row_len()
                    .saturating_mul(next_lp.ring_dimension)
                    .saturating_mul(elem_bytes)
            })
    };
    (v_bytes, u_bytes)
}

pub(crate) fn assign_schedule_compression_plans(
    policy: &PlannerPolicy,
    steps: &mut [Step],
) -> Result<Option<CommitmentCompressionPlan>, AkitaError> {
    let mut cursor = CompressionSetupCursor::default();
    let root_compression = match steps.first() {
        Some(Step::Fold(root_fold)) => {
            build_root_compression_plan(policy, &root_fold.params, &mut cursor)?
        }
        _ => None,
    };

    let fold_indices: Vec<usize> = steps
        .iter()
        .enumerate()
        .filter_map(|(idx, step)| matches!(step, Step::Fold(_)).then_some(idx))
        .collect();
    for &idx in &fold_indices {
        let next_fold_params = fold_indices
            .iter()
            .copied()
            .find(|next_idx| *next_idx > idx)
            .and_then(|next_idx| match &steps[next_idx] {
                Step::Fold(next_fold) => Some(next_fold.params.clone()),
                Step::Direct(_) => None,
            });
        let Step::Fold(fold) = &mut steps[idx] else {
            continue;
        };
        let successor_is_direct = next_fold_params.is_none();
        let next_params = next_fold_params.as_ref().unwrap_or(&fold.params);
        fold.compression = build_fold_compression_plans(
            policy,
            &fold.params,
            next_params,
            successor_is_direct,
            &mut cursor,
        )?;
    }

    Ok(root_compression)
}
