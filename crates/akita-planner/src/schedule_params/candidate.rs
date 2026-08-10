use super::*;

pub(crate) use akita_schedules::planner_support::planned_next_witness_len;
use akita_schedules::planner_support::{projected_collision_role_price, sis_key_at_dimension};

pub(crate) struct AbCommitmentCandidateRequest<'a> {
    pub(crate) policy: &'a PlannerPolicy,
    pub(crate) fold_policy: &'a dyn HonestFoldPolicy,
    pub(crate) ring_challenge_cfg: &'a SparseChallengeConfig,
    pub(crate) dimensions: CommitmentRingDims,
    pub(crate) payload_mode: akita_types::CommitmentPayloadMode,
    pub(crate) num_claims: usize,
    pub(crate) num_live_blocks: usize,
    pub(crate) num_chunks: usize,
    pub(crate) outer_slice_count: akita_types::CommitmentSliceCount,
    pub(crate) witness_norms: FoldWitnessNorms,
    pub(crate) log_basis_open: u32,
    pub(crate) width_s: usize,
    pub(crate) num_digits_outer: usize,
}

pub(crate) struct AbCommitmentCandidate {
    pub(crate) num_digits_fold: usize,
    pub(crate) inner_commit_matrix: InnerCommitMatrixParams,
    pub(crate) outer_commit_matrix: OuterCommitMatrixParams,
}

/// Derive the shared A/B commitment geometry for one planner candidate.
///
/// Root, recursive, and setup-prefix search own different enumeration and
/// scoring rules, but security sizing and complete-source admission are one
/// policy boundary. Returning `None` rejects a candidate that has no certified
/// rank or exceeds the canonical compression envelope.
pub(crate) fn derive_ab_commitment_candidate(
    request: AbCommitmentCandidateRequest<'_>,
) -> Result<Option<AbCommitmentCandidate>, AkitaError> {
    let AbCommitmentCandidateRequest {
        policy,
        fold_policy,
        ring_challenge_cfg,
        dimensions,
        payload_mode,
        num_claims,
        num_live_blocks,
        num_chunks,
        outer_slice_count,
        witness_norms,
        log_basis_open,
        width_s,
        num_digits_outer,
    } = request;
    let d_a = dimensions.d_a();
    let num_fold_coeffs = width_s
        .checked_mul(d_a)
        .and_then(|count| count.checked_mul(num_chunks))
        .ok_or_else(|| AkitaError::InvalidSetup("fold response width overflow".into()))?;
    let Ok(num_digits_fold) = fold_policy.num_digits_fold(HonestFoldSizingQuery {
        ring_dimension: d_a,
        num_claims,
        num_live_blocks,
        num_chunks,
        num_fold_coeffs,
        outer_slice_count,
        witness_norms,
        log_basis_response: log_basis_open,
        challenge_config: ring_challenge_cfg,
    }) else {
        return Ok(None);
    };
    let Some(norm_s) = rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_table_digest,
        policy.sis_modulus_profile,
        d_a,
        log_basis_open,
        ring_challenge_cfg,
        num_digits_fold,
        policy.ring_subfield_norm_bound,
    ) else {
        return Ok(None);
    };
    let Ok(inner_commit_matrix) = InnerCommitMatrixParams::try_new_with_min_rank(
        sis_key_at_dimension(policy, akita_types::SisMatrixRole::Inner, d_a, norm_s),
        width_s,
    ) else {
        return Ok(None);
    };
    let Ok(slice_geometry) = akita_types::CommitmentSliceGeometry::try_new(
        outer_slice_count,
        num_live_blocks,
        num_claims,
        inner_commit_matrix.output_rank(),
        num_digits_outer,
        d_a,
        dimensions.d_b(),
    ) else {
        return Ok(None);
    };
    let Some(norm_t) = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        akita_types::SisMatrixRole::Outer,
        dimensions.d_b(),
        log_basis_open,
    ) else {
        return Ok(None);
    };
    let Ok(outer_commit_matrix) = OuterCommitMatrixParams::try_new_with_min_rank(
        sis_key_at_dimension(
            policy,
            akita_types::SisMatrixRole::Outer,
            dimensions.d_b(),
            norm_t,
        ),
        slice_geometry.physical_input_width(),
    ) else {
        return Ok(None);
    };
    let complete_source_coefficients = outer_slice_count
        .complete_source_coefficients(outer_commit_matrix.output_rank(), dimensions.d_b())?;
    if payload_mode.is_compressed()
        && akita_types::CompressionChainPlan::try_for_complete_source(
            outer_commit_matrix.sis_modulus_profile(),
            complete_source_coefficients,
        )?
        .is_none()
    {
        return Ok(None);
    }
    Ok(Some(AbCommitmentCandidate {
        num_digits_fold,
        inner_commit_matrix,
        outer_commit_matrix,
    }))
}

mod recursive;
mod setup_prefix;

pub(crate) use recursive::{
    derive_candidate_level_params, derive_candidate_level_params_split_frontier,
    recursive_split_search_domain,
};
pub(crate) use setup_prefix::SetupPrefixSearchCache;
pub(super) use setup_prefix::{derive_setup_prefix_groups, SetupPrefixSearchRequest};

#[cfg(test)]
#[path = "../test/schedule_params_candidate.rs"]
mod tests;
