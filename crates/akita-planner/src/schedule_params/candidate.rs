use super::*;

pub(crate) use akita_schedules::planner_support::planned_next_witness_len;
use akita_schedules::planner_support::{
    projected_collision_role_price, selective_l2_inner_matrix, sis_key_at_dimension,
    SelectiveL2CandidateGeometry,
};

pub(crate) struct AbCommitmentCandidateRequest<'a> {
    pub(crate) policy: &'a PlannerPolicy,
    pub(crate) fold_policy: &'a dyn HonestFoldPolicy,
    pub(crate) ring_challenge_cfg: &'a SparseChallengeConfig,
    pub(crate) challenge_dimension: usize,
    pub(crate) dimensions: CommitmentRingDims,
    pub(crate) payload_mode: akita_types::CommitmentPayloadMode,
    pub(crate) num_claims: usize,
    pub(crate) num_live_ring_elements_per_claim: usize,
    pub(crate) num_live_blocks: usize,
    pub(crate) num_positions_per_block: usize,
    pub(crate) num_chunks: usize,
    pub(crate) outer_slice_count: akita_types::CommitmentSliceCount,
    pub(crate) witness_norms: FoldWitnessNorms,
    pub(crate) log_basis_open: u32,
    pub(crate) width_s: usize,
    pub(crate) num_digits_outer: usize,
    /// Optional typed-model cap for the Linf response. The universal policy
    /// remains the dominance guard.
    pub(crate) modeled_linf_cap: Option<u128>,
}

pub(crate) struct AbCommitmentCandidate {
    pub(crate) num_digits_fold: usize,
    pub(crate) inner_commit_matrix: InnerCommitMatrixParams,
    pub(crate) outer_commit_matrix: OuterCommitMatrixParams,
}

pub(super) struct InnerCommitmentCandidateRequest<'a> {
    pub(super) policy: &'a PlannerPolicy,
    pub(super) fold_policy: &'a dyn HonestFoldPolicy,
    pub(super) ring_challenge_cfg: &'a SparseChallengeConfig,
    pub(super) challenge_dimension: usize,
    pub(super) dimensions: CommitmentRingDims,
    pub(super) num_claims: usize,
    pub(super) num_live_ring_elements_per_claim: usize,
    pub(super) num_live_blocks: usize,
    pub(super) num_positions_per_block: usize,
    pub(super) num_chunks: usize,
    pub(super) witness_norms: FoldWitnessNorms,
    pub(super) log_basis_open: u32,
    pub(super) width_s: usize,
    pub(super) modeled_linf_cap: Option<u128>,
}

pub(super) struct InnerCommitmentCandidate {
    pub(super) num_digits_fold: usize,
    pub(super) inner_commit_matrix: InnerCommitMatrixParams,
}

pub(super) struct OuterCommitmentCandidateRequest<'a> {
    policy: &'a PlannerPolicy,
    dimensions: CommitmentRingDims,
    payload_mode: akita_types::CommitmentPayloadMode,
    num_claims: usize,
    num_live_blocks: usize,
    outer_slice_count: akita_types::CommitmentSliceCount,
    log_basis_open: u32,
    num_digits_outer: usize,
    inner_output_rank: usize,
}

/// Derive the one physical B matrix and complete logical B source after the
/// route-specific A rank is known.
pub(super) fn derive_outer_commitment_candidate(
    request: OuterCommitmentCandidateRequest<'_>,
) -> Result<Option<OuterCommitMatrixParams>, AkitaError> {
    let OuterCommitmentCandidateRequest {
        policy,
        dimensions,
        payload_mode,
        num_claims,
        num_live_blocks,
        outer_slice_count,
        log_basis_open,
        num_digits_outer,
        inner_output_rank,
    } = request;
    let Ok(slice_geometry) = akita_types::CommitmentSliceGeometry::try_new(
        outer_slice_count,
        num_live_blocks,
        num_claims,
        inner_output_rank,
        num_digits_outer,
        dimensions.d_a(),
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
    Ok(Some(outer_commit_matrix))
}

/// Derive the slice-independent response digit count and A matrix.
pub(super) fn derive_inner_commitment_candidate(
    request: InnerCommitmentCandidateRequest<'_>,
) -> Result<Option<InnerCommitmentCandidate>, AkitaError> {
    let InnerCommitmentCandidateRequest {
        policy,
        fold_policy,
        ring_challenge_cfg,
        challenge_dimension,
        dimensions,
        num_claims,
        num_live_ring_elements_per_claim,
        num_live_blocks,
        num_positions_per_block,
        num_chunks,
        witness_norms,
        log_basis_open,
        width_s,
        modeled_linf_cap,
    } = request;
    let d_a = dimensions.d_a();
    let num_fold_coeffs = width_s
        .checked_mul(d_a)
        .and_then(|count| count.checked_mul(num_chunks))
        .ok_or_else(|| AkitaError::InvalidSetup("fold response width overflow".into()))?;
    let Ok(universal_digits) = fold_policy.num_digits_fold(HonestFoldSizingQuery {
        ring_dimension: d_a,
        challenge_dimension,
        num_claims,
        num_live_ring_elements_per_claim,
        num_live_blocks,
        num_positions_per_block,
        num_chunks,
        num_fold_coeffs,
        witness_norms,
        log_basis_response: log_basis_open,
        challenge_config: ring_challenge_cfg,
    }) else {
        return Ok(None);
    };
    let num_digits_fold = modeled_linf_cap
        .map(|cap| {
            num_digits_for_linf_cap(cap, policy.decomposition.field_bits(), log_basis_open)
                .min(universal_digits)
        })
        .unwrap_or(universal_digits);
    let Some(norm_s) = rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_table_digest,
        policy.sis_modulus_profile,
        d_a,
        log_basis_open,
        ring_challenge_cfg,
        num_digits_fold,
        num_chunks,
    ) else {
        return Ok(None);
    };
    let Ok(inner_commit_matrix) = InnerCommitMatrixParams::try_new_with_min_rank(
        sis_key_at_dimension(policy, akita_types::SisMatrixRole::Inner, d_a, norm_s),
        width_s,
    ) else {
        return Ok(None);
    };
    Ok(Some(InnerCommitmentCandidate {
        num_digits_fold,
        inner_commit_matrix,
    }))
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
        challenge_dimension,
        dimensions,
        payload_mode,
        num_claims,
        num_live_ring_elements_per_claim,
        num_live_blocks,
        num_positions_per_block,
        num_chunks,
        outer_slice_count,
        witness_norms,
        log_basis_open,
        width_s,
        num_digits_outer,
        modeled_linf_cap,
    } = request;
    let Some(inner_candidate) =
        derive_inner_commitment_candidate(InnerCommitmentCandidateRequest {
            policy,
            fold_policy,
            ring_challenge_cfg,
            challenge_dimension,
            dimensions,
            num_claims,
            num_live_ring_elements_per_claim,
            num_live_blocks,
            num_positions_per_block,
            num_chunks,
            witness_norms,
            log_basis_open,
            width_s,
            modeled_linf_cap,
        })?
    else {
        return Ok(None);
    };
    let num_digits_fold = inner_candidate.num_digits_fold;
    let inner_commit_matrix = inner_candidate.inner_commit_matrix;
    let Some(outer_commit_matrix) =
        derive_outer_commitment_candidate(OuterCommitmentCandidateRequest {
            policy,
            dimensions,
            payload_mode,
            num_claims,
            num_live_blocks,
            outer_slice_count,
            log_basis_open,
            num_digits_outer,
            inner_output_rank: inner_commit_matrix.output_rank(),
        })?
    else {
        return Ok(None);
    };
    Ok(Some(AbCommitmentCandidate {
        num_digits_fold,
        inner_commit_matrix,
        outer_commit_matrix,
    }))
}

mod opening;
mod recursive;
mod setup_prefix;

pub(crate) use opening::PlannerOpeningCandidate;
pub(crate) use recursive::{
    derive_fold_candidates, derive_recursive_candidate_views, derive_terminal_candidates,
    recursive_split_search_domain, FoldCandidatePolicy, RecursiveCandidateRequest,
    RecursiveFoldWork, SplitBoundPolicy,
};
#[cfg(all(test, feature = "catalog-gen"))]
pub(crate) use recursive::{
    derive_unpruned_fold_candidates_for_oracle, derive_unpruned_terminal_candidates_for_oracle,
};
pub(crate) use setup_prefix::SetupPrefixSearchCache;
pub(super) use setup_prefix::{derive_setup_prefix_groups, SetupPrefixSearchRequest};

#[cfg(test)]
#[path = "../test/schedule_params_candidate.rs"]
mod tests;
