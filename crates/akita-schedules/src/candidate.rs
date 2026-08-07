//! Shared candidate-construction helpers.

use crate::runtime::PlannerPolicy;
use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;
use akita_types::sis::{
    min_secure_l2_rank, projected_role_ring_count, role_a_collision_l2_sq_for_response_bound,
    rounded_up_collision_inf_norm, sis_l2_table_key_for_collision_sq, FoldChallengeNorms,
    SisTableKey,
};
use akita_types::{InnerCommitMatrixParams, PhysicalL2NormProofShape, SisMatrixRole};

/// Exact public geometry that may admit one selective physical-L2 A matrix.
#[derive(Clone, Copy, Debug)]
pub struct SelectiveL2CandidateGeometry<'a> {
    pub fold_level: usize,
    pub input_witness_len: usize,
    pub num_claims: usize,
    pub num_chunks: usize,
    pub inner_width: usize,
    pub ring_dimension: usize,
    pub fold_basis: usize,
    pub fold_digit_count: usize,
    pub fold_challenge_config: &'a SparseChallengeConfig,
}

/// Derive the one canonical L2 A-matrix candidate for an exact fold geometry.
///
/// `Ok(None)` means the route is ineligible, has no measured cap, or has no
/// generated secure table row. Once a cap is present, malformed arithmetic or
/// proof geometry is a policy error rather than a silently different route.
pub fn selective_l2_inner_matrix(
    policy: &PlannerPolicy,
    geometry: SelectiveL2CandidateGeometry<'_>,
) -> Result<Option<InnerCommitMatrixParams>, AkitaError> {
    if geometry.fold_level < 3 || geometry.num_claims != 1 || geometry.num_chunks != 1 {
        return Ok(None);
    }
    let physical_response_len = geometry
        .inner_width
        .checked_mul(geometry.ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup("L2 physical response length overflow".into()))?;
    let Some(response_l2_sq_cap) = policy.selective_l2_cap_for_candidate(
        geometry.fold_level,
        geometry.input_witness_len,
        physical_response_len,
        geometry.fold_basis,
        geometry.fold_digit_count,
    ) else {
        return Ok(None);
    };
    let norm_proof_shape = PhysicalL2NormProofShape::derive(
        policy.sis_modulus_profile,
        physical_response_len,
        geometry.fold_basis,
        geometry.fold_digit_count,
    )?;
    let challenge_l1 = FoldChallengeNorms::new(geometry.fold_challenge_config).l1_norm;
    let collision_l2_sq =
        role_a_collision_l2_sq_for_response_bound(challenge_l1, response_l2_sq_cap)
            .ok_or_else(|| AkitaError::InvalidSetup("L2 collision bound overflow".into()))?;
    let Some(table_key) = sis_l2_table_key_for_collision_sq(
        policy.sis_security_policy,
        policy.sis_l2_table_digest,
        policy.sis_modulus_profile,
        geometry.ring_dimension as u32,
        collision_l2_sq,
    ) else {
        return Ok(None);
    };
    let width = u64::try_from(geometry.inner_width)
        .map_err(|_| AkitaError::InvalidSetup("L2 A matrix input width exceeds u64".into()))?;
    if min_secure_l2_rank(table_key, width).is_none() {
        return Ok(None);
    }
    InnerCommitMatrixParams::try_new_l2_with_min_rank(
        table_key,
        geometry.inner_width,
        response_l2_sq_cap,
        norm_proof_shape,
    )
    .map(Some)
}

/// Construct the canonical SIS-table key for one role and ring dimension.
pub fn sis_key_at_dimension(
    policy: &PlannerPolicy,
    role: SisMatrixRole,
    ring_dimension: usize,
    coeff_linf_bound: u128,
) -> SisTableKey {
    SisTableKey {
        policy: policy.sis_security_policy,
        table_digest: policy.sis_table_digest,
        modulus_profile: policy.sis_modulus_profile,
        role,
        ring_dimension: ring_dimension as u32,
        coeff_linf_bound,
    }
}

/// Price one projected B/D collision role using canonical physical width and
/// coefficient bounds.
pub fn projected_collision_role_price(
    policy: &PlannerPolicy,
    role: SisMatrixRole,
    carrier_dimension: usize,
    role_dimension: usize,
    native_width: usize,
    log_basis: u32,
) -> Option<(SisTableKey, usize)> {
    if role == SisMatrixRole::Inner
        || role_dimension == 0
        || !carrier_dimension.is_multiple_of(role_dimension)
    {
        return None;
    }
    let coeff_linf_bound = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        role,
        role_dimension,
        log_basis,
    )?;
    let physical_width =
        projected_role_ring_count(carrier_dimension, role_dimension, native_width)?;
    Some((
        sis_key_at_dimension(policy, role, role_dimension, coeff_linf_bound),
        physical_width,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PlannerCostModelId, SelectionPolicyId, SelectiveL2FoldCap};
    use akita_types::{
        ChunkedWitnessCfg, CommitmentRingDims, DecompositionParams, SisL2TableDigest,
        SisModulusProfileId, SisSecurityPolicyId, SisTableDigest,
    };

    #[test]
    fn missing_l2_rank_is_an_ineligible_candidate() {
        const INNER_WIDTH: usize = 6_400_000_000_001;
        const RING_DIMENSION: usize = 64;
        static DIMENSIONS: [CommitmentRingDims; 1] = [CommitmentRingDims::uniform(RING_DIMENSION)];
        static CAPS: [SelectiveL2FoldCap; 1] = [SelectiveL2FoldCap {
            fold_level: 3,
            input_witness_len: 7,
            physical_response_len: INNER_WIDTH * RING_DIMENSION,
            fold_basis: 2,
            fold_digit_count: 1,
            response_l2_sq_cap: 1,
        }];
        let policy = PlannerPolicy {
            cost_model: PlannerCostModelId::ExactPayloadAndSetupEnvelope,
            selection_policy: SelectionPolicyId::MinEstimatedProofPayload,
            setup_field_budget: None,
            min_offloaded_witness_contraction: 1,
            uniform_ring_dimension: RING_DIMENSION,
            setup_prefix_inner_ring_dimension: RING_DIMENSION,
            ring_dimension_candidates: &DIMENSIONS,
            decomposition: DecompositionParams {
                log_basis: 1,
                log_commit_bound: 1,
                log_open_bound: Some(1),
            },
            sis_modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
            sis_security_policy: SisSecurityPolicyId::Quantum128BitADPS16,
            sis_table_digest: SisTableDigest::CURRENT,
            sis_l2_table_digest: SisL2TableDigest::CURRENT,
            selective_l2_fold_caps: &CAPS,
            claim_ext_degree: 1,
            chal_ext_degree: 1,
            basis_range: (1, 1),
            witness_chunk: ChunkedWitnessCfg::default(),
            recursive_setup_planning: false,
        };
        let challenge = SparseChallengeConfig::pm1_only(3);

        let candidate = selective_l2_inner_matrix(
            &policy,
            SelectiveL2CandidateGeometry {
                fold_level: 3,
                input_witness_len: 7,
                num_claims: 1,
                num_chunks: 1,
                inner_width: INNER_WIDTH,
                ring_dimension: RING_DIMENSION,
                fold_basis: 2,
                fold_digit_count: 1,
                fold_challenge_config: &challenge,
            },
        )
        .expect("unsupported L2 rank must not abort the L-infinity frontier");

        assert!(candidate.is_none());
    }
}
