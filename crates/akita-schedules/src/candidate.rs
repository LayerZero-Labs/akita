//! Shared candidate-construction helpers.

use crate::runtime::PlannerPolicy;
use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
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
    pub num_claims: usize,
    pub num_chunks: usize,
    pub inner_width: usize,
    pub ring_dimension: usize,
    pub fold_basis: usize,
    pub fold_digit_count: usize,
    pub fold_challenge_config: &'a SparseChallengeConfig,
    /// Planner-frozen response cap. Artifact validation and resolved
    /// schedule audit replay this exact public value instead of rerunning an
    /// offline response model.
    pub response_l2_sq_cap: Option<u128>,
    /// Override the recursive proof shape when the response is checked in the
    /// clear, as it is at the terminal boundary.
    pub norm_proof_shape: Option<PhysicalL2NormProofShape>,
}

/// Derive the one canonical L2 A-matrix candidate for an exact fold geometry.
pub fn selective_l2_inner_matrix(
    policy: &PlannerPolicy,
    geometry: SelectiveL2CandidateGeometry<'_>,
) -> Result<Option<InnerCommitMatrixParams>, AkitaError> {
    // Basis 8 and above use the same L2 collision and norm-proof contract.
    // Smaller response bases are outside the generated schedule domain.
    if geometry.fold_level < 3
        || geometry.num_claims != 1
        || geometry.num_chunks != 1
        || geometry.fold_basis < 8
    {
        return Ok(None);
    }
    let physical_response_len = geometry
        .inner_width
        .checked_mul(geometry.ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup("L2 physical response length overflow".into()))?;
    let Some(response_l2_sq_cap) = geometry.response_l2_sq_cap else {
        return Ok(None);
    };
    let norm_proof_shape = match geometry.norm_proof_shape {
        Some(shape) => shape,
        None => PhysicalL2NormProofShape::derive(
            policy.sis_modulus_profile,
            physical_response_len,
            geometry.fold_basis,
            geometry.fold_digit_count,
        )?,
    };
    let challenge_mass = akita_challenges::selective_l2_operator_norm_rejection(
        geometry.ring_dimension,
        geometry.fold_challenge_config,
    )
    .map_or_else(
        || FoldChallengeNorms::new(geometry.fold_challenge_config).l1_norm,
        |rejection| u128::from(rejection.threshold),
    );
    let collision_l2_sq =
        role_a_collision_l2_sq_for_response_bound(challenge_mass, response_l2_sq_cap)
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
    source_dimension: usize,
    role_dimension: usize,
    native_width: usize,
    log_basis: u32,
) -> Option<(SisTableKey, usize)> {
    if role == SisMatrixRole::Inner
        || role_dimension == 0
        || !source_dimension.is_multiple_of(role_dimension)
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
    let physical_width = projected_role_ring_count(source_dimension, role_dimension, native_width)?;
    Some((
        sis_key_at_dimension(policy, role, role_dimension, coeff_linf_bound),
        physical_width,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PlannerCostModelId, SelectionPolicyId};
    use akita_types::{
        ChunkedWitnessCfg, DecompositionParams, SisL2TableDigest, SisModulusProfileId,
        SisSecurityPolicyId, SisTableDigest,
    };

    #[test]
    fn missing_l2_rank_is_an_ineligible_candidate() {
        const INNER_WIDTH: usize = 6_400_000_000_001;
        const RING_DIMENSION: usize = 64;
        let policy = PlannerPolicy {
            cost_model: PlannerCostModelId::ExactPayloadAndSetupEnvelope,
            selective_l2_response_model: crate::SelectiveL2ResponseModelId::Disabled,
            selection_policy: SelectionPolicyId::MinEstimatedProofPayloadV2,
            recursive_split_search_policy: crate::RecursiveSplitSearchPolicy::Exhaustive,
            recursive_setup_search_policy: crate::RecursiveSetupSearchPolicy::Exhaustive,
            setup_field_budget: None,
            min_offloaded_witness_contraction: 1,
            ring_dimension_schedule_mode: crate::RingDimensionScheduleMode::UniformDimension {
                ring_dimension: RING_DIMENSION,
            },
            decomposition: DecompositionParams {
                log_basis: 1,
                log_commit_bound: 1,
                log_open_bound: Some(1),
            },
            sis_modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
            sis_security_policy: SisSecurityPolicyId::Quantum128BitADPS16,
            sis_table_digest: SisTableDigest::CURRENT,
            sis_l2_table_digest: SisL2TableDigest::CURRENT,
            claim_ext_degree: 1,
            chal_ext_degree: 1,
            inner_basis_range: (1, 1),
            opening_basis_range: (1, 1),
            witness_chunk: ChunkedWitnessCfg::default(),
            recursive_setup_planning: false,
        };
        let challenge = akita_challenges::D64_SELECTIVE_L2_CHALLENGE_CONFIG;

        let candidate = selective_l2_inner_matrix(
            &policy,
            SelectiveL2CandidateGeometry {
                fold_level: 3,
                num_claims: 1,
                num_chunks: 1,
                inner_width: INNER_WIDTH,
                ring_dimension: RING_DIMENSION,
                fold_basis: 16,
                fold_digit_count: 3,
                fold_challenge_config: &challenge,
                response_l2_sq_cap: Some(1),
                norm_proof_shape: None,
            },
        )
        .expect("unsupported L2 rank must not abort the L-infinity frontier");

        assert!(candidate.is_none());
    }

    #[test]
    fn d128_l2_candidate_prices_the_certified_operator_threshold() {
        const INNER_WIDTH: usize = 128;
        const RING_DIMENSION: usize = 128;
        const RESPONSE_CAP: u128 = 1 << 20;
        let policy = PlannerPolicy {
            cost_model: PlannerCostModelId::ExactPayloadAndSetupEnvelope,
            selective_l2_response_model: crate::SelectiveL2ResponseModelId::Disabled,
            selection_policy: SelectionPolicyId::MinEstimatedProofPayloadV2,
            recursive_split_search_policy: crate::RecursiveSplitSearchPolicy::Exhaustive,
            recursive_setup_search_policy: crate::RecursiveSetupSearchPolicy::Exhaustive,
            setup_field_budget: None,
            min_offloaded_witness_contraction: 1,
            ring_dimension_schedule_mode: crate::RingDimensionScheduleMode::UniformDimension {
                ring_dimension: RING_DIMENSION,
            },
            decomposition: DecompositionParams {
                log_basis: 4,
                log_commit_bound: 128,
                log_open_bound: None,
            },
            sis_modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
            sis_security_policy: SisSecurityPolicyId::Quantum128BitADPS16,
            sis_table_digest: SisTableDigest::CURRENT,
            sis_l2_table_digest: SisL2TableDigest::CURRENT,
            claim_ext_degree: 1,
            chal_ext_degree: 1,
            inner_basis_range: (4, 4),
            opening_basis_range: (4, 4),
            witness_chunk: ChunkedWitnessCfg::default(),
            recursive_setup_planning: false,
        };
        let challenge = akita_challenges::D128_SELECTIVE_L2_CHALLENGE_CONFIG;
        let basis_eight = selective_l2_inner_matrix(
            &policy,
            SelectiveL2CandidateGeometry {
                fold_level: 3,
                num_claims: 1,
                num_chunks: 1,
                inner_width: INNER_WIDTH,
                ring_dimension: RING_DIMENSION,
                fold_basis: 8,
                fold_digit_count: 4,
                fold_challenge_config: &challenge,
                response_l2_sq_cap: Some(RESPONSE_CAP),
                norm_proof_shape: None,
            },
        )
        .expect("basis-eight eligibility check")
        .expect("basis-eight L2 table coverage");
        assert!(matches!(
            basis_eight.security_route(),
            akita_types::InnerCommitSecurityRoute::L2 { .. }
        ));
        let candidate = selective_l2_inner_matrix(
            &policy,
            SelectiveL2CandidateGeometry {
                fold_level: 3,
                num_claims: 1,
                num_chunks: 1,
                inner_width: INNER_WIDTH,
                ring_dimension: RING_DIMENSION,
                fold_basis: 16,
                fold_digit_count: 3,
                fold_challenge_config: &challenge,
                response_l2_sq_cap: Some(RESPONSE_CAP),
                norm_proof_shape: None,
            },
        )
        .expect("D128 L2 candidate")
        .expect("D128 L2 table coverage");
        let akita_types::InnerCommitSecurityRoute::L2 { table_key, .. } =
            candidate.security_route()
        else {
            panic!("expected L2 route")
        };
        let exact_collision = role_a_collision_l2_sq_for_response_bound(
            u128::from(akita_challenges::OperatorNormRejection::D128_SELECTIVE_L2.threshold),
            RESPONSE_CAP,
        )
        .expect("collision bound");
        assert_eq!(
            table_key.collision_l2_sq,
            exact_collision.next_power_of_two()
        );

        let l1_collision = role_a_collision_l2_sq_for_response_bound(
            FoldChallengeNorms::new(&challenge).l1_norm,
            RESPONSE_CAP,
        )
        .expect("L1 collision bound");
        assert!(table_key.collision_l2_sq < l1_collision.next_power_of_two());
    }
}
