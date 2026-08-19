//! Runtime helpers for materializing cataloged multi-group root precommits.

use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;
use akita_types::{
    AkitaScheduleLookupKey, PrecommittedGroupAdmissionPolicy, PrecommittedLevelParams,
};

use crate::generated::GeneratedRootPrecommittedGroup;
use crate::PlannerPolicy;

pub(crate) fn multi_group_root_precommitted_groups_for_open_basis(
    key: &AkitaScheduleLookupKey,
    generated_groups: &[GeneratedRootPrecommittedGroup],
    policy: &PlannerPolicy,
    ring_challenge_config: &dyn Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    log_basis_open: u32,
    open_ring_dimension: usize,
) -> Result<(Vec<PrecommittedLevelParams>, usize), AkitaError> {
    if key.precommitteds.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "multi-group root params require at least one precommitted group".to_string(),
        ));
    }
    if key.precommitteds.len() != generated_groups.len() {
        return Err(AkitaError::InvalidSetup(
            "generated precommitted group count does not match the schedule key".to_string(),
        ));
    }
    let admission_policy = PrecommittedGroupAdmissionPolicy {
        decomposition: policy.decomposition,
        sis_security_policy: policy.sis_security_policy,
        sis_table_digest: policy.sis_table_digest,
        sis_modulus_profile: policy.sis_modulus_profile,
    };
    let groups = key
        .precommitteds
        .iter()
        .zip(generated_groups)
        .map(|(layout, generated)| {
            let challenge_dimension = match generated.opening_method {
                akita_types::OpeningMethod::EvaluationTrace => {
                    layout.inner_commit_matrix.ring_dimension()
                }
                akita_types::OpeningMethod::SubringCoefficientPacking {
                    challenge_subring_dimension,
                } => challenge_subring_dimension,
            };
            let ring_challenge_cfg = ring_challenge_config(challenge_dimension)?;
            let num_digits_fold = usize::try_from(generated.num_digits_fold).map_err(|_| {
                AkitaError::InvalidSetup(
                    "generated precommitted fold depth does not fit the target platform"
                        .to_string(),
                )
            })?;
            let params = PrecommittedLevelParams::admit(
                *layout,
                num_digits_fold,
                admission_policy,
                generated.opening_method,
                ring_challenge_cfg,
                log_basis_open,
            )?;
            params.validate()?;
            Ok(params)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut d_width = 0usize;
    for group in &groups {
        d_width = d_width
            .checked_add(group.d_segment_width(policy.claim_ext_degree, open_ring_dimension)?)
            .ok_or_else(|| AkitaError::InvalidSetup("multi-group D width overflow".to_string()))?;
    }
    Ok((groups, d_width))
}
