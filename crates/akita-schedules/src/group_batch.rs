//! Runtime helpers for materializing cataloged multi-group root precommits.

use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_types::{AkitaScheduleLookupKey, GroupOpenPhaseParams, PrecommittedGroupAdmissionPolicy};

use crate::generated::GeneratedPrecommittedGroup;
use crate::PlannerPolicy;

pub(crate) fn multi_group_root_precommitted_groups_for_open_basis(
    key: &AkitaScheduleLookupKey,
    generated_groups: &[GeneratedPrecommittedGroup],
    policy: &PlannerPolicy,
    ring_challenge_config: &dyn Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    log_basis_open: u32,
) -> Result<Vec<GroupOpenPhaseParams>, AkitaError> {
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
        num_response_chunks: policy.chunks_at_level(0),
    };
    let groups = key
        .precommitteds
        .iter()
        .zip(generated_groups)
        .map(|(layout, generated)| {
            let generated = generated.group;
            let challenge_dimension = match generated.opening_method {
                akita_types::OpeningMethod::EvaluationTrace => layout.inner.matrix.ring_dimension(),
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
            let params = GroupOpenPhaseParams::admit(
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
    Ok(groups)
}
