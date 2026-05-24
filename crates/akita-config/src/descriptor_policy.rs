//! Config-owned instance-descriptor binding policy.

use crate::CommitmentConfig;
use akita_field::AkitaError;
use akita_transcript::Transcript;
use akita_types::{
    scheduled_next_level_params, AkitaInstanceDescriptor, AkitaScheduleInputs, AkitaSetupSeed,
    AlgebraSection, BasisMode, CallSection, ClaimIncidenceSummary, LevelParams, PlanSection,
    Schedule, SetupSection, Step,
};

/// Derive the setup-level list that the instance descriptor binds for a
/// config-selected schedule.
///
/// The descriptor needs the same level parameters the verifier will use, even
/// when the effective schedule contains direct steps. Keeping this next to
/// [`CommitmentConfig`] avoids duplicating config policy in scheme wrappers,
/// recursion guests, or tests.
///
/// # Errors
///
/// Returns an error if the schedule shape is inconsistent with the config
/// policy or if a direct step appears without enough preceding fold context.
pub fn descriptor_setup_levels_for_config<Cfg: CommitmentConfig>(
    incidence: &ClaimIncidenceSummary,
    schedule: &Schedule,
) -> Result<Vec<LevelParams>, AkitaError> {
    let mut setup_levels = Vec::new();
    let mut previous_next_w_len = None;
    for (level, step) in schedule.steps.iter().enumerate() {
        match step {
            Step::Fold(fold) => {
                setup_levels.push(fold.params.clone());
                previous_next_w_len = Some(fold.next_w_len);
            }
            Step::Direct(_) => {
                if level == 0 {
                    setup_levels.push(Cfg::get_params_for_batched_commitment(incidence)?);
                } else {
                    let current_w_len = previous_next_w_len.ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "direct schedule step has no preceding fold".to_string(),
                        )
                    })?;
                    setup_levels.push(scheduled_next_level_params(
                        schedule,
                        level,
                        AkitaScheduleInputs {
                            num_vars: incidence.num_vars(),
                            level,
                            current_w_len,
                        },
                        Cfg::level_params_with_log_basis,
                    )?);
                }
            }
        }
    }
    if setup_levels.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "descriptor cannot bind an empty schedule".to_string(),
        ));
    }
    Ok(setup_levels)
}

/// Build and bind the canonical instance descriptor for a config-selected
/// proof replay.
///
/// # Errors
///
/// Returns an error if descriptor construction or canonical serialization
/// fails.
pub fn bind_instance_descriptor_for_config<Cfg, T, const D: usize>(
    transcript: &mut T,
    setup_seed: &AkitaSetupSeed,
    incidence: &ClaimIncidenceSummary,
    schedule: &Schedule,
    basis: BasisMode,
) -> Result<(), AkitaError>
where
    Cfg: CommitmentConfig,
    T: Transcript<Cfg::Field>,
{
    let setup_levels = descriptor_setup_levels_for_config::<Cfg>(incidence, schedule)?;
    let descriptor = AkitaInstanceDescriptor::new(
        AlgebraSection::for_fields::<Cfg::Field, Cfg::ClaimField, Cfg::ChallengeField, D>()?,
        SetupSection::from_parts(
            Cfg::decomposition(),
            Cfg::sis_modulus_family(),
            setup_seed,
            &setup_levels,
        )
        .map_err(|err| AkitaError::InvalidSetup(format!("descriptor setup identity: {err}")))?,
        PlanSection::from_schedule(schedule),
        CallSection::from_incidence(incidence, basis)?,
    );
    let descriptor_bytes = descriptor
        .canonical_bytes()
        .map_err(|err| AkitaError::InvalidSetup(format!("descriptor serialization: {err}")))?;
    transcript.bind_instance_bytes(&descriptor_bytes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof_optimized::fp128;

    #[test]
    fn root_direct_descriptor_level_matches_commitment_policy() {
        type Cfg = fp128::D32Full;

        let incidence =
            ClaimIncidenceSummary::same_point(4, 1).expect("valid root-direct incidence");
        let schedule = Cfg::get_params_for_prove(&incidence).expect("prove schedule");
        assert!(
            matches!(schedule.steps.first(), Some(Step::Direct(_))),
            "test shape should exercise root-direct descriptor policy"
        );

        let descriptor_levels =
            descriptor_setup_levels_for_config::<Cfg>(&incidence, &schedule).unwrap();
        let verifier_root_params =
            Cfg::get_params_for_batched_commitment(&incidence).expect("root params");

        assert_eq!(descriptor_levels, vec![verifier_root_params]);
    }
}
