//! Canonical transcript descriptor binding shared by prover and verifier.
//!
//! Both `akita_prover::prove_batched` and `akita_verifier::verify_batched`
//! bind the same canonical [`AkitaInstanceDescriptor`] bytes into the
//! Fiat-Shamir transcript before protocol replay. The function lives here
//! (rather than in `akita-prover` or `akita-verifier`) so both sides reach
//! it without crossing through `akita-pcs`, and so the descriptor
//! construction is sourced from a single `Cfg`-driven implementation.

use crate::CommitmentConfig;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_transcript::Transcript;
use akita_types::{
    AkitaExpandedSetup, AkitaInstanceDescriptor, AlgebraSection, BasisMode, CallSection,
    ClaimIncidenceSummary, PlanSection, RingSubfieldEncoding, Schedule, SetupSection, Step,
    TerminalProofMode,
};

/// Bind the canonical [`AkitaInstanceDescriptor`] bytes into a transcript.
///
/// Both `prove_batched` (prover) and `verify_batched` (verifier) call this
/// helper after schedule selection and before protocol replay. The function
/// is `Cfg`-driven (algebra section, decomposition, SIS family), so both
/// sides produce byte-identical descriptor bytes for the same inputs and the
/// transcript-determinism invariant holds.
///
/// The per-proof effective `schedule` is digested into `PlanSection` and
/// binds every expanded `LevelParams` — including the root-direct commit
/// layout — so there is no separate setup-level digest to compute here.
///
/// # Errors
///
/// Returns an error when:
/// - the algebra section cannot be derived for the field tower, or
/// - canonical descriptor serialization fails.
pub fn bind_transcript_instance_descriptor<F, T, const D: usize, Cfg>(
    setup: &AkitaExpandedSetup<F>,
    incidence: &ClaimIncidenceSummary,
    schedule: &Schedule,
    basis: BasisMode,
    transcript: &mut T,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
    Cfg::ClaimField: RingSubfieldEncoding<F>,
    Cfg::ChallengeField: RingSubfieldEncoding<F>,
{
    let terminal_proof_mode = Cfg::terminal_proof_mode();
    validate_terminal_proof_mode_selectable(terminal_proof_mode)?;
    validate_schedule_terminal_proof_mode(schedule, terminal_proof_mode)?;

    let mut setup_levels = setup_level_params_from_runtime_schedule(&schedule.steps);
    if setup_levels.is_empty() {
        // Defensive fallback: empty schedules and root-direct edge entries
        // with no `commit_params` go through the same `Cfg`-driven path
        // setup uses to size the shared matrix.
        setup_levels.push(Cfg::get_params_for_batched_commitment(incidence)?);
    }

    let descriptor = AkitaInstanceDescriptor::new(
        AlgebraSection::for_fields::<F, Cfg::ClaimField, Cfg::ChallengeField, D>()?,
        SetupSection::from_parts(
            Cfg::decomposition(),
            Cfg::sis_modulus_family(),
            setup.seed(),
            &setup_levels,
            terminal_proof_mode,
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

fn validate_terminal_proof_mode_selectable(mode: TerminalProofMode) -> Result<(), AkitaError> {
    match mode {
        TerminalProofMode::RingSwitchSumcheck => Ok(()),
        TerminalProofMode::DirectRingRelations => {
            #[cfg(feature = "zk")]
            {
                Err(AkitaError::InvalidSetup(
                    "direct terminal proof mode is not supported with zk".to_string(),
                ))
            }
            #[cfg(not(feature = "zk"))]
            {
                Ok(())
            }
        }
    }
}

fn validate_schedule_terminal_proof_mode(
    schedule: &Schedule,
    mode: TerminalProofMode,
) -> Result<(), AkitaError> {
    if !schedule
        .steps
        .iter()
        .any(|step| matches!(step, Step::Fold(_)))
    {
        return Ok(());
    }

    let direct = match schedule.steps.last() {
        Some(Step::Direct(direct)) => direct,
        _ => {
            return Err(AkitaError::InvalidSetup(
                "folded schedule must terminate in a direct step".to_string(),
            ));
        }
    };
    if direct.terminal_proof_mode != mode {
        return Err(AkitaError::InvalidSetup(format!(
            "terminal proof mode mismatch between config and schedule: config={mode:?}, schedule={:?}",
            direct.terminal_proof_mode
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(feature = "zk"))]
    fn direct_terminal_mode_is_selectable_in_transparent_builds() {
        validate_terminal_proof_mode_selectable(TerminalProofMode::DirectRingRelations)
            .expect("direct mode is selectable once transparent prover/verifier routing is wired");
    }

    #[test]
    #[cfg(feature = "zk")]
    fn direct_terminal_mode_rejects_in_zk_builds() {
        let err = validate_terminal_proof_mode_selectable(TerminalProofMode::DirectRingRelations)
            .expect_err("direct mode has no zk masking contract");
        assert!(err
            .to_string()
            .contains("direct terminal proof mode is not supported with zk"));
    }

    #[test]
    fn ring_switch_terminal_mode_is_selectable() {
        validate_terminal_proof_mode_selectable(TerminalProofMode::RingSwitchSumcheck)
            .expect("existing terminal mode remains selectable");
    }
}
