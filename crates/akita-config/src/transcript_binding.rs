//! Canonical transcript descriptor binding shared by prover and verifier.
//!
//! Both `akita_prover::prove_batched` and `akita_verifier::verify_batched`
//! bind the same canonical [`AkitaInstanceDescriptor`] bytes into the
//! Fiat-Shamir transcript before protocol replay. The function lives here
//! (rather than in `akita-prover` or `akita-verifier`) so both sides reach
//! it without crossing through `akita-scheme`, and so the descriptor
//! construction is sourced from a single `Cfg`-driven implementation.

use crate::proof_optimized::setup_level_params_from_runtime_schedule;
use crate::CommitmentConfig;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_transcript::Transcript;
use akita_types::{
    AkitaExpandedSetup, AkitaInstanceDescriptor, AlgebraSection, BasisMode, CallSection,
    ClaimIncidenceSummary, PlanSection, RingSubfieldEncoding, Schedule, SetupSection,
};

/// Bind the canonical [`AkitaInstanceDescriptor`] bytes into a transcript.
///
/// Both `prove_batched` (prover) and `verify_batched` (verifier) call this
/// helper after schedule selection and before protocol replay. The function
/// is `Cfg`-driven (algebra section, decomposition, SIS family, root-fold
/// params for empty-fold schedules), so both sides produce byte-identical
/// descriptor bytes for the same inputs and the transcript-determinism
/// invariant holds.
///
/// # Errors
///
/// Returns an error when:
/// - the schedule contains no folded levels and
///   `Cfg::get_params_for_batched_commitment` fails for the supplied
///   incidence,
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
