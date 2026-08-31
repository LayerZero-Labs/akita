//! Canonical transcript descriptor binding shared by prover and verifier.
//!
//! Both `akita_prover::batched_prove` and `akita_verifier::batched_verify`
//! bind the same canonical [`AkitaInstanceDescriptor`] bytes into the
//! Fiat-Shamir transcript before protocol replay. The function lives here
//! (rather than in `akita-prover` or `akita-verifier`) so both sides reach
//! it without crossing through `akita-pcs`, and so the descriptor
//! construction is sourced from a single `Cfg`-driven implementation.

use crate::{derive_transcript_grinding_plan, CommitmentConfig};
use akita_error::AkitaError;
use akita_transcript::Transcript;
use akita_types::{
    AkitaExpandedSetup, AkitaInstanceDescriptor, AlgebraSection, BasisMode, CallSection,
    FoldSchedule, FpExtEncoding, GrindingPlan, OpeningClaimsLayout, OpeningScheduleSelection,
    PlanSection, SetupSection, TranscriptGrindingBinding,
};
use jolt_field::{CanonicalEncoding, Field};

/// Bind the canonical [`AkitaInstanceDescriptor`] bytes into a transcript.
///
/// Both `batched_prove` (prover) and `batched_verify` (verifier) call this
/// helper after schedule selection and before protocol replay. The function
/// is `Cfg`-driven (algebra section, decomposition, SIS family), so both
/// sides produce byte-identical descriptor bytes for the same inputs and the
/// transcript-determinism invariant holds.
///
/// The per-proof effective `schedule` is digested into `PlanSection` and
/// binds every expanded fold `CommittedGroupParams`, so there is no separate
/// setup-level digest to compute here.
///
/// # Errors
///
/// Returns an error when:
/// - the algebra section cannot be derived for the field tower, or
/// - canonical descriptor serialization fails.
pub fn bind_transcript_instance_descriptor<F, T, Cfg>(
    setup: &AkitaExpandedSetup<F>,
    opening_batch: &OpeningClaimsLayout,
    selection: OpeningScheduleSelection,
    schedule: &FoldSchedule,
    basis: BasisMode,
    transcript: &mut T,
) -> Result<GrindingPlan, AkitaError>
where
    F: Field + CanonicalEncoding,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
    Cfg::ExtField: FpExtEncoding<F>,
{
    let grinding_plan = derive_transcript_grinding_plan::<Cfg>(schedule, opening_batch)?;
    let descriptor = AkitaInstanceDescriptor::new(
        AlgebraSection::for_fields::<F, Cfg::ExtField>()?,
        SetupSection::from_parts(
            Cfg::decomposition(),
            Cfg::sis_modulus_profile(),
            &setup.seed().setup_seed,
        )
        .map_err(|err| AkitaError::InvalidSetup(format!("descriptor setup identity: {err}")))?,
        PlanSection::from_schedule(selection, schedule),
        TranscriptGrindingBinding::for_plan(&grinding_plan)?,
        CallSection::from_layout(opening_batch, basis)?,
    );
    let descriptor_bytes = descriptor
        .canonical_bytes()
        .map_err(|err| AkitaError::InvalidSetup(format!("descriptor serialization: {err}")))?;
    transcript.bind_instance_bytes(&descriptor_bytes);
    Ok(grinding_plan)
}
