//! Canonical transcript descriptor binding shared by prover and verifier.
//!
//! Both `akita_prover::batched_prove` and `AkitaVerifier::batched_verify`
//! bind the same canonical [`AkitaInstanceDescriptor`] bytes into the
//! Fiat-Shamir transcript before protocol replay. The function lives here
//! (rather than in `akita-prover` or `akita-verifier`) so both sides reach
//! it without crossing through `akita-pcs`, and so the descriptor
//! construction is sourced from a single `Cfg`-driven implementation.

use crate::transcript_grinding_plan::derive_zk_transcript_grinding_plan;
use crate::{derive_transcript_grinding_plan, CommitmentConfig};
use akita_error::AkitaError;
use akita_params::zk::ZkOpeningSchedule;
use akita_params::{
    BasisMode, FoldSchedule, GrindingPlan, OpeningClaimsLayout, OpeningScheduleSelection,
};
use akita_types::{
    AkitaInstanceDescriptor, AkitaSetupDescriptor, AlgebraSection, CallSection, FpExtEncoding,
    PlanSection, SetupSection, TranscriptGrindingBinding,
};
use jolt_algebra::{CanonicalEncoding, Field};

/// Construct the canonical [`AkitaInstanceDescriptor`] bytes and grinding plan.
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
/// With `zk`, the grinding plan is the zero-knowledge schedule's, and its
/// digest in the descriptor separates the two modes. `zk` must wrap
/// `schedule`.
///
/// # Errors
///
/// Returns an error when:
/// - the algebra section cannot be derived for the field tower,
/// - `zk` wraps a schedule other than `schedule`, or
/// - canonical descriptor serialization fails.
pub fn transcript_instance_descriptor<F, Cfg>(
    setup: &AkitaSetupDescriptor,
    opening_batch: &OpeningClaimsLayout,
    selection: OpeningScheduleSelection,
    schedule: &FoldSchedule,
    zk: Option<&ZkOpeningSchedule>,
    basis: BasisMode,
) -> Result<(GrindingPlan, Vec<u8>), AkitaError>
where
    F: Field + CanonicalEncoding,
    Cfg: CommitmentConfig<Field = F>,
    Cfg::ExtField: FpExtEncoding<F>,
{
    let grinding_plan = match zk {
        None => derive_transcript_grinding_plan::<Cfg>(schedule, opening_batch)?,
        Some(zk) if zk.schedule() == schedule => {
            derive_zk_transcript_grinding_plan::<Cfg>(zk, opening_batch)?
        }
        Some(_) => {
            return Err(AkitaError::InvalidInput(
                "zero-knowledge schedule differs from the selected schedule row".into(),
            ))
        }
    };
    let instance_descriptor = AkitaInstanceDescriptor::new(
        AlgebraSection::for_fields::<F, Cfg::ExtField>()?,
        SetupSection::from_parts(
            Cfg::decomposition(),
            Cfg::sis_modulus_profile(),
            &setup.setup_seed,
        )
        .map_err(|err| AkitaError::InvalidSetup(format!("descriptor setup identity: {err}")))?,
        PlanSection::from_schedule(selection, schedule),
        TranscriptGrindingBinding::for_plan(&grinding_plan)?,
        CallSection::from_layout(opening_batch, basis)?,
    );
    let descriptor_bytes = instance_descriptor
        .canonical_bytes()
        .map_err(|err| AkitaError::InvalidSetup(format!("descriptor serialization: {err}")))?;
    Ok((grinding_plan, descriptor_bytes))
}
