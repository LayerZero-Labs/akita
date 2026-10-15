//! Proving on a caller-owned transcript, for examples and tests that read the
//! transcript's event log or embed Akita after a caller prefix.

use akita_config::CommitmentConfig;
use akita_cpu_backend::{AkitaProverSetup, CommitmentHandle, CpuBackend};
use akita_error::AkitaError;
use akita_params::BasisMode;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{BackendRegistry, FixedFoldRoute, ProverBackend, SelectedProverOpeningData};
use akita_serialization::{AkitaSerialize, Valid};
use akita_types::FpExtEncoding;
use jolt_field::{
    AdditiveGroup, CanonicalEncoding, ExtField, Fold, MulBaseUnreduced, PseudoMersenne, Ring,
    Unreduced, WithCommitAccumulator,
};
use jolt_transcript::{ProverTranscript, Sponge};

/// Prove `opening` on `transcript` with every fold on `backend`: import the
/// schedule's setup prefixes, register the backend, and run the core prover.
/// The caller finishes the transcript.
#[allow(clippy::type_complexity)]
pub(crate) fn prove_on_transcript<'a, Cfg, H>(
    scheme: &AkitaCommitmentScheme<Cfg>,
    setup: &AkitaProverSetup<Cfg::Field>,
    opening: SelectedProverOpeningData<
        'a,
        Cfg::ExtField,
        CommitmentHandle<Cfg::Field, Cfg::ExtField>,
        Cfg::Field,
    >,
    backend: &CpuBackend<Cfg::Field, Cfg::ExtField>,
    transcript: &mut ProverTranscript<H>,
    basis: BasisMode,
) -> Result<(), AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: CanonicalEncoding
        + AkitaSerialize
        + Valid
        + Unreduced
        + PseudoMersenne
        + Ring
        + WithCommitAccumulator
        + 'static,
    <Cfg::Field as Unreduced>::Wide: From<Cfg::Field> + AdditiveGroup,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + Unreduced
        + Fold
        + Ring
        + MulBaseUnreduced<Cfg::Field>
        + AkitaSerialize
        + 'static,
    CpuBackend<Cfg::Field, Cfg::ExtField>: ProverBackend<
        Cfg::Field,
        Cfg::ExtField,
        CommitmentHandle = CommitmentHandle<Cfg::Field, Cfg::ExtField>,
    >,
    H: Sponge,
{
    let resolved = scheme.schedules().resolve_selection(opening.selection())?;
    let required_prefix_ids = akita_config::required_setup_prefix_slot_ids_for_schedule(
        resolved.schedule(),
        opening.opening_layout(),
    )?;
    let prefix_slots = backend.import_setup_prefixes(&setup.prefix_slots, &required_prefix_ids)?;
    let mut registry = BackendRegistry::<Cfg, H>::new()?;
    let root = registry.register(backend, &prefix_slots)?;
    let mut route = FixedFoldRoute::new(vec![root; resolved.schedule().num_fold_levels()]);
    akita_prover::batched_prove::<Cfg, _, _>(
        setup.expanded.descriptor(),
        scheme.schedules(),
        &registry,
        opening,
        transcript,
        basis,
        &mut route,
    )
}
