use super::*;
use crate::backend::{RecursiveFoldSource, RecursiveWitnessFlat};
use crate::compute::{
    prewarm_ntt_requirements, ComputeBackendSetup, DigitRowsComputeBackend, LevelProveStacks,
    NttExecutionRequirements, RuntimeCoefficientPackingBackendFor, RuntimeCommitBackendFor,
    RuntimeOpeningProveBackendFor, RuntimeRingSwitchProveBackend, RuntimeTensorBackendFor,
    SuffixOpeningProveBackend, SuffixTensorProveBackend,
};
use crate::SelectedProverOpeningData;
use akita_config::{ensure_prover_schedule_fits_setup, CommitmentConfig, TrustedScheduleCatalog};
use jolt_field::{AdditiveGroup, CanonicalEncoding};

/// Drive batched proving end-to-end under config `Cfg`.
///
/// Claim shape is guaranteed by `ProverOpeningData` construction. This entry
/// resolves the trusted catalog row, validates schedule execution and setup
/// capacity, binds the transcript, and proves the root and recursive suffix.
///
/// # Errors
///
/// Returns an error if claim preparation, schedule selection, transcript
/// binding, or folded proving fails.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn batched_prove<'a, Cfg, T, P, C, O, TS, R>(
    expanded: &Arc<AkitaExpandedSetup<Cfg::Field>>,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field>,
    schedules: &TrustedScheduleCatalog<Cfg>,
    stacks: &'a impl LevelProveStacks<
        'a,
        Cfg::Field,
        Commit = C,
        Opening = O,
        Tensor = TS,
        RingSwitch = R,
    >,
    opening: SelectedProverOpeningData<'a, Cfg::ExtField, P, Cfg::Field>,
    transcript: &mut T,
    basis: BasisMode,
) -> Result<AkitaBatchedProof<Cfg::Field, Cfg::ExtField>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: Field
        + CanonicalEncoding
        + akita_serialization::AkitaSerialize
        + Unreduced
        + Field
        + PseudoMersenne,
    Cfg::ExtField: FpExtEncoding<Cfg::Field> + MulBaseUnreduced<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + ExtField<Cfg::Field>
        + Unreduced
        + Fold
        + Ring
        + AkitaSerialize,
    T: Transcript<Cfg::Field> + TranscriptChallengePreview,
    Cfg::Field: Ring + 'static,
    <Cfg::Field as Unreduced>::Wide: From<Cfg::Field> + AdditiveGroup,
    P: PreparedGroupProveOps<Cfg::Field, Cfg::ExtField, O>,
    C: ComputeBackendSetup<Cfg::Field>
        + RuntimeCommitBackendFor<Cfg::Field, RecursiveWitnessFlat>
        + 'a,
    O: ComputeBackendSetup<Cfg::Field>
        + RuntimeOpeningProveBackendFor<Cfg::Field, RecursiveFoldSource<Cfg::Field>>
        + RuntimeCoefficientPackingBackendFor<
            Cfg::Field,
            RecursiveFoldSource<Cfg::Field>,
            Cfg::ExtField,
        > + SuffixOpeningProveBackend<Cfg::Field>
        + DigitRowsComputeBackend<Cfg::Field>
        + 'a,
    TS: ComputeBackendSetup<Cfg::Field>
        + RuntimeTensorBackendFor<Cfg::Field, RecursiveFoldSource<Cfg::Field>, Cfg::ExtField>
        + SuffixTensorProveBackend<Cfg::Field, Cfg::ExtField>
        + 'a,
    R: ComputeBackendSetup<Cfg::Field>
        + RuntimeRingSwitchProveBackend<Cfg::Field>
        + DigitRowsComputeBackend<Cfg::Field>
        + 'a,
    <C as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'a,
    <O as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'a,
    <TS as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'a,
    <R as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'a,
{
    let (selection, claims) = opening.into_low_level_parts();
    let opening_batch = claims.opening_layout();
    let resolved = schedules.resolve_selection(selection)?;
    resolved.validate_opening_layout(opening_batch)?;
    let schedule = resolved.schedule();
    schedule.validate_nonterminal_opening_execution(Cfg::EXT_DEGREE)?;
    ensure_prover_schedule_fits_setup::<Cfg>(expanded.as_ref(), schedule, opening_batch)?;
    let ntt_requirements = NttExecutionRequirements::from_prove_schedule(schedule)?;
    prewarm_ntt_requirements::<Cfg::Field, _>(stacks, &ntt_requirements)?;
    let grinding_plan = bind_transcript_instance_descriptor::<Cfg::Field, T, Cfg>(
        expanded.as_ref(),
        opening_batch,
        selection,
        schedule,
        basis,
        transcript,
    )?;

    let (next_params, next_binding) = schedule.recursive_folds.first().map_or(
        (
            super::fold::FoldSuccessorParams::Terminal(&schedule.terminal),
            akita_types::NextWitnessBindingPolicy::TerminalInnerState,
        ),
        |step| {
            (
                super::fold::FoldSuccessorParams::Recursive(step),
                akita_types::NextWitnessBindingPolicy::OuterPayload,
            )
        },
    );

    let mut grinding_transcript =
        akita_types::ProverGrindingTranscript::<T>::new(transcript, &grinding_plan)?;
    let root = prove_root::<Cfg::Field, Cfg::ExtField, _, P, C, O, TS, R, Cfg>(
        expanded,
        prefix_slots,
        stacks,
        &mut grinding_transcript,
        claims,
        &schedule.root,
        next_params,
        next_binding,
        basis,
    )
    .map_err(|err| AkitaError::InvalidInput(format!("root prove failed: {err:?}")))?;
    let next_state = root.next_state;
    let root = root.level_proof;

    // Prepared NTT state belongs to the supplied stack selector. Shared owners
    // retain it by default; an owner with an isolated root cache may release it
    // at this exact root/suffix boundary through the lifecycle hook.
    stacks.after_root_fold()?;

    let suffix = super::suffix::prove_suffix::<Cfg, _, C, O, TS, R>(
        expanded,
        prefix_slots,
        stacks,
        &mut grinding_transcript,
        next_state,
        schedule,
    )
    .map_err(|err| AkitaError::InvalidInput(format!("suffix prove failed: {err:?}")))?;
    let nonce_stream = grinding_transcript.finish()?;
    Ok(AkitaBatchedProof {
        nonce_stream,
        root,
        recursive_folds: suffix.recursive_folds,
        terminal: suffix.terminal,
    })
}
