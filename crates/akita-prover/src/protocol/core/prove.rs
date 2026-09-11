use super::*;
use crate::backend::RecursiveFoldSource;
use crate::commitment::{
    CommitmentExecutionPlan, CommitmentStatePolicy, InnerRelationState, OuterCompressionState,
    TerminalBindingState,
};
use crate::compute::{
    prewarm_ntt_requirements, ComputeBackendSetup, DigitRowsComputeBackend, LevelProveStacks,
    NttExecutionRequirements, RuntimeCoefficientPackingBackendFor, RuntimeOpeningProveBackendFor,
    RuntimeRingSwitchProveBackend, RuntimeTensorBackendFor, SuffixOpeningProveBackend,
    SuffixTensorProveBackend,
};
use crate::SelectedProverOpeningData;
use akita_config::{ensure_prover_schedule_fits_setup, CommitmentConfig, TrustedScheduleCatalog};
use jolt_field::{AdditiveGroup, CanonicalEncoding};

struct AdmittedProverInput<'opening, 'schedule, E: Clone, P, F: Field, S> {
    selection: akita_types::OpeningScheduleSelection,
    claims: ProverOpeningData<'opening, E, P, F, S>,
    schedule: &'schedule FoldSchedule,
}

impl<'stack, Stacks: ?Sized> ProverExecutor<'stack, Stacks> {
    fn validate_params<'opening, 'schedule, Cfg, P, S, O, TS, R, SP>(
        &self,
        expanded: &AkitaExpandedSetup<Cfg::Field>,
        schedules: &'schedule TrustedScheduleCatalog<Cfg>,
        opening: SelectedProverOpeningData<'opening, Cfg::ExtField, P, Cfg::Field, S>,
    ) -> Result<AdmittedProverInput<'opening, 'schedule, Cfg::ExtField, P, Cfg::Field, S>, AkitaError>
    where
        Cfg: CommitmentConfig,
        Cfg::Field: Field,
        Cfg::ExtField: Clone,
        P: RootProverGroupMeta<Cfg::Field>,
        S: InnerRelationState<Cfg::Field> + OuterCompressionState<Cfg::Field>,
        SP: CommitmentStatePolicy<Cfg::Field> + 'stack,
        SP::State: InnerRelationState<Cfg::Field>
            + OuterCompressionState<Cfg::Field>
            + TerminalBindingState<Cfg::Field>,
        Stacks: LevelProveStacks<
            'stack,
            Cfg::Field,
            Opening = O,
            Tensor = TS,
            RingSwitch = R,
            CommitmentStatePolicy = SP,
        >,
        O: ComputeBackendSetup<Cfg::Field> + 'stack,
        TS: ComputeBackendSetup<Cfg::Field> + 'stack,
        R: ComputeBackendSetup<Cfg::Field> + 'stack,
        <O as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'stack,
        <TS as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'stack,
        <R as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'stack,
    {
        let (selection, claims) = opening.into_low_level_parts();
        let opening_batch = claims.opening_layout();
        let resolved = schedules.resolve_selection(selection)?;
        resolved.validate_opening_layout(opening_batch)?;
        let schedule = resolved.schedule();
        schedule.validate_nonterminal_opening_execution(Cfg::EXT_DEGREE)?;
        ensure_prover_schedule_fits_setup::<Cfg>(expanded, schedule, opening_batch)?;
        let relation_geometry = akita_types::RelationWitnessGeometry::for_level(
            &schedule.root.params,
            opening_batch,
            Cfg::EXT_DEGREE,
        )?;
        for group_index in 0..claims.opening_claims().num_groups() {
            let state = claims.group_state(group_index)?;
            let group = schedule
                .root
                .params
                .group_params(opening_batch, group_index)?;
            let plan = CommitmentExecutionPlan::for_root(&group.profile)?;
            state.preflight_inner_relation(
                plan.inner(),
                opening_batch.group_layout(group_index)?.num_polynomials(),
            )?;
            if schedule.root.params.payload_mode.is_compressed() {
                state.preflight_outer_compression(
                    relation_geometry
                        .rhs_layout()
                        .compression_plan_for_group(group_index)?,
                    schedule.root.params.ring_relation_mode,
                )?;
            }
        }
        for (index, step) in schedule.recursive_folds.iter().enumerate() {
            let fold_level = index
                .checked_add(1)
                .ok_or_else(|| AkitaError::InvalidSetup("fold level overflow".into()))?;
            let plan = CommitmentExecutionPlan::for_recursive(&step.params, fold_level, 1)?;
            self.stacks
                .prove_stack_at_level(index)
                .commitment()
                .preflight_prover_state_consumers(&plan)?;
        }
        let terminal_plan = CommitmentExecutionPlan::for_terminal(&schedule.terminal)?;
        self.stacks
            .prove_stack_at_level(schedule.recursive_folds.len())
            .commitment()
            .preflight_prover_state_consumers(&terminal_plan)?;
        Ok(AdmittedProverInput {
            selection,
            claims,
            schedule,
        })
    }

    fn prepare_resources<Cfg, O, TS, R, SP>(
        &self,
        schedule: &FoldSchedule,
    ) -> Result<(), AkitaError>
    where
        Cfg: CommitmentConfig,
        Stacks: LevelProveStacks<
            'stack,
            Cfg::Field,
            Opening = O,
            Tensor = TS,
            RingSwitch = R,
            CommitmentStatePolicy = SP,
        >,
        O: ComputeBackendSetup<Cfg::Field> + 'stack,
        TS: ComputeBackendSetup<Cfg::Field> + 'stack,
        R: ComputeBackendSetup<Cfg::Field> + 'stack,
        SP: CommitmentStatePolicy<Cfg::Field> + 'stack,
        <O as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'stack,
        <TS as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'stack,
        <R as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'stack,
    {
        let requirements = NttExecutionRequirements::from_prove_schedule(schedule)?;
        prewarm_ntt_requirements::<Cfg::Field, _>(self.stacks, &requirements)
    }
}

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
pub fn batched_prove<'a, Cfg, T, P, S, O, TS, R, SP>(
    expanded: &Arc<AkitaExpandedSetup<Cfg::Field>>,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field>,
    schedules: &TrustedScheduleCatalog<Cfg>,
    stacks: &'a impl LevelProveStacks<
        'a,
        Cfg::Field,
        Opening = O,
        Tensor = TS,
        RingSwitch = R,
        CommitmentStatePolicy = SP,
    >,
    opening: SelectedProverOpeningData<'a, Cfg::ExtField, P, Cfg::Field, S>,
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
    S: InnerRelationState<Cfg::Field> + OuterCompressionState<Cfg::Field>,
    SP: CommitmentStatePolicy<Cfg::Field> + 'a,
    SP::State: InnerRelationState<Cfg::Field>
        + OuterCompressionState<Cfg::Field>
        + TerminalBindingState<Cfg::Field>,
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
    <O as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'a,
    <TS as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'a,
    <R as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'a,
{
    let executor = ProverExecutor { stacks };
    let admitted = executor.validate_params::<Cfg, P, S, O, TS, R, SP>(
        expanded.as_ref(),
        schedules,
        opening,
    )?;
    let AdmittedProverInput {
        selection,
        claims,
        schedule,
    } = admitted;
    let opening_batch = claims.opening_layout();
    executor.prepare_resources::<Cfg, O, TS, R, SP>(schedule)?;
    let grinding_plan = bind_transcript_instance_descriptor::<Cfg::Field, T, Cfg>(
        expanded.as_ref(),
        opening_batch,
        selection,
        schedule,
        basis,
        transcript,
    )?;

    let next_params = schedule.recursive_folds.first().map_or(
        super::fold::FoldSuccessorParams::Terminal(&schedule.terminal),
        super::fold::FoldSuccessorParams::Recursive,
    );

    let mut grinding_transcript =
        akita_types::ProverGrindingTranscript::<T>::new(transcript, &grinding_plan)?;
    let root = executor
        .prove_root::<Cfg::Field, Cfg::ExtField, _, P, S, O, TS, R, SP, Cfg>(
            expanded,
            prefix_slots,
            &mut grinding_transcript,
            claims,
            &schedule.root,
            next_params,
            basis,
        )
        .map_err(|err| AkitaError::InvalidInput(format!("root prove failed: {err:?}")))?;
    let next_state = root.next_state;
    let root = root.level_proof;

    // Prepared NTT state belongs to the supplied stack selector. Shared owners
    // retain it by default; an owner with an isolated root cache may release it
    // at this exact root/suffix boundary through the lifecycle hook.
    stacks.after_root_fold()?;

    let suffix = executor
        .prove_suffix::<Cfg, _, O, TS, R, SP>(
            expanded,
            prefix_slots,
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
