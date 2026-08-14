use super::*;
use crate::backend::{RecursiveFoldSource, RecursiveWitnessFlat};
use crate::compute::{
    prewarm_ntt_requirements, ComputeBackendSetup, DigitRowsComputeBackend, LevelProveStacks,
    NttExecutionRequirements, RuntimeCoefficientPackingBackendFor, RuntimeCommitBackendFor,
    RuntimeOpeningProveBackendFor, RuntimeRingSwitchProveBackend, RuntimeTensorBackendFor,
    SuffixOpeningProveBackend, SuffixTensorProveBackend,
};
use crate::RootTensorProjectionPoly;
use crate::SelectedProverOpeningData;
use akita_config::{
    effective_batched_schedule, ensure_prover_schedule_fits_setup, CommitmentConfig,
};
use akita_field::unreduced::ReduceTo;
use akita_field::{AdditiveGroup, CanonicalField};

/// Drive batched proving end-to-end under config `Cfg`.
///
/// This owns the full top-level prover work: validate/flatten public prover
/// claims, select the folded schedule from `Cfg`, bind the transcript instance
/// descriptor, and run the folded prover.
///
/// # Errors
///
/// Returns an error if claim preparation, schedule selection, transcript
/// binding, or folded proving fails.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn batched_prove<'a, Cfg, T, P, C, O, TS, R>(
    expanded: &Arc<AkitaExpandedSetup<Cfg::Field>>,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field>,
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
    Cfg::Field: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    Cfg::ExtField: FpExtEncoding<Cfg::Field> + MulBaseUnreduced<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + FrobeniusExtField<Cfg::Field>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<Cfg::Field> + ProverTranscriptGrind<Cfg::Field>,
    Cfg::Field: FromPrimitiveInt + 'static,
    <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field> + AdditiveGroup,
    P: PreparedGroupProveOps<Cfg::Field, Cfg::ExtField, O, TS>,
    C: ComputeBackendSetup<Cfg::Field>
        + RuntimeCommitBackendFor<Cfg::Field, RecursiveWitnessFlat>
        + 'a,
    O: ComputeBackendSetup<Cfg::Field>
        + RuntimeOpeningProveBackendFor<Cfg::Field, RecursiveFoldSource<Cfg::Field>>
        + RuntimeOpeningProveBackendFor<Cfg::Field, RootTensorProjectionPoly<Cfg::Field>>
        + RuntimeCoefficientPackingBackendFor<
            Cfg::Field,
            RecursiveFoldSource<Cfg::Field>,
            Cfg::ExtField,
        > + RuntimeCoefficientPackingBackendFor<
            Cfg::Field,
            RootTensorProjectionPoly<Cfg::Field>,
            Cfg::ExtField,
        > + SuffixOpeningProveBackend<Cfg::Field>
        + DigitRowsComputeBackend<Cfg::Field>
        + 'a,
    TS: ComputeBackendSetup<Cfg::Field>
        + RuntimeTensorBackendFor<Cfg::Field, RecursiveFoldSource<Cfg::Field>, Cfg::ExtField>
        + RuntimeTensorBackendFor<Cfg::Field, RootTensorProjectionPoly<Cfg::Field>, Cfg::ExtField>
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
    let opening_claims = claims.opening_claims();
    let opening_batch = claims.opening_layout()?;
    let final_group_point = opening_claims.group_point(opening_batch.root_final_group_index()?)?;
    let resolved = Cfg::resolve_schedule_selection(selection)?;
    let resolved = effective_batched_schedule::<Cfg>(resolved, &opening_batch, final_group_point)?;
    let schedule = resolved.schedule();
    schedule.validate_evaluation_trace_execution()?;
    ensure_prover_schedule_fits_setup::<Cfg>(expanded.as_ref(), schedule, &opening_batch)?;
    let ntt_requirements = NttExecutionRequirements::from_prove_schedule(schedule)?;
    prewarm_ntt_requirements::<Cfg::Field, _>(stacks, &ntt_requirements)?;
    bind_transcript_instance_descriptor::<Cfg::Field, T, Cfg>(
        expanded.as_ref(),
        &opening_batch,
        selection,
        schedule,
        basis,
        transcript,
    )?;

    prove::<Cfg, T, P, C, O, TS, R>(
        expanded,
        prefix_slots,
        stacks,
        transcript,
        claims,
        schedule,
        basis,
    )
    .map(|(proof, _total_levels)| proof)
}

/// Prove a folded batched root and assemble the recursive suffix under config
/// `Cfg`.
///
/// The prover crate owns folded-root preparation (root schedule shape checks,
/// opening-point reduction, commitment row shape validation), root fold
/// proving, the next-`w` commitment, recursive suffix proving, and final proof
/// assembly. All policy facts are obtained directly from `Cfg`.
///
/// # Errors
///
/// Returns an error if the schedule is not folded, root inputs are malformed,
/// root proving fails, or suffix construction fails.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[inline(never)]
pub fn prove<'a, Cfg, T, P, C, O, TS, R>(
    expanded: &Arc<AkitaExpandedSetup<Cfg::Field>>,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field>,
    stacks: &'a impl LevelProveStacks<
        'a,
        Cfg::Field,
        Commit = C,
        Opening = O,
        Tensor = TS,
        RingSwitch = R,
    >,
    transcript: &mut T,
    claims: ProverOpeningData<'a, Cfg::ExtField, P, Cfg::Field>,
    schedule: &FoldSchedule,
    basis: BasisMode,
) -> Result<(AkitaBatchedProof<Cfg::Field, Cfg::ExtField>, usize), AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    Cfg::ExtField: FpExtEncoding<Cfg::Field> + MulBaseUnreduced<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + FrobeniusExtField<Cfg::Field>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<Cfg::Field> + ProverTranscriptGrind<Cfg::Field>,
    Cfg::Field: FromPrimitiveInt + 'static,
    <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field> + AdditiveGroup,
    P: PreparedGroupProveOps<Cfg::Field, Cfg::ExtField, O, TS>,
    C: ComputeBackendSetup<Cfg::Field>
        + RuntimeCommitBackendFor<Cfg::Field, RecursiveWitnessFlat>
        + 'a,
    O: ComputeBackendSetup<Cfg::Field>
        + RuntimeOpeningProveBackendFor<Cfg::Field, RecursiveFoldSource<Cfg::Field>>
        + RuntimeOpeningProveBackendFor<Cfg::Field, RootTensorProjectionPoly<Cfg::Field>>
        + RuntimeCoefficientPackingBackendFor<
            Cfg::Field,
            RecursiveFoldSource<Cfg::Field>,
            Cfg::ExtField,
        > + RuntimeCoefficientPackingBackendFor<
            Cfg::Field,
            RootTensorProjectionPoly<Cfg::Field>,
            Cfg::ExtField,
        > + SuffixOpeningProveBackend<Cfg::Field>
        + DigitRowsComputeBackend<Cfg::Field>
        + 'a,
    TS: ComputeBackendSetup<Cfg::Field>
        + RuntimeTensorBackendFor<Cfg::Field, RecursiveFoldSource<Cfg::Field>, Cfg::ExtField>
        + RuntimeTensorBackendFor<Cfg::Field, RootTensorProjectionPoly<Cfg::Field>, Cfg::ExtField>
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
    schedule.validate_evaluation_trace_execution()?;
    let root_params = &schedule.root.params.final_group.commitment;
    {
        // Every public group commitment is the fixed terminal F payload. The
        // frozen B geometry derives the complete compression plan and therefore
        // the only accepted coefficient count.
        let opening_batch = claims.opening_layout()?;
        let commitments = claims.commitments();
        if commitments.len() != opening_batch.num_groups() {
            return Err(AkitaError::InvalidInput(
                "root commitment group count does not match opening batch".to_string(),
            ));
        }
        let relation_geometry =
            RelationWitnessGeometry::for_evaluation_trace_execution(root_params, &opening_batch)?;
        let relation_layout = relation_geometry.rhs_layout();
        for (group_index, commitment) in commitments.iter().enumerate() {
            let plan = relation_layout.compression_plan_for_group(group_index)?;
            if commitment.rows().coeff_len() != plan.terminal_coefficients() {
                return Err(AkitaError::InvalidInput(
                    "root compressed commitment does not match scheduled root params".to_string(),
                ));
            }
        }
    }

    let root_packed_w_len = root_input_witness_len(root_params);
    if root_packed_w_len != schedule.root.input_witness_len {
        return Err(AkitaError::InvalidSetup(
            "root input witness length does not match schedule".into(),
        ));
    }
    let (next_params, next_binding) = schedule.recursive_folds.first().map_or(
        (
            super::fold::FoldSuccessorParams::Terminal(&schedule.terminal.params.witness),
            akita_types::NextWitnessBindingPolicy::TerminalInnerState,
        ),
        |step| {
            (
                super::fold::FoldSuccessorParams::Recursive(&step.params),
                akita_types::NextWitnessBindingPolicy::OuterPayload,
            )
        },
    );

    let root = prove_root::<Cfg::Field, Cfg::ExtField, T, P, C, O, TS, R, Cfg>(
        expanded,
        prefix_slots,
        stacks,
        transcript,
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

    let suffix = crate::prove_suffix::<Cfg, T, C, O, TS, R>(
        expanded,
        prefix_slots,
        stacks,
        transcript,
        next_state,
        schedule,
    )
    .map_err(|err| AkitaError::InvalidInput(format!("suffix prove failed: {err:?}")))?;
    Ok((
        AkitaBatchedProof {
            root,
            recursive_folds: suffix.recursive_folds,
            terminal: suffix.terminal,
        },
        suffix.num_levels,
    ))
}
