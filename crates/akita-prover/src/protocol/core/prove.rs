use super::*;
use crate::api::commitment::validate_onehot_chunk_size_for_params;
use crate::backend::RecursiveFoldSource;
use crate::compute::{
    CommitmentComputeBackend, ComputeBackendSetup, DigitRowsComputeBackend, LevelProveStacks,
    ProveStackFor, RootPolyMeta, RuntimeOpeningProveBackendFor, RuntimeRingSwitchProveBackend,
    RuntimeRootProvePoly, RuntimeTensorBackendFor, SuffixOpeningProveBackend,
    SuffixTensorProveBackend,
};
use crate::RootTensorProjectionPoly;
use akita_config::{effective_batched_schedule, ensure_schedule_fits_setup, CommitmentConfig};
use akita_field::unreduced::ReduceTo;
use akita_field::{AdditiveGroup, CanonicalField};
use akita_types::{
    dispatch_for_field, should_reject_multi_group_root, validate_schedule_ring_dims,
};

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
    claims: ProverOpeningData<'a, Cfg::ExtField, P, Cfg::Field>,
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
    P: RuntimeRootProvePoly<Cfg::Field>,
    C: ComputeBackendSetup<Cfg::Field> + CommitmentComputeBackend<Cfg::Field> + 'a,
    O: ComputeBackendSetup<Cfg::Field>
        + RuntimeOpeningProveBackendFor<Cfg::Field, P>
        + RuntimeOpeningProveBackendFor<Cfg::Field, RecursiveFoldSource<Cfg::Field>>
        + RuntimeOpeningProveBackendFor<Cfg::Field, RootTensorProjectionPoly<Cfg::Field>>
        + SuffixOpeningProveBackend<Cfg::Field>
        + DigitRowsComputeBackend<Cfg::Field>
        + 'a,
    TS: ComputeBackendSetup<Cfg::Field>
        + RuntimeTensorBackendFor<Cfg::Field, P, Cfg::ExtField>
        + RuntimeTensorBackendFor<Cfg::Field, RecursiveFoldSource<Cfg::Field>, Cfg::ExtField>
        + RuntimeTensorBackendFor<Cfg::Field, RootTensorProjectionPoly<Cfg::Field>, Cfg::ExtField>
        + SuffixTensorProveBackend<Cfg::Field, Cfg::ExtField>
        + 'a,
    R: ComputeBackendSetup<Cfg::Field>
        + RuntimeRingSwitchProveBackend<Cfg::Field>
        + DigitRowsComputeBackend<Cfg::Field>
        + 'a,
    (): ProveStackFor<Cfg::Field, P, Cfg::ExtField, C, O, TS, R>,
    <C as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'a,
    <O as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'a,
    <TS as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'a,
    <R as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'a,
{
    claims.validate::<Cfg::Field>()?;
    let opening_claims = claims.opening_claims();
    opening_claims.validate(expanded.seed())?;
    let opening_batch = opening_claims.layout()?;
    let flat_polys = claims.flat_polys();
    if let Some(message) = should_reject_multi_group_root(
        &opening_batch,
        flat_polys
            .iter()
            .any(|poly| poly.onehot_chunk_size().is_none()),
    ) {
        return Err(AkitaError::InvalidInput(message.to_string()));
    }
    let schedule = effective_batched_schedule::<Cfg>(&opening_batch, claims.point())?;
    validate_schedule_ring_dims(&schedule, expanded.seed())?;
    ensure_schedule_fits_setup::<Cfg>(expanded.as_ref(), &schedule, &opening_batch)?;
    schedule.validate_structure()?;
    let root_step = schedule.root_fold();
    let root_commit_params = &root_step.params.final_group.commitment;
    validate_onehot_chunk_size_for_params::<Cfg::Field, &P>(&flat_polys, root_commit_params)?;

    // The transcript instance descriptor binds the setup-wide root ring
    // dimension (`gen_ring_dim`), NOT the root stack's const `D`. For uniform-D
    // presets `gen_ring_dim == Cfg::D == D`, so the descriptor bytes are
    // unchanged today; binding `gen_ring_dim` (via the canonical dispatcher,
    // exactly as the verifier's `batched_verify` does) keeps the prover and
    // verifier descriptors byte-identical under a future mixed-D preset. This is
    // the one absorption-parity point the compiler cannot check (S7/S9 caveat).
    dispatch_for_field!(
        ProtocolDispatchSlot::Envelope,
        Cfg::Field,
        expanded.seed().gen_ring_dim,
        |GEN_D| {
            bind_transcript_instance_descriptor::<Cfg::Field, T, GEN_D, Cfg>(
                expanded.as_ref(),
                &opening_batch,
                &schedule,
                basis,
                transcript,
            )
        }
    )?;

    prove::<Cfg, T, P, C, O, TS, R>(
        expanded,
        prefix_slots,
        stacks,
        transcript,
        claims,
        &schedule,
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
    P: RuntimeRootProvePoly<Cfg::Field>,
    C: ComputeBackendSetup<Cfg::Field> + CommitmentComputeBackend<Cfg::Field> + 'a,
    O: ComputeBackendSetup<Cfg::Field>
        + RuntimeOpeningProveBackendFor<Cfg::Field, P>
        + RuntimeOpeningProveBackendFor<Cfg::Field, RecursiveFoldSource<Cfg::Field>>
        + RuntimeOpeningProveBackendFor<Cfg::Field, RootTensorProjectionPoly<Cfg::Field>>
        + SuffixOpeningProveBackend<Cfg::Field>
        + DigitRowsComputeBackend<Cfg::Field>
        + 'a,
    TS: ComputeBackendSetup<Cfg::Field>
        + RuntimeTensorBackendFor<Cfg::Field, P, Cfg::ExtField>
        + RuntimeTensorBackendFor<Cfg::Field, RecursiveFoldSource<Cfg::Field>, Cfg::ExtField>
        + RuntimeTensorBackendFor<Cfg::Field, RootTensorProjectionPoly<Cfg::Field>, Cfg::ExtField>
        + SuffixTensorProveBackend<Cfg::Field, Cfg::ExtField>
        + 'a,
    R: ComputeBackendSetup<Cfg::Field>
        + RuntimeRingSwitchProveBackend<Cfg::Field>
        + DigitRowsComputeBackend<Cfg::Field>
        + 'a,
    (): ProveStackFor<Cfg::Field, P, Cfg::ExtField, C, O, TS, R>,
    <C as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'a,
    <O as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'a,
    <TS as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'a,
    <R as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'a,
{
    // Role dims were validated against the setup seed at batched_prove entry;
    // NTT pre-warm reads the same schedule-owned dims per level.
    let root_params = &schedule.root.params.final_group.commitment;
    stacks
        .prove_stack_at_level(0)
        .ensure_fold_level_role_ntt(expanded.as_ref(), root_params.role_dims())?;
    for (offset, step) in schedule.recursive_folds.iter().enumerate() {
        stacks
            .prove_stack_at_level(offset + 1)
            .ensure_fold_level_role_ntt(expanded.as_ref(), step.params.witness.role_dims())?;
    }
    stacks
        .prove_stack_at_level(schedule.num_fold_levels() - 1)
        .ensure_fold_level_envelope_ntt(
            expanded.as_ref(),
            schedule.terminal.params.witness.d_a(),
        )?;
    {
        // §6 invariant — commitment vector length == num_rings · ring_dim.
        // The flat `Commitment` stores raw coefficients; validate its ring count
        // against the scheduled root params under the schedule-derived ring
        // dimension via `RingView::new` (no-panic gate, mirrors the verifier's
        // commitment-length check) before interpreting it as ring rows.
        let root_ring_dim = root_params.role_dims().d_b();
        let opening_batch = claims.opening_claims().layout()?;
        let commitments = claims.commitments();
        if commitments.len() != opening_batch.num_groups() {
            return Err(AkitaError::InvalidInput(
                "root commitment group count does not match opening batch".to_string(),
            ));
        }
        for (group_index, commitment) in commitments.iter().enumerate() {
            let expected_rows = root_params.group_commitment_rows(&opening_batch, group_index)?;
            let view = RingView::new(commitment.rows().coeffs(), root_ring_dim)?;
            if view.num_rings() != expected_rows {
                return Err(AkitaError::InvalidInput(
                    "root commitment row count does not match scheduled root params".to_string(),
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
                akita_types::NextWitnessBindingPolicy::OuterCommitment,
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
