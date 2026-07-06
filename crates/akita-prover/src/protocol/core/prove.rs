use super::*;
use crate::api::commitment::validate_onehot_chunk_size_for_params;
use crate::compute::{
    CommitmentComputeBackend, ComputeBackendSetup, DigitRowsComputeBackend,
    DirectRootWitnessSource, LevelProveStacks, ProveStackFor, RootPolyMeta,
    RuntimeOpeningProveBackendFor, RuntimeRingSwitchProveBackend, RuntimeRootProvePoly,
    RuntimeTensorBackendFor, SuffixOpeningProveBackend, SuffixTensorProveBackend,
};
use crate::RootTensorProjectionPoly;
use akita_config::{effective_batched_schedule, CommitmentConfig};
use akita_field::unreduced::ReduceTo;
use akita_field::AdditiveGroup;
use akita_types::{
    schedule_terminal_direct_witness_shape, should_reject_grouped_root,
    validate_schedule_ring_dims, GROUPED_ROOT_RECURSIVE_SETUP_UNSUPPORTED,
};

fn grouped_root_prover_error(message: &'static str) -> AkitaError {
    if message == GROUPED_ROOT_RECURSIVE_SETUP_UNSUPPORTED {
        AkitaError::InvalidSetup(message.to_string())
    } else {
        AkitaError::InvalidInput(message.to_string())
    }
}

/// Build a root-direct batched proof from flattened polynomial references and
/// their commitment-group hints.
///
/// `ring_d` is the schedule-derived root commit ring dimension; the direct
/// witness materialization is the one typed operation and dispatches on it.
///
/// # Errors
///
/// Returns an error if any polynomial cannot produce a direct root witness.
pub fn prove_root_direct<F, E, P>(
    polys: &[&P],
    hints: &[AkitaCommitmentHint<F>],
    ring_d: usize,
) -> Result<AkitaBatchedProof<F, E>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
    P: DirectRootWitnessSource<F, 32>
        + DirectRootWitnessSource<F, 64>
        + DirectRootWitnessSource<F, 128>
        + DirectRootWitnessSource<F, 256>,
{
    let witnesses = dispatch_ring_dim_result!(ring_d, |D| {
        polys
            .iter()
            .map(|poly| DirectRootWitnessSource::<F, D>::direct_root_witness(*poly))
            .collect::<Result<Vec<_>, _>>()
    })?;
    let _ = hints;
    Ok(AkitaBatchedProof {
        root: AkitaBatchedRootProof::new_zero_fold(witnesses),
        steps: Vec::new(),
    })
}

/// Drive batched proving end-to-end under config `Cfg`.
///
/// This owns the full top-level prover work: validate/flatten public prover
/// claims, select the schedule from `Cfg`, apply the root-direct shortcut when
/// the selected schedule says no fold is needed, bind the transcript instance
/// descriptor, and either emit a root-direct proof or run the folded-root
/// prover.
///
/// # Errors
///
/// Returns an error if claim preparation, schedule selection, root-direct
/// witness construction, transcript binding, or folded-root proving fails.
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
    setup_contribution_mode: SetupContributionMode,
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
        + RuntimeOpeningProveBackendFor<Cfg::Field, RootTensorProjectionPoly<Cfg::Field>>
        + SuffixOpeningProveBackend<Cfg::Field>
        + DigitRowsComputeBackend<Cfg::Field>
        + 'a,
    TS: ComputeBackendSetup<Cfg::Field>
        + RuntimeTensorBackendFor<Cfg::Field, P, Cfg::ExtField>
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
    if let Some(message) = should_reject_grouped_root(
        &opening_batch,
        setup_contribution_mode,
        Some(
            flat_polys
                .iter()
                .any(|poly| poly.onehot_chunk_size().is_none()),
        ),
    ) {
        return Err(grouped_root_prover_error(message));
    }
    let schedule = effective_batched_schedule::<Cfg>(&opening_batch, claims.point())?;
    validate_schedule_ring_dims(&schedule, expanded.seed())?;
    let root_commit_params = match schedule.steps.first() {
        Some(Step::Fold(root)) => &root.params,
        Some(Step::Direct(root)) => root.params.as_ref().ok_or_else(|| {
            AkitaError::InvalidSetup("root-direct schedule missing commit params".to_string())
        })?,
        None => {
            return Err(AkitaError::InvalidSetup(
                "root schedule is empty".to_string(),
            ));
        }
    };
    validate_onehot_chunk_size_for_params::<Cfg::Field, &P>(&flat_polys, root_commit_params)?;

    // The transcript instance descriptor binds the setup-wide root ring
    // dimension (`gen_ring_dim`), NOT the root stack's const `D`. For uniform-D
    // presets `gen_ring_dim == Cfg::D == D`, so the descriptor bytes are
    // unchanged today; binding `gen_ring_dim` (via the canonical dispatcher,
    // exactly as the verifier's `batched_verify` does) keeps the prover and
    // verifier descriptors byte-identical under a future mixed-D preset. This is
    // the one absorption-parity point the compiler cannot check (S7/S9 caveat).
    dispatch_ring_dim_result!(expanded.seed().gen_ring_dim, |GEN_D| {
        bind_transcript_instance_descriptor::<Cfg::Field, T, GEN_D, Cfg>(
            expanded.as_ref(),
            &opening_batch,
            &schedule,
            basis,
            transcript,
        )
    })?;

    if schedule_is_root_direct(&schedule) {
        let commitment_hints = claims.hints().to_vec();
        return prove_root_direct::<Cfg::Field, Cfg::ExtField, P>(
            &flat_polys,
            &commitment_hints,
            root_commit_params.role_dims.d_a(),
        );
    }

    if schedule_root_fold_step(&schedule).is_none() {
        return Err(AkitaError::InvalidSetup(
            "root schedule does not start with a fold".to_string(),
        ));
    }
    prove::<Cfg, T, P, C, O, TS, R>(
        expanded,
        prefix_slots,
        stacks,
        transcript,
        claims,
        &schedule,
        basis,
        setup_contribution_mode,
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
    schedule: &Schedule,
    basis: BasisMode,
    setup_contribution_mode: SetupContributionMode,
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
        + RuntimeOpeningProveBackendFor<Cfg::Field, RootTensorProjectionPoly<Cfg::Field>>
        + SuffixOpeningProveBackend<Cfg::Field>
        + DigitRowsComputeBackend<Cfg::Field>
        + 'a,
    TS: ComputeBackendSetup<Cfg::Field>
        + RuntimeTensorBackendFor<Cfg::Field, P, Cfg::ExtField>
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
    for level in 0..schedule.num_fold_levels() {
        let role_dims = schedule.get_execution_schedule(level)?.params.role_dims();
        stacks
            .prove_stack_at_level(level)
            .ensure_fold_level_role_ntt(expanded.as_ref(), role_dims)?;
    }

    let root_scheduled = schedule.get_execution_schedule(0)?;
    {
        // §6 invariant — commitment vector length == num_rings · ring_dim.
        // The flat `Commitment` stores raw coefficients; validate its ring count
        // against the scheduled root params under the schedule-derived ring
        // dimension via `RingView::new` (no-panic gate, mirrors the verifier's
        // commitment-length check) before interpreting it as ring rows.
        let root_ring_dim = root_scheduled.params.role_dims().d_b();
        let expected_rows = root_scheduled.params.b_key.row_len();
        let commitments = claims.commitments();
        for commitment in commitments {
            let view = RingView::new(commitment.rows().coeffs(), root_ring_dim)?;
            if view.num_rings() != expected_rows {
                return Err(AkitaError::InvalidInput(
                    "root commitment row count does not match scheduled root params".to_string(),
                ));
            }
        }
    }

    let root_packed_w_len = root_current_w_len(&root_scheduled.params);
    root_scheduled.validate_current_w_len(root_packed_w_len)?;

    if root_scheduled.is_terminal {
        // Root is itself the terminal fold: no recursive suffix.
        let terminal_shape = schedule_terminal_direct_witness_shape(schedule)?;
        let terminal = prove_terminal_root_fold_with_params::<
            Cfg,
            Cfg::Field,
            Cfg::ExtField,
            T,
            P,
            C,
            O,
            TS,
            R,
        >(
            expanded,
            stacks,
            transcript,
            claims,
            &root_scheduled,
            terminal_shape,
            basis,
            setup_contribution_mode,
        )?;
        return Ok((
            AkitaBatchedProof {
                root: AkitaBatchedRootProof::new_terminal(terminal),
                steps: Vec::new(),
            },
            1,
        ));
    }

    let root = prove_root::<Cfg::Field, Cfg::ExtField, T, P, C, O, TS, R, Cfg>(
        expanded,
        prefix_slots,
        stacks,
        transcript,
        claims,
        &root_scheduled,
        basis,
        setup_contribution_mode,
    )?;
    let next_state = root.next_state;
    let root = AkitaBatchedRootProof::new(root.level_proof);

    let suffix = crate::prove_suffix::<Cfg, T, C, O, TS, R>(
        expanded,
        prefix_slots,
        stacks,
        transcript,
        next_state,
        schedule,
        setup_contribution_mode,
    )?;
    Ok((
        AkitaBatchedProof {
            root,
            steps: suffix.steps,
        },
        suffix.num_levels,
    ))
}
