use super::*;
use crate::backend::{RecursiveFoldSource, RecursiveWitnessFlat};
use crate::commitment::{
    CommitmentExecutionPlan, CommitmentStatePolicy, InnerRelationState, OuterCompressionState,
};
use crate::compute::{
    ComputeBackendSetup, DigitRowsComputeBackend, LevelProveStacks, ProverComputeStack,
    RuntimeCoefficientPackingBackendFor, RuntimeOpeningProveBackendFor,
    RuntimeRingSwitchProveBackend, RuntimeTensorBackendFor, SuffixOpeningProveBackend,
    SuffixTensorProveBackend,
};
use akita_types::AkitaCommitmentHint;
use jolt_field::AdditiveGroup;
use std::sync::Arc;

/// Prover state carried between suffix fold levels.
pub struct SuffixProverState<F: Field, E: Field, S = AkitaCommitmentHint<F>> {
    /// Current committed suffix witness representation.
    pub w: RecursiveWitnessFlat,
    /// Logical suffix witness when it differs from the committed representation.
    pub logical_w: Option<RecursiveWitnessFlat>,
    /// Transcript-bound public state for the current suffix witness.
    pub binding: NextWitnessState<F>,
    /// Persistent semantic A-ring rows for the current suffix commitment.
    pub prover_state: S,
    pub(crate) inner_relation_material: crate::commitment::InnerRelationStateMaterial<F>,
    pub(crate) compression_material: Option<crate::commitment::PortableCompressionState<F>>,
    /// Current digit basis, as `log2(b)`.
    pub log_basis: u32,
    /// Sumcheck challenges that become the next suffix opening point.
    pub sumcheck_challenges: Vec<E>,
    /// Claimed logical opening of `logical_w` at `sumcheck_challenges`.
    pub opening: E,
    /// Optional setup-prefix opening carried from the previous stage-3 proof.
    pub setup_prefix_opening: Option<(Vec<E>, E)>,
}

impl<F: Field, E: Field, S> SuffixProverState<F, E, S> {
    /// Logical witness represented by the carried opening claim.
    #[inline]
    pub fn logical_w(&self) -> &RecursiveWitnessFlat {
        self.logical_w.as_ref().unwrap_or(&self.w)
    }
}

impl<'stack, Stacks: ?Sized> ProverExecutor<'stack, Stacks> {
    #[allow(dead_code)] // Called by the native production entry point during cutover.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prove_suffix_native<Cfg, O, TS, R, SP>(
        &self,
        expanded: &Arc<AkitaExpandedSetup<Cfg::Field>>,
        prefix_slots: &SetupPrefixProverRegistry<Cfg::Field>,
        grinding: &mut akita_types::NativeProverGrinding<'_>,
        starting_state: SuffixProverState<Cfg::Field, Cfg::ExtField, SP::State>,
        schedule: &FoldSchedule,
    ) -> Result<NativeRecursiveSuffixOutcome, AkitaError>
    where
        Cfg: CommitmentConfig,
        Cfg::Field: Field
            + CanonicalEncoding
            + akita_serialization::AkitaSerialize
            + Unreduced
            + PseudoMersenne
            + Ring
            + 'static,
        <Cfg::Field as Unreduced>::Wide: From<Cfg::Field> + AdditiveGroup,
        Cfg::ExtField: FpExtEncoding<Cfg::Field>
            + ExtField<Cfg::Field>
            + Unreduced
            + Fold
            + Ring
            + AkitaSerialize
            + MulBaseUnreduced<Cfg::Field>,
        SP: CommitmentStatePolicy<Cfg::Field> + 'stack,
        SP::State: InnerRelationState<Cfg::Field> + OuterCompressionState<Cfg::Field>,
        O: SuffixOpeningProveBackend<Cfg::Field>
            + RuntimeOpeningProveBackendFor<Cfg::Field, RecursiveFoldSource<Cfg::Field>>
            + RuntimeCoefficientPackingBackendFor<
                Cfg::Field,
                RecursiveFoldSource<Cfg::Field>,
                Cfg::ExtField,
            > + DigitRowsComputeBackend<Cfg::Field>
            + ComputeBackendSetup<Cfg::Field>
            + 'stack,
        TS: SuffixTensorProveBackend<Cfg::Field, Cfg::ExtField>
            + RuntimeTensorBackendFor<Cfg::Field, RecursiveFoldSource<Cfg::Field>, Cfg::ExtField>
            + ComputeBackendSetup<Cfg::Field>
            + 'stack,
        R: RuntimeRingSwitchProveBackend<Cfg::Field>
            + DigitRowsComputeBackend<Cfg::Field>
            + ComputeBackendSetup<Cfg::Field>
            + 'stack,
        <O as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'stack,
        <TS as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'stack,
        <R as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'stack,
        Stacks: LevelProveStacks<
            'stack,
            Cfg::Field,
            Opening = O,
            Tensor = TS,
            RingSwitch = R,
            CommitmentStatePolicy = SP,
        >,
    {
        let planned_num_levels = schedule.num_fold_levels();
        if planned_num_levels < 2 {
            return Err(AkitaError::InvalidSetup(
                "prove_suffix expects a non-empty recursive suffix".into(),
            ));
        }
        let mut current_state = starting_state;
        let mut level = 1usize;
        for (recursive_index, step) in schedule.recursive_folds.iter().enumerate() {
            let level_params = &step.params;
            if current_state.w.live_coeff_len() != step.input_witness_len {
                return Err(AkitaError::InvalidSetup(format!(
                    "scheduled fold level {level} did not match runtime state"
                )));
            }
            let successor = schedule.recursive_folds.get(recursive_index + 1);
            let (next_params, next_binding) = successor.map_or(
                (
                    super::fold::FoldSuccessorParams::Terminal(&schedule.terminal),
                    akita_types::NextWitnessBindingPolicy::TerminalInnerState,
                ),
                |next| {
                    (
                        super::fold::FoldSuccessorParams::Recursive(next),
                        akita_types::NextWitnessBindingPolicy::OuterPayload,
                    )
                },
            );
            let stack = self.stacks.prove_stack_at_level(level);
            let prepared = prepare_suffix_native::<Cfg::Field, Cfg::ExtField, O, TS, R, SP, _>(
                stack,
                expanded,
                prefix_slots,
                grinding,
                current_state,
                level,
                level_params,
            )?;
            current_state =
                super::fold::prove_fold_native::<Cfg::Field, Cfg::ExtField, O, TS, R, SP, Cfg>(
                    expanded,
                    prefix_slots,
                    stack,
                    grinding,
                    level,
                    level_params,
                    next_params,
                    step.output_witness_len,
                    next_binding,
                    prepared,
                )?
                .next_state;
            level += 1;
        }
        if current_state.w.live_coeff_len() != schedule.terminal.input_witness_len {
            return Err(AkitaError::InvalidSetup(
                "scheduled terminal fold did not match runtime state".into(),
            ));
        }
        prove_terminal_suffix_native::<Cfg::Field, Cfg::ExtField, O, TS, R, SP, _>(
            self.stacks.prove_stack_at_level(level),
            grinding,
            level,
            current_state,
            &schedule.terminal,
        )?;
        Ok(NativeRecursiveSuffixOutcome {
            num_levels: planned_num_levels,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn prove_terminal_suffix_native<F, E, O, TS, R, SP, S>(
    stack: &ProverComputeStack<'_, F, O, TS, R, SP>,
    grinding: &mut akita_types::NativeProverGrinding<'_>,
    level: usize,
    current_state: SuffixProverState<F, E, S>,
    scheduled: &TerminalFoldParams,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Unreduced + Ring + 'static,
    <F as Unreduced>::Wide: From<F> + AdditiveGroup,
    E: FpExtEncoding<F>
        + ExtField<F>
        + Unreduced
        + Fold
        + Ring
        + AkitaSerialize
        + MulBaseUnreduced<F>,
    O: SuffixOpeningProveBackend<F>
        + DigitRowsComputeBackend<F>
        + RuntimeOpeningProveBackendFor<F, RecursiveFoldSource<F>>
        + ComputeBackendSetup<F>,
    TS: SuffixTensorProveBackend<F, E>
        + RuntimeTensorBackendFor<F, RecursiveFoldSource<F>, E>
        + ComputeBackendSetup<F>,
    R: ComputeBackendSetup<F>,
    SP: CommitmentStatePolicy<F>,
    S: InnerRelationState<F>,
{
    let SuffixProverState {
        w,
        logical_w,
        binding,
        prover_state,
        inner_relation_material,
        compression_material,
        sumcheck_challenges,
        opening,
        setup_prefix_opening,
        ..
    } = current_state;
    if setup_prefix_opening.is_some() || !matches!(binding, NextWitnessState::TerminalInnerState) {
        return Err(AkitaError::InvalidProof);
    }
    let terminal_plan = CommitmentExecutionPlan::for_terminal(scheduled, level)?;
    inner_relation_material.validate(terminal_plan.inner(), 1)?;
    if compression_material.is_some() {
        return Err(AkitaError::InvalidProof);
    }
    drop(prover_state);
    let mut terminal_rows = inner_relation_material.into_rows();
    let t_state = terminal_rows.pop().ok_or(AkitaError::InvalidProof)?;
    if !terminal_rows.is_empty() {
        return Err(AkitaError::InvalidProof);
    }
    let level_u32 = u32::try_from(level)
        .map_err(|_| AkitaError::InvalidSetup("fold level exceeds u32".into()))?;
    akita_transcript::public_native_fields_prover(
        grinding.state_mut(),
        akita_transcript::ProtocolSiteId {
            family: akita_transcript::SITE_FAMILY_TERMINAL,
            level: level_u32,
            stage: 2,
            ..akita_transcript::ProtocolSiteId::default()
        },
        t_state.coeffs(),
    )
    .map_err(|_| AkitaError::InvalidProof)?;

    let witness = Arc::new(w);
    let logical_witness = logical_w
        .map(Arc::new)
        .unwrap_or_else(|| Arc::clone(&witness));
    let witness_source = RecursiveFoldSource::witness(Arc::clone(&witness));
    let logical_source = RecursiveFoldSource::witness(logical_witness);
    let params = scheduled;
    let alpha_bits = params.d_a().trailing_zeros() as usize;
    let recursive_num_vars = params.recursive_opening_num_vars()?;
    if sumcheck_challenges.len() > recursive_num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: recursive_num_vars,
            actual: sumcheck_challenges.len(),
        });
    }
    let opening_batch = OpeningClaimsLayout::new(sumcheck_challenges.len(), 1)?;
    let polys = [&logical_source];
    let logical_group = PreparedProverGroup::from_refs(&polys)?;
    let (protocol_point, reduction) = if E::DEGREE > 1 {
        let eor_inputs = vec![ExtensionOpeningGroupInput {
            group: &logical_group,
            point: &sumcheck_challenges,
            ring_dimension: params.d_a(),
        }];
        let proved = prove_extension_opening_reduction_native::<F, E, _, TS>(
            stack.tensor().backend(),
            Some(stack.tensor().prepared()),
            &eor_inputs,
            grinding,
            level_u32,
            "terminal",
        )?;
        (
            proved
                .protocol_points
                .into_iter()
                .next()
                .ok_or(AkitaError::InvalidProof)?,
            Some(proved.reduction),
        )
    } else {
        (sumcheck_challenges, None)
    };
    akita_transcript::public_native_extensions_prover::<F, E>(
        grinding.state_mut(),
        akita_transcript::ProtocolSiteId {
            family: akita_transcript::SITE_FAMILY_FOLD_BINDING,
            level: level_u32,
            stage: 5,
            ..akita_transcript::ProtocolSiteId::default()
        },
        &protocol_point,
    )
    .map_err(|_| AkitaError::InvalidProof)?;

    let (e_folded, fold_output) = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        params.d_a(),
        |D| {
            let (prepared_point, (folded_rings, folded_blocks)) =
                prepare_and_evaluate_opening_group::<F, E, RecursiveFoldSource<F>, O, D>(
                    stack.opening().backend(),
                    Some(stack.opening().prepared()),
                    &[&witness_source],
                    &protocol_point,
                    BasisMode::Lagrange,
                    params.blocks.positions_per_block,
                    params.blocks.live_blocks,
                    alpha_bits,
                )?;
            let (trace, _) = compute_trace_target_native::<F, E, D>(
                reduction.as_ref(),
                &folded_rings,
                std::slice::from_ref(&prepared_point),
                &protocol_point,
                alpha_bits,
                BasisMode::Lagrange,
                &opening_batch,
                grinding,
                level_u32,
            )?;
            if reduction.is_none() && trace.trace_eval_target != opening {
                return Err(AkitaError::InvalidProof);
            }
            let folded = folded_blocks
                .into_iter()
                .next()
                .ok_or(AkitaError::InvalidProof)?;
            let e_folded = RingVec::from_ring_elems(&folded);
            akita_transcript::send_native_field_group(
                grinding.state_mut(),
                akita_transcript::ProtocolSiteId {
                    family: akita_transcript::SITE_FAMILY_TERMINAL,
                    level: level_u32,
                    stage: 1,
                    ..akita_transcript::ProtocolSiteId::default()
                },
                e_folded.coeffs(),
            )
            .map_err(|_| AkitaError::InvalidProof)?;
            let output = crate::protocol::fold_grind::sample_terminal_fold_response_native(
                stack.opening().backend(),
                Some(stack.opening().prepared()),
                grinding,
                level_u32,
                params,
                &scheduled.fold_challenge_config,
                &witness_source,
                &scheduled.response_shape,
            )?;
            Ok::<_, AkitaError>((e_folded, output))
        }
    )?;
    let terminal_response = akita_types::build_terminal_response(
        params,
        &scheduled.response_shape,
        &e_folded,
        t_state,
        fold_output.witness.centered_coeffs_flat(),
    )?;
    let group = scheduled
        .response_shape
        .layout
        .groups
        .first()
        .ok_or(AkitaError::InvalidProof)?;
    let z_payload = terminal_response
        .z_payloads
        .first()
        .ok_or(AkitaError::InvalidProof)?;
    akita_transcript::send_native_bounded_bytes(
        grinding.state_mut(),
        akita_transcript::ProtocolSiteId {
            family: akita_transcript::SITE_FAMILY_TERMINAL,
            level: level_u32,
            round: 3,
            ..akita_transcript::ProtocolSiteId::default()
        },
        z_payload,
        group.z_payload_bytes,
    )
    .map_err(|_| AkitaError::InvalidProof)
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(in crate::protocol::core) fn prepare_suffix_native<F, E, O, TS, R, SP, S>(
    stack: &ProverComputeStack<'_, F, O, TS, R, SP>,
    expanded: &Arc<AkitaExpandedSetup<F>>,
    prefix_slots: &SetupPrefixProverRegistry<F>,
    grinding: &mut akita_types::NativeProverGrinding<'_>,
    current_state: SuffixProverState<F, E, S>,
    level: usize,
    level_params: &CommittedGroupParams,
) -> Result<PreparedFold<F, E>, AkitaError>
where
    F: Field
        + CanonicalEncoding
        + akita_serialization::AkitaSerialize
        + Unreduced
        + PseudoMersenne
        + Ring
        + 'static,
    <F as Unreduced>::Wide: From<F> + AdditiveGroup,
    E: FpExtEncoding<F>
        + ExtField<F>
        + Unreduced
        + Fold
        + Ring
        + AkitaSerialize
        + MulBaseUnreduced<F>,
    TS: RuntimeTensorBackendFor<F, RecursiveWitnessFlat, E>
        + RuntimeTensorBackendFor<F, RecursiveFoldSource<F>, E>,
    O: DigitRowsComputeBackend<F>
        + RuntimeOpeningProveBackendFor<F, RecursiveWitnessFlat>
        + RuntimeOpeningProveBackendFor<F, RecursiveFoldSource<F>>
        + RuntimeCoefficientPackingBackendFor<F, RecursiveFoldSource<F>, E>,
    R: DigitRowsComputeBackend<F> + RuntimeRingSwitchProveBackend<F>,
    SP: CommitmentStatePolicy<F>,
    S: InnerRelationState<F> + OuterCompressionState<F>,
{
    let SuffixProverState {
        w,
        logical_w: optional_logical_w,
        binding,
        prover_state,
        inner_relation_material,
        compression_material,
        sumcheck_challenges,
        opening,
        setup_prefix_opening,
        ..
    } = current_state;
    let witness = Arc::new(w);
    let logical_witness = optional_logical_w
        .map(Arc::new)
        .unwrap_or_else(|| Arc::clone(&witness));
    let payload_geometry = level_params.outer_payload_geometry()?;
    let witness_commitment = match binding {
        NextWitnessState::OuterPayload(commitment) => {
            if commitment.coeff_len() != payload_geometry.transmitted_coefficients() {
                return Err(AkitaError::InvalidProof);
            }
            commitment
        }
        NextWitnessState::TerminalInnerState => return Err(AkitaError::InvalidProof),
    };
    let level_u32 = u32::try_from(level)
        .map_err(|_| AkitaError::InvalidSetup("fold level exceeds u32".into()))?;
    akita_transcript::public_native_fields_prover(
        grinding.state_mut(),
        akita_transcript::ProtocolSiteId {
            family: akita_transcript::SITE_FAMILY_FOLD_BINDING,
            level: level_u32,
            stage: 3,
            detail: u32::try_from(payload_geometry.transcript_ring_dimension())
                .map_err(|_| AkitaError::InvalidSetup("ring dimension exceeds u32".into()))?,
            ..akita_transcript::ProtocolSiteId::default()
        },
        witness_commitment.coeffs(),
    )
    .map_err(|_| AkitaError::InvalidProof)?;

    let opening_point = &sumcheck_challenges;
    let recursive_num_vars = level_params.recursive_opening_num_vars()?;
    let witness_source = RecursiveFoldSource::witness(Arc::clone(&witness));
    let logical_witness_source = RecursiveFoldSource::witness(logical_witness);
    let witness_polys = [&witness_source];
    let setup_slot = level_params
        .setup_prefix()
        .as_ref()
        .map(|id| {
            prefix_slots
                .get(&id.slot_id().expect("setup prefix group"))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("planned setup-prefix slot is missing".into())
                })
        })
        .transpose()?;
    let setup_source_storage = setup_slot.map(|slot| {
        RecursiveFoldSource::setup_prefix(Arc::clone(expanded), Arc::new(slot.clone()))
    });
    let setup_polys_storage = setup_source_storage.as_ref().map(|source| [source]);
    let block_claims = ProverOpeningData::new_recursive_suffix_fold(
        opening_point,
        recursive_num_vars,
        setup_prefix_opening,
        setup_slot,
        setup_polys_storage.as_ref().map(|polys| &polys[..]),
        opening,
        &witness_polys[..],
        (Commitment::new(witness_commitment), prover_state),
    )?;
    let witness_group_index = block_claims
        .opening_claims()
        .num_groups()
        .checked_sub(1)
        .ok_or(AkitaError::InvalidProof)?;
    let commitment_material = block_claims.prepare_commitment_relation_material(
        level_params,
        E::DEGREE,
        Some((
            witness_group_index,
            crate::types::PreparedCommitmentRelationMaterial {
                inner: inner_relation_material,
                compression: compression_material,
            },
        )),
    )?;
    let opening_method = level_params.opening_method();
    let needs_extension_reduction = opening_method.requires_extension_opening_reduction(E::DEGREE);
    let logical_polys = setup_source_storage
        .as_ref()
        .into_iter()
        .chain(std::iter::once(&logical_witness_source))
        .collect::<Vec<_>>();
    let logical_groups = logical_polys
        .iter()
        .map(|poly| PreparedProverGroup::from_ref_vec(vec![*poly]))
        .collect::<Result<Vec<_>, _>>()?;
    if const { <E as ExtField<F>>::DEGREE == 1 } {
        prepare_single_field_fold_native::<F, E, _, _, O, TS, R, SP>(
            stack,
            block_claims,
            commitment_material,
            true,
            grinding,
            level_u32,
            level_params,
            BasisMode::Lagrange,
        )
    } else {
        prepare_extension_claim_fold_native::<F, E, _, _, O, TS, R, SP>(
            stack,
            needs_extension_reduction,
            block_claims,
            commitment_material,
            ExtensionOpeningSource::Logical(&logical_groups),
            true,
            grinding,
            level_u32,
            level_params,
            BasisMode::Lagrange,
        )
    }
}
