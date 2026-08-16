use super::*;
use crate::backend::{RecursiveFoldSource, RecursiveWitnessFlat};
use crate::compute::{
    ComputeBackendSetup, DigitRowsComputeBackend, LevelProveStacks, ProverComputeStack,
    RuntimeCoefficientPackingBackendFor, RuntimeCommitBackendFor, RuntimeOpeningProveBackendFor,
    RuntimeRingSwitchProveBackend, RuntimeTensorBackendFor, SuffixOpeningProveBackend,
    SuffixTensorProveBackend,
};
use akita_field::unreduced::ReduceTo;
use akita_field::AdditiveGroup;
use akita_types::AkitaCommitmentHint;
use std::sync::Arc;

/// Prover state carried between suffix fold levels.
pub struct SuffixProverState<F: FieldCore, E: FieldCore> {
    /// Current committed suffix witness representation.
    pub w: RecursiveWitnessFlat,
    /// Logical suffix witness when it differs from the committed representation.
    pub logical_w: Option<RecursiveWitnessFlat>,
    /// Transcript-bound public state for the current suffix witness.
    pub binding: NextWitnessState<F>,
    /// Persistent semantic A-ring rows for the current suffix commitment.
    pub hint: AkitaCommitmentHint<F>,
    /// Current digit basis, as `log2(b)`.
    pub log_basis: u32,
    /// Sumcheck challenges that become the next suffix opening point.
    pub sumcheck_challenges: Vec<E>,
    /// Claimed logical opening of `logical_w` at `sumcheck_challenges`.
    pub opening: E,
    /// Optional setup-prefix opening carried from the previous stage-3 proof.
    pub setup_prefix_opening: Option<(Vec<E>, E)>,
}

impl<F: FieldCore, E: FieldCore> SuffixProverState<F, E> {
    /// Logical witness represented by the carried opening claim.
    #[inline]
    pub fn logical_w(&self) -> &RecursiveWitnessFlat {
        self.logical_w.as_ref().unwrap_or(&self.w)
    }
}

/// Drive the recursive fold suffix (after the root) under config `Cfg`.
///
/// The selected planner `schedule` is authoritative: it determines the fold
/// count, per-level `CommittedGroupParams`, successor params, and the terminal direct
/// witness basis. Earlier suffix levels run intermediate folds; the last
/// suffix level runs the terminal fold which ships the cleartext
/// `terminal_response`.
///
/// # Errors
///
/// Returns an error if level proving fails or the required recursive suffix is
/// absent.
#[allow(clippy::too_many_arguments)]
pub fn prove_suffix<'stack, Cfg, T, C, O, TS, R>(
    expanded: &Arc<AkitaExpandedSetup<Cfg::Field>>,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field>,
    stacks: &'stack impl LevelProveStacks<
        'stack,
        Cfg::Field,
        Commit = C,
        Opening = O,
        Tensor = TS,
        RingSwitch = R,
    >,
    transcript: &mut T,
    starting_state: SuffixProverState<Cfg::Field, Cfg::ExtField>,
    schedule: &FoldSchedule,
) -> Result<RecursiveSuffixOutcome<Cfg::Field, Cfg::ExtField>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField
        + FromPrimitiveInt
        + 'static,
    <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field> + AdditiveGroup,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + FrobeniusExtField<Cfg::Field>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize
        + MulBaseUnreduced<Cfg::Field>,
    T: Transcript<Cfg::Field> + ProverTranscriptGrind<Cfg::Field>,
    C: RuntimeCommitBackendFor<Cfg::Field, RecursiveWitnessFlat>
        + ComputeBackendSetup<Cfg::Field>
        + 'stack,
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
    <C as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'stack,
    <O as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'stack,
    <TS as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'stack,
    <R as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'stack,
{
    schedule.validate_nonterminal_opening_execution(Cfg::EXT_DEGREE)?;
    let planned_num_levels = schedule.num_fold_levels();
    if planned_num_levels < 2 {
        return Err(AkitaError::InvalidSetup(
            "prove_suffix expects a non-empty recursive suffix".to_string(),
        ));
    }
    let mut intermediate_levels = Vec::new();
    let mut current_state = starting_state;
    let mut level = 1usize;

    for (recursive_index, step) in schedule.recursive_folds.iter().enumerate() {
        let level_params = &step.params.witness;
        let input_witness_len = step.input_witness_len;
        let successor = schedule.recursive_folds.get(recursive_index + 1);
        let (next_params, next_binding) = successor.map_or(
            (
                super::fold::FoldSuccessorParams::Terminal(&schedule.terminal.params.witness),
                akita_types::NextWitnessBindingPolicy::TerminalInnerState,
            ),
            |next| {
                (
                    super::fold::FoldSuccessorParams::Recursive(&next.params),
                    akita_types::NextWitnessBindingPolicy::OuterPayload,
                )
            },
        );
        let current_witness_len = current_state.w.live_coeff_len();
        if current_witness_len != input_witness_len {
            return Err(AkitaError::InvalidSetup(format!(
                "scheduled fold level {level} did not match runtime state: expected_witness_len={input_witness_len}, actual_witness_len={}",
                current_witness_len
            )));
        }
        let role_dims = level_params.role_dims();
        let prepared_fold = {
            let stack = stacks.prove_stack_at_level(level);
            prepare_suffix::<Cfg::Field, Cfg::ExtField, T, C, O, TS, R>(
                stack,
                expanded,
                prefix_slots,
                transcript,
                current_state,
                level,
                level_params,
            )
            .map_err(|err| {
                AkitaError::InvalidInput(format!(
                    "suffix prepare level {level} d_a={} failed: {err:?}",
                    role_dims.d_a()
                ))
            })?
        };
        let out = super::fold::prove_fold::<Cfg::Field, Cfg::ExtField, T, C, O, TS, R, Cfg>(
            expanded,
            prefix_slots,
            stacks.prove_stack_at_level(level),
            transcript,
            level,
            level_params,
            Some(next_params),
            Some(step.output_witness_len),
            Some(next_binding),
            prepared_fold,
        )
        .map_err(|err| {
            AkitaError::InvalidInput(format!(
                "suffix fold level {level} d_a={} failed: {err:?}",
                role_dims.d_a()
            ))
        })?;
        intermediate_levels.push(out.level_proof);
        current_state = out.next_state;
        level += 1;
    }
    let current_witness_len = current_state.w.live_coeff_len();
    if current_witness_len != schedule.terminal.input_witness_len {
        return Err(AkitaError::InvalidSetup(format!(
            "scheduled terminal fold did not match runtime state: expected_witness_len={}, actual_witness_len={}",
            schedule.terminal.input_witness_len,
            current_witness_len,
        )));
    }
    let terminal = prove_terminal_suffix::<Cfg::Field, Cfg::ExtField, T, C, O, TS, R>(
        stacks.prove_stack_at_level(level),
        transcript,
        current_state,
        &schedule.terminal.params,
    )?;

    Ok(RecursiveSuffixOutcome {
        recursive_folds: intermediate_levels,
        terminal,
        num_levels: planned_num_levels,
    })
}

#[allow(clippy::too_many_arguments)]
fn prove_terminal_suffix<F, E, T, C, O, TS, R>(
    stack: &ProverComputeStack<'_, F, C, O, TS, R>,
    transcript: &mut T,
    current_state: SuffixProverState<F, E>,
    scheduled: &TerminalFoldParams,
) -> Result<TerminalLevelProof<F, E>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + FromPrimitiveInt
        + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F> + AdditiveGroup,
    E: FpExtEncoding<F>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize
        + MulBaseUnreduced<F>,
    T: Transcript<F> + ProverTranscriptGrind<F>,
    O: SuffixOpeningProveBackend<F>
        + DigitRowsComputeBackend<F>
        + RuntimeOpeningProveBackendFor<F, RecursiveFoldSource<F>>
        + ComputeBackendSetup<F>,
    TS: SuffixTensorProveBackend<F, E>
        + RuntimeTensorBackendFor<F, RecursiveFoldSource<F>, E>
        + ComputeBackendSetup<F>,
    C: ComputeBackendSetup<F>,
    R: ComputeBackendSetup<F>,
{
    let SuffixProverState {
        w,
        logical_w,
        binding,
        hint,
        sumcheck_challenges,
        opening,
        setup_prefix_opening,
        ..
    } = current_state;
    if setup_prefix_opening.is_some() {
        return Err(AkitaError::InvalidSetup(
            "terminal fold cannot receive a setup-prefix opening".into(),
        ));
    }
    match binding {
        NextWitnessState::TerminalInnerState => {}
        NextWitnessState::OuterPayload(_) => return Err(AkitaError::InvalidProof),
    }
    let mut terminal_rows = hint.into_rows();
    if terminal_rows.len() != 1 {
        return Err(AkitaError::InvalidProof);
    }
    let t_state = terminal_rows.pop().ok_or(AkitaError::InvalidProof)?;
    transcript.absorb_and_record_bytes(
        ABSORB_COMMITMENT,
        &akita_types::raw_field_segment_bytes(&t_state)?,
    );

    let witness = Arc::new(w);
    let logical_witness = logical_w
        .map(Arc::new)
        .unwrap_or_else(|| Arc::clone(&witness));
    let witness_source = RecursiveFoldSource::witness(Arc::clone(&witness));
    let logical_source = RecursiveFoldSource::witness(logical_witness);
    let params = &scheduled.witness;
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
    let needs_reduction = E::EXT_DEGREE > 1;
    let (protocol_point, reduction, row_coefficients) = if needs_reduction {
        let eor_inputs = vec![ExtensionOpeningGroupInput {
            group: &logical_group,
            point: &sumcheck_challenges,
            ring_dimension: params.d_a(),
        }];
        let proved = prove_extension_opening_reduction::<F, E, T, _, TS>(
            stack.tensor().backend(),
            Some(stack.tensor().prepared()),
            &eor_inputs,
            transcript,
            "terminal",
        )?;
        (
            proved
                .protocol_points
                .into_iter()
                .next()
                .ok_or(AkitaError::InvalidProof)?,
            Some(proved.reduction),
            Some(proved.row_coefficients),
        )
    } else {
        (sumcheck_challenges, None, None)
    };
    for coordinate in &protocol_point {
        append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, coordinate);
    }
    let (e_folded, fold_output, extension_opening_reduction) = dispatch_for_field!(
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
                    params.num_positions_per_block,
                    params.num_live_blocks,
                    alpha_bits,
                )?;
            let (trace, _) = compute_trace_target::<F, E, T, D>(
                &reduction,
                &folded_rings,
                std::slice::from_ref(&prepared_point),
                &protocol_point,
                alpha_bits,
                BasisMode::Lagrange,
                &opening_batch,
                row_coefficients,
                transcript,
            )?;
            // The EOR proof binds the carried extension-field opening to its
            // reduced final claim. `compute_trace_target` separately binds that
            // final claim to the directly evaluated base-field witness. Only a
            // degree-one opening can therefore be compared here verbatim.
            if reduction.is_none() && trace.trace_eval_target != opening {
                return Err(AkitaError::InvalidInput(
                    "terminal folded opening does not match the carried claim".into(),
                ));
            }
            let folded = folded_blocks
                .into_iter()
                .next()
                .ok_or(AkitaError::InvalidProof)?;
            let e_folded = RingVec::from_ring_elems(&folded);
            transcript.absorb_and_record_bytes(
                ABSORB_TERMINAL_E_HAT,
                &akita_types::raw_field_segment_bytes(&e_folded)?,
            );
            let output = crate::protocol::fold_grind::sample_terminal_fold_response(
                stack.opening().backend(),
                Some(stack.opening().prepared()),
                transcript,
                params,
                &scheduled.sparse_challenge_config,
                &witness_source,
                &scheduled.response_shape,
            )?;
            Ok::<_, AkitaError>((
                e_folded,
                output,
                reduction.as_ref().map(|value| value.proof.clone()),
            ))
        }
    )?;
    let terminal_response = akita_types::build_terminal_response(
        params,
        &scheduled.sparse_challenge_config,
        &scheduled.response_shape,
        &e_folded,
        t_state,
        fold_output.witness.centered_coeffs_flat(),
    )?;
    let transcript_parts = terminal_response.terminal_transcript_parts()?;
    transcript.absorb_and_record_bytes(ABSORB_TERMINAL_W_REMAINDER, &transcript_parts.response);
    Ok(TerminalLevelProof {
        extension_opening_reduction,
        fold_grind_nonce: fold_output.nonce,
        terminal_response,
    })
}
/// Prove one recursive fold level using already-selected current and next
/// level parameters.
///
/// The caller owns schedule/config selection and passes the next-level
/// commitment params. This function owns recursive opening-point reduction,
/// witness folding, public recursive transcript absorbs, recursive
/// ring-relation construction, and the folded-level prover mechanics.
///
/// # Errors
///
/// Returns an error if the recursive opening point has the wrong dimension,
/// witness folding or ring-relation construction fails, or the folded
/// prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(in crate::protocol::core) fn prepare_suffix<F, E, T, C, O, TS, R>(
    stack: &ProverComputeStack<'_, F, C, O, TS, R>,
    expanded: &Arc<AkitaExpandedSetup<F>>,
    prefix_slots: &SetupPrefixProverRegistry<F>,
    transcript: &mut T,
    current_state: SuffixProverState<F, E>,
    _level: usize,
    level_params: &CommittedGroupParams,
) -> Result<PreparedFold<F, E>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField
        + FromPrimitiveInt
        + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F> + AdditiveGroup,
    E: FpExtEncoding<F>
        + FrobeniusExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize
        + MulBaseUnreduced<F>,
    T: Transcript<F> + ProverTranscriptGrind<F>,
    TS: RuntimeTensorBackendFor<F, RecursiveWitnessFlat, E>
        + RuntimeTensorBackendFor<F, RecursiveFoldSource<F>, E>,
    O: DigitRowsComputeBackend<F>
        + RuntimeOpeningProveBackendFor<F, RecursiveWitnessFlat>
        + RuntimeOpeningProveBackendFor<F, RecursiveFoldSource<F>>
        + RuntimeCoefficientPackingBackendFor<F, RecursiveFoldSource<F>, E>,
    C: ComputeBackendSetup<F>,
    R: DigitRowsComputeBackend<F> + RuntimeRingSwitchProveBackend<F>,
{
    let SuffixProverState {
        w,
        logical_w: optional_logical_w,
        binding,
        hint,
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
                return Err(AkitaError::InvalidInput(format!(
                    "suffix commitment length {} does not match expected coefficient count {}",
                    commitment.coeffs().len(),
                    payload_geometry.transmitted_coefficients(),
                )));
            }
            commitment.append_flat_to_transcript::<T>(
                ABSORB_COMMITMENT,
                payload_geometry.transcript_ring_dimension(),
                transcript,
            )?;
            commitment
        }
        NextWitnessState::TerminalInnerState => return Err(AkitaError::InvalidProof),
    };
    let suffix_hint = hint;
    let opening_point = &sumcheck_challenges;

    let recursive_num_vars = level_params.recursive_opening_num_vars()?;
    let witness_source = RecursiveFoldSource::witness(Arc::clone(&witness));
    let logical_witness_source = RecursiveFoldSource::witness(logical_witness);
    let witness_polys = [&witness_source];
    let setup_slot = level_params
        .setup_prefix
        .as_ref()
        .map(|id| {
            prefix_slots.get(&id.slot_id()).ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "planned setup-prefix slot is missing from prover setup".into(),
                )
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
        (Commitment::new(witness_commitment), suffix_hint),
    )?;
    let opening_batch = block_claims.opening_layout()?;
    let opening_method = super::fold::uniform_opening_method(level_params, &opening_batch)?;
    let needs_extension_reduction =
        super::fold::extension_opening_reduction_enabled(opening_method, E::EXT_DEGREE > 1);
    let logical_polys = setup_source_storage
        .as_ref()
        .into_iter()
        .chain(std::iter::once(&logical_witness_source))
        .collect::<Vec<_>>();
    let logical_groups = logical_polys
        .iter()
        .map(|poly| PreparedProverGroup::from_ref_vec(vec![*poly]))
        .collect::<Result<Vec<_>, _>>()?;
    if const { <E as ExtField<F>>::EXT_DEGREE == 1 } {
        prepare_single_field_fold::<F, E, T, _, _, C, O, TS, R>(
            stack,
            block_claims,
            true,
            transcript,
            || Ok(()),
            level_params,
            BasisMode::Lagrange,
        )
    } else {
        prepare_extension_claim_fold::<F, E, T, _, _, C, O, TS, R>(
            stack,
            needs_extension_reduction,
            block_claims,
            ExtensionOpeningSource::Logical(&logical_groups),
            true,
            transcript,
            || Ok(()),
            level_params,
            BasisMode::Lagrange,
        )
    }
    .map_err(|err| AkitaError::InvalidInput(format!("suffix fold preparation failed: {err:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::core::fold_kernels::prepare_evaluation_trace_claim;
    use akita_field::Fp32;
    use akita_transcript::AkitaTranscript;

    type TestF = Fp32<251>;

    #[test]
    fn non_zk_eor_mismatch_is_rejected() {
        let openings = [TestF::zero()];
        let reduction = Some(ExtensionOpeningReduction {
            proof: ExtensionOpeningReductionProof {
                partials: Vec::new(),
                sumcheck: SumcheckProof {
                    round_polys: Vec::new(),
                },
            },
            final_claim: TestF::one(),
            final_factors: vec![TestF::one()],
        });

        let opening_batch = OpeningClaimsLayout::new(0, 1).expect("singleton opening batch");
        let mut transcript = AkitaTranscript::<TestF>::new(b"test/suffix-shared-trace-target");
        let err = match prepare_evaluation_trace_claim::<TestF, TestF, _>(
            &reduction,
            &openings,
            &opening_batch,
            Some(vec![TestF::one()]),
            &mut transcript,
        ) {
            Ok(_) => panic!("non-zk EOR mismatch should reject"),
            Err(err) => err,
        };

        assert!(
            matches!(err, AkitaError::InvalidProof),
            "unexpected error: {err:?}"
        );
    }
}
