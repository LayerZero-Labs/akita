use super::*;
use crate::backend::{RecursiveFoldSource, RecursiveWitnessFlat};
use crate::compute::{
    ComputeBackendSetup, DigitRowsComputeBackend, LevelProveStacks, ProverComputeStack,
    RuntimeOpeningProveBackendFor, RuntimeRingSwitchProveBackend, RuntimeTensorBackendFor,
    SuffixOpeningProveBackend, SuffixTensorProveBackend,
};
use crate::RootTensorProjectionPoly;
use akita_field::unreduced::ReduceTo;
use akita_field::AdditiveGroup;
use akita_types::schedule_terminal_direct_witness_shape;
use akita_types::terminal_golomb_grind_tail_t_vectors;
use std::sync::Arc;

/// Prover state carried between suffix fold levels.
pub struct SuffixProverState<F: FieldCore, E: FieldCore> {
    /// Current committed suffix witness representation.
    pub w: RecursiveWitnessFlat,
    /// Logical suffix witness when it differs from the committed representation.
    pub logical_w: Option<RecursiveWitnessFlat>,
    /// Current suffix witness commitment.
    pub commitment: RingVec<F>,
    /// D-erased suffix commitment hint cache.
    pub hint: RecursiveCommitmentHintCache<F>,
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
/// count, per-level `LevelParams`, successor params, and the terminal direct
/// witness basis. Earlier suffix levels run intermediate folds; the last
/// suffix level runs the terminal fold which ships the cleartext
/// `final_witness`.
///
/// # Errors
///
/// Returns an error if level proving fails, or an invalid-setup error when the
/// schedule's recursive suffix is empty (root-terminal proofs do not run this
/// helper).
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
    schedule: &Schedule,
    setup_contribution_mode: SetupContributionMode,
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
    C: crate::compute::CommitmentComputeBackend<Cfg::Field>
        + ComputeBackendSetup<Cfg::Field>
        + 'stack,
    O: SuffixOpeningProveBackend<Cfg::Field>
        + RuntimeOpeningProveBackendFor<Cfg::Field, RecursiveFoldSource<Cfg::Field>>
        + DigitRowsComputeBackend<Cfg::Field>
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
    let planned_num_levels = schedule_num_fold_levels(schedule);
    if planned_num_levels < 2 {
        return Err(AkitaError::InvalidSetup(
            "prove_suffix expects a non-empty recursive suffix".to_string(),
        ));
    }
    let mut intermediate_levels = Vec::new();
    let mut current_state = starting_state;
    let mut level = 1usize;

    let terminal_direct_witness_shape = schedule_terminal_direct_witness_shape(schedule)?;
    let terminal_tail_t_vectors = {
        let terminal_level = planned_num_levels - 1;
        let terminal_scheduled = schedule.get_execution_schedule(terminal_level)?;
        terminal_golomb_grind_tail_t_vectors(
            &terminal_scheduled.params,
            RelationMatrixRowLayout::WithoutDBlock,
            Some(terminal_direct_witness_shape),
        )?
    };
    let terminal_result = loop {
        let scheduled = schedule.get_execution_schedule(level)?;
        scheduled.validate_current_w_len(current_state.w.len())?;
        let level_params = &scheduled.params;
        let role_dims = level_params.role_dims();
        let is_terminal_level = scheduled.is_terminal;
        let relation_matrix_row_layout = if is_terminal_level {
            RelationMatrixRowLayout::WithoutDBlock
        } else {
            RelationMatrixRowLayout::WithDBlock
        };
        let tail_t_vectors = if is_terminal_level {
            terminal_tail_t_vectors
        } else {
            None
        };
        let prepared_fold = {
            let stack = stacks.prove_stack_at_level(level);
            stack.ensure_fold_level_role_ntt(expanded.as_ref(), role_dims)?;
            prepare_suffix::<Cfg::Field, Cfg::ExtField, T, C, O, TS, R>(
                stack,
                expanded,
                prefix_slots,
                transcript,
                current_state,
                level,
                level_params,
                relation_matrix_row_layout,
                tail_t_vectors,
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
            &scheduled,
            prepared_fold,
            setup_contribution_mode,
            is_terminal_level,
            if is_terminal_level {
                Some(terminal_direct_witness_shape)
            } else {
                None
            },
        )
        .map_err(|err| {
            AkitaError::InvalidInput(format!(
                "suffix fold level {level} d_a={} failed: {err:?}",
                role_dims.d_a()
            ))
        })?;
        if is_terminal_level {
            break out.get_terminal()?;
        }

        let out = out.get_intermediate()?;
        intermediate_levels.push(out.level_proof);
        current_state = out.next_state;
        level += 1;
    };
    let terminal = terminal_result;

    let mut steps = intermediate_levels;
    let final_w_len = terminal.final_witness().num_elems();
    steps.push(AkitaLevelProof::Terminal {
        extension_opening_reduction: terminal.extension_opening_reduction,
        fold_grind_nonce: terminal.fold_grind_nonce,
        stage2: terminal.stage2,
        final_w_len,
    });

    Ok(RecursiveSuffixOutcome {
        steps,
        num_levels: planned_num_levels,
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
    level_params: &LevelParams,
    relation_matrix_row_layout: RelationMatrixRowLayout,
    terminal_tail_t_vectors: Option<usize>,
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
        + RuntimeTensorBackendFor<F, RecursiveFoldSource<F>, E>
        + RuntimeTensorBackendFor<F, RootTensorProjectionPoly<F>, E>,
    O: DigitRowsComputeBackend<F>
        + RuntimeOpeningProveBackendFor<F, RecursiveWitnessFlat>
        + RuntimeOpeningProveBackendFor<F, RecursiveFoldSource<F>>
        + RuntimeOpeningProveBackendFor<F, RootTensorProjectionPoly<F>>,
    C: ComputeBackendSetup<F>,
    R: DigitRowsComputeBackend<F>,
{
    let SuffixProverState {
        w,
        logical_w: optional_logical_w,
        commitment,
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
    let role_dims = level_params.role_dims();
    let commit_d = role_dims.d_b();
    if !commitment.can_decode_vec(commit_d) {
        return Err(AkitaError::InvalidInput(format!(
            "suffix commitment length {} is not decodable at B-role dimension {}",
            commitment.coeffs().len(),
            commit_d,
        )));
    }
    // D-free suffix hint: the cache carries the flat `AkitaCommitmentHint<F>`
    // directly (Slice A re-homed the recomposed rows), so there is no typed
    // reconstruction here (the former `hint.to_typed::<D>()` bridge is gone).
    let suffix_hint = hint.into_hint();
    let opening_point = &sumcheck_challenges;

    // §6 invariant (H5 byte-parity) — absorb the suffix commitment through the
    // D-free flat coefficient encoder keyed on the level's B-role dimension.
    // This is byte-identical to the verifier's
    // `current_state.commitment.append_flat_to_transcript(...)` (S7
    // `prepare_fold_data`) and to the former typed `append_as_ring_commitment`
    // path (S2 byte-identity test). The same coefficient order, same
    // `ABSORB_COMMITMENT` label.
    commitment.append_flat_to_transcript::<T>(ABSORB_COMMITMENT, commit_d, transcript)?;

    let alpha = role_dims.d_a().trailing_zeros() as usize;
    let needs_extension_reduction =
        <E as ExtField<F>>::EXT_DEGREE != 1 && level_params.setup_prefix.is_none();
    let recursive_num_vars = level_params.recursive_opening_num_vars()?;
    let witness_source = RecursiveFoldSource::witness(Arc::clone(&witness));
    let logical_witness_source = RecursiveFoldSource::witness(logical_witness);
    let witness_polys = [&witness_source];
    let setup_slot = level_params
        .setup_prefix
        .as_ref()
        .map(|id| {
            prefix_slots.get(id).ok_or_else(|| {
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
    let (block_claims, eor_opening_batch, protocol_point) =
        ProverOpeningData::new_recursive_suffix_fold(
            opening_point,
            recursive_num_vars,
            setup_prefix_opening,
            setup_slot,
            setup_polys_storage.as_ref().map(|polys| &polys[..]),
            opening,
            &witness_polys[..],
            (Commitment::new(commitment), suffix_hint),
        )?;
    let logical_polys = setup_source_storage
        .as_ref()
        .into_iter()
        .chain(std::iter::once(&logical_witness_source))
        .collect::<Vec<_>>();
    prepare_fold_inner::<F, E, T, _, _, C, O, TS, R>(
        stack,
        needs_extension_reduction,
        block_claims,
        &logical_polys,
        &eor_opening_batch,
        true,
        transcript,
        protocol_point,
        || Ok(()),
        level_params,
        alpha,
        BasisMode::Lagrange,
        relation_matrix_row_layout,
        terminal_tail_t_vectors,
    )
    .map_err(|err| AkitaError::InvalidInput(format!("suffix fold preparation failed: {err:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::core::fold_kernels::compute_trace_target;
    use akita_field::Fp32;
    use akita_transcript::AkitaTranscript;
    use akita_types::RingOpeningPoint;

    type TestF = Fp32<251>;
    const D: usize = 4;

    #[test]
    fn non_zk_eor_mismatch_is_rejected() {
        let prepared_point = PreparedOpeningPoint::from_parts::<D>(
            Vec::new(),
            RingOpeningPoint {
                position_weights: vec![TestF::one()],
                block_weights: vec![TestF::one()],
            },
            RingMultiplierOpeningPoint::from_base(&RingOpeningPoint {
                position_weights: vec![TestF::one()],
                block_weights: vec![TestF::one()],
            }),
            CyclotomicRing::<TestF, D>::zero(),
        );
        let folded_rings = [CyclotomicRing::<TestF, D>::zero()];
        let reduction = Some(ExtensionOpeningReduction {
            proof: ExtensionOpeningReductionProof {
                partials: Vec::new(),
                sumcheck: SumcheckProof {
                    round_polys: Vec::new(),
                },
            },
            final_claim: TestF::one(),
            final_factor: TestF::one(),
        });

        let opening_batch = OpeningClaimsLayout::new(0, 1).expect("singleton opening batch");
        let mut transcript = AkitaTranscript::<TestF>::new(b"test/suffix-shared-trace-target");
        let err = match compute_trace_target::<TestF, TestF, _, D>(
            &reduction,
            &folded_rings,
            std::slice::from_ref(&prepared_point),
            &[],
            0,
            BasisMode::Lagrange,
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
