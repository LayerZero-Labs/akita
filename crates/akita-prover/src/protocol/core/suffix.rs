use super::*;
use crate::backend::RecursiveWitnessFlat;
use crate::compute::{
    CommitmentComputeBackend, ComputeBackendSetup, DigitRowsComputeBackend, LevelProveStacks,
    OpeningProveBackendFor, ProverComputeStack, RingSwitchProveBackend,
    SuffixDispatchOpeningProveBackendFor, SuffixDispatchTensorProveBackendFor,
    SuffixRingSwitchProveBackend, TensorBackendFor,
};
use crate::RootTensorProjectionPoly;
use akita_field::unreduced::ReduceTo;
use akita_field::AdditiveGroup;
use akita_types::schedule_terminal_direct_witness_shape;
use akita_types::terminal_golomb_grind_tail_t_vectors;

/// Prover state carried between suffix fold levels.
pub struct SuffixProverState<F: FieldCore, L: FieldCore> {
    /// Current committed suffix witness representation.
    pub w: RecursiveWitnessFlat,
    /// Logical suffix witness when it differs from the committed representation.
    pub logical_w: Option<RecursiveWitnessFlat>,
    /// Current suffix witness commitment.
    pub commitment: FlatRingVec<F>,
    /// D-erased suffix commitment hint cache.
    pub hint: RecursiveCommitmentHintCache<F>,
    /// Current digit basis, as `log2(b)`.
    pub log_basis: u32,
    /// Sumcheck challenges that become the next suffix opening point.
    pub sumcheck_challenges: Vec<L>,
    /// Claimed logical opening of `logical_w` at `sumcheck_challenges`.
    pub opening: L,
}

impl<F: FieldCore, L: FieldCore> SuffixProverState<F, L> {
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
pub fn prove_suffix<'stack, Cfg, T, C, O, TS, R, const D: usize>(
    expanded: &Arc<AkitaExpandedSetup<Cfg::Field>>,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, D>,
    stacks: &'stack impl LevelProveStacks<
        'stack,
        Cfg::Field,
        D,
        Commit = C,
        Opening = O,
        Tensor = TS,
        RingSwitch = R,
    >,
    transcript: &mut T,
    starting_state: SuffixProverState<Cfg::Field, Cfg::ExtField>,
    schedule: &Schedule,
    setup_contribution_mode: SetupContributionMode,
    a_ones_table: &FoldAOnesTable<Cfg::Field>,
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
    C: CommitmentComputeBackend<Cfg::Field> + ComputeBackendSetup<Cfg::Field> + 'stack,
    O: SuffixDispatchOpeningProveBackendFor<Cfg::Field, D>
        + DigitRowsComputeBackend<Cfg::Field>
        + ComputeBackendSetup<Cfg::Field>
        + 'stack,
    TS: SuffixDispatchTensorProveBackendFor<Cfg::Field, Cfg::ExtField, D>
        + ComputeBackendSetup<Cfg::Field>
        + 'stack,
    R: SuffixRingSwitchProveBackend<Cfg::Field>
        + RingSwitchProveBackend<Cfg::Field, D>
        + DigitRowsComputeBackend<Cfg::Field>
        + ComputeBackendSetup<Cfg::Field>
        + 'stack,
    <C as ComputeBackendSetup<Cfg::Field>>::PreparedSetup<D>: 'stack,
    <O as ComputeBackendSetup<Cfg::Field>>::PreparedSetup<D>: 'stack,
    <TS as ComputeBackendSetup<Cfg::Field>>::PreparedSetup<D>: 'stack,
    <R as ComputeBackendSetup<Cfg::Field>>::PreparedSetup<D>: 'stack,
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
            MRowLayout::WithoutDBlock,
            Some(terminal_direct_witness_shape),
        )?
    };
    let terminal_result = loop {
        let scheduled = schedule.get_execution_schedule(level)?;
        scheduled.validate_current_w_len(current_state.w.len())?;
        let level_params = &scheduled.params;
        let level_d = level_params.ring_dimension;
        let is_terminal_level = scheduled.is_terminal;
        let m_row_layout = if is_terminal_level {
            MRowLayout::WithoutDBlock
        } else {
            MRowLayout::WithDBlock
        };
        let tail_t_vectors = if is_terminal_level {
            terminal_tail_t_vectors
        } else {
            None
        };
        let out = if level_d == D {
            let stack = stacks.prove_stack_at_level(level);
            let prepared_fold = prepare_suffix::<Cfg::Field, Cfg::ExtField, T, C, O, TS, R, D>(
                stack,
                transcript,
                current_state,
                level,
                level_params,
                m_row_layout,
                tail_t_vectors,
                a_ones_table,
            )
            .map_err(|err| {
                AkitaError::InvalidInput(format!("suffix prepare level {level} failed: {err:?}"))
            })?;
            prove_fold::<Cfg::Field, Cfg::ExtField, T, C, O, TS, R, Cfg, D>(
                expanded,
                prefix_slots,
                stack,
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
                AkitaError::InvalidInput(format!("suffix prove_fold level {level} failed: {err:?}"))
            })
        } else {
            dispatch_ring_dim_result!(level_d, |D_LEVEL| {
                let tier_stack = stacks.prove_stack_at_level(level);
                let expanded_cloned = Arc::clone(expanded);
                let commit_backend = tier_stack.commit().backend();
                let opening_backend = tier_stack.opening().backend();
                let tensor_backend = tier_stack.tensor().backend();
                let ring_backend = tier_stack.ring_switch().backend();
                let commit_prepared =
                    commit_backend.prepare_expanded::<D_LEVEL>(Arc::clone(&expanded_cloned))?;
                let opening_prepared =
                    opening_backend.prepare_expanded::<D_LEVEL>(Arc::clone(&expanded_cloned))?;
                let tensor_prepared =
                    tensor_backend.prepare_expanded::<D_LEVEL>(Arc::clone(&expanded_cloned))?;
                let ring_prepared = ring_backend.prepare_expanded::<D_LEVEL>(expanded_cloned)?;
                let level_stack = ProverComputeStack::<Cfg::Field, D_LEVEL, C, O, TS, R>::new(
                    (commit_backend, &commit_prepared),
                    (opening_backend, &opening_prepared),
                    (tensor_backend, &tensor_prepared),
                    (ring_backend, &ring_prepared),
                    expanded.as_ref(),
                )?;
                let level_prefix_slots = SetupPrefixProverRegistry::new();
                let prepared_fold =
                    prepare_suffix::<Cfg::Field, Cfg::ExtField, T, C, O, TS, R, { D_LEVEL }>(
                        &level_stack,
                        transcript,
                        current_state,
                        level,
                        level_params,
                        m_row_layout,
                        tail_t_vectors,
                        a_ones_table,
                    )
                    .map_err(|err| {
                        AkitaError::InvalidInput(format!(
                            "suffix prepare level {level} D{D_LEVEL} failed: {err:?}"
                        ))
                    })?;
                prove_fold::<Cfg::Field, Cfg::ExtField, T, C, O, TS, R, Cfg, { D_LEVEL }>(
                    expanded,
                    &level_prefix_slots,
                    &level_stack,
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
                        "suffix prove_fold level {level} D{D_LEVEL} failed: {err:?}"
                    ))
                })
            })
        }?;
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
fn prepare_suffix<F, L, T, C, O, TS, R, const D: usize>(
    stack: &ProverComputeStack<'_, F, D, C, O, TS, R>,
    transcript: &mut T,
    current_state: SuffixProverState<F, L>,
    _level: usize,
    level_params: &LevelParams,
    m_row_layout: MRowLayout,
    terminal_tail_t_vectors: Option<usize>,
    a_ones_table: &FoldAOnesTable<F>,
) -> Result<PreparedFold<F, L, D>, AkitaError>
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
    L: FpExtEncoding<F>
        + FrobeniusExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize
        + MulBaseUnreduced<F>,
    T: Transcript<F> + ProverTranscriptGrind<F>,
    TS: TensorBackendFor<F, RecursiveWitnessFlat, L, D>,
    O: DigitRowsComputeBackend<F>
        + OpeningProveBackendFor<F, RecursiveWitnessFlat, D>
        + OpeningProveBackendFor<F, RootTensorProjectionPoly<F, D>, D>,
    C: ComputeBackendSetup<F>,
    R: DigitRowsComputeBackend<F>,
{
    let SuffixProverState {
        w,
        logical_w: optional_logical_w,
        commitment,
        hint,
        sumcheck_challenges,
        ..
    } = current_state;
    let logical_w = optional_logical_w.as_ref().unwrap_or(&w);
    let typed_hint = hint.to_typed::<D>()?;
    let opening_point = &sumcheck_challenges;

    commitment.append_as_ring_commitment::<T, D>(ABSORB_COMMITMENT, transcript)?;

    let alpha = level_params.ring_dimension.trailing_zeros() as usize;
    let needs_extension_reduction = <L as ExtField<F>>::EXT_DEGREE != 1;
    let logical_polys = [logical_w];
    let fold_polys = [&w];
    let eor_opening_batch =
        OpeningClaims::with_padded_point(opening_point, opening_point.len(), 1)?;
    let recursive_num_vars = level_params.recursive_opening_num_vars()?;
    let commitment_u = commitment.as_ring_slice::<D>()?;
    let suffix_commitment = (
        RingCommitment {
            u: commitment_u.to_vec(),
        },
        typed_hint,
    );
    let fold_claims = ProverOpeningData::new_suffix(
        opening_point,
        recursive_num_vars,
        &fold_polys,
        suffix_commitment,
    )?;
    prepare_fold_inner::<F, L, T, _, _, C, O, TS, R, D>(
        stack,
        needs_extension_reduction,
        fold_claims,
        &logical_polys[..],
        &eor_opening_batch,
        true,
        transcript,
        opening_point.to_vec(),
        || Ok(()),
        level_params,
        alpha,
        BasisMode::Lagrange,
        BlockOrder::ColumnMajor,
        m_row_layout,
        terminal_tail_t_vectors,
        a_ones_table,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::core::fold::compute_trace_target;
    use akita_field::Fp32;
    use akita_transcript::AkitaTranscript;
    use akita_types::RingOpeningPoint;

    type TestF = Fp32<251>;
    const D: usize = 4;

    #[test]
    fn non_zk_eor_mismatch_is_rejected() {
        let prepared_point: PreparedOpeningPoint<TestF, TestF, D> = PreparedOpeningPoint {
            padded_point: Vec::new(),
            ring_opening_point: RingOpeningPoint {
                a: vec![TestF::one()],
                b: vec![TestF::one()],
            },
            ring_multiplier_point: RingMultiplierOpeningPoint::from_base(&RingOpeningPoint {
                a: vec![TestF::one()],
                b: vec![TestF::one()],
            }),
            packed_inner_point: CyclotomicRing::<TestF, D>::zero(),
        };
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
            &prepared_point,
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

        assert!(matches!(err, AkitaError::InvalidProof));
    }
}
