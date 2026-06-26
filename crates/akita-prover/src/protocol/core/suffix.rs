use super::*;
use akita_field::unreduced::ReduceTo;
use akita_field::AdditiveGroup;
use crate::RootTensorProjectionPoly;
use crate::backend::{RecursiveCommitmentHintCache, RecursiveWitnessFlat};
use crate::compute::{
    CommitmentComputeBackend, ComputeBackendSetup, DigitRowsComputeBackend, LevelProveStacks,
    OpeningProveBackendFor, ProverComputeStack, RingSwitchProveBackend, RootPolyShape,
    SuffixOpeningProveBackend, SuffixRingSwitchProveBackend, SuffixTensorProveBackend,
    TensorBackendFor,
};
use akita_types::{
    padded_scalar_batch_num_vars, terminal_golomb_grind_tail_t_vectors,
    validate_scalar_point_matches_poly_arity, AkitaCommitmentHint,
    OpeningGroupShape, OpeningPoints, PointVariableSelection,
    schedule_terminal_direct_witness_shape,
};

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

/// Single-claim suffix fold input: flat commitment rows plus one typed hint.
///
/// Suffix levels store commitment as [`FlatRingVec`] in [`SuffixProverState`].
/// This type threads that representation into fold preparation without placing
/// rows in [`crate::ProverOpeningBatch`].
#[derive(Debug, Clone)]
pub(in crate::protocol::core) struct SuffixFoldClaims<'a, PointF: Clone, P, F: FieldCore, const D: usize>
{
    pub(in crate::protocol::core) point: OpeningPoints<'a, PointF>,
    pub(in crate::protocol::core) point_vars: PointVariableSelection,
    pub(in crate::protocol::core) polynomials: &'a [&'a P],
    pub(in crate::protocol::core) commitment: FlatRingVec<F>,
    pub(in crate::protocol::core) hint: AkitaCommitmentHint<F, D>,
}

impl<'a, PointF: Clone, P, F: FieldCore, const D: usize>
    SuffixFoldClaims<'a, PointF, P, F, D>
{
    /// Build the single-claim batch used by recursive suffix fold levels.
    pub(in crate::protocol::core) fn new(
        opening_point: &[PointF],
        recursive_num_vars: usize,
        polynomials: &'a [&'a P],
        commitment: FlatRingVec<F>,
        hint: RecursiveCommitmentHintCache<F>,
    ) -> Result<Self, AkitaError>
    where
        PointF: FieldCore,
    {
        let typed_hint = hint.to_typed::<D>()?;
        let opening_batch = OpeningBatchShape::new(recursive_num_vars, 1)?;
        let point_vars = opening_batch
            .groups()
            .first()
            .ok_or_else(|| {
                AkitaError::InvalidInput("recursive opening batch requires one group".to_string())
            })?
            .point_vars
            .clone();
        let mut padded_point = opening_point.to_vec();
        padded_point.resize(recursive_num_vars, PointF::zero());
        Ok(Self {
            point: padded_point.into(),
            point_vars,
            polynomials,
            commitment,
            hint: typed_hint,
        })
    }

    pub(in crate::protocol::core) fn point(&self) -> &[PointF] {
        self.point.as_ref()
    }

    pub(in crate::protocol::core) fn flat_polys(&self) -> Vec<&'a P> {
        self.polynomials.to_vec()
    }

    pub(in crate::protocol::core) fn to_opening_shape<PolyF>(&self) -> Result<OpeningBatchShape, AkitaError>
    where
        PolyF: FieldCore,
        P: RootPolyShape<PolyF, D>,
    {
        let padded_num_vars = padded_scalar_batch_num_vars(
            self.polynomials.iter().map(|poly| poly.num_vars()),
        )?;
        validate_scalar_point_matches_poly_arity(self.point().len(), padded_num_vars)?;
        OpeningBatchShape::from_groups(
            padded_num_vars,
            vec![OpeningGroupShape {
                point_vars: self.point_vars.clone(),
                num_polynomials: self.polynomials.len(),
            }],
        )
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
    prefix_slots: &SetupPrefixRegistry<Cfg::Field>,
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
    C: CommitmentComputeBackend<Cfg::Field> + ComputeBackendSetup<Cfg::Field> + 'stack,
    O: SuffixOpeningProveBackend<Cfg::Field>
        + DigitRowsComputeBackend<Cfg::Field>
        + ComputeBackendSetup<Cfg::Field>
        + 'stack,
    TS: SuffixTensorProveBackend<Cfg::Field, Cfg::ExtField>
        + ComputeBackendSetup<Cfg::Field>
        + 'stack,
    R: SuffixRingSwitchProveBackend<Cfg::Field>
        + RingSwitchProveBackend<Cfg::Field, D>
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
        let out = {
            let stack = stacks.prove_stack_at_level(level);
            stack.ensure_fold_level_envelope_ntt(expanded.as_ref(), level_d)?;
            super::fold::prove_suffix_fold_at_ring_d::<Cfg, T, C, O, TS, R>(
                expanded,
                prefix_slots,
                stack,
                transcript,
                level,
                level_d,
                current_state,
                level_params,
                &scheduled,
                m_row_layout,
                tail_t_vectors,
                setup_contribution_mode,
                is_terminal_level,
                if is_terminal_level {
                    Some(terminal_direct_witness_shape)
                } else {
                    None
                },
            )
            .map_err(|err| {
                AkitaError::InvalidInput(format!("suffix fold level {level} D{level_d} failed: {err:?}"))
            })?
        };
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
pub(in crate::protocol::core) fn prepare_suffix<F, L, T, C, O, TS, R, const D: usize>(
    stack: &ProverComputeStack<'_, F, C, O, TS, R>,
    transcript: &mut T,
    current_state: SuffixProverState<F, L>,
    _level: usize,
    level_params: &LevelParams,
    m_row_layout: MRowLayout,
    terminal_tail_t_vectors: Option<usize>,
) -> Result<PreparedFold<F, L>, AkitaError>
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
    let opening_point = &sumcheck_challenges;

    commitment.append_as_ring_commitment::<T, D>(ABSORB_COMMITMENT, transcript)?;

    let alpha = level_params.ring_dimension.trailing_zeros() as usize;
    let needs_extension_reduction = <L as ExtField<F>>::EXT_DEGREE != 1;
    let logical_polys = [logical_w];
    let fold_polys = [&w];
    let eor_opening_batch =
        VerifierOpeningBatch::with_padded_point(opening_point, opening_point.len(), 1)?;
    let recursive_num_vars = level_params.recursive_opening_num_vars()?;
    let fold_claims = SuffixFoldClaims::new(
        opening_point,
        recursive_num_vars,
        &fold_polys,
        commitment,
        hint,
    )?;
    super::fold::prepare_suffix_fold_inner::<F, L, T, _, C, O, TS, R, D>(
        stack,
        needs_extension_reduction,
        fold_claims,
        &logical_polys[..],
        &eor_opening_batch,
        transcript,
        opening_point.to_vec(),
        || Ok(()),
        level_params,
        alpha,
        BasisMode::Lagrange,
        BlockOrder::ColumnMajor,
        m_row_layout,
        terminal_tail_t_vectors,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::core::fold::compute_trace_target;
    use akita_field::Fp32;
    use akita_transcript::AkitaTranscript;
    use akita_types::{FlatDigitBlocks, RingOpeningPoint};

    type TestF = Fp32<251>;
    const D: usize = 4;

    #[test]
    fn suffix_fold_claims_keeps_flat_commitment_without_opening_batch() {
        let commitment = FlatRingVec::from_coeffs(vec![TestF::one(); D]);
        let polys: &[&CyclotomicRing<TestF, D>] = &[];
        let hint = AkitaCommitmentHint::<TestF, D>::singleton(
            FlatDigitBlocks::zeroed(vec![1]).expect("digit blocks"),
        );
        let claims = SuffixFoldClaims::<TestF, CyclotomicRing<TestF, D>, TestF, D> {
            point: vec![TestF::one()].into(),
            point_vars: PointVariableSelection::prefix(1, 1).expect("point vars"),
            polynomials: polys,
            commitment: commitment.clone(),
            hint,
        };
        assert_eq!(claims.commitment, commitment);
    }

    #[test]
    fn non_zk_eor_mismatch_is_rejected() {
        let prepared_point: PreparedOpeningPoint<TestF, TestF> = PreparedOpeningPoint {
            padded_point: Vec::new(),
            ring_opening_point: RingOpeningPoint {
                a: vec![TestF::one()],
                b: vec![TestF::one()],
            },
            ring_multiplier_point: RingMultiplierOpeningPoint::from_base(&RingOpeningPoint {
                a: vec![TestF::one()],
                b: vec![TestF::one()],
            }),
            packed_inner_point: RingBuf::from_single(&CyclotomicRing::<TestF, D>::zero()),
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

        let opening_batch = OpeningBatchShape::new(0, 1).expect("singleton opening batch");
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
