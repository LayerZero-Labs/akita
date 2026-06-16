use super::*;
#[cfg(not(feature = "zk"))]
use akita_types::CleartextWitnessShape;

fn append_shared_opening_point_to_transcript<F, E, T>(
    shared_opening_point: &[E],
    transcript: &mut T,
) where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
    T: Transcript<F>,
{
    for coord in shared_opening_point {
        append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, coord);
    }
}

pub(in crate::protocol::core) fn evaluate_claims_at_prepared_point<F, C, P, const D: usize>(
    polys: &[&P],
    prepared_point: &PreparedOpeningPoint<F, C, D>,
    block_len: usize,
) -> Result<FoldedClaimEvals<F, D>, AkitaError>
where
    F: FieldCore,
    C: FieldCore,
    P: AkitaPolyOps<F, D>,
{
    let _span = tracing::info_span!("root_evaluate_claims", num_claims = polys.len()).entered();
    let mut folded_rings = Vec::with_capacity(polys.len());
    let mut folded_blocks = Vec::with_capacity(polys.len());
    for poly in polys {
        let (folded_ring, folded_block) = evaluate_poly_at_multiplier_point(
            *poly,
            &prepared_point.ring_multiplier_point,
            block_len,
        )?;
        folded_rings.push(folded_ring);
        folded_blocks.push(folded_block);
    }
    Ok((folded_rings, folded_blocks))
}

fn multiplier_ring_weights<F: FieldCore, const D: usize>(
    point: &RingMultiplierOpeningPoint<F, D>,
) -> Result<MultiplierWeightSlices<'_, F, D>, AkitaError> {
    let b = point.b_rings().ok_or_else(|| {
        AkitaError::InvalidInput("ring multiplier must carry ring b weights".to_string())
    })?;
    let a = point.a_rings().ok_or_else(|| {
        AkitaError::InvalidInput("ring multiplier must carry ring a weights".to_string())
    })?;
    Ok((b, a))
}

fn evaluate_poly_at_multiplier_point<F, P, const D: usize>(
    poly: &P,
    point: &RingMultiplierOpeningPoint<F, D>,
    block_len: usize,
) -> Result<(CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>), AkitaError>
where
    F: FieldCore,
    P: AkitaPolyOps<F, D>,
{
    if let Some(base_point) = point.as_base() {
        return Ok(poly.evaluate_and_fold(&base_point.b, &base_point.a, block_len));
    }

    let (b, a) = multiplier_ring_weights(point)?;
    Ok(poly.evaluate_and_fold_ring(b, a, block_len))
}

fn validate_non_eor_root_opening_shape<F, E, const D: usize>(
    alpha_bits: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore,
    E: FpExtEncoding<F>,
{
    if !D.is_multiple_of(<E as ExtField<F>>::EXT_DEGREE)
        || !(D / <E as ExtField<F>>::EXT_DEGREE).is_power_of_two()
    {
        return Err(AkitaError::InvalidInput(
            "extension-field degree must divide the ring dimension into power-of-two slots"
                .to_string(),
        ));
    }

    let packed_slots = D / <E as ExtField<F>>::EXT_DEGREE;
    let packed_inner_bits = packed_slots.trailing_zeros() as usize;
    if packed_inner_bits > alpha_bits {
        return Err(AkitaError::InvalidPointDimension {
            expected: packed_inner_bits,
            actual: alpha_bits,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn prepare_fold_inner<
    'a,
    F,
    E,
    T,
    EorP,
    FoldP,
    V,
    B,
    const D: usize,
>(
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    needs_extension_reduction: bool,
    eor_polys: &[&EorP],
    fold_polys: &[&'a FoldP],
    opening_batch: &OpeningBatch,
    relation_opening_batch: OpeningBatch,
    trace_opening_batch: &OpeningBatch,
    opening_point: &[E],
    #[cfg(feature = "zk")] public_openings: Option<&[E]>,
    #[cfg(feature = "zk")] no_eor_trace_eval_target_public: Option<E>,
    pad_base_evals: bool,
    transcript: &mut T,
    #[cfg(feature = "zk")] mut zk_hiding: ZkHidingProverState<F>,
    expected_openings: Option<Vec<E>>,
    non_eor_protocol_point: Vec<E>,
    validate_non_eor: V,
    level_params: &LevelParams,
    alpha_bits: usize,
    basis: BasisMode,
    block_order: BlockOrder,
    commitment_hints: Vec<AkitaCommitmentHint<F, D>>,
    commitments: &[RingCommitment<F, D>],
    m_row_layout: MRowLayout,
    commitment: FlatRingVec<F>,
) -> Result<PreparedFold<F, E, D>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    E: FpExtEncoding<F>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + MulBaseUnreduced<F>
        + AkitaSerialize,
    T: Transcript<F>,
    EorP: AkitaPolyOps<F, D>,
    FoldP: AkitaPolyOps<F, D>,
    V: FnOnce() -> Result<(), AkitaError>,
    B: ProverComputeBackend<F>,
{
    let (fold_inputs, protocol_point, row_coefficients, reduction) = if needs_extension_reduction {
        let proved = prove_extension_opening_reduction::<F, E, T, EorP, D>(
            eor_polys,
            opening_batch,
            opening_point,
            #[cfg(feature = "zk")]
            public_openings,
            pad_base_evals,
            transcript,
            if pad_base_evals { "recursive" } else { "root" },
            #[cfg(feature = "zk")]
            &mut zk_hiding,
        )?;
        if let Some(expected_openings) = expected_openings.as_ref() {
            if proved.openings != *expected_openings {
                return Err(AkitaError::InvalidProof);
            }
        }
        let fold_inputs = {
            let _span =
                tracing::info_span!("extension_transform_polys", num_claims = fold_polys.len())
                    .entered();
            cfg_iter!(fold_polys)
                .map(|poly| {
                    <FoldP as AkitaPolyOps<F, D>>::tensor_packed_extension_fold_input::<E>(*poly)
                })
                .collect::<Result<Vec<FoldInputPoly<'a, F, FoldP, D>>, _>>()?
        };
        (
            fold_inputs,
            proved.protocol_point,
            Some(proved.row_coefficients),
            Some(proved.reduction),
        )
    } else {
        validate_non_eor()?;
        let fold_inputs = fold_polys
            .iter()
            .map(|poly| FoldInputPoly::Original(*poly))
            .collect::<Vec<_>>();
        let row_coefficients = if pad_base_evals {
            Some(vec![E::one()])
        } else {
            None
        };
        (fold_inputs, non_eor_protocol_point, row_coefficients, None)
    };
    let prepared_point = prepare_opening_point::<F, E, D>(
        &protocol_point,
        basis,
        level_params,
        alpha_bits,
        block_order,
    )?;
    let fold_refs = fold_inputs.iter().collect::<Vec<_>>();
    let (folded_rings, e_folded_by_claim) =
        evaluate_claims_at_prepared_point(&fold_refs, &prepared_point, level_params.block_len)?;
    for pt in &prepared_point.padded_point {
        append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
    }
    let (trace_target, row_coefficients) = compute_trace_target::<F, E, T, D>(
        &reduction,
        &folded_rings,
        &prepared_point,
        &protocol_point,
        alpha_bits,
        basis,
        trace_opening_batch,
        row_coefficients,
        transcript,
    )?;
    #[cfg(feature = "zk")]
    let mut trace_target = trace_target;
    #[cfg(feature = "zk")]
    if reduction.is_none() {
        if let Some(public_target) = no_eor_trace_eval_target_public {
            trace_target.trace_eval_target_public = public_target;
        }
    }
    let row_coefficient_rings = row_coefficient_rings::<F, E, D>(&row_coefficients)?;
    let (instance, witness) = RingRelationProver::new::<F, D, _, _, _>(
        backend,
        prepared,
        prepared_point.ring_opening_point.clone(),
        prepared_point.ring_multiplier_point.clone(),
        &fold_refs,
        e_folded_by_claim,
        relation_opening_batch,
        level_params.clone(),
        commitment_hints,
        transcript,
        commitments,
        row_coefficient_rings,
        m_row_layout,
    )?;

    let extension_opening_reduction = reduction.map(|reduction| reduction.proof);
    let row_coefficients = if pad_base_evals {
        None
    } else {
        Some(row_coefficients)
    };
    let trace_claim_scales = if pad_base_evals {
        None
    } else {
        trace_target.trace_claim_scales
    };

    Ok(PreparedFold {
        commitment,
        instance,
        witness,
        extension_opening_reduction,
        trace_eval_target: trace_target.trace_eval_target,
        trace_scale: trace_target.trace_scale,
        trace_prepared_point: Some(prepared_point),
        trace_claim_scales,
        #[cfg(feature = "zk")]
        trace_eval_target_public: trace_target.trace_eval_target_public,
        #[cfg(feature = "zk")]
        zk_hiding,
        row_coefficients,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn compute_trace_target<F, E, T, const D: usize>(
    reduction: &Option<ExtensionOpeningReduction<E>>,
    folded_rings: &[CyclotomicRing<F, D>],
    prepared_point: &PreparedOpeningPoint<F, E, D>,
    protocol_point: &[E],
    alpha_bits: usize,
    basis: BasisMode,
    opening_batch: &OpeningBatch,
    row_coefficients: Option<Vec<E>>,
    transcript: &mut T,
) -> Result<(TraceTarget<E>, Vec<E>), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F>,
    T: Transcript<F>,
{
    let inner_claim_point = &protocol_point[..protocol_point.len().min(alpha_bits)];
    let openings = folded_rings
        .iter()
        .map(|folded_ring| {
            scalar_opening_from_folded_ring::<F, E, E, D>(
                folded_ring,
                prepared_point,
                inner_claim_point,
                basis,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let row_coefficients = if let Some(row_coefficients) = row_coefficients {
        row_coefficients
    } else {
        append_claim_values_to_transcript::<F, E, T>(&openings, transcript);
        if opening_batch.num_claims() == 1 {
            vec![E::one()]
        } else {
            sample_public_row_coefficients::<F, E, T>(opening_batch, transcript)?
        }
    };
    let ordinary_trace_eval_target =
        batched_eval_target_from_opening_batch(opening_batch, &row_coefficients, &openings)?;
    let trace_eval_target =
        reduction
            .as_ref()
            .map_or(Ok(ordinary_trace_eval_target), |reduction| {
                check_extension_opening_reduction_output(
                    reduction.final_claim,
                    ordinary_trace_eval_target,
                    reduction.final_factor,
                )?;
                Ok(reduction.final_claim)
            })?;
    #[cfg(feature = "zk")]
    let trace_eval_target_public = reduction
        .as_ref()
        .map_or(trace_eval_target, |reduction| reduction.final_claim_public);
    let trace_claim_scales = reduction
        .as_ref()
        .map(|reduction| vec![reduction.final_factor; opening_batch.num_claims()]);
    let trace_scale = reduction
        .as_ref()
        .map_or(E::one(), |reduction| reduction.final_factor);

    Ok((
        TraceTarget {
            trace_eval_target,
            #[cfg(feature = "zk")]
            trace_eval_target_public,
            trace_claim_scales,
            trace_scale,
        },
        row_coefficients,
    ))
}

#[allow(clippy::too_many_arguments)]
fn prepare_root<F, E, T, P, B, const D: usize>(
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    polys: &[&P],
    opening_batch: OpeningBatch,
    shared_opening_point: &[E],
    commitments: &[RingCommitment<F, D>],
    commitment_hints: Vec<AkitaCommitmentHint<F, D>>,
    root_params: &LevelParams,
    m_row_layout: MRowLayout,
    #[cfg(feature = "zk")] zk_hiding: ZkHidingProverState<F>,
    basis: BasisMode,
) -> Result<PreparedFold<F, E, D>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasWide + HalvingField,
    E: FpExtEncoding<F>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + MulBaseUnreduced<F>
        + AkitaSerialize,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D>,
    B: ProverComputeBackend<F>,
{
    let num_claims = opening_batch.num_claims();
    let opening_num_vars = opening_batch.num_vars();
    let alpha_bits = root_params.ring_dimension.trailing_zeros() as usize;
    let needs_extension_reduction = root_tensor_projection_enabled::<F, E, E, D>(opening_num_vars);

    if shared_opening_point.len() > opening_num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: opening_num_vars,
            actual: shared_opening_point.len(),
        });
    }

    let expected_openings = if needs_extension_reduction {
        Some({
            let _span = tracing::info_span!("root_extension_check_openings", num_claims).entered();
            let mut padded_point = shared_opening_point.to_vec();
            padded_point.resize(opening_num_vars, E::zero());
            cfg_iter!(polys)
                .map(|poly| poly.evaluate_extension(&padded_point))
                .collect::<Result<Vec<_>, _>>()?
        })
    } else {
        None
    };
    let commitment_rows = flatten_batched_commitment_rows(commitments);
    prepare_fold_inner::<F, E, T, P, P, _, B, D>(
        backend,
        prepared,
        needs_extension_reduction,
        polys,
        polys,
        &opening_batch,
        opening_batch.clone(),
        &opening_batch,
        shared_opening_point,
        #[cfg(feature = "zk")]
        None,
        #[cfg(feature = "zk")]
        None,
        false,
        transcript,
        #[cfg(feature = "zk")]
        zk_hiding,
        expected_openings,
        shared_opening_point.to_vec(),
        || validate_non_eor_root_opening_shape::<F, E, D>(alpha_bits),
        root_params,
        alpha_bits,
        basis,
        BlockOrder::RowMajor,
        commitment_hints,
        commitments,
        m_row_layout,
        FlatRingVec::from_ring_elems(&commitment_rows),
    )
}

/// Prove the folded-root proof payload for an intermediate root.
///
/// The caller owns schedule/config selection and passes the validated schedule
/// execution for level 0. This function owns root polynomial folding, public
/// root transcript setup, root ring-relation construction, and the folded-root
/// prover mechanics.
///
/// # Errors
///
/// Returns an error if root inputs are malformed, polynomial folding or
/// ring-relation construction fails, or the folded-root prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_root<F, E, T, P, B, Cfg, const D: usize>(
    expanded: &Arc<AkitaExpandedSetup<F>>,
    prefix_slots: &SetupPrefixProverRegistry<F, D>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    polys: &[&P],
    opening_batch: OpeningBatch,
    shared_opening_point: &[E],
    commitments: &[RingCommitment<F, D>],
    commitment_hints: Vec<AkitaCommitmentHint<F, D>>,
    scheduled: &ExecutionSchedule,
    #[cfg(feature = "zk")] zk_hiding: ZkHidingProverState<F>,
    basis: BasisMode,
    setup_contribution_mode: SetupContributionMode,
) -> Result<ProveLevelOutput<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasWide + HalvingField + PseudoMersenneField,
    E: FpExtEncoding<F>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + MulBaseUnreduced<F>
        + AkitaSerialize,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D>,
    B: ProverComputeBackend<F>,
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
{
    let num_claims = opening_batch.num_claims();
    let root_params = &scheduled.params;

    if polys.len() != num_claims {
        return Err(AkitaError::InvalidInput(
            "invalid root-level inputs".to_string(),
        ));
    }

    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            level = 0usize,
            num_claims,
            "prove_root"
        );
    }

    append_opening_batch_shape_to_transcript::<F, T>(&opening_batch, transcript)?;
    append_batched_commitments_to_transcript(commitments, transcript);
    append_shared_opening_point_to_transcript::<F, E, T>(shared_opening_point, transcript);

    let prepared_fold = prepare_root::<F, E, T, P, B, D>(
        backend,
        prepared,
        transcript,
        polys,
        opening_batch,
        shared_opening_point,
        commitments,
        commitment_hints,
        root_params,
        MRowLayout::WithDBlock,
        #[cfg(feature = "zk")]
        zk_hiding,
        basis,
    )?;

    prove_fold::<F, E, T, B, Cfg, D>(
        expanded,
        prefix_slots,
        backend,
        prepared,
        transcript,
        0,
        scheduled,
        prepared_fold,
        setup_contribution_mode,
        false,
        #[cfg(not(feature = "zk"))]
        None,
    )?
    .get_intermediate()
}

/// Terminal-root analogue of [`prove_root`] used when the
/// schedule has exactly one fold level (the root is itself the terminal).
///
/// Mirrors the intermediate-root path through opening-batch absorbs,
/// optional extension-opening reduction, and ring-relation setup, then
/// emits a [`TerminalLevelProof`] through the shared fold prover instead of a
/// [`ProveLevelOutput`].
///
/// # Errors
///
/// Returns an error if opening-batch setup, EOR construction, or the inner
/// terminal-root prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_terminal_root_fold_with_params<Cfg, F, E, T, P, B, const D: usize>(
    expanded: &Arc<AkitaExpandedSetup<F>>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    polys: &[&P],
    opening_batch: OpeningBatch,
    shared_opening_point: &[E],
    commitments: &[RingCommitment<F, D>],
    commitment_hints: Vec<AkitaCommitmentHint<F, D>>,
    scheduled: &ExecutionSchedule,
    #[cfg(not(feature = "zk"))] terminal_direct_witness_shape: &CleartextWitnessShape,
    basis: BasisMode,
    setup_contribution_mode: SetupContributionMode,
    #[cfg(feature = "zk")] zk_hiding: &mut ZkHidingProverState<F>,
) -> Result<TerminalLevelProof<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasWide + HalvingField + PseudoMersenneField,
    E: FpExtEncoding<F>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + MulBaseUnreduced<F>
        + AkitaSerialize,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D>,
    B: ProverComputeBackend<F>,
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
{
    let num_claims = opening_batch.num_claims();
    let root_params = &scheduled.params;

    if polys.len() != num_claims {
        return Err(AkitaError::InvalidInput(
            "invalid root-level inputs".to_string(),
        ));
    }

    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            level = 0usize,
            num_claims,
            "prove_terminal_root_fold_with_params"
        );
    }

    append_opening_batch_shape_to_transcript::<F, T>(&opening_batch, transcript)?;
    append_batched_commitments_to_transcript(commitments, transcript);
    append_shared_opening_point_to_transcript::<F, E, T>(shared_opening_point, transcript);

    #[cfg(feature = "zk")]
    let owned_zk_hiding = std::mem::replace(zk_hiding, ZkHidingProverState::new(Vec::new()));
    let prepared_fold = prepare_root::<F, E, T, P, B, D>(
        backend,
        prepared,
        transcript,
        polys,
        opening_batch,
        shared_opening_point,
        commitments,
        commitment_hints,
        root_params,
        MRowLayout::WithoutDBlock,
        #[cfg(feature = "zk")]
        owned_zk_hiding,
        basis,
    )?;
    let prefix_slots = SetupPrefixProverRegistry::new();
    let terminal_result = prove_fold::<F, E, T, B, Cfg, D>(
        expanded,
        &prefix_slots,
        backend,
        prepared,
        transcript,
        0,
        scheduled,
        prepared_fold,
        setup_contribution_mode,
        true,
        #[cfg(not(feature = "zk"))]
        Some(terminal_direct_witness_shape),
    )?
    .get_terminal()?;

    #[cfg(not(feature = "zk"))]
    {
        Ok(terminal_result)
    }
    #[cfg(feature = "zk")]
    {
        let (terminal, returned_zk_hiding) = terminal_result;
        *zk_hiding = returned_zk_hiding;
        Ok(terminal)
    }
}
