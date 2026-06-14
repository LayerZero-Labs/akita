use super::*;

fn root_trace_claim_scales<C: Copy>(
    incidence_summary: &ClaimIncidenceSummary,
    factors_by_point: &[C],
) -> Result<Vec<C>, AkitaError> {
    incidence_summary
        .claim_to_point()
        .iter()
        .map(|&point_idx| {
            factors_by_point
                .get(point_idx)
                .copied()
                .ok_or(AkitaError::InvalidProof)
        })
        .collect()
}

pub(in crate::protocol::flow) fn evaluate_claims_at_prepared_points<F, C, P, const D: usize>(
    polys: &[&P],
    claim_to_point: &[usize],
    prepared_points: &[PreparedOpeningPoint<F, C, D>],
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
    for (poly, &point_idx) in polys.iter().zip(claim_to_point.iter()) {
        let prepared_point = &prepared_points[point_idx];
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

fn validate_non_eor_root_opening_shape<F, E, C, const D: usize>(
    alpha_bits: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore,
    E: RingSubfieldEncoding<F>,
    C: RingSubfieldEncoding<F> + ExtField<E>,
{
    if <C as ExtField<F>>::EXT_DEGREE != <E as ExtField<F>>::EXT_DEGREE {
        return Err(AkitaError::InvalidInput(
            "baseline extension root openings require claim and challenge fields to have the same base degree"
                .to_string(),
        ));
    }
    if !D.is_multiple_of(<E as ExtField<F>>::EXT_DEGREE)
        || !(D / <E as ExtField<F>>::EXT_DEGREE).is_power_of_two()
    {
        return Err(AkitaError::InvalidInput(
            "claim-field degree must divide the ring dimension into power-of-two slots".to_string(),
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
fn prepare_root_fold_from_evaluated_claims<F, C, T, P, B, const D: usize>(
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    commitments: &[RingCommitment<F, D>],
    hints: Vec<AkitaCommitmentHint<F, D>>,
    root_params: &LevelParams,
    m_row_layout: MRowLayout,
    prepared_points: &[PreparedOpeningPoint<F, C, D>],
    e_folded_by_poly: Vec<Vec<CyclotomicRing<F, D>>>,
    trace_eval_target: C,
    #[cfg(feature = "zk")] trace_eval_target_public: C,
    trace_claim_scales: Option<Vec<C>>,
    row_coefficients: Vec<C>,
    row_coefficient_rings: Vec<CyclotomicRing<F, D>>,
    extension_reduction: Option<RootExtensionOpeningReduction<C>>,
    #[cfg(feature = "zk")] zk_hiding: ZkHidingProverState<F>,
) -> Result<PreparedFold<F, C, D>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasWide + HalvingField,
    C: ExtField<F>
        + RingSubfieldEncoding<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D>,
    B: ProverComputeBackend<F>,
{
    let extension_opening_reduction = extension_reduction.map(|reduction| reduction.proof);

    let ring_opening_points = incidence_summary
        .public_rows()
        .iter()
        .map(|row| {
            prepared_points
                .get(row.point_idx())
                .map(|prepared_point| prepared_point.ring_opening_point.clone())
                .ok_or_else(|| {
                    AkitaError::InvalidInput("public row point index out of range".to_string())
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let ring_multiplier_points = incidence_summary
        .public_rows()
        .iter()
        .map(|row| {
            prepared_points
                .get(row.point_idx())
                .map(|prepared_point| prepared_point.ring_multiplier_point.clone())
                .ok_or_else(|| {
                    AkitaError::InvalidInput("public row point index out of range".to_string())
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (instance, witness) = RingRelationProver::new::<F, D, _, _, _>(
        backend,
        prepared,
        ring_opening_points,
        ring_multiplier_points,
        incidence_summary.claim_to_point().to_vec(),
        polys,
        e_folded_by_poly,
        incidence_summary,
        root_params.clone(),
        hints,
        transcript,
        commitments,
        row_coefficient_rings,
        m_row_layout,
    )?;

    let commitment_rows_owned: Option<Vec<CyclotomicRing<F, D>>> = if commitments.len() == 1 {
        None
    } else {
        Some(flatten_batched_commitment_rows(commitments))
    };
    let commitment_rows: &[CyclotomicRing<F, D>] = match &commitment_rows_owned {
        Some(v) => v.as_slice(),
        None => commitments[0].u.as_slice(),
    };
    Ok(PreparedFold {
        commitment: FlatRingVec::from_ring_elems(commitment_rows),
        instance,
        witness,
        extension_opening_reduction,
        trace_eval_target,
        #[cfg(feature = "zk")]
        trace_eval_target_public,
        trace_prepared_points: prepared_points.to_vec(),
        trace_claim_scales,
        trace_scale: C::one(),
        #[cfg(feature = "zk")]
        zk_hiding,
        row_coefficients: Some(row_coefficients),
    })
}
#[allow(clippy::too_many_arguments)]
fn prepare_root_fold_data<F, E, C, T, P, B, const D: usize>(
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    claim_points: &[&[E]],
    commitments: &[RingCommitment<F, D>],
    hints: Vec<AkitaCommitmentHint<F, D>>,
    root_params: &LevelParams,
    m_row_layout: MRowLayout,
    #[cfg(feature = "zk")] mut zk_hiding: ZkHidingProverState<F>,
    basis: BasisMode,
) -> Result<PreparedFold<F, C, D>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasWide + HalvingField,
    E: RingSubfieldEncoding<F> + MulBaseUnreduced<F>,
    C: RingSubfieldEncoding<F>
        + ExtField<E>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D>,
    B: ProverComputeBackend<F>,
{
    let claim_to_point = incidence_summary.claim_to_point();
    let num_claims = incidence_summary.num_claims();
    let alpha_bits = root_params.ring_dimension.trailing_zeros() as usize;
    let needs_extension_reduction =
        root_tensor_projection_enabled::<F, E, C, D>(incidence_summary.num_vars());

    if needs_extension_reduction {
        let (reduction, row_coefficients) =
            prove_root_extension_opening_reduction::<F, E, C, T, P, D>(
                polys,
                incidence_summary,
                claim_points,
                transcript,
                #[cfg(feature = "zk")]
                &mut zk_hiding,
            )?;
        let transformed_polys = {
            let _span = tracing::info_span!("root_extension_transform_polys", num_claims).entered();
            cfg_iter!(polys)
                .map(|poly| poly.tensor_packed_extension_root_poly::<C>())
                .collect::<Result<Vec<RootTensorProjectionPoly<F, D>>, _>>()?
        };
        let transformed_refs = transformed_polys.iter().collect::<Vec<_>>();
        let row_coefficient_rings = row_coefficient_rings::<F, C, D>(&row_coefficients)?;
        let protocol_point = {
            let _span = tracing::info_span!("root_extension_protocol_point").entered();
            ring_subfield_packed_extension_opening_point::<F, C, D>(
                reduction.rho.len(),
                &reduction.rho,
            )?
        };
        let prepared_protocol_point = prepare_opening_point::<F, C, D>(
            &protocol_point,
            basis,
            root_params,
            alpha_bits,
            BlockOrder::RowMajor,
        )?;
        let prepared_points = vec![prepared_protocol_point; incidence_summary.num_points()];

        let (_folded_rings, e_folded_by_poly) = evaluate_claims_at_prepared_points(
            &transformed_refs,
            claim_to_point,
            &prepared_points,
            root_params.block_len,
        )?;
        let trace_eval_target = reduction.final_claim;
        #[cfg(feature = "zk")]
        let trace_eval_target_public = reduction.final_claim_public;
        let trace_claim_scales = Some(root_trace_claim_scales(
            incidence_summary,
            &reduction.factors_by_point,
        )?);
        return prepare_root_fold_from_evaluated_claims::<
            F,
            C,
            T,
            RootTensorProjectionPoly<F, D>,
            B,
            D,
        >(
            backend,
            prepared,
            transcript,
            &transformed_refs,
            incidence_summary,
            commitments,
            hints,
            root_params,
            m_row_layout,
            &prepared_points,
            e_folded_by_poly,
            trace_eval_target,
            #[cfg(feature = "zk")]
            trace_eval_target_public,
            trace_claim_scales,
            row_coefficients,
            row_coefficient_rings,
            Some(reduction),
            #[cfg(feature = "zk")]
            zk_hiding,
        );
    }

    validate_non_eor_root_opening_shape::<F, E, C, D>(alpha_bits)?;
    let prepared_points = claim_points
        .iter()
        .map(|opening_point| {
            let challenge_point = opening_point
                .iter()
                .copied()
                .map(C::lift_base)
                .collect::<Vec<_>>();
            prepare_opening_point::<F, C, D>(
                &challenge_point,
                basis,
                root_params,
                alpha_bits,
                BlockOrder::RowMajor,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let (folded_rings, e_folded_by_poly) = evaluate_claims_at_prepared_points(
        polys,
        claim_to_point,
        &prepared_points,
        root_params.block_len,
    )?;

    let target_num_vars = root_params
        .m_vars
        .checked_add(root_params.r_vars)
        .and_then(|n| n.checked_add(alpha_bits))
        .ok_or_else(|| AkitaError::InvalidSetup("opening point length overflow".to_string()))?;
    let inner_claim_points = claim_points
        .iter()
        .map(|point| {
            if point.len() > target_num_vars {
                return Err(AkitaError::InvalidPointDimension {
                    expected: target_num_vars,
                    actual: point.len(),
                });
            }
            Ok(point[..point.len().min(alpha_bits)].to_vec())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let openings: Vec<E> = folded_rings
        .iter()
        .zip(claim_to_point.iter())
        .map(|(folded_ring, &point_idx)| {
            scalar_opening_from_folded_ring::<F, E, C, D>(
                folded_ring,
                &prepared_points[point_idx],
                &inner_claim_points[point_idx],
                basis,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    append_claim_values_to_transcript::<F, E, T>(&openings, transcript);
    let row_coefficients =
        sample_public_row_coefficients::<F, C, T>(incidence_summary, transcript)?;
    let row_coefficient_rings = row_coefficient_rings::<F, C, D>(&row_coefficients)?;
    let trace_eval_target =
        batched_eval_target_from_incidence(incidence_summary, &row_coefficients, &openings)?;
    #[cfg(feature = "zk")]
    let trace_eval_target_public = trace_eval_target;

    prepare_root_fold_from_evaluated_claims::<F, C, T, P, B, D>(
        backend,
        prepared,
        transcript,
        polys,
        incidence_summary,
        commitments,
        hints,
        root_params,
        m_row_layout,
        &prepared_points,
        e_folded_by_poly,
        trace_eval_target,
        #[cfg(feature = "zk")]
        trace_eval_target_public,
        None,
        row_coefficients,
        row_coefficient_rings,
        None,
        #[cfg(feature = "zk")]
        zk_hiding,
    )
}

/// Prove the folded root level using already-selected root and next-level
/// parameters.
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
pub fn prove_root_fold<F, E, C, T, P, B, Cfg, const D: usize>(
    expanded: &Arc<AkitaExpandedSetup<F>>,
    prefix_slots: &SetupPrefixProverRegistry<F, D>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    claim_points: &[&[E]],
    commitments: &[RingCommitment<F, D>],
    hints: Vec<AkitaCommitmentHint<F, D>>,
    scheduled: &ExecutionSchedule,
    #[cfg(feature = "zk")] zk_hiding: ZkHidingProverState<F>,
    basis: BasisMode,
    setup_contribution_mode: SetupContributionMode,
) -> Result<ProveLevelOutput<F, C>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasWide + HalvingField + PseudoMersenneField,
    E: RingSubfieldEncoding<F> + MulBaseUnreduced<F>,
    C: RingSubfieldEncoding<F>
        + ExtField<E>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D>,
    B: ProverComputeBackend<F>,
    Cfg: CommitmentConfig<Field = F, ClaimField = E, ChallengeField = C>,
{
    let claim_to_point = incidence_summary.claim_to_point();
    let num_claims = incidence_summary.num_claims();
    let root_params = &scheduled.params;

    if claim_points.is_empty()
        || claim_points.len() != incidence_summary.num_points()
        || claim_to_point.len() != num_claims
        || polys.len() != num_claims
        || commitments.len() != incidence_summary.num_points()
        || hints.len() != incidence_summary.num_points()
    {
        return Err(AkitaError::InvalidInput(
            "invalid root-level inputs".to_string(),
        ));
    }
    if claim_to_point
        .iter()
        .any(|&point_idx| point_idx >= claim_points.len())
    {
        return Err(AkitaError::InvalidInput(
            "root-level claim-to-point index out of range".to_string(),
        ));
    }

    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            level = 0usize,
            num_claims,
            num_points = claim_points.len(),
            "prove_root_fold"
        );
    }

    append_claim_incidence_shape_to_transcript::<F, T>(incidence_summary, transcript)?;
    append_batched_commitments_to_transcript(commitments, transcript);
    append_claim_points_to_transcript::<F, E, T>(claim_points, transcript);

    let prepared_fold = prepare_root_fold_data::<F, E, C, T, P, B, D>(
        backend,
        prepared,
        transcript,
        polys,
        incidence_summary,
        claim_points,
        commitments,
        hints,
        root_params,
        MRowLayout::WithDBlock,
        #[cfg(feature = "zk")]
        zk_hiding,
        basis,
    )?;

    prove_fold::<F, C, T, B, Cfg, D>(
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
    )?
    .get_intermediate()
}

/// Terminal-root analogue of [`prove_root_fold`] used when the
/// schedule has exactly one fold level (the root is itself the terminal).
///
/// Mirrors the intermediate-root path through claim-incidence absorbs,
/// optional extension-opening reduction, and ring-relation setup, then
/// emits a [`TerminalLevelProof`] through the shared fold prover instead of a
/// [`ProveLevelOutput`].
///
/// # Errors
///
/// Returns an error if claim-incidence setup, EOR construction, or the inner
/// terminal-root prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_terminal_root_fold_with_params<Cfg, F, E, C, T, P, B, const D: usize>(
    expanded: &Arc<AkitaExpandedSetup<F>>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    claim_points: &[&[E]],
    commitments: &[RingCommitment<F, D>],
    hints: Vec<AkitaCommitmentHint<F, D>>,
    scheduled: &ExecutionSchedule,
    basis: BasisMode,
    setup_contribution_mode: SetupContributionMode,
    #[cfg(feature = "zk")] zk_hiding: &mut ZkHidingProverState<F>,
) -> Result<TerminalLevelProof<F, C>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasWide + HalvingField + PseudoMersenneField,
    E: RingSubfieldEncoding<F> + MulBaseUnreduced<F>,
    C: RingSubfieldEncoding<F>
        + ExtField<E>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D>,
    B: ProverComputeBackend<F>,
    Cfg: CommitmentConfig<Field = F, ChallengeField = C>,
{
    let claim_to_point = incidence_summary.claim_to_point();
    let num_claims = incidence_summary.num_claims();
    let root_params = &scheduled.params;

    if claim_points.is_empty()
        || claim_points.len() != incidence_summary.num_points()
        || claim_to_point.len() != num_claims
        || polys.len() != num_claims
        || commitments.len() != incidence_summary.num_points()
        || hints.len() != incidence_summary.num_points()
    {
        return Err(AkitaError::InvalidInput(
            "invalid root-level inputs".to_string(),
        ));
    }
    if claim_to_point
        .iter()
        .any(|&point_idx| point_idx >= claim_points.len())
    {
        return Err(AkitaError::InvalidInput(
            "root-level claim-to-point index out of range".to_string(),
        ));
    }

    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            level = 0usize,
            num_claims,
            num_points = claim_points.len(),
            "prove_terminal_root_fold_with_params"
        );
    }

    append_claim_incidence_shape_to_transcript::<F, T>(incidence_summary, transcript)?;
    append_batched_commitments_to_transcript(commitments, transcript);
    append_claim_points_to_transcript::<F, E, T>(claim_points, transcript);

    #[cfg(feature = "zk")]
    let owned_zk_hiding = std::mem::replace(zk_hiding, ZkHidingProverState::new(Vec::new()));
    let prepared_fold = prepare_root_fold_data::<F, E, C, T, P, B, D>(
        backend,
        prepared,
        transcript,
        polys,
        incidence_summary,
        claim_points,
        commitments,
        hints,
        root_params,
        MRowLayout::WithoutDBlock,
        #[cfg(feature = "zk")]
        owned_zk_hiding,
        basis,
    )?;
    let prefix_slots = SetupPrefixProverRegistry::new();
    let terminal_result = prove_fold::<F, C, T, B, Cfg, D>(
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
