use super::*;

fn evaluate_root_claims_at_prepared_points<F, P, const D: usize>(
    polys: &[&P],
    claim_to_point: &[usize],
    prepared_points: &[PreparedRootOpeningPoint<F, D>],
    block_len: usize,
) -> Result<RootClaimEvaluations<F, D>, AkitaError>
where
    F: FieldCore,
    P: AkitaPolyOps<F, D>,
{
    let mut per_claim_y_rings = Vec::with_capacity(polys.len());
    let mut w_folded_by_poly = Vec::with_capacity(polys.len());
    for (poly, &point_idx) in polys.iter().zip(claim_to_point.iter()) {
        let prepared_point = &prepared_points[point_idx];
        let (y_ring, w_folded) = evaluate_poly_at_multiplier_point(
            *poly,
            &prepared_point.ring_multiplier_point,
            block_len,
        )?;
        per_claim_y_rings.push(y_ring);
        w_folded_by_poly.push(w_folded);
    }
    Ok((per_claim_y_rings, w_folded_by_poly))
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

pub(in crate::protocol::flow) fn evaluate_recursive_witness_at_multiplier_point<F, const D: usize>(
    witness: &RecursiveWitnessView<'_, F, D>,
    point: &RingMultiplierOpeningPoint<F, D>,
    block_len: usize,
    num_blocks: usize,
) -> Result<(CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if let Some(base_point) = point.as_base() {
        return Ok(witness.evaluate_and_fold(&base_point.b, &base_point.a, block_len, num_blocks));
    }

    let (b, a) = multiplier_ring_weights(point)?;
    Ok(witness.evaluate_and_fold_ring(b, a, block_len, num_blocks))
}

#[allow(clippy::too_many_arguments)]
fn finish_root_fold_with_prepared_openings<F, C, T, P, B, const D: usize, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    commitments: &[RingCommitment<F, D>],
    hints: Vec<AkitaCommitmentHint<F, D>>,
    root_params: &LevelParams,
    expected_w_len: usize,
    next_log_basis: u32,
    commit_w_for_next: CommitW,
    prepared_points: Vec<PreparedRootOpeningPoint<F, D>>,
    w_folded_by_poly: Vec<Vec<CyclotomicRing<F, D>>>,
    y_rings: Vec<CyclotomicRing<F, D>>,
    #[cfg(feature = "zk")] y_rings_masked: Vec<CyclotomicRing<F, D>>,
    row_coefficients: Vec<C>,
    row_coefficient_rings: Vec<CyclotomicRing<F, D>>,
    extension_opening_reduction: Option<ExtensionOpeningReductionProof<C>>,
    #[cfg(feature = "zk")] zk_hiding_commitment: ZkHidingCommitment<F>,
    #[cfg(feature = "zk")] zk_hiding: ZkHidingProverState<F>,
) -> Result<RootLevelRawOutput<F, C, D>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    C: ExtField<F> + RingSubfieldEncoding<F> + HasUnreducedOps + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D>,
    B: ProverComputeBackend<F>,
    CommitW: FnOnce(&RecursiveWitnessFlat) -> Result<NextWitnessCommitment<F>, AkitaError>,
{
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
    let quad_eq = Box::new(QuadraticEquation::<F, { D }>::new_prover(
        backend,
        prepared,
        ring_opening_points,
        ring_multiplier_points,
        incidence_summary.claim_to_point().to_vec(),
        polys,
        w_folded_by_poly,
        incidence_summary,
        root_params.clone(),
        hints,
        transcript,
        commitments,
        &y_rings,
        row_coefficient_rings,
        MRowLayout::Intermediate,
    )?);

    let commitment_rows_owned: Option<Vec<CyclotomicRing<F, D>>> = if commitments.len() == 1 {
        None
    } else {
        Some(flatten_batched_commitment_rows(commitments))
    };
    let commitment_rows: &[CyclotomicRing<F, D>] = match &commitment_rows_owned {
        Some(v) => v.as_slice(),
        None => commitments[0].u.as_slice(),
    };

    let mut raw = prove_root_fold_from_quadratic::<F, C, T, B, D, _>(
        expanded,
        backend,
        prepared,
        transcript,
        commitment_rows,
        root_params,
        expected_w_len,
        next_log_basis,
        #[cfg(feature = "zk")]
        zk_hiding_commitment,
        #[cfg(feature = "zk")]
        zk_hiding,
        quad_eq,
        y_rings,
        #[cfg(feature = "zk")]
        y_rings_masked,
        row_coefficients,
        commit_w_for_next,
    )?;
    raw.extension_opening_reduction = extension_opening_reduction;
    Ok(raw)
}

/// Prove the folded root level using already-selected root and next-level
/// parameters.
///
/// The caller owns schedule/config selection and passes the expected next
/// recursive witness length, next digit basis, and commitment policy for that
/// witness. This function owns root polynomial folding, public root transcript
/// setup, root quadratic-equation construction, and the folded-root prover
/// mechanics.
///
/// # Errors
///
/// Returns an error if root inputs are malformed, polynomial folding or
/// quadratic-equation construction fails, or the folded-root prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_root_fold_with_params<F, E, C, T, P, B, const D: usize, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    claim_points: &[&[E]],
    commitments: &[RingCommitment<F, D>],
    hints: Vec<AkitaCommitmentHint<F, D>>,
    root_params: &LevelParams,
    expected_w_len: usize,
    next_log_basis: u32,
    #[cfg(feature = "zk")] zk_hiding_commitment: ZkHidingCommitment<F>,
    #[cfg(feature = "zk")] mut zk_hiding: ZkHidingProverState<F>,
    basis: BasisMode,
    commit_w_for_next: CommitW,
) -> Result<RootLevelRawOutput<F, C, D>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    E: RingSubfieldEncoding<F>,
    C: RingSubfieldEncoding<F>
        + ExtField<E>
        + ExtField<F>
        + HasUnreducedOps
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D>,
    B: ProverComputeBackend<F>,
    CommitW: FnOnce(&RecursiveWitnessFlat) -> Result<NextWitnessCommitment<F>, AkitaError>,
{
    let claim_to_point = incidence_summary.claim_to_point();
    let num_claims = incidence_summary.num_claims();

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
            "prove_root_fold_with_params"
        );
    }

    append_claim_incidence_shape_to_transcript::<F, T>(incidence_summary, transcript)?;
    append_batched_commitments_to_transcript(commitments, transcript);
    append_claim_points_to_transcript::<F, E, T>(claim_points, transcript);

    let alpha_bits = root_params.ring_dimension.trailing_zeros() as usize;
    let needs_extension_reduction =
        root_tensor_projection_enabled::<F, E, C, D>(incidence_summary.num_vars());
    let extension_reduction_prepare = if !needs_extension_reduction {
        None
    } else {
        Some(prepare_root_extension_opening_reduction::<F, E, C, P, D>(
            polys,
            incidence_summary,
            claim_points,
        )?)
    };

    let openings: Vec<E>;
    let prepared_points: Vec<PreparedRootOpeningPoint<F, D>>;
    if let Some(prepared_reduction) = extension_reduction_prepare {
        openings = prepared_reduction.openings.clone();
        append_claim_values_to_transcript::<F, E, T>(&openings, transcript);
        let row_coefficients =
            sample_public_row_coefficients::<F, C, T>(incidence_summary, transcript)?;
        let row_coefficient_rings = row_coefficient_rings::<F, C, D>(&row_coefficients)?;
        let reduction = prove_prepared_root_extension_opening_reduction::<F, E, C, T, P, D>(
            polys,
            incidence_summary,
            root_params,
            basis,
            &row_coefficients,
            prepared_reduction,
            transcript,
            #[cfg(feature = "zk")]
            &mut zk_hiding,
        )?;
        let protocol_point = ring_subfield_packed_extension_opening_point::<F, C, D>(
            reduction.rho.len(),
            &reduction.rho,
        )?;
        let prepared_protocol_point = prepare_root_opening_point_ext::<F, C, C, D>(
            &protocol_point,
            basis,
            root_params,
            alpha_bits,
        )?;
        prepared_points = vec![prepared_protocol_point; incidence_summary.num_points()];
        let transformed_polys = cfg_iter!(polys)
            .map(|poly| poly.tensor_packed_extension_root_poly::<C>())
            .collect::<Result<Vec<RootTensorProjectionPoly<F, D>>, _>>()?;
        let transformed_refs = transformed_polys.iter().collect::<Vec<_>>();

        let (per_claim_y_rings, w_folded_by_poly) = evaluate_root_claims_at_prepared_points(
            &transformed_refs,
            claim_to_point,
            &prepared_points,
            root_params.block_len,
        )?;
        let y_rings = combine_root_y_rings::<F, D>(
            &per_claim_y_rings,
            incidence_summary,
            &row_coefficient_rings,
        )?;
        #[cfg(feature = "zk")]
        let y_rings_masked = {
            let mut masked = y_rings.clone();
            for y_ring in &mut masked {
                let (_, y_garbage) = zk_hiding.take_ring::<D>()?;
                *y_ring += y_garbage;
            }
            masked
        };
        #[cfg(not(feature = "zk"))]
        for y_ring in &y_rings {
            transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
        }
        #[cfg(feature = "zk")]
        for y_ring in &y_rings_masked {
            transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
        }
        let internal_claims = y_rings
            .iter()
            .zip(incidence_summary.public_rows().iter())
            .map(|(y_ring, row)| {
                recover_ring_subfield_inner_product::<F, C, D>(
                    y_ring,
                    &prepared_points[row.point_idx()].inner_reduction,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let final_opening = internal_claims
            .iter()
            .zip(incidence_summary.public_rows().iter())
            .fold(C::zero(), |acc, (&opening, row)| {
                acc + opening * reduction.factors_by_point[row.point_idx()]
            });
        check_extension_opening_reduction_output(reduction.final_claim, final_opening, C::one())?;
        let extension_opening_reduction = Some(reduction.proof);

        return finish_root_fold_with_prepared_openings::<
            F,
            C,
            T,
            RootTensorProjectionPoly<F, D>,
            B,
            D,
            _,
        >(
            expanded,
            backend,
            prepared,
            transcript,
            &transformed_refs,
            incidence_summary,
            commitments,
            hints,
            root_params,
            expected_w_len,
            next_log_basis,
            commit_w_for_next,
            prepared_points,
            w_folded_by_poly,
            y_rings,
            #[cfg(feature = "zk")]
            y_rings_masked,
            row_coefficients,
            row_coefficient_rings,
            extension_opening_reduction,
            #[cfg(feature = "zk")]
            zk_hiding_commitment,
            #[cfg(feature = "zk")]
            zk_hiding,
        );
    }

    prepared_points = claim_points
        .iter()
        .map(|opening_point| {
            prepare_root_opening_point_ext::<F, E, C, D>(
                opening_point,
                basis,
                root_params,
                alpha_bits,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let (per_claim_y_rings, w_folded_by_poly) = evaluate_root_claims_at_prepared_points(
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

    openings = per_claim_y_rings
        .iter()
        .zip(claim_to_point.iter())
        .map(|(y_ring, &point_idx)| {
            root_claim_opening_from_y_ring::<F, E, D>(
                y_ring,
                &prepared_points[point_idx],
                &inner_claim_points[point_idx],
                basis,
            )
        })
        .collect::<Result<_, _>>()?;
    append_claim_values_to_transcript::<F, E, T>(&openings, transcript);
    let row_coefficients =
        sample_public_row_coefficients::<F, C, T>(incidence_summary, transcript)?;
    let row_coefficient_rings = row_coefficient_rings::<F, C, D>(&row_coefficients)?;

    let y_rings = combine_root_y_rings::<F, D>(
        &per_claim_y_rings,
        incidence_summary,
        &row_coefficient_rings,
    )?;
    #[cfg(feature = "zk")]
    let y_rings_masked = {
        let mut masked = y_rings.clone();
        for y_ring in &mut masked {
            let (_, y_garbage) = zk_hiding.take_ring::<D>()?;
            *y_ring += y_garbage;
        }
        masked
    };
    #[cfg(feature = "zk")]
    for y_ring in &y_rings_masked {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }
    #[cfg(not(feature = "zk"))]
    for y_ring in &y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }

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
    let quad_eq = Box::new(QuadraticEquation::<F, { D }>::new_prover(
        backend,
        prepared,
        ring_opening_points,
        ring_multiplier_points,
        incidence_summary.claim_to_point().to_vec(),
        polys,
        w_folded_by_poly,
        incidence_summary,
        root_params.clone(),
        hints,
        transcript,
        commitments,
        &y_rings,
        row_coefficient_rings,
        MRowLayout::Intermediate,
    )?);

    let commitment_rows_owned: Option<Vec<CyclotomicRing<F, D>>> = if commitments.len() == 1 {
        None
    } else {
        Some(flatten_batched_commitment_rows(commitments))
    };
    let commitment_rows: &[CyclotomicRing<F, D>] = match &commitment_rows_owned {
        Some(v) => v.as_slice(),
        None => commitments[0].u.as_slice(),
    };

    prove_root_fold_from_quadratic::<F, C, T, B, D, _>(
        expanded,
        backend,
        prepared,
        transcript,
        commitment_rows,
        root_params,
        expected_w_len,
        next_log_basis,
        #[cfg(feature = "zk")]
        zk_hiding_commitment,
        #[cfg(feature = "zk")]
        zk_hiding,
        quad_eq,
        y_rings,
        #[cfg(feature = "zk")]
        y_rings_masked,
        row_coefficients,
        commit_w_for_next,
    )
}

/// Terminal-root analogue of [`prove_root_fold_with_params`] used when the
/// schedule has exactly one fold level (the root is itself the terminal).
///
/// Mirrors the intermediate-root path through claim-incidence absorbs,
/// optional extension-opening reduction, and quadratic-equation setup, then
/// emits a [`TerminalLevelProof`] via
/// [`prove_terminal_root_fold_from_quadratic`] instead of a
/// [`RootLevelRawOutput`].
///
/// # Errors
///
/// Returns an error if claim-incidence/transcript setup fails, the
/// extension-opening reduction proof construction fails, or the inner
/// terminal-root prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_terminal_root_fold_with_params<F, E, C, T, P, B, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    claim_points: &[&[E]],
    commitments: &[RingCommitment<F, D>],
    hints: Vec<AkitaCommitmentHint<F, D>>,
    root_params: &LevelParams,
    expected_w_len: usize,
    final_log_basis: u32,
    basis: BasisMode,
    #[cfg(feature = "zk")] zk_hiding: &mut ZkHidingProverState<F>,
) -> Result<TerminalLevelProof<F, C>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    E: RingSubfieldEncoding<F>,
    C: RingSubfieldEncoding<F> + ExtField<E> + HasUnreducedOps + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D>,
    B: ProverComputeBackend<F>,
{
    let claim_to_point = incidence_summary.claim_to_point();
    let num_claims = incidence_summary.num_claims();

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

    let alpha_bits = root_params.ring_dimension.trailing_zeros() as usize;
    let needs_extension_reduction =
        root_tensor_projection_enabled::<F, E, C, D>(incidence_summary.num_vars());
    let extension_reduction_prepare = if !needs_extension_reduction {
        None
    } else {
        Some(prepare_root_extension_opening_reduction::<F, E, C, P, D>(
            polys,
            incidence_summary,
            claim_points,
        )?)
    };

    let openings: Vec<E>;
    let prepared_points: Vec<PreparedRootOpeningPoint<F, D>>;
    if let Some(prepared_reduction) = extension_reduction_prepare {
        openings = prepared_reduction.openings.clone();
        append_claim_values_to_transcript::<F, E, T>(&openings, transcript);
        let row_coefficients =
            sample_public_row_coefficients::<F, C, T>(incidence_summary, transcript)?;
        let row_coefficient_rings = row_coefficient_rings::<F, C, D>(&row_coefficients)?;
        let reduction = prove_prepared_root_extension_opening_reduction::<F, E, C, T, P, D>(
            polys,
            incidence_summary,
            root_params,
            basis,
            &row_coefficients,
            prepared_reduction,
            transcript,
            #[cfg(feature = "zk")]
            zk_hiding,
        )?;
        let protocol_point = ring_subfield_packed_extension_opening_point::<F, C, D>(
            reduction.rho.len(),
            &reduction.rho,
        )?;
        let prepared_protocol_point = prepare_root_opening_point_ext::<F, C, C, D>(
            &protocol_point,
            basis,
            root_params,
            alpha_bits,
        )?;
        prepared_points = vec![prepared_protocol_point; incidence_summary.num_points()];
        let transformed_polys = polys
            .iter()
            .map(|poly| poly.tensor_packed_extension_root_poly::<C>())
            .collect::<Result<Vec<RootTensorProjectionPoly<F, D>>, _>>()?;
        let transformed_refs = transformed_polys.iter().collect::<Vec<_>>();

        let (per_claim_y_rings, w_folded_by_poly) = evaluate_root_claims_at_prepared_points(
            &transformed_refs,
            claim_to_point,
            &prepared_points,
            root_params.block_len,
        )?;
        let y_rings = combine_root_y_rings::<F, D>(
            &per_claim_y_rings,
            incidence_summary,
            &row_coefficient_rings,
        )?;
        #[cfg(feature = "zk")]
        let y_rings_masked = {
            let mut masked = y_rings.clone();
            for y_ring in &mut masked {
                let (_, y_garbage) = zk_hiding.take_ring::<D>()?;
                *y_ring += y_garbage;
            }
            masked
        };
        #[cfg(not(feature = "zk"))]
        for y_ring in &y_rings {
            transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
        }
        #[cfg(feature = "zk")]
        for y_ring in &y_rings_masked {
            transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
        }
        let internal_claims = y_rings
            .iter()
            .zip(incidence_summary.public_rows().iter())
            .map(|(y_ring, row)| {
                recover_ring_subfield_inner_product::<F, C, D>(
                    y_ring,
                    &prepared_points[row.point_idx()].inner_reduction,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let final_opening = internal_claims
            .iter()
            .zip(incidence_summary.public_rows().iter())
            .fold(C::zero(), |acc, (&opening, row)| {
                acc + opening * reduction.factors_by_point[row.point_idx()]
            });
        check_extension_opening_reduction_output(reduction.final_claim, final_opening, C::one())?;
        let extension_opening_reduction = Some(reduction.proof);

        return finish_terminal_root_fold_with_prepared_openings::<
            F,
            C,
            T,
            RootTensorProjectionPoly<F, D>,
            B,
            D,
        >(
            expanded,
            backend,
            prepared,
            transcript,
            &transformed_refs,
            incidence_summary,
            commitments,
            hints,
            root_params,
            expected_w_len,
            final_log_basis,
            prepared_points,
            w_folded_by_poly,
            y_rings,
            #[cfg(feature = "zk")]
            y_rings_masked,
            row_coefficients,
            row_coefficient_rings,
            extension_opening_reduction,
            #[cfg(feature = "zk")]
            zk_hiding,
        );
    }

    prepared_points = claim_points
        .iter()
        .map(|opening_point| {
            prepare_root_opening_point_ext::<F, E, C, D>(
                opening_point,
                basis,
                root_params,
                alpha_bits,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let (per_claim_y_rings, w_folded_by_poly) = evaluate_root_claims_at_prepared_points(
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

    openings = per_claim_y_rings
        .iter()
        .zip(claim_to_point.iter())
        .map(|(y_ring, &point_idx)| {
            root_claim_opening_from_y_ring::<F, E, D>(
                y_ring,
                &prepared_points[point_idx],
                &inner_claim_points[point_idx],
                basis,
            )
        })
        .collect::<Result<_, _>>()?;
    append_claim_values_to_transcript::<F, E, T>(&openings, transcript);
    let row_coefficients =
        sample_public_row_coefficients::<F, C, T>(incidence_summary, transcript)?;
    let row_coefficient_rings = row_coefficient_rings::<F, C, D>(&row_coefficients)?;

    let y_rings = combine_root_y_rings::<F, D>(
        &per_claim_y_rings,
        incidence_summary,
        &row_coefficient_rings,
    )?;
    #[cfg(feature = "zk")]
    let y_rings_masked = {
        let mut masked = y_rings.clone();
        for y_ring in &mut masked {
            let (_, y_garbage) = zk_hiding.take_ring::<D>()?;
            *y_ring += y_garbage;
        }
        masked
    };
    #[cfg(not(feature = "zk"))]
    for y_ring in &y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }
    #[cfg(feature = "zk")]
    for y_ring in &y_rings_masked {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }

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
    let quad_eq = Box::new(QuadraticEquation::<F, { D }>::new_prover(
        backend,
        prepared,
        ring_opening_points,
        ring_multiplier_points,
        incidence_summary.claim_to_point().to_vec(),
        polys,
        w_folded_by_poly,
        incidence_summary,
        root_params.clone(),
        hints,
        transcript,
        commitments,
        &y_rings,
        row_coefficient_rings,
        MRowLayout::Terminal,
    )?);

    let commitment_rows_owned: Option<Vec<CyclotomicRing<F, D>>> = if commitments.len() == 1 {
        None
    } else {
        Some(flatten_batched_commitment_rows(commitments))
    };
    let commitment_rows: &[CyclotomicRing<F, D>] = match &commitment_rows_owned {
        Some(v) => v.as_slice(),
        None => commitments[0].u.as_slice(),
    };

    prove_terminal_root_fold_from_quadratic::<F, C, T, B, D>(
        expanded,
        backend,
        prepared,
        transcript,
        commitment_rows,
        root_params,
        expected_w_len,
        final_log_basis,
        quad_eq,
        y_rings,
        #[cfg(feature = "zk")]
        y_rings_masked,
        row_coefficients,
        #[cfg(feature = "zk")]
        zk_hiding,
    )
}

#[allow(clippy::too_many_arguments)]
fn finish_terminal_root_fold_with_prepared_openings<F, C, T, P, B, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    commitments: &[RingCommitment<F, D>],
    hints: Vec<AkitaCommitmentHint<F, D>>,
    root_params: &LevelParams,
    expected_w_len: usize,
    final_log_basis: u32,
    prepared_points: Vec<PreparedRootOpeningPoint<F, D>>,
    w_folded_by_poly: Vec<Vec<CyclotomicRing<F, D>>>,
    y_rings: Vec<CyclotomicRing<F, D>>,
    #[cfg(feature = "zk")] y_rings_masked: Vec<CyclotomicRing<F, D>>,
    row_coefficients: Vec<C>,
    row_coefficient_rings: Vec<CyclotomicRing<F, D>>,
    extension_opening_reduction: Option<ExtensionOpeningReductionProof<C>>,
    #[cfg(feature = "zk")] zk_hiding: &mut ZkHidingProverState<F>,
) -> Result<TerminalLevelProof<F, C>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    C: ExtField<F> + RingSubfieldEncoding<F> + HasUnreducedOps + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D>,
    B: ProverComputeBackend<F>,
{
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
    let quad_eq = Box::new(QuadraticEquation::<F, { D }>::new_prover(
        backend,
        prepared,
        ring_opening_points,
        ring_multiplier_points,
        incidence_summary.claim_to_point().to_vec(),
        polys,
        w_folded_by_poly,
        incidence_summary,
        root_params.clone(),
        hints,
        transcript,
        commitments,
        &y_rings,
        row_coefficient_rings,
        MRowLayout::Terminal,
    )?);

    let commitment_rows_owned: Option<Vec<CyclotomicRing<F, D>>> = if commitments.len() == 1 {
        None
    } else {
        Some(flatten_batched_commitment_rows(commitments))
    };
    let commitment_rows: &[CyclotomicRing<F, D>] = match &commitment_rows_owned {
        Some(v) => v.as_slice(),
        None => commitments[0].u.as_slice(),
    };

    let mut terminal = prove_terminal_root_fold_from_quadratic::<F, C, T, B, D>(
        expanded,
        backend,
        prepared,
        transcript,
        commitment_rows,
        root_params,
        expected_w_len,
        final_log_basis,
        quad_eq,
        y_rings,
        #[cfg(feature = "zk")]
        y_rings_masked,
        row_coefficients,
        #[cfg(feature = "zk")]
        zk_hiding,
    )?;
    terminal.extension_opening_reduction = extension_opening_reduction;
    Ok(terminal)
}

/// Prove the folded root level after root orchestration has built its
/// quadratic equation and selected the next recursive commitment policy.
///
/// The root caller owns transcript setup for public openings and gamma
/// batching, schedule selection, and the commitment-row view used by the root
/// relation. It also passes the already-validated challenge sampler used for
/// the remaining base-field stage proofs. This function owns the config-free
/// prover mechanics from `w` construction through the stage proofs and next
/// recursive state.
///
/// # Errors
///
/// Returns an error if ring switching, recursive commitment, or either
/// sumcheck prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_root_fold_from_quadratic<F, C, T, B, const D: usize, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    commitment_rows: &[CyclotomicRing<F, D>],
    lp: &akita_types::LevelParams,
    expected_w_len: usize,
    next_log_basis: u32,
    #[cfg(feature = "zk")] zk_hiding_commitment: ZkHidingCommitment<F>,
    #[cfg(feature = "zk")] mut zk_hiding: ZkHidingProverState<F>,
    mut quad_eq: Box<QuadraticEquation<F, { D }>>,
    y_rings: Vec<CyclotomicRing<F, D>>,
    #[cfg(feature = "zk")] y_rings_masked: Vec<CyclotomicRing<F, D>>,
    row_coefficients: Vec<C>,
    commit_w_for_next: CommitW,
) -> Result<RootLevelRawOutput<F, C, D>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    C: ExtField<F> + RingSubfieldEncoding<F> + HasUnreducedOps + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
    B: ProverComputeBackend<F>,
    CommitW: FnOnce(&RecursiveWitnessFlat) -> Result<NextWitnessCommitment<F>, AkitaError>,
{
    let logical_w = ring_switch_build_w::<F, B, { D }>(&mut quad_eq, backend, prepared, lp)?;
    if logical_w.len() != expected_w_len {
        return Err(AkitaError::InvalidSetup(format!(
            "scheduled root next-w length did not match runtime witness: expected={expected_w_len}, actual={}",
            logical_w.len()
        )));
    }
    let next_commitment = {
        let _span = tracing::info_span!("commit_w_level", level = 0usize).entered();
        commit_w_for_next(&logical_w)?
    };
    let NextWitnessCommitment {
        witness: packed_witness,
        commitment: committed_commitment,
        hint: committed_hint,
    } = next_commitment;
    let w_commitment_proof = committed_commitment.clone();

    let rs = ring_switch_finalize_with_gamma::<F, C, T, { D }>(
        &quad_eq,
        expanded,
        transcript,
        &logical_w,
        &w_commitment_proof,
        lp,
        &row_coefficients,
        MRowLayout::Intermediate,
    )?;

    let relation_claim = relation_claim_from_rows_extension::<F, C, D>(
        &rs.tau1,
        rs.alpha,
        &quad_eq.v,
        commitment_rows,
        &y_rings,
    )?;
    #[cfg(feature = "zk")]
    let relation_claim_public = relation_claim_from_rows_extension::<F, C, D>(
        &rs.tau1,
        rs.alpha,
        &quad_eq.v,
        commitment_rows,
        &y_rings_masked,
    )?;
    let RingSwitchOutput {
        w_evals_compact,
        live_x_cols,
        m_evals_x,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau0,
        tau1: _,
        b,
        alpha: _,
    } = rs;
    let tau0_reordered = reorder_stage1_coords(&tau0, col_bits, ring_bits);
    #[cfg(feature = "zk")]
    let (stage1_round_pads, stage1_child_claim_masks, stage2_round_pads) =
        zk_hiding.take_current_level_pads::<C>(col_bits + ring_bits, b)?;
    let (stage1_proof, r_stage1, s_claim) = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        let stage1_prover = AkitaStage1Prover::new(
            &w_evals_compact,
            &tau0_reordered,
            b,
            live_x_cols,
            col_bits,
            ring_bits,
        )?;
        #[cfg(feature = "zk")]
        {
            stage1_prover.prove(transcript, stage1_round_pads, stage1_child_claim_masks)?
        }
        #[cfg(not(feature = "zk"))]
        {
            let (stage1_proof, r_stage1) = stage1_prover.prove(transcript)?;
            let s_claim = stage1_proof.s_claim;
            (stage1_proof, r_stage1, s_claim)
        }
    };

    transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &stage1_proof.s_claim);
    let batching_coeff: C = sample_ext_challenge::<F, C, T>(transcript, CHALLENGE_SUMCHECK_BATCH);
    #[cfg(feature = "zk")]
    let (stage2_sumcheck_proof_masked, sumcheck_challenges, w_eval, w_eval_masked) = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        let stage2_prover_result = AkitaStage2Prover::new(
            batching_coeff,
            w_evals_compact,
            &r_stage1,
            s_claim,
            b,
            alpha_evals_y,
            m_evals_x,
            live_x_cols,
            col_bits,
            ring_bits,
            relation_claim,
        );
        let mut stage2_prover = stage2_prover_result?;
        let stage2_public_input = batching_coeff * stage1_proof.s_claim + relation_claim_public;
        let (stage2_sumcheck_proof_masked, sumcheck_challenges) = stage2_prover
            .prove_zk::<F, T, _>(
                stage2_public_input,
                transcript,
                |tr| sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND),
                stage2_round_pads,
            )?;

        let w_eval = {
            let _span = tracing::info_span!("multilinear_eval", level = 0usize).entered();
            stage2_prover.final_w_eval()
        };
        let w_eval_masked = w_eval + zk_hiding.take_next_w_eval_mask::<C>()?;
        (
            stage2_sumcheck_proof_masked,
            sumcheck_challenges,
            w_eval,
            w_eval_masked,
        )
    };
    #[cfg(not(feature = "zk"))]
    let (stage2_sumcheck_proof, sumcheck_challenges, w_eval) = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        let mut stage2_prover = AkitaStage2Prover::new(
            batching_coeff,
            w_evals_compact,
            &r_stage1,
            s_claim,
            b,
            alpha_evals_y,
            m_evals_x,
            live_x_cols,
            col_bits,
            ring_bits,
            relation_claim,
        )?;
        let (stage2_sumcheck_proof, sumcheck_challenges, _) = stage2_prover
            .prove::<F, T, _>(transcript, |tr| {
                sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND)
            })?;

        let w_eval = {
            let _span = tracing::info_span!("multilinear_eval", level = 0usize).entered();
            stage2_prover.final_w_eval()
        };
        (stage2_sumcheck_proof, sumcheck_challenges, w_eval)
    };
    let (committed_witness, logical_w) = match packed_witness {
        Some(packed_witness) => (packed_witness, Some(logical_w)),
        None => (logical_w, None),
    };

    #[cfg(not(feature = "zk"))]
    let proof_w_eval = w_eval;
    #[cfg(feature = "zk")]
    let proof_w_eval = w_eval_masked;
    transcript.append_serde(ABSORB_STAGE2_NEXT_W_EVAL, &proof_w_eval);

    Ok(RootLevelRawOutput {
        #[cfg(feature = "zk")]
        zk_hiding_commitment,
        #[cfg(feature = "zk")]
        y_rings: y_rings_masked,
        #[cfg(not(feature = "zk"))]
        y_rings,
        extension_opening_reduction: None,
        v: quad_eq.v,
        stage1: stage1_proof,
        #[cfg(not(feature = "zk"))]
        stage2_sumcheck_proof,
        #[cfg(feature = "zk")]
        stage2_sumcheck_proof_masked,
        w_commitment_proof,
        w_eval: proof_w_eval,
        next_state: RecursiveProverState {
            w: committed_witness,
            logical_w,
            commitment: committed_commitment,
            hint: committed_hint,
            log_basis: next_log_basis,
            sumcheck_challenges,
            opening: w_eval,
            #[cfg(feature = "zk")]
            zk_hiding,
        },
    })
}

/// Terminal-root analogue of [`prove_root_fold_from_quadratic`] used when the
/// schedule has exactly one fold level (the root is itself the terminal).
///
/// Produces a [`TerminalLevelProof`] with cleartext `final_witness` instead
/// of a `RootLevelRawOutput`. There is no recursive suffix and no
/// `next_state` to thread.
///
/// # Errors
///
/// Returns an error if witness reconstruction does not match the schedule's
/// expected length, ring-switch replay fails, or the stage-2 sumcheck prover
/// fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_terminal_root_fold_from_quadratic<F, C, T, B, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    commitment_rows: &[CyclotomicRing<F, D>],
    lp: &akita_types::LevelParams,
    expected_w_len: usize,
    final_log_basis: u32,
    mut quad_eq: Box<QuadraticEquation<F, { D }>>,
    y_rings: Vec<CyclotomicRing<F, D>>,
    #[cfg(feature = "zk")] y_rings_masked: Vec<CyclotomicRing<F, D>>,
    row_coefficients: Vec<C>,
    #[cfg(feature = "zk")] zk_hiding: &mut ZkHidingProverState<F>,
) -> Result<TerminalLevelProof<F, C>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    C: ExtField<F> + RingSubfieldEncoding<F> + HasUnreducedOps + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
    B: ProverComputeBackend<F>,
{
    let terminal_layout = terminal_witness_segment_layout(
        lp,
        quad_eq.claim_to_point().len(),
        quad_eq.num_public_rows(),
    )?;
    let logical_w = ring_switch_build_w::<F, B, { D }>(&mut quad_eq, backend, prepared, lp)?;
    if logical_w.len() != expected_w_len {
        return Err(AkitaError::InvalidSetup(format!(
            "scheduled root next-w length did not match runtime witness: expected={expected_w_len}, actual={}",
            logical_w.len()
        )));
    }
    let final_witness = DirectWitnessProof::PackedDigits(
        PackedDigits::from_i8_digits_with_min_bits(logical_w.as_i8_digits(), final_log_basis),
    );

    let rs = ring_switch_finalize_terminal_with_gamma::<F, C, T, { D }>(
        &quad_eq,
        expanded,
        transcript,
        &logical_w,
        &final_witness,
        terminal_layout,
        lp,
        &row_coefficients,
    )?;

    // Terminal layout: the D-block is omitted, so the relation claim sums no
    // `v` rows. `quad_eq.v` is constructed as an empty vector under
    // `MRowLayout::Terminal`; pass `&[]` here for symmetry with the verifier.
    let relation_claim = relation_claim_from_rows_extension::<F, C, D>(
        &rs.tau1,
        rs.alpha,
        &[],
        commitment_rows,
        &y_rings,
    )?;
    #[cfg(feature = "zk")]
    let relation_claim_public = relation_claim_from_rows_extension::<F, C, D>(
        &rs.tau1,
        rs.alpha,
        &[],
        commitment_rows,
        &y_rings_masked,
    )?;

    let RingSwitchOutput {
        w_evals_compact,
        live_x_cols,
        m_evals_x,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau0: _,
        tau1: _,
        b,
        alpha: _,
    } = rs;

    let r_stage1 = vec![C::zero(); col_bits + ring_bits];
    #[cfg(feature = "zk")]
    let stage2_round_pads = zk_hiding.take_compressed_rounds::<C>(col_bits + ring_bits, 3)?;
    #[cfg(feature = "zk")]
    let stage2_sumcheck_proof_masked = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck_terminal_root").entered();
        let mut stage2_prover = AkitaStage2Prover::new(
            C::zero(),
            w_evals_compact,
            &r_stage1,
            C::zero(),
            b,
            alpha_evals_y,
            m_evals_x,
            live_x_cols,
            col_bits,
            ring_bits,
            relation_claim,
        )?;
        let (stage2_sumcheck_proof_masked, _sumcheck_challenges) = stage2_prover
            .prove_zk::<F, T, _>(
                relation_claim_public,
                transcript,
                |tr| sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND),
                stage2_round_pads,
            )?;
        stage2_sumcheck_proof_masked
    };
    #[cfg(not(feature = "zk"))]
    let stage2_sumcheck = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck_terminal_root").entered();
        let mut stage2_prover = AkitaStage2Prover::new(
            C::zero(),
            w_evals_compact,
            &r_stage1,
            C::zero(),
            b,
            alpha_evals_y,
            m_evals_x,
            live_x_cols,
            col_bits,
            ring_bits,
            relation_claim,
        )?;
        let (stage2_sumcheck, _sumcheck_challenges, _stage2_final_claim) = stage2_prover
            .prove::<F, T, _>(transcript, |tr| {
                sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND)
            })?;
        stage2_sumcheck
    };

    Ok(
        TerminalLevelProof::new_with_extension_opening_reduction::<D>(
            #[cfg(not(feature = "zk"))]
            y_rings,
            #[cfg(feature = "zk")]
            y_rings_masked,
            None,
            #[cfg(not(feature = "zk"))]
            stage2_sumcheck,
            #[cfg(feature = "zk")]
            stage2_sumcheck_proof_masked,
            final_witness,
        ),
    )
}
