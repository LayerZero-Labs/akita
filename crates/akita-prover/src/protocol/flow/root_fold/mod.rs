mod relation;
pub use relation::{prove_root_fold_from_ring_relation, prove_terminal_root_fold_from_ring_relation};

mod eval;
mod finish;

use eval::{eval_extension_reduction_post_transform, evaluate_root_claims_at_prepared_points};
use finish::{finish_root_fold_with_prepared_openings, finish_terminal_root_fold_with_prepared_openings};

pub(in crate::protocol::flow) use eval::evaluate_recursive_witness_at_multiplier_point;

use super::*;

/// Prove the folded root level using already-selected root and next-level
/// parameters.
///
/// The caller owns schedule/config selection and passes the expected next
/// recursive witness length, next digit basis, and commitment policy for that
/// witness. This function owns root polynomial folding, public root transcript
/// setup, root ring-relation construction, and the folded-root prover
/// mechanics.
///
/// # Errors
///
/// Returns an error if root inputs are malformed, polynomial folding or
/// ring-relation construction fails, or the folded-root prover fails.
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
    setup_contribution_mode: SetupContributionMode,
    commit_w_for_next: CommitW,
) -> Result<RootLevelRawOutput<F, C, D>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + FromPrimitiveInt
        + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: RingSubfieldEncoding<F> + MulBaseUnreduced<F>,
    C: RingSubfieldEncoding<F>
        + ExtField<E>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
    P: RootProvePoly<F, D>,
    B: ProverComputeBackend<F> + RootProveBackend<F, P, E, C, D>,
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
        Some(
            prepare_root_extension_opening_reduction::<F, E, C, P, B, D>(
                backend,
                polys,
                incidence_summary,
                claim_points,
            )?,
        )
    };

    if let Some(prepared_reduction) = extension_reduction_prepare {
        let openings = prepared_reduction.openings.clone();
        append_claim_values_to_transcript::<F, E, T>(&openings, transcript);
        let row_coefficients =
            sample_public_row_coefficients::<F, C, T>(incidence_summary, transcript)?;
        let row_coefficient_rings = row_coefficient_rings::<F, C, D>(&row_coefficients)?;
        let reduction = prove_prepared_root_extension_opening_reduction::<F, E, C, T, P, B, D>(
            backend,
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
        let post_transform = eval_extension_reduction_post_transform::<F, E, C, T, P, B, D>(
            backend,
            polys,
            incidence_summary,
            root_params,
            basis,
            alpha_bits,
            &row_coefficient_rings,
            reduction,
            transcript,
            #[cfg(feature = "zk")]
            &mut zk_hiding,
        )?;
        let transformed_refs = post_transform
            .transformed_polys
            .iter()
            .collect::<Vec<_>>();

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
            post_transform.prepared_points,
            post_transform.e_folded_by_poly,
            post_transform.y_rings,
            #[cfg(feature = "zk")]
            post_transform.y_rings_masked,
            row_coefficients,
            row_coefficient_rings,
            Some(post_transform.extension_opening_reduction),
            #[cfg(feature = "zk")]
            zk_hiding_commitment,
            #[cfg(feature = "zk")]
            zk_hiding,
            setup_contribution_mode,
        );
    }

    let prepared_points = claim_points
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

    let (per_claim_y_rings, e_folded_by_poly) = evaluate_root_claims_at_prepared_points(
        backend,
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

    let openings = per_claim_y_rings
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
        .collect::<Result<Vec<E>, _>>()?;
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
        &y_rings,
        row_coefficient_rings,
        MRowLayout::WithDBlock,
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

    prove_root_fold_from_ring_relation::<F, C, T, B, D, _>(
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
        instance,
        witness,
        y_rings,
        #[cfg(feature = "zk")]
        y_rings_masked,
        row_coefficients,
        setup_contribution_mode,
        commit_w_for_next,
    )
}

/// Terminal-root analogue of [`prove_root_fold_with_params`] used when the
/// schedule has exactly one fold level (the root is itself the terminal).
///
/// Mirrors the intermediate-root path through claim-incidence absorbs,
/// optional extension-opening reduction, and ring-relation setup, then
/// emits a [`TerminalLevelProof`] via
/// [`prove_terminal_root_fold_from_ring_relation`] instead of a
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
    _setup_contribution_mode: SetupContributionMode,
    #[cfg(feature = "zk")] zk_hiding: &mut ZkHidingProverState<F>,
) -> Result<TerminalLevelProof<F, C>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + FromPrimitiveInt
        + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: RingSubfieldEncoding<F> + MulBaseUnreduced<F>,
    C: RingSubfieldEncoding<F>
        + ExtField<E>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
    P: RootProvePoly<F, D>,
    B: ProverComputeBackend<F> + RootProveBackend<F, P, E, C, D>,
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
        Some(
            prepare_root_extension_opening_reduction::<F, E, C, P, B, D>(
                backend,
                polys,
                incidence_summary,
                claim_points,
            )?,
        )
    };

    if let Some(prepared_reduction) = extension_reduction_prepare {
        let openings = prepared_reduction.openings.clone();
        append_claim_values_to_transcript::<F, E, T>(&openings, transcript);
        let row_coefficients =
            sample_public_row_coefficients::<F, C, T>(incidence_summary, transcript)?;
        let row_coefficient_rings = row_coefficient_rings::<F, C, D>(&row_coefficients)?;
        let reduction = prove_prepared_root_extension_opening_reduction::<F, E, C, T, P, B, D>(
            backend,
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
        let post_transform = eval_extension_reduction_post_transform::<F, E, C, T, P, B, D>(
            backend,
            polys,
            incidence_summary,
            root_params,
            basis,
            alpha_bits,
            &row_coefficient_rings,
            reduction,
            transcript,
            #[cfg(feature = "zk")]
            zk_hiding,
        )?;
        let transformed_refs = post_transform
            .transformed_polys
            .iter()
            .collect::<Vec<_>>();

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
            post_transform.prepared_points,
            post_transform.e_folded_by_poly,
            post_transform.y_rings,
            #[cfg(feature = "zk")]
            post_transform.y_rings_masked,
            row_coefficients,
            row_coefficient_rings,
            Some(post_transform.extension_opening_reduction),
            #[cfg(feature = "zk")]
            zk_hiding,
        );
    }

    let prepared_points = claim_points
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

    let (per_claim_y_rings, e_folded_by_poly) = evaluate_root_claims_at_prepared_points(
        backend,
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

    let openings = per_claim_y_rings
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
        .collect::<Result<Vec<E>, _>>()?;
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
        &y_rings,
        row_coefficient_rings,
        MRowLayout::WithoutDBlock,
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

    prove_terminal_root_fold_from_ring_relation::<F, C, T, B, D>(
        expanded,
        backend,
        prepared,
        transcript,
        commitment_rows,
        root_params,
        expected_w_len,
        final_log_basis,
        instance,
        witness,
        y_rings,
        #[cfg(feature = "zk")]
        y_rings_masked,
        row_coefficients,
        #[cfg(feature = "zk")]
        zk_hiding,
    )
}
