use super::super::*;
use super::eval::{
    eval_extension_reduction_post_transform, evaluate_root_claims_at_prepared_points,
    ExtensionReductionPostTransform,
};

pub(super) fn validate_root_fold_inputs(
    num_polys: usize,
    incidence_summary: &ClaimIncidenceSummary,
    num_claim_points: usize,
    num_commitments: usize,
    num_hints: usize,
) -> Result<(), AkitaError> {
    let claim_to_point = incidence_summary.claim_to_point();
    let num_claims = incidence_summary.num_claims();

    if num_claim_points == 0
        || num_claim_points != incidence_summary.num_points()
        || claim_to_point.len() != num_claims
        || num_polys != num_claims
        || num_commitments != incidence_summary.num_points()
        || num_hints != incidence_summary.num_points()
    {
        return Err(AkitaError::InvalidInput(
            "invalid root-level inputs".to_string(),
        ));
    }
    if claim_to_point
        .iter()
        .any(|&point_idx| point_idx >= num_claim_points)
    {
        return Err(AkitaError::InvalidInput(
            "root-level claim-to-point index out of range".to_string(),
        ));
    }
    Ok(())
}

pub(super) fn trace_root_fold_entry(
    trace_name: &'static str,
    num_claims: usize,
    num_points: usize,
) {
    let x: u8 = 0;
    tracing::trace!(
        stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
        level = 0usize,
        num_claims,
        num_points,
        trace_name
    );
}

pub(super) fn append_root_fold_transcript_prefix<F, E, T, const D: usize>(
    incidence_summary: &ClaimIncidenceSummary,
    commitments: &[RingCommitment<F, D>],
    claim_points: &[&[E]],
    transcript: &mut T,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
    T: Transcript<F>,
{
    append_claim_incidence_shape_to_transcript::<F, T>(incidence_summary, transcript)?;
    append_batched_commitments_to_transcript(commitments, transcript);
    append_claim_points_to_transcript::<F, E, T>(claim_points, transcript);
    Ok(())
}

pub(super) fn maybe_prepare_root_extension_reduction<F, E, C, P, B, const D: usize>(
    backend: &B,
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    claim_points: &[&[E]],
    num_vars: usize,
) -> Result<Option<PreparedRootExtensionOpeningReduction<E, C>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: RingSubfieldEncoding<F> + MulBaseUnreduced<F>,
    C: RingSubfieldEncoding<F> + ExtField<E>,
    P: RootTensorSource<F, D>,
    B: for<'a> TensorProjectionBatchKernel<P::TensorBatchView<'a>, F, E, D>,
{
    if !root_tensor_projection_enabled::<F, E, C, D>(num_vars) {
        return Ok(None);
    }
    Ok(Some(prepare_root_extension_opening_reduction::<
        F,
        E,
        C,
        P,
        B,
        D,
    >(backend, polys, incidence_summary, claim_points)?))
}

pub(super) struct RootExtensionReductionPublicPhase<F: FieldCore, C: FieldCore, const D: usize> {
    pub post_transform: ExtensionReductionPostTransform<F, C, D>,
    pub row_coefficients: Vec<C>,
    pub row_coefficient_rings: Vec<CyclotomicRing<F, D>>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prove_root_extension_reduction_public_phase<F, E, C, T, P, B, const D: usize>(
    backend: &B,
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    root_params: &LevelParams,
    basis: BasisMode,
    alpha_bits: usize,
    prepared_reduction: PreparedRootExtensionOpeningReduction<E, C>,
    transcript: &mut T,
    #[cfg(feature = "zk")] zk_hiding: &mut ZkHidingProverState<F>,
) -> Result<RootExtensionReductionPublicPhase<F, C, D>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: RingSubfieldEncoding<F> + MulBaseUnreduced<F>,
    C: RingSubfieldEncoding<F>
        + ExtField<E>
        + ExtField<F>
        + FieldCore
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
    P: RootProvePoly<F, D>,
    B: RootProveFlowBackend<F, P, E, C, D>,
{
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
    Ok(RootExtensionReductionPublicPhase {
        post_transform,
        row_coefficients,
        row_coefficient_rings,
    })
}

pub(super) struct RootFoldDirectPublicPhase<F: FieldCore, C: FieldCore, const D: usize> {
    pub y_rings: Vec<CyclotomicRing<F, D>>,
    #[cfg(feature = "zk")]
    pub y_rings_masked: Vec<CyclotomicRing<F, D>>,
    pub row_coefficients: Vec<C>,
    pub instance: RingRelationInstance<F, D>,
    pub witness: RingRelationWitness<F, D>,
}

pub(super) fn batched_root_commitment_rows<'a, F: FieldCore, const D: usize>(
    commitments: &'a [RingCommitment<F, D>],
    commitment_rows_owned: &'a Option<Vec<CyclotomicRing<F, D>>>,
) -> &'a [CyclotomicRing<F, D>] {
    match commitment_rows_owned {
        Some(v) => v.as_slice(),
        None => commitments[0].u.as_slice(),
    }
}

pub(super) fn flatten_root_commitment_rows_if_needed<F: FieldCore, const D: usize>(
    commitments: &[RingCommitment<F, D>],
) -> Option<Vec<CyclotomicRing<F, D>>> {
    if commitments.len() == 1 {
        None
    } else {
        Some(flatten_batched_commitment_rows(commitments))
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_root_fold_direct_public_phase<F, E, C, T, P, B, const D: usize>(
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    claim_points: &[&[E]],
    commitments: &[RingCommitment<F, D>],
    hints: Vec<AkitaCommitmentHint<F, D>>,
    root_params: &LevelParams,
    basis: BasisMode,
    alpha_bits: usize,
    m_row_layout: MRowLayout,
    transcript: &mut T,
    #[cfg(feature = "zk")] zk_hiding: &mut ZkHidingProverState<F>,
) -> Result<RootFoldDirectPublicPhase<F, C, D>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: RingSubfieldEncoding<F> + MulBaseUnreduced<F>,
    C: RingSubfieldEncoding<F> + ExtField<E> + FieldCore,
    T: Transcript<F>,
    P: RootProvePoly<F, D>,
    B: RootProveFlowBackend<F, P, E, C, D>,
{
    let claim_to_point = incidence_summary.claim_to_point();

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
    append_root_fold_y_rings_to_transcript(transcript, &y_rings_masked);
    #[cfg(not(feature = "zk"))]
    append_root_fold_y_rings_to_transcript(transcript, &y_rings);

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
        m_row_layout,
    )?;

    Ok(RootFoldDirectPublicPhase {
        y_rings,
        #[cfg(feature = "zk")]
        y_rings_masked,
        row_coefficients,
        instance,
        witness,
    })
}

fn append_root_fold_y_rings_to_transcript<F, T, const D: usize>(
    transcript: &mut T,
    y_rings_for_transcript: &[CyclotomicRing<F, D>],
) where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    for y_ring in y_rings_for_transcript {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }
}
