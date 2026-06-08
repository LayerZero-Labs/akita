use super::super::*;

pub(super) fn evaluate_root_claims_at_prepared_points<F, Q, B, const D: usize>(
    backend: &B,
    polys: &[&Q],
    claim_to_point: &[usize],
    prepared_points: &[PreparedRootOpeningPoint<F, D>],
    block_len: usize,
) -> Result<RootClaimEvaluations<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
    Q: RootOpeningSource<F, D>,
    B: for<'a> OpeningFoldKernel<Q::OpeningView<'a>, F, D>,
{
    let _span = tracing::info_span!("root_evaluate_claims", num_claims = polys.len()).entered();
    let mut per_claim_y_rings = Vec::with_capacity(polys.len());
    let mut e_folded_by_poly = Vec::with_capacity(polys.len());
    for (poly, &point_idx) in polys.iter().zip(claim_to_point.iter()) {
        let prepared_point = &prepared_points[point_idx];
        let (y_ring, e_folded) = poly_kernels::evaluate_at_multiplier_point(
            backend,
            poly,
            &prepared_point.ring_multiplier_point,
            block_len,
        )?;
        per_claim_y_rings.push(y_ring);
        e_folded_by_poly.push(e_folded);
    }
    Ok((per_claim_y_rings, e_folded_by_poly))
}

pub(super) struct ExtensionReductionPostTransform<F: FieldCore, C: FieldCore, const D: usize> {
    pub(super) prepared_points: Vec<PreparedRootOpeningPoint<F, D>>,
    pub(super) transformed_polys: Vec<RootTensorProjectionPoly<F, D>>,
    pub(super) e_folded_by_poly: Vec<Vec<CyclotomicRing<F, D>>>,
    pub(super) y_rings: Vec<CyclotomicRing<F, D>>,
    #[cfg(feature = "zk")]
    pub(super) y_rings_masked: Vec<CyclotomicRing<F, D>>,
    pub(super) extension_opening_reduction: ExtensionOpeningReductionProof<C>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn eval_extension_reduction_post_transform<F, E, C, T, P, B, const D: usize>(
    backend: &B,
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    root_params: &LevelParams,
    basis: BasisMode,
    alpha_bits: usize,
    row_coefficient_rings: &[CyclotomicRing<F, D>],
    reduction: RootExtensionOpeningReduction<C>,
    transcript: &mut T,
    #[cfg(feature = "zk")] zk_hiding: &mut ZkHidingProverState<F>,
) -> Result<ExtensionReductionPostTransform<F, C, D>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: RingSubfieldEncoding<F>,
    C: RingSubfieldEncoding<F> + ExtField<E> + ExtField<F> + FieldCore,
    T: Transcript<F>,
    P: RootProvePoly<F, D>,
    B: RootProveFlowBackend<F, P, E, C, D>,
{
    let claim_to_point = incidence_summary.claim_to_point();
    let protocol_point = {
        let _span = tracing::info_span!("root_extension_protocol_point").entered();
        ring_subfield_packed_extension_opening_point::<F, C, D>(
            reduction.rho.len(),
            &reduction.rho,
        )?
    };
    let prepared_protocol_point = {
        let _span = tracing::info_span!("root_extension_prepare_protocol_point").entered();
        prepare_root_opening_point_ext::<F, C, C, D>(
            &protocol_point,
            basis,
            root_params,
            alpha_bits,
        )?
    };
    let prepared_points = vec![prepared_protocol_point; incidence_summary.num_points()];
    let transformed_polys = {
        let _span = tracing::info_span!("root_extension_transform_polys", num_claims = polys.len())
            .entered();
        cfg_iter!(polys)
            .map(|poly| poly_kernels::tensor_root_projection::<F, P, C, B, D>(backend, poly))
            .collect::<Result<Vec<RootTensorProjectionPoly<F, D>>, _>>()?
    };
    let transformed_refs = transformed_polys.iter().collect::<Vec<_>>();

    let (per_claim_y_rings, e_folded_by_poly) =
        evaluate_root_claims_at_prepared_points::<F, RootTensorProjectionPoly<F, D>, B, D>(
            backend,
            &transformed_refs,
            claim_to_point,
            &prepared_points,
            root_params.block_len,
        )?;
    let y_rings =
        combine_root_y_rings::<F, D>(&per_claim_y_rings, incidence_summary, row_coefficient_rings)?;
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

    Ok(ExtensionReductionPostTransform {
        prepared_points,
        transformed_polys,
        e_folded_by_poly,
        y_rings,
        #[cfg(feature = "zk")]
        y_rings_masked,
        extension_opening_reduction: reduction.proof,
    })
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
