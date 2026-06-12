#[cfg(not(feature = "zk"))]
use super::extension_opening_reduction::ExtensionOpeningReductionVerifier;
use akita_algebra::CyclotomicRing;
#[cfg(feature = "zk")]
use akita_algebra::EqPolynomial;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
};
#[cfg(feature = "zk")]
use akita_r1cs::{
    zk_ext_mask_lc, zk_ext_mask_lc_at, zk_masked_compressed_round_claim_mask,
    zk_row_masks_from_column_masks, ZkR1csLinearCombination, ZkRelationAccumulator,
};
use akita_serialization::AkitaSerialize;
#[cfg(not(feature = "zk"))]
use akita_sumcheck::SumcheckInstanceVerifierExt;
#[cfg(feature = "zk")]
use akita_transcript::labels::ABSORB_SUMCHECK_CLAIM;
use akita_transcript::labels::{
    ABSORB_EVALUATION_CLAIMS, CHALLENGE_SUMCHECK_BATCH, CHALLENGE_SUMCHECK_ROUND,
};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
#[cfg(not(feature = "zk"))]
use akita_types::check_tensor_extension_opening_claim;
use akita_types::{
    prepare_root_opening_point_ext, ring_subfield_packed_extension_opening_point,
    root_extension_opening_partials, tensor_reduction_claim_from_rows,
    tensor_row_partials_from_columns, BasisMode, ClaimIncidenceSummary,
    ExtensionOpeningReductionProof, LevelParams, PreparedRootOpeningPoint, RingSubfieldEncoding,
};
#[cfg(feature = "zk")]
use akita_types::{tensor_equality_factor_eval_at_point, EXTENSION_OPENING_REDUCTION_DEGREE};

pub(super) struct RootEorReplay<F: FieldCore, C: FieldCore, const D: usize> {
    pub(super) prepared_points: Vec<PreparedRootOpeningPoint<F, D>>,
    pub(super) reduction_challenges: Option<Vec<C>>,
    #[cfg(feature = "zk")]
    pub(super) final_relation: Option<(ZkR1csLinearCombination<C>, Vec<C>)>,
}

pub(super) struct RootEorInput<'a, F: FieldCore, E: FieldCore, C: FieldCore, const D: usize> {
    pub(super) extension_opening_reduction: Option<&'a ExtensionOpeningReductionProof<C>>,
    pub(super) y_rings: &'a [CyclotomicRing<F, D>],
    pub(super) claim_points: &'a [&'a [E]],
    pub(super) openings: &'a [E],
    pub(super) row_coefficients: &'a [C],
    pub(super) incidence_summary: &'a ClaimIncidenceSummary,
    pub(super) basis: BasisMode,
    pub(super) root_lp: &'a LevelParams,
}

#[cfg(feature = "zk")]
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_zk_extension_opening_reduction_sumcheck<F, E, T, S>(
    input_claim: E,
    num_rounds: usize,
    proof: &akita_sumcheck::SumcheckProofMasked<E>,
    input_claim_mask: ZkR1csLinearCombination<E>,
    transcript: &mut T,
    mut sample_challenge: S,
    relations: &mut ZkRelationAccumulator<E>,
    hiding_cursor: &mut usize,
) -> Result<(ZkR1csLinearCombination<E>, Vec<E>), AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore + ExtField<F> + AkitaSerialize,
    T: Transcript<F>,
    S: FnMut(&mut T) -> E,
{
    if proof.masked_round_polys.len() != num_rounds {
        return Err(AkitaError::InvalidSize {
            expected: num_rounds,
            actual: proof.masked_round_polys.len(),
        });
    }

    transcript.append_serde(ABSORB_SUMCHECK_CLAIM, &input_claim);
    let mut masked_claim = input_claim;
    let mut claim_mask = input_claim_mask;
    let mut challenges = Vec::with_capacity(num_rounds);
    for masked_poly in &proof.masked_round_polys {
        if masked_poly.degree() > EXTENSION_OPENING_REDUCTION_DEGREE {
            return Err(AkitaError::InvalidInput(format!(
                "extension-opening reduction round poly exceeds degree bound {}",
                EXTENSION_OPENING_REDUCTION_DEGREE
            )));
        }
        transcript.append_serde(akita_transcript::labels::ABSORB_SUMCHECK_ROUND, masked_poly);
        let r_i = sample_challenge(transcript);
        challenges.push(r_i);
        let next_claim_mask = zk_masked_compressed_round_claim_mask::<F, E>(
            &claim_mask,
            &masked_poly.coeffs_except_linear_term,
            r_i,
            hiding_cursor,
        );
        masked_claim = masked_poly.eval_from_hint(&masked_claim, &r_i);
        claim_mask = next_claim_mask;
    }

    Ok((
        relations.push_masked_claim_relation(
            "extension-opening reduction final claim",
            masked_claim,
            &claim_mask,
        ),
        challenges,
    ))
}

pub(super) fn verify_root_eor_and_prepare_points<F, E, C, T, const D: usize>(
    input: RootEorInput<'_, F, E, C, D>,
    transcript: &mut T,
    #[cfg(feature = "zk")] zk_hiding_cursor: &mut usize,
    #[cfg(feature = "zk")] zk_relations: &mut ZkRelationAccumulator<C>,
) -> Result<RootEorReplay<F, C, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: RingSubfieldEncoding<F>,
    C: RingSubfieldEncoding<F>
        + ExtField<E>
        + ExtField<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
{
    let RootEorInput {
        extension_opening_reduction,
        y_rings,
        claim_points,
        openings,
        row_coefficients,
        incidence_summary,
        basis,
        root_lp,
    } = input;
    #[cfg(feature = "zk")]
    let _ = y_rings;
    let alpha_bits = root_lp.ring_dimension.trailing_zeros() as usize;
    // The zk EOR final relation consumes the y-ring opening masks, which are a
    // shared resource with the downstream ring-switch binding, so it stays in
    // this outer flow rather than inside the sumcheck driver. These extras carry
    // `(final_claim_lc, factors_by_point)` for that deferred relation.
    #[cfg(feature = "zk")]
    let mut zk_eor_final: Option<(ZkR1csLinearCombination<C>, Vec<C>)> = None;
    let reduction_check = if let Some(reduction) = extension_opening_reduction {
        if <C as ExtField<F>>::EXT_DEGREE == 1 {
            return Err(AkitaError::InvalidProof);
        }
        if <C as ExtField<F>>::EXT_DEGREE != <E as ExtField<F>>::EXT_DEGREE {
            return Err(AkitaError::InvalidProof);
        }
        let (split_bits, width) = {
            let width = <C as ExtField<F>>::EXT_DEGREE;
            if width == 0 || !width.is_power_of_two() {
                return Err(AkitaError::InvalidProof);
            }
            (width.trailing_zeros() as usize, width)
        };
        let expected_partials =
            root_extension_opening_partials(width, incidence_summary.num_claims());
        if split_bits > incidence_summary.num_vars()
            || reduction.partials.len() != expected_partials
        {
            return Err(AkitaError::InvalidProof);
        }
        let padded_points = claim_points
            .iter()
            .map(|point| {
                if point.len() > incidence_summary.num_vars() {
                    return Err(AkitaError::InvalidProof);
                }
                let mut lifted = point.iter().copied().map(C::lift_base).collect::<Vec<_>>();
                lifted.resize(incidence_summary.num_vars(), C::zero());
                Ok(lifted)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut input_claim = C::zero();
        #[cfg(feature = "zk")]
        let mut input_claim_mask = ZkR1csLinearCombination::zero();
        for (claim_idx, opening) in openings
            .iter()
            .copied()
            .enumerate()
            .take(incidence_summary.num_claims())
        {
            let point_idx = incidence_summary.claim_to_point()[claim_idx];
            let partial_start = claim_idx * width;
            let partial_end = partial_start + width;
            let partials = &reduction.partials[partial_start..partial_end];
            #[cfg(feature = "zk")]
            let partial_masks = (0..width)
                .map(|_| zk_ext_mask_lc::<F, C>(zk_hiding_cursor))
                .collect::<Vec<_>>();
            #[cfg(not(feature = "zk"))]
            check_tensor_extension_opening_claim::<F, C>(
                &padded_points[point_idx],
                C::lift_base(opening),
                partials,
            )?;
            #[cfg(feature = "zk")]
            {
                let head_weights =
                    EqPolynomial::<C>::evals(&padded_points[point_idx][..split_bits])?;
                let mut residual = ZkR1csLinearCombination::constant(-C::lift_base(opening));
                for ((&partial, mask), weight) in
                    partials.iter().zip(partial_masks.iter()).zip(head_weights)
                {
                    let true_partial = ZkRelationAccumulator::unmask_lc(partial, mask);
                    residual.add_scaled(weight, &true_partial);
                }
                zk_relations.push_r1cs(
                    "root extension-opening partial claim",
                    residual,
                    ZkR1csLinearCombination::one(),
                    ZkR1csLinearCombination::zero(),
                )?;
            }
            for partial in partials {
                append_ext_field::<F, C, T>(transcript, ABSORB_EVALUATION_CLAIMS, partial);
            }
        }
        let eta = (0..split_bits)
            .map(|_| sample_ext_challenge::<F, C, T>(transcript, CHALLENGE_SUMCHECK_BATCH))
            .collect::<Vec<_>>();
        for (claim_idx, &row_coefficient) in row_coefficients
            .iter()
            .enumerate()
            .take(incidence_summary.num_claims())
        {
            let partial_start = claim_idx * width;
            let partial_end = partial_start + width;
            let row_partials = tensor_row_partials_from_columns::<F, C>(
                &reduction.partials[partial_start..partial_end],
            )?;
            let claim = tensor_reduction_claim_from_rows::<F, C>(&row_partials, &eta)?;
            input_claim += row_coefficient * claim;
            #[cfg(feature = "zk")]
            {
                let mut partial_masks = Vec::with_capacity(width);
                for offset in 0..width {
                    let mask_start = partial_start + offset;
                    let mask = zk_ext_mask_lc_at::<F, C>(
                        *zk_hiding_cursor
                            - reduction.partials.len() * <C as ExtField<F>>::EXT_DEGREE
                            + mask_start * <C as ExtField<F>>::EXT_DEGREE,
                    );
                    partial_masks.push(mask);
                }
                let row_masks = zk_row_masks_from_column_masks::<F, C>(&partial_masks)?;
                for (weight, row_mask) in EqPolynomial::<C>::evals(&eta)?.into_iter().zip(row_masks)
                {
                    input_claim_mask.add_scaled(row_coefficient * weight, &row_mask);
                }
            }
        }
        #[cfg(not(feature = "zk"))]
        {
            let rows = incidence_summary
                .public_rows()
                .iter()
                .enumerate()
                .map(|(row_idx, row)| {
                    let y_ring = y_rings.get(row_idx).ok_or(AkitaError::InvalidProof)?;
                    let point = padded_points
                        .get(row.point_idx())
                        .ok_or(AkitaError::InvalidProof)?;
                    let point_tail = point
                        .get(split_bits..)
                        .ok_or(AkitaError::InvalidProof)?
                        .to_vec();
                    Ok((y_ring, point_tail))
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            let eor_verifier = ExtensionOpeningReductionVerifier::<F, C, D>::new(
                incidence_summary.num_vars() - split_bits,
                input_claim,
                eta,
                rows,
                Box::new(
                    move |rho: &[C]| -> Result<CyclotomicRing<F, D>, AkitaError> {
                        let protocol_point = ring_subfield_packed_extension_opening_point::<F, C, D>(
                            rho.len(),
                            rho,
                        )?;
                        Ok(prepare_root_opening_point_ext::<F, C, C, D>(
                            &protocol_point,
                            basis,
                            root_lp,
                            alpha_bits,
                        )?
                        .inner_reduction)
                    },
                ),
            );
            let rho = eor_verifier.verify::<F, T, _>(&reduction.sumcheck, transcript, |tr| {
                sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND)
            })?;
            Some(rho)
        }
        #[cfg(feature = "zk")]
        {
            let (final_claim_lc, challenges) =
                verify_zk_extension_opening_reduction_sumcheck::<F, C, T, _>(
                    input_claim,
                    incidence_summary.num_vars() - split_bits,
                    &reduction.sumcheck_proof_masked,
                    input_claim_mask,
                    transcript,
                    |tr| sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND),
                    zk_relations,
                    zk_hiding_cursor,
                )?;
            let factors_by_point = padded_points
                .iter()
                .map(|point| {
                    tensor_equality_factor_eval_at_point::<F, C>(
                        &point[split_bits..],
                        &eta,
                        &challenges,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            zk_eor_final = Some((final_claim_lc, factors_by_point));
            Some(challenges)
        }
    } else {
        None
    };

    let prepared_points = if let Some(rho) = &reduction_check {
        let protocol_point =
            ring_subfield_packed_extension_opening_point::<F, C, D>(rho.len(), rho)?;
        let prepared = prepare_root_opening_point_ext::<F, C, C, D>(
            &protocol_point,
            basis,
            root_lp,
            alpha_bits,
        )?;
        vec![prepared; incidence_summary.num_points()]
    } else {
        claim_points
            .iter()
            .map(|opening_point| {
                prepare_root_opening_point_ext::<F, E, C, D>(
                    opening_point,
                    basis,
                    root_lp,
                    alpha_bits,
                )
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    Ok(RootEorReplay {
        prepared_points,
        reduction_challenges: reduction_check,
        #[cfg(feature = "zk")]
        final_relation: zk_eor_final,
    })
}
