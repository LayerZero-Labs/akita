use super::*;

pub(in crate::protocol::flow) struct PreparedRootExtensionOpeningReduction<
    E: FieldCore,
    C: FieldCore,
> {
    pub(in crate::protocol::flow) openings: Vec<E>,
    pub(in crate::protocol::flow) partials: Vec<C>,
    pub(in crate::protocol::flow) row_partials_by_claim: Vec<Vec<C>>,
    pub(in crate::protocol::flow) padded_points: Vec<Vec<C>>,
    pub(in crate::protocol::flow) split_bits: usize,
}

pub(in crate::protocol::flow) struct RootExtensionOpeningReduction<C: FieldCore> {
    pub(in crate::protocol::flow) proof: ExtensionOpeningReductionProof<C>,
    pub(in crate::protocol::flow) rho: Vec<C>,
    pub(in crate::protocol::flow) final_claim: C,
    pub(in crate::protocol::flow) factors_by_point: Vec<C>,
}

fn lift_claim_point<E, C>(point: &[E], num_vars: usize) -> Result<Vec<C>, AkitaError>
where
    E: FieldCore,
    C: ExtField<E>,
{
    if point.len() > num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: num_vars,
            actual: point.len(),
        });
    }
    let mut lifted = point.iter().copied().map(C::lift_base).collect::<Vec<_>>();
    lifted.resize(num_vars, C::zero());
    Ok(lifted)
}

fn prepare_root_extension_opening_reduction<F, E, C, P, const D: usize>(
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    claim_points: &[&[E]],
) -> Result<PreparedRootExtensionOpeningReduction<E, C>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: RingSubfieldEncoding<F> + MulBaseUnreduced<F>,
    C: RingSubfieldEncoding<F> + ExtField<E>,
    P: AkitaPolyOps<F, D>,
{
    let _span = tracing::info_span!(
        "prepare_root_extension_opening_reduction",
        num_claims = incidence_summary.num_claims(),
        num_points = incidence_summary.num_points(),
        num_vars = incidence_summary.num_vars()
    )
    .entered();
    if <C as ExtField<F>>::EXT_DEGREE != <E as ExtField<F>>::EXT_DEGREE {
        return Err(AkitaError::InvalidInput(
            "root extension-opening reduction currently requires claim and challenge fields to have the same base degree"
                .to_string(),
        ));
    }
    let num_vars = incidence_summary.num_vars();
    let (split_bits, width) = tensor_opening_split::<F, E>()?;
    if split_bits > num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: split_bits,
            actual: num_vars,
        });
    }
    if polys.len() != incidence_summary.num_claims()
        || claim_points.len() != incidence_summary.num_points()
    {
        return Err(AkitaError::InvalidInput(
            "root extension-opening reduction input lengths do not match".to_string(),
        ));
    }

    let padded_points_e = claim_points
        .iter()
        .map(|point| {
            if point.len() > num_vars {
                return Err(AkitaError::InvalidPointDimension {
                    expected: num_vars,
                    actual: point.len(),
                });
            }
            let mut padded = point.to_vec();
            padded.resize(num_vars, E::zero());
            Ok(padded)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let padded_points = claim_points
        .iter()
        .map(|point| lift_claim_point::<E, C>(point, num_vars))
        .collect::<Result<Vec<_>, _>>()?;

    let mut openings = Vec::with_capacity(incidence_summary.num_claims());
    let mut partials = Vec::with_capacity(root_extension_opening_partials(
        width,
        incidence_summary.num_claims(),
    ));
    let mut row_partials_by_claim = Vec::with_capacity(incidence_summary.num_claims());
    {
        let _span =
            tracing::info_span!("root_extension_prepare_partials", width, split_bits).entered();
        let mut column_partials_by_claim = (0..incidence_summary.num_claims())
            .map(|_| None)
            .collect::<Vec<_>>();
        for (point_idx, logical_point) in padded_points_e
            .iter()
            .enumerate()
            .take(incidence_summary.num_points())
        {
            let claim_indices = incidence_summary
                .claim_to_point()
                .iter()
                .enumerate()
                .filter_map(|(claim_idx, &claim_point_idx)| {
                    (claim_point_idx == point_idx).then_some(claim_idx)
                })
                .collect::<Vec<_>>();
            let point_polys = claim_indices
                .iter()
                .map(|&claim_idx| polys[claim_idx])
                .collect::<Vec<_>>();
            let point_partials = <P as AkitaPolyOps<F, D>>::tensor_extension_column_partials_batch::<
                E,
            >(&point_polys, logical_point)?;
            if point_partials.len() != claim_indices.len() {
                return Err(AkitaError::InvalidSize {
                    expected: claim_indices.len(),
                    actual: point_partials.len(),
                });
            }
            for (claim_idx, column_partials) in claim_indices.into_iter().zip(point_partials) {
                column_partials_by_claim[claim_idx] = Some(column_partials);
            }
        }
        for (claim_idx, column_partials) in column_partials_by_claim.into_iter().enumerate() {
            let point_idx = incidence_summary.claim_to_point()[claim_idx];
            let logical_point = &padded_points_e[point_idx];
            let column_partials = column_partials.ok_or_else(|| {
                AkitaError::InvalidInput(
                    "missing root extension-opening column partials for claim".to_string(),
                )
            })?;
            let opening =
                tensor_logical_claim_from_partials::<F, E>(logical_point, &column_partials)?;
            let row_partials = tensor_row_partials_from_columns::<F, E>(&column_partials)?
                .into_iter()
                .map(C::lift_base)
                .collect::<Vec<_>>();
            partials.extend(column_partials.into_iter().map(C::lift_base));
            openings.push(opening);
            row_partials_by_claim.push(row_partials);
        }
    }

    Ok(PreparedRootExtensionOpeningReduction {
        openings,
        partials,
        row_partials_by_claim,
        padded_points,
        split_bits,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::flow) fn prove_root_extension_opening_reduction<
    F,
    E,
    C,
    T,
    P,
    const D: usize,
>(
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    claim_points: &[&[E]],
    transcript: &mut T,
    #[cfg(feature = "zk")] zk_hiding: &mut ZkHidingProverState<F>,
) -> Result<(RootExtensionOpeningReduction<C>, Vec<C>), AkitaError>
where
    F: FieldCore + CanonicalField,
    E: RingSubfieldEncoding<F> + MulBaseUnreduced<F>,
    C: RingSubfieldEncoding<F>
        + ExtField<E>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + AkitaSerialize,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D>,
{
    let _span = tracing::info_span!(
        "prove_root_extension_opening_reduction",
        num_claims = incidence_summary.num_claims(),
        num_points = incidence_summary.num_points()
    )
    .entered();
    let prepared = prepare_root_extension_opening_reduction::<F, E, C, P, D>(
        polys,
        incidence_summary,
        claim_points,
    )?;
    append_claim_values_to_transcript::<F, E, T>(&prepared.openings, transcript);
    let row_coefficients =
        sample_public_row_coefficients::<F, C, T>(incidence_summary, transcript)?;
    let PreparedRootExtensionOpeningReduction {
        openings: _,
        partials,
        row_partials_by_claim,
        padded_points,
        split_bits,
    } = prepared;
    let width = <C as ExtField<F>>::EXT_DEGREE;
    #[cfg(feature = "zk")]
    let (partial_masks, sumcheck_pads) = zk_hiding.take_extension_opening_reduction_pads::<C>(
        partials.len(),
        incidence_summary.num_vars() - split_bits,
    )?;
    #[cfg(feature = "zk")]
    let proof_partials = partials
        .iter()
        .copied()
        .zip(partial_masks)
        .map(|(partial, mask)| partial + mask)
        .collect::<Vec<_>>();
    #[cfg(not(feature = "zk"))]
    let proof_partials = partials.clone();
    let proof_row_partials_by_claim = proof_partials
        .chunks_exact(width)
        .map(tensor_row_partials_from_columns::<F, C>)
        .collect::<Result<Vec<_>, _>>()?;
    {
        let _span = tracing::debug_span!(
            "root_extension_absorb_partials",
            partials_len = proof_partials.len()
        )
        .entered();
        for partial in &proof_partials {
            append_ext_field::<F, C, T>(transcript, ABSORB_EVALUATION_CLAIMS, partial);
        }
    }
    let eta = (0..split_bits)
        .map(|_| sample_ext_challenge::<F, C, T>(transcript, CHALLENGE_SUMCHECK_BATCH))
        .collect::<Vec<_>>();
    let input_claim = {
        let _span = tracing::debug_span!("root_extension_input_claim").entered();
        proof_row_partials_by_claim.iter().enumerate().try_fold(
            C::zero(),
            |acc, (claim_idx, row_partials)| {
                tensor_reduction_claim_from_rows::<F, C>(row_partials, &eta)
                    .map(|claim| acc + row_coefficients[claim_idx] * claim)
            },
        )?
    };
    let true_input_claim = row_partials_by_claim.iter().enumerate().try_fold(
        C::zero(),
        |acc, (claim_idx, row_partials)| {
            tensor_reduction_claim_from_rows::<F, C>(row_partials, &eta)
                .map(|claim| acc + row_coefficients[claim_idx] * claim)
        },
    )?;
    #[cfg(not(feature = "zk"))]
    debug_assert_eq!(input_claim, true_input_claim);
    let sparse_terms = {
        let _span = tracing::info_span!("root_extension_sparse_terms").entered();
        let mut terms = Vec::with_capacity(incidence_summary.num_points());
        let mut all_sparse = true;
        for (point_idx, padded_point) in padded_points
            .iter()
            .enumerate()
            .take(incidence_summary.num_points())
        {
            let tail_point = &padded_point[split_bits..];
            let mut point_polys = Vec::new();
            let mut point_coeffs = Vec::new();
            for (claim_idx, poly) in polys.iter().enumerate() {
                if incidence_summary.claim_to_point()[claim_idx] == point_idx {
                    point_polys.push(*poly);
                    point_coeffs.push(row_coefficients[claim_idx]);
                }
            }
            let witness_evals = {
                let _span = tracing::info_span!(
                    "root_extension_sparse_witnesses",
                    point_idx,
                    num_terms = point_polys.len()
                )
                .entered();
                <P as AkitaPolyOps<F, D>>::tensor_packed_extension_sparse_linear_combination::<C>(
                    &point_polys,
                    &point_coeffs,
                )?
            };
            let Some(witness_evals) = witness_evals else {
                all_sparse = false;
                break;
            };
            let lazy_rounds = tail_point.len().min(SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS);
            if lazy_rounds == 0 {
                let factor_evals = {
                    let _span = tracing::debug_span!(
                        "root_extension_factor_evals",
                        point_idx,
                        tail_vars = tail_point.len()
                    )
                    .entered();
                    tensor_equality_factor_evals::<F, C>(tail_point, &eta)?
                };
                terms.push(ExtensionOpeningReductionTerm::new_sparse(
                    witness_evals,
                    factor_evals,
                    C::one(),
                )?);
            } else {
                let _span = tracing::debug_span!(
                    "root_extension_lazy_tensor_factor",
                    point_idx,
                    tail_vars = tail_point.len(),
                    lazy_rounds
                )
                .entered();
                terms.push(
                    ExtensionOpeningReductionTerm::new_sparse_tensor_factor::<F>(
                        witness_evals,
                        tail_point.to_vec(),
                        eta.clone(),
                        C::one(),
                        lazy_rounds,
                    )?,
                );
            }
        }
        if all_sparse {
            Some(terms)
        } else {
            None
        }
    };
    let terms = if let Some(terms) = sparse_terms {
        terms
    } else {
        let mut terms = Vec::with_capacity(incidence_summary.num_claims());
        {
            let _span = tracing::info_span!("root_extension_dense_terms").entered();
            for (claim_idx, poly) in polys.iter().enumerate() {
                let point_idx = incidence_summary.claim_to_point()[claim_idx];
                let tail_point = &padded_points[point_idx][split_bits..];
                let factor_evals = tensor_equality_factor_evals::<F, C>(tail_point, &eta)?;
                let witness_evals = poly.tensor_packed_extension_evals::<C>()?;
                terms.push(ExtensionOpeningReductionTerm::new(
                    witness_evals,
                    factor_evals,
                    row_coefficients[claim_idx],
                )?);
            }
        }
        terms
    };
    let prover = {
        let _span = tracing::info_span!("root_extension_reduction_prover_new").entered();
        ExtensionOpeningReductionProver::new(terms, true_input_claim)?
    };
    let mut prover = prover;
    let _eor_sumcheck_span =
        tracing::info_span!("extension_opening_reduction_sumcheck", path = "root").entered();
    #[cfg(not(feature = "zk"))]
    let (sumcheck, rho, final_claim) = prover.prove::<F, T, _>(transcript, |tr| {
        sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND)
    })?;
    #[cfg(feature = "zk")]
    let (sumcheck_proof_masked, rho) = prover.prove_zk::<F, T, _>(
        input_claim,
        transcript,
        |tr| sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND),
        sumcheck_pads,
    )?;
    let final_terms = prover.final_terms().ok_or_else(|| {
        AkitaError::InvalidInput(
            "root extension-opening reduction has not reached a final point".to_string(),
        )
    })?;
    let expected_final = final_terms
        .into_iter()
        .fold(C::zero(), |acc, (coeff, witness, factor)| {
            acc + coeff * witness * factor
        });
    #[cfg(feature = "zk")]
    let final_claim = expected_final;
    #[cfg(not(feature = "zk"))]
    if final_claim != expected_final {
        return Err(AkitaError::InvalidInput(
            "root extension-opening reduction final oracle mismatch".to_string(),
        ));
    }

    let factors_by_point = {
        let _span = tracing::debug_span!("root_extension_final_factors").entered();
        padded_points
            .iter()
            .map(|point| {
                tensor_equality_factor_eval_at_point::<F, C>(&point[split_bits..], &eta, &rho)
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    Ok((
        RootExtensionOpeningReduction {
            proof: ExtensionOpeningReductionProof {
                partials: proof_partials,
                #[cfg(not(feature = "zk"))]
                sumcheck,
                #[cfg(feature = "zk")]
                sumcheck_proof_masked,
            },
            rho,
            final_claim,
            factors_by_point,
        },
        row_coefficients,
    ))
}

pub(in crate::protocol::flow) type MultiplierWeightSlices<'a, F, const D: usize> =
    (&'a [CyclotomicRing<F, D>], &'a [CyclotomicRing<F, D>]);
pub(in crate::protocol::flow) type FoldedRings<F, const D: usize> = Vec<CyclotomicRing<F, D>>;
pub(in crate::protocol::flow) type RootClaimEvaluations<F, const D: usize> =
    (Vec<CyclotomicRing<F, D>>, Vec<FoldedRings<F, D>>);
