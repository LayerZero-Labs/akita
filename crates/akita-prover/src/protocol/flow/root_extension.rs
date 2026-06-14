use super::*;

pub(in crate::protocol::flow) struct PreparedRootExtensionOpeningReduction<E: FieldCore> {
    pub(in crate::protocol::flow) openings: Vec<E>,
    pub(in crate::protocol::flow) partials: Vec<E>,
    pub(in crate::protocol::flow) row_partials_by_claim: Vec<Vec<E>>,
    pub(in crate::protocol::flow) padded_point: Vec<E>,
    pub(in crate::protocol::flow) split_bits: usize,
}

pub(in crate::protocol::flow) struct RootExtensionOpeningReduction<C: FieldCore> {
    pub(in crate::protocol::flow) proof: ExtensionOpeningReductionProof<C>,
    pub(in crate::protocol::flow) rho: Vec<C>,
    pub(in crate::protocol::flow) final_claim: C,
    #[cfg(feature = "zk")]
    pub(in crate::protocol::flow) final_claim_public: C,
    pub(in crate::protocol::flow) shared_factor: C,
}

fn prepare_root_extension_opening_reduction<F, E, P, const D: usize>(
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    shared_opening_point: &[E],
) -> Result<PreparedRootExtensionOpeningReduction<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: RingSubfieldEncoding<F> + MulBaseUnreduced<F>,
    P: AkitaPolyOps<F, D>,
{
    let _span = tracing::info_span!(
        "prepare_root_extension_opening_reduction",
        num_claims = incidence_summary.num_claims(),
        num_vars = incidence_summary.num_vars()
    )
    .entered();
    let num_vars = incidence_summary.num_vars();
    let (split_bits, width) = tensor_opening_split::<F, E>()?;
    if split_bits > num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: split_bits,
            actual: num_vars,
        });
    }
    if polys.len() != incidence_summary.num_claims() {
        return Err(AkitaError::InvalidInput(
            "root extension-opening reduction input lengths do not match".to_string(),
        ));
    }

    if shared_opening_point.len() > num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: num_vars,
            actual: shared_opening_point.len(),
        });
    }
    let mut padded_point = shared_opening_point.to_vec();
    padded_point.resize(num_vars, E::zero());

    let mut openings = Vec::with_capacity(incidence_summary.num_claims());
    let mut partials = Vec::with_capacity(root_extension_opening_partials(
        width,
        incidence_summary.num_claims(),
    ));
    let mut row_partials_by_claim = Vec::with_capacity(incidence_summary.num_claims());
    {
        let _span =
            tracing::info_span!("root_extension_prepare_partials", width, split_bits).entered();
        let point_partials = <P as AkitaPolyOps<F, D>>::tensor_extension_column_partials_batch::<E>(
            polys,
            &padded_point,
        )?;
        if point_partials.len() != incidence_summary.num_claims() {
            return Err(AkitaError::InvalidSize {
                expected: incidence_summary.num_claims(),
                actual: point_partials.len(),
            });
        }
        for column_partials in point_partials {
            let opening =
                tensor_logical_claim_from_partials::<F, E>(&padded_point, &column_partials)?;
            let row_partials = tensor_row_partials_from_columns::<F, E>(&column_partials)?;
            partials.extend(column_partials);
            openings.push(opening);
            row_partials_by_claim.push(row_partials);
        }
    }

    Ok(PreparedRootExtensionOpeningReduction {
        openings,
        partials,
        row_partials_by_claim,
        padded_point,
        split_bits,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::flow) fn prove_root_extension_opening_reduction<
    F,
    E,
    T,
    P,
    const D: usize,
>(
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    shared_opening_point: &[E],
    transcript: &mut T,
    #[cfg(feature = "zk")] zk_hiding: &mut ZkHidingProverState<F>,
) -> Result<(RootExtensionOpeningReduction<E>, Vec<E>), AkitaError>
where
    F: FieldCore + CanonicalField,
    E: RingSubfieldEncoding<F>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + MulBaseUnreduced<F>
        + AkitaSerialize,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D>,
{
    let _span = tracing::info_span!(
        "prove_root_extension_opening_reduction",
        num_claims = incidence_summary.num_claims()
    )
    .entered();
    let prepared = prepare_root_extension_opening_reduction::<F, E, P, D>(
        polys,
        incidence_summary,
        shared_opening_point,
    )?;
    append_claim_values_to_transcript::<F, E, T>(&prepared.openings, transcript);
    let row_coefficients =
        sample_public_row_coefficients::<F, E, T>(incidence_summary, transcript)?;
    let PreparedRootExtensionOpeningReduction {
        openings: _,
        partials,
        row_partials_by_claim,
        padded_point,
        split_bits,
    } = prepared;
    let width = <E as ExtField<F>>::EXT_DEGREE;
    #[cfg(feature = "zk")]
    let (partial_masks, sumcheck_pads) = zk_hiding.take_extension_opening_reduction_pads::<E>(
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
        .map(tensor_row_partials_from_columns::<F, E>)
        .collect::<Result<Vec<_>, _>>()?;
    {
        let _span = tracing::debug_span!(
            "root_extension_absorb_partials",
            partials_len = proof_partials.len()
        )
        .entered();
        for partial in &proof_partials {
            append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, partial);
        }
    }
    let eta = (0..split_bits)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH))
        .collect::<Vec<_>>();
    let input_claim = {
        let _span = tracing::debug_span!("root_extension_input_claim").entered();
        proof_row_partials_by_claim.iter().enumerate().try_fold(
            E::zero(),
            |acc, (claim_idx, row_partials)| {
                tensor_reduction_claim_from_rows::<F, E>(row_partials, &eta)
                    .map(|claim| acc + row_coefficients[claim_idx] * claim)
            },
        )?
    };
    let true_input_claim = row_partials_by_claim.iter().enumerate().try_fold(
        E::zero(),
        |acc, (claim_idx, row_partials)| {
            tensor_reduction_claim_from_rows::<F, E>(row_partials, &eta)
                .map(|claim| acc + row_coefficients[claim_idx] * claim)
        },
    )?;
    #[cfg(not(feature = "zk"))]
    debug_assert_eq!(input_claim, true_input_claim);
    let sparse_terms = {
        let _span = tracing::info_span!("root_extension_sparse_terms").entered();
        let mut terms = Vec::with_capacity(1);
        let tail_point = &padded_point[split_bits..];
        let witness_evals = {
            let _span =
                tracing::info_span!("root_extension_sparse_witnesses", num_terms = polys.len())
                    .entered();
            <P as AkitaPolyOps<F, D>>::tensor_packed_extension_sparse_linear_combination::<E>(
                polys,
                &row_coefficients,
            )?
        };
        if let Some(witness_evals) = witness_evals {
            let lazy_rounds = tail_point.len().min(SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS);
            if lazy_rounds == 0 {
                let factor_evals = {
                    let _span = tracing::debug_span!(
                        "root_extension_factor_evals",
                        tail_vars = tail_point.len()
                    )
                    .entered();
                    tensor_equality_factor_evals::<F, E>(tail_point, &eta)?
                };
                terms.push(ExtensionOpeningReductionTerm::new_sparse(
                    witness_evals,
                    factor_evals,
                    E::one(),
                )?);
            } else {
                let _span = tracing::debug_span!(
                    "root_extension_lazy_tensor_factor",
                    tail_vars = tail_point.len(),
                    lazy_rounds
                )
                .entered();
                terms.push(
                    ExtensionOpeningReductionTerm::new_sparse_tensor_factor::<F>(
                        witness_evals,
                        tail_point.to_vec(),
                        eta.clone(),
                        E::one(),
                        lazy_rounds,
                    )?,
                );
            }
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
                let tail_point = &padded_point[split_bits..];
                let factor_evals = tensor_equality_factor_evals::<F, E>(tail_point, &eta)?;
                let witness_evals = poly.tensor_packed_extension_evals::<E>()?;
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
        sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND)
    })?;
    #[cfg(feature = "zk")]
    let (sumcheck_proof_masked, rho) = prover.prove_zk::<F, T, _>(
        input_claim,
        transcript,
        |tr| sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND),
        sumcheck_pads,
    )?;
    #[cfg(feature = "zk")]
    let final_claim_public =
        masked_sumcheck_final_claim(input_claim, &sumcheck_proof_masked, &rho)?;
    let final_terms = prover.final_terms().ok_or_else(|| {
        AkitaError::InvalidInput(
            "root extension-opening reduction has not reached a final point".to_string(),
        )
    })?;
    let expected_final = final_terms
        .into_iter()
        .fold(E::zero(), |acc, (coeff, witness, factor)| {
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

    let shared_factor = {
        let _span = tracing::debug_span!("root_extension_final_factors").entered();
        tensor_equality_factor_eval_at_point::<F, E>(&padded_point[split_bits..], &eta, &rho)?
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
            #[cfg(feature = "zk")]
            final_claim_public,
            shared_factor,
        },
        row_coefficients,
    ))
}

pub(in crate::protocol::flow) type MultiplierWeightSlices<'a, F, const D: usize> =
    (&'a [CyclotomicRing<F, D>], &'a [CyclotomicRing<F, D>]);
pub(in crate::protocol::flow) type FoldedClaimEvals<F, const D: usize> =
    (Vec<CyclotomicRing<F, D>>, Vec<Vec<CyclotomicRing<F, D>>>);
