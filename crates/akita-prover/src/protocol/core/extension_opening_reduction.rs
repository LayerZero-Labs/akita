use super::*;
use crate::compute::{
    ComputeBackendSetup, RootTensorSource, TensorPackedWitness, TensorProjectionBatchKernel,
    TensorProjectionKernel,
};

pub(in crate::protocol::core) struct PreparedExtensionOpeningReduction<E: FieldCore> {
    pub(in crate::protocol::core) proof_partials: Vec<E>,
    pub(in crate::protocol::core) row_coefficients: Vec<E>,
    pub(in crate::protocol::core) terms: Vec<ExtensionOpeningReductionTerm<E>>,
    pub(in crate::protocol::core) padded_point: Vec<E>,
    pub(in crate::protocol::core) split_bits: usize,
    pub(in crate::protocol::core) eta: Vec<E>,
    pub(in crate::protocol::core) true_input_claim: E,
}

pub(in crate::protocol::core) struct ProvedExtensionOpeningReduction<E: FieldCore> {
    pub(in crate::protocol::core) reduction: ExtensionOpeningReduction<E>,
    pub(in crate::protocol::core) row_coefficients: Vec<E>,
    pub(in crate::protocol::core) protocol_point: Vec<E>,
}

pub(in crate::protocol::core) fn build_extension_opening_reduction_terms<
    F,
    E,
    P,
    B,
    const D: usize,
>(
    backend: &B,
    prepared: Option<&<B as ComputeBackendSetup<F>>::PreparedSetup>,
    polys: &[&P],
    row_coefficients: &[E],
    tail_point: &[E],
    eta: &[E],
) -> Result<Vec<ExtensionOpeningReductionTerm<E>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + MulBaseUnreduced<F>,
    P: RootTensorSource<F, D>,
    B: ComputeBackendSetup<F>
        + for<'a> TensorProjectionBatchKernel<P::TensorBatchView<'a>, F, E, D>
        + for<'a> TensorProjectionKernel<P::TensorView<'a>, F, E, D>,
{
    let _span =
        tracing::info_span!("extension_opening_reduction_terms", num_terms = polys.len()).entered();
    if polys.len() != row_coefficients.len() {
        return Err(AkitaError::InvalidSize {
            expected: polys.len(),
            actual: row_coefficients.len(),
        });
    }

    if let Some(terms) = try_sparse_extension_opening_reduction_terms::<F, E, P, B, D>(
        backend,
        prepared,
        polys,
        row_coefficients,
        tail_point,
        eta,
    )? {
        return Ok(terms);
    }

    build_dense_extension_opening_reduction_terms::<F, E, P, B, D>(
        backend,
        prepared,
        polys,
        row_coefficients,
        tail_point,
        eta,
    )
}

fn try_sparse_extension_opening_reduction_terms<F, E, P, B, const D: usize>(
    backend: &B,
    prepared: Option<&<B as ComputeBackendSetup<F>>::PreparedSetup>,
    polys: &[&P],
    row_coefficients: &[E],
    tail_point: &[E],
    eta: &[E],
) -> Result<Option<Vec<ExtensionOpeningReductionTerm<E>>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
    P: RootTensorSource<F, D>,
    B: ComputeBackendSetup<F>
        + for<'a> TensorProjectionBatchKernel<P::TensorBatchView<'a>, F, E, D>,
{
    let _span =
        tracing::info_span!("extension_opening_sparse_terms", num_terms = polys.len()).entered();
    let Some(witness_evals) = TensorProjectionBatchKernel::sparse_linear_combination(
        backend,
        prepared,
        P::tensor_batch(polys)?,
        row_coefficients,
    )?
    else {
        return Ok(None);
    };
    let lazy_rounds = tail_point.len().min(SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS);
    let term = if lazy_rounds == 0 {
        let factor_evals = {
            let _span = tracing::debug_span!(
                "extension_opening_factor_evals",
                tail_vars = tail_point.len()
            )
            .entered();
            tensor_equality_factor_evals::<F, E>(tail_point, eta)?
        };
        ExtensionOpeningReductionTerm::new_sparse(witness_evals, factor_evals, E::one())?
    } else {
        let _span = tracing::debug_span!(
            "extension_opening_lazy_tensor_factor",
            tail_vars = tail_point.len(),
            lazy_rounds
        )
        .entered();
        ExtensionOpeningReductionTerm::new_sparse_tensor_factor::<F>(
            witness_evals,
            tail_point.to_vec(),
            eta.to_vec(),
            E::one(),
            lazy_rounds,
        )?
    };
    Ok(Some(vec![term]))
}

fn extension_opening_term_from_packed_witness<F, E>(
    witness: TensorPackedWitness<E>,
    tail_point: &[E],
    eta: &[E],
    coeff: E,
) -> Result<ExtensionOpeningReductionTerm<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    let factor_evals = tensor_equality_factor_evals::<F, E>(tail_point, eta)?;
    match witness {
        TensorPackedWitness::Dense(witness_evals) => {
            ExtensionOpeningReductionTerm::new(witness_evals, factor_evals, coeff)
        }
        TensorPackedWitness::Sparse(witness) => {
            ExtensionOpeningReductionTerm::new_sparse(witness, factor_evals, coeff)
        }
    }
}

fn build_dense_extension_opening_reduction_terms<F, E, P, B, const D: usize>(
    backend: &B,
    prepared: Option<&<B as ComputeBackendSetup<F>>::PreparedSetup>,
    polys: &[&P],
    row_coefficients: &[E],
    tail_point: &[E],
    eta: &[E],
) -> Result<Vec<ExtensionOpeningReductionTerm<E>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + MulBaseUnreduced<F>,
    P: RootTensorSource<F, D>,
    B: ComputeBackendSetup<F> + for<'a> TensorProjectionKernel<P::TensorView<'a>, F, E, D>,
{
    let _span =
        tracing::info_span!("extension_opening_dense_witnesses", num_terms = polys.len()).entered();
    polys
        .iter()
        .zip(row_coefficients.iter().copied())
        .map(|(poly, coeff)| {
            let witness = {
                let _s = tracing::info_span!("eor_packed_witness").entered();
                TensorProjectionKernel::packed_witness(backend, prepared, poly.tensor_view()?)?
            };
            extension_opening_term_from_packed_witness::<F, E>(witness, tail_point, eta, coeff)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn prepare_extension_opening_reduction<
    F,
    E,
    T,
    P,
    B,
    const D: usize,
>(
    backend: &B,
    prepared: Option<&<B as ComputeBackendSetup<F>>::PreparedSetup>,
    polys: &[&P],
    opening_batch: &OpeningClaims<'_, E>,
    pad_base_evals: bool,
    transcript: &mut T,
) -> Result<PreparedExtensionOpeningReduction<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + MulBaseUnreduced<F>,
    T: Transcript<F>,
    P: RootTensorSource<F, D>,
    B: ComputeBackendSetup<F>
        + for<'a> TensorProjectionBatchKernel<P::TensorBatchView<'a>, F, E, D>
        + for<'a> TensorProjectionKernel<P::TensorView<'a>, F, E, D>,
{
    let num_claims = opening_batch.num_total_polynomials();
    let num_vars = opening_batch.num_vars();
    let _span =
        tracing::info_span!("prepare_extension_opening_reduction", num_claims, num_vars).entered();
    let (split_bits, width) = tensor_opening_split::<F, E>()?;
    if split_bits > num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: split_bits,
            actual: num_vars,
        });
    }
    if polys.len() != num_claims {
        return Err(AkitaError::InvalidInput(
            "extension-opening reduction input lengths do not match".to_string(),
        ));
    }

    let padded_point = opening_batch.point().to_vec();

    let mut openings = Vec::with_capacity(num_claims);
    let mut partials = Vec::with_capacity(width.saturating_mul(num_claims));
    let mut row_partials_by_claim = Vec::with_capacity(num_claims);
    {
        let _span =
            tracing::info_span!("extension_opening_prepare_partials", width, split_bits).entered();
        let point_partials = TensorProjectionBatchKernel::column_partials_batch(
            backend,
            prepared,
            P::tensor_batch(polys)?,
            &padded_point,
        )?;
        if point_partials.len() != num_claims {
            return Err(AkitaError::InvalidSize {
                expected: num_claims,
                actual: point_partials.len(),
            });
        }
        for column_partials in point_partials {
            let opening = derive_tensor_extension_opening_claim_from_partials::<F, E>(
                &padded_point,
                &column_partials,
            )?;
            let row_partials = tensor_row_partials_from_columns::<F, E>(&column_partials)?;
            partials.extend(column_partials);
            openings.push(opening);
            row_partials_by_claim.push(row_partials);
        }
    }
    let proof_partials = partials.clone();
    let row_coefficients = if pad_base_evals {
        if num_claims != 1 {
            return Err(AkitaError::InvalidInput(
                "recursive extension-opening reduction expects a single claim".to_string(),
            ));
        }
        vec![E::one()]
    } else {
        let transcript_openings = openings.as_slice();
        append_claim_values_to_transcript::<F, E, T>(transcript_openings, transcript);
        let opening_shape = opening_batch.layout();
        sample_public_row_coefficients::<F, E, T>(&opening_shape, transcript)?
    };
    if row_partials_by_claim.len() != row_coefficients.len() {
        return Err(AkitaError::InvalidSize {
            expected: row_partials_by_claim.len(),
            actual: row_coefficients.len(),
        });
    }
    let expected_partials = width
        .checked_mul(row_coefficients.len())
        .ok_or_else(|| AkitaError::InvalidInput("EOR partial count overflow".to_string()))?;
    if proof_partials.len() != expected_partials {
        return Err(AkitaError::InvalidSize {
            expected: expected_partials,
            actual: proof_partials.len(),
        });
    }
    let proof_row_partials_by_claim = proof_partials
        .chunks_exact(width)
        .map(tensor_row_partials_from_columns::<F, E>)
        .collect::<Result<Vec<_>, _>>()?;
    {
        let _span = tracing::debug_span!(
            "extension_opening_absorb_partials",
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
        let _span = tracing::debug_span!("extension_opening_input_claim").entered();
        proof_row_partials_by_claim
            .iter()
            .zip(row_coefficients.iter().copied())
            .try_fold(E::zero(), |acc, (row_partials, coeff)| {
                tensor_reduction_claim_from_rows::<F, E>(row_partials, &eta)
                    .map(|claim| acc + coeff * claim)
            })?
    };
    let true_input_claim = row_partials_by_claim
        .iter()
        .zip(row_coefficients.iter().copied())
        .try_fold(E::zero(), |acc, (row_partials, coeff)| {
            tensor_reduction_claim_from_rows::<F, E>(row_partials, &eta)
                .map(|claim| acc + coeff * claim)
        })?;
    debug_assert_eq!(input_claim, true_input_claim);

    let tail_point = &padded_point[split_bits..];
    let terms = build_extension_opening_reduction_terms::<F, E, P, B, D>(
        backend,
        prepared,
        polys,
        &row_coefficients,
        tail_point,
        &eta,
    )?;

    Ok(PreparedExtensionOpeningReduction {
        proof_partials,
        row_coefficients,
        terms,
        padded_point,
        split_bits,
        eta,
        true_input_claim,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn prove_extension_opening_reduction<F, E, T, P, B, const D: usize>(
    tensor_backend: &B,
    tensor_prepared: Option<&<B as ComputeBackendSetup<F>>::PreparedSetup>,
    polys: &[&P],
    opening_batch: &OpeningClaims<'_, E>,
    pad_base_evals: bool,
    transcript: &mut T,
    path: &'static str,
) -> Result<ProvedExtensionOpeningReduction<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + HasUnreducedOps + HasOptimizedFold + MulBaseUnreduced<F> + AkitaSerialize,
    T: Transcript<F>,
    P: RootTensorSource<F, D>,
    B: ComputeBackendSetup<F>
        + for<'a> TensorProjectionBatchKernel<P::TensorBatchView<'a>, F, E, D>
        + for<'a> TensorProjectionKernel<P::TensorView<'a>, F, E, D>,
{
    let _span = tracing::info_span!(
        "prove_extension_opening_reduction",
        path,
        num_claims = opening_batch.num_total_polynomials()
    )
    .entered();
    let backend = tensor_backend;
    let prepared = prepare_extension_opening_reduction::<F, E, T, P, B, D>(
        backend,
        tensor_prepared,
        polys,
        opening_batch,
        pad_base_evals,
        transcript,
    )?;
    let tail_point = &prepared.padded_point[prepared.split_bits..];
    let prover_claim =
        ExtensionOpeningReductionProver::input_claim_from_terms(prepared.terms.as_slice())?;
    if prover_claim != prepared.true_input_claim {
        return Err(AkitaError::InvalidInput(
            "extension-opening reduction input claim mismatch".to_string(),
        ));
    }
    let mut prover = {
        let _span = tracing::info_span!("extension_opening_reduction_prover_new", path).entered();
        ExtensionOpeningReductionProver::new(prepared.terms, prover_claim)?
    };
    let _eor_sumcheck_span = tracing::info_span!(
        "extension_opening_reduction_sumcheck",
        path = path,
        num_rounds = prover.num_rounds()
    )
    .entered();
    let (sumcheck_proof, rho, final_claim) = prover.prove::<F, T, _>(transcript, |tr| {
        sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND)
    })?;
    let final_terms = prover.final_terms().ok_or_else(|| {
        AkitaError::InvalidInput(format!(
            "{path} extension-opening reduction has not reached a final point"
        ))
    })?;
    let final_factor =
        tensor_equality_factor_eval_at_point::<F, E>(tail_point, &prepared.eta, &rho)?;
    if final_terms
        .iter()
        .any(|(_, _, factor)| *factor != final_factor)
    {
        return Err(AkitaError::InvalidInput(format!(
            "{path} extension-opening reduction transparent factor mismatch"
        )));
    }
    let expected_final = final_terms
        .into_iter()
        .fold(E::zero(), |acc, (coeff, witness, factor)| {
            acc + coeff * witness * factor
        });
    if final_claim != expected_final {
        return Err(AkitaError::InvalidInput(format!(
            "{path} extension-opening reduction final oracle mismatch"
        )));
    }
    let protocol_point = {
        let _span = tracing::info_span!("extension_opening_protocol_point").entered();
        ring_subfield_packed_extension_opening_point::<F, E, D>(rho.len(), &rho)?
    };
    let reduction = ExtensionOpeningReduction {
        proof: ExtensionOpeningReductionProof {
            partials: prepared.proof_partials,
            sumcheck: sumcheck_proof,
        },
        final_claim,
        final_factor,
    };

    Ok(ProvedExtensionOpeningReduction {
        reduction,
        row_coefficients: prepared.row_coefficients,
        protocol_point,
    })
}

pub(in crate::protocol::core) type MultiplierWeightSlices<'a, F, const D: usize> =
    (&'a [CyclotomicRing<F, D>], &'a [CyclotomicRing<F, D>]);
pub(in crate::protocol::core) type FoldedClaimEvals<F, const D: usize> =
    (Vec<CyclotomicRing<F, D>>, Vec<Vec<CyclotomicRing<F, D>>>);
