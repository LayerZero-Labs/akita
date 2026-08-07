use super::*;
use crate::compute::{
    ComputeBackendSetup, RootTensorSource, TensorPackedWitness, TensorProjectionBatchKernel,
    TensorProjectionKernel,
};
use crate::kernels::sumcheck::SumcheckTableOperations;
use akita_field::unreduced::ReduceTo;
use std::ops::Range;

pub(in crate::protocol::core) struct ProvedExtensionOpeningReduction<E: FieldCore> {
    pub(in crate::protocol::core) reduction: ExtensionOpeningReduction<E>,
    pub(in crate::protocol::core) row_coefficients: Vec<E>,
    pub(in crate::protocol::core) protocol_points: Vec<Vec<E>>,
}

pub(crate) struct PreparedExtensionOpeningGroup<E: FieldCore> {
    pub(crate) proof_partials: Vec<E>,
    pub(crate) row_partials_by_claim: Vec<Vec<E>>,
    pub(crate) openings: Vec<E>,
}

/// Truthful per-group input to extension-opening reduction.
///
/// EOR needs polynomial sources and point geometry, not claimed evaluations or
/// commitments. Keeping those values out prevents recursive suffix proving
/// from fabricating public claims merely to satisfy an adapter.
pub(in crate::protocol::core) struct ExtensionOpeningGroupInput<'group, 'point, E, G> {
    pub(in crate::protocol::core) group: &'group G,
    pub(in crate::protocol::core) point: &'point [E],
    pub(in crate::protocol::core) ring_dimension: usize,
}

pub(in crate::protocol::core) fn prepare_extension_opening_group<F, E, P, B, const D: usize>(
    backend: &B,
    prepared: Option<&<B as ComputeBackendSetup<F>>::PreparedSetup>,
    polys: &[&P],
    point: &[E],
) -> Result<PreparedExtensionOpeningGroup<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F> + MulBaseUnreduced<F>,
    P: RootTensorSource<F, D>,
    B: ComputeBackendSetup<F>
        + for<'a> TensorProjectionBatchKernel<P::TensorBatchView<'a>, F, E, D>,
{
    let (_split_bits, width) = tensor_opening_split::<F, E>()?;
    let point_partials = TensorProjectionBatchKernel::column_partials_batch(
        backend,
        prepared,
        P::tensor_batch(polys)?,
        point,
    )?;
    if point_partials.len() != polys.len() {
        return Err(AkitaError::InvalidSize {
            expected: polys.len(),
            actual: point_partials.len(),
        });
    }
    let mut proof_partials = Vec::with_capacity(width.saturating_mul(polys.len()));
    let mut row_partials_by_claim = Vec::with_capacity(polys.len());
    let mut openings = Vec::with_capacity(polys.len());
    for column_partials in point_partials {
        openings.push(derive_tensor_extension_opening_claim_from_partials::<F, E>(
            point,
            &column_partials,
        )?);
        row_partials_by_claim.push(tensor_row_partials_from_columns::<F, E>(&column_partials)?);
        proof_partials.extend(column_partials);
    }
    Ok(PreparedExtensionOpeningGroup {
        proof_partials,
        row_partials_by_claim,
        openings,
    })
}

/// Prove one extension-opening reduction over all opening groups.
///
/// Each group contributes native-dimension witness/factor terms. Terms with a
/// smaller tail arity are extended cylindrically over fixed zero coordinates,
/// so every group participates in one sumcheck challenge sequence without
/// materializing repeated witness tables.
#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn prove_extension_opening_reduction<F, E, T, G, B>(
    tensor_backend: &B,
    tensor_prepared: Option<&B::PreparedSetup>,
    group_inputs: &[ExtensionOpeningGroupInput<'_, '_, E, G>],
    transcript: &mut T,
    path: &'static str,
) -> Result<ProvedExtensionOpeningReduction<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + MulBaseUnreduced<F>
        + AkitaSerialize
        + SumcheckTableOperations<F>,
    T: Transcript<F>,
    G: RootProverGroupTensor<F, E, B>,
    B: ComputeBackendSetup<F>,
{
    let opening_batch = OpeningClaimsLayout::from_groups(
        group_inputs
            .iter()
            .map(|input| {
                PolynomialGroupLayout::new(input.point.len(), input.group.num_polynomials())
            })
            .collect(),
    )?;
    let _span = tracing::info_span!(
        "prove_extension_opening_reduction",
        path,
        num_claims = opening_batch.num_total_polynomials(),
        num_groups = opening_batch.num_groups(),
    )
    .entered();
    let (split_bits, width) = tensor_opening_split::<F, E>()?;
    let max_tail_vars = opening_batch.max_num_vars().checked_sub(split_bits).ok_or(
        AkitaError::InvalidPointDimension {
            expected: split_bits,
            actual: opening_batch.max_num_vars(),
        },
    )?;

    let mut prepared_groups = Vec::with_capacity(opening_batch.num_groups());
    for (group_index, input) in group_inputs.iter().enumerate() {
        let point = input.point;
        if point.len() < split_bits {
            return Err(AkitaError::InvalidPointDimension {
                expected: split_bits,
                actual: point.len(),
            });
        }
        let group = input
            .group
            .prepare_extension_opening(tensor_backend, tensor_prepared, input.ring_dimension, point)
            .map_err(|error| {
                AkitaError::InvalidInput(format!(
                    "extension-opening group {group_index} partials failed: {error:?}"
                ))
            })?;
        prepared_groups.push(group);
    }

    let openings = prepared_groups
        .iter()
        .flat_map(|group| group.openings.iter().copied())
        .collect::<Vec<_>>();
    let num_claims = opening_batch.num_total_polynomials();
    if openings.len() != num_claims {
        return Err(AkitaError::InvalidSize {
            expected: num_claims,
            actual: openings.len(),
        });
    }
    let row_coefficients =
        derive_public_row_coefficients::<F, E, T>(&opening_batch, &openings, transcript)?;
    let proof_partials = prepared_groups
        .iter()
        .flat_map(|group| group.proof_partials.iter().copied())
        .collect::<Vec<_>>();
    let expected_partials = width
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidInput("EOR partial count overflow".to_string()))?;
    if proof_partials.len() != expected_partials {
        return Err(AkitaError::InvalidSize {
            expected: expected_partials,
            actual: proof_partials.len(),
        });
    }
    for partial in &proof_partials {
        append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, partial);
    }
    let eta = (0..split_bits)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH))
        .collect::<Vec<_>>();
    let true_input_claim = prepared_groups
        .iter()
        .flat_map(|group| group.row_partials_by_claim.iter())
        .zip(row_coefficients.iter().copied())
        .try_fold(E::zero(), |acc, (row_partials, coeff)| {
            tensor_reduction_claim_from_rows::<F, E>(row_partials, &eta)
                .map(|claim| acc + coeff * claim)
        })?;

    let mut terms = Vec::new();
    let mut term_ranges = Vec::<Range<usize>>::with_capacity(group_inputs.len());
    for group_index in 0..group_inputs.len() {
        let claim_range = opening_batch.root_group_claim_range(group_index)?;
        let input = group_inputs
            .get(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        let point = input.point;
        let tail_point = &point[split_bits..];
        let extra_vars = max_tail_vars
            .checked_sub(tail_point.len())
            .ok_or(AkitaError::InvalidProof)?;
        let group_terms = input
            .group
            .extension_opening_terms(
                tensor_backend,
                tensor_prepared,
                input.ring_dimension,
                row_coefficients
                    .get(claim_range.clone())
                    .ok_or(AkitaError::InvalidProof)?,
                tail_point,
                &eta,
            )
            .map_err(|error| {
                AkitaError::InvalidInput(format!(
                    "extension-opening group {group_index} terms failed: {error:?}"
                ))
            })?;
        let start = terms.len();
        let expected_domain_len = checked_table_len(max_tail_vars)?;
        for term in group_terms {
            let term = term.extend_cylindrically(vec![E::zero(); extra_vars])?;
            if term.domain_len() != expected_domain_len {
                return Err(AkitaError::InvalidInput(format!(
                    "extension-opening group {group_index} domain mismatch: expected \
                     {expected_domain_len}, actual {}",
                    term.domain_len()
                )));
            }
            terms.push(term);
        }
        term_ranges.push(start..terms.len());
    }

    let mut prover = ExtensionOpeningReductionProver::new(terms, true_input_claim)?;
    let (sumcheck, rho, final_claim) = prover.prove::<F, T, _>(transcript, |tr| {
        sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND)
    })?;
    let final_terms = prover.final_terms().ok_or_else(|| {
        AkitaError::InvalidInput(format!(
            "{path} extension-opening reduction has not reached a final point"
        ))
    })?;
    let mut final_factors = Vec::with_capacity(group_inputs.len());
    let mut protocol_points = Vec::with_capacity(group_inputs.len());
    for group_index in 0..group_inputs.len() {
        let input = group_inputs
            .get(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        let point = &input.point;
        let tail_point = &point[split_bits..];
        let local_rho = rho
            .get(..tail_point.len())
            .ok_or(AkitaError::InvalidProof)?;
        let mut factor = tensor_equality_factor_eval_at_point::<F, E>(tail_point, &eta, local_rho)?;
        for &extra_challenge in rho
            .get(tail_point.len()..)
            .ok_or(AkitaError::InvalidProof)?
        {
            factor *= E::one() - extra_challenge;
        }
        let term_range = term_ranges
            .get(group_index)
            .cloned()
            .ok_or(AkitaError::InvalidProof)?;
        if final_terms
            .get(term_range)
            .ok_or(AkitaError::InvalidProof)?
            .iter()
            .any(|(_, _, term_factor)| *term_factor != factor)
        {
            return Err(AkitaError::InvalidInput(format!(
                "{path} extension-opening transparent factor mismatch"
            )));
        }
        let protocol_point = dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            input.ring_dimension,
            |D| ring_subfield_packed_extension_opening_point::<F, E, D>(local_rho.len(), local_rho,)
        )?;
        final_factors.push(factor);
        protocol_points.push(protocol_point);
    }
    let expected_final = final_terms
        .into_iter()
        .fold(E::zero(), |acc, (coeff, witness, factor)| {
            acc + coeff * witness * factor
        });
    if final_claim != expected_final {
        return Err(AkitaError::InvalidInput(format!(
            "{path} extension-opening final oracle mismatch"
        )));
    }

    Ok(ProvedExtensionOpeningReduction {
        reduction: ExtensionOpeningReduction {
            proof: ExtensionOpeningReductionProof {
                partials: proof_partials,
                sumcheck,
            },
            final_claim,
            final_factors,
        },
        row_coefficients,
        protocol_points,
    })
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
) -> Result<Vec<ExtensionOpeningReductionTerm<F, E>>, AkitaError>
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
) -> Result<Option<Vec<ExtensionOpeningReductionTerm<F, E>>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: MulBaseUnreduced<F>,
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
        ExtensionOpeningReductionTerm::new_sparse_tensor_factor(
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
    witness: TensorPackedWitness<F, E>,
    tail_point: &[E],
    eta: &[E],
    coeff: E,
) -> Result<ExtensionOpeningReductionTerm<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: MulBaseUnreduced<F>,
{
    match witness {
        TensorPackedWitness::Dense(witness_evals) => {
            ExtensionOpeningReductionTerm::new_tensor(witness_evals, tail_point, eta, coeff)
        }
        TensorPackedWitness::Sparse(witness) => ExtensionOpeningReductionTerm::new_sparse(
            witness,
            tensor_equality_factor_evals::<F, E>(tail_point, eta)?,
            coeff,
        ),
    }
}

fn build_dense_extension_opening_reduction_terms<F, E, P, B, const D: usize>(
    backend: &B,
    prepared: Option<&<B as ComputeBackendSetup<F>>::PreparedSetup>,
    polys: &[&P],
    row_coefficients: &[E],
    tail_point: &[E],
    eta: &[E],
) -> Result<Vec<ExtensionOpeningReductionTerm<F, E>>, AkitaError>
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

pub(in crate::protocol::core) type MultiplierWeightSlices<'a, F, const D: usize> =
    (&'a [CyclotomicRing<F, D>], &'a [CyclotomicRing<F, D>]);
pub(in crate::protocol::core) type FoldedClaimEvals<F, const D: usize> =
    (Vec<CyclotomicRing<F, D>>, Vec<Vec<CyclotomicRing<F, D>>>);
