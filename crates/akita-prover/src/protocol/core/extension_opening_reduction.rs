use super::*;
use crate::compute::{
    ComputeBackendSetup, RootTensorSource, TensorProjectionBatchKernel, TensorProjectionKernel,
};

pub(in crate::protocol::core) struct ProvedExtensionOpeningReduction<E: Field, R> {
    pub(in crate::protocol::core) reduction: R,
    pub(in crate::protocol::core) protocol_points: Vec<Vec<E>>,
}
pub(in crate::protocol::core) struct NativeExtensionOpeningReduction<E: Field> {
    pub(in crate::protocol::core) final_claims: Vec<E>,
    pub(in crate::protocol::core) final_factors: Vec<E>,
}

pub(crate) struct PreparedExtensionOpeningGroup<E: Field> {
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

trait EorProverStream<F, E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    type Sumcheck;
    type Reduction;

    fn prefix(
        &mut self,
        opening_batch: &OpeningClaimsLayout,
        openings: &[E],
        partials: &[E],
    ) -> Result<(Vec<E>, Vec<E>), AkitaError>;

    fn prove_sumcheck<P>(
        &mut self,
        prover: &mut P,
    ) -> Result<(Self::Sumcheck, Vec<E>, E), AkitaError>
    where
        P: akita_sumcheck::SumcheckInstanceProver<E> + ?Sized;

    fn final_claims(
        &mut self,
        opening_batch: &OpeningClaimsLayout,
        final_claims: &[E],
    ) -> Result<(), AkitaError>;

    fn build_reduction(
        partials: Vec<E>,
        sumcheck: Self::Sumcheck,
        final_claims: Vec<E>,
        final_factors: Vec<E>,
    ) -> Self::Reduction;
}
struct NativeEorProverStream<'a, 'plan> {
    grinding: &'a mut akita_types::NativeProverGrinding<'plan>,
    level: u32,
}

impl<F, E> EorProverStream<F, E> for NativeEorProverStream<'_, '_>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    type Sumcheck = ();
    type Reduction = NativeExtensionOpeningReduction<E>;

    fn prefix(
        &mut self,
        opening_batch: &OpeningClaimsLayout,
        openings: &[E],
        partials: &[E],
    ) -> Result<(Vec<E>, Vec<E>), AkitaError> {
        let prefix = akita_types::native_eor_prover_prefix::<F, E>(
            self.grinding,
            opening_batch,
            openings,
            partials,
            self.level,
        )?;
        Ok((prefix.eta, prefix.claim_coefficients))
    }

    fn prove_sumcheck<P>(
        &mut self,
        prover: &mut P,
    ) -> Result<(Self::Sumcheck, Vec<E>, E), AkitaError>
    where
        P: akita_sumcheck::SumcheckInstanceProver<E> + ?Sized,
    {
        let mut channel = akita_types::NativeGrindingSumcheckProver::<F, E>::new(
            self.grinding,
            akita_types::SumcheckProtocol::ExtensionOpeningReduction,
            self.level,
            0,
        );
        let shape =
            akita_sumcheck::NativeSumcheckShape::new(prover.num_rounds(), prover.degree_bound())?;
        let (point, final_claim) = akita_sumcheck::prove_sumcheck_native::<F, E, _, _>(
            prover,
            &mut channel,
            shape,
            akita_types::NATIVE_EOR_SUMCHECK_INVOCATION,
        )?;
        Ok(((), point, final_claim))
    }

    fn final_claims(
        &mut self,
        opening_batch: &OpeningClaimsLayout,
        final_claims: &[E],
    ) -> Result<(), AkitaError> {
        akita_types::native_eor_prover_final_claims::<F, E>(
            self.grinding,
            opening_batch,
            final_claims,
            self.level,
        )
    }

    fn build_reduction(
        _partials: Vec<E>,
        (): Self::Sumcheck,
        final_claims: Vec<E>,
        final_factors: Vec<E>,
    ) -> Self::Reduction {
        NativeExtensionOpeningReduction {
            final_claims,
            final_factors,
        }
    }
}

pub(in crate::protocol::core) fn prepare_extension_opening_group<F, E, P, B, const D: usize>(
    backend: &B,
    prepared: Option<&<B as ComputeBackendSetup<F>>::PreparedSetup>,
    polys: &[&P],
    point: &[E],
) -> Result<PreparedExtensionOpeningGroup<E>, AkitaError>
where
    F: Field + CanonicalEncoding + Ring + Unreduced + AkitaSerialize + 'static,
    <F as Unreduced>::Wide: From<F>,
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

/// Prove EOR directly into the authoritative native Spongefish stream.
#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn prove_extension_opening_reduction_native<F, E, G, B>(
    tensor_backend: &B,
    tensor_prepared: Option<&B::PreparedSetup>,
    group_inputs: &[ExtensionOpeningGroupInput<'_, '_, E, G>],
    grinding: &mut akita_types::NativeProverGrinding<'_>,
    level: u32,
    path: &'static str,
) -> Result<ProvedExtensionOpeningReduction<E, NativeExtensionOpeningReduction<E>>, AkitaError>
where
    F: Field + CanonicalEncoding + Ring + Unreduced + AkitaSerialize + 'static,
    <F as Unreduced>::Wide: From<F>,
    E: ExtField<F> + Unreduced + Fold + MulBaseUnreduced<F> + AkitaSerialize,
    G: RootProverGroupTensor<F, E, B>,
    B: ComputeBackendSetup<F>,
{
    let mut stream = NativeEorProverStream { grinding, level };
    prove_extension_opening_reduction_with_stream::<F, E, _, G, B>(
        tensor_backend,
        tensor_prepared,
        group_inputs,
        &mut stream,
        path,
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_extension_opening_reduction_with_stream<F, E, S, G, B>(
    tensor_backend: &B,
    tensor_prepared: Option<&B::PreparedSetup>,
    group_inputs: &[ExtensionOpeningGroupInput<'_, '_, E, G>],
    stream: &mut S,
    path: &'static str,
) -> Result<ProvedExtensionOpeningReduction<E, S::Reduction>, AkitaError>
where
    F: Field + CanonicalEncoding + Ring + Unreduced + AkitaSerialize + 'static,
    <F as Unreduced>::Wide: From<F>,
    E: ExtField<F> + Unreduced + Fold + MulBaseUnreduced<F> + AkitaSerialize,
    S: EorProverStream<F, E>,
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
    let (eta, claim_coefficients) = stream.prefix(&opening_batch, &openings, &proof_partials)?;
    if eta.len() != split_bits || claim_coefficients.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let true_input_claims = prepared_groups
        .iter()
        .flat_map(|group| group.row_partials_by_claim.iter())
        .map(|row_partials| tensor_reduction_claim_from_rows::<F, E>(row_partials, &eta))
        .collect::<Result<Vec<_>, _>>()?;
    let true_input_claim = true_input_claims
        .iter()
        .zip(&claim_coefficients)
        .fold(E::zero(), |acc, (&claim, &coefficient)| {
            acc + coefficient * claim
        });

    let mut groups = Vec::with_capacity(group_inputs.len());
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
        let group = input
            .group
            .extension_opening_group(
                tensor_backend,
                tensor_prepared,
                input.ring_dimension,
                claim_coefficients
                    .get(claim_range.clone())
                    .ok_or(AkitaError::InvalidProof)?,
                tail_point,
                &eta,
            )
            .map_err(|error| {
                AkitaError::InvalidInput(format!(
                    "extension-opening group {group_index} construction failed: {error:?}"
                ))
            })?;
        let expected_domain_len = reduction_table_len(max_tail_vars)?;
        let group = group.extend_cylindrically(vec![E::zero(); extra_vars])?;
        if group.domain_len() != expected_domain_len {
            return Err(AkitaError::InvalidInput(format!(
                "extension-opening group {group_index} domain mismatch: expected \
                 {expected_domain_len}, actual {}",
                group.domain_len()
            )));
        }
        if group.num_terms() != claim_range.len() {
            return Err(AkitaError::InvalidProof);
        }
        groups.push(group);
    }

    if groups
        .iter()
        .map(ExtensionOpeningReductionGroup::num_terms)
        .sum::<usize>()
        != num_claims
        || true_input_claims.len() != num_claims
    {
        return Err(AkitaError::InvalidProof);
    }
    #[cfg(debug_assertions)]
    {
        let prover_claim = ExtensionOpeningReductionProver::input_claim_from_groups(&groups)?;
        debug_assert_eq!(
            prover_claim, true_input_claim,
            "extension-opening reduction input claim mismatch"
        );
    }
    let mut prover = ExtensionOpeningReductionProver::new(groups, true_input_claim)?;
    let (sumcheck, rho, batched_final_claim) = stream.prove_sumcheck(&mut prover)?;
    let final_terms = prover.final_terms().ok_or_else(|| {
        AkitaError::InvalidInput(format!(
            "{path} extension-opening reduction has not reached a final point"
        ))
    })?;
    let final_claims = final_terms
        .iter()
        .map(|(_, witness, factor)| *witness * *factor)
        .collect::<Vec<_>>();
    let expected_batched_final = final_claims
        .iter()
        .zip(&claim_coefficients)
        .fold(E::zero(), |acc, (&claim, &coefficient)| {
            acc + coefficient * claim
        });
    if batched_final_claim != expected_batched_final {
        return Err(AkitaError::InvalidInput(format!(
            "{path} extension-opening final oracle mismatch"
        )));
    }
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
        let term_range = opening_batch.root_group_claim_range(group_index)?;
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
    stream.final_claims(&opening_batch, &final_claims)?;
    let reduction = S::build_reduction(proof_partials, sumcheck, final_claims, final_factors);

    Ok(ProvedExtensionOpeningReduction {
        reduction,
        protocol_points,
    })
}

pub(in crate::protocol::core) fn build_extension_opening_reduction_group<
    F,
    E,
    P,
    B,
    const D: usize,
>(
    backend: &B,
    prepared: Option<&<B as ComputeBackendSetup<F>>::PreparedSetup>,
    polys: &[&P],
    claim_coefficients: &[E],
    tail_point: &[E],
    eta: &[E],
) -> Result<ExtensionOpeningReductionGroup<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + MulBaseUnreduced<F>,
    P: RootTensorSource<F, D>,
    B: ComputeBackendSetup<F> + for<'a> TensorProjectionKernel<P::TensorView<'a>, F, E, D>,
{
    let _span =
        tracing::info_span!("extension_opening_reduction_group", num_terms = polys.len()).entered();
    if polys.len() != claim_coefficients.len() {
        return Err(AkitaError::InvalidSize {
            expected: polys.len(),
            actual: claim_coefficients.len(),
        });
    }
    let factor = tensor_equality_factor_evals::<F, E>(tail_point, eta)?;
    let mut terms = Vec::with_capacity(polys.len());
    for (poly, &coefficient) in polys.iter().zip(claim_coefficients) {
        let witness = {
            let _span = tracing::info_span!("eor_packed_witness").entered();
            TensorProjectionKernel::packed_witness(backend, prepared, poly.tensor_view()?)?
        };
        terms.push(ExtensionOpeningReductionTerm::new(witness, coefficient));
    }
    ExtensionOpeningReductionGroup::new(terms, factor)
}

pub(in crate::protocol::core) type FoldedClaimEvals<F, const D: usize> =
    (Vec<CyclotomicRing<F, D>>, Vec<Vec<CyclotomicRing<F, D>>>);
