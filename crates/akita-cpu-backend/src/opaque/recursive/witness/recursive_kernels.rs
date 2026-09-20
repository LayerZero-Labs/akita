use super::*;

macro_rules! impl_cpu_recursive_fold_kernels {
    ($backend:ty) => {
        impl<F, const D: usize>
            crate::opaque::consumer_kernels::RecursiveWitnessFoldKernel<OpaqueRecursiveWitness, F, D> for $backend
        where
            F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring + 'static,
            $backend: crate::opaque::ComputeBackendSetup<F>
                + for<'a> crate::opaque::FoldResponseKernel<
                    SuffixWitnessBatchView<'a, F, D>,
                    F,
                    D,
                    AcceptedFold = crate::opaque::CpuAcceptedFold<F>,
                >,
        {
            fn probe_recursive_witness(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                witness: &OpaqueRecursiveWitness,
                plan: &crate::opaque::ValidatedFoldProbePlan<'_>,
            ) -> Result<
                crate::opaque::FoldProbeOutcome<crate::opaque::CpuAcceptedFold<F>>,
                AkitaError,
            > {
                let source = SuffixWitnessBatchView {
                    polys: vec![witness.committed.as_ref().unwrap_or(&witness.logical)],
                    _marker: PhantomData,
                };
                crate::opaque::FoldResponseKernel::probe(self, prepared, source, plan)
            }
        }

        impl<F, const D: usize>
            crate::opaque::consumer_kernels::RecursiveWitnessTerminalFoldKernel<OpaqueRecursiveWitness, F, D>
            for $backend
        where
            F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring + 'static,
            $backend: crate::opaque::ComputeBackendSetup<F>
                + for<'a> crate::opaque::TerminalFoldResponseKernel<
                    SuffixWitnessBatchView<'a, F, D>,
                    F,
                    D,
                    AcceptedTerminalFold = crate::opaque::CpuAcceptedTerminalFold<F>,
                >,
        {
            fn probe_terminal_recursive_witness(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                witness: &OpaqueRecursiveWitness,
                plan: &crate::opaque::ValidatedTerminalFoldProbePlan<'_>,
            ) -> Result<
                crate::opaque::FoldProbeOutcome<
                    crate::opaque::CpuAcceptedTerminalFold<F>,
                >,
                AkitaError,
            > {
                let source = SuffixWitnessBatchView {
                    polys: vec![witness.committed.as_ref().unwrap_or(&witness.logical)],
                    _marker: PhantomData,
                };
                crate::opaque::TerminalFoldResponseKernel::probe_terminal(
                    self, prepared, source, plan,
                )
            }

            fn encode_terminal_recursive_witness(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                fold: &crate::opaque::CpuAcceptedTerminalFold<F>,
                plan: &crate::opaque::ValidatedTerminalZEncodingPlan,
            ) -> Result<Vec<u8>, AkitaError> {
                crate::opaque::TerminalFoldResponseKernel::encode_terminal_z(
                    self, prepared, fold, plan,
                )
            }
        }
    };
}

impl_cpu_recursive_fold_kernels!(crate::opaque::CpuBackend);

pub(crate) fn prepare_recursive_witness_opening<F, E, B, const D: usize>(
    backend: &B,
    prepared: Option<&B::PreparedSetup>,
    witness: &OpaqueRecursiveWitness,
    plan: &crate::opaque::ValidatedRecursiveGroupOpeningPlan<'_, E>,
) -> Result<
    crate::opaque::PreparedGroupOpening<
        E,
        <B as crate::opaque::PreparedOpeningHandleBackend<F, E>>::PreparedOpeningHandle,
    >,
    AkitaError,
>
where
    F: Field
        + CanonicalEncoding
        + akita_serialization::AkitaSerialize
        + Ring
        + jolt_field::Unreduced
        + 'static,
    <F as jolt_field::Unreduced>::Wide: From<F> + jolt_field::AdditiveGroup,
    E: akita_types::FpExtEncoding<F> + ExtField<F> + akita_serialization::AkitaSerialize,
    B: crate::opaque::ComputeBackendSetup<F>
        + crate::opaque::DigitRowsComputeBackend<F>
        + crate::opaque::PreparedGroupOpeningKernel<F, E>
        + for<'a> crate::opaque::OpeningFoldKernel<SuffixWitnessView<'a, F, D>, F, D>
        + for<'a> crate::opaque::SubringCoefficientPackingBatchKernel<
            SuffixWitnessBatchView<'a, F, D>,
            F,
            E,
            D,
        >,
{
    let prepared = prepared.ok_or_else(|| {
        AkitaError::InvalidInput("recursive opening requires prepared backend state".into())
    })?;
    let source = witness.committed.as_ref().unwrap_or(&witness.logical);
    if plan.ring_dimension() != D
        || plan.witness_len() != source.live_coeff_len()
    {
        return Err(AkitaError::InvalidInput(
            "recursive opening plan disagrees with its witness or operation context".into(),
        ));
    }
    let protocol_point = plan.point();
    let basis = plan.basis();
    let num_positions_per_block = plan.positions_per_block();
    let num_live_blocks = plan.live_blocks();
    let alpha_bits = plan.alpha_bits();
    let opening_method = plan.opening_method();
    if let akita_types::OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension,
    } = opening_method
    {
        let geometry = akita_types::SubringCoefficientPackingGeometry::try_new(
            E::DEGREE,
            D,
            challenge_subring_dimension,
        )?;
        let source_num_vars = <RecursiveWitnessFlat as RootPolyMeta<F>>::num_vars(source);
        let num_live_positions =
            <RecursiveWitnessFlat as RootPolyShape<F, D>>::num_live_ring_elems(source);
        if num_live_positions.div_ceil(num_positions_per_block) != num_live_blocks {
            return Err(AkitaError::InvalidInput(
                "coefficient-packing source shape disagrees with its group".into(),
            ));
        }
        let point = akita_types::PreparedSubringCoefficientPackingPoint::new(
            geometry,
            basis,
            num_live_positions,
            num_positions_per_block,
            source_num_vars,
            protocol_point,
        )?;
        let sources = [source];
        let batch = <RecursiveWitnessFlat as RootOpeningSource<F, D>>::opening_batch(&sources)?;
        let partials_by_claim = crate::opaque::SubringCoefficientPackingBatchKernel::coefficient_packing_partials_batch(
                backend,
                Some(prepared),
                batch,
                crate::opaque::SubringCoefficientPackingPlan { point: &point },
            )?;
        let [partials] = partials_by_claim.as_slice() else {
            return Err(AkitaError::InvalidProof);
        };
        let scalar = akita_types::coefficient_packing_scalar_opening::<F, E>(
            geometry,
            point.num_live_blocks(),
            core::slice::from_ref(partials),
            &[E::one()],
            point.live_block_weights(),
            point.tail_weights(),
        )?;
        return crate::opaque::PreparedGroupOpeningKernel::retain_coefficient_packing_opening(
            backend,
            None,
            Some(prepared),
            point,
            partials_by_claim,
            vec![scalar],
        );
    }

    let point = akita_types::prepare_opening_point::<F, E, D>(
        protocol_point,
        basis,
        num_positions_per_block,
        num_live_blocks,
        alpha_bits,
    )?;
    let plan = if let Some(base) = point.ring_multiplier_point.as_base() {
        crate::opaque::OpeningFoldPlan::Base {
            live_block_weights: &base.live_block_weights,
            position_weights: &base.position_weights,
            num_positions_per_block,
        }
    } else {
        crate::opaque::OpeningFoldPlan::Subfield {
            multipliers: point
                .ring_multiplier_point
                .as_subfield()
                .ok_or(AkitaError::InvalidProof)?,
            num_positions_per_block,
        }
    };
    let crate::opaque::OpeningFoldOutput { eval, folded } =
        crate::opaque::OpeningFoldKernel::evaluate_and_fold(
            backend,
            Some(prepared),
            source.view::<F, D>()?,
            plan,
        )?;
    let inner_point = &protocol_point[..protocol_point.len().min(alpha_bits)];
    let scalar = if E::DEGREE == 1 {
        (eval * point.packed_inner_trusted::<D>()?.sigma_m1())
            .coefficients()
            .first()
            .copied()
            .map(E::lift_base)
            .ok_or_else(|| AkitaError::InvalidInput("empty folded opening ring".into()))?
    } else {
        if !D.is_multiple_of(E::DEGREE) || !(D / E::DEGREE).is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "extension-field degree must divide the ring dimension into power-of-two slots"
                    .into(),
            ));
        }
        let packed_bits = (D / E::DEGREE).trailing_zeros() as usize;
        if inner_point.len() > packed_bits
            && inner_point[packed_bits..]
                .iter()
                .any(|value| !value.is_zero())
        {
            return Err(AkitaError::InvalidPointDimension {
                expected: packed_bits,
                actual: inner_point.len(),
            });
        }
        let mut reduced_point = inner_point[..inner_point.len().min(packed_bits)].to_vec();
        reduced_point.resize(packed_bits, E::zero());
        let weights = akita_types::basis_weights(&reduced_point, basis)?;
        let packed = akita_types::embed_ring_subfield_vector::<F, E, D>(
            &weights,
            AkitaError::InvalidInput(
                "root opening point does not encode in the ring-subfield basis".into(),
            ),
        )?;
        akita_types::recover_ring_subfield_inner_product::<F, E, D>(&eval, &packed)?
    };
    crate::opaque::PreparedGroupOpeningKernel::retain_evaluation_trace_opening(
        backend,
        None,
        Some(prepared),
        point,
        vec![RingVec::from_ring_elems(&folded).into_compact()],
        vec![scalar],
    )
}
