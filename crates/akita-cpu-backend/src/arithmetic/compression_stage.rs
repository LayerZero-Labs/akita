use super::{CpuBackend, CpuPreparedSetup};
use crate::commitment::{
    BackendStateRef, CommitmentStateBinding, CompressionOperation, CompressionStageOutput,
    CompressionState, PortableCompressionState, PortableCompressionStateExport,
    StateOwnerCapability,
};
use crate::opaque::compression::{
    execute_compression_chains, CompressionExecutionInput, CompressionRelationOutput,
};
use crate::opaque::OperationCtx;
use akita_error::AkitaError;
use akita_types::{
    AkitaExpandedSetup, CompressionChainPlan, CompressionChainWitness, RingRelationMode, RingVec,
};
use jolt_field::{CanonicalEncoding, Field};
use std::mem::size_of;
use std::sync::Arc;

struct CpuCompressionRetention<F: Field> {
    witness: CompressionChainWitness,
    relation: CompressionRelationOutput<F>,
}

impl<F: Field> CpuCompressionRetention<F> {
    fn retained_bytes(&self) -> Result<usize, AkitaError> {
        let quotient_coefficients = match &self.relation {
            CompressionRelationOutput::QuotientLift { quotients } => quotients
                .iter()
                .try_fold(0usize, |total, quotient| {
                    total.checked_add(quotient.coeff_len())
                })
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("CPU compression quotient extent overflow".into())
                })?,
            CompressionRelationOutput::ReducedEvaluation => 0,
        };
        self.witness
            .retained_bytes()?
            .checked_add(
                quotient_coefficients
                    .checked_mul(size_of::<F>())
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "CPU compression quotient byte extent overflow".into(),
                        )
                    })?,
            )
            .ok_or_else(|| {
                AkitaError::InvalidSetup("CPU compression retained bytes overflow".into())
            })
    }
}

struct CpuCompressionExporter<F: Field> {
    owner: StateOwnerCapability<CompressionState>,
    marker: std::marker::PhantomData<fn() -> F>,
}

impl<F: Field + 'static> PortableCompressionStateExport<F> for CpuCompressionExporter<F> {
    fn export_compression_state(
        &self,
        state: &BackendStateRef<CompressionState>,
    ) -> Result<PortableCompressionState<F>, AkitaError> {
        let retained = self.owner.value::<CpuCompressionRetention<F>>(state)?;
        match (state.binding().relation_mode(), &retained.relation) {
            (
                Some(RingRelationMode::QuotientLift),
                CompressionRelationOutput::QuotientLift { quotients },
            ) if quotients.len() == retained.witness.plan().maps().len() => {
                PortableCompressionState::quotient_lift(retained.witness.clone(), quotients.clone())
            }
            (
                Some(RingRelationMode::ReducedEvaluation),
                CompressionRelationOutput::ReducedEvaluation,
            ) => PortableCompressionState::reduced_evaluation(retained.witness.clone()),
            _ => Err(AkitaError::InvalidInput(
                "CPU compression state disagrees with its bound relation mode".into(),
            )),
        }
    }

    fn consume_compression_state(
        &self,
        state: BackendStateRef<CompressionState>,
    ) -> Result<PortableCompressionState<F>, AkitaError> {
        let relation_mode = state.binding().relation_mode();
        match self.owner.try_unwrap::<CpuCompressionRetention<F>>(state)? {
            Ok(retained) => match (relation_mode, retained.relation) {
                (
                    Some(RingRelationMode::QuotientLift),
                    CompressionRelationOutput::QuotientLift { quotients },
                ) if quotients.len() == retained.witness.plan().maps().len() => {
                    PortableCompressionState::quotient_lift(retained.witness, quotients)
                }
                (
                    Some(RingRelationMode::ReducedEvaluation),
                    CompressionRelationOutput::ReducedEvaluation,
                ) => PortableCompressionState::reduced_evaluation(retained.witness),
                _ => Err(AkitaError::InvalidInput(
                    "CPU compression state disagrees with its bound relation mode".into(),
                )),
            },
            Err(shared) => self.export_compression_state(&shared),
        }
    }
}

/// CPU compression operation whose returned state directly owns its witness.
pub(crate) struct CpuCompressionOperation<
    'a,
    F,
    Cfg: akita_config::CommitmentConfig = akita_config::proof_optimized::fp128::OneHot,
> where
    F: Field + CanonicalEncoding,
{
    context: OperationCtx<'a, F, CpuBackend<Cfg>>,
    owner: StateOwnerCapability<CompressionState>,
}

impl<'a, F, Cfg> CpuCompressionOperation<'a, F, Cfg>
where
    F: Field + CanonicalEncoding,
    Cfg: akita_config::CommitmentConfig,
{
    /// Construct a CPU compression operation for a custom commitment executor.
    pub(crate) fn new(
        backend: &'a CpuBackend<Cfg>,
        prepared: &'a CpuPreparedSetup<F>,
        expanded: &AkitaExpandedSetup<F>,
    ) -> Result<Self, AkitaError> {
        Ok(Self {
            context: OperationCtx::new(backend, prepared, expanded)?,
            owner: StateOwnerCapability::new(),
        })
    }

    /// State owner used by this compression operation.
    pub(crate) fn owner(&self) -> &StateOwnerCapability<CompressionState> {
        &self.owner
    }

    /// Exporter for the retained portable compression witness.
    pub(crate) fn portable_exporter(&self) -> Arc<dyn PortableCompressionStateExport<F>>
    where
        F: 'static,
    {
        Arc::new(CpuCompressionExporter {
            owner: self.owner.clone(),
            marker: std::marker::PhantomData,
        })
    }

    #[cfg(test)]
    fn retained_shape(
        &self,
        state: &BackendStateRef<CompressionState>,
    ) -> Result<(usize, usize), AkitaError> {
        let retained = self.owner.value::<CpuCompressionRetention<F>>(state)?;
        let quotient_count = match &retained.relation {
            CompressionRelationOutput::QuotientLift { quotients } => quotients.len(),
            CompressionRelationOutput::ReducedEvaluation => 0,
        };
        Ok((retained.witness.stages().len(), quotient_count))
    }
}

impl<F, Cfg> CompressionOperation<F> for CpuCompressionOperation<'_, F, Cfg>
where
    F: Field + CanonicalEncoding + 'static,
    Cfg: akita_config::CommitmentConfig,
{
    fn compress(
        &self,
        binding: &CommitmentStateBinding,
        plan: &CompressionChainPlan,
        relation_mode: RingRelationMode,
        u: RingVec<F>,
    ) -> Result<CompressionStageOutput<F>, AkitaError> {
        if binding.relation_mode() != Some(relation_mode) {
            return Err(AkitaError::InvalidInput(
                "CPU compression relation mode disagrees with its registration binding".into(),
            ));
        }
        if u.coeff_len() != plan.source_coefficients() {
            return Err(AkitaError::InvalidSize {
                expected: plan.source_coefficients(),
                actual: u.coeff_len(),
            });
        }
        let (mut outputs, _) = execute_compression_chains(
            &self.context,
            vec![CompressionExecutionInput {
                id: (),
                plan: plan.clone(),
                coefficients: u.into_coeffs(),
                relation_mode,
            }],
        )?;
        let output = outputs.pop().ok_or(AkitaError::InvalidProof)?;
        let terminal_ring_dim = output
            .witness
            .plan()
            .maps()
            .last()
            .ok_or(AkitaError::InvalidProof)?
            .ring_dimension();
        let terminal_payload = RingVec::from_coeffs_with_ring_dim(
            output.terminal.into_coefficients(),
            terminal_ring_dim,
        )?;
        let retention = CpuCompressionRetention {
            witness: output.witness,
            relation: output.relation,
        };
        let retained_bytes = retention.retained_bytes()?;
        let resident = self.owner.bind(binding.clone(), retained_bytes, retention);
        CompressionStageOutput::new(terminal_payload, resident, plan, relation_mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commitment::CommitmentStateBinding;
    use crate::opaque::{CommitInnerPlan, ComputeBackendSetup};
    use crate::AkitaProverSetup;
    use akita_types::{SetupMatrixCapacity, SisModulusProfileId};
    use jolt_field::{Prime128OffsetA7F7, Ring};

    type F = Prime128OffsetA7F7;

    #[test]
    fn cpu_compression_retains_mode_specific_relation_state() {
        let plan =
            CompressionChainPlan::for_complete_source(SisModulusProfileId::Q128OffsetA7F7, 64)
                .unwrap();
        let capacity = plan
            .maps()
            .iter()
            .map(|map| map.input_width() * map.ring_dimension())
            .max()
            .unwrap();
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            8,
            1,
            SetupMatrixCapacity {
                num_field_elements: capacity,
            },
        )
        .unwrap();
        let backend = CpuBackend::for_arithmetic_tests();
        let prepared = backend.prepare_setup(&setup).unwrap();
        let operation =
            CpuCompressionOperation::new(&backend, &prepared, setup.expanded.as_ref()).unwrap();
        for (mode, expected_quotients) in [
            (RingRelationMode::ReducedEvaluation, 0),
            (RingRelationMode::QuotientLift, plan.maps().len()),
        ] {
            let binding = CommitmentStateBinding::new(
                setup.expanded.descriptor.clone(),
                CommitInnerPlan {
                    ring_dimension: 64,
                    num_live_blocks: 1,
                    n_a: 1,
                    num_positions_per_block: 1,
                    num_digits_inner: 1,
                    log_basis_inner: 1,
                },
                1,
                Some(mode),
                Some(plan.clone()),
            )
            .unwrap();
            let source = RingVec::from_coeffs_with_ring_dim(vec![F::from_u64(1); 64], 64).unwrap();
            let output = operation.compress(&binding, &plan, mode, source).unwrap();

            assert_eq!(
                operation.retained_shape(output.state()).unwrap(),
                (plan.maps().len(), expected_quotients)
            );
            let exported = operation
                .portable_exporter()
                .export_compression_state(output.state())
                .unwrap();
            assert_eq!(exported.witness().plan(), &plan);
            match mode {
                RingRelationMode::ReducedEvaluation => assert!(exported.quotients().is_none()),
                RingRelationMode::QuotientLift => {
                    assert_eq!(exported.quotients().unwrap().len(), plan.maps().len());
                }
            }
        }
    }
}
