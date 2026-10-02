//! CPU compression of canonical B or D images without commitment orchestration.

use super::PortableCompressionState;
use crate::opaque::compression::{
    execute_compression_chains, CompressionExecutionInput, CompressionRelationOutput,
};
use crate::opaque::{CpuBackend, OperationCtx};
use akita_error::AkitaError;
use akita_params::{CompressionChainPlan, RingRelationMode};
use akita_types::RingVec;
use jolt_field::{CanonicalEncoding, Field};

/// Final compression-map payload (`p_F` or `p_H`) and private witness material.
/// The payload is unrelated to the protocol's terminal fold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortableCompressionOutput<F: Field> {
    compressed_payload: RingVec<F>,
    state: PortableCompressionState<F>,
}

impl<F: Field> PortableCompressionOutput<F> {
    /// Borrow the final compression-map payload.
    #[must_use]
    pub fn compressed_payload(&self) -> &RingVec<F> {
        &self.compressed_payload
    }

    /// Borrow the retained packed witnesses and mode-specific quotients.
    #[must_use]
    pub fn state(&self) -> &PortableCompressionState<F> {
        &self.state
    }

    /// Consume the output into its payload and private state.
    #[must_use]
    pub fn into_parts(self) -> (RingVec<F>, PortableCompressionState<F>) {
        (self.compressed_payload, self.state)
    }
}

impl<F: Field + CanonicalEncoding + 'static, E> CpuBackend<F, E> {
    /// Compress a canonical B or D relation image using the existing CPU kernels.
    /// No protocol sequencing or transcript operations are performed.
    ///
    /// # Errors
    /// Rejects incompatible field profiles, source sizes, setup capacity, or
    /// invalid compression results.
    pub fn compress_relation_image(
        &self,
        plan: &CompressionChainPlan,
        relation_mode: RingRelationMode,
        source: RingVec<F>,
    ) -> Result<PortableCompressionOutput<F>, AkitaError> {
        let prepared = self.prepared()?;
        let expanded = prepared.expanded();
        if plan.max_setup_field_elements()? > expanded.descriptor.num_field_elements {
            return Err(AkitaError::InvalidSetup(
                "relation compression exceeds setup capacity".into(),
            ));
        }
        let ctx = OperationCtx::new(self, prepared, expanded)?;
        let (mut outputs, _) = execute_compression_chains(
            &ctx,
            vec![CompressionExecutionInput {
                id: (),
                plan: plan.clone(),
                coefficients: source.into_coeffs(),
                relation_mode,
            }],
        )?;
        let output = outputs.pop().ok_or(AkitaError::InvalidProof)?;
        let ring_dimension = plan
            .maps()
            .last()
            .ok_or(AkitaError::InvalidProof)?
            .ring_dimension();
        let state = match output.relation {
            CompressionRelationOutput::QuotientLift { quotients } => {
                PortableCompressionState::quotient_lift(output.witness, quotients)?
            }
            CompressionRelationOutput::ReducedEvaluation => {
                PortableCompressionState::reduced_evaluation(output.witness)?
            }
        };
        state.validate(plan, relation_mode)?;
        Ok(PortableCompressionOutput {
            compressed_payload: RingVec::from_coeffs_with_ring_dim(
                output.terminal.into_coefficients(),
                ring_dimension,
            )?,
            state,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AkitaProverSetup;
    use akita_params::{SetupMatrixCapacity, SisModulusProfileId};
    use jolt_field::{Prime128OffsetA7F7, Ring};

    #[test]
    fn portable_relation_compression_matches_executor() {
        type F = Prime128OffsetA7F7;
        for source_len in [64, 512] {
            let plan = CompressionChainPlan::for_complete_source(
                SisModulusProfileId::Q128OffsetA7F7,
                source_len,
            )
            .unwrap();
            let setup = AkitaProverSetup::<F>::generate_with_capacity(
                8,
                1,
                SetupMatrixCapacity {
                    num_field_elements: plan.max_setup_field_elements().unwrap(),
                },
            )
            .unwrap();
            let backend = CpuBackend::<F, F>::new(setup.expanded.clone()).unwrap();
            let ctx =
                OperationCtx::new(&backend, backend.prepared().unwrap(), &setup.expanded).unwrap();
            for mode in [
                RingRelationMode::QuotientLift,
                RingRelationMode::ReducedEvaluation,
            ] {
                let coefficients = (0..source_len)
                    .map(|i| F::from_i64(i as i64 - 100))
                    .collect::<Vec<_>>();
                let actual = backend
                    .compress_relation_image(
                        &plan,
                        mode,
                        RingVec::from_coeffs_with_ring_dim(coefficients.clone(), 64).unwrap(),
                    )
                    .unwrap();
                let (mut expected, _) = execute_compression_chains(
                    &ctx,
                    vec![CompressionExecutionInput {
                        id: (),
                        plan: plan.clone(),
                        coefficients,
                        relation_mode: mode,
                    }],
                )
                .unwrap();
                let expected = expected.pop().unwrap();
                assert_eq!(
                    actual.compressed_payload().coeffs(),
                    expected.terminal.coefficients()
                );
                assert_eq!(
                    actual.compressed_payload().ring_dim(),
                    plan.maps().last().unwrap().ring_dimension()
                );
                assert_eq!(actual.state().witness(), &expected.witness);
                assert_eq!(actual.state().relation_mode(), mode);
                match expected.relation {
                    CompressionRelationOutput::QuotientLift { quotients } => {
                        assert_eq!(actual.state().quotients(), Some(quotients.as_slice()))
                    }
                    CompressionRelationOutput::ReducedEvaluation => {
                        assert!(actual.state().quotients().is_none())
                    }
                }
                assert!(backend
                    .compress_relation_image(
                        &plan,
                        mode,
                        RingVec::from_coeffs(vec![F::from_u64(1)])
                    )
                    .is_err());
                let wrong = CompressionChainPlan::for_complete_source(
                    SisModulusProfileId::Q64Offset59,
                    source_len,
                )
                .unwrap();
                assert!(backend
                    .compress_relation_image(
                        &wrong,
                        mode,
                        RingVec::from_coeffs(vec![F::from_u64(1); source_len])
                    )
                    .is_err());
                let (payload, state) = actual.clone().into_parts();
                assert_eq!(&payload, actual.compressed_payload());
                assert_eq!(&state, actual.state());
            }
            let small =
                AkitaProverSetup::<F>::generate_with_capacity(8, 1, SetupMatrixCapacity::minimum())
                    .unwrap();
            let backend = CpuBackend::<F, F>::new(small.expanded).unwrap();
            assert!(backend
                .compress_relation_image(
                    &plan,
                    RingRelationMode::QuotientLift,
                    RingVec::from_coeffs(vec![F::from_u64(1); source_len])
                )
                .is_err());
        }
    }
}
