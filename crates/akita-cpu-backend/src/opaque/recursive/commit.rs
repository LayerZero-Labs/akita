use super::{CpuWitnessHandle, RecursiveWitnessFlat};
use crate::commitment::CommitmentExecutor;

use akita_error::AkitaError;
use akita_types::dispatch_for_field;
use jolt_field::{CanonicalEncoding, ExtField, Field};

impl CpuWitnessHandle {
    pub(crate) fn from_cpu(
        inner: RecursiveWitnessFlat,
        commitment_ring_dimension: usize,
        binding: crate::opaque::OperationBinding,
    ) -> Result<Self, AkitaError> {
        let logical_len = inner.live_coeff_len();
        let commitment_domain_len =
            akita_types::witness_commitment_domain_len(logical_len, commitment_ring_dimension)?;
        Ok(Self {
            pending_successor: None,
            relation_plan: None,
            manifest: crate::opaque::RecursiveWitnessManifest::try_new(
                logical_len,
                commitment_domain_len,
                commitment_ring_dimension,
            )?,
            binding,
            logical: inner,
            committed: None,
        })
    }

    #[cfg(test)]
    pub(crate) fn live_coeff_len(&self) -> usize {
        self.logical.live_coeff_len()
    }

    pub(crate) fn align_for_commitment_ring_dim(self, ring_dim: usize) -> Result<Self, AkitaError> {
        let logical = self.logical.align_for_commitment_ring_dim(ring_dim)?;
        let manifest = crate::opaque::RecursiveWitnessManifest::try_new(
            logical.live_coeff_len(),
            akita_types::witness_commitment_domain_len(logical.live_coeff_len(), ring_dim)?,
            ring_dim,
        )?;
        Ok(Self {
            pending_successor: self.pending_successor,
            relation_plan: self.relation_plan,
            manifest,
            binding: self.binding,
            logical,
            committed: self
                .committed
                .map(|witness| witness.align_for_commitment_ring_dim(ring_dim))
                .transpose()?,
        })
    }

    pub(crate) fn tensor_pack<F, E, const D: usize>(&mut self) -> Result<(), AkitaError>
    where
        F: Field,
        E: ExtField<F>,
    {
        self.committed = Some(crate::opaque::tensor_pack_recursive_witness::<F, E, D>(
            &self.logical,
        )?);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn to_i8_digits(&self) -> Vec<i8> {
        self.logical.to_i8_digits()
    }
}

impl crate::opaque::RecursiveWitnessHandle for CpuWitnessHandle {
    fn manifest(&self) -> crate::opaque::RecursiveWitnessManifest {
        self.manifest
    }
}

pub(super) struct RecursiveCommitSource<'a>(&'a CpuWitnessHandle);

impl CpuWitnessHandle {
    pub(super) fn commitment_source(&self) -> RecursiveCommitSource<'_> {
        RecursiveCommitSource(self)
    }
}

impl<F: Field> crate::commitment::CommitmentSource<F> for RecursiveCommitSource<'_> {
    fn descriptor(&self) -> Result<crate::commitment::CommitSourceDescriptor, AkitaError> {
        crate::commitment::CommitmentSource::<F>::descriptor(
            self.0.committed.as_ref().unwrap_or(&self.0.logical),
        )
    }

    fn committed_centered_reach(
        &self,
        modulus: u128,
        centering_threshold: u128,
    ) -> Result<(u128, u128), AkitaError>
    where
        F: CanonicalEncoding,
    {
        crate::commitment::CommitmentSource::<F>::committed_centered_reach(
            self.0.committed.as_ref().unwrap_or(&self.0.logical),
            modulus,
            centering_threshold,
        )
    }

    fn available_polynomial_types(
        &self,
        plan: &crate::opaque::CommitInnerPlan,
    ) -> Result<crate::commitment::AvailablePolynomialTypes, AkitaError> {
        crate::commitment::CommitmentSource::<F>::available_polynomial_types(
            self.0.committed.as_ref().unwrap_or(&self.0.logical),
            plan,
        )
    }

    fn represent_as(
        &self,
        selected: crate::commitment::PolynomialTypeSelection,
        plan: &crate::opaque::CommitInnerPlan,
    ) -> Result<crate::commitment::PolynomialRepresentation<'_, F>, AkitaError> {
        crate::commitment::CommitmentSource::<F>::represent_as(
            self.0.committed.as_ref().unwrap_or(&self.0.logical),
            selected,
            plan,
        )
    }
}

impl<F, E> crate::opaque::OpaqueWitnessCommitKernel<F, E> for crate::opaque::CpuBackend<F, E>
where
    F: Field
        + CanonicalEncoding
        + akita_serialization::AkitaSerialize
        + jolt_field::Unreduced
        + jolt_field::WithCommitAccumulator
        + 'static,
    E: ExtField<F> + 'static,
{
    fn commit_witness(
        &self,
        mut witness: Self::WitnessHandle,
        plan: &crate::opaque::ValidatedRecursiveWitnessCommitPlan,
    ) -> Result<
        crate::opaque::WitnessCommitmentOutput<
            F,
            Self::WitnessHandle,
            Self::CommitmentMaterialHandle,
        >,
        AkitaError,
    > {
        let parent = witness.operation_binding();
        self.validate_binding(&parent)?;
        if witness.pending_successor.is_some() {
            return Err(AkitaError::InvalidInput(
                "witness already committed for its next level".into(),
            ));
        }
        let (schedule, _) = parent.scope_lease().proof_plan()?;
        let matches_schedule = match plan.parameters() {
            crate::opaque::WitnessCommitmentParameters::Recursive(parameters) => schedule
                .recursive_folds
                .get(parent.fold_level() as usize)
                .is_some_and(|fold| &fold.params == parameters),
            crate::opaque::WitnessCommitmentParameters::Terminal(parameters) => {
                parent.fold_level() as usize == schedule.recursive_folds.len()
                    && &schedule.terminal == parameters
            }
        };
        if !matches_schedule {
            return Err(AkitaError::InvalidInput(
                "witness commitment differs from the admitted successor".into(),
            ));
        }
        if witness.manifest.logical_len() != plan.logical_len()
            || akita_types::witness_commitment_domain_len(
                plan.logical_len(),
                plan.ring_dimension(),
            )? != plan.padded_len()
        {
            return Err(AkitaError::InvalidInput(
                "recursive commitment disagrees with witness geometry".into(),
            ));
        }
        let execution = match plan.parameters() {
            crate::opaque::WitnessCommitmentParameters::Recursive(parameters) => {
                crate::commitment::CommitmentExecutionPlan::for_recursive(
                    parameters,
                    parent.fold_level() as usize,
                    1,
                )?
            }
            crate::opaque::WitnessCommitmentParameters::Terminal(parameters) => {
                crate::commitment::CommitmentExecutionPlan::for_terminal(
                    parameters,
                    parent.fold_level() as usize,
                )?
            }
        };
        if execution.inner().ring_dimension != plan.ring_dimension() {
            return Err(AkitaError::InvalidProof);
        }
        witness = witness.align_for_commitment_ring_dim(plan.ring_dimension())?;
        let terminal = matches!(
            plan.parameters(),
            crate::opaque::WitnessCommitmentParameters::Terminal(_)
        );
        let tensor = match (terminal, plan.source_encoding()) {
            (true, _) => E::DEGREE != 1,
            (false, Some(akita_types::CommittedSourceEncoding::CanonicalCoefficientTable)) => false,
            (
                false,
                Some(akita_types::CommittedSourceEncoding::TensorSubfieldProjection {
                    extension_degree,
                }),
            ) if extension_degree == E::DEGREE => true,
            _ => {
                return Err(AkitaError::InvalidInput(
                    "invalid successor source encoding".into(),
                ))
            }
        };
        if tensor {
            dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Inner),
                F,
                plan.ring_dimension(),
                |D| witness.tensor_pack::<F, E, D>()
            )?;
        }
        let prepared = self.prepared()?;
        let executor = CommitmentExecutor::cpu(
            self,
            prepared,
            &prepared.expanded,
            Vec::new(),
            crate::commitment::PortableStatePolicy,
        )?;
        let source = witness.commitment_source();
        let sources: [&dyn crate::commitment::CommitmentSource<F>; 1] = [&source];
        let (public, state) = match execution.mode() {
            crate::commitment::CommitmentExecutionMode::Full => {
                let (public, state) = executor.execute_full(&execution, &sources)?.into_parts();
                (Some(public), state)
            }
            crate::commitment::CommitmentExecutionMode::Uncompressed => {
                let (public, state) = executor
                    .execute_uncompressed(&execution, &sources)?
                    .into_parts();
                (Some(public), state)
            }
            crate::commitment::CommitmentExecutionMode::InnerOnly => {
                (None, executor.execute_inner(&execution, &sources)?)
            }
        };
        let mut material = crate::opaque::CpuCommitmentMaterial::from_state(state, &execution, 1)?;
        let successor = self.next_level_binding(parent.clone())?;
        material.bind(successor.clone());
        material.bind_commitment(successor.operation_id(), public.clone());
        let binding = match (public, plan.binding()) {
            (Some(public), akita_types::NextWitnessBindingPolicy::OuterPayload) => {
                crate::opaque::NextWitnessBindingMessage::OuterPayload(public)
            }
            (None, akita_types::NextWitnessBindingPolicy::TerminalInnerState) => {
                crate::opaque::NextWitnessBindingMessage::TerminalInnerState(
                    crate::opaque::TerminalCommitmentMaterialKernel::terminal_message(
                        self, &material,
                    )?,
                )
            }
            _ => return Err(AkitaError::InvalidProof),
        };
        // One lineage identifies the next committed witness and its material.
        witness.set_operation_binding(parent.for_operation(successor.operation_id()));
        witness.pending_successor = Some(successor.fold_level());
        Ok(crate::opaque::WitnessCommitmentOutput::new(
            binding, witness, material,
        ))
    }
    fn advance_witness_level(&self, witness: &mut Self::WitnessHandle) -> Result<(), AkitaError> {
        let parent = witness.operation_binding();
        self.validate_binding(&parent)?;
        let next = parent
            .fold_level()
            .checked_add(1)
            .ok_or(AkitaError::InvalidProof)?;
        if witness.pending_successor != Some(next) {
            return Err(AkitaError::InvalidInput(
                "witness level transition requires its successor commitment".into(),
            ));
        }
        let binding = parent
            .for_level_operation(next, parent.operation_id())
            .with_group(None);
        self.validate_binding(&binding)?;
        witness.set_operation_binding(binding);
        witness.pending_successor = None;
        Ok(())
    }
}
