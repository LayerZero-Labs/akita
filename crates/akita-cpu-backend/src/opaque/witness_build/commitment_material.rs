use crate::commitment::{InnerRelationMaterial, OuterCompressionMaterial};
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::RingVec;
use jolt_field::{CanonicalEncoding, Field};

#[cfg(test)]
#[path = "commitment_binding_tests.rs"]
mod commitment_binding_tests;

/// Commitment-consumer state retained for recursive T construction.
/// Its coefficient rows are private to this consumer module.
#[derive(Clone)]
pub(crate) struct OpaqueInnerRelationState<F: Field> {
    material: crate::commitment::InnerRelationStateMaterial<F>,
    source_count: usize,
}

/// Retained compression state whose witness and quotient material is visible
/// only to the recursive-witness consumer.
pub(crate) struct OpaqueCompressionState<F: Field> {
    material: crate::commitment::PortableCompressionState<F>,
}

/// CPU consumer's private common material for setup-prefix and recursive state.
pub struct CpuCommitmentMaterialHandle<F: Field> {
    binding: crate::opaque::OperationBinding,
    pub(super) inner: crate::commitment::InnerRelationStateMaterial<F>,
    pub(super) compression: Option<crate::commitment::PortableCompressionState<F>>,
    pub(super) source_count: usize,
    commitment_id: Option<u128>,
    public_commitment: Option<RingVec<F>>,
}

impl<F: Field> CpuCommitmentMaterialHandle<F> {
    fn validate_terminal(
        &self,
        backend: &crate::opaque::CpuBackend<impl akita_config::CommitmentConfig>,
    ) -> Result<(), AkitaError> {
        backend.validate_binding(&self.binding)?;
        let (schedule, _) = backend.owner().proof_plan(self.binding.scope_id())?;
        if self.binding.fold_level() as usize != schedule.recursive_folds.len() + 1
            || self.commitment_id.is_none()
            || self.public_commitment.is_some()
            || self.compression.is_some()
        {
            return Err(AkitaError::InvalidInput(
                "material was not committed for terminal publication".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn bind_commitment(&mut self, id: u128, public: Option<RingVec<F>>) {
        self.commitment_id = Some(id);
        self.public_commitment = public;
    }

    pub(crate) fn validate_public_commitment(&self, public: &RingVec<F>) -> Result<(), AkitaError> {
        let expected = self.public_commitment.as_ref().ok_or_else(|| {
            AkitaError::InvalidInput("material has no admitted public commitment".into())
        })?;
        if expected.coeffs() != public.coeffs()
            || (public.ring_dim() != 0 && public.ring_dim() != expected.ring_dim())
        {
            return Err(AkitaError::InvalidInput(
                "public commitment differs from retained material".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn commitment_id(&self) -> Option<u128> {
        self.commitment_id
    }

    pub(crate) fn binding(&self) -> crate::opaque::OperationBinding {
        self.binding
    }

    pub(crate) fn bind(&mut self, binding: crate::opaque::OperationBinding) {
        self.binding = binding;
    }
}

impl<F> CpuCommitmentMaterialHandle<F>
where
    F: Field + CanonicalEncoding + AkitaSerialize + 'static,
{
    pub(crate) fn from_state<S>(
        state: S,
        plan: &crate::commitment::CommitmentExecutionPlan,
        source_count: usize,
    ) -> Result<Self, AkitaError>
    where
        S: crate::commitment::InnerRelationState<
                F,
                Material = crate::commitment::InnerRelationStateMaterial<F>,
            > + crate::commitment::OuterCompressionState<
                F,
                Material = crate::commitment::PortableCompressionState<F>,
            >,
    {
        let inner = state.inner_relation_material(plan.inner(), source_count)?;
        inner.validate_relation_material(plan.inner(), source_count)?;
        let compression = match (plan.compression(), plan.relation_mode()) {
            (Some(compression), Some(mode)) => {
                let material = state.outer_compression_material(compression, mode)?;
                material.validate_compression_material(compression, mode)?;
                Some(material)
            }
            (None, None) => None,
            _ => {
                return Err(AkitaError::InvalidSetup(
                    "commitment plan has inconsistent compression metadata".into(),
                ));
            }
        };
        Ok(Self {
            binding: crate::opaque::OperationBinding::legacy_unscoped(),
            commitment_id: None,
            public_commitment: None,
            inner,
            compression,
            source_count,
        })
    }
}

impl<F> crate::opaque::CommitmentRelationMaterial<F> for CpuCommitmentMaterialHandle<F>
where
    F: Field + CanonicalEncoding + Send + 'static,
{
    fn metadata(&self) -> crate::opaque::CommitmentMaterialMetadata {
        crate::opaque::CommitmentMaterialMetadata::try_new(
            self.inner.ring_dimension(),
            self.source_count,
            self.compression.is_some(),
        )
        .expect("CPU material has validated public geometry")
    }
}

pub(crate) type CpuCommitmentMaterial<F> = CpuCommitmentMaterialHandle<F>;

impl<F, Cfg> crate::opaque::TerminalCommitmentMaterialKernel<F, CpuCommitmentMaterial<F>>
    for crate::opaque::CpuBackend<Cfg>
where
    Cfg: akita_config::CommitmentConfig<Field = F>,
    F: Field + CanonicalEncoding + AkitaSerialize + Send + 'static,
{
    fn terminal_message(
        &self,
        material: &CpuCommitmentMaterial<F>,
    ) -> Result<crate::commitment::TerminalTFieldsMessage, AkitaError> {
        material.validate_terminal(self)?;
        crate::commitment::InnerRelationMaterial::terminal_message(&material.inner)
    }

    fn consume_terminal_row(
        &self,
        material: CpuCommitmentMaterial<F>,
    ) -> Result<RingVec<F>, AkitaError> {
        material.validate_terminal(self)?;
        crate::commitment::InnerRelationMaterial::into_terminal_row(material.inner)
    }
}

impl<F: Field> OpaqueCompressionState<F> {
    pub(crate) fn new(material: crate::commitment::PortableCompressionState<F>) -> Self {
        Self { material }
    }

    pub(crate) fn into_material(self) -> crate::commitment::PortableCompressionState<F> {
        self.material
    }
}

impl<F: Field> OpaqueInnerRelationState<F> {
    pub(crate) fn new(
        material: crate::commitment::InnerRelationStateMaterial<F>,
        source_count: usize,
    ) -> Self {
        Self {
            material,
            source_count,
        }
    }

    pub(crate) const fn ring_dimension(&self) -> usize {
        self.material.ring_dimension()
    }

    pub(crate) const fn source_count(&self) -> usize {
        self.source_count
    }

    pub(crate) fn rows(&self) -> &[RingVec<F>] {
        self.material.rows()
    }
}
