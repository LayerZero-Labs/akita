use crate::opaque::ProofScopeId;
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, Field};

/// Linear CPU state retained across transcript-owned fold grinding.
pub struct CpuWitnessBuildHandle<F: Field + CanonicalEncoding> {
    pub(crate) binding: OperationBinding,
    pub(crate) opening_bindings: Vec<OperationBinding>,
    pub(crate) assembly_state: crate::opaque::CpuRecursiveWitnessAssemblyState<F>,
    pub(crate) relation_rhs: akita_types::RingVec<F>,
    pub(crate) v: akita_types::RingVec<F>,
    pub(crate) level: akita_types::CommittedGroupParams,
    pub(crate) opening_batch: akita_types::OpeningClaimsLayout,
}

/// Owner, proof, setup, level, and operation identity attached to CPU state.
#[derive(Clone)]
pub(crate) struct OperationBinding {
    consumer_id: u64,
    scope_id: ProofScopeId,
    setup_digest: [u8; 32],
    fold_level: u32,
    operation_id: u128,
    group_index: Option<usize>,
    lease: crate::opaque::ScopeLease,
}

impl core::fmt::Debug for OperationBinding {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OperationBinding")
            .field("consumer_id", &self.consumer_id)
            .field("scope_id", &self.scope_id)
            .field("setup_digest", &self.setup_digest)
            .field("fold_level", &self.fold_level)
            .field("operation_id", &self.operation_id)
            .field("group_index", &self.group_index)
            .finish()
    }
}

impl PartialEq for OperationBinding {
    fn eq(&self, other: &Self) -> bool {
        self.consumer_id == other.consumer_id
            && self.scope_id == other.scope_id
            && self.setup_digest == other.setup_digest
            && self.fold_level == other.fold_level
            && self.operation_id == other.operation_id
            && self.group_index == other.group_index
    }
}

impl Eq for OperationBinding {}

impl OperationBinding {
    pub(crate) fn unbound() -> Self {
        Self::new(
            0,
            ProofScopeId::from_raw(0),
            [0; 32],
            0,
            0,
            crate::opaque::ScopeLease::unbound(),
        )
    }

    pub(crate) const fn group_index(&self) -> Option<usize> {
        self.group_index
    }
    pub(crate) const fn fold_level(&self) -> u32 {
        self.fold_level
    }
    pub(crate) fn with_group(mut self, group_index: Option<usize>) -> Self {
        self.group_index = group_index;
        self
    }
    pub(crate) fn validate_computation(&self, expected: &Self) -> Result<(), AkitaError> {
        self.validate_lineage(expected)?;
        if self.operation_id != expected.operation_id || self.group_index != expected.group_index {
            return Err(AkitaError::InvalidInput(
                "opaque handles belong to different computations".into(),
            ));
        }
        Ok(())
    }

    pub(crate) const fn operation_id(&self) -> u128 {
        self.operation_id
    }
    pub(crate) fn validate_group(
        &self,
        group_index: usize,
        group_count: usize,
    ) -> Result<(), AkitaError> {
        if self
            .group_index
            .map_or(group_index + 1 != group_count, |expected| {
                expected != group_index
            })
        {
            return Err(AkitaError::InvalidInput(
                "private handle belongs to a different source group".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn new(
        consumer_id: u64,
        scope_id: ProofScopeId,
        setup_digest: [u8; 32],
        fold_level: u32,
        operation_id: u128,
        lease: crate::opaque::ScopeLease,
    ) -> Self {
        Self {
            consumer_id,
            scope_id,
            setup_digest,
            fold_level,
            operation_id,
            group_index: None,
            lease,
        }
    }

    pub(crate) fn validate_owner(
        &self,
        consumer_id: u64,
        setup_digest: [u8; 32],
        fold_level: u32,
    ) -> Result<(), AkitaError> {
        if self.consumer_id != consumer_id
            || self.setup_digest != setup_digest
            || self.fold_level != fold_level
        {
            return Err(AkitaError::InvalidInput(
                "opaque handle belongs to a different consumer, setup, or level".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_lineage(&self, expected: &Self) -> Result<(), AkitaError> {
        if self.consumer_id != expected.consumer_id
            || self.scope_id != expected.scope_id
            || self.setup_digest != expected.setup_digest
            || self.fold_level != expected.fold_level
        {
            return Err(AkitaError::InvalidInput(
                "opaque handle belongs to a different consumer, scope, setup, or level".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn for_operation(&self, operation_id: u128) -> Self {
        Self {
            operation_id,
            ..self.clone()
        }
    }

    pub(crate) fn for_level_operation(&self, fold_level: u32, operation_id: u128) -> Self {
        Self {
            fold_level,
            operation_id,
            ..self.clone()
        }
    }

    pub(crate) const fn scope_id(&self) -> ProofScopeId {
        self.scope_id
    }

    pub(crate) const fn scope_lease(&self) -> &crate::opaque::ScopeLease {
        &self.lease
    }
}
