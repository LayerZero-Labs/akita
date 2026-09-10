use super::{
    BackendInstanceId, CommitmentExecutor, CommitmentOperationContext, CommitmentStatePolicy,
    PreparedCompression, PreparedFusedCommitment, PreparedInnerCommitment, PreparedOuterCommitment,
    StageResources,
};
use akita_error::AkitaError;
use akita_types::AkitaExpandedSetup;
use jolt_field::{CanonicalEncoding, Field};

/// Builder for independently registered commitment stage operations.
pub struct CommitmentExecutorBuilder<'a, F, SP>
where
    F: Field + CanonicalEncoding,
    SP: CommitmentStatePolicy<F>,
{
    setup: akita_types::AkitaSetupDescriptor,
    state_policy: SP,
    inner: Option<PreparedInnerCommitment<'a, F>>,
    outer: Option<PreparedOuterCommitment<'a, F>>,
    compression: Option<PreparedCompression<'a, F>>,
    fused: Option<PreparedFusedCommitment<'a, F>>,
}

impl<'a, F, SP> CommitmentExecutorBuilder<'a, F, SP>
where
    F: Field + CanonicalEncoding + 'static,
    SP: CommitmentStatePolicy<F>,
{
    /// Start a route builder bound to one explicit expanded setup.
    pub fn new(expanded: &AkitaExpandedSetup<F>, state_policy: SP) -> Self {
        Self {
            setup: expanded.descriptor().clone(),
            state_policy,
            inner: None,
            outer: None,
            compression: None,
            fused: None,
        }
    }

    /// Issue an opaque identity shared by registrations on one backend instance.
    pub fn issue_backend_instance(&self) -> BackendInstanceId {
        BackendInstanceId::issue()
    }

    /// Validate and bind reusable registration metadata to this executor setup.
    pub fn operation_context(
        &self,
        backend_instance: BackendInstanceId,
        name: &'static str,
        resources: StageResources<'a, F>,
    ) -> Result<CommitmentOperationContext<'a, F>, AkitaError> {
        if name.is_empty() {
            return Err(AkitaError::InvalidSetup(
                "commitment stage registration requires a diagnostic name".into(),
            ));
        }
        resources.validate_setup(&self.setup)?;
        Ok(CommitmentOperationContext {
            backend_instance,
            name,
            resources,
        })
    }

    /// Register the inner operation and its optional host-export edge.
    pub fn register_inner(
        &mut self,
        prepared: PreparedInnerCommitment<'a, F>,
    ) -> Result<(), AkitaError> {
        if self.inner.is_some() {
            return Err(AkitaError::InvalidSetup(
                "commitment executor inner operation is already registered".into(),
            ));
        }
        self.inner = Some(prepared);
        Ok(())
    }

    /// Register the independently selected outer operation.
    pub fn register_outer(
        &mut self,
        prepared: PreparedOuterCommitment<'a, F>,
    ) -> Result<(), AkitaError> {
        if self.outer.is_some() {
            return Err(AkitaError::InvalidSetup(
                "commitment executor outer operation is already registered".into(),
            ));
        }
        self.outer = Some(prepared);
        Ok(())
    }

    /// Register the independently selected compression operation.
    pub fn register_compression(
        &mut self,
        prepared: PreparedCompression<'a, F>,
    ) -> Result<(), AkitaError> {
        if self.compression.is_some() {
            return Err(AkitaError::InvalidSetup(
                "commitment executor compression operation is already registered".into(),
            ));
        }
        self.compression = Some(prepared);
        Ok(())
    }

    /// Explicitly select one fused inner-plus-outer operation for routes that
    /// contain both stages. Inner-only execution continues to use `inner`.
    pub fn register_fused(
        &mut self,
        prepared: PreparedFusedCommitment<'a, F>,
    ) -> Result<(), AkitaError> {
        if self.fused.is_some() {
            return Err(AkitaError::InvalidSetup(
                "commitment executor fused operation is already registered".into(),
            ));
        }
        self.fused = Some(prepared);
        Ok(())
    }

    /// Finish after at least one executable route has been registered.
    pub fn build(self) -> Result<CommitmentExecutor<'a, F, SP>, AkitaError> {
        let inner = self.inner;
        let outer = self.outer;
        let compression = self.compression;
        match (&inner, &outer, &self.fused) {
            (Some(inner), Some(outer), _) => {
                if !inner.owner.same_owner(&outer.owner) && inner.exporter.is_none() {
                    return Err(AkitaError::InvalidSetup(
                        "cross-owner outer route requires an inner-image exporter".into(),
                    ));
                }
            }
            (Some(_), None, _) => {}
            (None, None, Some(_)) => {}
            (None, None, None) => {
                return Err(AkitaError::InvalidSetup(
                    "commitment executor has no inner or fused operation".into(),
                ));
            }
            (None, Some(_), _) => {
                return Err(AkitaError::InvalidSetup(
                    "outer commitment registration requires a split inner operation".into(),
                ));
            }
        }
        Ok(CommitmentExecutor {
            setup: self.setup,
            inner,
            outer,
            compression,
            fused: self.fused,
            state_policy: self.state_policy,
        })
    }
}

#[cfg(test)]
#[path = "tests/builder.rs"]
mod tests;

#[cfg(test)]
#[path = "tests/fusion.rs"]
mod fusion_tests;

#[cfg(test)]
#[path = "tests/external_fusion.rs"]
mod external_fusion_tests;

#[cfg(test)]
#[path = "tests/source_selection.rs"]
mod source_selection_tests;
