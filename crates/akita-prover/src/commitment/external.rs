use crate::compute::CommitInnerPlan;
use crate::CommitInnerWitness;
use akita_error::AkitaError;
use jolt_field::Field;
use std::any::{Any, TypeId};
use std::hash::{Hash, Hasher};

use crate::compute::CpuPreparedSetup;

pub(super) struct CpuBackendKind;

/// Declare an external inner operation compatible with the canonical CPU executor.
///
/// The private backend marker remains owned by Akita while downstream sources can
/// advertise a capability that is guaranteed to match [`super::CommitmentExecutor::cpu`].
pub fn cpu_external_inner_commitment_capability<
    Family: 'static,
    Algorithm: 'static,
    F: Field + 'static,
>(
    diagnostic_name: &'static str,
) -> Result<ExternalInnerCommitmentCapability, AkitaError> {
    ExternalInnerCommitmentCapability::new::<Family, Algorithm, CpuPreparedSetup<F>>(
        BackendKindId::of::<CpuBackendKind>("cpu")?,
        diagnostic_name,
    )
}

/// Recover the prepared CPU setup supplied to an external inner operation.
pub fn cpu_external_inner_prepared_setup<F: Field + 'static>(
    context: &dyn Any,
) -> Result<&CpuPreparedSetup<F>, AkitaError> {
    context
        .downcast_ref()
        .ok_or_else(|| AkitaError::InvalidInput("external CPU commitment context mismatch".into()))
}

/// Open, process-local backend family identity.
#[derive(Clone, Copy)]
pub struct BackendKindId {
    type_id: TypeId,
    name: &'static str,
}

impl PartialEq for BackendKindId {
    fn eq(&self, other: &Self) -> bool {
        self.type_id == other.type_id
    }
}

impl Eq for BackendKindId {}

impl Hash for BackendKindId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.type_id.hash(state);
    }
}

impl BackendKindId {
    /// Identify a backend family by a private marker type and diagnostic name.
    pub fn of<T: 'static>(name: &'static str) -> Result<Self, AkitaError> {
        if name.is_empty() {
            return Err(AkitaError::InvalidInput(
                "backend kind diagnostic name must not be empty".into(),
            ));
        }
        Ok(Self {
            type_id: TypeId::of::<T>(),
            name,
        })
    }

    /// Sanitized diagnostic name.
    pub const fn name(self) -> &'static str {
        self.name
    }
}

impl std::fmt::Debug for BackendKindId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_tuple("BackendKindId")
            .field(&self.name)
            .finish()
    }
}

/// Type identities expected by one external operation object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternalOperationIdentity {
    family: TypeId,
    algorithm: TypeId,
    context: TypeId,
}

impl ExternalOperationIdentity {
    /// Bind an operation to payload-family, algorithm, and context marker types.
    pub fn of<Family: 'static, Algorithm: 'static, Context: 'static>() -> Self {
        Self {
            family: TypeId::of::<Family>(),
            algorithm: TypeId::of::<Algorithm>(),
            context: TypeId::of::<Context>(),
        }
    }
}

/// Side-effect-free external inner-operation declaration.
#[derive(Debug, Clone, Copy)]
pub struct ExternalInnerCommitmentCapability {
    backend: BackendKindId,
    identity: ExternalOperationIdentity,
    diagnostic_name: &'static str,
    fused_command_context: Option<TypeId>,
}

impl PartialEq for ExternalInnerCommitmentCapability {
    fn eq(&self, other: &Self) -> bool {
        self.backend == other.backend
            && self.identity == other.identity
            && self.fused_command_context == other.fused_command_context
    }
}

impl Eq for ExternalInnerCommitmentCapability {}

impl ExternalInnerCommitmentCapability {
    /// Declare an ordinary external inner operation.
    pub fn new<Family: 'static, Algorithm: 'static, Context: 'static>(
        backend: BackendKindId,
        diagnostic_name: &'static str,
    ) -> Result<Self, AkitaError> {
        Self::build(
            backend,
            ExternalOperationIdentity::of::<Family, Algorithm, Context>(),
            diagnostic_name,
            None,
        )
    }

    /// Declare an operation that can encode into a fused command context.
    pub fn new_fused<Family: 'static, Algorithm: 'static, Context: 'static, Command: 'static>(
        backend: BackendKindId,
        diagnostic_name: &'static str,
    ) -> Result<Self, AkitaError> {
        Self::build(
            backend,
            ExternalOperationIdentity::of::<Family, Algorithm, Context>(),
            diagnostic_name,
            Some(TypeId::of::<Command>()),
        )
    }

    fn build(
        backend: BackendKindId,
        identity: ExternalOperationIdentity,
        diagnostic_name: &'static str,
        fused_command_context: Option<TypeId>,
    ) -> Result<Self, AkitaError> {
        if diagnostic_name.is_empty() {
            return Err(AkitaError::InvalidInput(
                "external commitment algorithm name must not be empty".into(),
            ));
        }
        Ok(Self {
            backend,
            identity,
            diagnostic_name,
            fused_command_context,
        })
    }

    /// Backend family selected by this capability.
    pub const fn backend(self) -> BackendKindId {
        self.backend
    }

    /// Sanitized algorithm name.
    pub const fn diagnostic_name(self) -> &'static str {
        self.diagnostic_name
    }

    /// Whether this capability can append work to a fused command.
    pub const fn supports_fusion(self) -> bool {
        self.fused_command_context.is_some()
    }

    pub(crate) const fn context_type_id(self) -> TypeId {
        self.identity.context
    }

    pub(crate) const fn fused_command_context_type_id(self) -> Option<TypeId> {
        self.fused_command_context
    }
}

/// One checked erased external payload.
#[derive(Clone, Copy)]
pub struct ExternalInnerCommitmentInput<'a> {
    capability: ExternalInnerCommitmentCapability,
    payload: &'a (dyn Any + Send + Sync),
}

impl ExternalInnerCommitmentInput<'_> {
    /// Capability bound to this payload.
    pub const fn capability(&self) -> ExternalInnerCommitmentCapability {
        self.capability
    }

    /// Safely recover the declared payload family.
    pub fn payload<T: 'static>(&self) -> Result<&T, AkitaError> {
        if self.capability.identity.family != TypeId::of::<T>() {
            return Err(AkitaError::InvalidInput(
                "external commitment payload family mismatch".into(),
            ));
        }
        self.payload.downcast_ref().ok_or_else(|| {
            AkitaError::InvalidInput("external commitment payload downcast failed".into())
        })
    }
}

/// Source-provided external inner commitment implementation.
pub trait ExternalInnerCommitmentOperation<F: Field>: Send + Sync {
    /// Type identities implemented by this object.
    fn identity(&self) -> ExternalOperationIdentity;

    /// Commit one homogeneous external-family group in source order.
    fn commit_group(
        &self,
        plan: &CommitInnerPlan,
        sources: &[ExternalInnerCommitmentInput<'_>],
        context: &dyn Any,
    ) -> Result<Vec<CommitInnerWitness<F>>, AkitaError>;
}

/// Optional encoder that appends A work without submitting a fused command.
pub trait ExternalFusedInnerCommitmentEncoder: Send + Sync {
    /// Required fused command-builder type.
    fn command_context_type_id(&self) -> TypeId;

    /// Append source-owned A work to an in-progress command builder.
    fn encode(
        &self,
        plan: &CommitInnerPlan,
        source: &ExternalInnerCommitmentInput<'_>,
        command: &mut dyn Any,
    ) -> Result<(), AkitaError>;
}

/// Checked prepared external operation and borrowed source payload.
pub struct PreparedExternalInnerCommitment<'a, F: Field> {
    input: ExternalInnerCommitmentInput<'a>,
    operation: &'a dyn ExternalInnerCommitmentOperation<F>,
    fused_encoder: Option<&'a dyn ExternalFusedInnerCommitmentEncoder>,
}

impl<'a, F: Field> PreparedExternalInnerCommitment<'a, F> {
    /// Pair a selected capability with its payload and operation after checking identities.
    pub fn new<Payload: Any + Send + Sync>(
        capability: ExternalInnerCommitmentCapability,
        payload: &'a Payload,
        operation: &'a dyn ExternalInnerCommitmentOperation<F>,
        fused_encoder: Option<&'a dyn ExternalFusedInnerCommitmentEncoder>,
    ) -> Result<Self, AkitaError> {
        if capability.identity.family != TypeId::of::<Payload>()
            || operation.identity() != capability.identity
            || fused_encoder.map(|encoder| encoder.command_context_type_id())
                != capability.fused_command_context
        {
            return Err(AkitaError::InvalidInput(
                "prepared external commitment identities do not match the selected capability"
                    .into(),
            ));
        }
        Ok(Self {
            input: ExternalInnerCommitmentInput {
                capability,
                payload,
            },
            operation,
            fused_encoder,
        })
    }

    /// Checked erased input.
    pub const fn input(&self) -> &ExternalInnerCommitmentInput<'a> {
        &self.input
    }

    pub(crate) const fn capability(&self) -> ExternalInnerCommitmentCapability {
        self.input.capability
    }

    /// Selected operation object.
    pub const fn operation(&self) -> &'a dyn ExternalInnerCommitmentOperation<F> {
        self.operation
    }

    /// Optional fused command encoder.
    pub const fn fused_encoder(&self) -> Option<&'a dyn ExternalFusedInnerCommitmentEncoder> {
        self.fused_encoder
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Prime64Offset59;
    use std::collections::HashSet;

    struct Family;
    struct OtherFamily;
    struct Algorithm;
    struct OtherAlgorithm;
    struct Context;
    struct Command;
    struct OtherCommand;
    struct Backend;

    struct Operation;

    #[test]
    fn diagnostic_names_do_not_change_compatibility_identity() {
        let first_backend = BackendKindId::of::<Backend>("first label").unwrap();
        let second_backend = BackendKindId::of::<Backend>("second label").unwrap();
        assert_eq!(first_backend, second_backend);
        assert_eq!(HashSet::from([first_backend, second_backend]).len(), 1);

        let first = ExternalInnerCommitmentCapability::new::<Family, Algorithm, Context>(
            first_backend,
            "first algorithm label",
        )
        .unwrap();
        let second = ExternalInnerCommitmentCapability::new::<Family, Algorithm, Context>(
            second_backend,
            "second algorithm label",
        )
        .unwrap();
        assert_eq!(first, second);
    }

    impl ExternalInnerCommitmentOperation<Prime64Offset59> for Operation {
        fn identity(&self) -> ExternalOperationIdentity {
            ExternalOperationIdentity::of::<Family, Algorithm, Context>()
        }

        fn commit_group(
            &self,
            _plan: &CommitInnerPlan,
            _sources: &[ExternalInnerCommitmentInput<'_>],
            _context: &dyn Any,
        ) -> Result<Vec<CommitInnerWitness<Prime64Offset59>>, AkitaError> {
            Ok(Vec::new())
        }
    }

    struct WrongOperation;

    impl ExternalInnerCommitmentOperation<Prime64Offset59> for WrongOperation {
        fn identity(&self) -> ExternalOperationIdentity {
            ExternalOperationIdentity::of::<Family, OtherAlgorithm, Context>()
        }

        fn commit_group(
            &self,
            _plan: &CommitInnerPlan,
            _sources: &[ExternalInnerCommitmentInput<'_>],
            _context: &dyn Any,
        ) -> Result<Vec<CommitInnerWitness<Prime64Offset59>>, AkitaError> {
            Ok(Vec::new())
        }
    }

    struct Encoder;

    impl ExternalFusedInnerCommitmentEncoder for Encoder {
        fn command_context_type_id(&self) -> TypeId {
            TypeId::of::<OtherCommand>()
        }

        fn encode(
            &self,
            _plan: &CommitInnerPlan,
            _source: &ExternalInnerCommitmentInput<'_>,
            _command: &mut dyn Any,
        ) -> Result<(), AkitaError> {
            Ok(())
        }
    }

    #[test]
    fn prepared_external_input_rejects_forged_family_operation_and_context() {
        struct Backend;
        let backend = BackendKindId::of::<Backend>("test").unwrap();
        let split = ExternalInnerCommitmentCapability::new::<Family, Algorithm, Context>(
            backend,
            "algorithm",
        )
        .unwrap();
        assert!(PreparedExternalInnerCommitment::<Prime64Offset59>::new(
            split,
            &OtherFamily,
            &Operation,
            None,
        )
        .is_err());
        assert!(PreparedExternalInnerCommitment::<Prime64Offset59>::new(
            split,
            &Family,
            &WrongOperation,
            None,
        )
        .is_err());

        let fused =
            ExternalInnerCommitmentCapability::new_fused::<Family, Algorithm, Context, Command>(
                backend,
                "algorithm",
            )
            .unwrap();
        assert!(PreparedExternalInnerCommitment::<Prime64Offset59>::new(
            fused,
            &Family,
            &Operation,
            Some(&Encoder),
        )
        .is_err());

        let prepared = PreparedExternalInnerCommitment::<Prime64Offset59>::new(
            split, &Family, &Operation, None,
        )
        .unwrap();
        assert!(prepared.input().payload::<OtherFamily>().is_err());
    }
}
