use super::{BackendKindId, PolynomialType};
use akita_error::AkitaError;
use akita_types::{RingRelationMode, RingRole};
use jolt_field::{CanonicalEncoding, Field};

/// Inner-operation capabilities used while compiling one request.
pub struct CommitmentRequestCapabilities {
    pub(crate) backend: BackendKindId,
    pub(crate) standard_types: Vec<PolynomialType>,
    pub(crate) accepts_any_standard_type: bool,
    pub(crate) external_context: std::any::TypeId,
    pub(crate) fused_command_context: Option<std::any::TypeId>,
}

impl CommitmentRequestCapabilities {
    /// Describe a split inner operation.
    pub fn split<Context: 'static>(
        backend: BackendKindId,
        standard_types: Vec<PolynomialType>,
    ) -> Self {
        Self {
            backend,
            standard_types,
            accepts_any_standard_type: false,
            external_context: std::any::TypeId::of::<Context>(),
            fused_command_context: None,
        }
    }

    /// Describe an explicitly fused inner/outer operation.
    pub fn fused<Context: 'static, Command: 'static>(
        backend: BackendKindId,
        standard_types: Vec<PolynomialType>,
    ) -> Self {
        Self {
            backend,
            standard_types,
            accepts_any_standard_type: false,
            external_context: std::any::TypeId::of::<Context>(),
            fused_command_context: Some(std::any::TypeId::of::<Command>()),
        }
    }

    /// Selected backend family.
    pub const fn backend(&self) -> BackendKindId {
        self.backend
    }

    /// Standard representations in operation-preference order.
    pub fn standard_types(&self) -> &[PolynomialType] {
        &self.standard_types
    }

    pub(crate) fn is_fused(&self) -> bool {
        self.fused_command_context.is_some()
    }

    pub(crate) fn accept_any_standard_type(&mut self) {
        self.accepts_any_standard_type = true;
    }

    pub(crate) fn validate(&self) -> Result<(), AkitaError> {
        if self
            .standard_types
            .iter()
            .enumerate()
            .any(|(index, entry)| self.standard_types[..index].contains(entry))
        {
            return Err(AkitaError::InvalidSetup(
                "inner operation advertised a duplicate polynomial type".into(),
            ));
        }
        Ok(())
    }
}

/// Ring dimensions implemented by one commitment stage operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageDimensionCapabilities {
    dimensions: Vec<usize>,
}

impl StageDimensionCapabilities {
    /// Construct a nonempty, duplicate-free dimension declaration.
    pub fn new(dimensions: Vec<usize>) -> Result<Self, AkitaError> {
        if dimensions.is_empty()
            || dimensions.iter().any(|dimension| {
                *dimension == 0
                    || !dimension.is_power_of_two()
                    || dimensions
                        .iter()
                        .filter(|other| *other == dimension)
                        .count()
                        != 1
            })
        {
            return Err(AkitaError::InvalidSetup(
                "stage dimensions must be nonzero, power-of-two, and duplicate-free".into(),
            ));
        }
        Ok(Self { dimensions })
    }

    pub(crate) fn cpu_role<F: Field + CanonicalEncoding>(role: RingRole) -> Self {
        let tier = akita_types::protocol_dispatch_tier::<F>();
        Self {
            dimensions: akita_types::SUPPORTED_COMMITMENT_RING_DIMS
                .into_iter()
                .filter(|dimension| {
                    akita_types::dispatch::role_dim_supported_for_tier(tier, role, *dimension)
                })
                .collect(),
        }
    }

    pub(crate) fn cpu_compression<F: Field + CanonicalEncoding>() -> Self {
        let tier = akita_types::protocol_dispatch_tier::<F>();
        let mut dimensions = vec![8, 16, 32];
        dimensions.extend(akita_types::SUPPORTED_COMMITMENT_RING_DIMS);
        dimensions.retain(|dimension| {
            akita_types::compression_ring_dim_supported_for_tier(tier, *dimension)
        });
        Self { dimensions }
    }

    /// Whether the operation implements this runtime ring dimension.
    pub fn supports(&self, dimension: usize) -> bool {
        self.dimensions.contains(&dimension)
    }

    /// Declared dimensions.
    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }
}

/// Dimension and relation-mode capabilities of one compression operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressionOperationCapabilities {
    dimensions: StageDimensionCapabilities,
    relation_modes: Vec<RingRelationMode>,
}

impl CompressionOperationCapabilities {
    /// Construct a checked compression capability declaration.
    pub fn new(
        dimensions: StageDimensionCapabilities,
        relation_modes: Vec<RingRelationMode>,
    ) -> Result<Self, AkitaError> {
        if relation_modes.is_empty()
            || relation_modes
                .iter()
                .any(|mode| relation_modes.iter().filter(|other| *other == mode).count() != 1)
        {
            return Err(AkitaError::InvalidSetup(
                "compression relation modes must be nonempty and duplicate-free".into(),
            ));
        }
        Ok(Self {
            dimensions,
            relation_modes,
        })
    }

    pub(crate) fn cpu<F: Field + CanonicalEncoding>() -> Self {
        Self {
            dimensions: StageDimensionCapabilities::cpu_compression::<F>(),
            relation_modes: vec![
                RingRelationMode::QuotientLift,
                RingRelationMode::ReducedEvaluation,
            ],
        }
    }

    pub(crate) fn supports(
        &self,
        dimensions: impl IntoIterator<Item = usize>,
        relation_mode: RingRelationMode,
    ) -> bool {
        self.relation_modes.contains(&relation_mode)
            && dimensions
                .into_iter()
                .all(|dimension| self.dimensions.supports(dimension))
    }

    /// Supported compression ring dimensions.
    pub const fn dimensions(&self) -> &StageDimensionCapabilities {
        &self.dimensions
    }

    /// Supported schedule-selected relation modes.
    pub fn relation_modes(&self) -> &[RingRelationMode] {
        &self.relation_modes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_metadata_rejects_duplicates_and_invalid_dimensions() {
        assert!(StageDimensionCapabilities::new(Vec::new()).is_err());
        assert!(StageDimensionCapabilities::new(vec![64, 64]).is_err());
        assert!(StageDimensionCapabilities::new(vec![63]).is_err());
        let dimensions = StageDimensionCapabilities::new(vec![64]).unwrap();
        assert!(CompressionOperationCapabilities::new(
            dimensions,
            vec![
                RingRelationMode::QuotientLift,
                RingRelationMode::QuotientLift
            ],
        )
        .is_err());
    }
}
