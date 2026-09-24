use super::{
    BackendStateRef, CommitmentExecutionMode, CommitmentStateBinding, CompressionState,
    FullCommitmentOutput, InnerImage, InnerImageExportOperation,
};
use crate::commitment::PortableCommitmentHandle;
use crate::opaque::CommitInnerPlan;
use akita_error::AkitaError;
use akita_types::{CompressionChainPlan, CompressionChainWitness, RingRelationMode, RingVec};
use jolt_field::Field;
use std::sync::{Arc, Mutex};

/// Canonical shared host material exported from one retained compression state.
#[derive(Clone, PartialEq, Eq)]
pub struct PortableCompressionState<F: Field> {
    material: Arc<PortableCompressionMaterial<F>>,
}

#[derive(PartialEq, Eq)]
enum PortableCompressionMaterial<F: Field> {
    /// Packed witness and one quotient image per compression map.
    QuotientLift {
        /// Checked packed compression witness.
        witness: CompressionChainWitness,
        /// Canonical quotient images in map order.
        quotients: Vec<RingVec<F>>,
    },
    /// Packed witness with no quotient allocation.
    ReducedEvaluation {
        /// Checked packed compression witness.
        witness: CompressionChainWitness,
    },
}

impl<F: Field> PortableCompressionState<F> {
    /// Construct checked quotient-lift compression material.
    pub(crate) fn quotient_lift(
        witness: CompressionChainWitness,
        quotients: Vec<RingVec<F>>,
    ) -> Result<Self, AkitaError> {
        let material = Self {
            material: Arc::new(PortableCompressionMaterial::QuotientLift { witness, quotients }),
        };
        let plan = material.witness().plan().clone();
        material.validate(&plan, RingRelationMode::QuotientLift)?;
        Ok(material)
    }

    /// Construct checked reduced-evaluation compression material.
    pub(crate) fn reduced_evaluation(witness: CompressionChainWitness) -> Result<Self, AkitaError> {
        let material = Self {
            material: Arc::new(PortableCompressionMaterial::ReducedEvaluation { witness }),
        };
        let plan = material.witness().plan().clone();
        material.validate(&plan, RingRelationMode::ReducedEvaluation)?;
        Ok(material)
    }

    /// Borrow the checked compression-chain witness.
    pub fn witness(&self) -> &CompressionChainWitness {
        match self.material.as_ref() {
            PortableCompressionMaterial::QuotientLift { witness, .. }
            | PortableCompressionMaterial::ReducedEvaluation { witness } => witness,
        }
    }

    /// Borrow quotient rows when the material uses quotient-lift relations.
    pub fn quotients(&self) -> Option<&[RingVec<F>]> {
        match self.material.as_ref() {
            PortableCompressionMaterial::QuotientLift { quotients, .. } => Some(quotients),
            PortableCompressionMaterial::ReducedEvaluation { .. } => None,
        }
    }

    /// Relation mode represented by this checked compression material.
    #[cfg(test)]
    pub(crate) fn relation_mode(&self) -> RingRelationMode {
        match self.material.as_ref() {
            PortableCompressionMaterial::QuotientLift { .. } => RingRelationMode::QuotientLift,
            PortableCompressionMaterial::ReducedEvaluation { .. } => {
                RingRelationMode::ReducedEvaluation
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn shares_allocation_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.material, &other.material)
    }

    pub(crate) fn validate(
        &self,
        plan: &CompressionChainPlan,
        relation_mode: RingRelationMode,
    ) -> Result<(), AkitaError> {
        let valid = match (self.material.as_ref(), relation_mode) {
            (
                PortableCompressionMaterial::QuotientLift { witness, quotients },
                RingRelationMode::QuotientLift,
            ) => {
                witness.plan() == plan
                    && quotients.len() == plan.maps().len()
                    && quotients.iter().zip(plan.maps()).all(|(quotient, map)| {
                        quotient.ring_dim() == map.ring_dimension()
                            && quotient.coeff_len() == map.output_coefficients()
                    })
            }
            (
                PortableCompressionMaterial::ReducedEvaluation { witness },
                RingRelationMode::ReducedEvaluation,
            ) => witness.plan() == plan,
            _ => false,
        };
        if !valid {
            return Err(AkitaError::InvalidInput(
                "compression material does not match its relation mode or chain plan".into(),
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    fn retained_bytes(&self) -> Result<usize, AkitaError> {
        let quotient_coefficients = match self.quotients() {
            Some(quotients) => akita_error::checked::sum(quotients.iter().map(RingVec::coeff_len))
                .ok_or_else(|| {
                    AkitaError::InvalidInput("compression material retained bytes overflow".into())
                })?,
            None => 0,
        };
        self.witness()
            .retained_bytes()?
            .checked_add(
                quotient_coefficients
                    .checked_mul(size_of::<F>())
                    .ok_or_else(|| {
                        AkitaError::InvalidInput(
                            "compression material retained bytes overflow".into(),
                        )
                    })?,
            )
            .ok_or_else(|| {
                AkitaError::InvalidInput("compression material retained bytes overflow".into())
            })
    }
}

/// Owner-specific export edge for retained compression state.
pub(crate) trait PortableCompressionStateExport<F: Field>: Send + Sync {
    /// Export canonical mode-specific compression material.
    fn export_compression_state(
        &self,
        state: &BackendStateRef<CompressionState>,
    ) -> Result<PortableCompressionState<F>, AkitaError>;

    /// Consume resident compression state, moving its buffers when possible.
    fn consume_compression_state(
        &self,
        state: BackendStateRef<CompressionState>,
    ) -> Result<PortableCompressionState<F>, AkitaError> {
        self.export_compression_state(&state)
    }
}

/// Route-specific operations that can export retained stage state.
pub(super) struct CommitmentStateExporters<F: Field> {
    pub(super) inner: Option<Arc<dyn InnerImageExportOperation<F>>>,
    pub(super) compression: Option<Arc<dyn PortableCompressionStateExport<F>>>,
}

impl<F: Field> Clone for CommitmentStateExporters<F> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            compression: self.compression.clone(),
        }
    }
}

/// Checked consuming carrier for the state produced by one commitment route.
pub(crate) struct CommitmentStateComponents<F: Field> {
    mode: CommitmentExecutionMode,
    image: BackendStateRef<InnerImage>,
    compression: Option<BackendStateRef<CompressionState>>,
    exporters: CommitmentStateExporters<F>,
}

impl<F: Field> Clone for CommitmentStateComponents<F> {
    fn clone(&self) -> Self {
        Self {
            mode: self.mode,
            image: self.image.clone(),
            compression: self.compression.clone(),
            exporters: self.exporters.clone(),
        }
    }
}

impl<F: Field> CommitmentStateComponents<F> {
    /// Construct components after checking mode, relation, and request binding.
    pub(super) fn new(
        mode: CommitmentExecutionMode,
        image: BackendStateRef<InnerImage>,
        compression: Option<BackendStateRef<CompressionState>>,
        exporters: CommitmentStateExporters<F>,
    ) -> Result<Self, AkitaError> {
        let valid = match (mode, compression.as_ref()) {
            (CommitmentExecutionMode::Full, Some(state)) => {
                image.binding() == state.binding() && image.binding().relation_mode().is_some()
            }
            (CommitmentExecutionMode::Uncompressed | CommitmentExecutionMode::InnerOnly, None) => {
                image.binding().relation_mode().is_none()
            }
            _ => false,
        };
        if !valid {
            return Err(AkitaError::InvalidInput(
                "commitment state components disagree with their execution mode or binding".into(),
            ));
        }
        Ok(Self {
            mode,
            image,
            compression,
            exporters,
        })
    }
}

fn into_portable<F: Field>(
    components: CommitmentStateComponents<F>,
) -> Result<PortableCommitmentHandle<F>, AkitaError> {
    let CommitmentStateComponents {
        mode,
        image,
        compression,
        exporters,
    } = components;
    let binding = image.binding().clone();
    let rows = exporters
        .inner
        .as_ref()
        .ok_or_else(|| {
            AkitaError::InvalidInput("commitment route has no portable inner-image exporter".into())
        })?
        .consume_inner_rows(binding.inner_plan(), image)?;
    let rows = InnerRelationStateMaterial::from_binding(&binding, rows)?.into_rows();
    let compression = match compression {
        Some(state) => Some(
            exporters
                .compression
                .as_ref()
                .ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "commitment route has no portable compression exporter".into(),
                    )
                })?
                .consume_compression_state(state)?,
        ),
        None => None,
    };
    assemble_portable_hint(binding, mode, rows, compression)
}

pub(super) fn assemble_portable_hint<F: Field>(
    binding: CommitmentStateBinding,
    mode: CommitmentExecutionMode,
    rows: Vec<RingVec<F>>,
    compression: Option<PortableCompressionState<F>>,
) -> Result<PortableCommitmentHandle<F>, AkitaError> {
    match mode {
        CommitmentExecutionMode::InnerOnly | CommitmentExecutionMode::Uncompressed => {
            if compression.is_some() {
                return Err(AkitaError::InvalidInput(
                    "uncompressed portable state unexpectedly retained compression".into(),
                ));
            }
            PortableCommitmentHandle::new(binding.inner_plan().ring_dimension, rows)
        }
        CommitmentExecutionMode::Full => {
            let compression = compression.ok_or_else(|| {
                AkitaError::InvalidInput(
                    "compressed portable state omitted compression material".into(),
                )
            })?;
            let relation_mode = binding.relation_mode().ok_or_else(|| {
                AkitaError::InvalidInput(
                    "compressed portable state omitted its relation mode".into(),
                )
            })?;
            let compression_plan = binding.compression_plan().ok_or_else(|| {
                AkitaError::InvalidInput(
                    "compressed portable state omitted its compression plan".into(),
                )
            })?;
            compression.validate(compression_plan, relation_mode)?;
            match relation_mode {
                RingRelationMode::QuotientLift => {
                    PortableCommitmentHandle::new_with_outer_compression(
                        binding.inner_plan().ring_dimension,
                        rows,
                        compression.witness(),
                        compression.quotients().ok_or_else(|| {
                            AkitaError::InvalidInput(
                                "portable compression export omitted quotient rows".into(),
                            )
                        })?,
                    )
                }
                RingRelationMode::ReducedEvaluation if compression.quotients().is_none() => {
                    let [row] = rows.try_into().map_err(|_: Vec<_>| {
                        AkitaError::InvalidInput(
                            "reduced portable commitment state must contain one source".into(),
                        )
                    })?;
                    PortableCommitmentHandle::singleton_with_reduced_outer_compression(
                        row,
                        compression.witness(),
                    )
                }
                _ => Err(AkitaError::InvalidInput(
                    "portable compression export disagrees with the bound relation mode".into(),
                )),
            }
        }
    }
}

pub(super) fn validate_portable_export_route<F: Field>(
    exporters: &CommitmentStateExporters<F>,
    mode: CommitmentExecutionMode,
) -> Result<(), AkitaError> {
    if exporters.inner.is_none() {
        return Err(AkitaError::InvalidInput(
            "commitment route has no portable inner-image exporter".into(),
        ));
    }
    if mode == CommitmentExecutionMode::Full && exporters.compression.is_none() {
        return Err(AkitaError::InvalidInput(
            "commitment route has no portable compression exporter".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn validate_prover_state_consumer_route<F: Field>(
    exporters: &CommitmentStateExporters<F>,
    mode: CommitmentExecutionMode,
) -> Result<(), AkitaError> {
    if exporters.inner.is_none() {
        return Err(AkitaError::InvalidInput(
            "commitment route has no inner-relation state operation".into(),
        ));
    }
    if mode == CommitmentExecutionMode::Full && exporters.compression.is_none() {
        return Err(AkitaError::InvalidInput(
            "commitment route has no outer-compression state operation".into(),
        ));
    }
    Ok(())
}

/// Selects the prover-private state returned by commitment execution.
pub(crate) trait CommitmentStatePolicy<F: Field>: Send + Sync {
    /// Policy-selected prover state.
    type State;

    /// Consume checked stage components and bind the selected state.
    fn bind(&self, components: CommitmentStateComponents<F>) -> Result<Self::State, AkitaError>;
}

/// Retain the checked stage values through direct ownership.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ResidentStatePolicy;

/// Checked resident commitment state plus its owner-specific consumer routes.
#[derive(Clone)]
pub(crate) struct ResidentCommitmentState<F: Field> {
    binding: CommitmentStateBinding,
    mode: CommitmentExecutionMode,
    inner: Arc<Mutex<ResidentInnerState<F>>>,
    compression: Option<Arc<Mutex<ResidentCompressionState<F>>>>,
    exporters: CommitmentStateExporters<F>,
}

enum ResidentInnerState<F: Field> {
    Image(BackendStateRef<InnerImage>),
    Material(InnerRelationStateMaterial<F>),
    Failed,
}

enum ResidentCompressionState<F: Field> {
    Image(BackendStateRef<CompressionState>),
    Material(PortableCompressionState<F>),
    Failed,
}

impl<F: Field> std::fmt::Debug for ResidentCommitmentState<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ResidentCommitmentState(..)")
    }
}

impl<F: Field> ResidentCommitmentState<F> {
    /// Immutable checked request binding.
    #[cfg(test)]
    pub(crate) fn binding(&self) -> &CommitmentStateBinding {
        &self.binding
    }

    /// Backend-reported bytes retained by this complete commitment state.
    #[cfg(test)]
    pub(crate) fn retained_bytes(&self) -> Result<usize, AkitaError> {
        let inner = self.inner.lock().map_err(|_| {
            AkitaError::InvalidInput("resident inner-relation material lock is poisoned".into())
        })?;
        let inner_bytes = match &*inner {
            ResidentInnerState::Image(image) => image.retained_bytes(),
            ResidentInnerState::Material(material) => material.retained_bytes()?,
            ResidentInnerState::Failed => {
                return Err(AkitaError::InvalidInput(
                    "resident inner state export previously failed".into(),
                ));
            }
        };
        let compression_bytes = match self.compression.as_ref() {
            Some(compression) => {
                let compression = compression.lock().map_err(|_| {
                    AkitaError::InvalidInput(
                        "resident compression material lock is poisoned".into(),
                    )
                })?;
                match &*compression {
                    ResidentCompressionState::Image(image) => image.retained_bytes(),
                    ResidentCompressionState::Material(material) => material.retained_bytes()?,
                    ResidentCompressionState::Failed => {
                        return Err(AkitaError::InvalidInput(
                            "resident compression state export previously failed".into(),
                        ));
                    }
                }
            }
            None => 0,
        };
        akita_error::checked::sum([inner_bytes, compression_bytes]).ok_or_else(|| {
            AkitaError::InvalidInput("resident commitment retained-byte total overflow".into())
        })
    }

    fn frozen_inner_material(&self) -> Result<InnerRelationStateMaterial<F>, AkitaError> {
        let mut inner = self.inner.lock().map_err(|_| {
            AkitaError::InvalidInput("resident inner-relation material lock is poisoned".into())
        })?;
        let image = match std::mem::replace(&mut *inner, ResidentInnerState::Failed) {
            ResidentInnerState::Image(image) => image,
            ResidentInnerState::Material(material) => {
                let result = material.clone();
                *inner = ResidentInnerState::Material(material);
                return Ok(result);
            }
            ResidentInnerState::Failed => {
                return Err(AkitaError::InvalidInput(
                    "resident inner state export previously failed".into(),
                ));
            }
        };
        let rows = self
            .exporters
            .inner
            .as_ref()
            .ok_or_else(|| {
                AkitaError::InvalidInput(
                    "commitment route has no inner-relation state operation".into(),
                )
            })?
            .consume_inner_rows(self.binding.inner_plan(), image)?;
        let material = InnerRelationStateMaterial::from_binding(&self.binding, rows)?;
        *inner = ResidentInnerState::Material(material.clone());
        Ok(material)
    }

    fn frozen_compression_material(
        &self,
    ) -> Result<Option<PortableCompressionState<F>>, AkitaError> {
        let Some(compression) = self.compression.as_ref() else {
            return Ok(None);
        };
        let mut compression = compression.lock().map_err(|_| {
            AkitaError::InvalidInput("resident compression material lock is poisoned".into())
        })?;
        let image = match std::mem::replace(&mut *compression, ResidentCompressionState::Failed) {
            ResidentCompressionState::Image(image) => image,
            ResidentCompressionState::Material(material) => {
                let result = material.clone();
                *compression = ResidentCompressionState::Material(material);
                return Ok(Some(result));
            }
            ResidentCompressionState::Failed => {
                return Err(AkitaError::InvalidInput(
                    "resident compression state export previously failed".into(),
                ));
            }
        };
        let material = self
            .exporters
            .compression
            .as_ref()
            .ok_or_else(|| {
                AkitaError::InvalidInput(
                    "commitment route has no outer-compression state operation".into(),
                )
            })?
            .consume_compression_state(image)?;
        let plan = self.binding.compression_plan().ok_or_else(|| {
            AkitaError::InvalidInput("resident compression state omitted its chain plan".into())
        })?;
        let relation_mode = self.binding.relation_mode().ok_or_else(|| {
            AkitaError::InvalidInput("resident compression state omitted its relation mode".into())
        })?;
        material.validate(plan, relation_mode)?;
        *compression = ResidentCompressionState::Material(material.clone());
        Ok(Some(material))
    }
}

/// State capability required by consumers that need the portable CPU image.
#[cfg(test)]
pub(crate) trait PortableCommitmentState<F: Field> {
    /// Export the canonical legacy state at this explicit consumer boundary.
    fn portable_hint(&self) -> Result<PortableCommitmentHandle<F>, AkitaError>;
}

/// Consuming portable export used by persistence boundaries.
pub(crate) trait IntoPortableCommitmentState<F: Field> {
    /// Consume private state and return its canonical portable hint.
    fn into_portable_hint(self) -> Result<PortableCommitmentHandle<F>, AkitaError>;
}

/// Canonical inner rows derived for ring-relation construction.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct InnerRelationStateMaterial<F: Field> {
    plan: CommitInnerPlan,
    source_count: usize,
    rows: Arc<Vec<RingVec<F>>>,
}

impl<F: Field> InnerRelationStateMaterial<F> {
    /// Construct canonical rows for one exact inner relation request.
    pub(crate) fn new(
        plan: &CommitInnerPlan,
        source_count: usize,
        rows: Vec<RingVec<F>>,
    ) -> Result<Self, AkitaError> {
        let expected_coefficients =
            akita_error::checked::product([plan.num_live_blocks, plan.n_a, plan.ring_dimension])
                .ok_or_else(|| {
                    AkitaError::InvalidInput("inner-relation row length overflow".into())
                })?;
        if rows.len() != source_count
            || rows.iter().any(|row| {
                row.ring_dim() != plan.ring_dimension || row.coeff_len() != expected_coefficients
            })
        {
            return Err(AkitaError::InvalidInput(
                "inner-relation rows do not match their commitment plan".into(),
            ));
        }
        Ok(Self {
            plan: *plan,
            source_count,
            rows: Arc::new(rows),
        })
    }

    /// Validate and freeze rows exported for one exact commitment request.
    pub(crate) fn from_binding(
        binding: &CommitmentStateBinding,
        rows: Vec<RingVec<F>>,
    ) -> Result<Self, AkitaError> {
        Self::new(binding.inner_plan(), binding.source_count(), rows)
    }

    /// Runtime ring dimension of every row.
    pub(crate) const fn ring_dimension(&self) -> usize {
        self.plan.ring_dimension
    }

    /// Borrow the canonical rows in source order.
    pub(crate) fn rows(&self) -> &[RingVec<F>] {
        self.rows.as_slice()
    }

    /// Consume the carrier and return the canonical rows.
    pub(crate) fn into_rows(self) -> Vec<RingVec<F>> {
        Arc::try_unwrap(self.rows).unwrap_or_else(|rows| rows.as_ref().clone())
    }

    /// Validate this material against the relation request that consumes it.
    pub(crate) fn validate(
        &self,
        plan: &CommitInnerPlan,
        source_count: usize,
    ) -> Result<(), AkitaError> {
        if self.plan != *plan || self.source_count != source_count {
            return Err(AkitaError::InvalidInput(
                "inner-relation material does not match its commitment plan".into(),
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    fn retained_bytes(&self) -> Result<usize, AkitaError> {
        let coefficients = akita_error::checked::sum(self.rows.iter().map(RingVec::coeff_len))
            .ok_or_else(|| {
                AkitaError::InvalidInput("inner-relation coefficient total overflow".into())
            })?;
        akita_error::checked::product([coefficients, size_of::<F>()]).ok_or_else(|| {
            AkitaError::InvalidInput("inner-relation retained bytes overflow".into())
        })
    }
}

/// Independent state capability used by the inner ring relation.
pub(crate) trait InnerRelationState<F: Field> {
    /// Consumer-owned material transported to recursive witness assembly.
    type Material: InnerRelationMaterial<F>;

    /// Validate that inner-relation material can be produced without exporting it.
    fn preflight_inner_relation(
        &self,
        _plan: &CommitInnerPlan,
        _source_count: usize,
    ) -> Result<(), AkitaError> {
        Ok(())
    }

    /// Derive canonical inner rows without constructing a portable hint.
    fn inner_relation_material(
        &self,
        plan: &CommitInnerPlan,
        source_count: usize,
    ) -> Result<Self::Material, AkitaError>;
}

/// Validation-only contract for consumer-owned inner-relation material.
pub(crate) trait InnerRelationMaterial<F: Field>: Send + 'static {
    fn validate_relation_material(
        &self,
        plan: &CommitInnerPlan,
        source_count: usize,
    ) -> Result<(), AkitaError>;

    fn terminal_message(&self) -> Result<TerminalTFieldsMessage<F>, AkitaError>
    where
        F: jolt_field::CanonicalEncoding + akita_serialization::AkitaSerialize;

    fn into_terminal_row(self) -> Result<RingVec<F>, AkitaError>
    where
        Self: Sized;
}

impl<F: Field + 'static> InnerRelationMaterial<F> for InnerRelationStateMaterial<F> {
    fn validate_relation_material(
        &self,
        plan: &CommitInnerPlan,
        source_count: usize,
    ) -> Result<(), AkitaError> {
        self.validate(plan, source_count)
    }

    fn terminal_message(&self) -> Result<TerminalTFieldsMessage<F>, AkitaError>
    where
        F: jolt_field::CanonicalEncoding + akita_serialization::AkitaSerialize,
    {
        let [row] = self.rows() else {
            return Err(AkitaError::InvalidInput(
                "terminal inner relation material must contain one row".into(),
            ));
        };
        Ok(TerminalTFieldsMessage::from_row(row))
    }

    fn into_terminal_row(self) -> Result<RingVec<F>, AkitaError> {
        let [row] = self.into_rows().try_into().map_err(|_: Vec<_>| {
            AkitaError::InvalidInput("terminal inner relation material must contain one row".into())
        })?;
        Ok(row)
    }
}

/// Independent state capability used by outer-compression relations.
pub(crate) trait OuterCompressionState<F: Field> {
    /// Consumer-owned compression material transported to recursive assembly.
    type Material: OuterCompressionMaterial<F>;

    /// Validate that compression-relation material can be produced without exporting it.
    #[cfg(test)]
    fn preflight_outer_compression(
        &self,
        _plan: &CompressionChainPlan,
        _relation_mode: RingRelationMode,
    ) -> Result<(), AkitaError> {
        Ok(())
    }

    /// Derive the canonical compression witness and mode-specific relation data.
    fn outer_compression_material(
        &self,
        plan: &CompressionChainPlan,
        relation_mode: RingRelationMode,
    ) -> Result<Self::Material, AkitaError>;
}

/// Validation-only contract for consumer-owned outer-compression material.
pub(crate) trait OuterCompressionMaterial<F: Field>: Send + 'static {
    fn validate_compression_material(
        &self,
        plan: &CompressionChainPlan,
        relation_mode: RingRelationMode,
    ) -> Result<(), AkitaError>;
}

impl<F: Field + 'static> OuterCompressionMaterial<F> for PortableCompressionState<F> {
    fn validate_compression_material(
        &self,
        plan: &CompressionChainPlan,
        relation_mode: RingRelationMode,
    ) -> Result<(), AkitaError> {
        self.validate(plan, relation_mode)
    }
}

pub(crate) use akita_prover::backend::TerminalTFieldsMessage;

impl<F: Field + 'static> InnerRelationState<F> for PortableCommitmentHandle<F> {
    type Material = InnerRelationStateMaterial<F>;
    fn preflight_inner_relation(
        &self,
        plan: &CommitInnerPlan,
        source_count: usize,
    ) -> Result<(), AkitaError> {
        let expected_coefficients =
            akita_error::checked::product([plan.num_live_blocks, plan.n_a, plan.ring_dimension])
                .ok_or_else(|| {
                    AkitaError::InvalidInput("inner-relation row length overflow".into())
                })?;
        if self.ring_dim() != plan.ring_dimension
            || self.inner_rows().len() != source_count
            || self
                .inner_rows()
                .iter()
                .any(|row| row.coeff_len() != expected_coefficients)
        {
            return Err(AkitaError::InvalidInput(
                "portable commitment state does not match the requested inner plan".into(),
            ));
        }
        Ok(())
    }

    fn inner_relation_material(
        &self,
        plan: &CommitInnerPlan,
        source_count: usize,
    ) -> Result<InnerRelationStateMaterial<F>, AkitaError> {
        InnerRelationStateMaterial::new(plan, source_count, self.inner_rows().to_vec())
    }
}

impl<F: Field + 'static> InnerRelationState<F> for ResidentCommitmentState<F> {
    type Material = InnerRelationStateMaterial<F>;
    fn preflight_inner_relation(
        &self,
        plan: &CommitInnerPlan,
        source_count: usize,
    ) -> Result<(), AkitaError> {
        if self.binding.inner_plan() != plan || self.binding.source_count() != source_count {
            return Err(AkitaError::InvalidInput(
                "resident commitment state does not match the requested inner plan".into(),
            ));
        }
        if self.exporters.inner.is_none() {
            return Err(AkitaError::InvalidInput(
                "commitment route has no inner-relation state operation".into(),
            ));
        }
        Ok(())
    }

    fn inner_relation_material(
        &self,
        plan: &CommitInnerPlan,
        source_count: usize,
    ) -> Result<InnerRelationStateMaterial<F>, AkitaError> {
        self.preflight_inner_relation(plan, source_count)?;
        self.frozen_inner_material()
    }
}

impl<F> OuterCompressionState<F> for PortableCommitmentHandle<F>
where
    F: Field + jolt_field::CanonicalEncoding + akita_serialization::AkitaSerialize + 'static,
{
    type Material = PortableCompressionState<F>;
    #[cfg(test)]
    fn preflight_outer_compression(
        &self,
        plan: &CompressionChainPlan,
        relation_mode: RingRelationMode,
    ) -> Result<(), AkitaError> {
        self.validate_outer_compression_mode(plan, relation_mode)
    }

    fn outer_compression_material(
        &self,
        plan: &CompressionChainPlan,
        relation_mode: RingRelationMode,
    ) -> Result<PortableCompressionState<F>, AkitaError> {
        match relation_mode {
            RingRelationMode::QuotientLift => PortableCompressionState::quotient_lift(
                self.outer_compression_witness(plan)?,
                self.outer_compression_quotients(plan)?,
            ),
            RingRelationMode::ReducedEvaluation => PortableCompressionState::reduced_evaluation(
                self.reduced_outer_compression_witness(plan)?,
            ),
        }
    }
}

impl<F: Field + 'static> OuterCompressionState<F> for ResidentCommitmentState<F> {
    type Material = PortableCompressionState<F>;
    #[cfg(test)]
    fn preflight_outer_compression(
        &self,
        plan: &CompressionChainPlan,
        relation_mode: RingRelationMode,
    ) -> Result<(), AkitaError> {
        if self.mode != CommitmentExecutionMode::Full
            || self.binding().relation_mode() != Some(relation_mode)
            || self.binding().compression_plan() != Some(plan)
        {
            return Err(AkitaError::InvalidInput(
                "resident commitment state does not match the requested compression plan".into(),
            ));
        }
        if self.compression.is_none() || self.exporters.compression.is_none() {
            return Err(AkitaError::InvalidInput(
                "commitment route has no outer-compression state operation".into(),
            ));
        }
        Ok(())
    }

    fn outer_compression_material(
        &self,
        plan: &CompressionChainPlan,
        relation_mode: RingRelationMode,
    ) -> Result<PortableCompressionState<F>, AkitaError> {
        if self.mode != CommitmentExecutionMode::Full
            || self.binding.relation_mode() != Some(relation_mode)
            || self.binding.compression_plan() != Some(plan)
        {
            return Err(AkitaError::InvalidInput(
                "resident commitment state does not match the requested compression mode".into(),
            ));
        }
        let material = self.frozen_compression_material()?.ok_or_else(|| {
            AkitaError::InvalidInput("resident commitment state omitted compression state".into())
        })?;
        material.validate(plan, relation_mode)?;
        Ok(material)
    }
}

#[cfg(test)]
impl<F: Field> PortableCommitmentState<F> for PortableCommitmentHandle<F> {
    fn portable_hint(&self) -> Result<PortableCommitmentHandle<F>, AkitaError> {
        Ok(self.clone())
    }
}

#[cfg(test)]
impl<F: Field> PortableCommitmentState<F> for ResidentCommitmentState<F> {
    fn portable_hint(&self) -> Result<PortableCommitmentHandle<F>, AkitaError> {
        let rows = self.frozen_inner_material()?.rows().to_vec();
        let compression = self.frozen_compression_material()?;
        assemble_portable_hint(self.binding.clone(), self.mode, rows, compression)
    }
}

impl<F: Field> IntoPortableCommitmentState<F> for PortableCommitmentHandle<F> {
    fn into_portable_hint(self) -> Result<PortableCommitmentHandle<F>, AkitaError> {
        Ok(self)
    }
}

impl<F: Field> IntoPortableCommitmentState<F> for ResidentCommitmentState<F> {
    fn into_portable_hint(self) -> Result<PortableCommitmentHandle<F>, AkitaError> {
        let material = self.frozen_inner_material()?;
        let compression_material = self.frozen_compression_material()?;
        let Self {
            binding,
            mode,
            inner,
            compression,
            exporters,
        } = self;
        drop(inner);
        drop(compression);
        drop(exporters);
        assemble_portable_hint(binding, mode, material.into_rows(), compression_material)
    }
}

impl<F: Field> CommitmentStatePolicy<F> for ResidentStatePolicy {
    type State = ResidentCommitmentState<F>;

    fn bind(&self, components: CommitmentStateComponents<F>) -> Result<Self::State, AkitaError> {
        let CommitmentStateComponents {
            mode,
            image,
            compression,
            exporters,
        } = components;
        let binding = image.binding().clone();
        Ok(ResidentCommitmentState {
            binding,
            mode,
            inner: Arc::new(Mutex::new(ResidentInnerState::Image(image))),
            compression: compression
                .map(|state| Arc::new(Mutex::new(ResidentCompressionState::Image(state)))),
            exporters,
        })
    }
}

/// Explicitly release all prover-private commitment state after execution.
#[derive(Debug, Default, Clone, Copy)]
#[cfg(test)]
pub(crate) struct NoRetainedStatePolicy;

#[cfg(test)]
impl<F: Field> CommitmentStatePolicy<F> for NoRetainedStatePolicy {
    type State = ();

    fn bind(&self, components: CommitmentStateComponents<F>) -> Result<Self::State, AkitaError> {
        drop(components);
        Ok(())
    }
}

/// Export the existing portable CPU hint and release resident stage state.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct PortableStatePolicy;

impl<F: Field> CommitmentStatePolicy<F> for PortableStatePolicy {
    type State = PortableCommitmentHandle<F>;

    fn bind(&self, components: CommitmentStateComponents<F>) -> Result<Self::State, AkitaError> {
        into_portable(components)
    }
}

/// Public arithmetic output plus policy-selected prover-private state.
pub(crate) struct CommitmentExecutionOutput<F: Field, S> {
    terminal_payload: RingVec<F>,
    prover_state: S,
}

impl<F: Field, S> CommitmentExecutionOutput<F, S> {
    pub(super) fn from_full(
        output: FullCommitmentOutput<F>,
        policy: &impl CommitmentStatePolicy<F, State = S>,
        exporters: CommitmentStateExporters<F>,
    ) -> Result<Self, AkitaError> {
        let (image, compression) = output.into_parts();
        let (terminal_payload, compression) = compression.into_parts();
        let components = CommitmentStateComponents::new(
            CommitmentExecutionMode::Full,
            image,
            Some(compression),
            exporters,
        )?;
        let prover_state = policy.bind(components)?;
        Ok(Self {
            terminal_payload,
            prover_state,
        })
    }

    pub(super) fn from_uncompressed(
        output: super::UncompressedCommitmentOutput<F>,
        policy: &impl CommitmentStatePolicy<F, State = S>,
        exporters: CommitmentStateExporters<F>,
    ) -> Result<Self, AkitaError> {
        let (image, terminal_payload) = output.into_parts();
        let components = CommitmentStateComponents::new(
            CommitmentExecutionMode::Uncompressed,
            image,
            None,
            exporters,
        )?;
        let prover_state = policy.bind(components)?;
        Ok(Self {
            terminal_payload,
            prover_state,
        })
    }

    /// Public terminal compressed payload.
    #[cfg(test)]
    pub(crate) const fn terminal_payload(&self) -> &RingVec<F> {
        &self.terminal_payload
    }

    /// Policy-selected prover-private state.
    #[cfg(test)]
    pub(crate) const fn prover_state(&self) -> &S {
        &self.prover_state
    }

    /// Consume into the public payload and prover-private state.
    pub(crate) fn into_parts(self) -> (RingVec<F>, S) {
        (self.terminal_payload, self.prover_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commitment::StateOwnerCapability;
    use akita_types::{AkitaSetupDescriptor, AkitaSetupSeed};
    use jolt_field::Prime64Offset59;

    type F = Prime64Offset59;

    struct FailingInnerExport;

    impl InnerImageExportOperation<F> for FailingInnerExport {
        fn export_inner_rows(
            &self,
            _plan: &CommitInnerPlan,
            _image: &BackendStateRef<InnerImage>,
        ) -> Result<Vec<RingVec<F>>, AkitaError> {
            Err(AkitaError::InvalidInput(
                "intentional export failure".into(),
            ))
        }
    }

    #[test]
    fn failed_resident_freeze_has_stable_lifecycle_state() {
        let plan = CommitInnerPlan {
            ring_dimension: 64,
            num_live_blocks: 1,
            n_a: 1,
            num_positions_per_block: 1,
            num_digits_inner: 1,
            log_basis_inner: 1,
        };
        let binding = CommitmentStateBinding::new(
            AkitaSetupDescriptor {
                max_num_vars: 8,
                max_num_batched_polys: 1,
                num_field_elements: 64,
                setup_seed: AkitaSetupSeed::shake256_paged_v1([7; 32]),
            },
            plan,
            1,
            None,
            None,
        )
        .unwrap();
        let owner = StateOwnerCapability::<InnerImage>::new();
        let image = owner.bind(binding.clone(), 0, ());
        let state = ResidentCommitmentState {
            binding,
            mode: CommitmentExecutionMode::InnerOnly,
            inner: Arc::new(Mutex::new(ResidentInnerState::Image(image))),
            compression: None,
            exporters: CommitmentStateExporters {
                inner: Some(Arc::new(FailingInnerExport)),
                compression: None,
            },
        };

        let first = match state.inner_relation_material(&plan, 1) {
            Err(error) => error,
            Ok(_) => panic!("first export must fail"),
        };
        assert!(matches!(
            first,
            AkitaError::InvalidInput(message) if message.contains("intentional export failure")
        ));
        let second = match state.inner_relation_material(&plan, 1) {
            Err(error) => error,
            Ok(_) => panic!("failed state must stay failed"),
        };
        assert!(matches!(
            second,
            AkitaError::InvalidInput(message) if message.contains("previously failed")
        ));
        assert!(state.retained_bytes().is_err());
    }
}
