use super::{
    BackendStateRef, CommitmentExecutionMode, CommitmentStateBinding, CompressionState,
    FullCommitmentOutput, InnerImage, InnerImageExportOperation,
};
use crate::compute::CommitInnerPlan;
use akita_error::AkitaError;
use akita_types::{
    AkitaCommitmentHint, CompressionChainPlan, CompressionChainWitness, RingRelationMode, RingVec,
};
use jolt_field::Field;
use std::mem::size_of;
use std::sync::{Arc, Mutex};

/// Canonical host material exported from one retained compression state.
pub enum PortableCompressionState<F: Field> {
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

/// Owner-specific export edge for retained compression state.
pub trait PortableCompressionStateExport<F: Field>: Send + Sync {
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
pub struct CommitmentStateComponents<F: Field> {
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

    /// Complete immutable request binding.
    pub fn binding(&self) -> &CommitmentStateBinding {
        self.image.binding()
    }

    /// Execution mode represented by these components.
    pub const fn mode(&self) -> CommitmentExecutionMode {
        self.mode
    }

    /// Resident inner-image component.
    pub const fn image(&self) -> &BackendStateRef<InnerImage> {
        &self.image
    }

    /// Optional resident compression component.
    pub const fn compression(&self) -> Option<&BackendStateRef<CompressionState>> {
        self.compression.as_ref()
    }
}

fn into_portable<F: Field>(
    components: CommitmentStateComponents<F>,
) -> Result<AkitaCommitmentHint<F>, AkitaError> {
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

fn assemble_portable_hint<F: Field>(
    binding: CommitmentStateBinding,
    mode: CommitmentExecutionMode,
    rows: Vec<RingVec<F>>,
    compression: Option<PortableCompressionState<F>>,
) -> Result<AkitaCommitmentHint<F>, AkitaError> {
    match mode {
        CommitmentExecutionMode::InnerOnly | CommitmentExecutionMode::Uncompressed => {
            if compression.is_some() {
                return Err(AkitaError::InvalidInput(
                    "uncompressed portable state unexpectedly retained compression".into(),
                ));
            }
            AkitaCommitmentHint::new(binding.inner_plan().ring_dimension, rows)
        }
        CommitmentExecutionMode::Full => {
            let compression = compression.ok_or_else(|| {
                AkitaError::InvalidInput(
                    "compressed portable state omitted compression material".into(),
                )
            })?;
            match (binding.relation_mode(), compression) {
                (
                    Some(RingRelationMode::QuotientLift),
                    PortableCompressionState::QuotientLift { witness, quotients },
                ) => AkitaCommitmentHint::new_with_outer_compression(
                    binding.inner_plan().ring_dimension,
                    rows,
                    &witness,
                    &quotients,
                ),
                (
                    Some(RingRelationMode::ReducedEvaluation),
                    PortableCompressionState::ReducedEvaluation { witness },
                ) => {
                    let [row] = rows.try_into().map_err(|_: Vec<_>| {
                        AkitaError::InvalidInput(
                            "reduced portable commitment state must contain one source".into(),
                        )
                    })?;
                    AkitaCommitmentHint::singleton_with_reduced_outer_compression(row, &witness)
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
pub trait CommitmentStatePolicy<F: Field>: Send + Sync {
    /// Policy-selected prover state.
    type State;

    /// Consume checked stage components and bind the selected state.
    fn bind(&self, components: CommitmentStateComponents<F>) -> Result<Self::State, AkitaError>;
}

/// Retain the checked stage values through direct ownership.
#[derive(Debug, Default, Clone, Copy)]
pub struct ResidentStatePolicy;

/// Checked resident commitment state plus its owner-specific consumer routes.
#[derive(Clone)]
pub struct ResidentCommitmentState<F: Field> {
    binding: CommitmentStateBinding,
    mode: CommitmentExecutionMode,
    inner: Arc<Mutex<ResidentInnerState<F>>>,
    compression: Option<BackendStateRef<CompressionState>>,
    exporters: CommitmentStateExporters<F>,
}

struct ResidentInnerState<F: Field> {
    image: Option<BackendStateRef<InnerImage>>,
    material: Option<InnerRelationStateMaterial<F>>,
}

impl<F: Field> std::fmt::Debug for ResidentCommitmentState<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ResidentCommitmentState(..)")
    }
}

impl<F: Field> ResidentCommitmentState<F> {
    /// Immutable checked request binding.
    pub fn binding(&self) -> &CommitmentStateBinding {
        &self.binding
    }

    /// Backend-reported bytes retained by this complete commitment state.
    pub fn retained_bytes(&self) -> Result<usize, AkitaError> {
        let inner = self.inner.lock().map_err(|_| {
            AkitaError::InvalidInput("resident inner-relation material lock is poisoned".into())
        })?;
        let inner_bytes = match (&inner.image, &inner.material) {
            (Some(image), None) => image.retained_bytes(),
            (None, Some(material)) => material.retained_bytes()?,
            _ => {
                return Err(AkitaError::InvalidInput(
                    "resident inner state has inconsistent ownership".into(),
                ))
            }
        };
        let compression_bytes = self
            .compression
            .as_ref()
            .map_or(0, BackendStateRef::retained_bytes);
        akita_error::checked::sum([inner_bytes, compression_bytes]).ok_or_else(|| {
            AkitaError::InvalidInput("resident commitment retained-byte total overflow".into())
        })
    }

    fn frozen_inner_material(&self) -> Result<InnerRelationStateMaterial<F>, AkitaError> {
        let mut inner = self.inner.lock().map_err(|_| {
            AkitaError::InvalidInput("resident inner-relation material lock is poisoned".into())
        })?;
        if let Some(material) = inner.material.as_ref() {
            return Ok(material.clone());
        }
        let image = inner.image.take().ok_or_else(|| {
            AkitaError::InvalidInput("resident inner state omitted its image".into())
        })?;
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
        inner.material = Some(material.clone());
        Ok(material)
    }
}

/// State capability required by consumers that need the portable CPU image.
pub trait PortableCommitmentState<F: Field> {
    /// Export the canonical legacy state at this explicit consumer boundary.
    fn portable_hint(&self) -> Result<AkitaCommitmentHint<F>, AkitaError>;
}

/// Consuming portable export used by persistence boundaries.
pub trait IntoPortableCommitmentState<F: Field> {
    /// Consume private state and return its canonical portable hint.
    fn into_portable_hint(self) -> Result<AkitaCommitmentHint<F>, AkitaError>;
}

/// Canonical inner rows derived for ring-relation construction.
#[derive(Clone)]
pub struct InnerRelationStateMaterial<F: Field> {
    ring_dimension: usize,
    rows: Arc<Vec<RingVec<F>>>,
}

impl<F: Field> InnerRelationStateMaterial<F> {
    /// Construct canonical rows for one exact inner relation request.
    pub fn new(
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
            ring_dimension: plan.ring_dimension,
            rows: Arc::new(rows),
        })
    }

    pub(crate) fn from_rows(
        ring_dimension: usize,
        rows: Vec<RingVec<F>>,
    ) -> Result<Self, AkitaError> {
        if rows.iter().any(|row| !row.can_decode_vec(ring_dimension)) {
            return Err(AkitaError::InvalidInput(
                "inner-relation rows do not match their ring dimension".into(),
            ));
        }
        Ok(Self {
            ring_dimension,
            rows: Arc::new(rows),
        })
    }

    /// Validate and freeze rows exported for one exact commitment request.
    pub fn from_binding(
        binding: &CommitmentStateBinding,
        rows: Vec<RingVec<F>>,
    ) -> Result<Self, AkitaError> {
        Self::new(binding.inner_plan(), binding.source_count(), rows)
    }

    /// Runtime ring dimension of every row.
    pub const fn ring_dimension(&self) -> usize {
        self.ring_dimension
    }

    /// Borrow the canonical rows in source order.
    pub fn rows(&self) -> &[RingVec<F>] {
        self.rows.as_slice()
    }

    /// Consume the carrier and return the canonical rows.
    pub fn into_rows(self) -> Vec<RingVec<F>> {
        Arc::try_unwrap(self.rows).unwrap_or_else(|rows| rows.as_ref().clone())
    }

    /// Validate this material against the relation request that consumes it.
    pub fn validate(&self, plan: &CommitInnerPlan, source_count: usize) -> Result<(), AkitaError> {
        let expected_coefficients =
            akita_error::checked::product([plan.num_live_blocks, plan.n_a, plan.ring_dimension])
                .ok_or_else(|| {
                    AkitaError::InvalidInput("inner-relation row length overflow".into())
                })?;
        if self.ring_dimension != plan.ring_dimension
            || self.rows.len() != source_count
            || self.rows.iter().any(|row| {
                row.ring_dim() != plan.ring_dimension || row.coeff_len() != expected_coefficients
            })
        {
            return Err(AkitaError::InvalidInput(
                "inner-relation material does not match its commitment plan".into(),
            ));
        }
        Ok(())
    }

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
pub trait InnerRelationState<F: Field> {
    /// Validate that inner-relation material can be produced without exporting it.
    fn preflight_inner_relation(
        &self,
        plan: &CommitInnerPlan,
        source_count: usize,
    ) -> Result<(), AkitaError> {
        self.inner_relation_material()?.validate(plan, source_count)
    }

    /// Derive canonical inner rows without constructing a portable hint.
    fn inner_relation_material(&self) -> Result<InnerRelationStateMaterial<F>, AkitaError>;
}

/// Independent state capability used by outer-compression relations.
pub trait OuterCompressionState<F: Field> {
    /// Validate that compression-relation material can be produced without exporting it.
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
    ) -> Result<PortableCompressionState<F>, AkitaError>;
}

/// Canonical transcript message derived from terminal inner commitment state.
pub struct TerminalTFieldsMessage {
    bytes: Vec<u8>,
}

impl TerminalTFieldsMessage {
    /// Encode one canonical terminal inner row for transcript binding.
    pub fn from_row<F>(row: &RingVec<F>) -> Result<Self, AkitaError>
    where
        F: Field + jolt_field::CanonicalEncoding + akita_serialization::AkitaSerialize,
    {
        Ok(Self {
            bytes: akita_types::raw_field_segment_bytes(row)?,
        })
    }

    /// Canonical raw-field segment bytes absorbed by the transcript.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Independent state capability required by the terminal transcript binding.
pub trait TerminalBindingState<F: Field> {
    /// Derive the exact terminal inner-state transcript message.
    fn terminal_t_fields_message(&self) -> Result<TerminalTFieldsMessage, AkitaError>;
}

fn terminal_message_from_hint<F>(
    hint: &AkitaCommitmentHint<F>,
) -> Result<TerminalTFieldsMessage, AkitaError>
where
    F: Field + jolt_field::CanonicalEncoding + akita_serialization::AkitaSerialize,
{
    let [row] = hint.inner_rows() else {
        return Err(AkitaError::InvalidProof);
    };
    TerminalTFieldsMessage::from_row(row)
}

fn terminal_message_from_inner_material<F>(
    material: InnerRelationStateMaterial<F>,
) -> Result<TerminalTFieldsMessage, AkitaError>
where
    F: Field + jolt_field::CanonicalEncoding + akita_serialization::AkitaSerialize,
{
    let [row] = material
        .into_rows()
        .try_into()
        .map_err(|_: Vec<_>| AkitaError::InvalidProof)?;
    TerminalTFieldsMessage::from_row(&row)
}

impl<F> TerminalBindingState<F> for AkitaCommitmentHint<F>
where
    F: Field + jolt_field::CanonicalEncoding + akita_serialization::AkitaSerialize,
{
    fn terminal_t_fields_message(&self) -> Result<TerminalTFieldsMessage, AkitaError> {
        terminal_message_from_hint(self)
    }
}

impl<F> TerminalBindingState<F> for ResidentCommitmentState<F>
where
    F: Field + jolt_field::CanonicalEncoding + akita_serialization::AkitaSerialize,
{
    fn terminal_t_fields_message(&self) -> Result<TerminalTFieldsMessage, AkitaError> {
        terminal_message_from_inner_material(self.inner_relation_material()?)
    }
}

impl<F: Field> InnerRelationState<F> for AkitaCommitmentHint<F> {
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

    fn inner_relation_material(&self) -> Result<InnerRelationStateMaterial<F>, AkitaError> {
        InnerRelationStateMaterial::from_rows(self.ring_dim(), self.inner_rows().to_vec())
    }
}

impl<F: Field> InnerRelationState<F> for ResidentCommitmentState<F> {
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

    fn inner_relation_material(&self) -> Result<InnerRelationStateMaterial<F>, AkitaError> {
        self.frozen_inner_material()
    }
}

impl<F> OuterCompressionState<F> for AkitaCommitmentHint<F>
where
    F: Field + jolt_field::CanonicalEncoding + akita_serialization::AkitaSerialize,
{
    fn preflight_outer_compression(
        &self,
        plan: &CompressionChainPlan,
        relation_mode: RingRelationMode,
    ) -> Result<(), AkitaError> {
        self.outer_compression_material(plan, relation_mode)
            .map(drop)
    }

    fn outer_compression_material(
        &self,
        plan: &CompressionChainPlan,
        relation_mode: RingRelationMode,
    ) -> Result<PortableCompressionState<F>, AkitaError> {
        match relation_mode {
            RingRelationMode::QuotientLift => Ok(PortableCompressionState::QuotientLift {
                witness: self.outer_compression_witness(plan)?,
                quotients: self.outer_compression_quotients(plan)?,
            }),
            RingRelationMode::ReducedEvaluation => {
                Ok(PortableCompressionState::ReducedEvaluation {
                    witness: self.reduced_outer_compression_witness(plan)?,
                })
            }
        }
    }
}

impl<F: Field> OuterCompressionState<F> for ResidentCommitmentState<F> {
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
        let compression = self.compression.as_ref().ok_or_else(|| {
            AkitaError::InvalidInput("resident commitment state omitted compression state".into())
        })?;
        let material = self
            .exporters
            .compression
            .as_ref()
            .ok_or_else(|| {
                AkitaError::InvalidInput(
                    "commitment route has no outer-compression state operation".into(),
                )
            })?
            .export_compression_state(compression)?;
        let witness = match &material {
            PortableCompressionState::QuotientLift { witness, .. }
            | PortableCompressionState::ReducedEvaluation { witness } => witness,
        };
        if witness.plan() != plan {
            return Err(AkitaError::InvalidInput(
                "resident compression state does not match the requested chain plan".into(),
            ));
        }
        Ok(material)
    }
}

impl<F: Field> PortableCommitmentState<F> for AkitaCommitmentHint<F> {
    fn portable_hint(&self) -> Result<AkitaCommitmentHint<F>, AkitaError> {
        Ok(self.clone())
    }
}

impl<F: Field> PortableCommitmentState<F> for ResidentCommitmentState<F> {
    fn portable_hint(&self) -> Result<AkitaCommitmentHint<F>, AkitaError> {
        let rows = self.frozen_inner_material()?.rows().to_vec();
        let compression = match self.compression.as_ref() {
            Some(state) => Some(
                self.exporters
                    .compression
                    .as_ref()
                    .ok_or_else(|| {
                        AkitaError::InvalidInput(
                            "commitment route has no portable compression exporter".into(),
                        )
                    })?
                    .export_compression_state(state)?,
            ),
            None => None,
        };
        assemble_portable_hint(self.binding.clone(), self.mode, rows, compression)
    }
}

impl<F: Field> IntoPortableCommitmentState<F> for AkitaCommitmentHint<F> {
    fn into_portable_hint(self) -> Result<AkitaCommitmentHint<F>, AkitaError> {
        Ok(self)
    }
}

impl<F: Field> IntoPortableCommitmentState<F> for ResidentCommitmentState<F> {
    fn into_portable_hint(self) -> Result<AkitaCommitmentHint<F>, AkitaError> {
        let material = self.frozen_inner_material()?;
        let Self {
            binding,
            mode,
            inner,
            compression,
            exporters,
        } = self;
        drop(inner);
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
        assemble_portable_hint(binding, mode, material.into_rows(), compression)
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
            inner: Arc::new(Mutex::new(ResidentInnerState {
                image: Some(image),
                material: None,
            })),
            compression,
            exporters,
        })
    }
}

/// Explicitly release all prover-private commitment state after execution.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoRetainedStatePolicy;

impl<F: Field> CommitmentStatePolicy<F> for NoRetainedStatePolicy {
    type State = ();

    fn bind(&self, components: CommitmentStateComponents<F>) -> Result<Self::State, AkitaError> {
        drop(components);
        Ok(())
    }
}

/// Export the existing portable CPU hint and release resident stage state.
#[derive(Debug, Default, Clone, Copy)]
pub struct PortableStatePolicy;

impl<F: Field> CommitmentStatePolicy<F> for PortableStatePolicy {
    type State = AkitaCommitmentHint<F>;

    fn bind(&self, components: CommitmentStateComponents<F>) -> Result<Self::State, AkitaError> {
        into_portable(components)
    }
}

/// Public arithmetic output plus policy-selected prover-private state.
pub struct CommitmentExecutionOutput<F: Field, S> {
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
    pub const fn terminal_payload(&self) -> &RingVec<F> {
        &self.terminal_payload
    }

    /// Policy-selected prover-private state.
    pub const fn prover_state(&self) -> &S {
        &self.prover_state
    }

    /// Consume into the public payload and prover-private state.
    pub fn into_parts(self) -> (RingVec<F>, S) {
        (self.terminal_payload, self.prover_state)
    }
}
