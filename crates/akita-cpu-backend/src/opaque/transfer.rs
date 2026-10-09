//! CPU packets, directed bridges, and atomic adoption into local handles.
use super::*;
use crate::commitment::{
    CommitmentExecutionPlan, InnerRelationStateMaterial, PortableCompressionState,
};
use crate::sources::packed_digits::PackedSignedDigits;
use akita_error::checked;
use akita_params::{CompressionChainWitness, PackedNegativeBinary, RingRelationMode};
use akita_serialization::AkitaSerialize;
use akita_types::RingVec;
use jolt_field::ExtField;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SuccessorSection {
    LogicalDigits,
    InnerRows,
    CompressionDigits(usize),
    CompressionQuotient(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SuccessorEncoding {
    SignedI8,
    PackedSigned { bit_width: u8 },
    CanonicalField,
    NegativeBinary,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SuccessorSectionDescriptor {
    pub section: SuccessorSection,
    pub encoding: SuccessorEncoding,
    pub coefficients: usize,
    pub bytes: usize,
}

/// Public portable CPU packet format; native packets use the same shape description.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuPacketDescriptor {
    pub metadata: HandoffMetadata,
    pub sections: Vec<SuccessorSectionDescriptor>,
}

impl CpuPacketDescriptor {
    /// Validate identity, encodings, and every length before allocating or decoding sections.
    pub fn validate<F: Field + CanonicalEncoding>(
        &self,
        plan: &ValidatedSuccessorHandoffPlan<'_, F>,
    ) -> Result<(), AkitaError> {
        plan.validate_metadata(&self.metadata)?;
        let compression = plan.compression()?;
        let maps = compression.as_ref().map_or(&[][..], |(p, _)| p.maps());
        let max_sections = checked::sum([2, maps.len(), maps.len()])
            .ok_or_else(|| AkitaError::InvalidInput("successor section count overflow".into()))?;
        if self.sections.len() > max_sections {
            return Err(AkitaError::InvalidInput(
                "too many successor sections".into(),
            ));
        }
        for (index, section) in self.sections.iter().enumerate() {
            if self.sections[..index]
                .iter()
                .any(|s| s.section == section.section)
            {
                return Err(AkitaError::InvalidInput(
                    "duplicate successor section".into(),
                ));
            }
            let coefficients = match section.section {
                SuccessorSection::LogicalDigits => plan.commitment().logical_len(),
                SuccessorSection::InnerRows => plan.inner_coefficients()?,
                SuccessorSection::CompressionDigits(i) => maps
                    .get(i)
                    .ok_or_else(|| AkitaError::InvalidInput("unexpected compression stage".into()))?
                    .real_digit_count(),
                SuccessorSection::CompressionQuotient(i) => {
                    if !matches!(compression, Some((_, RingRelationMode::QuotientLift))) {
                        return Err(AkitaError::InvalidInput(
                            "unexpected compression quotient".into(),
                        ));
                    }
                    maps.get(i)
                        .ok_or_else(|| AkitaError::InvalidInput("unexpected quotient map".into()))?
                        .output_coefficients()
                }
            };
            let bytes = match (section.section, section.encoding) {
                (SuccessorSection::LogicalDigits, SuccessorEncoding::SignedI8) => coefficients,
                (
                    SuccessorSection::LogicalDigits,
                    SuccessorEncoding::PackedSigned { bit_width },
                ) if (1..=8).contains(&bit_width) => {
                    checked::product([coefficients, bit_width as usize])
                        .and_then(|n| checked::div_ceil(n, 8))
                        .ok_or_else(|| {
                            AkitaError::InvalidInput("packed successor length overflow".into())
                        })?
                }
                (
                    SuccessorSection::InnerRows | SuccessorSection::CompressionQuotient(_),
                    SuccessorEncoding::CanonicalField,
                ) => checked::product([coefficients, F::NUM_BYTES]).ok_or_else(|| {
                    AkitaError::InvalidInput("successor field section overflow".into())
                })?,
                (SuccessorSection::CompressionDigits(i), SuccessorEncoding::NegativeBinary) => maps
                    .get(i)
                    .ok_or_else(|| AkitaError::InvalidInput("missing compression map".into()))?
                    .packed_digit_bytes(),
                _ => {
                    return Err(AkitaError::InvalidInput(
                        "invalid successor section encoding".into(),
                    ))
                }
            };
            if section.coefficients != coefficients || section.bytes != bytes {
                return Err(AkitaError::InvalidInput(
                    "successor section length differs from canonical geometry".into(),
                ));
            }
        }
        for required in [SuccessorSection::LogicalDigits, SuccessorSection::InnerRows] {
            if !self.sections.iter().any(|s| s.section == required) {
                return Err(AkitaError::InvalidInput("missing successor section".into()));
            }
        }
        for i in 0..maps.len() {
            for required in [
                Some(SuccessorSection::CompressionDigits(i)),
                matches!(compression, Some((_, RingRelationMode::QuotientLift)))
                    .then_some(SuccessorSection::CompressionQuotient(i)),
            ]
            .into_iter()
            .flatten()
            {
                if !self.sections.iter().any(|s| s.section == required) {
                    return Err(AkitaError::InvalidInput(
                        "missing compression section".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

struct CpuSuccessorStorage<F: Field> {
    logical: RecursiveWitnessFlat,
    committed: Option<RecursiveWitnessFlat>,
    inner: InnerRelationStateMaterial<F>,
    compression: Option<PortableCompressionState<F>>,
    source: OperationBinding,
}

/// Encoded sections accepted by the public CPU import constructor.
pub type CpuPacketSections = Vec<(SuccessorSection, Vec<u8>)>;

/// An owned snapshot of a committed successor. Its native storage remains private.
/// Portable sections carry logical digits; CPU import derives tensor packing locally.
pub struct CpuExportPacket<F: Field> {
    descriptor: CpuPacketDescriptor,
    storage: CpuSuccessorStorage<F>,
}

/// CPU import representation. External bridges construct it from encoded sections.
/// CPU-to-CPU edges retain shared storage instead of encoding and copying it.
pub struct CpuImportPacket<F: Field> {
    descriptor: CpuPacketDescriptor,
    sections: CpuPacketSections,
    native: Option<CpuSuccessorStorage<F>>,
}

impl<F: Field> CpuImportPacket<F> {
    /// Check section presence, uniqueness, and byte lengths. Adoption additionally
    /// checks the admitted plan, canonical encodings, and witness digit bounds.
    /// Inner rows and compression material are trusted to reproduce the public
    /// outer commitment; import checks their shape and encoding, and the verifier
    /// checks commitment consistency. Terminal rows are cross-checked on import.
    pub fn new(
        descriptor: CpuPacketDescriptor,
        sections: CpuPacketSections,
    ) -> Result<Self, AkitaError> {
        if sections.len() != descriptor.sections.len() {
            return Err(AkitaError::InvalidInput(
                "CPU packet section count mismatch".into(),
            ));
        }
        for (index, (section, bytes)) in sections.iter().enumerate() {
            let mut matches = descriptor.sections.iter().filter(|s| s.section == *section);
            if matches.next().is_none_or(|s| s.bytes != bytes.len())
                || matches.next().is_some()
                || sections[..index].iter().any(|(s, _)| s == section)
            {
                return Err(AkitaError::InvalidInput(
                    "CPU packet section shape mismatch".into(),
                ));
            }
        }
        Ok(Self {
            descriptor,
            sections,
            native: None,
        })
    }

    fn take_section(&mut self, section: SuccessorSection) -> Result<Vec<u8>, AkitaError> {
        let index = self
            .sections
            .iter()
            .position(|(s, _)| *s == section)
            .ok_or_else(|| AkitaError::InvalidInput("CPU packet section is absent".into()))?;
        Ok(self.sections.swap_remove(index).1)
    }
}
impl<F: Field + Send + Sync + 'static> SuccessorImportPacket for CpuImportPacket<F> {
    fn metadata(&self) -> &HandoffMetadata {
        &self.descriptor.metadata
    }
}

impl<F: Field + CanonicalEncoding> CpuExportPacket<F> {
    /// Encode sections for a bridge targeting another backend representation.
    pub fn into_sections(self) -> Result<(CpuPacketDescriptor, CpuPacketSections), AkitaError> {
        let sections = encode_sections(&self.descriptor, &self.storage)?;
        Ok((self.descriptor, sections))
    }
}
impl<F: Field + CanonicalEncoding> CpuImportPacket<F> {
    /// Materialize a native packet into the same public representation accepted by `new`.
    pub fn into_sections(self) -> Result<(CpuPacketDescriptor, CpuPacketSections), AkitaError> {
        let sections = match self.native {
            Some(storage) => encode_sections(&self.descriptor, &storage)?,
            None => self.sections,
        };
        Ok((self.descriptor, sections))
    }
}
fn encode_sections<F: Field + CanonicalEncoding>(
    descriptor: &CpuPacketDescriptor,
    storage: &CpuSuccessorStorage<F>,
) -> Result<CpuPacketSections, AkitaError> {
    storage.source.scope_lease().validate(
        storage.source.scope_id(),
        storage.source.fold_level(),
        None,
    )?;
    descriptor
        .sections
        .iter()
        .map(|section| {
            let bytes = match section.section {
                SuccessorSection::LogicalDigits => {
                    storage.logical.packed_digits().encoded_bytes().to_vec()
                }
                SuccessorSection::CompressionDigits(index) => storage
                    .compression
                    .as_ref()
                    .and_then(|c| c.witness().stages().get(index))
                    .ok_or_else(|| {
                        AkitaError::InvalidInput("compression digit section is absent".into())
                    })?
                    .bytes()
                    .to_vec(),
                field_section => {
                    let fields = match field_section {
                        SuccessorSection::InnerRows => storage
                            .inner
                            .rows()
                            .first()
                            .ok_or_else(|| {
                                AkitaError::InvalidInput("inner rows are absent".into())
                            })?
                            .coeffs(),
                        SuccessorSection::CompressionQuotient(index) => storage
                            .compression
                            .as_ref()
                            .and_then(|c| c.quotients())
                            .and_then(|q| q.get(index))
                            .ok_or_else(|| {
                                AkitaError::InvalidInput("compression quotient is absent".into())
                            })?
                            .coeffs(),
                        _ => return Err(AkitaError::InvalidInput("unknown field section".into())),
                    };
                    let mut bytes = Vec::new();
                    bytes.try_reserve_exact(section.bytes).map_err(|_| {
                        AkitaError::InvalidInput("CPU packet allocation failed".into())
                    })?;
                    bytes.resize(section.bytes, 0);
                    for (field, output) in fields.iter().zip(bytes.chunks_exact_mut(F::NUM_BYTES)) {
                        field.to_bytes_le(output);
                    }
                    bytes
                }
            };
            Ok((section.section, bytes))
        })
        .collect()
}

impl<F, E> SuccessorBridge<CpuBackend<F, E>, CpuBackend<F, E>, F, E>
    for Edge<CpuBackend<F, E>, CpuBackend<F, E>>
where
    F: Field + CanonicalEncoding + AkitaSerialize + Send + Sync + 'static,
    E: ExtField<F> + Send + Sync + 'static,
{
    fn convert(
        packet: CpuExportPacket<F>,
        _plan: &ValidatedSuccessorHandoffPlan<'_, F>,
    ) -> Result<CpuImportPacket<F>, AkitaError> {
        Ok(CpuImportPacket {
            descriptor: packet.descriptor,
            sections: Vec::new(),
            native: Some(packet.storage),
        })
    }
}

fn execution<F: Field + CanonicalEncoding>(
    plan: &ValidatedSuccessorHandoffPlan<'_, F>,
) -> Result<CommitmentExecutionPlan, AkitaError> {
    match plan.commitment().parameters() {
        WitnessCommitmentParameters::Recursive(p) => {
            CommitmentExecutionPlan::for_recursive(p, plan.producer_level(), 1)
        }
        WitnessCommitmentParameters::Terminal(p) => {
            CommitmentExecutionPlan::for_terminal(p, plan.producer_level())
        }
    }
}

impl CpuWitnessHandle {
    fn from_imported_successor<F: Field + CanonicalEncoding>(
        logical: RecursiveWitnessFlat,
        committed: Option<RecursiveWitnessFlat>,
        plan: &ValidatedSuccessorHandoffPlan<'_, F>,
        binding: OperationBinding,
    ) -> Result<Self, AkitaError> {
        if binding.fold_level() as usize != plan.successor_level()
            || logical.live_coeff_len() != plan.commitment().logical_len()
        {
            return Err(AkitaError::InvalidInput(
                "imported witness differs from successor level or length".into(),
            ));
        }
        Ok(Self {
            phase: WitnessPhase::ReadyInput(binding.fold_level()),
            relation_plan: None,
            manifest: plan.manifest()?,
            binding,
            logical,
            committed,
        })
    }
}

impl<F, E> SuccessorExportKernel<F, E> for CpuBackend<F, E>
where
    F: Field + CanonicalEncoding + AkitaSerialize + Send + Sync + 'static,
    E: ExtField<F> + Send + Sync + 'static,
{
    type ExportPacket = CpuExportPacket<F>;

    fn export_successor(
        &self,
        session: &Self::ProofSessionHandle,
        witness: &Self::WitnessHandle,
        material: &Self::CommitmentMaterialHandle,
        plan: &ValidatedSuccessorHandoffPlan<'_, F>,
    ) -> Result<Self::ExportPacket, AkitaError> {
        let scope = session.validate_owner(self.owner())?;
        let parent = witness.operation_binding();
        self.validate_binding(&parent)?;
        let producer = u32::try_from(plan.producer_level())
            .map_err(|_| AkitaError::InvalidInput("producer level exceeds u32".into()))?;
        let successor = u32::try_from(plan.successor_level())
            .map_err(|_| AkitaError::InvalidInput("successor level exceeds u32".into()))?;
        if parent.scope_id() != scope.scope_id()
            || parent.fold_level() != producer
            || witness.phase
                != (WitnessPhase::CommittedForSuccessor {
                    producer,
                    successor,
                })
            || witness.manifest() != plan.manifest()?
        {
            return Err(AkitaError::InvalidInput(
                "export requires the pending committed successor in this scope".into(),
            ));
        }
        material
            .binding()
            .validate_computation(&parent.for_level_operation(successor, parent.operation_id()))?;
        if material.commitment_id() != Some(parent.operation_id()) {
            return Err(AkitaError::InvalidInput(
                "export witness and material are mispaired".into(),
            ));
        }
        let (schedule, _) = scope.proof_plan()?;
        if schedule.as_ref() != plan.schedule()
            || &self.prepared()?.expanded.descriptor != plan.setup()
        {
            return Err(AkitaError::InvalidInput(
                "export belongs to another admitted proof".into(),
            ));
        }
        match plan.binding() {
            SuccessorPublicBinding::Outer(rows) => material.validate_public_commitment(rows)?,
            SuccessorPublicBinding::Terminal(fields) => {
                if self.terminal_message(material)?.fields() != *fields {
                    return Err(AkitaError::InvalidInput(
                        "terminal export binding mismatch".into(),
                    ));
                }
            }
        }
        let (inner, compression) = material.transfer_parts();
        let storage = CpuSuccessorStorage {
            logical: witness.logical.clone(),
            committed: witness.committed.clone(),
            inner,
            compression,
            source: parent,
        };
        let digits = storage.logical.packed_digits();
        let mut sections = vec![SuccessorSectionDescriptor {
            section: SuccessorSection::LogicalDigits,
            encoding: SuccessorEncoding::PackedSigned {
                bit_width: digits.bit_width(),
            },
            coefficients: digits.len(),
            bytes: digits.encoded_bytes().len(),
        }];
        sections.push(SuccessorSectionDescriptor {
            section: SuccessorSection::InnerRows,
            encoding: SuccessorEncoding::CanonicalField,
            coefficients: plan.inner_coefficients()?,
            bytes: checked::product([plan.inner_coefficients()?, F::NUM_BYTES])
                .ok_or_else(|| AkitaError::InvalidInput("inner transfer size overflow".into()))?,
        });
        if let Some(compression) = &storage.compression {
            for (index, stage) in compression.witness().stages().iter().enumerate() {
                sections.push(SuccessorSectionDescriptor {
                    section: SuccessorSection::CompressionDigits(index),
                    encoding: SuccessorEncoding::NegativeBinary,
                    coefficients: stage.map().real_digit_count(),
                    bytes: stage.bytes().len(),
                });
            }
            if let Some(quotients) = compression.quotients() {
                for (index, row) in quotients.iter().enumerate() {
                    sections.push(SuccessorSectionDescriptor {
                        section: SuccessorSection::CompressionQuotient(index),
                        encoding: SuccessorEncoding::CanonicalField,
                        coefficients: row.coeff_len(),
                        bytes: checked::product([row.coeff_len(), F::NUM_BYTES]).ok_or_else(
                            || AkitaError::InvalidInput("quotient transfer size overflow".into()),
                        )?,
                    });
                }
            }
        }
        let descriptor = CpuPacketDescriptor {
            metadata: HandoffMetadata {
                handoff: plan.handoff_id(),
                manifest: witness.manifest(),
            },
            sections,
        };
        Ok(CpuExportPacket {
            descriptor,
            storage,
        })
    }
}

impl<F, E> SuccessorImportKernel<F, E> for CpuBackend<F, E>
where
    F: Field + CanonicalEncoding + AkitaSerialize + Send + Sync + 'static,
    E: ExtField<F> + Send + Sync + 'static,
{
    type ImportPacket = CpuImportPacket<F>;
    fn import_successor(
        &self,
        session: &Self::ProofSessionHandle,
        plan: &ValidatedSuccessorHandoffPlan<'_, F>,
        mut packet: Self::ImportPacket,
    ) -> Result<ImportedSuccessor<Self::WitnessHandle, Self::CommitmentMaterialHandle>, AkitaError>
    {
        let scope = session.validate_owner(self.owner())?;
        scope.require_prepared(plan.successor_level())?;
        let (schedule, _) = scope.proof_plan()?;
        if schedule.as_ref() != plan.schedule()
            || &self.prepared()?.expanded.descriptor != plan.setup()
        {
            return Err(AkitaError::InvalidInput(
                "import belongs to another admitted proof".into(),
            ));
        }
        packet.descriptor.validate(plan)?;
        let (logical, mut committed, inner, compression) =
            if let Some(native) = packet.native.take() {
                native.source.scope_lease().validate(
                    native.source.scope_id(),
                    native.source.fold_level(),
                    None,
                )?;
                if native.logical.live_coeff_len() != plan.commitment().logical_len()
                    || !native
                        .logical
                        .packed_digits()
                        .bounds()
                        .fits_balanced_log_basis(plan.log_basis())
                {
                    return Err(AkitaError::InvalidInput(
                        "native successor digit geometry mismatch".into(),
                    ));
                }
                (
                    native.logical,
                    native.committed,
                    native.inner,
                    native.compression,
                )
            } else {
                let (logical, inner, compression) = read_portable::<F>(plan, &mut packet)?;
                (logical, None, inner, compression)
            };
        let tensor = super::recursive::uses_tensor_source(plan.commitment(), E::DEGREE)?;
        if tensor && committed.is_none() {
            committed = Some(akita_params::dispatch_for_field!(
                akita_params::ProtocolDispatchSlot::Role(akita_params::RingRole::Inner),
                F,
                plan.commitment().ring_dimension(),
                |D| crate::opaque::tensor_pack_recursive_witness::<F, E, D>(&logical)
            )?);
        }
        if !tensor && committed.is_some() {
            return Err(AkitaError::InvalidInput(
                "canonical successor has an unexpected transformed source".into(),
            ));
        }
        let context = self.proof_context(
            session,
            u32::try_from(plan.successor_level())
                .map_err(|_| AkitaError::InvalidInput("imported level exceeds u32".into()))?,
        )?;
        let binding = self.binding(session, &context)?;
        let mut material =
            CpuCommitmentMaterialHandle::from_material(inner, compression, &execution(plan)?, 1)?;
        material.bind(binding.clone());
        material.bind_commitment(
            binding.operation_id(),
            match plan.binding() {
                SuccessorPublicBinding::Outer(rows) => Some((*rows).clone()),
                SuccessorPublicBinding::Terminal(_) => None,
            },
        );
        if let SuccessorPublicBinding::Terminal(fields) = plan.binding() {
            if self.terminal_message(&material)?.fields() != *fields {
                return Err(AkitaError::InvalidInput(
                    "adopted terminal rows differ from public binding".into(),
                ));
            }
        }
        let witness = CpuWitnessHandle::from_imported_successor(logical, committed, plan, binding)?;
        Ok(ImportedSuccessor::new(witness, material))
    }
}

fn read_digits<F: Field>(
    descriptor: &SuccessorSectionDescriptor,
    packet: &mut CpuImportPacket<F>,
) -> Result<PackedSignedDigits, AkitaError> {
    let bit_width = match descriptor.encoding {
        SuccessorEncoding::PackedSigned { bit_width } => bit_width,
        SuccessorEncoding::SignedI8 => 8,
        _ => {
            return Err(AkitaError::InvalidInput(
                "invalid portable digit encoding".into(),
            ))
        }
    };
    let bytes = packet.take_section(descriptor.section)?;
    PackedSignedDigits::import_encoded(descriptor.coefficients, bit_width, bytes)
}

fn read_fields<F: Field + CanonicalEncoding>(
    packet: &mut CpuImportPacket<F>,
    section: SuccessorSection,
    coefficients: usize,
    ring: usize,
) -> Result<RingVec<F>, AkitaError> {
    let mut fields = Vec::new();
    fields
        .try_reserve_exact(coefficients)
        .map_err(|_| AkitaError::InvalidInput("successor field allocation failed".into()))?;
    let bytes = packet.take_section(section)?;
    for coefficient in bytes.chunks_exact(F::NUM_BYTES) {
        fields.push(F::from_bytes_le_checked(coefficient).ok_or_else(|| {
            AkitaError::InvalidInput("noncanonical successor field coefficient".into())
        })?);
    }
    RingVec::from_coeffs_with_ring_dim(fields, ring)
}

type PortableParts<F> = (
    RecursiveWitnessFlat,
    InnerRelationStateMaterial<F>,
    Option<PortableCompressionState<F>>,
);

fn read_portable<F: Field + CanonicalEncoding>(
    plan: &ValidatedSuccessorHandoffPlan<'_, F>,
    packet: &mut CpuImportPacket<F>,
) -> Result<PortableParts<F>, AkitaError> {
    let logical_section = packet
        .descriptor
        .sections
        .iter()
        .find(|s| s.section == SuccessorSection::LogicalDigits)
        .copied()
        .ok_or_else(|| AkitaError::InvalidInput("logical section is missing".into()))?;
    let digits = read_digits(&logical_section, packet)?;
    let producer = plan.producer_parameters();
    for unit in plan.witness_layout().units() {
        let group = producer
            .groups()
            .get(unit.group_index())
            .ok_or_else(|| AkitaError::InvalidInput("producer witness group is missing".into()))?;
        for (range, log_basis) in [
            (unit.z_range(), group.log_basis_open()),
            (unit.e_range(), group.log_basis_open()),
            (unit.t_range(), group.log_basis_outer()),
        ] {
            if !digits
                .view()
                .slice(range)?
                .scan_bounds()
                .fits_balanced_log_basis(log_basis)
            {
                return Err(AkitaError::InvalidInput(
                    "successor digit exceeds its producer segment basis".into(),
                ));
            }
        }
    }
    for range in plan.witness_layout().negative_binary_support_intervals() {
        if !digits
            .view()
            .slice(range)?
            .scan_bounds()
            .fits_balanced_log_basis(1)
        {
            return Err(AkitaError::InvalidInput(
                "compression witness digit is not negative binary".into(),
            ));
        }
    }
    let logical =
        RecursiveWitnessFlat::from_witness_layout(digits, plan.witness_layout(), plan.log_basis())?
            .align_for_commitment_ring_dim(plan.commitment().ring_dimension())?;
    let execution = execution(plan)?;
    let inner = InnerRelationStateMaterial::new(
        execution.inner(),
        1,
        vec![read_fields(
            packet,
            SuccessorSection::InnerRows,
            plan.inner_coefficients()?,
            plan.commitment().ring_dimension(),
        )?],
    )?;
    let compression = match plan.compression()? {
        None => None,
        Some((chain, mode)) => {
            let mut stages = Vec::with_capacity(chain.maps().len());
            let mut quotients = Vec::new();
            for (index, map) in chain.maps().iter().enumerate() {
                let bytes = packet.take_section(SuccessorSection::CompressionDigits(index))?;
                stages.push(PackedNegativeBinary::from_bytes(*map, bytes)?);
                if mode == RingRelationMode::QuotientLift {
                    quotients.push(read_fields(
                        packet,
                        SuccessorSection::CompressionQuotient(index),
                        map.output_coefficients(),
                        map.ring_dimension(),
                    )?);
                }
            }
            let witness = CompressionChainWitness::new(chain, stages)?;
            Some(match mode {
                RingRelationMode::QuotientLift => {
                    PortableCompressionState::quotient_lift(witness, quotients)?
                }
                RingRelationMode::ReducedEvaluation => {
                    PortableCompressionState::reduced_evaluation(witness)?
                }
            })
        }
    };
    Ok((logical, inner, compression))
}

#[cfg(test)]
#[path = "transfer/tests.rs"]
mod tests;
