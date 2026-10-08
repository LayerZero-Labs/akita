//! CPU packets, directed bridges, and atomic adoption into local handles.
use super::*;
use crate::commitment::{
    CommitmentExecutionPlan, InnerRelationStateMaterial, PortableCompressionState,
};
use crate::sources::packed_digits::{PackedSignedDigitWriter, PackedSignedDigits};
use akita_error::checked;
use akita_params::{CompressionChainWitness, PackedNegativeBinary, RingRelationMode};
use akita_serialization::AkitaSerialize;
use akita_types::RingVec;
use jolt_field::ExtField;

struct CpuSuccessorStorage<F: Field> {
    logical: RecursiveWitnessFlat,
    committed: Option<RecursiveWitnessFlat>,
    inner: InnerRelationStateMaterial<F>,
    compression: Option<PortableCompressionState<F>>,
    source: OperationBinding,
}

impl<F: Field + CanonicalEncoding> CpuSuccessorStorage<F> {
    fn read(
        &self,
        section: SuccessorSection,
        offset: usize,
        output: &mut [u8],
    ) -> Result<(), AkitaError> {
        self.source.scope_lease().validate(
            self.source.scope_id(),
            self.source.fold_level(),
            None,
        )?;
        let end = checked::sum([offset, output.len()])
            .ok_or_else(|| AkitaError::InvalidInput("successor section range overflow".into()))?;
        let bytes = match section {
            SuccessorSection::LogicalDigits => Some(self.logical.packed_digits().encoded_bytes()),
            SuccessorSection::CompressionDigits(index) => Some(
                self.compression
                    .as_ref()
                    .and_then(|c| c.witness().stages().get(index))
                    .ok_or_else(|| {
                        AkitaError::InvalidInput("compression digit section is absent".into())
                    })?
                    .bytes(),
            ),
            _ => None,
        };
        if let Some(bytes) = bytes {
            output.copy_from_slice(bytes.get(offset..end).ok_or_else(|| {
                AkitaError::InvalidInput("successor read is outside its section".into())
            })?);
            return Ok(());
        }
        if !offset.is_multiple_of(F::NUM_BYTES) || !output.len().is_multiple_of(F::NUM_BYTES) {
            return Err(AkitaError::InvalidInput(
                "field section read is not coefficient aligned".into(),
            ));
        }
        let fields = match section {
            SuccessorSection::InnerRows => self
                .inner
                .rows()
                .first()
                .ok_or_else(|| AkitaError::InvalidInput("inner rows are absent".into()))?
                .coeffs(),
            SuccessorSection::CompressionQuotient(index) => self
                .compression
                .as_ref()
                .and_then(|c| c.quotients())
                .and_then(|q| q.get(index))
                .ok_or_else(|| AkitaError::InvalidInput("compression quotient is absent".into()))?
                .coeffs(),
            _ => return Err(AkitaError::InvalidInput("unknown field section".into())),
        };
        let fields = fields
            .get(offset / F::NUM_BYTES..end / F::NUM_BYTES)
            .ok_or_else(|| AkitaError::InvalidInput("field read is outside its section".into()))?;
        for (field, bytes) in fields.iter().zip(output.chunks_exact_mut(F::NUM_BYTES)) {
            field.to_bytes_le(bytes);
        }
        Ok(())
    }
}

/// Encoded sections accepted by the public CPU import constructor.
pub type CpuPacketSections = Vec<(SuccessorSection, Vec<u8>)>;

/// An owned snapshot of a committed successor. Its native storage remains private.
/// Portable sections carry logical digits; CPU import derives tensor packing locally.
pub struct CpuExportPacket<F: Field> {
    descriptor: SuccessorExportDescriptor,
    storage: CpuSuccessorStorage<F>,
}

/// CPU import representation. External bridges construct it from encoded sections.
/// CPU-to-CPU edges retain shared storage instead of encoding and copying it.
pub struct CpuImportPacket<F: Field> {
    descriptor: SuccessorExportDescriptor,
    sections: CpuPacketSections,
    native: Option<CpuSuccessorStorage<F>>,
}

impl<F: Field> CpuImportPacket<F> {
    /// Check section presence, uniqueness, and byte lengths. Adoption additionally
    /// checks the admitted plan, canonical encodings, and witness digit bounds.
    pub fn new(
        descriptor: SuccessorExportDescriptor,
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

    fn read(
        &self,
        section: SuccessorSection,
        offset: usize,
        output: &mut [u8],
    ) -> Result<(), AkitaError> {
        let end = checked::sum([offset, output.len()])
            .ok_or_else(|| AkitaError::InvalidInput("CPU packet range overflow".into()))?;
        let bytes = self
            .sections
            .iter()
            .find(|(s, _)| *s == section)
            .and_then(|(_, bytes)| bytes.get(offset..end))
            .ok_or_else(|| AkitaError::InvalidInput("CPU packet section range is absent".into()))?;
        output.copy_from_slice(bytes);
        Ok(())
    }
}
impl<F: Field + Send + Sync + 'static> SuccessorImportPacket for CpuImportPacket<F> {
    fn descriptor(&self) -> &SuccessorExportDescriptor {
        &self.descriptor
    }
}

impl<F: Field + CanonicalEncoding> CpuExportPacket<F> {
    /// Encode sections for a bridge targeting another backend representation.
    pub fn into_sections(
        self,
    ) -> Result<(SuccessorExportDescriptor, CpuPacketSections), AkitaError> {
        let sections = encode_sections(&self.descriptor, &self.storage)?;
        Ok((self.descriptor, sections))
    }
}
impl<F: Field + CanonicalEncoding> CpuImportPacket<F> {
    /// Materialize a native packet into the same public representation accepted by `new`.
    pub fn into_sections(
        self,
    ) -> Result<(SuccessorExportDescriptor, CpuPacketSections), AkitaError> {
        let sections = match self.native {
            Some(storage) => encode_sections(&self.descriptor, &storage)?,
            None => self.sections,
        };
        Ok((self.descriptor, sections))
    }
}
fn encode_sections<F: Field + CanonicalEncoding>(
    descriptor: &SuccessorExportDescriptor,
    storage: &CpuSuccessorStorage<F>,
) -> Result<CpuPacketSections, AkitaError> {
    descriptor
        .sections
        .iter()
        .map(|section| {
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(section.bytes)
                .map_err(|_| AkitaError::InvalidInput("CPU packet allocation failed".into()))?;
            bytes.resize(section.bytes, 0);
            storage.read(section.section, 0, &mut bytes)?;
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
        plan: &ValidatedSuccessorHandoffPlan<'_, F>,
    ) -> Result<CpuImportPacket<F>, AkitaError> {
        plan.validate_export(&packet.descriptor)?;
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
    fn instance_identity(&self) -> BackendInstanceIdentity {
        BackendInstanceIdentity::new::<Self>(u128::from(self.owner_id()))
    }

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
        let descriptor = SuccessorExportDescriptor {
            handoff: plan.handoff_id(),
            manifest: witness.manifest(),
            sections,
        };
        plan.validate_export(&descriptor)?;
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
        plan.validate_export(&packet.descriptor)?;
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
                let (logical, inner, compression) =
                    read_portable::<F>(plan, &packet.descriptor, &packet)?;
                (logical, None, inner, compression)
            };
        let tensor = matches!(
            plan.commitment().source_encoding(),
            Some(akita_params::CommittedSourceEncoding::TensorSubfieldProjection { .. })
        ) || (matches!(
            plan.commitment().parameters(),
            WitnessCommitmentParameters::Terminal(_)
        ) && E::DEGREE != 1);
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
    packet: &CpuImportPacket<F>,
) -> Result<PackedSignedDigits, AkitaError> {
    match descriptor.encoding {
        SuccessorEncoding::PackedSigned { bit_width } => {
            PackedSignedDigits::import_encoded(descriptor.coefficients, bit_width, |bytes| {
                packet.read(descriptor.section, 0, bytes)
            })
        }
        SuccessorEncoding::SignedI8 => {
            let mut writer = PackedSignedDigitWriter::new(descriptor.coefficients, 8)?;
            let mut bytes = [0u8; 4096];
            let mut digits = [0i8; 4096];
            for start in (0..descriptor.coefficients).step_by(bytes.len()) {
                let count = (descriptor.coefficients - start).min(bytes.len());
                packet.read(descriptor.section, start, &mut bytes[..count])?;
                for (digit, byte) in digits[..count].iter_mut().zip(&bytes[..count]) {
                    *digit = *byte as i8;
                }
                writer.write_at(start, &digits[..count])?;
            }
            writer.finish()
        }
        _ => Err(AkitaError::InvalidInput(
            "invalid portable digit encoding".into(),
        )),
    }
}

fn read_fields<F: Field + CanonicalEncoding>(
    packet: &CpuImportPacket<F>,
    section: SuccessorSection,
    coefficients: usize,
    ring: usize,
) -> Result<RingVec<F>, AkitaError> {
    let mut fields = Vec::new();
    fields
        .try_reserve_exact(coefficients)
        .map_err(|_| AkitaError::InvalidInput("successor field allocation failed".into()))?;
    let mut buffer = vec![
        0u8;
        checked::product([4096, F::NUM_BYTES]).ok_or_else(|| {
            AkitaError::InvalidInput("field staging size overflow".into())
        })?
    ];
    for start in (0..coefficients).step_by(4096) {
        let count = (coefficients - start).min(4096);
        let bytes = checked::product([count, F::NUM_BYTES])
            .ok_or_else(|| AkitaError::InvalidInput("field read size overflow".into()))?;
        let offset = checked::product([start, F::NUM_BYTES])
            .ok_or_else(|| AkitaError::InvalidInput("field read offset overflow".into()))?;
        packet.read(section, offset, &mut buffer[..bytes])?;
        for bytes in buffer[..bytes].chunks_exact(F::NUM_BYTES) {
            fields.push(F::from_bytes_le_checked(bytes).ok_or_else(|| {
                AkitaError::InvalidInput("noncanonical successor field coefficient".into())
            })?);
        }
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
    descriptor: &SuccessorExportDescriptor,
    packet: &CpuImportPacket<F>,
) -> Result<PortableParts<F>, AkitaError> {
    let logical_section = descriptor
        .sections
        .iter()
        .find(|s| s.section == SuccessorSection::LogicalDigits)
        .ok_or_else(|| AkitaError::InvalidInput("logical section is missing".into()))?;
    let digits = read_digits(logical_section, packet)?;
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
            let half = akita_params::balanced_signed_digit_abs_bound(log_basis)
                .ok_or_else(|| AkitaError::InvalidInput("invalid producer digit basis".into()))?;
            if digits.view().slice(range)?.iter().any(|digit| {
                i128::from(digit) < -i128::from(half) || i128::from(digit) >= i128::from(half)
            }) {
                return Err(AkitaError::InvalidInput(
                    "successor digit exceeds its producer segment basis".into(),
                ));
            }
        }
    }
    let binary = plan.witness_layout().negative_binary_support_intervals();
    let mut span = 0;
    for (offset, digit) in digits.iter().enumerate() {
        while binary.get(span).is_some_and(|range| offset >= range.end) {
            span += 1;
        }
        if binary
            .get(span)
            .is_some_and(|range| range.contains(&offset))
            && digit != 0
            && digit != -1
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
                let mut bytes = vec![0; map.packed_digit_bytes()];
                packet.read(SuccessorSection::CompressionDigits(index), 0, &mut bytes)?;
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
