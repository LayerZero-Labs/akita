// A distinct handle family; CPU storage is private to this test backend.
use super::{Cfg, E, F, NV};
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_cpu_backend::{CpuBackend, CpuImportPacket, DensePoly, GroupContext, OneHotPoly};
use akita_error::AkitaError;
use akita_params::{FoldSchedule, GroupCommitPhaseParams, OpeningClaimsLayout};
use akita_prover::backend::*;
use akita_prover::{PreparedSetupPrefix, SetupPrefixProverRegistry};
use akita_types::{AkitaSetupDescriptor, Commitment};
use jolt_field::One;
use jolt_poly::UnivariatePoly;
use std::cell::{Cell, RefCell};

type Cpu = CpuBackend<F, E>;
#[derive(Clone)]
pub(crate) struct OwnedCommitment(<Cpu as ProverHandleFamily<F, E>>::CommitmentHandle);
pub(crate) struct OwnedWitness(<Cpu as ProverHandleFamily<F, E>>::WitnessHandle);
pub(crate) struct OwnedMaterial(<Cpu as ProverHandleFamily<F, E>>::CommitmentMaterialHandle);
impl CommitmentHandleMetadata for OwnedCommitment {
    fn metadata(&self) -> SourceMetadata {
        self.0.metadata()
    }
    fn producer_contract(&self) -> akita_params::sis::CommittedSourceContract {
        self.0.producer_contract()
    }
}
impl RecursiveWitnessHandle for OwnedWitness {
    fn manifest(&self) -> RecursiveWitnessManifest {
        self.0.manifest()
    }
}
impl CommitmentRelationMaterial<F> for OwnedMaterial {
    fn metadata(&self) -> CommitmentMaterialMetadata {
        self.0.metadata()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fault {
    None,
    Preparation,
    Export,
    Import,
    SourceStage1,
    DestinationOpening,
    Descriptor,
    Packet,
    InvalidDigits,
    InvalidFields,
    SignedDigits,
}

pub(crate) struct PrivateCpu {
    inner: Cpu,
    pub(crate) portable: Cell<bool>,
    pub(crate) fault: Cell<Fault>,
    pub(crate) supported: RefCell<Vec<usize>>,
    pub(crate) events: RefCell<Vec<&'static str>>,
    pub(crate) admitted: Cell<usize>,
    pub(crate) finished: Cell<usize>,
    pub(crate) aborted: Cell<usize>,
    pub(crate) commitments: Cell<usize>,
    pub(crate) releases: Cell<usize>,
}
impl PrivateCpu {
    pub(crate) fn new(inner: Cpu) -> Self {
        Self {
            inner,
            portable: Cell::new(false),
            fault: Cell::new(Fault::None),
            supported: RefCell::default(),
            events: RefCell::default(),
            admitted: Cell::new(0),
            finished: Cell::new(0),
            aborted: Cell::new(0),
            commitments: Cell::new(0),
            releases: Cell::new(0),
        }
    }
    pub(crate) fn prefixes(
        &self,
        setup: &akita_cpu_backend::AkitaProverSetup<F>,
        ids: &[akita_params::SetupPrefixSlotId],
    ) -> SetupPrefixProverRegistry<F, OwnedCommitment> {
        let imported = self
            .inner
            .import_setup_prefixes(&setup.prefix_slots, ids)
            .unwrap();
        let mut prefixes = SetupPrefixProverRegistry::default();
        for id in ids {
            let slot = imported.get(id).unwrap();
            prefixes
                .insert(PreparedSetupPrefix {
                    public: slot.public.clone(),
                    commitment_handle: OwnedCommitment(slot.commitment_handle.clone()),
                })
                .unwrap();
        }
        prefixes
    }
    // Shared by tests (`commit_root`) and the dynamic_onehot example (`commit_onehot`).
    #[allow(dead_code)]
    pub(crate) fn commit_root(
        &self,
        schedules: &TrustedScheduleCatalog<Cfg>,
    ) -> (akita_types::CommittedGroup<F>, OwnedCommitment) {
        let source = if let Ok(chunk) = akita_config::unit_onehot_source_chunk_size::<Cfg>() {
            self.inner.import_source(vec![OneHotPoly::<F, u8>::new(
                chunk,
                vec![Some(0); (1 << NV) / chunk],
            )
            .unwrap()])
        } else {
            self.inner.import_source(vec![DensePoly::from_field_evals(
                NV,
                vec![F::one(); 1 << NV],
            )
            .unwrap()])
        }
        .unwrap();
        let output = self
            .inner
            .commit(
                schedules,
                &source,
                GroupContext::scheduler_without_precommitted_groups(),
            )
            .unwrap();
        (
            output.committed_group,
            OwnedCommitment(output.private_handle),
        )
    }
    #[allow(dead_code)]
    pub(crate) fn commit_onehot(
        &self,
        schedules: &TrustedScheduleCatalog<Cfg>,
        polys: Vec<OneHotPoly<F, u8>>,
    ) -> Result<(akita_types::CommittedGroup<F>, OwnedCommitment), AkitaError> {
        let source = self.inner.import_source(polys)?;
        let output = self.inner.commit(
            schedules,
            &source,
            GroupContext::scheduler_without_precommitted_groups(),
        )?;
        Ok((
            output.committed_group,
            OwnedCommitment(output.private_handle),
        ))
    }
}
impl ProofScopeConsumer for PrivateCpu {
    type ProofSessionHandle = <Cpu as ProofScopeConsumer>::ProofSessionHandle;
    fn finish_scope(&self, session: &Self::ProofSessionHandle) -> Result<(), AkitaError> {
        self.inner.finish_scope(session)?;
        self.finished.set(self.finished.get() + 1);
        Ok(())
    }
    fn abort_scope_best_effort(&self, session: &Self::ProofSessionHandle) {
        self.inner.abort_scope_best_effort(session);
        self.aborted.set(self.aborted.get() + 1);
    }
}
impl ProverHandleFamily<F, E> for PrivateCpu {
    type CommitmentHandle = OwnedCommitment;
    type EorPreparationHandle = <Cpu as ProverHandleFamily<F, E>>::EorPreparationHandle;
    type EorSessionHandle = <Cpu as ProverHandleFamily<F, E>>::EorSessionHandle;
    type PreparedOpeningHandle = <Cpu as ProverHandleFamily<F, E>>::PreparedOpeningHandle;
    type CommitmentMaterialHandle = OwnedMaterial;
    type AcceptedFoldHandle = <Cpu as ProverHandleFamily<F, E>>::AcceptedFoldHandle;
    type AcceptedTerminalFoldHandle = <Cpu as ProverHandleFamily<F, E>>::AcceptedTerminalFoldHandle;
    type WitnessBuildHandle = <Cpu as ProverHandleFamily<F, E>>::WitnessBuildHandle;
    type WitnessHandle = OwnedWitness;
    type RelationHandle = <Cpu as ProverHandleFamily<F, E>>::RelationHandle;
    type Stage1SessionHandle = <Cpu as ProverHandleFamily<F, E>>::Stage1SessionHandle;
    type Stage2SessionHandle = <Cpu as ProverHandleFamily<F, E>>::Stage2SessionHandle;
}
impl OpaqueProverConsumer<F, E> for PrivateCpu {}
impl ProofAdmission<F, E> for PrivateCpu {
    fn prepare_executor<C: CommitmentConfig<Field = F, ExtField = E>>(
        &self,
        setup: &AkitaSetupDescriptor,
        schedules: &TrustedScheduleCatalog<C>,
        plan: &FoldSchedule,
        layout: &OpeningClaimsLayout,
    ) -> Result<Self::ProofSessionHandle, AkitaError> {
        self.events.borrow_mut().push("prepare_executor");
        if self.fault.get() == Fault::Preparation {
            return Err(AkitaError::InvalidInput(
                "injected executor preparation failure".into(),
            ));
        }
        let session = self
            .inner
            .prepare_executor(setup, schedules, plan, layout)?;
        self.admitted.set(self.admitted.get() + 1);
        Ok(session)
    }
    fn begin_fold(
        &self,
        session: &Self::ProofSessionHandle,
        requirements: &FoldExecutionRequirements<'_>,
    ) -> Result<(), AkitaError> {
        self.events.borrow_mut().push("begin_fold");
        if !self.supported.borrow().is_empty()
            && !self.supported.borrow().contains(&requirements.level())
        {
            return Err(AkitaError::UnsupportedSchedule(
                "private backend cannot execute this level".into(),
            ));
        }
        self.inner.begin_fold(session, requirements)
    }
    fn proof_context(
        &self,
        session: &Self::ProofSessionHandle,
        level: u32,
    ) -> Result<ProofContext, AkitaError> {
        self.inner.proof_context(session, level)
    }
    fn validate_commitment(
        &self,
        session: &Self::ProofSessionHandle,
        context: &ProofContext,
        handle: &Self::CommitmentHandle,
        parameters: &GroupCommitPhaseParams,
        commitment: &Commitment<F>,
    ) -> Result<Self::CommitmentMaterialHandle, AkitaError> {
        self.inner
            .validate_commitment(session, context, &handle.0, parameters, commitment)
            .map(OwnedMaterial)
    }
}
impl TerminalCommitmentMaterialKernel<F, OwnedMaterial> for PrivateCpu {
    fn terminal_message(
        &self,
        material: &OwnedMaterial,
    ) -> Result<TerminalTFieldsMessage<F>, AkitaError> {
        self.inner.terminal_message(&material.0)
    }
    fn consume_terminal_row(
        &self,
        material: OwnedMaterial,
    ) -> Result<akita_types::RingVec<F>, AkitaError> {
        self.inner.consume_terminal_row(material.0)
    }
}
impl OpaqueTerminalFoldKernel<F, E> for PrivateCpu {
    fn probe_terminal_fold(
        &self,
        witness_handle: &Self::WitnessHandle,
        plan: &ValidatedTerminalFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<Self::AcceptedTerminalFoldHandle>, AkitaError> {
        self.inner.probe_terminal_fold(&witness_handle.0, plan)
    }
    fn encode_terminal_fold(
        &self,
        terminal_fold_handle: Self::AcceptedTerminalFoldHandle,
        plan: &ValidatedTerminalZEncodingPlan,
    ) -> Result<Vec<u8>, AkitaError> {
        self.inner.encode_terminal_fold(terminal_fold_handle, plan)
    }
}
impl OpaqueRecursiveWitnessBuildKernel<F, E> for PrivateCpu {
    fn begin_recursive_witness(
        &self,
        session: &Self::ProofSessionHandle,
        context: &ProofContext,
        prepared_opening_handles: &[Self::PreparedOpeningHandle],
        commitment_material_handles: Vec<Self::CommitmentMaterialHandle>,
        level: &akita_params::CommittedGroupParams,
        opening_batch: &akita_params::OpeningClaimsLayout,
        relation_rhs_layout: &akita_params::RelationRhsLayout,
        group_commitments: &[akita_types::RingVec<F>],
    ) -> Result<RecursiveWitnessBuildStart<F, E, Self::WitnessBuildHandle>, AkitaError> {
        self.inner.begin_recursive_witness(
            session,
            context,
            prepared_opening_handles,
            commitment_material_handles
                .into_iter()
                .map(|material| material.0)
                .collect(),
            level,
            opening_batch,
            relation_rhs_layout,
            group_commitments,
        )
    }
    fn finish_recursive_witness(
        &self,
        build_handle: Self::WitnessBuildHandle,
        fold_inputs: Vec<RecursiveWitnessFoldInput<Self::AcceptedFoldHandle>>,
        relation: &akita_types::RingRelationInstance<F>,
        plan: &ValidatedRecursiveWitnessPlan<'_, F>,
    ) -> Result<Self::WitnessHandle, AkitaError> {
        self.inner
            .finish_recursive_witness(build_handle, fold_inputs, relation, plan)
            .map(OwnedWitness)
    }
}
impl OpaqueWitnessCommitKernel<F, E> for PrivateCpu {
    fn commit_witness(
        &self,
        witness_handle: Self::WitnessHandle,
        plan: &ValidatedRecursiveWitnessCommitPlan,
    ) -> Result<
        WitnessCommitmentOutput<F, Self::WitnessHandle, Self::CommitmentMaterialHandle>,
        AkitaError,
    > {
        self.commitments.set(self.commitments.get() + 1);
        self.inner
            .commit_witness(witness_handle.0, plan)
            .map(|output| {
                let (binding, witness, material) = output.into_commitment_parts();
                WitnessCommitmentOutput::new(
                    binding,
                    OwnedWitness(witness),
                    OwnedMaterial(material),
                )
            })
    }
    fn advance_witness_level(
        &self,
        witness_handle: &mut Self::WitnessHandle,
    ) -> Result<(), AkitaError> {
        self.inner.advance_witness_level(&mut witness_handle.0)
    }
}
impl OpaqueRelationWitnessKernel<F, E> for PrivateCpu {
    fn prepare_relation_witness(
        &self,
        witness_handle: &Self::WitnessHandle,
        plan: &ValidatedRelationWitnessPlan,
    ) -> Result<PreparedRelationHandle<Self::RelationHandle>, AkitaError> {
        self.inner.prepare_relation_witness(&witness_handle.0, plan)
    }
}
impl OpaqueStage1Kernel<F, E> for PrivateCpu {
    fn begin_stage1(
        &self,
        relation_handle: &Self::RelationHandle,
        plan: &ValidatedStage1Plan<E>,
    ) -> Result<Self::Stage1SessionHandle, AkitaError> {
        self.events.borrow_mut().push("stage1");
        if self.fault.get() == Fault::SourceStage1 {
            return Err(AkitaError::InvalidInput(
                "injected source failure after adoption".into(),
            ));
        }
        self.inner.begin_stage1(relation_handle, plan)
    }
    fn stage1_round_polynomial(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        step: Stage1Step,
        round: usize,
        previous_claim: E,
    ) -> Result<Stage1RoundPolynomial<E>, AkitaError> {
        self.inner
            .stage1_round_polynomial(session_handle, step, round, previous_claim)
    }
    fn bind_stage1_challenge(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        step: Stage1Step,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError> {
        self.inner
            .bind_stage1_challenge(session_handle, step, round, challenge)
    }
    fn stage1_public_transition(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        step: Stage1Step,
    ) -> Result<Stage1PublicTransition<E>, AkitaError> {
        self.inner.stage1_public_transition(session_handle, step)
    }
    fn bind_stage1_batch_challenge(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        transition: Stage1Transition,
        challenge: E,
    ) -> Result<(), AkitaError> {
        self.inner
            .bind_stage1_batch_challenge(session_handle, transition, challenge)
    }
    fn finish_stage1(
        &self,
        session_handle: Self::Stage1SessionHandle,
    ) -> Result<Stage1FinalClaims<E>, AkitaError> {
        self.inner.finish_stage1(session_handle)
    }
}
impl OpaqueStage2Kernel<F, E> for PrivateCpu {
    fn begin_stage2(
        &self,
        relation_handle: Self::RelationHandle,
        plan: ValidatedRelationSessionPlan<'_, F, E>,
    ) -> Result<Self::Stage2SessionHandle, AkitaError> {
        self.inner.begin_stage2(relation_handle, plan)
    }
    fn stage2_input_claim(
        &self,
        session_handle: &Self::Stage2SessionHandle,
    ) -> Result<E, AkitaError> {
        self.inner.stage2_input_claim(session_handle)
    }
    fn stage2_num_rounds(
        &self,
        session_handle: &Self::Stage2SessionHandle,
    ) -> Result<usize, AkitaError> {
        self.inner.stage2_num_rounds(session_handle)
    }
    fn stage2_round_polynomial(
        &self,
        session_handle: &mut Self::Stage2SessionHandle,
        round: usize,
        previous_claim: E,
    ) -> Result<UnivariatePoly<E>, AkitaError> {
        self.inner
            .stage2_round_polynomial(session_handle, round, previous_claim)
    }
    fn bind_stage2_challenge(
        &self,
        session_handle: &mut Self::Stage2SessionHandle,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError> {
        self.inner
            .bind_stage2_challenge(session_handle, round, challenge)
    }
    fn finish_stage2(
        &self,
        session_handle: Self::Stage2SessionHandle,
    ) -> Result<RelationWitnessFinalClaims<E>, AkitaError> {
        self.inner.finish_stage2(session_handle)
    }
}
impl OpaqueWitnessOpeningKernel<F, E> for PrivateCpu {
    fn prepare_native_witness_opening(
        &self,
        witness_handle: &Self::WitnessHandle,
        plan: &ValidatedRecursiveGroupOpeningPlan<'_, E>,
    ) -> Result<PreparedGroupOpening<E, Self::PreparedOpeningHandle>, AkitaError> {
        self.inner
            .prepare_native_witness_opening(&witness_handle.0, plan)
    }
    fn terminal_native_witness_opening(
        &self,
        opening_handle: Self::PreparedOpeningHandle,
    ) -> Result<Vec<akita_types::RingVec<F>>, AkitaError> {
        self.inner.terminal_native_witness_opening(opening_handle)
    }
}
impl OpaqueResourceReleaseKernel<F, E> for PrivateCpu {
    fn release_witness_handle(
        &self,
        witness_handle: Self::WitnessHandle,
    ) -> Result<(), AkitaError> {
        self.releases.set(self.releases.get() + 1);
        self.inner.release_witness_handle(witness_handle.0)
    }
}
impl OpaqueOpeningKernel<F, E> for PrivateCpu {
    fn prepare_openings(
        &self,
        session: &Self::ProofSessionHandle,
        requests: &[GroupOpeningRequest<'_, E, Self::CommitmentHandle, Self::WitnessHandle>],
    ) -> Result<Vec<PreparedGroupOpening<E, Self::PreparedOpeningHandle>>, AkitaError> {
        if self.fault.get() == Fault::DestinationOpening {
            return Err(AkitaError::InvalidInput(
                "injected destination opening failure".into(),
            ));
        }
        let requests = requests
            .iter()
            .map(|r| GroupOpeningRequest {
                context: r.context,
                source: match r.source {
                    OpeningSource::Commitment(c) => OpeningSource::Commitment(&c.0),
                    OpeningSource::Witness(w) => OpeningSource::Witness(&w.0),
                },
                plan: r.plan,
            })
            .collect::<Vec<_>>();
        self.inner.prepare_openings(session, &requests)
    }
    fn probe_opening_fold(
        &self,
        context: &ProofContext,
        opening: &Self::PreparedOpeningHandle,
        plan: &ValidatedFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<Self::AcceptedFoldHandle>, AkitaError> {
        self.inner.probe_opening_fold(context, opening, plan)
    }
}
impl OpaqueEorKernel<F, E> for PrivateCpu {
    fn prepare_eor(
        &self,
        session: &Self::ProofSessionHandle,
        context: &ProofContext,
        layout: &OpeningClaimsLayout,
        groups: &[EorGroupRequest<'_, E, Self::CommitmentHandle, Self::WitnessHandle>],
    ) -> Result<PreparedEor<E, Self::EorPreparationHandle>, AkitaError> {
        let groups = groups
            .iter()
            .map(|g| EorGroupRequest {
                source: match g.source {
                    OpeningSource::Commitment(c) => OpeningSource::Commitment(&c.0),
                    OpeningSource::Witness(w) => OpeningSource::Witness(&w.0),
                },
                point: g.point,
                ring_dimension: g.ring_dimension,
            })
            .collect::<Vec<_>>();
        self.inner.prepare_eor(session, context, layout, &groups)
    }
    fn begin_eor(
        &self,
        preparation: Self::EorPreparationHandle,
        eta: &[E],
        coefficients: &[E],
    ) -> Result<(E, Self::EorSessionHandle), AkitaError> {
        self.inner.begin_eor(preparation, eta, coefficients)
    }
    fn eor_round(
        &self,
        session: &mut Self::EorSessionHandle,
        round: usize,
        claim: E,
    ) -> Result<UnivariatePoly<E>, AkitaError> {
        self.inner.eor_round(session, round, claim)
    }
    fn bind_eor_round(
        &self,
        session: &mut Self::EorSessionHandle,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError> {
        self.inner.bind_eor_round(session, round, challenge)
    }
    fn finish_eor(&self, session: Self::EorSessionHandle) -> Result<Vec<E>, AkitaError> {
        self.inner.finish_eor(session)
    }
}
impl OpaqueStage3Kernel<F, E> for PrivateCpu {
    type Stage3SessionHandle = <Cpu as OpaqueStage3Kernel<F, E>>::Stage3SessionHandle;
    fn begin_stage3(
        &self,
        request: Stage3Request<'_, F, E, Self::ProofSessionHandle>,
    ) -> Result<(E, Self::Stage3SessionHandle), AkitaError> {
        self.inner.begin_stage3(request)
    }
    fn stage3_round_polynomial(
        &self,
        session: &mut Self::Stage3SessionHandle,
        round: usize,
        claim: E,
    ) -> Result<UnivariatePoly<E>, AkitaError> {
        self.inner.stage3_round_polynomial(session, round, claim)
    }
    fn bind_stage3_challenge(
        &self,
        session: &mut Self::Stage3SessionHandle,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError> {
        self.inner.bind_stage3_challenge(session, round, challenge)
    }
    fn finish_stage3(&self, session: Self::Stage3SessionHandle) -> Result<E, AkitaError> {
        self.inner.finish_stage3(session)
    }
}

impl SuccessorExportKernel<F, E> for PrivateCpu {
    type ExportPacket = CpuImportPacket<F>;
    fn instance_identity(&self) -> BackendInstanceIdentity {
        BackendInstanceIdentity::new::<Self>(self as *const Self as usize as u128)
    }
    fn export_successor(
        &self,
        session: &Self::ProofSessionHandle,
        witness: &Self::WitnessHandle,
        material: &Self::CommitmentMaterialHandle,
        plan: &ValidatedSuccessorHandoffPlan<'_, F>,
    ) -> Result<Self::ExportPacket, AkitaError> {
        self.events.borrow_mut().push("export");
        if self.fault.get() == Fault::Export {
            return Err(AkitaError::InvalidInput("injected export failure".into()));
        }
        let export = self
            .inner
            .export_successor(session, &witness.0, &material.0, plan)?;
        if !self.portable.get()
            && !matches!(
                self.fault.get(),
                Fault::Descriptor
                    | Fault::Packet
                    | Fault::InvalidDigits
                    | Fault::InvalidFields
                    | Fault::SignedDigits
            )
        {
            return Edge::<Cpu, Cpu>::convert(export, plan);
        }
        let (mut descriptor, mut sections) = export.into_sections()?;
        if self.fault.get() == Fault::Descriptor {
            for alteration in 0..5 {
                let mut invalid = descriptor.clone();
                match alteration {
                    0 => invalid.sections.clear(),
                    1 => invalid.sections[0].coefficients = usize::MAX,
                    2 => invalid.handoff += 1,
                    3 => invalid.sections[0].encoding = SuccessorEncoding::CanonicalField,
                    _ => invalid.sections.push(invalid.sections[0]),
                }
                assert!(plan.validate_export(&invalid).is_err());
            }
            // Shape remains self-consistent but cannot be used for this handoff.
            descriptor.handoff += 1;
        }
        if self.fault.get() == Fault::Packet {
            assert!(CpuImportPacket::<F>::new(descriptor.clone(), Vec::new()).is_err());
            let mut duplicate = sections.clone();
            duplicate[1] = duplicate[0].clone();
            assert!(CpuImportPacket::<F>::new(descriptor.clone(), duplicate).is_err());
            sections[0].1.pop();
        }
        if self.fault.get() == Fault::InvalidFields {
            sections
                .iter_mut()
                .find(|(s, _)| *s == SuccessorSection::InnerRows)
                .unwrap()
                .1
                .fill(0xff);
        }
        if self.fault.get() == Fault::InvalidDigits {
            sections
                .iter_mut()
                .find(|(s, _)| *s == SuccessorSection::LogicalDigits)
                .unwrap()
                .1
                .fill(0x55);
        }
        if self.fault.get() == Fault::SignedDigits {
            let logical = descriptor
                .sections
                .iter_mut()
                .find(|s| s.section == SuccessorSection::LogicalDigits)
                .unwrap();
            let SuccessorEncoding::PackedSigned { bit_width } = logical.encoding else {
                panic!("CPU source must be packed")
            };
            let bytes = &mut sections
                .iter_mut()
                .find(|(s, _)| *s == SuccessorSection::LogicalDigits)
                .unwrap()
                .1;
            let mut signed = Vec::with_capacity(logical.coefficients);
            let mask = (1u16 << bit_width) - 1;
            let sign = 1u16 << (bit_width - 1);
            for index in 0..logical.coefficients {
                let bit = index * bit_width as usize;
                let word = u16::from(bytes[bit / 8])
                    | (u16::from(bytes.get(bit / 8 + 1).copied().unwrap_or(0)) << 8);
                let value = (word >> (bit % 8)) & mask;
                signed.push(((value ^ sign) as i16 - sign as i16) as i8 as u8);
            }
            *bytes = signed;
            logical.encoding = SuccessorEncoding::SignedI8;
            logical.bytes = logical.coefficients;
        }
        // This is the public constructor available to any external backend/bridge.
        CpuImportPacket::new(descriptor, sections)
    }
}
impl SuccessorImportKernel<F, E> for PrivateCpu {
    type ImportPacket = CpuImportPacket<F>;
    fn import_successor(
        &self,
        session: &Self::ProofSessionHandle,
        plan: &ValidatedSuccessorHandoffPlan<'_, F>,
        packet: Self::ImportPacket,
    ) -> Result<ImportedSuccessor<Self::WitnessHandle, Self::CommitmentMaterialHandle>, AkitaError>
    {
        self.events.borrow_mut().push("import");
        if self.fault.get() == Fault::Import {
            return Err(AkitaError::InvalidInput("injected import failure".into()));
        }
        let packet = if self.portable.get() {
            let (descriptor, sections) = packet.into_sections()?;
            CpuImportPacket::new(descriptor, sections)?
        } else {
            packet
        };
        let (mut witness, material) = self
            .inner
            .import_successor(session, plan, packet)?
            .into_parts();
        assert!(self.inner.advance_witness_level(&mut witness).is_err());
        Ok(ImportedSuccessor::new(
            OwnedWitness(witness),
            OwnedMaterial(material),
        ))
    }
}
impl SuccessorBridge<PrivateCpu, PrivateCpu, F, E> for Edge<PrivateCpu, PrivateCpu> {
    fn convert(
        packet: CpuImportPacket<F>,
        _: &ValidatedSuccessorHandoffPlan<'_, F>,
    ) -> Result<CpuImportPacket<F>, AkitaError> {
        Ok(packet)
    }
}
impl SuccessorBridge<PrivateCpu, Cpu, F, E> for Edge<PrivateCpu, Cpu> {
    fn convert(
        packet: CpuImportPacket<F>,
        plan: &ValidatedSuccessorHandoffPlan<'_, F>,
    ) -> Result<CpuImportPacket<F>, AkitaError> {
        // Exercise a bridge building CPU imports using only its public section API.
        plan.validate_export(packet.descriptor())?;
        let (descriptor, sections) = packet.into_sections()?;
        CpuImportPacket::new(descriptor, sections)
    }
}
impl SuccessorBridge<Cpu, PrivateCpu, F, E> for Edge<Cpu, PrivateCpu> {
    fn convert(
        packet: akita_cpu_backend::CpuExportPacket<F>,
        plan: &ValidatedSuccessorHandoffPlan<'_, F>,
    ) -> Result<CpuImportPacket<F>, AkitaError> {
        Edge::<Cpu, Cpu>::convert(packet, plan)
    }
}
