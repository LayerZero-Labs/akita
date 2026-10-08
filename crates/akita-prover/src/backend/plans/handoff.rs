use super::{ValidatedRecursiveWitnessCommitPlan, WitnessCommitmentParameters};
use crate::backend::{
    RecursiveWitnessManifest, SuccessorEncoding, SuccessorExportDescriptor, SuccessorSection,
};
use akita_error::{checked, AkitaError};
use akita_params::{
    CompressionChainPlan, FoldSchedule, OpeningClaimsLayout, RingRelationMode, WitnessLayout,
};
use akita_types::{AkitaSetupDescriptor, RingVec};
use jolt_field::{CanonicalEncoding, Field};

/// Entire work performed by a producer, including committing its successor.
pub struct FoldExecutionRequirements<'a> {
    schedule: &'a FoldSchedule,
    root_layout: &'a OpeningClaimsLayout,
    level: usize,
}

impl<'a> FoldExecutionRequirements<'a> {
    pub(crate) fn new(
        schedule: &'a FoldSchedule,
        root_layout: &'a OpeningClaimsLayout,
        level: usize,
    ) -> Result<Self, AkitaError> {
        if level >= schedule.num_fold_levels() {
            return Err(AkitaError::InvalidInput(
                "execution level is outside the schedule".into(),
            ));
        }
        Ok(Self {
            schedule,
            root_layout,
            level,
        })
    }
    pub fn schedule(&self) -> &FoldSchedule {
        self.schedule
    }
    pub fn root_layout(&self) -> &OpeningClaimsLayout {
        self.root_layout
    }
    pub fn level(&self) -> usize {
        self.level
    }
    pub fn is_terminal(&self) -> bool {
        self.level + 1 == self.schedule.num_fold_levels()
    }
}

pub enum SuccessorPublicBinding<'a, F: Field> {
    Outer(&'a RingVec<F>),
    Terminal(&'a [F]),
}

/// Protocol-issued authority derived from the exact producer and commitment plans.
pub struct ValidatedSuccessorHandoffPlan<'a, F: Field> {
    pub(crate) handoff: u128,
    pub(crate) setup: &'a AkitaSetupDescriptor,
    pub(crate) schedule: &'a FoldSchedule,
    pub(crate) producer: usize,
    pub(crate) layout: WitnessLayout,
    pub(crate) producer_parameters: &'a akita_params::CommittedGroupParams,
    pub(crate) commitment: &'a ValidatedRecursiveWitnessCommitPlan,
    pub(crate) binding: SuccessorPublicBinding<'a, F>,
}

impl<F: Field + CanonicalEncoding> ValidatedSuccessorHandoffPlan<'_, F> {
    pub fn handoff_id(&self) -> u128 {
        self.handoff
    }
    pub fn setup(&self) -> &AkitaSetupDescriptor {
        self.setup
    }
    pub fn schedule(&self) -> &FoldSchedule {
        self.schedule
    }
    pub fn producer_level(&self) -> usize {
        self.producer
    }
    pub fn successor_level(&self) -> usize {
        self.producer + 1
    }
    pub fn witness_layout(&self) -> &WitnessLayout {
        &self.layout
    }
    pub fn log_basis(&self) -> u32 {
        self.producer_parameters.witness_log_basis()
    }
    pub fn producer_parameters(&self) -> &akita_params::CommittedGroupParams {
        self.producer_parameters
    }
    pub fn commitment(&self) -> &ValidatedRecursiveWitnessCommitPlan {
        self.commitment
    }
    pub fn binding(&self) -> &SuccessorPublicBinding<'_, F> {
        &self.binding
    }
    pub fn manifest(&self) -> Result<RecursiveWitnessManifest, AkitaError> {
        RecursiveWitnessManifest::try_new(
            self.commitment.logical_len(),
            self.commitment.padded_len(),
            self.commitment.ring_dimension(),
        )
    }
    pub fn inner_coefficients(&self) -> Result<usize, AkitaError> {
        let (blocks, rows, ring) = match self.commitment.parameters() {
            WitnessCommitmentParameters::Recursive(p) => {
                let p = &p.own_group().profile;
                (
                    p.blocks.live_blocks,
                    p.inner.matrix.output_rank(),
                    p.inner.matrix.ring_dimension(),
                )
            }
            WitnessCommitmentParameters::Terminal(p) => {
                (p.blocks.live_blocks, p.inner.matrix.output_rank(), p.d_a())
            }
        };
        checked::product([blocks, rows, ring])
            .ok_or_else(|| AkitaError::InvalidInput("successor inner geometry overflow".into()))
    }
    pub fn compression(
        &self,
    ) -> Result<Option<(CompressionChainPlan, RingRelationMode)>, AkitaError> {
        let WitnessCommitmentParameters::Recursive(p) = self.commitment.parameters() else {
            return Ok(None);
        };
        if !p.payload_mode.is_compressed() {
            return Ok(None);
        }
        let profile = &p.own_group().profile;
        let count = profile.outer_slice_count.complete_source_coefficients(
            profile.outer.matrix.output_rank(),
            profile.outer.matrix.ring_dimension(),
        )?;
        Ok(Some((
            CompressionChainPlan::for_complete_source(
                profile.outer.matrix.sis_modulus_profile(),
                count,
            )?,
            p.ring_relation_mode,
        )))
    }
    /// Validate all lengths before a backend allocates from an export descriptor.
    pub fn validate_export(
        &self,
        descriptor: &SuccessorExportDescriptor,
    ) -> Result<(), AkitaError> {
        if descriptor.handoff != self.handoff
            || descriptor.manifest != self.manifest()?
            || self.layout.live_coeff_len() != self.commitment.logical_len()
        {
            return Err(AkitaError::InvalidInput(
                "successor export differs from its handoff plan".into(),
            ));
        }
        let compression = self.compression()?;
        let maps = compression.as_ref().map_or(&[][..], |(p, _)| p.maps());
        let max_sections = checked::sum([2, maps.len(), maps.len()])
            .ok_or_else(|| AkitaError::InvalidInput("successor section count overflow".into()))?;
        if descriptor.sections.len() > max_sections {
            return Err(AkitaError::InvalidInput(
                "too many successor sections".into(),
            ));
        }
        for (index, section) in descriptor.sections.iter().enumerate() {
            if descriptor.sections[..index]
                .iter()
                .any(|s| s.section == section.section)
            {
                return Err(AkitaError::InvalidInput(
                    "duplicate successor section".into(),
                ));
            }
            let coefficients = match section.section {
                SuccessorSection::LogicalDigits => self.commitment.logical_len(),
                SuccessorSection::InnerRows => self.inner_coefficients()?,
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
            if !descriptor.sections.iter().any(|s| s.section == required) {
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
                if !descriptor.sections.iter().any(|s| s.section == required) {
                    return Err(AkitaError::InvalidInput(
                        "missing compression section".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}
