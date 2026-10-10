use super::{ValidatedRecursiveWitnessCommitPlan, WitnessCommitmentParameters};
use crate::backend::{HandoffMetadata, RecursiveWitnessManifest};
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
    /// Check packet identity and witness geometry independently of its representation.
    pub fn validate_metadata(&self, metadata: &HandoffMetadata) -> Result<(), AkitaError> {
        if metadata.handoff != self.handoff || metadata.manifest != self.manifest()? {
            return Err(AkitaError::InvalidInput(
                "successor export differs from its handoff plan".into(),
            ));
        }
        Ok(())
    }
}
