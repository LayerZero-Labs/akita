//! Strict runtime schedule resolution.

use crate::audit::audit_resolved_schedule;
use crate::runtime::planned_next_witness_len;
use crate::PlannerPolicy;
use akita_error::AkitaError;
use akita_params::{
    root_input_witness_len, schedule_row_digest, validate_schedule_ring_dims,
    CommittedGroupBatchProfile, FoldSchedule, OpeningScheduleSelection,
};

/// One artifact row resolved to the exact verifier schedule and public identity.
#[derive(Clone, Debug)]
pub struct ResolvedScheduleRow {
    selection: OpeningScheduleSelection,
    profiles: CommittedGroupBatchProfile,
    schedule: FoldSchedule,
}

impl ResolvedScheduleRow {
    /// Semantically audit one expanded row and derive its public identity.
    ///
    /// This validates the exact committed profiles and expanded schedule before
    /// deriving the public row digest. It does not establish artifact trust or
    /// provenance. Admission happens when callers construct a
    /// [`ValidatedScheduleCatalog`](crate::ValidatedScheduleCatalog) from an
    /// application-chosen trusted source.
    ///
    /// The row must use only opening methods that proving and verification
    /// execute: coefficient packing at levels 0 and 1, evaluation trace at later
    /// nonterminal levels. A row that would open by extension opening reduction
    /// at level 0 or 1 is rejected here, before its geometry is audited, rather
    /// than on the first proof.
    pub fn try_new(
        profiles: CommittedGroupBatchProfile,
        schedule: FoldSchedule,
        policy: &PlannerPolicy,
    ) -> Result<Self, AkitaError> {
        schedule.validate_nonterminal_opening_execution(policy.claim_ext_degree)?;
        audit_resolved_schedule(&profiles, &schedule, policy)?;
        validate_schedule_ring_dims(&schedule)?;
        validate_canonical_transition_lengths(&profiles.opening_layout()?, &schedule, policy)?;
        let selection = OpeningScheduleSelection {
            row_digest: schedule_row_digest(&profiles, &schedule)?,
        };
        Ok(Self {
            selection,
            profiles,
            schedule,
        })
    }

    /// Batch-level public schedule selection.
    pub const fn selection(&self) -> OpeningScheduleSelection {
        self.selection
    }

    /// Exact ordered committed profiles accepted by this row.
    pub fn profiles(&self) -> &CommittedGroupBatchProfile {
        &self.profiles
    }

    /// Exact expanded schedule consumed by proving and verification.
    pub fn schedule(&self) -> &FoldSchedule {
        &self.schedule
    }

    /// Check that opening claims have the exact layout authorized by this row.
    pub fn validate_opening_layout(
        &self,
        opening_batch: &akita_params::OpeningClaimsLayout,
    ) -> Result<(), AkitaError> {
        if self.profiles.opening_layout()? != *opening_batch {
            return Err(AkitaError::InvalidInput(
                "committed-group descriptors do not match the opening layout".to_string(),
            ));
        }
        Ok(())
    }
}

pub(crate) fn validate_canonical_transition_lengths(
    root_layout: &akita_params::OpeningClaimsLayout,
    schedule: &FoldSchedule,
    policy: &PlannerPolicy,
) -> Result<(), AkitaError> {
    for (index, producer) in std::iter::once(&schedule.root)
        .chain(&schedule.recursive_folds)
        .enumerate()
    {
        let successor = schedule.recursive_folds.get(index).map_or(
            akita_params::FoldSuccessor::Terminal(&schedule.terminal),
            |fold| akita_params::FoldSuccessor::Recursive(&fold.params),
        );
        if producer.params.successor_block_len
            != (producer.params.witness_chunk.num_chunks > 1)
                .then_some(successor.source_block_len()?)
        {
            return Err(AkitaError::InvalidSetup(
                "witness padding disagrees with successor block".into(),
            ));
        }
    }
    let field_bits = policy.decomposition.field_bits();
    let root_params = &schedule.root.params;
    if !root_params.witness_chunk_ends.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "root witness ownership must be proportional".into(),
        ));
    }
    let expected_root_input = root_input_witness_len(root_params);
    if schedule.root.input_witness_len != expected_root_input {
        return Err(AkitaError::InvalidSetup(format!(
            "root input witness length {} is not canonical; expected {expected_root_input}",
            schedule.root.input_witness_len
        )));
    }
    let expected_root_output = root_params.output_witness_len_for_field_bits(
        field_bits,
        policy.claim_ext_degree,
        root_layout,
    )?;
    if schedule.root.output_witness_len != expected_root_output {
        return Err(AkitaError::InvalidSetup(format!(
            "root output witness length {} is not canonical; expected {expected_root_output}",
            schedule.root.output_witness_len
        )));
    }

    let mut expected_input = expected_root_output;
    let mut producer = &schedule.root;
    let mut producer_opening = root_layout.clone();
    for (index, step) in schedule.recursive_folds.iter().enumerate() {
        let geometry = akita_params::RelationWitnessGeometry::for_level(
            &producer.params,
            &producer_opening,
            policy.claim_ext_degree,
        )?;
        let layout = akita_params::WitnessLayout::new(
            &producer.params,
            &producer_opening,
            &geometry,
            producer.params.witness_chunk.num_chunks,
            akita_params::RelationQuotientPlan::for_field_bits(&producer.params, field_bits)?,
        )?;
        let (len, ends) = layout.chunk_shape()?.align(
            akita_params::FoldSuccessor::Recursive(&step.params).source_block_len()?,
            step.params.witness_chunk.num_chunks,
        )?;
        if len != expected_input || ends != step.params.witness_chunk_ends {
            return Err(AkitaError::InvalidSetup(
                "recursive witness ownership is not inherited from its producer".into(),
            ));
        }
        if step.input_witness_len != expected_input {
            return Err(AkitaError::InvalidSetup(format!(
                "recursive fold {index} input witness length {} is not canonical; expected {expected_input}",
                step.input_witness_len
            )));
        }
        let expected_output = planned_next_witness_len(
            field_bits,
            policy.claim_ext_degree,
            &step.params,
            1,
            step.params.witness_chunk.num_chunks,
        )?
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "recursive fold {index} uses unsupported compression source geometry"
            ))
        })?;
        if step.output_witness_len != expected_output {
            return Err(AkitaError::InvalidSetup(format!(
                "recursive fold {index} output witness length {} is not canonical; expected {expected_output}",
                step.output_witness_len
            )));
        }
        expected_input = expected_output;
        producer = step;
        producer_opening = akita_params::suffix_opening_layout(
            step.input_witness_len,
            step.params
                .setup_prefix()
                .and_then(|prefix| prefix.setup_natural_len),
        )?;
    }
    if schedule.terminal.input_witness_len != expected_input {
        return Err(AkitaError::InvalidSetup(format!(
            "terminal input witness length {} is not canonical; expected {expected_input}",
            schedule.terminal.input_witness_len
        )));
    }
    Ok(())
}
