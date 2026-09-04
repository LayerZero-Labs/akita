//! Strict runtime schedule resolution.

use crate::audit::audit_resolved_schedule;
use crate::runtime::planned_next_witness_len;
use crate::PlannerPolicy;
use akita_error::AkitaError;
use akita_types::{
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
    /// [`TrustedScheduleCatalog`](crate::TrustedScheduleCatalog) from an
    /// application-chosen trusted source.
    pub fn try_new(
        profiles: CommittedGroupBatchProfile,
        schedule: FoldSchedule,
        policy: &PlannerPolicy,
    ) -> Result<Self, AkitaError> {
        audit_resolved_schedule(&profiles, &schedule, policy)?;
        validate_schedule_ring_dims(&schedule)?;
        validate_canonical_transition_lengths(&profiles, &schedule, policy)?;
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
        opening_batch: &akita_types::OpeningClaimsLayout,
    ) -> Result<(), AkitaError> {
        if self.profiles.opening_layout()? != *opening_batch {
            return Err(AkitaError::InvalidInput(
                "committed-group descriptors do not match the opening layout".to_string(),
            ));
        }
        Ok(())
    }
}

fn validate_canonical_transition_lengths(
    profiles: &CommittedGroupBatchProfile,
    schedule: &FoldSchedule,
    policy: &PlannerPolicy,
) -> Result<(), AkitaError> {
    let field_bits = policy.decomposition.field_bits();
    let root_params = &schedule.root.params;
    let expected_root_input = root_input_witness_len(root_params);
    if schedule.root.input_witness_len != expected_root_input {
        return Err(AkitaError::InvalidSetup(format!(
            "root input witness length {} is not canonical; expected {expected_root_input}",
            schedule.root.input_witness_len
        )));
    }
    let expected_root_output = if root_params.has_preceding_groups() {
        root_params.output_witness_len_for_field_bits(
            field_bits,
            policy.claim_ext_degree,
            &profiles.opening_layout()?,
        )?
    } else {
        planned_next_witness_len(
            field_bits,
            policy.claim_ext_degree,
            root_params,
            profiles.final_group.group.num_polynomials(),
            root_params.witness_chunk.num_chunks,
        )?
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "root schedule uses unsupported compression source geometry".to_string(),
            )
        })?
    };
    if schedule.root.output_witness_len != expected_root_output {
        return Err(AkitaError::InvalidSetup(format!(
            "root output witness length {} is not canonical; expected {expected_root_output}",
            schedule.root.output_witness_len
        )));
    }

    let mut expected_input = expected_root_output;
    for (index, step) in schedule.recursive_folds.iter().enumerate() {
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
    }
    if schedule.terminal.input_witness_len != expected_input {
        return Err(AkitaError::InvalidSetup(format!(
            "terminal input witness length {} is not canonical; expected {expected_input}",
            schedule.terminal.input_witness_len
        )));
    }
    Ok(())
}
