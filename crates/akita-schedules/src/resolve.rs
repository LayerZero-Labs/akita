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
    canonical_root_transition: bool,
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
    /// execute: coefficient packing at folds 0 and 1 and every multi-chunk fold,
    /// either method at later single-chunk nonterminal folds. Unsupported
    /// evaluation trace is rejected here before geometry audit or the first proof.
    pub fn try_new(
        profiles: CommittedGroupBatchProfile,
        schedule: FoldSchedule,
        policy: &PlannerPolicy,
    ) -> Result<Self, AkitaError> {
        Self::with_root_transition(profiles, schedule, policy, &OrdinaryRootTransition)
    }

    /// Extension seam: audit a selected root transition while sharing every
    /// suffix invariant; <https://github.com/LayerZero-Labs/akita/issues/285>.
    pub(crate) fn with_root_transition(
        profiles: CommittedGroupBatchProfile,
        schedule: FoldSchedule,
        policy: &PlannerPolicy,
        transition: &impl RootTransitionValidation,
    ) -> Result<Self, AkitaError> {
        schedule.validate_nonterminal_opening_execution(policy.claim_ext_degree)?;
        audit_resolved_schedule(&profiles, &schedule, policy)?;
        validate_schedule_ring_dims(&schedule)?;
        validate_canonical_transition_lengths(
            &profiles.opening_layout()?,
            &schedule,
            policy,
            transition,
        )?;
        let selection = OpeningScheduleSelection {
            row_digest: schedule_row_digest(&profiles, &schedule)?,
        };
        Ok(Self {
            selection,
            profiles,
            schedule,
            canonical_root_transition: transition.is_ordinary(),
        })
    }

    /// Extension seam: extension rows require explicit admission and cannot be
    /// encoded as ordinary artifacts; <https://github.com/LayerZero-Labs/akita/issues/285>.
    pub(crate) fn require_ordinary_artifact(&self) -> Result<(), AkitaError> {
        if !self.canonical_root_transition {
            return Err(AkitaError::InvalidSetup(
                "selected root transition requires explicit catalog admission".into(),
            ));
        }
        Ok(())
    }

    /// Whether this row uses an explicitly validated native-root transition.
    /// The PCS reads this only after resolving a trusted catalog selection.
    /// Ordinary artifact decoding cannot create native rows. This checks geometry
    /// and the suffix; production use also requires the open security-ledger items.
    /// There is no separate runtime production-admission gate.
    pub const fn is_native_root(&self) -> bool {
        !self.canonical_root_transition
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

/// Extension seam: a root producer supplies its transition contract while the
/// recursive suffix stays canonical; <https://github.com/LayerZero-Labs/akita/issues/285>.
pub(crate) trait RootTransitionValidation {
    fn validate_root(
        &self,
        root_layout: &akita_params::OpeningClaimsLayout,
        schedule: &FoldSchedule,
        policy: &PlannerPolicy,
    ) -> Result<usize, AkitaError>;

    fn validate_first_successor(
        &self,
        producer: &akita_params::FoldParams,
        producer_opening: &akita_params::OpeningClaimsLayout,
        step: &akita_params::FoldParams,
        expected_input: usize,
        policy: &PlannerPolicy,
    ) -> Result<(), AkitaError>;

    fn is_ordinary(&self) -> bool;
}

pub(crate) struct OrdinaryRootTransition;

impl RootTransitionValidation for OrdinaryRootTransition {
    fn validate_root(
        &self,
        root_layout: &akita_params::OpeningClaimsLayout,
        schedule: &FoldSchedule,
        policy: &PlannerPolicy,
    ) -> Result<usize, AkitaError> {
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
            policy.decomposition.field_bits(),
            policy.claim_ext_degree,
            root_layout,
        )?;
        if schedule.root.output_witness_len != expected_root_output {
            return Err(AkitaError::InvalidSetup(format!(
                "root output witness length {} is not canonical; expected {expected_root_output}",
                schedule.root.output_witness_len
            )));
        }
        if let Some(prefix) = schedule
            .recursive_folds
            .first()
            .and_then(|fold| fold.params.setup_prefix())
        {
            let expected = akita_params::active_setup_field_len(root_params, root_layout)?
                .max(schedule.root_accumulated_setup_prefix_len);
            if prefix.setup_natural_len != Some(expected) {
                return Err(AkitaError::InvalidSetup(
                    "root setup prefix does not match the combined setup footprint".into(),
                ));
            }
        }

        Ok(expected_root_output)
    }

    fn validate_first_successor(
        &self,
        producer: &akita_params::FoldParams,
        producer_opening: &akita_params::OpeningClaimsLayout,
        step: &akita_params::FoldParams,
        expected_input: usize,
        policy: &PlannerPolicy,
    ) -> Result<(), AkitaError> {
        validate_inherited_chunks(producer, producer_opening, step, expected_input, policy)
    }

    fn is_ordinary(&self) -> bool {
        true
    }
}

/// Extension seam: share canonical suffix validation with an explicitly selected
/// root transition; <https://github.com/LayerZero-Labs/akita/issues/285>.
pub(crate) fn validate_canonical_transition_lengths(
    root_layout: &akita_params::OpeningClaimsLayout,
    schedule: &FoldSchedule,
    policy: &PlannerPolicy,
    transition: &impl RootTransitionValidation,
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
    let expected_root_output = transition.validate_root(root_layout, schedule, policy)?;

    let mut expected_input = expected_root_output;
    let mut producer = &schedule.root;
    let mut producer_opening = root_layout.clone();
    for (index, step) in schedule.recursive_folds.iter().enumerate() {
        if index == 0 {
            transition.validate_first_successor(
                producer,
                &producer_opening,
                step,
                expected_input,
                policy,
            )?;
        } else {
            validate_inherited_chunks(producer, &producer_opening, step, expected_input, policy)?;
        }
        if step.input_witness_len != expected_input {
            return Err(AkitaError::InvalidSetup(format!(
                "recursive fold {index} input witness length {} is not canonical; expected \
                 {expected_input}",
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
                "recursive fold {index} output witness length {} is not canonical; expected \
                 {expected_output}",
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

fn validate_inherited_chunks(
    producer: &akita_params::FoldParams,
    producer_opening: &akita_params::OpeningClaimsLayout,
    step: &akita_params::FoldParams,
    expected_input: usize,
    policy: &PlannerPolicy,
) -> Result<(), AkitaError> {
    let field_bits = policy.decomposition.field_bits();
    let geometry = akita_params::RelationWitnessGeometry::for_level(
        &producer.params,
        producer_opening,
        policy.claim_ext_degree,
    )?;
    let layout = akita_params::WitnessLayout::new(
        &producer.params,
        producer_opening,
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
    Ok(())
}
