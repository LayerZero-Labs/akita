//! Setup-independent capacity validation for the grouped typed prefix.

use std::ops::Range;

use akita_error::{checked, AkitaError};
use akita_params::{
    commit_only_setup_field_elements, setup_prefix_slot_field_elements, GroupCommitPhaseParams,
    SetupMatrixCapacity, SetupPrefixSlotId, SisModulusProfileId,
};
use akita_serialization::Valid;

use super::{BinaryTypedPrefix, GroupedNativeRootPlan};
use crate::root_repacking::RootCoefficientRepacking;

/// Validated native resources before choosing the shared setup descriptor.
/// This is capacity metadata, not native opening or security admission.
#[derive(Clone, Debug)]
pub struct GroupedPrefixCapacity {
    pub(super) plan: GroupedNativeRootPlan,
    pub(super) parts: Vec<(Range<usize>, BinaryTypedPrefix)>,
    pub(super) natural_len: usize,
    pub(super) successor: GroupCommitPhaseParams,
    pub(super) outgoing: RootCoefficientRepacking,
    pub(super) first: SetupPrefixSlotId,
    matrix_capacity: SetupMatrixCapacity,
}

impl GroupedPrefixCapacity {
    /// Validate the full grouped stream, selected successor and first commitment.
    pub fn new(
        plan: &GroupedNativeRootPlan,
        successor: GroupCommitPhaseParams,
        first: SetupPrefixSlotId,
    ) -> Result<Self, AkitaError> {
        successor.validate_frozen_precommit(64)?;
        first.check().map_err(|_| AkitaError::InvalidProof)?;
        for profile in [&successor, &first.commitment_profile] {
            if [
                profile.inner.matrix.sis_modulus_profile(),
                profile.outer.matrix.sis_modulus_profile(),
            ]
            .iter()
            .any(|&field| field != SisModulusProfileId::Q64Offset59)
            {
                return Err(AkitaError::InvalidInput(
                    "grouped successor field mismatch".into(),
                ));
            }
        }
        let outgoing = checked::pow2(successor.group.num_vars()).ok_or(AkitaError::InvalidProof)?;
        if successor.group.num_polynomials() != 1
            || successor.inner.digits.log_basis != 3
            || successor.inner.digits.num_digits != 1
            || successor.outer.digits.num_digits
                != akita_params::sis::compute_num_digits_field_width(
                    64,
                    successor.outer.digits.log_basis,
                )
            || successor.inner.matrix.ring_dimension()
                != plan.repacking().destination_ring_dimension()
            || outgoing < plan.repacking().domain_len()
        {
            return Err(AkitaError::InvalidInput(
                "grouped successor capacity mismatch".into(),
            ));
        }
        let last = plan
            .repacking()
            .segments()
            .len()
            .checked_sub(1)
            .ok_or(AkitaError::InvalidProof)?;
        let physical_cells = plan.repacking().segment_range(last)?.end;
        let committed_cells = checked::product([
            successor.blocks.live_ring_elements_per_claim,
            successor.inner.matrix.ring_dimension(),
        ])
        .ok_or(AkitaError::InvalidProof)?;
        if committed_cells < physical_cells {
            return Err(AkitaError::InvalidInput(
                "short grouped successor witness".into(),
            ));
        }
        let outgoing = plan.repacking().clone().with_domain_len(outgoing)?;
        let mut parts = Vec::new();
        parts
            .try_reserve_exact(plan.binary().len())
            .map_err(|_| AkitaError::InvalidProof)?;
        let mut natural_len = 0;
        for binary in plan.binary() {
            let typed = BinaryTypedPrefix::from_plan(&binary.source().manifest, binary.local())?;
            let range =
                checked::range(natural_len, typed.natural_len).ok_or(AkitaError::InvalidProof)?;
            natural_len = range.end;
            parts.push((range, typed));
        }
        if first.natural_len != natural_len {
            return Err(AkitaError::InvalidInput(
                "grouped prefix capacity mismatch".into(),
            ));
        }
        let mut required = setup_prefix_slot_field_elements(&first)?;
        for profile in std::iter::once(&successor)
            .chain(plan.binary().iter().map(|binary| &binary.source().carrier))
            .chain(
                plan.ordinary()
                    .into_iter()
                    .flat_map(|batch| batch.plans().iter().map(|plan| plan.profile())),
            )
        {
            required = required.max(commit_only_setup_field_elements(
                &profile.inner.matrix,
                &profile.outer.matrix,
                profile.outer_slice_count,
            )?);
        }
        Ok(Self {
            plan: plan.clone(),
            parts,
            natural_len,
            successor,
            outgoing,
            first,
            matrix_capacity: SetupMatrixCapacity {
                num_field_elements: required,
            },
        })
    }

    pub fn plan(&self) -> &GroupedNativeRootPlan {
        &self.plan
    }

    pub fn successor(&self) -> &GroupCommitPhaseParams {
        &self.successor
    }

    pub fn first(&self) -> &SetupPrefixSlotId {
        &self.first
    }

    pub fn matrix_capacity(&self) -> SetupMatrixCapacity {
        self.matrix_capacity
    }
}
