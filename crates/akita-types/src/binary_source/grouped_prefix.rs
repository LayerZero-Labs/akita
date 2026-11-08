//! Capacity-bound concatenation of existing typed setup streams.

use akita_algebra::{CanonicalEncoding, Field};
use akita_error::{checked, AkitaError};
use akita_params::{
    commit_only_setup_field_elements, setup_prefix_slot_field_elements, GroupCommitPhaseParams,
    SetupPrefixSlotId, SisModulusProfileId,
};
use akita_serialization::{AkitaSerialize, Valid};
use std::ops::Range;

use super::{BinaryCommittedGroupManifest, BinaryTypedPrefix, GroupedNativeRootPlan};
use crate::root_repacking::RootCoefficientRepacking;
use crate::{AkitaSetupDescriptor, AkitaVerifierSetup};

/// One immutable setup/capacity binding. No opening admission is represented.
#[derive(Clone, Debug)]
pub struct GroupedTypedPrefix {
    plan: GroupedNativeRootPlan,
    parts: Vec<(Range<usize>, BinaryTypedPrefix)>,
    natural_len: usize,
    successor: GroupCommitPhaseParams,
    outgoing: RootCoefficientRepacking,
    first: SetupPrefixSlotId,
    setup: AkitaSetupDescriptor,
    identity: Vec<u8>,
}

impl GroupedTypedPrefix {
    /// Compile each A/B/D stream unchanged and concatenate its natural cells.
    /// The complete selected successor and prefix profiles are validated and bound.
    pub fn new(
        plan: &GroupedNativeRootPlan,
        successor: GroupCommitPhaseParams,
        first: SetupPrefixSlotId,
        setup: AkitaSetupDescriptor,
    ) -> Result<Self, AkitaError> {
        successor.validate_frozen_precommit(64)?;
        setup.check().map_err(|_| AkitaError::InvalidProof)?;
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
            .chain(plan.ordinary().into_iter().flat_map(|batch| {
                batch.plans().iter().map(|plan| plan.profile())
            }))
        {
            required = required.max(commit_only_setup_field_elements(
                &profile.inner.matrix,
                &profile.outer.matrix,
                profile.outer_slice_count,
            )?);
        }
        if setup.num_field_elements < required {
            return Err(AkitaError::InvalidInput(
                "grouped setup capacity mismatch".into(),
            ));
        }
        let mut identity = b"akita/grouped-root/typed-prefix/p64/v1".to_vec();
        plan.identity()
            .to_vec()
            .serialize_compressed(&mut identity)
            .map_err(|_| AkitaError::InvalidProof)?;
        identity.extend(BinaryCommittedGroupManifest::carrier_profile_identity(
            &successor,
        )?);
        first
            .serialize_compressed(&mut identity)
            .map_err(|_| AkitaError::InvalidProof)?;
        setup
            .serialize_compressed(&mut identity)
            .map_err(|_| AkitaError::InvalidProof)?;
        for ((range, typed), binary) in parts.iter().zip(plan.binary()) {
            range.start
                .serialize_compressed(&mut identity)
                .map_err(|_| AkitaError::InvalidProof)?;
            typed
                .identity(&binary.source().manifest, &first)?
                .serialize_compressed(&mut identity)
                .map_err(|_| AkitaError::InvalidProof)?;
        }
        let identity = crate::digest_serializable(&identity)
            .map_err(|_| AkitaError::InvalidProof)?
            .to_vec();
        Ok(Self {
            plan: plan.clone(),
            parts,
            natural_len,
            successor,
            outgoing,
            first,
            setup,
            identity,
        })
    }

    pub fn plan(&self) -> &GroupedNativeRootPlan {
        &self.plan
    }

    pub fn parts(&self) -> &[(Range<usize>, BinaryTypedPrefix)] {
        &self.parts
    }

    pub fn natural_len(&self) -> usize {
        self.natural_len
    }

    pub fn successor(&self) -> &GroupCommitPhaseParams {
        &self.successor
    }

    pub fn outgoing(&self) -> &RootCoefficientRepacking {
        &self.outgoing
    }

    pub fn first(&self) -> &SetupPrefixSlotId {
        &self.first
    }

    pub fn setup(&self) -> &AkitaSetupDescriptor {
        &self.setup
    }

    pub fn identity(&self) -> &[u8] {
        &self.identity
    }

    /// Reject substitution of sources, claims, signed profiles or their order.
    pub fn validate_plan(&self, plan: &GroupedNativeRootPlan) -> Result<(), AkitaError> {
        if plan.identity() != self.plan.identity() {
            return Err(AkitaError::InvalidInput(
                "grouped prefix plan mismatch".into(),
            ));
        }
        Ok(())
    }

    /// Regenerate from the bound seed. No caller-supplied stream or source list.
    pub fn setup_coefficients<F: Field + CanonicalEncoding>(&self) -> Result<Vec<F>, AkitaError> {
        if akita_params::field_modulus::<F>()? != SisModulusProfileId::Q64Offset59.modulus() {
            return Err(AkitaError::InvalidInput(
                "grouped setup field mismatch".into(),
            ));
        }
        let len = self.first.n_prefix()?;
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(len)
            .map_err(|_| AkitaError::InvalidProof)?;
        coefficients.resize(len, F::zero());
        for ((range, typed), binary) in self.parts.iter().zip(self.plan.binary()) {
            let local = typed.setup_coefficients(
                &self.setup.setup_seed.seed,
                &binary.source().manifest,
                typed.domain_len,
            )?;
            coefficients
                .get_mut(range.clone())
                .ok_or(AkitaError::InvalidProof)?
                .copy_from_slice(local.get(..typed.natural_len).ok_or(AkitaError::InvalidProof)?);
        }
        Ok(coefficients)
    }

    /// Require the exact context installed by authenticated grouped preparation.
    pub fn validate_verifier_setup<F: Field + CanonicalEncoding>(
        &self,
        setup: &AkitaVerifierSetup<F>,
    ) -> Result<(), AkitaError> {
        if akita_params::field_modulus::<F>()? != SisModulusProfileId::Q64Offset59.modulus()
            || setup.expanded().descriptor() != &self.setup
            || setup.prefix_context(&self.first) != Some(self.identity())
            || setup.prefix_slots().get(&self.first).is_none()
        {
            return Err(AkitaError::InvalidInput(
                "foreign grouped prefix context".into(),
            ));
        }
        Ok(())
    }
}
