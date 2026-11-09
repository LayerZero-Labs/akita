//! Capacity-bound concatenation of existing typed setup streams.

use akita_algebra::{CanonicalEncoding, Field};
use akita_error::AkitaError;
use akita_params::{GroupCommitPhaseParams, SetupPrefixSlotId, SisModulusProfileId};
use akita_serialization::{AkitaSerialize, Valid};
use std::ops::Range;

use super::{
    BinaryCommittedGroupManifest, BinaryTypedPrefix, GroupedNativeRootPlan, GroupedPrefixCapacity,
};
use crate::root_repacking::RootCoefficientRepacking;
use crate::{AkitaSetupDescriptor, AkitaVerifierSetup};

/// Immutable setup/capacity binding authenticated by the grouped opening drivers.
#[derive(Clone, Debug)]
pub struct GroupedTypedPrefix {
    capacity: GroupedPrefixCapacity,
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
        let capacity = GroupedPrefixCapacity::new(plan, successor, first)?;
        Self::bind_capacity(capacity, setup)
    }

    /// Bind validated capacity to the final descriptor without recompiling streams.
    pub fn bind_capacity(
        capacity: GroupedPrefixCapacity,
        setup: AkitaSetupDescriptor,
    ) -> Result<Self, AkitaError> {
        setup.check().map_err(|_| AkitaError::InvalidProof)?;
        let required = capacity.matrix_capacity().num_field_elements;
        if setup.num_field_elements < required {
            return Err(AkitaError::InvalidInput(
                "grouped setup capacity mismatch".into(),
            ));
        }
        let plan = &capacity.plan;
        let successor = &capacity.successor;
        let first = &capacity.first;
        let parts = &capacity.parts;
        let mut identity = b"akita/grouped-root/typed-prefix/p64/v1".to_vec();
        plan.identity()
            .to_vec()
            .serialize_compressed(&mut identity)
            .map_err(|_| AkitaError::InvalidProof)?;
        identity.extend(BinaryCommittedGroupManifest::carrier_profile_identity(
            successor,
        )?);
        first
            .serialize_compressed(&mut identity)
            .map_err(|_| AkitaError::InvalidProof)?;
        setup
            .serialize_compressed(&mut identity)
            .map_err(|_| AkitaError::InvalidProof)?;
        for ((range, typed), binary) in parts.iter().zip(plan.binary()) {
            range
                .start
                .serialize_compressed(&mut identity)
                .map_err(|_| AkitaError::InvalidProof)?;
            typed
                .identity(&binary.source().manifest, first)?
                .serialize_compressed(&mut identity)
                .map_err(|_| AkitaError::InvalidProof)?;
        }
        let identity = crate::digest_serializable(&identity)
            .map_err(|_| AkitaError::InvalidProof)?
            .to_vec();
        Ok(Self {
            capacity,
            setup,
            identity,
        })
    }

    pub fn plan(&self) -> &GroupedNativeRootPlan {
        &self.capacity.plan
    }

    pub fn parts(&self) -> &[(Range<usize>, BinaryTypedPrefix)] {
        &self.capacity.parts
    }

    pub fn natural_len(&self) -> usize {
        self.capacity.natural_len
    }

    pub fn successor(&self) -> &GroupCommitPhaseParams {
        &self.capacity.successor
    }

    pub fn outgoing(&self) -> &RootCoefficientRepacking {
        &self.capacity.outgoing
    }

    pub fn first(&self) -> &SetupPrefixSlotId {
        &self.capacity.first
    }

    pub fn setup(&self) -> &AkitaSetupDescriptor {
        &self.setup
    }

    pub fn identity(&self) -> &[u8] {
        &self.identity
    }

    /// Reject substitution of sources, claims, signed profiles or their order.
    pub fn validate_plan(&self, plan: &GroupedNativeRootPlan) -> Result<(), AkitaError> {
        if plan.identity() != self.capacity.plan.identity() {
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
        let len = self.capacity.first.n_prefix()?;
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(len)
            .map_err(|_| AkitaError::InvalidProof)?;
        coefficients.resize(len, F::zero());
        for ((range, typed), binary) in self.capacity.parts.iter().zip(self.capacity.plan.binary())
        {
            let local = typed.setup_coefficients(
                &self.setup.setup_seed.seed,
                &binary.source().manifest,
                typed.domain_len,
            )?;
            coefficients
                .get_mut(range.clone())
                .ok_or(AkitaError::InvalidProof)?
                .copy_from_slice(
                    local
                        .get(..typed.natural_len)
                        .ok_or(AkitaError::InvalidProof)?,
                );
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
            || setup.prefix_context(&self.capacity.first) != Some(self.identity())
            || setup.prefix_slots().get(&self.capacity.first).is_none()
        {
            return Err(AkitaError::InvalidInput(
                "foreign grouped prefix context".into(),
            ));
        }
        Ok(())
    }
}
