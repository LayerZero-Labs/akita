//! Reusable setup-prefix commitments with backend-owned retained sources.

use crate::backend::{CommitmentHandleMetadata, ProverHandleFamily};
use akita_error::AkitaError;
use akita_serialization::Valid;
use akita_types::{
    AkitaSetupDescriptor, AkitaSetupSeed, SetupPrefixSlotId, SetupPrefixVerifierSlot,
};
use jolt_field::{CanonicalEncoding, Field};
use std::collections::BTreeMap;

/// Public prefix commitment paired with the ordinary opaque commitment handle.
#[derive(Clone)]
pub struct PreparedSetupPrefix<F: Field, H> {
    pub public: SetupPrefixVerifierSlot<F>,
    pub commitment_handle: H,
}

/// Prepare an exact prefix without exposing its source or retained material.
pub trait SetupPrefixKernel<F: Field + CanonicalEncoding, E: Field>:
    ProverHandleFamily<F, E>
{
    fn prepare_setup_prefix(
        &self,
        setup: &AkitaSetupDescriptor,
        prefix: &SetupPrefixSlotId,
    ) -> Result<PreparedSetupPrefix<F, Self::CommitmentHandle>, AkitaError>;
}

/// Immutable reusable commitments indexed by their complete public prefix parameters.
#[derive(Clone)]
pub struct SetupPrefixProverRegistry<F: Field, H> {
    setup_seed: AkitaSetupSeed,
    slots: BTreeMap<SetupPrefixSlotId, PreparedSetupPrefix<F, H>>,
}

impl<F: Field, H> SetupPrefixProverRegistry<F, H> {
    pub fn new(setup_seed: AkitaSetupSeed) -> Self {
        Self {
            setup_seed,
            slots: BTreeMap::new(),
        }
    }
    pub fn setup_seed(&self) -> &AkitaSetupSeed {
        &self.setup_seed
    }
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
    pub fn len(&self) -> usize {
        self.slots.len()
    }
    pub fn get(&self, id: &SetupPrefixSlotId) -> Option<&PreparedSetupPrefix<F, H>> {
        self.slots.get(id)
    }
    pub fn iter(&self) -> impl Iterator<Item = (&SetupPrefixSlotId, &PreparedSetupPrefix<F, H>)> {
        self.slots.iter()
    }
    pub fn verifier_slots(&self) -> Vec<SetupPrefixVerifierSlot<F>> {
        self.slots
            .values()
            .map(|slot| slot.public.clone())
            .collect()
    }

    pub fn insert(&mut self, slot: PreparedSetupPrefix<F, H>) -> Result<(), AkitaError>
    where
        F: Valid,
        H: CommitmentHandleMetadata,
    {
        slot.public
            .check()
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        let metadata = slot.commitment_handle.metadata();
        let domain = slot.public.id.n_prefix()?;
        if metadata.num_polynomials() != 1
            || metadata.num_vars() != domain.trailing_zeros() as usize
        {
            return Err(AkitaError::InvalidSetup(
                "setup prefix source shape does not match its public domain".into(),
            ));
        }
        if self.slots.contains_key(&slot.public.id) {
            return Err(AkitaError::InvalidSetup(
                "duplicate setup prefix slot id".into(),
            ));
        }
        self.slots.insert(slot.public.id.clone(), slot);
        Ok(())
    }
}
