//! Reusable setup-prefix commitments with backend-owned retained sources.

use crate::backend::CommitmentHandleMetadata;
use akita_error::AkitaError;
use akita_serialization::Valid;
use akita_types::{SetupPrefixSlotId, SetupPrefixVerifierSlot};
use jolt_field::Field;
use std::collections::BTreeMap;

/// Public prefix commitment paired with the ordinary opaque commitment handle.
#[derive(Clone)]
pub struct PreparedSetupPrefix<F: Field, H> {
    pub public: SetupPrefixVerifierSlot<F>,
    pub commitment_handle: H,
}

/// Immutable reusable commitments indexed by their complete public prefix parameters.
#[derive(Clone)]
pub struct SetupPrefixProverRegistry<F: Field, H> {
    slots: BTreeMap<SetupPrefixSlotId, PreparedSetupPrefix<F, H>>,
}

impl<F: Field, H> Default for SetupPrefixProverRegistry<F, H> {
    fn default() -> Self {
        Self {
            slots: BTreeMap::new(),
        }
    }
}

impl<F: Field, H> SetupPrefixProverRegistry<F, H> {
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn get(&self, id: &SetupPrefixSlotId) -> Option<&PreparedSetupPrefix<F, H>> {
        self.slots.get(id)
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
