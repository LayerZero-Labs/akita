//! Backend-owned setup-prefix material and persistence.
use super::PortableCommitmentHandle;
use akita_error::AkitaError;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use akita_types::{
    setup_prefix_compression_plan, AkitaSetupSeed, SetupPrefixPublicCommitment, SetupPrefixSlotId,
    SetupPrefixVerifierSlot,
};
use jolt_field::Field;
use std::{
    collections::BTreeMap,
    io::{Read, Write},
};
const MAX_SETUP_PREFIX_SLOTS: usize = 4096;
fn read_limited_usize<R: Read>(
    reader: R,
    compress: Compress,
    validate: Validate,
    max: usize,
) -> Result<usize, SerializationError> {
    let len = usize::deserialize_with_mode(reader, compress, validate, &())?;
    if len > max {
        return Err(SerializationError::LengthLimitExceeded {
            len: u64::try_from(len).unwrap_or(u64::MAX),
            max,
        });
    }
    Ok(len)
}
/// Persisted setup-prefix artifact for validated import into a CPU backend.
///
/// The public commitment uses flat ring-coefficient rows. Private retained
/// commitment material is validated against `id.d_setup` and the per-row
/// coefficient width by [`Valid::check`] and checked against the backend's
/// setup when imported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupPrefixSlot<F: Field> {
    pub id: SetupPrefixSlotId,
    pub commitment: SetupPrefixPublicCommitment<F>,
    pub(crate) hint: PortableCommitmentHandle<F>,
}

impl<F: Field + Valid> Valid for SetupPrefixSlot<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.id.check()?;
        self.commitment.check()?;
        let compression_plan = setup_prefix_compression_plan(&self.id.commitment_profile)?;
        let expected_payload_coefficients = compression_plan.terminal_coefficients();
        if self.commitment.rows.len() != 1 {
            return Err(SerializationError::InvalidData(
                "setup prefix commitment must contain one compressed payload".into(),
            ));
        }
        for row in &self.commitment.rows {
            if row.coeff_len() != expected_payload_coefficients {
                return Err(SerializationError::InvalidData(format!(
                    "setup prefix prover slot commitment row has {} coefficients, expected {}",
                    row.coeff_len(),
                    expected_payload_coefficients
                )));
            }
        }
        self.hint.check()?;
        self.hint
            .validate_outer_compression(&compression_plan)
            .map_err(|error| SerializationError::InvalidData(error.to_string()))
    }
}

impl<F: Field + AkitaSerialize> AkitaSerialize for SetupPrefixSlot<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.id.serialize_with_mode(&mut writer, compress)?;
        self.commitment.serialize_with_mode(&mut writer, compress)?;
        self.hint.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.id.serialized_size(compress)
            + self.commitment.serialized_size(compress)
            + self.hint.serialized_size(compress)
    }
}

impl<F> AkitaDeserialize for SetupPrefixSlot<F>
where
    F: Field + Valid + AkitaDeserialize<Context = ()>,
{
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let id = SetupPrefixSlotId::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let commitment = SetupPrefixPublicCommitment::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &(),
        )?;
        let hint =
            PortableCommitmentHandle::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self {
            id,
            commitment,
            hint,
        };
        if validate == Validate::Yes {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: Field> SetupPrefixSlot<F> {
    fn validate_compression_hint(&self) -> Result<(), AkitaError> {
        let plan = setup_prefix_compression_plan(&self.id.commitment_profile)
            .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
        self.hint.validate_outer_compression(&plan)
    }

    /// Strip prover-only hint material for verifier metadata.
    #[must_use]
    pub fn verifier_slot(&self) -> SetupPrefixVerifierSlot<F> {
        SetupPrefixVerifierSlot {
            id: self.id.clone(),
            commitment: self.commitment.clone(),
        }
    }
}

/// In-memory registry of prover-ready setup-prefix slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupPrefixProverRegistry<F: Field> {
    setup_seed: AkitaSetupSeed,
    slots: BTreeMap<SetupPrefixSlotId, SetupPrefixSlot<F>>,
}

impl<F: Field> SetupPrefixProverRegistry<F> {
    #[must_use]
    pub fn new(setup_seed: AkitaSetupSeed) -> Self {
        Self {
            setup_seed,
            slots: BTreeMap::new(),
        }
    }

    /// Public field stream to which every committed prefix belongs.
    #[must_use]
    pub fn setup_seed(&self) -> &AkitaSetupSeed {
        &self.setup_seed
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    #[must_use]
    pub fn get(&self, id: &SetupPrefixSlotId) -> Option<&SetupPrefixSlot<F>> {
        self.slots.get(id)
    }

    pub fn insert(&mut self, slot: SetupPrefixSlot<F>) -> Result<(), AkitaError>
    where
        F: Valid,
    {
        slot.check()
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        slot.validate_compression_hint()?;
        if self.slots.contains_key(&slot.id) {
            return Err(AkitaError::InvalidSetup(
                "duplicate setup prefix slot id".to_string(),
            ));
        }
        self.slots.insert(slot.id.clone(), slot);
        Ok(())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&SetupPrefixSlotId, &SetupPrefixSlot<F>)> {
        self.slots.iter()
    }

    #[must_use]
    pub fn verifier_slots(&self) -> Vec<SetupPrefixVerifierSlot<F>> {
        self.slots
            .values()
            .map(SetupPrefixSlot::verifier_slot)
            .collect()
    }
}

impl<F: Field + Valid> Valid for SetupPrefixProverRegistry<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.setup_seed.check()?;
        if self.slots.len() > MAX_SETUP_PREFIX_SLOTS {
            return Err(SerializationError::LengthLimitExceeded {
                len: u64::try_from(self.slots.len()).unwrap_or(u64::MAX),
                max: MAX_SETUP_PREFIX_SLOTS,
            });
        }
        for (id, slot) in &self.slots {
            if id != &slot.id {
                return Err(SerializationError::InvalidData(
                    "setup prefix prover registry key does not match slot id".to_string(),
                ));
            }
            slot.check()?;
        }
        Ok(())
    }
}

impl<F: Field + AkitaSerialize> AkitaSerialize for SetupPrefixProverRegistry<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.setup_seed.serialize_with_mode(&mut writer, compress)?;
        self.slots
            .len()
            .serialize_with_mode(&mut writer, compress)?;
        for slot in self.slots.values() {
            slot.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.setup_seed.serialized_size(compress)
            + self.slots.len().serialized_size(compress)
            + self
                .slots
                .values()
                .map(|slot| slot.serialized_size(compress))
                .sum::<usize>()
    }
}

impl<F> AkitaDeserialize for SetupPrefixProverRegistry<F>
where
    F: Field + Valid + AkitaDeserialize<Context = ()>,
{
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let setup_seed =
            AkitaSetupSeed::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let slot_count =
            read_limited_usize(&mut reader, compress, validate, MAX_SETUP_PREFIX_SLOTS)?;
        let mut out = Self::new(setup_seed);
        for _ in 0..slot_count {
            let slot =
                SetupPrefixSlot::deserialize_with_mode(&mut reader, compress, validate, &())?;
            out.insert(slot)
                .map_err(|err| SerializationError::InvalidData(err.to_string()))?;
        }
        if validate == Validate::Yes {
            out.check()?;
        }
        Ok(out)
    }
}

#[cfg(test)]
#[path = "setup_prefix_tests.rs"]
mod tests;
