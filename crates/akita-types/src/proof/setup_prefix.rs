//! Setup-prefix commitment artifacts for setup-claim offloading (slice 02B).
//!
//! This module defines preprocessing metadata for actual power-of-two flat
//! coefficient prefixes of the shared setup vector `S`. It does not run a setup
//! product sumcheck or change proof semantics.

use crate::layout::setup_prefix_slots::{
    padded_setup_prefix_len, setup_prefix_compression_plan, SetupPrefixSlotId,
};
use crate::proof::{RingVec, MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS};
use crate::{AkitaSetupSeed, CommittedGroupParams};
use akita_error::AkitaError;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use jolt_field::Field;
use std::collections::BTreeMap;
use std::io::{Read, Write};

const MAX_SETUP_PREFIX_SLOTS: usize = 4096;

/// Public commitment half of a setup-prefix slot, stored without `D` const generics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupPrefixPublicCommitment<F: Field> {
    /// Commitment rows in flattened ring-coefficient form.
    pub rows: Vec<RingVec<F>>,
}

impl<F: Field + Valid> Valid for SetupPrefixPublicCommitment<F> {
    fn check(&self) -> Result<(), SerializationError> {
        if self.rows.is_empty() {
            return Err(SerializationError::InvalidData(
                "setup prefix commitment must contain at least one row".to_string(),
            ));
        }
        let mut total_coeffs = 0usize;
        for row in &self.rows {
            if row.coeff_len() == 0 {
                return Err(SerializationError::InvalidData(
                    "setup prefix commitment rows must be non-empty".to_string(),
                ));
            }
            total_coeffs = total_coeffs.checked_add(row.coeff_len()).ok_or_else(|| {
                SerializationError::InvalidData(
                    "setup prefix commitment coefficient count overflow".to_string(),
                )
            })?;
            row.check()?;
        }
        if total_coeffs > MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS {
            return Err(SerializationError::LengthLimitExceeded {
                len: u64::try_from(total_coeffs).unwrap_or(u64::MAX),
                max: MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS,
            });
        }
        Ok(())
    }
}

impl<F: Field + AkitaSerialize> AkitaSerialize for SetupPrefixPublicCommitment<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.rows.len().serialize_with_mode(&mut writer, compress)?;
        for row in &self.rows {
            row.coeff_len().serialize_with_mode(&mut writer, compress)?;
            row.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.rows.len().serialized_size(compress)
            + self
                .rows
                .iter()
                .map(|row| {
                    row.coeff_len().serialized_size(compress) + row.serialized_size(compress)
                })
                .sum::<usize>()
    }
}

impl<F> AkitaDeserialize for SetupPrefixPublicCommitment<F>
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
        let row_count = read_limited_usize(
            &mut reader,
            compress,
            validate,
            MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS,
        )?;
        let mut rows = Vec::new();
        crate::wire_limits::reserve_shape_len(&mut rows, row_count)?;
        let mut total_coeffs = 0usize;
        for _ in 0..row_count {
            let coeff_count = read_limited_usize(
                &mut reader,
                compress,
                validate,
                MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS,
            )?;
            if coeff_count == 0 {
                return Err(SerializationError::InvalidData(
                    "setup prefix commitment rows must be non-empty".to_string(),
                ));
            }
            total_coeffs = total_coeffs.checked_add(coeff_count).ok_or_else(|| {
                SerializationError::InvalidData(
                    "setup prefix commitment coefficient count overflow".to_string(),
                )
            })?;
            if total_coeffs > MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS {
                return Err(SerializationError::LengthLimitExceeded {
                    len: u64::try_from(total_coeffs).unwrap_or(u64::MAX),
                    max: MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS,
                });
            }
            rows.push(RingVec::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &coeff_count,
            )?);
        }
        let out = Self { rows };
        if validate == Validate::Yes {
            out.check()?;
        }
        Ok(out)
    }
}

/// Verifier-visible metadata for one setup-prefix slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupPrefixVerifierSlot<F: Field> {
    pub id: SetupPrefixSlotId,
    pub commitment: SetupPrefixPublicCommitment<F>,
}

impl<F: Field + Valid> Valid for SetupPrefixVerifierSlot<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.id.check()?;
        self.commitment.check()?;
        let expected_payload_coefficients =
            setup_prefix_compression_plan(&self.id.commitment_profile)?.terminal_coefficients();
        if self.commitment.rows.len() != 1 {
            return Err(SerializationError::InvalidData(
                "setup prefix commitment must contain one compressed payload".into(),
            ));
        }
        for row in &self.commitment.rows {
            if row.coeff_len() != expected_payload_coefficients {
                return Err(SerializationError::InvalidData(format!(
                    "setup prefix commitment row has {} coefficients, expected {}",
                    row.coeff_len(),
                    expected_payload_coefficients
                )));
            }
        }
        Ok(())
    }
}

impl<F: Field + AkitaSerialize> AkitaSerialize for SetupPrefixVerifierSlot<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.id.serialize_with_mode(&mut writer, compress)?;
        self.commitment.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.id.serialized_size(compress) + self.commitment.serialized_size(compress)
    }
}

impl<F> AkitaDeserialize for SetupPrefixVerifierSlot<F>
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
        let out = Self { id, commitment };
        if validate == Validate::Yes {
            out.check()?;
        }
        Ok(out)
    }
}

/// In-memory registry of verifier-visible setup-prefix slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupPrefixVerifierRegistry<F: Field> {
    setup_seed: AkitaSetupSeed,
    slots: BTreeMap<SetupPrefixSlotId, SetupPrefixVerifierSlot<F>>,
}

impl<F: Field> SetupPrefixVerifierRegistry<F> {
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
    pub fn get(&self, id: &SetupPrefixSlotId) -> Option<&SetupPrefixVerifierSlot<F>> {
        self.slots.get(id)
    }

    pub fn insert(&mut self, slot: SetupPrefixVerifierSlot<F>) -> Result<(), AkitaError>
    where
        F: Valid,
    {
        slot.check()
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        if self.slots.contains_key(&slot.id) {
            return Err(AkitaError::InvalidSetup(
                "duplicate setup prefix slot id".to_string(),
            ));
        }
        self.slots.insert(slot.id.clone(), slot);
        Ok(())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&SetupPrefixSlotId, &SetupPrefixVerifierSlot<F>)> {
        self.slots.iter()
    }
}

impl<F: Field + Valid> Valid for SetupPrefixVerifierRegistry<F> {
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
                    "setup prefix verifier registry key does not match slot id".to_string(),
                ));
            }
            slot.check()?;
        }
        Ok(())
    }
}

impl<F: Field + AkitaSerialize> AkitaSerialize for SetupPrefixVerifierRegistry<F> {
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

impl<F> AkitaDeserialize for SetupPrefixVerifierRegistry<F>
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
            let slot = SetupPrefixVerifierSlot::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?;
            out.insert(slot)
                .map_err(|err| SerializationError::InvalidData(err.to_string()))?;
        }
        if validate == Validate::Yes {
            out.check()?;
        }
        Ok(out)
    }
}

/// Validate that a selected setup-prefix slot covers one setup-product footprint.
///
/// This centralizes the checks shared by prover and verifier: full-prefix
/// length, planned prefix commitment parameters, selected slot identity, active
/// source support, and the producer-ring evaluation length used for setup MLEs.
/// The slot's commitment dimension is independent of that producer view.
///
/// `shared_matrix_field_elements` is `Some` when the full source prefix must
/// be resident in the shared matrix, as in the prover. It is `None` when the
/// source is represented by the registered setup-prefix commitment, as in the verifier.
/// In both cases the slot's active support and full-prefix lengths are checked.
pub fn setup_prefix_coverage_eval_len(
    shared_matrix_field_elements: Option<usize>,
    selected_slot_id: &SetupPrefixSlotId,
    level_params: &CommittedGroupParams,
    natural_field_len: usize,
    source_ring_dimension: usize,
    coverage_error: &'static str,
) -> Result<usize, AkitaError> {
    let Some(template) = &level_params.setup_prefix() else {
        return Err(AkitaError::InvalidSetup(
            "Stage 3 requires a selected setup-prefix slot".to_string(),
        ));
    };
    template.validate()?;
    selected_slot_id
        .commitment_profile
        .validate_setup_prefix_geometry(selected_slot_id.natural_len)?;
    let template_slot_id = template.slot_id().ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "{coverage_error}: planned setup-prefix template is not a prefix group"
        ))
    })?;
    if selected_slot_id != &template_slot_id {
        return Err(AkitaError::InvalidSetup(format!(
            "{coverage_error}: selected setup-prefix slot id does not match planned slot"
        )));
    }
    let n_prefix = padded_setup_prefix_len(natural_field_len);
    if let Some(shared_matrix_field_elements) = shared_matrix_field_elements {
        if n_prefix > shared_matrix_field_elements {
            return Err(AkitaError::InvalidSetup(
                "setup prefix request exceeds shared matrix capacity".to_string(),
            ));
        }
    }
    let template_n_prefix = template.n_prefix()?;
    if template_slot_id.natural_len != natural_field_len || template_n_prefix != n_prefix {
        return Err(AkitaError::InvalidSetup(format!(
            "{coverage_error}: planned natural/full-prefix lengths are {}/{template_n_prefix}, \
             active lengths are {natural_field_len}/{n_prefix}",
            template_slot_id.natural_len,
        )));
    }

    if source_ring_dimension == 0 || !template_n_prefix.is_multiple_of(source_ring_dimension) {
        return Err(AkitaError::InvalidSetup(
            "setup prefix full length must be divisible by the producer ring dimension".to_string(),
        ));
    }
    let setup_eval_len = template_n_prefix / source_ring_dimension;
    Ok(setup_eval_len)
}

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

#[cfg(test)]
#[path = "setup_prefix_tests.rs"]
mod tests;
