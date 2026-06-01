//! Setup-prefix commitment artifacts for setup-claim offloading (slice 02B).
//!
//! This module defines the preprocessing metadata for power-of-two flat
//! coefficient prefixes of the shared setup vector `S`. It does not run a setup
//! product sumcheck or change proof semantics.

use crate::instance_descriptor::DescriptorDigest;
use crate::proof::{
    setup::MAX_SETUP_MATRIX_FIELD_ELEMENTS, AkitaCommitmentHint, FlatRingVec, RingCommitment,
};
use crate::{ClaimIncidenceSummary, LevelParams};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, FieldCore};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use std::collections::BTreeMap;
use std::io::{Read, Write};

const MAX_SETUP_PREFIX_SLOTS: usize = 4096;

/// Ring dimension used when delegating setup claims to a flat coefficient prefix.
pub const SETUP_OFFLOAD_D_SETUP: usize = 32;

/// Minimum flat coefficient prefix length eligible for setup delegation.
pub const SETUP_OFFLOAD_N_MIN: usize = 1 << 23;

/// Identity for one committed setup-prefix slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SetupPrefixSlotId {
    /// Digest of the deterministic setup seed / layout identity.
    pub setup_seed_digest: DescriptorDigest,
    /// Coefficient-axis ring dimension for the delegated prefix object.
    pub d_setup: usize,
    /// Padded flat coefficient length committed for this slot.
    pub n_prefix: usize,
    /// Digest of the commitment parameters used to build the slot.
    pub level_params_digest: DescriptorDigest,
}

impl Valid for SetupPrefixSlotId {
    fn check(&self) -> Result<(), SerializationError> {
        if self.d_setup == 0 {
            return Err(SerializationError::InvalidData(
                "setup prefix slot d_setup must be non-zero".to_string(),
            ));
        }
        if self.n_prefix == 0 || !self.n_prefix.is_power_of_two() {
            return Err(SerializationError::InvalidData(
                "setup prefix slot n_prefix must be a non-zero power of two".to_string(),
            ));
        }
        if !self.n_prefix.is_multiple_of(self.d_setup) {
            return Err(SerializationError::InvalidData(
                "setup prefix slot n_prefix must be a multiple of d_setup".to_string(),
            ));
        }
        Ok(())
    }
}

impl AkitaSerialize for SetupPrefixSlotId {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        writer.write_all(&self.setup_seed_digest)?;
        self.d_setup.serialize_with_mode(&mut writer, compress)?;
        self.n_prefix.serialize_with_mode(&mut writer, compress)?;
        writer.write_all(&self.level_params_digest)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.setup_seed_digest.len()
            + self.d_setup.serialized_size(compress)
            + self.n_prefix.serialized_size(compress)
            + self.level_params_digest.len()
    }
}

impl AkitaDeserialize for SetupPrefixSlotId {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let mut setup_seed_digest = [0u8; 32];
        reader.read_exact(&mut setup_seed_digest)?;
        let d_setup = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let n_prefix = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let mut level_params_digest = [0u8; 32];
        reader.read_exact(&mut level_params_digest)?;
        let out = Self {
            setup_seed_digest,
            d_setup,
            n_prefix,
            level_params_digest,
        };
        if validate == Validate::Yes {
            out.check()?;
        }
        Ok(out)
    }
}

/// Policy for which prefix slots preprocessing should populate.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SetupPrefixPopulatePolicy {
    /// Do not generate setup-prefix commitments.
    #[default]
    Disabled,
    /// Generate every power-of-two prefix in `[n_min, n_max]`.
    FullLadder {
        /// Minimum prefix length (inclusive).
        n_min: usize,
        /// Maximum prefix length (inclusive).
        n_max: usize,
    },
    /// Generate only the listed padded prefix lengths.
    SelectedSlots(Vec<usize>),
}

/// Behavior when a requested prefix slot is absent at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingSetupPrefixSlotPolicy {
    /// Fail with a setup/policy error.
    #[default]
    StrictError,
    /// Prover-side convenience: create and persist the missing slot.
    GenerateAndPersist,
    /// Skip delegation and keep the direct setup scan.
    DirectFallback,
}

/// Public commitment half of a setup-prefix slot, stored without `D` const generics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupPrefixPublicCommitment<F: FieldCore> {
    /// Commitment rows in flattened ring-coefficient form.
    pub rows: Vec<FlatRingVec<F>>,
}

impl<F: FieldCore + Valid> Valid for SetupPrefixPublicCommitment<F> {
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
        if total_coeffs > MAX_SETUP_MATRIX_FIELD_ELEMENTS {
            return Err(SerializationError::LengthLimitExceeded {
                len: u64::try_from(total_coeffs).unwrap_or(u64::MAX),
                max: MAX_SETUP_MATRIX_FIELD_ELEMENTS,
            });
        }
        Ok(())
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for SetupPrefixPublicCommitment<F> {
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
    F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
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
            MAX_SETUP_MATRIX_FIELD_ELEMENTS,
        )?;
        let mut rows = Vec::new();
        super::reserve_shape_len(&mut rows, row_count)?;
        let mut total_coeffs = 0usize;
        for _ in 0..row_count {
            let coeff_count = read_limited_usize(
                &mut reader,
                compress,
                validate,
                MAX_SETUP_MATRIX_FIELD_ELEMENTS,
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
            if total_coeffs > MAX_SETUP_MATRIX_FIELD_ELEMENTS {
                return Err(SerializationError::LengthLimitExceeded {
                    len: u64::try_from(total_coeffs).unwrap_or(u64::MAX),
                    max: MAX_SETUP_MATRIX_FIELD_ELEMENTS,
                });
            }
            rows.push(FlatRingVec::deserialize_with_mode(
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

impl<F: FieldCore, const D: usize> From<RingCommitment<F, D>> for SetupPrefixPublicCommitment<F> {
    fn from(commitment: RingCommitment<F, D>) -> Self {
        Self {
            rows: commitment
                .u
                .into_iter()
                .map(|row| FlatRingVec::from_coeffs(row.coeffs.to_vec()))
                .collect(),
        }
    }
}

impl<F: FieldCore, const D: usize> TryFrom<&SetupPrefixPublicCommitment<F>>
    for RingCommitment<F, D>
{
    type Error = AkitaError;

    fn try_from(commitment: &SetupPrefixPublicCommitment<F>) -> Result<Self, AkitaError> {
        let u = commitment
            .rows
            .iter()
            .map(|row| {
                if row.coeffs().len() != D {
                    return Err(AkitaError::InvalidSetup(format!(
                        "setup prefix commitment row has {} coefficients, expected {D}",
                        row.coeffs().len()
                    )));
                }
                let mut coeffs = [F::zero(); D];
                for (dst, src) in coeffs.iter_mut().zip(row.coeffs()) {
                    *dst = *src;
                }
                Ok(CyclotomicRing::from_coefficients(coeffs))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(RingCommitment { u })
    }
}

/// Verifier-visible metadata for one setup-prefix slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupPrefixVerifierSlot<F: FieldCore> {
    pub id: SetupPrefixSlotId,
    pub natural_len: usize,
    pub padded_len: usize,
    pub commitment: SetupPrefixPublicCommitment<F>,
}

impl<F: FieldCore + Valid> Valid for SetupPrefixVerifierSlot<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.id.check()?;
        if self.natural_len == 0 || self.natural_len > self.padded_len {
            return Err(SerializationError::InvalidData(
                "setup prefix verifier slot natural_len must be in 1..=padded_len".to_string(),
            ));
        }
        if self.padded_len != self.id.n_prefix {
            return Err(SerializationError::InvalidData(
                "setup prefix verifier slot padded_len must match slot id".to_string(),
            ));
        }
        self.commitment.check()?;
        for row in &self.commitment.rows {
            if row.coeff_len() != self.id.d_setup {
                return Err(SerializationError::InvalidData(format!(
                    "setup prefix commitment row has {} coefficients, expected {}",
                    row.coeff_len(),
                    self.id.d_setup
                )));
            }
        }
        Ok(())
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for SetupPrefixVerifierSlot<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.id.serialize_with_mode(&mut writer, compress)?;
        self.natural_len
            .serialize_with_mode(&mut writer, compress)?;
        self.padded_len.serialize_with_mode(&mut writer, compress)?;
        self.commitment.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.id.serialized_size(compress)
            + self.natural_len.serialized_size(compress)
            + self.padded_len.serialized_size(compress)
            + self.commitment.serialized_size(compress)
    }
}

impl<F> AkitaDeserialize for SetupPrefixVerifierSlot<F>
where
    F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
{
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let id = SetupPrefixSlotId::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let natural_len = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let padded_len = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let commitment = SetupPrefixPublicCommitment::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &(),
        )?;
        let out = Self {
            id,
            natural_len,
            padded_len,
            commitment,
        };
        if validate == Validate::Yes {
            out.check()?;
        }
        Ok(out)
    }
}

/// Prover-ready metadata for one setup-prefix slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupPrefixSlot<F: FieldCore, const D: usize> {
    pub id: SetupPrefixSlotId,
    pub natural_len: usize,
    pub padded_len: usize,
    pub commitment: RingCommitment<F, D>,
    pub hint: AkitaCommitmentHint<F, D>,
}

impl<F: FieldCore + Valid, const D: usize> Valid for SetupPrefixSlot<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        self.id.check()?;
        if self.id.d_setup != D {
            return Err(SerializationError::InvalidData(format!(
                "setup prefix prover slot d_setup {} does not match D={D}",
                self.id.d_setup
            )));
        }
        if self.natural_len == 0 || self.natural_len > self.padded_len {
            return Err(SerializationError::InvalidData(
                "setup prefix prover slot natural_len must be in 1..=padded_len".to_string(),
            ));
        }
        if self.padded_len != self.id.n_prefix {
            return Err(SerializationError::InvalidData(
                "setup prefix prover slot padded_len must match slot id".to_string(),
            ));
        }
        self.commitment.check()?;
        self.hint.check()
    }
}

impl<F: FieldCore + AkitaSerialize, const D: usize> AkitaSerialize for SetupPrefixSlot<F, D> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.id.serialize_with_mode(&mut writer, compress)?;
        self.natural_len
            .serialize_with_mode(&mut writer, compress)?;
        self.padded_len.serialize_with_mode(&mut writer, compress)?;
        self.commitment.serialize_with_mode(&mut writer, compress)?;
        self.hint.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.id.serialized_size(compress)
            + self.natural_len.serialized_size(compress)
            + self.padded_len.serialized_size(compress)
            + self.commitment.serialized_size(compress)
            + self.hint.serialized_size(compress)
    }
}

impl<F, const D: usize> AkitaDeserialize for SetupPrefixSlot<F, D>
where
    F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
{
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let id = SetupPrefixSlotId::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let natural_len = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let padded_len = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let commitment =
            RingCommitment::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let hint =
            AkitaCommitmentHint::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self {
            id,
            natural_len,
            padded_len,
            commitment,
            hint,
        };
        if validate == Validate::Yes {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore, const D: usize> SetupPrefixSlot<F, D> {
    /// Strip prover-only hint material for verifier metadata.
    #[must_use]
    pub fn verifier_slot(&self) -> SetupPrefixVerifierSlot<F> {
        SetupPrefixVerifierSlot {
            id: self.id,
            natural_len: self.natural_len,
            padded_len: self.padded_len,
            commitment: self.commitment.clone().into(),
        }
    }
}

/// In-memory registry of prover-ready setup-prefix slots.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SetupPrefixProverRegistry<F: FieldCore, const D: usize> {
    slots: BTreeMap<SetupPrefixSlotId, SetupPrefixSlot<F, D>>,
}

impl<F: FieldCore, const D: usize> SetupPrefixProverRegistry<F, D> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
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
    pub fn get(&self, id: &SetupPrefixSlotId) -> Option<&SetupPrefixSlot<F, D>> {
        self.slots.get(id)
    }

    pub fn insert(&mut self, slot: SetupPrefixSlot<F, D>) -> Result<(), AkitaError> {
        if self.slots.contains_key(&slot.id) {
            return Err(AkitaError::InvalidSetup(
                "duplicate setup prefix slot id".to_string(),
            ));
        }
        self.slots.insert(slot.id, slot);
        Ok(())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&SetupPrefixSlotId, &SetupPrefixSlot<F, D>)> {
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

impl<F: FieldCore + Valid, const D: usize> Valid for SetupPrefixProverRegistry<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
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

impl<F: FieldCore + AkitaSerialize, const D: usize> AkitaSerialize
    for SetupPrefixProverRegistry<F, D>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.slots
            .len()
            .serialize_with_mode(&mut writer, compress)?;
        for slot in self.slots.values() {
            slot.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.slots.len().serialized_size(compress)
            + self
                .slots
                .values()
                .map(|slot| slot.serialized_size(compress))
                .sum::<usize>()
    }
}

impl<F, const D: usize> AkitaDeserialize for SetupPrefixProverRegistry<F, D>
where
    F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
{
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let slot_count =
            read_limited_usize(&mut reader, compress, validate, MAX_SETUP_PREFIX_SLOTS)?;
        let mut out = Self::new();
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

/// In-memory registry of verifier-visible setup-prefix slots.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SetupPrefixVerifierRegistry<F: FieldCore> {
    slots: BTreeMap<SetupPrefixSlotId, SetupPrefixVerifierSlot<F>>,
}

impl<F: FieldCore> SetupPrefixVerifierRegistry<F> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
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

    pub fn insert(&mut self, slot: SetupPrefixVerifierSlot<F>) -> Result<(), AkitaError> {
        if self.slots.contains_key(&slot.id) {
            return Err(AkitaError::InvalidSetup(
                "duplicate setup prefix slot id".to_string(),
            ));
        }
        self.slots.insert(slot.id, slot);
        Ok(())
    }

    pub fn replace_from_prover_registry<const D: usize>(
        &mut self,
        prover_registry: &SetupPrefixProverRegistry<F, D>,
    ) -> Result<(), AkitaError> {
        self.slots.clear();
        for slot in prover_registry.verifier_slots() {
            self.insert(slot)?;
        }
        Ok(())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&SetupPrefixSlotId, &SetupPrefixVerifierSlot<F>)> {
        self.slots.iter()
    }
}

impl<F: FieldCore + Valid> Valid for SetupPrefixVerifierRegistry<F> {
    fn check(&self) -> Result<(), SerializationError> {
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

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for SetupPrefixVerifierRegistry<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.slots
            .len()
            .serialize_with_mode(&mut writer, compress)?;
        for slot in self.slots.values() {
            slot.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.slots.len().serialized_size(compress)
            + self
                .slots
                .values()
                .map(|slot| slot.serialized_size(compress))
                .sum::<usize>()
    }
}

impl<F> AkitaDeserialize for SetupPrefixVerifierRegistry<F>
where
    F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
{
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let slot_count =
            read_limited_usize(&mut reader, compress, validate, MAX_SETUP_PREFIX_SLOTS)?;
        let mut out = Self::new();
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

/// Why setup delegation fell back to the direct scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetupPrefixDirectReason {
    BelowMinimum {
        n_prefix: usize,
        n_min: usize,
    },
    DSetupMismatch {
        ring_dimension: usize,
        d_setup: usize,
    },
    MissingSlot(SetupPrefixSlotId),
}

/// Inputs needed to select a setup-prefix slot for one active shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SetupPrefixSelectionRequest {
    pub d_setup: usize,
    pub natural_field_len: usize,
    pub level_params_digest: DescriptorDigest,
}

/// Result of attempting to select a setup-prefix slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetupPrefixSelectionOutcome<F: FieldCore, const D: usize> {
    DirectScan { reason: SetupPrefixDirectReason },
    Selected(SetupPrefixSlot<F, D>),
}

/// Return the packed role widths `(W_A, W_B, W_D)` for one active level shape.
pub fn active_setup_role_widths(
    level_params: &LevelParams,
    incidence: &ClaimIncidenceSummary,
) -> Result<(usize, usize, usize), AkitaError> {
    let w_a = level_params
        .block_len
        .checked_mul(level_params.num_digits_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("A setup width overflow".to_string()))?;
    let num_claims = incidence.num_claims();
    let max_group_poly_count = incidence
        .num_polys_per_point()
        .iter()
        .copied()
        .max()
        .ok_or_else(|| AkitaError::InvalidSetup("empty claim incidence".to_string()))?;
    let w_d = num_claims
        .checked_mul(level_params.num_blocks)
        .and_then(|n| n.checked_mul(level_params.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("D setup width overflow".to_string()))?;
    let w_b = max_group_poly_count
        .checked_mul(level_params.a_key.row_len())
        .and_then(|n| n.checked_mul(level_params.num_blocks))
        .and_then(|n| n.checked_mul(level_params.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("B setup width overflow".to_string()))?;
    Ok((w_a, w_b, w_d))
}

/// Active packed setup footprint in ring slots: `max(n_a W_A, n_b W_B, n_d W_D)`.
pub fn active_setup_ring_slots(
    level_params: &LevelParams,
    incidence: &ClaimIncidenceSummary,
) -> Result<usize, AkitaError> {
    let (w_a, w_b, w_d) = active_setup_role_widths(level_params, incidence)?;
    let a_slots = level_params
        .a_key
        .row_len()
        .checked_mul(w_a)
        .ok_or_else(|| AkitaError::InvalidSetup("A setup footprint overflow".to_string()))?;
    let b_slots = level_params
        .b_key
        .row_len()
        .checked_mul(w_b)
        .ok_or_else(|| AkitaError::InvalidSetup("B setup footprint overflow".to_string()))?;
    let d_slots = level_params
        .d_key
        .row_len()
        .checked_mul(w_d)
        .ok_or_else(|| AkitaError::InvalidSetup("D setup footprint overflow".to_string()))?;
    Ok(a_slots.max(b_slots).max(d_slots))
}

/// Active flat coefficient count `N_active^F = D_setup * N_active^R`.
pub fn active_setup_field_len(
    level_params: &LevelParams,
    incidence: &ClaimIncidenceSummary,
    d_setup: usize,
) -> Result<usize, AkitaError> {
    active_setup_ring_slots(level_params, incidence)?
        .checked_mul(d_setup)
        .ok_or_else(|| AkitaError::InvalidSetup("active setup field length overflow".to_string()))
}

/// Smallest power-of-two flat prefix length covering `natural_field_len`.
#[must_use]
pub fn padded_setup_prefix_len(natural_field_len: usize) -> usize {
    natural_field_len.max(1).next_power_of_two()
}

/// Return the eligible padded prefix length, if any.
#[must_use]
pub fn select_prefix_len(natural_field_len: usize, n_min: usize) -> Option<usize> {
    let n_prefix = padded_setup_prefix_len(natural_field_len);
    (n_prefix >= n_min).then_some(n_prefix)
}

/// Ring-slot count for a flat prefix of `n_prefix` field coefficients at `d_setup`.
pub fn setup_prefix_commit_ring_slots(
    n_prefix: usize,
    d_setup: usize,
) -> Result<usize, AkitaError> {
    if d_setup == 0 || !n_prefix.is_multiple_of(d_setup) {
        return Err(AkitaError::InvalidSetup(
            "setup prefix length must be a positive multiple of d_setup".to_string(),
        ));
    }
    Ok(n_prefix / d_setup)
}

/// Whether `level_params` witness shape matches one committed prefix length.
#[must_use]
pub fn level_params_matches_setup_prefix(
    level_params: &LevelParams,
    n_prefix: usize,
    d_setup: usize,
) -> bool {
    setup_prefix_commit_ring_slots(n_prefix, d_setup).is_ok_and(|ring_slots| {
        level_params
            .num_blocks
            .checked_mul(level_params.block_len)
            .is_some_and(|witness| witness == ring_slots)
    })
}

/// Keep only prefix lengths compatible with the supplied commitment parameters.
#[must_use]
pub fn filter_prefix_lengths_for_level_params(
    lengths: &[usize],
    level_params: &LevelParams,
    d_setup: usize,
) -> Vec<usize> {
    lengths
        .iter()
        .copied()
        .filter(|&n_prefix| level_params_matches_setup_prefix(level_params, n_prefix, d_setup))
        .collect()
}

/// Enumerate padded prefix lengths requested by a populate policy.
pub fn prefix_lengths_for_policy(
    policy: &SetupPrefixPopulatePolicy,
) -> Result<Vec<usize>, AkitaError> {
    match policy {
        SetupPrefixPopulatePolicy::Disabled => Ok(Vec::new()),
        SetupPrefixPopulatePolicy::FullLadder { n_min, n_max } => {
            if *n_min == 0 || !n_min.is_power_of_two() {
                return Err(AkitaError::InvalidSetup(
                    "setup prefix ladder n_min must be a non-zero power of two".to_string(),
                ));
            }
            if *n_max < *n_min || !n_max.is_power_of_two() {
                return Err(AkitaError::InvalidSetup(
                    "setup prefix ladder n_max must be a power of two >= n_min".to_string(),
                ));
            }
            let mut lengths = Vec::new();
            let mut current = *n_min;
            while current <= *n_max {
                lengths.push(current);
                current = current.checked_mul(2).ok_or_else(|| {
                    AkitaError::InvalidSetup("prefix ladder overflow".to_string())
                })?;
            }
            Ok(lengths)
        }
        SetupPrefixPopulatePolicy::SelectedSlots(lengths) => {
            for &len in lengths {
                if len == 0 || !len.is_power_of_two() {
                    return Err(AkitaError::InvalidSetup(format!(
                        "selected setup prefix length {len} must be a non-zero power of two"
                    )));
                }
            }
            Ok(lengths.clone())
        }
    }
}

/// Build the slot id for one committed setup prefix.
pub fn setup_prefix_slot_id(
    setup_seed_digest: DescriptorDigest,
    d_setup: usize,
    n_prefix: usize,
    level_params_digest: DescriptorDigest,
) -> SetupPrefixSlotId {
    SetupPrefixSlotId {
        setup_seed_digest,
        d_setup,
        n_prefix,
        level_params_digest,
    }
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

/// Select the tightest populated prover slot for one active shape.
pub fn select_setup_prefix_slot<F: FieldCore, const D: usize>(
    registry: &SetupPrefixProverRegistry<F, D>,
    setup_seed_digest: DescriptorDigest,
    ring_dimension: usize,
    request: SetupPrefixSelectionRequest,
    n_min: usize,
) -> SetupPrefixSelectionOutcome<F, D> {
    if ring_dimension != request.d_setup {
        return SetupPrefixSelectionOutcome::DirectScan {
            reason: SetupPrefixDirectReason::DSetupMismatch {
                ring_dimension,
                d_setup: request.d_setup,
            },
        };
    }
    let Some(n_prefix) = select_prefix_len(request.natural_field_len, n_min) else {
        return SetupPrefixSelectionOutcome::DirectScan {
            reason: SetupPrefixDirectReason::BelowMinimum {
                n_prefix: padded_setup_prefix_len(request.natural_field_len),
                n_min,
            },
        };
    };
    let id = setup_prefix_slot_id(
        setup_seed_digest,
        request.d_setup,
        n_prefix,
        request.level_params_digest,
    );
    match registry.get(&id) {
        Some(slot) => SetupPrefixSelectionOutcome::Selected(slot.clone()),
        None => SetupPrefixSelectionOutcome::DirectScan {
            reason: SetupPrefixDirectReason::MissingSlot(id),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance_descriptor::digest_level_params;
    use crate::{ClaimIncidenceSummary, LevelParams, SisModulusFamily};
    use akita_challenges::SparseChallengeConfig;

    fn sample_level_params() -> LevelParams {
        LevelParams::params_only(
            SisModulusFamily::Q32,
            32,
            3,
            2,
            3,
            2,
            SparseChallengeConfig::Uniform {
                weight: 3,
                nonzero_coeffs: vec![-1, 1],
            },
        )
        .with_decomp(2, 3, 2, 2, 3, 0)
        .expect("sample level params")
    }

    #[test]
    fn active_setup_field_len_matches_packed_role_maximum() {
        let lp = sample_level_params();
        let incidence = ClaimIncidenceSummary::from_point_polys(5, vec![2, 1]).expect("incidence");
        let (w_a, w_b, w_d) = active_setup_role_widths(&lp, &incidence).expect("widths");
        let expected_ring_slots = lp
            .a_key
            .row_len()
            .checked_mul(w_a)
            .unwrap()
            .max(lp.b_key.row_len().checked_mul(w_b).unwrap())
            .max(lp.d_key.row_len().checked_mul(w_d).unwrap());
        assert_eq!(
            active_setup_ring_slots(&lp, &incidence).expect("ring slots"),
            expected_ring_slots
        );
        assert_eq!(
            active_setup_field_len(&lp, &incidence, SETUP_OFFLOAD_D_SETUP).expect("field len"),
            expected_ring_slots * SETUP_OFFLOAD_D_SETUP
        );
    }

    #[test]
    fn select_prefix_len_honors_n_min_gate() {
        assert_eq!(select_prefix_len(10, 17), None);
        assert_eq!(select_prefix_len(10, 16), Some(16));
        assert_eq!(select_prefix_len(100, SETUP_OFFLOAD_N_MIN), None);
        assert_eq!(
            select_prefix_len(SETUP_OFFLOAD_N_MIN, SETUP_OFFLOAD_N_MIN),
            Some(SETUP_OFFLOAD_N_MIN)
        );
    }

    #[test]
    fn prefix_lengths_for_selected_slots_rejects_non_power_of_two() {
        let err = prefix_lengths_for_policy(&SetupPrefixPopulatePolicy::SelectedSlots(vec![12]))
            .expect_err("non power-of-two");
        assert!(err.to_string().contains("power of two"));
    }

    #[test]
    fn select_setup_prefix_slot_reports_below_minimum() {
        use akita_field::Prime32Offset99 as F;

        let registry = SetupPrefixProverRegistry::<F, 32>::new();
        let outcome = select_setup_prefix_slot(
            &registry,
            [7u8; 32],
            32,
            SetupPrefixSelectionRequest {
                d_setup: 32,
                natural_field_len: 100,
                level_params_digest: [1u8; 32],
            },
            SETUP_OFFLOAD_N_MIN,
        );
        match outcome {
            SetupPrefixSelectionOutcome::DirectScan {
                reason: SetupPrefixDirectReason::BelowMinimum { n_min, .. },
            } => assert_eq!(n_min, SETUP_OFFLOAD_N_MIN),
            other => panic!("expected below minimum, got {other:?}"),
        }
    }

    #[test]
    fn filter_prefix_lengths_keeps_only_matching_witness_shape() {
        let lp = sample_level_params();
        let witness_field_len = lp
            .num_blocks
            .checked_mul(lp.block_len)
            .unwrap()
            .checked_mul(SETUP_OFFLOAD_D_SETUP)
            .unwrap();
        let filtered = filter_prefix_lengths_for_level_params(
            &[witness_field_len, witness_field_len * 2],
            &lp,
            SETUP_OFFLOAD_D_SETUP,
        );
        assert_eq!(filtered, vec![witness_field_len]);
        assert!(level_params_matches_setup_prefix(
            &lp,
            witness_field_len,
            SETUP_OFFLOAD_D_SETUP
        ));
    }

    #[test]
    fn select_setup_prefix_slot_reports_missing_slot() {
        use akita_field::Prime32Offset99 as F;

        let registry = SetupPrefixProverRegistry::<F, 32>::new();
        let incidence = ClaimIncidenceSummary::same_point(4, 1).expect("incidence");
        let lp = sample_level_params();
        let natural = active_setup_field_len(&lp, &incidence, 32).expect("natural");
        let outcome = select_setup_prefix_slot(
            &registry,
            [7u8; 32],
            32,
            SetupPrefixSelectionRequest {
                d_setup: 32,
                natural_field_len: natural,
                level_params_digest: digest_level_params(&[lp]),
            },
            1,
        );
        match outcome {
            SetupPrefixSelectionOutcome::DirectScan {
                reason: SetupPrefixDirectReason::MissingSlot(_),
            } => {}
            other => panic!("expected missing slot, got {other:?}"),
        }
    }

    #[test]
    fn prover_registry_duplicate_insert_does_not_replace_existing_slot() {
        use crate::proof::FlatDigitBlocks;
        use akita_algebra::CyclotomicRing;
        use akita_field::Prime32Offset99 as F;

        let id = SetupPrefixSlotId {
            setup_seed_digest: [7u8; 32],
            d_setup: 32,
            n_prefix: 32,
            level_params_digest: [9u8; 32],
        };
        let slot = |natural_len| {
            let decomposed = FlatDigitBlocks::<32>::from_blocks(vec![Vec::new()]);
            let recomposed = vec![Vec::new()];
            #[cfg(feature = "zk")]
            let hint = AkitaCommitmentHint::singleton_with_recomposed_inner_rows(
                decomposed,
                recomposed,
                FlatDigitBlocks::empty(),
            );
            #[cfg(not(feature = "zk"))]
            let hint =
                AkitaCommitmentHint::singleton_with_recomposed_inner_rows(decomposed, recomposed);
            SetupPrefixSlot {
                id,
                natural_len,
                padded_len: id.n_prefix,
                commitment: RingCommitment {
                    u: vec![CyclotomicRing::<F, 32>::zero()],
                },
                hint,
            }
        };

        let mut registry = SetupPrefixProverRegistry::<F, 32>::new();
        registry.insert(slot(1)).expect("first insert");
        registry
            .insert(slot(2))
            .expect_err("duplicate insert must fail");

        assert_eq!(registry.get(&id).expect("stored slot").natural_len, 1);
    }

    #[test]
    fn verifier_registry_duplicate_insert_does_not_replace_existing_slot() {
        use akita_field::Prime32Offset99 as F;

        let id = SetupPrefixSlotId {
            setup_seed_digest: [7u8; 32],
            d_setup: 32,
            n_prefix: 32,
            level_params_digest: [9u8; 32],
        };
        let slot = |natural_len| SetupPrefixVerifierSlot {
            id,
            natural_len,
            padded_len: id.n_prefix,
            commitment: SetupPrefixPublicCommitment {
                rows: vec![FlatRingVec::from_coeffs(vec![F::zero()])],
            },
        };

        let mut registry = SetupPrefixVerifierRegistry::<F>::new();
        registry.insert(slot(1)).expect("first insert");
        registry
            .insert(slot(2))
            .expect_err("duplicate insert must fail");

        assert_eq!(registry.get(&id).expect("stored slot").natural_len, 1);
    }
}
