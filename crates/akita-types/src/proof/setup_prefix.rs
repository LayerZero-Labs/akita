//! Setup-prefix commitment artifacts for setup-claim offloading (slice 02B).
//!
//! This module defines preprocessing metadata for exact flat coefficient
//! prefixes of the shared setup vector `S`, zero-padded to power-of-two
//! commitment domains. It does not run a setup product sumcheck or change proof
//! semantics.

use crate::instance_descriptor::{digest_level_params, setup_seed_digest, DescriptorDigest};
use crate::proof::{
    setup::{AkitaSetupSeed, MAX_SETUP_MATRIX_FIELD_ELEMENTS},
    AkitaCommitmentHint, FlatRingVec, RingCommitment,
};
use crate::LevelParams;
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, FieldCore};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use std::collections::BTreeMap;
use std::io::{Read, Write};

const MAX_SETUP_PREFIX_SLOTS: usize = 4096;

/// Ring dimension used when delegating setup claims to a flat coefficient prefix.
pub const SETUP_OFFLOAD_D_SETUP: usize = 64;

/// Identity for one committed setup-prefix slot.
///
/// `natural_len` distinguishes exact prefixes that share a padded commitment
/// domain, while `n_prefix` binds the power-of-two domain and derived commitment
/// parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SetupPrefixSlotId {
    /// Digest of the deterministic setup seed / layout identity.
    pub setup_seed_digest: DescriptorDigest,
    /// Coefficient-axis ring dimension for the delegated prefix object.
    pub d_setup: usize,
    /// Exact flat coefficient length represented before zero padding.
    pub natural_len: usize,
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
        if self.natural_len == 0 || self.natural_len > self.n_prefix {
            return Err(SerializationError::InvalidData(
                "setup prefix slot natural_len must be in 1..=n_prefix".to_string(),
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
        self.natural_len
            .serialize_with_mode(&mut writer, compress)?;
        self.n_prefix.serialize_with_mode(&mut writer, compress)?;
        writer.write_all(&self.level_params_digest)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.setup_seed_digest.len()
            + self.d_setup.serialized_size(compress)
            + self.natural_len.serialized_size(compress)
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
        let natural_len = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let n_prefix = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let mut level_params_digest = [0u8; 32];
        reader.read_exact(&mut level_params_digest)?;
        let out = Self {
            setup_seed_digest,
            d_setup,
            natural_len,
            n_prefix,
            level_params_digest,
        };
        if validate == Validate::Yes {
            out.check()?;
        }
        Ok(out)
    }
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
        if self.natural_len != self.id.natural_len {
            return Err(SerializationError::InvalidData(
                "setup prefix verifier slot natural_len must match slot id".to_string(),
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
        if self.natural_len != self.id.natural_len {
            return Err(SerializationError::InvalidData(
                "setup prefix prover slot natural_len must match slot id".to_string(),
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

/// Erased setup-prefix slot: commitment/hint stay D-typed; keying uses `id.d_setup`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetupPrefixSlotAny<F: FieldCore> {
    D32(SetupPrefixSlot<F, 32>),
    D64(SetupPrefixSlot<F, 64>),
    D128(SetupPrefixSlot<F, 128>),
    D256(SetupPrefixSlot<F, 256>),
}

impl<F: FieldCore> SetupPrefixSlotAny<F> {
    #[must_use]
    pub fn id(&self) -> &SetupPrefixSlotId {
        match self {
            Self::D32(slot) => &slot.id,
            Self::D64(slot) => &slot.id,
            Self::D128(slot) => &slot.id,
            Self::D256(slot) => &slot.id,
        }
    }

    #[must_use]
    pub fn natural_len(&self) -> usize {
        match self {
            Self::D32(slot) => slot.natural_len,
            Self::D64(slot) => slot.natural_len,
            Self::D128(slot) => slot.natural_len,
            Self::D256(slot) => slot.natural_len,
        }
    }

    #[must_use]
    pub fn padded_len(&self) -> usize {
        match self {
            Self::D32(slot) => slot.padded_len,
            Self::D64(slot) => slot.padded_len,
            Self::D128(slot) => slot.padded_len,
            Self::D256(slot) => slot.padded_len,
        }
    }

    #[must_use]
    pub fn verifier_slot(&self) -> SetupPrefixVerifierSlot<F> {
        match self {
            Self::D32(slot) => slot.verifier_slot(),
            Self::D64(slot) => slot.verifier_slot(),
            Self::D128(slot) => slot.verifier_slot(),
            Self::D256(slot) => slot.verifier_slot(),
        }
    }

    /// Ring dimension of this slot (`id.d_setup`).
    #[must_use]
    pub fn ring_d(&self) -> usize {
        match self {
            Self::D32(_) => 32,
            Self::D64(_) => 64,
            Self::D128(_) => 128,
            Self::D256(_) => 256,
        }
    }

    /// View the slot at compile-time ring degree `D` when it matches [`Self::ring_d`].
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when `D` does not match the stored variant.
    pub fn as_d<const D: usize>(&self) -> Result<&SetupPrefixSlot<F, D>, AkitaError> {
        if self.ring_d() != D {
            return Err(AkitaError::InvalidSetup(format!(
                "setup prefix slot ring_d={} does not match requested D={D}",
                self.ring_d()
            )));
        }
        Ok(match self {
            Self::D32(slot) => {
                let slot = slot as *const SetupPrefixSlot<F, 32> as *const SetupPrefixSlot<F, D>;
                // SAFETY: `ring_d()` check guarantees `D == 32`.
                unsafe { &*slot }
            }
            Self::D64(slot) => {
                let slot = slot as *const SetupPrefixSlot<F, 64> as *const SetupPrefixSlot<F, D>;
                unsafe { &*slot }
            }
            Self::D128(slot) => {
                let slot = slot as *const SetupPrefixSlot<F, 128> as *const SetupPrefixSlot<F, D>;
                unsafe { &*slot }
            }
            Self::D256(slot) => {
                let slot = slot as *const SetupPrefixSlot<F, 256> as *const SetupPrefixSlot<F, D>;
                unsafe { &*slot }
            }
        })
    }
}

impl<F: FieldCore> From<SetupPrefixSlot<F, 32>> for SetupPrefixSlotAny<F> {
    fn from(slot: SetupPrefixSlot<F, 32>) -> Self {
        Self::D32(slot)
    }
}

impl<F: FieldCore> From<SetupPrefixSlot<F, 64>> for SetupPrefixSlotAny<F> {
    fn from(slot: SetupPrefixSlot<F, 64>) -> Self {
        Self::D64(slot)
    }
}

impl<F: FieldCore> From<SetupPrefixSlot<F, 128>> for SetupPrefixSlotAny<F> {
    fn from(slot: SetupPrefixSlot<F, 128>) -> Self {
        Self::D128(slot)
    }
}

impl<F: FieldCore> From<SetupPrefixSlot<F, 256>> for SetupPrefixSlotAny<F> {
    fn from(slot: SetupPrefixSlot<F, 256>) -> Self {
        Self::D256(slot)
    }
}

impl<F: FieldCore + Valid> Valid for SetupPrefixSlotAny<F> {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            Self::D32(slot) => slot.check(),
            Self::D64(slot) => slot.check(),
            Self::D128(slot) => slot.check(),
            Self::D256(slot) => slot.check(),
        }
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for SetupPrefixSlotAny<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::D32(slot) => slot.serialize_with_mode(writer, compress),
            Self::D64(slot) => slot.serialize_with_mode(writer, compress),
            Self::D128(slot) => slot.serialize_with_mode(writer, compress),
            Self::D256(slot) => slot.serialize_with_mode(writer, compress),
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            Self::D32(slot) => slot.serialized_size(compress),
            Self::D64(slot) => slot.serialized_size(compress),
            Self::D128(slot) => slot.serialized_size(compress),
            Self::D256(slot) => slot.serialized_size(compress),
        }
    }
}

impl<F> AkitaDeserialize for SetupPrefixSlotAny<F>
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
        crate::dispatch_ring_dim_result!(id.d_setup, |D| {
            let commitment = RingCommitment::<F, D>::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?;
            let hint = AkitaCommitmentHint::<F, D>::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?;
            let slot = SetupPrefixSlot {
                id,
                natural_len,
                padded_len,
                commitment,
                hint,
            };
            if validate == Validate::Yes {
                slot.check()?;
            }
            Ok(slot.into())
        })
        .map_err(|err| SerializationError::InvalidData(err.to_string()))
    }
}

/// In-memory registry of prover-ready setup-prefix slots (keyed on `SetupPrefixSlotId`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SetupPrefixRegistry<F: FieldCore> {
    slots: BTreeMap<SetupPrefixSlotId, SetupPrefixSlotAny<F>>,
}

impl<F: FieldCore> SetupPrefixRegistry<F> {
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
    pub fn get(&self, id: &SetupPrefixSlotId) -> Option<&SetupPrefixSlotAny<F>> {
        self.slots.get(id)
    }

    pub fn insert(&mut self, slot: SetupPrefixSlotAny<F>) -> Result<(), AkitaError> {
        if self.slots.contains_key(slot.id()) {
            return Err(AkitaError::InvalidSetup(
                "duplicate setup prefix slot id".to_string(),
            ));
        }
        self.slots.insert(*slot.id(), slot);
        Ok(())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&SetupPrefixSlotId, &SetupPrefixSlotAny<F>)> {
        self.slots.iter()
    }

    #[must_use]
    pub fn verifier_slots(&self) -> Vec<SetupPrefixVerifierSlot<F>> {
        self.slots
            .values()
            .map(SetupPrefixSlotAny::verifier_slot)
            .collect()
    }
}

impl<F: FieldCore + Valid> Valid for SetupPrefixRegistry<F> {
    fn check(&self) -> Result<(), SerializationError> {
        if self.slots.len() > MAX_SETUP_PREFIX_SLOTS {
            return Err(SerializationError::LengthLimitExceeded {
                len: u64::try_from(self.slots.len()).unwrap_or(u64::MAX),
                max: MAX_SETUP_PREFIX_SLOTS,
            });
        }
        for (id, slot) in &self.slots {
            if id != slot.id() {
                return Err(SerializationError::InvalidData(
                    "setup prefix prover registry key does not match slot id".to_string(),
                ));
            }
            slot.check()?;
        }
        Ok(())
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for SetupPrefixRegistry<F> {
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

impl<F> AkitaDeserialize for SetupPrefixRegistry<F>
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
                SetupPrefixSlotAny::deserialize_with_mode(&mut reader, compress, validate, &())?;
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

    pub fn replace_from_prover_registry(
        &mut self,
        prover_registry: &SetupPrefixRegistry<F>,
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

/// Smallest power-of-two flat prefix length covering `natural_field_len`.
#[must_use]
pub fn padded_setup_prefix_len(natural_field_len: usize) -> usize {
    natural_field_len.max(1).next_power_of_two()
}

/// Repack `level_params` into a witness shape that commits a flat prefix of
/// `n_prefix` setup coefficients at ring dimension `d_setup`.
pub fn setup_prefix_level_params(
    level_params: &LevelParams,
    n_prefix: usize,
    d_setup: usize,
) -> Result<Option<LevelParams>, AkitaError> {
    let ring_slots = n_prefix.checked_div(d_setup).ok_or_else(|| {
        AkitaError::InvalidSetup("setup prefix length has invalid dimension".to_string())
    })?;
    let mut prefix_params = level_params.clone();
    let mut num_blocks = 1usize;
    while num_blocks <= ring_slots {
        if ring_slots.is_multiple_of(num_blocks) {
            let block_len = ring_slots / num_blocks;
            let inner_width = block_len
                .checked_mul(level_params.num_digits_commit)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("prefix inner width overflow".to_string())
                })?;
            let outer_width = num_blocks
                .checked_mul(level_params.a_key.row_len())
                .and_then(|n| n.checked_mul(level_params.num_digits_open))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("prefix outer width overflow".to_string())
                })?;
            if inner_width <= level_params.a_key.col_len()
                && outer_width <= level_params.b_key.col_len()
            {
                prefix_params.num_blocks = num_blocks;
                prefix_params.m_vars = num_blocks.trailing_zeros() as usize;
                prefix_params.block_len = block_len;
                prefix_params.r_vars = block_len.next_power_of_two().trailing_zeros() as usize;
                return Ok(Some(prefix_params));
            }
        }
        num_blocks = num_blocks
            .checked_mul(2)
            .ok_or_else(|| AkitaError::InvalidSetup("prefix block count overflow".to_string()))?;
    }
    Ok(None)
}

/// Build the slot id for one committed setup prefix.
pub fn setup_prefix_slot_id(
    setup_seed_digest: DescriptorDigest,
    d_setup: usize,
    natural_len: usize,
    n_prefix: usize,
    level_params_digest: DescriptorDigest,
) -> SetupPrefixSlotId {
    SetupPrefixSlotId {
        setup_seed_digest,
        d_setup,
        natural_len,
        n_prefix,
        level_params_digest,
    }
}

/// Select a setup-prefix slot that covers one setup-product footprint.
///
/// This centralizes the derivation shared by prover and verifier: setup seed
/// digest, padded prefix length, prefix commitment parameters, slot id, coverage
/// check, and the ring-slot evaluation length used for setup MLEs.
pub fn select_setup_prefix_slot<'a, Slot, Lookup>(
    setup_seed: &AkitaSetupSeed,
    setup_ring_slots_at_d: usize,
    lookup_slot: Lookup,
    level_params: &LevelParams,
    natural_field_len: usize,
    d_setup: usize,
    coverage_error: &'static str,
) -> Result<Option<(&'a Slot, usize)>, AkitaError>
where
    Slot: ?Sized,
    Lookup: FnOnce(&SetupPrefixSlotId) -> Option<(&'a Slot, usize, usize)>,
{
    let seed_digest = setup_seed_digest(setup_seed)
        .map_err(|err| AkitaError::InvalidSetup(format!("setup seed digest failed: {err}")))?;
    let n_prefix = padded_setup_prefix_len(natural_field_len);
    let setup_field_len = setup_ring_slots_at_d.checked_mul(d_setup).ok_or_else(|| {
        AkitaError::InvalidSetup("setup matrix field length overflow".to_string())
    })?;
    if natural_field_len > setup_field_len {
        return Err(AkitaError::InvalidSetup(
            "setup prefix request exceeds shared matrix capacity".to_string(),
        ));
    }
    if n_prefix > setup_field_len {
        return Ok(None);
    }
    let Some(prefix_params) = setup_prefix_level_params(level_params, n_prefix, d_setup)? else {
        return Ok(None);
    };
    let slot_id = setup_prefix_slot_id(
        seed_digest,
        d_setup,
        natural_field_len,
        n_prefix,
        digest_level_params(std::slice::from_ref(&prefix_params)),
    );
    let Some((slot, slot_natural_len, slot_padded_len)) = lookup_slot(&slot_id) else {
        return Ok(None);
    };
    if slot_natural_len != natural_field_len || slot_padded_len != n_prefix {
        return Err(AkitaError::InvalidSetup(coverage_error.to_string()));
    }
    let setup_eval_len = n_prefix.checked_div(d_setup).ok_or_else(|| {
        AkitaError::InvalidSetup("setup prefix padded length has invalid dimension".to_string())
    })?;
    Ok(Some((slot, setup_eval_len)))
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
mod tests {
    use super::*;
    use crate::{LevelParams, SisModulusFamily};
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
        .with_decomp(2, 3, 2, 2, 3)
        .expect("sample level params")
    }

    #[test]
    fn select_setup_prefix_slot_rejects_coverage_mismatch() {
        use akita_field::Prime32Offset99 as F;

        let level_params = sample_level_params();
        let d_setup = 32usize;
        let natural_len = 33usize;
        let n_prefix = padded_setup_prefix_len(natural_len);
        let seed = AkitaSetupSeed {
            max_num_vars: 1,
            max_num_batched_polys: 1,
            gen_ring_dim: d_setup,
            max_setup_len: 2,
            public_matrix_seed: [3u8; 32],
        };
        let prefix_params =
            setup_prefix_level_params(&level_params, n_prefix, d_setup).expect("prefix params");
        let id = setup_prefix_slot_id(
            setup_seed_digest(&seed).expect("seed digest"),
            d_setup,
            natural_len,
            n_prefix,
            digest_level_params(std::slice::from_ref(
                &prefix_params.expect("eligible prefix params"),
            )),
        );
        let slot = SetupPrefixVerifierSlot {
            id,
            natural_len,
            padded_len: n_prefix,
            commitment: SetupPrefixPublicCommitment {
                rows: vec![FlatRingVec::from_coeffs(vec![F::zero()])],
            },
        };
        let mut registry = SetupPrefixVerifierRegistry::<F>::new();
        registry.insert(slot).expect("insert slot");

        let err = select_setup_prefix_slot(
            &seed,
            2,
            |candidate| {
                registry
                    .get(candidate)
                    .map(|slot| (slot, slot.natural_len + 1, slot.padded_len))
            },
            &level_params,
            natural_len,
            d_setup,
            "slot does not cover request",
        )
        .expect_err("mismatched lookup metadata must fail closed");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn select_setup_prefix_slot_uses_canonical_id_and_checks_coverage() {
        use akita_field::Prime32Offset99 as F;

        let level_params = sample_level_params();
        let d_setup = 32usize;
        let natural_len = 33usize;
        let n_prefix = padded_setup_prefix_len(natural_len);
        let seed = AkitaSetupSeed {
            max_num_vars: 1,
            max_num_batched_polys: 1,
            gen_ring_dim: d_setup,
            max_setup_len: 2,
            public_matrix_seed: [3u8; 32],
        };
        let prefix_params =
            setup_prefix_level_params(&level_params, n_prefix, d_setup).expect("prefix params");
        let id = setup_prefix_slot_id(
            setup_seed_digest(&seed).expect("seed digest"),
            d_setup,
            natural_len,
            n_prefix,
            digest_level_params(std::slice::from_ref(
                &prefix_params.expect("eligible prefix params"),
            )),
        );
        let slot = SetupPrefixVerifierSlot {
            id,
            natural_len,
            padded_len: n_prefix,
            commitment: SetupPrefixPublicCommitment {
                rows: vec![FlatRingVec::from_coeffs(vec![F::zero()])],
            },
        };
        let mut registry = SetupPrefixVerifierRegistry::<F>::new();
        registry.insert(slot).expect("insert slot");

        let selection = select_setup_prefix_slot(
            &seed,
            2,
            |candidate| {
                registry
                    .get(candidate)
                    .map(|slot| (slot, slot.natural_len, slot.padded_len))
            },
            &level_params,
            natural_len,
            d_setup,
            "slot does not cover request",
        )
        .expect("selection succeeds")
        .expect("slot selected");
        assert_eq!(selection.0.id, id);
        assert_eq!(selection.1, 2);

        let selection = select_setup_prefix_slot(
            &seed,
            2,
            |candidate| {
                registry
                    .get(candidate)
                    .map(|slot| (slot, slot.natural_len, slot.padded_len))
            },
            &level_params,
            natural_len + 1,
            d_setup,
            "slot does not cover request",
        )
        .expect("different natural_len slot falls back");
        assert!(
            selection.is_none(),
            "same padded prefix with different natural_len should use a different id"
        );
    }

    #[test]
    fn prover_registry_duplicate_insert_does_not_replace_existing_slot() {
        use crate::proof::FlatDigitBlocks;
        use akita_algebra::CyclotomicRing;
        use akita_field::Prime32Offset99 as F;

        let id = SetupPrefixSlotId {
            setup_seed_digest: [7u8; 32],
            d_setup: 32,
            natural_len: 1,
            n_prefix: 32,
            level_params_digest: [9u8; 32],
        };
        let slot = || {
            let decomposed = FlatDigitBlocks::<32>::from_blocks(vec![Vec::new()]);
            let recomposed = vec![Vec::new()];
            let hint =
                AkitaCommitmentHint::singleton_with_recomposed_inner_rows(decomposed, recomposed);
            SetupPrefixSlot {
                id,
                natural_len: id.natural_len,
                padded_len: id.n_prefix,
                commitment: RingCommitment {
                    u: vec![CyclotomicRing::<F, 32>::zero()],
                },
                hint,
            }
        };

        let mut registry = SetupPrefixRegistry::<F>::new();
        registry
            .insert(SetupPrefixSlotAny::D32(slot()))
            .expect("first insert");
        registry
            .insert(SetupPrefixSlotAny::D32(slot()))
            .expect_err("duplicate insert must fail");

        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn verifier_registry_duplicate_insert_does_not_replace_existing_slot() {
        use akita_field::Prime32Offset99 as F;

        let id = SetupPrefixSlotId {
            setup_seed_digest: [7u8; 32],
            d_setup: 32,
            natural_len: 1,
            n_prefix: 32,
            level_params_digest: [9u8; 32],
        };
        let slot = || SetupPrefixVerifierSlot {
            id,
            natural_len: id.natural_len,
            padded_len: id.n_prefix,
            commitment: SetupPrefixPublicCommitment {
                rows: vec![FlatRingVec::from_coeffs(vec![F::zero()])],
            },
        };

        let mut registry = SetupPrefixVerifierRegistry::<F>::new();
        registry.insert(slot()).expect("first insert");
        registry
            .insert(slot())
            .expect_err("duplicate insert must fail");

        assert_eq!(registry.len(), 1);
    }
}
