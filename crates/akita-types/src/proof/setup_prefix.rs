//! Setup-prefix commitment artifacts for setup-claim offloading (slice 02B).
//!
//! This module defines preprocessing metadata for exact flat coefficient
//! prefixes of the shared setup vector `S`, zero-padded to power-of-two
//! commitment domains. It does not run a setup product sumcheck or change proof
//! semantics.

use crate::proof::{setup::MAX_SETUP_MATRIX_FIELD_ELEMENTS, AkitaCommitmentHint, RingVec};
use crate::{
    AjtaiKeyParams, LevelParams, OpeningClaimsLayout, PolynomialGroupLayout,
    PrecommittedGroupParams, PrecommittedLevelParams, SisModulusFamily,
};
use akita_field::{AkitaError, FieldCore};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};

const MAX_SETUP_PREFIX_SLOTS: usize = 4096;

/// Ring dimension used when delegating setup claims to a flat coefficient prefix.
pub const SETUP_OFFLOAD_D_SETUP: usize = 64;

/// Minimum padded setup-prefix field length for recursive setup offloading.
pub const SETUP_OFFLOAD_MIN_PREFIX_FIELD_LEN: usize = 1 << 10;

/// Identity for one committed setup-prefix slot.
///
/// `natural_len` distinguishes exact prefixes that share the padded commitment
/// domain derived from `commitment_params`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupPrefixSlotId {
    /// Coefficient-axis ring dimension for the delegated prefix object.
    pub d_setup: usize,
    /// Exact flat coefficient length represented before zero padding.
    pub natural_len: usize,
    /// Commitment parameters used to build the setup-prefix object.
    pub commitment_params: PrecommittedLevelParams,
}

impl SetupPrefixSlotId {
    /// Padded flat coefficient length committed for this slot.
    pub fn n_prefix(&self) -> Result<usize, AkitaError> {
        n_prefix_from_commitment_params(&self.commitment_params).map_err(|err| {
            AkitaError::InvalidSetup(format!("invalid setup-prefix commitment domain: {err}"))
        })
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        crate::descriptor_bytes::push_usize(bytes, self.d_setup);
        crate::descriptor_bytes::push_usize(bytes, self.natural_len);
        self.commitment_params.append_descriptor_bytes(bytes);
    }
}

fn precommitted_level_params_descriptor_bytes(params: &PrecommittedLevelParams) -> Vec<u8> {
    let mut bytes = Vec::new();
    params.append_descriptor_bytes(&mut bytes);
    bytes
}

fn n_prefix_from_commitment_params(
    params: &PrecommittedLevelParams,
) -> Result<usize, SerializationError> {
    1usize
        .checked_shl(params.layout.group.num_vars() as u32)
        .ok_or_else(|| {
            SerializationError::InvalidData(
                "setup prefix slot commitment domain overflows usize".to_string(),
            )
        })
}

impl Ord for SetupPrefixSlotId {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.d_setup, self.natural_len)
            .cmp(&(other.d_setup, other.natural_len))
            .then_with(|| {
                precommitted_level_params_descriptor_bytes(&self.commitment_params).cmp(
                    &precommitted_level_params_descriptor_bytes(&other.commitment_params),
                )
            })
    }
}

impl PartialOrd for SetupPrefixSlotId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Hash for SetupPrefixSlotId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.d_setup.hash(state);
        self.natural_len.hash(state);
        precommitted_level_params_descriptor_bytes(&self.commitment_params).hash(state);
    }
}

impl Valid for SetupPrefixSlotId {
    fn check(&self) -> Result<(), SerializationError> {
        if self.d_setup == 0 {
            return Err(SerializationError::InvalidData(
                "setup prefix slot d_setup must be non-zero".to_string(),
            ));
        }
        let n_prefix = n_prefix_from_commitment_params(&self.commitment_params)?;
        if self.natural_len == 0 || self.natural_len > n_prefix {
            return Err(SerializationError::InvalidData(
                "setup prefix slot natural_len must be in 1..=n_prefix".to_string(),
            ));
        }
        if n_prefix == 0 || !n_prefix.is_power_of_two() {
            return Err(SerializationError::InvalidData(
                "setup prefix slot n_prefix must be a non-zero power of two".to_string(),
            ));
        }
        if !n_prefix.is_multiple_of(self.d_setup) {
            return Err(SerializationError::InvalidData(
                "setup prefix slot n_prefix must be a multiple of d_setup".to_string(),
            ));
        }
        self.commitment_params
            .layout
            .validate()
            .map_err(|err| SerializationError::InvalidData(err.to_string()))?;
        if self.commitment_params.layout.group.num_polynomials() != 1 {
            return Err(SerializationError::InvalidData(
                "setup prefix slot commitment params must be singleton".to_string(),
            ));
        }
        Ok(())
    }
}

fn serialize_sis_family<W: Write>(
    family: SisModulusFamily,
    mut writer: W,
) -> Result<(), SerializationError> {
    let tag = match family {
        SisModulusFamily::Q32 => 0u8,
        SisModulusFamily::Q64 => 1u8,
        SisModulusFamily::Q128 => 2u8,
    };
    writer.write_all(&[tag])?;
    Ok(())
}

fn deserialize_sis_family<R: Read>(mut reader: R) -> Result<SisModulusFamily, SerializationError> {
    let mut tag = [0u8; 1];
    reader.read_exact(&mut tag)?;
    match tag[0] {
        0 => Ok(SisModulusFamily::Q32),
        1 => Ok(SisModulusFamily::Q64),
        2 => Ok(SisModulusFamily::Q128),
        _ => Err(SerializationError::InvalidData(
            "invalid SIS modulus family tag".to_string(),
        )),
    }
}

fn serialize_ajtai_key<W: Write>(
    key: &AjtaiKeyParams,
    mut writer: W,
    compress: Compress,
) -> Result<(), SerializationError> {
    key.row_len().serialize_with_mode(&mut writer, compress)?;
    key.col_len().serialize_with_mode(&mut writer, compress)?;
    key.min_security_bits()
        .serialize_with_mode(&mut writer, compress)?;
    serialize_sis_family(key.sis_family(), &mut writer)?;
    (key.sis_table_key().ring_dimension as usize).serialize_with_mode(&mut writer, compress)?;
    key.coeff_linf_bound()
        .serialize_with_mode(&mut writer, compress)?;
    Ok(())
}

fn deserialize_ajtai_key<R: Read>(
    mut reader: R,
    compress: Compress,
    validate: Validate,
) -> Result<AjtaiKeyParams, SerializationError> {
    let row_len = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let col_len = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let min_security_bits = u16::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let family = deserialize_sis_family(&mut reader)?;
    let ring_dimension = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let coeff_linf_bound = u128::deserialize_with_mode(&mut reader, compress, validate, &())?;
    Ok(AjtaiKeyParams::new_unchecked(
        min_security_bits,
        family,
        row_len,
        col_len,
        coeff_linf_bound,
        ring_dimension,
    ))
}

fn serialize_precommitted_level_params<W: Write>(
    params: &PrecommittedLevelParams,
    mut writer: W,
    compress: Compress,
) -> Result<(), SerializationError> {
    params
        .layout
        .group
        .num_vars()
        .serialize_with_mode(&mut writer, compress)?;
    params
        .layout
        .group
        .num_polynomials()
        .serialize_with_mode(&mut writer, compress)?;
    params
        .layout
        .m_vars
        .serialize_with_mode(&mut writer, compress)?;
    params
        .layout
        .r_vars
        .serialize_with_mode(&mut writer, compress)?;
    params
        .layout
        .log_basis
        .serialize_with_mode(&mut writer, compress)?;
    params
        .layout
        .n_a
        .serialize_with_mode(&mut writer, compress)?;
    params
        .layout
        .conservative_n_b
        .serialize_with_mode(&mut writer, compress)?;
    serialize_ajtai_key(&params.a_key, &mut writer, compress)?;
    serialize_ajtai_key(&params.b_key, &mut writer, compress)?;
    params
        .num_blocks
        .serialize_with_mode(&mut writer, compress)?;
    params
        .block_len
        .serialize_with_mode(&mut writer, compress)?;
    params
        .num_digits_commit
        .serialize_with_mode(&mut writer, compress)?;
    params
        .num_digits_open
        .serialize_with_mode(&mut writer, compress)?;
    params
        .num_digits_fold_one
        .serialize_with_mode(&mut writer, compress)?;
    Ok(())
}

fn deserialize_precommitted_level_params<R: Read>(
    mut reader: R,
    compress: Compress,
    validate: Validate,
) -> Result<PrecommittedLevelParams, SerializationError> {
    let group_num_vars = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let group_num_polynomials = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let m_vars = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let r_vars = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let log_basis = u32::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let n_a = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let conservative_n_b = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let a_key = deserialize_ajtai_key(&mut reader, compress, validate)?;
    let b_key = deserialize_ajtai_key(&mut reader, compress, validate)?;
    let num_blocks = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let block_len = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let num_digits_commit = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let num_digits_open = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let num_digits_fold_one = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    Ok(PrecommittedLevelParams {
        layout: PrecommittedGroupParams {
            group: PolynomialGroupLayout::new(group_num_vars, group_num_polynomials),
            m_vars,
            r_vars,
            log_basis,
            n_a,
            conservative_n_b,
        },
        a_key,
        b_key,
        num_blocks,
        block_len,
        num_digits_commit,
        num_digits_open,
        num_digits_fold_one,
    })
}

fn precommitted_level_params_serialized_size(
    params: &PrecommittedLevelParams,
    compress: Compress,
) -> usize {
    params.layout.group.num_vars().serialized_size(compress)
        + params
            .layout
            .group
            .num_polynomials()
            .serialized_size(compress)
        + params.layout.m_vars.serialized_size(compress)
        + params.layout.r_vars.serialized_size(compress)
        + params.layout.log_basis.serialized_size(compress)
        + params.layout.n_a.serialized_size(compress)
        + params.layout.conservative_n_b.serialized_size(compress)
        + params.a_key.row_len().serialized_size(compress)
        + params.a_key.col_len().serialized_size(compress)
        + params.a_key.min_security_bits().serialized_size(compress)
        + 1
        + (params.a_key.sis_table_key().ring_dimension as usize).serialized_size(compress)
        + params.a_key.coeff_linf_bound().serialized_size(compress)
        + params.b_key.row_len().serialized_size(compress)
        + params.b_key.col_len().serialized_size(compress)
        + params.b_key.min_security_bits().serialized_size(compress)
        + 1
        + (params.b_key.sis_table_key().ring_dimension as usize).serialized_size(compress)
        + params.b_key.coeff_linf_bound().serialized_size(compress)
        + params.num_blocks.serialized_size(compress)
        + params.block_len.serialized_size(compress)
        + params.num_digits_commit.serialized_size(compress)
        + params.num_digits_open.serialized_size(compress)
        + params.num_digits_fold_one.serialized_size(compress)
}

impl AkitaSerialize for SetupPrefixSlotId {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.d_setup.serialize_with_mode(&mut writer, compress)?;
        self.natural_len
            .serialize_with_mode(&mut writer, compress)?;
        serialize_precommitted_level_params(&self.commitment_params, &mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.d_setup.serialized_size(compress)
            + self.natural_len.serialized_size(compress)
            + precommitted_level_params_serialized_size(&self.commitment_params, compress)
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
        let d_setup = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let natural_len = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let commitment_params =
            deserialize_precommitted_level_params(&mut reader, compress, validate)?;
        let out = Self {
            d_setup,
            natural_len,
            commitment_params,
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
    pub rows: Vec<RingVec<F>>,
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
pub struct SetupPrefixVerifierSlot<F: FieldCore> {
    pub id: SetupPrefixSlotId,
    pub natural_len: usize,
    pub padded_len: usize,
    pub commitment: SetupPrefixPublicCommitment<F>,
}

impl<F: FieldCore + Valid> Valid for SetupPrefixVerifierSlot<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.id.check()?;
        let id_n_prefix = n_prefix_from_commitment_params(&self.id.commitment_params)?;
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
        if self.padded_len != id_n_prefix {
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
///
/// S4: D-free. The commitment is stored as the D-free
/// [`SetupPrefixPublicCommitment`] (flat ring-coefficient rows) rather than a
/// typed `RingCommitment<F, D>`, and the hint is the D-free
/// [`AkitaCommitmentHint<F>`]. The former compile-time `d_setup == D` guarantee
/// is re-asserted at runtime against `id.d_setup` and the per-row coefficient
/// width (see [`Valid::check`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupPrefixSlot<F: FieldCore> {
    pub id: SetupPrefixSlotId,
    pub natural_len: usize,
    pub padded_len: usize,
    pub commitment: SetupPrefixPublicCommitment<F>,
    pub hint: AkitaCommitmentHint<F>,
}

impl<F: FieldCore + Valid> Valid for SetupPrefixSlot<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.id.check()?;
        let id_n_prefix = n_prefix_from_commitment_params(&self.id.commitment_params)?;
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
        if self.padded_len != id_n_prefix {
            return Err(SerializationError::InvalidData(
                "setup prefix prover slot padded_len must match slot id".to_string(),
            ));
        }
        self.commitment.check()?;
        // Re-assert the invariant the const generic `D` used to enforce: every
        // commitment row must have exactly `d_setup` coefficients.
        for row in &self.commitment.rows {
            if row.coeff_len() != self.id.d_setup {
                return Err(SerializationError::InvalidData(format!(
                    "setup prefix prover slot commitment row has {} coefficients, expected \
                     d_setup={}",
                    row.coeff_len(),
                    self.id.d_setup
                )));
            }
        }
        self.hint.check()
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for SetupPrefixSlot<F> {
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

impl<F> AkitaDeserialize for SetupPrefixSlot<F>
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

impl<F: FieldCore> SetupPrefixSlot<F> {
    /// Strip prover-only hint material for verifier metadata.
    #[must_use]
    pub fn verifier_slot(&self) -> SetupPrefixVerifierSlot<F> {
        SetupPrefixVerifierSlot {
            id: self.id.clone(),
            natural_len: self.natural_len,
            padded_len: self.padded_len,
            commitment: self.commitment.clone(),
        }
    }
}

/// In-memory registry of prover-ready setup-prefix slots.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SetupPrefixProverRegistry<F: FieldCore> {
    slots: BTreeMap<SetupPrefixSlotId, SetupPrefixSlot<F>>,
}

impl<F: FieldCore> SetupPrefixProverRegistry<F> {
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
    pub fn get(&self, id: &SetupPrefixSlotId) -> Option<&SetupPrefixSlot<F>> {
        self.slots.get(id)
    }

    pub fn insert(&mut self, slot: SetupPrefixSlot<F>) -> Result<(), AkitaError> {
        if self.slots.contains_key(&slot.id) {
            return Err(AkitaError::InvalidSetup(
                "duplicate setup prefix slot id".to_string(),
            ));
        }
        self.slots.insert(slot.id.clone(), slot);
        Ok(())
    }

    pub fn replace_from(&mut self, other: Self) {
        self.slots = other.slots;
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

impl<F: FieldCore + Valid> Valid for SetupPrefixProverRegistry<F> {
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

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for SetupPrefixProverRegistry<F> {
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

impl<F> AkitaDeserialize for SetupPrefixProverRegistry<F>
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
        self.slots.insert(slot.id.clone(), slot);
        Ok(())
    }

    pub fn replace_from_prover_registry(
        &mut self,
        prover_registry: &SetupPrefixProverRegistry<F>,
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

/// Active packed setup footprint in ring slots.
fn active_setup_ring_slots(
    level_params: &LevelParams,
    opening_batch: &OpeningClaimsLayout,
) -> Result<usize, AkitaError> {
    opening_batch.check()?;
    level_params.validate_root_opening_batch(opening_batch)?;

    let mut max_slots = 0usize;
    let mut shared_d_width = 0usize;
    for group_index in 0..opening_batch.num_groups() {
        let group_layout = opening_batch.group_layout(group_index)?;
        let group_params = level_params.root_group_params(opening_batch, group_index)?;
        let a_width = group_params
            .block_len()
            .checked_mul(group_params.num_digits_commit())
            .ok_or_else(|| AkitaError::InvalidSetup("A setup width overflow".to_string()))?;
        let a_slots = group_params
            .a_rows_len()
            .checked_mul(a_width)
            .ok_or_else(|| AkitaError::InvalidSetup("A setup footprint overflow".to_string()))?;

        let b_width = group_layout
            .num_polynomials()
            .checked_mul(group_params.a_rows_len())
            .and_then(|n| n.checked_mul(group_params.num_blocks()))
            .and_then(|n| n.checked_mul(group_params.num_digits_open()))
            .ok_or_else(|| AkitaError::InvalidSetup("B setup width overflow".to_string()))?;
        let b_slots = group_params
            .b_rows_len()
            .checked_mul(b_width)
            .ok_or_else(|| AkitaError::InvalidSetup("B setup footprint overflow".to_string()))?;

        let d_width = group_layout
            .num_polynomials()
            .checked_mul(group_params.num_blocks())
            .and_then(|n| n.checked_mul(group_params.num_digits_open()))
            .ok_or_else(|| AkitaError::InvalidSetup("D setup width overflow".to_string()))?;
        shared_d_width = shared_d_width
            .checked_add(d_width)
            .ok_or_else(|| AkitaError::InvalidSetup("D setup width overflow".to_string()))?;

        max_slots = max_slots.max(a_slots).max(b_slots);
    }

    let d_slots = level_params
        .d_key
        .row_len()
        .checked_mul(shared_d_width)
        .ok_or_else(|| AkitaError::InvalidSetup("D setup footprint overflow".to_string()))?;
    Ok(max_slots.max(d_slots))
}

/// Active flat coefficient count `N_active^F = D_setup * N_active^R`.
pub fn active_setup_field_len(
    level_params: &LevelParams,
    opening_batch: &OpeningClaimsLayout,
    d_setup: usize,
) -> Result<usize, AkitaError> {
    active_setup_ring_slots(level_params, opening_batch)?
        .checked_mul(d_setup)
        .ok_or_else(|| AkitaError::InvalidSetup("active setup field length overflow".to_string()))
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
    let num_digits_commit = crate::sis::compute_num_digits_full_field(
        level_params.field_bits_for_cache(),
        level_params.log_basis,
    );
    let mut num_blocks = 1usize;
    while num_blocks <= ring_slots {
        if ring_slots.is_multiple_of(num_blocks) {
            let block_len = ring_slots / num_blocks;
            let inner_width = block_len.checked_mul(num_digits_commit).ok_or_else(|| {
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
                prefix_params.num_digits_commit = num_digits_commit;
                return Ok(Some(prefix_params));
            }
        }
        num_blocks = num_blocks
            .checked_mul(2)
            .ok_or_else(|| AkitaError::InvalidSetup("prefix block count overflow".to_string()))?;
    }
    Ok(None)
}

/// Convert committed setup-prefix `LevelParams` into the precommitted-group
/// metadata stored on the consuming fold.
pub fn setup_prefix_precommitted_params(
    prefix_params: &LevelParams,
    n_prefix: usize,
) -> Result<PrecommittedLevelParams, AkitaError> {
    if n_prefix == 0 || !n_prefix.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "setup prefix length must be a nonzero power of two".to_string(),
        ));
    }
    Ok(PrecommittedLevelParams {
        layout: PrecommittedGroupParams {
            group: PolynomialGroupLayout::singleton(n_prefix.trailing_zeros() as usize),
            m_vars: prefix_params.m_vars,
            r_vars: prefix_params.r_vars,
            log_basis: prefix_params.log_basis,
            n_a: prefix_params.a_key.row_len(),
            conservative_n_b: prefix_params.b_key.row_len(),
        },
        a_key: prefix_params.a_key.clone(),
        b_key: prefix_params.b_key.clone(),
        num_blocks: prefix_params.num_blocks,
        block_len: prefix_params.block_len,
        num_digits_commit: prefix_params.num_digits_commit,
        num_digits_open: prefix_params.num_digits_open,
        num_digits_fold_one: prefix_params.num_digits_fold_one,
    })
}

/// Build the slot id for one committed setup prefix.
pub fn setup_prefix_slot_id(
    d_setup: usize,
    natural_len: usize,
    commitment_params: PrecommittedLevelParams,
) -> SetupPrefixSlotId {
    SetupPrefixSlotId {
        d_setup,
        natural_len,
        commitment_params,
    }
}

/// Select a setup-prefix slot that covers one setup-product footprint.
///
/// This centralizes the derivation shared by prover and verifier: setup seed
/// digest, padded prefix length, prefix commitment parameters, slot id, coverage
/// check, and the ring-slot evaluation length used for setup MLEs.
pub fn select_setup_prefix_slot<'a, Slot, Lookup>(
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
    let slot_id = if let Some(template) = &level_params.setup_prefix {
        let template_n_prefix = template.n_prefix()?;
        if template.natural_len != natural_field_len || template_n_prefix != n_prefix {
            return Err(AkitaError::InvalidSetup(coverage_error.to_string()));
        }
        template.clone()
    } else {
        let Some(prefix_params) = setup_prefix_level_params(level_params, n_prefix, d_setup)?
        else {
            return Ok(None);
        };
        let commitment_params = setup_prefix_precommitted_params(&prefix_params, n_prefix)?;
        setup_prefix_slot_id(d_setup, natural_field_len, commitment_params)
    };
    let slot_n_prefix = slot_id.n_prefix()?;
    if slot_n_prefix > setup_field_len {
        return Ok(None);
    };
    let Some((slot, slot_natural_len, slot_padded_len)) = lookup_slot(&slot_id) else {
        return Ok(None);
    };
    if slot_natural_len != natural_field_len || slot_padded_len != n_prefix {
        return Err(AkitaError::InvalidSetup(coverage_error.to_string()));
    }
    let setup_eval_len = slot_n_prefix.checked_div(d_setup).ok_or_else(|| {
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
    use crate::{LevelParams, OpeningClaimsLayout, SisModulusFamily};
    use akita_challenges::SparseChallengeConfig;

    fn sample_level_params() -> LevelParams {
        LevelParams::params_only(
            SisModulusFamily::Q32,
            32,
            3,
            2,
            3,
            2,
            SparseChallengeConfig::pm1_only(3),
        )
        .with_decomp(2, 3, 2, 2, 3)
        .expect("sample level params")
    }

    fn prefix_eligible_level_params() -> LevelParams {
        let full_field_digits = crate::sis::compute_num_digits_full_field(128, 3);
        LevelParams::params_only(
            SisModulusFamily::Q32,
            32,
            3,
            2,
            3,
            2,
            SparseChallengeConfig::pm1_only(3),
        )
        .with_decomp(2, 3, full_field_digits, 2, 3)
        .expect("prefix eligible level params")
    }

    #[test]
    fn active_setup_field_len_matches_packed_role_maximum() {
        let lp = sample_level_params();
        let opening_batch = OpeningClaimsLayout::new(5, 3).expect("opening batch");
        let w_a = lp.block_len * lp.num_digits_commit;
        let w_b = opening_batch.num_total_polynomials()
            * lp.a_key.row_len()
            * lp.num_blocks
            * lp.num_digits_open;
        let w_d = opening_batch.num_total_polynomials() * lp.num_blocks * lp.num_digits_open;
        let expected_ring_slots = lp
            .a_key
            .row_len()
            .checked_mul(w_a)
            .unwrap()
            .max(lp.b_key.row_len().checked_mul(w_b).unwrap())
            .max(lp.d_key.row_len().checked_mul(w_d).unwrap());
        assert_eq!(
            active_setup_ring_slots(&lp, &opening_batch).expect("ring slots"),
            expected_ring_slots
        );
        assert_eq!(
            active_setup_field_len(&lp, &opening_batch, SETUP_OFFLOAD_D_SETUP).expect("field len"),
            expected_ring_slots * SETUP_OFFLOAD_D_SETUP
        );
    }

    #[test]
    fn select_setup_prefix_slot_uses_canonical_id_and_checks_coverage() {
        use akita_field::Prime32Offset99 as F;

        let level_params = prefix_eligible_level_params();
        let d_setup = 32usize;
        let natural_len = 33usize;
        let n_prefix = padded_setup_prefix_len(natural_len);
        let prefix_params =
            setup_prefix_level_params(&level_params, n_prefix, d_setup).expect("prefix params");
        let prefix_params = prefix_params.expect("eligible prefix params");
        let id = setup_prefix_slot_id(
            d_setup,
            natural_len,
            setup_prefix_precommitted_params(&prefix_params, n_prefix)
                .expect("precommitted prefix params"),
        );
        let slot = SetupPrefixVerifierSlot {
            id: id.clone(),
            natural_len,
            padded_len: n_prefix,
            commitment: SetupPrefixPublicCommitment {
                rows: vec![RingVec::from_coeffs(vec![F::zero()])],
            },
        };
        let mut registry = SetupPrefixVerifierRegistry::<F>::new();
        registry.insert(slot).expect("insert slot");

        let selection = select_setup_prefix_slot(
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
        assert_eq!(&selection.0.id, &id);
        assert_eq!(selection.1, 2);

        let selection = select_setup_prefix_slot(
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
        use crate::proof::DigitBlocks;
        use akita_field::Prime32Offset99 as F;

        let commitment_params =
            setup_prefix_precommitted_params(&sample_level_params(), 32).expect("prefix params");
        let id = setup_prefix_slot_id(32, 1, commitment_params);
        let slot = || {
            // D-free hint: one empty digit block at stride 32 (the former D).
            let decomposed = DigitBlocks::from_blocks(vec![Vec::new()], 32).expect("digit blocks");
            let hint = AkitaCommitmentHint::<F>::singleton(decomposed);
            SetupPrefixSlot {
                id: id.clone(),
                natural_len: id.natural_len,
                padded_len: id.n_prefix().expect("padded len"),
                // One commitment row of d_setup = 32 coefficients.
                commitment: SetupPrefixPublicCommitment {
                    rows: vec![RingVec::from_coeffs(vec![F::zero(); 32])],
                },
                hint,
            }
        };

        let mut registry = SetupPrefixProverRegistry::<F>::new();
        registry.insert(slot()).expect("first insert");
        registry
            .insert(slot())
            .expect_err("duplicate insert must fail");

        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn verifier_registry_duplicate_insert_does_not_replace_existing_slot() {
        use akita_field::Prime32Offset99 as F;

        let commitment_params =
            setup_prefix_precommitted_params(&sample_level_params(), 32).expect("prefix params");
        let id = setup_prefix_slot_id(32, 1, commitment_params);
        let slot = || SetupPrefixVerifierSlot {
            id: id.clone(),
            natural_len: id.natural_len,
            padded_len: id.n_prefix().expect("padded len"),
            commitment: SetupPrefixPublicCommitment {
                rows: vec![RingVec::from_coeffs(vec![F::zero()])],
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
