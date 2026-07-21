//! Setup-prefix commitment artifacts for setup-claim offloading (slice 02B).
//!
//! This module defines preprocessing metadata for exact flat coefficient
//! prefixes of the shared setup vector `S`, zero-padded to power-of-two
//! commitment domains. It does not run a setup product sumcheck or change proof
//! semantics.

use crate::descriptor_bytes::sis_modulus_profile_tag;
use crate::proof::{setup::MAX_SETUP_MATRIX_FIELD_ELEMENTS, AkitaCommitmentHint, RingVec};
use crate::sis::{SisMatrixRole, SisModulusProfileId, SisSecurityPolicyId, SisTableDigest};
use crate::{
    AjtaiKeyParams, LevelParams, OpeningClaimsLayout, PolynomialGroupLayout,
    PrecommittedGroupParams, PrecommittedLevelParams,
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

fn serialize_sis_modulus_profile<W: Write>(
    profile: SisModulusProfileId,
    mut writer: W,
) -> Result<(), SerializationError> {
    writer.write_all(&[sis_modulus_profile_tag(profile)])?;
    Ok(())
}

fn deserialize_sis_modulus_profile<R: Read>(
    mut reader: R,
) -> Result<SisModulusProfileId, SerializationError> {
    let mut tag = [0u8; 1];
    reader.read_exact(&mut tag)?;
    match tag[0] {
        0 => Ok(SisModulusProfileId::Q32Offset99),
        1 => Ok(SisModulusProfileId::Q64Offset59),
        2 => Ok(SisModulusProfileId::Q128OffsetA7F7),
        _ => Err(SerializationError::InvalidData(
            "invalid SIS modulus profile tag".to_string(),
        )),
    }
}

fn serialize_sis_security_policy<W: Write>(
    policy: SisSecurityPolicyId,
    mut writer: W,
) -> Result<(), SerializationError> {
    writer.write_all(&[policy.tag()])?;
    Ok(())
}

fn deserialize_sis_security_policy<R: Read>(
    mut reader: R,
) -> Result<SisSecurityPolicyId, SerializationError> {
    let mut tag = [0u8; 1];
    reader.read_exact(&mut tag)?;
    match tag[0] {
        1 => Ok(SisSecurityPolicyId::Quantum128BitADPS16),
        _ => Err(SerializationError::InvalidData(
            "invalid SIS security policy tag".to_string(),
        )),
    }
}

fn serialize_sis_matrix_role<W: Write>(
    role: SisMatrixRole,
    mut writer: W,
) -> Result<(), SerializationError> {
    writer.write_all(&[role.tag()])?;
    Ok(())
}

fn deserialize_sis_matrix_role<R: Read>(
    mut reader: R,
) -> Result<SisMatrixRole, SerializationError> {
    let mut tag = [0u8; 1];
    reader.read_exact(&mut tag)?;
    match tag[0] {
        1 => Ok(SisMatrixRole::A),
        2 => Ok(SisMatrixRole::B),
        3 => Ok(SisMatrixRole::D),
        _ => Err(SerializationError::InvalidData(
            "invalid SIS matrix role tag".to_string(),
        )),
    }
}

fn serialize_sis_table_digest<W: Write>(
    digest: SisTableDigest,
    mut writer: W,
) -> Result<(), SerializationError> {
    writer.write_all(&digest.0)?;
    Ok(())
}

fn deserialize_sis_table_digest<R: Read>(
    mut reader: R,
) -> Result<SisTableDigest, SerializationError> {
    let mut bytes = [0u8; 32];
    reader.read_exact(&mut bytes)?;
    Ok(SisTableDigest(bytes))
}

/// Wire layout mirrors [`AjtaiKeyParams::append_descriptor_bytes`]:
/// profile tag, policy tag, role tag, table digest, ring dim, row, col, linf.
fn serialize_ajtai_key<W: Write>(
    key: &AjtaiKeyParams,
    mut writer: W,
    compress: Compress,
) -> Result<(), SerializationError> {
    let table_key = key.sis_table_key();
    serialize_sis_modulus_profile(key.sis_modulus_profile(), &mut writer)?;
    serialize_sis_security_policy(key.security_policy(), &mut writer)?;
    serialize_sis_matrix_role(table_key.role, &mut writer)?;
    serialize_sis_table_digest(table_key.table_digest, &mut writer)?;
    (table_key.ring_dimension as usize).serialize_with_mode(&mut writer, compress)?;
    key.row_len().serialize_with_mode(&mut writer, compress)?;
    key.col_len().serialize_with_mode(&mut writer, compress)?;
    key.coeff_linf_bound()
        .serialize_with_mode(&mut writer, compress)?;
    Ok(())
}

fn deserialize_ajtai_key<R: Read>(
    mut reader: R,
    compress: Compress,
    validate: Validate,
) -> Result<AjtaiKeyParams, SerializationError> {
    let sis_modulus_profile = deserialize_sis_modulus_profile(&mut reader)?;
    let policy = deserialize_sis_security_policy(&mut reader)?;
    let role = deserialize_sis_matrix_role(&mut reader)?;
    let table_digest = deserialize_sis_table_digest(&mut reader)?;
    let ring_dimension = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let row_len = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let col_len = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let coeff_linf_bound = u128::deserialize_with_mode(&mut reader, compress, validate, &())?;
    Ok(AjtaiKeyParams::new_unchecked(
        policy,
        table_digest,
        sis_modulus_profile,
        role,
        row_len,
        col_len,
        coeff_linf_bound,
        ring_dimension,
    ))
}

fn ajtai_key_serialized_size(key: &AjtaiKeyParams, compress: Compress) -> usize {
    1 // profile tag
        + 1 // policy tag
        + 1 // role tag
        + 32 // table digest
        + (key.sis_table_key().ring_dimension as usize).serialized_size(compress)
        + key.row_len().serialized_size(compress)
        + key.col_len().serialized_size(compress)
        + key.coeff_linf_bound().serialized_size(compress)
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
        .num_live_ring_elements_per_claim
        .serialize_with_mode(&mut writer, compress)?;
    params
        .layout
        .num_positions_per_block
        .serialize_with_mode(&mut writer, compress)?;
    params
        .layout
        .num_live_blocks
        .serialize_with_mode(&mut writer, compress)?;
    match params.layout.fold_challenge_shape {
        akita_challenges::TensorChallengeShape::Flat => writer.write_all(&[0])?,
        akita_challenges::TensorChallengeShape::Tensor { fold_low_len } => {
            writer.write_all(&[1])?;
            fold_low_len.serialize_with_mode(&mut writer, compress)?;
        }
    }
    params
        .layout
        .log_basis_inner
        .serialize_with_mode(&mut writer, compress)?;
    params
        .layout
        .log_basis_outer
        .serialize_with_mode(&mut writer, compress)?;
    params
        .log_basis_open
        .serialize_with_mode(&mut writer, compress)?;
    params
        .layout
        .n_a
        .serialize_with_mode(&mut writer, compress)?;
    params
        .layout
        .n_b
        .serialize_with_mode(&mut writer, compress)?;
    serialize_ajtai_key(&params.a_key, &mut writer, compress)?;
    serialize_ajtai_key(&params.b_key, &mut writer, compress)?;
    params
        .num_digits_inner
        .serialize_with_mode(&mut writer, compress)?;
    params
        .num_digits_outer
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
    let num_live_ring_elements_per_claim =
        usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let num_positions_per_block =
        usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let num_live_blocks = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let mut shape_tag = [0u8; 1];
    reader.read_exact(&mut shape_tag)?;
    let fold_challenge_shape = match shape_tag[0] {
        0 => akita_challenges::TensorChallengeShape::Flat,
        1 => akita_challenges::TensorChallengeShape::Tensor {
            fold_low_len: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
        },
        _ => {
            return Err(SerializationError::InvalidData(
                "invalid setup-prefix fold challenge shape tag".to_string(),
            ))
        }
    };
    let log_basis_inner = u32::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let log_basis_outer = u32::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let log_basis_open = u32::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let n_a = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let n_b = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let a_key = deserialize_ajtai_key(&mut reader, compress, validate)?;
    let b_key = deserialize_ajtai_key(&mut reader, compress, validate)?;
    let num_digits_inner = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let num_digits_outer = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let num_digits_open = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let num_digits_fold_one = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    Ok(PrecommittedLevelParams {
        layout: PrecommittedGroupParams {
            group: PolynomialGroupLayout::new(group_num_vars, group_num_polynomials),
            num_live_ring_elements_per_claim,
            num_positions_per_block,
            num_live_blocks,
            fold_challenge_shape,
            log_basis_inner,
            log_basis_outer,
            n_a,
            a_coeff_linf_bound: a_key.coeff_linf_bound(),
            n_b,
            b_coeff_linf_bound: b_key.coeff_linf_bound(),
        },
        a_key,
        b_key,
        log_basis_open,
        num_digits_inner,
        num_digits_outer,
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
        + params
            .layout
            .num_live_ring_elements_per_claim
            .serialized_size(compress)
        + params
            .layout
            .num_positions_per_block
            .serialized_size(compress)
        + params.layout.num_live_blocks.serialized_size(compress)
        + 1
        + match params.layout.fold_challenge_shape {
            akita_challenges::TensorChallengeShape::Flat => 0,
            akita_challenges::TensorChallengeShape::Tensor { fold_low_len } => {
                fold_low_len.serialized_size(compress)
            }
        }
        + params.layout.log_basis_inner.serialized_size(compress)
        + params.layout.log_basis_outer.serialized_size(compress)
        + params.log_basis_open.serialized_size(compress)
        + params.layout.n_a.serialized_size(compress)
        + params.layout.n_b.serialized_size(compress)
        + ajtai_key_serialized_size(&params.a_key, compress)
        + ajtai_key_serialized_size(&params.b_key, compress)
        + params.num_digits_inner.serialized_size(compress)
        + params.num_digits_outer.serialized_size(compress)
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

fn active_setup_projection_geometry(
    level_params: &LevelParams,
    opening_batch: &OpeningClaimsLayout,
) -> Result<crate::SetupProjectionGeometry, AkitaError> {
    opening_batch.check()?;
    level_params.validate_opening_batch(opening_batch)?;

    let mut max_a_slots = 0usize;
    let mut max_b_slots = 0usize;
    let mut shared_d_width = 0usize;
    for group_index in 0..opening_batch.num_groups() {
        let group_layout = opening_batch.group_layout(group_index)?;
        let group_params = level_params.group_params(opening_batch, group_index)?;
        let a_width = group_params
            .num_positions_per_block()
            .checked_mul(group_params.num_digits_inner())
            .ok_or_else(|| AkitaError::InvalidSetup("A setup width overflow".to_string()))?;
        let a_slots = group_params
            .a_rows_len()
            .checked_mul(a_width)
            .ok_or_else(|| AkitaError::InvalidSetup("A setup footprint overflow".to_string()))?;

        let b_width = group_layout
            .num_polynomials()
            .checked_mul(group_params.a_rows_len())
            .and_then(|n| n.checked_mul(group_params.num_live_blocks()))
            .and_then(|n| n.checked_mul(group_params.num_digits_outer()))
            .ok_or_else(|| AkitaError::InvalidSetup("B setup width overflow".to_string()))?;
        let b_slots = group_params
            .b_rows_len()
            .checked_mul(b_width)
            .ok_or_else(|| AkitaError::InvalidSetup("B setup footprint overflow".to_string()))?;

        let d_width = group_layout
            .num_polynomials()
            .checked_mul(group_params.num_live_blocks())
            .and_then(|n| n.checked_mul(group_params.num_digits_open()))
            .ok_or_else(|| AkitaError::InvalidSetup("D setup width overflow".to_string()))?;
        shared_d_width = shared_d_width
            .checked_add(d_width)
            .ok_or_else(|| AkitaError::InvalidSetup("D setup width overflow".to_string()))?;

        max_a_slots = max_a_slots.max(a_slots);
        max_b_slots = max_b_slots.max(b_slots);
    }
    let d_slots = level_params
        .d_key
        .row_len()
        .checked_mul(shared_d_width)
        .ok_or_else(|| AkitaError::InvalidSetup("D setup footprint overflow".to_string()))?;
    crate::SetupProjectionGeometry::from_role_footprints(
        level_params.role_dims(),
        max_a_slots,
        max_b_slots,
        d_slots,
    )
}

/// Active flat coefficient count under the canonical Stage 3 base projection.
pub fn active_setup_field_len(
    level_params: &LevelParams,
    opening_batch: &OpeningClaimsLayout,
) -> Result<usize, AkitaError> {
    Ok(active_setup_projection_geometry(level_params, opening_batch)?.natural_field_len())
}

/// Smallest power-of-two flat prefix length covering `natural_field_len`.
#[must_use]
pub fn padded_setup_prefix_len(natural_field_len: usize) -> usize {
    natural_field_len.max(1).next_power_of_two()
}

/// Repack `level_params` into the precommitted-group metadata stored on the
/// consuming fold.
pub fn setup_prefix_precommitted_params(
    prefix_params: &LevelParams,
    n_prefix: usize,
) -> Result<PrecommittedLevelParams, AkitaError> {
    let d_setup = SETUP_OFFLOAD_D_SETUP;
    if n_prefix == 0 || !n_prefix.is_power_of_two() || !n_prefix.is_multiple_of(d_setup) {
        return Err(AkitaError::InvalidSetup(
            "setup prefix length must be a nonzero power-of-two multiple of d_setup".to_string(),
        ));
    }
    let ring_slots = n_prefix / d_setup;
    let mut num_positions_per_block = 1usize;
    while num_positions_per_block <= ring_slots.max(1) {
        let num_live_blocks = ring_slots.div_ceil(num_positions_per_block);
        let inner_width = num_positions_per_block
            .checked_mul(prefix_params.num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("prefix inner width overflow".to_string()))?;
        let outer_width = num_live_blocks
            .checked_mul(prefix_params.a_key.row_len())
            .and_then(|n| n.checked_mul(prefix_params.num_digits_outer))
            .ok_or_else(|| AkitaError::InvalidSetup("prefix outer width overflow".to_string()))?;
        if inner_width <= prefix_params.a_key.col_len()
            && outer_width <= prefix_params.b_key.col_len()
        {
            return Ok(PrecommittedLevelParams {
                layout: PrecommittedGroupParams {
                    group: PolynomialGroupLayout::singleton(n_prefix.trailing_zeros() as usize),
                    num_live_ring_elements_per_claim: ring_slots,
                    num_positions_per_block,
                    num_live_blocks,
                    fold_challenge_shape: prefix_params.fold_challenge_shape,
                    log_basis_inner: prefix_params.log_basis_inner,
                    log_basis_outer: prefix_params.log_basis_outer,
                    n_a: prefix_params.a_key.row_len(),
                    a_coeff_linf_bound: prefix_params.a_key.coeff_linf_bound(),
                    n_b: prefix_params.b_key.row_len(),
                    b_coeff_linf_bound: prefix_params.b_key.coeff_linf_bound(),
                },
                a_key: prefix_params.a_key.clone(),
                b_key: prefix_params.b_key.clone(),
                log_basis_open: prefix_params.log_basis_open,
                num_digits_inner: prefix_params.num_digits_inner,
                num_digits_outer: prefix_params.num_digits_outer,
                num_digits_open: prefix_params.num_digits_open,
                num_digits_fold_one: prefix_params.num_digits_fold_one,
            });
        }
        num_positions_per_block = num_positions_per_block.checked_mul(2).ok_or_else(|| {
            AkitaError::InvalidSetup("prefix position count overflow".to_string())
        })?;
    }
    Err(AkitaError::InvalidSetup(
        "setup prefix does not fit successor commitment widths".to_string(),
    ))
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
/// digest, padded prefix length, prefix commitment parameters, slot id, natural
/// source coverage, and the ring-slot evaluation length used for setup MLEs.
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
    let Some(template) = &level_params.setup_prefix else {
        return Ok(None);
    };

    let n_prefix = padded_setup_prefix_len(natural_field_len);
    let setup_field_len = setup_ring_slots_at_d.checked_mul(d_setup).ok_or_else(|| {
        AkitaError::InvalidSetup("setup matrix field length overflow".to_string())
    })?;
    if natural_field_len > setup_field_len {
        return Err(AkitaError::InvalidSetup(
            "setup prefix request exceeds shared matrix capacity".to_string(),
        ));
    }
    let template_n_prefix = template.n_prefix()?;
    if template.natural_len != natural_field_len || template_n_prefix != n_prefix {
        return Err(AkitaError::InvalidSetup(coverage_error.to_string()));
    }

    let Some((slot, slot_natural_len, slot_padded_len)) = lookup_slot(template) else {
        return Err(AkitaError::InvalidSetup(
            "required setup prefix slot is missing from registry".to_string(),
        ));
    };
    if slot_natural_len != natural_field_len || slot_padded_len != n_prefix {
        return Err(AkitaError::InvalidSetup(coverage_error.to_string()));
    }
    let setup_eval_len = template_n_prefix.checked_div(d_setup).ok_or_else(|| {
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
#[path = "setup_prefix_tests.rs"]
mod tests;
