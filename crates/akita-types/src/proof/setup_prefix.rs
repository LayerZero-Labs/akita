//! Setup-prefix commitment artifacts for setup-claim offloading (slice 02B).
//!
//! This module defines preprocessing metadata for actual power-of-two flat
//! coefficient prefixes of the shared setup vector `S`. It does not run a setup
//! product sumcheck or change proof semantics.

use crate::descriptor_bytes::sis_modulus_profile_tag;
use crate::proof::{AkitaCommitmentHint, RingVec, MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS};
use crate::sis::{SisMatrixRole, SisModulusProfileId, SisSecurityPolicyId, SisTableDigest};
use crate::{
    AkitaSetupSeed, CommitmentSliceCount, CommitmentSliceGeometry, CommittedGroupParams,
    CommittedGroupProfile, InnerCommitMatrixParams, OpeningClaimsLayout, OuterCommitMatrixParams,
    PolynomialGroupLayout, PrecommittedLevelParams,
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
pub const SETUP_PREFIX_CONTENT_TAG: &[u8; 4] = b"SPF1";

#[path = "setup_prefix_helpers.rs"]
mod helpers;
use helpers::setup_prefix_compression_plan;
pub use helpers::suffix_opening_layout;

/// Identity for one committed setup-prefix slot.
///
/// `natural_len` distinguishes active setup-weight supports that share the
/// full-prefix commitment domain derived from `commitment_params`.
#[derive(Debug, Clone)]
pub struct SetupPrefixSlotId {
    /// Active setup-weight support in flat field coefficients.
    pub natural_len: usize,
    /// Commitment parameters used to build the setup-prefix object.
    pub commitment_params: PrecommittedLevelParams,
}

impl PartialEq for SetupPrefixSlotId {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for SetupPrefixSlotId {}

impl SetupPrefixSlotId {
    /// Ring dimension used to commit the setup-prefix coefficient vector.
    #[must_use]
    pub fn d_setup(&self) -> usize {
        self.commitment_params
            .layout
            .inner_commit_matrix
            .ring_dimension()
    }

    /// Full power-of-two flat coefficient length committed for this slot.
    pub fn n_prefix(&self) -> Result<usize, AkitaError> {
        n_prefix_from_commitment_params(&self.commitment_params).map_err(|err| {
            AkitaError::InvalidSetup(format!("invalid setup-prefix commitment domain: {err}"))
        })
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(SETUP_PREFIX_CONTENT_TAG);
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
        self.natural_len.cmp(&other.natural_len).then_with(|| {
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
        self.natural_len.hash(state);
        precommitted_level_params_descriptor_bytes(&self.commitment_params).hash(state);
    }
}

impl Valid for SetupPrefixSlotId {
    fn check(&self) -> Result<(), SerializationError> {
        let d_setup = self.d_setup();
        if d_setup == 0 {
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
        if !n_prefix.is_multiple_of(d_setup) {
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
        1 => Ok(SisMatrixRole::Inner),
        2 => Ok(SisMatrixRole::Outer),
        3 => Ok(SisMatrixRole::Open),
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

#[path = "setup_prefix_commit_matrix.rs"]
mod commit_matrix;
use commit_matrix::{
    commit_matrix_serialized_size, deserialize_commit_matrix, serialize_commit_matrix,
};

fn serialize_precommitted_level_params<W: Write>(
    params: &PrecommittedLevelParams,
    mut writer: W,
    compress: Compress,
) -> Result<(), SerializationError> {
    params
        .layout
        .version
        .serialize_with_mode(&mut writer, compress)?;
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
    let outer_slice_count = params.layout.outer_slice_count.get();
    outer_slice_count.serialize_with_mode(&mut writer, compress)?;
    params
        .layout
        .log_basis_inner
        .serialize_with_mode(&mut writer, compress)?;
    params
        .layout
        .num_digits_inner
        .serialize_with_mode(&mut writer, compress)?;
    serialize_commit_matrix(&params.layout.inner_commit_matrix, &mut writer, compress)?;
    params
        .layout
        .log_basis_outer
        .serialize_with_mode(&mut writer, compress)?;
    params
        .layout
        .num_digits_outer
        .serialize_with_mode(&mut writer, compress)?;
    serialize_commit_matrix(&params.layout.outer_commit_matrix, &mut writer, compress)?;
    params
        .log_basis_open
        .serialize_with_mode(&mut writer, compress)?;
    params
        .fold_challenge_config
        .count_pm1
        .serialize_with_mode(&mut writer, compress)?;
    params
        .fold_challenge_config
        .count_pm2
        .serialize_with_mode(&mut writer, compress)?;
    params
        .num_digits_open
        .serialize_with_mode(&mut writer, compress)?;
    params
        .num_digits_fold
        .serialize_with_mode(&mut writer, compress)?;
    Ok(())
}

fn deserialize_precommitted_level_params<R: Read>(
    mut reader: R,
    compress: Compress,
    validate: Validate,
) -> Result<PrecommittedLevelParams, SerializationError> {
    let version = u8::deserialize_with_mode(&mut reader, compress, validate, &())?;
    if version != CommittedGroupProfile::VERSION {
        return Err(SerializationError::InvalidData(format!(
            "unknown committed-group profile version {version}"
        )));
    }
    let group_num_vars = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let group_num_polynomials = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let group = PolynomialGroupLayout::new(group_num_vars, group_num_polynomials);
    let num_live_ring_elements_per_claim =
        usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let num_positions_per_block =
        usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let num_live_blocks = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let raw_slice_count = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let outer_slice_count = CommitmentSliceCount::try_new(raw_slice_count)
        .map_err(|err| SerializationError::InvalidData(err.to_string()))?;
    let log_basis_inner = u32::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let num_digits_inner = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let inner_commit_matrix: InnerCommitMatrixParams =
        deserialize_commit_matrix(&mut reader, compress, validate)?;
    let log_basis_outer = u32::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let num_digits_outer = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let outer_commit_matrix: OuterCommitMatrixParams =
        deserialize_commit_matrix(&mut reader, compress, validate)?;
    let log_basis_open = u32::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let challenge_count_pm1 = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let challenge_count_pm2 = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let num_digits_open = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let num_digits_fold = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    Ok(PrecommittedLevelParams {
        layout: CommittedGroupProfile {
            version,
            group,
            num_live_ring_elements_per_claim,
            num_positions_per_block,
            num_live_blocks,
            outer_slice_count,
            log_basis_inner,
            num_digits_inner,
            inner_commit_matrix,
            log_basis_outer,
            num_digits_outer,
            outer_commit_matrix,
        },
        log_basis_open,
        fold_challenge_config: akita_challenges::SparseChallengeConfig {
            count_pm1: challenge_count_pm1,
            count_pm2: challenge_count_pm2,
        },
        num_digits_open,
        num_digits_fold,
    })
}

fn precommitted_level_params_serialized_size(
    params: &PrecommittedLevelParams,
    compress: Compress,
) -> usize {
    let outer_slice_count = params.layout.outer_slice_count.get();
    params.layout.version.serialized_size(compress)
        + params.layout.group.num_vars().serialized_size(compress)
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
        + outer_slice_count.serialized_size(compress)
        + params.layout.log_basis_inner.serialized_size(compress)
        + params.layout.num_digits_inner.serialized_size(compress)
        + commit_matrix_serialized_size(&params.layout.inner_commit_matrix, compress)
        + params.layout.log_basis_outer.serialized_size(compress)
        + params.layout.num_digits_outer.serialized_size(compress)
        + commit_matrix_serialized_size(&params.layout.outer_commit_matrix, compress)
        + params.log_basis_open.serialized_size(compress)
        + params
            .fold_challenge_config
            .count_pm1
            .serialized_size(compress)
        + params
            .fold_challenge_config
            .count_pm2
            .serialized_size(compress)
        + params.num_digits_open.serialized_size(compress)
        + params.num_digits_fold.serialized_size(compress)
}

impl AkitaSerialize for SetupPrefixSlotId {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.check()?;
        writer.write_all(SETUP_PREFIX_CONTENT_TAG)?;
        self.natural_len
            .serialize_with_mode(&mut writer, compress)?;
        serialize_precommitted_level_params(&self.commitment_params, &mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        SETUP_PREFIX_CONTENT_TAG.len()
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
        let mut content_tag = [0u8; SETUP_PREFIX_CONTENT_TAG.len()];
        reader.read_exact(&mut content_tag)?;
        if &content_tag != SETUP_PREFIX_CONTENT_TAG {
            return Err(SerializationError::InvalidData(
                "unsupported setup-prefix content format".to_string(),
            ));
        }
        let natural_len = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let commitment_params =
            deserialize_precommitted_level_params(&mut reader, compress, validate)?;
        let out = Self {
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
        if total_coeffs > MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS {
            return Err(SerializationError::LengthLimitExceeded {
                len: u64::try_from(total_coeffs).unwrap_or(u64::MAX),
                max: MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS,
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
            MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS,
        )?;
        let mut rows = Vec::new();
        super::reserve_shape_len(&mut rows, row_count)?;
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
pub struct SetupPrefixVerifierSlot<F: FieldCore> {
    pub id: SetupPrefixSlotId,
    pub commitment: SetupPrefixPublicCommitment<F>,
}

impl<F: FieldCore + Valid> Valid for SetupPrefixVerifierSlot<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.id.check()?;
        self.commitment.check()?;
        let expected_payload_coefficients =
            setup_prefix_compression_plan(&self.id.commitment_params)?.terminal_coefficients();
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

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for SetupPrefixVerifierSlot<F> {
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
    pub commitment: SetupPrefixPublicCommitment<F>,
    pub hint: AkitaCommitmentHint<F>,
}

impl<F: FieldCore + Valid> Valid for SetupPrefixSlot<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.id.check()?;
        self.commitment.check()?;
        let compression_plan = setup_prefix_compression_plan(&self.id.commitment_params)?;
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

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for SetupPrefixSlot<F> {
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
    fn validate_compression_hint(&self) -> Result<(), AkitaError> {
        let plan = setup_prefix_compression_plan(&self.id.commitment_params)
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
pub struct SetupPrefixProverRegistry<F: FieldCore> {
    setup_seed: AkitaSetupSeed,
    slots: BTreeMap<SetupPrefixSlotId, SetupPrefixSlot<F>>,
}

impl<F: FieldCore> SetupPrefixProverRegistry<F> {
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

impl<F: FieldCore + Valid> Valid for SetupPrefixProverRegistry<F> {
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

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for SetupPrefixProverRegistry<F> {
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
    F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
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

/// In-memory registry of verifier-visible setup-prefix slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupPrefixVerifierRegistry<F: FieldCore> {
    setup_seed: AkitaSetupSeed,
    slots: BTreeMap<SetupPrefixSlotId, SetupPrefixVerifierSlot<F>>,
}

impl<F: FieldCore> SetupPrefixVerifierRegistry<F> {
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

    pub fn replace_from_prover_registry(
        &mut self,
        prover_registry: &SetupPrefixProverRegistry<F>,
    ) -> Result<(), AkitaError>
    where
        F: Valid,
    {
        if self.setup_seed != *prover_registry.setup_seed() {
            return Err(AkitaError::InvalidSetup(
                "setup-prefix registries belong to different public matrices".to_string(),
            ));
        }
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

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for SetupPrefixVerifierRegistry<F> {
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
    F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
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

fn active_setup_projection_geometry(
    level_params: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
) -> Result<crate::SetupProjectionGeometry, AkitaError> {
    opening_batch.check()?;
    level_params.validate_opening_batch(opening_batch)?;

    let mut d_physical_cols = 0usize;
    let mut groups = Vec::with_capacity(opening_batch.num_groups());
    for group_index in 0..opening_batch.num_groups() {
        let group_layout = opening_batch.group_layout(group_index)?;
        let group_params = level_params.group_params(opening_batch, group_index)?;
        let group_role_dims = level_params.group_role_dims(opening_batch, group_index)?;
        let (_, d_subcolumns) =
            crate::SetupProjectionGeometry::native_role_subcolumn_counts(group_role_dims)?;
        let a_cols = group_params
            .num_positions_per_block()
            .checked_mul(group_params.num_digits_inner())
            .ok_or_else(|| AkitaError::InvalidSetup("A setup width overflow".to_string()))?;

        let b_cols = group_params.b_col_len();

        let d_active_cols = group_layout
            .num_polynomials()
            .checked_mul(group_params.num_live_blocks())
            .and_then(|n| n.checked_mul(group_params.num_digits_open()))
            .and_then(|n| n.checked_mul(d_subcolumns))
            .ok_or_else(|| AkitaError::InvalidSetup("D setup width overflow".to_string()))?;
        d_physical_cols = d_physical_cols
            .checked_add(d_active_cols)
            .ok_or_else(|| AkitaError::InvalidSetup("D setup width overflow".to_string()))?;

        groups.push(crate::setup_contribution::SetupProjectionGroupGeometry {
            role_dims: group_role_dims,
            a_rows: group_params.a_rows_len(),
            a_cols,
            b_rows: group_params.b_rows_len(),
            b_cols,
            d_active_cols,
        });
    }
    crate::SetupProjectionGeometry::from_groups(
        level_params.role_dims(),
        level_params.open_commit_matrix.output_rank(),
        d_physical_cols,
        &groups,
    )
}

/// Active flat coefficient count under the canonical Stage 3 base projection.
pub fn active_setup_field_len(
    level_params: &CommittedGroupParams,
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
    prefix_params: &CommittedGroupParams,
    n_prefix: usize,
) -> Result<PrecommittedLevelParams, AkitaError> {
    let d_setup = prefix_params.inner_commit_matrix.ring_dimension();
    let d_outer = prefix_params.outer_commit_matrix.ring_dimension();
    if d_outer == 0 || !d_setup.is_multiple_of(d_outer) {
        return Err(AkitaError::InvalidSetup(
            "setup prefix A dimension must be a multiple of its B dimension".to_string(),
        ));
    }
    if n_prefix == 0 || !n_prefix.is_power_of_two() || !n_prefix.is_multiple_of(d_setup) {
        return Err(AkitaError::InvalidSetup(
            "setup prefix length must be a nonzero power-of-two multiple of d_setup".to_string(),
        ));
    }
    let ring_slots = n_prefix / d_setup;
    let mut num_positions_per_block = 1usize;
    while num_positions_per_block <= ring_slots.max(1) {
        let num_live_blocks = ring_slots.div_ceil(num_positions_per_block);
        if prefix_params.outer_slice_count.get() > num_live_blocks {
            break;
        }
        let inner_width = num_positions_per_block
            .checked_mul(prefix_params.num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("prefix inner width overflow".to_string()))?;
        let outer_width = CommitmentSliceGeometry::try_new(
            prefix_params.outer_slice_count,
            num_live_blocks,
            1,
            prefix_params.inner_commit_matrix.output_rank(),
            prefix_params.num_digits_outer,
            d_setup,
            d_outer,
        )?
        .physical_input_width();
        if inner_width <= prefix_params.inner_commit_matrix.input_width()
            && outer_width <= prefix_params.outer_commit_matrix.input_width()
        {
            if prefix_params.inner_commit_matrix.sis_table_key().is_none() {
                return Err(AkitaError::InvalidSetup(
                    "setup prefix cannot be derived from an L2 A security route".into(),
                ));
            }
            let inner_commit_matrix = prefix_params
                .inner_commit_matrix
                .try_with_input_width(inner_width)?;
            let outer_commit_matrix = OuterCommitMatrixParams::new_unchecked(
                prefix_params.outer_commit_matrix.security_policy(),
                prefix_params
                    .outer_commit_matrix
                    .sis_table_key()
                    .table_digest,
                prefix_params.outer_commit_matrix.sis_modulus_profile(),
                prefix_params.outer_commit_matrix.output_rank(),
                outer_width,
                prefix_params.outer_commit_matrix.coeff_linf_bound(),
                prefix_params.outer_commit_matrix.ring_dimension(),
            );
            return Ok(PrecommittedLevelParams {
                layout: CommittedGroupProfile {
                    version: CommittedGroupProfile::VERSION,
                    group: PolynomialGroupLayout::singleton(n_prefix.trailing_zeros() as usize),
                    num_live_ring_elements_per_claim: ring_slots,
                    num_positions_per_block,
                    num_live_blocks,
                    outer_slice_count: prefix_params.outer_slice_count,
                    log_basis_inner: prefix_params.log_basis_inner,
                    num_digits_inner: prefix_params.num_digits_inner,
                    inner_commit_matrix,
                    log_basis_outer: prefix_params.log_basis_outer,
                    num_digits_outer: prefix_params.num_digits_outer,
                    outer_commit_matrix,
                },
                log_basis_open: prefix_params.log_basis_open,
                fold_challenge_config: prefix_params.fold_challenge_config,
                num_digits_open: prefix_params.num_digits_open,
                num_digits_fold: prefix_params.num_digits_fold,
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
    natural_len: usize,
    commitment_params: PrecommittedLevelParams,
) -> SetupPrefixSlotId {
    SetupPrefixSlotId {
        natural_len,
        commitment_params,
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
    let Some(template) = &level_params.setup_prefix else {
        return Err(AkitaError::InvalidSetup(
            "Stage 3 requires a selected setup-prefix slot".to_string(),
        ));
    };
    if selected_slot_id != template {
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
    if template.natural_len != natural_field_len || template_n_prefix != n_prefix {
        return Err(AkitaError::InvalidSetup(format!(
            "{coverage_error}: planned natural/full-prefix lengths are {}/{template_n_prefix}, \
             active lengths are {natural_field_len}/{n_prefix}",
            template.natural_len,
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
