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
    GroupCommitPhaseParams, GroupOpenPhaseParams, InnerCommitMatrixParams, OpeningClaimsLayout,
    OuterCommitMatrixParams, PolynomialGroupLayout,
};
use akita_error::AkitaError;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use jolt_field::Field;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};

const MAX_SETUP_PREFIX_SLOTS: usize = 4096;
pub const SETUP_PREFIX_CONTENT_TAG: &[u8; 4] = b"SPF4";

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
    /// Frozen commitment profile used to build the setup-prefix object.
    pub commitment_profile: GroupCommitPhaseParams,
}

impl GroupOpenPhaseParams {
    /// Descriptor bytes for this group in its role as a fold's setup prefix.
    ///
    /// Byte-identical to the encoder on the deleted `GroupOpenPhaseParams`:
    /// slot id, then the consuming fold's opening plan.
    pub(crate) fn append_setup_prefix_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        if let Some(slot) = self.slot_id() {
            slot.append_descriptor_bytes(bytes);
        }
        self.opening.append_descriptor_bytes(bytes);
    }
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
        self.commitment_profile.inner.matrix.ring_dimension()
    }

    /// Full power-of-two flat coefficient length committed for this slot.
    pub fn n_prefix(&self) -> Result<usize, AkitaError> {
        n_prefix_from_commitment_profile(&self.commitment_profile).map_err(|err| {
            AkitaError::InvalidSetup(format!("invalid setup-prefix commitment domain: {err}"))
        })
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(SETUP_PREFIX_CONTENT_TAG);
        crate::descriptor_bytes::push_usize(bytes, self.natural_len);
        self.commitment_profile.append_descriptor_bytes(bytes);
    }
}

fn committed_group_profile_descriptor_bytes(params: &GroupCommitPhaseParams) -> Vec<u8> {
    let mut bytes = Vec::new();
    params.append_descriptor_bytes(&mut bytes);
    bytes
}

fn n_prefix_from_commitment_profile(
    params: &GroupCommitPhaseParams,
) -> Result<usize, SerializationError> {
    1usize
        .checked_shl(params.group.num_vars() as u32)
        .ok_or_else(|| {
            SerializationError::InvalidData(
                "setup prefix slot commitment domain overflows usize".to_string(),
            )
        })
}

/// Validate that a setup prefix uses the smallest power-of-two commitment
/// domain covering its active support.
pub fn validate_setup_prefix_domain(natural_len: usize, n_prefix: usize) -> Result<(), AkitaError> {
    let expected = natural_len.checked_next_power_of_two().ok_or_else(|| {
        AkitaError::InvalidSetup("setup-prefix natural length overflows its padded domain".into())
    })?;
    if natural_len == 0 || n_prefix != expected {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix commitment domain is not the canonical padded natural length".into(),
        ));
    }
    Ok(())
}

impl Ord for SetupPrefixSlotId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.natural_len.cmp(&other.natural_len).then_with(|| {
            committed_group_profile_descriptor_bytes(&self.commitment_profile).cmp(
                &committed_group_profile_descriptor_bytes(&other.commitment_profile),
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
        committed_group_profile_descriptor_bytes(&self.commitment_profile).hash(state);
    }
}

impl Valid for SetupPrefixSlotId {
    fn check(&self) -> Result<(), SerializationError> {
        self.commitment_profile
            .validate(
                self.commitment_profile
                    .inner
                    .matrix
                    .sis_modulus_profile()
                    .field_bits(),
            )
            .and_then(|()| {
                self.commitment_profile
                    .validate_setup_prefix_geometry(self.natural_len)
            })
            .map_err(|err| SerializationError::InvalidData(err.to_string()))?;
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
    SisSecurityPolicyId::from_tag(tag[0]).ok_or_else(|| {
        SerializationError::InvalidData("invalid SIS security policy tag".to_string())
    })
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

fn serialize_committed_group_profile<W: Write>(
    params: &GroupCommitPhaseParams,
    mut writer: W,
    compress: Compress,
) -> Result<(), SerializationError> {
    params.version.serialize_with_mode(&mut writer, compress)?;
    params
        .group
        .num_vars()
        .serialize_with_mode(&mut writer, compress)?;
    params
        .group
        .num_polynomials()
        .serialize_with_mode(&mut writer, compress)?;
    params
        .blocks
        .live_ring_elements_per_claim
        .serialize_with_mode(&mut writer, compress)?;
    params
        .blocks
        .positions_per_block
        .serialize_with_mode(&mut writer, compress)?;
    params
        .blocks
        .live_blocks
        .serialize_with_mode(&mut writer, compress)?;
    let outer_slice_count = params.outer_slice_count.get();
    outer_slice_count.serialize_with_mode(&mut writer, compress)?;
    params
        .inner
        .digits
        .log_basis
        .serialize_with_mode(&mut writer, compress)?;
    params
        .inner
        .digits
        .num_digits
        .serialize_with_mode(&mut writer, compress)?;
    serialize_commit_matrix(&params.inner.matrix, &mut writer, compress)?;
    params
        .outer
        .digits
        .log_basis
        .serialize_with_mode(&mut writer, compress)?;
    params
        .outer
        .digits
        .num_digits
        .serialize_with_mode(&mut writer, compress)?;
    serialize_commit_matrix(&params.outer.matrix, &mut writer, compress)?;
    Ok(())
}

fn deserialize_committed_group_profile<R: Read>(
    mut reader: R,
    compress: Compress,
    validate: Validate,
) -> Result<GroupCommitPhaseParams, SerializationError> {
    let version = u8::deserialize_with_mode(&mut reader, compress, validate, &())?;
    if version != GroupCommitPhaseParams::VERSION {
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
    Ok(GroupCommitPhaseParams {
        version,
        group,

        blocks: crate::BlockGeometry::new(
            num_live_ring_elements_per_claim,
            num_positions_per_block,
            num_live_blocks,
        ),

        outer_slice_count,
        inner: crate::RoleParams::new(
            crate::GadgetDigits::new(log_basis_inner, num_digits_inner),
            inner_commit_matrix,
        ),
        outer: crate::RoleParams::new(
            crate::GadgetDigits::new(log_basis_outer, num_digits_outer),
            outer_commit_matrix,
        ),
    })
}

fn committed_group_profile_serialized_size(
    params: &GroupCommitPhaseParams,
    compress: Compress,
) -> usize {
    let outer_slice_count = params.outer_slice_count.get();
    params.version.serialized_size(compress)
        + params.group.num_vars().serialized_size(compress)
        + params.group.num_polynomials().serialized_size(compress)
        + params
            .blocks
            .live_ring_elements_per_claim
            .serialized_size(compress)
        + params.blocks.positions_per_block.serialized_size(compress)
        + params.blocks.live_blocks.serialized_size(compress)
        + outer_slice_count.serialized_size(compress)
        + params.inner.digits.log_basis.serialized_size(compress)
        + params.inner.digits.num_digits.serialized_size(compress)
        + commit_matrix_serialized_size(&params.inner.matrix, compress)
        + params.outer.digits.log_basis.serialized_size(compress)
        + params.outer.digits.num_digits.serialized_size(compress)
        + commit_matrix_serialized_size(&params.outer.matrix, compress)
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
        serialize_committed_group_profile(&self.commitment_profile, &mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        SETUP_PREFIX_CONTENT_TAG.len()
            + self.natural_len.serialized_size(compress)
            + committed_group_profile_serialized_size(&self.commitment_profile, compress)
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
        let commitment_profile =
            deserialize_committed_group_profile(&mut reader, compress, validate)?;
        let out = Self {
            natural_len,
            commitment_profile,
        };
        if validate == Validate::Yes {
            out.check()?;
        }
        Ok(out)
    }
}

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

/// Prover-ready metadata for one setup-prefix slot.
///
/// S4: D-free. The commitment is stored as the D-free
/// [`SetupPrefixPublicCommitment`] (flat ring-coefficient rows) rather than a
/// typed `RingCommitment<F, D>`, and the hint is the D-free
/// [`AkitaCommitmentHint<F>`]. The former compile-time `d_setup == D` guarantee
/// is re-asserted at runtime against `id.d_setup` and the per-row coefficient
/// width (see [`Valid::check`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupPrefixSlot<F: Field> {
    pub id: SetupPrefixSlotId,
    pub commitment: SetupPrefixPublicCommitment<F>,
    pub hint: AkitaCommitmentHint<F>,
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

fn active_setup_projection_geometry(
    level_params: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
) -> Result<crate::SetupProjectionGeometry, AkitaError> {
    let final_group_index = level_params.validate_opening_batch(opening_batch)?;

    let d_physical_cols = level_params.open().matrix.input_width();
    let mut groups = Vec::with_capacity(opening_batch.num_groups());
    for group_index in 0..opening_batch.num_groups() {
        // The batch-wide validation above already checked every preceding
        // group and its layout. Resolving both views through the public accessors
        // would repeat that validation twice per group.
        let (group_params, group_role_dims) = if group_index == final_group_index {
            (level_params.final_group(), level_params.role_dims())
        } else {
            let group_params = *level_params
                .preceding_group_params(group_index)
                .ok_or(AkitaError::InvalidProof)?;
            (
                group_params,
                group_params.role_dims(level_params.open().matrix.ring_dimension()),
            )
        };
        group_role_dims.validate_role_projection()?;
        let a_cols = group_params
            .num_positions_per_block()
            .checked_mul(group_params.num_digits_inner())
            .ok_or_else(|| AkitaError::InvalidSetup("A setup width overflow".to_string()))?;

        let b_cols = group_params.b_col_len();

        groups.push(crate::setup_contribution::SetupProjectionGroupGeometry {
            role_dims: group_role_dims,
            a_rows: group_params.a_rows_len(),
            a_cols,
            b_rows: group_params.b_rows_len(),
            b_cols,
        });
    }
    crate::SetupProjectionGeometry::from_groups(
        level_params.role_dims(),
        level_params.open().matrix.output_rank(),
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
) -> Result<GroupOpenPhaseParams, AkitaError> {
    let d_setup = prefix_params.inner().matrix.ring_dimension();
    let d_outer = prefix_params.outer().matrix.ring_dimension();
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
    let setup_num_digits = crate::sis::compute_num_digits_field_width(
        prefix_params
            .inner()
            .matrix
            .sis_modulus_profile()
            .field_bits(),
        prefix_params.inner().digits.log_basis,
    );
    let ring_slots = n_prefix / d_setup;
    let mut num_positions_per_block = 1usize;
    while num_positions_per_block <= ring_slots.max(1) {
        let num_live_blocks = ring_slots.div_ceil(num_positions_per_block);
        if prefix_params.outer_slice_count().get() > num_live_blocks {
            break;
        }
        let inner_width = num_positions_per_block
            .checked_mul(setup_num_digits)
            .ok_or_else(|| AkitaError::InvalidSetup("prefix inner width overflow".to_string()))?;
        let outer_width = CommitmentSliceGeometry::try_new(
            prefix_params.outer_slice_count(),
            num_live_blocks,
            1,
            prefix_params.inner().matrix.output_rank(),
            prefix_params.outer().digits.num_digits,
            d_setup,
            d_outer,
        )?
        .physical_input_width();
        if inner_width <= prefix_params.inner().matrix.input_width()
            && outer_width <= prefix_params.outer().matrix.input_width()
        {
            if prefix_params.inner().matrix.sis_table_key().is_none() {
                return Err(AkitaError::InvalidSetup(
                    "setup prefix cannot be derived from an L2 A security route".into(),
                ));
            }
            let inner_commit_matrix = prefix_params
                .inner()
                .matrix
                .try_with_input_width(inner_width)?;
            let outer_commit_matrix = OuterCommitMatrixParams::new_unchecked(
                prefix_params.outer().matrix.security_policy(),
                prefix_params.outer().matrix.sis_table_key().table_digest,
                prefix_params.outer().matrix.sis_modulus_profile(),
                prefix_params.outer().matrix.output_rank(),
                outer_width,
                prefix_params.outer().matrix.coeff_linf_bound(),
                prefix_params.outer().matrix.ring_dimension(),
            );
            return Ok(GroupOpenPhaseParams {
                setup_natural_len: None,
                profile: GroupCommitPhaseParams {
                    version: GroupCommitPhaseParams::VERSION,
                    group: PolynomialGroupLayout::singleton(n_prefix.trailing_zeros() as usize),

                    blocks: crate::BlockGeometry::new(
                        ring_slots,
                        num_positions_per_block,
                        num_live_blocks,
                    ),

                    outer_slice_count: prefix_params.outer_slice_count(),
                    inner: crate::RoleParams::new(
                        crate::GadgetDigits::new(
                            prefix_params.inner().digits.log_basis,
                            setup_num_digits,
                        ),
                        inner_commit_matrix,
                    ),
                    outer: crate::RoleParams::new(
                        crate::GadgetDigits::new(
                            prefix_params.outer().digits.log_basis,
                            prefix_params.outer().digits.num_digits,
                        ),
                        outer_commit_matrix,
                    ),
                },
                opening: crate::GroupOpeningPlan::evaluation_trace(
                    prefix_params.fold_challenge_config(),
                    prefix_params.open().digits.log_basis,
                    prefix_params.open().digits.num_digits,
                    prefix_params.num_digits_fold(),
                ),
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

/// Mark one committed group as a fold's incoming setup prefix.
///
/// Presence of `setup_natural_len` is the sole record that this group is a
/// prefix and the sole record of its active support length, so there is no
/// second field for a mirror audit to compare against.
#[must_use]
pub fn scheduled_setup_prefix(
    natural_len: usize,
    commitment_params: GroupOpenPhaseParams,
) -> GroupOpenPhaseParams {
    GroupOpenPhaseParams {
        setup_natural_len: Some(natural_len),
        ..commitment_params
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
