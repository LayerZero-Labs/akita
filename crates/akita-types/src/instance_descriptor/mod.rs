//! Fiat-Shamir instance descriptor bound into the transcript preamble.
//!
//! The descriptor is intentionally smaller than the proof or verifier setup:
//! large structured inputs are represented by Blake2b digests of canonical
//! Akita encodings. The top-level descriptor remains self-describing and
//! round-trippable so both prover and verifier can compare preamble bytes.
//!
//! ## Descriptor version policy
//!
//! Each incompatible change increments the version while the protocol is under
//! active development. Integrators must pin an exact revision because the
//! project does not guarantee backward compatibility.

#[cfg(test)]
mod tests;
mod transcript_grinding_binding;

pub use transcript_grinding_binding::TranscriptGrindingBinding;

use crate::descriptor_bytes::{push_usize, sis_modulus_profile_tag};
use crate::narrowing::{usize_to_u32, usize_to_u8};
use crate::{
    AkitaSetupSeed, BasisMode, CommittedGroupParams, CompressionPolicyId, DecompositionParams,
    FoldSchedule, OpeningClaimsLayout, SisModulusProfileId, COMPRESSION_POLICY,
};
use akita_error::AkitaError;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
    DEFAULT_MAX_SEQUENCE_LEN,
};
use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};
use jolt_field::{CanonicalEncoding, ExtField, Field};
use std::io::{Read, Write};

/// Descriptor schema version for the in-development transcript preamble.
pub const AKITA_INSTANCE_DESCRIPTOR_VERSION: u32 = 4;

/// Fixed-size Blake2b digest used inside the descriptor.
pub type DescriptorDigest = [u8; 32];

/// Compute the descriptor digest for a public matrix identity.
///
/// The expanded shared matrix and NTT views are deterministic caches derived
/// from the setup seed, so the transcript descriptor binds the seed and the
/// schedule/layout metadata that determine how those caches are used.
///
/// # Errors
///
/// Returns a serialization error if the seed cannot be canonically serialized.
pub fn setup_seed_digest(
    setup_seed: &AkitaSetupSeed,
) -> Result<DescriptorDigest, SerializationError> {
    digest_serializable(setup_seed)
}

/// Canonical transcript preamble for one Akita proof instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaInstanceDescriptor {
    /// Schema version.
    pub version: u32,
    /// Algebraic substrate for this binary/proof family.
    pub algebra: AlgebraSection,
    /// Setup-bound parameters and deterministic setup identity.
    pub setup: SetupSection,
    /// Final effective verifier schedule for this proof.
    pub plan: PlanSection,
    /// Protocol-wide transcript-grinding policy and plan identity.
    pub grinding: TranscriptGrindingBinding,
    /// Per-call public shape and batching data.
    pub call: CallSection,
}

impl AkitaInstanceDescriptor {
    /// Construct a descriptor from its canonical sections.
    pub fn new(
        algebra: AlgebraSection,
        setup: SetupSection,
        plan: PlanSection,
        grinding: TranscriptGrindingBinding,
        call: CallSection,
    ) -> Self {
        Self {
            version: AKITA_INSTANCE_DESCRIPTOR_VERSION,
            algebra,
            setup,
            plan,
            grinding,
            call,
        }
    }

    /// Return canonical uncompressed descriptor bytes.
    ///
    /// # Errors
    ///
    /// Returns serialization errors from the underlying Akita serializer.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, SerializationError> {
        let mut out = Vec::with_capacity(self.uncompressed_size());
        self.serialize_uncompressed(&mut out)?;
        Ok(out)
    }
}

/// Algebraic substrate that determines the field towers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlgebraSection {
    /// Characteristic `p` of the base prime field, big-endian and 32-byte
    /// padded.
    pub prime_modulus_be: [u8; 32],
    /// Extension degree of the message field over the base prime field.
    pub field_extension_degree: u8,
    /// Extension degree of the protocol extension field over the base prime field.
    pub extension_degree: u8,
}

impl AlgebraSection {
    /// Build the algebra section for base field `F` and extension field `E`.
    ///
    /// # Errors
    ///
    /// Returns an error if an extension degree does not fit the descriptor's
    /// fixed-width integer fields.
    pub fn for_fields<F, E>() -> Result<Self, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        Ok(Self {
            prime_modulus_be: modulus_be_32::<F>()?,
            field_extension_degree: usize_to_u8(1, "field extension degree")?,
            extension_degree: usize_to_u8(E::DEGREE, "extension degree")?,
        })
    }
}

/// Compile-time features that change protocol transcript behavior.
///
/// After the zk-strip cutover the product is transparent-only; the wire field
/// remains for transcript layout stability and must deserialize as `zk = false`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProtocolFeatureSet {
    /// Whether zk hiding was active (always `false` after zk-strip).
    pub zk: bool,
}

impl ProtocolFeatureSet {
    /// Return the protocol feature set of the current build.
    #[inline]
    pub const fn current() -> Self {
        Self { zk: false }
    }
}

/// Setup-bound descriptor fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupSection {
    /// Gadget decomposition parameters.
    pub decomposition: DecompositionParams,
    /// SIS modulus family used for security sizing.
    pub sis_modulus_profile: SisModulusProfileId,
    /// Commitment-compression protocol bound into the transcript preamble.
    pub compression_policy: CompressionPolicyId,
    /// Digest of the canonical [`AkitaSetupSeed`] bytes.
    pub setup_seed_digest: DescriptorDigest,
    /// Protocol-affecting feature mode (transparent-only after zk-strip).
    pub protocol_features: ProtocolFeatureSet,
}

impl SetupSection {
    /// Build setup fields from existing setup/layout data.
    ///
    /// The per-level `CommittedGroupParams` are intentionally *not* digested here: the
    /// per-proof effective schedule (`PlanSection`) already binds every
    /// expanded fold `CommittedGroupParams`, and
    /// the public-matrix identity is bound separately from local provisioning.
    ///
    /// # Errors
    ///
    /// Returns a serialization error if the setup seed fails to serialize.
    pub fn from_parts(
        decomposition: DecompositionParams,
        sis_modulus_profile: SisModulusProfileId,
        setup_seed: &AkitaSetupSeed,
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            decomposition,
            sis_modulus_profile,
            compression_policy: COMPRESSION_POLICY,
            setup_seed_digest: setup_seed_digest(setup_seed)?,
            protocol_features: ProtocolFeatureSet::current(),
        })
    }
}

/// Per-proof effective schedule binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanSection {
    /// Explicit v1 catalog and row identity selected by the public statement.
    pub schedule_selection: crate::OpeningScheduleSelection,
    /// Digest of the final effective verifier schedule.
    pub effective_schedule_digest: DescriptorDigest,
}

impl PlanSection {
    /// Build a plan section from the runtime schedule the verifier will replay.
    pub fn from_schedule(
        schedule_selection: crate::OpeningScheduleSelection,
        schedule: &FoldSchedule,
    ) -> Self {
        Self {
            schedule_selection,
            effective_schedule_digest: digest_effective_schedule(schedule),
        }
    }
}

/// Per commit-and-open call descriptor fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallSection {
    /// Number of commitment groups opened by the call.
    pub num_commitment_groups: u32,
    /// Per-group opening-point arities in descriptor/transcript order.
    pub num_vars_per_commitment_group: Vec<u32>,
    /// Per-group polynomial counts in descriptor/transcript order.
    pub num_polys_per_commitment_group: Vec<u32>,
    /// Public basis mode for opening-point weights.
    pub basis_mode: BasisMode,
    /// Digest of normalized opening layout.
    pub opening_batch_digest: DescriptorDigest,
}

impl CallSection {
    /// Build call fields from normalized public opening layout.
    ///
    /// # Errors
    ///
    /// Returns an error if a count does not fit the descriptor's fixed-width
    /// integer fields.
    pub fn from_layout(
        layout: &OpeningClaimsLayout,
        basis_mode: BasisMode,
    ) -> Result<Self, AkitaError> {
        layout.check()?;
        let num_vars_per_commitment_group = layout
            .groups()
            .iter()
            .map(|group| usize_to_u32(group.num_vars(), "num_vars_per_commitment_group"))
            .collect::<Result<Vec<_>, _>>()?;
        let num_polys_per_commitment_group = layout
            .groups()
            .iter()
            .map(|group| usize_to_u32(group.num_polynomials(), "num_polys_per_commitment_group"))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            num_commitment_groups: usize_to_u32(layout.num_groups(), "num_commitment_groups")?,
            num_vars_per_commitment_group,
            num_polys_per_commitment_group,
            basis_mode,
            opening_batch_digest: layout.opening_batch_digest(),
        })
    }
}

/// Return the Blake2b-256 digest of an Akita-serializable value.
///
/// # Errors
///
/// Returns serialization errors from the value's canonical encoder.
pub fn digest_serializable<S: AkitaSerialize>(
    value: &S,
) -> Result<DescriptorDigest, SerializationError> {
    let mut bytes = Vec::with_capacity(value.uncompressed_size());
    value.serialize_uncompressed(&mut bytes)?;
    Ok(blake2b_256(&bytes))
}

/// Digest a normalized list of commitment level parameters.
pub fn digest_level_params(params: &[CommittedGroupParams]) -> DescriptorDigest {
    let mut bytes = Vec::new();
    push_usize(&mut bytes, params.len());
    for params in params {
        params.append_descriptor_bytes(&mut bytes);
    }
    blake2b_256(&bytes)
}

/// Digest the final effective runtime verifier schedule.
pub fn digest_effective_schedule(schedule: &FoldSchedule) -> DescriptorDigest {
    let mut bytes = Vec::new();
    schedule.append_descriptor_bytes(&mut bytes);
    blake2b_256(&bytes)
}

impl Valid for AkitaInstanceDescriptor {
    fn check(&self) -> Result<(), SerializationError> {
        if self.version != AKITA_INSTANCE_DESCRIPTOR_VERSION {
            return Err(SerializationError::InvalidData(format!(
                "unsupported Akita instance descriptor version {}",
                self.version
            )));
        }
        self.algebra.check()?;
        self.setup.check()?;
        self.plan.check()?;
        self.grinding.check()?;
        self.call.check()?;
        Ok(())
    }
}

impl AkitaSerialize for AkitaInstanceDescriptor {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.version.serialize_with_mode(&mut writer, compress)?;
        self.algebra.serialize_with_mode(&mut writer, compress)?;
        self.setup.serialize_with_mode(&mut writer, compress)?;
        self.plan.serialize_with_mode(&mut writer, compress)?;
        self.grinding.serialize_with_mode(&mut writer, compress)?;
        self.call.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.version.serialized_size(compress)
            + self.algebra.serialized_size(compress)
            + self.setup.serialized_size(compress)
            + self.plan.serialized_size(compress)
            + self.grinding.serialized_size(compress)
            + self.call.serialized_size(compress)
    }
}

impl AkitaDeserialize for AkitaInstanceDescriptor {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &Self::Context,
    ) -> Result<Self, SerializationError> {
        let out = Self {
            version: u32::deserialize_with_mode(&mut reader, compress, validate, &())?,
            algebra: AlgebraSection::deserialize_with_mode(&mut reader, compress, validate, &())?,
            setup: SetupSection::deserialize_with_mode(&mut reader, compress, validate, &())?,
            plan: PlanSection::deserialize_with_mode(&mut reader, compress, validate, &())?,
            grinding: TranscriptGrindingBinding::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?,
            call: CallSection::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl Valid for AlgebraSection {
    fn check(&self) -> Result<(), SerializationError> {
        if self.field_extension_degree == 0 || self.extension_degree == 0 {
            return Err(SerializationError::InvalidData(
                "descriptor extension degrees must be non-zero".to_string(),
            ));
        }
        Ok(())
    }
}

impl AkitaSerialize for AlgebraSection {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        writer.write_all(&self.prime_modulus_be)?;
        self.field_extension_degree
            .serialize_with_mode(&mut writer, compress)?;
        self.extension_degree
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        32 + self.field_extension_degree.serialized_size(compress)
            + self.extension_degree.serialized_size(compress)
    }
}

impl AkitaDeserialize for AlgebraSection {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &Self::Context,
    ) -> Result<Self, SerializationError> {
        let mut prime_modulus_be = [0u8; 32];
        reader.read_exact(&mut prime_modulus_be)?;
        let out = Self {
            prime_modulus_be,
            field_extension_degree: u8::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?,
            extension_degree: u8::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl Valid for ProtocolFeatureSet {
    fn check(&self) -> Result<(), SerializationError> {
        if *self != Self::current() {
            return Err(SerializationError::InvalidData(
                "descriptor protocol features do not match active build".to_string(),
            ));
        }
        Ok(())
    }
}

impl AkitaSerialize for ProtocolFeatureSet {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.zk.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.zk.serialized_size(compress)
    }
}

impl AkitaDeserialize for ProtocolFeatureSet {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &Self::Context,
    ) -> Result<Self, SerializationError> {
        let out = Self {
            zk: bool::deserialize_with_mode(reader, compress, validate, &())?,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl Valid for SetupSection {
    fn check(&self) -> Result<(), SerializationError> {
        // Covers `log_basis`, the field width, and the committed source bound
        // (`1 <= log_commit_bound <= field_bits`) in one place, so a bounded
        // source cannot arrive with a bound the digit math cannot represent.
        self.decomposition
            .validate()
            .map_err(|error| SerializationError::InvalidData(error.to_string()))?;
        if self.compression_policy != COMPRESSION_POLICY {
            return Err(SerializationError::InvalidData(
                "descriptor compression policy does not match active protocol cutover".to_string(),
            ));
        }
        self.protocol_features.check()?;
        Ok(())
    }
}

impl AkitaSerialize for SetupSection {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        encode_decomposition(&self.decomposition, &mut writer, compress)?;
        encode_sis_modulus_profile(self.sis_modulus_profile, &mut writer, compress)?;
        self.compression_policy
            .tag()
            .serialize_with_mode(&mut writer, compress)?;
        writer.write_all(&self.setup_seed_digest)?;
        self.protocol_features
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        decomposition_size(&self.decomposition, compress)
            + sis_modulus_profile_size(compress)
            + 1
            + 32
            + self.protocol_features.serialized_size(compress)
    }
}

impl AkitaDeserialize for SetupSection {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &Self::Context,
    ) -> Result<Self, SerializationError> {
        let decomposition = decode_decomposition(&mut reader, compress, validate)?;
        let sis_modulus_profile = decode_sis_modulus_profile(&mut reader, compress, validate)?;
        let compression_policy_tag =
            u8::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let compression_policy =
            CompressionPolicyId::from_tag(compression_policy_tag).ok_or_else(|| {
                SerializationError::InvalidData(format!(
                    "unsupported compression policy tag {compression_policy_tag}"
                ))
            })?;
        let setup_seed_digest = read_digest(&mut reader)?;
        let protocol_features =
            ProtocolFeatureSet::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self {
            decomposition,
            sis_modulus_profile,
            compression_policy,
            setup_seed_digest,
            protocol_features,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl Valid for PlanSection {
    fn check(&self) -> Result<(), SerializationError> {
        self.schedule_selection.check()
    }
}

impl AkitaSerialize for PlanSection {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.schedule_selection
            .serialize_with_mode(&mut writer, compress)?;
        writer.write_all(&self.effective_schedule_digest)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.schedule_selection.serialized_size(compress) + 32
    }
}

impl AkitaDeserialize for PlanSection {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &Self::Context,
    ) -> Result<Self, SerializationError> {
        let out = Self {
            schedule_selection: crate::OpeningScheduleSelection::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?,
            effective_schedule_digest: read_digest(&mut reader)?,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl Valid for CallSection {
    fn check(&self) -> Result<(), SerializationError> {
        if self.num_commitment_groups == 0 {
            return Err(SerializationError::InvalidData(
                "descriptor call counts must be non-zero".to_string(),
            ));
        }
        if self.num_vars_per_commitment_group.len() != self.num_commitment_groups as usize {
            return Err(SerializationError::InvalidData(
                "descriptor group arity count mismatch".to_string(),
            ));
        }
        if self.num_polys_per_commitment_group.len() != self.num_commitment_groups as usize {
            return Err(SerializationError::InvalidData(
                "descriptor commitment-group count mismatch".to_string(),
            ));
        }
        if self.num_polys_per_commitment_group.contains(&0) {
            return Err(SerializationError::InvalidData(
                "descriptor commitment groups must be non-empty".to_string(),
            ));
        }
        Ok(())
    }
}

impl AkitaSerialize for CallSection {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.num_commitment_groups
            .serialize_with_mode(&mut writer, compress)?;
        let arity_count =
            u32::try_from(self.num_vars_per_commitment_group.len()).map_err(|_| {
                SerializationError::InvalidData(
                    "descriptor group arity vector length does not fit u32".to_string(),
                )
            })?;
        arity_count.serialize_with_mode(&mut writer, compress)?;
        for &num_vars in &self.num_vars_per_commitment_group {
            num_vars.serialize_with_mode(&mut writer, compress)?;
        }
        let group_count =
            u32::try_from(self.num_polys_per_commitment_group.len()).map_err(|_| {
                SerializationError::InvalidData(
                    "descriptor commitment-group vector length does not fit u32".to_string(),
                )
            })?;
        group_count.serialize_with_mode(&mut writer, compress)?;
        for &group_size in &self.num_polys_per_commitment_group {
            group_size.serialize_with_mode(&mut writer, compress)?;
        }
        encode_basis_mode(self.basis_mode, &mut writer, compress)?;
        writer.write_all(&self.opening_batch_digest)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.num_commitment_groups.serialized_size(compress)
            + 0u32.serialized_size(compress)
            + self
                .num_vars_per_commitment_group
                .iter()
                .map(|num_vars| num_vars.serialized_size(compress))
                .sum::<usize>()
            + 0u32.serialized_size(compress)
            + self
                .num_polys_per_commitment_group
                .iter()
                .map(|group_size| group_size.serialized_size(compress))
                .sum::<usize>()
            + basis_mode_size(compress)
            + 32
    }
}

impl AkitaDeserialize for CallSection {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &Self::Context,
    ) -> Result<Self, SerializationError> {
        let num_commitment_groups =
            u32::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let expected_count = usize::try_from(num_commitment_groups).map_err(|_| {
            SerializationError::InvalidData(
                "descriptor commitment-group count does not fit usize".to_string(),
            )
        })?;
        if expected_count == 0 || expected_count > DEFAULT_MAX_SEQUENCE_LEN {
            return Err(SerializationError::InvalidData(format!(
                "descriptor commitment-group count {expected_count} is outside 1..={DEFAULT_MAX_SEQUENCE_LEN}"
            )));
        }
        let arity_count = u32::deserialize_with_mode(&mut reader, compress, validate, &())?;
        if arity_count != num_commitment_groups {
            return Err(SerializationError::InvalidData(
                "descriptor group arity count mismatch".to_string(),
            ));
        }
        let mut num_vars_per_commitment_group = Vec::new();
        num_vars_per_commitment_group
            .try_reserve_exact(expected_count)
            .map_err(|_| {
                SerializationError::InvalidData(
                    "descriptor group arity allocation failed".to_string(),
                )
            })?;
        for _ in 0..expected_count {
            num_vars_per_commitment_group.push(u32::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?);
        }
        let group_count = u32::deserialize_with_mode(&mut reader, compress, validate, &())?;
        if group_count != num_commitment_groups {
            return Err(SerializationError::InvalidData(
                "descriptor commitment-group count mismatch".to_string(),
            ));
        }
        let mut num_polys_per_commitment_group = Vec::new();
        num_polys_per_commitment_group
            .try_reserve_exact(expected_count)
            .map_err(|_| {
                SerializationError::InvalidData(
                    "descriptor commitment-group allocation failed".to_string(),
                )
            })?;
        for _ in 0..expected_count {
            num_polys_per_commitment_group.push(u32::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?);
        }
        let out = Self {
            num_commitment_groups,
            num_vars_per_commitment_group,
            num_polys_per_commitment_group,
            basis_mode: decode_basis_mode(&mut reader, compress, validate)?,
            opening_batch_digest: read_digest(&mut reader)?,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

fn modulus_be_32<F: Field + CanonicalEncoding>() -> Result<[u8; 32], AkitaError> {
    crate::field_modulus_be_bytes::<F>()
}

fn blake2b_256(bytes: &[u8]) -> DescriptorDigest {
    type Blake2b256 = Blake2b<U32>;
    let digest = Blake2b256::digest(bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

/// Hash canonical descriptor bytes with Akita's Blake2b-256 primitive.
///
/// Domain separation and version bytes are owned by the caller's canonical
/// descriptor. This shared primitive prevents catalog and transcript identity
/// code from implementing divergent hash truncation rules.
pub fn digest_descriptor_bytes(bytes: &[u8]) -> DescriptorDigest {
    blake2b_256(bytes)
}

fn read_digest<R: Read>(mut reader: R) -> Result<DescriptorDigest, SerializationError> {
    let mut digest = [0u8; 32];
    reader.read_exact(&mut digest)?;
    Ok(digest)
}

fn encode_decomposition<W: Write>(
    decomp: &DecompositionParams,
    mut writer: W,
    compress: Compress,
) -> Result<(), SerializationError> {
    decomp
        .log_basis
        .serialize_with_mode(&mut writer, compress)?;
    decomp
        .log_commit_bound
        .serialize_with_mode(&mut writer, compress)?;
    decomp
        .log_open_bound
        .is_some()
        .serialize_with_mode(&mut writer, compress)?;
    if let Some(log_open_bound) = decomp.log_open_bound {
        log_open_bound.serialize_with_mode(&mut writer, compress)?;
    }
    Ok(())
}

fn decode_decomposition<R: Read>(
    mut reader: R,
    compress: Compress,
    validate: Validate,
) -> Result<DecompositionParams, SerializationError> {
    let log_basis = u32::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let log_commit_bound = u32::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let has_log_open_bound = bool::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let log_open_bound = if has_log_open_bound {
        Some(u32::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &(),
        )?)
    } else {
        None
    };
    Ok(DecompositionParams {
        log_basis,
        log_commit_bound,
        log_open_bound,
    })
}

fn decomposition_size(decomp: &DecompositionParams, compress: Compress) -> usize {
    let mut size = 0u32.serialized_size(compress)
        + 0u32.serialized_size(compress)
        + false.serialized_size(compress);
    if decomp.log_open_bound.is_some() {
        size += 0u32.serialized_size(compress);
    }
    size
}

fn encode_sis_modulus_profile<W: Write>(
    family: SisModulusProfileId,
    writer: W,
    compress: Compress,
) -> Result<(), SerializationError> {
    sis_modulus_profile_tag(family).serialize_with_mode(writer, compress)
}

fn decode_sis_modulus_profile<R: Read>(
    reader: R,
    compress: Compress,
    validate: Validate,
) -> Result<SisModulusProfileId, SerializationError> {
    match u8::deserialize_with_mode(reader, compress, validate, &())? {
        0 => Ok(SisModulusProfileId::Q32Offset99),
        1 => Ok(SisModulusProfileId::Q64Offset59),
        2 => Ok(SisModulusProfileId::Q128OffsetA7F7),
        other => Err(SerializationError::InvalidData(format!(
            "unknown SisModulusProfileId tag {other}"
        ))),
    }
}

fn sis_modulus_profile_size(compress: Compress) -> usize {
    0u8.serialized_size(compress)
}

fn encode_basis_mode<W: Write>(
    basis: BasisMode,
    writer: W,
    compress: Compress,
) -> Result<(), SerializationError> {
    basis_mode_tag(basis).serialize_with_mode(writer, compress)
}

fn decode_basis_mode<R: Read>(
    reader: R,
    compress: Compress,
    validate: Validate,
) -> Result<BasisMode, SerializationError> {
    match u8::deserialize_with_mode(reader, compress, validate, &())? {
        0 => Ok(BasisMode::Lagrange),
        1 => Ok(BasisMode::Monomial),
        other => Err(SerializationError::InvalidData(format!(
            "unknown BasisMode tag {other}"
        ))),
    }
}

fn basis_mode_tag(basis: BasisMode) -> u8 {
    match basis {
        BasisMode::Lagrange => 0,
        BasisMode::Monomial => 1,
    }
}

fn basis_mode_size(compress: Compress) -> usize {
    0u8.serialized_size(compress)
}
