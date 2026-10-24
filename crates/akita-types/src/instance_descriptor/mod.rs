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
//!
//! ## Word-aligned layout
//!
//! Version 6 encodes every scalar field, including flags and small tags, as
//! one `u64` little-endian word, and every 32-byte digest or modulus as its
//! bare bytes. A variable-length vector is a length word followed by one word
//! per element. The descriptor is therefore a whole number of words.

#[cfg(test)]
mod tests;
mod transcript_grinding_binding;

pub use transcript_grinding_binding::TranscriptGrindingBinding;

use crate::AkitaSetupSeed;
use akita_error::{
    narrowing::{usize_to_u32, usize_to_u8},
    AkitaError,
};
use akita_params::descriptor_bytes::{
    digest_descriptor_bytes, sis_modulus_profile_tag, DescriptorDigest,
    AKITA_INSTANCE_DESCRIPTOR_VERSION,
};
use akita_params::{
    BasisMode, CompressionPolicyId, DecompositionParams, FoldSchedule, OpeningClaimsLayout,
    SisModulusProfileId, COMPRESSION_POLICY,
};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
    DEFAULT_MAX_SEQUENCE_LEN,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};
use std::io::{Read, Write};

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
    pub schedule_selection: akita_params::OpeningScheduleSelection,
    /// Digest of the final effective verifier schedule.
    pub effective_schedule_digest: DescriptorDigest,
}

impl PlanSection {
    /// Build a plan section from the runtime schedule the verifier will replay.
    pub fn from_schedule(
        schedule_selection: akita_params::OpeningScheduleSelection,
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
    Ok(digest_descriptor_bytes(&bytes))
}

/// Digest the final effective runtime verifier schedule.
pub fn digest_effective_schedule(schedule: &FoldSchedule) -> DescriptorDigest {
    let mut bytes = Vec::new();
    schedule.append_descriptor_bytes(&mut bytes);
    digest_descriptor_bytes(&bytes)
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
        write_word(&mut writer, u64::from(self.version))?;
        self.algebra.serialize_with_mode(&mut writer, compress)?;
        self.setup.serialize_with_mode(&mut writer, compress)?;
        self.plan.serialize_with_mode(&mut writer, compress)?;
        self.grinding.serialize_with_mode(&mut writer, compress)?;
        self.call.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        WORD_BYTES
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
            version: read_word_as(&mut reader, "descriptor version")?,
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
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        writer.write_all(&self.prime_modulus_be)?;
        write_word(&mut writer, u64::from(self.field_extension_degree))?;
        write_word(&mut writer, u64::from(self.extension_degree))?;
        Ok(())
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        32 + 2 * WORD_BYTES
    }
}

impl AkitaDeserialize for AlgebraSection {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        validate: Validate,
        _ctx: &Self::Context,
    ) -> Result<Self, SerializationError> {
        let mut prime_modulus_be = [0u8; 32];
        reader.read_exact(&mut prime_modulus_be)?;
        let out = Self {
            prime_modulus_be,
            field_extension_degree: read_word_as(&mut reader, "field extension degree")?,
            extension_degree: read_word_as(&mut reader, "extension degree")?,
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
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        write_word(writer, u64::from(self.zk))
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        WORD_BYTES
    }
}

impl AkitaDeserialize for ProtocolFeatureSet {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        reader: R,
        _compress: Compress,
        validate: Validate,
        _ctx: &Self::Context,
    ) -> Result<Self, SerializationError> {
        let out = Self {
            zk: read_bool_word(reader, "zk feature flag")?,
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
        encode_decomposition(&self.decomposition, &mut writer)?;
        write_word(
            &mut writer,
            u64::from(sis_modulus_profile_tag(self.sis_modulus_profile)),
        )?;
        write_word(&mut writer, u64::from(self.compression_policy.tag()))?;
        writer.write_all(&self.setup_seed_digest)?;
        self.protocol_features
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        decomposition_size(&self.decomposition)
            + 2 * WORD_BYTES
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
        let decomposition = decode_decomposition(&mut reader)?;
        let sis_modulus_profile = decode_sis_modulus_profile(&mut reader)?;
        let compression_policy_tag: u8 = read_word_as(&mut reader, "compression policy tag")?;
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
            schedule_selection: akita_params::OpeningScheduleSelection::deserialize_with_mode(
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
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        write_word(&mut writer, u64::from(self.num_commitment_groups))?;
        write_word_vector(&mut writer, &self.num_vars_per_commitment_group)?;
        write_word_vector(&mut writer, &self.num_polys_per_commitment_group)?;
        write_word(&mut writer, u64::from(basis_mode_tag(self.basis_mode)))?;
        writer.write_all(&self.opening_batch_digest)?;
        Ok(())
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        WORD_BYTES
            + WORD_BYTES * (1 + self.num_vars_per_commitment_group.len())
            + WORD_BYTES * (1 + self.num_polys_per_commitment_group.len())
            + WORD_BYTES
            + 32
    }
}

impl AkitaDeserialize for CallSection {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        validate: Validate,
        _ctx: &Self::Context,
    ) -> Result<Self, SerializationError> {
        let num_commitment_groups: u32 = read_word_as(&mut reader, "commitment-group count")?;
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
        let num_vars_per_commitment_group =
            read_word_vector(&mut reader, expected_count, "group arity")?;
        let num_polys_per_commitment_group =
            read_word_vector(&mut reader, expected_count, "commitment-group size")?;
        let out = Self {
            num_commitment_groups,
            num_vars_per_commitment_group,
            num_polys_per_commitment_group,
            basis_mode: decode_basis_mode(&mut reader)?,
            opening_batch_digest: read_digest(&mut reader)?,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

fn modulus_be_32<F: Field + CanonicalEncoding>() -> Result<[u8; 32], AkitaError> {
    akita_params::field_modulus_be_bytes::<F>()
}

fn read_digest<R: Read>(mut reader: R) -> Result<DescriptorDigest, SerializationError> {
    let mut digest = [0u8; 32];
    reader.read_exact(&mut digest)?;
    Ok(digest)
}

/// Bytes in one descriptor word.
const WORD_BYTES: usize = 8;

/// Write one descriptor word: `value` as a `u64` little-endian.
fn write_word<W: Write>(mut writer: W, value: u64) -> Result<(), SerializationError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

/// Read one descriptor word and narrow it to `T`, rejecting a value `T`
/// cannot hold.
fn read_word_as<T: TryFrom<u64>, R: Read>(
    mut reader: R,
    field: &str,
) -> Result<T, SerializationError> {
    let mut word = [0u8; WORD_BYTES];
    reader.read_exact(&mut word)?;
    let value = u64::from_le_bytes(word);
    T::try_from(value).map_err(|_| {
        SerializationError::InvalidData(format!("descriptor {field} word {value} is out of range"))
    })
}

/// Read one descriptor flag word, accepting only 0 or 1.
fn read_bool_word<R: Read>(reader: R, field: &str) -> Result<bool, SerializationError> {
    match read_word_as::<u64, _>(reader, field)? {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(SerializationError::InvalidData(format!(
            "descriptor {field} word {other} is not a flag"
        ))),
    }
}

/// Write a vector as its length word, then one word per element.
fn write_word_vector<W: Write>(mut writer: W, values: &[u32]) -> Result<(), SerializationError> {
    write_word(&mut writer, values.len() as u64)?;
    for &value in values {
        write_word(&mut writer, u64::from(value))?;
    }
    Ok(())
}

/// Read a vector written by [`write_word_vector`] whose length must equal
/// `expected_len`, checked before allocating.
fn read_word_vector<R: Read>(
    mut reader: R,
    expected_len: usize,
    field: &str,
) -> Result<Vec<u32>, SerializationError> {
    let len: u64 = read_word_as(&mut reader, field)?;
    if usize::try_from(len).ok() != Some(expected_len) {
        return Err(SerializationError::InvalidData(format!(
            "descriptor {field} count mismatch"
        )));
    }
    let mut values = Vec::new();
    values.try_reserve_exact(expected_len).map_err(|_| {
        SerializationError::InvalidData(format!("descriptor {field} allocation failed"))
    })?;
    for _ in 0..expected_len {
        values.push(read_word_as(&mut reader, field)?);
    }
    Ok(values)
}

fn encode_decomposition<W: Write>(
    decomp: &DecompositionParams,
    mut writer: W,
) -> Result<(), SerializationError> {
    write_word(&mut writer, u64::from(decomp.log_basis))?;
    write_word(&mut writer, u64::from(decomp.log_commit_bound))?;
    write_word(&mut writer, u64::from(decomp.log_open_bound.is_some()))?;
    if let Some(log_open_bound) = decomp.log_open_bound {
        write_word(&mut writer, u64::from(log_open_bound))?;
    }
    Ok(())
}

fn decode_decomposition<R: Read>(mut reader: R) -> Result<DecompositionParams, SerializationError> {
    let log_basis = read_word_as(&mut reader, "log basis")?;
    let log_commit_bound = read_word_as(&mut reader, "log commit bound")?;
    let log_open_bound = if read_bool_word(&mut reader, "log open bound presence")? {
        Some(read_word_as(&mut reader, "log open bound")?)
    } else {
        None
    };
    Ok(DecompositionParams {
        log_basis,
        log_commit_bound,
        log_open_bound,
    })
}

fn decomposition_size(decomp: &DecompositionParams) -> usize {
    WORD_BYTES * (3 + usize::from(decomp.log_open_bound.is_some()))
}

fn decode_sis_modulus_profile<R: Read>(reader: R) -> Result<SisModulusProfileId, SerializationError> {
    match read_word_as::<u64, _>(reader, "SIS modulus profile tag")? {
        0 => Ok(SisModulusProfileId::Q32Offset99),
        1 => Ok(SisModulusProfileId::Q64Offset59),
        2 => Ok(SisModulusProfileId::Q128OffsetA7F7),
        other => Err(SerializationError::InvalidData(format!(
            "unknown SisModulusProfileId tag {other}"
        ))),
    }
}

fn decode_basis_mode<R: Read>(reader: R) -> Result<BasisMode, SerializationError> {
    match read_word_as::<u64, _>(reader, "basis mode tag")? {
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
