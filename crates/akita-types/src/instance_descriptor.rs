//! Fiat-Shamir instance descriptor bound into the transcript preamble.
//!
//! The descriptor is intentionally smaller than the proof or verifier setup:
//! large structured inputs are represented by Blake2b digests of canonical
//! Akita encodings. The top-level descriptor remains self-describing and
//! round-trippable so both prover and verifier can compare preamble bytes.

use crate::{
    detect_field_modulus, AkitaSetupSeed, BasisMode, ClaimIncidenceSummary, DecompositionParams,
    DirectWitnessShape, FlatMatrix, FoldStep, LevelParams, Schedule, SisModulusFamily, Step,
};
use akita_challenges::SparseChallengeConfig;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use blake2::{Blake2b512, Digest};
use std::io::{Read, Write};

/// Descriptor schema version for transcript-hardening v1.
pub const AKITA_INSTANCE_DESCRIPTOR_VERSION: u32 = 1;

/// Fixed-size Blake2b digest used inside the descriptor.
pub type DescriptorDigest = [u8; 32];

/// Canonical transcript preamble for one Akita proof instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaInstanceDescriptor {
    /// Schema version.
    pub version: u32,
    /// Algebraic substrate for this binary/proof family.
    pub algebra: AlgebraSection,
    /// Setup-bound parameters and setup artifact identities.
    pub setup: SetupSection,
    /// Final effective verifier schedule for this proof.
    pub plan: PlanSection,
    /// Per-call public shape and batching data.
    pub call: CallSection,
}

impl AkitaInstanceDescriptor {
    /// Construct a version-1 descriptor from its four sections.
    pub fn new(
        algebra: AlgebraSection,
        setup: SetupSection,
        plan: PlanSection,
        call: CallSection,
    ) -> Self {
        Self {
            version: AKITA_INSTANCE_DESCRIPTOR_VERSION,
            algebra,
            setup,
            plan,
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

/// Algebraic substrate that determines the ring and field towers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlgebraSection {
    /// Characteristic `p` of the base prime field, big-endian and 32-byte
    /// padded.
    pub prime_modulus_be: [u8; 32],
    /// Cyclotomic index `D` defining the ring.
    pub ring_dimension_d: u32,
    /// Extension degree of the message field over the base prime field.
    pub field_extension_degree: u8,
    /// Extension degree of the claim field over the base prime field.
    pub claim_extension_degree: u8,
    /// Extension degree of the challenge field over the base prime field.
    pub challenge_extension_degree: u8,
}

impl AlgebraSection {
    /// Build the algebra section for base field `F`, claim field `L`, and
    /// challenge field `C` in cyclotomic ring dimension `D`.
    ///
    /// # Errors
    ///
    /// Returns an error if `D` or an extension degree does not fit the
    /// descriptor's fixed-width integer fields.
    pub fn for_fields<F, L, C, const D: usize>() -> Result<Self, AkitaError>
    where
        F: CanonicalField,
        L: ExtField<F>,
        C: ExtField<F>,
    {
        Ok(Self {
            prime_modulus_be: modulus_be_32::<F>(),
            ring_dimension_d: usize_to_u32(D, "ring dimension")?,
            field_extension_degree: usize_to_u8(1, "field extension degree")?,
            claim_extension_degree: usize_to_u8(L::EXT_DEGREE, "claim extension degree")?,
            challenge_extension_degree: usize_to_u8(C::EXT_DEGREE, "challenge extension degree")?,
        })
    }
}

/// Compile-time features that change protocol transcript behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProtocolFeatureSet {
    /// Whether the `zk` feature is active.
    pub zk: bool,
}

impl ProtocolFeatureSet {
    /// Return the protocol feature set of the current build.
    #[inline]
    pub const fn current() -> Self {
        Self {
            zk: cfg!(feature = "zk"),
        }
    }
}

/// Setup-bound descriptor fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupSection {
    /// Gadget decomposition parameters.
    pub decomposition: DecompositionParams,
    /// SIS modulus family used for security sizing.
    pub sis_modulus_family: SisModulusFamily,
    /// Digest of the canonical `AkitaSetupSeed` bytes.
    pub setup_seed_digest: DescriptorDigest,
    /// Digest of the canonical expanded shared-matrix artifact bytes.
    pub shared_matrix_digest: DescriptorDigest,
    /// Protocol-affecting feature mode.
    pub protocol_features: ProtocolFeatureSet,
    /// Digest of the full `Vec<LevelParams>` envelope.
    pub level_params_digest: DescriptorDigest,
}

impl SetupSection {
    /// Build setup fields from existing setup/layout data.
    ///
    /// # Errors
    ///
    /// Returns a serialization error if any canonical digest input fails to
    /// serialize.
    pub fn from_parts<F>(
        decomposition: DecompositionParams,
        sis_modulus_family: SisModulusFamily,
        setup_seed: &AkitaSetupSeed,
        shared_matrix: &FlatMatrix<F>,
        level_params: &[LevelParams],
    ) -> Result<Self, SerializationError>
    where
        F: FieldCore + AkitaSerialize,
    {
        Ok(Self {
            decomposition,
            sis_modulus_family,
            setup_seed_digest: digest_serializable(setup_seed)?,
            shared_matrix_digest: digest_serializable(shared_matrix)?,
            protocol_features: ProtocolFeatureSet::current(),
            level_params_digest: digest_level_params(level_params),
        })
    }
}

/// Per-proof effective schedule binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanSection {
    /// Digest of the final effective verifier schedule.
    pub effective_schedule_digest: DescriptorDigest,
}

impl PlanSection {
    /// Build a plan section from the runtime schedule the verifier will replay.
    pub fn from_schedule(schedule: &Schedule) -> Self {
        Self {
            effective_schedule_digest: digest_effective_schedule(schedule),
        }
    }
}

/// Per commit-and-open call descriptor fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallSection {
    /// Number of distinct opening points.
    pub num_points: u32,
    /// Total number of committed polynomials addressed by the call.
    pub num_polys: u32,
    /// Total number of claimed openings addressed by the call.
    pub num_claims: u32,
    /// Public basis mode for opening-point weights.
    pub basis_mode: BasisMode,
    /// Common opening-point arity.
    pub opening_point_arity: u32,
    /// Digest of normalized batch incidence.
    pub incidence_digest: DescriptorDigest,
}

impl CallSection {
    /// Build call fields from normalized public incidence.
    ///
    /// # Errors
    ///
    /// Returns an error if a count does not fit the descriptor's fixed-width
    /// integer fields.
    pub fn from_incidence(
        incidence: &ClaimIncidenceSummary,
        basis_mode: BasisMode,
    ) -> Result<Self, AkitaError> {
        Ok(Self {
            num_points: usize_to_u32(incidence.num_points(), "num_points")?,
            num_polys: usize_to_u32(incidence.num_polynomials(), "num_polys")?,
            num_claims: usize_to_u32(incidence.num_claims(), "num_claims")?,
            basis_mode,
            opening_point_arity: usize_to_u32(incidence.num_vars(), "opening_point_arity")?,
            incidence_digest: digest_incidence(incidence),
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

/// Digest the normalized claim incidence summary.
pub fn digest_incidence(summary: &ClaimIncidenceSummary) -> DescriptorDigest {
    let mut bytes = Vec::new();
    push_usize(&mut bytes, summary.num_vars());
    push_usize_vec(&mut bytes, summary.num_polys_per_point());
    push_usize_vec(&mut bytes, summary.claim_to_point());
    push_usize_vec(&mut bytes, summary.claim_poly_indices());
    push_usize(&mut bytes, summary.public_rows().len());
    for row in summary.public_rows() {
        push_usize(&mut bytes, row.point_idx());
        push_usize_vec(&mut bytes, row.claim_indices());
    }
    blake2b_256(&bytes)
}

/// Digest the full `LevelParams` vector used by a setup/config envelope.
pub fn digest_level_params(levels: &[LevelParams]) -> DescriptorDigest {
    let mut bytes = Vec::new();
    push_usize(&mut bytes, levels.len());
    for lp in levels {
        encode_level_params(&mut bytes, lp);
    }
    blake2b_256(&bytes)
}

/// Digest the final effective runtime verifier schedule.
pub fn digest_effective_schedule(schedule: &Schedule) -> DescriptorDigest {
    let mut bytes = Vec::new();
    push_usize(&mut bytes, schedule.steps.len());
    for step in &schedule.steps {
        match step {
            Step::Fold(fold) => {
                bytes.push(0);
                encode_fold_step(&mut bytes, fold);
            }
            Step::Direct(direct) => {
                bytes.push(1);
                push_usize(&mut bytes, direct.current_w_len);
                encode_direct_witness_shape(&mut bytes, &direct.witness_shape);
                push_usize(&mut bytes, direct.direct_bytes);
            }
        }
    }
    push_usize(&mut bytes, schedule.total_bytes);
    blake2b_256(&bytes)
}

impl Valid for AkitaInstanceDescriptor {
    fn check(&self) -> Result<(), SerializationError> {
        self.algebra.check()?;
        self.setup.check()?;
        self.plan.check()?;
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
        self.call.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.version.serialized_size(compress)
            + self.algebra.serialized_size(compress)
            + self.setup.serialized_size(compress)
            + self.plan.serialized_size(compress)
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
        if self.ring_dimension_d == 0 {
            return Err(SerializationError::InvalidData(
                "descriptor ring dimension must be non-zero".to_string(),
            ));
        }
        if self.field_extension_degree == 0
            || self.claim_extension_degree == 0
            || self.challenge_extension_degree == 0
        {
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
        self.ring_dimension_d
            .serialize_with_mode(&mut writer, compress)?;
        self.field_extension_degree
            .serialize_with_mode(&mut writer, compress)?;
        self.claim_extension_degree
            .serialize_with_mode(&mut writer, compress)?;
        self.challenge_extension_degree
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        32 + self.ring_dimension_d.serialized_size(compress)
            + self.field_extension_degree.serialized_size(compress)
            + self.claim_extension_degree.serialized_size(compress)
            + self.challenge_extension_degree.serialized_size(compress)
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
            ring_dimension_d: u32::deserialize_with_mode(&mut reader, compress, validate, &())?,
            field_extension_degree: u8::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?,
            claim_extension_degree: u8::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?,
            challenge_extension_degree: u8::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl Valid for ProtocolFeatureSet {
    fn check(&self) -> Result<(), SerializationError> {
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
        if self.decomposition.log_basis == 0 {
            return Err(SerializationError::InvalidData(
                "descriptor log_basis must be non-zero".to_string(),
            ));
        }
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
        encode_sis_family(self.sis_modulus_family, &mut writer, compress)?;
        writer.write_all(&self.setup_seed_digest)?;
        writer.write_all(&self.shared_matrix_digest)?;
        self.protocol_features
            .serialize_with_mode(&mut writer, compress)?;
        writer.write_all(&self.level_params_digest)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        decomposition_size(&self.decomposition, compress)
            + sis_family_size(compress)
            + 32
            + 32
            + self.protocol_features.serialized_size(compress)
            + 32
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
        let sis_modulus_family = decode_sis_family(&mut reader, compress, validate)?;
        let setup_seed_digest = read_digest(&mut reader)?;
        let shared_matrix_digest = read_digest(&mut reader)?;
        let protocol_features =
            ProtocolFeatureSet::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let level_params_digest = read_digest(&mut reader)?;
        let out = Self {
            decomposition,
            sis_modulus_family,
            setup_seed_digest,
            shared_matrix_digest,
            protocol_features,
            level_params_digest,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl Valid for PlanSection {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl AkitaSerialize for PlanSection {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        writer.write_all(&self.effective_schedule_digest)?;
        Ok(())
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        32
    }
}

impl AkitaDeserialize for PlanSection {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        validate: Validate,
        _ctx: &Self::Context,
    ) -> Result<Self, SerializationError> {
        let out = Self {
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
        if self.num_points == 0 || self.num_polys == 0 || self.num_claims == 0 {
            return Err(SerializationError::InvalidData(
                "descriptor call counts must be non-zero".to_string(),
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
        self.num_points.serialize_with_mode(&mut writer, compress)?;
        self.num_polys.serialize_with_mode(&mut writer, compress)?;
        self.num_claims.serialize_with_mode(&mut writer, compress)?;
        encode_basis_mode(self.basis_mode, &mut writer, compress)?;
        self.opening_point_arity
            .serialize_with_mode(&mut writer, compress)?;
        writer.write_all(&self.incidence_digest)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.num_points.serialized_size(compress)
            + self.num_polys.serialized_size(compress)
            + self.num_claims.serialized_size(compress)
            + basis_mode_size(compress)
            + self.opening_point_arity.serialized_size(compress)
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
        let out = Self {
            num_points: u32::deserialize_with_mode(&mut reader, compress, validate, &())?,
            num_polys: u32::deserialize_with_mode(&mut reader, compress, validate, &())?,
            num_claims: u32::deserialize_with_mode(&mut reader, compress, validate, &())?,
            basis_mode: decode_basis_mode(&mut reader, compress, validate)?,
            opening_point_arity: u32::deserialize_with_mode(&mut reader, compress, validate, &())?,
            incidence_digest: read_digest(&mut reader)?,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

fn modulus_be_32<F: CanonicalField>() -> [u8; 32] {
    let modulus = detect_field_modulus::<F>();
    let mut out = [0u8; 32];
    out[16..].copy_from_slice(&modulus.to_be_bytes());
    out
}

fn usize_to_u32(value: usize, name: &'static str) -> Result<u32, AkitaError> {
    u32::try_from(value).map_err(|_| AkitaError::InvalidInput(format!("{name} does not fit u32")))
}

fn usize_to_u8(value: usize, name: &'static str) -> Result<u8, AkitaError> {
    u8::try_from(value).map_err(|_| AkitaError::InvalidInput(format!("{name} does not fit u8")))
}

fn blake2b_256(bytes: &[u8]) -> DescriptorDigest {
    let digest = Blake2b512::digest(bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest[..32]);
    out
}

fn read_digest<R: Read>(mut reader: R) -> Result<DescriptorDigest, SerializationError> {
    let mut digest = [0u8; 32];
    reader.read_exact(&mut digest)?;
    Ok(digest)
}

fn push_usize(bytes: &mut Vec<u8>, value: usize) {
    bytes.extend_from_slice(&(value as u64).to_le_bytes());
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_i8(bytes: &mut Vec<u8>, value: i8) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_usize_vec(bytes: &mut Vec<u8>, values: &[usize]) {
    push_usize(bytes, values.len());
    for &value in values {
        push_usize(bytes, value);
    }
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

fn encode_sis_family<W: Write>(
    family: SisModulusFamily,
    writer: W,
    compress: Compress,
) -> Result<(), SerializationError> {
    sis_family_tag(family).serialize_with_mode(writer, compress)
}

fn decode_sis_family<R: Read>(
    reader: R,
    compress: Compress,
    validate: Validate,
) -> Result<SisModulusFamily, SerializationError> {
    match u8::deserialize_with_mode(reader, compress, validate, &())? {
        0 => Ok(SisModulusFamily::Q32),
        1 => Ok(SisModulusFamily::Q64),
        2 => Ok(SisModulusFamily::Q128),
        other => Err(SerializationError::InvalidData(format!(
            "unknown SisModulusFamily tag {other}"
        ))),
    }
}

fn sis_family_tag(family: SisModulusFamily) -> u8 {
    match family {
        SisModulusFamily::Q32 => 0,
        SisModulusFamily::Q64 => 1,
        SisModulusFamily::Q128 => 2,
    }
}

fn sis_family_size(compress: Compress) -> usize {
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

fn encode_level_params(bytes: &mut Vec<u8>, lp: &LevelParams) {
    push_usize(bytes, lp.ring_dimension);
    push_u32(bytes, lp.log_basis);
    encode_ajtai_key(bytes, &lp.a_key);
    encode_ajtai_key(bytes, &lp.b_key);
    encode_ajtai_key(bytes, &lp.d_key);
    push_usize(bytes, lp.num_blocks);
    push_usize(bytes, lp.block_len);
    push_usize(bytes, lp.m_vars);
    push_usize(bytes, lp.r_vars);
    encode_sparse_challenge_config(bytes, &lp.stage1_config);
    push_usize(bytes, lp.num_digits_commit);
    push_usize(bytes, lp.num_digits_open);
    push_usize(bytes, lp.num_digits_fold);
}

fn encode_ajtai_key(bytes: &mut Vec<u8>, key: &crate::AjtaiKeyParams) {
    bytes.push(sis_family_tag(key.sis_family()));
    push_usize(bytes, key.row_len());
    push_usize(bytes, key.col_len());
    push_u32(bytes, key.collision_inf());
}

fn encode_sparse_challenge_config(bytes: &mut Vec<u8>, config: &SparseChallengeConfig) {
    match config {
        SparseChallengeConfig::Uniform {
            weight,
            nonzero_coeffs,
        } => {
            bytes.push(0);
            push_usize(bytes, *weight);
            push_usize(bytes, nonzero_coeffs.len());
            for &coeff in nonzero_coeffs {
                push_i8(bytes, coeff);
            }
        }
        SparseChallengeConfig::ExactShell {
            count_mag1,
            count_mag2,
        } => {
            bytes.push(1);
            push_usize(bytes, *count_mag1);
            push_usize(bytes, *count_mag2);
        }
        SparseChallengeConfig::BoundedL1Norm => {
            bytes.push(2);
        }
    }
}

fn encode_fold_step(bytes: &mut Vec<u8>, fold: &FoldStep) {
    encode_level_params(bytes, &fold.params);
    push_usize(bytes, fold.current_w_len);
    push_usize(bytes, fold.delta_fold_per_poly);
    push_usize(bytes, fold.w_ring);
    push_usize(bytes, fold.next_w_len);
    push_usize(bytes, fold.level_bytes);
}

fn encode_direct_witness_shape(bytes: &mut Vec<u8>, shape: &DirectWitnessShape) {
    match shape {
        DirectWitnessShape::PackedDigits((num_elems, bits_per_elem)) => {
            bytes.push(0);
            push_usize(bytes, *num_elems);
            push_u32(bytes, *bits_per_elem);
        }
        DirectWitnessShape::FieldElements(coeff_len) => {
            bytes.push(1);
            push_usize(bytes, *coeff_len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::{Prime32Offset99, Prime64Offset59};

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

    fn sample_descriptor() -> AkitaInstanceDescriptor {
        let incidence =
            ClaimIncidenceSummary::from_point_polys(5, vec![2, 1]).expect("valid incidence");
        let schedule = Schedule {
            steps: vec![
                Step::Fold(FoldStep {
                    params: sample_level_params(),
                    current_w_len: 256,
                    delta_fold_per_poly: 3,
                    w_ring: 8,
                    next_w_len: 256,
                    level_bytes: 123,
                }),
                Step::Direct(crate::DirectStep {
                    current_w_len: 256,
                    witness_shape: DirectWitnessShape::PackedDigits((64, 3)),
                    direct_bytes: 32,
                }),
            ],
            total_bytes: 155,
        };

        AkitaInstanceDescriptor::new(
            AlgebraSection::for_fields::<Prime32Offset99, Prime32Offset99, Prime32Offset99, 32>()
                .expect("algebra"),
            SetupSection {
                decomposition: DecompositionParams {
                    log_basis: 3,
                    log_commit_bound: 32,
                    log_open_bound: Some(32),
                },
                sis_modulus_family: SisModulusFamily::Q32,
                setup_seed_digest: [1; 32],
                shared_matrix_digest: [2; 32],
                protocol_features: ProtocolFeatureSet::current(),
                level_params_digest: digest_level_params(&[sample_level_params()]),
            },
            PlanSection::from_schedule(&schedule),
            CallSection::from_incidence(&incidence, BasisMode::Lagrange).expect("call"),
        )
    }

    #[test]
    fn canonical_encoding_roundtrip() {
        let descriptor = sample_descriptor();
        let bytes = descriptor.canonical_bytes().expect("serialize descriptor");
        assert_eq!(bytes.len(), descriptor.uncompressed_size());

        let decoded = AkitaInstanceDescriptor::deserialize_uncompressed(&bytes[..], &())
            .expect("deserialize descriptor");
        assert_eq!(decoded, descriptor);
    }

    #[test]
    fn algebra_section_binds_prime_and_extension_shape() {
        let fp32 =
            AlgebraSection::for_fields::<Prime32Offset99, Prime32Offset99, Prime32Offset99, 32>()
                .expect("fp32 algebra");
        let fp64 =
            AlgebraSection::for_fields::<Prime64Offset59, Prime64Offset59, Prime64Offset59, 32>()
                .expect("fp64 algebra");

        assert_ne!(fp32.prime_modulus_be, fp64.prime_modulus_be);
        assert_eq!(fp32.ring_dimension_d, 32);
        assert_eq!(fp32.field_extension_degree, 1);
        assert_eq!(fp32.claim_extension_degree, 1);
        assert_eq!(fp32.challenge_extension_degree, 1);
    }

    #[test]
    fn incidence_digest_binds_grouping_order() {
        let left = ClaimIncidenceSummary::from_point_polys(4, vec![2, 1]).expect("left");
        let right = ClaimIncidenceSummary::from_point_polys(4, vec![1, 2]).expect("right");

        assert_ne!(digest_incidence(&left), digest_incidence(&right));
    }

    #[test]
    fn effective_schedule_digest_binds_direct_shape() {
        let schedule_a = Schedule {
            steps: vec![Step::Direct(crate::DirectStep {
                current_w_len: 8,
                witness_shape: DirectWitnessShape::FieldElements(8),
                direct_bytes: 8,
            })],
            total_bytes: 8,
        };
        let schedule_b = Schedule {
            steps: vec![Step::Direct(crate::DirectStep {
                current_w_len: 8,
                witness_shape: DirectWitnessShape::PackedDigits((8, 3)),
                direct_bytes: 3,
            })],
            total_bytes: 3,
        };

        assert_ne!(
            digest_effective_schedule(&schedule_a),
            digest_effective_schedule(&schedule_b)
        );
    }
}
