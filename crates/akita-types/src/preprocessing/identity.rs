//! Identities of streams, commitment algorithms, profiles, and commitments.

use super::encoding::{
    hex, malformed, push_bytes, push_str, push_u32_le, push_u64, push_u8, Reader,
};
use crate::{AkitaSetupSeed, SetupPrefixPublicCommitment, SetupPrefixVerifierSlot};
use akita_error::AkitaError;
use akita_params::sis::{InnerCommitSecurityRoute, SisModulusProfileId};
use akita_params::{
    digest_descriptor_bytes, field_modulus_be_bytes, DescriptorHasher, GroupCommitPhaseParams,
    COMPRESSION_POLICY,
};
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Compress, Validate};
use jolt_field::{CanonicalEncoding, Field};

const STREAM_TAG: &str = "akita/preprocessing/stream/v1";
const PROFILE_TAG: &str = "akita/preprocessing/profile/v1";
const COMMITMENT_TAG: &str = "akita/preprocessing/commitment/v1";
pub(crate) const KNOWN_ANSWER_TAG: &str = "akita/preprocessing/known-answer/v1";

/// Version of the setup-prefix commitment payload encoding and its
/// computation. Bump it with any change to how a payload is computed.
pub const SETUP_PREFIX_PAYLOAD_VERSION: u32 = 1;

/// Version of the known-answer corpus whose outputs the compiled
/// known-answer digests pin.
pub const KNOWN_ANSWER_CORPUS_VERSION: u32 = 1;

macro_rules! digest_newtype {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name([u8; 32]);

        impl $name {
            /// Wrap raw digest bytes.
            #[must_use]
            pub const fn from_bytes(bytes: [u8; 32]) -> Self {
                Self(bytes)
            }

            /// Raw digest bytes.
            #[must_use]
            pub const fn as_bytes(&self) -> &[u8; 32] {
                &self.0
            }

            /// Lowercase hex, as used in file names.
            #[must_use]
            pub fn to_hex(&self) -> String {
                hex(&self.0)
            }
        }

        impl core::fmt::Debug for $name {
            fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(formatter, "{}({})", stringify!($name), self.to_hex())
            }
        }

        impl core::fmt::Display for $name {
            fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str(&self.to_hex())
            }
        }
    };
}

digest_newtype!(
    /// `H(tag || stream_identity)`: names one public field stream.
    StreamId
);
digest_newtype!(
    /// `H(tag || profile_descriptor)`: names one commitment profile.
    ProfileKey
);
digest_newtype!(
    /// Binds one setup-prefix commitment to its stream, algorithm, and
    /// profile. Distinct from the transcript's root `CommitmentDigest`.
    PrefixCommitmentDigest
);
digest_newtype!(
    /// `H(tag || str(name) || bytes(content))` of one release file.
    FileHash
);
digest_newtype!(
    /// `H(tag || manifest)`: the identity of one release.
    BundleId
);

/// The public field stream: field, seed, and page size.
///
/// Page `i` of the stream is a function of this identity and `i` alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamIdentity {
    prime_modulus_be: [u8; 32],
    setup_seed: AkitaSetupSeed,
}

impl StreamIdentity {
    /// Encoded length: modulus, serialized seed, and page size.
    pub const ENCODED_LEN: usize = 32 + 33 + 8;

    /// The stream of base field `F` under `setup_seed`.
    pub fn for_field<F: Field + CanonicalEncoding>(
        setup_seed: &AkitaSetupSeed,
    ) -> Result<Self, AkitaError> {
        Ok(Self {
            prime_modulus_be: field_modulus_be_bytes::<F>()?,
            setup_seed: setup_seed.clone(),
        })
    }

    /// Big-endian base field modulus, as in `AlgebraSection`.
    #[must_use]
    pub const fn prime_modulus_be(&self) -> &[u8; 32] {
        &self.prime_modulus_be
    }

    /// The seed that fixes the public matrix.
    #[must_use]
    pub const fn setup_seed(&self) -> &AkitaSetupSeed {
        &self.setup_seed
    }

    /// Field elements per independently derived page.
    #[must_use]
    pub const fn page_field_elements(&self) -> usize {
        self.setup_seed.derivation.page_field_elements()
    }

    /// `prime_modulus_be || ser(setup_seed) || u64(page_field_elements)`.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(Self::ENCODED_LEN);
        out.extend_from_slice(&self.prime_modulus_be);
        // The seed encoding is a one-byte tag and 32 bytes; writing to a
        // vector cannot fail.
        let _ = self.setup_seed.serialize_uncompressed(&mut out);
        push_u64(&mut out, self.page_field_elements() as u64);
        out
    }

    pub(crate) fn decode(reader: &mut Reader<'_>) -> Result<Self, AkitaError> {
        let bytes = reader.take(Self::ENCODED_LEN, "stream identity")?;
        let mut prime_modulus_be = [0u8; 32];
        prime_modulus_be.copy_from_slice(&bytes[..32]);
        let setup_seed =
            AkitaSetupSeed::deserialize_with_mode(&bytes[32..65], Compress::No, Validate::Yes, &())
                .map_err(|_| malformed("stream seed"))?;
        let out = Self {
            prime_modulus_be,
            setup_seed,
        };
        if out.to_bytes() != bytes {
            return Err(malformed("stream identity is not canonical"));
        }
        Ok(out)
    }

    /// `H(tag("akita/preprocessing/stream/v1") || stream_identity)`.
    #[must_use]
    pub fn stream_id(&self) -> StreamId {
        let mut input = Vec::with_capacity(STREAM_TAG.len() + 8 + Self::ENCODED_LEN);
        push_str(&mut input, STREAM_TAG);
        input.extend_from_slice(&self.to_bytes());
        StreamId(digest_descriptor_bytes(&input))
    }
}

/// How a commitment is computed from a profile and a stream.
///
/// Versions that only affect the transcript are not part of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlgorithmIdentity {
    compression_policy_tag: u8,
    group_profile_version: u8,
    payload_version: u32,
    known_answer_digest: [u8; 32],
}

impl AlgorithmIdentity {
    /// Encoded length.
    pub const ENCODED_LEN: usize = 1 + 1 + 4 + 32;

    /// The algorithm compiled into this binary for fields of `profile`.
    #[must_use]
    pub const fn compiled(profile: SisModulusProfileId) -> Self {
        Self {
            compression_policy_tag: COMPRESSION_POLICY.tag(),
            group_profile_version: GroupCommitPhaseParams::VERSION,
            payload_version: SETUP_PREFIX_PAYLOAD_VERSION,
            known_answer_digest: known_answer_digest(profile),
        }
    }

    /// The pinned known-answer digest.
    #[must_use]
    pub const fn known_answer_digest(&self) -> &[u8; 32] {
        &self.known_answer_digest
    }

    /// `u8(policy) || u8(profile version) || u32(payload version) || digest`.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(Self::ENCODED_LEN);
        push_u8(&mut out, self.compression_policy_tag);
        push_u8(&mut out, self.group_profile_version);
        push_u32_le(&mut out, self.payload_version);
        out.extend_from_slice(&self.known_answer_digest);
        out
    }

    pub(crate) fn decode(reader: &mut Reader<'_>) -> Result<Self, AkitaError> {
        Ok(Self {
            compression_policy_tag: reader.u8("algorithm policy")?,
            group_profile_version: reader.u8("algorithm profile version")?,
            payload_version: reader.u32("algorithm payload version")?,
            known_answer_digest: reader.digest("algorithm known-answer digest")?,
        })
    }
}

/// Known-answer digests pinned by the compiled commitment code, one per field.
///
/// Each is `H(tag || u32(corpus version) || test stream || list(profile ||
/// payload))` over the corpus entries of that field. The generator crate's
/// tests recompute them on every supported instruction set.
#[must_use]
pub const fn known_answer_digest(profile: SisModulusProfileId) -> [u8; 32] {
    match profile {
        SisModulusProfileId::Q32Offset99 => KNOWN_ANSWER_Q32,
        SisModulusProfileId::Q64Offset59 => KNOWN_ANSWER_Q64,
        SisModulusProfileId::Q128OffsetA7F7 => KNOWN_ANSWER_Q128,
    }
}

const KNOWN_ANSWER_Q32: [u8; 32] = super::encoding::digest_from_hex(
    "b9d89fbf6c8395bcde7efb35707d4206192bf8dcc542dfe7afa3fbee4bda2a2d",
);
const KNOWN_ANSWER_Q64: [u8; 32] = super::encoding::digest_from_hex(
    "60794d92812d0ca59a490a68dc6bdf5e10bd9aecb9c3456b9f65ca68d2413a64",
);
const KNOWN_ANSWER_Q128: [u8; 32] = super::encoding::digest_from_hex(
    "592ed04e776ff8c6e3b4ca85b83ddeff42d5c838dfba8ffd3112eaf871a6f9b0",
);

/// Inner security route byte of a profile descriptor.
#[must_use]
pub fn profile_route(profile: &GroupCommitPhaseParams) -> u8 {
    match profile.inner.matrix.security_route() {
        InnerCommitSecurityRoute::Linf(_) => 0,
        InnerCommitSecurityRoute::L2 { .. } => 1,
    }
}

/// `u8(route) || bytes(GroupCommitPhaseParams::canonical_descriptor_bytes())`.
#[must_use]
pub fn profile_descriptor(profile: &GroupCommitPhaseParams) -> Vec<u8> {
    let descriptor = profile.canonical_descriptor_bytes();
    let mut out = Vec::with_capacity(9 + descriptor.len());
    push_u8(&mut out, profile_route(profile));
    push_bytes(&mut out, &descriptor);
    out
}

/// `H(tag("akita/preprocessing/profile/v1") || profile_descriptor)`.
#[must_use]
pub fn profile_key(profile: &GroupCommitPhaseParams) -> ProfileKey {
    let mut input = Vec::new();
    push_str(&mut input, PROFILE_TAG);
    input.extend_from_slice(&profile_descriptor(profile));
    ProfileKey(digest_descriptor_bytes(&input))
}

/// `prefix_commitment_digest` over already encoded parts.
#[must_use]
pub fn prefix_commitment_digest(
    stream: &StreamIdentity,
    algorithm: &AlgorithmIdentity,
    profile_descriptor: &[u8],
    payload: &[u8],
) -> PrefixCommitmentDigest {
    let mut hasher = DescriptorHasher::new();
    let mut head = Vec::new();
    push_str(&mut head, COMMITMENT_TAG);
    head.extend_from_slice(&stream.to_bytes());
    head.extend_from_slice(&algorithm.to_bytes());
    push_bytes(&mut head, profile_descriptor);
    push_u64(&mut head, payload.len() as u64);
    hasher.update(&head);
    hasher.update(payload);
    PrefixCommitmentDigest(hasher.finalize())
}

/// `payload = ser(SetupPrefixPublicCommitment<F>)`, uncompressed.
pub fn setup_prefix_payload<F: Field + AkitaSerialize>(
    commitment: &SetupPrefixPublicCommitment<F>,
) -> Result<Vec<u8>, AkitaError> {
    let mut out = Vec::with_capacity(commitment.serialized_size(Compress::No));
    commitment
        .serialize_uncompressed(&mut out)
        .map_err(|error| AkitaError::InvalidSetup(format!("setup-prefix payload: {error}")))?;
    Ok(out)
}

/// Prefix commitment digest of one setup-prefix slot of the stream
/// `setup_seed` under the compiled commitment algorithm.
///
/// Prover and verifier recompute it at protocol Stage 3 from the registry
/// entry the fold reads; loaders compare it with the manifest.
pub fn slot_prefix_commitment_digest<F>(
    setup_seed: &AkitaSetupSeed,
    slot: &SetupPrefixVerifierSlot<F>,
) -> Result<PrefixCommitmentDigest, AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize,
{
    let profile = &slot.id.commitment_profile;
    let stream = StreamIdentity::for_field::<F>(setup_seed)?;
    let algorithm = AlgorithmIdentity::compiled(profile.inner.matrix.sis_modulus_profile());
    Ok(prefix_commitment_digest(
        &stream,
        &algorithm,
        &profile_descriptor(profile),
        &setup_prefix_payload(&slot.commitment)?,
    ))
}
