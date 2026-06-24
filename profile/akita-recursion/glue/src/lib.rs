//! Shared verifier-input blob shipped from a host artifact generator into a
//! Jolt guest program.
//!
//! The host serializes the bundle once (`AkitaJoltInputs::write_to_bytes`) and
//! the Jolt guest deserializes it as the very first step of the program.
//! Per-component encoding is the existing [`AkitaSerialize`] /
//! [`AkitaDeserialize`] machinery in [`akita_serialization`]. The recursion
//! benchmark can opt into an explicitly trusted cached-matrix setup decoder;
//! strict decoding remains the default.

#![allow(clippy::missing_errors_doc)]

use akita_field::{CanonicalField, FieldCore, RandomSampling};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use akita_types::{
    AkitaBatchedProof, AkitaBatchedProofShape, AkitaExpandedSetup, AkitaSetupSeed,
    AkitaVerifierSetup, CommitmentGroup, FlatMatrix, RingCommitment, SetupContributionMode,
    SetupPrefixVerifierRegistry, VerifierOpeningBatch,
    MAX_SETUP_MATRIX_FIELD_ELEMENTS,
};
use std::sync::Arc;

/// Encoding mode used for the verifier-input blob. Held constant on both ends
/// so the host and guest don't have to negotiate compression.
pub const BLOB_COMPRESS: Compress = Compress::No;

/// Validation mode used when decoding on the guest side. The blob is verifier
/// input, so malformed shape headers must be rejected before they drive
/// allocation or proof replay.
pub const BLOB_VALIDATE: Validate = Validate::Yes;

/// Maximum verifier-input blob bytes accepted by host and guest.
///
/// Mirrors the Jolt guest `max_input_size` literal in `guest/src/lib.rs`.
pub const MAX_JOLT_BLOB_BYTES: u64 = 805_306_368;

/// Magic header so the guest fails fast if it gets the wrong bytes.
const BLOB_MAGIC: [u8; 8] = *b"AKJOLTv1";
const MAX_TRANSCRIPT_DOMAIN_BYTES: usize = 1024;
const MAX_BLOB_NUM_VARS: usize = 64;

fn setup_mode_to_u8(mode: SetupContributionMode) -> u8 {
    match mode {
        SetupContributionMode::Direct => 0,
        SetupContributionMode::Recursive => 1,
    }
}

fn setup_mode_from_u8(byte: u8) -> Result<SetupContributionMode, SerializationError> {
    match byte {
        0 => Ok(SetupContributionMode::Direct),
        1 => Ok(SetupContributionMode::Recursive),
        other => Err(SerializationError::InvalidData(format!(
            "akita-jolt blob has invalid setup-contribution mode byte {other}"
        ))),
    }
}

fn reject_trailing_bytes(rest: &[u8]) -> Result<(), SerializationError> {
    if rest.is_empty() {
        return Ok(());
    }
    Err(SerializationError::InvalidData(format!(
        "akita-jolt blob has {} trailing bytes",
        rest.len()
    )))
}

/// Bundled verifier inputs that travel from the host to the Jolt guest.
///
/// `D` is the cyclotomic ring dimension picked by the host config. The
/// guest must use the same `D` to decode `commitment`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaJoltInputs<F: FieldCore, const D: usize> {
    /// Domain label both prover and verifier transcripts were initialized with.
    pub transcript_domain: Vec<u8>,
    /// Number of variables of the public polynomial (informational; sanity).
    pub num_vars: u64,
    /// Setup-contribution mode the proof was generated under. Held in the blob
    /// so host preflight and guest replay verify under the same mode without a
    /// separate flag.
    pub setup_contribution_mode: SetupContributionMode,
    /// Opening point in the multilinear basis.
    pub opening_point: Vec<F>,
    /// Claimed opening value at `opening_point`.
    pub opening: F,
    /// Single committed-poly group: one ring commitment per (poly, point) pair.
    pub commitment: RingCommitment<F, D>,
    /// Expanded verifier setup (matrix prefix usable by the verifier kernel).
    pub verifier_setup: AkitaVerifierSetup<F>,
    /// Proof shape descriptor; needed to deserialize `proof` without
    /// reconstructing a `Schedule` first.
    pub proof_shape: AkitaBatchedProofShape,
    /// The Akita batched proof itself. The extension field collapses to `F`
    /// for the fp128 D32OneHot profile (`EXT_DEGREE == 1`).
    pub proof: AkitaBatchedProof<F, F>,
}

impl<F: FieldCore, const D: usize> AkitaJoltInputs<F, D> {
    /// Build the singleton verifier claim represented by this blob.
    ///
    /// The recursion profile currently ships exactly one opening for one
    /// commitment. Keeping this projection here prevents host and guest replay
    /// from growing independent claim-shaping code.
    pub fn verifier_opening_batch<'a>(
        &'a self,
        openings: &'a [F; 1],
    ) -> VerifierOpeningBatch<'static, F, &'a RingCommitment<F, D>> {
        VerifierOpeningBatch::from_groups(
            self.opening_point.clone(),
            vec![CommitmentGroup {
                claims: openings.to_vec(),
                commitment: &self.commitment,
            }],
        )
        .expect("singleton recursion opening batch is valid")
    }

    fn validate_blob_header_bounds(
        transcript_domain_len: usize,
        num_vars: usize,
        opening_point_len: usize,
    ) -> Result<(), SerializationError> {
        if transcript_domain_len > MAX_TRANSCRIPT_DOMAIN_BYTES {
            return Err(SerializationError::LengthLimitExceeded {
                len: u64::try_from(transcript_domain_len).unwrap_or(u64::MAX),
                max: MAX_TRANSCRIPT_DOMAIN_BYTES,
            });
        }
        if num_vars > MAX_BLOB_NUM_VARS {
            return Err(SerializationError::LengthLimitExceeded {
                len: u64::try_from(num_vars).unwrap_or(u64::MAX),
                max: MAX_BLOB_NUM_VARS,
            });
        }
        if opening_point_len != num_vars {
            return Err(SerializationError::InvalidData(format!(
                "akita-jolt blob num_vars={num_vars} does not match opening-point arity {opening_point_len}"
            )));
        }
        Ok(())
    }
}

impl<F, const D: usize> AkitaJoltInputs<F, D>
where
    F: FieldCore + CanonicalField + AkitaSerialize,
{
    /// Encode the bundle into a single contiguous byte vector.
    pub fn write_to_bytes(&self) -> Result<Vec<u8>, SerializationError> {
        Self::validate_blob_header_bounds(
            self.transcript_domain.len(),
            usize::try_from(self.num_vars).map_err(|_| {
                SerializationError::LengthLimitExceeded {
                    len: self.num_vars,
                    max: usize::MAX,
                }
            })?,
            self.opening_point.len(),
        )?;
        let encoded_size = self.encoded_size();
        if encoded_size as u64 > MAX_JOLT_BLOB_BYTES {
            return Err(SerializationError::LengthLimitExceeded {
                len: encoded_size as u64,
                max: MAX_JOLT_BLOB_BYTES as usize,
            });
        }
        let mut bytes = Vec::with_capacity(self.encoded_size());
        bytes.extend_from_slice(&BLOB_MAGIC);
        // D is encoded so the guest can fail loudly on a mismatched
        // monomorphization.
        (D as u64).serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.transcript_domain
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.num_vars
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        setup_mode_to_u8(self.setup_contribution_mode)
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.opening_point
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.opening
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.commitment
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.verifier_setup
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.proof_shape
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.proof.serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        Ok(bytes)
    }

    /// Total encoded size in bytes (cheap pre-allocation sizing).
    pub fn encoded_size(&self) -> usize {
        BLOB_MAGIC.len()
            + (D as u64).serialized_size(BLOB_COMPRESS)
            + self.transcript_domain.serialized_size(BLOB_COMPRESS)
            + self.num_vars.serialized_size(BLOB_COMPRESS)
            + setup_mode_to_u8(self.setup_contribution_mode).serialized_size(BLOB_COMPRESS)
            + self.opening_point.serialized_size(BLOB_COMPRESS)
            + self.opening.serialized_size(BLOB_COMPRESS)
            + self.commitment.serialized_size(BLOB_COMPRESS)
            + self.verifier_setup.serialized_size(BLOB_COMPRESS)
            + self.proof_shape.serialized_size(BLOB_COMPRESS)
            + self.proof.serialized_size(BLOB_COMPRESS)
    }
}

impl<F, const D: usize> AkitaJoltInputs<F, D>
where
    F: FieldCore + AkitaSerialize + AkitaDeserialize<Context = ()> + Valid,
{
    fn decode_capped_bytes(
        rest: &mut &[u8],
        max_len: usize,
        context: &'static str,
    ) -> Result<Vec<u8>, SerializationError> {
        let len = Self::decode_capped_len(rest, max_len)?;
        Self::ensure_remaining(rest, len, context)?;
        let (bytes, tail) = rest.split_at(len);
        *rest = tail;
        Ok(bytes.to_vec())
    }

    fn decode_capped_len(rest: &mut &[u8], max_len: usize) -> Result<usize, SerializationError> {
        let encoded = u64::deserialize_with_mode(rest, BLOB_COMPRESS, BLOB_VALIDATE, &())?;
        let len =
            usize::try_from(encoded).map_err(|_| SerializationError::LengthLimitExceeded {
                len: encoded,
                max: usize::MAX,
            })?;
        if len > max_len {
            return Err(SerializationError::LengthLimitExceeded {
                len: encoded,
                max: max_len,
            });
        }
        Ok(len)
    }

    fn decode_num_vars(rest: &mut &[u8]) -> Result<usize, SerializationError> {
        Self::decode_capped_len(rest, MAX_BLOB_NUM_VARS)
    }

    fn ensure_remaining(
        rest: &[u8],
        len: usize,
        context: &'static str,
    ) -> Result<(), SerializationError> {
        if rest.len() < len {
            return Err(SerializationError::InvalidData(format!(
                "{context} claims {len} bytes but only {} remain",
                rest.len()
            )));
        }
        Ok(())
    }

    fn encoded_field_payload_len(field_elements: usize) -> Result<usize, SerializationError> {
        let field_size = F::zero().serialized_size(BLOB_COMPRESS);
        field_elements.checked_mul(field_size).ok_or_else(|| {
            SerializationError::InvalidData(
                "akita-jolt blob field payload length overflow".to_string(),
            )
        })
    }

    fn decode_opening_point(
        rest: &mut &[u8],
        transcript_domain_len: usize,
        num_vars: usize,
    ) -> Result<Vec<F>, SerializationError> {
        let len = Self::decode_capped_len(rest, MAX_BLOB_NUM_VARS)?;
        Self::validate_blob_header_bounds(transcript_domain_len, num_vars, len)?;
        let payload_len = Self::encoded_field_payload_len(len)?;
        Self::ensure_remaining(rest, payload_len, "akita-jolt opening point")?;
        let mut point = Vec::with_capacity(len);
        for _ in 0..len {
            point.push(F::deserialize_with_mode(
                &mut *rest,
                BLOB_COMPRESS,
                BLOB_VALIDATE,
                &(),
            )?);
        }
        Ok(point)
    }

    fn setup_matrix_encoded_len(matrix_fields: usize) -> Result<usize, SerializationError> {
        let header_len = 0usize
            .serialized_size(BLOB_COMPRESS)
            .checked_mul(2)
            .ok_or_else(|| {
                SerializationError::InvalidData(
                    "akita-jolt setup matrix header length overflow".to_string(),
                )
            })?;
        let payload_len = Self::encoded_field_payload_len(matrix_fields)?;
        header_len.checked_add(payload_len).ok_or_else(|| {
            SerializationError::InvalidData(
                "akita-jolt setup matrix encoded length overflow".to_string(),
            )
        })
    }

    fn check_setup_matrix_bytes_available(
        rest: &[u8],
        matrix_fields: usize,
    ) -> Result<(), SerializationError> {
        let matrix_len = Self::setup_matrix_encoded_len(matrix_fields)?;
        if rest.len() < matrix_len {
            return Err(SerializationError::InvalidData(format!(
                "akita-jolt setup matrix claims {matrix_len} bytes but only {} remain",
                rest.len()
            )));
        }
        Ok(())
    }

    fn decode_seed_and_matrix(
        rest: &mut &[u8],
    ) -> Result<(AkitaSetupSeed, FlatMatrix<F>), SerializationError> {
        let seed =
            AkitaSetupSeed::deserialize_with_mode(&mut *rest, BLOB_COMPRESS, BLOB_VALIDATE, &())?;
        if seed.gen_ring_dim != D {
            return Err(SerializationError::InvalidData(format!(
                "akita-jolt setup D={} does not match guest D={D}",
                seed.gen_ring_dim
            )));
        }
        let matrix_fields = seed.matrix_field_elements()?;
        if matrix_fields > MAX_SETUP_MATRIX_FIELD_ELEMENTS {
            return Err(SerializationError::LengthLimitExceeded {
                len: u64::try_from(matrix_fields).unwrap_or(u64::MAX),
                max: MAX_SETUP_MATRIX_FIELD_ELEMENTS,
            });
        }
        Self::check_setup_matrix_bytes_available(rest, matrix_fields)?;
        let shared_matrix = FlatMatrix::<F>::deserialize_with_expected_shape(
            &mut *rest,
            BLOB_COMPRESS,
            BLOB_VALIDATE,
            seed.max_setup_len,
            seed.gen_ring_dim,
            MAX_SETUP_MATRIX_FIELD_ELEMENTS,
        )?;
        Ok((seed, shared_matrix))
    }

    fn decode_prefix_slots(
        rest: &mut &[u8],
    ) -> Result<SetupPrefixVerifierRegistry<F>, SerializationError> {
        SetupPrefixVerifierRegistry::deserialize_with_mode(
            &mut *rest,
            BLOB_COMPRESS,
            BLOB_VALIDATE,
            &(),
        )
    }

    fn decode_from_bytes_with_setup(
        bytes: &[u8],
        decode_setup: impl FnOnce(&mut &[u8]) -> Result<AkitaVerifierSetup<F>, SerializationError>,
    ) -> Result<Self, SerializationError> {
        if bytes.len() < BLOB_MAGIC.len() {
            return Err(SerializationError::InvalidData(
                "akita-jolt blob shorter than magic header".to_string(),
            ));
        }
        if bytes.len() as u64 > MAX_JOLT_BLOB_BYTES {
            return Err(SerializationError::LengthLimitExceeded {
                len: bytes.len() as u64,
                max: MAX_JOLT_BLOB_BYTES as usize,
            });
        }
        let (magic, mut rest) = bytes.split_at(BLOB_MAGIC.len());
        if magic != BLOB_MAGIC {
            return Err(SerializationError::InvalidData(
                "akita-jolt blob magic mismatch".to_string(),
            ));
        }
        let encoded_d = u64::deserialize_with_mode(&mut rest, BLOB_COMPRESS, BLOB_VALIDATE, &())?;
        if encoded_d != D as u64 {
            return Err(SerializationError::InvalidData(format!(
                "akita-jolt blob D={encoded_d} doesn't match guest D={D}"
            )));
        }
        let transcript_domain = Self::decode_capped_bytes(
            &mut rest,
            MAX_TRANSCRIPT_DOMAIN_BYTES,
            "akita-jolt transcript domain",
        )?;
        let num_vars = Self::decode_num_vars(&mut rest)?;
        let setup_mode_byte =
            u8::deserialize_with_mode(&mut rest, BLOB_COMPRESS, BLOB_VALIDATE, &())?;
        let setup_contribution_mode = setup_mode_from_u8(setup_mode_byte)?;
        let opening_point =
            Self::decode_opening_point(&mut rest, transcript_domain.len(), num_vars)?;
        let opening = F::deserialize_with_mode(&mut rest, BLOB_COMPRESS, BLOB_VALIDATE, &())?;
        let commitment = RingCommitment::<F, D>::deserialize_with_mode(
            &mut rest,
            BLOB_COMPRESS,
            BLOB_VALIDATE,
            &(),
        )?;
        let verifier_setup = decode_setup(&mut rest)?;
        let proof_shape = AkitaBatchedProofShape::deserialize_with_mode(
            &mut rest,
            BLOB_COMPRESS,
            BLOB_VALIDATE,
            &(),
        )?;
        let proof = AkitaBatchedProof::<F, F>::deserialize_with_mode(
            &mut rest,
            BLOB_COMPRESS,
            BLOB_VALIDATE,
            &proof_shape,
        )?;
        reject_trailing_bytes(rest)?;
        Ok(Self {
            transcript_domain,
            num_vars: num_vars as u64,
            setup_contribution_mode,
            opening_point,
            opening,
            commitment,
            verifier_setup,
            proof_shape,
            proof,
        })
    }
}

impl<F, const D: usize> AkitaJoltInputs<F, D>
where
    F: FieldCore + RandomSampling + AkitaSerialize + AkitaDeserialize<Context = ()> + Valid,
{
    fn deserialize_strict_host_setup(
        rest: &mut &[u8],
    ) -> Result<AkitaVerifierSetup<F>, SerializationError> {
        let (seed, shared_matrix) = Self::decode_seed_and_matrix(rest)?;
        let prefix_slots = Self::decode_prefix_slots(rest)?;
        Ok(AkitaVerifierSetup {
            expanded: Arc::new(AkitaExpandedSetup::from_verified_parts(
                seed,
                shared_matrix,
            )?),
            prefix_slots,
        })
    }

    /// Strictly decode the bundle from bytes produced by [`Self::write_to_bytes`].
    ///
    /// This path rederives the public setup matrix from its seed and rejects
    /// stale or corrupted cached matrix bytes. Host-side artifact checks should
    /// use this path.
    pub fn read_from_bytes(bytes: &[u8]) -> Result<Self, SerializationError> {
        Self::decode_from_bytes_with_setup(bytes, Self::deserialize_strict_host_setup)
    }
}

#[cfg(any(
    feature = "trusted-benchmark-artifact",
    akita_trusted_benchmark_artifact
))]
impl<F, const D: usize> AkitaJoltInputs<F, D>
where
    F: FieldCore + AkitaSerialize + AkitaDeserialize<Context = ()> + Valid,
{
    fn deserialize_trusted_host_setup(
        rest: &mut &[u8],
    ) -> Result<AkitaVerifierSetup<F>, SerializationError> {
        let (seed, shared_matrix) = Self::decode_seed_and_matrix(rest)?;
        let prefix_slots = Self::decode_prefix_slots(rest)?;
        Ok(AkitaVerifierSetup {
            expanded: Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(seed, shared_matrix),
            ),
            prefix_slots,
        })
    }

    /// Decode a host-produced recursion artifact while trusting the cached
    /// setup matrix.
    ///
    /// This is a benchmark/profile fast path, not a general recursion security
    /// boundary. It still validates the blob magic, ring dimension, serialized
    /// structure, field elements, and seed/matrix shape equality, but it
    /// deliberately skips checking that the expanded setup matrix coefficients
    /// equal the matrix derived from the seed.
    pub fn read_trusted_host_artifact_bytes(bytes: &[u8]) -> Result<Self, SerializationError> {
        Self::decode_from_bytes_with_setup(bytes, Self::deserialize_trusted_host_setup)
    }
}

// `akita-algebra` is pulled in only so that downstream consumers can rely on
// `RingCommitment<F, D>` having all of its trait bounds satisfied; declare it
// here to avoid a `cargo machete` style trim.
#[doc(hidden)]
pub use akita_algebra as _akita_algebra_dep;

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128Offset275;
    use akita_types::{
        derive_public_matrix_flat, sample_public_matrix_seed, FlatRingVec,
        SetupPrefixPublicCommitment, SetupPrefixSlotId, SetupPrefixVerifierSlot,
    };

    type TestF = Prime128Offset275;
    const TEST_D: usize = 32;

    fn blob_prefix() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&BLOB_MAGIC);
        (TEST_D as u64)
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();
        bytes
    }

    #[test]
    fn trailing_blob_bytes_are_rejected() {
        let err = reject_trailing_bytes(&[0]).unwrap_err();
        assert!(err.to_string().contains("trailing bytes"));
        reject_trailing_bytes(&[]).unwrap();
    }

    #[test]
    fn transcript_domain_len_is_capped_before_allocation() {
        let mut bytes = blob_prefix();
        ((MAX_TRANSCRIPT_DOMAIN_BYTES + 1) as u64)
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();

        let err = AkitaJoltInputs::<TestF, TEST_D>::read_from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("length"));
    }

    #[test]
    fn num_vars_is_capped_before_opening_point_allocation() {
        let mut bytes = blob_prefix();
        Vec::<u8>::new()
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();
        ((MAX_BLOB_NUM_VARS + 1) as u64)
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();

        let err = AkitaJoltInputs::<TestF, TEST_D>::read_from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("length"));
    }

    #[test]
    fn opening_point_len_must_match_num_vars_before_allocation() {
        let mut bytes = blob_prefix();
        Vec::<u8>::new()
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();
        2u64.serialize_with_mode(&mut bytes, BLOB_COMPRESS).unwrap();
        setup_mode_to_u8(SetupContributionMode::Direct)
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();
        3u64.serialize_with_mode(&mut bytes, BLOB_COMPRESS).unwrap();

        let err = AkitaJoltInputs::<TestF, TEST_D>::read_from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("opening-point arity 3"));
    }

    #[test]
    fn strict_setup_decoder_preserves_prefix_slots() {
        let public_matrix_seed = sample_public_matrix_seed();
        let seed = AkitaSetupSeed {
            max_num_vars: 8,
            max_num_batched_polys: 1,
            gen_ring_dim: TEST_D,
            max_setup_len: 2,
            public_matrix_seed,
        };
        let shared_matrix = derive_public_matrix_flat::<TestF, TEST_D>(2, &public_matrix_seed);
        let id = SetupPrefixSlotId {
            setup_seed_digest: [1u8; 32],
            d_setup: TEST_D,
            natural_len: 1,
            n_prefix: TEST_D,
            level_params_digest: [2u8; 32],
        };
        let mut prefix_slots = SetupPrefixVerifierRegistry::new();
        prefix_slots
            .insert(SetupPrefixVerifierSlot {
                id,
                natural_len: 1,
                padded_len: TEST_D,
                commitment: SetupPrefixPublicCommitment {
                    rows: vec![FlatRingVec::from_coeffs(vec![TestF::zero(); TEST_D])],
                },
            })
            .expect("insert prefix slot");

        let mut bytes = Vec::new();
        seed.serialize_with_mode(&mut bytes, BLOB_COMPRESS).unwrap();
        shared_matrix
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();
        prefix_slots
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();

        let mut rest = &bytes[..];
        let decoded = AkitaJoltInputs::<TestF, TEST_D>::deserialize_strict_host_setup(&mut rest)
            .expect("decode setup");

        assert!(rest.is_empty());
        assert!(decoded.prefix_slots.get(&id).is_some());
        assert_eq!(decoded.prefix_slots.len(), 1);
    }

    #[test]
    fn setup_matrix_payload_must_fit_remaining_blob_before_allocation() {
        let err = AkitaJoltInputs::<TestF, TEST_D>::check_setup_matrix_bytes_available(&[], 1)
            .unwrap_err();
        assert!(err.to_string().contains("setup matrix claims"));
    }
}
