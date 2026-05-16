//! Shared verifier-input blob shipped from a host artifact generator into a
//! Jolt guest program.
//!
//! The host serializes the bundle once (`AkitaJoltInputs::write_to_bytes`) and
//! the Jolt guest deserializes it as the very first step of the program
//! (`AkitaJoltInputs::read_from_bytes`). Per-component encoding is the existing
//! [`AkitaSerialize`] / [`AkitaDeserialize`] machinery in
//! [`akita_serialization`].

#![allow(clippy::missing_errors_doc)]

use akita_field::FieldCore;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use akita_types::{AkitaBatchedProof, AkitaBatchedProofShape, AkitaVerifierSetup, RingCommitment};

/// Encoding mode used for the verifier-input blob. Held constant on both ends
/// so the host and guest don't have to negotiate compression.
pub const BLOB_COMPRESS: Compress = Compress::No;

/// Validation mode used when decoding on the guest side. The blob is verifier
/// input, so malformed shape headers must be rejected before they drive
/// allocation or proof replay.
pub const BLOB_VALIDATE: Validate = Validate::Yes;

/// Magic header so the guest fails fast if it gets the wrong bytes.
const BLOB_MAGIC: [u8; 8] = *b"AKJOLTv1";

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
    /// The Akita batched proof itself.
    pub proof: AkitaBatchedProof<F, F>,
}

impl<F, const D: usize> AkitaJoltInputs<F, D>
where
    F: FieldCore + AkitaSerialize + AkitaDeserialize<Context = ()> + Valid,
{
    /// Encode the bundle into a single contiguous byte vector.
    pub fn write_to_bytes(&self) -> Result<Vec<u8>, SerializationError> {
        let mut bytes = Vec::with_capacity(self.encoded_size());
        bytes.extend_from_slice(&BLOB_MAGIC);
        // D is encoded so the guest can fail loudly on a mismatched
        // monomorphization.
        (D as u64).serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.transcript_domain
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.num_vars
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

    /// Decode the bundle from bytes produced by [`Self::write_to_bytes`].
    pub fn read_from_bytes(bytes: &[u8]) -> Result<Self, SerializationError> {
        if bytes.len() < BLOB_MAGIC.len() {
            return Err(SerializationError::InvalidData(
                "akita-jolt blob shorter than magic header".to_string(),
            ));
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
        let transcript_domain =
            Vec::<u8>::deserialize_with_mode(&mut rest, BLOB_COMPRESS, BLOB_VALIDATE, &())?;
        let num_vars = u64::deserialize_with_mode(&mut rest, BLOB_COMPRESS, BLOB_VALIDATE, &())?;
        let opening_point =
            Vec::<F>::deserialize_with_mode(&mut rest, BLOB_COMPRESS, BLOB_VALIDATE, &())?;
        let opening = F::deserialize_with_mode(&mut rest, BLOB_COMPRESS, BLOB_VALIDATE, &())?;
        let commitment = RingCommitment::<F, D>::deserialize_with_mode(
            &mut rest,
            BLOB_COMPRESS,
            BLOB_VALIDATE,
            &(),
        )?;
        let verifier_setup = AkitaVerifierSetup::<F>::deserialize_with_mode(
            &mut rest,
            BLOB_COMPRESS,
            BLOB_VALIDATE,
            &(),
        )?;
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
        Ok(Self {
            transcript_domain,
            num_vars,
            opening_point,
            opening,
            commitment,
            verifier_setup,
            proof_shape,
            proof,
        })
    }

    /// Total encoded size in bytes (cheap pre-allocation sizing).
    pub fn encoded_size(&self) -> usize {
        BLOB_MAGIC.len()
            + (D as u64).serialized_size(BLOB_COMPRESS)
            + self.transcript_domain.serialized_size(BLOB_COMPRESS)
            + self.num_vars.serialized_size(BLOB_COMPRESS)
            + self.opening_point.serialized_size(BLOB_COMPRESS)
            + self.opening.serialized_size(BLOB_COMPRESS)
            + self.commitment.serialized_size(BLOB_COMPRESS)
            + self.verifier_setup.serialized_size(BLOB_COMPRESS)
            + self.proof_shape.serialized_size(BLOB_COMPRESS)
            + self.proof.serialized_size(BLOB_COMPRESS)
    }
}

// `akita-algebra` is pulled in only so that downstream consumers can rely on
// `RingCommitment<F, D>` having all of its trait bounds satisfied; declare it
// here to avoid a `cargo machete` style trim.
#[doc(hidden)]
pub use akita_algebra as _akita_algebra_dep;
