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

use akita_field::{FieldCore, RandomSampling};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
#[cfg(any(
    feature = "trusted-benchmark-artifact",
    akita_trusted_benchmark_artifact
))]
use akita_types::AkitaExpandedSetup;
use akita_types::{AkitaBatchedProof, AkitaBatchedProofShape, AkitaVerifierSetup, RingCommitment};
#[cfg(any(
    feature = "trusted-benchmark-artifact",
    akita_trusted_benchmark_artifact
))]
use std::sync::Arc;

/// Encoding mode used for the verifier-input blob. Held constant on both ends
/// so the host and guest don't have to negotiate compression.
pub const BLOB_COMPRESS: Compress = Compress::No;

/// Validation mode used when decoding on the guest side. The blob is verifier
/// input, so malformed shape headers must be rejected before they drive
/// allocation or proof replay.
pub const BLOB_VALIDATE: Validate = Validate::Yes;

/// Magic header so the guest fails fast if it gets the wrong bytes.
const BLOB_MAGIC: [u8; 8] = *b"AKJOLTv1";

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
    /// The Akita batched proof itself. The claim/challenge field collapses to
    /// `F` for the fp128 D32OneHot profile (`CLAIM_EXT_DEGREE == 1`).
    pub proof: AkitaBatchedProof<F, F>,
}

impl<F, const D: usize> AkitaJoltInputs<F, D>
where
    F: FieldCore + AkitaSerialize,
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

impl<F, const D: usize> AkitaJoltInputs<F, D>
where
    F: FieldCore + AkitaSerialize + AkitaDeserialize<Context = ()> + Valid,
{
    fn decode_from_bytes_with_setup(
        bytes: &[u8],
        decode_setup: impl FnOnce(&mut &[u8]) -> Result<AkitaVerifierSetup<F>, SerializationError>,
    ) -> Result<Self, SerializationError> {
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
        if num_vars != opening_point.len() as u64 {
            return Err(SerializationError::InvalidData(format!(
                "akita-jolt blob num_vars={num_vars} does not match opening-point arity {}",
                opening_point.len()
            )));
        }
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
            num_vars,
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
    /// Strictly decode the bundle from bytes produced by [`Self::write_to_bytes`].
    ///
    /// This path rederives the public setup matrix from its seed and rejects
    /// stale or corrupted cached matrix bytes. Host-side artifact checks should
    /// use this path.
    pub fn read_from_bytes(bytes: &[u8]) -> Result<Self, SerializationError> {
        Self::decode_from_bytes_with_setup(bytes, |rest| {
            AkitaVerifierSetup::<F>::deserialize_with_mode(rest, BLOB_COMPRESS, BLOB_VALIDATE, &())
        })
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
    /// Decode a host-produced recursion artifact while trusting the cached
    /// setup matrix.
    ///
    /// This is a benchmark/profile fast path, not a general recursion security
    /// boundary. It still validates the blob magic, ring dimension, serialized
    /// structure, field elements, and seed/matrix shape equality, but it
    /// deliberately skips checking that the expanded setup matrix coefficients
    /// equal the matrix derived from the seed.
    pub fn read_trusted_host_artifact_bytes(bytes: &[u8]) -> Result<Self, SerializationError> {
        Self::decode_from_bytes_with_setup(bytes, |rest| {
            Ok(AkitaVerifierSetup {
                expanded: Arc::new(AkitaExpandedSetup::deserialize_trusted_cached_matrix(
                    rest,
                    BLOB_COMPRESS,
                )?),
            })
        })
    }
}

// `akita-algebra` is pulled in only so that downstream consumers can rely on
// `RingCommitment<F, D>` having all of its trait bounds satisfied; declare it
// here to avoid a `cargo machete` style trim.
#[doc(hidden)]
pub use akita_algebra as _akita_algebra_dep;

#[cfg(test)]
mod tests {
    use super::reject_trailing_bytes;

    #[test]
    fn trailing_blob_bytes_are_rejected() {
        let err = reject_trailing_bytes(&[0]).unwrap_err();
        assert!(err.to_string().contains("trailing bytes"));
        reject_trailing_bytes(&[]).unwrap();
    }
}
