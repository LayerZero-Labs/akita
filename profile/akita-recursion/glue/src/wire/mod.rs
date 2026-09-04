//! Canonical verifier-input blob framing and bounded decoding.
//!
//! The host serializes [`AkitaJoltInputs`] once and the Jolt guest deserializes
//! it as the first step of the program.
//! Per-component encoding is the existing [`AkitaSerialize`] /
//! [`AkitaDeserialize`] machinery in [`akita_serialization`]. The recursion
//! benchmark can opt into an explicitly trusted cached-matrix setup decoder;
//! strict decoding remains the default.

use crate::{AkitaJoltCase, AkitaJoltInputs, AkitaJoltOpeningGroup};
use akita_config::{
    derive_transcript_grinding_plan, CommitmentConfig, TrustedScheduleCatalog,
    MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES,
};
use akita_error::checked;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use akita_types::{
    canonical_proof_shape, AkitaBatchedProof, AkitaBatchedProofShape, AkitaExpandedSetup,
    AkitaSetupDescriptor, AkitaVerifierSetup, CommittedGroup, FlatMatrix, OpeningScheduleSelection,
    SetupPrefixVerifierRegistry, MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};
use std::sync::Arc;

#[cfg(any(
    feature = "trusted-benchmark-artifact",
    akita_trusted_benchmark_artifact
))]
mod trusted_fixed_width;

mod jolt_postcard_adapter;
use jolt_postcard_adapter::{setup_matrix_padding, SETUP_MATRIX_MAX_PADDING_BYTES};

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
const BLOB_MAGIC: [u8; 8] = *b"AKJOLTv5";
const CATALOG_FRAME_MAGIC: [u8; 8] = *b"AKCATF01";
const CATALOG_FRAME_HEADER_BYTES: usize = CATALOG_FRAME_MAGIC.len() + 8;
const MAX_TRANSCRIPT_DOMAIN_BYTES: usize = 1024;
const MAX_BLOB_NUM_VARS: usize = 64;
const MAX_BLOB_GROUPS: usize = 16;
const MAX_BLOB_OPENINGS_PER_GROUP: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BlobEncodingLayout {
    encoded_size: usize,
    setup_matrix_padding: usize,
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

/// Read the case identity without decoding the full verifier artifact.
pub fn read_blob_case(bytes: &[u8]) -> Result<AkitaJoltCase, SerializationError> {
    let bytes = inner_blob_bytes(bytes)?;
    if bytes.len() < BLOB_MAGIC.len() + 1 {
        return Err(SerializationError::InvalidData(
            "akita-jolt blob is too short to contain a case identity".to_string(),
        ));
    }
    if bytes.len() as u64 > MAX_JOLT_BLOB_BYTES {
        return Err(SerializationError::LengthLimitExceeded {
            len: bytes.len() as u64,
            max: MAX_JOLT_BLOB_BYTES as usize,
        });
    }
    let (magic, payload) = bytes.split_at(BLOB_MAGIC.len());
    if magic != BLOB_MAGIC {
        return Err(SerializationError::InvalidData(
            "akita-jolt blob magic mismatch".to_string(),
        ));
    }
    AkitaJoltCase::from_tag(payload[0])
}

fn inner_blob_bytes(bytes: &[u8]) -> Result<&[u8], SerializationError> {
    if !bytes.starts_with(&CATALOG_FRAME_MAGIC) {
        return Ok(bytes);
    }
    if bytes.len() < CATALOG_FRAME_HEADER_BYTES {
        return Err(SerializationError::InvalidData(
            "akita-jolt catalog frame is truncated".to_string(),
        ));
    }
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&bytes[CATALOG_FRAME_MAGIC.len()..CATALOG_FRAME_HEADER_BYTES]);
    let artifact_len_u64 = u64::from_le_bytes(len_bytes);
    let artifact_len =
        usize::try_from(artifact_len_u64).map_err(|_| SerializationError::LengthLimitExceeded {
            len: artifact_len_u64,
            max: MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES,
        })?;
    if artifact_len == 0 || artifact_len > MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES {
        return Err(SerializationError::LengthLimitExceeded {
            len: artifact_len_u64,
            max: MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES,
        });
    }
    let inner_offset = CATALOG_FRAME_HEADER_BYTES
        .checked_add(artifact_len)
        .ok_or_else(|| SerializationError::InvalidData("catalog frame length overflow".into()))?;
    if inner_offset >= bytes.len() {
        return Err(SerializationError::InvalidData(
            "akita-jolt catalog frame has no complete inner blob".to_string(),
        ));
    }
    Ok(&bytes[inner_offset..])
}

/// Frame one verifier-input blob with a full external schedule artifact.
///
/// This is a benchmark bring-up format, not an authenticated production
/// recursion format. It keeps schedule rows out of the guest executable.
pub fn frame_with_schedule_catalog<Cfg: CommitmentConfig>(
    inner_blob: &[u8],
    schedules: &TrustedScheduleCatalog,
) -> Result<Vec<u8>, SerializationError> {
    akita_config::validate_trusted_schedule_catalog::<Cfg>(schedules)
        .map_err(|error| SerializationError::InvalidData(error.to_string()))?;
    let artifact = schedules
        .to_artifact_bytes()
        .map_err(|error| SerializationError::InvalidData(error.to_string()))?;
    if artifact.is_empty() || artifact.len() > MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES {
        return Err(SerializationError::LengthLimitExceeded {
            len: artifact.len() as u64,
            max: MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES,
        });
    }
    let framed_len =
        checked::sum([CATALOG_FRAME_HEADER_BYTES, artifact.len(), inner_blob.len()])
            .ok_or_else(|| SerializationError::InvalidData("catalog frame size overflow".into()))?;
    if framed_len as u64 > MAX_JOLT_BLOB_BYTES {
        return Err(SerializationError::LengthLimitExceeded {
            len: framed_len as u64,
            max: MAX_JOLT_BLOB_BYTES as usize,
        });
    }
    let mut framed = Vec::with_capacity(framed_len);
    framed.extend_from_slice(&CATALOG_FRAME_MAGIC);
    framed.extend_from_slice(&(artifact.len() as u64).to_le_bytes());
    framed.extend_from_slice(&artifact);
    framed.extend_from_slice(inner_blob);
    Ok(framed)
}

/// Decode the full benchmark catalog and return the inner verifier-input blob.
pub fn split_schedule_catalog<Cfg: CommitmentConfig>(
    bytes: &[u8],
) -> Result<(TrustedScheduleCatalog, &[u8]), SerializationError> {
    if !bytes.starts_with(&CATALOG_FRAME_MAGIC) {
        return Err(SerializationError::InvalidData(
            "akita-jolt input is missing the external schedule catalog frame".to_string(),
        ));
    }
    let inner = inner_blob_bytes(bytes)?;
    let artifact_end = bytes.len() - inner.len();
    let artifact = &bytes[CATALOG_FRAME_HEADER_BYTES..artifact_end];
    let schedules = akita_config::trusted_schedule_catalog_from_bytes::<Cfg>(artifact)
        .map_err(|error| SerializationError::InvalidData(error.to_string()))?;
    Ok((schedules, inner))
}

impl<F: Field, const D: usize, E: Field> AkitaJoltInputs<F, D, E> {
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

impl<F, const D: usize, E> AkitaJoltInputs<F, D, E>
where
    F: Field + CanonicalEncoding + AkitaSerialize + Valid,
    E: Field + AkitaSerialize + Valid,
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
        let layout = self.encoding_layout()?;
        let encoded_size = layout.encoded_size;
        if encoded_size as u64 > MAX_JOLT_BLOB_BYTES {
            return Err(SerializationError::LengthLimitExceeded {
                len: encoded_size as u64,
                max: MAX_JOLT_BLOB_BYTES as usize,
            });
        }
        let mut bytes = Vec::with_capacity(encoded_size);
        bytes.extend_from_slice(&BLOB_MAGIC);
        bytes.push(self.case.tag());
        // D is encoded so the guest can fail loudly on a mismatched
        // monomorphization.
        (D as u64).serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.transcript_domain
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.num_vars
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.opening_point
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.openings
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        (self.precommitted_groups.len() as u64).serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        for group in &self.precommitted_groups {
            group
                .opening_point
                .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
            group
                .openings
                .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
            group
                .commitment
                .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        }
        self.schedule_selection
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.commitment
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.verifier_setup
            .expanded
            .descriptor
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        bytes.push(u8::try_from(layout.setup_matrix_padding).map_err(|_| {
            SerializationError::InvalidData(
                "akita-jolt setup matrix padding does not fit u8".to_string(),
            )
        })?);
        bytes.resize(
            bytes
                .len()
                .checked_add(layout.setup_matrix_padding)
                .ok_or_else(|| {
                    SerializationError::InvalidData(
                        "akita-jolt setup matrix padding overflow".to_string(),
                    )
                })?,
            0,
        );
        self.verifier_setup
            .expanded
            .shared_matrix
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.verifier_setup
            .prefix_slots
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.proof_shape
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.proof.serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        if bytes.len() != encoded_size {
            return Err(SerializationError::InvalidData(format!(
                "akita-jolt encoded-size mismatch: expected {encoded_size}, wrote {}",
                bytes.len()
            )));
        }
        Ok(bytes)
    }

    fn encoding_layout(&self) -> Result<BlobEncodingLayout, SerializationError> {
        let groups_size = checked::sum(self.precommitted_groups.iter().flat_map(|group| {
            [
                group.opening_point.serialized_size(BLOB_COMPRESS),
                group.openings.serialized_size(BLOB_COMPRESS),
                group.commitment.serialized_size(BLOB_COMPRESS),
            ]
        }))
        .ok_or_else(|| {
            SerializationError::InvalidData("akita-jolt opening-group size overflow".to_string())
        })?;
        let padding_record_offset = checked::sum([
            BLOB_MAGIC.len(),
            1,
            (D as u64).serialized_size(BLOB_COMPRESS),
            self.transcript_domain.serialized_size(BLOB_COMPRESS),
            self.num_vars.serialized_size(BLOB_COMPRESS),
            self.opening_point.serialized_size(BLOB_COMPRESS),
            self.openings.serialized_size(BLOB_COMPRESS),
            (self.precommitted_groups.len() as u64).serialized_size(BLOB_COMPRESS),
            groups_size,
            self.schedule_selection.serialized_size(BLOB_COMPRESS),
            self.commitment.serialized_size(BLOB_COMPRESS),
            self.verifier_setup
                .expanded
                .descriptor
                .serialized_size(BLOB_COMPRESS),
        ])
        .ok_or_else(|| {
            SerializationError::InvalidData("akita-jolt blob prefix size overflow".to_string())
        })?;
        let unpadded_size = checked::sum([
            padding_record_offset,
            self.verifier_setup
                .expanded
                .shared_matrix
                .serialized_size(BLOB_COMPRESS),
            self.verifier_setup
                .prefix_slots
                .serialized_size(BLOB_COMPRESS),
            self.proof_shape.serialized_size(BLOB_COMPRESS),
            self.proof.serialized_size(BLOB_COMPRESS),
        ])
        .ok_or_else(|| {
            SerializationError::InvalidData("akita-jolt blob size overflow".to_string())
        })?;
        let setup_matrix_padding = setup_matrix_padding(unpadded_size, padding_record_offset)?;
        let encoded_size =
            checked::sum([unpadded_size, 1, setup_matrix_padding]).ok_or_else(|| {
                SerializationError::InvalidData("akita-jolt blob size overflow".to_string())
            })?;
        Ok(BlobEncodingLayout {
            encoded_size,
            setup_matrix_padding,
        })
    }
}

impl<F, const D: usize, E> AkitaJoltInputs<F, D, E>
where
    F: Field + CanonicalEncoding + AkitaSerialize + AkitaDeserialize<Context = ()> + Valid,
    E: ExtField<F> + AkitaSerialize + AkitaDeserialize<Context = ()> + Valid,
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

    fn encoded_field_payload_len<T: Field + AkitaSerialize>(
        field_elements: usize,
    ) -> Result<usize, SerializationError> {
        let field_size = T::zero().serialized_size(BLOB_COMPRESS);
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
    ) -> Result<Vec<E>, SerializationError> {
        let len = Self::decode_capped_len(rest, MAX_BLOB_NUM_VARS)?;
        Self::validate_blob_header_bounds(transcript_domain_len, num_vars, len)?;
        let payload_len = Self::encoded_field_payload_len::<E>(len)?;
        Self::ensure_remaining(rest, payload_len, "akita-jolt opening point")?;
        let mut point = Vec::with_capacity(len);
        for _ in 0..len {
            point.push(E::deserialize_with_mode(
                &mut *rest,
                BLOB_COMPRESS,
                BLOB_VALIDATE,
                &(),
            )?);
        }
        Ok(point)
    }

    fn decode_ext_field_vec(
        rest: &mut &[u8],
        max_len: usize,
        context: &'static str,
    ) -> Result<Vec<E>, SerializationError> {
        let len = Self::decode_capped_len(rest, max_len)?;
        let payload_len = Self::encoded_field_payload_len::<E>(len)?;
        Self::ensure_remaining(rest, payload_len, context)?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(E::deserialize_with_mode(
                &mut *rest,
                BLOB_COMPRESS,
                BLOB_VALIDATE,
                &(),
            )?);
        }
        Ok(values)
    }

    fn decode_opening_group(
        rest: &mut &[u8],
    ) -> Result<AkitaJoltOpeningGroup<F, E>, SerializationError> {
        let opening_point = Self::decode_ext_field_vec(
            rest,
            MAX_BLOB_NUM_VARS,
            "akita-jolt precommitted opening point",
        )?;
        let openings = Self::decode_ext_field_vec(
            rest,
            MAX_BLOB_OPENINGS_PER_GROUP,
            "akita-jolt precommitted openings",
        )?;
        if openings.is_empty() {
            return Err(SerializationError::InvalidData(
                "akita-jolt precommitted opening group is empty".to_string(),
            ));
        }
        let commitment = CommittedGroup::<F>::deserialize_with_mode(
            &mut *rest,
            BLOB_COMPRESS,
            BLOB_VALIDATE,
            &(),
        )?;
        Ok(AkitaJoltOpeningGroup {
            opening_point,
            openings,
            commitment,
        })
    }

    fn setup_matrix_encoded_len(matrix_fields: usize) -> Result<usize, SerializationError> {
        let header_len = 0usize.serialized_size(BLOB_COMPRESS);
        let payload_len = Self::encoded_field_payload_len::<F>(matrix_fields)?;
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

    fn decode_seed_and_matrix_with(
        rest: &mut &[u8],
        total_blob_len: usize,
        decode_matrix: impl FnOnce(&mut &[u8], usize) -> Result<FlatMatrix<F>, SerializationError>,
    ) -> Result<(AkitaSetupDescriptor, FlatMatrix<F>), SerializationError> {
        let seed = AkitaSetupDescriptor::deserialize_with_mode(
            &mut *rest,
            BLOB_COMPRESS,
            BLOB_VALIDATE,
            &(),
        )?;
        let matrix_fields = seed.num_field_elements;
        if matrix_fields > MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS {
            return Err(SerializationError::LengthLimitExceeded {
                len: u64::try_from(matrix_fields).unwrap_or(u64::MAX),
                max: MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
            });
        }
        Self::decode_setup_matrix_padding(rest, total_blob_len)?;
        Self::check_setup_matrix_bytes_available(rest, matrix_fields)?;
        let shared_matrix = decode_matrix(rest, seed.num_field_elements)?;
        Ok((seed, shared_matrix))
    }

    fn decode_setup_matrix_padding(
        rest: &mut &[u8],
        total_blob_len: usize,
    ) -> Result<(), SerializationError> {
        let padding_record_offset = total_blob_len.checked_sub(rest.len()).ok_or_else(|| {
            SerializationError::InvalidData(
                "akita-jolt setup matrix padding offset is outside the blob".to_string(),
            )
        })?;
        let (&encoded_padding, tail) = rest.split_first().ok_or_else(|| {
            SerializationError::InvalidData(
                "akita-jolt setup matrix padding record is missing".to_string(),
            )
        })?;
        let padding = usize::from(encoded_padding);
        if padding > SETUP_MATRIX_MAX_PADDING_BYTES {
            return Err(SerializationError::InvalidData(format!(
                "akita-jolt setup matrix padding {padding} exceeds {SETUP_MATRIX_MAX_PADDING_BYTES}"
            )));
        }
        if tail.len() < padding {
            return Err(SerializationError::InvalidData(format!(
                "akita-jolt setup matrix padding claims {padding} bytes but only {} remain",
                tail.len()
            )));
        }
        let (padding_bytes, matrix_bytes) = tail.split_at(padding);
        if padding_bytes.iter().any(|byte| *byte != 0) {
            return Err(SerializationError::InvalidData(
                "akita-jolt setup matrix padding must be zero".to_string(),
            ));
        }
        let unpadded_blob_len = total_blob_len
            .checked_sub(padding.checked_add(1).ok_or_else(|| {
                SerializationError::InvalidData(
                    "akita-jolt setup matrix padding length overflow".to_string(),
                )
            })?)
            .ok_or_else(|| {
                SerializationError::InvalidData(
                    "akita-jolt setup matrix padding exceeds the blob".to_string(),
                )
            })?;
        let expected_padding = setup_matrix_padding(unpadded_blob_len, padding_record_offset)?;
        if padding != expected_padding {
            return Err(SerializationError::InvalidData(format!(
                "akita-jolt setup matrix padding {padding} does not match expected {expected_padding}"
            )));
        }
        *rest = matrix_bytes;
        Ok(())
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

    fn decode_from_bytes_with_setup<Cfg>(
        bytes: &[u8],
        schedules: &TrustedScheduleCatalog,
        decode_setup: impl FnOnce(
            &mut &[u8],
            usize,
        ) -> Result<AkitaVerifierSetup<F>, SerializationError>,
    ) -> Result<Self, SerializationError>
    where
        Cfg: CommitmentConfig<Field = F, ExtField = E>,
    {
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
        let (&case_tag, tail) = rest.split_first().ok_or_else(|| {
            SerializationError::InvalidData(
                "akita-jolt blob is missing its case identity".to_string(),
            )
        })?;
        let case = AkitaJoltCase::from_tag(case_tag)?;
        rest = tail;
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
        let num_vars = Self::decode_capped_len(&mut rest, MAX_BLOB_NUM_VARS)?;
        let opening_point =
            Self::decode_opening_point(&mut rest, transcript_domain.len(), num_vars)?;
        let openings = Self::decode_ext_field_vec(
            &mut rest,
            MAX_BLOB_OPENINGS_PER_GROUP,
            "akita-jolt final openings",
        )?;
        if openings.is_empty() {
            return Err(SerializationError::InvalidData(
                "akita-jolt final opening group is empty".to_string(),
            ));
        }
        let precommitted_count = Self::decode_capped_len(&mut rest, MAX_BLOB_GROUPS)?;
        let mut precommitted_groups = Vec::with_capacity(precommitted_count);
        for _ in 0..precommitted_count {
            precommitted_groups.push(Self::decode_opening_group(&mut rest)?);
        }
        let schedule_selection = OpeningScheduleSelection::deserialize_with_mode(
            &mut rest,
            BLOB_COMPRESS,
            BLOB_VALIDATE,
            &(),
        )?;
        let commitment = CommittedGroup::<F>::deserialize_with_mode(
            &mut rest,
            BLOB_COMPRESS,
            BLOB_VALIDATE,
            &(),
        )?;
        let verifier_setup = decode_setup(&mut rest, bytes.len())?;
        let proof_shape = AkitaBatchedProofShape::deserialize_with_mode(
            &mut rest,
            BLOB_COMPRESS,
            BLOB_VALIDATE,
            &(),
        )?;
        Self::validate_proof_shape_before_allocation::<Cfg>(
            schedule_selection,
            &proof_shape,
            rest.len(),
            schedules,
        )?;
        let proof = AkitaBatchedProof::<F, E>::deserialize_with_mode(
            &mut rest,
            BLOB_COMPRESS,
            BLOB_VALIDATE,
            &proof_shape,
        )?;
        reject_trailing_bytes(rest)?;
        let inputs = Self {
            case,
            transcript_domain,
            num_vars: num_vars as u64,
            opening_point,
            openings,
            precommitted_groups,
            schedule_selection,
            commitment,
            verifier_setup,
            proof_shape,
            proof,
        };
        inputs
            .verifier_statement()
            .map_err(|error| SerializationError::InvalidData(error.to_string()))?;
        Ok(inputs)
    }

    fn validate_proof_shape_before_allocation<Cfg>(
        schedule_selection: OpeningScheduleSelection,
        proof_shape: &AkitaBatchedProofShape,
        proof_bytes_available: usize,
        schedules: &TrustedScheduleCatalog,
    ) -> Result<(), SerializationError>
    where
        Cfg: CommitmentConfig<Field = F, ExtField = E>,
    {
        proof_shape.validate_decode_budget(
            proof_bytes_available,
            F::zero().serialized_size(BLOB_COMPRESS),
            E::zero().serialized_size(BLOB_COMPRESS),
        )?;
        akita_config::validate_trusted_schedule_catalog::<Cfg>(schedules)
            .map_err(|error| SerializationError::InvalidData(error.to_string()))?;
        let resolved = schedules
            .resolve_selection(schedule_selection)
            .map_err(|error| SerializationError::InvalidData(error.to_string()))?;
        let root_opening_layout = resolved
            .profiles()
            .opening_layout()
            .map_err(|error| SerializationError::InvalidData(error.to_string()))?;
        let grinding_plan =
            derive_transcript_grinding_plan::<Cfg>(resolved.schedule(), &root_opening_layout)
                .map_err(|error| SerializationError::InvalidData(error.to_string()))?;
        proof_shape.validate_grinding_plan(&grinding_plan)?;
        let expected_shape = canonical_proof_shape(
            resolved.schedule(),
            &root_opening_layout,
            E::DEGREE,
            &grinding_plan,
        )
        .map_err(|error| SerializationError::InvalidData(error.to_string()))?;
        if *proof_shape != expected_shape {
            return Err(SerializationError::InvalidData(
                "proof shape does not match the selected canonical schedule".to_string(),
            ));
        }
        Ok(())
    }
}

impl<F, const D: usize, E> AkitaJoltInputs<F, D, E>
where
    F: Field + CanonicalEncoding + AkitaSerialize + AkitaDeserialize<Context = ()> + Valid,
    E: ExtField<F> + AkitaSerialize + AkitaDeserialize<Context = ()> + Valid,
{
    fn deserialize_strict_host_setup(
        rest: &mut &[u8],
        total_blob_len: usize,
    ) -> Result<AkitaVerifierSetup<F>, SerializationError> {
        let (seed, shared_matrix) =
            Self::decode_seed_and_matrix_with(rest, total_blob_len, |rest, matrix_fields| {
                FlatMatrix::<F>::deserialize_with_expected_shape(
                    &mut *rest,
                    BLOB_COMPRESS,
                    BLOB_VALIDATE,
                    matrix_fields,
                    MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
                )
            })?;
        let prefix_slots = Self::decode_prefix_slots(rest)?;
        AkitaVerifierSetup::from_parts(
            Arc::new(AkitaExpandedSetup::from_verified_parts(
                seed,
                shared_matrix,
            )?),
            prefix_slots,
        )
        .map_err(|err| SerializationError::InvalidData(err.to_string()))
    }

    /// Strictly decode the bundle from bytes produced by [`Self::write_to_bytes`].
    ///
    /// This path rederives the public setup matrix from its seed and rejects
    /// stale or corrupted cached matrix bytes. Host-side artifact checks should
    /// use this path.
    pub fn read_from_bytes<Cfg>(
        bytes: &[u8],
        schedules: &TrustedScheduleCatalog,
    ) -> Result<Self, SerializationError>
    where
        Cfg: CommitmentConfig<Field = F, ExtField = E>,
    {
        Self::decode_from_bytes_with_setup::<Cfg>(
            bytes,
            schedules,
            Self::deserialize_strict_host_setup,
        )
    }
}

#[cfg(test)]
mod tests;
