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

use akita_config::CommitmentConfig;
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, RandomSampling};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use akita_types::{
    canonical_base_field_proof_shape, AkitaBatchedProof, AkitaBatchedProofShape,
    AkitaExpandedSetup, AkitaSetupDescriptor, AkitaVerifierSetup, CommittedGroup, FlatMatrix,
    GroupBatchStatement, OpeningClaims, OpeningScheduleSelection, PolynomialGroupClaims,
    SetupPrefixVerifierRegistry, MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
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
const BLOB_MAGIC: [u8; 8] = *b"AKJOLTv2";
const MAX_TRANSCRIPT_DOMAIN_BYTES: usize = 1024;
const MAX_BLOB_NUM_VARS: usize = 64;

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
/// `D` is the cyclotomic root-envelope dimension pinned by the host config.
/// The guest must use the same value to reject blobs built for a different
/// verifier monomorphization; per-level dimensions remain schedule-owned.
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
    /// Exact generated schedule row accepted for this opening batch.
    pub schedule_selection: OpeningScheduleSelection,
    /// Single committed-poly group: one ring commitment per (poly, point) pair.
    pub commitment: CommittedGroup<F>,
    /// Expanded verifier setup (matrix prefix usable by the verifier kernel).
    pub verifier_setup: AkitaVerifierSetup<F>,
    /// Proof shape descriptor; needed to deserialize `proof` without
    /// reconstructing a `Schedule` first.
    pub proof_shape: AkitaBatchedProofShape,
    /// The Akita batched proof itself. The extension field collapses to `F`
    /// for the fp128 OneHot profile (`EXT_DEGREE == 1`).
    pub proof: AkitaBatchedProof<F, F>,
}

impl<F: FieldCore, const D: usize> AkitaJoltInputs<F, D> {
    /// Build the singleton verifier claim represented by this blob.
    ///
    /// The recursion profile currently ships exactly one opening for one
    /// commitment. Keeping this projection here prevents host and guest replay
    /// from growing independent claim-shaping code.
    pub fn verifier_statement<'a>(
        &'a self,
        openings: &'a [F; 1],
    ) -> Result<GroupBatchStatement<'a, F, F>, AkitaError> {
        let num_vars = usize::try_from(self.num_vars).map_err(|_| {
            AkitaError::InvalidInput("recursion blob num_vars does not fit usize".to_string())
        })?;
        if num_vars != self.opening_point.len() {
            return Err(AkitaError::InvalidInput(
                "singleton recursion opening point does not cover all variables".to_string(),
            ));
        }
        let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            self.opening_point.clone(),
            openings.to_vec(),
            &self.commitment,
        )?])?;
        GroupBatchStatement::new(self.schedule_selection, claims)
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
    F: FieldCore + CanonicalField + AkitaSerialize + Valid,
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
        self.opening_point
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.opening
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.schedule_selection
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
            + self.schedule_selection.serialized_size(BLOB_COMPRESS)
            + self.commitment.serialized_size(BLOB_COMPRESS)
            + self.verifier_setup.serialized_size(BLOB_COMPRESS)
            + self.proof_shape.serialized_size(BLOB_COMPRESS)
            + self.proof.serialized_size(BLOB_COMPRESS)
    }
}

impl<F, const D: usize> AkitaJoltInputs<F, D>
where
    F: FieldCore
        + CanonicalField
        + FromPrimitiveInt
        + AkitaSerialize
        + AkitaDeserialize<Context = ()>
        + Valid,
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
        let header_len = 0usize.serialized_size(BLOB_COMPRESS);
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
        Self::check_setup_matrix_bytes_available(rest, matrix_fields)?;
        let shared_matrix = FlatMatrix::<F>::deserialize_with_expected_shape(
            &mut *rest,
            BLOB_COMPRESS,
            BLOB_VALIDATE,
            seed.num_field_elements,
            MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
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

    fn decode_from_bytes_with_setup<Cfg>(
        bytes: &[u8],
        decode_setup: impl FnOnce(&mut &[u8]) -> Result<AkitaVerifierSetup<F>, SerializationError>,
    ) -> Result<Self, SerializationError>
    where
        Cfg: CommitmentConfig<Field = F, ExtField = F>,
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
        let opening = F::deserialize_with_mode(&mut rest, BLOB_COMPRESS, BLOB_VALIDATE, &())?;
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
        let verifier_setup = decode_setup(&mut rest)?;
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
            opening_point,
            opening,
            schedule_selection,
            commitment,
            verifier_setup,
            proof_shape,
            proof,
        })
    }

    fn validate_proof_shape_before_allocation<Cfg>(
        schedule_selection: OpeningScheduleSelection,
        proof_shape: &AkitaBatchedProofShape,
        proof_bytes_available: usize,
    ) -> Result<(), SerializationError>
    where
        Cfg: CommitmentConfig<Field = F, ExtField = F>,
    {
        proof_shape.validate_base_field_decode_budget(
            proof_bytes_available,
            F::zero().serialized_size(BLOB_COMPRESS),
        )?;
        let resolved = Cfg::resolve_schedule_selection(schedule_selection)
            .map_err(|error| SerializationError::InvalidData(error.to_string()))?;
        let expected_shape = canonical_base_field_proof_shape(resolved.schedule())
            .map_err(|error| SerializationError::InvalidData(error.to_string()))?;
        if *proof_shape != expected_shape {
            return Err(SerializationError::InvalidData(
                "proof shape does not match the selected canonical schedule".to_string(),
            ));
        }
        Ok(())
    }
}

impl<F, const D: usize> AkitaJoltInputs<F, D>
where
    F: FieldCore
        + CanonicalField
        + FromPrimitiveInt
        + RandomSampling
        + AkitaSerialize
        + AkitaDeserialize<Context = ()>
        + Valid,
{
    fn deserialize_strict_host_setup(
        rest: &mut &[u8],
    ) -> Result<AkitaVerifierSetup<F>, SerializationError> {
        let (seed, shared_matrix) = Self::decode_seed_and_matrix(rest)?;
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
    pub fn read_from_bytes<Cfg>(bytes: &[u8]) -> Result<Self, SerializationError>
    where
        Cfg: CommitmentConfig<Field = F, ExtField = F>,
    {
        Self::decode_from_bytes_with_setup::<Cfg>(bytes, Self::deserialize_strict_host_setup)
    }
}

#[cfg(any(
    feature = "trusted-benchmark-artifact",
    akita_trusted_benchmark_artifact
))]
impl<F, const D: usize> AkitaJoltInputs<F, D>
where
    F: FieldCore
        + CanonicalField
        + FromPrimitiveInt
        + AkitaSerialize
        + AkitaDeserialize<Context = ()>
        + Valid,
{
    fn deserialize_trusted_host_setup(
        rest: &mut &[u8],
    ) -> Result<AkitaVerifierSetup<F>, SerializationError> {
        let (seed, shared_matrix) = Self::decode_seed_and_matrix(rest)?;
        let prefix_slots = Self::decode_prefix_slots(rest)?;
        AkitaVerifierSetup::from_parts(
            Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(seed, shared_matrix),
            ),
            prefix_slots,
        )
        .map_err(|err| SerializationError::InvalidData(err.to_string()))
    }

    /// Decode a host-produced recursion artifact while trusting the cached
    /// setup matrix.
    ///
    /// This is a benchmark/profile fast path, not a general recursion security
    /// boundary. It still validates the blob magic, serialized structure,
    /// field elements, and seed/matrix shape equality, but it
    /// deliberately skips checking that the expanded setup matrix coefficients
    /// equal the matrix derived from the seed.
    pub fn read_trusted_host_artifact_bytes<Cfg>(bytes: &[u8]) -> Result<Self, SerializationError>
    where
        Cfg: CommitmentConfig<Field = F, ExtField = F>,
    {
        Self::decode_from_bytes_with_setup::<Cfg>(bytes, Self::deserialize_trusted_host_setup)
    }
}

// `akita-algebra` is pulled in only so that downstream consumers can rely on
// `CommittedGroup<F>` having all of its trait bounds satisfied; declare it
// here to avoid a `cargo machete` style trim.
#[doc(hidden)]
pub use akita_algebra as _akita_algebra_dep;

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_config::proof_optimized::fp128;
    use akita_types::{
        derive_public_matrix_prefix, sample_akita_setup_seed, scheduled_setup_prefix,
        CommittedGroupProfile, CompressionChainPlan, GroupOpeningPlan, InnerCommitMatrixParams,
        OuterCommitMatrixParams, PolynomialGroupLayout, PrecommittedLevelParams, RingVec,
        SetupPrefixPublicCommitment, SetupPrefixVerifierSlot, SisMatrixRole, SisModulusProfileId,
        SisTableDigest, SisTableKey, DEFAULT_SIS_SECURITY_POLICY,
    };

    type TestCfg = fp128::OneHot;
    type TestF = fp128::Field;
    const TEST_D: usize = 256;
    const PREFIX_D: usize = 64;

    fn blob_prefix() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&BLOB_MAGIC);
        (TEST_D as u64)
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();
        bytes
    }

    fn prefix_commitment_params() -> PrecommittedLevelParams {
        let inner_commit_matrix = InnerCommitMatrixParams::try_new_with_min_rank(
            SisTableKey {
                policy: DEFAULT_SIS_SECURITY_POLICY,
                table_digest: SisTableDigest::CURRENT,
                modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
                role: SisMatrixRole::Inner,
                ring_dimension: u32::try_from(PREFIX_D).expect("test prefix ring dimension"),
                coeff_linf_bound: 32_767,
            },
            1,
        )
        .expect("audited prefix A matrix");
        let outer_commit_matrix = OuterCommitMatrixParams::try_new_with_min_rank(
            SisTableKey {
                policy: DEFAULT_SIS_SECURITY_POLICY,
                table_digest: SisTableDigest::CURRENT,
                modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
                role: SisMatrixRole::Outer,
                ring_dimension: u32::try_from(PREFIX_D).expect("test prefix ring dimension"),
                coeff_linf_bound: 3,
            },
            inner_commit_matrix.output_rank(),
        )
        .expect("audited prefix B matrix");
        PrecommittedLevelParams {
            layout: CommittedGroupProfile {
                version: CommittedGroupProfile::VERSION,
                group: PolynomialGroupLayout::singleton(PREFIX_D.trailing_zeros() as usize),
                num_live_ring_elements_per_claim: 1,
                num_positions_per_block: 1,
                num_live_blocks: 1,
                outer_slice_count: akita_types::CommitmentSliceCount::ONE,
                log_basis_inner: 1,
                num_digits_inner: 1,
                inner_commit_matrix,
                log_basis_outer: 1,
                num_digits_outer: 1,
                outer_commit_matrix,
            },
            opening: GroupOpeningPlan::evaluation_trace(
                SparseChallengeConfig::pm1_only(0),
                1,
                1,
                1,
            ),
        }
    }

    #[test]
    fn trailing_blob_bytes_are_rejected() {
        let err = reject_trailing_bytes(&[0]).unwrap_err();
        assert!(err.to_string().contains("trailing bytes"));
        reject_trailing_bytes(&[]).unwrap();
    }

    #[test]
    fn previous_blob_version_is_rejected_at_the_magic_boundary() {
        let mut bytes = blob_prefix();
        bytes[..BLOB_MAGIC.len()].copy_from_slice(b"AKJOLTv1");
        let error = AkitaJoltInputs::<TestF, TEST_D>::read_from_bytes::<TestCfg>(&bytes)
            .expect_err("v1 blob must not reach payload decoding");
        assert!(error.to_string().contains("magic mismatch"));
    }

    #[test]
    fn transcript_domain_len_is_capped_before_allocation() {
        let mut bytes = blob_prefix();
        ((MAX_TRANSCRIPT_DOMAIN_BYTES + 1) as u64)
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();

        let err = AkitaJoltInputs::<TestF, TEST_D>::read_from_bytes::<TestCfg>(&bytes).unwrap_err();
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

        let err = AkitaJoltInputs::<TestF, TEST_D>::read_from_bytes::<TestCfg>(&bytes).unwrap_err();
        assert!(err.to_string().contains("length"));
    }

    #[test]
    fn opening_point_len_must_match_num_vars_before_allocation() {
        let mut bytes = blob_prefix();
        Vec::<u8>::new()
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();
        2u64.serialize_with_mode(&mut bytes, BLOB_COMPRESS).unwrap();
        3u64.serialize_with_mode(&mut bytes, BLOB_COMPRESS).unwrap();

        let err = AkitaJoltInputs::<TestF, TEST_D>::read_from_bytes::<TestCfg>(&bytes).unwrap_err();
        assert!(err.to_string().contains("opening-point arity 3"));
    }

    #[test]
    fn strict_setup_decoder_preserves_prefix_slots() {
        let setup_seed = sample_akita_setup_seed();
        let seed = AkitaSetupDescriptor {
            max_num_vars: 8,
            max_num_batched_polys: 1,
            num_field_elements: 2 * TEST_D,
            setup_seed: setup_seed.clone(),
        };
        let shared_matrix = derive_public_matrix_prefix::<TestF>(2 * TEST_D, &setup_seed);
        let commitment_params = prefix_commitment_params();
        let matrix = &commitment_params.layout.outer_commit_matrix;
        let payload_coefficients = CompressionChainPlan::for_complete_source(
            matrix.sis_modulus_profile(),
            matrix.output_rank() * matrix.ring_dimension(),
        )
        .expect("setup-prefix compression plan")
        .terminal_coefficients();
        let id = scheduled_setup_prefix(1, commitment_params).slot_id();
        let mut prefix_slots = SetupPrefixVerifierRegistry::new(setup_seed.clone());
        prefix_slots
            .insert(SetupPrefixVerifierSlot {
                id: id.clone(),
                commitment: SetupPrefixPublicCommitment {
                    rows: vec![RingVec::from_coeffs(vec![
                        TestF::zero();
                        payload_coefficients
                    ])],
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

    #[test]
    fn proof_shape_budget_and_schedule_identity_precede_proof_allocation() {
        let row = TestCfg::resolve_catalog_row_for_opening(
            &akita_types::OpeningClaimsLayout::new(14, 1).expect("opening layout"),
        )
        .expect("generated singleton row");
        let canonical = canonical_base_field_proof_shape(row.schedule()).expect("canonical shape");

        let mut huge = canonical.clone();
        huge.root.opening_payload_coeffs = usize::MAX;
        let budget_error =
            AkitaJoltInputs::<TestF, TEST_D>::validate_proof_shape_before_allocation::<TestCfg>(
                row.selection(),
                &huge,
                0,
            )
            .expect_err("huge shape must fail against remaining bytes");
        let budget_message = budget_error.to_string();
        assert!(
            budget_message.contains("remaining proof bytes") || budget_message.contains("overflow"),
            "unexpected budget error: {budget_message}"
        );

        let mut noncanonical = canonical;
        noncanonical.root.opening_payload_coeffs += 1;
        let identity_error =
            AkitaJoltInputs::<TestF, TEST_D>::validate_proof_shape_before_allocation::<TestCfg>(
                row.selection(),
                &noncanonical,
                MAX_JOLT_BLOB_BYTES as usize,
            )
            .expect_err("noncanonical shape must fail before proof decoding");
        assert!(identity_error.to_string().contains("canonical schedule"));
    }
}
