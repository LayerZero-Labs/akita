//! Shared setup data shapes for Akita prover and verifier APIs.

use super::setup_prefix::SetupPrefixVerifierRegistry;
use akita_error::AkitaError;
use akita_params::FlatMatrix;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use jolt_algebra::{CanonicalEncoding, Field};
use jolt_crypto::lattice::{check_public_matrix_prefix, PublicMatrixSeed};
use jolt_crypto::CryptoError;
use std::io::{Read, Write};
use std::sync::Arc;

/// Maximum setup matrix field elements accepted by self-describing setup
/// deserialization.
///
/// This cap protects generic verifier-facing setup decoding from allocating
/// directly from attacker-controlled seed metadata. It is not a protocol limit
/// on the deterministic public stream. Context-backed decoders should instead
/// enforce an expected shape and caller-supplied resource budget.
pub const MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS: usize = 1 << 26;

/// Seed-only stage for deterministic setup expansion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaSetupDescriptor {
    /// Provisioning variable bound inherited from setup construction.
    pub max_num_vars: usize,
    /// Provisioning polynomial-count bound inherited from setup construction.
    pub max_num_batched_polys: usize,
    /// Number of materialized public-matrix field elements.
    pub num_field_elements: usize,
    /// Semantic identity of the infinite public field stream.
    pub setup_seed: PublicMatrixSeed,
}

/// Expanded setup stage containing materialized public matrices.
///
/// Base role matrices (A, B, D) are packed row/column prefix views of
/// `shared_matrix`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaExpandedSetup<F: Field> {
    /// Setup seed and runtime layout metadata.
    pub descriptor: AkitaSetupDescriptor,
    /// Shared 1D flat backing vector.
    pub shared_matrix: FlatMatrix<F>,
}

/// Verifier setup artifact derived from prover setup.
///
/// Semantic state is immutable so clones can safely share prepared matrix
/// caches. To replace a setup, construct a new value with [`Self::from_parts`].
///
/// ```compile_fail
/// use akita_types::AkitaVerifierSetup;
/// use jolt_algebra::Prime32Offset99;
/// fn replace(setup: &mut AkitaVerifierSetup<Prime32Offset99>,
///            other: AkitaVerifierSetup<Prime32Offset99>) {
///     setup.expanded = other.expanded;
/// }
/// ```
///
/// ```compile_fail
/// use akita_types::AkitaVerifierSetup;
/// use jolt_algebra::Prime32Offset99;
/// fn replace(setup: &mut AkitaVerifierSetup<Prime32Offset99>,
///            other: AkitaVerifierSetup<Prime32Offset99>) {
///     setup.prefix_slots = other.prefix_slots;
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaVerifierSetup<F: Field> {
    /// Expanded matrix stage used for verification.
    expanded: Arc<AkitaExpandedSetup<F>>,
    /// Public setup-prefix commitment metadata for setup-claim offloading.
    prefix_slots: SetupPrefixVerifierRegistry<F>,
}

impl<F: Field> AkitaVerifierSetup<F> {
    /// Borrow the immutable expanded matrix stage.
    pub fn expanded(&self) -> &Arc<AkitaExpandedSetup<F>> {
        &self.expanded
    }

    /// Borrow the public setup-prefix commitment metadata.
    pub fn prefix_slots(&self) -> &SetupPrefixVerifierRegistry<F> {
        &self.prefix_slots
    }

    /// Construct verifier setup state from expanded setup and structurally checked prefix metadata.
    ///
    /// This constructor binds the registry to the public matrix identity. It
    /// does not prove that each stored prefix commitment was derived from that
    /// matrix. Callers loading external registries must establish that
    /// provenance at their setup-installation boundary.
    pub fn from_parts(
        expanded: Arc<AkitaExpandedSetup<F>>,
        prefix_slots: SetupPrefixVerifierRegistry<F>,
    ) -> Result<Self, AkitaError> {
        if prefix_slots.setup_seed() != &expanded.descriptor.setup_seed {
            return Err(AkitaError::InvalidSetup(
                "setup-prefix registry belongs to a different public matrix".to_string(),
            ));
        }
        Ok(Self {
            expanded,
            prefix_slots,
        })
    }
}

impl<F: Field> AkitaExpandedSetup<F> {
    /// Build an expanded setup from a trusted matrix the caller has already
    /// derived from `descriptor.setup_seed`.
    ///
    /// This constructor deliberately does not rederive or validate the matrix. Use
    /// [`Self::from_verified_parts`] for untrusted serialized setup bytes.
    #[must_use]
    pub fn from_trusted_seed_derived_parts_unchecked(
        descriptor: AkitaSetupDescriptor,
        shared_matrix: FlatMatrix<F>,
    ) -> Self {
        Self {
            descriptor,
            shared_matrix,
        }
    }

    /// Setup seed and runtime layout metadata.
    #[must_use]
    pub fn descriptor(&self) -> &AkitaSetupDescriptor {
        &self.descriptor
    }

    /// Shared coefficient-form matrix backing all setup roles.
    #[must_use]
    pub fn shared_matrix(&self) -> &FlatMatrix<F> {
        &self.shared_matrix
    }
}

impl<F> AkitaExpandedSetup<F>
where
    F: Field + CanonicalEncoding + Valid,
{
    /// Build an expanded setup from untrusted parts and verify the materialized
    /// matrix against the public seed.
    ///
    /// # Errors
    ///
    /// Returns a serialization error if the seed/matrix shape is malformed or
    /// the matrix was not deterministically derived from the seed.
    pub fn from_verified_parts(
        descriptor: AkitaSetupDescriptor,
        shared_matrix: FlatMatrix<F>,
    ) -> Result<Self, SerializationError> {
        let out = Self {
            descriptor,
            shared_matrix,
        };
        out.check()?;
        Ok(out)
    }
}

/// Fixed public seed for deterministic, reproducible setup.
#[must_use]
pub fn sample_akita_setup_seed() -> PublicMatrixSeed {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&0xDEAD_BEEF_CAFE_BABEu64.to_le_bytes());
    PublicMatrixSeed::shake256_paged_v1(seed)
}

/// Check that a materialized public matrix has exactly the shape declared by
/// `descriptor`.
///
/// # Errors
///
/// Returns an error if either side is structurally malformed or if the matrix
/// field-element count differs from the descriptor.
pub fn validate_public_matrix_shape_matches_seed<F: Field + Valid>(
    shared_matrix: &FlatMatrix<F>,
    descriptor: &AkitaSetupDescriptor,
) -> Result<(), SerializationError> {
    descriptor.check()?;
    shared_matrix.check()?;
    if shared_matrix.num_field_elements() != descriptor.num_field_elements {
        return Err(SerializationError::InvalidData(
            "setup shared_matrix field count does not match setup descriptor".to_string(),
        ));
    }
    Ok(())
}

/// Check that a materialized public matrix is exactly the deterministic matrix
/// derived from `seed`.
///
/// # Errors
///
/// Returns an error if the matrix shape is malformed or if any coefficient
/// differs from the seed-derived public matrix.
pub fn validate_public_matrix_matches_seed<F: Field + CanonicalEncoding + Valid>(
    shared_matrix: &FlatMatrix<F>,
    descriptor: &AkitaSetupDescriptor,
) -> Result<(), SerializationError> {
    // The prefix check accepts any truncation of the derived stream; the shape
    // check pins the matrix length to the descriptor.
    validate_public_matrix_shape_matches_seed(shared_matrix, descriptor)?;
    check_public_matrix_prefix(shared_matrix.as_field_slice(), &descriptor.setup_seed).map_err(
        |error| match error {
            CryptoError::InvalidInput(_) => SerializationError::InvalidData(
                "setup shared_matrix does not match public matrix seed".to_string(),
            ),
            error => SerializationError::InvalidData(error.to_string()),
        },
    )
}

impl Valid for AkitaSetupDescriptor {
    fn check(&self) -> Result<(), SerializationError> {
        if self.max_num_batched_polys == 0 {
            return Err(SerializationError::InvalidData(
                "setup descriptor max_num_batched_polys must be at least 1".to_string(),
            ));
        }
        if self.num_field_elements == 0 {
            return Err(SerializationError::InvalidData(
                "setup descriptor num_field_elements must be non-zero".to_string(),
            ));
        }
        self.setup_seed.check()?;
        Ok(())
    }
}

impl AkitaSerialize for AkitaSetupDescriptor {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.max_num_vars
            .serialize_with_mode(&mut writer, compress)?;
        self.max_num_batched_polys
            .serialize_with_mode(&mut writer, compress)?;
        self.num_field_elements
            .serialize_with_mode(&mut writer, compress)?;
        self.setup_seed.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.max_num_vars.serialized_size(compress)
            + self.max_num_batched_polys.serialized_size(compress)
            + self.num_field_elements.serialized_size(compress)
            + self.setup_seed.serialized_size(compress)
    }
}

impl AkitaDeserialize for AkitaSetupDescriptor {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let max_num_vars = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let max_num_batched_polys =
            usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let num_field_elements =
            usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let setup_seed =
            PublicMatrixSeed::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self {
            max_num_vars,
            max_num_batched_polys,
            num_field_elements,
            setup_seed,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: Field + CanonicalEncoding + Valid> Valid for AkitaExpandedSetup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.descriptor.check()?;
        self.shared_matrix.check()?;
        validate_public_matrix_matches_seed(&self.shared_matrix, &self.descriptor)?;
        Ok(())
    }
}

impl<F: Field + AkitaSerialize> AkitaSerialize for AkitaExpandedSetup<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.descriptor.serialize_with_mode(&mut writer, compress)?;
        self.shared_matrix
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.descriptor.serialized_size(compress) + self.shared_matrix.serialized_size(compress)
    }
}

impl<F: Field + CanonicalEncoding + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for AkitaExpandedSetup<F>
{
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let descriptor =
            AkitaSetupDescriptor::deserialize_with_mode(&mut reader, compress, validate, &())?;
        descriptor.check()?;
        let shared_matrix = FlatMatrix::deserialize_with_expected_shape(
            &mut reader,
            compress,
            validate,
            descriptor.num_field_elements,
            MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
        )?;
        if matches!(validate, Validate::Yes) {
            Self::from_verified_parts(descriptor, shared_matrix)
        } else {
            Ok(Self::from_trusted_seed_derived_parts_unchecked(
                descriptor,
                shared_matrix,
            ))
        }
    }
}

impl<F: Field + CanonicalEncoding + Valid> Valid for AkitaVerifierSetup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.expanded.check()?;
        if self.prefix_slots.setup_seed() != &self.expanded.descriptor.setup_seed {
            return Err(SerializationError::InvalidData(
                "setup-prefix registry belongs to a different public matrix".to_string(),
            ));
        }
        self.prefix_slots.check()
    }
}

impl<F: Field + AkitaSerialize> AkitaSerialize for AkitaVerifierSetup<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let mut writer = writer;
        self.expanded.serialize_with_mode(&mut writer, compress)?;
        self.prefix_slots.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.expanded.serialized_size(compress) + self.prefix_slots.serialized_size(compress)
    }
}

impl<F: Field + CanonicalEncoding + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for AkitaVerifierSetup<F>
{
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let mut reader = reader;
        let expanded = Arc::new(AkitaExpandedSetup::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &(),
        )?);
        let prefix_slots =
            SetupPrefixVerifierRegistry::deserialize_with_mode(reader, compress, validate, &())?;
        Self::from_parts(expanded, prefix_slots)
            .map_err(|err| SerializationError::InvalidData(err.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_algebra::Zero;
    use jolt_algebra::{
        Fp64, Prime128Offset275, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59,
    };
    use jolt_crypto::lattice::derive_public_matrix_prefix;

    type F = Prime128OffsetA7F7;
    const D: usize = 4;
    type SmallF = Fp64<4294967197>;
    const SMALL_D: usize = 64;

    fn prefix_commitment_params(
        n_prefix: usize,
        d_setup: usize,
    ) -> akita_params::GroupOpenPhaseParams {
        let a_bound = *akita_params::sis::inner_coeff_linf_bounds(
            akita_params::sis::SisModulusProfileId::Q128OffsetA7F7,
            u32::try_from(d_setup).expect("test ring dimension"),
        )
        .first()
        .expect("exact prefix A bounds");
        let inner_commit_matrix = akita_params::InnerCommitMatrixParams::try_new_with_min_rank(
            akita_params::SisTableKey {
                policy: akita_params::sis::DEFAULT_SIS_SECURITY_POLICY,
                table_digest: akita_params::sis::SisTableDigest::CURRENT,
                modulus_profile: akita_params::sis::SisModulusProfileId::Q128OffsetA7F7,
                role: akita_params::sis::SisMatrixRole::Inner,
                ring_dimension: u32::try_from(d_setup).expect("test ring dimension"),
                coeff_linf_bound: a_bound,
            },
            1,
        )
        .expect("audited prefix A matrix");
        let outer_commit_matrix = akita_params::OuterCommitMatrixParams::try_new_with_min_rank(
            akita_params::SisTableKey {
                policy: akita_params::sis::DEFAULT_SIS_SECURITY_POLICY,
                table_digest: akita_params::sis::SisTableDigest::CURRENT,
                modulus_profile: akita_params::sis::SisModulusProfileId::Q128OffsetA7F7,
                role: akita_params::sis::SisMatrixRole::Outer,
                ring_dimension: u32::try_from(d_setup).expect("test ring dimension"),
                coeff_linf_bound: 3,
            },
            inner_commit_matrix.output_rank() * (n_prefix / d_setup),
        )
        .expect("audited prefix B matrix");
        akita_params::GroupOpenPhaseParams {
            setup_natural_len: None,
            profile: akita_params::GroupCommitPhaseParams {
                randomness_columns: 0,
                version: akita_params::GroupCommitPhaseParams::VERSION,
                group: akita_params::PolynomialGroupLayout::singleton(
                    n_prefix.trailing_zeros() as usize
                ),
                blocks: akita_params::BlockGeometry::new(n_prefix / d_setup, 1, n_prefix / d_setup),
                outer_slice_count: akita_params::CommitmentSliceCount::ONE,
                inner: akita_params::RoleParams::new(
                    akita_params::GadgetDigits::new(1, 1),
                    inner_commit_matrix,
                ),
                outer: akita_params::RoleParams::new(
                    akita_params::GadgetDigits::new(1, 1),
                    outer_commit_matrix,
                ),
            },
            opening: akita_params::GroupOpeningPlan::evaluation_trace(
                jolt_crypto::challenge::SparseChallengeConfig::pm1_only(0),
                1,
                1,
                1,
            ),
        }
    }

    fn derive_matrix<F: Field + CanonicalEncoding>(
        num_field_elements: usize,
        seed: &PublicMatrixSeed,
    ) -> FlatMatrix<F> {
        FlatMatrix::from_flat_data(
            derive_public_matrix_prefix::<F>(num_field_elements, seed).expect("derive matrix"),
        )
    }

    fn seed(public_matrix_seed: [u8; 32]) -> AkitaSetupDescriptor {
        AkitaSetupDescriptor {
            max_num_vars: 8,
            max_num_batched_polys: 1,
            num_field_elements: 2 * D,
            setup_seed: public_matrix_seed.into(),
        }
    }

    #[test]
    fn verifier_setup_prefix_slots_roundtrip() {
        use crate::proof::{RingVec, SetupPrefixPublicCommitment, SetupPrefixVerifierSlot};

        let setup_seed = seed([7u8; 32]);
        let shared_matrix = derive_matrix::<F>(2 * D, &setup_seed.setup_seed);
        let mut prefix_slots = SetupPrefixVerifierRegistry::new(setup_seed.setup_seed.clone());
        let d_setup = 64;
        let commitment_params = prefix_commitment_params(d_setup, d_setup);
        let matrix = &commitment_params.profile.outer.matrix;
        let payload_coefficients = akita_params::CompressionChainPlan::for_complete_source(
            matrix.sis_modulus_profile(),
            matrix.output_rank() * matrix.ring_dimension(),
        )
        .expect("setup-prefix compression plan")
        .terminal_coefficients();
        let slot = SetupPrefixVerifierSlot {
            id: akita_params::scheduled_setup_prefix(d_setup - 1, commitment_params)
                .slot_id()
                .expect("setup prefix group"),
            commitment: SetupPrefixPublicCommitment {
                rows: vec![RingVec::from_coeffs(vec![F::zero(); payload_coefficients])],
            },
        };
        prefix_slots.insert(slot).expect("insert prefix slot");
        let setup = AkitaVerifierSetup {
            expanded: Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    setup_seed,
                    shared_matrix,
                ),
            ),
            prefix_slots,
        };

        let mut bytes = Vec::new();
        setup.serialize_compressed(&mut bytes).expect("serialize");
        let decoded = AkitaVerifierSetup::<F>::deserialize_compressed_exact(&bytes, &())
            .expect("deserialize");

        assert_eq!(decoded.prefix_slots.len(), 1);
        assert_eq!(decoded, setup);

        for suffix in [0, 0xa5] {
            let mut suffixed = bytes.clone();
            suffixed.push(suffix);
            assert!(AkitaVerifierSetup::<F>::deserialize_compressed_exact(&suffixed, &()).is_err());
        }

        let mut expanded_bytes = Vec::new();
        setup
            .expanded
            .serialize_compressed(&mut expanded_bytes)
            .expect("serialize expanded setup");
        assert!(
            AkitaExpandedSetup::<F>::deserialize_compressed_exact(&expanded_bytes, &()).is_ok()
        );
        for suffix in [0, 0xa5] {
            let mut suffixed = expanded_bytes.clone();
            suffixed.push(suffix);
            assert!(AkitaExpandedSetup::<F>::deserialize_compressed_exact(&suffixed, &()).is_err());
        }
    }

    #[test]
    fn verifier_setup_rejects_prefix_registry_from_another_public_matrix() {
        let setup_seed = seed([7u8; 32]);
        let shared_matrix = derive_matrix::<F>(2 * D, &setup_seed.setup_seed);
        let expanded = Arc::new(
            AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                setup_seed,
                shared_matrix,
            ),
        );
        let foreign_registry =
            SetupPrefixVerifierRegistry::new(PublicMatrixSeed::shake256_paged_v1([9u8; 32]));

        let err = AkitaVerifierSetup::from_parts(expanded, foreign_registry)
            .expect_err("cross-seed prefix registry must be rejected");
        assert!(err.to_string().contains("different public matrix"));
    }

    #[test]
    fn strict_verifier_setup_decode_rejects_matrix_not_derived_from_seed() {
        let descriptor = seed([7u8; 32]);
        let setup_seed = descriptor.setup_seed.clone();
        let wrong_seed = PublicMatrixSeed::shake256_paged_v1([9u8; 32]);
        let wrong_matrix = derive_matrix::<F>(2 * D, &wrong_seed);
        let setup = AkitaVerifierSetup {
            expanded: Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    descriptor,
                    wrong_matrix,
                ),
            ),
            prefix_slots: SetupPrefixVerifierRegistry::new(setup_seed),
        };

        let mut bytes = Vec::new();
        setup.serialize_compressed(&mut bytes).unwrap();
        let err = AkitaVerifierSetup::<F>::deserialize_compressed(&bytes[..], &()).unwrap_err();

        assert!(err
            .to_string()
            .contains("setup shared_matrix does not match public matrix seed"));
    }

    #[test]
    fn strict_verifier_setup_decode_rejects_truncated_seed_prefix_matrix() {
        let descriptor = seed([7u8; 32]);
        let setup_seed = descriptor.setup_seed.clone();
        let short_matrix = derive_matrix::<F>(D, &setup_seed);
        let setup = AkitaVerifierSetup {
            expanded: Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    descriptor,
                    short_matrix,
                ),
            ),
            prefix_slots: SetupPrefixVerifierRegistry::new(setup_seed),
        };

        let mut bytes = Vec::new();
        setup.serialize_compressed(&mut bytes).unwrap();
        let err = AkitaVerifierSetup::<F>::deserialize_compressed(&bytes[..], &()).unwrap_err();

        assert!(err
            .to_string()
            .contains("flat matrix field count does not match expected setup shape"));
    }

    #[test]
    fn strict_setup_decode_rejects_matrix_shape_before_payload() {
        let setup_seed = seed([7u8; 32]);
        let mut bytes = Vec::new();
        setup_seed.serialize_compressed(&mut bytes).unwrap();
        usize::MAX.serialize_compressed(&mut bytes).unwrap();
        let err = AkitaExpandedSetup::<F>::deserialize_compressed(&bytes[..], &()).unwrap_err();

        assert!(err
            .to_string()
            .contains("flat matrix field count does not match expected setup shape"));
    }

    #[test]
    fn setup_seed_validity_is_not_the_generic_decode_allocation_cap() {
        let setup_seed = AkitaSetupDescriptor {
            max_num_vars: 32,
            max_num_batched_polys: 1,
            num_field_elements: MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS + 1,
            setup_seed: [7u8; 32].into(),
        };

        setup_seed.check().unwrap();
        assert!(setup_seed.num_field_elements > MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS);
    }

    #[test]
    fn generic_setup_decode_still_rejects_shapes_above_allocation_cap() {
        let setup_seed = AkitaSetupDescriptor {
            max_num_vars: 32,
            max_num_batched_polys: 1,
            num_field_elements: MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS + 1,
            setup_seed: [7u8; 32].into(),
        };
        let mut bytes = Vec::new();
        setup_seed.serialize_compressed(&mut bytes).unwrap();

        let err = AkitaExpandedSetup::<F>::deserialize_compressed(&bytes[..], &()).unwrap_err();

        assert!(matches!(
            err,
            SerializationError::LengthLimitExceeded { max, .. }
                if max == MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS
        ));
    }

    #[test]
    fn flat_derivation_is_prefix_stable() {
        let seed = PublicMatrixSeed::shake256_paged_v1([7u8; 32]);
        let small = derive_matrix::<SmallF>(6 * SMALL_D, &seed);
        let large = derive_matrix::<SmallF>(24 * SMALL_D, &seed);
        let small_view = small.ring_view::<SMALL_D>(1, 6).unwrap();
        let large_view = large.ring_view::<SMALL_D>(1, 6).unwrap();
        for c in 0..6 {
            assert_eq!(small_view.row(0).unwrap()[c], large_view.row(0).unwrap()[c]);
        }
    }

    #[test]
    fn paged_derivation_golden_vector() {
        let seed = PublicMatrixSeed::shake256_paged_v1([31u8; 32]);
        fn samples<F: Field + CanonicalEncoding>(seed: &PublicMatrixSeed) -> [u128; 3] {
            let page_field_elements = seed.derivation.page_field_elements();
            let derived = derive_matrix::<F>(page_field_elements + 64, seed);
            let canonical = derived
                .as_field_slice()
                .iter()
                .map(|value| {
                    value
                        .to_u128_checked()
                        .expect("Akita field element must fit in u128")
                })
                .collect::<Vec<_>>();
            [
                canonical[0],
                canonical[page_field_elements - 1],
                canonical[page_field_elements],
            ]
        }

        assert_eq!(
            samples::<Prime32Offset99>(&seed),
            [985_701_565, 215_851_758, 196_317_274]
        );
        assert_eq!(
            samples::<Prime64Offset59>(&seed),
            [
                15_459_661_060_209_904_737,
                1_106_841_764_157_043_686,
                11_841_567_322_073_738_392,
            ]
        );
        assert_eq!(
            samples::<Prime128Offset275>(&seed),
            [
                9_840_922_769_526_400_152_209_492_837_491_680_711,
                87_058_165_705_274_552_119_584_186_413_843_782_366,
                214_441_952_004_995_181_775_787_633_634_410_275_750,
            ]
        );
    }

    #[test]
    fn different_shapes_from_same_flat() {
        let seed = PublicMatrixSeed::shake256_paged_v1([13u8; 32]);
        let flat = derive_matrix::<SmallF>(12 * SMALL_D, &seed);
        let view_3x4 = flat.ring_view::<SMALL_D>(3, 4).unwrap();
        let view_2x6 = flat.ring_view::<SMALL_D>(2, 6).unwrap();

        assert_eq!(view_3x4.row(0).unwrap()[0], view_2x6.row(0).unwrap()[0]);
        assert_eq!(view_3x4.row(0).unwrap()[3], view_2x6.row(0).unwrap()[3]);
        assert_ne!(view_3x4.row(1).unwrap()[0], view_2x6.row(1).unwrap()[0]);
    }
}
