//! Shared setup data shapes for Akita prover and verifier APIs.

use crate::FlatMatrix;
use akita_algebra::CyclotomicRing;
#[allow(unused_imports)]
use akita_field::parallel::*;
use akita_field::{FieldCore, RandomSampling};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use rand_core::{CryptoRng, RngCore};
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake256;
use std::io::{Read, Write};
use std::sync::Arc;

/// Public seed used to derive commitment matrices.
pub type PublicMatrixSeed = [u8; 32];

/// Maximum setup matrix field elements accepted by self-describing setup
/// deserialization.
///
/// Config-backed cache paths should enforce tighter exact shape bounds before
/// decoding the matrix body. This cap protects generic verifier-facing setup
/// decoding from allocating directly from attacker-controlled seed metadata.
pub const MAX_SETUP_MATRIX_FIELD_ELEMENTS: usize = 1 << 26;

const PUBLIC_MATRIX_DOMAIN: &[u8] = b"akita/commitment/public-matrix-1d";
const SHARED_MATRIX_LABEL: &[u8] = b"shared";

/// Seed-only stage for deterministic setup expansion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaSetupSeed {
    /// Maximum supported variable count.
    pub max_num_vars: usize,
    /// Maximum number of batched polynomials supported by setup.
    pub max_num_batched_polys: usize,
    /// Maximum number of distinct opening points.
    ///
    /// Together with `max_num_batched_polys` this bounds the outer/D matrix
    /// widths the setup can serve; a multi-point batched opening that exceeds
    /// this bound would otherwise silently read past the shared matrix prefix.
    pub max_num_points: usize,
    /// Global row stride for the flat NTT cache.
    pub max_stride: usize,
    /// Ring dimension used to generate `shared_matrix`.
    pub gen_ring_dim: usize,
    /// Number of generated ring elements at `gen_ring_dim`.
    pub total_ring_elements: usize,
    /// Public seed used to derive commitment matrices.
    pub public_matrix_seed: PublicMatrixSeed,
}

impl AkitaSetupSeed {
    /// Number of field elements in the serialized shared matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if the seed shape overflows `usize`.
    pub fn matrix_field_elements(&self) -> Result<usize, SerializationError> {
        self.total_ring_elements
            .checked_mul(self.gen_ring_dim)
            .ok_or_else(|| {
                SerializationError::InvalidData(
                    "setup seed matrix field count overflow".to_string(),
                )
            })
    }
}

/// Expanded setup stage containing a single shared coefficient-form matrix.
///
/// All role matrices (A, B, D) are row/column prefixes of this shared vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaExpandedSetup<F: FieldCore> {
    /// Setup seed and runtime layout metadata.
    pub seed: AkitaSetupSeed,
    /// Shared 1D flat backing vector.
    pub shared_matrix: FlatMatrix<F>,
}

/// Verifier setup artifact derived from prover setup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaVerifierSetup<F: FieldCore> {
    /// Expanded matrix stage used for verification.
    pub expanded: Arc<AkitaExpandedSetup<F>>,
}

impl<F: FieldCore> AkitaExpandedSetup<F> {
    /// Build an expanded setup from a trusted matrix the caller has already
    /// derived from `seed.public_matrix_seed`.
    ///
    /// This constructor deliberately does not rederive or validate the matrix. Use
    /// [`Self::from_verified_parts`] for untrusted serialized setup bytes.
    #[must_use]
    pub fn from_trusted_seed_derived_parts_unchecked(
        seed: AkitaSetupSeed,
        shared_matrix: FlatMatrix<F>,
    ) -> Self {
        Self {
            seed,
            shared_matrix,
        }
    }

    /// Setup seed and runtime layout metadata.
    #[must_use]
    pub fn seed(&self) -> &AkitaSetupSeed {
        &self.seed
    }

    /// Shared coefficient-form matrix backing all setup roles.
    #[must_use]
    pub fn shared_matrix(&self) -> &FlatMatrix<F> {
        &self.shared_matrix
    }
}

impl<F> AkitaExpandedSetup<F>
where
    F: FieldCore + RandomSampling + Valid,
{
    /// Build an expanded setup from untrusted parts and verify the materialized
    /// matrix against the public seed.
    ///
    /// # Errors
    ///
    /// Returns a serialization error if the seed/matrix shape is malformed or
    /// the matrix was not deterministically derived from the seed.
    pub fn from_verified_parts(
        seed: AkitaSetupSeed,
        shared_matrix: FlatMatrix<F>,
    ) -> Result<Self, SerializationError> {
        let out = Self {
            seed,
            shared_matrix,
        };
        out.check()?;
        Ok(out)
    }
}

/// Fixed public seed for deterministic, reproducible setup.
#[must_use]
pub fn sample_public_matrix_seed() -> PublicMatrixSeed {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&0xDEAD_BEEF_CAFE_BABEu64.to_le_bytes());
    seed
}

/// Derive a flat public vector of ring elements from a seed.
///
/// All role matrices (A, B, D) share one backing vector with a fixed label
/// (`"shared"`). Each role views a prefix of this vector reshaped with its
/// own `(num_rows, num_cols)` dimensions.
///
/// Domain separation uses a single flat index so that a vector of length N is
/// a prefix of any vector of length M > N derived from the same seed.
#[tracing::instrument(skip_all, name = "derive_public_matrix_flat")]
#[must_use]
pub fn derive_public_matrix_flat<F: FieldCore + RandomSampling, const D: usize>(
    total_ring_elements: usize,
    seed: &PublicMatrixSeed,
) -> FlatMatrix<F> {
    let ring_elements: Vec<CyclotomicRing<F, D>> = cfg_into_iter!(0..total_ring_elements)
        .map(|idx| {
            let mut entry_rng = ShakeXofRng::new(seed, idx);
            CyclotomicRing::random(&mut entry_rng)
        })
        .collect();

    let mut data = Vec::with_capacity(total_ring_elements * D);
    for ring in ring_elements {
        data.extend(ring.coeffs);
    }

    FlatMatrix::from_flat_data(data, D)
}

fn fill_public_matrix_coefficients<F: FieldCore + RandomSampling>(
    seed: &PublicMatrixSeed,
    flat_index: usize,
    coeffs: &mut [F],
) {
    let mut entry_rng = ShakeXofRng::new(seed, flat_index);
    for coeff in coeffs {
        *coeff = F::random(&mut entry_rng);
    }
}

/// Check that a materialized public matrix has exactly the shape declared by
/// `seed`.
///
/// # Errors
///
/// Returns an error if either side is structurally malformed or if the matrix
/// generation dimension / length differs from the seed.
pub fn validate_public_matrix_shape_matches_seed<F: FieldCore + Valid>(
    shared_matrix: &FlatMatrix<F>,
    seed: &AkitaSetupSeed,
) -> Result<(), SerializationError> {
    seed.check()?;
    shared_matrix.check()?;
    let gen_ring_dim = shared_matrix.gen_ring_dim();
    if gen_ring_dim != seed.gen_ring_dim {
        return Err(SerializationError::InvalidData(
            "setup shared_matrix generation dimension does not match setup seed".to_string(),
        ));
    }
    if shared_matrix.total_ring_elements() != seed.total_ring_elements {
        return Err(SerializationError::InvalidData(
            "setup shared_matrix length does not match setup seed".to_string(),
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
pub fn validate_public_matrix_matches_seed<F: FieldCore + RandomSampling + Valid>(
    shared_matrix: &FlatMatrix<F>,
    seed: &AkitaSetupSeed,
) -> Result<(), SerializationError> {
    validate_public_matrix_shape_matches_seed(shared_matrix, seed)?;
    let gen_ring_dim = shared_matrix.gen_ring_dim();
    for (idx, coeffs) in shared_matrix
        .as_field_slice()
        .chunks_exact(gen_ring_dim)
        .enumerate()
    {
        let mut expected = vec![F::zero(); gen_ring_dim];
        fill_public_matrix_coefficients(&seed.public_matrix_seed, idx, &mut expected);
        if coeffs != expected.as_slice() {
            return Err(SerializationError::InvalidData(
                "setup shared_matrix does not match public matrix seed".to_string(),
            ));
        }
    }
    Ok(())
}

struct ShakeXofRng {
    reader: Box<dyn XofReader>,
}

impl ShakeXofRng {
    fn new(seed: &PublicMatrixSeed, flat_index: usize) -> Self {
        let mut xof = Shake256::default();
        absorb_len_prefixed(&mut xof, b"domain", PUBLIC_MATRIX_DOMAIN);
        absorb_len_prefixed(&mut xof, b"seed", seed);
        absorb_len_prefixed(&mut xof, b"matrix", SHARED_MATRIX_LABEL);
        absorb_len_prefixed(&mut xof, b"index", &(flat_index as u64).to_le_bytes());
        Self {
            reader: Box::new(xof.finalize_xof()),
        }
    }
}

impl RngCore for ShakeXofRng {
    fn next_u32(&mut self) -> u32 {
        let mut buf = [0u8; 4];
        self.fill_bytes(&mut buf);
        u32::from_le_bytes(buf)
    }

    fn next_u64(&mut self) -> u64 {
        let mut buf = [0u8; 8];
        self.fill_bytes(&mut buf);
        u64::from_le_bytes(buf)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.reader.read(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl CryptoRng for ShakeXofRng {}

fn absorb_len_prefixed(xof: &mut Shake256, label: &[u8], data: &[u8]) {
    xof.update(&(label.len() as u64).to_le_bytes());
    xof.update(label);
    xof.update(&(data.len() as u64).to_le_bytes());
    xof.update(data);
}

impl Valid for AkitaSetupSeed {
    fn check(&self) -> Result<(), SerializationError> {
        if self.max_stride == 0 {
            return Err(SerializationError::InvalidData(
                "setup seed max_stride must be non-zero".to_string(),
            ));
        }
        if self.max_num_batched_polys == 0 {
            return Err(SerializationError::InvalidData(
                "setup seed max_num_batched_polys must be at least 1".to_string(),
            ));
        }
        if self.max_num_points == 0 {
            return Err(SerializationError::InvalidData(
                "setup seed max_num_points must be at least 1".to_string(),
            ));
        }
        if self.gen_ring_dim == 0 {
            return Err(SerializationError::InvalidData(
                "setup seed gen_ring_dim must be non-zero".to_string(),
            ));
        }
        if self.total_ring_elements == 0 {
            return Err(SerializationError::InvalidData(
                "setup seed total_ring_elements must be non-zero".to_string(),
            ));
        }
        if !self.total_ring_elements.is_multiple_of(self.max_stride) {
            return Err(SerializationError::InvalidData(
                "setup seed total_ring_elements must be a multiple of max_stride".to_string(),
            ));
        }
        let matrix_field_elements = self.matrix_field_elements()?;
        if matrix_field_elements > MAX_SETUP_MATRIX_FIELD_ELEMENTS {
            return Err(SerializationError::LengthLimitExceeded {
                len: u64::try_from(matrix_field_elements).unwrap_or(u64::MAX),
                max: MAX_SETUP_MATRIX_FIELD_ELEMENTS,
            });
        }
        Ok(())
    }
}

impl AkitaSerialize for AkitaSetupSeed {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.max_num_vars
            .serialize_with_mode(&mut writer, compress)?;
        self.max_num_batched_polys
            .serialize_with_mode(&mut writer, compress)?;
        self.max_num_points
            .serialize_with_mode(&mut writer, compress)?;
        self.max_stride.serialize_with_mode(&mut writer, compress)?;
        self.gen_ring_dim
            .serialize_with_mode(&mut writer, compress)?;
        self.total_ring_elements
            .serialize_with_mode(&mut writer, compress)?;
        writer.write_all(&self.public_matrix_seed)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.max_num_vars.serialized_size(compress)
            + self.max_num_batched_polys.serialized_size(compress)
            + self.max_num_points.serialized_size(compress)
            + self.max_stride.serialized_size(compress)
            + self.gen_ring_dim.serialized_size(compress)
            + self.total_ring_elements.serialized_size(compress)
            + 32
    }
}

impl AkitaDeserialize for AkitaSetupSeed {
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
        let max_num_points = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let max_stride = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let gen_ring_dim = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let total_ring_elements =
            usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let mut public_matrix_seed = [0u8; 32];
        reader.read_exact(&mut public_matrix_seed)?;
        let out = Self {
            max_num_vars,
            max_num_batched_polys,
            max_num_points,
            max_stride,
            gen_ring_dim,
            total_ring_elements,
            public_matrix_seed,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + RandomSampling + Valid> Valid for AkitaExpandedSetup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.seed.check()?;
        self.shared_matrix.check()?;
        validate_public_matrix_matches_seed(&self.shared_matrix, &self.seed)?;
        Ok(())
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for AkitaExpandedSetup<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.seed.serialize_with_mode(&mut writer, compress)?;
        self.shared_matrix
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.seed.serialized_size(compress) + self.shared_matrix.serialized_size(compress)
    }
}

impl<F: FieldCore + RandomSampling + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for AkitaExpandedSetup<F>
{
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let seed = AkitaSetupSeed::deserialize_with_mode(&mut reader, compress, validate, &())?;
        seed.check()?;
        let shared_matrix = FlatMatrix::deserialize_with_expected_shape(
            &mut reader,
            compress,
            validate,
            seed.total_ring_elements,
            seed.gen_ring_dim,
            MAX_SETUP_MATRIX_FIELD_ELEMENTS,
        )?;
        if matches!(validate, Validate::Yes) {
            Self::from_verified_parts(seed, shared_matrix)
        } else {
            Ok(Self::from_trusted_seed_derived_parts_unchecked(
                seed,
                shared_matrix,
            ))
        }
    }
}

impl<F: FieldCore + RandomSampling + Valid> Valid for AkitaVerifierSetup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.expanded.check()
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for AkitaVerifierSetup<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.expanded.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.expanded.serialized_size(compress)
    }
}

impl<F: FieldCore + RandomSampling + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for AkitaVerifierSetup<F>
{
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            expanded: Arc::new(AkitaExpandedSetup::deserialize_with_mode(
                reader,
                compress,
                validate,
                &(),
            )?),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{Fp64, Prime128Offset275};

    type F = Prime128Offset275;
    const D: usize = 4;
    type SmallF = Fp64<4294967197>;
    const SMALL_D: usize = 64;

    fn seed(public_matrix_seed: PublicMatrixSeed) -> AkitaSetupSeed {
        AkitaSetupSeed {
            max_num_vars: 8,
            max_num_batched_polys: 1,
            max_num_points: 1,
            max_stride: 2,
            gen_ring_dim: D,
            total_ring_elements: 2,
            public_matrix_seed,
        }
    }

    #[test]
    fn strict_verifier_setup_decode_rejects_matrix_not_derived_from_seed() {
        let setup_seed = seed([7u8; 32]);
        let wrong_seed = [9u8; 32];
        let wrong_matrix = derive_public_matrix_flat::<F, D>(2, &wrong_seed);
        let setup = AkitaVerifierSetup {
            expanded: Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    setup_seed,
                    wrong_matrix,
                ),
            ),
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
        let setup_seed = seed([7u8; 32]);
        let short_matrix = derive_public_matrix_flat::<F, D>(1, &setup_seed.public_matrix_seed);
        let setup = AkitaVerifierSetup {
            expanded: Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    setup_seed,
                    short_matrix,
                ),
            ),
        };

        let mut bytes = Vec::new();
        setup.serialize_compressed(&mut bytes).unwrap();
        let err = AkitaVerifierSetup::<F>::deserialize_compressed(&bytes[..], &()).unwrap_err();

        assert!(err
            .to_string()
            .contains("flat matrix total_ring_elements does not match expected setup shape"));
    }

    #[test]
    fn strict_setup_decode_rejects_matrix_shape_before_payload() {
        let setup_seed = seed([7u8; 32]);
        let mut bytes = Vec::new();
        setup_seed.serialize_compressed(&mut bytes).unwrap();
        usize::MAX.serialize_compressed(&mut bytes).unwrap();
        D.serialize_compressed(&mut bytes).unwrap();
        let err = AkitaExpandedSetup::<F>::deserialize_compressed(&bytes[..], &()).unwrap_err();

        assert!(err
            .to_string()
            .contains("flat matrix total_ring_elements does not match expected setup shape"));
    }

    #[test]
    fn flat_derivation_is_deterministic_for_same_seed() {
        let seed = [42u8; 32];
        let m1 = derive_public_matrix_flat::<SmallF, SMALL_D>(15, &seed);
        let m2 = derive_public_matrix_flat::<SmallF, SMALL_D>(15, &seed);
        assert_eq!(m1, m2);
    }

    #[test]
    fn flat_derivation_is_prefix_stable() {
        let seed = [7u8; 32];
        let small = derive_public_matrix_flat::<SmallF, SMALL_D>(6, &seed);
        let large = derive_public_matrix_flat::<SmallF, SMALL_D>(24, &seed);
        let small_view = small.ring_view::<SMALL_D>(1, 6).unwrap();
        let large_view = large.ring_view::<SMALL_D>(1, 6).unwrap();
        for c in 0..6 {
            assert_eq!(small_view.row(0).unwrap()[c], large_view.row(0).unwrap()[c]);
        }
    }

    #[test]
    fn flat_derivation_matches_ring_random_stream() {
        let seed = [5u8; 32];
        let got = derive_public_matrix_flat::<SmallF, SMALL_D>(6, &seed);
        let expected = (0..6)
            .flat_map(|idx| {
                let mut rng = ShakeXofRng::new(&seed, idx);
                CyclotomicRing::<SmallF, SMALL_D>::random(&mut rng).coeffs
            })
            .collect::<Vec<_>>();

        assert_eq!(got.as_field_slice(), expected.as_slice());
    }

    #[test]
    fn different_shapes_from_same_flat() {
        let seed = [13u8; 32];
        let flat = derive_public_matrix_flat::<SmallF, SMALL_D>(12, &seed);
        let view_3x4 = flat.ring_view::<SMALL_D>(3, 4).unwrap();
        let view_2x6 = flat.ring_view::<SMALL_D>(2, 6).unwrap();

        assert_eq!(view_3x4.row(0).unwrap()[0], view_2x6.row(0).unwrap()[0]);
        assert_eq!(view_3x4.row(0).unwrap()[3], view_2x6.row(0).unwrap()[3]);
        assert_ne!(view_3x4.row(1).unwrap()[0], view_2x6.row(1).unwrap()[0]);
    }
}
