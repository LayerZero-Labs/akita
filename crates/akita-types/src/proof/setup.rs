//! Shared setup data shapes for Akita prover and verifier APIs.

use crate::{FlatMatrix, SetupIdentityDigests};
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
    /// Public seed used to derive commitment matrices.
    pub public_matrix_seed: PublicMatrixSeed,
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
    /// Cached descriptor digest for deterministic setup identity.
    pub descriptor_digests: SetupIdentityDigests,
}

/// Verifier setup artifact derived from prover setup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaVerifierSetup<F: FieldCore> {
    /// Expanded matrix stage used for verification.
    pub expanded: Arc<AkitaExpandedSetup<F>>,
}

impl<F> AkitaExpandedSetup<F>
where
    F: FieldCore + AkitaSerialize,
{
    /// Build an expanded setup and compute its cached descriptor digests.
    ///
    /// # Errors
    ///
    /// Returns a serialization error if the setup seed cannot be canonically
    /// serialized for descriptor hashing.
    pub fn from_parts(
        seed: AkitaSetupSeed,
        shared_matrix: FlatMatrix<F>,
    ) -> Result<Self, SerializationError> {
        let descriptor_digests = SetupIdentityDigests::from_seed(&seed)?;
        Ok(Self {
            seed,
            shared_matrix,
            descriptor_digests,
        })
    }

    /// Validate seed, structure, field elements, and cached descriptor digests
    /// without re-deriving the public matrix from the seed.
    ///
    /// This is only for trusted cache paths where the host intentionally ships
    /// a previously materialized matrix. Ordinary setup deserialization uses
    /// [`Valid::check`] and revalidates matrix determinism.
    pub fn check_trusted_cached_matrix(&self) -> Result<(), SerializationError>
    where
        F: Valid,
    {
        self.seed.check()?;
        self.shared_matrix.check()?;
        self.descriptor_digests.check_seed(&self.seed)?;
        Ok(())
    }
}

impl<F> AkitaExpandedSetup<F>
where
    F: FieldCore + Valid + AkitaDeserialize<Context = ()> + AkitaSerialize,
{
    /// Deserialize a trusted cached setup while skipping public-matrix
    /// rederivation.
    ///
    /// Field elements, shape metadata, and cached seed digests are still
    /// validated when `validate == Validate::Yes`; only the expensive
    /// seed-to-matrix recomputation is skipped.
    pub fn deserialize_with_trusted_cached_matrix<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let seed = AkitaSetupSeed::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let shared_matrix =
            FlatMatrix::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self::from_parts(seed, shared_matrix)?;
        if matches!(validate, Validate::Yes) {
            out.check_trusted_cached_matrix()?;
        }
        Ok(out)
    }
}

impl<F> AkitaVerifierSetup<F>
where
    F: FieldCore + Valid + AkitaDeserialize<Context = ()> + AkitaSerialize,
{
    /// Deserialize a trusted cached verifier setup while skipping public-matrix
    /// rederivation.
    ///
    /// Use this only when the serialized setup blob was produced inside a
    /// trusted host-side cache boundary. General verifier setup decoding should
    /// use [`AkitaDeserialize`] so the matrix is checked against the seed.
    pub fn deserialize_with_trusted_cached_matrix<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            expanded: Arc::new(AkitaExpandedSetup::deserialize_with_trusted_cached_matrix(
                reader, compress, validate,
            )?),
        })
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
    let mut data = Vec::with_capacity(total_ring_elements.saturating_mul(D));
    for idx in 0..total_ring_elements {
        let mut entry_rng = ShakeXofRng::new(seed, idx);
        for _ in 0..D {
            data.push(F::random(&mut entry_rng));
        }
    }
    FlatMatrix::from_flat_data(data, D)
}

/// Fallible public matrix derivation for runtime-selected generation
/// dimensions.
///
/// # Errors
///
/// Returns an error if `gen_ring_dim` is zero, if the field-element count
/// overflows, or if the output allocation cannot be reserved.
pub fn derive_public_matrix_flat_with_dimension<F: FieldCore + RandomSampling>(
    total_ring_elements: usize,
    gen_ring_dim: usize,
    seed: &PublicMatrixSeed,
) -> Result<FlatMatrix<F>, SerializationError> {
    if gen_ring_dim == 0 {
        return Err(SerializationError::InvalidData(
            "public matrix generation dimension must be non-zero".to_string(),
        ));
    }
    let total_fields = total_ring_elements
        .checked_mul(gen_ring_dim)
        .ok_or_else(|| {
            SerializationError::InvalidData("public matrix field count overflow".to_string())
        })?;
    let mut data = Vec::new();
    data.try_reserve_exact(total_fields).map_err(|_| {
        SerializationError::InvalidData("public matrix allocation failed".to_string())
    })?;
    for idx in 0..total_ring_elements {
        let mut entry_rng = ShakeXofRng::new(seed, idx);
        for _ in 0..gen_ring_dim {
            data.push(F::random(&mut entry_rng));
        }
    }
    Ok(FlatMatrix::from_flat_data(data, gen_ring_dim))
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
    seed: &PublicMatrixSeed,
) -> Result<(), SerializationError> {
    shared_matrix.check()?;
    let gen_ring_dim = shared_matrix.gen_ring_dim();
    if gen_ring_dim == 0 {
        return Err(SerializationError::InvalidData(
            "public matrix generation dimension must be non-zero".to_string(),
        ));
    }
    for (idx, coeffs) in shared_matrix
        .as_field_slice()
        .chunks_exact(gen_ring_dim)
        .enumerate()
    {
        let mut entry_rng = ShakeXofRng::new(seed, idx);
        for coeff in coeffs {
            if *coeff != F::random(&mut entry_rng) {
                return Err(SerializationError::InvalidData(
                    "setup shared_matrix does not match public matrix seed".to_string(),
                ));
            }
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
        writer.write_all(&self.public_matrix_seed)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.max_num_vars.serialized_size(compress)
            + self.max_num_batched_polys.serialized_size(compress)
            + self.max_num_points.serialized_size(compress)
            + self.max_stride.serialized_size(compress)
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
        let mut public_matrix_seed = [0u8; 32];
        reader.read_exact(&mut public_matrix_seed)?;
        let out = Self {
            max_num_vars,
            max_num_batched_polys,
            max_num_points,
            max_stride,
            public_matrix_seed,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + RandomSampling + Valid + AkitaSerialize> Valid for AkitaExpandedSetup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.check_trusted_cached_matrix()?;
        validate_public_matrix_matches_seed(&self.shared_matrix, &self.seed.public_matrix_seed)?;
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
where
    F: AkitaSerialize,
{
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let seed = AkitaSetupSeed::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let shared_matrix =
            FlatMatrix::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self::from_parts(seed, shared_matrix)?;
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + RandomSampling + Valid + AkitaSerialize> Valid for AkitaVerifierSetup<F> {
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
where
    F: AkitaSerialize,
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
    use akita_field::Prime128Offset275;

    type F = Prime128Offset275;
    const D: usize = 4;

    fn seed(public_matrix_seed: PublicMatrixSeed) -> AkitaSetupSeed {
        AkitaSetupSeed {
            max_num_vars: 8,
            max_num_batched_polys: 1,
            max_num_points: 1,
            max_stride: 2,
            public_matrix_seed,
        }
    }

    #[test]
    fn strict_verifier_setup_decode_rejects_matrix_not_derived_from_seed() {
        let setup_seed = seed([7u8; 32]);
        let wrong_seed = [9u8; 32];
        let wrong_matrix = derive_public_matrix_flat::<F, D>(2, &wrong_seed);
        let setup = AkitaVerifierSetup {
            expanded: Arc::new(AkitaExpandedSetup::from_parts(setup_seed, wrong_matrix).unwrap()),
        };

        let mut bytes = Vec::new();
        setup.serialize_compressed(&mut bytes).unwrap();
        let err = AkitaVerifierSetup::<F>::deserialize_compressed(&bytes[..], &()).unwrap_err();

        assert!(err
            .to_string()
            .contains("setup shared_matrix does not match public matrix seed"));
    }

    #[test]
    fn trusted_cached_verifier_setup_decode_skips_matrix_rederivation() {
        let setup_seed = seed([7u8; 32]);
        let wrong_seed = [9u8; 32];
        let wrong_matrix = derive_public_matrix_flat::<F, D>(2, &wrong_seed);
        let setup = AkitaVerifierSetup {
            expanded: Arc::new(AkitaExpandedSetup::from_parts(setup_seed, wrong_matrix).unwrap()),
        };

        let mut bytes = Vec::new();
        setup.serialize_compressed(&mut bytes).unwrap();
        let decoded = AkitaVerifierSetup::<F>::deserialize_with_trusted_cached_matrix(
            &bytes[..],
            Compress::Yes,
            Validate::Yes,
        )
        .unwrap();

        assert_eq!(decoded, setup);
    }
}
