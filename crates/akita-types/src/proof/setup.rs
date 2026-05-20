//! Shared setup data shapes for Akita prover and verifier APIs.

use crate::FlatMatrix;
use akita_field::FieldCore;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use std::io::{Read, Write};
use std::sync::{Arc, OnceLock};

/// Public seed used to derive commitment matrices.
pub type PublicMatrixSeed = [u8; 32];

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
///
/// Tiered roots additionally need a per-setup `F` matrix
/// (`specs/tiered_commit.md` §3). It is deterministically derived from
/// `seed.public_matrix_seed` under the `b"tier1-f"` domain-separation
/// label and depends only on the schedule's outer SIS shape — so we
/// cache it lazily here behind an `Arc<OnceLock<_>>`. The first caller
/// at a given `num_rings` populates the cache; every subsequent
/// `verify` reuses it without re-running SHAKE. For legacy roots the
/// cache stays empty forever. Clones share the cache via the Arc so a
/// `verifier.clone()` keeps amortising across calls.
#[derive(Debug)]
pub struct AkitaExpandedSetup<F: FieldCore> {
    /// Setup seed and runtime layout metadata.
    pub seed: AkitaSetupSeed,
    /// Shared 1D flat backing vector.
    pub shared_matrix: FlatMatrix<F>,
    /// Lazily cached tiered `F` matrix.
    ///
    /// Holds `(num_rings, FlatMatrix)` so a debug-only mismatch check
    /// can detect setups being reused at unexpected tiered shapes.
    /// Skipped by `PartialEq`/`Eq` and by all serialisation paths.
    tier1_f_cache: Arc<OnceLock<(usize, FlatMatrix<F>)>>,
}

impl<F: FieldCore> AkitaExpandedSetup<F> {
    /// Build an expanded setup with an empty tier-1 F cache.
    pub fn new(seed: AkitaSetupSeed, shared_matrix: FlatMatrix<F>) -> Self {
        Self {
            seed,
            shared_matrix,
            tier1_f_cache: Arc::new(OnceLock::new()),
        }
    }

    /// Retrieve (lazily) the tiered `F` matrix of `num_rings` ring
    /// elements derived from the public matrix seed. The first call
    /// populates the cache; every subsequent call with the same
    /// `num_rings` returns the cached entry without rerunning SHAKE.
    ///
    /// `derive` is a single-arg closure taking `&PublicMatrixSeed`. We
    /// pass it in (rather than calling `derive_tier1_f_matrix_flat`
    /// directly) so this method can live in `akita-types` without
    /// taking a dependency on the verifier-side derivation crate.
    ///
    /// # Panics
    ///
    /// Panics in debug if a subsequent caller passes a different
    /// `num_rings` than the first one — the setup is being reused at
    /// an unexpected tiered SIS shape.
    pub fn tier1_f_matrix<DeriveFn>(
        &self,
        num_rings: usize,
        derive: DeriveFn,
    ) -> &FlatMatrix<F>
    where
        DeriveFn: FnOnce(&PublicMatrixSeed) -> FlatMatrix<F>,
    {
        let entry = self
            .tier1_f_cache
            .get_or_init(|| (num_rings, derive(&self.seed.public_matrix_seed)));
        debug_assert_eq!(
            entry.0, num_rings,
            "tier1 F cache shape mismatch: cached {} ring elements but caller requested {}",
            entry.0, num_rings,
        );
        &entry.1
    }
}

impl<F: FieldCore> Clone for AkitaExpandedSetup<F> {
    fn clone(&self) -> Self {
        Self {
            seed: self.seed.clone(),
            shared_matrix: self.shared_matrix.clone(),
            // Share the cache across clones — the F matrix is a pure
            // function of the seed, so any clone of this setup would
            // derive the same matrix.
            tier1_f_cache: Arc::clone(&self.tier1_f_cache),
        }
    }
}

impl<F: FieldCore> PartialEq for AkitaExpandedSetup<F> {
    fn eq(&self, other: &Self) -> bool {
        // Cache state is irrelevant to setup equality (it's a function
        // of `seed.public_matrix_seed` already covered below).
        self.seed == other.seed && self.shared_matrix == other.shared_matrix
    }
}

impl<F: FieldCore> Eq for AkitaExpandedSetup<F> {}

/// Verifier setup artifact derived from prover setup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaVerifierSetup<F: FieldCore> {
    /// Expanded matrix stage used for verification.
    pub expanded: Arc<AkitaExpandedSetup<F>>,
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

impl<F: FieldCore + Valid> Valid for AkitaExpandedSetup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.seed.check()?;
        self.shared_matrix.check()?;
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

impl<F: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
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
        let shared_matrix =
            FlatMatrix::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self::new(seed, shared_matrix);
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + Valid> Valid for AkitaVerifierSetup<F> {
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

impl<F: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
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
