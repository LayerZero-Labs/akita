//! Shared setup data shapes for Akita prover and verifier APIs.

use crate::FlatMatrix;
use akita_field::FieldCore;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use std::io::{Read, Write};
use std::sync::Arc;

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaExpandedSetup<F: FieldCore> {
    /// Setup seed and runtime layout metadata.
    pub seed: AkitaSetupSeed,
    /// Shared 1D flat backing vector.
    pub shared_matrix: FlatMatrix<F>,
}

/// Verifier setup artifact derived from prover setup.
///
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaVerifierSetup<F: FieldCore> {
    /// Expanded matrix stage used for verification.
    pub expanded: Arc<AkitaExpandedSetup<F>>,
}

impl<F: FieldCore> AkitaVerifierSetup<F> {
    /// Build a verifier setup wrapping the given expanded setup with a
    /// fresh empty schedule cache.
    #[must_use]
    pub fn new(expanded: Arc<AkitaExpandedSetup<F>>) -> Self {
        Self { expanded }
    }
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
        let out = Self {
            seed: AkitaSetupSeed::deserialize_with_mode(&mut reader, compress, validate, &())?,
            shared_matrix: FlatMatrix::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
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
        let expanded = Arc::new(AkitaExpandedSetup::deserialize_with_mode(
            reader,
            compress,
            validate,
            &(),
        )?);
        Ok(Self::new(expanded))
    }
}
