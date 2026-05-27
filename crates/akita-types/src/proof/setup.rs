//! Shared setup data shapes for Akita prover and verifier APIs.

use crate::{FlatMatrix, SetupArtifactDigests};
use akita_field::{AkitaError, FieldCore};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use std::io::{Read, Write};
use std::sync::Arc;

/// Public seed used to derive commitment matrices.
pub type PublicMatrixSeed = [u8; 32];

/// Public seed used to derive feature-gated ZK blinding setup terms.
pub type ZkBlindingSeed = [u8; 32];

const SETUP_LAYOUT_TAG: [u8; 16] = *b"AKITA_SETUP_V002";

/// Config-derived setup matrix capacity.
///
/// `max_setup_len` is the physical number of ring elements generated at the
/// setup generation dimension. `max_rows` and `max_stride` are retained while
/// the current implementation still has stride-based role views; later packed
/// role-view slices remove the stride dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SetupMatrixEnvelope {
    /// Physical shared setup length at the generation ring dimension.
    pub max_setup_len: usize,
    /// Diagnostic/current-layout maximum row count.
    pub max_rows: usize,
    /// Diagnostic/current-layout maximum row stride.
    pub max_stride: usize,
}

impl SetupMatrixEnvelope {
    /// Build an envelope from the current row/stride layout.
    ///
    /// # Errors
    ///
    /// Returns `InvalidData` if either dimension is zero or if the physical
    /// setup length overflows.
    pub fn from_rows_stride(max_rows: usize, max_stride: usize) -> Result<Self, AkitaError> {
        if max_rows == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup envelope max_rows must be non-zero".to_string(),
            ));
        }
        if max_stride == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup envelope max_stride must be non-zero".to_string(),
            ));
        }
        let max_setup_len = max_rows.checked_mul(max_stride).ok_or_else(|| {
            AkitaError::InvalidSetup("setup envelope length overflow".to_string())
        })?;
        Ok(Self {
            max_setup_len,
            max_rows,
            max_stride,
        })
    }
}

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
    /// Physical shared setup length at the generation ring dimension.
    pub max_setup_len: usize,
    /// Public seed used to derive commitment matrices.
    pub public_matrix_seed: PublicMatrixSeed,
    /// Public seed/domain for ZK blinding setup terms.
    pub zk_blinding_seed: ZkBlindingSeed,
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
    /// Cached descriptor digests for the setup artifacts.
    pub descriptor_digests: SetupArtifactDigests,
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
    /// Returns a serialization error if the setup seed or shared matrix cannot
    /// be canonically serialized for descriptor hashing.
    pub fn from_parts(
        seed: AkitaSetupSeed,
        shared_matrix: FlatMatrix<F>,
    ) -> Result<Self, SerializationError> {
        let descriptor_digests = SetupArtifactDigests::from_parts(&seed, &shared_matrix)?;
        Ok(Self {
            seed,
            shared_matrix,
            descriptor_digests,
        })
    }
}

impl Valid for AkitaSetupSeed {
    fn check(&self) -> Result<(), SerializationError> {
        if self.max_setup_len == 0 {
            return Err(SerializationError::InvalidData(
                "setup seed max_setup_len must be non-zero".to_string(),
            ));
        }
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
        writer.write_all(&SETUP_LAYOUT_TAG)?;
        self.max_setup_len
            .serialize_with_mode(&mut writer, compress)?;
        self.max_stride.serialize_with_mode(&mut writer, compress)?;
        writer.write_all(&self.public_matrix_seed)?;
        writer.write_all(&self.zk_blinding_seed)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.max_num_vars.serialized_size(compress)
            + self.max_num_batched_polys.serialized_size(compress)
            + self.max_num_points.serialized_size(compress)
            + SETUP_LAYOUT_TAG.len()
            + self.max_setup_len.serialized_size(compress)
            + self.max_stride.serialized_size(compress)
            + 64
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
        let mut setup_layout_tag = [0u8; SETUP_LAYOUT_TAG.len()];
        reader.read_exact(&mut setup_layout_tag)?;
        if setup_layout_tag != SETUP_LAYOUT_TAG {
            return Err(SerializationError::InvalidData(
                "unsupported setup layout tag".to_string(),
            ));
        }
        let max_setup_len = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let max_stride = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let mut public_matrix_seed = [0u8; 32];
        reader.read_exact(&mut public_matrix_seed)?;
        let mut zk_blinding_seed = [0u8; 32];
        reader.read_exact(&mut zk_blinding_seed)?;
        let out = Self {
            max_num_vars,
            max_num_batched_polys,
            max_num_points,
            max_stride,
            max_setup_len,
            public_matrix_seed,
            zk_blinding_seed,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + Valid + AkitaSerialize> Valid for AkitaExpandedSetup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.seed.check()?;
        self.shared_matrix.check()?;
        if self.shared_matrix.total_ring_elements() != self.seed.max_setup_len {
            return Err(SerializationError::InvalidData(format!(
                "shared setup length {} does not match seed max_setup_len {}",
                self.shared_matrix.total_ring_elements(),
                self.seed.max_setup_len
            )));
        }
        self.descriptor_digests
            .check_parts(&self.seed, &self.shared_matrix)?;
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

impl<F: FieldCore + Valid + AkitaSerialize> Valid for AkitaVerifierSetup<F> {
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
