//! Shared setup data shapes for Akita prover and verifier APIs.

use crate::{FlatMatrix, LevelParams, RingMatrixView, SetupArtifactDigests};
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

const SETUP_LAYOUT_TAG: [u8; 16] = *b"AKITA_SETUP_FLAT";

/// Config-derived setup matrix capacity.
///
/// `max_setup_len` is the physical number of ring elements generated at the
/// setup generation dimension. It is the maximum packed A/B/D role footprint
/// over the shapes the setup supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SetupMatrixEnvelope {
    /// Physical shared setup length at the generation ring dimension.
    pub max_setup_len: usize,
}

/// Base A/B/D setup role dimensions for one proof shape.
///
/// These dimensions intentionally exclude feature-gated ZK blinding tails.
/// They describe how the shared base setup matrix is viewed once the global
/// setup matrix is packed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SetupRoleDimensions {
    /// A-role row count.
    pub n_a: usize,
    /// B-role row count.
    pub n_b: usize,
    /// D-role row count.
    pub n_d: usize,
    /// A-role packed setup width.
    pub a_setup_width: usize,
    /// B-role packed setup width.
    pub b_setup_width: usize,
    /// D-role packed setup width.
    pub d_setup_width: usize,
}

impl SetupRoleDimensions {
    /// Dimensions implied by a concrete `LevelParams` value.
    ///
    /// This is the schedule-table/key shape and is useful for existing setup
    /// capacity scans. Batched runtime shapes should use
    /// [`Self::for_batched_shape`] so grouped B/D widths are computed from the
    /// actual claim/point incidence.
    #[inline]
    #[must_use]
    pub fn from_level_params(lp: &LevelParams) -> Self {
        Self {
            n_a: lp.a_key.row_len(),
            n_b: lp.b_key.row_len(),
            n_d: lp.d_key.row_len(),
            a_setup_width: lp.inner_width(),
            b_setup_width: lp.outer_width(),
            d_setup_width: lp.d_matrix_width(),
        }
    }

    /// Dimensions for the batched ring-switch shape used by verifier replay.
    ///
    /// # Errors
    ///
    /// Returns `InvalidSetup` if shape arithmetic overflows, if the runtime
    /// incidence is empty, or if the `LevelParams` key widths cannot cover the
    /// packed runtime widths.
    pub fn for_batched_shape(
        lp: &LevelParams,
        num_polys_per_point: &[usize],
        num_claims: usize,
    ) -> Result<Self, AkitaError> {
        if num_polys_per_point.is_empty() {
            return Err(AkitaError::InvalidSetup(
                "setup role dimensions require at least one point".to_string(),
            ));
        }
        if num_claims == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup role dimensions require at least one claim".to_string(),
            ));
        }
        let a_setup_width = lp.inner_width();
        let d_setup_width = lp
            .num_digits_open
            .checked_mul(lp.num_blocks)
            .and_then(|width| width.checked_mul(num_claims))
            .ok_or_else(|| AkitaError::InvalidSetup("D setup width overflow".to_string()))?;
        let max_point_poly_count = num_polys_per_point.iter().copied().max().unwrap_or(0);
        let b_setup_width = max_point_poly_count
            .checked_mul(lp.a_key.row_len())
            .and_then(|width| width.checked_mul(lp.num_digits_open))
            .and_then(|width| width.checked_mul(lp.num_blocks))
            .ok_or_else(|| AkitaError::InvalidSetup("B setup width overflow".to_string()))?;
        let out = Self {
            n_a: lp.a_key.row_len(),
            n_b: lp.b_key.row_len(),
            n_d: lp.d_key.row_len(),
            a_setup_width,
            b_setup_width,
            d_setup_width,
        };
        out.validate_key_widths(lp)?;
        Ok(out)
    }

    /// Validate that verifier-reachable key widths cover these setup views.
    ///
    /// # Errors
    ///
    /// Returns `InvalidSetup` if any key column count is too small.
    pub fn validate_key_widths(&self, lp: &LevelParams) -> Result<(), AkitaError> {
        if lp.a_key.col_len() < self.a_setup_width {
            return Err(AkitaError::InvalidSetup(
                "A-key column width is too small for setup role dimensions".to_string(),
            ));
        }
        if lp.b_key.col_len() < self.b_setup_width {
            return Err(AkitaError::InvalidSetup(
                "B-key column width is too small for setup role dimensions".to_string(),
            ));
        }
        if lp.d_key.col_len() < self.d_setup_width {
            return Err(AkitaError::InvalidSetup(
                "D-key column width is too small for setup role dimensions".to_string(),
            ));
        }
        Ok(())
    }

    /// A-role packed footprint.
    ///
    /// # Errors
    ///
    /// Returns `InvalidSetup` on arithmetic overflow.
    pub fn a_footprint(&self) -> Result<usize, AkitaError> {
        self.n_a
            .checked_mul(self.a_setup_width)
            .ok_or_else(|| AkitaError::InvalidSetup("A setup footprint overflow".to_string()))
    }

    /// B-role packed footprint.
    ///
    /// # Errors
    ///
    /// Returns `InvalidSetup` on arithmetic overflow.
    pub fn b_footprint(&self) -> Result<usize, AkitaError> {
        self.n_b
            .checked_mul(self.b_setup_width)
            .ok_or_else(|| AkitaError::InvalidSetup("B setup footprint overflow".to_string()))
    }

    /// D-role packed footprint.
    ///
    /// # Errors
    ///
    /// Returns `InvalidSetup` on arithmetic overflow.
    pub fn d_footprint(&self) -> Result<usize, AkitaError> {
        self.n_d
            .checked_mul(self.d_setup_width)
            .ok_or_else(|| AkitaError::InvalidSetup("D setup footprint overflow".to_string()))
    }

    /// Maximum packed base setup footprint across A/B/D.
    ///
    /// # Errors
    ///
    /// Returns `InvalidSetup` on arithmetic overflow.
    pub fn max_footprint(&self) -> Result<usize, AkitaError> {
        Ok(self
            .a_footprint()?
            .max(self.b_footprint()?)
            .max(self.d_footprint()?))
    }
}

impl SetupMatrixEnvelope {
    /// Build an envelope from a packed physical setup length.
    ///
    /// # Errors
    ///
    /// Returns `InvalidSetup` if the physical setup length is zero.
    pub fn from_max_setup_len(max_setup_len: usize) -> Result<Self, AkitaError> {
        if max_setup_len == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup envelope max_setup_len must be non-zero".to_string(),
            ));
        }
        Ok(Self { max_setup_len })
    }

    /// Build a packed envelope from base role dimensions.
    ///
    /// # Errors
    ///
    /// Returns `InvalidSetup` on arithmetic overflow.
    pub fn from_role_dimensions(dimensions: SetupRoleDimensions) -> Result<Self, AkitaError> {
        Self::from_max_setup_len(dimensions.max_footprint()?)
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

impl<F: FieldCore> AkitaExpandedSetup<F> {
    /// Borrow the packed A-role setup prefix at ring dimension `D`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidSetup` if the packed A footprint is not available at
    /// the requested ring dimension.
    #[inline]
    pub fn a_setup_view<const D: usize>(
        &self,
        dimensions: SetupRoleDimensions,
    ) -> Result<RingMatrixView<'_, F, D>, AkitaError> {
        self.shared_matrix
            .ring_view::<D>(dimensions.n_a, dimensions.a_setup_width)
    }

    /// Borrow the packed B-role setup prefix at ring dimension `D`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidSetup` if the packed B footprint is not available at
    /// the requested ring dimension.
    #[inline]
    pub fn b_setup_view<const D: usize>(
        &self,
        dimensions: SetupRoleDimensions,
    ) -> Result<RingMatrixView<'_, F, D>, AkitaError> {
        self.shared_matrix
            .ring_view::<D>(dimensions.n_b, dimensions.b_setup_width)
    }

    /// Borrow the packed D-role setup prefix at ring dimension `D`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidSetup` if the packed D footprint is not available at
    /// the requested ring dimension.
    #[inline]
    pub fn d_setup_view<const D: usize>(
        &self,
        dimensions: SetupRoleDimensions,
    ) -> Result<RingMatrixView<'_, F, D>, AkitaError> {
        self.shared_matrix
            .ring_view::<D>(dimensions.n_d, dimensions.d_setup_width)
    }
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
        let mut public_matrix_seed = [0u8; 32];
        reader.read_exact(&mut public_matrix_seed)?;
        let mut zk_blinding_seed = [0u8; 32];
        reader.read_exact(&mut zk_blinding_seed)?;
        let out = Self {
            max_num_vars,
            max_num_batched_polys,
            max_num_points,
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{AjtaiKeyParams, SisModulusFamily};
    use akita_algebra::CyclotomicRing;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::fields::Prime128Offset275;

    const D: usize = 32;
    type F = Prime128Offset275;

    fn stage1_config() -> SparseChallengeConfig {
        SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![-1, 1],
        }
    }

    fn sample_level() -> LevelParams {
        LevelParams::params_only(SisModulusFamily::Q32, D, 3, 3, 5, 7, stage1_config())
            .with_decomp(4, 3, 2, 6, 1, 0)
            .expect("sample level should be valid")
    }

    fn batched_sample_level() -> (LevelParams, SetupRoleDimensions) {
        let mut lp = sample_level();
        let t_cols_per_claim = lp.a_key.row_len() * lp.num_digits_open * lp.num_blocks;
        lp.b_key = AjtaiKeyParams::new_unchecked(
            SisModulusFamily::Q32,
            lp.b_key.row_len(),
            4 * t_cols_per_claim,
            0,
            D,
        );
        lp.d_key = AjtaiKeyParams::new_unchecked(
            SisModulusFamily::Q32,
            lp.d_key.row_len(),
            lp.num_digits_open * lp.num_blocks * 3,
            0,
            D,
        );
        let dims = SetupRoleDimensions::for_batched_shape(&lp, &[2, 1, 4], 3)
            .expect("shape should fit key widths");
        (lp, dims)
    }

    fn setup_from_dimensions(dimensions: SetupRoleDimensions) -> AkitaExpandedSetup<F> {
        let total = dimensions
            .max_footprint()
            .expect("sample dimensions fit usize");
        let rings = vec![CyclotomicRing::<F, D>::zero(); total];
        let shared_matrix = FlatMatrix::from_ring_slice(&rings);
        let seed = AkitaSetupSeed {
            max_num_vars: 8,
            max_num_batched_polys: 4,
            max_num_points: 3,
            max_setup_len: total,
            public_matrix_seed: [1u8; 32],
            zk_blinding_seed: [2u8; 32],
        };
        AkitaExpandedSetup::from_parts(seed, shared_matrix).expect("setup descriptor digest")
    }

    #[test]
    fn level_role_dimensions_match_key_widths() {
        let lp = sample_level();
        let dims = SetupRoleDimensions::from_level_params(&lp);

        assert_eq!(dims.n_a, 3);
        assert_eq!(dims.n_b, 5);
        assert_eq!(dims.n_d, 7);
        assert_eq!(dims.a_setup_width, lp.inner_width());
        assert_eq!(dims.b_setup_width, lp.outer_width());
        assert_eq!(dims.d_setup_width, lp.d_matrix_width());
    }

    #[test]
    fn batched_role_dimensions_use_grouped_b_width_and_claim_d_width() {
        let (lp, dims) = batched_sample_level();
        let t_cols_per_claim = lp.a_key.row_len() * lp.num_digits_open * lp.num_blocks;

        assert_eq!(dims.a_setup_width, lp.inner_width());
        assert_eq!(dims.b_setup_width, 4 * t_cols_per_claim);
        assert_eq!(dims.d_setup_width, lp.num_digits_open * lp.num_blocks * 3);
        assert_eq!(dims.a_footprint().unwrap(), dims.n_a * dims.a_setup_width);
        assert_eq!(dims.b_footprint().unwrap(), dims.n_b * dims.b_setup_width);
        assert_eq!(dims.d_footprint().unwrap(), dims.n_d * dims.d_setup_width);

        let envelope = SetupMatrixEnvelope::from_role_dimensions(dims).unwrap();
        assert_eq!(envelope.max_setup_len, dims.max_footprint().unwrap());
    }

    #[test]
    fn batched_role_dimensions_reject_key_width_mismatch() {
        let mut lp = sample_level();
        lp.b_key =
            AjtaiKeyParams::new_unchecked(SisModulusFamily::Q32, lp.b_key.row_len(), 1, 0, D);

        let err = SetupRoleDimensions::for_batched_shape(&lp, &[2, 1], 2)
            .expect_err("B key cannot cover grouped B setup width");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn expanded_setup_role_views_use_packed_widths() {
        let (_lp, dims) = batched_sample_level();
        let setup = setup_from_dimensions(dims);

        let a_view = setup.a_setup_view::<D>(dims).unwrap();
        let b_view = setup.b_setup_view::<D>(dims).unwrap();
        let d_view = setup.d_setup_view::<D>(dims).unwrap();

        assert_eq!(a_view.num_rows(), dims.n_a);
        assert_eq!(a_view.num_cols(), dims.a_setup_width);
        assert_eq!(b_view.num_rows(), dims.n_b);
        assert_eq!(b_view.num_cols(), dims.b_setup_width);
        assert_eq!(d_view.num_rows(), dims.n_d);
        assert_eq!(d_view.num_cols(), dims.d_setup_width);
    }

    #[test]
    fn expanded_setup_role_views_reject_insufficient_prefix() {
        let (_lp, dims) = batched_sample_level();
        let total = dims
            .b_footprint()
            .expect("sample B footprint fits usize")
            .checked_sub(1)
            .expect("sample B footprint is non-zero");
        let rings = vec![CyclotomicRing::<F, D>::zero(); total];
        let shared_matrix = FlatMatrix::from_ring_slice(&rings);
        let seed = AkitaSetupSeed {
            max_num_vars: 8,
            max_num_batched_polys: 4,
            max_num_points: 3,
            max_setup_len: total,
            public_matrix_seed: [1u8; 32],
            zk_blinding_seed: [2u8; 32],
        };
        let setup =
            AkitaExpandedSetup::from_parts(seed, shared_matrix).expect("setup descriptor digest");

        let err = setup
            .b_setup_view::<D>(dims)
            .expect_err("undersized B setup prefix must be rejected");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }
}
