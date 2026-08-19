//! Checked subring coefficient packing geometry.

use akita_challenges::SparseChallengeConfig;
use akita_field::{AkitaError, FieldCore};

#[cfg(test)]
use akita_field::{ExtField, FromPrimitiveInt};

use crate::{basis_weights, basis_weights_prefix, BasisMode};

mod fold;

#[cfg(any(test, feature = "test-support"))]
pub use fold::coefficient_packing_partials;
pub use fold::{
    coefficient_packing_scalar_opening, fold_coefficient_packing_partials,
    CoefficientPackingFoldProduct,
};

#[cfg(test)]
use fold::{
    coefficient_packing_map, embed_subring_challenge_in_a_ring,
    multiply_a_ring_by_subring_challenge,
};

/// Canonical geometry for one subring coefficient packing opening.
///
/// The A ring dimension satisfies `d_a = k * h * s`. The subring challenge
/// embeds through `Y -> X^(k * h)`, while one partial opening contains `k * s`
/// base field coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubringCoefficientPackingGeometry {
    extension_degree: usize,
    a_ring_dimension: usize,
    challenge_subring_dimension: usize,
    packing_factor: usize,
    subring_embedding_stride: usize,
    partial_base_field_width: usize,
    fold_challenge_config: SparseChallengeConfig,
}

impl SubringCoefficientPackingGeometry {
    /// Construct checked production subring packing geometry.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] unless all dimensions are powers of
    /// two, `k * s` divides `d_a`, and `s` has a production sparse challenge
    /// family that passes the existing entropy audit.
    pub fn try_new(
        extension_degree: usize,
        a_ring_dimension: usize,
        challenge_subring_dimension: usize,
    ) -> Result<Self, AkitaError> {
        if !extension_degree.is_power_of_two()
            || !a_ring_dimension.is_power_of_two()
            || !challenge_subring_dimension.is_power_of_two()
        {
            return Err(AkitaError::InvalidSetup(
                "subring packing dimensions must be nonzero powers of two".into(),
            ));
        }

        let fold_challenge_config = SparseChallengeConfig::production_for_ring_dim(
            challenge_subring_dimension,
        )
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "challenge subring dimension {challenge_subring_dimension} has no production challenge family"
            ))
        })?;
        fold_challenge_config
            .validate_for_ring_dim(challenge_subring_dimension)
            .map_err(|reason| {
                AkitaError::InvalidSetup(format!(
                    "subring challenge family fails its entropy audit: {reason}"
                ))
            })?;

        let partial_base_field_width = extension_degree
            .checked_mul(challenge_subring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("partial opening width overflow".into()))?;
        if !a_ring_dimension.is_multiple_of(partial_base_field_width) {
            return Err(AkitaError::InvalidSetup(format!(
                "partial opening width {partial_base_field_width} does not divide A ring dimension {a_ring_dimension}"
            )));
        }
        let packing_factor = a_ring_dimension / partial_base_field_width;
        let subring_embedding_stride = extension_degree
            .checked_mul(packing_factor)
            .ok_or_else(|| AkitaError::InvalidSetup("subring embedding stride overflow".into()))?;

        Ok(Self {
            extension_degree,
            a_ring_dimension,
            challenge_subring_dimension,
            packing_factor,
            subring_embedding_stride,
            partial_base_field_width,
            fold_challenge_config,
        })
    }

    /// Extension degree `k = [E:K]`.
    #[must_use]
    pub const fn extension_degree(self) -> usize {
        self.extension_degree
    }

    /// A ring dimension `d_a`.
    #[must_use]
    pub const fn a_ring_dimension(self) -> usize {
        self.a_ring_dimension
    }

    /// Challenge subring dimension `s`.
    #[must_use]
    pub const fn challenge_subring_dimension(self) -> usize {
        self.challenge_subring_dimension
    }

    /// Packing factor `h = d_a / (k * s)`.
    #[must_use]
    pub const fn packing_factor(self) -> usize {
        self.packing_factor
    }

    /// Exponent stride `k * h` in the embedding `Y -> X^(k * h)`.
    #[must_use]
    pub const fn subring_embedding_stride(self) -> usize {
        self.subring_embedding_stride
    }

    /// Base field coordinate width `k * s` of one partial opening.
    #[must_use]
    pub const fn partial_base_field_width(self) -> usize {
        self.partial_base_field_width
    }

    /// Production sparse challenge family fixed by `s`.
    #[must_use]
    pub const fn fold_challenge_config(self) -> SparseChallengeConfig {
        self.fold_challenge_config
    }

    /// Flatten `(low_coefficient_index, subring_coefficient_index)` into one A
    /// ring coefficient index.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when either coordinate is outside
    /// the checked geometry.
    pub fn a_ring_coefficient_index(
        self,
        low_coefficient_index: usize,
        subring_coefficient_index: usize,
    ) -> Result<usize, AkitaError> {
        if low_coefficient_index >= self.subring_embedding_stride
            || subring_coefficient_index >= self.challenge_subring_dimension
        {
            return Err(AkitaError::InvalidSetup(
                "subring coefficient coordinates are outside the A ring".into(),
            ));
        }
        subring_coefficient_index
            .checked_mul(self.subring_embedding_stride)
            .and_then(|offset| offset.checked_add(low_coefficient_index))
            .filter(|&index| index < self.a_ring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("A ring coefficient index overflow".into()))
    }

    /// Split one A ring coefficient into
    /// `(low_coefficient_index, subring_coefficient_index)`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when `index` is outside the A ring.
    pub fn a_ring_coefficient_coordinates(
        self,
        index: usize,
    ) -> Result<(usize, usize), AkitaError> {
        if index >= self.a_ring_dimension {
            return Err(AkitaError::InvalidSetup(
                "A ring coefficient lies outside subring packing geometry".into(),
            ));
        }
        Ok((
            index % self.subring_embedding_stride,
            index / self.subring_embedding_stride,
        ))
    }

    /// Flatten `(extension_coordinate, subring_coefficient_index)` in the
    /// canonical partial opening layout.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when either coordinate is outside
    /// the checked geometry.
    pub fn partial_base_field_coordinate_index(
        self,
        extension_coordinate: usize,
        subring_coefficient_index: usize,
    ) -> Result<usize, AkitaError> {
        if extension_coordinate >= self.extension_degree
            || subring_coefficient_index >= self.challenge_subring_dimension
        {
            return Err(AkitaError::InvalidSetup(
                "partial opening coordinates are outside subring packing geometry".into(),
            ));
        }
        extension_coordinate
            .checked_mul(self.challenge_subring_dimension)
            .and_then(|offset| offset.checked_add(subring_coefficient_index))
            .filter(|&index| index < self.partial_base_field_width)
            .ok_or_else(|| AkitaError::InvalidSetup("partial opening index overflow".into()))
    }

    /// Split one partial opening index into
    /// `(extension_coordinate, subring_coefficient_index)`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when `index` is outside the partial
    /// opening.
    pub fn partial_base_field_coordinates(
        self,
        index: usize,
    ) -> Result<(usize, usize), AkitaError> {
        if index >= self.partial_base_field_width {
            return Err(AkitaError::InvalidSetup(
                "partial opening coefficient lies outside subring packing geometry".into(),
            ));
        }
        Ok((
            index / self.challenge_subring_dimension,
            index % self.challenge_subring_dimension,
        ))
    }
}

/// Canonical opening-point split for subring coefficient packing.
///
/// The source point order is `[r_pack | r_tail | r_M | r_B]`. Full domains
/// are retained for the first three axes; only the exact live block prefix is
/// retained from the padded block domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedSubringCoefficientPackingPoint<E: FieldCore> {
    geometry: SubringCoefficientPackingGeometry,
    basis: BasisMode,
    source_num_vars: usize,
    num_live_positions: usize,
    num_positions_per_block: usize,
    num_live_blocks: usize,
    packing_point: Vec<E>,
    tail_point: Vec<E>,
    position_point: Vec<E>,
    block_point: Vec<E>,
    packing_weights: Vec<E>,
    tail_weights: Vec<E>,
    position_weights: Vec<E>,
    live_block_weights: Vec<E>,
}

impl<E: FieldCore> PreparedSubringCoefficientPackingPoint<E> {
    /// Split one public opening point into the canonical packing axes.
    pub fn new(
        geometry: SubringCoefficientPackingGeometry,
        basis: BasisMode,
        num_live_positions: usize,
        num_positions_per_block: usize,
        source_num_vars: usize,
        point: &[E],
    ) -> Result<Self, AkitaError> {
        if num_live_positions == 0 || !num_positions_per_block.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "coefficient-packing opening requires live positions and a power-of-two position domain"
                    .into(),
            ));
        }
        let num_live_blocks = num_live_positions.div_ceil(num_positions_per_block);
        let block_domain = num_live_blocks.checked_next_power_of_two().ok_or_else(|| {
            AkitaError::InvalidSetup("coefficient-packing block domain overflow".into())
        })?;
        let axis_bits = [
            geometry.subring_embedding_stride().trailing_zeros() as usize,
            geometry.challenge_subring_dimension().trailing_zeros() as usize,
            num_positions_per_block.trailing_zeros() as usize,
            block_domain.trailing_zeros() as usize,
        ];
        let expected = axis_bits.iter().try_fold(0usize, |sum, &bits| {
            sum.checked_add(bits).ok_or_else(|| {
                AkitaError::InvalidSetup("coefficient-packing point length overflow".into())
            })
        })?;
        if point.len() != source_num_vars {
            return Err(AkitaError::InvalidPointDimension {
                expected: source_num_vars,
                actual: point.len(),
            });
        }
        if source_num_vars > expected {
            return Err(AkitaError::InvalidSetup(
                "coefficient-packing source exceeds prepared opening domain".into(),
            ));
        }
        let source_domain = 1usize.checked_shl(source_num_vars as u32).ok_or_else(|| {
            AkitaError::InvalidSetup("coefficient-packing source domain overflow".into())
        })?;
        let padded_ring_positions = source_domain.div_ceil(geometry.a_ring_dimension());
        if num_live_positions
            .checked_next_power_of_two()
            .filter(|&positions| positions == padded_ring_positions)
            .is_none()
        {
            return Err(AkitaError::InvalidSetup(
                "coefficient-packing live source prefix disagrees with source arity".into(),
            ));
        }
        let mut padded_point = point.to_vec();
        padded_point.resize(expected, E::zero());
        let mut offset = 0usize;
        let mut take_axis = |bits: usize| -> Result<&[E], AkitaError> {
            let end = offset.checked_add(bits).ok_or(AkitaError::InvalidProof)?;
            let axis = padded_point
                .get(offset..end)
                .ok_or(AkitaError::InvalidProof)?;
            offset = end;
            Ok(axis)
        };
        let packing_point = take_axis(axis_bits[0])?.to_vec();
        let tail_point = take_axis(axis_bits[1])?.to_vec();
        let position_point = take_axis(axis_bits[2])?.to_vec();
        let block_point = take_axis(axis_bits[3])?.to_vec();
        let packing_weights = basis_weights(&packing_point, basis)?;
        let tail_weights = basis_weights(&tail_point, basis)?;
        let position_weights = basis_weights(&position_point, basis)?;
        let live_block_weights = basis_weights_prefix(&block_point, basis, num_live_blocks)?;
        Ok(Self {
            geometry,
            basis,
            source_num_vars,
            num_live_positions,
            num_positions_per_block,
            num_live_blocks,
            packing_point,
            tail_point,
            position_point,
            block_point,
            packing_weights,
            tail_weights,
            position_weights,
            live_block_weights,
        })
    }

    /// Checked coefficient-packing geometry.
    pub const fn geometry(&self) -> SubringCoefficientPackingGeometry {
        self.geometry
    }

    /// Polynomial basis used for every tensor-product opening axis.
    pub const fn basis(&self) -> BasisMode {
        self.basis
    }

    /// Authenticated public-point arity before preparation-only padding.
    pub const fn source_num_vars(&self) -> usize {
        self.source_num_vars
    }

    /// Number of live A-ring positions in the source.
    pub const fn num_live_positions(&self) -> usize {
        self.num_live_positions
    }

    /// Fixed position domain within each block.
    pub const fn num_positions_per_block(&self) -> usize {
        self.num_positions_per_block
    }

    /// Number of live partial-opening blocks.
    pub const fn num_live_blocks(&self) -> usize {
        self.num_live_blocks
    }

    /// Public point coordinates for the low A-ring coefficient axis.
    pub fn packing_point(&self) -> &[E] {
        &self.packing_point
    }

    /// Public point coordinates for the challenge-subring coefficient axis.
    pub fn tail_point(&self) -> &[E] {
        &self.tail_point
    }

    /// Public point coordinates for positions within one block.
    pub fn position_point(&self) -> &[E] {
        &self.position_point
    }

    /// Public point coordinates for the padded block domain.
    pub fn block_point(&self) -> &[E] {
        &self.block_point
    }

    /// Weights for the low A-ring coefficient axis.
    pub fn packing_weights(&self) -> &[E] {
        &self.packing_weights
    }

    /// Weights for the challenge-subring coefficient axis.
    pub fn tail_weights(&self) -> &[E] {
        &self.tail_weights
    }

    /// Weights for positions within one block.
    pub fn position_weights(&self) -> &[E] {
        &self.position_weights
    }

    /// Exact live prefix of the padded block-domain weights.
    pub fn live_block_weights(&self) -> &[E] {
        &self.live_block_weights
    }
}

#[cfg(test)]
#[path = "subring_coefficient_packing_reference_tests.rs"]
mod reference_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::PRODUCTION_FOLD_CHALLENGE_RING_DIMS;
    use akita_field::Prime128OffsetA7F7;

    #[test]
    fn derives_every_production_subring_packing_geometry() {
        for &s in PRODUCTION_FOLD_CHALLENGE_RING_DIMS {
            for k in [1usize, 2, 4] {
                for h in [1usize, 2, 4] {
                    let d_a = k * h * s;
                    let geometry =
                        SubringCoefficientPackingGeometry::try_new(k, d_a, s).expect("geometry");
                    assert_eq!(geometry.extension_degree(), k);
                    assert_eq!(geometry.a_ring_dimension(), d_a);
                    assert_eq!(geometry.challenge_subring_dimension(), s);
                    assert_eq!(geometry.packing_factor(), h);
                    assert_eq!(geometry.subring_embedding_stride(), k * h);
                    assert_eq!(geometry.partial_base_field_width(), k * s);
                    assert!(geometry
                        .fold_challenge_config()
                        .matches_production_ladder(s));
                }
            }
        }
    }

    #[test]
    fn rejects_malformed_or_unregistered_geometry() {
        for (k, d_a, s) in [
            (0, 256, 64),
            (3, 384, 64),
            (2, 0, 64),
            (2, 192, 64),
            (2, 256, 0),
            (2, 256, 32),
            (4, 128, 64),
        ] {
            assert!(
                SubringCoefficientPackingGeometry::try_new(k, d_a, s).is_err(),
                "{k}/{d_a}/{s}"
            );
        }

        let high_bit = 1usize << (usize::BITS - 1);
        assert!(SubringCoefficientPackingGeometry::try_new(high_bit, high_bit, 64).is_err());
    }

    #[test]
    fn opening_point_uses_pack_tail_position_block_order() {
        type F = Prime128OffsetA7F7;
        let geometry = SubringCoefficientPackingGeometry::try_new(1, 128, 64).unwrap();
        // log(kh)=1, log(s)=6, log(M)=2, log(B-domain)=1.
        let point = (1..=10).map(F::from_u64).collect::<Vec<_>>();
        for basis in [BasisMode::Lagrange, BasisMode::Monomial] {
            let prepared =
                PreparedSubringCoefficientPackingPoint::new(geometry, basis, 6, 4, 10, &point)
                    .unwrap();
            assert_eq!(prepared.basis(), basis);
            assert_eq!(
                prepared.packing_weights(),
                basis_weights(&point[..1], basis).unwrap()
            );
            assert_eq!(
                prepared.tail_weights(),
                basis_weights(&point[1..7], basis).unwrap()
            );
            assert_eq!(
                prepared.position_weights(),
                basis_weights(&point[7..9], basis).unwrap()
            );
            assert_eq!(
                prepared.live_block_weights(),
                basis_weights_prefix(&point[9..], basis, 2).unwrap()
            );
            assert_eq!(prepared.num_live_blocks(), 2);
        }
        assert!(PreparedSubringCoefficientPackingPoint::new(
            geometry,
            BasisMode::Lagrange,
            6,
            4,
            10,
            &point[..9],
        )
        .is_err());
        assert!(PreparedSubringCoefficientPackingPoint::new(
            geometry,
            BasisMode::Lagrange,
            6,
            4,
            10,
            &[point.as_slice(), &[F::zero()]].concat(),
        )
        .is_err());

        let short_source_geometry = SubringCoefficientPackingGeometry::try_new(4, 256, 64).unwrap();
        let short_source_point = point[..9].to_vec();
        let padded = PreparedSubringCoefficientPackingPoint::new(
            short_source_geometry,
            BasisMode::Lagrange,
            2,
            4,
            9,
            &short_source_point,
        )
        .unwrap();
        assert_eq!(padded.num_live_blocks(), 1);
        assert!(PreparedSubringCoefficientPackingPoint::new(
            short_source_geometry,
            BasisMode::Lagrange,
            2,
            4,
            9,
            &short_source_point[..8],
        )
        .is_err());

        let low_arity_geometry = SubringCoefficientPackingGeometry::try_new(1, 128, 64).unwrap();
        let low_arity_point = point[..6].to_vec();
        assert!(PreparedSubringCoefficientPackingPoint::new(
            low_arity_geometry,
            BasisMode::Lagrange,
            1,
            1,
            6,
            &low_arity_point,
        )
        .is_ok());
    }

    #[test]
    fn a_ring_coefficient_indices_round_trip() {
        for &s in PRODUCTION_FOLD_CHALLENGE_RING_DIMS {
            for k in [1usize, 2, 4] {
                for h in [1usize, 2, 4] {
                    let geometry = SubringCoefficientPackingGeometry::try_new(k, k * h * s, s)
                        .expect("geometry");
                    for index in 0..geometry.a_ring_dimension() {
                        let (low_coefficient_index, subring_coefficient_index) = geometry
                            .a_ring_coefficient_coordinates(index)
                            .expect("coordinates");
                        assert_eq!(
                            geometry
                                .a_ring_coefficient_index(
                                    low_coefficient_index,
                                    subring_coefficient_index,
                                )
                                .expect("index"),
                            index
                        );
                        assert_eq!(
                            index,
                            low_coefficient_index
                                + geometry.subring_embedding_stride() * subring_coefficient_index
                        );
                    }
                    assert!(geometry
                        .a_ring_coefficient_index(geometry.subring_embedding_stride(), 0)
                        .is_err());
                    assert!(geometry
                        .a_ring_coefficient_index(0, geometry.challenge_subring_dimension())
                        .is_err());
                    assert!(geometry
                        .a_ring_coefficient_coordinates(geometry.a_ring_dimension())
                        .is_err());
                }
            }
        }
    }

    #[test]
    fn partial_base_field_coordinates_round_trip() {
        for &s in PRODUCTION_FOLD_CHALLENGE_RING_DIMS {
            for k in [1usize, 2, 4] {
                for h in [1usize, 2, 4] {
                    let geometry = SubringCoefficientPackingGeometry::try_new(k, k * h * s, s)
                        .expect("geometry");
                    for index in 0..geometry.partial_base_field_width() {
                        let (extension_coordinate, subring_coefficient_index) = geometry
                            .partial_base_field_coordinates(index)
                            .expect("coordinates");
                        assert_eq!(
                            geometry
                                .partial_base_field_coordinate_index(
                                    extension_coordinate,
                                    subring_coefficient_index,
                                )
                                .expect("index"),
                            index
                        );
                        assert_eq!(index, extension_coordinate * s + subring_coefficient_index);
                    }
                    assert!(geometry
                        .partial_base_field_coordinate_index(geometry.extension_degree(), 0)
                        .is_err());
                    assert!(geometry
                        .partial_base_field_coordinate_index(
                            0,
                            geometry.challenge_subring_dimension()
                        )
                        .is_err());
                    assert!(geometry
                        .partial_base_field_coordinates(geometry.partial_base_field_width())
                        .is_err());
                }
            }
        }
    }
}
