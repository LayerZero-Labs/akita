//! Subring coefficient packing opening points and partial folds.
//!
//! The checked packing geometry lives in `layout/subring_packing_geometry.rs`.

use akita_error::{checked, AkitaError};
use jolt_field::Field;

#[cfg(test)]
use jolt_field::{Ring, Zero};

use crate::layout::subring_packing_geometry::SubringCoefficientPackingGeometry;
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

/// Canonical opening-point split for subring coefficient packing.
///
/// The source point order is `[r_pack | r_tail | r_M | r_B]`. Full domains
/// are retained for the first three axes; only the exact live block prefix is
/// retained from the padded block domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedSubringCoefficientPackingPoint<E: Field> {
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

impl<E: Field> PreparedSubringCoefficientPackingPoint<E> {
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
        let source_domain = checked::pow2(source_num_vars).ok_or_else(|| {
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
    use jolt_field::Prime128OffsetA7F7;

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
}
