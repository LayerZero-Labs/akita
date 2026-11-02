//! Public outgoing layout and direct relation assembly for a binary root.

use akita_challenges::Challenges;
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::BINARY_SETUP_P;
use akita_params::{RelationPolynomial, RelationRowGeometry};
use jolt_field::Field;

use super::{
    binary_scalar_geometry, BinaryEvaluationBounds, BinaryEvaluationSegments,
    BinaryProjectionBounds, BinarySourceDescriptor, BinaryTerminalClaim,
    BINARY_PARITY_CARRY_DIGITS, BINARY_VALUE_CARRY_DIGITS,
};
use crate::native_root_relation::NativeRootWeights;

#[path = "root_projection_weights.rs"]
mod projection_weights;
use crate::root_repacking::{RootCoefficientRepacking, RootRepackingSegment};
pub use projection_weights::binary_direct_a_weights;

/// One public layout shared by A, parity, value, range and successor checks.
///
/// Its five segments are response, image, U, K and K_val.
/// All later coefficient maps address this plan; enlarging its domain preserves
/// every live address and adds mandatory authenticated zero padding.
#[derive(Clone, Debug)]
pub struct BinaryRootWitnessPlan {
    descriptor: BinarySourceDescriptor,
    rank: usize,
    pub(super) repacking: RootCoefficientRepacking,
    projection_bounds: BinaryProjectionBounds,
    pub(super) word: bool,
    pub(super) prime_planes: usize,
}

impl BinaryRootWitnessPlan {
    pub fn new(
        descriptor: &BinarySourceDescriptor,
        rank: usize,
        destination_ring_dimension: usize,
    ) -> Result<Self, AkitaError> {
        if rank == 0 {
            return Err(AkitaError::InvalidSetup(
                "binary root matrix has zero rank".into(),
            ));
        }
        let columns = descriptor.index_map().columns();
        let scalar_rows = checked::exact_div(descriptor.index_map().padded_carriers(), columns)
            .ok_or(AkitaError::InvalidProof)?;
        let ring_rows = checked::exact_div(scalar_rows, 4).ok_or(AkitaError::InvalidProof)?;
        let owners = descriptor.polynomials().len();
        let projection_bounds =
            BinaryProjectionBounds::new(ring_rows, columns, rank, owners, BINARY_SETUP_P)?;
        BinaryEvaluationBounds::new(scalar_rows, columns, BINARY_SETUP_P)?;
        let ring = RelationRowGeometry::for_polynomial(
            RelationPolynomial::minus_trinomial(648)?,
            1,
            1024,
        )?;
        let scalar = binary_scalar_geometry()?;
        let segments = [
            (ring, checked::product([owners, ring_rows]), 4),
            (ring, checked::product([owners, columns, rank]), 8),
            (scalar, checked::product([owners, columns]), 1),
            (scalar, Some(owners), BINARY_PARITY_CARRY_DIGITS),
            (scalar, Some(owners), BINARY_VALUE_CARRY_DIGITS),
        ]
        .into_iter()
        .map(|(geometry, elements, digits)| {
            Ok(RootRepackingSegment {
                geometry,
                elements: elements.ok_or(AkitaError::InvalidProof)?,
                digits,
            })
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
        Ok(Self {
            descriptor: descriptor.clone(),
            rank,
            projection_bounds,
            word: true,
            prime_planes: 0,
            repacking: RootCoefficientRepacking::new(segments, destination_ring_dimension)?,
        })
    }

    pub fn projection_bounds(&self) -> &BinaryProjectionBounds {
        &self.projection_bounds
    }

    pub fn descriptor(&self) -> &BinarySourceDescriptor {
        &self.descriptor
    }
    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn repacking(&self) -> &RootCoefficientRepacking {
        &self.repacking
    }

    pub fn with_domain_len(mut self, domain_len: usize) -> Result<Self, AkitaError> {
        self.repacking = self.repacking.with_domain_len(domain_len)?;
        Ok(self)
    }

    /// Append one admitted ordinary source's complete commitment preimage
    /// before selecting the common outgoing coefficient domain.
    pub fn with_ordinary(
        mut self,
        ordinary: &super::OrdinaryNativeRootPlan,
    ) -> Result<Self, AkitaError> {
        if ordinary.first_segment != self.repacking.segments().len() {
            return Err(AkitaError::InvalidProof);
        }
        let mut segments = self.repacking.segments().to_vec();
        segments
            .try_reserve_exact(ordinary.segments().len())
            .map_err(|_| AkitaError::InvalidProof)?;
        segments.extend_from_slice(ordinary.segments());
        self.repacking =
            RootCoefficientRepacking::new(segments, self.repacking.destination_ring_dimension())?;
        Ok(self)
    }

    /// Assemble all binary identities after outgoing commitment and batching.
    /// B/D and any ordinary-group identities join these same linear weights.
    #[allow(clippy::too_many_arguments)]
    pub fn relation_weights<E: Field>(
        &self,
        claim: Option<&BinaryTerminalClaim>,
        matrix: Option<&[Vec<[i128; 648]>]>,
        challenges: &Challenges,
        alpha: E,
        rho: &[[E; 648]],
        terminal_batches: impl AsRef<[E]>,
        padding_weights: &[E],
    ) -> Result<NativeRootWeights<E>, AkitaError> {
        let a = binary_direct_a_weights(self, matrix, challenges, rho, padding_weights)?;
        let Some(claim) = claim else {
            if !terminal_batches.as_ref().is_empty() {
                return Err(AkitaError::InvalidProof);
            }
            return Ok(a);
        };
        let terminal = claim.relation_weights(
            &self.descriptor,
            &self.repacking,
            BinaryEvaluationSegments {
                response: 0,
                partials: 2,
                parity_carry: 3,
                value_carry: 4,
            },
            challenges,
            alpha,
            terminal_batches
                .as_ref()
                .try_into()
                .map_err(|_| AkitaError::InvalidProof)?,
            BINARY_SETUP_P,
        )?;
        let mut terms = a.terms().to_vec();
        terms
            .try_reserve_exact(terminal.terms().len())
            .map_err(|_| AkitaError::InvalidProof)?;
        terms.extend_from_slice(terminal.terms());
        NativeRootWeights::new(
            self.repacking.domain_len(),
            terms,
            a.claim() + terminal.claim(),
        )
    }
}

#[cfg(test)]
#[path = "root_plan_tests.rs"]
mod tests;
