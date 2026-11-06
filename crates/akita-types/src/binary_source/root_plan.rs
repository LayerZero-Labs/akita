//! Public outgoing layout and direct relation assembly for a binary root.

use akita_challenges::Challenges;
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::BinaryFieldProfile;
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
pub use projection_weights::{binary_direct_a_weights, BinaryAOperand};
pub(crate) use projection_weights::{DirectAWeights, ImageWeights};

/// One public layout shared by A, parity, value, range and successor checks.
///
/// Its word segments are response, image, U, parity carry and value carry.
/// Selected prime and ordinary segments follow. P64 appends the coefficient
/// quotient segment last, with one degree-648 element per owner and A row.
/// All later coefficient maps address this plan; enlarging its domain preserves
/// every live address and adds padding cells that no root row reads. Honest
/// construction writes zero there; the range check, the successor's
/// authentication and its norm checks bound whatever the prover commits.
#[derive(Clone, Debug)]
pub struct BinaryRootWitnessPlan {
    descriptor: BinarySourceDescriptor,
    rank: usize,
    pub(super) repacking: RootCoefficientRepacking,
    projection_bounds: BinaryProjectionBounds,
    field_profile: BinaryFieldProfile,
    pub(super) word: bool,
    pub(super) prime_planes: usize,
}

impl BinaryRootWitnessPlan {
    pub fn new(
        descriptor: &BinarySourceDescriptor,
        rank: usize,
        destination_ring_dimension: usize,
    ) -> Result<Self, AkitaError> {
        Self::for_profile(
            descriptor,
            rank,
            destination_ring_dimension,
            BinaryFieldProfile::P128,
            true,
            0,
        )
    }

    /// Compile a selected field and view layout, without admitting a runtime root.
    /// Quotients follow every selected source segment; they are never part of D.
    pub fn for_profile(
        descriptor: &BinarySourceDescriptor,
        rank: usize,
        destination_ring_dimension: usize,
        field_profile: BinaryFieldProfile,
        word: bool,
        prime_planes: usize,
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
            BinaryProjectionBounds::for_profile(ring_rows, columns, rank, owners, field_profile)?;
        BinaryEvaluationBounds::new(scalar_rows, columns, field_profile.modulus())?;
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
        let mut plan = Self {
            descriptor: descriptor.clone(),
            rank,
            projection_bounds,
            field_profile,
            word: true,
            prime_planes: 0,
            repacking: RootCoefficientRepacking::new(segments, destination_ring_dimension)?,
        }
        .with_prime_planes(word, prime_planes)?;
        if let Some(bounds) = plan.projection_bounds.quotient() {
            let mut segments = plan.repacking.segments().to_vec();
            segments.push(RootRepackingSegment {
                geometry: ring,
                elements: checked::product([owners, rank]).ok_or(AkitaError::InvalidProof)?,
                digits: bounds.digits,
            });
            plan.repacking = RootCoefficientRepacking::new(segments, destination_ring_dimension)?;
        }
        Ok(plan)
    }

    pub fn field_profile(&self) -> BinaryFieldProfile {
        self.field_profile
    }

    pub fn quotient_segment(&self) -> Option<usize> {
        self.projection_bounds
            .quotient()
            .map(|_| self.repacking.segments().len() - 1)
    }

    /// Independent coefficient coordinates, or the frozen seven P128 projections.
    pub fn coefficient_batch_count(&self) -> Result<usize, AkitaError> {
        if self.quotient_segment().is_some() {
            checked::product([self.projection_bounds.residual_rows(), 648])
                .ok_or(AkitaError::InvalidProof)
        } else {
            Ok(super::BINARY_PROJECTION_COUNT)
        }
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
        let first = self
            .quotient_segment()
            .unwrap_or(self.repacking.segments().len());
        if ordinary
            .profile()
            .inner
            .matrix
            .sis_modulus_profile()
            .modulus()
            != self.field_profile.modulus()
        {
            return Err(AkitaError::InvalidSetup(
                "mixed root field profile mismatch".into(),
            ));
        }
        if ordinary.first_segment != first {
            return Err(AkitaError::InvalidProof);
        }
        let mut segments = self.repacking.segments().to_vec();
        segments
            .try_reserve_exact(ordinary.segments().len())
            .map_err(|_| AkitaError::InvalidProof)?;
        let quotient = self.quotient_segment().and_then(|_| segments.pop());
        segments.extend_from_slice(ordinary.segments());
        segments.extend(quotient);
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
        matrix: Option<BinaryAOperand<'_, E>>,
        challenges: &Challenges,
        alpha: E,
        rho: &[[E; 648]],
        terminal_batches: impl AsRef<[E]>,
    ) -> Result<NativeRootWeights<E>, AkitaError> {
        let a = binary_direct_a_weights(self, matrix, challenges, rho)?;
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
            self.field_profile.modulus(),
        )?;
        NativeRootWeights::merge(self.repacking.domain_len(), [a, terminal], E::zero())
    }
}

#[cfg(test)]
#[path = "root_plan_tests.rs"]
mod tests;
