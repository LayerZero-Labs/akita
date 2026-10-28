//! Public outgoing layout and direct relation assembly for a binary root.

use akita_algebra::ring::trinomial::module_map::pack_signed_components;
use akita_challenges::Challenges;
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::BINARY_SETUP_P;
use akita_params::{RelationPolynomial, RelationRowGeometry};
use jolt_field::Field;

use super::{
    binary_scalar_challenges, binary_scalar_geometry, BinaryEvaluationBounds,
    BinaryEvaluationSegments, BinaryQuotientBounds, BinarySourceDescriptor, BinaryTerminalClaim,
    BINARY_PARITY_CARRY_DIGITS, BINARY_SOURCE_Q, BINARY_VALUE_CARRY_DIGITS, CANONICAL_IMAGE_WINDOW,
};
use crate::native_root_relation::{
    binary_commitment_group, NativeRootRelation, NativeRootTerm, NativeRootWeights,
};
use crate::root_repacking::{RootCoefficientRepacking, RootRepackingSegment};

/// One public layout shared by A, parity, value, range and successor checks.
///
/// Its six segments are response, image, committed q quotient, U, K and K_val.
/// All later coefficient maps address this plan; enlarging its domain preserves
/// every live address and adds mandatory authenticated zero padding.
#[derive(Clone, Debug)]
pub struct BinaryRootWitnessPlan {
    descriptor: BinarySourceDescriptor,
    rank: usize,
    repacking: RootCoefficientRepacking,
    quotient_bounds: BinaryQuotientBounds,
    evaluation_bounds: BinaryEvaluationBounds,
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
        let quotient_bounds = BinaryQuotientBounds::new(ring_rows, columns, BINARY_SETUP_P)?;
        let evaluation_bounds = BinaryEvaluationBounds::new(scalar_rows, columns, BINARY_SETUP_P)?;
        let ring = RelationRowGeometry::for_polynomial(
            RelationPolynomial::minus_trinomial(648)?,
            1,
            1024,
        )?;
        let scalar = binary_scalar_geometry()?;
        let segments = [
            (ring, checked::product([owners, ring_rows]), 4),
            (ring, checked::product([owners, columns, rank]), 8),
            (
                ring,
                checked::product([owners, rank]),
                quotient_bounds.digit_count(),
            ),
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
            repacking: RootCoefficientRepacking::new(segments, destination_ring_dimension)?,
            quotient_bounds,
            evaluation_bounds,
        })
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
    pub fn quotient_bounds(&self) -> BinaryQuotientBounds {
        self.quotient_bounds
    }
    pub fn evaluation_bounds(&self) -> BinaryEvaluationBounds {
        self.evaluation_bounds
    }

    pub fn with_domain_len(mut self, domain_len: usize) -> Result<Self, AkitaError> {
        self.repacking = self.repacking.with_domain_len(domain_len)?;
        Ok(self)
    }

    /// Append one ordinary source's complete commitment preimage before
    /// selecting the common outgoing coefficient domain.
    pub fn with_ordinary(
        mut self,
        profile: &akita_params::GroupCommitPhaseParams,
    ) -> Result<Self, AkitaError> {
        let ordinary =
            super::OrdinaryNativeRootPlan::new(*profile, self.repacking.segments().len())?;
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
        claim: &BinaryTerminalClaim,
        matrix: &[Vec<[i128; 648]>],
        challenges: &Challenges,
        alpha: E,
        row_batches: &[E],
        terminal_batches: [E; 2],
        padding_weights: &[E],
    ) -> Result<NativeRootWeights<E>, AkitaError> {
        let a = binary_direct_a_weights(
            self,
            matrix,
            challenges,
            alpha,
            row_batches,
            padding_weights,
        )?;
        let terminal = claim.relation_weights(
            &self.descriptor,
            &self.repacking,
            BinaryEvaluationSegments {
                response: 0,
                partials: 3,
                parity_carry: 4,
                value_carry: 5,
            },
            challenges,
            alpha,
            terminal_batches,
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

/// Evaluate the binary A term directly from the typed q24 setup matrix.
///
/// The future typed adjoint replaces this function. It compiles only the
/// source-independent identity `rem_Psi(A*z-sum c*T)-q*KA=0`, never a clear
/// residual or carry claim. The field E must extend the admitted proof field.
/// Setup uniformity belongs to the typed setup producer; this boundary checks
/// its dimensions and centered representatives before compiling public weights.
pub fn binary_direct_a_weights<E: Field>(
    plan: &BinaryRootWitnessPlan,
    matrix: &[Vec<[i128; 648]>],
    challenges: &Challenges,
    alpha: E,
    row_batches: &[E],
    padding_weights: &[E],
) -> Result<NativeRootWeights<E>, AkitaError> {
    let descriptor = plan.descriptor();
    let columns = descriptor.index_map().columns();
    let rows = checked::product([4, columns])
        .and_then(|divisor| checked::exact_div(descriptor.index_map().padded_carriers(), divisor))
        .ok_or(AkitaError::InvalidProof)?;
    if matrix.len() != plan.rank
        || matrix.iter().any(|row| row.len() != rows)
        || matrix.iter().flatten().flatten().any(|&coefficient| {
            coefficient < i128::from(CANONICAL_IMAGE_WINDOW.0)
                || coefficient > i128::from(CANONICAL_IMAGE_WINDOW.1)
        })
    {
        return Err(AkitaError::InvalidSetup(
            "binary A matrix differs from its declared geometry or centered range".into(),
        ));
    }
    let scalar = binary_scalar_challenges(descriptor, challenges)?;
    let embedded = scalar
        .into_iter()
        .map(|scalar| {
            Ok(pack_signed_components(&[scalar, [0; 162], [0; 162], [0; 162]])?.map(E::from_i128))
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let matrix = matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|coefficients| coefficients.map(E::from_i128))
                .collect()
        })
        .collect::<Vec<Vec<_>>>();
    let row_count = checked::product([descriptor.polynomials().len(), plan.rank])
        .ok_or(AkitaError::InvalidProof)?;
    let mut residuals = Vec::new();
    residuals
        .try_reserve_exact(row_count)
        .map_err(|_| AkitaError::InvalidProof)?;
    residuals.resize(row_count, [E::zero(); 648]);
    let mut group =
        binary_commitment_group(descriptor.clone(), &matrix, &embedded, 0, 1, &residuals)?;
    for (element, row) in group.rows.iter_mut().enumerate() {
        let mut multiplier = vec![E::zero(); 648];
        *multiplier.first_mut().ok_or(AkitaError::InvalidProof)? =
            -E::from_u64(u64::from(BINARY_SOURCE_Q));
        row.terms.push(NativeRootTerm {
            segment: 2,
            element,
            multiplier,
        });
    }
    NativeRootRelation::new(plan.repacking.clone(), vec![group])?.ring_switch(
        alpha,
        row_batches,
        padding_weights,
    )
}

#[cfg(test)]
#[path = "root_plan_tests.rs"]
mod tests;
