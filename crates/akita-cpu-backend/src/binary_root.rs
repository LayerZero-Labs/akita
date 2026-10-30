//! Bounded q24 root witnesses using the canonical trinomial relation kernel.
//!
//! The proof field is `2^128-275`, which has no degree-648 smooth transform.
//! Convolution uses the existing kernel over an auxiliary smooth field, with
//! a checked integer no-wrap bound, then embeds the recovered integers into
//! the proof field. A q-divisible residual is retained explicitly: it is never
//! asserted to vanish over the proof field.

use akita_algebra::ring::trinomial::{
    image::{Q24Image, IMAGE_MIN, Q24, Q24_CENTERED_BOUND},
    module_map::pack_signed_components,
    signed::signed_trinomial_product,
};
use akita_algebra::{MinusTrinomial, TrinomialRing};
use akita_challenges::{Challenges, FoldChallengeFamily};
use akita_error::{checked, AkitaError};
use akita_params::{RelationPolynomial, RelationRowGeometry};
use akita_types::binary_source::{BinarySourceDescriptor, BINARY_RESPONSE_INTERVAL};
use akita_types::native_root_relation::{binary_commitment_group, NativeRootGroup};
use akita_types::root_repacking::{RootCoefficientRepacking, RootRepackingSegment};
use jolt_field::{CanonicalEncoding, Field, One, Prime128Offset275, Prime128OffsetA7F7, Ring};

use crate::kernels::trinomial_relation::TrinomialRelationQuotientBuilder;

const D: usize = 648;
type AuxiliaryRing = TrinomialRing<Prime128OffsetA7F7, D, MinusTrinomial>;

mod evaluation;
mod quotient;
pub use evaluation::{binary_row_partials, BinaryEvaluationWitness};
pub use quotient::BinaryQuotientWitness;

/// An integer identity `sum A*z - sum iota(c)*T = remainder + Psi*quotient`.
/// These values are construction witnesses, not unauthenticated proof messages.
#[derive(Clone, Debug)]
pub struct BinaryRootResidualRow {
    pub remainder: [i128; D],
    pub quotient: Vec<i128>,
}

/// Fixed root witness at the checkpoint immediately preceding projection J.
///
/// Source and image elements use polynomial/column/row order. Response elements
/// use polynomial/row order. Residual rows use polynomial/matrix-row order.
/// No original source coefficients enter the ordinary successor witness.
#[derive(Clone, Debug)]
pub struct BinaryRootRelation {
    pub response: Vec<[i128; D]>,
    pub images: Vec<[Q24Image; D]>,
    pub rows: Vec<BinaryRootResidualRow>,
    pub embedded_challenges: Vec<[i128; D]>,
}

impl BinaryRootRelation {
    /// Fold column-contiguous sources, then construct the fixed root relation.
    /// The descriptor's agreement contract remains the caller's obligation;
    /// this routine does not test binary or prime source agreement.
    pub fn build(
        descriptor: &BinarySourceDescriptor,
        matrix: &[Vec<[i128; D]>],
        sources: &[[i128; D]],
        images: &[[Q24Image; D]],
        challenges: &Challenges,
        family: FoldChallengeFamily,
    ) -> Result<Self, AkitaError> {
        Self::probe(descriptor, matrix, sources, images, challenges, family)?
            .ok_or_else(|| invalid("binary response exceeds its four-digit interval"))
    }

    /// Probe one honest fold; a clipped bounded response rejects only this nonce.
    /// All source/challenge geometry and arithmetic failures remain errors.
    pub fn probe(
        descriptor: &BinarySourceDescriptor,
        matrix: &[Vec<[i128; D]>],
        sources: &[[i128; D]],
        images: &[[Q24Image; D]],
        challenges: &Challenges,
        family: FoldChallengeFamily,
    ) -> Result<Option<Self>, AkitaError> {
        let (polynomials, columns, rows) = dimensions(descriptor, matrix)?;
        let expected = checked::product([polynomials, columns, rows])
            .ok_or_else(|| invalid("binary source count overflow"))?;
        if sources.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: sources.len(),
            });
        }
        let embedded = embed_challenges(descriptor, challenges, family)?;
        let count = checked::product([polynomials, rows])
            .ok_or_else(|| invalid("binary response count overflow"))?;
        let mut response = Vec::new();
        response
            .try_reserve_exact(count)
            .map_err(|_| invalid("binary response allocation failed"))?;
        response.resize(count, [0i128; D]);
        for (polynomial_sources, (polynomial_challenges, output)) in
            sources.chunks_exact(columns * rows).zip(
                embedded
                    .chunks_exact(columns)
                    .zip(response.chunks_exact_mut(rows)),
            )
        {
            for (column_sources, challenge) in polynomial_sources
                .chunks_exact(rows)
                .zip(polynomial_challenges)
            {
                for (source, destination) in column_sources.iter().zip(output.iter_mut()) {
                    let product = signed_trinomial_product::<D, MinusTrinomial>(source, challenge)?;
                    for (coefficient, value) in destination.iter_mut().zip(product) {
                        *coefficient = coefficient
                            .checked_add(value)
                            .ok_or_else(|| invalid("binary response integer overflow"))?;
                    }
                }
            }
        }
        let (low, high) = BINARY_RESPONSE_INTERVAL;
        if response
            .iter()
            .flatten()
            .any(|&coefficient| coefficient < i128::from(low) || coefficient > i128::from(high))
        {
            return Ok(None);
        }
        Self::from_response(descriptor, matrix, response, images, challenges, family).map(Some)
    }

    /// Validate a supplied bounded response and compute source-independent rows.
    ///
    /// q-divisibility here is an honest-prover construction check. Protocol
    /// verification must authenticate the residual through the shared Stage 2
    /// and a committed q quotient or fresh exact-lift projections.
    /// `challenges` must come from the caller's replay of the binary-family
    /// fold draw. This arithmetic boundary checks geometry and coefficient
    /// bounds; it neither samples challenges nor replays transcript state.
    pub fn from_response(
        descriptor: &BinarySourceDescriptor,
        matrix: &[Vec<[i128; D]>],
        response: Vec<[i128; D]>,
        images: &[[Q24Image; D]],
        challenges: &Challenges,
        family: FoldChallengeFamily,
    ) -> Result<Self, AkitaError> {
        let (polynomials, columns, source_rows) = dimensions(descriptor, matrix)?;
        let rank = matrix.len();
        let expected_response = checked::product([polynomials, source_rows])
            .ok_or_else(|| invalid("binary response count overflow"))?;
        let expected_images = checked::product([polynomials, columns, rank])
            .ok_or_else(|| invalid("binary image count overflow"))?;
        for (expected, actual) in [
            (expected_response, response.len()),
            (expected_images, images.len()),
        ] {
            if expected != actual {
                return Err(AkitaError::InvalidSize { expected, actual });
            }
        }
        let (low, high) = BINARY_RESPONSE_INTERVAL;
        if response
            .iter()
            .flatten()
            .any(|&x| x < i128::from(low) || x > i128::from(high))
        {
            return Err(invalid("binary response exceeds its four-digit interval"));
        }
        let embedded_challenges = embed_challenges(descriptor, challenges, family)?;
        // Each unreduced product coefficient contains at most D products.
        // Division by Psi combines at most two such coefficients in Q and at
        // most three in the remainder. This conservative bound covers both.
        let product_bound = (source_rows as u128)
            .checked_mul(D as u128)
            .and_then(|x| x.checked_mul(Q24_CENTERED_BOUND as u128))
            .and_then(|x| x.checked_mul(low.unsigned_abs() as u128))
            .and_then(|x| {
                x.checked_add(
                    (columns as u128)
                        * (family.l1_norm() as u128)
                        * (IMAGE_MIN.unsigned_abs() as u128),
                )
            })
            .and_then(|x| x.checked_mul(3))
            .ok_or_else(|| invalid("binary relation integer bound overflow"))?;
        let auxiliary_modulus = (-Prime128OffsetA7F7::one())
            .to_u128_checked()
            .and_then(|x| x.checked_add(1))
            .ok_or(AkitaError::InvalidProof)?;
        if product_bound >= auxiliary_modulus / 2 || product_bound > i128::MAX as u128 {
            return Err(invalid(
                "binary relation cannot be lifted uniquely from its transform field",
            ));
        }
        let mut rows = Vec::new();
        let row_count = checked::product([polynomials, rank])
            .ok_or_else(|| invalid("binary residual row count overflow"))?;
        rows.try_reserve_exact(row_count)
            .map_err(|_| invalid("binary residual allocation failed"))?;
        for (polynomial, response) in response.chunks_exact(source_rows).enumerate() {
            let response = response
                .iter()
                .map(auxiliary_ring)
                .collect::<Result<Vec<_>, _>>()?;
            let challenge_start =
                checked::product([polynomial, columns]).ok_or(AkitaError::InvalidProof)?;
            let challenge_range =
                checked::range(challenge_start, columns).ok_or(AkitaError::InvalidProof)?;
            let common = embedded_challenges
                .get(challenge_range)
                .ok_or(AkitaError::InvalidProof)?;
            let common = common
                .iter()
                .map(auxiliary_ring)
                .collect::<Result<Vec<_>, _>>()?;
            let mut builder = TrinomialRelationQuotientBuilder::new(&response, &common)?;
            for (row_index, matrix_row) in matrix.iter().enumerate() {
                let matrix_row = matrix_row
                    .iter()
                    .map(auxiliary_ring)
                    .collect::<Result<Vec<_>, _>>()?;
                let mut image_row = Vec::with_capacity(columns);
                for column in 0..columns {
                    let offset = checked::mul_add(polynomial, columns, column)
                        .and_then(|x| checked::mul_add(x, rank, row_index))
                        .ok_or(AkitaError::InvalidProof)?;
                    let image = images.get(offset).ok_or(AkitaError::InvalidProof)?;
                    image_row.push(auxiliary_ring(&image.map(|x| i128::from(x.centered())))?);
                }
                let (remainder, quotient) = builder.decompose(&matrix_row, &image_row)?;
                let mut lifted_remainder = [0; D];
                for (destination, &coefficient) in
                    lifted_remainder.iter_mut().zip(remainder.coefficients())
                {
                    *destination = lift_auxiliary(coefficient, auxiliary_modulus, product_bound)?;
                    if *destination % i128::from(Q24) != 0 {
                        return Err(invalid(
                            "binary commitment residual is not divisible by q24",
                        ));
                    }
                }
                let quotient = quotient
                    .into_iter()
                    .map(|x| lift_auxiliary(x, auxiliary_modulus, product_bound))
                    .collect::<Result<Vec<_>, _>>()?;
                rows.push(BinaryRootResidualRow {
                    remainder: lifted_remainder,
                    quotient,
                });
            }
        }
        Ok(Self {
            response,
            images: images.to_vec(),
            rows,
            embedded_challenges,
        })
    }

    /// Repack only the bounded response and image digits into an ordinary ring.
    /// The returned map is also the map used by the public root row compiler.
    pub fn successor_table(
        &self,
        destination_ring_dimension: usize,
    ) -> Result<(RootCoefficientRepacking, Vec<i8>), AkitaError> {
        let geometry =
            RelationRowGeometry::for_polynomial(RelationPolynomial::minus_trinomial(D)?, 1, 1024)?;
        let plan = RootCoefficientRepacking::new(
            vec![
                RootRepackingSegment {
                    geometry,
                    elements: self.response.len(),
                    digits: 4,
                },
                RootRepackingSegment {
                    geometry,
                    elements: self.images.len(),
                    digits: 8,
                },
            ],
            destination_ring_dimension,
        )?;
        let response_len = checked::product([self.response.len(), 4, D])
            .ok_or_else(|| invalid("binary response digit count overflow"))?;
        let image_len = checked::product([self.images.len(), 8, D])
            .ok_or_else(|| invalid("binary image digit count overflow"))?;
        let mut response_digits = Vec::with_capacity(response_len);
        let mut image_digits = Vec::with_capacity(image_len);
        for element in &self.response {
            for digit in 0..4 {
                for &coefficient in element {
                    let word = coefficient
                        .checked_sub(i128::from(BINARY_RESPONSE_INTERVAL.0))
                        .filter(|&x| (0..4096).contains(&x))
                        .ok_or_else(|| invalid("binary response digit decomposition failed"))?;
                    response_digits.push(((word >> (3 * digit)) & 7) as i8 - 4);
                }
            }
        }
        for element in &self.images {
            for digit in 0..8 {
                image_digits.extend(
                    element
                        .iter()
                        .map(|coefficient| coefficient.digits()[digit]),
                );
            }
        }
        let table = plan.repack(&[&response_digits, &image_digits])?;
        Ok((plan, table))
    }

    /// Embed an integer residual row into the actual proof field.
    /// Its q-divisibility must still be authenticated by the caller's lane.
    pub fn residual_in_proof_field(
        &self,
        row: usize,
    ) -> Result<[Prime128Offset275; D], AkitaError> {
        let row = self.rows.get(row).ok_or(AkitaError::InvalidProof)?;
        Ok(row.remainder.map(Prime128Offset275::from_i128))
    }

    /// Assemble typed public A rows against successor segments 0 and 1.
    /// Setup, challenge, and residual integers are embedded in the requested
    /// proof field. Residual targets are construction values awaiting the
    /// authenticated q-quotient or projection lane, never clear proof claims.
    pub fn commitment_group<E: Field>(
        &self,
        descriptor: &BinarySourceDescriptor,
        matrix: &[Vec<[i128; D]>],
    ) -> Result<NativeRootGroup<E>, AkitaError> {
        dimensions(descriptor, matrix)?;
        let matrix = matrix
            .iter()
            .map(|row| {
                row.iter()
                    .map(|coefficients| coefficients.map(E::from_i128))
                    .collect()
            })
            .collect::<Vec<Vec<_>>>();
        let challenges = self
            .embedded_challenges
            .iter()
            .map(|coefficients| coefficients.map(E::from_i128))
            .collect::<Vec<_>>();
        let residuals = self
            .rows
            .iter()
            .map(|row| row.remainder.map(E::from_i128))
            .collect::<Vec<_>>();
        binary_commitment_group(descriptor.clone(), &matrix, &challenges, 0, 1, &residuals)
    }
}

fn dimensions(
    descriptor: &BinarySourceDescriptor,
    matrix: &[Vec<[i128; D]>],
) -> Result<(usize, usize, usize), AkitaError> {
    let index = descriptor.index_map();
    let columns = index.columns();
    let rows = checked::product([4, columns])
        .and_then(|x| checked::exact_div(index.padded_carriers(), x))
        .filter(|&x| x > 0)
        .ok_or_else(|| invalid("binary descriptor has no complete source rows"))?;
    if matrix.is_empty() || matrix.iter().any(|row| row.len() != rows) {
        return Err(invalid(
            "binary matrix does not match descriptor source rows",
        ));
    }
    if matrix
        .iter()
        .flatten()
        .flatten()
        .any(|&x| x < -i128::from(Q24_CENTERED_BOUND) || x > i128::from(Q24_CENTERED_BOUND))
    {
        return Err(invalid(
            "binary matrix coefficient is not centered modulo q24",
        ));
    }
    Ok((descriptor.polynomials().len(), columns, rows))
}

fn embed_challenges(
    descriptor: &BinarySourceDescriptor,
    challenges: &Challenges,
    family: FoldChallengeFamily,
) -> Result<Vec<[i128; D]>, AkitaError> {
    if family != FoldChallengeFamily::BinaryRoot
        || challenges.num_claims() != descriptor.polynomials().len()
        || challenges.num_live_blocks_per_claim() != descriptor.index_map().columns()
    {
        return Err(invalid(
            "binary root requires its selected family and descriptor challenge geometry",
        ));
    }
    let mut embedded = Vec::with_capacity(challenges.len());
    for challenge in challenges.as_slice() {
        challenge.validate_dyn(162)?;
        if challenge.positions.len() > family.weight()
            || challenge.coeffs.iter().any(|&x| x != 1 && x != -1)
        {
            return Err(invalid(
                "binary root challenge is outside the bounded-weight signed family",
            ));
        }
        let mut scalar = [0i128; 162];
        for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
            *scalar
                .get_mut(position as usize)
                .ok_or(AkitaError::InvalidProof)? = i128::from(coefficient);
        }
        embedded.push(pack_signed_components(&[
            scalar, [0; 162], [0; 162], [0; 162],
        ])?);
    }
    Ok(embedded)
}

fn auxiliary_ring(coefficients: &[i128; D]) -> Result<AuxiliaryRing, AkitaError> {
    AuxiliaryRing::from_coefficients(coefficients.map(Prime128OffsetA7F7::from_i128))
        .map_err(|error| invalid(&error.to_string()))
}

fn lift_auxiliary(
    value: Prime128OffsetA7F7,
    modulus: u128,
    bound: u128,
) -> Result<i128, AkitaError> {
    let canonical = value.to_u128_checked().ok_or(AkitaError::InvalidProof)?;
    let (negative, magnitude) = if canonical > modulus / 2 {
        (true, modulus - canonical)
    } else {
        (false, canonical)
    };
    if magnitude > bound {
        return Err(invalid(
            "binary transform coefficient exceeds its verified integer bound",
        ));
    }
    let magnitude = i128::try_from(magnitude).map_err(|_| AkitaError::InvalidProof)?;
    Ok(if negative { -magnitude } else { magnitude })
}

fn invalid(message: &str) -> AkitaError {
    AkitaError::InvalidInput(message.into())
}

#[cfg(test)]
#[path = "binary_root_tests.rs"]
mod tests;
