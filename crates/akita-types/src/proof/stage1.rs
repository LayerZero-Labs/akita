//! Shared stage-1 tree shape and polynomial helpers.

use crate::{AkitaStage1Proof, AkitaStage1StageShape};
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt};

const MAX_PRODUCT_STAGES: usize = 2;

/// Checked flat Boolean domain for the compact digit witness.
///
/// The first `live_len` addresses contain digits and the remaining addresses
/// up to `domain_len()` are public zero padding. Variables bind in increasing
/// physical-address bit order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FlatBooleanDomain {
    live_len: usize,
    num_vars: usize,
}

impl FlatBooleanDomain {
    /// Construct a checked live prefix inside a Boolean hypercube.
    ///
    /// # Errors
    ///
    /// Returns an error if the domain width overflows, the live prefix is
    /// empty, or the live prefix exceeds the Boolean domain.
    pub fn new(live_len: usize, num_vars: usize) -> Result<Self, AkitaError> {
        let shift = u32::try_from(num_vars)
            .map_err(|_| AkitaError::InvalidInput("Boolean domain width overflow".to_string()))?;
        let domain_len = 1usize
            .checked_shl(shift)
            .ok_or_else(|| AkitaError::InvalidInput("Boolean domain width overflow".to_string()))?;
        if live_len == 0 || live_len > domain_len {
            return Err(AkitaError::InvalidSize {
                expected: domain_len,
                actual: live_len,
            });
        }
        Ok(Self { live_len, num_vars })
    }

    /// Number of explicit witness entries before zero padding.
    #[must_use]
    pub fn live_len(self) -> usize {
        self.live_len
    }

    /// Total number of Boolean variables.
    #[must_use]
    pub fn num_vars(self) -> usize {
        self.num_vars
    }

    /// Padded Boolean-domain length.
    #[must_use]
    pub fn domain_len(self) -> usize {
        1usize << self.num_vars
    }

    /// Number of live blocks after grouping by `block_variable_count` low bits.
    ///
    /// # Errors
    ///
    /// Returns an error if the block width is larger than the domain or the
    /// live prefix is not block-aligned.
    pub fn live_block_count(self, block_variable_count: usize) -> Result<usize, AkitaError> {
        if block_variable_count > self.num_vars {
            return Err(AkitaError::InvalidSize {
                expected: self.num_vars,
                actual: block_variable_count,
            });
        }
        let block_len = 1usize << block_variable_count;
        if !self.live_len.is_multiple_of(block_len) {
            return Err(AkitaError::InvalidInput(format!(
                "live digit prefix {} is not aligned to block length {block_len}",
                self.live_len
            )));
        }
        Ok(self.live_len / block_len)
    }
}

/// Checked equality point in physical digit-table binding order.
///
/// Ring-switch draws column challenges before ring-slot challenges, while the
/// flat digit table binds its contiguous low address bits first. Construction
/// performs that one protocol-defined permutation and records the low-bit
/// boundary needed by the current compact kernels.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DigitRangeEqualityPoint<E: FieldCore> {
    coordinates: Vec<E>,
    low_variable_count: usize,
}

impl<E: FieldCore> DigitRangeEqualityPoint<E> {
    /// Reorder checked column-then-ring challenges into flat binding order.
    ///
    /// # Errors
    ///
    /// Returns an error if the declared widths overflow or do not consume the
    /// complete challenge vector.
    pub fn from_column_then_ring_challenges(
        challenges: &[E],
        column_variable_count: usize,
        ring_variable_count: usize,
    ) -> Result<Self, AkitaError> {
        let expected = column_variable_count
            .checked_add(ring_variable_count)
            .ok_or_else(|| {
                AkitaError::InvalidInput("digit-range point width overflow".to_string())
            })?;
        if challenges.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: challenges.len(),
            });
        }
        let mut coordinates = Vec::with_capacity(expected);
        coordinates.extend_from_slice(&challenges[column_variable_count..]);
        coordinates.extend_from_slice(&challenges[..column_variable_count]);
        Ok(Self {
            coordinates,
            low_variable_count: ring_variable_count,
        })
    }

    /// Coordinates in physical-address binding order.
    #[must_use]
    pub fn coordinates(&self) -> &[E] {
        &self.coordinates
    }

    /// Consume the checked point and return its ordered coordinates.
    #[must_use]
    pub fn into_coordinates(self) -> Vec<E> {
        self.coordinates
    }

    /// Number of initially contiguous low-address variables.
    #[must_use]
    pub fn low_variable_count(&self) -> usize {
        self.low_variable_count
    }

    /// Validate that this point spans `domain` exactly.
    ///
    /// # Errors
    ///
    /// Returns an error if the point width differs from the domain width.
    pub fn validate_domain(&self, domain: FlatBooleanDomain) -> Result<(), AkitaError> {
        if self.coordinates.len() != domain.num_vars() {
            return Err(AkitaError::InvalidSize {
                expected: domain.num_vars(),
                actual: self.coordinates.len(),
            });
        }
        Ok(())
    }
}

/// Checked topology for the balanced digit range proof.
///
/// This is the single authority for the supported basis, product-stage
/// arities, leaf factorization, proof shape, and child-claim order. It is
/// constructed from the concrete range basis already selected by the
/// ring-switch boundary and has no dependency on level parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DigitRangePlan {
    log_basis: u8,
    product_stage_arities: [usize; MAX_PRODUCT_STAGES],
    product_stage_count: u8,
    leaf_factor_count: usize,
    leaf_degree: usize,
}

impl DigitRangePlan {
    /// Construct the canonical range topology for a supported concrete basis.
    ///
    /// # Errors
    ///
    /// Returns an error unless `basis` is one of `4, 8, 16, 32, 64`.
    pub fn new(basis: usize) -> Result<Self, AkitaError> {
        let (product_stage_arities, product_stage_count, leaf_factor_count, leaf_degree) =
            match basis {
                4 => ([0, 0], 0, 1, 2),
                8 => ([0, 0], 0, 1, 4),
                16 => ([2, 0], 1, 2, 4),
                32 => ([4, 0], 1, 4, 4),
                64 => ([2, 4], 2, 8, 4),
                _ => {
                    return Err(AkitaError::InvalidInput(format!(
                        "digit range basis must be one of 4, 8, 16, 32, 64; got {basis}"
                    )))
                }
            };
        Ok(Self {
            log_basis: basis.trailing_zeros() as u8,
            product_stage_arities,
            product_stage_count,
            leaf_factor_count,
            leaf_degree,
        })
    }

    /// Concrete balanced decomposition basis.
    #[must_use]
    pub fn basis(self) -> usize {
        1usize << self.log_basis
    }

    /// Base-2 logarithm of [`Self::basis`].
    #[must_use]
    pub fn log_basis(self) -> u8 {
        self.log_basis
    }

    /// Product-stage arities in transcript order, before the leaf stage.
    #[must_use]
    pub fn product_stage_arities(&self) -> &[usize] {
        &self.product_stage_arities[..usize::from(self.product_stage_count)]
    }

    /// Number of quartic (or smaller for basis four) leaf factors.
    #[must_use]
    pub fn leaf_factor_count(self) -> usize {
        self.leaf_factor_count
    }

    /// Degree of the final range leaf.
    #[must_use]
    pub fn leaf_degree(self) -> usize {
        self.leaf_degree
    }

    /// Number of range subproofs in transcript order.
    #[must_use]
    pub fn stage_count(self) -> usize {
        usize::from(self.product_stage_count) + 1
    }

    /// Wire shape of one range subproof in transcript order.
    #[must_use]
    pub fn stage_shape(self, rounds: usize, stage_index: usize) -> Option<AkitaStage1StageShape> {
        if stage_index < usize::from(self.product_stage_count) {
            let mut parent_count = 1usize;
            for &arity in &self.product_stage_arities[..stage_index] {
                parent_count *= arity;
            }
            let arity = self.product_stage_arities[stage_index];
            return Some(AkitaStage1StageShape {
                sumcheck_proof: (rounds, arity),
                child_claims: parent_count * arity,
            });
        }
        (stage_index == usize::from(self.product_stage_count)).then_some(AkitaStage1StageShape {
            sumcheck_proof: (rounds, self.leaf_degree),
            child_claims: 0,
        })
    }

    /// Wire shapes of all range subproofs in transcript order.
    #[must_use]
    pub fn stage_shapes(self, rounds: usize) -> Vec<AkitaStage1StageShape> {
        (0..self.stage_count())
            .filter_map(|stage_index| self.stage_shape(rounds, stage_index))
            .collect()
    }

    /// Validate the complete in-memory range-proof shape without allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of substages, rounds, polynomial degree,
    /// or child claims differs from this plan.
    pub fn validate_proof_shape<E: FieldCore>(
        self,
        proof: &AkitaStage1Proof<E>,
        rounds: usize,
    ) -> Result<(), AkitaError> {
        if proof.stages.len() != self.stage_count() {
            return Err(AkitaError::InvalidSize {
                expected: self.stage_count(),
                actual: proof.stages.len(),
            });
        }
        for (stage_index, stage) in proof.stages.iter().enumerate() {
            let expected = self
                .stage_shape(rounds, stage_index)
                .ok_or(AkitaError::InvalidProof)?;
            if stage.sumcheck_proof.round_polys.len() != expected.sumcheck_proof.0 {
                return Err(AkitaError::InvalidSize {
                    expected: expected.sumcheck_proof.0,
                    actual: stage.sumcheck_proof.round_polys.len(),
                });
            }
            for round_poly in &stage.sumcheck_proof.round_polys {
                if round_poly.coeffs_except_linear_term.len() != expected.sumcheck_proof.1 {
                    return Err(AkitaError::InvalidSize {
                        expected: expected.sumcheck_proof.1,
                        actual: round_poly.coeffs_except_linear_term.len(),
                    });
                }
            }
            if stage.child_claims.len() != expected.child_claims {
                return Err(AkitaError::InvalidSize {
                    expected: expected.child_claims,
                    actual: stage.child_claims.len(),
                });
            }
        }
        Ok(())
    }

    /// Coefficients of the final range-leaf polynomials.
    pub fn leaf_coeffs<E: FieldCore + FromPrimitiveInt>(self) -> Vec<Vec<E>> {
        stage1_root_values::<E>(self.basis())
            .chunks(4)
            .map(poly_coeffs_from_roots)
            .collect()
    }

    /// Evaluate the complete balanced-digit range polynomial at `range_image`.
    pub fn evaluate_range_polynomial<E: FieldCore + FromPrimitiveInt>(self, range_image: E) -> E {
        let mut value = E::one();
        for root in stage1_root_values::<E>(self.basis()) {
            value *= range_image - root;
        }
        value
    }

    /// Evaluate one leaf polynomial at a range-image value.
    pub fn evaluate_leaf_polynomial<E: FieldCore>(self, coeffs: &[E], range_image: E) -> E {
        coeffs
            .iter()
            .rev()
            .copied()
            .fold(E::zero(), |acc, coeff| acc * range_image + coeff)
    }

    /// Return powers of the interstage batching challenge.
    pub fn interstage_batch_weights<E: FieldCore>(self, gamma: E, count: usize) -> Vec<E> {
        let mut weights = Vec::with_capacity(count);
        let mut weight = E::one();
        for _ in 0..count {
            weights.push(weight);
            weight *= gamma;
        }
        weights
    }

    /// Batch child claims using the current interstage weights.
    pub fn batch_claims<E: FieldCore>(self, weights: &[E], claims: &[E]) -> E {
        debug_assert_eq!(weights.len(), claims.len());
        weights
            .iter()
            .zip(claims.iter())
            .fold(E::zero(), |acc, (&weight, &claim)| acc + weight * claim)
    }

    /// Batch leaf-polynomial coefficient vectors using interstage weights.
    pub fn batch_leaf_polynomials<E: FieldCore>(
        self,
        weights: &[E],
        leaf_polynomials: &[Vec<E>],
    ) -> Vec<E> {
        debug_assert_eq!(weights.len(), leaf_polynomials.len());
        let max_len = leaf_polynomials.iter().map(Vec::len).max().unwrap_or(0);
        let mut batched = vec![E::zero(); max_len];
        for (weight, polynomial) in weights.iter().zip(leaf_polynomials.iter()) {
            for (coefficient, &term) in batched.iter_mut().zip(polynomial.iter()) {
                *coefficient += *weight * term;
            }
        }
        batched
    }
}

fn stage1_root_values<E: FieldCore + FromPrimitiveInt>(b: usize) -> Vec<E> {
    let half = b / 2;
    (0..half)
        .map(|k| {
            let k = k as i64;
            E::from_i64(k * (k + 1))
        })
        .collect()
}

fn poly_coeffs_from_roots<E: FieldCore>(roots: &[E]) -> Vec<E> {
    let mut coeffs = vec![E::one()];
    for &root in roots {
        let mut next = vec![E::zero(); coeffs.len() + 1];
        for (idx, &coeff) in coeffs.iter().enumerate() {
            next[idx] -= coeff * root;
            next[idx + 1] += coeff;
        }
        coeffs = next;
    }
    coeffs
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128Offset275;

    type F = Prime128Offset275;

    #[test]
    fn digit_range_plan_exhausts_supported_topologies() {
        let expected = [
            (4, &[(2, 0)][..]),
            (8, &[(4, 0)][..]),
            (16, &[(2, 2), (4, 0)][..]),
            (32, &[(4, 4), (4, 0)][..]),
            (64, &[(2, 2), (4, 8), (4, 0)][..]),
        ];
        for (basis, expected_stages) in expected {
            let plan = DigitRangePlan::new(basis).expect("supported basis");
            let actual = plan
                .stage_shapes(7)
                .into_iter()
                .map(|shape| (shape.sumcheck_proof.1, shape.child_claims))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected_stages);
            assert_eq!(plan.leaf_coeffs::<F>().len(), plan.leaf_factor_count());
        }
    }

    #[test]
    fn digit_range_plan_rejects_every_nearby_unsupported_basis() {
        for basis in [0, 1, 2, 3, 5, 6, 7, 9, 15, 63, 65, 128] {
            assert!(DigitRangePlan::new(basis).is_err(), "basis {basis}");
        }
    }

    #[test]
    fn flat_domain_checks_count_width_and_alignment() {
        let domain = FlatBooleanDomain::new(24, 5).expect("live prefix");
        assert_eq!(domain.live_len(), 24);
        assert_eq!(domain.num_vars(), 5);
        assert_eq!(domain.domain_len(), 32);
        assert_eq!(domain.live_block_count(2).unwrap(), 6);
        assert!(domain.live_block_count(4).is_err());
        assert!(FlatBooleanDomain::new(0, 5).is_err());
        assert!(FlatBooleanDomain::new(33, 5).is_err());
        assert!(FlatBooleanDomain::new(1, usize::MAX).is_err());
    }

    #[test]
    fn equality_point_owns_every_column_then_ring_permutation() {
        let transcript_point = [
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
            F::from_u64(5),
        ];
        let domain = FlatBooleanDomain::new(24, transcript_point.len()).unwrap();
        for column_variable_count in 0..=transcript_point.len() {
            let ring_variable_count = transcript_point.len() - column_variable_count;
            let point = DigitRangeEqualityPoint::from_column_then_ring_challenges(
                &transcript_point,
                column_variable_count,
                ring_variable_count,
            )
            .expect("checked point");
            let expected = transcript_point[column_variable_count..]
                .iter()
                .chain(&transcript_point[..column_variable_count])
                .copied()
                .collect::<Vec<_>>();
            assert_eq!(point.coordinates(), expected);
            assert_eq!(point.low_variable_count(), ring_variable_count);
            point.validate_domain(domain).expect("matching domain");
        }

        assert!(DigitRangeEqualityPoint::from_column_then_ring_challenges(
            &transcript_point[..4],
            3,
            2,
        )
        .is_err());
        let short_point =
            DigitRangeEqualityPoint::from_column_then_ring_challenges(&transcript_point[..4], 2, 2)
                .unwrap();
        assert!(short_point.validate_domain(domain).is_err());
    }
}
