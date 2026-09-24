//! Shared stage-1 tree shape and polynomial helpers.

use crate::proof::PhysicalL2NormProofWireShape;
use crate::{AkitaStage1StageShape, InnerCommitSecurityRoute};
use akita_error::AkitaError;
use jolt_field::{Field, Ring};

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
pub struct DigitRangeEqualityPoint<E: Field> {
    coordinates: Vec<E>,
    low_variable_count: usize,
}

impl<E: Field> DigitRangeEqualityPoint<E> {
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
}

impl DigitRangePlan {
    /// Construct the canonical range topology for a supported concrete basis.
    ///
    /// # Errors
    ///
    /// Returns an error unless `basis` is one of `4, 8, 16, 32, 64`.
    pub fn new(basis: usize) -> Result<Self, AkitaError> {
        if !matches!(basis, 4 | 8 | 16 | 32 | 64) {
            return Err(AkitaError::InvalidInput(format!(
                "digit range basis must be one of 4, 8, 16, 32, 64; got {basis}"
            )));
        }
        Ok(Self {
            log_basis: basis.trailing_zeros() as u8,
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
    pub fn product_stage_arities(&self) -> &'static [usize] {
        match self.log_basis {
            2 | 3 => &[],
            4 => &[2],
            5 => &[4],
            6 => &[2, 4],
            _ => unreachable!("DigitRangePlan construction validates log basis"),
        }
    }

    /// Number of child lanes emitted by one product substage.
    #[must_use]
    pub fn product_stage_lane_count(self, stage_index: usize) -> Option<usize> {
        self.stage_shapes_iter(0)
            .take(self.product_stage_arities().len())
            .nth(stage_index)
            .map(|shape| shape.child_claims)
    }

    /// Number of quartic (or smaller for basis four) leaf factors.
    #[must_use]
    pub fn leaf_factor_count(self) -> usize {
        match self.log_basis {
            2 | 3 => 1,
            4 => 2,
            5 => 4,
            6 => 8,
            _ => unreachable!("DigitRangePlan construction validates log basis"),
        }
    }

    /// Degree of the final range leaf.
    #[must_use]
    pub fn leaf_degree(self) -> usize {
        if self.log_basis == 2 {
            2
        } else {
            4
        }
    }

    /// Number of range subproofs in transcript order.
    #[must_use]
    pub fn stage_count(self) -> usize {
        self.product_stage_arities().len() + 1
    }

    /// The supported bases have at most two product stages and eight lanes.
    /// Every yielded stage is therefore complete, with no per-index failure.
    fn stage_shapes_iter(self, rounds: usize) -> impl Iterator<Item = AkitaStage1StageShape> {
        let mut child_claims = 1;
        let products = self
            .product_stage_arities()
            .iter()
            .copied()
            .map(move |arity| {
                child_claims *= arity;
                AkitaStage1StageShape {
                    sumcheck_proof: (rounds, arity),
                    child_claims,
                }
            });
        products.chain(std::iter::once(AkitaStage1StageShape {
            sumcheck_proof: (rounds, self.leaf_degree()),
            child_claims: 0,
        }))
    }

    /// Wire shape of one range subproof in transcript order.
    /// Returns `None` only when `stage_index` is outside this plan.
    #[must_use]
    pub fn stage_shape(self, rounds: usize, stage_index: usize) -> Option<AkitaStage1StageShape> {
        self.stage_shapes_iter(rounds).nth(stage_index)
    }

    /// Wire shapes of all range subproofs in transcript order.
    #[must_use]
    pub fn stage_shapes(self, rounds: usize) -> Vec<AkitaStage1StageShape> {
        self.stage_shapes_iter(rounds).collect()
    }

    /// Derive the headerless Stage 1 wire shape from the scheduled A route.
    pub fn proof_shapes_for_route(
        self,
        rounds: usize,
        route: InnerCommitSecurityRoute,
    ) -> Result<
        (
            Vec<AkitaStage1StageShape>,
            Option<PhysicalL2NormProofWireShape>,
        ),
        AkitaError,
    > {
        let shape = self.route_shape(rounds, route)?;
        Ok((
            shape.stages().collect(),
            shape.norm.map(|norm| PhysicalL2NormProofWireShape {
                subclaims: norm.subclaims,
                virtual_evaluations: norm.virtual_evaluations,
                sumcheck: vec![norm.degree; norm.rounds],
            }),
        ))
    }

    /// Return the selected Stage 1 product stages and optional L2 norm shape.
    ///
    /// Returns an error if the selected L2 norm geometry is invalid.
    pub(crate) fn route_shape(
        self,
        rounds: usize,
        route: InnerCommitSecurityRoute,
    ) -> Result<DigitRangeRouteShape, AkitaError> {
        let norm = match route {
            InnerCommitSecurityRoute::Linf(_) => None,
            InnerCommitSecurityRoute::L2 {
                norm_proof_shape, ..
            } => {
                norm_proof_shape.validate()?;
                Some(PhysicalL2NormShape {
                    subclaims: norm_proof_shape.subclaim_count().ok_or_else(|| {
                        AkitaError::InvalidSetup("L2 norm subclaim count overflow".into())
                    })?,
                    virtual_evaluations: norm_proof_shape.virtual_evaluation_count(),
                    rounds,
                    degree: self.leaf_degree() + 1,
                })
            }
        };
        Ok(DigitRangeRouteShape {
            plan: self,
            rounds,
            norm,
        })
    }

    /// Coefficients of the final range-leaf polynomials.
    pub fn leaf_coeffs<E: Field + Ring>(self) -> Vec<Vec<E>> {
        stage1_root_values::<E>(self.basis())
            .chunks(4)
            .map(poly_coeffs_from_roots)
            .collect()
    }

    /// Evaluate the complete balanced-digit range polynomial at `range_image`.
    pub fn evaluate_range_polynomial<E: Field + Ring>(self, range_image: E) -> E {
        let mut value = E::one();
        for root in stage1_root_values::<E>(self.basis()) {
            value *= range_image - root;
        }
        value
    }

    /// Evaluate one leaf polynomial at a range-image value.
    pub fn evaluate_leaf_polynomial<E: Field>(self, coeffs: &[E], range_image: E) -> E {
        coeffs
            .iter()
            .rev()
            .copied()
            .fold(E::zero(), |acc, coeff| acc * range_image + coeff)
    }

    /// Return powers of the interstage batching challenge.
    pub fn interstage_batch_weights<E: Field>(self, gamma: E, count: usize) -> Vec<E> {
        let mut weights = Vec::with_capacity(count);
        let mut weight = E::one();
        for _ in 0..count {
            weights.push(weight);
            weight *= gamma;
        }
        weights
    }

    /// Batch child claims using the current interstage weights.
    pub fn batch_claims<E: Field>(self, weights: &[E], claims: &[E]) -> Result<E, AkitaError> {
        if weights.len() != claims.len() {
            return Err(AkitaError::InvalidSize {
                expected: weights.len(),
                actual: claims.len(),
            });
        }
        Ok(weights
            .iter()
            .zip(claims.iter())
            .fold(E::zero(), |acc, (&weight, &claim)| acc + weight * claim))
    }

    /// Batch leaf-polynomial coefficient vectors using interstage weights.
    pub fn batch_leaf_polynomials<E: Field>(
        self,
        weights: &[E],
        leaf_polynomials: &[Vec<E>],
    ) -> Result<Vec<E>, AkitaError> {
        if weights.len() != leaf_polynomials.len() {
            return Err(AkitaError::InvalidSize {
                expected: weights.len(),
                actual: leaf_polynomials.len(),
            });
        }
        let max_len = leaf_polynomials.iter().map(Vec::len).max().unwrap_or(0);
        let mut batched = vec![E::zero(); max_len];
        for (weight, polynomial) in weights.iter().zip(leaf_polynomials.iter()) {
            for (coefficient, &term) in batched.iter_mut().zip(polynomial.iter()) {
                *coefficient += *weight * term;
            }
        }
        Ok(batched)
    }
}

/// Stage 1 shape for a selected A route, retaining only the optional L2 norm
/// payload in addition to the canonical digit-range stages.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DigitRangeRouteShape {
    plan: DigitRangePlan,
    rounds: usize,
    pub(crate) norm: Option<PhysicalL2NormShape>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PhysicalL2NormShape {
    pub(crate) subclaims: usize,
    pub(crate) virtual_evaluations: usize,
    pub(crate) rounds: usize,
    pub(crate) degree: usize,
}

impl DigitRangeRouteShape {
    pub(crate) fn stages(self) -> impl Iterator<Item = AkitaStage1StageShape> {
        let count = if self.norm.is_some() {
            self.plan.product_stage_arities().len()
        } else {
            self.plan.stage_count()
        };
        self.plan.stage_shapes_iter(self.rounds).take(count)
    }
}

fn stage1_root_values<E: Field + Ring>(b: usize) -> Vec<E> {
    let half = b / 2;
    (0..half)
        .map(|k| {
            let k = k as i64;
            E::from_i64(k * (k + 1))
        })
        .collect()
}

fn poly_coeffs_from_roots<E: Field>(roots: &[E]) -> Vec<E> {
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
    use crate::{
        InnerCommitSecurityRoute, PhysicalL2NormProofShape, SisL2TableDigest, SisL2TableKey,
        SisModulusProfileId, SisSecurityPolicyId, SisTableDigest, SisTableKey,
    };
    use jolt_field::One;
    use jolt_field::Prime128Offset275;

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
            assert!(plan.stage_shape(7, plan.stage_count()).is_none());
            assert!(plan
                .product_stage_lane_count(plan.product_stage_arities().len())
                .is_none());
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
    fn digit_range_batching_rejects_mismatched_lengths() {
        let plan = DigitRangePlan::new(16).unwrap();
        let weights = [F::one(), F::from_u64(2)];
        assert!(plan.batch_claims(&weights, &[F::one()]).is_err());
        assert!(plan
            .batch_leaf_polynomials(&weights, &[vec![F::one()]])
            .is_err());
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

    #[test]
    fn route_authority_derives_exact_headerless_stage1_shapes() {
        let plan = DigitRangePlan::new(64).unwrap();
        let linf = InnerCommitSecurityRoute::Linf(SisTableKey {
            policy: SisSecurityPolicyId::Quantum128BitADPS16,
            table_digest: SisTableDigest::CURRENT,
            modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
            role: crate::SisMatrixRole::Inner,
            ring_dimension: 64,
            coeff_linf_bound: 1,
        });
        let (linf_stages, linf_norm) = plan.proof_shapes_for_route(7, linf).unwrap();
        assert_eq!(linf_stages, plan.stage_shapes(7));
        assert!(linf_norm.is_none());

        let l2 = InnerCommitSecurityRoute::L2 {
            table_key: SisL2TableKey {
                policy: SisSecurityPolicyId::Quantum128BitADPS16,
                table_digest: SisL2TableDigest::CURRENT,
                modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
                ring_dimension: 64,
                collision_l2_sq: 1,
            },
            response_l2_sq_cap: 1,
            norm_proof_shape: PhysicalL2NormProofShape::Direct {
                physical_response_len: 64,
            },
        };
        let (l2_stages, l2_norm) = plan.proof_shapes_for_route(7, l2).unwrap();
        assert_eq!(l2_stages.len(), plan.product_stage_arities().len());
        let l2_norm = l2_norm.expect("L2 wire shape");
        assert_eq!(l2_norm.subclaims, 0);
        assert_eq!(l2_norm.virtual_evaluations, 1);
        assert_eq!(l2_norm.sumcheck, vec![plan.leaf_degree() + 1; 7]);
    }

    #[test]
    fn compact_route_shape_matches_materialized_shapes_and_rejects_bad_norms() {
        let key = SisL2TableKey {
            policy: SisSecurityPolicyId::Quantum128BitADPS16,
            table_digest: SisL2TableDigest::CURRENT,
            modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
            ring_dimension: 64,
            collision_l2_sq: 1,
        };
        let shapes = [
            PhysicalL2NormProofShape::Direct {
                physical_response_len: 512,
            },
            PhysicalL2NormProofShape::LimbGram {
                physical_response_len: 512,
                block_len: 32,
                limb_count: 3,
            },
        ];
        for basis in [4, 8, 16, 32, 64] {
            let plan = DigitRangePlan::new(basis).unwrap();
            for rounds in [0, 1, 7, 32] {
                for norm in shapes {
                    let route = InnerCommitSecurityRoute::L2 {
                        table_key: key,
                        response_l2_sq_cap: 1,
                        norm_proof_shape: norm,
                    };
                    let compact = plan.route_shape(rounds, route).unwrap();
                    let (stages, wire_norm) = plan.proof_shapes_for_route(rounds, route).unwrap();
                    // Reconstruct the former materialized rule independently.
                    let expected: Vec<_> = plan
                        .stage_shapes(rounds)
                        .into_iter()
                        .take(plan.product_stage_arities().len())
                        .collect();
                    assert_eq!(compact.stages().collect::<Vec<_>>(), expected);
                    assert_eq!(stages, expected);
                    let compact_norm = compact.norm.unwrap();
                    let wire_norm = wire_norm.unwrap();
                    assert_eq!(wire_norm.subclaims, norm.subclaim_count().unwrap());
                    assert_eq!(
                        wire_norm.virtual_evaluations,
                        norm.virtual_evaluation_count()
                    );
                    assert_eq!(wire_norm.sumcheck, vec![plan.leaf_degree() + 1; rounds]);
                    assert_eq!(compact_norm.subclaims, wire_norm.subclaims);
                    assert_eq!(
                        compact_norm.virtual_evaluations,
                        wire_norm.virtual_evaluations
                    );
                    assert_eq!(
                        vec![compact_norm.degree; compact_norm.rounds],
                        wire_norm.sumcheck
                    );
                }
            }
            let bad = InnerCommitSecurityRoute::L2 {
                table_key: key,
                response_l2_sq_cap: 1,
                norm_proof_shape: PhysicalL2NormProofShape::Direct {
                    physical_response_len: 0,
                },
            };
            assert!(plan.route_shape(7, bad).is_err());
            assert!(plan.proof_shapes_for_route(7, bad).is_err());
        }
    }
}
