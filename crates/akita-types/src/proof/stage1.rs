//! Shared stage-1 tree shape and polynomial helpers.

use crate::layout::digit_range::FlatBooleanDomain;
use akita_error::AkitaError;
use jolt_field::Field;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::digit_range::DigitRangePlan;
    use crate::{
        InnerCommitSecurityRoute, PhysicalL2NormProofShape, SisL2TableDigest, SisL2TableKey,
        SisModulusProfileId, SisSecurityPolicyId, SisTableDigest, SisTableKey,
    };
    use jolt_field::Prime128Offset275;
    use jolt_field::{One, Ring};

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
