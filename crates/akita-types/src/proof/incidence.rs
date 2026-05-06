//! Normalized point/group/claim incidence for batched openings.

use super::{MultiPointBatchShape, VerifierClaims};
use akita_field::AkitaError;
use akita_transcript::labels::ABSORB_BATCH_SHAPE;
use akita_transcript::Transcript;
use std::collections::BTreeSet;

/// One committed group in a normalized opening incidence graph.
#[derive(Debug, Clone, Copy)]
pub struct IncidenceGroup<'a, C> {
    /// Commitment for the group.
    pub commitment: &'a C,
    /// Number of committed polynomials addressable within this group.
    pub poly_count: usize,
}

/// One claimed opening edge from a point to a committed group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IncidenceClaim<F> {
    /// Opening-point index.
    pub point_idx: usize,
    /// Committed-group index.
    pub group_idx: usize,
    /// Polynomial index within the committed group.
    pub poly_idx: usize,
    /// Claimed evaluation at `points[point_idx]`.
    pub claimed_eval: F,
}

/// Verifier-safe normalized incidence graph for batched openings.
#[derive(Debug, Clone)]
pub struct ClaimIncidence<'a, F, C> {
    /// Distinct opening points.
    pub points: Vec<&'a [F]>,
    /// Distinct committed groups.
    pub groups: Vec<IncidenceGroup<'a, C>>,
    /// Individual claimed openings.
    pub claims: Vec<IncidenceClaim<F>>,
}

/// Normalize the current verifier claim input shape into an incidence graph.
///
/// The existing ergonomic input is grouped by opening point, then by committed
/// group. This preserves that order by materializing one incidence group for
/// each caller-provided group occurrence.
pub fn verifier_claims_to_incidence<'a, F, C>(
    claims: &VerifierClaims<'a, F, C>,
) -> ClaimIncidence<'a, F, C>
where
    F: Copy,
{
    let points = claims.iter().map(|(point, _)| *point).collect();
    let mut groups = Vec::new();
    let mut incidence_claims = Vec::new();

    for (point_idx, (_, groups_at_point)) in claims.iter().enumerate() {
        for group in groups_at_point {
            let group_idx = groups.len();
            groups.push(IncidenceGroup {
                commitment: group.commitment,
                poly_count: group.openings.len(),
            });
            incidence_claims.extend(group.openings.iter().enumerate().map(
                |(poly_idx, &claimed_eval)| IncidenceClaim {
                    point_idx,
                    group_idx,
                    poly_idx,
                    claimed_eval,
                },
            ));
        }
    }

    ClaimIncidence {
        points,
        groups,
        claims: incidence_claims,
    }
}

/// Capacity and dimension limits for incidence validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClaimIncidenceLimits {
    /// Maximum supported number of variables in an opening point.
    pub max_num_vars: usize,
    /// Maximum supported number of distinct opening points.
    pub max_num_points: usize,
    /// Maximum supported number of claimed openings.
    pub max_num_claims: usize,
}

/// Derived routing and count data for a normalized incidence graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimIncidenceSummary {
    /// Number of variables in every opening point.
    pub num_vars: usize,
    /// Number of distinct opening points.
    pub num_points: usize,
    /// Number of distinct committed groups.
    pub num_groups: usize,
    /// Number of individual claimed openings.
    pub num_claims: usize,
    /// Opening-point index for each flattened claim.
    pub claim_to_point: Vec<usize>,
    /// Committed-group index for each flattened claim.
    pub claim_to_group: Vec<usize>,
    /// Polynomial index within its committed group for each flattened claim.
    pub claim_poly_indices: Vec<usize>,
    /// Number of polynomials in each committed group.
    pub group_poly_counts: Vec<usize>,
    /// Number of claims referencing each committed group.
    pub group_claim_counts: Vec<usize>,
    /// Number of claims evaluated at each opening point.
    pub point_claim_counts: Vec<usize>,
    /// Number of distinct committed groups touched by each opening point.
    pub point_group_counts: Vec<usize>,
}

impl ClaimIncidenceSummary {
    /// Derive the legacy point/group batch shape consumed by root proof code.
    ///
    /// Groups are emitted in canonical `(point_idx, group_idx)` order. The
    /// incidence summary retains exact claim routing, while this shape carries
    /// only the counts needed by the existing batched root layout.
    pub fn multi_point_batch_shape(&self) -> MultiPointBatchShape {
        debug_assert_eq!(self.claim_to_point.len(), self.num_claims);
        debug_assert_eq!(self.claim_to_group.len(), self.num_claims);

        let mut groups_by_point = vec![BTreeSet::new(); self.num_points];
        for claim_idx in 0..self.num_claims {
            groups_by_point[self.claim_to_point[claim_idx]].insert(self.claim_to_group[claim_idx]);
        }

        let mut point_group_sizes = Vec::with_capacity(self.num_points);
        let mut claim_group_sizes = Vec::new();
        let mut claim_to_point = Vec::with_capacity(self.num_claims);

        for (point_idx, groups) in groups_by_point.iter().enumerate() {
            point_group_sizes.push(groups.len());
            for &group_idx in groups {
                let group_claim_count = (0..self.num_claims)
                    .filter(|&claim_idx| {
                        self.claim_to_point[claim_idx] == point_idx
                            && self.claim_to_group[claim_idx] == group_idx
                    })
                    .count();
                claim_group_sizes.push(group_claim_count);
                claim_to_point.extend(std::iter::repeat_n(point_idx, group_claim_count));
            }
        }

        MultiPointBatchShape {
            point_group_sizes,
            claim_group_sizes,
            claim_to_point,
        }
    }
}

impl<'a, F, C> ClaimIncidence<'a, F, C> {
    /// Validate the incidence graph and derive its flattened routing summary.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph is empty, exceeds supplied capacities, has
    /// inconsistent point dimensions, references invalid point/group/poly
    /// indices, contains duplicate claim edges, or contains unused points or
    /// groups.
    pub fn validate(
        &self,
        limits: ClaimIncidenceLimits,
    ) -> Result<ClaimIncidenceSummary, AkitaError> {
        if self.points.is_empty() {
            return Err(AkitaError::InvalidInput(
                "claim incidence requires at least one opening point".to_string(),
            ));
        }
        if self.groups.is_empty() {
            return Err(AkitaError::InvalidInput(
                "claim incidence requires at least one committed group".to_string(),
            ));
        }
        if self.claims.is_empty() {
            return Err(AkitaError::InvalidInput(
                "claim incidence requires at least one claim".to_string(),
            ));
        }

        let num_vars = self.points[0].len();
        if self.points.iter().any(|point| point.len() != num_vars) {
            return Err(AkitaError::InvalidInput(
                "claim incidence requires all opening points to have the same length".to_string(),
            ));
        }
        if num_vars > limits.max_num_vars {
            return Err(AkitaError::InvalidPointDimension {
                expected: limits.max_num_vars,
                actual: num_vars,
            });
        }
        if self.points.len() > limits.max_num_points {
            return Err(AkitaError::InvalidSize {
                expected: limits.max_num_points,
                actual: self.points.len(),
            });
        }
        if self.claims.len() > limits.max_num_claims {
            return Err(AkitaError::InvalidSize {
                expected: limits.max_num_claims,
                actual: self.claims.len(),
            });
        }

        let group_poly_counts: Vec<usize> = self
            .groups
            .iter()
            .enumerate()
            .map(|(idx, group)| {
                if group.poly_count == 0 {
                    Err(AkitaError::InvalidInput(format!(
                        "claim incidence group {idx} must contain at least one polynomial",
                    )))
                } else {
                    Ok(group.poly_count)
                }
            })
            .collect::<Result<_, _>>()?;

        let mut claim_to_point = Vec::with_capacity(self.claims.len());
        let mut claim_to_group = Vec::with_capacity(self.claims.len());
        let mut claim_poly_indices = Vec::with_capacity(self.claims.len());
        let mut group_claim_counts = vec![0usize; self.groups.len()];
        let mut point_claim_counts = vec![0usize; self.points.len()];
        let mut point_group_sets = vec![BTreeSet::new(); self.points.len()];
        let mut seen_edges = BTreeSet::new();

        for claim in &self.claims {
            if claim.point_idx >= self.points.len() {
                return Err(AkitaError::InvalidInput(format!(
                    "claim incidence point index {} out of range",
                    claim.point_idx
                )));
            }
            if claim.group_idx >= self.groups.len() {
                return Err(AkitaError::InvalidInput(format!(
                    "claim incidence group index {} out of range",
                    claim.group_idx
                )));
            }
            if claim.poly_idx >= group_poly_counts[claim.group_idx] {
                return Err(AkitaError::InvalidInput(format!(
                    "claim incidence polynomial index {} out of range for group {}",
                    claim.poly_idx, claim.group_idx
                )));
            }
            if !seen_edges.insert((claim.point_idx, claim.group_idx, claim.poly_idx)) {
                return Err(AkitaError::InvalidInput(
                    "claim incidence contains duplicate point/group/poly claim".to_string(),
                ));
            }

            claim_to_point.push(claim.point_idx);
            claim_to_group.push(claim.group_idx);
            claim_poly_indices.push(claim.poly_idx);
            group_claim_counts[claim.group_idx] = group_claim_counts[claim.group_idx]
                .checked_add(1)
                .ok_or_else(|| {
                    AkitaError::InvalidInput("claim incidence group count overflow".to_string())
                })?;
            point_claim_counts[claim.point_idx] = point_claim_counts[claim.point_idx]
                .checked_add(1)
                .ok_or_else(|| {
                    AkitaError::InvalidInput("claim incidence point count overflow".to_string())
                })?;
            point_group_sets[claim.point_idx].insert(claim.group_idx);
        }

        if let Some(point_idx) = point_claim_counts.iter().position(|&count| count == 0) {
            return Err(AkitaError::InvalidInput(format!(
                "claim incidence point {point_idx} is unused",
            )));
        }
        if let Some(group_idx) = group_claim_counts.iter().position(|&count| count == 0) {
            return Err(AkitaError::InvalidInput(format!(
                "claim incidence group {group_idx} is unused",
            )));
        }

        let point_group_counts = point_group_sets
            .iter()
            .map(BTreeSet::len)
            .collect::<Vec<_>>();

        Ok(ClaimIncidenceSummary {
            num_vars,
            num_points: self.points.len(),
            num_groups: self.groups.len(),
            num_claims: self.claims.len(),
            claim_to_point,
            claim_to_group,
            claim_poly_indices,
            group_poly_counts,
            group_claim_counts,
            point_claim_counts,
            point_group_counts,
        })
    }
}

/// Absorb normalized incidence shape and routing into the transcript.
pub fn append_claim_incidence_shape_to_transcript<F, T>(
    summary: &ClaimIncidenceSummary,
    transcript: &mut T,
) where
    F: akita_field::FieldCore + akita_field::CanonicalField,
    T: Transcript<F>,
{
    transcript.append_serde(ABSORB_BATCH_SHAPE, &summary.num_vars);
    transcript.append_serde(ABSORB_BATCH_SHAPE, &summary.num_points);
    transcript.append_serde(ABSORB_BATCH_SHAPE, &summary.num_groups);
    transcript.append_serde(ABSORB_BATCH_SHAPE, &summary.num_claims);
    for count in &summary.group_poly_counts {
        transcript.append_serde(ABSORB_BATCH_SHAPE, count);
    }
    for count in &summary.point_claim_counts {
        transcript.append_serde(ABSORB_BATCH_SHAPE, count);
    }
    for claim_idx in 0..summary.num_claims {
        transcript.append_serde(ABSORB_BATCH_SHAPE, &summary.claim_to_point[claim_idx]);
        transcript.append_serde(ABSORB_BATCH_SHAPE, &summary.claim_to_group[claim_idx]);
        transcript.append_serde(ABSORB_BATCH_SHAPE, &summary.claim_poly_indices[claim_idx]);
    }
}

#[cfg(test)]
mod tests {
    use super::super::CommittedOpenings;
    use super::*;

    fn generous_limits() -> ClaimIncidenceLimits {
        ClaimIncidenceLimits {
            max_num_vars: 8,
            max_num_points: 8,
            max_num_claims: 16,
        }
    }

    #[test]
    fn incidence_summary_tracks_routing_counts() {
        let p0 = [1u64, 2];
        let p1 = [3u64, 4];
        let c0 = "commitment-0";
        let c1 = "commitment-1";
        let incidence = ClaimIncidence {
            points: vec![&p0, &p1],
            groups: vec![
                IncidenceGroup {
                    commitment: &c0,
                    poly_count: 2,
                },
                IncidenceGroup {
                    commitment: &c1,
                    poly_count: 1,
                },
            ],
            claims: vec![
                IncidenceClaim {
                    point_idx: 0,
                    group_idx: 0,
                    poly_idx: 1,
                    claimed_eval: 10u64,
                },
                IncidenceClaim {
                    point_idx: 0,
                    group_idx: 1,
                    poly_idx: 0,
                    claimed_eval: 11u64,
                },
                IncidenceClaim {
                    point_idx: 1,
                    group_idx: 0,
                    poly_idx: 0,
                    claimed_eval: 12u64,
                },
            ],
        };

        let summary = incidence
            .validate(generous_limits())
            .expect("valid incidence");

        assert_eq!(summary.num_vars, 2);
        assert_eq!(summary.num_points, 2);
        assert_eq!(summary.num_groups, 2);
        assert_eq!(summary.num_claims, 3);
        assert_eq!(summary.claim_to_point, vec![0, 0, 1]);
        assert_eq!(summary.claim_to_group, vec![0, 1, 0]);
        assert_eq!(summary.claim_poly_indices, vec![1, 0, 0]);
        assert_eq!(summary.group_poly_counts, vec![2, 1]);
        assert_eq!(summary.group_claim_counts, vec![2, 1]);
        assert_eq!(summary.point_claim_counts, vec![2, 1]);
        assert_eq!(summary.point_group_counts, vec![2, 1]);

        let batch_shape = summary.multi_point_batch_shape();
        assert_eq!(batch_shape.point_group_sizes, vec![2, 1]);
        assert_eq!(batch_shape.claim_group_sizes, vec![1, 1, 1]);
        assert_eq!(batch_shape.claim_to_point, vec![0, 0, 1]);
    }

    #[test]
    fn one_group_can_be_opened_at_many_points_without_duplicate_group_input() {
        let p0 = [1u64];
        let p1 = [2u64];
        let commitment = "shared";
        let incidence = ClaimIncidence {
            points: vec![&p0, &p1],
            groups: vec![IncidenceGroup {
                commitment: &commitment,
                poly_count: 1,
            }],
            claims: vec![
                IncidenceClaim {
                    point_idx: 0,
                    group_idx: 0,
                    poly_idx: 0,
                    claimed_eval: 3u64,
                },
                IncidenceClaim {
                    point_idx: 1,
                    group_idx: 0,
                    poly_idx: 0,
                    claimed_eval: 4u64,
                },
            ],
        };

        let summary = incidence
            .validate(generous_limits())
            .expect("valid incidence");

        assert_eq!(summary.num_groups, 1);
        assert_eq!(summary.group_claim_counts, vec![2]);
        assert_eq!(summary.point_group_counts, vec![1, 1]);
        assert_eq!(summary.claim_to_group, vec![0, 0]);
    }

    #[test]
    fn batch_shape_groups_multiple_claims_for_one_point_group_pair() {
        let p0 = [1u64];
        let commitment = "shared";
        let incidence = ClaimIncidence {
            points: vec![&p0],
            groups: vec![IncidenceGroup {
                commitment: &commitment,
                poly_count: 2,
            }],
            claims: vec![
                IncidenceClaim {
                    point_idx: 0,
                    group_idx: 0,
                    poly_idx: 0,
                    claimed_eval: 3u64,
                },
                IncidenceClaim {
                    point_idx: 0,
                    group_idx: 0,
                    poly_idx: 1,
                    claimed_eval: 4u64,
                },
            ],
        };

        let summary = incidence
            .validate(generous_limits())
            .expect("valid same-point group incidence");
        let batch_shape = summary.multi_point_batch_shape();

        assert_eq!(batch_shape.point_group_sizes, vec![1]);
        assert_eq!(batch_shape.claim_group_sizes, vec![2]);
        assert_eq!(batch_shape.claim_to_point, vec![0, 0]);
    }

    #[test]
    fn verifier_claims_normalize_to_incidence_graph() {
        let p0 = [1u64, 2];
        let p1 = [3u64, 4];
        let c0 = 10usize;
        let c1 = 11usize;
        let c2 = 12usize;
        let openings0 = [20u64, 21];
        let openings1 = [22u64];
        let openings2 = [23u64, 24, 25];
        let claims = vec![
            (
                &p0[..],
                vec![
                    CommittedOpenings {
                        commitment: &c0,
                        openings: &openings0,
                    },
                    CommittedOpenings {
                        commitment: &c1,
                        openings: &openings1,
                    },
                ],
            ),
            (
                &p1[..],
                vec![CommittedOpenings {
                    commitment: &c2,
                    openings: &openings2,
                }],
            ),
        ];

        let incidence = verifier_claims_to_incidence(&claims);

        assert_eq!(incidence.points, vec![&p0[..], &p1[..]]);
        assert_eq!(incidence.groups.len(), 3);
        assert_eq!(incidence.groups[0].commitment, &c0);
        assert_eq!(incidence.groups[0].poly_count, 2);
        assert_eq!(incidence.groups[1].commitment, &c1);
        assert_eq!(incidence.groups[1].poly_count, 1);
        assert_eq!(incidence.groups[2].commitment, &c2);
        assert_eq!(incidence.groups[2].poly_count, 3);
        assert_eq!(
            incidence.claims,
            vec![
                IncidenceClaim {
                    point_idx: 0,
                    group_idx: 0,
                    poly_idx: 0,
                    claimed_eval: 20,
                },
                IncidenceClaim {
                    point_idx: 0,
                    group_idx: 0,
                    poly_idx: 1,
                    claimed_eval: 21,
                },
                IncidenceClaim {
                    point_idx: 0,
                    group_idx: 1,
                    poly_idx: 0,
                    claimed_eval: 22,
                },
                IncidenceClaim {
                    point_idx: 1,
                    group_idx: 2,
                    poly_idx: 0,
                    claimed_eval: 23,
                },
                IncidenceClaim {
                    point_idx: 1,
                    group_idx: 2,
                    poly_idx: 1,
                    claimed_eval: 24,
                },
                IncidenceClaim {
                    point_idx: 1,
                    group_idx: 2,
                    poly_idx: 2,
                    claimed_eval: 25,
                },
            ]
        );

        let summary = incidence
            .validate(generous_limits())
            .expect("normalized verifier claims should validate");
        assert_eq!(summary.claim_to_point, vec![0, 0, 0, 1, 1, 1]);
        assert_eq!(summary.claim_to_group, vec![0, 0, 1, 2, 2, 2]);
        assert_eq!(summary.claim_poly_indices, vec![0, 1, 0, 0, 1, 2]);
        assert_eq!(
            summary.multi_point_batch_shape(),
            MultiPointBatchShape {
                point_group_sizes: vec![2, 1],
                claim_group_sizes: vec![2, 1, 3],
                claim_to_point: vec![0, 0, 0, 1, 1, 1],
            }
        );
    }

    #[test]
    fn incidence_validation_rejects_malformed_shapes() {
        let p0 = [1u64];
        let p1 = [2u64, 3];
        let commitment = "commitment";

        let mismatched_points = ClaimIncidence {
            points: vec![&p0, &p1],
            groups: vec![IncidenceGroup {
                commitment: &commitment,
                poly_count: 1,
            }],
            claims: vec![IncidenceClaim {
                point_idx: 0,
                group_idx: 0,
                poly_idx: 0,
                claimed_eval: 5u64,
            }],
        };
        assert!(matches!(
            mismatched_points.validate(generous_limits()),
            Err(AkitaError::InvalidInput(_))
        ));

        let invalid_poly = ClaimIncidence {
            points: vec![&p0],
            groups: vec![IncidenceGroup {
                commitment: &commitment,
                poly_count: 1,
            }],
            claims: vec![IncidenceClaim {
                point_idx: 0,
                group_idx: 0,
                poly_idx: 1,
                claimed_eval: 5u64,
            }],
        };
        assert!(matches!(
            invalid_poly.validate(generous_limits()),
            Err(AkitaError::InvalidInput(_))
        ));

        let duplicate_edge = ClaimIncidence {
            points: vec![&p0],
            groups: vec![IncidenceGroup {
                commitment: &commitment,
                poly_count: 1,
            }],
            claims: vec![
                IncidenceClaim {
                    point_idx: 0,
                    group_idx: 0,
                    poly_idx: 0,
                    claimed_eval: 5u64,
                },
                IncidenceClaim {
                    point_idx: 0,
                    group_idx: 0,
                    poly_idx: 0,
                    claimed_eval: 6u64,
                },
            ],
        };
        assert!(matches!(
            duplicate_edge.validate(generous_limits()),
            Err(AkitaError::InvalidInput(_))
        ));
    }
}
