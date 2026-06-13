//! Normalized point/claim incidence for batched openings.
//!
//! Each opening point in a batched opening references exactly one commitment.
//! That commitment may bundle multiple polynomials, so a point can carry
//! several claimed openings, and the same commitment may be referenced by
//! multiple points (multipoint opening of a shared commitment).

use super::VerifierClaims;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, MulBase};
use akita_transcript::labels::{ABSORB_BATCH_SHAPE, CHALLENGE_EVAL_BATCH};
use akita_transcript::{sample_ext_challenge, Transcript};
use std::collections::BTreeSet;

/// One claimed opening edge from a point to a polynomial within the point's
/// commitment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IncidenceClaim<F> {
    /// Opening-point index.
    pub point_idx: usize,
    /// Polynomial index within the point's commitment.
    pub poly_idx: usize,
    /// Claimed evaluation at `points[point_idx]`.
    pub claimed_eval: F,
}

/// Verifier-safe normalized incidence graph for batched openings.
///
/// Each opening point cites exactly one commitment via `claims[i].poly_idx`
/// into the point's bundled polynomial family.
#[derive(Debug, Clone)]
pub struct ClaimIncidence<'a, F> {
    /// Distinct opening points.
    pub points: Vec<&'a [F]>,
    /// Individual claimed openings.
    pub claims: Vec<IncidenceClaim<F>>,
}

/// Normalize the public verifier-claim input shape into an incidence graph.
pub fn verifier_claims_to_incidence<'a, F, C>(
    claims: &VerifierClaims<'a, F, C>,
) -> ClaimIncidence<'a, F>
where
    F: Copy,
{
    let points = claims.iter().map(|(point, _)| *point).collect();
    let mut incidence_claims = Vec::new();
    for (point_idx, (_, openings)) in claims.iter().enumerate() {
        incidence_claims.extend(openings.openings.iter().enumerate().map(
            |(poly_idx, &claimed_eval)| IncidenceClaim {
                point_idx,
                poly_idx,
                claimed_eval,
            },
        ));
    }
    ClaimIncidence {
        points,
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

/// One public quotient row: all claims at one opening point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicOpeningRow {
    point_idx: usize,
    claim_indices: Vec<usize>,
}

impl PublicOpeningRow {
    /// Opening-point index used by every claim in this row.
    pub fn point_idx(&self) -> usize {
        self.point_idx
    }

    /// Flattened claim indices combined into this row, in poly-index order.
    pub fn claim_indices(&self) -> &[usize] {
        &self.claim_indices
    }
}

/// Commitment-side routing: which committed-polynomial bundle holds each claim's
/// witness columns in the ring-switch `t`-segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitmentRouting {
    claim_to_commitment_group: Vec<usize>,
    claim_poly_in_commitment_group: Vec<usize>,
    num_polys_per_commitment_group: Vec<usize>,
}

impl CommitmentRouting {
    /// Copy routing from incidence where each opening point has its own
    /// commitment group (point index equals group index).
    pub fn copy_incidence(incidence: &ClaimIncidenceSummary) -> Result<Self, AkitaError> {
        let num_claims = incidence.num_claims();
        Self {
            claim_to_commitment_group: incidence.claim_to_point().to_vec(),
            claim_poly_in_commitment_group: incidence.claim_poly_indices().to_vec(),
            num_polys_per_commitment_group: incidence.num_polys_per_point().to_vec(),
        }
        .check(num_claims)
    }

    /// Build routing for recursive multipoint: one shared commitment opened at
    /// many points (all claims map to group 0).
    pub fn from_recursive_multipoint(num_claims: usize) -> Result<Self, AkitaError> {
        Self {
            claim_to_commitment_group: vec![0; num_claims],
            claim_poly_in_commitment_group: vec![0; num_claims],
            num_polys_per_commitment_group: vec![1],
        }
        .check(num_claims)
    }

    /// Validate commitment routing tables.
    pub fn check(self, num_claims: usize) -> Result<Self, AkitaError> {
        if self.claim_to_commitment_group.len() != num_claims
            || self.claim_poly_in_commitment_group.len() != num_claims
        {
            return Err(AkitaError::InvalidInput(
                "commitment routing lengths do not match claim count".to_string(),
            ));
        }
        if self.num_polys_per_commitment_group.is_empty() {
            return Err(AkitaError::InvalidInput(
                "commitment routing requires at least one group".to_string(),
            ));
        }
        for (claim_idx, &group_idx) in self.claim_to_commitment_group.iter().enumerate() {
            if group_idx >= self.num_polys_per_commitment_group.len() {
                return Err(AkitaError::InvalidInput(format!(
                    "commitment routing group index {group_idx} out of range for claim {claim_idx}"
                )));
            }
            let poly_idx = self.claim_poly_in_commitment_group[claim_idx];
            if poly_idx >= self.num_polys_per_commitment_group[group_idx] {
                return Err(AkitaError::InvalidInput(format!(
                    "commitment routing poly index {poly_idx} out of range for claim {claim_idx}"
                )));
            }
        }
        Ok(self)
    }

    /// Validate that commitment routing is the same as opening-point incidence.
    ///
    /// This is the only relation-routing shape supported by the current
    /// ring-switch row-evaluation layout.
    pub fn check_matches_incidence(
        &self,
        incidence: &ClaimIncidenceSummary,
    ) -> Result<(), AkitaError> {
        if self.claim_to_commitment_group.as_slice() != incidence.claim_to_point()
            || self.claim_poly_in_commitment_group.as_slice() != incidence.claim_poly_indices()
            || self.num_polys_per_commitment_group.as_slice() != incidence.num_polys_per_point()
        {
            return Err(AkitaError::InvalidInput(
                "split opening/commitment routing is not supported by ring relation layout"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Flattened claim index to committed-bundle index.
    pub fn claim_to_commitment_group(&self) -> &[usize] {
        &self.claim_to_commitment_group
    }

    /// Polynomial index within the claim's committed bundle.
    pub fn claim_poly_in_commitment_group(&self) -> &[usize] {
        &self.claim_poly_in_commitment_group
    }

    /// Number of polynomials bundled in each commitment group.
    pub fn num_polys_per_commitment_group(&self) -> &[usize] {
        &self.num_polys_per_commitment_group
    }
}

/// Derived routing and count data for a normalized incidence graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimIncidenceSummary {
    num_vars: usize,
    claim_to_point: Vec<usize>,
    claim_poly_indices: Vec<usize>,
    num_polys_per_point: Vec<usize>,
    public_rows: Vec<PublicOpeningRow>,
}

impl ClaimIncidenceSummary {
    /// Validate that routing and count tables are internally consistent.
    ///
    /// # Errors
    ///
    /// Returns an error if any routing/count vector has the wrong length,
    /// routes outside the declared point shape, references a missing
    /// polynomial slot, or disagrees with the derived claim counts.
    pub fn check(&self) -> Result<(), AkitaError> {
        let num_points = self.num_points();
        let num_claims = self.num_claims();
        if num_points == 0 || num_claims == 0 {
            return Err(AkitaError::InvalidProof);
        }
        if self.claim_poly_indices.len() != num_claims
            || self.num_polys_per_point.len() != num_points
            || self.public_rows.len() != num_points
            || self.num_polys_per_point.contains(&0)
        {
            return Err(AkitaError::InvalidProof);
        }

        let mut point_claim_counts = vec![0usize; num_points];
        let mut point_poly_sets = vec![BTreeSet::new(); num_points];
        for claim_idx in 0..num_claims {
            let point_idx = self.claim_to_point[claim_idx];
            if point_idx >= num_points {
                return Err(AkitaError::InvalidProof);
            }
            let poly_idx = self.claim_poly_indices[claim_idx];
            if poly_idx >= self.num_polys_per_point[point_idx] {
                return Err(AkitaError::InvalidProof);
            }
            point_claim_counts[point_idx] = point_claim_counts[point_idx]
                .checked_add(1)
                .ok_or(AkitaError::InvalidProof)?;
            if !point_poly_sets[point_idx].insert(poly_idx) {
                return Err(AkitaError::InvalidProof);
            }
        }
        if point_claim_counts != self.num_polys_per_point {
            return Err(AkitaError::InvalidProof);
        }

        let mut row_claims = BTreeSet::new();
        for (row_idx, row) in self.public_rows.iter().enumerate() {
            if row.point_idx != row_idx
                || row.point_idx >= num_points
                || row.claim_indices.len() != self.num_polys_per_point[row.point_idx]
            {
                return Err(AkitaError::InvalidProof);
            }
            for &claim_idx in &row.claim_indices {
                if claim_idx >= num_claims
                    || self.claim_to_point[claim_idx] != row.point_idx
                    || !row_claims.insert(claim_idx)
                {
                    return Err(AkitaError::InvalidProof);
                }
            }
        }
        if row_claims.len() != num_claims {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }

    /// Build an incidence summary from per-point polynomial counts.
    ///
    /// `num_polys_per_point[p]` is the number of polynomials opened at point
    /// `p`, all bundled into the single commitment cited by that point.
    /// Claims are emitted in (point, poly) lexicographic order.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_polys_per_point` is empty, contains a zero, or
    /// the total claim count overflows `usize`.
    pub fn from_point_polys(
        num_vars: usize,
        num_polys_per_point: Vec<usize>,
    ) -> Result<Self, AkitaError> {
        if num_polys_per_point.is_empty() {
            return Err(AkitaError::InvalidInput(
                "claim incidence requires at least one opening point".to_string(),
            ));
        }
        if let Some(point_idx) = num_polys_per_point.iter().position(|&count| count == 0) {
            return Err(AkitaError::InvalidInput(format!(
                "claim incidence point {point_idx} must bundle at least one polynomial"
            )));
        }
        let num_claims = num_polys_per_point.iter().try_fold(0usize, |acc, &count| {
            acc.checked_add(count).ok_or_else(|| {
                AkitaError::InvalidInput("claim incidence claim count overflow".to_string())
            })
        })?;

        let mut claim_to_point = Vec::with_capacity(num_claims);
        let mut claim_poly_indices = Vec::with_capacity(num_claims);
        let mut public_rows = Vec::with_capacity(num_polys_per_point.len());
        for (point_idx, &polys_at_point) in num_polys_per_point.iter().enumerate() {
            let mut row_claim_indices = Vec::with_capacity(polys_at_point);
            for poly_idx in 0..polys_at_point {
                let claim_idx = claim_to_point.len();
                claim_to_point.push(point_idx);
                claim_poly_indices.push(poly_idx);
                row_claim_indices.push(claim_idx);
            }
            public_rows.push(PublicOpeningRow {
                point_idx,
                claim_indices: row_claim_indices,
            });
        }

        Ok(Self {
            num_vars,
            claim_to_point,
            claim_poly_indices,
            num_polys_per_point,
            public_rows,
        })
    }

    /// Build an incidence summary for one commitment opened at one point.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_polys` is zero.
    pub fn same_point(num_vars: usize, num_polys: usize) -> Result<Self, AkitaError> {
        Self::from_point_polys(num_vars, vec![num_polys])
    }

    /// Build a synthetic incidence from aggregate counts.
    ///
    /// Used by schedule-table and setup-envelope enumeration when only the
    /// aggregate root shape limits are known. Claims are distributed
    /// round-robin so every requested point is exercised.
    ///
    /// # Errors
    ///
    /// Returns an error if any count is zero, points exceed claims, or counts
    /// overflow.
    pub fn from_counts(
        num_vars: usize,
        num_claims: usize,
        num_points: usize,
    ) -> Result<Self, AkitaError> {
        if num_claims == 0 || num_points == 0 {
            return Err(AkitaError::InvalidInput(
                "claim incidence counts must be nonzero".to_string(),
            ));
        }
        if num_points > num_claims {
            return Err(AkitaError::InvalidInput(format!(
                "claim incidence has {num_points} points but only {num_claims} claims"
            )));
        }
        let mut num_polys_per_point = vec![0usize; num_points];
        for claim_idx in 0..num_claims {
            num_polys_per_point[claim_idx % num_points] += 1;
        }
        Self::from_point_polys(num_vars, num_polys_per_point)
    }

    /// Number of variables in every opening point.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Number of distinct opening points.
    pub fn num_points(&self) -> usize {
        self.num_polys_per_point.len()
    }

    /// Number of individual claimed openings.
    pub fn num_claims(&self) -> usize {
        self.claim_to_point.len()
    }

    /// Number of public quotient rows (one per opening point).
    pub fn num_public_rows(&self) -> usize {
        self.public_rows.len()
    }

    /// Opening-point index for each flattened claim.
    pub fn claim_to_point(&self) -> &[usize] {
        &self.claim_to_point
    }

    /// Polynomial index within the point's commitment for each flattened claim.
    pub fn claim_poly_indices(&self) -> &[usize] {
        &self.claim_poly_indices
    }

    /// Number of polynomials bundled at each opening point.
    pub fn num_polys_per_point(&self) -> &[usize] {
        &self.num_polys_per_point
    }

    /// Public-row records, one per opening point.
    pub fn public_rows(&self) -> &[PublicOpeningRow] {
        &self.public_rows
    }

    /// Total number of polynomials addressed by the incidence summary.
    ///
    /// Identical to `num_claims()` under the one-commitment-per-point
    /// invariant; both names are retained for call-site clarity.
    pub fn num_polynomials(&self) -> usize {
        self.num_claims()
    }
}

impl<'a, F> ClaimIncidence<'a, F> {
    /// Validate the incidence graph and derive its flattened routing summary.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph is empty, exceeds supplied capacities, has
    /// inconsistent point dimensions, references invalid point/poly indices,
    /// contains duplicate claim edges, or contains unused points.
    pub fn validate(
        &self,
        limits: ClaimIncidenceLimits,
    ) -> Result<ClaimIncidenceSummary, AkitaError> {
        if self.points.is_empty() {
            return Err(AkitaError::InvalidInput(
                "claim incidence requires at least one opening point".to_string(),
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

        let mut claim_to_point = Vec::with_capacity(self.claims.len());
        let mut claim_poly_indices = Vec::with_capacity(self.claims.len());
        let mut public_rows = (0..self.points.len())
            .map(|point_idx| PublicOpeningRow {
                point_idx,
                claim_indices: Vec::new(),
            })
            .collect::<Vec<_>>();
        let mut num_polys_per_point = vec![0usize; self.points.len()];
        let mut seen_edges = BTreeSet::new();
        let mut max_poly_idx_per_point = vec![0usize; self.points.len()];

        for (claim_idx, claim) in self.claims.iter().enumerate() {
            if claim.point_idx >= self.points.len() {
                return Err(AkitaError::InvalidInput(format!(
                    "claim incidence point index {} out of range",
                    claim.point_idx
                )));
            }
            if !seen_edges.insert((claim.point_idx, claim.poly_idx)) {
                return Err(AkitaError::InvalidInput(
                    "claim incidence contains duplicate point/poly claim".to_string(),
                ));
            }
            claim_to_point.push(claim.point_idx);
            claim_poly_indices.push(claim.poly_idx);
            public_rows[claim.point_idx].claim_indices.push(claim_idx);
            num_polys_per_point[claim.point_idx] = num_polys_per_point[claim.point_idx]
                .checked_add(1)
                .ok_or_else(|| {
                    AkitaError::InvalidInput("claim incidence point count overflow".to_string())
                })?;
            if claim.poly_idx >= max_poly_idx_per_point[claim.point_idx] {
                max_poly_idx_per_point[claim.point_idx] = claim.poly_idx + 1;
            }
        }

        if let Some(point_idx) = num_polys_per_point.iter().position(|&count| count == 0) {
            return Err(AkitaError::InvalidInput(format!(
                "claim incidence point {point_idx} is unused",
            )));
        }
        // The poly indices at each point must be a dense `0..count` range so
        // the point's commitment bundle is fully accounted for.
        for (point_idx, &count) in num_polys_per_point.iter().enumerate() {
            if max_poly_idx_per_point[point_idx] != count {
                return Err(AkitaError::InvalidInput(format!(
                    "claim incidence point {point_idx} polynomials must be a dense 0..k range"
                )));
            }
        }

        Ok(ClaimIncidenceSummary {
            num_vars,
            claim_to_point,
            claim_poly_indices,
            num_polys_per_point,
            public_rows,
        })
    }
}

/// Absorb normalized incidence shape and routing into the transcript.
///
/// This is a migration bridge, not proof serialization: verifier and prover
/// both derive incidence from public claim inputs. Once public claim
/// absorption canonicalizes and binds the same routing unambiguously, this
/// separate shape append should be removed.
pub fn append_claim_incidence_shape_to_transcript<F, T>(
    summary: &ClaimIncidenceSummary,
    transcript: &mut T,
) -> Result<(), AkitaError>
where
    F: akita_field::FieldCore + akita_field::CanonicalField,
    T: Transcript<F>,
{
    summary.check()?;

    transcript.append_serde(ABSORB_BATCH_SHAPE, &summary.num_vars());
    transcript.append_serde(ABSORB_BATCH_SHAPE, &summary.num_points());
    transcript.append_serde(ABSORB_BATCH_SHAPE, &summary.num_claims());
    for count in summary.num_polys_per_point() {
        transcript.append_serde(ABSORB_BATCH_SHAPE, count);
    }
    for claim_idx in 0..summary.num_claims() {
        transcript.append_serde(ABSORB_BATCH_SHAPE, &summary.claim_to_point()[claim_idx]);
        transcript.append_serde(ABSORB_BATCH_SHAPE, &summary.claim_poly_indices()[claim_idx]);
    }
    Ok(())
}

/// Sample row-local public-row batching coefficients.
///
/// Singleton public rows (points with one polynomial) use coefficient one.
/// Non-singleton rows sample one extension-field coefficient per claim in
/// row order.
///
/// # Errors
///
/// Returns an invalid-input error if the incidence summary is internally
/// inconsistent.
pub fn sample_public_row_coefficients<F, L, T>(
    summary: &ClaimIncidenceSummary,
    transcript: &mut T,
) -> Result<Vec<L>, AkitaError>
where
    F: FieldCore + CanonicalField,
    L: ExtField<F>,
    T: Transcript<F>,
{
    let mut coeffs = vec![L::zero(); summary.num_claims()];
    let mut seen = vec![false; summary.num_claims()];
    for row in summary.public_rows() {
        if row.point_idx() >= summary.num_points() || row.claim_indices().is_empty() {
            return Err(AkitaError::InvalidInput(
                "public-row incidence contains an invalid row".to_string(),
            ));
        }
        for &claim_idx in row.claim_indices() {
            if claim_idx >= summary.num_claims()
                || summary.claim_to_point()[claim_idx] != row.point_idx()
                || seen[claim_idx]
            {
                return Err(AkitaError::InvalidInput(
                    "public-row incidence term is inconsistent".to_string(),
                ));
            }
            seen[claim_idx] = true;
            coeffs[claim_idx] = if row.claim_indices().len() == 1 {
                L::one()
            } else {
                sample_ext_challenge::<F, L, T>(transcript, CHALLENGE_EVAL_BATCH)
            };
        }
    }
    if seen.iter().any(|&present| !present) {
        return Err(AkitaError::InvalidInput(
            "public-row incidence does not cover every claim".to_string(),
        ));
    }
    Ok(coeffs)
}

/// Sum batched public opening claims under per-claim row coefficients.
pub fn batched_eval_target_from_incidence<E, L>(
    incidence: &ClaimIncidenceSummary,
    row_coefficients: &[L],
    openings: &[E],
) -> Result<L, AkitaError>
where
    E: FieldCore,
    L: ExtField<E> + MulBase<E> + FieldCore,
{
    if row_coefficients.len() != incidence.num_claims() {
        return Err(AkitaError::InvalidSize {
            expected: incidence.num_claims(),
            actual: row_coefficients.len(),
        });
    }
    if openings.len() != incidence.num_claims() {
        return Err(AkitaError::InvalidSize {
            expected: incidence.num_claims(),
            actual: openings.len(),
        });
    }
    incidence
        .public_rows()
        .iter()
        .flat_map(|row| row.claim_indices())
        .try_fold(L::zero(), |acc, &claim_idx| {
            let coefficient = *row_coefficients
                .get(claim_idx)
                .ok_or(AkitaError::InvalidProof)?;
            let opening = *openings.get(claim_idx).ok_or(AkitaError::InvalidProof)?;
            Ok(acc + coefficient.mul_base(opening))
        })
}

#[cfg(test)]
mod tests {
    use super::super::CommittedOpenings;
    use super::*;
    use akita_field::{Fp64, FpExt2, NegOneNr};
    use akita_transcript::{labels, AkitaTranscript, Transcript};

    type TranscriptField = Fp64<4294967197>;

    fn generous_limits() -> ClaimIncidenceLimits {
        ClaimIncidenceLimits {
            max_num_vars: 8,
            max_num_points: 8,
            max_num_claims: 16,
        }
    }

    fn incidence_shape_challenge(summary: &ClaimIncidenceSummary) -> TranscriptField {
        let mut transcript = AkitaTranscript::<TranscriptField>::new(labels::DOMAIN_AKITA_PROTOCOL);
        append_claim_incidence_shape_to_transcript(summary, &mut transcript).unwrap();
        transcript.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION)
    }

    #[test]
    fn incidence_summary_tracks_routing_counts() {
        let p0 = [1u64, 2];
        let p1 = [3u64, 4];
        let incidence = ClaimIncidence {
            points: vec![&p0, &p1],
            claims: vec![
                IncidenceClaim {
                    point_idx: 0,
                    poly_idx: 0,
                    claimed_eval: 10u64,
                },
                IncidenceClaim {
                    point_idx: 0,
                    poly_idx: 1,
                    claimed_eval: 11u64,
                },
                IncidenceClaim {
                    point_idx: 1,
                    poly_idx: 0,
                    claimed_eval: 12u64,
                },
            ],
        };

        let summary = incidence
            .validate(generous_limits())
            .expect("valid incidence");

        assert_eq!(summary.num_vars(), 2);
        assert_eq!(summary.num_points(), 2);
        assert_eq!(summary.num_claims(), 3);
        assert_eq!(summary.claim_to_point(), &[0, 0, 1]);
        assert_eq!(summary.claim_poly_indices(), &[0, 1, 0]);
        assert_eq!(summary.num_polys_per_point(), &[2, 1]);
        assert_eq!(summary.num_polynomials(), 3);
    }

    #[test]
    fn one_commitment_can_be_opened_at_many_points() {
        let p0 = [1u64];
        let p1 = [2u64];
        let incidence = ClaimIncidence {
            points: vec![&p0, &p1],
            claims: vec![
                IncidenceClaim {
                    point_idx: 0,
                    poly_idx: 0,
                    claimed_eval: 3u64,
                },
                IncidenceClaim {
                    point_idx: 1,
                    poly_idx: 0,
                    claimed_eval: 4u64,
                },
            ],
        };

        let summary = incidence
            .validate(generous_limits())
            .expect("valid incidence");

        assert_eq!(summary.num_polys_per_point(), &[1, 1]);
        assert_eq!(summary.claim_to_point(), &[0, 1]);
        assert_eq!(summary.num_claims(), 2);
    }

    #[test]
    fn root_commitment_routing_matches_incidence_axis() {
        let summary =
            ClaimIncidenceSummary::from_point_polys(8, vec![2, 1]).expect("valid incidence");
        let routing = CommitmentRouting::copy_incidence(&summary).expect("root routing");

        assert_eq!(routing.claim_to_commitment_group(), &[0, 0, 1]);
        assert_eq!(routing.claim_poly_in_commitment_group(), &[0, 1, 0]);
        assert_eq!(routing.num_polys_per_commitment_group(), &[2, 1]);
        routing
            .check_matches_incidence(&summary)
            .expect("root routing is same-axis");
    }

    #[test]
    fn split_commitment_routing_is_rejected_by_same_axis_contract() {
        let summary =
            ClaimIncidenceSummary::from_point_polys(8, vec![1, 1]).expect("valid incidence");
        let routing =
            CommitmentRouting::from_recursive_multipoint(summary.num_claims()).expect("routing");

        let err = routing
            .check_matches_incidence(&summary)
            .expect_err("split-axis routing must be rejected");
        assert!(
            format!("{err:?}").contains("split opening/commitment routing is not supported"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn public_rows_group_same_point_polys() {
        let p0 = [1u64];
        let incidence = ClaimIncidence {
            points: vec![&p0],
            claims: vec![
                IncidenceClaim {
                    point_idx: 0,
                    poly_idx: 0,
                    claimed_eval: 3u64,
                },
                IncidenceClaim {
                    point_idx: 0,
                    poly_idx: 1,
                    claimed_eval: 4u64,
                },
            ],
        };

        let summary = incidence
            .validate(generous_limits())
            .expect("valid same-point incidence");

        assert_eq!(summary.num_public_rows(), 1);
        assert_eq!(summary.public_rows()[0].point_idx(), 0);
        assert_eq!(summary.public_rows()[0].claim_indices(), &[0, 1]);
    }

    #[test]
    fn public_rows_split_across_points() {
        let p0 = [1u64];
        let p1 = [2u64];
        let incidence = ClaimIncidence {
            points: vec![&p0, &p1],
            claims: vec![
                IncidenceClaim {
                    point_idx: 0,
                    poly_idx: 0,
                    claimed_eval: 3u64,
                },
                IncidenceClaim {
                    point_idx: 0,
                    poly_idx: 1,
                    claimed_eval: 4u64,
                },
                IncidenceClaim {
                    point_idx: 1,
                    poly_idx: 0,
                    claimed_eval: 5u64,
                },
            ],
        };

        let summary = incidence
            .validate(generous_limits())
            .expect("valid mixed incidence");

        assert_eq!(summary.num_public_rows(), 2);
        assert_eq!(summary.public_rows()[0].point_idx(), 0);
        assert_eq!(summary.public_rows()[0].claim_indices(), &[0, 1]);
        assert_eq!(summary.public_rows()[1].point_idx(), 1);
        assert_eq!(summary.public_rows()[1].claim_indices(), &[2]);
    }

    #[test]
    fn row_local_coefficients_sample_only_for_non_singleton_rows() {
        let summary =
            ClaimIncidenceSummary::from_point_polys(1, vec![2, 1]).expect("valid incidence");
        let mut transcript = AkitaTranscript::<TranscriptField>::new(labels::DOMAIN_AKITA_PROTOCOL);
        append_claim_incidence_shape_to_transcript(&summary, &mut transcript).unwrap();

        let coeffs = sample_public_row_coefficients::<TranscriptField, TranscriptField, _>(
            &summary,
            &mut transcript,
        )
        .expect("row coefficients should sample");

        assert_eq!(coeffs.len(), 3);
        assert_eq!(coeffs[2], TranscriptField::one());
        assert_ne!(coeffs[0], TranscriptField::zero());
        assert_ne!(coeffs[1], TranscriptField::zero());
    }

    #[test]
    fn incidence_transcript_binds_claim_order() {
        let p0 = [1u64];
        let p1 = [2u64];
        let forward = ClaimIncidence {
            points: vec![&p0, &p1],
            claims: vec![
                IncidenceClaim {
                    point_idx: 0,
                    poly_idx: 0,
                    claimed_eval: 3u64,
                },
                IncidenceClaim {
                    point_idx: 1,
                    poly_idx: 0,
                    claimed_eval: 4u64,
                },
            ],
        };
        let reversed = ClaimIncidence {
            claims: forward.claims.iter().copied().rev().collect(),
            ..forward.clone()
        };

        let forward_summary = forward
            .validate(generous_limits())
            .expect("valid incidence");
        let reversed_summary = reversed
            .validate(generous_limits())
            .expect("valid incidence");

        // Reversing the claim order produces a different `claim_to_point`
        // routing in the summary, which must be transcript-bound so adversarial
        // reordering is caught.
        assert_ne!(
            incidence_shape_challenge(&forward_summary),
            incidence_shape_challenge(&reversed_summary)
        );
    }

    #[test]
    fn incidence_transcript_binds_same_point_poly_idx_order() {
        // Two claims at the same opening point with poly indices `{0, 1}`.
        // The forward ordering is `[poly_idx=0, poly_idx=1]`, the swapped
        // ordering is `[poly_idx=1, poly_idx=0]`. Both produce the same
        // `claim_to_point` routing (`[0, 0]`), but `claim_poly_indices`
        // differs (`[0, 1]` vs `[1, 0]`); the transcript must bind the
        // per-claim poly index so adversarial reordering inside a point is
        // caught.
        let p0 = [1u64];
        let forward = ClaimIncidence {
            points: vec![&p0],
            claims: vec![
                IncidenceClaim {
                    point_idx: 0,
                    poly_idx: 0,
                    claimed_eval: 3u64,
                },
                IncidenceClaim {
                    point_idx: 0,
                    poly_idx: 1,
                    claimed_eval: 4u64,
                },
            ],
        };
        let swapped = ClaimIncidence {
            points: vec![&p0],
            claims: vec![
                IncidenceClaim {
                    point_idx: 0,
                    poly_idx: 1,
                    claimed_eval: 4u64,
                },
                IncidenceClaim {
                    point_idx: 0,
                    poly_idx: 0,
                    claimed_eval: 3u64,
                },
            ],
        };

        let forward_summary = forward
            .validate(generous_limits())
            .expect("valid forward incidence");
        let swapped_summary = swapped
            .validate(generous_limits())
            .expect("valid swapped incidence");

        assert_eq!(
            forward_summary.claim_to_point(),
            swapped_summary.claim_to_point(),
            "swapping poly order at one point must not change `claim_to_point`"
        );
        assert_ne!(
            forward_summary.claim_poly_indices(),
            swapped_summary.claim_poly_indices(),
            "swapping poly order must change `claim_poly_indices`"
        );
        assert_ne!(
            incidence_shape_challenge(&forward_summary),
            incidence_shape_challenge(&swapped_summary)
        );
    }

    #[test]
    fn extension_row_coefficients_sample_for_non_singleton_rows() {
        type E = FpExt2<TranscriptField, NegOneNr>;
        let summary = ClaimIncidenceSummary::same_point(1, 2).expect("valid same-point incidence");
        let mut transcript = AkitaTranscript::<TranscriptField>::new(labels::DOMAIN_AKITA_PROTOCOL);

        let coeffs =
            sample_public_row_coefficients::<TranscriptField, E, _>(&summary, &mut transcript)
                .expect("extension row coefficients should sample");

        assert_eq!(coeffs.len(), 2);
        assert_ne!(coeffs[0], E::zero());
        assert_ne!(coeffs[1], E::zero());
    }

    #[test]
    fn verifier_claims_normalize_to_incidence_graph() {
        let p0 = [1u64, 2];
        let p1 = [3u64, 4];
        let c0 = 10usize;
        let c1 = 12usize;
        let openings0 = [20u64, 21];
        let openings1 = [23u64, 24, 25];
        let claims = vec![
            (
                &p0[..],
                CommittedOpenings {
                    commitment: &c0,
                    openings: &openings0,
                },
            ),
            (
                &p1[..],
                CommittedOpenings {
                    commitment: &c1,
                    openings: &openings1,
                },
            ),
        ];

        let incidence = verifier_claims_to_incidence(&claims);

        assert_eq!(incidence.points, vec![&p0[..], &p1[..]]);
        assert_eq!(
            incidence.claims,
            vec![
                IncidenceClaim {
                    point_idx: 0,
                    poly_idx: 0,
                    claimed_eval: 20,
                },
                IncidenceClaim {
                    point_idx: 0,
                    poly_idx: 1,
                    claimed_eval: 21,
                },
                IncidenceClaim {
                    point_idx: 1,
                    poly_idx: 0,
                    claimed_eval: 23,
                },
                IncidenceClaim {
                    point_idx: 1,
                    poly_idx: 1,
                    claimed_eval: 24,
                },
                IncidenceClaim {
                    point_idx: 1,
                    poly_idx: 2,
                    claimed_eval: 25,
                },
            ]
        );

        let summary = incidence
            .validate(generous_limits())
            .expect("normalized verifier claims should validate");
        assert_eq!(summary.claim_to_point(), &[0, 0, 1, 1, 1]);
        assert_eq!(summary.claim_poly_indices(), &[0, 1, 0, 1, 2]);
        assert_eq!(summary.num_polys_per_point(), &[2, 3]);
    }

    #[test]
    fn incidence_validation_rejects_malformed_shapes() {
        let p0 = [1u64];
        let p1 = [2u64, 3];

        let mismatched_points = ClaimIncidence {
            points: vec![&p0, &p1],
            claims: vec![IncidenceClaim {
                point_idx: 0,
                poly_idx: 0,
                claimed_eval: 5u64,
            }],
        };
        assert!(matches!(
            mismatched_points.validate(generous_limits()),
            Err(AkitaError::InvalidInput(_))
        ));

        let sparse_poly = ClaimIncidence {
            points: vec![&p0],
            claims: vec![IncidenceClaim {
                point_idx: 0,
                poly_idx: 1,
                claimed_eval: 5u64,
            }],
        };
        assert!(matches!(
            sparse_poly.validate(generous_limits()),
            Err(AkitaError::InvalidInput(_))
        ));

        let duplicate_edge = ClaimIncidence {
            points: vec![&p0],
            claims: vec![
                IncidenceClaim {
                    point_idx: 0,
                    poly_idx: 0,
                    claimed_eval: 5u64,
                },
                IncidenceClaim {
                    point_idx: 0,
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
