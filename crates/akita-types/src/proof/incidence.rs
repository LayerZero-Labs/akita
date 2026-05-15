//! Normalized point/claim incidence for batched openings.
//!
//! Hachi commits to **one** logical bundle of polynomials with a single
//! commitment. A batched opening proof opens that bundle at one or more
//! opening points, with each point referencing some subset of the committed
//! polynomials by global index.

use akita_field::AkitaError;
use akita_transcript::labels::ABSORB_BATCH_SHAPE;
use akita_transcript::Transcript;

/// Normalized routing summary for a batched opening.
///
/// Every claim opens polynomial `claim_poly_indices[claim_idx]` (an index into
/// the single committed bundle of `num_polys` polynomials) at opening point
/// `claim_to_point[claim_idx]`. Claims are stored in opening-point order
/// (point 0 first, then point 1, etc.).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimIncidenceSummary {
    /// Number of variables in every opening point.
    pub num_vars: usize,
    /// Number of distinct opening points.
    pub num_points: usize,
    /// Total number of polynomials in the single committed bundle.
    pub num_polys: usize,
    /// Number of individual claimed openings (`= Σ point_claim_counts`).
    pub num_claims: usize,
    /// Opening-point index for each flattened claim (length `num_claims`).
    pub claim_to_point: Vec<usize>,
    /// Polynomial index inside the single committed bundle for each flattened
    /// claim (length `num_claims`, each value in `[0, num_polys)`).
    pub claim_poly_indices: Vec<usize>,
    /// Number of claims evaluated at each opening point (`l_i`, length
    /// `num_points`).
    pub point_claim_counts: Vec<usize>,
}

impl ClaimIncidenceSummary {
    /// Build an incidence summary from per-point polynomial-index lists.
    ///
    /// `per_point_poly_indices[p]` carries the global polynomial indices opened
    /// at opening point `p`, in claim order. Every index must be in
    /// `[0, num_polys)`. Each per-point list must be non-empty.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_polys` is zero, the per-point list is empty,
    /// any per-point list is empty, any index is out of range, or counts
    /// overflow.
    pub fn from_per_point_polys(
        num_vars: usize,
        num_polys: usize,
        per_point_poly_indices: &[&[usize]],
    ) -> Result<Self, AkitaError> {
        if num_polys == 0 {
            return Err(AkitaError::InvalidInput(
                "claim incidence requires at least one committed polynomial".to_string(),
            ));
        }
        if per_point_poly_indices.is_empty() {
            return Err(AkitaError::InvalidInput(
                "claim incidence requires at least one opening point".to_string(),
            ));
        }
        let num_points = per_point_poly_indices.len();
        let mut point_claim_counts = Vec::with_capacity(num_points);
        let mut num_claims = 0usize;
        for (point_idx, point_polys) in per_point_poly_indices.iter().enumerate() {
            if point_polys.is_empty() {
                return Err(AkitaError::InvalidInput(format!(
                    "claim incidence point {point_idx} must have at least one claim"
                )));
            }
            for &poly_idx in *point_polys {
                if poly_idx >= num_polys {
                    return Err(AkitaError::InvalidInput(format!(
                        "claim incidence claim references poly {poly_idx} but only {num_polys} polynomials are committed"
                    )));
                }
            }
            point_claim_counts.push(point_polys.len());
            num_claims = num_claims.checked_add(point_polys.len()).ok_or_else(|| {
                AkitaError::InvalidInput("claim incidence claim count overflow".to_string())
            })?;
        }

        let mut claim_to_point = Vec::with_capacity(num_claims);
        let mut claim_poly_indices = Vec::with_capacity(num_claims);
        for (point_idx, point_polys) in per_point_poly_indices.iter().enumerate() {
            for &poly_idx in *point_polys {
                claim_to_point.push(point_idx);
                claim_poly_indices.push(poly_idx);
            }
        }

        Ok(Self {
            num_vars,
            num_points,
            num_polys,
            num_claims,
            claim_to_point,
            claim_poly_indices,
            point_claim_counts,
        })
    }

    /// Build an incidence summary for one opening point opening polynomials
    /// `0..num_polys` of the bundle in order.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_polys` is zero.
    pub fn same_point(num_vars: usize, num_polys: usize) -> Result<Self, AkitaError> {
        let polys: Vec<usize> = (0..num_polys).collect();
        Self::from_per_point_polys(num_vars, num_polys, &[polys.as_slice()])
    }

    /// Build a synthetic valid incidence from aggregate counts.
    ///
    /// Used by schedule-table and setup-envelope enumeration when only the
    /// shape limits are known. Claims are assigned round-robin across points
    /// so every point is touched and every polynomial is referenced at least
    /// once.
    ///
    /// # Errors
    ///
    /// Returns an error if any count is zero, `num_polys > num_claims`, or
    /// `num_points > num_claims`.
    pub fn from_counts(
        num_vars: usize,
        num_claims: usize,
        num_polys: usize,
        num_points: usize,
    ) -> Result<Self, AkitaError> {
        if num_claims == 0 || num_polys == 0 || num_points == 0 {
            return Err(AkitaError::InvalidInput(
                "claim incidence counts must be nonzero".to_string(),
            ));
        }
        if num_polys > num_claims {
            return Err(AkitaError::InvalidInput(format!(
                "claim incidence has {num_polys} polys but only {num_claims} claims"
            )));
        }
        if num_points > num_claims {
            return Err(AkitaError::InvalidInput(format!(
                "claim incidence has {num_points} points but only {num_claims} claims"
            )));
        }

        let mut per_point: Vec<Vec<usize>> = vec![Vec::new(); num_points];
        for claim_idx in 0..num_claims {
            let point_idx = claim_idx % num_points;
            let poly_idx = claim_idx % num_polys;
            per_point[point_idx].push(poly_idx);
        }
        // Make sure every point has at least one claim. Round-robin guarantees
        // this when num_points <= num_claims, which we checked above.
        let per_point_refs: Vec<&[usize]> = per_point.iter().map(Vec::as_slice).collect();
        Self::from_per_point_polys(num_vars, num_polys, &per_point_refs)
    }
}

/// Absorb normalized incidence shape and routing into the transcript.
///
/// Binds the public claim layout to the transcript so prover and verifier
/// agree on point/claim/polynomial routing without explicitly re-deriving it
/// from the rest of the absorbed inputs.
pub fn append_claim_incidence_shape_to_transcript<F, T>(
    summary: &ClaimIncidenceSummary,
    transcript: &mut T,
) where
    F: akita_field::FieldCore + akita_field::CanonicalField,
    T: Transcript<F>,
{
    transcript.append_serde(ABSORB_BATCH_SHAPE, &summary.num_vars);
    transcript.append_serde(ABSORB_BATCH_SHAPE, &summary.num_points);
    transcript.append_serde(ABSORB_BATCH_SHAPE, &summary.num_polys);
    transcript.append_serde(ABSORB_BATCH_SHAPE, &summary.num_claims);
    for count in &summary.point_claim_counts {
        transcript.append_serde(ABSORB_BATCH_SHAPE, count);
    }
    for claim_idx in 0..summary.num_claims {
        transcript.append_serde(ABSORB_BATCH_SHAPE, &summary.claim_to_point[claim_idx]);
        transcript.append_serde(ABSORB_BATCH_SHAPE, &summary.claim_poly_indices[claim_idx]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Fp64;
    use akita_transcript::{labels, Blake2bTranscript, Transcript};

    type TranscriptField = Fp64<4294967197>;

    fn shape_challenge(summary: &ClaimIncidenceSummary) -> TranscriptField {
        let mut transcript =
            Blake2bTranscript::<TranscriptField>::new(labels::DOMAIN_AKITA_PROTOCOL);
        append_claim_incidence_shape_to_transcript(summary, &mut transcript);
        transcript.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION)
    }

    #[test]
    fn from_per_point_polys_tracks_routing_counts() {
        // 4 committed polynomials. Point 0 opens polys [0, 1]; point 1 opens
        // polys [0, 2, 3].
        let p0 = [0usize, 1];
        let p1 = [0usize, 2, 3];
        let summary = ClaimIncidenceSummary::from_per_point_polys(2, 4, &[&p0, &p1])
            .expect("valid incidence");

        assert_eq!(summary.num_vars, 2);
        assert_eq!(summary.num_points, 2);
        assert_eq!(summary.num_polys, 4);
        assert_eq!(summary.num_claims, 5);
        assert_eq!(summary.claim_to_point, vec![0, 0, 1, 1, 1]);
        assert_eq!(summary.claim_poly_indices, vec![0, 1, 0, 2, 3]);
        assert_eq!(summary.point_claim_counts, vec![2, 3]);
    }

    #[test]
    fn same_point_assigns_polys_in_order() {
        let summary = ClaimIncidenceSummary::same_point(3, 4).expect("valid same-point incidence");
        assert_eq!(summary.num_points, 1);
        assert_eq!(summary.num_polys, 4);
        assert_eq!(summary.num_claims, 4);
        assert_eq!(summary.claim_to_point, vec![0, 0, 0, 0]);
        assert_eq!(summary.claim_poly_indices, vec![0, 1, 2, 3]);
        assert_eq!(summary.point_claim_counts, vec![4]);
    }

    #[test]
    fn one_poly_can_be_opened_at_many_points() {
        let p0 = [0usize];
        let p1 = [0usize];
        let summary =
            ClaimIncidenceSummary::from_per_point_polys(1, 1, &[&p0, &p1]).expect("valid");
        assert_eq!(summary.num_polys, 1);
        assert_eq!(summary.num_claims, 2);
        assert_eq!(summary.claim_to_point, vec![0, 1]);
        assert_eq!(summary.claim_poly_indices, vec![0, 0]);
    }

    #[test]
    fn rejects_out_of_range_poly_index() {
        let p0 = [3usize];
        assert!(matches!(
            ClaimIncidenceSummary::from_per_point_polys(1, 2, &[&p0]),
            Err(AkitaError::InvalidInput(_))
        ));
    }

    #[test]
    fn rejects_empty_point() {
        let p0: [usize; 0] = [];
        let p1 = [0usize];
        assert!(matches!(
            ClaimIncidenceSummary::from_per_point_polys(1, 2, &[&p0, &p1]),
            Err(AkitaError::InvalidInput(_))
        ));
    }

    #[test]
    fn rejects_zero_polys() {
        assert!(matches!(
            ClaimIncidenceSummary::from_per_point_polys(1, 0, &[&[0usize]]),
            Err(AkitaError::InvalidInput(_))
        ));
    }

    #[test]
    fn from_counts_round_robin_assignment() {
        let summary = ClaimIncidenceSummary::from_counts(4, 5, 3, 2).expect("valid");
        assert_eq!(summary.num_points, 2);
        assert_eq!(summary.num_polys, 3);
        assert_eq!(summary.num_claims, 5);
        // Round robin: claim 0 -> (point 0, poly 0), claim 1 -> (point 1, poly 1),
        //              claim 2 -> (point 0, poly 2), claim 3 -> (point 1, poly 0),
        //              claim 4 -> (point 0, poly 1).
        // After grouping by point (claims sorted by point then arrival order):
        //   point 0: polys [0, 2, 1]
        //   point 1: polys [1, 0]
        assert_eq!(summary.claim_to_point, vec![0, 0, 0, 1, 1]);
        assert_eq!(summary.claim_poly_indices, vec![0, 2, 1, 1, 0]);
        assert_eq!(summary.point_claim_counts, vec![3, 2]);
    }

    #[test]
    fn transcript_shape_binds_routing_order() {
        let p0 = [0usize, 1];
        let p1 = [2usize];
        let forward =
            ClaimIncidenceSummary::from_per_point_polys(1, 3, &[&p0, &p1]).expect("valid");
        // Same shape, swapped point order.
        let reversed =
            ClaimIncidenceSummary::from_per_point_polys(1, 3, &[&p1, &p0]).expect("valid");

        assert_eq!(shape_challenge(&forward), shape_challenge(&forward));
        assert_ne!(shape_challenge(&forward), shape_challenge(&reversed));
    }
}
