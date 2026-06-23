//! Normalized single-point opening batches.
//!
//! # Protocol contract
//!
//! A batched prove/verify call uses exactly **one shared opening point**. Each
//! commitment group chooses an ordered subset of coordinates from that point and
//! carries dense claimed evaluations for the polynomials in that commitment.
//!
//! The current folded-root protocol constructs one full-point commitment group.
//! The type also records the future multi-group shape directly, without the old
//! flattened slot/routing vocabulary.

use super::VerifierClaims;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_transcript::labels::{ABSORB_BATCH_SHAPE, CHALLENGE_EVAL_BATCH};
use akita_transcript::{sample_ext_challenge, Transcript};
use std::collections::BTreeSet;

/// Ordered coordinate selection into an [`OpeningBatch`]'s shared point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointVariableSelection {
    indices: Vec<usize>,
}

impl PointVariableSelection {
    /// Build an ordered, duplicate-free selection into a point of length `point_len`.
    pub fn new(indices: Vec<usize>, point_len: usize) -> Result<Self, AkitaError> {
        let mut seen = BTreeSet::new();
        for &index in &indices {
            if index >= point_len || !seen.insert(index) {
                return Err(AkitaError::InvalidInput(
                    "opening batch point-variable selection is malformed".to_string(),
                ));
            }
        }
        Ok(Self { indices })
    }

    /// Select the first `num_vars` coordinates of the shared point.
    pub fn prefix(num_vars: usize, point_len: usize) -> Result<Self, AkitaError> {
        if num_vars > point_len {
            return Err(AkitaError::InvalidPointDimension {
                expected: point_len,
                actual: num_vars,
            });
        }
        Ok(Self {
            indices: (0..num_vars).collect(),
        })
    }

    /// Selected point-coordinate indices, in evaluation order.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Number of variables selected for this group.
    pub fn num_vars(&self) -> usize {
        self.indices.len()
    }

    fn check(&self, point_len: usize) -> Result<(), AkitaError> {
        let mut seen = BTreeSet::new();
        for &index in &self.indices {
            if index >= point_len || !seen.insert(index) {
                return Err(AkitaError::InvalidProof);
            }
        }
        Ok(())
    }
}

/// One commitment group opened at an ordered subset of the shared point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitmentGroup<F> {
    /// Coordinates of [`OpeningBatch::point`] used by every polynomial in this group.
    pub point_vars: PointVariableSelection,
    /// Claimed evaluations, one per committed polynomial, in commitment order.
    pub claims: Vec<F>,
}

impl<F> CommitmentGroup<F> {
    /// Number of variables used by the polynomials in this group.
    pub fn num_vars(&self) -> usize {
        self.point_vars.num_vars()
    }

    /// Number of claimed polynomial openings in this group.
    pub fn num_claims(&self) -> usize {
        self.claims.len()
    }
}

/// Derived count limits for opening-batch validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpeningBatchLimits {
    /// Maximum supported number of variables in the shared opening point.
    pub max_num_vars: usize,
    /// Maximum supported number of claimed openings.
    pub max_num_claims: usize,
}

/// Shared opening point plus commitment groups opened at ordered point subsets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpeningBatch<F = ()> {
    /// Padded/shared opening point.
    pub point: Vec<F>,
    /// Commitment groups in transcript order.
    pub groups: Vec<CommitmentGroup<F>>,
}

/// Normalize public verifier claims into a one-group full-point batch.
pub fn verifier_claims_to_opening_batch<'a, F, C>(
    claims: &VerifierClaims<'a, F, C>,
) -> OpeningBatch<F>
where
    F: Copy,
{
    let (point, openings) = claims;
    OpeningBatch {
        point: point.to_vec(),
        groups: vec![CommitmentGroup {
            point_vars: PointVariableSelection {
                indices: (0..point.len()).collect(),
            },
            claims: openings.openings.to_vec(),
        }],
    }
}

impl OpeningBatch<()> {
    /// Build a one-group shape for `num_polys` polynomials opened at a point of `num_vars`.
    pub fn same_point(num_vars: usize, num_polys: usize) -> Result<Self, AkitaError> {
        Self::from_commitment_groups(num_vars, &[num_polys])
    }

    /// Build a shape from commitment-group sizes.
    pub fn from_commitment_groups(
        num_vars: usize,
        num_polys_per_commitment_group: &[usize],
    ) -> Result<Self, AkitaError> {
        if num_polys_per_commitment_group.is_empty() {
            return Err(AkitaError::InvalidInput(
                "opening batch requires at least one commitment group".to_string(),
            ));
        }
        let point = vec![(); num_vars];
        let mut groups = Vec::with_capacity(num_polys_per_commitment_group.len());
        for &group_size in num_polys_per_commitment_group {
            if group_size == 0 {
                return Err(AkitaError::InvalidInput(
                    "opening batch commitment groups must be nonempty".to_string(),
                ));
            }
            groups.push(CommitmentGroup {
                point_vars: PointVariableSelection::prefix(num_vars, num_vars)?,
                claims: vec![(); group_size],
            });
        }
        OpeningBatch::new(point, groups)
    }
}

impl<F> OpeningBatch<F> {
    /// Build a validated opening batch from a shared point and commitment groups.
    pub fn new(point: Vec<F>, groups: Vec<CommitmentGroup<F>>) -> Result<Self, AkitaError> {
        let batch = Self { point, groups };
        batch.check()?;
        Ok(batch)
    }

    /// Build one full-point commitment group with explicit claimed evaluations.
    pub fn same_point_with_claims(point: Vec<F>, claims: Vec<F>) -> Result<Self, AkitaError> {
        let point_vars = PointVariableSelection::prefix(point.len(), point.len())?;
        Self::new(point, vec![CommitmentGroup { point_vars, claims }])
    }

    /// Erase field values and retain only the normalized shape.
    pub fn to_shape(&self) -> OpeningBatch<()> {
        OpeningBatch {
            point: vec![(); self.point.len()],
            groups: self
                .groups
                .iter()
                .map(|group| CommitmentGroup {
                    point_vars: group.point_vars.clone(),
                    claims: vec![(); group.claims.len()],
                })
                .collect(),
        }
    }

    /// Validate public limits and return the shape-only summary used by schedules.
    pub fn validate(&self, limits: OpeningBatchLimits) -> Result<OpeningBatch<()>, AkitaError> {
        if self.point.len() > limits.max_num_vars {
            return Err(AkitaError::InvalidPointDimension {
                expected: limits.max_num_vars,
                actual: self.point.len(),
            });
        }
        let num_claims = self.num_claims();
        if num_claims > limits.max_num_claims {
            return Err(AkitaError::InvalidSize {
                expected: limits.max_num_claims,
                actual: num_claims,
            });
        }
        self.check()?;
        Ok(self.to_shape())
    }
}

impl<F> OpeningBatch<F> {
    /// Validate that routing and count tables are internally consistent.
    pub fn check(&self) -> Result<(), AkitaError> {
        if self.groups.is_empty() || self.num_claims() == 0 {
            return Err(AkitaError::InvalidProof);
        }
        for group in &self.groups {
            if group.claims.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
            group.point_vars.check(self.point.len())?;
        }
        Ok(())
    }

    /// Number of variables in the shared padded opening point.
    pub fn num_vars(&self) -> usize {
        self.point.len()
    }

    /// Number of individual claimed openings.
    pub fn num_claims(&self) -> usize {
        self.groups.iter().map(CommitmentGroup::num_claims).sum()
    }

    /// Number of commitment groups represented by the batch.
    pub fn num_commitment_groups(&self) -> usize {
        self.groups.len()
    }

    /// Number of polynomials committed in each commitment group.
    pub fn num_polys_per_commitment_group(&self) -> Vec<usize> {
        self.groups
            .iter()
            .map(CommitmentGroup::num_claims)
            .collect()
    }

    /// Total number of committed polynomials addressed by the batch.
    pub fn num_polynomials(&self) -> usize {
        self.num_claims()
    }
}

/// Absorb normalized opening-batch shape and routing into the transcript.
pub fn append_opening_batch_shape_to_transcript<F, T>(
    batch: &OpeningBatch,
    transcript: &mut T,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    batch.check()?;

    transcript.append_serde(ABSORB_BATCH_SHAPE, &batch.num_vars());
    transcript.append_serde(ABSORB_BATCH_SHAPE, &batch.num_claims());
    transcript.append_serde(ABSORB_BATCH_SHAPE, &batch.num_commitment_groups());
    for group in &batch.groups {
        transcript.append_serde(ABSORB_BATCH_SHAPE, &group.num_claims());
        transcript.append_serde(ABSORB_BATCH_SHAPE, &group.point_vars.num_vars());
        for &index in group.point_vars.indices() {
            transcript.append_serde(ABSORB_BATCH_SHAPE, &index);
        }
    }
    Ok(())
}

/// Sample gamma coefficients for the one public row.
pub fn sample_public_row_coefficients<F, L, T>(
    batch: &OpeningBatch,
    transcript: &mut T,
) -> Result<Vec<L>, AkitaError>
where
    F: FieldCore + CanonicalField,
    L: ExtField<F>,
    T: Transcript<F>,
{
    batch.check()?;
    if batch.num_claims() == 1 {
        return Ok(vec![L::one()]);
    }
    Ok((0..batch.num_claims())
        .map(|_| sample_ext_challenge::<F, L, T>(transcript, CHALLENGE_EVAL_BATCH))
        .collect())
}

/// Sum batched public opening claims under per-slot gamma coefficients.
pub fn batched_eval_target_from_opening_batch<E>(
    batch: &OpeningBatch,
    row_coefficients: &[E],
    openings: &[E],
) -> Result<E, AkitaError>
where
    E: FieldCore,
{
    if row_coefficients.len() != batch.num_claims() {
        return Err(AkitaError::InvalidSize {
            expected: batch.num_claims(),
            actual: row_coefficients.len(),
        });
    }
    if openings.len() != batch.num_claims() {
        return Err(AkitaError::InvalidSize {
            expected: batch.num_claims(),
            actual: openings.len(),
        });
    }
    row_coefficients
        .iter()
        .zip(openings.iter())
        .try_fold(E::zero(), |acc, (&coefficient, &opening)| {
            Ok(acc + coefficient * opening)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{Fp64, FpExt2, NegOneNr};
    use akita_transcript::{labels, AkitaTranscript};

    type TranscriptField = Fp64<4294967197>;

    fn generous_limits() -> OpeningBatchLimits {
        OpeningBatchLimits {
            max_num_vars: 8,
            max_num_claims: 16,
        }
    }

    #[test]
    fn opening_batch_tracks_single_point_group() {
        let batch = OpeningBatch::same_point_with_claims(vec![1u64, 2], vec![10u64, 11])
            .expect("valid batch");

        let summary = batch.validate(generous_limits()).expect("valid shape");

        assert_eq!(summary.num_vars(), 2);
        assert_eq!(summary.num_claims(), 2);
        assert_eq!(summary.num_commitment_groups(), 1);
        assert_eq!(summary.num_polys_per_commitment_group(), vec![2]);
        assert_eq!(summary.groups[0].point_vars.indices(), &[0, 1]);
    }

    #[test]
    fn opening_batch_represents_multi_commitment_groups() {
        let batch = OpeningBatch::from_commitment_groups(3, &[1, 2])
            .expect("multi-group batches have a direct shape");

        assert_eq!(batch.num_commitment_groups(), 2);
        assert_eq!(batch.num_polys_per_commitment_group(), vec![1, 2]);
        assert_eq!(batch.groups[0].num_claims(), 1);
        assert_eq!(batch.groups[1].num_claims(), 2);
    }

    #[test]
    fn opening_batch_single_group_normalizes_to_same_point() {
        let batch = OpeningBatch::from_commitment_groups(3, &[2]).expect("single group");

        assert_eq!(batch.num_commitment_groups(), 1);
        assert_eq!(batch.num_polys_per_commitment_group(), vec![2]);
        assert_eq!(batch.groups[0].num_claims(), 2);
    }

    #[test]
    fn point_variable_selection_preserves_custom_order() {
        let selection = PointVariableSelection::new(vec![2, 0], 3).expect("custom order");
        let batch = OpeningBatch::new(
            vec![1u64, 2, 3],
            vec![CommitmentGroup {
                point_vars: selection,
                claims: vec![7u64],
            }],
        )
        .expect("valid custom point subset");

        assert_eq!(batch.groups[0].num_vars(), 2);
        assert_eq!(batch.groups[0].point_vars.indices(), &[2, 0]);
    }

    #[test]
    fn row_coefficients_batch_all_claims_once() {
        type E = FpExt2<TranscriptField, NegOneNr>;
        let batch = OpeningBatch::same_point(1, 2).expect("valid same-point batch");
        let mut transcript = AkitaTranscript::<TranscriptField>::new(labels::DOMAIN_AKITA_PROTOCOL);

        let coeffs =
            sample_public_row_coefficients::<TranscriptField, E, _>(&batch, &mut transcript)
                .expect("row coefficients should sample");

        assert_eq!(coeffs.len(), 2);
        assert_ne!(coeffs[0], E::zero());
        assert_ne!(coeffs[1], E::zero());
    }

    #[test]
    fn transcript_binds_point_variable_order() {
        let forward = OpeningBatch::new(
            vec![1u64, 2],
            vec![CommitmentGroup {
                point_vars: PointVariableSelection::new(vec![0, 1], 2).expect("forward vars"),
                claims: vec![10u64],
            }],
        )
        .expect("forward batch")
        .to_shape();
        let swapped = OpeningBatch::new(
            vec![1u64, 2],
            vec![CommitmentGroup {
                point_vars: PointVariableSelection::new(vec![1, 0], 2).expect("swapped vars"),
                claims: vec![10u64],
            }],
        )
        .expect("swapped batch")
        .to_shape();
        let mut t1 = AkitaTranscript::<TranscriptField>::new(labels::DOMAIN_AKITA_PROTOCOL);
        let mut t2 = AkitaTranscript::<TranscriptField>::new(labels::DOMAIN_AKITA_PROTOCOL);

        append_opening_batch_shape_to_transcript(&forward, &mut t1).unwrap();
        append_opening_batch_shape_to_transcript(&swapped, &mut t2).unwrap();

        assert_ne!(
            t1.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION),
            t2.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION)
        );
    }
}
