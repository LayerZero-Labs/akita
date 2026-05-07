//! Shared batching and root-opening helper types.

use crate::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, AkitaRootBatchSummary,
    AppendToTranscript, BasisMode, BlockOrder, LevelParams, RingCommitment, RingOpeningPoint,
};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_transcript::labels::{ABSORB_BATCH_SHAPE, ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS};
use akita_transcript::Transcript;

/// A mapping that records which opening point belongs to which polynomial, and vice versa.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointToPolynomialMap {
    /// Opening-point index used by this claim.
    pub point_idx: usize,
    /// Polynomial index used by the prover.
    pub polynomial_idx: usize,
}

/// Public opening statement shared by the prover and verifier.
///
/// This type stores only the shared part of an opening claim: opening points,
/// commitments, claimed scalar evaluations, and the map from each point to the
/// opened polynomial. Prover-only witnesses such as polynomials and commitment
/// hints live outside this statement.
#[derive(Debug, Clone)]
pub struct OpeningStatement<'a, F: FieldCore, C> {
    opening_points: Vec<&'a [F]>,
    commitments: Vec<C>,
    claims: Vec<F>,
    map: Vec<Vec<PointToPolynomialMap>>,
}

impl<'a, F, C> OpeningStatement<'a, F, C>
where
    F: FieldCore,
{
    /// Build a public opening statement.
    ///
    /// # Errors
    ///
    /// Returns an error if the public shape is empty or inconsistent.
    pub fn new(
        opening_points: Vec<&'a [F]>,
        commitments: Vec<C>,
        claims: Vec<F>,
        map: Vec<Vec<PointToPolynomialMap>>,
    ) -> Result<Self, AkitaError> {
        if opening_points.is_empty() {
            return Err(AkitaError::InvalidInput(
                "opening statement requires at least one opening point".to_string(),
            ));
        }
        if commitments.is_empty() {
            return Err(AkitaError::InvalidInput(
                "opening statement requires at least one commitment".to_string(),
            ));
        }
        if claims.is_empty() || map.is_empty() {
            return Err(AkitaError::InvalidInput(
                "opening statement requires at least one opening claim".to_string(),
            ));
        }
        if commitments.len() != map.len() {
            return Err(AkitaError::InvalidInput(
                "opening statement commitments and map groups must have the same length"
                    .to_string(),
            ));
        }
        let mapped_claims = map.iter().try_fold(0usize, |acc, group| {
            if group.is_empty() {
                return Err(AkitaError::InvalidInput(
                    "opening statement map groups must be nonempty".to_string(),
                ));
            }
            acc.checked_add(group.len()).ok_or_else(|| {
                AkitaError::InvalidInput("opening statement claim count overflow".to_string())
            })
        })?;
        if claims.len() != mapped_claims {
            return Err(AkitaError::InvalidInput(
                "opening statement claims and map entries must have the same length".to_string(),
            ));
        }
        let num_vars = opening_points[0].len();
        if opening_points.iter().any(|point| point.len() != num_vars) {
            return Err(AkitaError::InvalidInput(
                "opening statement requires all opening points to have the same length".to_string(),
            ));
        }
        for entry in map.iter().flatten() {
            if entry.point_idx >= opening_points.len() {
                return Err(AkitaError::InvalidInput(
                    "opening statement map point index out of range".to_string(),
                ));
            }
        }
        let mut point_group_sizes = vec![0usize; opening_points.len()];
        for group in &map {
            point_group_sizes[group[0].point_idx] += 1;
        }
        if point_group_sizes.contains(&0) {
            return Err(AkitaError::InvalidInput(
                "opening statement requires every opening point to have a commitment group"
                    .to_string(),
            ));
        }
        for (expected_poly_idx, entry) in map.iter().flatten().enumerate() {
            if entry.polynomial_idx != expected_poly_idx {
                return Err(AkitaError::InvalidInput(
                    "opening statement map must use contiguous flat polynomial indices".to_string(),
                ));
            }
        }

        Ok(Self {
            opening_points,
            commitments,
            claims,
            map,
        })
    }

    /// Check that this statement matches setup capacity limits.
    ///
    /// # Errors
    ///
    /// Returns an error when the statement exceeds the supplied setup limits.
    pub fn matches_setup(
        &self,
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<(), AkitaError> {
        if self.num_vars() > max_num_vars {
            return Err(AkitaError::InvalidInput(format!(
                "opening statement has {} variables but setup supports at most {}",
                self.num_vars(),
                max_num_vars
            )));
        }
        if self.num_points() > max_num_points {
            return Err(AkitaError::InvalidInput(format!(
                "opening statement has {} opening points but setup supports at most {}",
                self.num_points(),
                max_num_points
            )));
        }
        if self.num_claims() > max_num_batched_polys {
            return Err(AkitaError::InvalidInput(format!(
                "opening statement has {} claims but setup supports at most {}",
                self.num_claims(),
                max_num_batched_polys
            )));
        }
        Ok(())
    }

    /// Opening points in caller order.
    pub fn opening_points(&self) -> &[&'a [F]] {
        &self.opening_points
    }

    /// Commitments in commitment-group order.
    pub fn commitments(&self) -> &[C] {
        &self.commitments
    }

    /// Claimed scalar evaluations in map order.
    pub fn claims(&self) -> &[F] {
        &self.claims
    }

    /// Point-to-polynomial map grouped by commitment.
    pub fn map(&self) -> &[Vec<PointToPolynomialMap>] {
        &self.map
    }

    /// Number of variables in every opening point.
    pub fn num_vars(&self) -> usize {
        self.opening_points[0].len()
    }

    /// Number of distinct opening points.
    pub fn num_points(&self) -> usize {
        self.opening_points.len()
    }

    /// Number of commitment groups.
    pub fn num_commitments(&self) -> usize {
        self.commitments.len()
    }

    /// Number of flattened opening claims.
    pub fn num_claims(&self) -> usize {
        self.claims.len()
    }

    /// Number of flat prover polynomials required by the current map.
    pub fn num_polynomials(&self) -> usize {
        self.map.iter().map(Vec::len).sum()
    }

    /// Number of commitment groups attached to each opening point.
    pub fn point_group_sizes(&self) -> Vec<usize> {
        let mut point_group_sizes = vec![0usize; self.opening_points.len()];
        for group in &self.map {
            point_group_sizes[group[0].point_idx] += 1;
        }
        point_group_sizes
    }

    /// Number of scalar claims inside each commitment group.
    pub fn claim_group_sizes(&self) -> Vec<usize> {
        self.map.iter().map(Vec::len).collect()
    }

    /// Opening-point index for each flattened scalar claim.
    pub fn claim_to_point(&self) -> Vec<usize> {
        self.map
            .iter()
            .flat_map(|group| group.iter().map(|entry| entry.point_idx))
            .collect()
    }

    /// Derive the root batch summary used for schedule lookup.
    ///
    /// # Errors
    ///
    /// Returns an error if the current claim groups are malformed.
    pub fn batch_summary(&self) -> Result<AkitaRootBatchSummary, AkitaError> {
        AkitaRootBatchSummary::from_claim_group_sizes(&self.claim_group_sizes(), self.num_points())
    }
}

/// Root-level opening point prepared for ring-level replay.
#[derive(Debug, Clone)]
pub struct PreparedRootOpeningPoint<F: FieldCore, const D: usize> {
    /// Opening point padded to the root verifier's target variable count.
    pub padded_point: Vec<F>,
    /// Ring-level outer opening point.
    pub ring_opening_point: RingOpeningPoint<F>,
    /// Inner ring-slot reduction.
    pub inner_reduction: CyclotomicRing<F, D>,
}

/// Flatten commitment rows in group order.
pub fn flatten_batched_commitment_rows<F: FieldCore, const D: usize>(
    commitments: &[RingCommitment<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    commitments
        .iter()
        .flat_map(|commitment| commitment.u.iter().copied())
        .collect()
}

/// Absorb batched commitments into the transcript in group order.
pub fn append_batched_commitments_to_transcript<F, T, const D: usize>(
    commitments: &[RingCommitment<F, D>],
    transcript: &mut T,
) where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    for commitment in commitments {
        commitment.append_to_transcript(ABSORB_COMMITMENT, transcript);
    }
}

/// Sum claim-group sizes with overflow checking.
///
/// # Errors
///
/// Returns an error if the total claim count overflows `usize`.
pub fn checked_total_claims(group_sizes: &[usize], label: &str) -> Result<usize, AkitaError> {
    group_sizes.iter().try_fold(0usize, |acc, &group_size| {
        acc.checked_add(group_size)
            .ok_or_else(|| AkitaError::InvalidInput(format!("{label} total claim count overflow")))
    })
}

/// Absorb the multipoint batch shape into the transcript.
pub fn append_batch_shape_to_transcript<F, T>(
    point_group_sizes: &[usize],
    claim_group_sizes: &[usize],
    transcript: &mut T,
) where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    transcript.append_serde(ABSORB_BATCH_SHAPE, &point_group_sizes.len());
    for group_count in point_group_sizes {
        transcript.append_serde(ABSORB_BATCH_SHAPE, group_count);
    }
    for claim_count in claim_group_sizes {
        transcript.append_serde(ABSORB_BATCH_SHAPE, claim_count);
    }
}

/// Sum point-group sizes with non-empty and overflow checks.
///
/// # Errors
///
/// Returns an error if any point group is empty or the total group count
/// overflows `usize`.
pub fn checked_total_groups(point_group_sizes: &[usize], label: &str) -> Result<usize, AkitaError> {
    if point_group_sizes.is_empty() || point_group_sizes.contains(&0) {
        return Err(AkitaError::InvalidInput(format!(
            "{label} requires nonempty point group sizes"
        )));
    }
    point_group_sizes.iter().try_fold(0usize, |acc, &size| {
        acc.checked_add(size)
            .ok_or_else(|| AkitaError::InvalidInput(format!("{label} group count overflow")))
    })
}

/// Prepare a root opening point for ring-level verification/proving.
///
/// # Errors
///
/// Returns an error if the target variable count overflows, the opening point
/// is too long, or the field-to-ring reduction rejects the point dimensions.
pub fn prepare_root_opening_point<F, const D: usize>(
    opening_point: &[F],
    basis: BasisMode,
    lp: &LevelParams,
    alpha_bits: usize,
) -> Result<PreparedRootOpeningPoint<F, D>, AkitaError>
where
    F: FieldCore,
{
    let target_num_vars = lp
        .m_vars
        .checked_add(lp.r_vars)
        .and_then(|n| n.checked_add(alpha_bits))
        .ok_or_else(|| AkitaError::InvalidSetup("opening point length overflow".to_string()))?;
    if opening_point.len() > target_num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: target_num_vars,
            actual: opening_point.len(),
        });
    }
    let mut padded_point = opening_point.to_vec();
    padded_point.resize(target_num_vars, F::zero());
    let inner_point = &padded_point[..alpha_bits];
    let outer_point = &padded_point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field::<F>(
        outer_point,
        lp.r_vars,
        lp.m_vars,
        basis,
        BlockOrder::RowMajor,
    )?;
    let inner_reduction = reduce_inner_opening_to_ring_element::<F, D>(inner_point, basis)?;
    Ok(PreparedRootOpeningPoint {
        padded_point,
        ring_opening_point,
        inner_reduction,
    })
}

/// Append a prepared root opening point to the transcript.
pub fn append_prepared_root_opening_point<F, T, const D: usize>(
    prepared_point: &PreparedRootOpeningPoint<F, D>,
    transcript: &mut T,
) where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    for pt in &prepared_point.padded_point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
    }
}
