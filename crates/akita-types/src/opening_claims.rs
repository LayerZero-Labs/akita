//! Public opening claims and layout-only opening geometry.

use crate::config::SetupContributionMode;
use crate::descriptor_bytes::{push_usize, push_usize_vec};
use crate::instance_descriptor::DescriptorDigest;
use crate::proof::scheme::OpeningPoints;
use crate::proof::setup::AkitaSetupSeed;
use crate::AppendToTranscript;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_transcript::labels::{
    ABSORB_BATCH_SHAPE, ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, CHALLENGE_EVAL_BATCH,
};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};
use std::collections::BTreeSet;

/// Tiered presets cannot open multi-group root batches yet.
pub const GROUPED_ROOT_TIERED_UNSUPPORTED: &str =
    "tiered multi-group root batching is not supported; see specs/multi-group-batching.md";

/// Recursive setup contribution cannot open multi-group root batches yet.
pub const GROUPED_ROOT_RECURSIVE_SETUP_UNSUPPORTED: &str =
    "recursive setup contribution with multiple commitment groups is not supported; see specs/multi-group-batching.md";

/// Dense polynomials cannot open multi-group root batches yet.
pub const GROUPED_ROOT_DENSE_UNSUPPORTED: &str =
    "dense polynomial multi-group root batching is not supported; see specs/multi-group-batching.md";

/// Grouped root prove/verify is not implemented yet.
pub const GROUPED_ROOT_UNSUPPORTED: &str =
    "multi-group root batching is not supported yet; see specs/multi-group-batching.md";

/// Return the grouped-root rejection message, if the layout should be rejected.
///
/// `includes_dense_polynomial` is `Some(true)` when the prover knows the batch
/// includes a dense polynomial; verifier callers pass `None` and skip the check.
pub fn should_reject_grouped_root(
    layout: &OpeningClaimsLayout,
    tiered_commitment: bool,
    setup_contribution_mode: SetupContributionMode,
    includes_dense_polynomial: Option<bool>,
) -> Option<&'static str> {
    if layout.num_groups() <= 1 {
        return None;
    }
    if tiered_commitment {
        return Some(GROUPED_ROOT_TIERED_UNSUPPORTED);
    }
    if setup_contribution_mode == SetupContributionMode::Recursive {
        return Some(GROUPED_ROOT_RECURSIVE_SETUP_UNSUPPORTED);
    }
    if includes_dense_polynomial == Some(true) {
        return Some(GROUPED_ROOT_DENSE_UNSUPPORTED);
    }
    Some(GROUPED_ROOT_UNSUPPORTED)
}

/// Ordered coordinate selection into an opening batch's shared point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointVariableSelection {
    indices: Vec<usize>,
}

impl PointVariableSelection {
    /// Build an ordered, duplicate-free selection into a point of length `point_len`.
    pub fn new(indices: Vec<usize>, point_len: usize) -> Result<Self, AkitaError> {
        let selection = Self { indices };
        selection.check(point_len)?;
        Ok(selection)
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

    fn is_prefix(&self) -> bool {
        self.indices.iter().copied().eq(0..self.indices.len())
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

/// Per-group opening geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PolynomialGroupLayout {
    num_vars: usize,
    num_polynomials: usize,
}

impl PolynomialGroupLayout {
    /// Build a per-group layout. Runtime callers should pair this with `validate`.
    pub const fn new(num_vars: usize, num_polynomials: usize) -> Self {
        Self {
            num_vars,
            num_polynomials,
        }
    }

    /// Scalar default: one polynomial at `num_vars`.
    pub const fn singleton(num_vars: usize) -> Self {
        Self::new(num_vars, 1)
    }

    /// Active variable count for this group.
    pub const fn num_vars(self) -> usize {
        self.num_vars
    }

    /// Number of polynomials in this group.
    pub const fn num_polynomials(self) -> usize {
        self.num_polynomials
    }

    /// Validate that the group carries at least one polynomial.
    pub fn validate(self) -> Result<(), AkitaError> {
        if self.num_polynomials == 0 {
            return Err(AkitaError::InvalidSetup(
                "opening group layouts must be nonempty".to_string(),
            ));
        }
        Ok(())
    }
}

/// Batch structure without point values, evaluations, commitments, or routing values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpeningClaimsLayout {
    groups: Vec<PolynomialGroupLayout>,
}

impl OpeningClaimsLayout {
    /// Build a one-group layout for `num_total_polynomials` at `num_vars`.
    pub fn new(num_vars: usize, num_total_polynomials: usize) -> Result<Self, AkitaError> {
        Self::from_groups(vec![PolynomialGroupLayout::new(
            num_vars,
            num_total_polynomials,
        )])
    }

    /// Build a layout from group sizes, all sharing the same active variable count.
    pub fn from_group_sizes(
        num_vars: usize,
        polynomials_per_group: &[usize],
    ) -> Result<Self, AkitaError> {
        Self::from_groups(
            polynomials_per_group
                .iter()
                .map(|&num_polynomials| PolynomialGroupLayout::new(num_vars, num_polynomials))
                .collect(),
        )
    }

    /// Build a validated layout from per-group geometry.
    pub fn from_groups(groups: Vec<PolynomialGroupLayout>) -> Result<Self, AkitaError> {
        let layout = Self { groups };
        layout.check()?;
        Ok(layout)
    }

    /// Worst-case setup envelope as a one-group layout.
    pub fn from_setup_seed(seed: &AkitaSetupSeed) -> Result<Self, AkitaError> {
        Self::new(seed.max_num_vars, seed.max_num_batched_polys)
    }

    /// Validate layout count consistency.
    pub fn check(&self) -> Result<(), AkitaError> {
        if self.groups.is_empty() || self.checked_num_total_polynomials()? == 0 {
            return Err(AkitaError::InvalidProof);
        }
        for group in &self.groups {
            group.validate()?;
        }
        Ok(())
    }

    /// Maximum active variable count across groups.
    pub fn max_num_vars(&self) -> usize {
        self.groups
            .iter()
            .map(|group| group.num_vars())
            .max()
            .unwrap_or(0)
    }

    /// Commitment groups in transcript order.
    pub fn groups(&self) -> &[PolynomialGroupLayout] {
        &self.groups
    }

    /// Number of commitment groups represented by the batch.
    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }

    /// Total polynomials opened across all groups.
    pub fn num_total_polynomials(&self) -> usize {
        self.groups
            .iter()
            .map(|group| group.num_polynomials())
            .sum()
    }

    fn checked_num_total_polynomials(&self) -> Result<usize, AkitaError> {
        self.groups.iter().try_fold(0usize, |acc, group| {
            acc.checked_add(group.num_polynomials())
                .ok_or(AkitaError::InvalidProof)
        })
    }

    /// Number of polynomials in each group.
    pub fn group_sizes(&self) -> Vec<usize> {
        self.groups
            .iter()
            .map(|group| group.num_polynomials())
            .collect()
    }

    /// Borrow one group layout by index.
    pub fn group_layout(&self, g: usize) -> Result<&PolynomialGroupLayout, AkitaError> {
        self.groups.get(g).ok_or(AkitaError::InvalidProof)
    }

    /// Digest layout-only opening geometry.
    pub fn opening_batch_digest(&self) -> DescriptorDigest {
        let mut bytes = Vec::new();
        push_usize(&mut bytes, self.max_num_vars());
        push_usize(&mut bytes, self.num_total_polynomials());
        push_usize(&mut bytes, self.num_groups());
        for group in &self.groups {
            push_usize(&mut bytes, group.num_polynomials());
            let prefix_indices = (0..group.num_vars()).collect::<Vec<_>>();
            push_usize_vec(&mut bytes, &prefix_indices);
        }
        blake2b_256(&bytes)
    }

    /// Absorb normalized batch-shape fields into the transcript.
    pub fn append_batch_shape_to_transcript<F, T>(
        &self,
        transcript: &mut T,
    ) -> Result<(), AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
    {
        self.check()?;

        transcript.append_serde(ABSORB_BATCH_SHAPE, &self.max_num_vars());
        transcript.append_serde(ABSORB_BATCH_SHAPE, &self.num_total_polynomials());
        transcript.append_serde(ABSORB_BATCH_SHAPE, &self.num_groups());
        for group in self.groups() {
            transcript.append_serde(ABSORB_BATCH_SHAPE, &group.num_polynomials());
            transcript.append_serde(ABSORB_BATCH_SHAPE, &group.num_vars());
            for index in 0..group.num_vars() {
                transcript.append_serde(ABSORB_BATCH_SHAPE, &index);
            }
        }
        Ok(())
    }

    /// Sum batched public opening claims under per-slot gamma coefficients.
    pub fn batched_eval_target<E>(
        &self,
        row_coefficients: &[E],
        openings: &[E],
    ) -> Result<E, AkitaError>
    where
        E: FieldCore,
    {
        if row_coefficients.len() != self.num_total_polynomials() {
            return Err(AkitaError::InvalidSize {
                expected: self.num_total_polynomials(),
                actual: row_coefficients.len(),
            });
        }
        if openings.len() != self.num_total_polynomials() {
            return Err(AkitaError::InvalidSize {
                expected: self.num_total_polynomials(),
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
}

/// Public claims and commitment payload for one polynomial group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolynomialGroupClaims<'a, F, C = ()> {
    point_vars: PointVariableSelection,
    evaluations: Vec<F>,
    commitment: C,
    _marker: std::marker::PhantomData<&'a F>,
}

impl<'a, F, C> PolynomialGroupClaims<'a, F, C> {
    /// Build one group of public claims.
    pub fn new(
        point_vars: PointVariableSelection,
        evaluations: Vec<F>,
        commitment: C,
    ) -> Result<Self, AkitaError> {
        if evaluations.is_empty() {
            return Err(AkitaError::InvalidInput(
                "opening claim groups must be nonempty".to_string(),
            ));
        }
        Ok(Self {
            point_vars,
            evaluations,
            commitment,
            _marker: std::marker::PhantomData,
        })
    }

    /// Ordered point-variable selection for this group.
    pub fn point_vars(&self) -> &PointVariableSelection {
        &self.point_vars
    }

    /// Claimed evaluations, one per committed polynomial.
    pub fn evaluations(&self) -> &[F] {
        &self.evaluations
    }

    /// Group commitment.
    pub fn commitment(&self) -> &C {
        &self.commitment
    }

    /// Number of evaluations in this group.
    pub fn num_evaluations(&self) -> usize {
        self.evaluations.len()
    }
}

/// Public opening claims: one shared point and polynomial groups in transcript order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpeningClaims<'a, F: Clone, C = ()> {
    point: OpeningPoints<'a, F>,
    groups: Vec<PolynomialGroupClaims<'a, F, C>>,
}

impl<'a, F: Clone, C> OpeningClaims<'a, F, C> {
    /// Build public claims from a shared point and ordered groups.
    pub fn from_groups(
        point: impl Into<OpeningPoints<'a, F>>,
        groups: Vec<PolynomialGroupClaims<'a, F, C>>,
    ) -> Result<Self, AkitaError> {
        let claims = Self {
            point: point.into(),
            groups,
        };
        claims.check()?;
        Ok(claims)
    }

    /// Validate internal routing/count consistency.
    pub fn check(&self) -> Result<(), AkitaError> {
        if self.groups.is_empty() || self.checked_num_total_polynomials()? == 0 {
            return Err(AkitaError::InvalidProof);
        }
        let point_len = self.point.as_ref().len();
        let mut max_group_vars = 0usize;
        for group in &self.groups {
            if group.evaluations.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
            group.point_vars.check(point_len)?;
            if !group.point_vars.is_prefix() {
                return Err(AkitaError::InvalidInput(
                    "custom point-variable routing is not supported by instance descriptors"
                        .to_string(),
                ));
            }
            max_group_vars = max_group_vars.max(group.point_vars.num_vars());
        }
        if max_group_vars != point_len {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }

    /// Validate consistency plus public capacity against the setup envelope.
    pub fn validate(&self, seed: &AkitaSetupSeed) -> Result<(), AkitaError> {
        self.check()?;
        if self.num_vars() > seed.max_num_vars {
            return Err(AkitaError::InvalidPointDimension {
                expected: seed.max_num_vars,
                actual: self.num_vars(),
            });
        }
        let num_polynomials = self.checked_num_total_polynomials()?;
        if num_polynomials > seed.max_num_batched_polys {
            return Err(AkitaError::InvalidSize {
                expected: seed.max_num_batched_polys,
                actual: num_polynomials,
            });
        }
        Ok(())
    }

    /// Shared opening point.
    pub fn point(&self) -> &[F] {
        self.point.as_ref()
    }

    /// Number of coordinates in the shared opening point.
    pub fn num_vars(&self) -> usize {
        self.point.as_ref().len()
    }

    /// Number of polynomial groups.
    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }

    /// Total polynomials opened across all groups.
    pub fn num_total_polynomials(&self) -> usize {
        self.groups
            .iter()
            .map(|group| group.evaluations.len())
            .sum()
    }

    fn checked_num_total_polynomials(&self) -> Result<usize, AkitaError> {
        self.groups.iter().try_fold(0usize, |acc, group| {
            acc.checked_add(group.evaluations.len())
                .ok_or(AkitaError::InvalidProof)
        })
    }

    /// Number of polynomials/evaluations in each group.
    pub fn group_sizes(&self) -> Vec<usize> {
        self.groups
            .iter()
            .map(PolynomialGroupClaims::num_evaluations)
            .collect()
    }

    /// Borrow one group's evaluations.
    pub fn group_evaluations(&self, g: usize) -> Result<&[F], AkitaError> {
        self.groups
            .get(g)
            .map(PolynomialGroupClaims::evaluations)
            .ok_or(AkitaError::InvalidProof)
    }

    /// Borrow one group's point-variable selection.
    pub fn group_point_vars(&self, g: usize) -> Result<&PointVariableSelection, AkitaError> {
        self.groups
            .get(g)
            .map(PolynomialGroupClaims::point_vars)
            .ok_or(AkitaError::InvalidProof)
    }

    /// Borrow one group's commitment.
    pub fn group_commitment(&self, g: usize) -> Result<&C, AkitaError> {
        self.groups
            .get(g)
            .map(PolynomialGroupClaims::commitment)
            .ok_or(AkitaError::InvalidProof)
    }

    /// Commitment groups in transcript order.
    pub fn groups(&self) -> &[PolynomialGroupClaims<'a, F, C>] {
        &self.groups
    }

    /// Structural view for setup, planner, and config code.
    pub fn layout(&self) -> Result<OpeningClaimsLayout, AkitaError> {
        self.check()?;
        OpeningClaimsLayout::from_groups(
            self.groups
                .iter()
                .map(|group| {
                    PolynomialGroupLayout::new(group.point_vars.num_vars(), group.evaluations.len())
                })
                .collect(),
        )
    }

    /// Layout digest for this claim set.
    pub fn opening_batch_digest(&self) -> Result<DescriptorDigest, AkitaError> {
        Ok(self.layout()?.opening_batch_digest())
    }
}

impl<'a, F: Clone, C> OpeningClaims<'a, F, C> {
    /// Claimed openings flattened in canonical claim order.
    pub fn flat_evaluations(&self) -> Vec<F> {
        self.groups
            .iter()
            .flat_map(|group| group.evaluations.iter().cloned())
            .collect()
    }
}

impl<'a, F: FieldCore> OpeningClaims<'a, F, ()> {
    /// Commitment-less, full-point claims used by internal extension-opening replay.
    pub fn with_padded_point(
        point: &[F],
        num_vars: usize,
        num_total_polynomials: usize,
    ) -> Result<Self, AkitaError> {
        if point.len() > num_vars {
            return Err(AkitaError::InvalidPointDimension {
                expected: num_vars,
                actual: point.len(),
            });
        }
        let mut padded_point = point.to_vec();
        padded_point.resize(num_vars, F::zero());
        let group = PolynomialGroupClaims::new(
            PointVariableSelection::prefix(num_vars, num_vars)?,
            vec![F::zero(); num_total_polynomials],
            (),
        )?;
        Self::from_groups(padded_point, vec![group])
    }
}

impl<'a, F: Clone, C> OpeningClaims<'a, F, C> {
    /// Return the only commitment when the current single-group path applies.
    pub fn single_group_commitment(&self) -> Option<&C> {
        self.groups
            .first()
            .filter(|_| self.groups.len() == 1)
            .map(PolynomialGroupClaims::commitment)
    }
}

impl<'a, F: Clone, C> OpeningClaims<'a, F, C> {
    /// Absorb the normalized batch shape, commitments, and shared point.
    pub fn append_to_transcript<TranscriptF, T>(&self, transcript: &mut T) -> Result<(), AkitaError>
    where
        TranscriptF: FieldCore + CanonicalField,
        F: ExtField<TranscriptF>,
        C: AppendToTranscript<TranscriptF>,
        T: Transcript<TranscriptF>,
    {
        self.layout()?
            .append_batch_shape_to_transcript::<TranscriptF, T>(transcript)?;
        for group in &self.groups {
            group
                .commitment
                .append_to_transcript(ABSORB_COMMITMENT, transcript);
        }
        for coord in self.point() {
            append_ext_field::<TranscriptF, F, T>(transcript, ABSORB_EVALUATION_CLAIMS, coord);
        }
        Ok(())
    }
}

/// Sample gamma coefficients for the one public row.
pub fn sample_public_row_coefficients<F, L, T>(
    layout: &OpeningClaimsLayout,
    transcript: &mut T,
) -> Result<Vec<L>, AkitaError>
where
    F: FieldCore + CanonicalField,
    L: ExtField<F>,
    T: Transcript<F>,
{
    layout.check()?;
    if layout.num_total_polynomials() == 1 {
        return Ok(vec![L::one()]);
    }
    Ok((0..layout.num_total_polynomials())
        .map(|_| sample_ext_challenge::<F, L, T>(transcript, CHALLENGE_EVAL_BATCH))
        .collect())
}

fn blake2b_256(bytes: &[u8]) -> DescriptorDigest {
    type Blake2b256 = Blake2b<U32>;
    let digest = Blake2b256::digest(bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128OffsetA7F7;

    type F = Prime128OffsetA7F7;

    fn prefix_claims(num_vars: usize, evals: usize) -> OpeningClaims<'static, F, ()> {
        let point_vars = PointVariableSelection::prefix(num_vars, num_vars).expect("prefix");
        let group =
            PolynomialGroupClaims::new(point_vars, vec![F::zero(); evals], ()).expect("group");
        OpeningClaims::from_groups(vec![F::zero(); num_vars], vec![group]).expect("claims")
    }

    #[test]
    fn check_rejects_duplicate_point_indices() {
        let err = PointVariableSelection::new(vec![0, 0], 2).expect_err("duplicate index");
        assert!(matches!(err, AkitaError::InvalidProof));
    }

    #[test]
    fn check_rejects_out_of_range_point_indices() {
        let err = PointVariableSelection::new(vec![2], 2).expect_err("out of range");
        assert!(matches!(err, AkitaError::InvalidProof));
    }

    #[test]
    fn check_rejects_non_prefix_routing() {
        let point_vars = PointVariableSelection::new(vec![1, 0], 2).expect("custom routing");
        let group = PolynomialGroupClaims::new(point_vars, vec![F::zero()], ()).expect("group");
        let err = OpeningClaims::from_groups(vec![F::zero(), F::zero()], vec![group])
            .expect_err("non-prefix routing");
        assert!(matches!(err, AkitaError::InvalidInput(_)));
    }

    #[test]
    fn check_rejects_short_point_relative_to_max_group_vars() {
        let claims = prefix_claims(3, 1);
        let short_point = vec![F::zero(); 2];
        let err = OpeningClaims::from_groups(short_point, claims.groups().to_vec())
            .expect_err("short point");
        assert!(matches!(err, AkitaError::InvalidProof));
    }

    #[test]
    fn layout_digest_matches_layout_view() {
        let claims = prefix_claims(4, 2);
        assert_eq!(
            claims.opening_batch_digest().expect("claims digest"),
            claims.layout().expect("layout").opening_batch_digest()
        );
    }

    #[test]
    fn with_padded_point_rejects_longer_point() {
        let err = OpeningClaims::with_padded_point(&[F::zero(); 3], 2, 1)
            .expect_err("point longer than num_vars");
        assert!(matches!(err, AkitaError::InvalidPointDimension { .. }));
    }

    #[test]
    fn should_reject_grouped_root_returns_canonical_messages() {
        let layout = OpeningClaimsLayout::from_group_sizes(4, &[1, 1]).expect("layout");
        assert_eq!(
            should_reject_grouped_root(&layout, false, SetupContributionMode::Direct, None),
            Some(GROUPED_ROOT_UNSUPPORTED)
        );
        assert_eq!(
            should_reject_grouped_root(
                &OpeningClaimsLayout::new(4, 1).expect("single group"),
                true,
                SetupContributionMode::Direct,
                None,
            ),
            None
        );
    }
}
