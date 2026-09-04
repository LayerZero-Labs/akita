//! Public opening claims and layout-only opening geometry.

use crate::descriptor_bytes::push_usize;
use crate::instance_descriptor::DescriptorDigest;
#[cfg(test)]
use crate::proof::batch::append_claim_values_to_transcript;
use crate::proof::scheme::OpeningPoints;
use crate::proof::setup::AkitaSetupDescriptor;
use crate::{CommittedGroup, GrindingSite, OpeningScheduleSelection, TranscriptGrinding};
use akita_error::{checked, AkitaError};
use akita_transcript::labels::ABSORB_BATCH_SHAPE;
use akita_transcript::{sample_ext_challenge, Transcript};
use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};
use jolt_field::{CanonicalEncoding, ExtField, Field};

/// Per-group opening geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

    /// Build a root-opening layout from precommitted groups plus the final/new group.
    pub fn from_root_groups(
        precommitteds: &[PolynomialGroupLayout],
        final_group: PolynomialGroupLayout,
    ) -> Result<Self, AkitaError> {
        let mut groups = Vec::with_capacity(precommitteds.len() + 1);
        groups.extend_from_slice(precommitteds);
        groups.push(final_group);
        Self::from_groups(groups)
    }

    /// Worst-case setup-capacity request as a one-group layout.
    pub fn from_setup_seed(seed: &AkitaSetupDescriptor) -> Result<Self, AkitaError> {
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

    /// Whether transcript batching needs a sampled row coefficient challenge.
    pub fn requires_row_batch_challenge(&self) -> bool {
        self.num_total_polynomials() > 1
    }

    /// Collapse this batch into the single group shape used by extension
    /// opening reduction sizing.
    ///
    /// The opening point uses the maximum group-local arity, while the partial
    /// count uses the checked sum of polynomials across every group.
    pub fn aggregate_polynomial_group_layout(&self) -> Result<PolynomialGroupLayout, AkitaError> {
        self.check()?;
        Ok(PolynomialGroupLayout::new(
            self.max_num_vars(),
            self.checked_num_total_polynomials()?,
        ))
    }

    fn checked_num_total_polynomials(&self) -> Result<usize, AkitaError> {
        checked::sum(self.groups.iter().map(|group| group.num_polynomials()))
            .ok_or(AkitaError::InvalidProof)
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

    /// Commitment-group index used as the final/new group for multi-group root schedules.
    pub fn root_final_group_index(&self) -> Result<usize, AkitaError> {
        self.check()?;
        self.groups
            .len()
            .checked_sub(1)
            .ok_or(AkitaError::InvalidProof)
    }

    /// Group processing order for multi-group root schedules: final/new group first.
    pub fn root_group_order(&self) -> Result<Vec<usize>, AkitaError> {
        let final_group_index = self.root_final_group_index()?;
        let mut order = Vec::with_capacity(self.num_groups());
        order.push(final_group_index);
        for group_index in 0..self.num_groups() {
            if group_index != final_group_index {
                order.push(group_index);
            }
        }
        Ok(order)
    }

    /// Layouts of precommitted groups in root transcript order.
    pub fn root_precommitted_group_layouts(&self) -> Result<&[PolynomialGroupLayout], AkitaError> {
        self.check()?;
        let final_index = self.root_final_group_index()?;
        Ok(&self.groups[..final_index])
    }

    /// Final/new group layout for multi-group root schedule lookup.
    pub fn root_final_group_layout(&self) -> Result<PolynomialGroupLayout, AkitaError> {
        Ok(*self.group_layout(self.root_final_group_index()?)?)
    }

    /// Flat claim range covered by one commitment group.
    pub fn root_group_claim_range(
        &self,
        group_index: usize,
    ) -> Result<std::ops::Range<usize>, AkitaError> {
        self.check()?;
        if group_index >= self.groups.len() {
            return Err(AkitaError::InvalidProof);
        }
        let start = checked::sum(
            self.groups[..group_index]
                .iter()
                .map(|group| group.num_polynomials()),
        )
        .ok_or(AkitaError::InvalidProof)?;
        let end = start
            .checked_add(self.groups[group_index].num_polynomials())
            .ok_or(AkitaError::InvalidProof)?;
        Ok(start..end)
    }

    /// Digest layout-only opening geometry.
    pub fn opening_batch_digest(&self) -> DescriptorDigest {
        let mut bytes = Vec::new();
        push_usize(&mut bytes, self.num_groups());
        for group in &self.groups {
            push_usize(&mut bytes, group.num_vars());
            push_usize(&mut bytes, group.num_polynomials());
        }
        blake2b_256(&bytes)
    }

    /// Absorb normalized batch-shape fields into the transcript.
    pub fn append_batch_shape_to_transcript<F, T>(
        &self,
        transcript: &mut T,
    ) -> Result<(), AkitaError>
    where
        F: Field + CanonicalEncoding,
        T: Transcript<F>,
    {
        self.check()?;

        transcript.append_serde(ABSORB_BATCH_SHAPE, &self.num_groups());
        for group in self.groups() {
            transcript.append_serde(ABSORB_BATCH_SHAPE, &group.num_vars());
            transcript.append_serde(ABSORB_BATCH_SHAPE, &group.num_polynomials());
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
        E: Field,
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

    /// Scale flat row coefficients by one transparent reduction factor per
    /// opening group.
    ///
    /// The returned coefficients remain in canonical flat-claim order. This is
    /// the shared prover/verifier definition of the final grouped
    /// extension-opening relation.
    pub fn scale_row_coefficients_by_group<E>(
        &self,
        row_coefficients: &[E],
        group_factors: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: Field,
    {
        if row_coefficients.len() != self.num_total_polynomials() {
            return Err(AkitaError::InvalidSize {
                expected: self.num_total_polynomials(),
                actual: row_coefficients.len(),
            });
        }
        if group_factors.len() != self.num_groups() {
            return Err(AkitaError::InvalidSize {
                expected: self.num_groups(),
                actual: group_factors.len(),
            });
        }
        let mut scaled = Vec::with_capacity(row_coefficients.len());
        for (group_index, &factor) in group_factors.iter().enumerate() {
            let range = self.root_group_claim_range(group_index)?;
            scaled.extend(
                row_coefficients
                    .get(range)
                    .ok_or(AkitaError::InvalidProof)?
                    .iter()
                    .map(|&coefficient| coefficient * factor),
            );
        }
        Ok(scaled)
    }
}

/// Public claims and commitment payload for one polynomial group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolynomialGroupClaims<'a, F: Clone, C = ()> {
    point: OpeningPoints<'a, F>,
    evaluations: Vec<F>,
    commitment: C,
}

impl<'a, F: Clone, C> PolynomialGroupClaims<'a, F, C> {
    /// Build one group of public claims.
    pub fn new(
        point: impl Into<OpeningPoints<'a, F>>,
        evaluations: Vec<F>,
        commitment: C,
    ) -> Result<Self, AkitaError> {
        if evaluations.is_empty() {
            return Err(AkitaError::InvalidInput(
                "opening claim groups must be nonempty".to_string(),
            ));
        }
        Ok(Self {
            point: point.into(),
            evaluations,
            commitment,
        })
    }

    /// Complete opening point owned by this group.
    pub fn point(&self) -> &[F] {
        self.point.as_ref()
    }

    /// Number of variables in this group's opening point.
    pub fn num_vars(&self) -> usize {
        self.point.len()
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

/// Public opening claims: polynomial groups in transcript order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpeningClaims<'a, F: Clone, C = ()> {
    groups: Vec<PolynomialGroupClaims<'a, F, C>>,
}

/// Public opening statement bound to one exact verifier-approved schedule row.
///
/// The schedule selection is batch-level. Individual commitments remain
/// reusable and carry only their exact algebraic profiles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupBatchStatement<'a, E: Clone, F: Field> {
    selection: OpeningScheduleSelection,
    claims: OpeningClaims<'a, E, &'a CommittedGroup<F>>,
}

impl<'a, E: Clone, F: Field> GroupBatchStatement<'a, E, F> {
    /// Bind ordered self-describing claims to an approved schedule row.
    ///
    /// # Errors
    ///
    /// Returns an error when the claim set is empty or structurally malformed.
    pub fn new(
        selection: OpeningScheduleSelection,
        claims: OpeningClaims<'a, E, &'a CommittedGroup<F>>,
    ) -> Result<Self, AkitaError> {
        claims.check()?;
        Ok(Self { selection, claims })
    }

    /// Exact catalog and row identity selected for this batch.
    pub const fn selection(&self) -> OpeningScheduleSelection {
        self.selection
    }

    /// Ordered public opening claims.
    pub fn claims(&self) -> &OpeningClaims<'a, E, &'a CommittedGroup<F>> {
        &self.claims
    }

    /// Consume the statement into its ordered public claims.
    pub fn into_claims(self) -> OpeningClaims<'a, E, &'a CommittedGroup<F>> {
        self.claims
    }
}

impl<'a, E: Clone, F: Field> OpeningClaims<'a, E, &'a CommittedGroup<F>> {
    /// Layout reconstructed from the frozen commitment profiles.
    pub fn committed_layout(&self) -> Result<OpeningClaimsLayout, AkitaError> {
        self.check()?;
        let mut groups = Vec::with_capacity(self.groups.len());
        for group in &self.groups {
            let declared = group.commitment.profile().group;
            if declared.num_vars() != group.point.len()
                || declared.num_polynomials() != group.evaluations.len()
            {
                return Err(AkitaError::InvalidProof);
            }
            groups.push(declared);
        }
        OpeningClaimsLayout::from_groups(groups)
    }
}

impl<'a, E: Clone, F: Field> OpeningClaims<'a, E, CommittedGroup<F>> {
    /// Layout reconstructed from the frozen commitment profiles.
    pub fn committed_layout(&self) -> Result<OpeningClaimsLayout, AkitaError> {
        self.check()?;
        let mut groups = Vec::with_capacity(self.groups.len());
        for group in &self.groups {
            let declared = group.commitment.profile().group;
            if declared.num_vars() != group.point.len()
                || declared.num_polynomials() != group.evaluations.len()
            {
                return Err(AkitaError::InvalidProof);
            }
            groups.push(declared);
        }
        OpeningClaimsLayout::from_groups(groups)
    }
}

impl<'a, F: Clone, C> OpeningClaims<'a, F, C> {
    /// Build public claims from ordered groups.
    pub fn from_groups(groups: Vec<PolynomialGroupClaims<'a, F, C>>) -> Result<Self, AkitaError> {
        let claims = Self { groups };
        claims.check()?;
        Ok(claims)
    }

    /// Validate group and count consistency.
    pub fn check(&self) -> Result<(), AkitaError> {
        if self.groups.is_empty() || self.checked_num_total_polynomials()? == 0 {
            return Err(AkitaError::InvalidProof);
        }
        for group in &self.groups {
            if group.evaluations.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
        }
        Ok(())
    }

    /// Validate consistency plus public capacity against the setup limits.
    pub fn validate(&self, seed: &AkitaSetupDescriptor) -> Result<(), AkitaError> {
        self.check()?;
        let max_num_vars = self.layout()?.max_num_vars();
        if max_num_vars > seed.max_num_vars {
            return Err(AkitaError::InvalidPointDimension {
                expected: seed.max_num_vars,
                actual: max_num_vars,
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
        checked::sum(
            self.groups
                .iter()
                .map(PolynomialGroupClaims::num_evaluations),
        )
        .ok_or(AkitaError::InvalidProof)
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

    /// Borrow one group's complete opening point.
    pub fn group_point(&self, g: usize) -> Result<&[F], AkitaError> {
        self.groups
            .get(g)
            .map(PolynomialGroupClaims::point)
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
                .map(|group| PolynomialGroupLayout::new(group.point.len(), group.evaluations.len()))
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

impl<'a, F: Clone, C> OpeningClaims<'a, F, C> {
    /// Return the only commitment when the current single-group path applies.
    pub fn single_group_commitment(&self) -> Option<&C> {
        self.groups
            .first()
            .filter(|_| self.groups.len() == 1)
            .map(PolynomialGroupClaims::commitment)
    }
}

/// Apply the scheduled work and sample row-batching coefficients for a protocol site.
///
/// Claimed values must already have been absorbed. Keeping message binding
/// separate makes the phase boundary explicit. The caller chooses whether this
/// is an early protocol-local compression or the later application batch and
/// must use the corresponding transcript phase.
pub fn sample_row_coefficients<F, L, T>(
    layout: &OpeningClaimsLayout,
    site: GrindingSite,
    transcript: &mut T,
) -> Result<Vec<L>, AkitaError>
where
    F: Field + CanonicalEncoding,
    L: ExtField<F>,
    T: TranscriptGrinding<F>,
{
    layout.check()?;
    if !layout.requires_row_batch_challenge() {
        return Ok(vec![L::one()]);
    }
    transcript.grind_query(site)?;
    let challenge_label = site.proof_of_work_label().ok_or_else(|| {
        AkitaError::InvalidInput("row batching requires a proof-of-work grinding site".into())
    })?;
    Ok((0..layout.num_total_polynomials())
        .map(|_| sample_ext_challenge::<F, L, T>(transcript, challenge_label))
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
    use akita_transcript::AkitaTranscript;
    use jolt_field::{Prime128OffsetA7F7, Ring, Zero};

    type F = Prime128OffsetA7F7;

    fn prefix_claims(num_vars: usize, evals: usize) -> OpeningClaims<'static, F, ()> {
        let group =
            PolynomialGroupClaims::new(vec![F::zero(); num_vars], vec![F::zero(); evals], ())
                .expect("group");
        OpeningClaims::from_groups(vec![group]).expect("claims")
    }

    #[test]
    fn groups_own_independent_points() {
        let first = vec![F::from_u64(1), F::from_u64(2)];
        let second = vec![F::from_u64(3), F::from_u64(4), F::from_u64(5)];
        let claims = OpeningClaims::from_groups(vec![
            PolynomialGroupClaims::new(first.clone(), vec![F::zero()], ()).expect("first group"),
            PolynomialGroupClaims::new(second.clone(), vec![F::zero()], ()).expect("second group"),
        ])
        .expect("claims");

        assert_eq!(claims.group_point(0).expect("first point"), first);
        assert_eq!(claims.group_point(1).expect("second point"), second);
        assert_eq!(claims.layout().expect("layout").max_num_vars(), 3);
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
    fn layout_digest_binds_group_arity_count_and_order() {
        let baseline = OpeningClaimsLayout::from_groups(vec![
            PolynomialGroupLayout::new(2, 1),
            PolynomialGroupLayout::new(3, 2),
        ])
        .expect("baseline");
        let changed_arity = OpeningClaimsLayout::from_groups(vec![
            PolynomialGroupLayout::new(1, 1),
            PolynomialGroupLayout::new(3, 2),
        ])
        .expect("changed arity");
        let changed_count = OpeningClaimsLayout::from_groups(vec![
            PolynomialGroupLayout::new(2, 2),
            PolynomialGroupLayout::new(3, 1),
        ])
        .expect("changed count");
        let reversed = OpeningClaimsLayout::from_groups(vec![
            PolynomialGroupLayout::new(3, 2),
            PolynomialGroupLayout::new(2, 1),
        ])
        .expect("reversed");

        let digest = baseline.opening_batch_digest();
        assert_ne!(digest, changed_arity.opening_batch_digest());
        assert_ne!(digest, changed_count.opening_batch_digest());
        assert_ne!(digest, reversed.opening_batch_digest());
    }

    #[test]
    fn root_layout_helpers_preserve_group_order_and_lengths() {
        let layout = OpeningClaimsLayout::from_groups(vec![
            PolynomialGroupLayout::new(2, 1),
            PolynomialGroupLayout::new(3, 2),
            PolynomialGroupLayout::new(4, 1),
        ])
        .expect("multi-group layout");

        assert_eq!(layout.root_final_group_index().expect("final index"), 2);
        assert_eq!(
            layout
                .root_precommitted_group_layouts()
                .expect("precommitted layouts"),
            &[
                PolynomialGroupLayout::new(2, 1),
                PolynomialGroupLayout::new(3, 2),
            ]
        );
        assert_eq!(
            layout.root_final_group_layout().expect("final layout"),
            PolynomialGroupLayout::new(4, 1)
        );
        assert_eq!(
            layout.root_group_order().expect("group order"),
            vec![2, 0, 1]
        );
        assert_eq!(layout.root_group_claim_range(0).expect("range"), 0..1);
        assert_eq!(layout.root_group_claim_range(1).expect("range"), 1..3);
        assert_eq!(layout.root_group_claim_range(2).expect("range"), 3..4);
    }

    #[test]
    fn from_root_groups_appends_final_group_after_precommitted_groups() {
        let precommitteds = [
            PolynomialGroupLayout::new(2, 1),
            PolynomialGroupLayout::new(3, 2),
        ];
        let final_group = PolynomialGroupLayout::new(4, 3);
        let layout = OpeningClaimsLayout::from_root_groups(&precommitteds, final_group)
            .expect("root layout");

        assert_eq!(
            layout.groups(),
            &[
                PolynomialGroupLayout::new(2, 1),
                PolynomialGroupLayout::new(3, 2),
                PolynomialGroupLayout::new(4, 3),
            ]
        );
        assert_eq!(
            layout.root_final_group_layout().expect("final group"),
            final_group
        );
    }

    #[test]
    fn aggregate_opening_layout_preserves_grouped_claim_count_for_eor() {
        let precommitteds = [PolynomialGroupLayout::new(10, 2)];
        let final_group = PolynomialGroupLayout::new(8, 1);
        let aggregate = OpeningClaimsLayout::from_root_groups(&precommitteds, final_group)
            .expect("root layout")
            .aggregate_polynomial_group_layout()
            .expect("aggregate layout");

        assert_eq!(aggregate, PolynomialGroupLayout::new(10, 3));
        let final_only_bytes = crate::extension_opening_reduction_level_bytes(128, 4, final_group)
            .expect("final-only EOR bytes");
        let aggregate_bytes = crate::extension_opening_reduction_level_bytes(128, 4, aggregate)
            .expect("aggregate EOR bytes");
        assert!(final_only_bytes > 0);
        let extra_partial_bytes = 2 * 4 * crate::field_bytes(128);
        let extra_round_bytes =
            2 * crate::EXTENSION_OPENING_REDUCTION_DEGREE * crate::field_bytes(128);
        let extra_terminal_claim_bytes = 2 * crate::field_bytes(128);
        assert_eq!(
            aggregate_bytes - final_only_bytes,
            extra_partial_bytes + extra_round_bytes + extra_terminal_claim_bytes
        );
    }

    #[test]
    fn public_row_coefficients_bind_against_claim_cancellation() {
        let layout = OpeningClaimsLayout::from_groups(vec![
            PolynomialGroupLayout::singleton(2),
            PolynomialGroupLayout::singleton(3),
        ])
        .expect("two-group layout");
        let openings = [F::from_u64(5), F::from_u64(11)];
        let delta = F::from_u64(3);
        let tampered = [openings[0] + delta, openings[1] - delta];

        let target = |values: &[F]| {
            let mut inner = AkitaTranscript::<F>::new(b"test/public-row-claim-binding");
            append_claim_values_to_transcript::<F, F, _>(values, &mut inner);
            let plan = crate::GrindingPlan::new(
                vec![crate::GrindingRun::proof_of_work(
                    GrindingSite::EvaluationBatch { level: 0 },
                    1,
                    128,
                )
                .expect("grinding run")],
                128,
            )
            .expect("grinding plan");
            let mut transcript = crate::ProverGrindingTranscript::new(&mut inner, &plan)
                .expect("grinding transcript");
            let coefficients = sample_row_coefficients::<F, F, _>(
                &layout,
                GrindingSite::EvaluationBatch { level: 0 },
                &mut transcript,
            )
            .expect("derive coefficients");
            transcript.finish().expect("finish grinding");
            layout
                .batched_eval_target(&coefficients, values)
                .expect("batched target")
        };

        assert_ne!(target(&openings), target(&tampered));
    }
}
