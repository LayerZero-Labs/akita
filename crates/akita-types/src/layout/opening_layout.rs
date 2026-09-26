//! Layout-only opening geometry shared by planning, sizing, and claims.

use crate::descriptor_bytes::{digest_descriptor_bytes, push_usize, DescriptorDigest};
use akita_error::{checked, AkitaError};
use jolt_field::Field;

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
    #[cfg(any(test, feature = "test-support"))]
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
        digest_descriptor_bytes(&bytes)
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
