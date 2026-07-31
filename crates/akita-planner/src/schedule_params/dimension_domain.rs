use akita_field::AkitaError;
use akita_types::CommitmentRingDims;

use crate::PlannerPolicy;

/// Objective used to select one schedule from the mixed-dimension Pareto frontier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MixedScheduleObjective {
    /// Preserve the original experimental policy: setup envelope first, then proof bytes.
    #[default]
    MinimumSetupThenProof,
    /// Minimize the worst relative regression from the independently best proof,
    /// setup-envelope, and direct-verifier matrix-work values.
    Balanced,
}

/// Search policy for an explicit mixed-ring dimension domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MixedDimensionPolicy {
    /// Number of leading fold levels at which the full mixed domain is searched.
    pub mixed_fold_levels: usize,
    /// Fixed role dimensions used after the leading mixed levels.
    pub suffix_dimensions: CommitmentRingDims,
    /// Optional cap on the A-role output dimension `n_A * d_A`.
    pub max_inner_total_dimension: Option<usize>,
    /// Reject a larger B dimension once a smaller admitted B dimension has rank one.
    pub stop_outer_at_rank_one: bool,
    /// Reject a larger D dimension once a smaller admitted D dimension has rank one.
    pub stop_opening_at_rank_one: bool,
    /// Final selection rule over nondominated complete schedules.
    pub objective: MixedScheduleObjective,
}

impl Default for MixedDimensionPolicy {
    fn default() -> Self {
        Self {
            mixed_fold_levels: 2,
            suffix_dimensions: CommitmentRingDims::uniform(64),
            max_inner_total_dimension: None,
            stop_outer_at_rank_one: false,
            stop_opening_at_rank_one: false,
            objective: MixedScheduleObjective::MinimumSetupThenProof,
        }
    }
}

/// Explicit A/B/D dimensions admitted by mixed-D planner search.
///
/// `PlannerPolicy::ring_dimension` remains the setup generation dimension and
/// the implicit singleton domain used by [`super::find_schedule`]. This separate
/// offline-only value makes mixed-D search opt-in without changing runtime
/// policy or existing catalog identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingDimensionSearchDomain {
    setup_generation_dimension: usize,
    candidates: Vec<CommitmentRingDims>,
    mixed_policy: MixedDimensionPolicy,
}

impl RingDimensionSearchDomain {
    /// Construct and canonicalize a non-empty dimension domain.
    ///
    /// Every tuple must satisfy the A-carrier invariant, and every role
    /// dimension must divide `setup_generation_dimension`.
    pub fn new(
        setup_generation_dimension: usize,
        candidates: impl IntoIterator<Item = CommitmentRingDims>,
    ) -> Result<Self, AkitaError> {
        if setup_generation_dimension == 0 || !setup_generation_dimension.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "setup generation dimension must be a nonzero power of two".into(),
            ));
        }
        let mut candidates = candidates.into_iter().collect::<Vec<_>>();
        candidates.sort_by_key(|dims| (dims.d_a(), dims.d_b(), dims.d_d()));
        candidates.dedup();
        if candidates.is_empty() {
            return Err(AkitaError::InvalidSetup(
                "ring-dimension search domain must be nonempty".into(),
            ));
        }
        for dims in &candidates {
            dims.validate_a_carrier()?;
            for d in [dims.d_a(), dims.d_b(), dims.d_d()] {
                if !setup_generation_dimension.is_multiple_of(d) {
                    return Err(AkitaError::InvalidSetup(format!(
                        "candidate dimension D{d} does not divide setup generation dimension \
                         D{setup_generation_dimension}"
                    )));
                }
            }
        }
        Ok(Self {
            setup_generation_dimension,
            candidates,
            mixed_policy: MixedDimensionPolicy::default(),
        })
    }

    /// Construct the explicit singleton domain used by a uniform policy.
    pub fn uniform(setup_generation_dimension: usize) -> Result<Self, AkitaError> {
        Self::new(
            setup_generation_dimension,
            [CommitmentRingDims::uniform(setup_generation_dimension)],
        )
    }

    /// Construct every nested `(d_A, d_B, d_D)` triple from role ladders.
    ///
    /// `outer_opening_dimensions` is shared by B and D; only triples satisfying
    /// `d_D | d_B | d_A` are admitted.
    pub fn nested(
        setup_generation_dimension: usize,
        inner_dimensions: impl IntoIterator<Item = usize>,
        outer_opening_dimensions: impl IntoIterator<Item = usize>,
    ) -> Result<Self, AkitaError> {
        let inner_dimensions = inner_dimensions.into_iter().collect::<Vec<_>>();
        let outer_opening_dimensions = outer_opening_dimensions.into_iter().collect::<Vec<_>>();
        let mut candidates = Vec::new();
        for inner in inner_dimensions {
            for &outer in &outer_opening_dimensions {
                for &opening in &outer_opening_dimensions {
                    if inner.is_multiple_of(outer) && outer.is_multiple_of(opening) {
                        candidates.push(CommitmentRingDims {
                            inner,
                            outer,
                            opening,
                        });
                    }
                }
            }
        }
        Self::new(setup_generation_dimension, candidates)
    }

    /// Select the policy used when this domain contains mixed dimensions.
    #[must_use]
    pub fn with_mixed_policy(mut self, mixed_policy: MixedDimensionPolicy) -> Self {
        self.mixed_policy = mixed_policy;
        self
    }

    /// Canonically ordered admitted A/B/D tuples.
    pub fn candidates(&self) -> &[CommitmentRingDims] {
        &self.candidates
    }

    /// Setup generation dimension against which this domain was validated.
    pub fn setup_generation_dimension(&self) -> usize {
        self.setup_generation_dimension
    }

    /// Policy controlling mixed-level enumeration and final selection.
    pub fn mixed_policy(&self) -> MixedDimensionPolicy {
        self.mixed_policy
    }

    pub(super) fn validate_for_policy(&self, policy: &PlannerPolicy) -> Result<(), AkitaError> {
        if self.setup_generation_dimension != policy.ring_dimension {
            return Err(AkitaError::InvalidSetup(format!(
                "ring-dimension domain uses setup generation D{}, but policy uses D{}",
                self.setup_generation_dimension, policy.ring_dimension
            )));
        }
        Ok(())
    }

    pub(super) fn is_uniform_policy_domain(&self, policy: &PlannerPolicy) -> bool {
        self.candidates.as_slice() == [CommitmentRingDims::uniform(policy.ring_dimension)]
    }
}
