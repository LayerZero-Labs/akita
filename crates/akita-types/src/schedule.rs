//! Runtime schedule shapes shared by configs, prover, verifier, and planner.

use crate::config::SetupContributionMode;
use crate::descriptor_bytes::{push_u32, push_usize};
use crate::{CleartextWitnessShape, LevelParams, OpeningClaimsLayout, PolynomialGroupLayout};
use akita_field::{AkitaError, CanonicalField};

/// Public inputs that deterministically select one level's active Akita params.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AkitaScheduleInputs {
    /// Root polynomial variable count.
    pub num_vars: usize,
    /// Fold level, where `0` is the original polynomial.
    pub level: usize,
    /// Current witness length in field elements before this level runs.
    pub current_w_len: usize,
}

/// Transcript binding used for one fold's outgoing witness state.
///
/// This is schedule-owned because the same intermediate proof body may either
/// recurse through an outer commitment or hand its witness to the final
/// suffix fold as a public inner `t` state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NextWitnessBindingPolicy {
    /// Bind `u = B * decompose(t)` and recurse through another committed fold.
    OuterCommitment,
    /// Bind canonical inner-state `t` bytes for the following suffix-terminal
    /// fold. No outer `u` is present on this edge.
    TerminalInnerState,
}

/// Schedule facts for one fold level.
#[derive(Debug, Clone)]
pub struct ExecutionSchedule {
    /// Fold level, where `0` is the root.
    pub level: usize,
    /// Witness length expected before this fold runs.
    pub current_w_len: usize,
    /// Active level parameters for this fold.
    pub params: LevelParams,
    /// Successor parameters for the next committed level, or a log-basis stub
    /// for the terminal direct witness.
    pub next_params: LevelParams,
    /// Witness length expected after this fold's ring-switch relation builds
    /// the next `w`.
    pub next_w_len: usize,
    /// Whether this fold hands off to the terminal direct witness.
    pub is_terminal: bool,
    /// Transcript/wire policy for the witness state leaving this fold.
    pub next_witness_binding: Option<NextWitnessBindingPolicy>,
}

impl ExecutionSchedule {
    /// Canonical physical relation-row layout for this fold.
    ///
    /// A terminal fold receives public `t` from its predecessor and therefore
    /// has exactly `consistency | A`.
    #[must_use]
    pub fn relation_matrix_row_layout(&self) -> crate::RelationMatrixRowLayout {
        if self.is_terminal {
            crate::RelationMatrixRowLayout::WithoutCommitmentBlocks
        } else {
            crate::RelationMatrixRowLayout::WithDBlock
        }
    }

    /// Validate the witness length entering this fold.
    ///
    /// # Errors
    ///
    /// Returns an error if the runtime witness length does not match the
    /// planner schedule.
    pub fn validate_current_w_len(&self, actual_current_w_len: usize) -> Result<(), AkitaError> {
        if actual_current_w_len != self.current_w_len {
            return Err(AkitaError::InvalidSetup(format!(
                "scheduled fold level {} did not match runtime state: \
                 expected_w_len={}, actual_w_len={}",
                self.level, self.current_w_len, actual_current_w_len
            )));
        }
        Ok(())
    }

    /// Validate the next witness length produced by this fold.
    ///
    /// # Errors
    ///
    /// Returns an error if the post-ring-switch witness length does not match
    /// the planner schedule.
    pub fn validate_next_w_len(&self, actual_next_w_len: usize) -> Result<(), AkitaError> {
        if actual_next_w_len != self.next_w_len {
            return Err(AkitaError::InvalidSetup(format!(
                "scheduled fold level {} produced unexpected next-w length: expected={}, actual={actual_next_w_len}",
                self.level, self.next_w_len
            )));
        }
        Ok(())
    }
}

/// Root layout metadata frozen when a standalone commitment group is created.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrecommittedGroupParams {
    /// Per-group root schedule entry shape.
    pub group: PolynomialGroupLayout,
    /// Exact number of live source ring elements per claim (`N`).
    pub num_live_ring_elements_per_claim: usize,
    /// Number of positions per block (`M`), power-of-two in the current Boolean layout.
    pub num_positions_per_block: usize,
    /// Exact number of live blocks (`B = ceil(N / M)`).
    pub num_live_blocks: usize,
    /// Group-local flat or tensor fold challenge shape.
    pub fold_challenge_shape: akita_challenges::TensorChallengeShape,
    /// Gadget basis selected for the standalone group commit.
    pub log_basis: u32,
    /// A-role row count selected for the committed inner rows.
    pub n_a: usize,
    /// Conservative B-role row count used by the standalone precommit.
    pub conservative_n_b: usize,
}

impl PrecommittedGroupParams {
    /// Build frozen group metadata from the concrete commit params.
    pub fn from_params(group: PolynomialGroupLayout, params: &LevelParams) -> Self {
        Self {
            group,
            num_live_ring_elements_per_claim: params.num_live_ring_elements_per_claim,
            num_positions_per_block: params.num_positions_per_block,
            num_live_blocks: params.num_live_blocks,
            fold_challenge_shape: params.fold_challenge_shape,
            log_basis: params.log_basis,
            n_a: params.a_key.row_len(),
            conservative_n_b: params.b_key.row_len(),
        }
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        push_usize(bytes, self.group.num_vars());
        push_usize(bytes, self.group.num_polynomials());
        push_usize(bytes, self.num_live_ring_elements_per_claim);
        push_usize(bytes, self.num_positions_per_block);
        push_usize(bytes, self.num_live_blocks);
        bytes.push(match self.fold_challenge_shape {
            akita_challenges::TensorChallengeShape::Flat => 0,
            akita_challenges::TensorChallengeShape::Tensor { .. } => 1,
        });
        if let akita_challenges::TensorChallengeShape::Tensor { fold_low_len } =
            self.fold_challenge_shape
        {
            push_usize(bytes, fold_low_len);
        }
        push_u32(bytes, self.log_basis);
        push_usize(bytes, self.n_a);
        push_usize(bytes, self.conservative_n_b);
    }

    /// Validate that this layout is a well-formed standalone commitment group.
    pub fn validate(&self) -> Result<(), AkitaError> {
        self.group.validate()?;
        if self.group.num_polynomials() != 1 {
            return Err(AkitaError::InvalidSetup(format!(
                "precommitted groups must contain exactly one polynomial, got {}",
                self.group.num_polynomials()
            )));
        }
        if self.n_a == 0 || self.conservative_n_b == 0 {
            return Err(AkitaError::InvalidSetup(
                "commitment group layout requires nonzero A rows and conservative B rows"
                    .to_string(),
            ));
        }
        if self.log_basis == 0 {
            return Err(AkitaError::InvalidSetup(
                "commitment group layout requires nonzero log_basis".to_string(),
            ));
        }
        Ok(())
    }

    /// Validate that frozen exact block geometry matches `group.num_vars`.
    pub fn validate_root_geometry(&self, ring_dimension: usize) -> Result<(), AkitaError> {
        let alpha = ring_dimension.trailing_zeros() as usize;
        let Some(source_field_len) = self
            .num_live_ring_elements_per_claim
            .checked_mul(ring_dimension)
        else {
            return Err(AkitaError::InvalidSetup(
                "commitment group layout geometry overflow".to_string(),
            ));
        };
        let expected_field_len = 1usize
            .checked_shl(self.group.num_vars() as u32)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("commitment group field length overflow".to_string())
            })?;
        if source_field_len != expected_field_len
            || self.num_positions_per_block == 0
            || !self.num_positions_per_block.is_power_of_two()
            || self.num_live_blocks
                != self
                    .num_live_ring_elements_per_claim
                    .div_ceil(self.num_positions_per_block)
        {
            return Err(AkitaError::InvalidSetup(format!(
                "precommitted group geometry does not match group.num_vars: \
                 N={} L={} F={} alpha={} group.num_vars={}",
                self.num_live_ring_elements_per_claim,
                self.num_positions_per_block,
                self.num_live_blocks,
                alpha,
                self.group.num_vars()
            )));
        }
        Ok(())
    }

    /// Validate metadata frozen by a precommitted group at precommit time.
    pub fn validate_frozen_precommit(&self, ring_dimension: usize) -> Result<(), AkitaError> {
        self.validate()?;
        self.validate_root_geometry(ring_dimension)?;
        Ok(())
    }
}

/// Freezes conservative root-commit metadata for each precommitted group when
/// building a schedule lookup key from an opening layout.
pub trait ScheduleKeyPrecommitSource {
    /// Resolve frozen standalone-commit params for one precommitted group.
    fn precommitted_group_params(
        group: PolynomialGroupLayout,
    ) -> Result<PrecommittedGroupParams, AkitaError>;
}

/// Canonical runtime schedule lookup key.
///
/// Scalar same-point openings use an empty `precommitteds` vector and store the
/// sole group in `final_group`. Multi-group roots list earlier groups in
/// `precommitteds` and the final group in `final_group`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AkitaScheduleLookupKey {
    /// Final group shape for the multi-group root commitment.
    pub final_group: PolynomialGroupLayout,
    /// Previously committed groups in caller-supplied transcript order.
    pub precommitteds: Vec<PrecommittedGroupParams>,
}

impl AkitaScheduleLookupKey {
    /// Scalar root-opening context with no precommitted groups.
    pub fn single(final_group: PolynomialGroupLayout) -> Self {
        Self {
            final_group,
            precommitteds: Vec::new(),
        }
    }

    /// Build the canonical schedule lookup key for `layout`.
    ///
    /// Scalar layouts leave `precommitteds` empty. Grouped layouts freeze each
    /// earlier group through `S` (production uses the conservative commit
    /// adapter wired by `akita-config`'s `opening_schedule_key`).
    pub fn from_layout<S: ScheduleKeyPrecommitSource>(
        layout: &OpeningClaimsLayout,
    ) -> Result<Self, AkitaError> {
        layout.check()?;
        let precommitteds = if layout.num_groups() == 1 {
            Vec::new()
        } else {
            layout
                .root_precommitted_group_layouts()?
                .iter()
                .copied()
                .map(S::precommitted_group_params)
                .collect::<Result<Vec<_>, _>>()?
        };
        let key = Self {
            final_group: layout.root_final_group_layout()?,
            precommitteds,
        };
        key.validate()?;
        Ok(key)
    }

    /// Build a multi-group opening layout from this schedule lookup key.
    pub fn opening_layout(&self) -> Result<OpeningClaimsLayout, AkitaError> {
        let mut groups: Vec<PolynomialGroupLayout> = self
            .precommitteds
            .iter()
            .map(|layout| layout.group)
            .collect();
        groups.push(self.final_group);
        OpeningClaimsLayout::from_groups(groups)
    }

    /// Number of commitment groups in this schedule key.
    pub fn num_commitment_groups(&self) -> usize {
        self.precommitteds.len() + 1
    }

    /// Total number of polynomials across the final and precommitted groups.
    pub fn num_polynomials(&self) -> Result<usize, AkitaError> {
        let mut total = self.final_group.num_polynomials();
        for layout in &self.precommitteds {
            total = total
                .checked_add(layout.group.num_polynomials())
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "multi-group root polynomial count overflow".to_string(),
                    )
                })?;
        }
        Ok(total)
    }

    /// Validate per-group metadata.
    pub fn validate(&self) -> Result<(), AkitaError> {
        self.final_group.validate()?;
        if self.final_group.num_vars() == 0 {
            return Err(AkitaError::InvalidSetup(
                "schedule lookup key dimensions must be at least 1".to_string(),
            ));
        }
        for layout in &self.precommitteds {
            layout.group.validate()?;
            if layout.group.num_vars() > self.final_group.num_vars() / 2 {
                return Err(AkitaError::InvalidInput(
                    "multi-group root requires precommitted groups to have at most half the final num_vars"
                        .to_string(),
                ));
            }
            layout.validate()?;
        }
        Ok(())
    }
}

/// Number of gadget decomposition levels needed for `r` over field `F`.
pub fn r_decomp_levels<F: CanonicalField>(log_basis: u32) -> usize {
    let modulus = detect_field_modulus::<F>();
    let field_bits = 128 - (modulus.saturating_sub(1)).leading_zeros();
    crate::sis::compute_num_digits_full_field(field_bits, log_basis)
}

/// Detect the field modulus from the canonical representation.
///
/// Uses the identity: the canonical form of `-1` in `Z_q` is `q - 1`.
#[inline]
pub fn detect_field_modulus<F: CanonicalField>() -> u128 {
    crate::dispatch::field_modulus::<F>()
}

/// Total ring elements in an intermediate recursive witness polynomial.
/// Terminal witnesses are quotient-free and must be sized from their
/// [`crate::CleartextWitnessShape`] instead.
pub fn intermediate_w_ring_element_count_with_counts<F: CanonicalField>(
    lp: &LevelParams,
    num_polynomials: usize,
    num_z_segments: usize,
) -> Result<usize, AkitaError> {
    let modulus = detect_field_modulus::<F>();
    let field_bits = 128 - (modulus.saturating_sub(1)).leading_zeros();
    intermediate_w_ring_element_count_with_counts_bits(
        field_bits,
        lp,
        num_polynomials,
        num_z_segments,
    )
}

/// Non-generic variant of [`intermediate_w_ring_element_count_with_counts`] for
/// callers that already know the effective field bit width. The planner
/// search uses this to keep its API free of a base-field type parameter.
pub fn intermediate_w_ring_element_count_with_counts_bits(
    field_bits: u32,
    lp: &LevelParams,
    num_polynomials: usize,
    num_z_segments: usize,
) -> Result<usize, AkitaError> {
    lp.require_scalar_level("intermediate_w_ring_element_count_with_counts_bits")?;
    let e_hat_count = num_polynomials
        .checked_mul(lp.num_live_blocks)
        .and_then(|n| n.checked_mul(lp.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("witness W width overflow".to_string()))?;
    let t_hat_count = num_polynomials
        .checked_mul(lp.num_live_blocks)
        .and_then(|n| n.checked_mul(lp.a_key.row_len()))
        .and_then(|n| n.checked_mul(lp.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("witness T width overflow".to_string()))?;
    let num_digits_fold = lp.num_digits_fold(num_polynomials, field_bits)?;
    let z_pre_count = num_z_segments
        .checked_mul(lp.inner_width())
        .and_then(|n| n.checked_mul(num_digits_fold))
        .ok_or_else(|| AkitaError::InvalidSetup("witness Z width overflow".to_string()))?;
    let r_rows =
        lp.relation_matrix_row_count_for(1, crate::layout::RelationMatrixRowLayout::WithDBlock)?;
    let r_count = r_rows
        .checked_mul(crate::sis::compute_num_digits_full_field(
            field_bits,
            lp.log_basis,
        ))
        .ok_or_else(|| AkitaError::InvalidSetup("witness r-tail width overflow".to_string()))?;

    e_hat_count
        .checked_add(t_hat_count)
        .and_then(|n| n.checked_add(z_pre_count))
        .and_then(|n| n.checked_add(r_count))
        .ok_or_else(|| AkitaError::InvalidSetup("witness width overflow".to_string()))
}

/// Witness ring-element count for a chunked (multi-chunk) or single-chunk layout.
///
/// `num_chunks == 1` delegates to
/// [`intermediate_w_ring_element_count_with_counts_bits`] with `num_public_rows = 1`,
/// so it is byte-identical to the historical single-chunk pricing.
///
/// `num_chunks > 1` prices the multi-chunk witness layout used by the distributed
/// prover: `num_chunks` chunks each holding a partitioned slice of `ê`/`t̂` plus a
/// **replicated full-width** `ẑ`, followed by a single shared `r`-tail. The
/// per-node relations stack *horizontally* (`M = [M_0 | … | M_{num_chunks-1}]`),
/// sharing the same row blocks (concatenation adds columns, not rows) and summing
/// the partial commitments `u_j` into one `u`, so the quotient `r = Σ_j r_j` keeps
/// the **single-machine shape** — its row count is priced with `num_commitments =
/// 1`, unchanged from the single-chunk layout. The **only** extra cost over the
/// single-chunk layout is `(num_chunks - 1) · z_chunk` ring elements (the
/// replicated `ẑ`).
///
/// The exact `ê`/`t̂` live-block prefix is partitioned without padding. Its
/// total width and the shared `r` tail therefore stay unchanged.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when `num_chunks == 0`, `num_chunks > 1`
/// is not a power of two, there are fewer live blocks than chunks, or
/// any width product overflows. Never panics — verifier-reachable through the runtime DP fallback.
pub fn intermediate_w_ring_element_count_for_chunks(
    field_bits: u32,
    lp: &LevelParams,
    num_polynomials: usize,
    num_chunks: usize,
) -> Result<usize, AkitaError> {
    if num_chunks == 0 {
        return Err(AkitaError::InvalidSetup(
            "intermediate_w_ring_element_count_for_chunks: num_chunks must be >= 1".to_string(),
        ));
    }
    if num_chunks == 1 {
        return intermediate_w_ring_element_count_with_counts_bits(
            field_bits,
            lp,
            num_polynomials,
            1,
        );
    }
    if !num_chunks.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "intermediate_w_ring_element_count_for_chunks: num_chunks must be a power of two"
                .to_string(),
        ));
    }
    if lp.num_live_blocks < num_chunks {
        return Err(AkitaError::InvalidSetup(format!(
            "intermediate_w_ring_element_count_for_chunks: num_live_blocks={} smaller than num_chunks={num_chunks}",
            lp.num_live_blocks
        )));
    }
    let overflow = || AkitaError::InvalidSetup("chunked witness width overflow".to_string());
    let single =
        intermediate_w_ring_element_count_with_counts_bits(field_bits, lp, num_polynomials, 1)?;
    let num_digits_fold = lp.num_digits_fold(num_polynomials, field_bits)?;
    let z_chunk = lp
        .inner_width()
        .checked_mul(num_digits_fold)
        .ok_or_else(overflow)?;
    num_chunks
        .checked_sub(1)
        .and_then(|copies| copies.checked_mul(z_chunk))
        .and_then(|extra| single.checked_add(extra))
        .ok_or_else(overflow)
}

/// Parameters for one fold level in the computed schedule.
#[derive(Clone, Debug)]
pub struct FoldStep {
    /// Unified level parameters (ring dimension, Ajtai keys, block geometry,
    /// digit depths, challenge config).
    pub params: LevelParams,
    /// Witness length entering this level.
    pub current_w_len: usize,
    /// Witness length leaving this level.
    pub next_w_len: usize,
    /// Proof bytes for this level.
    pub level_bytes: usize,
}

/// Terminal direct-send step.
#[derive(Clone, Debug)]
pub struct DirectStep {
    /// Witness length entering the direct step.
    pub current_w_len: usize,
    /// Serialized terminal witness payload shape.
    pub witness_shape: CleartextWitnessShape,
    /// Direct witness bytes.
    pub direct_bytes: usize,
}

impl DirectStep {
    /// Active terminal log-basis for segment-typed direct witnesses.
    pub fn log_basis(&self) -> Result<u32, AkitaError> {
        match &self.witness_shape {
            CleartextWitnessShape::SegmentTyped(shape) => Ok(shape.layout.log_basis),
            CleartextWitnessShape::FieldElements(_) => Err(AkitaError::InvalidSetup(
                "terminal direct witness must use the segment-typed encoding".to_string(),
            )),
        }
    }
}

/// A single step in the schedule.
#[derive(Clone, Debug)]
pub enum Step {
    /// Fold through one recursive level.
    Fold(Box<FoldStep>),
    /// Send the terminal witness directly.
    Direct(DirectStep),
}

impl Step {
    /// Construct a fold step while keeping its boxed storage private to the
    /// schedule representation.
    #[must_use]
    pub fn fold(step: FoldStep) -> Self {
        Self::Fold(Box::new(step))
    }
}

/// Complete schedule with step-by-step parameters.
#[derive(Clone, Debug)]
pub struct Schedule {
    /// Ordered proof schedule steps.
    pub steps: Vec<Step>,
    /// Exact total proof bytes for the schedule.
    pub total_bytes: usize,
}

impl Schedule {
    /// Iterate over the fold steps in execution order.
    pub fn fold_steps(&self) -> impl Iterator<Item = &FoldStep> + '_ {
        self.steps.iter().filter_map(|step| match step {
            Step::Fold(fold) => Some(fold.as_ref()),
            Step::Direct(_) => None,
        })
    }

    /// Number of fold levels before the terminal direct step.
    pub fn num_fold_levels(&self) -> usize {
        self.fold_steps().count()
    }

    /// Return the root fold carried by every supported schedule.
    pub fn root_fold(&self) -> Result<&FoldStep, AkitaError> {
        match self.steps.first() {
            Some(Step::Fold(step)) => Ok(step.as_ref()),
            Some(Step::Direct(_)) | None => Err(AkitaError::UnsupportedSchedule(
                "schedule must begin with a root fold".to_string(),
            )),
        }
    }

    /// Mutably borrow the root fold carried by every supported schedule.
    pub fn root_fold_mut(&mut self) -> Result<&mut FoldStep, AkitaError> {
        match self.steps.first_mut() {
            Some(Step::Fold(step)) => Ok(step.as_mut()),
            Some(Step::Direct(_)) | None => Err(AkitaError::UnsupportedSchedule(
                "schedule must begin with a root fold".to_string(),
            )),
        }
    }

    /// Validate protocol-level schedule topology before any witness is interpreted.
    ///
    /// This boundary owns stable step adjacency and grouped-fold shape. Planner
    /// eligibility, setup-slot identity, and proof-object validation remain at
    /// their respective boundaries.
    pub fn validate_structure(&self) -> Result<(), AkitaError> {
        if self.steps.is_empty() {
            return Err(AkitaError::InvalidSetup(
                "schedule must contain at least one step".to_string(),
            ));
        }
        if !matches!(self.steps.first(), Some(Step::Fold(_))) || self.num_fold_levels() < 2 {
            return Err(AkitaError::InvalidSetup(
                "schedule must start with a fold and contain at least two fold levels".to_string(),
            ));
        }

        let last_index = self.steps.len() - 1;
        for (index, step) in self.steps.iter().enumerate() {
            match step {
                Step::Fold(fold) => {
                    if fold.current_w_len == 0 || fold.next_w_len == 0 {
                        return Err(AkitaError::InvalidSetup(
                            "fold witness lengths must be nonzero".to_string(),
                        ));
                    }

                    let Some(successor) = self.steps.get(index + 1) else {
                        return Err(AkitaError::InvalidSetup(
                            "schedule must end with a direct step".to_string(),
                        ));
                    };
                    let successor_w_len = match successor {
                        Step::Fold(next_fold) => next_fold.current_w_len,
                        Step::Direct(direct) => direct.current_w_len,
                    };
                    if fold.next_w_len != successor_w_len {
                        return Err(AkitaError::InvalidSetup(format!(
                            "schedule witness length mismatch between steps {index} and {}",
                            index + 1
                        )));
                    }
                    if fold.params.has_precommitted_groups() && !matches!(successor, Step::Fold(_))
                    {
                        return Err(AkitaError::InvalidSetup(
                            "grouped fold must be followed by another fold".to_string(),
                        ));
                    }

                    if index == 0 && fold.params.setup_prefix.is_some() {
                        return Err(AkitaError::InvalidSetup(
                            "root fold must not carry an incoming setup prefix".to_string(),
                        ));
                    }

                    let successor_is_direct = matches!(successor, Step::Direct(_));
                    if successor_is_direct {
                        if fold.params.setup_contribution_mode != SetupContributionMode::Direct {
                            return Err(AkitaError::InvalidSetup(
                                "terminal fold must use direct setup contribution".to_string(),
                            ));
                        }
                        if fold.params.has_precommitted_groups() {
                            return Err(AkitaError::InvalidSetup(
                                "terminal fold must be scalar".to_string(),
                            ));
                        }
                    }

                    if let Step::Fold(successor_fold) = successor {
                        let successor_carries_setup_prefix_only =
                            successor_fold.params.setup_prefix.is_some()
                                && successor_fold.params.precommitted_groups.is_empty()
                                && successor_fold.params.precommitted_group_count() == 1;

                        match fold.params.setup_contribution_mode {
                            SetupContributionMode::Recursive => {
                                if successor_fold.params.setup_prefix.is_none() {
                                    return Err(AkitaError::InvalidSetup(
                                        "recursive fold successor must carry a setup prefix"
                                            .to_string(),
                                    ));
                                }
                                if !successor_fold.params.precommitted_groups.is_empty()
                                    || successor_fold.params.precommitted_group_count() != 1
                                {
                                    return Err(AkitaError::InvalidSetup(
                                        "recursive fold successor must carry only the setup prefix group"
                                            .to_string(),
                                    ));
                                }
                            }
                            SetupContributionMode::Direct => {
                                if successor_fold.params.setup_prefix.is_some() {
                                    return Err(AkitaError::InvalidSetup(
                                        "direct fold must not forward a setup prefix".to_string(),
                                    ));
                                }
                            }
                        }

                        if successor_carries_setup_prefix_only
                            && fold.params.setup_contribution_mode
                                != SetupContributionMode::Recursive
                        {
                            return Err(AkitaError::InvalidSetup(
                                "setup-prefix successor requires a recursive predecessor"
                                    .to_string(),
                            ));
                        }
                    }
                }
                Step::Direct(direct) => {
                    if index != last_index {
                        return Err(AkitaError::InvalidSetup(
                            "direct step must be the final schedule step".to_string(),
                        ));
                    }
                    if direct.current_w_len == 0 {
                        return Err(AkitaError::InvalidSetup(
                            "direct witness length must be nonzero".to_string(),
                        ));
                    }

                    let CleartextWitnessShape::SegmentTyped(shape) = &direct.witness_shape else {
                        return Err(AkitaError::InvalidSetup(
                            "terminal direct step requires a segment-typed witness".to_string(),
                        ));
                    };
                    if shape.layout.logical_num_elems != direct.current_w_len {
                        return Err(AkitaError::InvalidSetup(
                            "terminal direct witness shape does not match current witness length"
                                .to_string(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Resolve one fold's execution schedule from the static schedule.
    ///
    /// # Errors
    ///
    /// Returns an error if `level` is not a fold step or if the scheduled
    /// successor step cannot provide next-level params.
    pub fn get_execution_schedule(&self, level: usize) -> Result<ExecutionSchedule, AkitaError> {
        let Some(Step::Fold(step)) = self.steps.get(level) else {
            return Err(AkitaError::InvalidSetup(format!(
                "schedule is missing fold step at level {level}"
            )));
        };
        let is_terminal = matches!(self.steps.get(level + 1), Some(Step::Direct(_)));
        let next_witness_binding = if is_terminal {
            None
        } else if matches!(self.steps.get(level + 2), Some(Step::Direct(_))) {
            Some(NextWitnessBindingPolicy::TerminalInnerState)
        } else {
            Some(NextWitnessBindingPolicy::OuterCommitment)
        };
        let next_level_params = scheduled_next_level_params(self, level + 1)?;
        Ok(ExecutionSchedule {
            level,
            current_w_len: step.current_w_len,
            params: step.params.clone(),
            next_params: next_level_params,
            next_w_len: step.next_w_len,
            is_terminal,
            next_witness_binding,
        })
    }

    /// Witness length (field elements) entering the first step, or `None`
    /// when the schedule has no steps.
    pub fn initial_w_len(&self) -> Option<usize> {
        self.steps.first().map(|step| match step {
            Step::Fold(fold) => fold.current_w_len,
            Step::Direct(direct) => direct.current_w_len,
        })
    }

    /// Append the descriptor digest encoding for this effective schedule.
    ///
    /// Kept next to [`Schedule`] so protocol-affecting step field changes are
    /// reviewed with their Fiat-Shamir binding.
    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        push_usize(bytes, self.steps.len());
        for step in &self.steps {
            match step {
                Step::Fold(fold) => {
                    bytes.push(0);
                    fold.params.append_descriptor_bytes(bytes);
                    push_usize(bytes, fold.current_w_len);
                    push_usize(bytes, fold.next_w_len);
                    push_usize(bytes, fold.level_bytes);
                }
                Step::Direct(direct) => {
                    bytes.push(1);
                    push_usize(bytes, direct.current_w_len);
                    append_direct_witness_shape_descriptor_bytes(bytes, &direct.witness_shape);
                    push_usize(bytes, direct.direct_bytes);
                }
            }
        }
        push_usize(bytes, self.total_bytes);
    }
}

fn append_direct_witness_shape_descriptor_bytes(
    bytes: &mut Vec<u8>,
    shape: &CleartextWitnessShape,
) {
    match shape {
        CleartextWitnessShape::FieldElements(coeff_len) => {
            bytes.push(1);
            push_usize(bytes, *coeff_len);
        }
        CleartextWitnessShape::SegmentTyped(shape) => {
            bytes.push(2);
            shape.append_descriptor_bytes(bytes);
        }
    }
}

/// Witness length entering the root fold, in field elements.
pub fn root_current_w_len(lp: &LevelParams) -> usize {
    lp.num_live_blocks
        .checked_mul(lp.num_positions_per_block)
        .and_then(|len| len.checked_mul(lp.ring_dimension))
        .unwrap_or(0)
}

/// Return the terminal direct witness shape from a runtime schedule.
///
/// # Errors
///
/// Returns an error if the schedule does not end in a direct witness handoff.
pub fn schedule_terminal_direct_witness_shape(
    schedule: &Schedule,
) -> Result<&CleartextWitnessShape, AkitaError> {
    match schedule.steps.last() {
        Some(Step::Direct(step)) => Ok(&step.witness_shape),
        Some(Step::Fold(_)) => Err(AkitaError::InvalidSetup(
            "schedule must end in a terminal direct witness step".to_string(),
        )),
        None => Err(AkitaError::InvalidSetup(
            "schedule is missing terminal direct witness step".to_string(),
        )),
    }
}

/// Resolve one scheduled level's active Akita params.
///
/// `Fold` steps return the baked-in `params` set by the planner DP and
/// table materializer. A terminal `Direct(SegmentTyped)` step has no
/// commitment of its own (the cleartext witness is absorbed into the
/// transcript directly), so it ships no `LevelParams`; this function
/// instead returns a [`LevelParams::log_basis_stub`] carrying only the
/// active `log_basis` read off `witness_shape`. The only caller that
/// actually consumes a field of the terminal-Direct successor is the
/// prover's terminal-fold path, which reads `log_basis`.
///
/// # Errors
///
/// Returns an error when `step_index` is outside the schedule or when a
/// recursive schedule transitions into a `Direct(FieldElements)`.
pub fn scheduled_next_level_params(
    schedule: &Schedule,
    step_index: usize,
) -> Result<LevelParams, AkitaError> {
    match schedule.steps.get(step_index) {
        Some(Step::Fold(step)) => Ok(step.params.clone()),
        Some(Step::Direct(step)) => match &step.witness_shape {
            CleartextWitnessShape::SegmentTyped(shape) => {
                Ok(LevelParams::log_basis_stub(shape.layout.log_basis))
            }
            CleartextWitnessShape::FieldElements(_) => Err(AkitaError::InvalidSetup(
                "recursive schedule cannot transition into a field-element direct step".to_string(),
            )),
        },
        None => Err(AkitaError::InvalidSetup(
            "schedule is missing successor step".to_string(),
        )),
    }
}

#[cfg(test)]
#[path = "schedule_tests.rs"]
mod tests;
