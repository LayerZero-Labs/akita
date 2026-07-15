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

#[cfg(test)]
#[path = "schedule_tests.rs"]
mod topology_tests;

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
}

impl ExecutionSchedule {
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
    /// Exact live source ring elements per claim (`N`).
    pub source_ring_len_per_claim: usize,
    /// Power-of-two positions per fold slice (`L`).
    pub fold_position_count: usize,
    /// Exact live fold count (`F`).
    pub live_fold_count: usize,
    /// Power-of-two shard ownership granule (`S`).
    pub shard_granule: usize,
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
            source_ring_len_per_claim: params.source_ring_len_per_claim,
            fold_position_count: params.fold_position_count,
            live_fold_count: params.live_fold_count,
            shard_granule: params.shard_granule,
            fold_challenge_shape: params.fold_challenge_shape,
            log_basis: params.log_basis,
            n_a: params.a_key.row_len(),
            conservative_n_b: params.b_key.row_len(),
        }
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        push_usize(bytes, self.group.num_vars());
        push_usize(bytes, self.group.num_polynomials());
        push_usize(bytes, self.source_ring_len_per_claim);
        push_usize(bytes, self.fold_position_count);
        push_usize(bytes, self.live_fold_count);
        push_usize(bytes, self.shard_granule);
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

    /// Validate that frozen exact fold geometry matches `group.num_vars`.
    pub fn validate_root_geometry(&self, ring_dimension: usize) -> Result<(), AkitaError> {
        let alpha = ring_dimension.trailing_zeros() as usize;
        let Some(source_field_len) = self.source_ring_len_per_claim.checked_mul(ring_dimension)
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
            || self.fold_position_count == 0
            || !self.fold_position_count.is_power_of_two()
            || self.live_fold_count
                != self
                    .source_ring_len_per_claim
                    .div_ceil(self.fold_position_count)
            || self.shard_granule == 0
            || !self.shard_granule.is_power_of_two()
        {
            return Err(AkitaError::InvalidSetup(format!(
                "precommitted group geometry does not match group.num_vars: \
                 N={} L={} F={} S={} alpha={} group.num_vars={}",
                self.source_ring_len_per_claim,
                self.fold_position_count,
                self.live_fold_count,
                self.shard_granule,
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

/// Total ring elements in a recursive witness polynomial for an explicit
/// relation-matrix row layout. The terminal layout drops the D-block from the M-matrix,
/// which shrinks the per-row `r` quotients by `n_d * r_decomp_levels` ring
/// elements relative to the intermediate layout.
pub fn w_ring_element_count_with_counts_for_layout<F: CanonicalField>(
    lp: &LevelParams,
    num_polynomials: usize,
    num_z_segments: usize,
    layout: crate::layout::RelationMatrixRowLayout,
) -> Result<usize, AkitaError> {
    let modulus = detect_field_modulus::<F>();
    let field_bits = 128 - (modulus.saturating_sub(1)).leading_zeros();
    w_ring_element_count_with_counts_for_layout_bits(
        field_bits,
        lp,
        num_polynomials,
        num_z_segments,
        layout,
    )
}

/// Non-generic variant of [`w_ring_element_count_with_counts_for_layout`] for
/// callers that already know the effective field bit width. The planner
/// search uses this to keep its API free of a base-field type parameter.
pub fn w_ring_element_count_with_counts_for_layout_bits(
    field_bits: u32,
    lp: &LevelParams,
    num_polynomials: usize,
    num_z_segments: usize,
    layout: crate::layout::RelationMatrixRowLayout,
) -> Result<usize, AkitaError> {
    lp.require_scalar_level("w_ring_element_count_with_counts_for_layout_bits")?;
    let e_hat_count = num_polynomials
        .checked_mul(lp.live_fold_count)
        .and_then(|n| n.checked_mul(lp.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("witness W width overflow".to_string()))?;
    let t_hat_count = num_polynomials
        .checked_mul(lp.live_fold_count)
        .and_then(|n| n.checked_mul(lp.a_key.row_len()))
        .and_then(|n| n.checked_mul(lp.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("witness T width overflow".to_string()))?;
    let num_digits_fold = lp.num_digits_fold(num_polynomials, field_bits)?;
    let z_pre_count = num_z_segments
        .checked_mul(lp.inner_width())
        .and_then(|n| n.checked_mul(num_digits_fold))
        .ok_or_else(|| AkitaError::InvalidSetup("witness Z width overflow".to_string()))?;
    let r_rows = lp.relation_matrix_row_count_for(1, layout)?;
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
/// [`w_ring_element_count_with_counts_for_layout_bits`] with `num_public_rows = 1`,
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
/// The exact `ê`/`t̂` live-fold prefix is partitioned without padding. Its
/// total width and the shared `r` tail therefore stay unchanged.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when `num_chunks == 0`, `num_chunks > 1`
/// is not a power of two, there are fewer live folds than chunks, or
/// any width product overflows. Never panics — verifier-reachable through the runtime DP fallback.
pub fn w_ring_element_count_for_chunks(
    field_bits: u32,
    lp: &LevelParams,
    num_polynomials: usize,
    layout: crate::layout::RelationMatrixRowLayout,
    num_chunks: usize,
) -> Result<usize, AkitaError> {
    if num_chunks == 0 {
        return Err(AkitaError::InvalidSetup(
            "w_ring_element_count_for_chunks: num_chunks must be >= 1".to_string(),
        ));
    }
    if num_chunks == 1 {
        return w_ring_element_count_with_counts_for_layout_bits(
            field_bits,
            lp,
            num_polynomials,
            1,
            layout,
        );
    }
    if !num_chunks.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "w_ring_element_count_for_chunks: num_chunks must be a power of two".to_string(),
        ));
    }
    if lp.live_fold_count < num_chunks {
        return Err(AkitaError::InvalidSetup(format!(
            "w_ring_element_count_for_chunks: live_fold_count={} smaller than num_chunks={num_chunks}",
            lp.live_fold_count
        )));
    }
    let overflow = || AkitaError::InvalidSetup("chunked witness width overflow".to_string());
    let single = w_ring_element_count_with_counts_for_layout_bits(
        field_bits,
        lp,
        num_polynomials,
        1,
        layout,
    )?;
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
    /// Root commit layout for root-direct schedules (schedule starts
    /// with this `Direct`, `witness_shape = FieldElements`).
    ///
    /// `Some(_)` is the root commit layout — the verifier replays
    /// commitments against it and the transcript binds it through the
    /// per-proof effective-schedule digest (`PlanSection`). `None` is the
    /// *uncommittable* edge: a table-recorded large-`num_vars` entry
    /// whose singleton root layout exceeds the audited SIS floor. The
    /// schedule is intentionally usable for proof-size exploration and
    /// DP planning, but `get_params_for_batched_commitment` rejects it
    /// loudly and `setup_level_params_from_runtime_schedule` returns
    /// an empty list. Don't commit through such a schedule.
    ///
    /// Terminal-direct steps (`witness_shape = SegmentTyped`, schedule
    /// is `[Fold, …, Fold, Direct]`) ship the cleartext witness without
    /// committing — the verifier absorbs the bytes into the transcript
    /// and re-evaluates the witness directly. They always carry
    /// `params = None`. The active `log_basis` lives on
    /// [`Self::witness_shape`]; `scheduled_next_level_params`
    /// synthesizes a [`LevelParams::log_basis_stub`] from it so the
    /// prover's terminal-fold path still receives a `LevelParams`-shaped
    /// successor (only `log_basis` is consulted there).
    pub params: Option<LevelParams>,
}

impl DirectStep {
    /// Active terminal log-basis for segment-typed direct witnesses.
    pub fn log_basis(&self, field_bits: u32) -> u32 {
        match &self.witness_shape {
            CleartextWitnessShape::FieldElements(_) => field_bits,
            CleartextWitnessShape::SegmentTyped(shape) => shape.layout.log_basis,
        }
    }
}

/// A single step in the schedule.
#[derive(Clone, Debug)]
pub enum Step {
    /// Fold through one recursive level.
    Fold(FoldStep),
    /// Send the terminal witness directly.
    Direct(DirectStep),
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
            Step::Fold(fold) => Some(fold),
            Step::Direct(_) => None,
        })
    }

    /// Number of fold levels before the terminal direct step.
    pub fn num_fold_levels(&self) -> usize {
        self.fold_steps().count()
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

                    if index == 0 {
                        let CleartextWitnessShape::FieldElements(witness_len) =
                            &direct.witness_shape
                        else {
                            return Err(AkitaError::InvalidSetup(
                                "root direct step requires a field-element witness".to_string(),
                            ));
                        };
                        if *witness_len != direct.current_w_len {
                            return Err(AkitaError::InvalidSetup(
                                "root direct witness shape does not match current witness length"
                                    .to_string(),
                            ));
                        }
                        if direct
                            .params
                            .as_ref()
                            .is_some_and(LevelParams::has_precommitted_groups)
                        {
                            return Err(AkitaError::InvalidSetup(
                                "root direct step must be scalar".to_string(),
                            ));
                        }
                    } else {
                        let CleartextWitnessShape::SegmentTyped(shape) = &direct.witness_shape
                        else {
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
                        if direct.params.is_some() {
                            return Err(AkitaError::InvalidSetup(
                                "terminal direct step must not carry commitment params".to_string(),
                            ));
                        }
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
        let next_level_params = scheduled_next_level_params(self, level + 1)?;
        Ok(ExecutionSchedule {
            level,
            current_w_len: step.current_w_len,
            params: step.params.clone(),
            next_params: next_level_params,
            next_w_len: step.next_w_len,
            is_terminal,
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
                    // Root-direct commit layout (`Some` for committable root
                    // entries, `None` for terminal-direct handoffs). Binding it
                    // here is what lets the transcript drop the redundant
                    // setup-level `level_params_digest`: the per-proof schedule
                    // digest now pins the root-direct commit params directly.
                    match &direct.params {
                        Some(params) => {
                            bytes.push(1);
                            params.append_descriptor_bytes(bytes);
                        }
                        None => bytes.push(0),
                    }
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
    lp.live_fold_count
        .checked_mul(lp.fold_position_count)
        .and_then(|len| len.checked_mul(lp.ring_dimension))
        .unwrap_or(0)
}

/// Build the root-direct schedule for roots that do not admit a fold step.
///
/// `current_w_len` is the flattened witness length in field elements for a
/// single scalar group (`2^num_vars`). `commit_params` carries the root commit
/// layout that `Cfg::get_params_for_batched_commitment` returns for this
/// schedule shape and must itself be scalar.
///
/// # Errors
///
/// Returns an error if `current_w_len` is zero or `commit_params` carries
/// precommitted groups.
pub fn root_direct_schedule(
    current_w_len: usize,
    commit_params: LevelParams,
) -> Result<Schedule, AkitaError> {
    if current_w_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "root-direct witness length is zero".to_string(),
        ));
    }
    if commit_params.has_precommitted_groups() {
        return Err(AkitaError::InvalidSetup(
            "root direct step must be scalar".to_string(),
        ));
    }
    Ok(Schedule {
        steps: vec![Step::Direct(DirectStep {
            current_w_len,
            witness_shape: CleartextWitnessShape::FieldElements(current_w_len),
            direct_bytes: 0,
            // Root-direct: stores the root commit layout.
            params: Some(commit_params),
        })],
        total_bytes: 0,
    })
}

/// Return the number of fold levels in a runtime schedule.
pub fn schedule_num_fold_levels(schedule: &Schedule) -> usize {
    schedule
        .steps
        .iter()
        .filter(|step| matches!(step, Step::Fold(_)))
        .count()
}

/// Return whether a runtime schedule uses the root-direct fast path.
pub fn schedule_is_root_direct(schedule: &Schedule) -> bool {
    matches!(schedule.steps.first(), Some(Step::Direct(_)))
}

/// Return the root fold step when a runtime schedule starts with one.
pub fn schedule_root_fold_step(schedule: &Schedule) -> Option<&FoldStep> {
    match schedule.steps.first() {
        Some(Step::Fold(step)) => Some(step),
        Some(Step::Direct(_)) | None => None,
    }
}

/// Root commit layout read from the first step of a multi-group runtime schedule.
pub fn multi_group_root_commit_params(schedule: &Schedule) -> Result<LevelParams, AkitaError> {
    match schedule.steps.first() {
        Some(Step::Fold(root_step)) => Ok(root_step.params.clone()),
        Some(Step::Direct(direct)) => direct.params.clone().ok_or_else(|| {
            AkitaError::InvalidSetup(
                "multi-group root-direct schedule is missing commit params".to_string(),
            )
        }),
        None => Err(AkitaError::InvalidSetup(
            "multi-group schedule has no steps".to_string(),
        )),
    }
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
/// recursive schedule transitions into a `Direct(FieldElements)` (only
/// the *first* step of a root-direct schedule may carry that shape).
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
