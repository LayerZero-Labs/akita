//! Runtime schedule shapes shared by configs, prover, verifier, and planner.

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

    /// Reject any fold level that combines precommitted groups with multi-chunk
    /// witness layout.
    pub fn reject_multi_group_multi_chunk(&self, context: &str) -> Result<(), AkitaError> {
        for fold in self.fold_steps() {
            fold.params.reject_multi_group_multi_chunk(context)?;
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
/// `current_w_len` is the flattened witness length in field elements (for a
/// singleton group, `2^num_vars`; for multi-group batches, the per-group hypercube
/// sizes summed over polynomials). `commit_params` carries the root commit
/// layout that `Cfg::get_params_for_batched_commitment` returns for this
/// schedule shape.
///
/// # Errors
///
/// Returns an error if `current_w_len` is zero.
pub fn root_direct_schedule(
    current_w_len: usize,
    commit_params: LevelParams,
) -> Result<Schedule, AkitaError> {
    if current_w_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "root-direct witness length is zero".to_string(),
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
mod tests {
    use super::*;
    use crate::golomb_rice::golomb_rice_encode_vec;
    use crate::proof::{segment_typed_witness_shape, SegmentTypedWitness};
    use crate::tail_golomb_rice_z_params;
    use crate::{
        direct_witness_bytes, extension_opening_reduction_proof_bytes, level_proof_bytes,
        stage1_tree_stage_shapes, sumcheck_rounds, AkitaBatchedRootProof,
        AkitaIntermediateStage2Proof, AkitaLevelProof, AkitaStage1Proof, AkitaStage1StageProof,
        AkitaStage2Proof, CleartextWitnessProof, ExtensionOpeningReductionProof,
        RelationMatrixRowLayout, RingVec, SisModulusFamily, TerminalLevelProof,
        EXTENSION_OPENING_REDUCTION_DEGREE,
    };
    use akita_algebra::CyclotomicRing;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::{AkitaError, CanonicalField, FieldCore, Prime128OffsetA7F7};
    use akita_serialization::{AkitaSerialize, Compress};
    use akita_sumcheck::EqFactoredUniPoly;
    use akita_sumcheck::{CompressedUniPoly, EqFactoredSumcheckProof, SumcheckProof};

    type F = Prime128OffsetA7F7;

    #[test]
    fn chunked_witness_count_matches_chunk_layout_arithmetic() {
        const D: usize = 64;
        let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
        // live_fold_count = 2^3 = 8, divisible by {1, 2, 4, 8}.
        let lp =
            LevelParams::params_only(SisModulusFamily::Q128, D, 3, 2, 2, 2, fold_challenge_config)
                .with_decomp(4, 32, 2, 2)
                .unwrap();
        let field_bits = 128u32;
        let num_poly = 3usize;

        for layout in [
            RelationMatrixRowLayout::WithDBlock,
            RelationMatrixRowLayout::WithoutDBlock,
        ] {
            let single = w_ring_element_count_with_counts_for_layout_bits(
                field_bits, &lp, num_poly, 1, layout,
            )
            .unwrap();
            // num_chunks = 1 must be byte-identical to the single-chunk delegate.
            assert_eq!(
                w_ring_element_count_for_chunks(field_bits, &lp, num_poly, layout, 1).unwrap(),
                single
            );

            let z_pre = lp.inner_width() * lp.num_digits_fold(num_poly, field_bits).unwrap();
            for num_chunks in [2usize, 4, 8] {
                let chunked =
                    w_ring_element_count_for_chunks(field_bits, &lp, num_poly, layout, num_chunks)
                        .unwrap();
                // ê/t̂ totals are unchanged (partitioned), and the shared r-tail is
                // a single summed quotient that keeps the single-machine row count
                // (num_commitments = 1). So the ONLY growth is the replicated ẑ:
                // (num_chunks - 1) full-width copies.
                assert_eq!(chunked, single + (num_chunks - 1) * z_pre);
                assert!(chunked > single, "chunked layout must grow vs single chunk");
            }
        }
    }

    #[test]
    fn chunked_witness_count_rejects_invalid_chunk_counts() {
        const D: usize = 64;
        let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
        // live_fold_count = 2^3 = 8.
        let lp =
            LevelParams::params_only(SisModulusFamily::Q128, D, 3, 2, 2, 2, fold_challenge_config)
                .with_decomp(4, 32, 2, 2)
                .unwrap();
        // Non-power-of-two chunk count.
        assert!(matches!(
            w_ring_element_count_for_chunks(128, &lp, 1, RelationMatrixRowLayout::WithDBlock, 6),
            Err(AkitaError::InvalidSetup(_))
        ));
        // num_chunks does not divide live_fold_count (8 % 16 != 0).
        assert!(matches!(
            w_ring_element_count_for_chunks(128, &lp, 1, RelationMatrixRowLayout::WithDBlock, 16),
            Err(AkitaError::InvalidSetup(_))
        ));
        // Zero chunks.
        assert!(matches!(
            w_ring_element_count_for_chunks(128, &lp, 1, RelationMatrixRowLayout::WithDBlock, 0),
            Err(AkitaError::InvalidSetup(_))
        ));
    }

    fn segment_typed_final_witness(
        lp: &LevelParams,
        num_claims: usize,
    ) -> (CleartextWitnessProof<F>, CleartextWitnessShape) {
        let field_bits = F::modulus_bits();
        let shape = segment_typed_witness_shape(lp, field_bits, num_claims, num_claims, 1, 1)
            .expect("segment-typed witness shape");
        let CleartextWitnessShape::SegmentTyped(ref segment_shape) = shape else {
            panic!("expected segment-typed witness shape");
        };
        let layout = segment_shape.layout;
        let (rice_low_bits, zigzag_w) =
            tail_golomb_rice_z_params(lp, num_claims).expect("golomb z params");
        let z_payload =
            golomb_rice_encode_vec(&vec![0i64; layout.z_coords], rice_low_bits, zigzag_w)
                .expect("encode zero z segment");
        let witness = SegmentTypedWitness {
            layout,
            z_payload,
            e_fields: RingVec::from_coeffs(vec![F::zero(); layout.e_field_elems]),
            t_fields: RingVec::from_coeffs(vec![F::zero(); layout.t_field_elems]),
            r_fields: RingVec::from_coeffs(vec![F::zero(); layout.r_field_elems]),
        };
        (CleartextWitnessProof::SegmentTyped(witness), shape)
    }

    #[test]
    fn root_direct_schedule_uses_field_element_payload() {
        let dummy_commit_params = LevelParams::params_only(
            crate::SisModulusFamily::Q128,
            64,
            3,
            1,
            1,
            1,
            akita_challenges::SparseChallengeConfig::pm1_only(1),
        );
        let schedule =
            root_direct_schedule(8, dummy_commit_params.clone()).expect("root-direct schedule");
        assert_eq!(schedule.total_bytes, 0);

        let [Step::Direct(step)] = schedule.steps.as_slice() else {
            panic!("root-direct schedule should contain one direct step");
        };
        assert_eq!(step.current_w_len, 8);
        assert_eq!(step.witness_shape, CleartextWitnessShape::FieldElements(8));
        assert_eq!(step.direct_bytes, 0);
        assert_eq!(step.params.as_ref(), Some(&dummy_commit_params));
    }

    #[test]
    fn root_direct_schedule_uses_multi_group_witness_len() {
        let layout = OpeningClaimsLayout::from_groups(vec![
            PolynomialGroupLayout::new(2, 1),
            PolynomialGroupLayout::new(3, 2),
            PolynomialGroupLayout::new(4, 1),
        ])
        .expect("multi-group layout");
        let witness_len = layout.root_direct_witness_len().expect("witness len");
        assert_eq!(witness_len, 4 + 16 + 16);

        let dummy_commit_params = LevelParams::params_only(
            crate::SisModulusFamily::Q128,
            64,
            3,
            1,
            1,
            1,
            akita_challenges::SparseChallengeConfig::pm1_only(3),
        );
        let schedule =
            root_direct_schedule(witness_len, dummy_commit_params).expect("root-direct schedule");
        let [Step::Direct(step)] = schedule.steps.as_slice() else {
            panic!("root-direct schedule should contain one direct step");
        };
        assert_eq!(step.current_w_len, witness_len);
        assert_eq!(
            step.witness_shape,
            CleartextWitnessShape::FieldElements(witness_len)
        );
    }

    fn dummy_sumcheck<F: FieldCore>(rounds: usize, degree: usize) -> SumcheckProof<F> {
        SumcheckProof {
            round_polys: (0..rounds)
                .map(|_| CompressedUniPoly {
                    coeffs_except_linear_term: vec![F::zero(); degree],
                })
                .collect(),
        }
    }

    fn dummy_eq_factored_sumcheck<F: FieldCore>(
        rounds: usize,
        degree: usize,
    ) -> EqFactoredSumcheckProof<F> {
        EqFactoredSumcheckProof {
            round_polys: (0..rounds)
                .map(|_| EqFactoredUniPoly {
                    coeffs_except_linear_term: vec![
                        F::zero();
                        EqFactoredUniPoly::<F>::stored_coeff_count_for_degree(degree)
                    ],
                })
                .collect(),
        }
    }

    fn dummy_stage1_proof<F: FieldCore>(rounds: usize, b: usize) -> AkitaStage1Proof<F> {
        AkitaStage1Proof {
            stages: stage1_tree_stage_shapes(rounds, b)
                .into_iter()
                .map(|shape| AkitaStage1StageProof {
                    sumcheck_proof: dummy_eq_factored_sumcheck(rounds, shape.sumcheck_proof.1),
                    child_claims: vec![F::zero(); shape.child_claims],
                })
                .collect(),
            s_claim: F::zero(),
        }
    }

    fn exact_level_proof_bytes<F: FieldCore + CanonicalField + AkitaSerialize>(
        lp: &LevelParams,
        next_lp: &LevelParams,
        next_w_len: usize,
    ) -> Result<usize, AkitaError> {
        let current_coeffs = lp
            .d_key
            .row_len()
            .checked_mul(lp.ring_dimension)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("recursive proof sizing overflow".to_string())
            })?;
        let next_commit_coeffs = next_lp
            .b_key
            .row_len()
            .checked_mul(next_lp.ring_dimension)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("recursive proof sizing overflow".to_string())
            })?;
        let rounds = sumcheck_rounds(lp.ring_dimension, next_w_len);
        let b = 1usize << lp.log_basis;

        let proof = AkitaLevelProof::Intermediate {
            extension_opening_reduction: None,
            v: RingVec::from_coeffs(vec![F::zero(); current_coeffs]),
            fold_grind_nonce: 0,
            stage1: dummy_stage1_proof(rounds, b),
            stage2: AkitaStage2Proof::Intermediate(AkitaIntermediateStage2Proof {
                sumcheck_proof: dummy_sumcheck(rounds, 3),
                next_w_commitment: RingVec::from_coeffs(vec![F::zero(); next_commit_coeffs]),
                next_w_eval: F::zero(),
            }),
            stage3_sumcheck_proof: None,
        };
        Ok(proof.serialized_size(Compress::No))
    }

    #[test]
    fn planned_level_bytes_match_two_stage_payload_at_all_bases() {
        const D: usize = 64;
        let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
        let next_lp =
            LevelParams::params_only(SisModulusFamily::Q128, D, 2, 2, 3, 2, fold_challenge_config);
        let next_w_len = D * 8;

        for log_basis in 2..=6 {
            let lp = LevelParams::params_only(
                SisModulusFamily::Q128,
                D,
                log_basis,
                2,
                2,
                2,
                fold_challenge_config,
            )
            .with_decomp(1, 1, 1, 1)
            .unwrap();
            assert_eq!(
                level_proof_bytes(
                    128,
                    128,
                    &lp,
                    Some(&next_lp),
                    next_w_len,
                    1,
                    RelationMatrixRowLayout::WithDBlock,
                ),
                exact_level_proof_bytes::<F>(&lp, &next_lp, next_w_len).unwrap(),
                "planned level bytes should match the serialized two-stage body at log_basis={log_basis}"
            );
        }
    }

    #[test]
    fn planned_terminal_level_bytes_match_terminal_payload_at_all_bases() {
        const D: usize = 64;
        let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
        let next_w_len = D * 8;
        let num_claims = 3;

        for log_basis in 2..=6 {
            let lp = LevelParams::params_only(
                SisModulusFamily::Q128,
                D,
                log_basis,
                2,
                2,
                2,
                fold_challenge_config,
            )
            .with_decomp(1, 1, 1, 1)
            .unwrap();
            let rounds = sumcheck_rounds(D, next_w_len);

            let (final_witness, witness_shape) = segment_typed_final_witness(&lp, num_claims);
            let final_witness_bytes_runtime = final_witness.serialized_size(Compress::No);
            let terminal_proof = TerminalLevelProof::<F, F>::new_with_extension_opening_reduction(
                None,
                dummy_sumcheck(rounds, 3),
                final_witness,
                0,
            );

            // The planner accounts for the final witness separately
            // (`direct_witness_bytes` on the terminal direct step). Subtract
            // it from the serialized terminal level to compare against
            // `terminal_level_proof_bytes`.
            let serialized_without_witness =
                terminal_proof.serialized_size(Compress::No) - final_witness_bytes_runtime;

            assert_eq!(
                level_proof_bytes(
                    128,
                    128,
                    &lp,
                    None,
                    next_w_len,
                    num_claims,
                    RelationMatrixRowLayout::WithoutDBlock,
                ),
                serialized_without_witness,
                "planned terminal-level bytes should match the serialized terminal body \
                 (less final_witness) at log_basis={log_basis}"
            );

            let scheduled_bytes = direct_witness_bytes(128, &witness_shape);
            assert!(
                scheduled_bytes >= final_witness_bytes_runtime,
                "scheduled direct witness budget must cover serialized segment-typed witness \
                 at log_basis={log_basis}"
            );
        }
    }

    #[test]
    fn planned_batched_root_bytes_match_two_stage_payload_at_all_bases() {
        const D: usize = 64;
        let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
        let next_lp =
            LevelParams::params_only(SisModulusFamily::Q128, D, 2, 2, 3, 2, fold_challenge_config);
        let next_w_len = D * 8;

        for log_basis in 2..=6 {
            let lp = LevelParams::params_only(
                SisModulusFamily::Q128,
                D,
                log_basis,
                2,
                2,
                2,
                fold_challenge_config,
            )
            .with_decomp(1, 1, 1, 1)
            .unwrap();
            let rounds = sumcheck_rounds(D, next_w_len);
            let b = 1usize << log_basis;
            let next_commitment = RingVec::from_ring_elems(&vec![
                CyclotomicRing::<F, D>::zero();
                next_lp.b_key.row_len()
            ])
            .into_compact();
            let level_proof =
                AkitaLevelProof::new_two_stage_many_with_extension_opening_reduction::<D>(
                    None,
                    vec![CyclotomicRing::<F, D>::zero(); lp.d_key.row_len()],
                    dummy_stage1_proof(rounds, b),
                    dummy_sumcheck(rounds, 3),
                    next_commitment,
                    F::zero(),
                );
            let root_proof = AkitaBatchedRootProof::new(level_proof);

            assert_eq!(
                level_proof_bytes(
                    128,
                    128,
                    &lp,
                    Some(&next_lp),
                    next_w_len,
                    1,
                    RelationMatrixRowLayout::WithDBlock,
                ),
                root_proof.serialized_size(Compress::No),
                "planned batched root bytes should match the serialized two-stage body at log_basis={log_basis}"
            );
        }
    }

    #[test]
    fn planned_root_extension_reduction_bytes_match_payload() {
        let extension_width = 4usize;
        let num_claims = 3usize;
        let opening_vars = 12usize;
        let partials = extension_width.saturating_mul(num_claims);
        let reduction = ExtensionOpeningReductionProof {
            partials: vec![F::zero(); partials],
            sumcheck: dummy_sumcheck(
                opening_vars - extension_width.trailing_zeros() as usize,
                EXTENSION_OPENING_REDUCTION_DEGREE,
            ),
        };
        let sumcheck_bytes = reduction.sumcheck.serialized_size(Compress::No);

        assert_eq!(
            extension_opening_reduction_proof_bytes(128, partials, opening_vars, extension_width)
                .unwrap(),
            reduction
                .partials
                .iter()
                .map(|partial| partial.serialized_size(Compress::No))
                .sum::<usize>()
                + sumcheck_bytes,
            "planned root EOR bytes should match the headerless serialized payload"
        );
    }

    #[test]
    fn from_layout_accepts_scalar_layout() {
        let layout = OpeningClaimsLayout::new(4, 2).expect("scalar layout");
        let key = AkitaScheduleLookupKey::from_layout::<NoPrecommitSource>(&layout)
            .expect("scalar layout lookup");
        assert_eq!(key.final_group, PolynomialGroupLayout::new(4, 2));
        assert!(key.precommitteds.is_empty());
        assert_eq!(key.num_commitment_groups(), 1);
    }

    struct NoPrecommitSource;

    impl ScheduleKeyPrecommitSource for NoPrecommitSource {
        fn precommitted_group_params(
            _group: PolynomialGroupLayout,
        ) -> Result<PrecommittedGroupParams, AkitaError> {
            Err(AkitaError::InvalidSetup(
                "NoPrecommitSource is only valid for scalar layouts".to_string(),
            ))
        }
    }

    #[test]
    fn validate_rejects_zero_dimensions() {
        assert!(
            AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(0, 1))
                .validate()
                .is_err()
        );
        assert!(
            AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(20, 0))
                .validate()
                .is_err()
        );
        assert!(
            AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(20, 4))
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn group_batch_key_rejects_precommitted_num_vars_above_main() {
        let multi_group_key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(20, 3),
            precommitteds: vec![PrecommittedGroupParams {
                group: PolynomialGroupLayout::new(24, 1),
                source_ring_len_per_claim: 1usize << 18,
                fold_position_count: 16,
                live_fold_count: 1usize << 14,
                shard_granule: 1,
                fold_challenge_shape: akita_challenges::TensorChallengeShape::Flat,
                log_basis: 2,
                n_a: 3,
                conservative_n_b: 4,
            }],
        };

        let err = multi_group_key
            .validate()
            .expect_err("precommitted groups above the main num_vars must be rejected");
        assert!(matches!(err, AkitaError::InvalidInput(_)));
    }

    #[test]
    fn group_batch_key_rejects_precommitted_num_vars_above_half_main() {
        let multi_group_key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(20, 3),
            precommitteds: vec![PrecommittedGroupParams {
                group: PolynomialGroupLayout::new(12, 1),
                source_ring_len_per_claim: 64,
                fold_position_count: 16,
                live_fold_count: 4,
                shard_granule: 1,
                fold_challenge_shape: akita_challenges::TensorChallengeShape::Flat,
                log_basis: 2,
                n_a: 3,
                conservative_n_b: 4,
            }],
        };

        multi_group_key
            .validate()
            .expect_err("precommitted groups above half the main key must be rejected");
    }

    #[test]
    fn group_batch_key_allows_mixed_polynomial_counts() {
        let multi_group_key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(20, 3),
            precommitteds: vec![PrecommittedGroupParams {
                group: PolynomialGroupLayout::new(10, 1),
                source_ring_len_per_claim: 16,
                fold_position_count: 4,
                live_fold_count: 4,
                shard_granule: 1,
                fold_challenge_shape: akita_challenges::TensorChallengeShape::Flat,
                log_basis: 2,
                n_a: 3,
                conservative_n_b: 4,
            }],
        };

        multi_group_key
            .validate()
            .expect("unequal K_g is allowed for a supported precommitted dimension");
        assert_eq!(multi_group_key.num_commitment_groups(), 2);
    }

    #[test]
    fn validate_frozen_precommit_rejects_geometry_mismatch() {
        let layout = PrecommittedGroupParams {
            group: PolynomialGroupLayout::new(20, 1),
            source_ring_len_per_claim: 1,
            fold_position_count: 16,
            live_fold_count: 1,
            shard_granule: 1,
            fold_challenge_shape: akita_challenges::TensorChallengeShape::Flat,
            log_basis: 2,
            n_a: 3,
            conservative_n_b: 4,
        };
        let err = layout
            .validate_frozen_precommit(64)
            .expect_err("geometry must match num_vars");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }
}
