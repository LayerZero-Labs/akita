//! Phase D-full setup-claim-reduction wrapper config.
//!
//! [`ClaimReductionCfg`] adapts any production base [`CommitmentConfig`] to
//! enable the book §5.3 setup-side claim-reduction sumcheck via the
//! [`LevelParams::with_setup_claim_reduction`] hook. The wrapper is generic
//! over a `SHRINK` const that names the tiered shrink factor `f` from book
//! §5.4 (line 762, T2 ratio): `SHRINK = 1` keeps the un-tiered design (the
//! pre-Slice-G baseline) and `SHRINK = 8` activates the book's sweet-spot
//! tiered shape that drops the cascade penalty to `|S|/8`.
//!
//! All [`CommitmentConfig`] and [`akita_planner::PlannerConfig`] hooks
//! delegate to `Base`; the only thing the wrapper changes is:
//!
//! 1. Every level-params method routes through
//!    [`apply_claim_reduction`] so the active level enables
//!    [`LevelParams::use_setup_claim_reduction`] (and uses tensor stage-1
//!    challenges, which is the only stage-1 shape the claim-reduction
//!    proof currently supports — see book §5.3 line 558 "stage-1 batched
//!    range + relation under tensor challenges").
//! 2. [`max_setup_matrix_size`](CommitmentConfig::max_setup_matrix_size)
//!    runs a probe planner pass under the wrapper to compute the matrix
//!    envelope the claim-reduction schedule actually needs (which can
//!    exceed `Base`'s envelope because the cascade-aware schedule picks
//!    different `(m, r)` splits). This was the test-local custom logic
//!    promoted to a real preset as called out in the Phase D-full hygiene
//!    list.
//! 3. [`planner_setup_shrink_factor`](akita_planner::PlannerConfig::planner_setup_shrink_factor)
//!    returns the `SHRINK` const, which the planner consults to compute
//!    the `|S|/f` cascade penalty per book §5.4. `SHRINK = 1` is
//!    bit-equivalent to the un-tiered baseline; `SHRINK = 8` activates
//!    the tiered cascade.
//!
//! [`commitment_layout`](CommitmentConfig::commitment_layout) and the
//! related helpers re-run the planner under the wrapper so the returned
//! root layout matches the schedule the prover will actually execute.

use crate::CommitmentConfig;
use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;
use akita_planner::PlannerConfig;
use akita_types::{
    tiered_setup_group_lp, untiered_setup_group_lp, AjtaiRole, AkitaRootBatchSummary,
    AkitaScheduleInputs, AkitaScheduleLookupKey, AkitaSchedulePlan, CommitmentEnvelope,
    DecompositionParams, LevelParams, Schedule, ScheduleProvider, Step, TieredSetupParams,
    WitnessShape,
};
use std::marker::PhantomData;

/// Per Phase D-full Slice G's book-aligned tiered design, [`apply_claim_reduction`]
/// patches a base [`LevelParams`] into the post-Slice-F shape: tensor
/// stage-1 challenges + setup-side claim-reduction sumcheck enabled.
///
/// This is the single LP-modification site every wrapper hook delegates
/// through, so cascade-routing through the planner sees a consistent LP
/// shape regardless of which method built the layout.
fn apply_claim_reduction(lp: LevelParams) -> LevelParams {
    lp.with_tensor_stage1_challenges()
        .with_setup_claim_reduction()
}

/// Phase D-full setup-claim-reduction wrapper over a base
/// [`CommitmentConfig`].
///
/// `SHRINK` is the tiered shrink factor `f` from book §5.4. Production
/// presets use `SHRINK = 8` (sweet spot, `k = f² = 64` chunks, T2 ratio
/// `≈ 1`); `SHRINK = 1` is the un-tiered baseline that exists for
/// regression coverage of the Slice F (`f = 1, k = 1`) routing path.
#[derive(Clone, Copy, Debug, Default)]
pub struct ClaimReductionCfg<Base, const SHRINK: usize>(PhantomData<Base>);

/// Un-tiered claim-reduction preset (`f = 1`). Equivalent to the Slice F
/// baseline; retained for regression coverage of the un-tiered cascade
/// path.
pub type UntieredClaimReductionCfg<Base> = ClaimReductionCfg<Base, 1>;

/// Tiered claim-reduction preset at the book's sweet spot `f = 8`,
/// `k = 64`.
pub type TieredClaimReductionCfg<Base> = ClaimReductionCfg<Base, 8>;

fn cascade_schedule<Cfg>(
    max_num_vars: usize,
    num_vars: usize,
    batch: AkitaRootBatchSummary,
) -> Result<Schedule, AkitaError>
where
    Cfg: CommitmentConfig + PlannerConfig,
{
    akita_planner::find_optimal_schedule_with_max::<Cfg>(
        max_num_vars,
        num_vars,
        WitnessShape::new(
            batch.num_claims,
            batch.num_commitment_groups,
            batch.num_points,
        ),
    )
}

fn cascade_root_layout<Cfg, Base>(
    max_num_vars: usize,
    num_vars: usize,
    batch: AkitaRootBatchSummary,
) -> Result<LevelParams, AkitaError>
where
    Cfg: CommitmentConfig + PlannerConfig,
    Base: CommitmentConfig + PlannerConfig,
{
    let schedule = cascade_schedule::<Cfg>(max_num_vars, num_vars, batch)?;
    match schedule.steps.first() {
        Some(Step::Fold(root_step)) => Ok(root_step.params.clone()),
        Some(Step::Direct(_)) | None => {
            Base::commitment_layout(num_vars).map(apply_claim_reduction)
        }
    }
}

fn claim_reduction_schedule<Base, const SHRINK: usize>(
    max_num_vars: usize,
    num_vars: usize,
    batch: AkitaRootBatchSummary,
) -> Result<Schedule, AkitaError>
where
    Base: CommitmentConfig + PlannerConfig,
{
    cascade_schedule::<ClaimReductionCfg<Base, SHRINK>>(max_num_vars, num_vars, batch)
}

fn claim_reduction_root_layout<Base, const SHRINK: usize>(
    max_num_vars: usize,
    num_vars: usize,
    batch: AkitaRootBatchSummary,
) -> Result<LevelParams, AkitaError>
where
    Base: CommitmentConfig + PlannerConfig,
{
    cascade_root_layout::<ClaimReductionCfg<Base, SHRINK>, Base>(max_num_vars, num_vars, batch)
}

/// Envelope the matrix sizes the wrapper schedule actually visits.
///
/// The wrapper enables `use_setup_claim_reduction`, which makes the
/// planner consider cascade routing (book §5.3 lines 627-660). Under
/// tiered shrinkage (`SHRINK > 1`) the planner may pick a recursive
/// suffix whose per-level `(m, r)` and rank profile is wider than
/// `Base`'s un-tiered envelope. This helper re-runs the planner under
/// the wrapper to pin the post-cascade matrix dimensions.
fn cascade_matrix_size<Cfg>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
    max_num_points: usize,
) -> Result<(usize, usize), AkitaError>
where
    Cfg: CommitmentConfig + PlannerConfig,
{
    let mut max_rows = 1usize;
    let mut max_stride = 1usize;
    let mut visit = |lp: &LevelParams| {
        max_rows = max_rows
            .max(lp.a_key.row_len())
            .max(lp.b_key.row_len())
            .max(lp.d_key.row_len());
        max_stride = max_stride
            .max(lp.inner_width().next_power_of_two())
            .max(lp.outer_width().next_power_of_two())
            .max(lp.d_matrix_width().next_power_of_two());
    };

    let singleton_layout = Cfg::commitment_layout(max_num_vars)?;
    visit(&singleton_layout);
    let batched_layout = if max_num_batched_polys > 1 {
        let batch = AkitaRootBatchSummary::new(
            max_num_batched_polys,
            max_num_batched_polys,
            max_num_points.min(max_num_batched_polys).max(1),
        )?;
        Some((
            batch,
            Cfg::get_params_for_batched_commitment(max_num_vars, max_num_vars, batch)?,
        ))
    } else {
        None
    };
    if let Some((_, layout)) = &batched_layout {
        visit(layout);
    }

    let mut visit_schedule = |schedule: &Schedule| -> Result<(), AkitaError> {
        let mut incoming_setup: Option<(usize, TieredSetupParams)> = None;
        for step in &schedule.steps {
            let Step::Fold(fold) = step else {
                continue;
            };
            visit(&fold.params);
            if let Some((setup_field_len, tier)) = incoming_setup.take() {
                if setup_field_len > 0 {
                    let s_lp = tiered_setup_group_lp(&fold.params, setup_field_len, tier)?;
                    visit(&s_lp);
                    if tier.is_tiered() {
                        let meta_field_len =
                            tier.num_chunks * s_lp.b_key.row_len() * fold.params.ring_dimension;
                        let meta_lp = untiered_setup_group_lp(
                            &fold.params,
                            meta_field_len.next_power_of_two(),
                        )?;
                        visit(&meta_lp);
                    }
                }
            }
            if fold.s_field_len_emitted > 0 {
                let s_lp = tiered_setup_group_lp(
                    &fold.params,
                    fold.s_field_len_emitted,
                    fold.tier_setup_params,
                )?;
                visit(&s_lp);
                if fold.tier_setup_params.is_tiered() {
                    let meta_field_len = fold.tier_setup_params.num_chunks
                        * s_lp.b_key.row_len()
                        * fold.params.ring_dimension;
                    let meta_lp =
                        untiered_setup_group_lp(&fold.params, meta_field_len.next_power_of_two())?;
                    visit(&meta_lp);
                }
            }
            incoming_setup = (fold.s_field_len_emitted > 0)
                .then_some((fold.s_field_len_emitted, fold.tier_setup_params));
        }
        Ok(())
    };

    let singleton = cascade_schedule::<Cfg>(
        max_num_vars,
        max_num_vars,
        AkitaRootBatchSummary::singleton(),
    )?;
    visit_schedule(&singleton)?;

    if let Some((batch, _)) = batched_layout {
        let batched = cascade_schedule::<Cfg>(max_num_vars, max_num_vars, batch)?;
        visit_schedule(&batched)?;
    }

    Ok((max_rows, max_stride))
}

fn claim_reduction_matrix_size<Base, const SHRINK: usize>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
    max_num_points: usize,
) -> Result<(usize, usize), AkitaError>
where
    Base: CommitmentConfig + PlannerConfig,
{
    cascade_matrix_size::<ClaimReductionCfg<Base, SHRINK>>(
        max_num_vars,
        max_num_batched_polys,
        max_num_points,
    )
}

impl<Base, const SHRINK: usize> ScheduleProvider for ClaimReductionCfg<Base, SHRINK>
where
    Base: CommitmentConfig + PlannerConfig,
{
    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        None
    }

    fn schedule_key(key: AkitaScheduleLookupKey) -> String {
        format!("claim-reduction/f={SHRINK}/{}", Base::schedule_key(key))
    }

    fn schedule_plan(
        _key: AkitaScheduleLookupKey,
    ) -> Result<Option<AkitaSchedulePlan>, AkitaError> {
        Ok(None)
    }
}

impl<Base, const SHRINK: usize> PlannerConfig for ClaimReductionCfg<Base, SHRINK>
where
    Base: CommitmentConfig + PlannerConfig,
{
    const PLANNER_D: usize = Base::PLANNER_D;

    fn planner_field_bits() -> u32 {
        Base::planner_field_bits()
    }

    fn planner_stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        Base::planner_stage1_challenge_config(d)
    }

    fn planner_schedule_plan(
        _key: AkitaScheduleLookupKey,
    ) -> Result<Option<AkitaSchedulePlan>, AkitaError> {
        Ok(None)
    }

    fn planner_root_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, AkitaError> {
        Base::planner_root_level_layout_with_log_basis(inputs, log_basis).map(apply_claim_reduction)
    }

    fn planner_current_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, AkitaError> {
        Base::planner_current_level_layout_with_log_basis(inputs, log_basis)
            .map(apply_claim_reduction)
    }

    fn planner_root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, AkitaError> {
        let params = Base::planner_root_level_params_for_layout_with_log_basis(inputs, lp)?;
        Ok(apply_claim_reduction(params))
    }

    fn planner_log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
        Base::planner_log_basis_search_range(inputs)
    }

    fn planner_stage1_prover_weight() -> usize {
        Base::planner_stage1_prover_weight()
    }

    /// Phase D-full Slice G hook: report the size of the shared setup
    /// polynomial `S` so the planner can compute the cascade penalty
    /// `|S|/f` per book §5.4. We delegate to the base config's
    /// `max_setup_matrix_size(_, 1, 1)` because that captures the
    /// shared-matrix envelope the prover/verifier actually allocate.
    fn planner_setup_polynomial_size(max_num_vars: usize) -> usize {
        let (rows, stride) =
            <Self as CommitmentConfig>::max_setup_matrix_size(max_num_vars, 1, 1).unwrap_or((0, 0));
        rows.saturating_mul(stride)
    }

    /// Phase D-full Slice G hook: report the tiered shrink factor `f`
    /// per book §5.4. The planner consults this together with
    /// `planner_setup_polynomial_size` to compute the additive cascade
    /// penalty `w_fold_L + |S|/f`. `SHRINK = 1` is the un-tiered case.
    fn planner_setup_shrink_factor() -> usize {
        SHRINK
    }

    /// Uniform-tier wrapper: the cascade lives at the root only. The
    /// planner force-routes wherever `_at_level > 1`, so reporting
    /// `SHRINK` at every level would request an infinite cascade. The
    /// book §5.4 single-tier sweet spot tiers `S` once at L0 and lets
    /// the recursive levels run un-tiered (or skip routing entirely);
    /// cascade configs that want a second tiered level use
    /// [`ClaimReductionCascadeCfg`].
    fn planner_setup_shrink_factor_at_level(level: usize) -> usize {
        match level {
            0 => SHRINK,
            _ => 1,
        }
    }
}

impl<Base, const SHRINK: usize> CommitmentConfig for ClaimReductionCfg<Base, SHRINK>
where
    Base: CommitmentConfig + PlannerConfig,
{
    type Field = Base::Field;
    type ClaimField = Base::ClaimField;
    type ChallengeField = Base::ChallengeField;
    const D: usize = Base::D;

    fn decomposition() -> DecompositionParams {
        Base::decomposition()
    }

    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        Base::stage1_challenge_config(d)
    }

    fn audited_root_rank(role: AjtaiRole, max_num_vars: usize) -> usize {
        Base::audited_root_rank(role, max_num_vars)
    }

    fn envelope(max_num_vars: usize) -> CommitmentEnvelope {
        Base::envelope(max_num_vars)
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<(usize, usize), AkitaError> {
        // We deliberately do NOT call `Base::max_setup_matrix_size` here.
        // Post audit B-1 (production fp128 presets default CR-on with
        // their own `planner_setup_shrink_factor`), calling the base
        // would re-derive the envelope under the BASE's tier shape,
        // which may differ from `SHRINK` (e.g. base defaults to f=2,
        // wrapper is `UntieredClaimReductionCfg = SHRINK=1`). When the
        // base's tier shape doesn't fit at small NV, the base call
        // errors out and breaks otherwise-valid wrapper configs. The
        // wrapper KNOWS its tier shape (`SHRINK`) and computes the
        // envelope itself via `claim_reduction_matrix_size`, which
        // walks the planner schedule under the wrapper's `_at_level`
        // policy.
        let (cr_rows, cr_stride) = claim_reduction_matrix_size::<Base, SHRINK>(
            max_num_vars,
            max_num_batched_polys,
            max_num_points,
        )?;
        // Tiered routing commits the dim-derived per-chunk view. Its column
        // axis can be wider than the planner's old total-length split,
        // especially for row-thin setup tables.
        let tier_boundary_stride = if SHRINK > 1 {
            cr_stride.saturating_mul(SHRINK.saturating_mul(8))
        } else {
            cr_stride
        };
        Ok((cr_rows, cr_stride.max(tier_boundary_stride)))
    }

    fn level_params_with_log_basis(inputs: AkitaScheduleInputs, log_basis: u32) -> LevelParams {
        apply_claim_reduction(Base::level_params_with_log_basis(inputs, log_basis))
    }

    fn root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, AkitaError> {
        let params = Base::root_level_params_for_layout_with_log_basis(inputs, lp)?;
        Ok(apply_claim_reduction(params))
    }

    fn root_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, AkitaError> {
        Base::root_level_layout_with_log_basis(inputs, log_basis).map(apply_claim_reduction)
    }

    fn log_basis_at_level(inputs: AkitaScheduleInputs) -> u32 {
        Base::log_basis_at_level(inputs)
    }

    fn log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
        Base::log_basis_search_range(inputs)
    }

    fn commitment_layout(max_num_vars: usize) -> Result<LevelParams, AkitaError> {
        claim_reduction_root_layout::<Base, SHRINK>(
            max_num_vars,
            max_num_vars,
            AkitaRootBatchSummary::singleton(),
        )
    }

    fn get_params_for_commitment(
        num_vars: usize,
        num_polys_per_point: usize,
    ) -> Result<LevelParams, AkitaError> {
        let batch = AkitaRootBatchSummary::new(num_polys_per_point, 1, 1)?;
        claim_reduction_root_layout::<Base, SHRINK>(num_vars, num_vars, batch)
    }

    fn get_params_for_batched_commitment(
        max_num_vars: usize,
        num_vars: usize,
        batch: AkitaRootBatchSummary,
    ) -> Result<LevelParams, AkitaError> {
        claim_reduction_root_layout::<Base, SHRINK>(max_num_vars, num_vars, batch)
    }

    fn get_params_for_prove(
        max_num_vars: usize,
        num_vars: usize,
        layout_num_claims: usize,
        batch: AkitaRootBatchSummary,
    ) -> Result<Schedule, AkitaError> {
        if layout_num_claims != batch.num_claims {
            return Err(AkitaError::InvalidSetup(format!(
                "claim-reduction schedule requires layout_num_claims ({layout_num_claims}) to match total claims ({})",
                batch.num_claims
            )));
        }
        claim_reduction_schedule::<Base, SHRINK>(max_num_vars, num_vars, batch)
    }
}

// ---------------------------------------------------------------------
// Cascade preset — per-level tier `f` selection per book §5.8 line 1170.
// ---------------------------------------------------------------------

/// Cascade-aware claim-reduction wrapper that selects a different
/// tiered shrink factor at each recursion level.
///
/// Book §5.8 line 1170 prescribes `f_{L0} = 8` + `f_{L1} = 4` to keep
/// the T2 cascade ratio `≲ 1` across the two levels where the bulk of
/// the asymptotic gain materialises (Table 1141–1158 row "T1+T2 @
/// L0+L1": 16× / 35× / 265× speedup at NV=32/38/44). Levels ≥ 2 fall
/// back to un-tiered (`f = 1`); the cascade has already paid down
/// `|S|` enough by then that further tiering hurts more than it helps.
///
/// Use [`TieredCascadeCfg`] for the book's headline `f_{L0}=8`,
/// `f_{L1}=4` shape; instantiate `ClaimReductionCascadeCfg` directly
/// for experiment configs with different per-level `f`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ClaimReductionCascadeCfg<Base, const F_L0: usize, const F_L1: usize>(PhantomData<Base>);

/// Book §5.8 headline cascade preset: `f_{L0} = 8`, `f_{L1} = 4`.
pub type TieredCascadeCfg<Base> = ClaimReductionCascadeCfg<Base, 8, 4>;

impl<Base, const F_L0: usize, const F_L1: usize> ScheduleProvider
    for ClaimReductionCascadeCfg<Base, F_L0, F_L1>
where
    Base: CommitmentConfig + PlannerConfig,
{
    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        None
    }

    fn schedule_key(key: AkitaScheduleLookupKey) -> String {
        format!(
            "claim-reduction-cascade/f0={F_L0}/f1={F_L1}/{}",
            Base::schedule_key(key)
        )
    }

    fn schedule_plan(
        _key: AkitaScheduleLookupKey,
    ) -> Result<Option<AkitaSchedulePlan>, AkitaError> {
        Ok(None)
    }
}

impl<Base, const F_L0: usize, const F_L1: usize> PlannerConfig
    for ClaimReductionCascadeCfg<Base, F_L0, F_L1>
where
    Base: CommitmentConfig + PlannerConfig,
{
    const PLANNER_D: usize = Base::PLANNER_D;

    fn planner_field_bits() -> u32 {
        Base::planner_field_bits()
    }

    fn planner_stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        Base::planner_stage1_challenge_config(d)
    }

    fn planner_schedule_plan(
        _key: AkitaScheduleLookupKey,
    ) -> Result<Option<AkitaSchedulePlan>, AkitaError> {
        Ok(None)
    }

    fn planner_root_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, AkitaError> {
        Base::planner_root_level_layout_with_log_basis(inputs, log_basis).map(apply_claim_reduction)
    }

    fn planner_current_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, AkitaError> {
        Base::planner_current_level_layout_with_log_basis(inputs, log_basis)
            .map(apply_claim_reduction)
    }

    fn planner_root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, AkitaError> {
        let params = Base::planner_root_level_params_for_layout_with_log_basis(inputs, lp)?;
        Ok(apply_claim_reduction(params))
    }

    fn planner_log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
        Base::planner_log_basis_search_range(inputs)
    }

    fn planner_stage1_prover_weight() -> usize {
        Base::planner_stage1_prover_weight()
    }

    fn planner_setup_polynomial_size(max_num_vars: usize) -> usize {
        let (rows, stride) =
            <Self as CommitmentConfig>::max_setup_matrix_size(max_num_vars, 1, 1).unwrap_or((0, 0));
        rows.saturating_mul(stride)
    }

    /// Uniform-tier shorthand (book §5.4 sweet spot). Cascade configs
    /// should consult [`Self::planner_setup_shrink_factor_at_level`]
    /// instead, which returns the per-level `f`. We report `F_L0` here
    /// so legacy planner sites that still take the uniform value get
    /// the root tier, which is the closest single-value proxy.
    fn planner_setup_shrink_factor() -> usize {
        F_L0
    }

    /// Book §5.8 line 1170 per-level cascade: `f_{L0} = F_L0`,
    /// `f_{L1} = F_L1`, `f_{Lk} = 1` for `k ≥ 2`.
    fn planner_setup_shrink_factor_at_level(level: usize) -> usize {
        match level {
            0 => F_L0,
            1 => F_L1,
            _ => 1,
        }
    }
}

impl<Base, const F_L0: usize, const F_L1: usize> CommitmentConfig
    for ClaimReductionCascadeCfg<Base, F_L0, F_L1>
where
    Base: CommitmentConfig + PlannerConfig,
{
    type Field = Base::Field;
    type ClaimField = Base::ClaimField;
    type ChallengeField = Base::ChallengeField;
    const D: usize = Base::D;

    fn decomposition() -> DecompositionParams {
        Base::decomposition()
    }

    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        Base::stage1_challenge_config(d)
    }

    fn audited_root_rank(role: AjtaiRole, max_num_vars: usize) -> usize {
        Base::audited_root_rank(role, max_num_vars)
    }

    fn envelope(max_num_vars: usize) -> CommitmentEnvelope {
        Base::envelope(max_num_vars)
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<(usize, usize), AkitaError> {
        let (base_rows, base_stride) =
            Base::max_setup_matrix_size(max_num_vars, max_num_batched_polys, max_num_points)?;
        let (cr_rows, cr_stride) =
            cascade_matrix_size::<Self>(max_num_vars, max_num_batched_polys, max_num_points)?;
        // Boundary widening: per-level chunks can produce strides
        // wider than the planner's split. Take the largest of all
        // tier factors so the envelope covers every level.
        let max_tier = F_L0.max(F_L1).max(1);
        let tier_boundary_stride = if max_tier > 1 {
            cr_stride.saturating_mul(max_tier.saturating_mul(8))
        } else {
            cr_stride
        };
        Ok((
            base_rows.max(cr_rows),
            base_stride.max(cr_stride).max(tier_boundary_stride),
        ))
    }

    fn level_params_with_log_basis(inputs: AkitaScheduleInputs, log_basis: u32) -> LevelParams {
        apply_claim_reduction(Base::level_params_with_log_basis(inputs, log_basis))
    }

    fn root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, AkitaError> {
        let params = Base::root_level_params_for_layout_with_log_basis(inputs, lp)?;
        Ok(apply_claim_reduction(params))
    }

    fn root_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, AkitaError> {
        Base::root_level_layout_with_log_basis(inputs, log_basis).map(apply_claim_reduction)
    }

    fn log_basis_at_level(inputs: AkitaScheduleInputs) -> u32 {
        Base::log_basis_at_level(inputs)
    }

    fn log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
        Base::log_basis_search_range(inputs)
    }

    fn commitment_layout(max_num_vars: usize) -> Result<LevelParams, AkitaError> {
        cascade_root_layout::<Self, Base>(
            max_num_vars,
            max_num_vars,
            AkitaRootBatchSummary::singleton(),
        )
    }

    fn get_params_for_commitment(
        num_vars: usize,
        num_polys_per_point: usize,
    ) -> Result<LevelParams, AkitaError> {
        let batch = AkitaRootBatchSummary::new(num_polys_per_point, 1, 1)?;
        cascade_root_layout::<Self, Base>(num_vars, num_vars, batch)
    }

    fn get_params_for_batched_commitment(
        max_num_vars: usize,
        num_vars: usize,
        batch: AkitaRootBatchSummary,
    ) -> Result<LevelParams, AkitaError> {
        cascade_root_layout::<Self, Base>(max_num_vars, num_vars, batch)
    }

    fn get_params_for_prove(
        max_num_vars: usize,
        num_vars: usize,
        layout_num_claims: usize,
        batch: AkitaRootBatchSummary,
    ) -> Result<Schedule, AkitaError> {
        if layout_num_claims != batch.num_claims {
            return Err(AkitaError::InvalidSetup(format!(
                "cascade claim-reduction schedule requires layout_num_claims ({layout_num_claims}) to match total claims ({})",
                batch.num_claims
            )));
        }
        cascade_schedule::<Self>(max_num_vars, num_vars, batch)
    }
}
