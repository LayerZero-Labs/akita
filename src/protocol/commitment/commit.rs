//! Ring-native §4.1 commitment core implementation.

#[cfg(test)]
use super::config::validate_and_derive_layout;
use super::config::{ensure_block_layout, ensure_layout_supported_num_vars};
use super::onehot::{inner_ajtai_onehot_wide, map_onehot_to_sparse_blocks};
use super::schedule::HachiScheduleInputs;
use super::schedule::{
    hachi_root_runtime_plan_from_root_layout, HachiRootBatchSummary, HachiScheduleLookupKey,
};
use super::scheme::{CommitWitness, RingCommitmentScheme};
use super::types::RingCommitment;
use super::utils::crt_ntt::build_ntt_slot;
use super::utils::linear::{
    decompose_rows_i8_into, mat_vec_mul_ntt_i8_dense, mat_vec_mul_ntt_i8_dense_single_row,
    mat_vec_mul_ntt_single_i8,
};
use super::utils::matrix::{
    derive_public_matrix_flat, sample_public_matrix_seed, PublicMatrixSeed,
};
use super::CommitmentConfig;
use crate::algebra::fields::wide::HasWide;
use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::planner::digit_math::compute_num_digits_fold_with_claims;
use crate::primitives::serialization::Valid;
use crate::protocol::commitment_scheme::should_stop_folding;
use crate::protocol::hachi_poly_ops::OneHotIndex;
use crate::protocol::params::{AjtaiKeyParams, LevelParams};
use crate::protocol::preprocessing::{
    HachiExpandedSetup, HachiProverSetup, HachiSetupSeed, HachiVerifierSetup,
};
use crate::protocol::proof::FlatDigitBlocks;
use crate::protocol::ring_switch::w_ring_element_count;
use crate::{CanonicalField, FieldCore, FieldSampling};
use std::sync::Arc;

// Unused re-import anchor so that existing `use crate::protocol::commitment::HachiSetupSeed`
// paths continue to work. The canonical home is `crate::protocol::preprocessing`.

pub(crate) fn root_current_w_len<const D: usize>(lp: &LevelParams) -> usize {
    lp.num_blocks
        .checked_mul(lp.block_len)
        .and_then(|len| len.checked_mul(D))
        .unwrap_or(0)
}

#[derive(Debug, Clone, Copy, Default)]
struct LayoutChainStats {
    max_inner_width: usize,
    max_outer_width: usize,
    max_d_matrix_width: usize,
    max_n_a: usize,
    max_n_b: usize,
    max_n_d: usize,
    max_r_vars: usize,
    max_num_digits_open: usize,
    max_num_digits_fold: usize,
    max_log_basis: u32,
}

impl LayoutChainStats {
    fn include(&mut self, lp: &LevelParams) {
        self.max_inner_width = self.max_inner_width.max(lp.inner_width());
        self.max_outer_width = self.max_outer_width.max(lp.outer_width());
        self.max_d_matrix_width = self.max_d_matrix_width.max(lp.d_matrix_width());
        self.max_r_vars = self.max_r_vars.max(lp.r_vars);
        self.max_num_digits_open = self.max_num_digits_open.max(lp.num_digits_open);
        self.max_num_digits_fold = self.max_num_digits_fold.max(lp.num_digits_fold);
        self.max_log_basis = self.max_log_basis.max(lp.log_basis);
        self.max_n_a = self.max_n_a.max(lp.a_key.row_len());
        self.max_n_b = self.max_n_b.max(lp.b_key.row_len());
        self.max_n_d = self.max_n_d.max(lp.d_key.row_len());
    }
}

pub(crate) fn scale_batched_root_layout<Cfg, const D: usize>(
    max_num_vars: usize,
    root_lp: &LevelParams,
    num_claims: usize,
) -> Result<LevelParams, HachiError>
where
    Cfg: CommitmentConfig,
{
    if num_claims == 0 {
        return Err(HachiError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }

    let root_inputs = HachiScheduleInputs {
        max_num_vars,
        level: 0,
        current_w_len: root_current_w_len::<D>(root_lp),
    };
    let root_stage1_config =
        Cfg::stage1_challenge_config(Cfg::d_at_level(0, root_inputs.current_w_len));
    let mut scaled = root_lp.clone();
    let d = scaled.ring_dimension;
    // Root batching concatenates the outer binding roles across claims.
    // The inner A role stays per-claim, so only B and D widen here.
    scaled.b_key = AjtaiKeyParams::new_unchecked(
        scaled.b_key.row_len(),
        root_lp
            .b_key
            .col_len()
            .checked_mul(num_claims)
            .ok_or_else(|| HachiError::InvalidSetup("batched outer width overflow".to_string()))?,
        scaled.b_key.collision_inf(),
        d,
    );
    scaled.d_key = AjtaiKeyParams::new_unchecked(
        scaled.d_key.row_len(),
        root_lp
            .d_key
            .col_len()
            .checked_mul(num_claims)
            .ok_or_else(|| HachiError::InvalidSetup("batched D width overflow".to_string()))?,
        scaled.d_key.collision_inf(),
        d,
    );
    // `num_claims` amplifies the folded root witness bound. Public point count
    // is handled later when sizing the explicit y rows and serialized y_rings.
    scaled.num_digits_fold = root_lp
        .num_digits_fold
        .max(compute_num_digits_fold_with_claims(
            root_lp.r_vars,
            root_stage1_config.l1_mass(),
            root_lp.log_basis,
            num_claims,
        ));
    Ok(scaled)
}

/// Shared batched-root derivation used by planner and runtime.
///
/// `level_lp` is the batch-effective root layout that widens the `B/D` widths
/// and fold-digit budget for the concrete root batch. `root_lp` is the active
/// root parameter set derived against that widened layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BatchedRootLevelDerivation {
    pub level_lp: LevelParams,
    pub root_lp: LevelParams,
}

pub(crate) fn derive_batched_root_level_derivation<Cfg, const D: usize>(
    max_num_vars: usize,
    root_lp: &LevelParams,
    num_claims: usize,
) -> Result<BatchedRootLevelDerivation, HachiError>
where
    Cfg: CommitmentConfig,
{
    let inputs = HachiScheduleInputs {
        max_num_vars,
        level: 0,
        current_w_len: root_current_w_len::<D>(root_lp),
    };
    let level_lp = scale_batched_root_layout::<Cfg, D>(max_num_vars, root_lp, num_claims)?;
    let root_lp = Cfg::root_level_params_for_layout_with_log_basis(inputs, &level_lp)?;
    Ok(BatchedRootLevelDerivation { level_lp, root_lp })
}

/// Planner-derived batched root split parameters.
pub(crate) struct BatchedRootSplit {
    /// Per-polynomial root params/layout for the chosen `(log_basis, m, r)`.
    pub params: LevelParams,
    /// Batched fold digits (from the planner's candidate layout).
    /// May differ from `params.num_digits_fold` when batching amplifies
    /// the fold bound.
    pub num_digits_fold_batched: usize,
}

/// Extract `BatchedRootSplit` from a pre-computed `HachiSchedulePlan`'s
/// first fold level, if one exists.
fn split_from_schedule_plan(plan: &super::schedule::HachiSchedulePlan) -> Option<BatchedRootSplit> {
    use super::config::compute_num_digits_fold;

    let root_level = plan.fold_levels().next()?;
    let per_poly_fold = compute_num_digits_fold(
        root_level.lp.r_vars,
        root_level.lp.challenge_l1_mass(),
        root_level.lp.log_basis,
    );
    let mut lp = root_level.lp.clone();
    let batched_fold = lp.num_digits_fold;
    lp.num_digits_fold = per_poly_fold;
    Some(BatchedRootSplit {
        params: lp,
        num_digits_fold_batched: batched_fold,
    })
}

fn fallback_batched_root_split<Cfg, const D: usize>(
    max_num_vars: usize,
    num_claims: usize,
) -> Result<BatchedRootSplit, HachiError>
where
    Cfg: CommitmentConfig,
{
    let root_lp = Cfg::commitment_layout(max_num_vars)?;
    let scaled_lp = scale_batched_root_layout::<Cfg, D>(max_num_vars, &root_lp, num_claims)?;
    Ok(BatchedRootSplit {
        num_digits_fold_batched: scaled_lp.num_digits_fold,
        params: root_lp,
    })
}

fn per_poly_root_split_from_batched_level(
    root_lp: &LevelParams,
    per_poly_fold: usize,
    num_claims: usize,
) -> Result<BatchedRootSplit, HachiError> {
    if num_claims == 0 {
        return Err(HachiError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }
    let b_cols = root_lp
        .b_key
        .col_len()
        .checked_div(num_claims)
        .filter(|cols| cols.saturating_mul(num_claims) == root_lp.b_key.col_len())
        .ok_or_else(|| {
            HachiError::InvalidSetup(format!(
                "batched root B width {} is not divisible by num_claims={num_claims}",
                root_lp.b_key.col_len()
            ))
        })?;
    let d_cols = root_lp
        .d_key
        .col_len()
        .checked_div(num_claims)
        .filter(|cols| cols.saturating_mul(num_claims) == root_lp.d_key.col_len())
        .ok_or_else(|| {
            HachiError::InvalidSetup(format!(
                "batched root D width {} is not divisible by num_claims={num_claims}",
                root_lp.d_key.col_len()
            ))
        })?;
    let d = root_lp.ring_dimension;
    let mut lp = root_lp.clone();
    let batched_fold = lp.num_digits_fold;
    lp.b_key =
        AjtaiKeyParams::new_unchecked(lp.b_key.row_len(), b_cols, lp.b_key.collision_inf(), d);
    lp.d_key =
        AjtaiKeyParams::new_unchecked(lp.d_key.row_len(), d_cols, lp.d_key.collision_inf(), d);
    lp.num_digits_fold = per_poly_fold;
    Ok(BatchedRootSplit {
        params: lp,
        num_digits_fold_batched: batched_fold,
    })
}

/// Find the optimal `(log_basis, m, r)` triple for a batched root opening.
///
/// First checks the pre-computed generated tables.  Falls back to the DP
/// planner only when no table entry exists.
pub(crate) fn optimal_root_batch_split<Cfg, const D: usize>(
    max_num_vars: usize,
    num_claims: usize,
) -> Result<BatchedRootSplit, HachiError>
where
    Cfg: CommitmentConfig,
{
    let lookup_key = HachiScheduleLookupKey::with_batch(
        max_num_vars,
        max_num_vars,
        num_claims,
        HachiRootBatchSummary::new(num_claims, 1, 1)?,
    );
    if let Some(plan) = Cfg::schedule_plan(lookup_key)? {
        if let Some(split) = split_from_schedule_plan(&plan) {
            tracing::info!(
                max_num_vars,
                num_claims,
                total_bytes = plan.exact_proof_bytes,
                root_m = split.params.log_block_len(),
                root_r = split.params.log_num_blocks(),
                root_lb = split.params.log_basis,
                "batched root split: read from pre-computed table"
            );
            return Ok(split);
        }
        let split = fallback_batched_root_split::<Cfg, D>(max_num_vars, num_claims)?;
        tracing::info!(
            max_num_vars,
            num_claims,
            "batched root split: schedule is direct-only, falling back to config root layout"
        );
        return Ok(split);
    }

    use crate::planner::schedule_params::{find_optimal_batched_schedule, BatchConfig, Step};

    let batch = BatchConfig {
        num_claims,
        num_commitment_groups: 1,
        num_points: 1,
    };
    let schedule = find_optimal_batched_schedule::<Cfg, D>(max_num_vars, batch)?;

    let root_step = match schedule.steps.first() {
        Some(Step::Fold(step)) => step,
        _ => return fallback_batched_root_split::<Cfg, D>(max_num_vars, num_claims),
    };

    let split = per_poly_root_split_from_batched_level(
        &root_step.params,
        root_step.delta_fold_per_poly,
        num_claims,
    )?;

    tracing::info!(
        max_num_vars,
        num_claims,
        total_bytes = schedule.total_bytes,
        root_m = split.params.log_block_len(),
        root_r = split.params.log_num_blocks(),
        root_lb = split.params.log_basis,
        "batched root split: computed from scratch by DP planner (no pre-computed table)"
    );

    Ok(split)
}

pub(crate) fn root_batched_layout<Cfg, const D: usize>(
    max_num_vars: usize,
    root_lp: &LevelParams,
    max_num_batched_polys: usize,
) -> Result<LevelParams, HachiError>
where
    Cfg: CommitmentConfig,
{
    let optimized_root_lp = if max_num_batched_polys > 1 {
        let split = optimal_root_batch_split::<Cfg, D>(max_num_vars, max_num_batched_polys)?;
        let mut lp = split.params.clone();
        lp.num_digits_fold = split.num_digits_fold_batched;
        lp
    } else {
        root_lp.clone()
    };
    scale_batched_root_layout::<Cfg, D>(max_num_vars, &optimized_root_lp, max_num_batched_polys)
}

/// Derive the per-polynomial commitment layout optimized for a batch of
/// `num_claims` polynomials with `max_num_vars` variables.
///
/// When `num_claims <= 1` this returns the singleton layout from
/// [`CommitmentConfig::commitment_layout`]. For larger batches the
/// `m_vars`/`r_vars` split is optimized to minimize proof size.
///
/// # Errors
///
/// Returns an error if the layout parameters overflow or are invalid.
pub fn hachi_batched_root_layout<Cfg, const D: usize>(
    max_num_vars: usize,
    num_claims: usize,
) -> Result<LevelParams, HachiError>
where
    Cfg: CommitmentConfig,
{
    if num_claims <= 1 {
        return Cfg::commitment_layout(max_num_vars);
    }

    let split = optimal_root_batch_split::<Cfg, D>(max_num_vars, num_claims)?;
    Ok(split.params)
}

fn scan_layout_chain<F, const D: usize, Cfg>(
    max_num_vars: usize,
    root_lp: &LevelParams,
    max_num_batched_polys: usize,
) -> Result<LayoutChainStats, HachiError>
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig<Field = F>,
{
    let mut stats = LayoutChainStats::default();
    let batched_root_lp =
        root_batched_layout::<Cfg, D>(max_num_vars, root_lp, max_num_batched_polys)?;
    stats.include(&batched_root_lp);
    if max_num_batched_polys <= 1 {
        let root_inputs = HachiScheduleInputs {
            max_num_vars,
            level: 0,
            current_w_len: root_current_w_len::<D>(root_lp),
        };
        let singleton_lp = Cfg::root_level_params_for_layout_with_log_basis(root_inputs, root_lp)?;
        stats.include(&singleton_lp);
    }

    let singleton_schedule_key = HachiScheduleLookupKey::singleton(max_num_vars, max_num_vars, 1);
    let singleton_plan = Cfg::schedule_plan(singleton_schedule_key)?;

    let can_use_planned_root =
        Cfg::commitment_layout(max_num_vars).is_ok_and(|planned_root| planned_root == *root_lp);
    if can_use_planned_root && max_num_batched_polys == 1 {
        if let Some(plan) = singleton_plan.as_ref() {
            for level in plan.fold_levels().skip(1) {
                stats.include(&level.lp);
            }
            return Ok(stats);
        }
    }

    // Batched roots can hand off a larger recursive witness than the
    // singleton schedule, so the suffix must follow the concrete runtime path.
    let root_plan = hachi_root_runtime_plan_from_root_layout::<Cfg, D>(
        HachiScheduleLookupKey::with_batch(
            max_num_vars,
            max_num_vars,
            max_num_batched_polys,
            HachiRootBatchSummary::new(
                max_num_batched_polys,
                max_num_batched_polys,
                max_num_batched_polys,
            )?,
        ),
        root_lp,
    )?;
    let mut prev_w_len = root_plan
        .inputs
        .current_w_len
        .saturating_mul(root_plan.batch.num_claims);
    let mut level = 1usize;
    let mut current_w_len = root_plan.next_w_len();
    let mut current_lp = root_plan.next_level_params.clone();
    stats.include(&current_lp);

    loop {
        if should_stop_folding(current_w_len, prev_w_len) {
            break;
        }

        let next_w_len = w_ring_element_count::<F>(&current_lp) * current_lp.ring_dimension;
        let next_level = level + 1;
        let next_lp_partial = Cfg::level_params(HachiScheduleInputs {
            max_num_vars,
            level: next_level,
            current_w_len: next_w_len,
        });
        let next_lp =
            super::hachi_recursive_level_layout_from_params::<Cfg>(&next_lp_partial, next_w_len)?;
        stats.include(&next_lp);

        prev_w_len = current_w_len;
        current_w_len = next_w_len;
        current_lp = next_lp;
        level = next_level;
    }

    Ok(stats)
}

/// Concrete §4.1 commitment core.
#[derive(Clone, Copy, Default)]
pub struct HachiCommitmentCore;

impl<F, const D: usize, Cfg> RingCommitmentScheme<F, D, Cfg> for HachiCommitmentCore
where
    F: FieldCore + CanonicalField + FieldSampling + HasWide + Valid,
    Cfg: CommitmentConfig<Field = F>,
{
    type Commitment = RingCommitment<F, D>;

    fn layout(setup: &HachiProverSetup<F, D>) -> Result<LevelParams, HachiError> {
        hachi_batched_root_layout::<Cfg, D>(
            setup.expanded.seed.max_num_vars,
            setup.expanded.seed.max_num_batched_polys,
        )
    }

    #[tracing::instrument(skip_all, name = "RingCommitmentScheme::commit_ring_blocks")]
    fn commit_ring_blocks(
        f_blocks: &[Vec<CyclotomicRing<F, D>>],
        setup: &HachiProverSetup<F, D>,
    ) -> Result<CommitWitness<Self::Commitment, F, D>, HachiError> {
        let root_lp = <Self as RingCommitmentScheme<F, D, Cfg>>::layout(setup)?;
        ensure_layout_supported_num_vars::<D>(setup.expanded.seed.max_num_vars, &root_lp)?;
        ensure_block_layout(f_blocks, &root_lp)?;
        let max_stride = setup.expanded.seed.max_stride;

        let depth_commit = root_lp.num_digits_commit;
        let depth_open = root_lp.num_digits_open;
        let log_basis = root_lp.log_basis;
        let block_slices: Vec<&[CyclotomicRing<F, D>]> =
            f_blocks.iter().map(|b| b.as_slice()).collect();
        let t_hat = if root_lp.a_key.row_len() == 1 {
            let t_single = mat_vec_mul_ntt_i8_dense_single_row(
                &setup.ntt_shared,
                max_stride,
                &block_slices,
                depth_commit,
                log_basis,
            );
            let mut t_hat = FlatDigitBlocks::zeroed(vec![depth_open; t_single.len()])?;
            let dst_blocks = t_hat.split_blocks_mut();
            #[cfg(feature = "parallel")]
            cfg_into_iter!(dst_blocks)
                .zip(cfg_iter!(t_single))
                .for_each(|(dst, t_i)| {
                    decompose_rows_i8_into(std::slice::from_ref(t_i), dst, depth_open, log_basis)
                });
            #[cfg(not(feature = "parallel"))]
            dst_blocks
                .into_iter()
                .zip(t_single.iter())
                .for_each(|(dst, t_i)| {
                    decompose_rows_i8_into(std::slice::from_ref(t_i), dst, depth_open, log_basis)
                });
            t_hat
        } else {
            let t_all = mat_vec_mul_ntt_i8_dense(
                &setup.ntt_shared,
                root_lp.a_key.row_len(),
                max_stride,
                &block_slices,
                depth_commit,
                log_basis,
            );
            let block_sizes: Vec<usize> = t_all.iter().map(|t_i| t_i.len() * depth_open).collect();
            let mut t_hat = FlatDigitBlocks::zeroed(block_sizes)?;
            let dst_blocks = t_hat.split_blocks_mut();
            #[cfg(feature = "parallel")]
            cfg_into_iter!(dst_blocks)
                .zip(cfg_iter!(t_all))
                .for_each(|(dst, t_i)| decompose_rows_i8_into(t_i, dst, depth_open, log_basis));
            #[cfg(not(feature = "parallel"))]
            dst_blocks
                .into_iter()
                .zip(t_all.iter())
                .for_each(|(dst, t_i)| decompose_rows_i8_into(t_i, dst, depth_open, log_basis));
            t_hat
        };

        let u: Vec<CyclotomicRing<F, D>> = mat_vec_mul_ntt_single_i8(
            &setup.ntt_shared,
            root_lp.b_key.row_len(),
            max_stride,
            t_hat.flat_digits(),
        );
        Ok(CommitWitness::new(RingCommitment { u }, t_hat))
    }

    #[tracing::instrument(skip_all, name = "RingCommitmentScheme::commit_onehot")]
    fn commit_onehot<I: OneHotIndex>(
        onehot_k: usize,
        indices: &[Option<I>],
        setup: &HachiProverSetup<F, D>,
    ) -> Result<CommitWitness<Self::Commitment, F, D>, HachiError> {
        let root_lp = <Self as RingCommitmentScheme<F, D, Cfg>>::layout(setup)?;
        ensure_layout_supported_num_vars::<D>(setup.expanded.seed.max_num_vars, &root_lp)?;
        let max_stride = setup.expanded.seed.max_stride;

        let sparse_blocks =
            map_onehot_to_sparse_blocks(onehot_k, indices, root_lp.r_vars, root_lp.m_vars, D)?;

        let depth_commit = root_lp.num_digits_commit;
        let depth_open = root_lp.num_digits_open;
        let log_basis = root_lp.log_basis;
        let zero_block_len = root_lp.a_key.row_len().checked_mul(depth_open).unwrap();
        let a_view = setup
            .expanded
            .shared_matrix
            .ring_view::<D>(root_lp.a_key.row_len(), max_stride);
        let block_len = root_lp.block_len;

        let block_sizes = vec![zero_block_len; sparse_blocks.len()];
        let mut t_hat = FlatDigitBlocks::zeroed(block_sizes)?;
        let dst_blocks = t_hat.split_blocks_mut();
        #[cfg(feature = "parallel")]
        cfg_into_iter!(dst_blocks)
            .zip(cfg_iter!(sparse_blocks))
            .for_each(|(dst, block_entries)| {
                if !block_entries.is_empty() {
                    let mut t_i =
                        inner_ajtai_onehot_wide(&a_view, block_entries, block_len, depth_commit);
                    t_i.truncate(root_lp.a_key.row_len());
                    decompose_rows_i8_into(&t_i, dst, depth_open, log_basis);
                }
            });
        #[cfg(not(feature = "parallel"))]
        dst_blocks
            .into_iter()
            .zip(sparse_blocks.iter())
            .for_each(|(dst, block_entries)| {
                if !block_entries.is_empty() {
                    let mut t_i =
                        inner_ajtai_onehot_wide(&a_view, block_entries, block_len, depth_commit);
                    t_i.truncate(root_lp.a_key.row_len());
                    decompose_rows_i8_into(&t_i, dst, depth_open, log_basis);
                }
            });

        let u: Vec<CyclotomicRing<F, D>> = mat_vec_mul_ntt_single_i8(
            &setup.ntt_shared,
            root_lp.b_key.row_len(),
            max_stride,
            t_hat.flat_digits(),
        );
        Ok(CommitWitness::new(RingCommitment { u }, t_hat))
    }
}

impl HachiCommitmentCore {
    /// Create a setup with a caller-specified layout, bypassing
    /// `CommitmentConfig::commitment_layout`.
    ///
    /// Use this when the desired `(m_vars, r_vars)` split differs from what
    /// the config's heuristic would choose (e.g. mega-polynomial commitments
    /// where each sub-polynomial occupies one block).
    ///
    /// # Errors
    ///
    /// Returns `HachiError` on invalid layout or matrix generation failures.
    pub fn setup_with_layout<F, const D: usize, Cfg>(
        lp: &LevelParams,
    ) -> Result<(HachiProverSetup<F, D>, HachiVerifierSetup<F>), HachiError>
    where
        F: FieldCore + CanonicalField + FieldSampling,
        Cfg: CommitmentConfig<Field = F>,
    {
        let alpha = D.trailing_zeros() as usize;
        let max_num_vars = lp.m_vars + lp.r_vars + alpha;
        let public_matrix_seed = sample_public_matrix_seed();
        Self::setup_with_lp_and_seed::<F, D, Cfg>(lp, max_num_vars, public_matrix_seed)
    }

    /// Create a setup that supports any of the provided runtime layouts.
    ///
    /// This sizes the public matrices from the exact per-layout maxima
    /// (including recursive `w` commitments) instead of inflating through a
    /// synthetic max layout.
    ///
    /// # Errors
    ///
    /// Returns `HachiError` if `layouts` is empty, uses inconsistent
    /// decomposition parameters, or matrix generation fails.
    pub fn setup_with_layouts<F, const D: usize, Cfg>(
        layouts: &[LevelParams],
    ) -> Result<(HachiProverSetup<F, D>, HachiVerifierSetup<F>), HachiError>
    where
        F: FieldCore + CanonicalField + FieldSampling,
        Cfg: CommitmentConfig<Field = F>,
    {
        let Some((first_lp, _)) = layouts.split_first() else {
            return Err(HachiError::InvalidSetup(
                "setup_with_layouts requires at least one layout".to_string(),
            ));
        };

        let alpha = D.trailing_zeros() as usize;
        let mut max_num_vars = 0usize;
        let mut max_inner_width = 0usize;
        let mut max_outer_width = 0usize;
        let mut max_d_matrix_width = 0usize;
        let mut max_r_vars = 0usize;
        let mut max_num_digits_open = 0usize;
        let mut max_num_digits_fold = 0usize;
        let mut max_log_basis = first_lp.log_basis;
        let mut max_n_a = 0usize;
        let mut max_n_b = 0usize;
        let mut max_n_d = 0usize;

        for lp in layouts {
            let layout_num_vars = lp.m_vars + lp.r_vars + alpha;
            let chain_stats = scan_layout_chain::<F, D, Cfg>(layout_num_vars, lp, 1)?;
            tracing::debug!(?lp, ?chain_stats, "setup layout chain");
            max_num_vars = max_num_vars.max(layout_num_vars);
            max_inner_width = max_inner_width.max(chain_stats.max_inner_width);
            max_outer_width = max_outer_width.max(chain_stats.max_outer_width);
            max_d_matrix_width = max_d_matrix_width.max(chain_stats.max_d_matrix_width);
            max_r_vars = max_r_vars.max(chain_stats.max_r_vars);
            max_num_digits_open = max_num_digits_open.max(chain_stats.max_num_digits_open);
            max_num_digits_fold = max_num_digits_fold.max(chain_stats.max_num_digits_fold);
            max_log_basis = max_log_basis.max(chain_stats.max_log_basis);
            max_n_a = max_n_a.max(chain_stats.max_n_a);
            max_n_b = max_n_b.max(chain_stats.max_n_b);
            max_n_d = max_n_d.max(chain_stats.max_n_d);
        }

        tracing::debug!(
            max_inner_width,
            max_outer_width,
            max_d_matrix_width,
            max_r_vars,
            max_num_vars,
            "setup envelope"
        );
        let public_matrix_seed = sample_public_matrix_seed();
        Self::setup_with_matrix_widths_and_seed::<F, D, Cfg>(
            max_num_vars,
            1,
            public_matrix_seed,
            max_inner_width,
            max_outer_width,
            max_d_matrix_width,
            max_n_a,
            max_n_b,
            max_n_d,
        )
    }

    fn setup_with_lp_and_seed<F, const D: usize, Cfg>(
        lp: &LevelParams,
        max_num_vars: usize,
        public_matrix_seed: PublicMatrixSeed,
    ) -> Result<(HachiProverSetup<F, D>, HachiVerifierSetup<F>), HachiError>
    where
        F: FieldCore + CanonicalField + FieldSampling,
        Cfg: CommitmentConfig<Field = F>,
    {
        let alpha = D.trailing_zeros() as usize;
        let layout_num_vars = lp.m_vars + lp.r_vars + alpha;
        let chain_stats = scan_layout_chain::<F, D, Cfg>(layout_num_vars, lp, 1)?;
        let a_cols = chain_stats.max_inner_width;
        let b_cols = chain_stats.max_outer_width;
        let d_cols = chain_stats.max_d_matrix_width;

        Self::setup_with_matrix_widths_and_seed::<F, D, Cfg>(
            max_num_vars,
            1,
            public_matrix_seed,
            a_cols,
            b_cols,
            d_cols,
            chain_stats.max_n_a,
            chain_stats.max_n_b,
            chain_stats.max_n_d,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn setup_with_matrix_widths_and_seed<F, const D: usize, Cfg>(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        public_matrix_seed: PublicMatrixSeed,
        a_cols: usize,
        b_cols: usize,
        d_cols: usize,
        max_n_a: usize,
        max_n_b: usize,
        max_n_d: usize,
    ) -> Result<(HachiProverSetup<F, D>, HachiVerifierSetup<F>), HachiError>
    where
        F: FieldCore + CanonicalField + FieldSampling,
        Cfg: CommitmentConfig<Field = F>,
    {
        let envelope = Cfg::envelope(max_num_vars);
        let max_stride = a_cols.max(b_cols).max(d_cols);
        let max_rows = max_n_a
            .max(max_n_b)
            .max(max_n_d)
            .max(envelope.max_n_a)
            .max(envelope.max_n_b)
            .max(envelope.max_n_d);
        let max_total = max_rows * max_stride;
        {
            let ring_bytes = std::mem::size_of::<CyclotomicRing<F, D>>();
            let shared_mb = (max_total * ring_bytes) as f64 / (1024.0_f64 * 1024.0_f64);
            tracing::debug!(
                a_cols,
                b_cols,
                d_cols,
                max_stride,
                max_total,
                ring_bytes,
                shared_mb,
                "setup shared matrix size"
            );
        }
        let shared_flat = derive_public_matrix_flat::<F, D>(max_total, &public_matrix_seed);
        let ntt_shared = build_ntt_slot(shared_flat.ring_view::<D>(1, max_total))?;

        let expanded = Arc::new(HachiExpandedSetup {
            seed: HachiSetupSeed {
                max_num_vars,
                max_num_batched_polys,
                max_stride,
                public_matrix_seed,
            },
            shared_matrix: shared_flat,
        });
        let prover_setup = HachiProverSetup {
            expanded: Arc::clone(&expanded),
            ntt_shared,
        };
        let verifier_setup = HachiVerifierSetup { expanded };
        Ok((prover_setup, verifier_setup))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::{HachiDeserialize, HachiSerialize};
    use crate::protocol::commitment::{hachi_recursive_level_layout_from_params, presets::fp128};
    use crate::protocol::ring_switch::w_ring_element_count_with_num_claims_and_points;
    use crate::test_utils::{TinyConfig, F as TestF};

    #[test]
    fn expanded_setup_roundtrips_and_derives_same_verifier() {
        const TEST_D: usize = 64;
        let prover_setup = HachiProverSetup::<TestF, TEST_D>::new::<TinyConfig>(16, 3).unwrap();
        let verifier_setup = HachiVerifierSetup {
            expanded: Arc::clone(&prover_setup.expanded),
        };

        let mut bytes = Vec::new();
        prover_setup
            .expanded
            .serialize_compressed(&mut bytes)
            .unwrap();
        let decoded = HachiExpandedSetup::<TestF>::deserialize_compressed(&bytes[..], &()).unwrap();

        assert_eq!(decoded, prover_setup.expanded.as_ref().clone());
        assert_eq!(decoded.seed.max_num_batched_polys, 3);

        let derived_verifier = HachiVerifierSetup {
            expanded: Arc::new(decoded.clone()),
        };
        assert_eq!(derived_verifier, verifier_setup);
    }

    #[test]
    fn setup_scales_root_batch_capacity() {
        const TEST_D: usize = 64;
        const MAX_NUM_VARS: usize = 16;
        const MAX_BATCH: usize = 3;

        let root_lp =
            validate_and_derive_layout::<TestF, TinyConfig, TEST_D>(MAX_NUM_VARS).unwrap();
        let single_stats =
            scan_layout_chain::<TestF, TEST_D, TinyConfig>(MAX_NUM_VARS, &root_lp, 1).unwrap();
        let batched_stats =
            scan_layout_chain::<TestF, TEST_D, TinyConfig>(MAX_NUM_VARS, &root_lp, MAX_BATCH)
                .unwrap();
        let scaled_root =
            root_batched_layout::<TinyConfig, TEST_D>(MAX_NUM_VARS, &root_lp, MAX_BATCH).unwrap();
        let worst_case_multipoint_w_len = w_ring_element_count_with_num_claims_and_points::<TestF>(
            &scaled_root,
            MAX_BATCH,
            MAX_BATCH,
        ) * TEST_D;
        let multipoint_level1_params = TinyConfig::level_params(HachiScheduleInputs {
            max_num_vars: MAX_NUM_VARS,
            level: 1,
            current_w_len: worst_case_multipoint_w_len,
        });
        let multipoint_level1_lp = hachi_recursive_level_layout_from_params::<TinyConfig>(
            &multipoint_level1_params,
            worst_case_multipoint_w_len,
        )
        .unwrap();

        let setup =
            HachiProverSetup::<TestF, TEST_D>::new::<TinyConfig>(MAX_NUM_VARS, MAX_BATCH).unwrap();
        let seed = &setup.expanded.seed;

        assert_eq!(setup.expanded.seed.max_num_batched_polys, MAX_BATCH);
        assert!(batched_stats.max_outer_width >= single_stats.max_outer_width);
        assert!(batched_stats.max_d_matrix_width >= single_stats.max_d_matrix_width);
        assert!(batched_stats.max_outer_width >= scaled_root.outer_width());
        assert!(batched_stats.max_d_matrix_width >= scaled_root.d_matrix_width());
        let max_stride = seed.max_stride;
        assert!(max_stride >= scaled_root.inner_width());
        assert!(max_stride >= scaled_root.outer_width());
        assert!(max_stride >= scaled_root.d_matrix_width());
        assert!(batched_stats.max_inner_width >= multipoint_level1_lp.inner_width());
        assert!(batched_stats.max_outer_width >= multipoint_level1_lp.outer_width());
        assert!(batched_stats.max_d_matrix_width >= multipoint_level1_lp.d_matrix_width());
        let batch_summary = HachiRootBatchSummary::new(MAX_BATCH, MAX_BATCH, MAX_BATCH).unwrap();
        let batched_root_plan = hachi_root_runtime_plan_from_root_layout::<TinyConfig, TEST_D>(
            HachiScheduleLookupKey::with_batch(
                MAX_NUM_VARS,
                MAX_NUM_VARS,
                MAX_BATCH,
                batch_summary,
            ),
            &root_lp,
        )
        .unwrap();
        let mut prev_w_len = batched_root_plan
            .inputs
            .current_w_len
            .saturating_mul(batched_root_plan.batch.num_claims);
        let mut current_w_len = batched_root_plan.next_w_len();
        let mut current_lp = batched_root_plan.next_level_params.clone();
        let mut level = 1usize;
        loop {
            assert!(batched_stats.max_inner_width >= current_lp.inner_width());
            assert!(batched_stats.max_outer_width >= current_lp.outer_width());
            assert!(batched_stats.max_d_matrix_width >= current_lp.d_matrix_width());
            assert!(batched_stats.max_n_a >= current_lp.a_key.row_len());
            assert!(batched_stats.max_n_b >= current_lp.b_key.row_len());
            assert!(batched_stats.max_n_d >= current_lp.d_key.row_len());
            if should_stop_folding(current_w_len, prev_w_len) {
                break;
            }
            let next_w_len = w_ring_element_count::<TestF>(&current_lp) * current_lp.ring_dimension;
            let next_level = level + 1;
            let next_lp_partial = TinyConfig::level_params(HachiScheduleInputs {
                max_num_vars: MAX_NUM_VARS,
                level: next_level,
                current_w_len: next_w_len,
            });
            let next_lp = hachi_recursive_level_layout_from_params::<TinyConfig>(
                &next_lp_partial,
                next_w_len,
            )
            .unwrap();
            prev_w_len = current_w_len;
            current_w_len = next_w_len;
            current_lp = next_lp;
            level = next_level;
        }
        assert!(seed.max_stride >= multipoint_level1_lp.inner_width());
        assert!(seed.max_stride >= multipoint_level1_lp.outer_width());
        assert!(seed.max_stride >= multipoint_level1_lp.d_matrix_width());
        let envelope = TinyConfig::envelope(MAX_NUM_VARS);
        let total_elements = setup
            .expanded
            .shared_matrix
            .total_ring_elements_at::<TEST_D>();
        assert!(total_elements >= envelope.max_n_a * batched_stats.max_inner_width);
        assert!(total_elements >= envelope.max_n_b * batched_stats.max_outer_width);
        assert!(total_elements >= envelope.max_n_d * batched_stats.max_d_matrix_width);
    }

    #[test]
    fn onehot_batched_helper_matches_setup_root_layout() {
        use crate::planner::schedule_params::{BatchConfig, Step};

        type Cfg = fp128::D64OneHot;
        const TEST_D: usize = Cfg::D;
        const NV: usize = 15;
        const BATCH: usize = 2;

        let root_lp = Cfg::commitment_layout(NV).unwrap();
        let setup_root = root_batched_layout::<Cfg, TEST_D>(NV, &root_lp, BATCH).unwrap();
        let helper_root = hachi_batched_root_layout::<Cfg, TEST_D>(NV, BATCH).unwrap();
        let setup = HachiProverSetup::<fp128::Field, TEST_D>::new::<Cfg>(NV, BATCH).unwrap();
        let runtime_lp =
            <HachiCommitmentCore as RingCommitmentScheme<fp128::Field, TEST_D, Cfg>>::layout(
                &setup,
            )
            .unwrap();
        let schedule =
            crate::planner::schedule_params::find_optimal_batched_schedule::<Cfg, TEST_D>(
                NV,
                BatchConfig {
                    num_claims: BATCH,
                    num_commitment_groups: 1,
                    num_points: 1,
                },
            )
            .unwrap();
        let root_step = match schedule.steps.first() {
            Some(Step::Fold(step)) => step,
            _ => panic!("batch-2 onehot schedule should start with a fold"),
        };

        assert_eq!(helper_root.m_vars, setup_root.m_vars);
        assert_eq!(helper_root.r_vars, setup_root.r_vars);
        assert_eq!(runtime_lp, helper_root);
        assert_eq!(
            helper_root.outer_width() * BATCH,
            root_step.params.outer_width()
        );
        assert_eq!(
            helper_root.d_matrix_width() * BATCH,
            root_step.params.d_matrix_width()
        );
        assert_eq!(helper_root.num_digits_fold, root_step.delta_fold_per_poly);
        assert_eq!(setup_root.outer_width(), root_step.params.outer_width());
        assert_eq!(helper_root.outer_width() * BATCH, setup_root.outer_width());
        assert_eq!(
            helper_root.d_matrix_width() * BATCH,
            setup_root.d_matrix_width()
        );
        assert!(
            helper_root.num_digits_fold <= setup_root.num_digits_fold,
            "per-poly num_digits_fold ({}) must not exceed batched value ({})",
            helper_root.num_digits_fold,
            setup_root.num_digits_fold,
        );
        let max_stride = setup.expanded.seed.max_stride;
        assert!(max_stride >= setup_root.outer_width());
        assert!(max_stride >= setup_root.d_matrix_width());
    }

    #[test]
    fn direct_only_batched_helper_stays_per_poly() {
        use crate::planner::schedule_params::{BatchConfig, Step};

        type Cfg = fp128::D64OneHot;
        const TEST_D: usize = Cfg::D;
        const BATCH: usize = 2;
        let batch = BatchConfig {
            num_claims: BATCH,
            num_commitment_groups: 1,
            num_points: 1,
        };
        let nv = (1usize..=12)
            .find(|&nv| {
                matches!(
                    crate::planner::schedule_params::find_optimal_batched_schedule::<Cfg, TEST_D>(
                        nv, batch
                    )
                    .unwrap()
                    .steps
                    .first(),
                    Some(Step::Direct(_))
                )
            })
            .expect("should find a direct-only batch-2 onehot root");
        let root_lp = Cfg::commitment_layout(nv).unwrap();
        let helper_root = hachi_batched_root_layout::<Cfg, TEST_D>(nv, BATCH).unwrap();
        let setup_root = root_batched_layout::<Cfg, TEST_D>(nv, &root_lp, BATCH).unwrap();

        assert_eq!(helper_root, root_lp);
        assert_eq!(helper_root.outer_width() * BATCH, setup_root.outer_width());
        assert_eq!(
            helper_root.d_matrix_width() * BATCH,
            setup_root.d_matrix_width()
        );
    }

    #[test]
    fn setup_with_layouts_uses_exact_width_envelope() {
        use crate::protocol::commitment::{compute_num_digits_fold, num_digits_for_bound};
        const TEST_D: usize = 64;
        const ALPHA: usize = 6; // TEST_D.trailing_zeros()

        let decomp = TinyConfig::decomposition();
        let depth_commit = num_digits_for_bound(decomp.log_commit_bound, decomp.log_basis);
        let open_bound = decomp.log_open_bound.unwrap_or(decomp.log_commit_bound);
        let depth_open = num_digits_for_bound(open_bound, decomp.log_basis);

        let nv_a = 4 + 2 + ALPHA;
        let nv_b = 1 + 6 + ALPHA;
        let params_a = TinyConfig::level_params(HachiScheduleInputs {
            max_num_vars: nv_a,
            level: 0,
            current_w_len: 1usize << nv_a,
        });
        let params_b = TinyConfig::level_params(HachiScheduleInputs {
            max_num_vars: nv_b,
            level: 0,
            current_w_len: 1usize << nv_b,
        });
        let depth_fold_a =
            compute_num_digits_fold(2, params_a.challenge_l1_mass(), decomp.log_basis);
        let depth_fold_b =
            compute_num_digits_fold(6, params_b.challenge_l1_mass(), decomp.log_basis);
        let lp_a = params_a
            .with_decomp(4, 2, depth_commit, depth_open, depth_fold_a, 0)
            .unwrap();
        let lp_b = params_b
            .with_decomp(1, 6, depth_commit, depth_open, depth_fold_b, 0)
            .unwrap();
        let w_len_a = w_ring_element_count::<TestF>(&lp_a) * TEST_D;
        let w_len_b = w_ring_element_count::<TestF>(&lp_b) * TEST_D;
        let w_lp_a =
            hachi_recursive_level_layout_from_params::<TinyConfig>(&params_a, w_len_a).unwrap();
        let w_lp_b =
            hachi_recursive_level_layout_from_params::<TinyConfig>(&params_b, w_len_b).unwrap();

        let expected_inner = [
            lp_a.inner_width(),
            lp_b.inner_width(),
            w_lp_a.inner_width(),
            w_lp_b.inner_width(),
        ]
        .into_iter()
        .max()
        .unwrap();
        let expected_outer = [
            lp_a.outer_width(),
            lp_b.outer_width(),
            w_lp_a.outer_width(),
            w_lp_b.outer_width(),
        ]
        .into_iter()
        .max()
        .unwrap();
        let expected_d = [
            lp_a.d_matrix_width(),
            lp_b.d_matrix_width(),
            w_lp_a.d_matrix_width(),
            w_lp_b.d_matrix_width(),
        ]
        .into_iter()
        .max()
        .unwrap();
        let expected_max_num_vars = nv_a.max(nv_b);

        let (setup, _) =
            HachiCommitmentCore::setup_with_layouts::<TestF, TEST_D, TinyConfig>(&[lp_a, lp_b])
                .unwrap();
        let seed = &setup.expanded.seed;

        assert_eq!(seed.max_num_vars, expected_max_num_vars);
        let total_elements = setup
            .expanded
            .shared_matrix
            .total_ring_elements_at::<TEST_D>();
        let envelope = TinyConfig::envelope(expected_max_num_vars);
        assert!(total_elements >= envelope.max_n_a * expected_inner);
        assert!(total_elements >= envelope.max_n_b * expected_outer);
        assert!(total_elements >= envelope.max_n_d * expected_d);
    }

    #[test]
    fn setup_accepts_field_coupled_presets() {
        HachiProverSetup::<fp128::Field, 128>::new::<fp128::D128Full>(12, 1)
            .expect("legacy fp128 preset should accept the legacy field");

        HachiProverSetup::<fp128::Field, 128>::new::<fp128::D128Full>(12, 1)
            .expect("default fp128 fixed-D preset should accept the default field");

        HachiProverSetup::<fp128::Field, 32>::new::<fp128::D32Full>(12, 1)
            .expect("small-D fp128 preset should accept the default field");
    }

    #[cfg(feature = "disk-persistence")]
    mod disk_persistence {
        use super::*;
        use std::fs;
        use std::sync::{LazyLock, Mutex};

        static DISK_TEST_ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

        fn cleanup_setup_file(max_num_vars: usize) {
            if let Some(path) = get_storage_path::<TinyConfig>(max_num_vars, 1) {
                let _ = fs::remove_file(path);
            }
        }

        fn with_test_cache_dir<T>(test_name: &str, f: impl FnOnce() -> T) -> T {
            let _guard = DISK_TEST_ENV_LOCK.lock().unwrap();
            let cache_root = std::env::temp_dir().join(format!("hachi-disk-tests-{test_name}"));
            fs::create_dir_all(&cache_root).unwrap();

            let old_local_app_data = std::env::var_os("LOCALAPPDATA");
            std::env::set_var("LOCALAPPDATA", &cache_root);
            let out = f();
            match old_local_app_data {
                Some(path) => std::env::set_var("LOCALAPPDATA", path),
                None => std::env::remove_var("LOCALAPPDATA"),
            }
            out
        }

        #[test]
        fn save_and_load_roundtrips() {
            with_test_cache_dir("roundtrip", || {
                const TEST_D: usize = 64;
                const MAX_VARS: usize = 100;

                cleanup_setup_file(MAX_VARS);

                let prover_setup =
                    HachiProverSetup::<TestF, TEST_D>::new::<TinyConfig>(MAX_VARS, 1).unwrap();

                let loaded = load_expanded_setup::<TestF, TinyConfig>(MAX_VARS, 1).unwrap();
                assert_eq!(loaded, prover_setup.expanded.as_ref().clone());

                cleanup_setup_file(MAX_VARS);
            });
        }

        #[test]
        fn setup_uses_cache_on_second_call() {
            with_test_cache_dir("second-call", || {
                const TEST_D: usize = 64;
                const MAX_VARS: usize = 101;

                cleanup_setup_file(MAX_VARS);

                let first =
                    HachiProverSetup::<TestF, TEST_D>::new::<TinyConfig>(MAX_VARS, 1).unwrap();

                let second =
                    HachiProverSetup::<TestF, TEST_D>::new::<TinyConfig>(MAX_VARS, 1).unwrap();

                assert_eq!(first.expanded, second.expanded);

                cleanup_setup_file(MAX_VARS);
            });
        }

        #[test]
        fn ntt_caches_rebuilt_correctly_from_disk() {
            with_test_cache_dir("ntt-rebuild", || {
                use crate::algebra::CyclotomicRing;

                const TEST_D: usize = 64;
                const MAX_VARS: usize = 102;

                cleanup_setup_file(MAX_VARS);

                let fresh_setup =
                    HachiProverSetup::<TestF, TEST_D>::new::<TinyConfig>(MAX_VARS, 1).unwrap();

                let loaded_expanded =
                    load_expanded_setup::<TestF, TinyConfig>(MAX_VARS, 1).unwrap();
                let (disk_setup, _) =
                    setup_from_expanded::<TestF, TEST_D>(loaded_expanded).unwrap();

                let lp = TinyConfig::commitment_layout(MAX_VARS).unwrap();
                let num_coeffs = lp.num_blocks * lp.block_len;
                let coeffs = vec![CyclotomicRing::<TestF, TEST_D>::zero(); num_coeffs];

                let fresh_commit = <HachiCommitmentCore as RingCommitmentScheme<
                    TestF,
                    TEST_D,
                    TinyConfig,
                >>::commit_coeffs(&coeffs, &fresh_setup)
                .unwrap();
                let disk_commit = <HachiCommitmentCore as RingCommitmentScheme<
                    TestF,
                    TEST_D,
                    TinyConfig,
                >>::commit_coeffs(&coeffs, &disk_setup)
                .unwrap();

                assert_eq!(fresh_commit.commitment, disk_commit.commitment);

                cleanup_setup_file(MAX_VARS);
            });
        }
    }
}
