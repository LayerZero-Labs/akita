//! Ring-native §4.1 commitment core implementation.

use super::config::{ensure_block_layout, ensure_layout_supported_num_vars};
use super::onehot::{inner_ajtai_onehot_wide, map_onehot_to_sparse_blocks};
use super::schedule::HachiScheduleInputs;
use super::schedule::{HachiRootBatchSummary, HachiScheduleLookupKey};
use super::scheme::{CommitWitness, RingCommitmentScheme};
use super::types::RingCommitment;
use super::utils::linear::{
    decompose_rows_i8_into, mat_vec_mul_ntt_i8_dense, mat_vec_mul_ntt_i8_dense_single_row,
    mat_vec_mul_ntt_single_i8,
};
use super::CommitmentConfig;
use crate::algebra::fields::wide::HasWide;
use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::planner::digit_math::compute_num_digits_fold_with_claims;
use crate::primitives::serialization::Valid;
use crate::protocol::hachi_poly_ops::OneHotIndex;
use crate::protocol::params::{AjtaiKeyParams, LevelParams};
use crate::protocol::proof::FlatDigitBlocks;
use crate::protocol::setup::HachiProverSetup;
use crate::{CanonicalField, FieldCore, FieldSampling};

pub(crate) fn root_current_w_len<const D: usize>(lp: &LevelParams) -> usize {
    lp.num_blocks
        .checked_mul(lp.block_len)
        .and_then(|len| len.checked_mul(D))
        .unwrap_or(0)
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
    lp.num_digits_fold = per_poly_fold;
    Some(BatchedRootSplit { params: lp })
}

pub(crate) fn fallback_batched_root_split<Cfg, const D: usize>(
    max_num_vars: usize,
    num_claims: usize,
) -> Result<BatchedRootSplit, HachiError>
where
    Cfg: CommitmentConfig,
{
    let root_lp = Cfg::commitment_layout(max_num_vars)?;
    let params = if num_claims <= 1 {
        root_lp
    } else {
        scale_batched_root_layout::<Cfg, D>(max_num_vars, &root_lp, num_claims)?
    };
    Ok(BatchedRootSplit { params })
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
    lp.b_key =
        AjtaiKeyParams::new_unchecked(lp.b_key.row_len(), b_cols, lp.b_key.collision_inf(), d);
    lp.d_key =
        AjtaiKeyParams::new_unchecked(lp.d_key.row_len(), d_cols, lp.d_key.collision_inf(), d);
    lp.num_digits_fold = per_poly_fold;
    Ok(BatchedRootSplit { params: lp })
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
        let split = fallback_batched_root_split::<Cfg, D>(max_num_vars, 1)?;
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
        _ => return fallback_batched_root_split::<Cfg, D>(max_num_vars, 1),
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

        let sparse_blocks = map_onehot_to_sparse_blocks(onehot_k, indices, root_lp.block_len, D)?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::{HachiDeserialize, HachiSerialize};
    use crate::protocol::commitment::presets::fp128;
    use crate::protocol::setup::{HachiExpandedSetup, HachiVerifierSetup};
    use crate::test_utils::{TinyConfig, F as TestF};
    use std::sync::Arc;

    #[test]
    fn expanded_setup_roundtrips_and_derives_same_verifier() {
        const TEST_D: usize = 64;
        let prover_setup = HachiProverSetup::<TestF, TEST_D>::new::<TinyConfig>(16, 3, 1).unwrap();
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
    fn setup_accepts_field_coupled_presets() {
        HachiProverSetup::<fp128::Field, 128>::new::<fp128::D128Full>(12, 1, 1)
            .expect("legacy fp128 preset should accept the legacy field");

        HachiProverSetup::<fp128::Field, 128>::new::<fp128::D128Full>(12, 1, 1)
            .expect("default fp128 fixed-D preset should accept the default field");

        HachiProverSetup::<fp128::Field, 32>::new::<fp128::D32Full>(12, 1, 1)
            .expect("small-D fp128 preset should accept the default field");
    }

    #[cfg(feature = "disk-persistence")]
    mod disk_persistence {
        use super::*;
        use crate::protocol::setup::{get_storage_path, load_expanded_setup};
        use std::fs;
        use std::sync::{LazyLock, Mutex};

        static DISK_TEST_ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

        fn cleanup_setup_file(max_num_vars: usize) {
            if let Some(path) = get_storage_path::<TinyConfig>(max_num_vars, 1, 1) {
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
                    HachiProverSetup::<TestF, TEST_D>::new::<TinyConfig>(MAX_VARS, 1, 1).unwrap();

                let loaded = load_expanded_setup::<TestF, TinyConfig>(MAX_VARS, 1, 1).unwrap();
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
                    HachiProverSetup::<TestF, TEST_D>::new::<TinyConfig>(MAX_VARS, 1, 1).unwrap();

                let second =
                    HachiProverSetup::<TestF, TEST_D>::new::<TinyConfig>(MAX_VARS, 1, 1).unwrap();

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
                    HachiProverSetup::<TestF, TEST_D>::new::<TinyConfig>(MAX_VARS, 1, 1).unwrap();

                let loaded_expanded =
                    load_expanded_setup::<TestF, TinyConfig>(MAX_VARS, 1, 1).unwrap();
                let disk_setup =
                    HachiProverSetup::<TestF, TEST_D>::from_expanded(loaded_expanded).unwrap();

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
