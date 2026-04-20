//! Ring-native §4.1 commitment layout helpers.
//!
//! These helpers used to back a `RingCommitmentScheme` trait that materialised
//! commitments from explicit `t_hat` layouts. The production flow commits via
//! `HachiPolyOps::commit_inner_witness` (see `commitment_scheme.rs`), so only
//! the layout-selection helpers remain here.

use super::schedule::HachiScheduleInputs;
use super::schedule::{HachiRootBatchSummary, HachiScheduleLookupKey};
use super::CommitmentConfig;
use crate::error::HachiError;
use crate::planner::digit_math::compute_num_digits_fold_with_claims;
use crate::protocol::params::{AjtaiKeyParams, LevelParams};

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

#[cfg(test)]
mod tests {
    use crate::primitives::{HachiDeserialize, HachiSerialize};
    use crate::protocol::commitment::presets::fp128;
    use crate::protocol::setup::{HachiExpandedSetup, HachiProverSetup, HachiVerifierSetup};
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
        use crate::protocol::commitment::CommitmentConfig;
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
                use crate::protocol::commitment::utils::linear::mat_vec_mul_ntt_single_i8;
                use crate::protocol::hachi_poly_ops::{DensePoly, HachiPolyOps};

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
                let poly = DensePoly::<TestF, TEST_D>::from_ring_coeffs(coeffs);

                // Commit via the production path on both setups and compare.
                // Both should yield the same `u = B · t_hat` because the
                // disk-loaded expanded setup must rebuild its NTT caches to
                // match the fresh one exactly.
                let commit_u = |setup: &HachiProverSetup<TestF, TEST_D>| {
                    let inner = poly
                        .commit_inner_witness(
                            &setup.expanded.shared_matrix,
                            &setup.ntt_shared,
                            lp.a_key.row_len(),
                            lp.block_len,
                            lp.num_digits_commit,
                            lp.num_digits_open,
                            lp.log_basis,
                            setup.expanded.seed.max_stride,
                        )
                        .unwrap();
                    mat_vec_mul_ntt_single_i8::<TestF, TEST_D>(
                        &setup.ntt_shared,
                        lp.b_key.row_len(),
                        setup.expanded.seed.max_stride,
                        inner.t_hat.flat_digits(),
                    )
                };

                let fresh_u = commit_u(&fresh_setup);
                let disk_u = commit_u(&disk_setup);

                assert_eq!(fresh_u, disk_u);

                cleanup_setup_file(MAX_VARS);
            });
        }
    }
}
