//! Per-fold-level operator-norm rejection policy and A-role witness-cost
//! scoring.
//!
//! `norm_bound` owns the pure norm→collision-bucket primitives. This module
//! layers the *decision* on top: given a fold geometry, should the level price
//! its A-role SIS collision with the operator-norm cap `Γ` (rejection on) or the
//! L1 mass `ω` (rejection off)? Rejection is enabled only when `Γ` pricing
//! yields a strictly smaller audited rank, a strictly cheaper next-level witness
//! scoring cost, and a bounded number of sparse draws per fold.

use akita_challenges::fold_sparse_challenge_sample_count;
use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};

use super::ajtai_key::{min_secure_rank, SisModulusFamily};
use super::decomposition_digits::{num_digits_fold, num_digits_for_bound};
use super::norm_bound::{
    committed_fold_collision_l2_sq, fold_challenge_norms, fold_witness_linf_cap_policy,
    FoldChallengeNorms, FoldWitnessLinfCapConfig, FoldWitnessNorms,
};
use crate::DecompositionParams;

/// Lemma-7 outer multiplier for A-role collision sizing at one fold level.
///
/// When `op_norm_rejection` is false, prices with L1 mass `ω`. When true,
/// prices with the operator-norm cap `Γ` (flat `Γ`, tensor `Γ²`).
#[inline]
#[must_use]
pub fn committed_fold_a_role_mass(
    fold_shape: TensorChallengeShape,
    stage1_config: &SparseChallengeConfig,
    op_norm_rejection: bool,
) -> u128 {
    if op_norm_rejection {
        fold_shape.effective_operator_norm_cap(stage1_config) as u128
    } else {
        fold_shape.effective_l1_mass(stage1_config) as u128
    }
}

/// Next-level witness scoring cost for one fold geometry, matching
/// [`crate::layout::digit_math::optimal_m_r_split`]:
///
/// ```text
///   (1 + n_a) · δ_open · 2^r  +  δ_commit · δ_fold · m_eff
/// ```
///
/// `m_eff` is the S-block row count implied by `inner_width / δ_commit`.
#[allow(clippy::too_many_arguments)]
pub fn fold_level_witness_scoring_cost(
    n_a: usize,
    op_norm_rejection: bool,
    r_vars: usize,
    num_claims: usize,
    inner_width: usize,
    decomposition: DecompositionParams,
    stage1_config: &SparseChallengeConfig,
    fold_shape: TensorChallengeShape,
    ring_dimension: usize,
    fold_challenge: FoldChallengeNorms,
    fold_witness: FoldWitnessNorms,
) -> Option<u64> {
    let field_bits = decomposition.field_bits();
    let log_basis = decomposition.log_basis;
    let log_commit_bound = decomposition.log_commit_bound;
    let open_bound = log_commit_bound.max(field_bits);
    let delta_open = num_digits_for_bound(open_bound, field_bits, log_basis) as u64;
    let delta_commit = num_digits_for_bound(log_commit_bound, field_bits, log_basis) as u64;
    let block_len = inner_width.checked_div(delta_commit as usize)?;
    if block_len == 0 {
        return None;
    }
    let num_blocks = 1u64.checked_shl(r_vars as u32)?;
    let m_eff = block_len as u64;
    let cap_policy = fold_witness_linf_cap_policy(stage1_config, fold_shape, ring_dimension);
    let binding = crate::FoldLinfProtocolBinding::CURRENT;
    let (grind_target_accept_num, grind_target_accept_den) = binding.grind_target_accept_prob();
    let cap_config = FoldWitnessLinfCapConfig::for_fold_level_scoring(
        cap_policy,
        stage1_config,
        fold_shape,
        ring_dimension,
        op_norm_rejection,
        inner_width,
        grind_target_accept_num,
        grind_target_accept_den,
    )
    .ok()?;
    let delta_fold = num_digits_fold(
        r_vars,
        num_claims,
        field_bits,
        log_basis,
        fold_challenge,
        fold_witness,
        cap_config,
    )
    .ok()? as u64;
    let per_block_cost = delta_open.saturating_add((n_a as u64).saturating_mul(delta_open));
    let opening_cost = per_block_cost.saturating_mul(num_blocks);
    let folding_cost = delta_commit
        .saturating_mul(delta_fold)
        .saturating_mul(m_eff);
    Some(opening_cost.saturating_add(folding_cost))
}

/// Maximum sparse challenges in one fold draw for which the planner may enable
/// operator-norm rejection (verifier replays the certified predicate per slot).
pub const OP_NORM_REJECTION_MAX_SPARSE_SAMPLES: usize = 1 << 12;

/// Override via `AKITA_OP_NORM_MAX_SPARSE_SAMPLES` for planner what-if runs.
#[inline]
fn effective_op_norm_rejection_max_sparse_samples() -> usize {
    std::env::var("AKITA_OP_NORM_MAX_SPARSE_SAMPLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(OP_NORM_REJECTION_MAX_SPARSE_SAMPLES)
}

#[derive(Hash, PartialEq, Eq)]
struct OpNormRejectionCacheKey {
    sis_family: SisModulusFamily,
    d: usize,
    decomposition: DecompositionParams,
    stage1_config: SparseChallengeConfig,
    fold_shape: TensorChallengeShape,
    is_root: bool,
    onehot_chunk_size: usize,
    ring_subfield_norm_bound: u32,
    r_vars: usize,
    num_claims: usize,
    inner_width: u64,
    max_sparse_samples: usize,
}

type OpNormRejectionCache =
    std::collections::HashMap<OpNormRejectionCacheKey, Option<(bool, u128, usize)>>;

thread_local! {
    static OP_NORM_REJECTION_CACHE: std::cell::RefCell<OpNormRejectionCache> =
        std::cell::RefCell::new(OpNormRejectionCache::new());
}

/// Like [`choose_op_norm_rejection_for_a_role`] with an explicit sparse-draw cap
/// (for planner what-if analysis).
#[allow(clippy::too_many_arguments)]
pub fn choose_op_norm_rejection_for_a_role_with_max_sparse_samples(
    sis_family: SisModulusFamily,
    d: usize,
    decomposition: DecompositionParams,
    stage1_config: &SparseChallengeConfig,
    fold_shape: TensorChallengeShape,
    is_root: bool,
    onehot_chunk_size: usize,
    ring_subfield_norm_bound: u32,
    r_vars: usize,
    num_claims: usize,
    inner_width: u64,
    max_sparse_samples: usize,
) -> Option<(bool, u128, usize)> {
    let key = OpNormRejectionCacheKey {
        sis_family,
        d,
        decomposition,
        stage1_config: stage1_config.clone(),
        fold_shape,
        is_root,
        onehot_chunk_size,
        ring_subfield_norm_bound,
        r_vars,
        num_claims,
        inner_width,
        max_sparse_samples,
    };
    OP_NORM_REJECTION_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(hit) = cache.get(&key) {
            return *hit;
        }
        let result = choose_op_norm_rejection_for_a_role_inner(
            sis_family,
            d,
            decomposition,
            stage1_config,
            fold_shape,
            is_root,
            onehot_chunk_size,
            ring_subfield_norm_bound,
            r_vars,
            num_claims,
            inner_width,
            max_sparse_samples,
        );
        cache.insert(key, result);
        result
    })
}

/// Choose per-level operator-norm rejection for A-role SIS sizing.
///
/// Rejection is enabled only when pricing with the operator-norm cap `Γ` yields
/// a strictly smaller audited [`min_secure_rank`] than L1-mass `ω` pricing at
/// the same inner width **and** [`fold_level_witness_scoring_cost`] is strictly
/// lower with rejection on at that geometry **and** the fold draw samples at most
/// [`OP_NORM_REJECTION_MAX_SPARSE_SAMPLES`] sparse challenges (flat
/// `2^{r_vars} · num_claims`, or the tensor left+right total). When both ranks
/// match, rejection is off (no proof-size benefit; avoids wasteful rejection sampling).
///
/// Production binding presets exist only at D=64 today (`ExactShell` with
/// `T < ||c||_1`). D=32 (`BoundedL1Norm`) and D=128/D=256 (`Uniform`) keep
/// `operator_norm_cap == ω`, so this function returns `false` for all current
/// proof-optimized ring challenge configs except D=64 levels where Γ wins.
///
/// Returns `(op_norm_rejection, collision_bucket, n_a)`.
#[allow(clippy::too_many_arguments)]
pub fn choose_op_norm_rejection_for_a_role(
    sis_family: SisModulusFamily,
    d: usize,
    decomposition: DecompositionParams,
    stage1_config: &SparseChallengeConfig,
    fold_shape: TensorChallengeShape,
    is_root: bool,
    onehot_chunk_size: usize,
    ring_subfield_norm_bound: u32,
    r_vars: usize,
    num_claims: usize,
    inner_width: u64,
) -> Option<(bool, u128, usize)> {
    choose_op_norm_rejection_for_a_role_with_max_sparse_samples(
        sis_family,
        d,
        decomposition,
        stage1_config,
        fold_shape,
        is_root,
        onehot_chunk_size,
        ring_subfield_norm_bound,
        r_vars,
        num_claims,
        inner_width,
        effective_op_norm_rejection_max_sparse_samples(),
    )
}

#[allow(clippy::too_many_arguments)]
fn choose_op_norm_rejection_for_a_role_inner(
    sis_family: SisModulusFamily,
    d: usize,
    decomposition: DecompositionParams,
    stage1_config: &SparseChallengeConfig,
    fold_shape: TensorChallengeShape,
    is_root: bool,
    onehot_chunk_size: usize,
    ring_subfield_norm_bound: u32,
    r_vars: usize,
    num_claims: usize,
    inner_width: u64,
    max_sparse_samples: usize,
) -> Option<(bool, u128, usize)> {
    let omega = committed_fold_a_role_mass(fold_shape, stage1_config, false);
    let gamma = committed_fold_a_role_mass(fold_shape, stage1_config, true);
    if omega == 0 {
        return None;
    }

    let is_onehot = is_root && decomposition.log_commit_bound == 1;
    let witness = FoldWitnessNorms::new(decomposition.log_basis, d, onehot_chunk_size, is_onehot);
    let challenge = fold_challenge_norms(stage1_config, fold_shape);

    // ω and Γ share the same Lemma-7 collision shape (only the outer mass
    // differs), priced through the canonical `committed_fold_collision_l2_sq`.
    let rank_for_mass = |mass: u128| -> Option<(u128, usize)> {
        let bucket = committed_fold_collision_l2_sq(
            sis_family,
            d as u32,
            mass,
            challenge,
            witness,
            r_vars,
            num_claims,
            ring_subfield_norm_bound,
        )?;
        let rank = min_secure_rank(sis_family, d as u32, bucket, inner_width)?;
        Some((bucket, rank))
    };

    let (bucket_l1, rank_l1) = rank_for_mass(omega)?;

    let sample_count = fold_sparse_challenge_sample_count(fold_shape, r_vars, num_claims);
    let rejection_allowed = sample_count.is_some_and(|n| n <= max_sparse_samples);

    if gamma == omega || !rejection_allowed {
        return Some((false, bucket_l1, rank_l1));
    }

    let (bucket_gamma, rank_gamma) = rank_for_mass(gamma)?;
    if rank_gamma >= rank_l1 {
        return Some((false, bucket_l1, rank_l1));
    }

    let inner_width_usize = usize::try_from(inner_width).ok()?;
    let witness_cost = |n_a: usize, op_norm_rejection: bool| -> Option<u64> {
        fold_level_witness_scoring_cost(
            n_a,
            op_norm_rejection,
            r_vars,
            num_claims,
            inner_width_usize,
            decomposition,
            stage1_config,
            fold_shape,
            d,
            challenge,
            witness,
        )
    };
    if witness_cost(rank_gamma, true)? < witness_cost(rank_l1, false)? {
        Some((true, bucket_gamma, rank_gamma))
    } else {
        Some((false, bucket_l1, rank_l1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sis::ajtai_key::min_secure_rank;
    use crate::DecompositionParams;
    use akita_challenges::{
        fold_sparse_challenge_sample_count, D64_PRODUCTION_EXACT_SHELL_MAG1,
        D64_PRODUCTION_EXACT_SHELL_MAG2, D64_PRODUCTION_OPERATOR_NORM_THRESHOLD,
    };

    fn production_shell() -> SparseChallengeConfig {
        SparseChallengeConfig::ExactShell {
            count_mag1: D64_PRODUCTION_EXACT_SHELL_MAG1,
            count_mag2: D64_PRODUCTION_EXACT_SHELL_MAG2,
            operator_norm_threshold: D64_PRODUCTION_OPERATOR_NORM_THRESHOLD,
        }
    }

    fn production_decomp() -> DecompositionParams {
        DecompositionParams {
            log_basis: 3,
            log_commit_bound: 1,
            log_open_bound: Some(128),
        }
    }

    /// Independent A-role bucket for `mass`, re-derived through the canonical
    /// collision primitive to cross-check `choose_*`.
    fn bucket_for_mass(
        shell: &SparseChallengeConfig,
        decomp: DecompositionParams,
        r_vars: usize,
        mass: u128,
    ) -> u128 {
        let witness = FoldWitnessNorms::new(decomp.log_basis, 64, 256, true);
        let challenge = fold_challenge_norms(shell, TensorChallengeShape::Flat);
        committed_fold_collision_l2_sq(
            SisModulusFamily::Q128,
            64,
            mass,
            challenge,
            witness,
            r_vars,
            1,
            1,
        )
        .expect("collision bucket")
    }

    #[test]
    fn choose_op_norm_rejection_only_when_gamma_lowers_rank_and_witness_cost() {
        let shell = production_shell();
        let decomp = production_decomp();
        let challenge = fold_challenge_norms(&shell, TensorChallengeShape::Flat);
        let witness = FoldWitnessNorms::new(3, 64, 256, true);
        let omega = TensorChallengeShape::Flat.effective_l1_mass(&shell) as u128;
        let gamma = TensorChallengeShape::Flat.effective_operator_norm_cap(&shell) as u128;
        let mut saw_gamma_win = false;
        let mut saw_rank_win_declined_by_witness_cost = false;
        for r_vars in 1usize..=12 {
            for &inner_width in &[
                1_000_000u64,
                5_000_000,
                20_000_000,
                50_000_000,
                80_000_000,
                120_000_000,
                500_000_000,
            ] {
                let inner_width_usize = inner_width as usize;
                let Some((reject, bucket, rank)) = choose_op_norm_rejection_for_a_role(
                    SisModulusFamily::Q128,
                    64,
                    decomp,
                    &shell,
                    TensorChallengeShape::Flat,
                    true,
                    256,
                    1,
                    r_vars,
                    1,
                    inner_width,
                ) else {
                    continue;
                };
                let bucket_l1 = bucket_for_mass(&shell, decomp, r_vars, omega);
                let bucket_gamma = bucket_for_mass(&shell, decomp, r_vars, gamma);
                let rank_l1 =
                    min_secure_rank(SisModulusFamily::Q128, 64, bucket_l1, inner_width).unwrap();
                let rank_gamma =
                    min_secure_rank(SisModulusFamily::Q128, 64, bucket_gamma, inner_width).unwrap();
                let witness_cost_gamma = fold_level_witness_scoring_cost(
                    rank_gamma,
                    true,
                    r_vars,
                    1,
                    inner_width_usize,
                    decomp,
                    &shell,
                    TensorChallengeShape::Flat,
                    64,
                    challenge,
                    witness,
                )
                .expect("gamma witness score");
                let witness_cost_l1 = fold_level_witness_scoring_cost(
                    rank_l1,
                    false,
                    r_vars,
                    1,
                    inner_width_usize,
                    decomp,
                    &shell,
                    TensorChallengeShape::Flat,
                    64,
                    challenge,
                    witness,
                )
                .expect("l1 witness score");
                if reject {
                    saw_gamma_win = true;
                    assert_eq!(bucket, bucket_gamma);
                    assert_eq!(rank, rank_gamma);
                    assert!(rank_gamma < rank_l1);
                    assert!(witness_cost_gamma < witness_cost_l1);
                } else {
                    assert_eq!(bucket, bucket_l1);
                    assert_eq!(rank, rank_l1);
                    if rank_gamma < rank_l1 {
                        let draw_within_cap = fold_sparse_challenge_sample_count(
                            TensorChallengeShape::Flat,
                            r_vars,
                            1,
                        )
                        .is_some_and(|n| n <= OP_NORM_REJECTION_MAX_SPARSE_SAMPLES);
                        if draw_within_cap {
                            saw_rank_win_declined_by_witness_cost = true;
                            assert!(witness_cost_gamma >= witness_cost_l1);
                        }
                    } else {
                        assert!(rank_gamma >= rank_l1);
                    }
                }
            }
        }
        assert!(
            saw_gamma_win || saw_rank_win_declined_by_witness_cost,
            "expected some (r, width) where Gamma=18 lowers n_a vs omega=53"
        );

        let loose = SparseChallengeConfig::ExactShell {
            count_mag1: 30,
            count_mag2: 12,
            operator_norm_threshold: 54,
        };
        let (reject_loose, _, _) = choose_op_norm_rejection_for_a_role(
            SisModulusFamily::Q128,
            64,
            decomp,
            &loose,
            TensorChallengeShape::Flat,
            true,
            256,
            1,
            3,
            1,
            10_000,
        )
        .expect("main shell should size");
        assert!(
            !reject_loose,
            "non-binding T=54 should not enable rejection when ranks tie"
        );
    }

    #[test]
    fn choose_op_norm_rejection_disabled_when_sparse_draw_exceeds_cap() {
        let shell = production_shell();
        let decomp = production_decomp();
        let inner_width = 50_000_000u64;
        let (reject_within_cap, _, _) =
            choose_op_norm_rejection_for_a_role_with_max_sparse_samples(
                SisModulusFamily::Q128,
                64,
                decomp,
                &shell,
                TensorChallengeShape::Flat,
                true,
                256,
                1,
                10,
                1,
                inner_width,
                OP_NORM_REJECTION_MAX_SPARSE_SAMPLES,
            )
            .expect("within-cap draw should size");
        let (reject_over_cap, _, _) = choose_op_norm_rejection_for_a_role_with_max_sparse_samples(
            SisModulusFamily::Q128,
            64,
            decomp,
            &shell,
            TensorChallengeShape::Flat,
            true,
            256,
            1,
            13,
            1,
            inner_width,
            OP_NORM_REJECTION_MAX_SPARSE_SAMPLES,
        )
        .expect("over-cap draw should size");
        assert_eq!(
            fold_sparse_challenge_sample_count(TensorChallengeShape::Flat, 12, 1),
            Some(1 << 12)
        );
        assert_eq!(
            fold_sparse_challenge_sample_count(TensorChallengeShape::Flat, 13, 1),
            Some(1 << 13)
        );
        assert!(
            fold_sparse_challenge_sample_count(TensorChallengeShape::Flat, 12, 1)
                .is_some_and(|n| n <= OP_NORM_REJECTION_MAX_SPARSE_SAMPLES)
        );
        assert!(
            reject_within_cap,
            "2^10 draws are within the 2^12 cap and should still explore rejection"
        );
        assert!(
            !reject_over_cap,
            "above 2^12 flat draws must not enable rejection"
        );

        let (left_len, right_len) = akita_challenges::tensor_split(1 << 12).unwrap();
        assert_eq!(
            fold_sparse_challenge_sample_count(TensorChallengeShape::Tensor, 12, 1),
            Some(left_len + right_len)
        );
        let tensor_inner = 5_000_000u64;
        let (reject_tensor, _, _) = choose_op_norm_rejection_for_a_role_with_max_sparse_samples(
            SisModulusFamily::Q128,
            64,
            decomp,
            &shell,
            TensorChallengeShape::Tensor,
            true,
            256,
            1,
            12,
            1,
            tensor_inner,
            OP_NORM_REJECTION_MAX_SPARSE_SAMPLES,
        )
        .expect("tensor draw within cap should size");
        assert!(
            reject_tensor,
            "tensor totals use left+right factor draws, not num_blocks"
        );
    }
}
