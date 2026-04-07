//! Root layout computation using planner building blocks.
//!
//! Computes the `(m_vars, r_vars)` split and digit counts for the root
//! commitment level by combining digit math with SIS security constraints.
//! The iterative `n_a` convergence loop ensures that the SIS-derived rank
//! is self-consistent with the layout it determines.

use super::digit_math::{
    compute_num_digits_fold, compute_num_digits_fold_batched, num_digits_for_bound,
    optimal_m_r_split,
};
use super::sis_security::{ceil_supported_collision, min_rank_for_secure_width, MAX_RANK};

/// Computed root layout dimensions returned by the planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootLayoutDimensions {
    pub m_vars: usize,
    pub r_vars: usize,
    pub n_a: usize,
    pub num_digits_commit: usize,
    pub num_digits_open: usize,
    pub num_digits_fold: usize,
    pub log_basis: u32,
}

fn inner_width_for_m_vars(m_vars: usize, num_digits_commit: usize) -> Option<u64> {
    let block_len = 1usize.checked_shl(m_vars as u32)?;
    let inner_width = block_len.checked_mul(num_digits_commit)?;
    inner_width.try_into().ok()
}

fn root_split_for_claims(
    n_a: u32,
    challenge_l1_mass: usize,
    log_commit_bound: u32,
    log_basis: u32,
    reduced_vars: usize,
    num_claims: usize,
) -> (usize, usize) {
    if reduced_vars == 0 {
        (0, 0)
    } else if num_claims > 1 {
        batched_optimal_m_r_split(
            n_a,
            challenge_l1_mass,
            log_commit_bound,
            log_basis,
            reduced_vars,
            num_claims,
        )
    } else {
        optimal_m_r_split(
            n_a,
            challenge_l1_mass,
            log_commit_bound,
            log_basis,
            reduced_vars,
            0,
        )
    }
}

/// Compute root commitment layout dimensions from first principles.
///
/// Iterates the SIS rank `n_a` until the layout-derived inner width is
/// consistent with the SIS security table. Returns `None` only when the
/// SIS table has no entry for the resulting widths.
///
/// When `max_num_vars <= log2(d)` (the tiny-root case), the layout uses
/// `m_vars = 0, r_vars = 0` (a single block of one ring element).
///
/// When `num_claims > 1`, the `(m_vars, r_vars)` split is re-optimized
/// using a batched fold-digit cost model that factors in the claim count,
/// and `n_a` is converged against that same batched split so the returned
/// rank is secure for the final inner width. The returned `num_digits_fold`
/// uses single-poly fold digits (matching the per-polynomial layout
/// convention).
///
/// Parameters:
/// - `max_num_vars`: total polynomial variable count
/// - `d`: ring dimension (32, 64, or 128)
/// - `log_basis`: gadget decomposition base exponent
/// - `log_commit_bound`: bit-width of committed coefficients
/// - `log_open_bound`: bit-width of opening-time coefficients
/// - `challenge_l1_mass`: L1 mass of the stage-1 challenge
/// - `max_abs_challenge_coeff`: maximum absolute challenge coefficient
/// - `num_claims`: number of polynomials in the batch (1 for single-poly)
#[allow(clippy::too_many_arguments)]
pub fn compute_root_layout_dimensions(
    max_num_vars: usize,
    d: usize,
    log_basis: u32,
    log_commit_bound: u32,
    log_open_bound: u32,
    challenge_l1_mass: usize,
    max_abs_challenge_coeff: u32,
    num_claims: usize,
) -> Option<RootLayoutDimensions> {
    let alpha = d.trailing_zeros() as usize;
    let reduced_vars = max_num_vars.saturating_sub(alpha);

    let bd_collision = (1u32 << log_basis) - 1;
    let a_raw = if log_commit_bound == 1 {
        2
    } else {
        bd_collision
    };
    let a_collision = ceil_supported_collision(d as u32, a_raw * max_abs_challenge_coeff)?;

    let num_digits_commit = num_digits_for_bound(log_commit_bound, log_basis);
    let num_digits_open = num_digits_for_bound(log_open_bound, log_basis);

    // Converge n_a against the same split objective that the final layout
    // will use. In batched mode, the batched objective can pick a larger
    // m_vars, which widens the inner Ajtai matrix and must be reflected in
    // the SIS rank.
    let mut n_a = 1u32;
    for _ in 0..MAX_RANK {
        let (m_vars, _) = root_split_for_claims(
            n_a,
            challenge_l1_mass,
            log_commit_bound,
            log_basis,
            reduced_vars,
            num_claims,
        );
        let inner_width = inner_width_for_m_vars(m_vars, num_digits_commit)?;
        let derived_n_a = min_rank_for_secure_width(d as u32, a_collision, inner_width as u64)?;
        if derived_n_a == n_a {
            break;
        }
        n_a = derived_n_a;
    }

    let (m_vars, r_vars) = root_split_for_claims(
        n_a,
        challenge_l1_mass,
        log_commit_bound,
        log_basis,
        reduced_vars,
        num_claims,
    );

    let num_digits_fold = compute_num_digits_fold(r_vars, challenge_l1_mass, log_basis);
    Some(RootLayoutDimensions {
        m_vars,
        r_vars,
        n_a: n_a as usize,
        num_digits_commit,
        num_digits_open,
        num_digits_fold,
        log_basis,
    })
}

/// Batched `(m, r)` split that uses the batched fold-digit cost model.
///
/// Same witness-size objective as [`optimal_m_r_split`] but `delta_fold`
/// accounts for `num_claims` via [`compute_num_digits_fold_batched`].
fn batched_optimal_m_r_split(
    n_a: u32,
    challenge_l1_mass: usize,
    log_commit_bound: u32,
    log_basis: u32,
    reduced_vars: usize,
    num_claims: usize,
) -> (usize, usize) {
    if reduced_vars <= 2 || reduced_vars >= 53 {
        let r = reduced_vars / 2;
        return (reduced_vars - r, r);
    }

    let open_bound = log_commit_bound.max(128);
    let delta_open = num_digits_for_bound(open_bound, log_basis) as u64;
    let delta_commit = num_digits_for_bound(log_commit_bound, log_basis) as u64;
    let per_block_cost = delta_open + n_a as u64 * delta_open;

    let mut best = (u64::MAX, reduced_vars / 2);

    for r in 1..reduced_vars {
        let m_eff = 1u64 << (reduced_vars - r);
        let num_blocks = 1u64 << r;

        let delta_fold =
            compute_num_digits_fold_batched(r, challenge_l1_mass, log_basis, num_claims) as u64;

        let opening_cost = per_block_cost.saturating_mul(num_blocks);
        let folding_cost = delta_commit
            .saturating_mul(delta_fold)
            .saturating_mul(m_eff);
        let total = opening_cost.saturating_add(folding_cost);

        if total < best.0 {
            best = (total, r);
        }
    }

    let best_r = best.1;
    (reduced_vars - best_r, best_r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn legacy_root_layout_dimensions(
        max_num_vars: usize,
        d: usize,
        log_basis: u32,
        log_commit_bound: u32,
        log_open_bound: u32,
        challenge_l1_mass: usize,
        max_abs_challenge_coeff: u32,
        num_claims: usize,
    ) -> Option<RootLayoutDimensions> {
        let alpha = d.trailing_zeros() as usize;
        let reduced_vars = max_num_vars.saturating_sub(alpha);

        let bd_collision = (1u32 << log_basis) - 1;
        let a_raw = if log_commit_bound == 1 {
            2
        } else {
            bd_collision
        };
        let a_collision = ceil_supported_collision(d as u32, a_raw * max_abs_challenge_coeff)?;

        let num_digits_commit = num_digits_for_bound(log_commit_bound, log_basis);
        let num_digits_open = num_digits_for_bound(log_open_bound, log_basis);

        let mut n_a = 1u32;
        for _ in 0..MAX_RANK {
            let (m_vars, _) = if reduced_vars == 0 {
                (0, 0)
            } else {
                optimal_m_r_split(
                    n_a,
                    challenge_l1_mass,
                    log_commit_bound,
                    log_basis,
                    reduced_vars,
                    0,
                )
            };

            let inner_width = inner_width_for_m_vars(m_vars, num_digits_commit)?;
            let derived_n_a = min_rank_for_secure_width(d as u32, a_collision, inner_width)?;
            if derived_n_a == n_a {
                break;
            }
            n_a = derived_n_a;
        }

        let (m_vars, r_vars) = root_split_for_claims(
            n_a,
            challenge_l1_mass,
            log_commit_bound,
            log_basis,
            reduced_vars,
            num_claims,
        );
        let num_digits_fold = compute_num_digits_fold(r_vars, challenge_l1_mass, log_basis);
        Some(RootLayoutDimensions {
            m_vars,
            r_vars,
            n_a: n_a as usize,
            num_digits_commit,
            num_digits_open,
            num_digits_fold,
            log_basis,
        })
    }

    #[test]
    fn d128_onehot_32_produces_layout() {
        let dims = compute_root_layout_dimensions(32, 128, 3, 1, 128, 31, 1, 1);
        let dims = dims.expect("should produce layout for D=128 onehot nv=32");
        assert!(dims.m_vars > 0);
        assert!(dims.r_vars > 0);
        assert_eq!(dims.m_vars + dims.r_vars, 32 - 7);
        assert!(dims.n_a >= 1);
        assert!(dims.num_digits_commit >= 1);
        assert!(dims.num_digits_open >= 1);
        assert!(dims.num_digits_fold >= 1);
    }

    #[test]
    fn d128_dense_32_produces_layout() {
        let dims = compute_root_layout_dimensions(32, 128, 3, 128, 128, 31, 1, 1);
        let dims = dims.expect("should produce layout for D=128 dense nv=32");
        assert!(dims.m_vars > 0);
        assert!(dims.r_vars > 0);
        assert_eq!(dims.m_vars + dims.r_vars, 32 - 7);
    }

    #[test]
    fn full_field_bounds_use_num_digits_for_bound() {
        let dims = compute_root_layout_dimensions(32, 128, 4, 1, 128, 31, 1, 1)
            .expect("full-field layout should be derivable");
        assert_eq!(dims.num_digits_open, 32);
    }

    #[test]
    fn tiny_root_produces_degenerate_layout() {
        let dims = compute_root_layout_dimensions(7, 128, 3, 128, 128, 31, 1, 1)
            .expect("tiny root at alpha boundary should produce layout");
        assert_eq!(dims.m_vars, 0);
        assert_eq!(dims.r_vars, 0);

        let dims = compute_root_layout_dimensions(5, 128, 3, 128, 128, 31, 1, 1)
            .expect("tiny root below alpha should produce layout");
        assert_eq!(dims.m_vars, 0);
        assert_eq!(dims.r_vars, 0);

        let dims = compute_root_layout_dimensions(4, 32, 3, 128, 128, 256, 8, 1)
            .expect("D=32 tiny root should produce layout");
        assert_eq!(dims.m_vars, 0);
        assert_eq!(dims.r_vars, 0);
    }

    #[test]
    fn batched_layout_preserves_public_shape() {
        let single =
            compute_root_layout_dimensions(32, 128, 3, 1, 128, 31, 1, 1).expect("single layout");
        let batched =
            compute_root_layout_dimensions(32, 128, 3, 1, 128, 31, 1, 4).expect("batched layout");
        assert_eq!(single.num_digits_commit, batched.num_digits_commit);
        assert_eq!(single.num_digits_open, batched.num_digits_open);
        assert_eq!(single.log_basis, batched.log_basis);
        assert_eq!(
            single.m_vars + single.r_vars,
            batched.m_vars + batched.r_vars
        );
    }

    #[test]
    fn batched_convergence_sizes_rank_for_final_inner_width() {
        let legacy = legacy_root_layout_dimensions(8, 32, 2, 128, 128, 2, 3, 2)
            .expect("legacy batched layout should exist");
        let legacy_collision =
            ceil_supported_collision(32, ((1u32 << 2) - 1) * 3).expect("supported collision");
        let legacy_inner_width =
            inner_width_for_m_vars(legacy.m_vars, legacy.num_digits_commit).expect("inner width");
        let legacy_required = min_rank_for_secure_width(32, legacy_collision, legacy_inner_width)
            .expect("legacy final width should fit security table");
        assert_eq!(legacy.m_vars, 2);
        assert_eq!(legacy.n_a, 1);
        assert_eq!(legacy_required, 2);
        assert!(legacy.n_a < legacy_required as usize);

        let fixed = compute_root_layout_dimensions(8, 32, 2, 128, 128, 2, 3, 2)
            .expect("fixed batched layout should exist");
        let fixed_inner_width =
            inner_width_for_m_vars(fixed.m_vars, fixed.num_digits_commit).expect("inner width");
        let fixed_required = min_rank_for_secure_width(32, legacy_collision, fixed_inner_width)
            .expect("fixed final width should fit security table");
        assert!(fixed.n_a >= fixed_required as usize);
        assert!(fixed.n_a > legacy.n_a);
    }
}
