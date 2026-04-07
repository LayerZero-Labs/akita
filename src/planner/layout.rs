//! Root layout computation using planner building blocks.
//!
//! Computes the `(m_vars, r_vars)` split and digit counts for the root
//! commitment level by combining digit math with SIS security constraints.
//! The iterative `n_a` convergence loop ensures that the SIS-derived rank
//! is self-consistent with the layout it determines.

use super::digit_math::{compute_num_digits, compute_num_digits_fold, optimal_m_r_split};
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

/// Compute root commitment layout dimensions from first principles.
///
/// Iterates the SIS rank `n_a` until the layout-derived inner width is
/// consistent with the SIS security table. Returns `None` only when the
/// SIS table has no entry for the resulting widths.
///
/// When `max_num_vars <= log2(d)` (the tiny-root case), the layout uses
/// `m_vars = 0, r_vars = 0` (a single block of one ring element).
///
/// Parameters:
/// - `max_num_vars`: total polynomial variable count
/// - `d`: ring dimension (32, 64, or 128)
/// - `log_basis`: gadget decomposition base exponent
/// - `log_commit_bound`: bit-width of committed coefficients
/// - `log_open_bound`: bit-width of opening-time coefficients
/// - `challenge_l1_mass`: L1 mass of the stage-1 challenge
/// - `max_abs_challenge_coeff`: maximum absolute challenge coefficient
pub fn compute_root_layout_dimensions(
    max_num_vars: usize,
    d: usize,
    log_basis: u32,
    log_commit_bound: u32,
    log_open_bound: u32,
    challenge_l1_mass: usize,
    max_abs_challenge_coeff: u32,
) -> Option<RootLayoutDimensions> {
    let alpha = d.trailing_zeros() as usize;
    let reduced_vars = max_num_vars.saturating_sub(alpha);

    let bd_collision = (1u32 << log_basis) - 1;
    let a_raw = if log_commit_bound == 1 { 2 } else { bd_collision };
    let a_collision = ceil_supported_collision(d as u32, a_raw * max_abs_challenge_coeff)?;

    let num_digits_commit = compute_num_digits(log_commit_bound, log_basis);

    let mut n_a = 1u32;
    for _ in 0..MAX_RANK {
        let (m_vars, r_vars) = if reduced_vars == 0 {
            (0, 0)
        } else {
            optimal_m_r_split(n_a, challenge_l1_mass, log_commit_bound, log_basis, reduced_vars, 0)
        };

        let block_len = 1usize.checked_shl(m_vars as u32)?;
        let inner_width = block_len.checked_mul(num_digits_commit)?;

        let derived_n_a = min_rank_for_secure_width(d as u32, a_collision, inner_width as u64)?;
        if derived_n_a == n_a {
            let num_digits_open = compute_num_digits(log_open_bound, log_basis);
            let num_digits_fold = compute_num_digits_fold(r_vars, challenge_l1_mass, log_basis);
            return Some(RootLayoutDimensions {
                m_vars,
                r_vars,
                n_a: n_a as usize,
                num_digits_commit,
                num_digits_open,
                num_digits_fold,
                log_basis,
            });
        }
        n_a = derived_n_a;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d128_onehot_32_produces_layout() {
        let dims = compute_root_layout_dimensions(32, 128, 3, 1, 128, 31, 1);
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
        let dims = compute_root_layout_dimensions(32, 128, 3, 128, 128, 31, 1);
        let dims = dims.expect("should produce layout for D=128 dense nv=32");
        assert!(dims.m_vars > 0);
        assert!(dims.r_vars > 0);
        assert_eq!(dims.m_vars + dims.r_vars, 32 - 7);
    }

    #[test]
    fn tiny_root_produces_degenerate_layout() {
        let dims = compute_root_layout_dimensions(7, 128, 3, 128, 128, 31, 1)
            .expect("tiny root at alpha boundary should produce layout");
        assert_eq!(dims.m_vars, 0);
        assert_eq!(dims.r_vars, 0);

        let dims = compute_root_layout_dimensions(5, 128, 3, 128, 128, 31, 1)
            .expect("tiny root below alpha should produce layout");
        assert_eq!(dims.m_vars, 0);
        assert_eq!(dims.r_vars, 0);

        let dims = compute_root_layout_dimensions(4, 32, 3, 128, 128, 256, 8)
            .expect("D=32 tiny root should produce layout");
        assert_eq!(dims.m_vars, 0);
        assert_eq!(dims.r_vars, 0);
    }
}
