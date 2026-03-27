/// Number of base-`2^log_basis` digits to represent a value with `log_bound` bits.
///
/// Returns `ceil(log_bound / log_basis)`, with an extra level when the
/// balanced-digit range would not cover the full bound.
pub fn compute_num_digits(log_bound: u32, log_basis: u32) -> usize {
    assert!(log_basis > 0 && log_basis < 128, "invalid log_basis");
    if log_bound == 0 {
        return 1;
    }
    let mut levels = (log_bound as usize).div_ceil(log_basis as usize);

    let total_bits = (levels as u32).saturating_mul(log_basis);
    if total_bits <= log_bound {
        let b: u128 = 1u128 << log_basis;
        let half_b_minus_1 = b / 2 - 1;
        let b_minus_1 = b - 1;
        let mut b_pow = 1u128;
        for _ in 0..levels {
            b_pow = b_pow.saturating_mul(b);
        }
        let max_positive = half_b_minus_1.saturating_mul(b_pow.saturating_sub(1) / b_minus_1);
        let required = if log_bound > 128 {
            u128::MAX / 2
        } else if log_bound == 0 {
            0
        } else {
            (1u128 << (log_bound - 1)).saturating_sub(1)
        };
        if max_positive < required {
            levels += 1;
        }
    }
    levels.max(1)
}

/// Decomposition depth for the folded witness `z_pre`.
///
/// beta = 2^r_vars * challenge_l1_mass * 2^(log_basis - 1).
pub fn compute_num_digits_fold(r_vars: usize, challenge_l1_mass: usize, log_basis: u32) -> usize {
    let shift = r_vars + (log_basis as usize) - 1;
    if shift >= 127 || challenge_l1_mass == 0 {
        return compute_num_digits(128, log_basis);
    }
    let beta = (challenge_l1_mass as u128).saturating_mul(1u128 << shift);
    if beta == 0 {
        return 1;
    }
    let log_beta = 128 - beta.leading_zeros();
    compute_num_digits(log_beta, log_basis)
}

/// Number of r-decomposition levels for quotient rows.
pub fn r_decomp_levels(field_bits: u32, half_field_bound: u128, log_basis: u32) -> usize {
    let bits = field_bits as usize;
    let lb = log_basis as usize;
    let mut levels = compute_num_digits(field_bits, log_basis);
    if levels == 0 {
        levels = 1;
    }

    let total_bits = levels * lb;
    if total_bits <= bits {
        let b = 1u128 << log_basis;
        let half_b_minus_1 = b / 2 - 1;
        let b_minus_1 = b - 1;
        let mut b_pow = 1u128;
        for _ in 0..levels {
            b_pow = b_pow.saturating_mul(b);
        }
        let max_positive = half_b_minus_1.saturating_mul((b_pow - 1) / b_minus_1);
        if max_positive < half_field_bound {
            levels += 1;
        }
    }

    levels
}

/// Find the (m, r) split of `reduced_vars` that minimizes witness size.
///
/// When `num_ring > 0` (tight z_pre mode), `m_eff` uses the actual ring count
/// instead of `2^m`. Pass `num_ring = 0` for the standard power-of-two fallback.
pub fn optimal_m_r_split(
    n_a: u32,
    challenge_l1_mass: usize,
    log_commit_bound: u32,
    log_basis: u32,
    reduced_vars: usize,
    num_ring: usize,
) -> (usize, usize) {
    if reduced_vars <= 2 || reduced_vars >= 53 {
        let r = reduced_vars / 2;
        return (reduced_vars - r, r);
    }

    let open_bound = if log_commit_bound < 128 {
        128
    } else {
        log_commit_bound
    };
    let delta_open = compute_num_digits(open_bound, log_basis) as u64;
    let delta_commit = compute_num_digits(log_commit_bound, log_basis) as u64;
    let c1 = delta_open + n_a as u64 * delta_commit;

    let mut best_r = reduced_vars / 2;
    let mut best_cost = u64::MAX;

    for r in 1..reduced_vars {
        let m = reduced_vars - r;
        let delta_fold = compute_num_digits_fold(r, challenge_l1_mass, log_basis) as u64;
        let m_eff = if num_ring > 0 {
            num_ring.div_ceil(1usize << r) as u64
        } else {
            1u64 << m
        };
        let cost = c1.saturating_mul(1u64 << r)
            + delta_commit
                .saturating_mul(delta_fold)
                .saturating_mul(m_eff);
        if cost < best_cost {
            best_cost = cost;
            best_r = r;
        }
    }

    (reduced_vars - best_r, best_r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digits_basic() {
        assert_eq!(compute_num_digits(128, 2), 65);
        assert_eq!(compute_num_digits(128, 3), 43);
        assert_eq!(compute_num_digits(1, 2), 1);
        assert_eq!(compute_num_digits(0, 2), 1);
    }

    #[test]
    fn digits_fold_basic() {
        let got_2 = compute_num_digits_fold(12, 54, 2);
        let got_3 = compute_num_digits_fold(12, 54, 3);
        assert!(got_2 > 0);
        assert!(got_3 > 0);
        assert!(got_2 >= got_3);
    }

    #[test]
    fn r_decomp_p275() {
        let half_q: u128 = (u128::MAX - 274) / 2; // (2^128 - 275) / 2
        let r2 = r_decomp_levels(128, half_q, 2);
        let r3 = r_decomp_levels(128, half_q, 3);
        assert!(r2 > 0);
        assert!(r3 > 0);
        assert!(r2 >= r3);
    }
}
