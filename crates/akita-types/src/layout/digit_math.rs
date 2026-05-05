//! Digit decomposition helpers used by schedule planning and runtime layout.

use akita_field::{CanonicalField, FieldCore};

/// Maximum positive value representable by `num_digits` balanced base-`b` digits,
/// where `b = 2^log_basis`.
///
/// Each balanced digit lies in `[-b/2, b/2 - 1]`. The positional weights
/// are `1, b, b^2, …, b^(n-1)`, so the maximum positive value is:
///
///   max_pos = (b/2 - 1) · (b^n - 1) / (b - 1)       (geometric series)
///
/// When `b^n` overflows `u128`, the result is a conservative lower bound
/// (uses `u128::MAX` as a stand-in for the true `b^n`). This is safe because
/// the caller only asks "is max_pos < required?" -- an underestimate can only
/// cause an extra digit to be added, never one to be missed.
fn balanced_digit_max(log_basis: u32, num_digits: usize) -> u128 {
    let base: u128 = 1u128 << log_basis;
    let max_digit = base / 2 - 1; // b/2 - 1
    let base_minus_1 = base - 1; // b - 1

    let mut base_pow = 1u128; // will become b^num_digits (saturating)
    for _ in 0..num_digits {
        base_pow = base_pow.saturating_mul(base);
    }

    max_digit.saturating_mul(base_pow.saturating_sub(1) / base_minus_1)
}

/// Minimum number of balanced base-`2^log_basis` digits needed to represent
/// any value in `[-V, V]` where `V < 2^log_bound`, using symmetric centering.
///
/// **Balanced digits:** each digit `d_i ∈ [-b/2, b/2-1]` with `b = 2^log_basis`.
/// A number is represented as `Σ d_i · b^i` for `i = 0..n-1`.
///
/// **Algorithm:**
/// 1. Start with the unsigned estimate `n = ⌈log_bound / log_basis⌉`.
/// 2. When the bit budget is tight (`n * log_basis ≤ log_bound`), verify that
///    the balanced range actually covers `2^(log_bound-1) - 1`. If not, add
///    one more digit.
///
/// For full-field bounds (128 bits), prefer [`num_digits_for_bound`] which
/// uses asymmetric centering to avoid the +1 correction.
///
/// # Panics
///
/// Panics if `log_basis` is 0 or at least 128, or if `log_bound` exceeds 128.
///
/// # Examples
///
/// ```ignore
/// # use akita_types::layout::digit_math::compute_num_digits;
/// // 128-bit value in balanced base-4 (log_basis=2): needs 65 digits
/// assert_eq!(compute_num_digits(128, 2), 65);
///
/// // 128-bit value in balanced base-8 (log_basis=3): 43 suffices
/// assert_eq!(compute_num_digits(128, 3), 43);
/// ```
pub fn compute_num_digits(log_bound: u32, log_basis: u32) -> usize {
    assert!(log_basis > 0 && log_basis < 128, "invalid log_basis");
    assert!(
        log_bound <= 128,
        "log_bound={log_bound} exceeds 128-bit field"
    );

    if log_bound == 0 {
        return 1;
    }

    let mut num_digits = (log_bound as usize).div_ceil(log_basis as usize);

    // When the bit budget is tight (no slack bits), the balanced range may be
    // smaller than the unsigned range. Verify explicitly.
    let total_bits = (num_digits as u32).saturating_mul(log_basis);
    if total_bits <= log_bound {
        let required_positive = (1u128 << (log_bound - 1)).saturating_sub(1);

        if balanced_digit_max(log_basis, num_digits) < required_positive {
            num_digits += 1;
        }
    }

    num_digits.max(1)
}

/// Decomposition depth for full-field values using asymmetric centering:
/// `ceil(field_bits / log_basis)` with no +1 correction.
///
/// # Panics
///
/// Panics if `log_basis` is 0 or >= 128.
pub fn compute_num_digits_full_field(field_bits: u32, log_basis: u32) -> usize {
    assert!(log_basis > 0 && log_basis < 128, "invalid log_basis");
    if field_bits == 0 {
        return 1;
    }
    (field_bits as usize).div_ceil(log_basis as usize).max(1)
}

/// Choose the correct digit-count function for a given bit-width bound.
///
/// Full-field bounds (>=128 bits) use asymmetric centering (no +1 correction).
/// Smaller bounds use symmetric centering (possible +1 correction).
///
/// # Panics
///
/// Panics if `log_basis` is 0 or at least 128.
pub fn num_digits_for_bound(log_bound: u32, log_basis: u32) -> usize {
    num_digits_for_bound_with_field_bits(log_bound, 128, log_basis)
}

/// Choose the correct digit-count function for an explicit field bit width.
///
/// Full-field bounds (`log_bound >= field_bits`) use asymmetric centering
/// against `field_bits`. Smaller bounds use symmetric centering.
///
/// # Panics
///
/// Panics if `log_basis` is 0 or at least 128, or if the effective symmetric
/// bound exceeds 128 bits.
pub fn num_digits_for_bound_with_field_bits(
    log_bound: u32,
    field_bits: u32,
    log_basis: u32,
) -> usize {
    if log_bound >= field_bits {
        compute_num_digits_full_field(field_bits, log_basis)
    } else {
        compute_num_digits(log_bound, log_basis)
    }
}

/// Return the row gadget scalars `1, b, b^2, ...` for `b = 2^log_basis`.
pub fn gadget_row_scalars<F: FieldCore + CanonicalField>(levels: usize, log_basis: u32) -> Vec<F> {
    let base = F::from_canonical_u128_reduced(1u128 << log_basis);
    let mut out = Vec::with_capacity(levels);
    let mut power = F::one();
    for _ in 0..levels {
        out.push(power);
        power *= base;
    }
    out
}

/// Number of balanced digits needed to decompose the folded witness `z_pre`.
///
/// The folding step multiplies the witness by a sparse challenge vector with
/// L1-mass `challenge_l1_mass` and applies `r_vars` levels of block folding,
/// producing entries bounded by:
///
///   β = challenge_l1_mass · num_claims · 2^(r_vars + log_basis - 1)
///
/// Used by both singleton and batched root paths; singleton callers pass
/// `num_claims = 1`.
///
/// Falls back to the field-width ceiling when the shift overflows or the
/// mass is zero.
///
/// # Panics
///
/// Panics if `log_basis` is 0 or at least 128.
pub fn compute_num_digits_fold_with_claims(
    r_vars: usize,
    challenge_l1_mass: usize,
    log_basis: u32,
    num_claims: usize,
) -> usize {
    compute_num_digits_fold_with_claims_for_field(
        r_vars,
        challenge_l1_mass,
        log_basis,
        num_claims,
        128,
    )
}

/// Field-width-aware variant of [`compute_num_digits_fold_with_claims`].
///
/// # Panics
///
/// Panics if `log_basis` is 0 or at least 128.
pub fn compute_num_digits_fold_with_claims_for_field(
    r_vars: usize,
    challenge_l1_mass: usize,
    log_basis: u32,
    num_claims: usize,
    field_bits: u32,
) -> usize {
    let shift = r_vars + (log_basis as usize) - 1;
    if shift >= 127 || challenge_l1_mass == 0 {
        return compute_num_digits_full_field(field_bits, log_basis);
    }
    let beta = (challenge_l1_mass as u128)
        .saturating_mul(num_claims as u128)
        .saturating_mul(1u128 << shift);
    if beta == 0 {
        return 1;
    }
    let log_beta = 128 - beta.leading_zeros();
    num_digits_for_bound_with_field_bits(log_beta, field_bits, log_basis)
}

/// Find the `(m, r)` split of `reduced_vars` that minimizes next-level witness size.
///
/// # Background (Akita paper, Section 4.5)
///
/// After removing the ring dimension (`α = log2(D)` variables), the remaining
/// `reduced_vars = ℓ - α` variables are partitioned as `m + r = reduced_vars`.
/// The witness is viewed as a matrix: `2^r` block-columns and `m_eff` rows.
///
/// The paper's witness-size formula (in ring elements, dropping the constant
/// quotient term `|r|` which doesn't depend on the split):
///
/// ```text
///   witness_size = |t̂| + |ŵ|           +  |ẑ|
///           = (n_A+1)·2^r · δ     +  (τ+1)·2^m · δ
/// ```
///
/// The planner refines the paper's single `δ` into three concrete digit counts,
/// since they have different magnitude bounds in practice:
///
/// ```text
///   witness_size = (1 + n_A) · δ_open · 2^r  +  δ_commit · δ_fold · m_eff
///              ─────────────────────────     ────────────────────────
///              |t̂| + |ŵ|  (opening)         |ẑ|  (folded witness)
/// ```
///
/// where:
/// - `δ_open` = digits to decompose opening entries (bound `max(log_commit_bound, 128)`),
///   used for both `ŵ` and `t̂` since both are opened at the field level
/// - `δ_commit` = digits to decompose commitment entries (bound `log_commit_bound`)
/// - `δ_fold` = digits to decompose folded entries (bound `β = 2^r · ω · b`,
///   corresponds to paper's `τ`)
/// - `m_eff` = effective row count (see below)
///
/// # The tradeoff
///
/// As `r` increases: term 1 grows exponentially (`2^r`), but term 2 shrinks
/// because `m_eff` decreases. However, `δ_fold` also grows with `r` (larger
/// `β`), partially counteracting the shrinkage. There is no closed-form
/// optimum, so we brute-force all valid splits.
///
/// The paper sets `m = r` for asymptotic analysis; the planner finds the true
/// optimum which is generally asymmetric (especially for onehot where
/// `δ_commit = 1`).
///
/// # Tight z_pre mode
///
/// - `num_ring > 0`: `m_eff = ⌈num_ring / 2^r⌉` — the actual occupied row
///   count, which can be smaller than `2^m` when the ring-element count isn't
///   a power of two.
/// - `num_ring = 0`: `m_eff = 2^m` — the standard power-of-two upper bound.
///
/// # Panics
///
/// Panics if `log_basis` is 0 or at least 128.
pub fn optimal_m_r_split(
    n_a: u32,
    challenge_l1_mass: usize,
    log_commit_bound: u32,
    log_basis: u32,
    reduced_vars: usize,
    num_ring: usize,
) -> (usize, usize) {
    optimal_m_r_split_with_field_bits(
        n_a,
        challenge_l1_mass,
        log_commit_bound,
        log_basis,
        reduced_vars,
        num_ring,
        128,
    )
}

/// Field-width-aware variant of [`optimal_m_r_split`].
///
/// # Panics
///
/// Panics if `log_basis` is 0 or at least 128.
#[allow(clippy::too_many_arguments)]
pub fn optimal_m_r_split_with_field_bits(
    n_a: u32,
    challenge_l1_mass: usize,
    log_commit_bound: u32,
    log_basis: u32,
    reduced_vars: usize,
    num_ring: usize,
    field_bits: u32,
) -> (usize, usize) {
    // Too few variables to optimize; too many would overflow `2^r` in u64.
    // Fall back to the paper's symmetric split m = r.
    if reduced_vars <= 2 || reduced_vars >= 53 {
        let r = reduced_vars / 2;
        return (reduced_vars - r, r);
    }

    let open_bound = log_commit_bound.max(field_bits);
    let delta_open = num_digits_for_bound_with_field_bits(open_bound, field_bits, log_basis) as u64;
    let delta_commit =
        num_digits_for_bound_with_field_bits(log_commit_bound, field_bits, log_basis) as u64;

    // Per-block cost from |t̂| + |ŵ|: each of the 2^r blocks contributes
    // (1 + n_A) · δ_open ring elements, matching the witness construction
    // where both ŵ and t̂ use δ_open.
    let per_block_cost = delta_open + n_a as u64 * delta_open;

    let mut best = (u64::MAX, reduced_vars / 2); // (cost, r)

    for r in 1..reduced_vars {
        let num_blocks = 1u64 << r;

        // Effective row count: exact when we know the ring count, else 2^m.
        let m_eff = if num_ring > 0 {
            num_ring.div_ceil(1usize << r) as u64
        } else {
            1u64 << (reduced_vars - r)
        };

        // δ_fold grows with r because β = 2^r · challenge_l1_mass · 2^(lb-1).
        let delta_fold = compute_num_digits_fold_with_claims_for_field(
            r,
            challenge_l1_mass,
            log_basis,
            1,
            field_bits,
        ) as u64;

        // |t̂| + |ŵ|                    +  |ẑ|
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

/// Baseline variant of [`optimal_m_r_split`] with `num_ring = 0` (standard
/// power-of-two upper bound for `m_eff`).
///
/// # Panics
///
/// Panics if `log_basis` is 0 or at least 128.
pub fn baseline_optimal_m_r_split(
    n_a: u32,
    challenge_l1_mass: usize,
    log_commit_bound: u32,
    log_basis: u32,
    reduced_vars: usize,
) -> (usize, usize) {
    optimal_m_r_split(
        n_a,
        challenge_l1_mass,
        log_commit_bound,
        log_basis,
        reduced_vars,
        0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_digit_max_base4() {
        // base=4, 2 digits: max digit=1, weights 1+4=5, max_pos = 1*5 = 5
        assert_eq!(balanced_digit_max(2, 2), 5);
    }

    #[test]
    fn balanced_digit_max_base8() {
        // base=8, 1 digit: max digit=3, max_pos = 3
        assert_eq!(balanced_digit_max(3, 1), 3);
    }

    #[test]
    fn digits_basic() {
        assert_eq!(compute_num_digits(128, 2), 65);
        assert_eq!(compute_num_digits(128, 3), 43);
        assert_eq!(compute_num_digits(1, 2), 1);
        assert_eq!(compute_num_digits(0, 2), 1);
    }

    #[test]
    fn full_field_digits() {
        assert_eq!(compute_num_digits_full_field(128, 2), 64);
        assert_eq!(compute_num_digits_full_field(128, 3), 43);
        assert_eq!(compute_num_digits_full_field(128, 4), 32);
        assert_eq!(compute_num_digits_full_field(128, 8), 16);
    }

    #[test]
    fn num_digits_for_bound_selects_correctly() {
        // Full-field (128 bits): asymmetric centering, no +1
        assert_eq!(num_digits_for_bound(128, 2), 64);
        // Sub-field bound: symmetric centering
        assert_eq!(num_digits_for_bound(10, 2), compute_num_digits(10, 2));
        // Full-field with log_basis=3: asymmetric gives same as symmetric
        assert_eq!(num_digits_for_bound(128, 3), 43);
    }

    #[test]
    fn digits_fold_basic() {
        let got_2 = compute_num_digits_fold_with_claims(12, 54, 2, 1);
        let got_3 = compute_num_digits_fold_with_claims(12, 54, 3, 1);
        assert!(got_2 > 0);
        assert!(got_3 > 0);
        assert!(got_2 >= got_3);
    }

    #[test]
    fn digits_fold_monotonic_in_claims() {
        let (r, mass, lb) = (8, 54, 3);
        let d1 = compute_num_digits_fold_with_claims(r, mass, lb, 1);
        let d4 = compute_num_digits_fold_with_claims(r, mass, lb, 4);
        let d16 = compute_num_digits_fold_with_claims(r, mass, lb, 16);
        assert!(d1 <= d4, "more claims should need >= digits");
        assert!(d4 <= d16, "more claims should need >= digits");
    }
}
