//! Gadget-decomposition digit counts and the committed-matrix widths derived
//! from them.
//!
//! Three layers live here, lowest to highest:
//!
//! 1. **Core digit-count math** — how many balanced base-`2^log_basis` digits
//!    represent a bound. Two centering conventions exist:
//!    - `compute_num_digits` (crate-private): the *symmetric* signed range
//!      `[-2^(k-1), 2^(k-1) - 1]`, including the sign-bit correction. Reached
//!      only through the router below.
//!    - [`compute_num_digits_field_width`]: an arbitrary field element's
//!      *asymmetric* residue, using plain `ceil(field_bits / log_basis)` with no
//!      correction.
//!    - [`num_digits_for_bound`]: the router. Field-width bounds
//!      (`log_bound >= field_bits`) use the asymmetric count; smaller bounds use
//!      the symmetric one. This is the *only* symmetric entry point, so a caller
//!      cannot accidentally request the symmetric count of a field-width bound
//!      (the historical `compute_num_digits(128, _)` footgun).
//!
//! 2. **Per-role selectors** — map a [`DecompositionParams`] to the digit depth
//!    of a specific witness role, encoding which bound applies to each:
//!    - [`num_digits_inner`]: committed witness `s` (`log_commit_bound` at
//!      the root, `log_basis` at recursive levels).
//!    - [`num_digits_open`]: opening witnesses `t̂` / `ŵ` (`log_open_bound`).
//!    - [`super::honest_fold_policy::HonestFoldPolicy`]: folded witness `z` — the
//!      group-owned offline policy returns its exact scheduled digit count.
//!
//! 3. **Committed-matrix widths** — name the `checked_mul` products that turn a
//!    digit depth plus block geometry into a matrix's ring-column count:
//!    [`decomposed_s_block_ring_count`] (A), [`decomposed_t_ring_count`] (B),
//!    [`decomposed_w_ring_count`] (D). These are layout arithmetic, not digit
//!    math; they sit here so each width formula lives beside the depth it
//!    multiplies.

use crate::DecompositionParams;

/// Signed coefficient interval represented by `num_digits` balanced
/// base-`2^log_basis` digits, returned as `(negative_abs_reach, positive_reach)`.
///
/// This is the accepted envelope for **any** balanced-digit plane: the folded
/// response `z` the verifier admits, and equally the committed source `s` a
/// bounded commitment can represent. A source coefficient outside this interval
/// cannot be recovered from its `num_digits` digits, so a producer must reject it
/// rather than commit the truncation
/// (see [`crate::DecompositionParams::log_commit_bound`]).
///
/// Both sides **saturate to a conservative lower bound** once the true reach
/// exceeds `u128::MAX`, inheriting that behavior from the underlying
/// `balanced_digit_max` / [`balanced_digit_abs_max`] series. A caller that needs
/// to distinguish "the reach is this value" from "the reach is beyond `u128`" must
/// use [`checked_balanced_digit_representable_bounds`]; comparing a saturated
/// value against a real coefficient rejects legitimate inputs.
#[inline]
#[must_use]
pub fn balanced_digit_representable_bounds(log_basis: u32, num_digits: usize) -> (u128, u128) {
    let num_digits = num_digits.max(1);
    (
        balanced_digit_abs_max(log_basis, num_digits),
        balanced_digit_max(log_basis, num_digits),
    )
}

/// Exact signed coefficient interval, as
/// `(negative_abs_reach, positive_reach)`, with `None` on a side whose true reach
/// exceeds `u128::MAX`.
///
/// `None` means "unbounded for any `u128` coefficient", which is the distinction
/// [`balanced_digit_representable_bounds`] loses by saturating. A commit-side
/// range check needs it: a full-field decomposition can span more than 128 bits
/// (`ceil(128 / 11) * 11 = 132`), and treating a saturated lower bound as the
/// accepted interval would reject valid field elements.
///
/// Total function: a degenerate `log_basis` (`0`, or `>= 128`) yields a
/// `(0, 0)`-reach answer rather than panicking, so a verifier-reachable caller can
/// pass unvalidated schedule data through it.
#[must_use]
pub fn checked_balanced_digit_representable_bounds(
    log_basis: u32,
    num_digits: usize,
) -> (Option<u128>, Option<u128>) {
    if log_basis == 0 || log_basis >= 128 {
        return (Some(0), Some(0));
    }
    let base: u128 = 1u128 << log_basis;
    // `(b^n - 1) / (b - 1)` accumulated as `1 + b + ... + b^(n-1)`, so overflow is
    // detected rather than folded into a saturating quotient.
    let num_digits = num_digits.max(1);
    let mut series: Option<u128> = Some(0);
    let mut power: Option<u128> = Some(1);
    for _ in 0..num_digits {
        series = match (series, power) {
            (Some(total), Some(term)) => total.checked_add(term),
            _ => None,
        };
        if series.is_none() {
            break;
        }
        power = power.and_then(|term| term.checked_mul(base));
    }
    let scale = |factor: u128| series.and_then(|total| factor.checked_mul(total));
    // Balanced digits span `[-b/2, b/2 - 1]`, so the positive factor is one less.
    (scale(base / 2), scale(base / 2 - 1))
}

/// Centered interval the **declared** committed-source bound alone admits, as
/// `(negative_abs_reach, positive_reach)`.
///
/// Following [`crate::DecompositionParams::log_commit_bound`], a bound of `k`
/// signed bits is the range `[-2^(k-1), 2^(k-1) - 1]`. Two declarations are
/// deliberately *unconstrained* here and return `(None, None)`:
///
/// - **The full-field endpoint** (`log_commit_bound == field_bits`). Every field
///   element is in range by construction, and the decomposition switches to
///   asymmetric centering there, so the symmetric interval would not describe it.
/// - **The unit one-hot endpoint** (`log_commit_bound == 1`). Its source is
///   structurally `{0, 1}` rather than range-checked, and the signed reading of
///   `k = 1` is `[-1, 0]`, which would reject a hot position. The `1` there
///   selects a depth of one digit; it was never an acceptance interval.
///
/// So this constrains exactly the interior, `1 < log_commit_bound < field_bits` —
/// the bounded-source case, which is the only one whose schedule is priced for a
/// range narrower than what its digits can represent.
#[must_use]
pub fn declared_committed_source_bounds(
    decomposition: DecompositionParams,
) -> (Option<u128>, Option<u128>) {
    if !decomposition.has_bounded_committed_source() || decomposition.log_commit_bound <= 1 {
        return (None, None);
    }
    // `k` signed bits is one sign bit plus `k - 1` magnitude bits.
    let negative_abs = 1u128.checked_shl(decomposition.log_commit_bound - 1);
    (negative_abs, negative_abs.map(|reach| reach - 1))
}

/// Centered interval a committed source must fit, as
/// `(negative_abs_reach, positive_reach)` with `None` meaning "beyond every
/// `u128`, so it cannot be exceeded".
///
/// A source has to satisfy **two independent** constraints, and this is their
/// intersection:
///
/// 1. **Representability** — it must be recoverable from `num_digits_inner`
///    balanced base-`2^log_basis_inner` digits, i.e. lie inside
///    [`checked_balanced_digit_representable_bounds`]. Outside it the
///    decomposition keeps only the scheduled digits and the commitment binds a
///    truncation.
/// 2. **Declaration** — it must lie inside
///    [`declared_committed_source_bounds`], the range the schedule was *planned*
///    for. The planner prices a bounded source's final digit plane at only the
///    range its bound leaves, so a coefficient beyond the declaration inflates
///    the level-1 witness past the L2 response caps frozen into the suffix.
///
/// The two differ because `num_digits_inner` rounds up: 13 base-`2^5` digits span
/// 65 bits, so they *represent* about `±2^64` while a `log_commit_bound = 64`
/// schedule is *priced* for `±2^63`. Checking only representability would accept
/// coefficients the schedule never declared — up to 256x the declared bound at
/// the shipped `log_basis_inner = 9` geometry.
#[must_use]
pub fn accepted_committed_source_bounds(
    decomposition: DecompositionParams,
    log_basis_inner: u32,
    num_digits_inner: usize,
) -> (Option<u128>, Option<u128>) {
    let (representable_negative, representable_positive) =
        checked_balanced_digit_representable_bounds(log_basis_inner, num_digits_inner);
    let (declared_negative, declared_positive) = declared_committed_source_bounds(decomposition);
    (
        tighter_reach(representable_negative, declared_negative),
        tighter_reach(representable_positive, declared_positive),
    )
}

/// Smaller of two reaches, where `None` is "beyond every `u128`" and therefore
/// never the tighter side.
fn tighter_reach(left: Option<u128>, right: Option<u128>) -> Option<u128> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(only), None) | (None, Some(only)) => Some(only),
        (None, None) => None,
    }
}

/// Minimum balanced-digit depth whose exact signed range contains
/// `[-cap, cap]`.
///
/// Unlike [`num_digits_for_bound`], this function accepts an exact integer
/// magnitude instead of a power-of-two bit width. The positive reach is the
/// binding side because balanced digits extend farther in the negative
/// direction.
#[inline]
#[must_use]
pub fn num_digits_for_linf_cap(cap: u128, field_bits: u32, log_basis: u32) -> usize {
    assert!(log_basis > 0 && log_basis < 128, "invalid log_basis");
    if cap == 0 {
        return 1;
    }
    let signed_bits = (u128::BITS - cap.leading_zeros()).saturating_add(1);
    if signed_bits >= field_bits {
        return compute_num_digits_field_width(field_bits, log_basis);
    }
    let fallback = num_digits_for_bound(signed_bits, field_bits, log_basis);
    for digits in 1..fallback {
        if balanced_digit_max(log_basis, digits) >= cap {
            return digits;
        }
    }
    fallback
}

/// Maximum positive value representable by `num_digits` balanced base-`b`
/// digits, where `b = 2^log_basis`. Each balanced digit lies in
/// `[-b/2, b/2 - 1]`; the max positive value is the geometric series
/// `(b/2 - 1) · (b^n - 1) / (b - 1)`. When `b^n` overflows `u128` the result is
/// a conservative lower bound (safe: it can only add a digit, never drop one).
pub(crate) fn balanced_digit_max(log_basis: u32, num_digits: usize) -> u128 {
    let base: u128 = 1u128 << log_basis;
    let max_digit = base / 2 - 1;
    let base_minus_1 = base - 1;

    let mut base_pow = 1u128;
    for _ in 0..num_digits {
        base_pow = base_pow.saturating_mul(base);
    }

    max_digit.saturating_mul(base_pow.saturating_sub(1) / base_minus_1)
}

/// Maximum absolute value accepted by `num_digits` balanced base-`b` digits,
/// i.e. the negative reach `(b/2) · (b^n - 1)/(b - 1)`.
///
/// This is the coefficient-`L∞` envelope the verifier accepts for the folded
/// witness `z`: stage-1 digit membership admits every balanced `num_digits`-digit
/// string, and balanced digits `[-b/2, b/2 - 1]` reach further on the negative
/// side, so the absolute envelope is this negative reach.
#[inline]
#[must_use]
pub fn balanced_digit_abs_max(log_basis: u32, num_digits: usize) -> u128 {
    let base: u128 = 1u128 << log_basis;
    let max_abs_digit = base / 2;

    let mut pow = 1u128;
    let mut series = 0u128;
    for _ in 0..num_digits {
        series = series.saturating_add(pow);
        pow = pow.saturating_mul(base);
    }

    max_abs_digit.saturating_mul(series)
}

/// Minimum number of balanced base-`2^log_basis` digits needed to represent a
/// `log_bound`-bit *signed* coefficient, using symmetric centering.
///
/// Following [`crate::DecompositionParams::log_commit_bound`], a bound of `k`
/// bits denotes the centered range `[-2^(k-1), 2^(k-1) - 1]`, i.e. one sign bit
/// plus `k-1` magnitude bits. The binding constraint is the positive end,
/// `2^(k-1) - 1`, since the balanced digit range `[-b/2, b/2 - 1]` reaches
/// further on the negative side. This is *not* `2^log_bound - 1`: the leading
/// bit is the sign, so callers that mean "magnitude up to `2^m`" must pass
/// `log_bound = m + 1`.
///
/// The count is `ceil(log_bound / log_basis)`, plus one more digit when the
/// balanced-digit positive reach `balanced_digit_max` still falls short of
/// `2^(log_bound-1) - 1`. The extra digit is only ever needed when `log_basis`
/// divides `log_bound` exactly (otherwise `ceil(log_bound/log_basis)·log_basis
/// > log_bound`, so the reach already clears `2^(log_bound-1)`); the check is
/// run unconditionally because it is cheap and self-evidently correct. Both the
/// coverage and the minimality of the result are pinned by the
/// `compute_num_digits_covers_signed_range` unit test.
///
/// This symmetric count is for *small* bounds (`log_bound < field_bits`):
/// one-hot `log_commit_bound = 1`, recursive `log_basis`, and fold `log_beta`.
/// It is crate-private and reached only through [`num_digits_for_bound`], which
/// routes field-width bounds to the asymmetric [`compute_num_digits_field_width`]
/// instead — so no caller can ask for the symmetric count of a field-width bound.
///
/// # Panics
///
/// Panics if `log_basis` is 0 or at least 128, or if `log_bound` exceeds 128.
pub(crate) fn compute_num_digits(log_bound: u32, log_basis: u32) -> usize {
    assert!(log_basis > 0 && log_basis < 128, "invalid log_basis");
    assert!(
        log_bound <= 128,
        "log_bound={log_bound} exceeds 128-bit field"
    );

    if log_bound == 0 {
        return 1;
    }

    let mut num_digits = (log_bound as usize).div_ceil(log_basis as usize);
    let required_positive = (1u128 << (log_bound - 1)).saturating_sub(1);
    if balanced_digit_max(log_basis, num_digits) < required_positive {
        num_digits += 1;
    }
    num_digits.max(1)
}

/// Decomposition depth for arbitrary field elements using asymmetric centering:
/// `ceil(field_bits / log_basis)` with no +1 correction.
///
/// # Panics
///
/// Panics if `log_basis` is 0 or >= 128.
pub fn compute_num_digits_field_width(field_bits: u32, log_basis: u32) -> usize {
    assert!(log_basis > 0 && log_basis < 128, "invalid log_basis");
    if field_bits == 0 {
        return 1;
    }
    (field_bits as usize).div_ceil(log_basis as usize).max(1)
}

/// Choose the correct digit-count function for an explicit field bit width.
/// Field-width bounds (`log_bound >= field_bits`) use asymmetric centering;
/// smaller bounds use symmetric centering.
///
/// # Panics
///
/// Panics if `log_basis` is 0 or at least 128, or if the effective symmetric
/// bound exceeds 128 bits.
pub fn num_digits_for_bound(log_bound: u32, field_bits: u32, log_basis: u32) -> usize {
    if log_bound >= field_bits {
        compute_num_digits_field_width(field_bits, log_basis)
    } else {
        compute_num_digits(log_bound, log_basis)
    }
}

/// `δ_commit`: digits per coefficient of the committed witness `s`, using the
/// level's gadget base `decomposition.log_basis`.
///
/// The root commits against its configured `log_commit_bound`; a recursive
/// level commits the balanced-digit witness, whose commit bound collapses to
/// `log_basis`.
pub fn num_digits_inner(decomposition: DecompositionParams, is_root: bool) -> usize {
    let bound = if is_root {
        decomposition.log_commit_bound
    } else {
        decomposition.log_basis
    };
    num_digits_inner_for_bound(decomposition, bound)
}

/// `δ_commit` for an explicit source coefficient bound.
///
/// This is the general planner entry point when the source bound and selected
/// A decomposition basis are independent. `source_log_bound` follows the same
/// signed-bound convention as [`num_digits_for_bound`].
pub fn num_digits_inner_for_bound(
    decomposition: DecompositionParams,
    source_log_bound: u32,
) -> usize {
    num_digits_for_bound(
        source_log_bound,
        decomposition.field_bits(),
        decomposition.log_basis,
    )
}

/// `δ_setup`: digits per coefficient for setup-prefix commitments.
///
/// Setup prefixes commit raw shared-setup field elements, not the already-small
/// recursive witness digits. Their commit-side decomposition must therefore
/// cover the full configured field width.
pub fn num_digits_setup_prefix_commit(decomposition: DecompositionParams) -> usize {
    compute_num_digits_field_width(decomposition.field_bits(), decomposition.log_basis)
}

/// `δ_open`: digits per coefficient of the opening witnesses `t̂` / `ŵ`,
/// which are opened at the field level (`log_open_bound`).
pub fn num_digits_open(decomposition: DecompositionParams) -> usize {
    let field_bits = decomposition.field_bits();
    let bound = decomposition
        .log_open_bound
        .unwrap_or(decomposition.log_commit_bound);
    num_digits_for_bound(bound, field_bits, decomposition.log_basis)
}

/// A-matrix committed width (ring columns): `num_positions_per_block · δ_commit`.
#[inline]
pub fn decomposed_s_block_ring_count(
    num_positions_per_block: usize,
    num_digits_inner: usize,
) -> Option<usize> {
    num_positions_per_block.checked_mul(num_digits_inner)
}

/// B-matrix committed width (ring columns): `n_a · δ_open · num_live_blocks · num_polynomials`.
#[inline]
pub fn decomposed_t_ring_count(
    n_a: usize,
    num_digits_open: usize,
    num_live_blocks: usize,
    num_polynomials: usize,
) -> Option<usize> {
    n_a.checked_mul(num_digits_open)?
        .checked_mul(num_live_blocks)?
        .checked_mul(num_polynomials)
}

/// D-matrix committed width (ring columns): `δ_open · num_live_blocks · num_polynomials`.
#[inline]
pub fn decomposed_w_ring_count(
    num_digits_open: usize,
    num_live_blocks: usize,
    num_polynomials: usize,
) -> Option<usize> {
    num_digits_open
        .checked_mul(num_live_blocks)?
        .checked_mul(num_polynomials)
}

/// Convert an A-native ring-column count into the physical column count of a
/// projected B- or D-native role.
///
/// The role dimension must divide the source dimension exactly.
#[inline]
pub fn projected_role_ring_count(
    source_dimension: usize,
    role_dimension: usize,
    native_ring_count: usize,
) -> Option<usize> {
    if role_dimension == 0 || !source_dimension.is_multiple_of(role_dimension) {
        return None;
    }
    native_ring_count.checked_mul(source_dimension / role_dimension)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sis::{fold_witness_beta_inf, FoldChallengeNorms, FoldWitnessNorms};

    #[test]
    fn balanced_digit_max_cases() {
        assert_eq!(balanced_digit_max(2, 2), 5);
        assert_eq!(balanced_digit_max(3, 1), 3);
    }

    #[test]
    fn balanced_digit_abs_max_uses_negative_reach() {
        // b = 4, δ = 3 digits represent [-42, 21]; A-role pricing must use
        // the accepted absolute envelope, not the shorter positive side.
        assert_eq!(balanced_digit_max(2, 3), 21);
        assert_eq!(balanced_digit_abs_max(2, 3), 42);
        // b = 8, δ = 2 digits represent [-36, 27].
        assert_eq!(balanced_digit_max(3, 2), 27);
        assert_eq!(balanced_digit_abs_max(3, 2), 36);
    }

    /// The checked reaches agree with the saturating ones inside `u128` and
    /// report `None` beyond it.
    ///
    /// The saturating pair is a deliberate conservative *lower* bound
    /// (`balanced_digit_max` divides a saturated `b^n`), which is correct for
    /// choosing a digit depth but wrong as an acceptance interval. A commit-side
    /// range check must not read it as exact.
    #[test]
    fn checked_reaches_are_exact_where_the_saturating_reaches_are() {
        for (log_basis, num_digits) in [(2u32, 3usize), (3, 2), (5, 13), (11, 11)] {
            let (checked_negative, checked_positive) =
                checked_balanced_digit_representable_bounds(log_basis, num_digits);
            let (negative, positive) = balanced_digit_representable_bounds(log_basis, num_digits);
            assert_eq!(checked_negative, Some(negative));
            assert_eq!(checked_positive, Some(positive));
        }

        // `ceil(128 / 11) = 12` digits of base 2^11 span 132 bits, so both true
        // reaches exceed `u128::MAX` and the saturating pair understates them.
        let (negative, positive) = checked_balanced_digit_representable_bounds(11, 12);
        assert_eq!((negative, positive), (None, None));
        let (saturating_negative, saturating_positive) =
            balanced_digit_representable_bounds(11, 12);
        assert!(saturating_positive < u128::MAX / 2);
        assert_eq!(saturating_negative, u128::MAX);

        // Total over degenerate bases: verifier-reachable callers pass unvalidated
        // schedule data, so this must answer rather than panic.
        for log_basis in [0u32, 128, u32::MAX] {
            assert_eq!(
                checked_balanced_digit_representable_bounds(log_basis, 4),
                (Some(0), Some(0))
            );
        }
        // Base 2 balanced digits are `{-1, 0}`: four of them reach `-15` and
        // nothing positive. Not a production basis, but the reach must still be
        // the exact one rather than a wrapped `base / 2 - 1`.
        assert_eq!(
            checked_balanced_digit_representable_bounds(1, 4),
            (Some(15), Some(0))
        );
    }

    fn decomposition(log_commit_bound: u32, log_open_bound: Option<u32>) -> DecompositionParams {
        DecompositionParams {
            log_basis: 3,
            log_commit_bound,
            log_open_bound,
        }
    }

    /// Only an interior bound constrains; both endpoints are unconstrained, and
    /// for opposite reasons.
    #[test]
    fn declared_bounds_constrain_only_the_interior_of_the_range() {
        // Full-field endpoint: every field element is in range by construction.
        assert_eq!(
            declared_committed_source_bounds(decomposition(128, Some(128))),
            (None, None)
        );
        assert_eq!(
            declared_committed_source_bounds(decomposition(128, None)),
            (None, None)
        );
        // Unit one-hot endpoint: the source is structurally `{0, 1}`. Reading
        // `k = 1` as the signed range `[-1, 0]` would reject a hot position, so
        // this must stay unconstrained.
        assert_eq!(
            declared_committed_source_bounds(decomposition(1, Some(128))),
            (None, None)
        );
        // Interior: the signed reading, one sign bit plus `k - 1` magnitude bits.
        assert_eq!(
            declared_committed_source_bounds(decomposition(64, Some(128))),
            (Some(1 << 63), Some((1u128 << 63) - 1))
        );
        // A `u64` workload declares 65, not 64: its magnitude reaches `2^64 - 1`.
        assert_eq!(
            declared_committed_source_bounds(decomposition(65, Some(128))),
            (Some(1 << 64), Some(u128::from(u64::MAX)))
        );
    }

    /// The accepted interval is the intersection of representability and
    /// declaration, and the declaration is the binding side for every shipped
    /// bounded geometry.
    #[test]
    fn accepted_bounds_intersect_representability_with_the_declaration() {
        // `log_basis_inner` / `num_digits_inner` of the shipped bounded rows at
        // `log_commit_bound = 65`.
        for (log_basis_inner, num_digits_inner) in [(5u32, 14usize), (9, 8), (7, 10)] {
            let accepted = accepted_committed_source_bounds(
                decomposition(65, Some(128)),
                log_basis_inner,
                num_digits_inner,
            );
            assert_eq!(
                accepted,
                (Some(1 << 64), Some(u128::from(u64::MAX))),
                "the declaration must bind at lb={log_basis_inner} delta={num_digits_inner}"
            );
            // A full `u64` sits exactly on the positive endpoint, and nothing above
            // it is admitted. This is the property the bounded preset exists for.
            assert_eq!(accepted.1, Some(u128::from(u64::MAX)));
        }

        // Representability binds when the digits cannot even reach the
        // declaration: 2 base-2^5 digits span 10 bits, far below a 65-bit bound.
        let (negative, positive) =
            accepted_committed_source_bounds(decomposition(65, Some(128)), 5, 2);
        assert_eq!(
            (negative, positive),
            (
                Some(balanced_digit_abs_max(5, 2)),
                Some(balanced_digit_max(5, 2))
            )
        );

        // Full-field: only representability applies, and a 12-digit base-2^11
        // decomposition spans 132 bits, beyond every `u128` on both sides.
        assert_eq!(
            accepted_committed_source_bounds(decomposition(128, Some(128)), 11, 12),
            (None, None)
        );

        // Unit one-hot: the hot value `1` must stay inside the accepted interval.
        let (_, one_hot_positive) =
            accepted_committed_source_bounds(decomposition(1, Some(128)), 3, 1);
        assert!(
            one_hot_positive.is_some_and(|reach| reach >= 1),
            "a one-hot source must be able to commit its hot position"
        );
    }

    /// The gap this intersection closes, stated in the numbers that motivated it.
    ///
    /// `num_digits_for_bound` rounds up, so the representable envelope overshoots
    /// the declaration — by 256x at the `log_basis_inner = 9` geometry the nv=24
    /// row selects. Checking representability alone would accept coefficients the
    /// schedule was never priced for.
    #[test]
    fn representable_envelope_overshoots_the_declaration_it_was_sized_from() {
        for (log_basis_inner, expected_ratio) in [(5u32, 1), (9, 255), (7, 63)] {
            let params = decomposition(64, Some(128));
            let num_digits_inner = num_digits_inner_for_bound(
                DecompositionParams {
                    log_basis: log_basis_inner,
                    ..params
                },
                params.log_commit_bound,
            );
            let (_, representable) =
                checked_balanced_digit_representable_bounds(log_basis_inner, num_digits_inner);
            let (_, declared) = declared_committed_source_bounds(params);
            let representable = representable.expect("shipped geometries fit u128");
            let declared = declared.expect("an interior bound is constrained");
            assert!(
                representable / declared >= expected_ratio,
                "lb={log_basis_inner}: envelope {representable} vs declaration {declared}"
            );
            // ...and the intersection is what the declaration says.
            assert_eq!(
                accepted_committed_source_bounds(params, log_basis_inner, num_digits_inner).1,
                Some(declared)
            );
        }
    }

    #[test]
    fn exact_linf_cap_does_not_round_through_a_power_of_two_range() {
        assert_eq!(num_digits_for_linf_cap(20, 64, 3), 2);
        assert_eq!(num_digits_for_linf_cap(27, 64, 3), 2);
        assert_eq!(num_digits_for_linf_cap(28, 64, 3), 3);
        assert_eq!(num_digits_for_linf_cap(1_755, 128, 3), 4);
        assert_eq!(num_digits_for_linf_cap(1_756, 128, 3), 5);
    }

    #[test]
    fn digits_basic() {
        // Production `compute_num_digits` inputs are small symmetric bounds:
        // one-hot `log_commit_bound = 1`, recursive `log_basis`, fold
        // `log_beta`. Field-width bounds go through `num_digits_for_bound` to
        // `compute_num_digits_field_width`, not here.
        assert_eq!(compute_num_digits(1, 2), 1);
        assert_eq!(compute_num_digits(0, 2), 1);
        // `log_basis` itself (the recursive commit bound): one base-`2^lb`
        // digit covers the balanced range `[-2^(lb-1), 2^(lb-1) - 1]` exactly.
        assert_eq!(compute_num_digits(2, 2), 1);
        assert_eq!(compute_num_digits(3, 3), 1);
    }

    /// The returned digit count must actually cover the signed range
    /// `[-2^(log_bound-1), 2^(log_bound-1) - 1]` its contract promises, for
    /// every production base and bound. This pins the invariant the previous
    /// conditional guard left unchecked whenever `log_basis ∤ log_bound`.
    #[test]
    fn compute_num_digits_covers_signed_range() {
        for log_basis in 2u32..=8 {
            for log_bound in 1u32..=120 {
                let n = compute_num_digits(log_bound, log_basis);
                let required_positive = (1u128 << (log_bound - 1)).saturating_sub(1);
                assert!(
                    balanced_digit_max(log_basis, n) >= required_positive,
                    "log_bound={log_bound} log_basis={log_basis} n={n} \
                     reach={} < required={required_positive}",
                    balanced_digit_max(log_basis, n),
                );
                // Minimality: one fewer digit must be insufficient (unless n==1).
                if n > 1 {
                    assert!(
                        balanced_digit_max(log_basis, n - 1) < required_positive,
                        "non-minimal: log_bound={log_bound} log_basis={log_basis} n={n}",
                    );
                }
            }
        }
    }

    #[test]
    fn field_element_digits() {
        assert_eq!(compute_num_digits_field_width(128, 2), 64);
        assert_eq!(compute_num_digits_field_width(128, 3), 43);
        assert_eq!(compute_num_digits_field_width(128, 4), 32);
        assert_eq!(compute_num_digits_field_width(128, 8), 16);
    }

    #[test]
    fn num_digits_for_bound_selects_correctly() {
        assert_eq!(num_digits_for_bound(128, 128, 2), 64);
        assert_eq!(num_digits_for_bound(10, 128, 2), compute_num_digits(10, 2));
        assert_eq!(num_digits_for_bound(128, 128, 3), 43);
    }

    #[test]
    fn widths_are_checked() {
        assert_eq!(decomposed_s_block_ring_count(4, 3), Some(12));
        assert_eq!(decomposed_t_ring_count(2, 3, 4, 5), Some(120));
        assert_eq!(decomposed_w_ring_count(3, 4, 5), Some(60));
        assert_eq!(decomposed_s_block_ring_count(usize::MAX, 2), None);
        assert_eq!(projected_role_ring_count(256, 64, 7), Some(28));
        assert_eq!(projected_role_ring_count(256, 128, 7), Some(14));
        assert_eq!(projected_role_ring_count(128, 256, 7), None);
        assert_eq!(projected_role_ring_count(256, 0, 7), None);
    }

    #[test]
    fn fold_witness_digit_plan_derives_beta() {
        // Dense witness (||s||_inf = b/2, ||s||_1 = D·b/2) picks the
        // ||c||_1·||s||_inf side; one-hot (||s||_1 = 1) picks ||c||_inf and
        // needs strictly fewer digits.
        let challenge = FoldChallengeNorms {
            infinity_norm: 8,
            l1_norm: 54,
        };
        // dense: log_basis=3 ⇒ ||s||_inf = b/2 = 4, ||s||_1 = D·b/2 = 64·4.
        let dense = FoldWitnessNorms::bounded(3, 64);
        // one-hot single-chunk: ||s||_inf = 1, ||s||_1 = 1.
        let onehot = FoldWitnessNorms::sparse_binary(64, 64).unwrap();
        let dense_beta = fold_witness_beta_inf(8, 1, challenge, dense).unwrap();
        let onehot_beta = fold_witness_beta_inf(8, 1, challenge, onehot).unwrap();
        let dense_digits =
            num_digits_for_bound((128 - dense_beta.leading_zeros()).saturating_add(1), 128, 3);
        let onehot_digits = num_digits_for_bound(
            (128 - onehot_beta.leading_zeros()).saturating_add(1),
            128,
            3,
        );
        assert!(dense_digits > 0 && onehot_digits > 0);
        assert!(onehot_digits < dense_digits);
        // More claims never reduce the digit count.
        let batched_beta = fold_witness_beta_inf(8, 4, challenge, dense).unwrap();
        let batched_digits = num_digits_for_bound(
            (128 - batched_beta.leading_zeros()).saturating_add(1),
            128,
            3,
        );
        assert!(batched_digits >= dense_digits);
    }

    #[test]
    fn fold_witness_digit_plan_rejects_zero_live_blocks() {
        let challenge = FoldChallengeNorms {
            infinity_norm: 8,
            l1_norm: 54,
        };
        let witness = FoldWitnessNorms::bounded(3, 64);
        // A fold must contain at least one live block.
        assert!(fold_witness_beta_inf(0, 1, challenge, witness).is_err());
    }
}
