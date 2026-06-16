//! Sum-of-four-squares solver for turning a bound into an equality.
//!
//! A non-strict integer bound `value <= bound` can be rewritten as an exact
//! equality using Lagrange's four-square theorem: the non-negative slack
//! `bound - value` is a sum of four integer squares, so
//!
//! ```text
//! value + ell_0^2 + ell_1^2 + ell_2^2 + ell_3^2 = bound.
//! ```
//!
//! [`four_squares`] is the pure solver for that step. It consumes only the slack
//! integer `target = bound - value` and returns the four witnesses
//! `ell_0..ell_3`.
//!
//! The prover uses it to certify a Euclidean-norm bound on the folded witness
//! `z`: there `value` is the realized squared norm `Σ z[i]^2` and `bound` is the
//! proven upper bound, so the four squares become the committed slack witnesses
//! of the certificate.
//!
//! ## Bound guarantee
//!
//! For `target < 2^64`, each returned `ell_j` is one of four non-negative squares
//! summing to `target`, hence `ell_j^2 <= target` and
//! `ell_j <= floor(sqrt(target)) <= 2^32 - 1 < 2^32`, so every slack witness
//! fits a 32-bit budget with no runtime clamp needed.
//!
//! [`four_squares_u128`] accepts any `target <= u128::MAX`. Witnesses still fit
//! `u64` because `floor(sqrt(u128::MAX)) < 2^64`; callers that encode slack in a
//! smaller digit budget must reject oversize witnesses separately.
//!
//! ## Algorithm
//!
//! The solver has two paths.
//!
//! The fast path is the Rabin–Shallit-style prime hunt. It reduces solving to
//! two-squares of a prime, which is cheap and avoids general integer
//! factorization on the common path:
//!
//! 1. Strip factors of 4: write `target = 4^e * m` with `m` not divisible by 4
//!    (so `m mod 4 in {1, 2, 3}`). A four-square representation of `m` scales by
//!    `2^e` into one of `target`.
//! 2. Hunt for `a, b` (scanning `a` downward from `floor(sqrt(m))`, then `b`
//!    downward so the residue `p = m - a^2 - b^2` grows from `0`) until `p` is a
//!    value we can split into two squares directly: `p in {0, 1, 2}`, or `p` is a
//!    prime `≡ 1 (mod 4)`.
//! 3. Split that `p` into `c^2 + d^2` (trivially for `{0,1,2}`; via the
//!    Hermite–Serret Euclidean descent on a square root of `-1 (mod p)` for the
//!    prime case).
//!
//! The totality path is a finite residual search backed by the two-squares
//! theorem. It enumerates `a, b`, factors the residual `p = m - a^2 - b^2` by
//! exact trial division, and constructs `p = c^2 + d^2` iff every prime
//! `q ≡ 3 (mod 4)` appears with even exponent. Lagrange's four-square theorem
//! guarantees that some enumerated residual is a sum of two squares, so this
//! fallback is total for every `u64` target.
//!
//! The decision path is integer-only (no floating point): [`u128::isqrt`],
//! deterministic Miller-Rabin for `u64`, Baillie-PSW probable-prime testing for
//! larger `u128` values, exact trial division, and widening modular arithmetic.

use akita_field::AkitaError;

/// Decompose `target` into four squares
/// `ell_0^2 + ell_1^2 + ell_2^2 + ell_3^2 = target`.
///
/// Every returned `ell_j` satisfies `ell_j < 2^32` when `target < 2^64` (see the
/// module docs). The returned tuple is unordered; callers must not depend on a
/// particular slot carrying the largest square.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidInput`] only on internal inconsistency: a
/// self-check on the returned squares failing, or the theorem-backed total
/// fallback finding no representable residual. Neither is reachable for a valid
/// `u64` target (Lagrange's theorem guarantees the fallback succeeds), and the
/// result is always re-verified before return.
pub fn four_squares(target: u64) -> Result<[u64; 4], AkitaError> {
    four_squares_u128(u128::from(target))
}

/// Decompose `target` into four squares when the slack may exceed `2^64`.
///
/// This is the entry point for small-field certificates where `B_l2 - Z_SQUARED`
/// can be a full `u128`. Each returned witness fits `u64` (see module docs).
///
/// # Errors
///
/// Same contract as [`four_squares`], extended to every `u128` target.
pub fn four_squares_u128(target: u128) -> Result<[u64; 4], AkitaError> {
    if target == 0 {
        return Ok([0, 0, 0, 0]);
    }

    // target = 4^e * m with m not divisible by 4; solve m, then scale by 2^e.
    let mut m = target;
    let mut scale = 1u64;
    while m.is_multiple_of(4) {
        m /= 4;
        scale = scale.checked_mul(2).ok_or_else(|| {
            AkitaError::InvalidInput("four_squares: factor-of-4 scale overflowed".to_string())
        })?;
    }

    if let Some(result) = fast_prime_hunt(m, scale, target)? {
        return Ok(result);
    }
    four_squares_via_two_square_residuals(m, scale, target)
}

/// Fast path: look for a residual that is either trivial or prime `1 mod 4`.
///
/// This usually succeeds quickly, but totality does not depend on it. If it
/// misses, [`four_squares_via_two_square_residuals`] performs the theorem-backed
/// finite fallback.
fn fast_prime_hunt(m: u128, scale: u64, target: u128) -> Result<Option<[u64; 4]>, AkitaError> {
    let a_max = m.isqrt();
    for a in (0..=a_max).rev() {
        let rem_a = m - a * a;
        let b_max = rem_a.isqrt();
        for b in (0..=b_max).rev() {
            let p = rem_a - b * b;
            let split = match p {
                0 => Some((0, 0)),
                1 => Some((1, 0)),
                2 => Some((1, 1)),
                _ if p % 4 == 1 && is_prime(p) => two_squares_prime(p).ok(),
                _ => None,
            };
            if let Some((c, d)) = split {
                let result = scale_decomposition([witness_u64(a)?, witness_u64(b)?, c, d], scale)?;
                return Ok(Some(verify_decomposition(result, target)?));
            }
        }
    }

    Ok(None)
}

/// Guaranteed fallback.
///
/// Proof of totality:
///
/// - By Lagrange's four-square theorem, `m = a^2 + b^2 + c^2 + d^2` for some
///   non-negative integers `a, b, c, d`.
/// - The nested loops enumerate every possible non-negative `a, b` with
///   `a^2 + b^2 <= m`, so they eventually visit the first two coordinates of
///   one such representation.
/// - For that visit, the residual is `p = c^2 + d^2`.
/// - [`two_squares`] is exact by the two-squares theorem, so it accepts that
///   residual and constructs a valid pair.
fn four_squares_via_two_square_residuals(
    m: u128,
    scale: u64,
    target: u128,
) -> Result<[u64; 4], AkitaError> {
    let a_max = m.isqrt();
    for a in (0..=a_max).rev() {
        let rem_a = m - a * a;
        let b_max = rem_a.isqrt();
        for b in (0..=b_max).rev() {
            let p = rem_a - b * b;
            let split = two_squares(p).unwrap_or_default();
            if let Some((c, d)) = split {
                let result = scale_decomposition([witness_u64(a)?, witness_u64(b)?, c, d], scale)?;
                return verify_decomposition(result, target);
            }
        }
    }

    Err(AkitaError::InvalidInput(format!(
        "four_squares: total fallback failed for target {target}"
    )))
}

fn witness_u64(v: u128) -> Result<u64, AkitaError> {
    u64::try_from(v).map_err(|_| {
        AkitaError::InvalidInput(format!(
            "four_squares: witness {v} exceeds u64 (internal inconsistency)"
        ))
    })
}

fn scale_decomposition(values: [u64; 4], scale: u64) -> Result<[u64; 4], AkitaError> {
    let mut out = [0u64; 4];
    for (dst, value) in out.iter_mut().zip(values) {
        *dst = value.checked_mul(scale).ok_or_else(|| {
            AkitaError::InvalidInput("four_squares: scale multiplication overflowed".to_string())
        })?;
    }
    Ok(out)
}

/// Re-check `sum result_j^2 == target` over `u128` before returning, so a
/// solver bug can never emit an invalid certificate.
fn verify_decomposition(result: [u64; 4], target: u128) -> Result<[u64; 4], AkitaError> {
    let sum: u128 = result.iter().map(|&v| u128::from(v) * u128::from(v)).sum();
    if sum == target {
        Ok(result)
    } else {
        Err(AkitaError::InvalidInput(format!(
            "four_squares: internal self-check failed (sum of squares {sum} != target {target})"
        )))
    }
}

/// Construct `n = a^2 + b^2` iff such a representation exists.
///
/// This is the constructive form of the two-squares theorem: a non-negative
/// integer is a sum of two squares iff every prime `3 mod 4` has even exponent
/// in its factorization. Prime `1 mod 4` factors are split by
/// [`two_squares_prime`], and representations are multiplied with the
/// Brahmagupta-Fibonacci identity.
fn two_squares(n: u128) -> Result<Option<(u64, u64)>, AkitaError> {
    match n {
        0 => return Ok(Some((0, 0))),
        1 => return Ok(Some((1, 0))),
        2 => return Ok(Some((1, 1))),
        _ => {}
    }

    if is_prime(n) {
        return if n % 4 == 1 {
            Ok(Some(two_squares_prime(n)?))
        } else {
            Ok(None)
        };
    }

    let factors = factor_counts_trial_division(n);
    let mut rep = (1u64, 0u64);
    for (p, exp) in factors {
        if p == 2 {
            rep = scale_two_square_rep(rep, checked_pow_u128(2, exp / 2)?)?;
            if exp % 2 == 1 {
                rep = multiply_two_square_reps(rep, (1, 1))?;
            }
        } else if p % 4 == 1 {
            let prime_rep = two_squares_prime(p)?;
            for _ in 0..exp {
                rep = multiply_two_square_reps(rep, prime_rep)?;
            }
        } else {
            if exp % 2 == 1 {
                return Ok(None);
            }
            rep = scale_two_square_rep(rep, checked_pow_u128(p, exp / 2)?)?;
        }
    }

    let sum = u128::from(rep.0) * u128::from(rep.0) + u128::from(rep.1) * u128::from(rep.1);
    if sum == n {
        Ok(Some(rep))
    } else {
        Err(AkitaError::InvalidInput(format!(
            "two_squares: internal self-check failed ({}^2 + {}^2 != {n})",
            rep.0, rep.1
        )))
    }
}

fn factor_counts_trial_division(mut n: u128) -> Vec<(u128, u32)> {
    let mut factors = Vec::new();
    let mut exp = 0u32;
    while n.is_multiple_of(2) {
        n /= 2;
        exp += 1;
    }
    if exp > 0 {
        factors.push((2, exp));
    }

    let mut p = 3u128;
    while p <= n / p {
        let mut exp = 0u32;
        while n.is_multiple_of(p) {
            n /= p;
            exp += 1;
        }
        if exp > 0 {
            factors.push((p, exp));
        }
        p += 2;
    }
    if n > 1 {
        factors.push((n, 1));
    }
    factors
}

fn checked_pow_u128(base: u128, exp: u32) -> Result<u128, AkitaError> {
    let mut acc = 1u128;
    for _ in 0..exp {
        acc = acc.checked_mul(base).ok_or_else(|| {
            AkitaError::InvalidInput("two_squares: prime-power scale overflowed".to_string())
        })?;
    }
    Ok(acc)
}

fn scale_two_square_rep(rep: (u64, u64), scale: u128) -> Result<(u64, u64), AkitaError> {
    let scale = u64::try_from(scale).map_err(|_| {
        AkitaError::InvalidInput("two_squares: representation scale exceeds u64".to_string())
    })?;
    Ok((
        rep.0.checked_mul(scale).ok_or_else(|| {
            AkitaError::InvalidInput("two_squares: representation scale overflowed".to_string())
        })?,
        rep.1.checked_mul(scale).ok_or_else(|| {
            AkitaError::InvalidInput("two_squares: representation scale overflowed".to_string())
        })?,
    ))
}

fn multiply_two_square_reps(lhs: (u64, u64), rhs: (u64, u64)) -> Result<(u64, u64), AkitaError> {
    let (a, b) = (i128::from(lhs.0), i128::from(lhs.1));
    let (c, d) = (i128::from(rhs.0), i128::from(rhs.1));
    let x = a * c - b * d;
    let y = a * d + b * c;
    let x = u64::try_from(x.unsigned_abs()).map_err(|_| {
        AkitaError::InvalidInput("two_squares: product component overflowed".to_string())
    })?;
    let y = u64::try_from(y.unsigned_abs()).map_err(|_| {
        AkitaError::InvalidInput("two_squares: product component overflowed".to_string())
    })?;
    Ok((x, y))
}

/// `(a * b) mod m` via additive doubling; exact for any `a, b, m <= u128::MAX`.
#[inline]
fn mul_mod(a: u128, b: u128, m: u128) -> u128 {
    if m <= 1 {
        return 0;
    }
    let mut a = a % m;
    let mut b = b;
    let mut acc = 0u128;
    while b > 0 {
        if b & 1 == 1 {
            acc = add_mod(acc, a, m);
        }
        a = add_mod(a, a, m);
        b >>= 1;
    }
    acc
}

#[inline]
fn add_mod(a: u128, b: u128, m: u128) -> u128 {
    let a = a % m;
    let b = b % m;
    if a > m - b {
        a + b - m
    } else {
        a + b
    }
}

#[inline]
fn sub_mod(a: u128, b: u128, m: u128) -> u128 {
    let a = a % m;
    let b = b % m;
    if a >= b {
        a - b
    } else {
        m - (b - a)
    }
}

/// `base^exp mod m` by square-and-multiply.
#[inline]
fn pow_mod(mut base: u128, mut exp: u128, m: u128) -> u128 {
    let mut acc = 1u128 % m;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mul_mod(acc, base, m);
        }
        base = mul_mod(base, base, m);
        exp >>= 1;
    }
    acc
}

/// Deterministic Miller-Rabin for `n <= u64::MAX`; Baillie-PSW above that.
const MILLER_RABIN_BASES_U64: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

fn is_prime(n: u128) -> bool {
    if n < 2 {
        return false;
    }
    if n <= u128::from(u64::MAX) {
        return is_prime_u64(n as u64);
    }
    for &base in &MILLER_RABIN_BASES_U64 {
        let base = u128::from(base);
        if n.is_multiple_of(base) {
            return false;
        }
    }
    is_baillie_psw_probable_prime(n)
}

fn is_prime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for &base in &MILLER_RABIN_BASES_U64 {
        if n == base {
            return true;
        }
        if n.is_multiple_of(base) {
            return false;
        }
    }
    let trailing = (n - 1).trailing_zeros();
    let odd = (n - 1) >> trailing;
    'witness: for &base in &MILLER_RABIN_BASES_U64 {
        let mut x = pow_mod(u128::from(base), u128::from(odd), u128::from(n));
        if x == 1 || x == n as u128 - 1 {
            continue;
        }
        for _ in 1..trailing {
            x = mul_mod(x, x, u128::from(n));
            if x == n as u128 - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// Strong probable prime test (base `a`).
fn is_strong_probable_prime(a: u128, n: u128) -> bool {
    if n < 2 {
        return false;
    }
    if a.is_multiple_of(n) {
        return true;
    }
    let mut d = n - 1;
    let s = d.trailing_zeros();
    d >>= s;
    let mut x = pow_mod(a, d, n);
    if x == 1 || x == n - 1 {
        return true;
    }
    for _ in 1..s {
        x = mul_mod(x, x, n);
        if x == n - 1 {
            return true;
        }
    }
    false
}

fn is_baillie_psw_probable_prime(n: u128) -> bool {
    if is_square(n) || !is_strong_probable_prime(2, n) {
        return false;
    }
    is_strong_lucas_selfridge_probable_prime(n)
}

fn is_square(n: u128) -> bool {
    let root = n.isqrt();
    root * root == n
}

fn is_strong_lucas_selfridge_probable_prime(n: u128) -> bool {
    let (d_abs, d_negative) = match selfridge_discriminant(n) {
        Some(d) => d,
        None => return false,
    };
    let q_mod = if d_negative {
        ((d_abs + 1) / 4) % n
    } else {
        signed_mod((d_abs - 1) / 4, true, n)
    };
    let d_mod = signed_mod(d_abs, d_negative, n);
    let Some(mut odd) = n.checked_add(1) else {
        return false;
    };
    let trailing = odd.trailing_zeros();
    odd >>= trailing;

    let (u, mut v, mut q_k) = lucas_sequence_mod(odd, d_mod, q_mod, n);
    if u == 0 || v == 0 {
        return true;
    }
    for _ in 1..trailing {
        v = sub_mod(mul_mod(v, v, n), mul_mod(2, q_k, n), n);
        q_k = mul_mod(q_k, q_k, n);
        if v == 0 {
            return true;
        }
    }
    false
}

fn selfridge_discriminant(n: u128) -> Option<(u128, bool)> {
    let mut d_abs = 5u128;
    let mut d_negative = false;
    loop {
        match jacobi_small_signed(d_abs, d_negative, n) {
            -1 => return Some((d_abs, d_negative)),
            0 => return None,
            _ => {}
        }
        d_abs = d_abs.checked_add(2)?;
        d_negative = !d_negative;
    }
}

fn jacobi_small_signed(abs_a: u128, negative: bool, n: u128) -> i8 {
    debug_assert!(n % 2 == 1);
    let mut sign = 1i8;
    if negative && n % 4 == 3 {
        sign = -sign;
    }
    let mut a = abs_a % n;
    let mut n = n;
    while a != 0 {
        while a.is_multiple_of(2) {
            a >>= 1;
            let n_mod_8 = n % 8;
            if n_mod_8 == 3 || n_mod_8 == 5 {
                sign = -sign;
            }
        }
        core::mem::swap(&mut a, &mut n);
        if a % 4 == 3 && n % 4 == 3 {
            sign = -sign;
        }
        a %= n;
    }
    if n == 1 {
        sign
    } else {
        0
    }
}

fn signed_mod(abs_value: u128, negative: bool, modulus: u128) -> u128 {
    let value = abs_value % modulus;
    if negative && value != 0 {
        modulus - value
    } else {
        value
    }
}

fn lucas_sequence_mod(k: u128, d_mod: u128, q_mod: u128, n: u128) -> (u128, u128, u128) {
    debug_assert!(k > 0);
    let inv_two = n.div_ceil(2);
    let mut u = 0u128;
    let mut v = 2u128 % n;
    let mut q_k = 1u128;
    let top_bit = 127 - k.leading_zeros();
    for bit in (0..=top_bit).rev() {
        let u_doubled = mul_mod(u, v, n);
        let v_doubled = sub_mod(mul_mod(v, v, n), mul_mod(2, q_k, n), n);
        u = u_doubled;
        v = v_doubled;
        q_k = mul_mod(q_k, q_k, n);
        if (k >> bit) & 1 == 1 {
            u = mul_mod(add_mod(u_doubled, v_doubled, n), inv_two, n);
            v = mul_mod(
                add_mod(mul_mod(d_mod, u_doubled, n), v_doubled, n),
                inv_two,
                n,
            );
            q_k = mul_mod(q_k, q_mod, n);
        }
    }
    (u, v, q_k)
}

/// Write a prime `p` that is `2` or `≡ 1 (mod 4)` as `c^2 + d^2`.
///
/// For the odd case this finds a square root `x` of `-1 (mod p)` and runs the
/// Hermite–Serret Euclidean descent: the first remainder of `gcd(p, x)` that
/// drops below `sqrt(p)`, paired with the next remainder, are the two squares.
fn two_squares_prime(p: u128) -> Result<(u64, u64), AkitaError> {
    if p == 2 {
        return Ok((1, 1));
    }
    let x = sqrt_neg_one_mod_p(p)?;
    let limit = p.isqrt();
    let (mut prev, mut cur) = (p, x);
    while cur > limit {
        let next = prev % cur;
        prev = cur;
        cur = next;
    }
    let c = u64::try_from(cur).map_err(|_| {
        AkitaError::InvalidInput(format!(
            "two_squares_prime: Euclidean component {cur} exceeds u64 for prime {p}"
        ))
    })?;
    let d = u64::try_from(prev % cur).map_err(|_| {
        AkitaError::InvalidInput(format!(
            "two_squares_prime: Euclidean component {} exceeds u64 for prime {p}",
            prev % cur
        ))
    })?;
    if u128::from(c) * u128::from(c) + u128::from(d) * u128::from(d) == p {
        Ok((c, d))
    } else {
        Err(AkitaError::InvalidInput(format!(
            "two_squares_prime: Euclidean descent did not certify prime {p}"
        )))
    }
}

/// Square root of `-1 (mod p)` for a prime `p ≡ 1 (mod 4)`.
fn sqrt_neg_one_mod_p(p: u128) -> Result<u128, AkitaError> {
    let half = (p - 1) / 2;
    let quarter = (p - 1) / 4;
    let mut z = 2u128;
    while z < p {
        if pow_mod(z, half, p) == p - 1 {
            return Ok(pow_mod(z, quarter, p));
        }
        z += 1;
    }
    Err(AkitaError::InvalidInput(format!(
        "sqrt_neg_one_mod_p: no quadratic non-residue found below prime {p}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    /// Trial-division oracle for cross-checking [`is_prime`] on small inputs.
    fn trial_division_prime(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        let mut d = 2u64;
        while d * d <= n {
            if n.is_multiple_of(d) {
                return false;
            }
            d += 1;
        }
        true
    }

    fn assert_valid_decomposition(target: u64, squares: [u64; 4]) {
        assert_valid_decomposition_u128(u128::from(target), squares);
    }

    fn assert_valid_decomposition_u128(target: u128, squares: [u64; 4]) {
        let sum: u128 = squares.iter().map(|&v| u128::from(v) * u128::from(v)).sum();
        assert_eq!(sum, target, "sum of squares != target {target}");
        if target < u128::from(u64::MAX) {
            for v in squares {
                assert!(
                    u128::from(v) < (1u128 << 32),
                    "slack witness {v} exceeds 2^32 for target {target}"
                );
            }
        }
    }

    #[test]
    fn miller_rabin_matches_trial_division() {
        for n in 0u64..20_000 {
            assert_eq!(
                is_prime(u128::from(n)),
                trial_division_prime(n),
                "primality mismatch at {n}"
            );
        }
    }

    #[test]
    fn miller_rabin_known_large_values() {
        // Mersenne / large primes and obvious composites near them.
        assert!(is_prime(2_147_483_647)); // 2^31 - 1
        assert!(is_prime(2_305_843_009_213_693_951)); // 2^61 - 1
        assert!(is_prime(18_446_744_073_709_551_557)); // largest prime < 2^64
        assert!(!is_prime(2_147_483_647 * 3));
        assert!(!is_prime(18_446_744_073_709_551_557 - 1));
        // Carmichael numbers must not fool the test.
        assert!(!is_prime(561));
        assert!(!is_prime(41_041));
        // Strong probable-prime path above `u64::MAX` (M127).
        let m127 = (1u128 << 127) - 1;
        assert!(is_prime(m127));
        assert!(!is_prime(m127 - 1));
        let large_composite = u128::from(4_294_967_291u64) * u128::from(4_294_967_311u64);
        assert!(large_composite > u128::from(u64::MAX));
        assert!(!is_prime(large_composite));
    }

    #[test]
    fn baillie_psw_rejects_base_two_strong_pseudoprimes() {
        for &n in &[
            2047u128, 3277, 4033, 4681, 8321, 15_841, 29_341, 42_799, 49_141,
        ] {
            assert!(is_strong_probable_prime(2, n), "{n} should be base-2 SPRP");
            assert!(!is_prime(n), "{n} is composite");
        }
    }

    #[test]
    fn exhaustive_small_targets() {
        for target in 0u64..=50_000 {
            let squares = four_squares(target).expect("decomposition exists");
            assert_valid_decomposition(target, squares);
        }
    }

    #[test]
    fn powers_of_four_times_seven_class() {
        // 4^e * (8k + 7): the class that forces all four squares to be nonzero,
        // and exercises the factor-of-4 stripping plus the rescale.
        for e in 0u32..16 {
            for k in 0u64..50 {
                let base = 8 * k + 7;
                let target = base * 4u64.pow(e);
                let squares = four_squares(target).expect("decomposition exists");
                assert_valid_decomposition(target, squares);
            }
        }
    }

    #[test]
    fn two_squares_prime_known() {
        assert_valid_two_square(2);
        for &p in &[5u64, 13, 17, 29, 97, 101, 65_537, 1_000_000_009] {
            assert!(is_prime(u128::from(p)));
            assert!(p == 2 || p % 4 == 1);
            assert_valid_two_square(u128::from(p));
        }
    }

    fn assert_valid_two_square(p: u128) {
        let (c, d) = two_squares_prime(p).expect("prime is a sum of two squares");
        assert_eq!(
            u128::from(c) * u128::from(c) + u128::from(d) * u128::from(d),
            p
        );
    }

    #[test]
    fn two_squares_general_matches_theorem() {
        for &n in &[0u64, 1, 2, 5, 25, 45, 50, 65, 325, 845, 16_900] {
            let (c, d) = two_squares(u128::from(n))
                .expect("two-square construction should not fail")
                .expect("n should be a sum of two squares");
            assert_eq!(
                u128::from(c) * u128::from(c) + u128::from(d) * u128::from(d),
                u128::from(n),
                "invalid two-square representation for {n}"
            );
        }

        for &n in &[3u64, 6, 7, 11, 12, 21, 28, 44, 3 * 5 * 13] {
            assert!(
                two_squares(u128::from(n))
                    .expect("two-square rejection should not fail")
                    .is_none(),
                "{n} has an odd 3 mod 4 prime exponent and should be rejected"
            );
        }
    }

    #[test]
    fn total_fallback_constructs_four_squares() {
        for &target in &[7u64, 15, 23, 31, 79, 255, 1023, 65_535, 4 * 65_535] {
            let mut m = u128::from(target);
            let mut scale = 1u64;
            while m.is_multiple_of(4) {
                m /= 4;
                scale <<= 1;
            }
            let squares = four_squares_via_two_square_residuals(m, scale, u128::from(target))
                .expect("fallback decomposition exists");
            assert_valid_decomposition(target, squares);
        }
    }

    #[test]
    fn boundary_targets_near_u64_max() {
        let targets = [
            u64::MAX,
            u64::MAX - 1,
            1u64 << 63,
            (1u64 << 63) + 1,
            (1u64 << 32) - 1,
            1u64 << 32,
            (1u64 << 32) + 1,
            u64::MAX / 4 * 4, // multiple of 4 near the top
        ];
        for target in targets {
            let squares = four_squares(target).expect("decomposition exists");
            assert_valid_decomposition(target, squares);
        }
    }

    #[test]
    fn randomized_full_range() {
        let mut rng = StdRng::seed_from_u64(0xA51A_F0F0_1234_5678);
        for _ in 0..20_000 {
            let target: u64 = rng.r#gen();
            let squares = four_squares(target).expect("decomposition exists");
            assert_valid_decomposition(target, squares);
        }
    }

    #[test]
    fn randomized_certificate_sized_range() {
        // A realistic slack-target size (up to ~2^40): the regime the certificate
        // feeds this solver in practice, well below the full u64 range above.
        let mut rng = StdRng::seed_from_u64(0x0BAD_C0DE_DEAD_BEEF);
        for _ in 0..20_000 {
            let target: u64 = rng.gen_range(0..(1u64 << 40));
            let squares = four_squares(target).expect("decomposition exists");
            assert_valid_decomposition(target, squares);
        }
    }

    #[test]
    fn slack_targets_above_u64_max() {
        let targets = [
            (1u128 << 64) + 7,
            (1u128 << 64) + 15,
            (1u128 << 64) + 79,
            (1u128 << 80) + 12_345,
            (1u128 << 96) - 1,
            u128::MAX - 1,
        ];
        for target in targets {
            let squares = four_squares_u128(target).expect("decomposition exists");
            assert_valid_decomposition_u128(target, squares);
        }
    }

    #[test]
    fn randomized_u128_slack_above_u64_max() {
        let mut rng = StdRng::seed_from_u64(0xCAFE_BABE_1234_5678);
        for _ in 0..2_000 {
            let low: u64 = rng.r#gen();
            let high: u64 = rng.gen_range(1..(1u64 << 24));
            let target = (u128::from(high) << 64) | u128::from(low);
            let squares = four_squares_u128(target).expect("decomposition exists");
            assert_valid_decomposition_u128(target, squares);
        }
    }
}
