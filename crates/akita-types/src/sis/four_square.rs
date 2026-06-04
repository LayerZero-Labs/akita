//! Lagrange four-square slack certificate solver (spec slice S7).
//!
//! The L2 folded-witness certificate turns the inequality `Z_SQUARED <= B_l2`
//! into the equality
//!
//! ```text
//! Z_SQUARED + ell_0^2 + ell_1^2 + ell_2^2 + ell_3^2 = B_l2
//! ```
//!
//! by representing the non-negative slack `target = B_l2 - Z_SQUARED` as a sum
//! of four squares (Lagrange's four-square theorem). [`four_squares`] is that
//! pure, prover-side solver: it consumes only the target integer and returns the
//! four slack witnesses `ell_0..ell_3`.
//!
//! ## Bound guarantee
//!
//! The input is a `u64`, so `target < 2^64`. Each returned `ell_j` is one of
//! four non-negative squares summing to `target`, hence `ell_j^2 <= target` and
//! `ell_j <= floor(sqrt(target)) <= 2^32 - 1 < 2^32`. This is exactly the
//! `ell_j < 2^32` ceiling the spec pins for the certificate payload, with no
//! runtime clamp needed.
//!
//! ## Algorithm (Rabin–Shallit prime hunt)
//!
//! Solving is reduced to two-squares of a prime, which is cheap and avoids
//! general integer factorization:
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
//! Because `m mod 4 != 0`, residues `p ≡ 1 (mod 4)` are reachable, and such
//! primes are dense among the scanned residues, so the hunt terminates in a
//! handful of residue inspections in practice. The decision path is integer-only
//! (no floating point): [`u64::isqrt`], deterministic Miller–Rabin (exact for all
//! `u64`), and `u128` modular arithmetic.
//!
//! The solver has no production caller yet; spec slice S8 (prover certificate
//! assembly) is the first consumer.

use akita_field::AkitaError;

/// Bases that make Miller–Rabin a deterministic primality test for every
/// `u64`: `{2,3,5,7,11,13,17,19,23,29,31,37}` is a proven witness set for all
/// `n < 3.3 * 10^24 > 2^64`.
const MILLER_RABIN_BASES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

/// Decompose `target` into four squares
/// `ell_0^2 + ell_1^2 + ell_2^2 + ell_3^2 = target`.
///
/// Every returned `ell_j` satisfies `ell_j < 2^32` (see the module docs). The
/// returned tuple is unordered; callers must not depend on a particular slot
/// carrying the largest square.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidInput`] only on internal inconsistency (the
/// prime hunt exhausting its search space, or a self-check on the returned
/// squares failing). Neither is reachable for a valid `u64` target; the result
/// is always re-verified before return.
pub fn four_squares(target: u64) -> Result<[u64; 4], AkitaError> {
    if target == 0 {
        return Ok([0, 0, 0, 0]);
    }

    // target = 4^e * m with m not divisible by 4; solve m, then scale by 2^e.
    let mut m = target;
    let mut scale = 1u64;
    while m.is_multiple_of(4) {
        m /= 4;
        scale <<= 1;
    }

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
                _ if p % 4 == 1 && is_prime(p) => Some(two_squares_prime(p)?),
                _ => None,
            };
            if let Some((c, d)) = split {
                let result = [a * scale, b * scale, c * scale, d * scale];
                return verify_decomposition(result, target);
            }
        }
    }

    Err(AkitaError::InvalidInput(format!(
        "four_squares: no four-square decomposition found for target {target}"
    )))
}

/// Re-check `sum result_j^2 == target` over `u128` before returning, so a
/// solver bug can never emit an invalid certificate.
fn verify_decomposition(result: [u64; 4], target: u64) -> Result<[u64; 4], AkitaError> {
    let sum: u128 = result.iter().map(|&v| u128::from(v) * u128::from(v)).sum();
    if sum == u128::from(target) {
        Ok(result)
    } else {
        Err(AkitaError::InvalidInput(format!(
            "four_squares: internal self-check failed (sum of squares {sum} != target {target})"
        )))
    }
}

/// `(a * b) mod m` via a `u128` widening; exact for any `a, b, m < 2^64`.
#[inline]
fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((u128::from(a) * u128::from(b)) % u128::from(m)) as u64
}

/// `base^exp mod m` by square-and-multiply.
#[inline]
fn pow_mod(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut acc = 1u64 % m;
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

/// Deterministic Miller–Rabin primality test, exact for every `u64`.
fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for &base in &MILLER_RABIN_BASES {
        if n == base {
            return true;
        }
        if n.is_multiple_of(base) {
            return false;
        }
    }
    // n is now coprime to every base and larger than all of them (>= 41), so
    // each base is a valid witness with `base < n`.
    let trailing = (n - 1).trailing_zeros();
    let odd = (n - 1) >> trailing;
    'witness: for &base in &MILLER_RABIN_BASES {
        let mut x = pow_mod(base, odd, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..trailing {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// Write a prime `p` that is `2` or `≡ 1 (mod 4)` as `c^2 + d^2`.
///
/// For the odd case this finds a square root `x` of `-1 (mod p)` and runs the
/// Hermite–Serret Euclidean descent: the first remainder of `gcd(p, x)` that
/// drops below `sqrt(p)`, paired with the next remainder, are the two squares.
fn two_squares_prime(p: u64) -> Result<(u64, u64), AkitaError> {
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
    let c = cur;
    let d = prev % cur;
    if u128::from(c) * u128::from(c) + u128::from(d) * u128::from(d) == u128::from(p) {
        Ok((c, d))
    } else {
        Err(AkitaError::InvalidInput(format!(
            "two_squares_prime: Euclidean descent did not certify prime {p}"
        )))
    }
}

/// Square root of `-1 (mod p)` for a prime `p ≡ 1 (mod 4)`.
///
/// `x = z^((p-1)/4) mod p` for any quadratic non-residue `z`, found by Euler's
/// criterion. The least non-residue is tiny in practice, so the scan is short.
fn sqrt_neg_one_mod_p(p: u64) -> Result<u64, AkitaError> {
    let half = (p - 1) / 2;
    let quarter = (p - 1) / 4;
    let mut z = 2u64;
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
        let sum: u128 = squares.iter().map(|&v| u128::from(v) * u128::from(v)).sum();
        assert_eq!(sum, u128::from(target), "sum of squares != target {target}");
        for v in squares {
            assert!(
                u128::from(v) < (1u128 << 32),
                "slack witness {v} exceeds 2^32 for target {target}"
            );
        }
    }

    #[test]
    fn miller_rabin_matches_trial_division() {
        for n in 0u64..20_000 {
            assert_eq!(
                is_prime(n),
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
            assert!(is_prime(p));
            assert!(p == 2 || p % 4 == 1);
            assert_valid_two_square(p);
        }
    }

    fn assert_valid_two_square(p: u64) {
        let (c, d) = two_squares_prime(p).expect("prime is a sum of two squares");
        assert_eq!(
            u128::from(c) * u128::from(c) + u128::from(d) * u128::from(d),
            u128::from(p)
        );
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
            let target: u64 = rng.gen();
            let squares = four_squares(target).expect("decomposition exists");
            assert_valid_decomposition(target, squares);
        }
    }

    #[test]
    fn randomized_certificate_sized_range() {
        // The realistic certificate regime: slack near the calibration's
        // Z_SQUARED ~ 2^32, well inside the field-capacity gate.
        let mut rng = StdRng::seed_from_u64(0x0BAD_C0DE_DEAD_BEEF);
        for _ in 0..20_000 {
            let target: u64 = rng.gen_range(0..(1u64 << 40));
            let squares = four_squares(target).expect("decomposition exists");
            assert_valid_decomposition(target, squares);
        }
    }
}
