//! Exact operator-norm acceptance predicate `gamma_D(c) <= T` (spec slice S1).
//!
//! For `c in Z[X]/(X^D + 1)` the negacyclic convolution operator norm is
//!
//! ```text
//! gamma_D(c) = max_{0 <= k < D} |c_hat(k)|,
//! c_hat(k)   = sum_{j} c_j * zeta_k^j,   zeta_k = exp((2k + 1) * pi * i / D),
//! ```
//!
//! and `accept(c) iff gamma_D(c) <= T iff max_k |c_hat(k)|^2 <= T^2`.
//!
//! Soundness contract (the only invariant callers depend on): a returned
//! [`Decision::Accept`] means `gamma_D(c) <= T` over the reals. To honor that
//! without trusting machine floating-point FFTs (forbidden by the spec), the
//! decision is integer-only:
//!
//! - The `2*D` distinct trig values `cos(pi*t/D), sin(pi*t/D)` are tabulated as
//!   signed fixed-point integers at scale `2^q`, each carried with a *sound*
//!   per-entry error bound `eps_root` (`|C[t] - 2^q cos(pi t/D)| <= eps_root`).
//!   The table is built by interval Taylor evaluation over a rigorous rational
//!   enclosure of `pi` (Machin series in `i128`); no floating point enters the
//!   table or the decision.
//! - For each frequency the predicate forms integer accumulators `R_k, I_k`,
//!   then the conservative upper bound
//!   `upper_k = R_k^2 + I_k^2 + 2(|R_k| + |I_k|) r + 2 r^2` with `r = ||c||_1 *
//!   eps_root`, which provably bounds `|2^q c_hat(k)|^2` from above.
//!   [`Decision::Accept`] is returned only when every `upper_k <= T^2 2^{2q}`,
//!   so acceptance is always sound. A frequency whose *lower* bound already
//!   exceeds the threshold forces [`Decision::Reject`]; anything in between is
//!   [`Decision::Indeterminate`] (the spec sanctions treating the boundary band
//!   as a reject, at the cost of a slightly smaller accepted support).
//!
//! Only the `D/2` frequencies `k = 0..D/2` are scanned: `zeta_{D-1-k} =
//! conj(zeta_k)` and `c` is real, so `|c_hat(D-1-k)| = |c_hat(k)|`.
//!
//! The predicate itself has no production caller yet; slice S3 (operator-norm
//! rejection sampling) is the first consumer.
#![allow(dead_code)]

use akita_field::AkitaError;

use crate::SparseChallenge;

/// Internal fixed-point scale for the certified `pi` and trig interval math.
/// Chosen so that products of two scaled values stay within `i128`
/// (`2 * TRIG_SCALE < 127`).
const TRIG_SCALE: u32 = 61;
/// `1.0` at [`TRIG_SCALE`].
const TRIG_ONE: i128 = 1i128 << TRIG_SCALE;
/// Taylor terms for the small-angle (`<= pi/4`) cos/sin enclosures. The first
/// omitted term is below `2^-29` at [`TRIG_SCALE`] for `phi <= pi/4`, and the
/// largest factorial used for the remainder bound (`26!`) fits `i128`.
const NTERMS: usize = 12;

/// Decision of the operator-norm acceptance predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Decision {
    /// Proven `gamma_D(c) <= T`.
    Accept,
    /// Proven `gamma_D(c) > T`.
    Reject,
    /// Inside the certified uncertainty band; neither bound is decisive.
    Indeterminate,
}

/// Certified fixed-point cos/sin tables for one `(D, q)` plus the global sound
/// error bound `eps_root`, with accumulator overflow validated for a stated
/// `max_l1` / `max_t`.
#[derive(Debug, Clone)]
pub(crate) struct OpNormTable {
    d: usize,
    q: u32,
    /// `base_cos[t] ~= 2^q cos(pi t / D)` for `t in 0..2D`.
    base_cos: Vec<i128>,
    /// `base_sin[t] ~= 2^q sin(pi t / D)` for `t in 0..2D`.
    base_sin: Vec<i128>,
    /// Sound bound `|base_*[t] - 2^q trig(pi t / D)| <= eps_root`.
    eps_root: i128,
    /// Largest `||c||_1` the overflow validation covered.
    max_l1: i128,
}

impl OpNormTable {
    /// Build the certified table for ring degree `d` at fixed-point scale `q`,
    /// validating that the predicate's `i128` accumulators cannot overflow for
    /// any challenge with `||c||_1 <= max_l1` and any threshold `t <= max_t`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] if `d` is not a positive multiple of
    /// 4, if `q` is outside `1..=56`, or if the worst-case accumulator/square
    /// bounds for `(max_l1, max_t)` do not fit `i128`.
    pub(crate) fn new(d: usize, q: u32, max_l1: u64, max_t: u64) -> Result<Self, AkitaError> {
        if d < 4 || !d.is_multiple_of(4) {
            return Err(AkitaError::InvalidSetup(format!(
                "OpNormTable: ring degree d = {d} must be a positive multiple of 4"
            )));
        }
        if !(1..=56).contains(&q) {
            return Err(AkitaError::InvalidSetup(format!(
                "OpNormTable: scale q = {q} out of supported range 1..=56"
            )));
        }

        let (base_cos, base_sin, eps_root) = build_tables(d, q);

        let max_l1 = i128::from(max_l1);
        let max_t = i128::from(max_t);
        validate_no_overflow(q, eps_root, max_l1, max_t)?;

        Ok(Self {
            d,
            q,
            base_cos,
            base_sin,
            eps_root,
            max_l1,
        })
    }

    /// Decide `gamma_D(c) <= t` for a sparse challenge.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidInput`] if a position is `>= D`, the
    /// positions/coefficients lengths disagree, or `||c||_1` exceeds the
    /// `max_l1` this table validated (which could overflow the accumulators).
    pub(crate) fn decide(
        &self,
        challenge: &SparseChallenge,
        t: u64,
    ) -> Result<Decision, AkitaError> {
        self.decide_with_freqs(challenge, t, self.d / 2)
    }

    /// `true` iff [`Decision::Accept`]; the production (strict) predicate, which
    /// treats the certified boundary band as a reject.
    ///
    /// # Errors
    ///
    /// Propagates the validation errors of [`Self::decide`].
    pub(crate) fn accept_strict(
        &self,
        challenge: &SparseChallenge,
        t: u64,
    ) -> Result<bool, AkitaError> {
        Ok(self.decide(challenge, t)? == Decision::Accept)
    }

    /// Shared scan over the first `num_freqs` frequencies. Production uses
    /// `num_freqs = D/2` (conjugate symmetry); tests pass `D` to confirm the
    /// reduced scan agrees with the full spectrum.
    fn decide_with_freqs(
        &self,
        challenge: &SparseChallenge,
        t: u64,
        num_freqs: usize,
    ) -> Result<Decision, AkitaError> {
        if challenge.positions.len() != challenge.coeffs.len() {
            return Err(AkitaError::InvalidInput(
                "operator-norm predicate: positions/coeffs length mismatch".to_string(),
            ));
        }
        let mut l1: i128 = 0;
        for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
            if pos as usize >= self.d {
                return Err(AkitaError::InvalidInput(format!(
                    "operator-norm predicate: position {pos} out of range for D = {}",
                    self.d
                )));
            }
            l1 += i128::from(coeff.unsigned_abs());
        }
        if l1 > self.max_l1 {
            return Err(AkitaError::InvalidInput(format!(
                "operator-norm predicate: ||c||_1 = {l1} exceeds validated max_l1 = {}",
                self.max_l1
            )));
        }

        // Angle index math stays in `usize`: `mult * pos < 2 d^2`, so this avoids
        // an `i128` modulo (a compiler-rt libcall) on the inner loop while the
        // accumulators and squarings stay `i128` for precision.
        let two_d = 2 * self.d;
        let r = l1 * self.eps_root;
        let two_r = 2 * r;
        let two_r_sq = 2 * r * r;
        let threshold = (i128::from(t) * i128::from(t)) << (2 * self.q);

        let mut indeterminate = false;
        for k in 0..num_freqs {
            let mult = (2 * k + 1) % two_d;
            let mut acc_re: i128 = 0;
            let mut acc_im: i128 = 0;
            for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
                let idx = (mult * pos as usize) % two_d;
                let coeff = i128::from(coeff);
                acc_re += coeff * self.base_cos[idx];
                acc_im += coeff * self.base_sin[idx];
            }
            let abs_re = acc_re.abs();
            let abs_im = acc_im.abs();
            let center = acc_re * acc_re + acc_im * acc_im;
            let upper = center + two_r * (abs_re + abs_im) + two_r_sq;
            if upper <= threshold {
                continue;
            }
            let low_re = (abs_re - r).max(0);
            let low_im = (abs_im - r).max(0);
            let lower = low_re * low_re + low_im * low_im;
            if lower > threshold {
                return Ok(Decision::Reject);
            }
            indeterminate = true;
        }
        if indeterminate {
            Ok(Decision::Indeterminate)
        } else {
            Ok(Decision::Accept)
        }
    }
}

/// Validate the worst-case `i128` accumulator and square bounds for the
/// predicate, given a sound `eps_root` and the caller's `(max_l1, max_t)`.
fn validate_no_overflow(
    q: u32,
    eps_root: i128,
    max_l1: i128,
    max_t: i128,
) -> Result<(), AkitaError> {
    let overflow = || {
        AkitaError::InvalidSetup(
            "OpNormTable: worst-case accumulator does not fit i128".to_string(),
        )
    };
    // max |R_k| = max_l1 * 2^q  (triangle bound over base_* in [-2^q, 2^q]).
    let max_acc = max_l1
        .checked_mul(1i128.checked_shl(q).ok_or_else(overflow)?)
        .ok_or_else(overflow)?;
    let r = max_l1.checked_mul(eps_root).ok_or_else(overflow)?;
    // upper_k = R^2 + I^2 + 2(|R| + |I|) r + 2 r^2, worst case |R| = |I| = max_acc.
    let center = max_acc.checked_mul(max_acc).ok_or_else(overflow)?;
    let center2 = center.checked_mul(2).ok_or_else(overflow)?;
    let cross = max_acc
        .checked_mul(4)
        .and_then(|v| v.checked_mul(r))
        .ok_or_else(overflow)?;
    let slack = r
        .checked_mul(r)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(overflow)?;
    let upper = center2
        .checked_add(cross)
        .and_then(|v| v.checked_add(slack))
        .ok_or_else(overflow)?;
    // threshold = max_t^2 * 2^{2q}.
    let threshold = max_t
        .checked_mul(max_t)
        .and_then(|v| v.checked_shl(2 * q))
        .ok_or_else(overflow)?;
    let _ = upper.checked_add(threshold).ok_or_else(overflow)?;
    Ok(())
}

/// Build `base_cos`, `base_sin` at scale `2^q` for `t in 0..2d`, returning the
/// sound global error bound `eps_root`.
fn build_tables(d: usize, q: u32) -> (Vec<i128>, Vec<i128>, i128) {
    let two_d = 2 * d;
    let (pi_lo, pi_hi) = pi_enclosure();
    let shift = TRIG_SCALE - q;

    let mut base_cos = vec![0i128; two_d];
    let mut base_sin = vec![0i128; two_d];
    let mut eps_root: i128 = 0;

    for t in 0..two_d {
        let (r, swap, cos_sign, sin_sign) = octant_reduce(t, d);
        // phi = pi * r / d, enclosed at TRIG_SCALE; phi in [0, pi/4].
        let phi_lo = fdiv(pi_lo * r as i128, d as i128);
        let phi_hi = cdiv(pi_hi * r as i128, d as i128);
        let cos_phi = taylor_cos((phi_lo, phi_hi));
        let sin_phi = taylor_sin((phi_lo, phi_hi));
        let (cos_a, sin_a) = if swap {
            (sin_phi, cos_phi)
        } else {
            (cos_phi, sin_phi)
        };
        let cos_a = apply_sign(cos_a, cos_sign);
        let sin_a = apply_sign(sin_a, sin_sign);

        let (c_val, c_err) = downscale_round(cos_a, shift);
        let (s_val, s_err) = downscale_round(sin_a, shift);
        base_cos[t] = c_val;
        base_sin[t] = s_val;
        eps_root = eps_root.max(c_err).max(s_err);
    }
    (base_cos, base_sin, eps_root)
}

/// Reduce angle index `t` (`angle = pi t / d`, period `2d`) to the first octant:
/// returns `(r, swap, cos_sign, sin_sign)` with `r in [0, d/4]` such that
/// `cos(angle) = cos_sign * (swap ? sin : cos)(pi r / d)` and likewise for sin.
fn octant_reduce(t: usize, d: usize) -> (usize, bool, i128, i128) {
    let mut t = t % (2 * d);
    let mut cos_sign = 1i128;
    let mut sin_sign = 1i128;
    if t > d {
        // angle in (pi, 2pi): cos(2pi - a) = cos a, sin(2pi - a) = -sin a.
        t = 2 * d - t;
        sin_sign = -1;
    }
    if t > d / 2 {
        // angle in (pi/2, pi]: cos(pi - a) = -cos a, sin(pi - a) = sin a.
        t = d - t;
        cos_sign = -cos_sign;
    }
    let swap = if t > d / 4 {
        // angle in (pi/4, pi/2]: cos a = sin(pi/2 - a), sin a = cos(pi/2 - a).
        t = d / 2 - t;
        true
    } else {
        false
    };
    (t, swap, cos_sign, sin_sign)
}

/// `cos(phi)` enclosure at [`TRIG_SCALE`] for `phi in [0, pi/4]`.
fn taylor_cos(phi: (i128, i128)) -> (i128, i128) {
    let phi2 = imul(phi, phi);
    let mut sum = (TRIG_ONE, TRIG_ONE);
    let mut pow = (TRIG_ONE, TRIG_ONE); // phi^{2n}
    let mut fact: i128 = 1; // (2n)!
    for n in 1..=NTERMS {
        pow = imul(pow, phi2);
        let two_n = 2 * n as i128;
        fact = fact * (two_n - 1) * two_n;
        let term = (fdiv(pow.0, fact), cdiv(pow.1, fact));
        sum = if n % 2 == 1 {
            isub(sum, term)
        } else {
            iadd(sum, term)
        };
    }
    add_remainder(sum, phi2, pow, fact, NTERMS + 1)
}

/// `sin(phi)` enclosure at [`TRIG_SCALE`] for `phi in [0, pi/4]`.
fn taylor_sin(phi: (i128, i128)) -> (i128, i128) {
    let phi2 = imul(phi, phi);
    let mut sum = phi; // n = 0 term: phi^1 / 1!
    let mut pow = phi; // phi^{2n+1}
    let mut fact: i128 = 1; // (2n+1)!
    for n in 1..=NTERMS {
        pow = imul(pow, phi2);
        let two_n = 2 * n as i128;
        fact = fact * two_n * (two_n + 1);
        let term = (fdiv(pow.0, fact), cdiv(pow.1, fact));
        sum = if n % 2 == 1 {
            isub(sum, term)
        } else {
            iadd(sum, term)
        };
    }
    add_remainder(sum, phi2, pow, fact, 2 * NTERMS + 2)
}

/// Widen an alternating-series partial sum by the first omitted term magnitude,
/// a sound bound on the truncation error for a decreasing alternating series.
fn add_remainder(
    sum: (i128, i128),
    phi2: (i128, i128),
    pow: (i128, i128),
    fact: i128,
    next_factor: usize,
) -> (i128, i128) {
    let next_pow = imul(pow, phi2);
    let next_fact = fact * (next_factor as i128) * (next_factor as i128 + 1);
    let rem = cdiv(next_pow.1, next_fact);
    (sum.0 - rem, sum.1 + rem)
}

/// Negate / sign-apply an interval (`sign in {-1, +1}`).
fn apply_sign(v: (i128, i128), sign: i128) -> (i128, i128) {
    if sign < 0 {
        (-v.1, -v.0)
    } else {
        v
    }
}

/// Downscale an interval from [`TRIG_SCALE`] to scale `2^(TRIG_SCALE - shift)`
/// (i.e. to the table scale `2^q`), pick the midpoint integer entry, and return
/// `(entry, error_bound)` with `|entry - true_value| <= error_bound`.
fn downscale_round(v: (i128, i128), shift: u32) -> (i128, i128) {
    let lo = fshr(v.0, shift);
    let hi = cshr(v.1, shift);
    let entry = fdiv(lo + hi, 2);
    let err = (entry - lo).max(hi - entry);
    (entry, err)
}

/// Interval multiply at [`TRIG_SCALE`] for non-negative operands.
fn imul(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
    debug_assert!(a.0 >= 0 && b.0 >= 0);
    (fshr(a.0 * b.0, TRIG_SCALE), cshr(a.1 * b.1, TRIG_SCALE))
}

fn iadd(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
    (a.0 + b.0, a.1 + b.1)
}

fn isub(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
    (a.0 - b.1, a.1 - b.0)
}

/// Floor division (`b > 0`).
fn fdiv(a: i128, b: i128) -> i128 {
    a.div_euclid(b)
}

/// Ceil division (`b > 0`).
fn cdiv(a: i128, b: i128) -> i128 {
    -((-a).div_euclid(b))
}

/// Arithmetic floor shift `floor(x / 2^sh)`.
fn fshr(x: i128, sh: u32) -> i128 {
    x >> sh
}

/// Ceil shift `ceil(x / 2^sh)`.
fn cshr(x: i128, sh: u32) -> i128 {
    -((-x) >> sh)
}

/// Rigorous rational enclosure of `pi` at [`TRIG_SCALE`] via Machin's formula
/// `pi = 16 arctan(1/5) - 4 arctan(1/239)`, all in `i128` fixed point.
fn pi_enclosure() -> (i128, i128) {
    let a5 = arctan_inv(5);
    let a239 = arctan_inv(239);
    // pi = 16 a5 - 4 a239 (interval arithmetic, outward).
    let lo = 16 * a5.0 - 4 * a239.1;
    let hi = 16 * a5.1 - 4 * a239.0;
    (lo, hi)
}

/// Enclosure of `arctan(1/x)` at [`TRIG_SCALE`] from its alternating series
/// `sum_n (-1)^n / ((2n+1) x^{2n+1})`. Each scaled term is floor-divided, and
/// the result is widened by (terms summed + 1) ULPs to cover both the per-term
/// truncation and the alternating-series remainder.
fn arctan_inv(x: i128) -> (i128, i128) {
    let mut sum: i128 = 0;
    let mut x_pow = x; // x^{2n+1}
    let x2 = x * x;
    let mut count: i128 = 0;
    let mut n: i128 = 0;
    while let Some(denom) = (2 * n + 1).checked_mul(x_pow) {
        let term = TRIG_ONE / denom;
        if term == 0 {
            break;
        }
        if n % 2 == 0 {
            sum += term;
        } else {
            sum -= term;
        }
        count += 1;
        n += 1;
        match x_pow.checked_mul(x2) {
            Some(v) => x_pow = v,
            None => break,
        }
    }
    let slack = count + 1;
    (sum - slack, sum + slack)
}

#[cfg(all(test, not(feature = "zk")))]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const Q: u32 = 48;

    fn table(d: usize) -> OpNormTable {
        // ||c||_1 <= 2 * D covers every shipping shell at D <= 128; T <= 64.
        OpNormTable::new(d, Q, (2 * d) as u64, 64).unwrap()
    }

    /// f64 reference, used ONLY to sanity-check the certified integer path.
    fn gamma_sq_f64(d: usize, ch: &SparseChallenge) -> f64 {
        let mut max = 0.0f64;
        for k in 0..d {
            let base = (2 * k + 1) as f64 * PI / d as f64;
            let mut re = 0.0;
            let mut im = 0.0;
            for (&pos, &coeff) in ch.positions.iter().zip(ch.coeffs.iter()) {
                let theta = base * pos as f64;
                re += coeff as f64 * theta.cos();
                im += coeff as f64 * theta.sin();
            }
            max = max.max(re * re + im * im);
        }
        max
    }

    #[test]
    fn pi_encloses_pi() {
        let (lo, hi) = pi_enclosure();
        let pi_scaled = PI * TRIG_ONE as f64;
        assert!((lo as f64) <= pi_scaled && pi_scaled <= (hi as f64));
        // Enclosure should be tight (well under 2^-40 absolute).
        assert!(hi - lo < (1i128 << (TRIG_SCALE - 40)));
    }

    #[test]
    fn table_entries_match_reference_within_eps() {
        for &d in &[4usize, 32, 64, 128] {
            let t = table(d);
            for idx in 0..(2 * d) {
                let angle = PI * idx as f64 / d as f64;
                let ref_cos = (angle.cos() * (1i128 << Q) as f64).round() as i128;
                let ref_sin = (angle.sin() * (1i128 << Q) as f64).round() as i128;
                assert!(
                    (t.base_cos[idx] - ref_cos).abs() <= t.eps_root + 1,
                    "cos mismatch d={d} idx={idx}: {} vs {ref_cos} (eps={})",
                    t.base_cos[idx],
                    t.eps_root
                );
                assert!(
                    (t.base_sin[idx] - ref_sin).abs() <= t.eps_root + 1,
                    "sin mismatch d={d} idx={idx}: {} vs {ref_sin} (eps={})",
                    t.base_sin[idx],
                    t.eps_root
                );
            }
            // eps_root stays tiny: a couple of units in 2^48.
            assert!(
                t.eps_root <= 4,
                "eps_root too large for d={d}: {}",
                t.eps_root
            );
        }
    }

    #[test]
    fn pythagorean_identity_holds() {
        let scale_sq = (1i128 << Q) * (1i128 << Q);
        for &d in &[32usize, 64, 128] {
            let t = table(d);
            for idx in 0..(2 * d) {
                let s = t.base_cos[idx] * t.base_cos[idx] + t.base_sin[idx] * t.base_sin[idx];
                let diff = (s - scale_sq).abs();
                // Each factor is within eps_root; product error is O(eps * 2^q).
                assert!(diff <= (t.eps_root + 2) * (1i128 << (Q + 2)));
            }
        }
    }

    #[test]
    fn unit_coefficient_has_unit_norm() {
        let d = 64;
        let t = table(d);
        let ch = SparseChallenge {
            positions: vec![0],
            coeffs: vec![1],
        };
        // A single coefficient of magnitude m gives gamma = m exactly, so the
        // predicate accepts strictly above, rejects strictly below, and reports
        // the exact tie (T = m) as the indeterminate boundary band.
        assert_eq!(t.decide(&ch, 0).unwrap(), Decision::Reject);
        assert_eq!(t.decide(&ch, 1).unwrap(), Decision::Indeterminate);
        assert_eq!(t.decide(&ch, 2).unwrap(), Decision::Accept);
        assert_eq!(t.decide(&ch, 5).unwrap(), Decision::Accept);
    }

    #[test]
    fn large_single_coefficient_rejects() {
        let d = 64;
        let t = table(d);
        // gamma = 5 exactly: reject below, indeterminate at the tie, accept above.
        let ch = SparseChallenge {
            positions: vec![3],
            coeffs: vec![5],
        };
        assert_eq!(t.decide(&ch, 4).unwrap(), Decision::Reject);
        assert_eq!(t.decide(&ch, 5).unwrap(), Decision::Indeterminate);
        assert_eq!(t.decide(&ch, 6).unwrap(), Decision::Accept);
    }

    /// Tiny deterministic PRNG (xorshift) so the test is fully reproducible.
    fn rng_next(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    fn random_shell(state: &mut u64, d: usize, mag1: usize, mag2: usize) -> SparseChallenge {
        let total = mag1 + mag2;
        let mut positions = Vec::with_capacity(total);
        while positions.len() < total {
            let p = (rng_next(state) % d as u64) as u32;
            if !positions.contains(&p) {
                positions.push(p);
            }
        }
        let mut coeffs = Vec::with_capacity(total);
        for i in 0..total {
            let sign = if rng_next(state) & 1 == 0 { 1i8 } else { -1 };
            let mag = if i < mag1 { 1 } else { 2 };
            coeffs.push(sign * mag);
        }
        SparseChallenge { positions, coeffs }
    }

    #[test]
    fn reduced_scan_matches_full_spectrum() {
        let d = 64;
        let t = table(d);
        let mut state = 0x1234_5678_9abc_def0u64;
        for _ in 0..2000 {
            let ch = random_shell(&mut state, d, 31, 11);
            let reduced = t.decide_with_freqs(&ch, 16, d / 2).unwrap();
            let full = t.decide_with_freqs(&ch, 16, d).unwrap();
            assert_eq!(reduced, full, "challenge {ch:?}");
        }
    }

    #[test]
    fn accept_implies_sound_and_reject_implies_over() {
        let d = 64;
        let t = table(d);
        let mut state = 0xfeed_face_dead_beefu64;
        let threshold = 16u64;
        let mut accepts = 0;
        let mut rejects = 0;
        for _ in 0..5000 {
            let ch = random_shell(&mut state, d, 31, 11);
            let g2 = gamma_sq_f64(d, &ch);
            match t.decide(&ch, threshold).unwrap() {
                Decision::Accept => {
                    accepts += 1;
                    assert!(
                        g2 <= (threshold * threshold) as f64 + 1e-3,
                        "unsound accept: gamma^2={g2} for {ch:?}"
                    );
                }
                Decision::Reject => {
                    rejects += 1;
                    assert!(
                        g2 >= (threshold * threshold) as f64 - 1e-3,
                        "unsound reject: gamma^2={g2} for {ch:?}"
                    );
                }
                Decision::Indeterminate => {}
            }
        }
        // The (31, 11) shell at T = 16 should produce a healthy mix.
        assert!(accepts > 0, "no accepts sampled");
        assert!(rejects > 0, "no rejects sampled");
    }

    #[test]
    fn rejects_oversized_l1() {
        let d = 32;
        let t = OpNormTable::new(d, Q, 4, 16).unwrap();
        let ch = SparseChallenge {
            positions: vec![0, 1, 2, 3, 4],
            coeffs: vec![1, 1, 1, 1, 1],
        };
        assert!(t.decide(&ch, 8).is_err());
    }

    #[test]
    fn construction_rejects_bad_params() {
        assert!(OpNormTable::new(6, Q, 16, 16).is_err()); // not a multiple of 4
        assert!(OpNormTable::new(64, 60, 16, 16).is_err()); // q out of range
        assert!(OpNormTable::new(64, Q, u64::MAX, u64::MAX).is_err()); // overflow
    }
}

/// Microbenchmark (ignored by default) for the operator-norm predicate and
/// `D=64` exact-shell sampling, including the verifier-side rejection-sampling
/// replay cost. Not a correctness test; prints a timing report.
///
/// ```text
/// cargo test -p akita-challenges --release op_norm::perf -- --ignored --nocapture
/// ```
#[cfg(all(test, not(feature = "zk")))]
mod perf {
    use super::{Decision, OpNormTable};
    use crate::sampler::exact_shell::sample_exact_shell_challenge;
    use crate::sampler::xof::XofCursor;
    use crate::{SparseChallenge, SparseChallengeConfig};
    use akita_field::Prime128OffsetA7F7;
    use akita_transcript::labels::DOMAIN_AKITA_PROTOCOL;
    use akita_transcript::{AkitaTranscript, Transcript};
    use std::hint::black_box;
    use std::time::Instant;

    type F = Prime128OffsetA7F7;

    const D: usize = 64;
    const Q: u32 = 48;
    const T: u64 = 16;
    const C1: usize = 31;
    const C2: usize = 11;

    fn build_table() -> OpNormTable {
        OpNormTable::new(D, Q, (2 * D) as u64, 64).unwrap()
    }

    /// Float reference for the operator norm `gamma(c) = max_k |c(zeta_k)|`,
    /// used ONLY to study the acceptance-probability vs threshold tradeoff (not
    /// the protocol predicate, which stays integer-certified).
    fn gamma_f64(ch: &SparseChallenge) -> f64 {
        use std::f64::consts::PI;
        let mut maxsq = 0.0f64;
        for k in 0..D / 2 {
            let base = (2 * k + 1) as f64 * PI / D as f64;
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (&pos, &coeff) in ch.positions.iter().zip(ch.coeffs.iter()) {
                let theta = base * pos as f64;
                re += coeff as f64 * theta.cos();
                im += coeff as f64 * theta.sin();
            }
            let s = re * re + im * im;
            if s > maxsq {
                maxsq = s;
            }
        }
        maxsq.sqrt()
    }

    #[test]
    #[ignore = "measurement: run with --release --ignored --nocapture"]
    fn perf_gamma_distribution() {
        let mut cur = warm_cursor();
        let n: usize = 4_000_000;
        let mut gammas: Vec<f64> = Vec::with_capacity(n);
        for _ in 0..n {
            let ch = sample_exact_shell_challenge(&mut cur, D, C1, C2);
            gammas.push(gamma_f64(&ch));
        }
        gammas.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pct = |q: f64| gammas[((n as f64 * q) as usize).min(n - 1)];
        let mean: f64 = gammas.iter().sum::<f64>() / n as f64;
        println!("\n=== gamma(c) distribution, D={D} shell=({C1},{C2}), N={n} ===");
        println!("mean   = {mean:.3}");
        println!("p50    = {:.3}", pct(0.50));
        println!("p90    = {:.3}", pct(0.90));
        println!("p99    = {:.3}", pct(0.99));
        println!("p99.9  = {:.3}", pct(0.999));
        println!("p99.99 = {:.3}", pct(0.9999));
        println!("max    = {:.3}", gammas[n - 1]);
        println!("(||c||_1 = {} is the trivial deterministic bound)", C1 + 2 * C2);
        println!("--- acceptance p(T) = Pr[gamma <= T] and avg candidates 1/p ---");
        for t in 14..=30u32 {
            let acc = gammas.partition_point(|&g| g <= t as f64);
            let p = acc as f64 / n as f64;
            let cand = if p > 0.0 { 1.0 / p } else { f64::INFINITY };
            println!("T={t:>2}: p={p:.5}  candidates/accept={cand:>7.3}");
        }
        println!();
    }

    fn warm_cursor() -> XofCursor {
        XofCursor::from_seed(&[0x42u8; 32])
    }

    fn time_ns(iters: u64, mut f: impl FnMut()) -> f64 {
        let start = Instant::now();
        for _ in 0..iters {
            f();
        }
        start.elapsed().as_nanos() as f64 / iters as f64
    }

    /// Reference variant showing the remaining headroom past production: on top
    /// of the `usize` index (now in production), accumulators are also `i64`
    /// (`|R_k|, |I_k| <= ||c||_1 2^q < 2^55` for `q <= 48`), with `i128` used
    /// only for the `O(d/2)` squarings. Not adopted: it recouples the accumulator
    /// budget to `(q, ||c||_1)` for a gain the verifier-side cost does not need.
    fn decide_opt(tbl: &OpNormTable, ch: &SparseChallenge, t: u64) -> Decision {
        let two_d = 2 * tbl.d;
        let mut l1: i128 = 0;
        for &c in &ch.coeffs {
            l1 += i128::from(c.unsigned_abs());
        }
        let r = l1 * tbl.eps_root;
        let two_r = 2 * r;
        let two_r_sq = 2 * r * r;
        let threshold = (i128::from(t) * i128::from(t)) << (2 * tbl.q);
        let mut indeterminate = false;
        for k in 0..tbl.d / 2 {
            let mult = 2 * k + 1;
            let mut acc_re: i64 = 0;
            let mut acc_im: i64 = 0;
            for (&pos, &coeff) in ch.positions.iter().zip(ch.coeffs.iter()) {
                let idx = (mult * pos as usize) % two_d;
                acc_re += coeff as i64 * tbl.base_cos[idx] as i64;
                acc_im += coeff as i64 * tbl.base_sin[idx] as i64;
            }
            let abs_re = i128::from(acc_re.abs());
            let abs_im = i128::from(acc_im.abs());
            let re = i128::from(acc_re);
            let im = i128::from(acc_im);
            let center = re * re + im * im;
            let upper = center + two_r * (abs_re + abs_im) + two_r_sq;
            if upper <= threshold {
                continue;
            }
            let low_re = (abs_re - r).max(0);
            let low_im = (abs_im - r).max(0);
            if low_re * low_re + low_im * low_im > threshold {
                return Decision::Reject;
            }
            indeterminate = true;
        }
        if indeterminate {
            Decision::Indeterminate
        } else {
            Decision::Accept
        }
    }

    #[test]
    #[ignore = "microbenchmark: run with --release --ignored --nocapture"]
    fn perf_op_norm_d64() {
        let tbl = build_table();
        let mut cur = warm_cursor();

        // (E) one-time certified table construction.
        let build_ns = time_ns(2_000, || {
            black_box(OpNormTable::new(D, Q, (2 * D) as u64, 64).unwrap());
        });

        // (A) per-candidate sampling: decode one (31,11) shell from a warm XOF.
        for _ in 0..50_000 {
            black_box(sample_exact_shell_challenge(&mut cur, D, C1, C2));
        }
        let sample_ns = time_ns(1_000_000, || {
            black_box(sample_exact_shell_challenge(&mut cur, D, C1, C2));
        });

        // Pool of sampled challenges for decide-only timing (realistic mix of
        // accept / reject / indeterminate).
        let pool: Vec<SparseChallenge> = (0..4096)
            .map(|_| sample_exact_shell_challenge(&mut cur, D, C1, C2))
            .collect();
        let accepted: SparseChallenge = pool
            .iter()
            .find(|ch| tbl.accept_strict(ch, T).unwrap())
            .cloned()
            .expect("some (31,11) shell accepts at T=16");
        // The reference variant must agree with production on every challenge.
        for ch in &pool {
            assert_eq!(decide_opt(&tbl, ch, T), tbl.decide(ch, T).unwrap());
        }

        // (B) op-norm check, production d/2 scan, averaged over the pool.
        for ch in &pool {
            black_box(tbl.decide(ch, T).unwrap());
        }
        let mut i = 0usize;
        let decide_ns = time_ns(1_000_000, || {
            let ch = &pool[i & (pool.len() - 1)];
            i += 1;
            black_box(tbl.decide(ch, T).unwrap());
        });

        // (B-opt) reference: inner loop also moved to i64 accumulators.
        let mut io = 0usize;
        let decide_opt_ns = time_ns(1_000_000, || {
            let ch = &pool[io & (pool.len() - 1)];
            io += 1;
            black_box(decide_opt(&tbl, ch, T));
        });

        // (B') worst case: an accepted challenge always scans all d/2 frequencies.
        let decide_worst_ns = time_ns(1_000_000, || {
            black_box(tbl.decide(&accepted, T).unwrap());
        });

        // (B'') full-spectrum scan (all d frequencies) for the symmetry saving.
        let mut j = 0usize;
        let decide_full_ns = time_ns(1_000_000, || {
            let ch = &pool[j & (pool.len() - 1)];
            j += 1;
            black_box(tbl.decide_with_freqs(ch, T, D).unwrap());
        });

        // (D) rejection sampling end-to-end (the verifier-side replay): draw and
        // check candidates until one is accepted.
        let n_accepted = 100_000u64;
        let (mut attempts, mut accepts, mut rejects, mut indet) = (0u64, 0u64, 0u64, 0u64);
        let start = Instant::now();
        while accepts < n_accepted {
            attempts += 1;
            let ch = sample_exact_shell_challenge(&mut cur, D, C1, C2);
            match tbl.decide(&ch, T).unwrap() {
                Decision::Accept => accepts += 1,
                Decision::Reject => rejects += 1,
                Decision::Indeterminate => indet += 1,
            }
        }
        let per_accepted_ns = start.elapsed().as_nanos() as f64 / n_accepted as f64;
        let p = n_accepted as f64 / attempts as f64;

        // Full public sampling path (transcript absorb + SHAKE seed + decode),
        // amortized per challenge, for n=1 and n=1024 batches.
        let cfg = SparseChallengeConfig::ExactShell {
            count_mag1: C1,
            count_mag2: C2,
        };
        let batch_ns = |n: usize, iters: u64| -> f64 {
            time_ns(iters, || {
                let mut tr = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
                tr.append_field(b"perf-seed", &F::from_u64(0xC0FFEE));
                let chs =
                    crate::sample_sparse_challenges::<F, _, D>(&mut tr, b"perf", n, &cfg).unwrap();
                black_box(chs);
            }) / n as f64
        };
        let cold1_ns = batch_ns(1, 20_000);
        let amort1024_ns = batch_ns(1024, 2_000);

        println!("\n=== operator-norm microbench (D={D}, q={Q}, T={T}, shell=({C1},{C2})) ===");
        println!(
            "certified table build (one-time) : {build_ns:>10.0} ns  ({:.2} us)",
            build_ns / 1e3
        );
        println!("sample 1 candidate (warm XOF)    : {sample_ns:>10.2} ns");
        println!("  full path, n=1 (cold + SHAKE)  : {cold1_ns:>10.2} ns");
        println!("  full path, per chal. @ n=1024  : {amort1024_ns:>10.2} ns");
        println!("op-norm check, pool avg (d/2)    : {decide_ns:>10.2} ns  (production: usize idx, i128 accum)");
        println!(
            "op-norm check, +i64 accumulators : {decide_opt_ns:>10.2} ns  (reference; not adopted)"
        );
        println!("op-norm check, accepted (d/2)    : {decide_worst_ns:>10.2} ns");
        println!("op-norm check, full d scan       : {decide_full_ns:>10.2} ns");
        println!(
            "sample + check, one candidate    : {:>10.2} ns",
            sample_ns + decide_ns
        );
        println!("--- rejection sampling (verifier replay) ---");
        println!("empirical accept prob p          : {p:.4}  ({accepts} acc / {rejects} rej / {indet} indet, {attempts} attempts)");
        println!("avg attempts / accepted          : {:.3}", 1.0 / p);
        println!(
            "time / accepted challenge        : {per_accepted_ns:>10.2} ns  ({:.3} us)",
            per_accepted_ns / 1e3
        );
        println!();
    }
}
