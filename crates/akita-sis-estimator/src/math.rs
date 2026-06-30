//! Log-space helpers and elementary special functions for estimator formulas.

use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};

/// Return `log2(q)` for a positive integer modulus.
#[must_use]
pub fn log2_biguint(q: &BigUint) -> f64 {
    if q.is_zero() {
        return f64::NEG_INFINITY;
    }
    let bit_len = q.bits();
    if bit_len <= 64 {
        return (q.to_u64().unwrap_or(1) as f64).log2();
    }
    let shift = bit_len - 64;
    let top = (q >> shift).to_u64().unwrap_or(1);
    shift as f64 + (top as f64).log2()
}

/// Return `log2(x)` for a positive `f64`, mapping non-finite or non-positive inputs to `-inf`.
#[must_use]
pub fn log2_positive(x: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        f64::NEG_INFINITY
    } else {
        x.log2()
    }
}

/// Error function approximation matching Sage `erf` closely enough for golden parity.
#[must_use]
pub fn erf(x: f64) -> f64 {
    if x.abs() < 0.5 {
        let x2 = x * x;
        let mut term = x;
        let mut sum = x;
        for n in 1..32 {
            term *= -x2 / n as f64;
            let addend = term / (2 * n + 1) as f64;
            sum += addend;
            if addend.abs() < 1e-18 * sum.abs().max(1.0) {
                break;
            }
        }
        return std::f64::consts::FRAC_2_SQRT_PI * sum;
    }

    // Abramowitz and Stegun 7.1.26
    let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

/// Compute `log(1 - 2^log_x)` from `log_x = log2(x)` with `x <= 1`.
#[must_use]
pub fn log1mexp2(log_x: f64) -> f64 {
    if log_x > 0.0 {
        return f64::NAN;
    }
    if log_x == 0.0 {
        return f64::NEG_INFINITY;
    }
    let x = 2.0_f64.powf(log_x);
    (-x).ln_1p()
}

/// Return `(q - 1) / 2` as `f64` when it fits, otherwise approximate from bit length.
#[must_use]
pub fn half_q_minus_one(q: &BigUint) -> f64 {
    if q.bits() <= 64 {
        let q_u64 = q.to_u64().unwrap_or(0);
        return ((q_u64.saturating_sub(1)) as f64) / 2.0;
    }
    // For large q, `(q-1)/2` is dominated by `q/2`.
    2.0_f64.powi(q.bits() as i32 - 1)
}

/// Return whether `length_bound >= (q - 1) / 2`.
#[must_use]
pub fn sis_trivially_easy(q: &BigUint, length_bound: f64) -> bool {
    if q.bits() <= 64 {
        let q_u64 = q.to_u64().unwrap_or(0);
        let half = (q_u64.saturating_sub(1)) as f64 / 2.0;
        return length_bound >= half;
    }
    // Conservative for large q: integer bound cannot exceed half the modulus unless huge.
    length_bound >= half_q_minus_one(q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::akita_q32;

    #[test]
    fn log2_biguint_matches_known_moduli() {
        let log_q32 = log2_biguint(&akita_q32());
        assert!(log_q32 > 31.999_999);
        assert!(log_q32 < 32.0);
    }

    #[test]
    fn erf_at_zero_is_zero() {
        assert!(erf(0.0).abs() < 1e-12);
    }
}
