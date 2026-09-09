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

/// Upper numerical estimate of `log2(erf(2^log2_arg))`.
///
/// Use the complementary tail above one: rounding `erf(x)` to one before
/// taking its logarithm loses probability mass that can matter after millions
/// of coordinates. The tiny-argument branch uses the analytic upper bound
/// `erf(x) <= 2*x/sqrt(pi)` and never materializes an underflowed argument.
///
/// The fdlibm approximations in libm have near-ulp accuracy. Allow 32 epsilons
/// for their rational/exponential evaluation and eight for the elementary
/// functions. Apply these allowances *before* multiplying by the coordinate
/// count, toward greater attack probability (hence lower attack cost). This
/// guards special-function rounding; it is not an interval certification of
/// the preceding floating-point lattice simulation.
#[must_use]
pub fn log2_erf_from_log2_arg(log2_arg: f64) -> f64 {
    const SPECIAL_FUNCTION_ALLOWANCE: f64 = 32.0 * f64::EPSILON;
    const ELEMENTARY_ALLOWANCE: f64 = 8.0 * f64::EPSILON;
    // Rounded upward from 1 - log2(pi)/2.
    const LOG2_TWO_OVER_SQRT_PI_UPPER: f64 = 0.174_251_935_263_840_63;
    if log2_arg.is_nan() || log2_arg == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if log2_arg < -20.0 {
        return (log2_arg + LOG2_TWO_OVER_SQRT_PI_UPPER).next_up();
    }
    let x = (libm::exp2(log2_arg) * (1.0 + ELEMENTARY_ALLOWANCE)).next_up();
    let log_probability = if x < 1.0 {
        let mass = (libm::erf(x) * (1.0 + SPECIAL_FUNCTION_ALLOWANCE)).next_up();
        libm::log2(mass.min(1.0))
    } else {
        let tail = (libm::erfc(x) * (1.0 - SPECIAL_FUNCTION_ALLOWANCE))
            .next_down()
            .max(0.0);
        libm::log1p(-tail) * std::f64::consts::LOG2_E
    };
    (log_probability * (1.0 - ELEMENTARY_ALLOWANCE))
        .next_up()
        .min(0.0)
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
    fn log_erf_preserves_tails_and_extreme_arguments() {
        assert!(log2_erf_from_log2_arg(3.0) < 0.0);
        assert!(log2_erf_from_log2_arg(-10_000.0).is_finite());
        assert_eq!(log2_erf_from_log2_arg(1_024.0), 0.0);
    }
}
