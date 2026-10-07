//! Success-probability amplification helpers.

use crate::math::log1mexp2;

/// Return repetitions needed to amplify `success_probability` to
/// `target_success_probability`.
///
/// Mirrors `estimator.prob.amplify` for computational (non-majority) amplification.
#[must_use]
pub fn amplify(target_success_probability: f64, success_probability: f64) -> f64 {
    if target_success_probability < success_probability {
        return 1.0;
    }
    if success_probability == 0.0 {
        return f64::INFINITY;
    }
    let log_success = success_probability.log2();
    let denom = log1mexp2(log_success);
    if !denom.is_finite() || denom == 0.0 {
        return f64::INFINITY;
    }
    ((1.0 - target_success_probability).ln() / denom).ceil()
}

/// Return `log2` of the repetition count needed to amplify a trial with
/// `log2(success_probability)`.
#[must_use]
pub fn log2_amplify(target_success_probability: f64, log2_success_probability: f64) -> f64 {
    if log2_success_probability >= 0.0 {
        return 0.0;
    }
    if !log2_success_probability.is_finite() {
        return f64::INFINITY;
    }

    if log2_success_probability >= target_success_probability.log2() {
        return 0.0;
    }

    // Below machine epsilon, -ln(1-p) = p to binary64 precision.
    // Stay in log space before either the quotient overflows or p underflows.
    if log2_success_probability < -54.0 {
        let log_repetitions =
            (-(-target_success_probability).ln_1p()).log2() - log2_success_probability;
        // A similarly tiny target can still require a small integer count.
        return if log_repetitions < 53.0 {
            log_repetitions.exp2().ceil().log2()
        } else {
            log_repetitions
        };
    }

    amplify(
        target_success_probability,
        2.0_f64.powf(log2_success_probability),
    )
    .log2()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amplify_matches_representable_tiny_probability_example() {
        let reps = amplify(0.99, 2.0_f64.powi(-100));
        assert!((reps.log2() - 102.203).abs() < 0.01);
    }

    #[test]
    fn log2_amplify_handles_underflowed_probability() {
        let reps_log2 = log2_amplify(0.99, -10_000.0);
        assert!((reps_log2 - 10_002.203).abs() < 0.01);
    }
}
