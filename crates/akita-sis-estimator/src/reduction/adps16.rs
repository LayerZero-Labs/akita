//! ADPS16 reduction cost model.

use crate::{
    config::Adps16Mode,
    cost::CostValue,
    reduction::short_vectors::{sieve_short_vectors, ShortVectors},
};

/// ADPS16 exponent for each cost mode.
#[must_use]
pub const fn adps16_exponent(mode: Adps16Mode) -> f64 {
    match mode {
        Adps16Mode::Classical => 0.2920,
        Adps16Mode::Quantum => 0.2650,
        Adps16Mode::Paranoid => 0.2075,
    }
}

/// BKZ cost `2^(c * beta)` in log2 space.
#[must_use]
pub fn adps16_log2_cost(beta: u32, mode: Adps16Mode) -> f64 {
    adps16_exponent(mode) * beta as f64
}

/// Cost short vectors using the ADPS16 sieve model from lattice-estimator.
#[must_use]
pub fn adps16_short_vectors(beta: u32, _d: u32, mode: Adps16Mode) -> ShortVectors {
    sieve_short_vectors(beta, adps16_log2_cost(beta, mode))
}

/// Convert a log2 cost to [`CostValue`], treating non-finite values as infinity.
#[must_use]
pub fn log2_to_cost_value(log2: f64) -> CostValue {
    if !log2.is_finite() {
        CostValue::Infinity
    } else {
        CostValue::finite_log2(log2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adps16_policy_costs_scale_with_beta() {
        assert!((adps16_log2_cost(500, Adps16Mode::Classical) - 146.0).abs() < 1e-9);
        assert!((adps16_log2_cost(500, Adps16Mode::Quantum) - 132.5).abs() < 1e-9);
    }
}
