//! ADPS16 reduction cost model.

use crate::{
    config::Adps16Mode,
    cost::CostValue,
    math::log2_positive,
    reduction::{delta::delta, short_vectors::ShortVectors},
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
    let sieve_dim = beta;
    let bkz_log2 = adps16_log2_cost(beta, mode);
    let n_default = 2.0_f64.powf(0.2075 * beta as f64);
    let c0 = n_default;
    let c1 = n_default;
    let c = c0 / c1;
    if c > 2.0_f64.powi(1000) {
        return ShortVectors {
            rho: f64::INFINITY,
            cost_red_log2: f64::INFINITY,
            count: f64::INFINITY,
            sieve_dim,
        };
    }
    let ceil_c = c.ceil();
    let rho = (4.0_f64 / 3.0).sqrt()
        * delta(sieve_dim).powi(sieve_dim as i32 - 1)
        * delta(beta).powf(1.0 - sieve_dim as f64);
    ShortVectors {
        rho,
        cost_red_log2: log2_positive(ceil_c) + bkz_log2,
        count: ceil_c * c1.floor(),
        sieve_dim,
    }
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
    fn adps16_classical_cost_scales_with_beta() {
        assert!((adps16_log2_cost(500, Adps16Mode::Classical) - 146.0).abs() < 1e-9);
    }
}
