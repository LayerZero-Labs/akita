//! BDGL16 reduction cost model from lattice-estimator `estimator.reduction.BDGL16`.

use crate::{
    math::log2_positive,
    reduction::short_vectors::{sieve_short_vectors, ShortVectors},
};

/// Number of SVP calls in BKZ-β (Chen13 experiments, loosely).
#[must_use]
pub const fn svp_repeat(beta: u32, d: u32) -> u64 {
    if beta < d {
        8 * d as u64
    } else {
        1
    }
}

/// LLL preprocessing cost with `B=None` (entry bit-size ignored for SIS paths).
#[must_use]
pub fn lll(d: u32) -> f64 {
    (d as f64).powi(3)
}

/// BDGL16 asymptotic sieve cost `LLL(d) + 2^(0.292·β + 16.4 + log₂ repeat)`.
#[must_use]
pub fn bdgl16_cost(beta: u32, d: u32) -> f64 {
    let repeat = svp_repeat(beta, d) as f64;
    let exponent = 0.292 * beta as f64 + 16.4 + repeat.log2();
    lll(d) + 2.0_f64.powf(exponent)
}

/// BDGL16 BKZ cost in log₂ space.
#[must_use]
pub fn bdgl16_log2_cost(beta: u32, d: u32) -> f64 {
    log2_positive(bdgl16_cost(beta, d))
}

/// Cost short vectors using BDGL16 BKZ preprocessing and the shared sieve formula.
#[must_use]
pub fn bdgl16_short_vectors(beta: u32, d: u32) -> ShortVectors {
    super::short_vectors::sieve_short_vectors(beta, bdgl16_log2_cost(beta, d))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bdgl16_asymptotic_matches_lattice_estimator_doctest() {
        let log2 = bdgl16_log2_cost(500, 1024);
        assert!((log2 - 175.4).abs() < 1e-9);
    }

    #[test]
    fn svp_repeat_switches_at_beta_equals_d() {
        assert_eq!(svp_repeat(63, 64), 8 * 64);
        assert_eq!(svp_repeat(64, 64), 1);
    }
}
