//! Idealized BCSS23 quantum-sieving reduction cost.
//!
//! This asymptotic model assumes writable QRAQM with unit-cost or
//! polylogarithmic-cost coherent access. It is an offline diagnostic, not a
//! concrete fault-tolerant quantum resource estimate.

use crate::reduction::short_vectors::{sieve_short_vectors, ShortVectors};

/// BCSS23 idealized quantum-sieving time exponent.
pub const BCSS23_IDEALIZED_EXPONENT: f64 = 0.2563;

/// Idealized BCSS23 BKZ cost `2^(0.2563 * beta)` in log2 space.
#[must_use]
pub fn bcss23_idealized_log2_cost(beta: u32) -> f64 {
    BCSS23_IDEALIZED_EXPONENT * beta as f64
}

/// Cost short vectors by transferring the idealized BCSS23 SVP cost to the
/// shared BKZ/repeated-short-vector sieve path.
#[must_use]
pub fn bcss23_idealized_short_vectors(beta: u32) -> ShortVectors {
    sieve_short_vectors(beta, bcss23_idealized_log2_cost(beta))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idealized_bcss23_cost_scales_with_beta() {
        assert!((bcss23_idealized_log2_cost(500) - 128.15).abs() < 1e-9);
    }
}
