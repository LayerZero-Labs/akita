//! Short-vector sieve output shared across reduction cost models.

use crate::{
    math::log2_positive,
    reduction::delta::delta,
};

/// Output of a reduction model's short-vector sieve path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShortVectors {
    /// Scaling factor ρ relative to the shortest BKZ vector.
    pub rho: f64,
    /// Total short-vector generation cost in log2 space.
    pub cost_red_log2: f64,
    /// Number of output vectors.
    pub count: f64,
    /// Sieving dimension η.
    pub sieve_dim: u32,
}

/// Shared sieve amortization from lattice-estimator `ReductionCost._short_vectors_sieve`.
#[must_use]
pub fn sieve_short_vectors(beta: u32, bkz_log2: f64) -> ShortVectors {
    let sieve_dim = beta;
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
