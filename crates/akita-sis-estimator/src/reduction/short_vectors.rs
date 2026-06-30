//! Short-vector sieve output shared across reduction cost models.

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
