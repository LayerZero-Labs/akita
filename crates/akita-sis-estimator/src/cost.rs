//! Estimator cost output types.

use crate::numeric::Probability;

/// A base-2 logarithmic cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogCost {
    /// `log2(cost)`.
    pub log2: f64,
}

impl LogCost {
    /// Create a finite log-space cost.
    #[must_use]
    pub const fn new(log2: f64) -> Self {
        Self { log2 }
    }
}

/// Cost value in log space, with explicit infinity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CostValue {
    /// Finite cost.
    Finite(LogCost),
    /// Infinite or unbounded cost.
    Infinity,
}

impl CostValue {
    /// Create a finite log-space cost.
    #[must_use]
    pub const fn finite_log2(log2: f64) -> Self {
        Self::Finite(LogCost::new(log2))
    }

    /// Return the finite log2 value, if present.
    #[must_use]
    pub const fn log2(self) -> Option<f64> {
        match self {
            Self::Finite(cost) => Some(cost.log2),
            Self::Infinity => None,
        }
    }
}

/// Caller-visible tag copied from input parameters into output.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EstimateTag(Option<String>);

impl EstimateTag {
    /// Create an empty tag.
    #[must_use]
    pub const fn empty() -> Self {
        Self(None)
    }

    /// Create a non-empty tag.
    #[must_use]
    pub fn new(tag: impl Into<String>) -> Self {
        Self(Some(tag.into()))
    }

    /// Return the tag string, if present.
    #[must_use]
    pub fn as_deref(&self) -> Option<&str> {
        self.0.as_deref()
    }
}

/// Lattice-estimator style cost output.
#[derive(Clone, Debug, PartialEq)]
pub struct LatticeCost {
    /// Total ring operations in log space.
    pub rop: CostValue,
    /// Reduction cost in log space.
    pub red: Option<CostValue>,
    /// Sieve cost in log space.
    pub sieve: Option<CostValue>,
    /// Root-Hermite factor or equivalent shape parameter.
    pub delta: Option<f64>,
    /// BKZ block size.
    pub beta: Option<u32>,
    /// Final short-vector/sieve dimension.
    pub eta: Option<u32>,
    /// Number of zeroed coordinates.
    pub zeta: Option<u32>,
    /// Effective lattice dimension after zeta handling.
    pub d: u32,
    /// One-shot or amplified success probability.
    pub prob: Option<Probability>,
    /// Repetition count in log space.
    pub repetitions: Option<CostValue>,
    /// Caller-visible tag.
    pub tag: EstimateTag,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_value_keeps_infinity_explicit() {
        assert_eq!(CostValue::finite_log2(138.0).log2(), Some(138.0));
        assert_eq!(CostValue::Infinity.log2(), None);
    }

    #[test]
    fn estimate_tag_debug_is_serialization_free() {
        let tag = EstimateTag::new("akita_infinity_golden");
        assert_eq!(tag.as_deref(), Some("akita_infinity_golden"));
        assert!(format!("{tag:?}").contains("akita_infinity_golden"));
    }
}
