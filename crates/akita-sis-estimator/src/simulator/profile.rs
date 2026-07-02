//! Reduced-basis shape profiles for infinity-norm probability analysis.

/// Squared Gram-Schmidt norms for one effective lattice dimension.
#[derive(Clone, Debug, PartialEq)]
pub struct ShapeProfile {
    squared_norms: Vec<f64>,
}

impl ShapeProfile {
    /// Wrap an already-computed squared-GSO profile.
    #[must_use]
    pub fn from_squared_norms(squared_norms: Vec<f64>) -> Self {
        Self { squared_norms }
    }

    /// Squared Gram-Schmidt norms in descending profile order.
    #[must_use]
    pub fn squared_norms(&self) -> &[f64] {
        &self.squared_norms
    }
}
