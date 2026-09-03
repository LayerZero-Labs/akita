//! Reduced-basis shape profiles for infinity-norm probability analysis.

const Q_VECTOR_RELATIVE_TOLERANCE: f64 = 8.0 * f64::EPSILON;

/// Whether a reconstructed Gram-Schmidt length represents a q-vector.
///
/// Shape simulators round through logarithms and exponentials, so an absolute
/// tolerance cannot recognize q-vectors at q64 scale. The relative tolerance
/// covers a small number of binary64 ulps without accepting materially shorter
/// vectors.
pub(crate) fn is_q_vector_length(length: f64, q: f64) -> bool {
    length.is_finite()
        && q.is_finite()
        && q > 0.0
        && (length - q).abs() <= Q_VECTOR_RELATIVE_TOLERANCE * length.abs().max(q)
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q_vector_detection_scales_with_q64() {
        let q = 2.0_f64.powi(64);
        assert!(is_q_vector_length(q + 28_672.0, q));
        assert!(!is_q_vector_length(q - 2.0_f64.powi(40), q));
    }
}
