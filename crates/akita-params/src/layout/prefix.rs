//! Exact public-matrix prefix requirements, independent of NTT storage.

use akita_error::AkitaError;

/// Exact public-matrix prefix required at one ring dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NttPrefixRequirement {
    /// Ring dimension used to interpret the flat public field stream.
    pub ring_dimension: usize,
    /// Number of ring elements in the required prefix.
    pub num_ring_elements: usize,
}

impl NttPrefixRequirement {
    /// Derive an exact prefix from the matrix rows and active row width passed
    /// to a consuming kernel.
    pub fn from_matrix_shape(
        ring_dimension: usize,
        num_rows: usize,
        active_width: usize,
    ) -> Result<Self, AkitaError> {
        if ring_dimension == 0 {
            return Err(AkitaError::InvalidSetup(
                "NTT prefix ring dimension must be nonzero".into(),
            ));
        }
        let num_ring_elements = num_rows
            .checked_mul(active_width)
            .filter(|count| *count > 0)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("NTT prefix matrix shape overflows or is empty".into())
            })?;
        Ok(Self {
            ring_dimension,
            num_ring_elements,
        })
    }

    /// Join overlapping public-matrix prefixes by maximum length.
    pub fn join(self, other: Self) -> Result<Self, AkitaError> {
        if self.ring_dimension != other.ring_dimension {
            return Err(AkitaError::InvalidSetup(
                "cannot join NTT prefixes at different ring dimensions".into(),
            ));
        }
        Ok(Self {
            ring_dimension: self.ring_dimension,
            num_ring_elements: self.num_ring_elements.max(other.num_ring_elements),
        })
    }

    /// Exact number of public field elements covered by this prefix.
    pub fn num_field_elements(self) -> Result<usize, AkitaError> {
        self.num_ring_elements
            .checked_mul(self.ring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("NTT prefix field count overflow".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_requirements_join_by_maximum_in_one_dimension() {
        let short = NttPrefixRequirement::from_matrix_shape(64, 2, 3).expect("short prefix");
        let long = NttPrefixRequirement::from_matrix_shape(64, 4, 5).expect("long prefix");
        assert_eq!(short.join(long).expect("join"), long);
        assert_eq!(long.num_field_elements().expect("field count"), 20 * 64);

        let other_dimension =
            NttPrefixRequirement::from_matrix_shape(128, 1, 1).expect("other dimension");
        assert!(short.join(other_dimension).is_err());
    }

    #[test]
    fn prefix_requirements_reject_empty_and_overflowing_shapes() {
        for (dimension, rows, width) in [(0, 1, 1), (64, 0, 1), (64, 1, 0), (64, usize::MAX, 2)] {
            assert!(NttPrefixRequirement::from_matrix_shape(dimension, rows, width).is_err());
        }
        let prefix = NttPrefixRequirement::from_matrix_shape(64, 1, usize::MAX).unwrap();
        assert!(prefix.num_field_elements().is_err());
    }
}
