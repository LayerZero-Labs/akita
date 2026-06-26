//! Flat witness hypercube layout for the JL consistency sumcheck.

use akita_algebra::PaddedHypercube;
use akita_field::{AkitaError, FieldCore};

/// Degree bound for the JL product sumcheck.
pub const JL_CONSISTENCY_DEGREE: usize = 2;

/// Witness layout for the flattened JL consistency table.
///
/// The compact witness order is `w[x * 2^ring_bits + y]`, with `x` as the
/// outer column index and `y` as the ring-slot index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JlWitnessLayout {
    /// Number of live outer columns before power-of-two padding.
    pub live_x_cols: usize,
    /// Number of bits in the padded outer-column hypercube.
    pub col_bits: usize,
    /// Number of bits in the ring-slot hypercube.
    pub ring_bits: usize,
    ring_len: usize,
    padded_len: usize,
}

impl JlWitnessLayout {
    /// Construct and validate the flat JL witness layout for `matrix_cols` live columns.
    ///
    /// # Errors
    ///
    /// Returns an error if the live shape does not match `matrix_cols` or if any
    /// power-of-two layout size overflows.
    pub fn new(
        matrix_cols: usize,
        live_x_cols: usize,
        col_bits: usize,
        ring_bits: usize,
    ) -> Result<Self, AkitaError> {
        if live_x_cols == 0 {
            return Err(AkitaError::InvalidInput(
                "JL witness layout requires a non-zero live column count".to_string(),
            ));
        }
        let ring_len = pow2(ring_bits, "JL witness ring dimension")?;
        let padded_x_cols = pow2(col_bits, "JL witness padded column dimension")?;
        if live_x_cols > padded_x_cols {
            return Err(AkitaError::InvalidInput(format!(
                "JL witness live columns {live_x_cols} exceed padded column capacity {padded_x_cols}"
            )));
        }
        let live_len = live_x_cols.checked_mul(ring_len).ok_or_else(|| {
            AkitaError::InvalidInput("JL witness live length overflow".to_string())
        })?;
        if matrix_cols != live_len {
            return Err(AkitaError::InvalidSize {
                expected: live_len,
                actual: matrix_cols,
            });
        }
        let padded_len = padded_x_cols.checked_mul(ring_len).ok_or_else(|| {
            AkitaError::InvalidInput("JL witness padded length overflow".to_string())
        })?;
        Ok(Self {
            live_x_cols,
            col_bits,
            ring_bits,
            ring_len,
            padded_len,
        })
    }

    /// Number of live flat witness entries, equal to the JL matrix column count.
    pub fn live_len(&self) -> usize {
        self.live_x_cols * self.ring_len
    }

    /// Number of padded flat witness entries in the sumcheck hypercube.
    pub fn padded_len(&self) -> usize {
        self.padded_len
    }

    /// Number of variables in the flat witness hypercube.
    pub fn num_vars(&self) -> usize {
        self.col_bits + self.ring_bits
    }

    /// Flat index for compact witness order `w[x * 2^ring_bits + y]`.
    pub fn flat_index(&self, x: usize, y: usize) -> Result<usize, AkitaError> {
        if x >= self.live_x_cols || y >= self.ring_len {
            return Err(AkitaError::InvalidInput(
                "JL witness flat index out of range".to_string(),
            ));
        }
        Ok(x * self.ring_len + y)
    }
}

/// Pad live witness or row-weight evals to the sumcheck hypercube with trailing zeros.
pub fn padded_live_table<F: FieldCore>(
    layout: JlWitnessLayout,
    live_evals: &[F],
) -> Result<Vec<F>, AkitaError> {
    if live_evals.len() != layout.live_len() {
        return Err(AkitaError::InvalidSize {
            expected: layout.live_len(),
            actual: live_evals.len(),
        });
    }
    let mut table = vec![F::zero(); layout.padded_len()];
    table[..live_evals.len()].copy_from_slice(live_evals);
    Ok(table)
}

/// Validate that `layout` matches the matrix column count and MLE hypercube geometry.
pub fn validate_layout_for_matrix_mle(
    matrix_cols: usize,
    layout: JlWitnessLayout,
) -> Result<(), AkitaError> {
    if layout.live_len() != matrix_cols {
        return Err(AkitaError::InvalidSize {
            expected: matrix_cols,
            actual: layout.live_len(),
        });
    }
    let col_hyper = PaddedHypercube::from_live_len(matrix_cols)?;
    if layout.padded_len() != col_hyper.padded_len {
        return Err(AkitaError::InvalidInput(format!(
            "JL layout padded length {} does not match matrix MLE hypercube {}",
            layout.padded_len(),
            col_hyper.padded_len
        )));
    }
    Ok(())
}

fn pow2(bits: usize, name: &str) -> Result<usize, AkitaError> {
    1usize
        .checked_shl(bits as u32)
        .ok_or_else(|| AkitaError::InvalidInput(format!("{name} overflows usize")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_pins_flat_x_outer_y_inner_order() {
        let live_x_cols = 3;
        let ring_bits = 2;
        let ring_len = 1usize << ring_bits;
        let col_bits = 2;
        let layout =
            JlWitnessLayout::new(live_x_cols * ring_len, live_x_cols, col_bits, ring_bits).unwrap();

        assert_eq!(layout.live_len(), 12);
        assert_eq!(layout.padded_len(), 16);
        assert_eq!(layout.num_vars(), 4);
        assert_eq!(layout.flat_index(0, 0).unwrap(), 0);
        assert_eq!(layout.flat_index(0, 3).unwrap(), 3);
        assert_eq!(layout.flat_index(1, 0).unwrap(), 4);
        assert_eq!(layout.flat_index(2, 3).unwrap(), 11);
    }

    #[test]
    fn rejects_nonminimal_layout_for_matrix_mle() {
        let layout = JlWitnessLayout::new(8, 2, 2, 2).unwrap();
        assert!(validate_layout_for_matrix_mle(8, layout).is_err());
    }

    #[test]
    fn rejects_layout_whose_live_len_differs_from_matrix_cols() {
        let layout = JlWitnessLayout::new(12, 3, 2, 2).unwrap();
        assert!(validate_layout_for_matrix_mle(10, layout).is_err());
    }
}
