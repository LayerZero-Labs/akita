//! Borrowed multilinear-evaluation view.
//!
//! [`PolynomialView`] is the single shared witness-view vocabulary that both
//! lanes of the stack resolve openings to. It owns no data: it borrows an
//! existing evaluation table and records its multilinear shape (`num_vars`).

use akita_field::AkitaError;

/// A borrowed multilinear polynomial, given by its evaluations over the boolean
/// hypercube `{0,1}^num_vars`.
///
/// Evaluations are indexed in little-endian (LSB-first) order: entry `i` holds
/// the value at the assignment whose `k`-th variable is bit `k` of `i`. The
/// view borrows the backing slice and stores no copy, so it is cheap to pass to
/// [`SumcheckEngine`] and the polyops standard views.
///
/// Construction is fallible: [`PolynomialView::new`] rejects a slice whose
/// length is not exactly `2^num_vars`, so malformed shapes from untrusted data
/// surface as [`AkitaError`] rather than a panic later in a hot path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolynomialView<'a, F> {
    num_vars: usize,
    evals: &'a [F],
}

impl<'a, F> PolynomialView<'a, F> {
    /// Wraps `evals` as a multilinear view over `num_vars` variables.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidInput`] when `num_vars` is too large for the
    /// hypercube size to be representable as a `usize`, and
    /// [`AkitaError::InvalidSize`] when `evals.len()` is not exactly
    /// `2^num_vars`.
    pub fn new(num_vars: usize, evals: &'a [F]) -> Result<Self, AkitaError> {
        let expected = hypercube_size(num_vars)?;
        if evals.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: evals.len(),
            });
        }
        Ok(Self { num_vars, evals })
    }

    /// Number of variables of the multilinear polynomial.
    pub const fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Borrowed evaluation table over the boolean hypercube (length `2^num_vars`).
    pub const fn evals(&self) -> &'a [F] {
        self.evals
    }

    /// Length of the evaluation table (`2^num_vars`).
    pub const fn len(&self) -> usize {
        self.evals.len()
    }

    /// Always `false`: a valid view has at least the single `num_vars == 0` entry.
    pub const fn is_empty(&self) -> bool {
        self.evals.is_empty()
    }
}

/// Returns `2^num_vars`, or an error when it overflows `usize`.
fn hypercube_size(num_vars: usize) -> Result<usize, AkitaError> {
    u32::try_from(num_vars)
        .ok()
        .and_then(|shift| 1usize.checked_shl(shift))
        .ok_or_else(|| {
            AkitaError::InvalidInput(format!(
                "num_vars {num_vars} too large: 2^num_vars overflows usize"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_accepts_matching_shape() {
        let evals = [1u64, 2, 3, 4];
        let view = PolynomialView::new(2, &evals).expect("2^2 == 4");

        assert_eq!(view.num_vars(), 2);
        assert_eq!(view.len(), 4);
        assert!(!view.is_empty());
        assert_eq!(view.evals(), &evals);
    }

    #[test]
    fn new_accepts_zero_variable_constant() {
        let evals = [7u64];
        let view = PolynomialView::new(0, &evals).expect("2^0 == 1");

        assert_eq!(view.num_vars(), 0);
        assert_eq!(view.len(), 1);
        assert!(!view.is_empty());
    }

    #[test]
    fn new_rejects_too_short_slice() {
        let evals = [1u64, 2, 3];
        let err = PolynomialView::new(2, &evals).expect_err("3 != 2^2");

        assert_eq!(
            err,
            AkitaError::InvalidSize {
                expected: 4,
                actual: 3,
            }
        );
    }

    #[test]
    fn new_rejects_too_long_slice() {
        let evals = [1u64, 2, 3, 4, 5];
        let err = PolynomialView::new(2, &evals).expect_err("5 != 2^2");

        assert_eq!(
            err,
            AkitaError::InvalidSize {
                expected: 4,
                actual: 5,
            }
        );
    }

    #[test]
    fn new_rejects_overflowing_num_vars() {
        let evals = [1u64];
        let err = PolynomialView::new(usize::BITS as usize, &evals)
            .expect_err("2^usize::BITS overflows usize");

        assert!(matches!(err, AkitaError::InvalidInput(_)));
    }
}
