//! Fallible, panic-free witness-view provider.
//!
//! A [`WitnessProvider`] resolves an opening identifier to a borrowed
//! [`PolynomialView`]. It is the single seam where a sumcheck descriptor's
//! `Source::Opening(id)` becomes a concrete witness table on the prover side:
//! the Tier-A kernel reads the resulting view while walking the declared
//! expression. Resolution is fallible so unknown openings or malformed backing
//! tables are rejected with [`AkitaError`] instead of panicking.

use akita_error::AkitaError;
use jolt_field::FieldCore;

use crate::PolynomialView;

/// A source of borrowed multilinear witness views, addressed by an opening
/// identifier.
///
/// The identifier type is an associated type so a concrete provider names a
/// single opening vocabulary (e.g. the protocol layer's opening enum). The view
/// borrows from the provider, so a provider materializes or stores its backing
/// evaluation tables and lends them out for the duration of a resolution.
pub trait WitnessProvider<F: FieldCore> {
    /// Identifier selecting which witness oracle to view.
    type OpeningId;

    /// Borrows the multilinear view for `opening`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError`] when `opening` is unknown to the provider or its
    /// backing evaluation table has a malformed shape.
    fn multilinear_view(
        &self,
        opening: Self::OpeningId,
    ) -> Result<PolynomialView<'_, F>, AkitaError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fp64;

    type F = Fp64<4294967197>;

    struct PairProvider {
        left: Vec<F>,
        right: Vec<F>,
    }

    #[derive(Clone, Copy)]
    enum Opening {
        Left,
        Right,
        Missing,
    }

    impl WitnessProvider<F> for PairProvider {
        type OpeningId = Opening;

        fn multilinear_view(
            &self,
            opening: Self::OpeningId,
        ) -> Result<PolynomialView<'_, F>, AkitaError> {
            match opening {
                Opening::Left => PolynomialView::new(1, &self.left),
                Opening::Right => PolynomialView::new(1, &self.right),
                Opening::Missing => Err(AkitaError::InvalidInput("unknown opening".to_owned())),
            }
        }
    }

    fn evals(values: [u64; 2]) -> Vec<F> {
        values.into_iter().map(F::from_u64).collect()
    }

    #[test]
    fn provider_yields_requested_views() {
        let provider = PairProvider {
            left: evals([1, 2]),
            right: evals([3, 4]),
        };

        let left = provider.multilinear_view(Opening::Left).expect("left view");
        assert_eq!(left.num_vars(), 1);
        assert_eq!(left.evals(), evals([1, 2]).as_slice());

        let right = provider
            .multilinear_view(Opening::Right)
            .expect("right view");
        assert_eq!(right.evals(), evals([3, 4]).as_slice());
    }

    #[test]
    fn provider_rejects_unknown_opening() {
        let provider = PairProvider {
            left: evals([1, 2]),
            right: evals([3, 4]),
        };

        let err = provider
            .multilinear_view(Opening::Missing)
            .expect_err("missing opening");
        assert!(matches!(err, AkitaError::InvalidInput(_)));
    }
}
