//! Sumcheck accumulation policies shared by protocol traversals.

use akita_field::unreduced::HasUnreducedOps;
use akita_field::{FieldCore, Zero};

/// Accumulate the constant and quadratic coefficients of a product round.
///
/// Implementations differ only in when they reduce field products. Protocol
/// traversals use this common interface so the field's exactness policy has one
/// implementation.
pub trait ProductRoundAccumulator<E: FieldCore + HasUnreducedOps>: Sized + Send {
    /// Return an accumulator with both coefficients set to zero.
    fn zero() -> Self;

    /// Add `lhs * rhs` to the constant coefficient.
    fn add_constant_product(&mut self, lhs: E, rhs: E);

    /// Add `lhs * rhs` to the quadratic coefficient.
    fn add_quadratic_product(&mut self, lhs: E, rhs: E);

    /// Combine two partial accumulators.
    #[cfg(feature = "parallel")]
    fn merge(self, other: Self) -> Self;

    /// Reduce both coefficients to field elements.
    fn finish(self) -> (E, E);
}

/// Product round accumulator that reduces once after summing wide products.
///
/// Construction rejects fields that have not declared delayed product sums
/// exact.
pub struct DelayedProductRoundAccumulator<E: HasUnreducedOps> {
    constant: E::ProductAccum,
    quadratic: E::ProductAccum,
}

impl<E: FieldCore + HasUnreducedOps> ProductRoundAccumulator<E>
    for DelayedProductRoundAccumulator<E>
{
    #[inline]
    fn zero() -> Self {
        assert!(
            E::DELAYED_PRODUCT_SUM_IS_EXACT,
            "delayed product round accumulation requires an exact field accumulator"
        );
        Self {
            constant: E::ProductAccum::zero(),
            quadratic: E::ProductAccum::zero(),
        }
    }

    #[inline]
    fn add_constant_product(&mut self, lhs: E, rhs: E) {
        self.constant += lhs.mul_to_product_accum(rhs);
    }

    #[inline]
    fn add_quadratic_product(&mut self, lhs: E, rhs: E) {
        self.quadratic += lhs.mul_to_product_accum(rhs);
    }

    #[cfg(feature = "parallel")]
    #[inline]
    fn merge(self, other: Self) -> Self {
        Self {
            constant: self.constant + other.constant,
            quadratic: self.quadratic + other.quadratic,
        }
    }

    #[inline]
    fn finish(self) -> (E, E) {
        (
            E::reduce_product_accum(self.constant),
            E::reduce_product_accum(self.quadratic),
        )
    }
}

/// Product round accumulator that reduces each product before summing it.
///
/// This is the exact fallback when a field does not permit delayed product
/// reduction for the batch.
pub struct DirectProductRoundAccumulator<E> {
    constant: E,
    quadratic: E,
}

impl<E: FieldCore + HasUnreducedOps> ProductRoundAccumulator<E>
    for DirectProductRoundAccumulator<E>
{
    #[inline]
    fn zero() -> Self {
        Self {
            constant: E::zero(),
            quadratic: E::zero(),
        }
    }

    #[inline]
    fn add_constant_product(&mut self, lhs: E, rhs: E) {
        self.constant += lhs * rhs;
    }

    #[inline]
    fn add_quadratic_product(&mut self, lhs: E, rhs: E) {
        self.quadratic += lhs * rhs;
    }

    #[cfg(feature = "parallel")]
    #[inline]
    fn merge(self, other: Self) -> Self {
        Self {
            constant: self.constant + other.constant,
            quadratic: self.quadratic + other.quadratic,
        }
    }

    #[inline]
    fn finish(self) -> (E, E) {
        (self.constant, self.quadratic)
    }
}

#[inline]
/// Reduce separated positive and negative unreduced accumulators into one field
/// element.
pub fn reduce_signed_accum<E: FieldCore + HasUnreducedOps>(
    pos: E::MulU64Accum,
    neg: E::MulU64Accum,
) -> E {
    E::reduce_mul_u64_accum(pos) - E::reduce_mul_u64_accum(neg)
}

#[cfg(test)]
mod tests {
    use super::{DelayedProductRoundAccumulator, ProductRoundAccumulator};
    use akita_field::Prime128Offset275;

    #[test]
    #[should_panic(
        expected = "delayed product round accumulation requires an exact field accumulator"
    )]
    fn delayed_product_round_rejects_inexact_field() {
        let _ = <DelayedProductRoundAccumulator<Prime128Offset275> as ProductRoundAccumulator<
            Prime128Offset275,
        >>::zero();
    }
}
