//! Sumcheck accumulation policies shared by protocol traversals.

use akita_field::unreduced::HasUnreducedOps;
use akita_field::{FieldCore, Zero};

/// Accumulate an arbitrary sum of field products under the field's selected
/// reduction policy.
pub trait ProductSumAccumulator<E: FieldCore + HasUnreducedOps>: Sized + Send {
    /// Return an empty product sum.
    fn zero() -> Self;

    /// Add `lhs * rhs` to the sum.
    fn add_product(&mut self, lhs: E, rhs: E);

    /// Combine two partial sums.
    #[cfg(feature = "parallel")]
    fn merge(self, other: Self) -> Self;

    /// Reduce the complete sum to one field element.
    fn finish(self) -> E;
}

/// Product sum that reduces once after all wide products have been added.
pub struct DelayedProductSum<E: HasUnreducedOps> {
    sum: E::ProductAccum,
}

impl<E: FieldCore + HasUnreducedOps> ProductSumAccumulator<E> for DelayedProductSum<E> {
    #[inline]
    fn zero() -> Self {
        assert!(
            E::DELAYED_PRODUCT_SUM_IS_EXACT,
            "delayed product accumulation requires an exact field accumulator"
        );
        Self {
            sum: E::ProductAccum::zero(),
        }
    }

    #[inline]
    fn add_product(&mut self, lhs: E, rhs: E) {
        self.sum += lhs.mul_to_product_accum(rhs);
    }

    #[cfg(feature = "parallel")]
    #[inline]
    fn merge(self, other: Self) -> Self {
        Self {
            sum: self.sum + other.sum,
        }
    }

    #[inline]
    fn finish(self) -> E {
        E::reduce_product_accum(self.sum)
    }
}

/// Product sum that reduces every product before adding it.
pub struct DirectProductSum<E> {
    sum: E,
}

impl<E: FieldCore + HasUnreducedOps> ProductSumAccumulator<E> for DirectProductSum<E> {
    #[inline]
    fn zero() -> Self {
        Self { sum: E::zero() }
    }

    #[inline]
    fn add_product(&mut self, lhs: E, rhs: E) {
        self.sum += lhs * rhs;
    }

    #[cfg(feature = "parallel")]
    #[inline]
    fn merge(self, other: Self) -> Self {
        Self {
            sum: self.sum + other.sum,
        }
    }

    #[inline]
    fn finish(self) -> E {
        self.sum
    }
}

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
    constant: DelayedProductSum<E>,
    quadratic: DelayedProductSum<E>,
}

impl<E: FieldCore + HasUnreducedOps> ProductRoundAccumulator<E>
    for DelayedProductRoundAccumulator<E>
{
    #[inline]
    fn zero() -> Self {
        Self {
            constant: DelayedProductSum::zero(),
            quadratic: DelayedProductSum::zero(),
        }
    }

    #[inline]
    fn add_constant_product(&mut self, lhs: E, rhs: E) {
        self.constant.add_product(lhs, rhs);
    }

    #[inline]
    fn add_quadratic_product(&mut self, lhs: E, rhs: E) {
        self.quadratic.add_product(lhs, rhs);
    }

    #[cfg(feature = "parallel")]
    #[inline]
    fn merge(self, other: Self) -> Self {
        Self {
            constant: self.constant.merge(other.constant),
            quadratic: self.quadratic.merge(other.quadratic),
        }
    }

    #[inline]
    fn finish(self) -> (E, E) {
        (self.constant.finish(), self.quadratic.finish())
    }
}

/// Product round accumulator that reduces each product before summing it.
///
/// This is the exact fallback when a field does not permit delayed product
/// reduction for the batch.
pub struct DirectProductRoundAccumulator<E> {
    constant: DirectProductSum<E>,
    quadratic: DirectProductSum<E>,
}

impl<E: FieldCore + HasUnreducedOps> ProductRoundAccumulator<E>
    for DirectProductRoundAccumulator<E>
{
    #[inline]
    fn zero() -> Self {
        Self {
            constant: DirectProductSum::zero(),
            quadratic: DirectProductSum::zero(),
        }
    }

    #[inline]
    fn add_constant_product(&mut self, lhs: E, rhs: E) {
        self.constant.add_product(lhs, rhs);
    }

    #[inline]
    fn add_quadratic_product(&mut self, lhs: E, rhs: E) {
        self.quadratic.add_product(lhs, rhs);
    }

    #[cfg(feature = "parallel")]
    #[inline]
    fn merge(self, other: Self) -> Self {
        Self {
            constant: self.constant.merge(other.constant),
            quadratic: self.quadratic.merge(other.quadratic),
        }
    }

    #[inline]
    fn finish(self) -> (E, E) {
        (self.constant.finish(), self.quadratic.finish())
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
    #[should_panic(expected = "delayed product accumulation requires an exact field accumulator")]
    fn delayed_product_round_rejects_inexact_field() {
        let _ = <DelayedProductRoundAccumulator<Prime128Offset275> as ProductRoundAccumulator<
            Prime128Offset275,
        >>::zero();
    }
}
