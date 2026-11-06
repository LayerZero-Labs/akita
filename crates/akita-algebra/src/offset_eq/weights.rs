//! Coefficient weights for exact affine interval contractions.
use crate::Field;
use akita_error::AkitaError;

/// Coefficient algebra used by [`super::eval_affine_digit_intervals`].
///
/// The equality and digit factors live in `F`; outer high/low factors may live
/// either in `F` itself or in a small coordinate algebra that is linear over
/// `F`. Keeping these operations abstract lets the trace evaluator preserve
/// its factored extension coordinates without introducing another address
/// kernel.
pub trait AffineWeight<F: Field>: Clone + Send + Sync {
    /// Additive identity carrying the same algebra metadata as `self`.
    fn zero_like(&self) -> Self;

    /// Add `factor * scale` to `self`.
    fn add_scaled(&mut self, factor: &Self, scale: F);

    /// Add a batch of paired products. Backends may defer reduction across
    /// the batch; the default preserves the coordinate algebra's scalar path.
    fn add_scaled_batch(&mut self, factors: &[Self], scales: &[F]) {
        for (factor, &scale) in factors.iter().zip(scales) {
            self.add_scaled(factor, scale);
        }
    }

    /// Add `factor` to `self` without a unit scalar multiplication.
    fn add(&mut self, factor: &Self);

    /// Add the embedded scalar `scale` to `self`.
    fn add_scalar(&mut self, scale: F) -> Result<(), AkitaError>;

    /// Whether multiplying by this factor is the identity. The conservative
    /// default keeps arbitrary coordinate algebras on their ordinary path.
    fn is_one(&self) -> bool {
        false
    }

    /// A checked scalar-bit factorization of a bounded power-of-two vector.
    /// Returns its common outer factor and one scalar pair per index bit.
    fn scalar_bit_factors(_weights: &[Self]) -> Option<(Self, Vec<[F; 2]>)> {
        None
    }

    /// Multiply two outer factors.
    fn multiply(&self, rhs: &Self) -> Self;
}

/// Random-access outer weights consumed by [`super::eval_affine_digit_intervals`].
///
/// Implementations may expose an existing dense slice or compute a factored
/// weight only when the contraction reaches its row. The callback keeps dense
/// values borrowed while allowing computed values to remain stack-local.
#[allow(clippy::len_without_is_empty)]
pub trait AffineWeightSource<F: Field, A: AffineWeight<F>>: Sync {
    /// Number of available outer weights.
    fn len(&self) -> usize;

    /// Borrow or compute weight `index` for the duration of `consume`.
    fn with_weight<R>(&self, index: usize, consume: impl FnOnce(&A) -> R) -> Option<R>;
}

impl<F, A, H> AffineWeightSource<F, A> for H
where
    F: Field,
    A: AffineWeight<F>,
    H: AsRef<[A]> + Sync + ?Sized,
{
    fn len(&self) -> usize {
        self.as_ref().len()
    }

    fn with_weight<R>(&self, index: usize, consume: impl FnOnce(&A) -> R) -> Option<R> {
        self.as_ref().get(index).map(consume)
    }
}

/// Cartesian product of two affine weight slices in outer-major order.
///
/// Weight `i` is `outer[i / inner.len()] * inner[i % inner.len()]`. This is the
/// native representation for block-by-row factors and avoids materializing the
/// full product before a single contraction.
pub struct AffineWeightProduct<'a, A> {
    outer: &'a [A],
    inner: &'a [A],
    len: usize,
}

impl<'a, A> AffineWeightProduct<'a, A> {
    /// Construct a checked outer-major Cartesian product.
    pub fn new(outer: &'a [A], inner: &'a [A]) -> Result<Self, AkitaError> {
        let len = outer
            .len()
            .checked_mul(inner.len())
            .ok_or_else(|| AkitaError::InvalidInput("affine weight product overflow".into()))?;
        Ok(Self { outer, inner, len })
    }
}

impl<F: Field, A: AffineWeight<F>> AffineWeightSource<F, A> for AffineWeightProduct<'_, A> {
    fn len(&self) -> usize {
        self.len
    }

    fn with_weight<R>(&self, index: usize, consume: impl FnOnce(&A) -> R) -> Option<R> {
        if self.inner.is_empty() {
            return None;
        }
        let outer = self.outer.get(index / self.inner.len())?;
        let inner = self.inner.get(index % self.inner.len())?;
        if inner.is_one() {
            return Some(consume(outer));
        }
        let product = outer.multiply(inner);
        Some(consume(&product))
    }
}

impl<F: Field> AffineWeight<F> for F {
    fn scalar_bit_factors(weights: &[Self]) -> Option<(Self, Vec<[F; 2]>)> {
        if !weights.len().is_power_of_two() || weights.len() > super::MAX_PREPARED_STRIDE {
            return None;
        }
        let pivot = weights.iter().position(|w| !w.is_zero())?;
        let root = *weights.get(pivot)?;
        let inverse = root.inverse()?;
        let mut factors = Vec::new();
        for bit in 0..weights.len().trailing_zeros() {
            let other = *weights.get(pivot ^ (1usize << bit))? * inverse;
            factors.push(if pivot & (1usize << bit) == 0 {
                [F::one(), other]
            } else {
                [other, F::one()]
            });
        }
        let mut values = vec![root];
        for &[lo, hi] in &factors {
            let old = values.len();
            values.resize(2 * old, F::zero());
            for i in 0..old {
                let v = values[i];
                values[i + old] = if hi == F::one() { v } else { v * hi };
                values[i] = if lo == F::one() { v } else { v * lo };
            }
        }
        (values == weights).then_some((root, factors))
    }

    fn zero_like(&self) -> Self {
        Self::zero()
    }

    fn add_scaled(&mut self, factor: &Self, scale: F) {
        *self = factor.mul_add(scale, *self);
    }

    fn add_scaled_batch(&mut self, factors: &[Self], scales: &[F]) {
        *self += F::dot_product(factors, scales);
    }

    fn add(&mut self, factor: &Self) {
        *self += *factor;
    }

    fn add_scalar(&mut self, scale: F) -> Result<(), AkitaError> {
        *self += scale;
        Ok(())
    }

    fn is_one(&self) -> bool {
        *self == F::one()
    }

    fn multiply(&self, rhs: &Self) -> Self {
        *self * *rhs
    }
}
