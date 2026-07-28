use super::*;

/// Wide (unreduced) cyclotomic ring element for carry-free accumulation.
///
/// Coefficients are wide accumulators (`W: AdditiveGroup`) that support
/// addition/subtraction without modular reduction. After accumulation,
/// call [`reduce`](Self::reduce) to convert back to `CyclotomicRing<F, D>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct WideCyclotomicRing<W: AdditiveGroup, const D: usize> {
    pub(crate) coeffs: [W; D],
}

impl<W: AdditiveGroup, const D: usize> WideCyclotomicRing<W, D> {
    /// Returns the zero ring element.
    #[inline]
    pub fn zero() -> Self {
        Self {
            coeffs: [W::zero(); D],
        }
    }

    /// Convert a reduced `CyclotomicRing<F, D>` into wide form.
    #[inline]
    pub fn from_ring<F: FieldCore>(ring: &CyclotomicRing<F, D>) -> Self
    where
        W: From<F>,
    {
        Self {
            coeffs: from_fn(|i| W::from(ring.coeffs[i])),
        }
    }

    /// Reduce all coefficients back to canonical field form.
    #[inline]
    pub fn reduce<F: FieldCore>(&self) -> CyclotomicRing<F, D>
    where
        W: ReduceTo<F>,
    {
        CyclotomicRing {
            coeffs: from_fn(|i| self.coeffs[i].reduce()),
        }
    }

    /// Fused negacyclic shift + accumulate: `dst += self * X^k`.
    ///
    /// Requires `k < D`.
    /// Wide version of [`CyclotomicRing::shift_accumulate_into`].
    /// `WideCyclotomicRing` has no support for general negacyclic shifts (`k >= D`).
    /// For `k >= D`, reduce to `CyclotomicRing` and use [`CyclotomicRing::negacyclic_shift`].
    #[inline]
    pub fn shift_accumulate_into(&self, dst: &mut Self, k: usize) {
        debug_assert!(
            k < D,
            "fused method shift_accumulate_into: k={k} must be < D={D}"
        );

        let (lo, hi) = dst.coeffs.split_at_mut(k);
        let (self_lo, self_hi) = self.coeffs.split_at(D - k);
        for (d, s) in hi.iter_mut().zip(self_lo) {
            *d += *s; // i + k < D
        }
        for (d, s) in lo.iter_mut().zip(self_hi) {
            *d -= *s; // i + k >= D
        }
    }

    /// Fused negacyclic shifts + accumulate: `dst += Σ_i self * X^shifts[i]`.
    ///
    /// Requires every shift to be smaller than `D`. The fixed-size shift array
    /// lets callers fuse several contributions to the same destination into
    /// one coefficient pass.
    #[inline]
    pub fn shift_accumulate_array_into<const N: usize>(&self, dst: &mut Self, shifts: &[usize; N]) {
        debug_assert!(
            shifts.iter().all(|&shift| shift < D),
            "fused method shift_accumulate_array_into requires every shift to be below D={D}"
        );

        for (coefficient, dst) in dst.coeffs.iter_mut().enumerate() {
            let mut sum = W::zero();
            for &shift in shifts {
                if coefficient >= shift {
                    sum += self.coeffs[coefficient - shift];
                } else {
                    sum -= self.coeffs[D + coefficient - shift];
                }
            }
            *dst += sum;
        }
    }

    /// Fused negacyclic shift + subtract: `dst -= self * X^k`.
    ///
    /// Requires `k < D`.
    /// Wide version of [`CyclotomicRing::shift_sub_into`].
    /// `WideCyclotomicRing` has no support for general negacyclic shifts (`k >= D`).
    /// For `k >= D`, reduce to `CyclotomicRing` and use [`CyclotomicRing::negacyclic_shift`].
    #[inline]
    pub fn shift_sub_into(&self, dst: &mut Self, k: usize) {
        debug_assert!(k < D, "fused method shift_sub_into: k={k} must be < D={D}");

        let (lo, hi) = dst.coeffs.split_at_mut(k);
        let (self_lo, self_hi) = self.coeffs.split_at(D - k);
        for (d, s) in hi.iter_mut().zip(self_lo) {
            *d -= *s; // i + k < D
        }
        for (d, s) in lo.iter_mut().zip(self_hi) {
            *d += *s; // i + k >= D
        }
    }
}

impl<W: AdditiveGroup, const D: usize> Add for WideCyclotomicRing<W, D> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        for i in 0..D {
            self.coeffs[i] += rhs.coeffs[i];
        }
        self
    }
}

impl<W: AdditiveGroup, const D: usize> AddAssign for WideCyclotomicRing<W, D> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..D {
            self.coeffs[i] += rhs.coeffs[i];
        }
    }
}

impl<W: AdditiveGroup, const D: usize> Sub for WideCyclotomicRing<W, D> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self {
        for i in 0..D {
            self.coeffs[i] -= rhs.coeffs[i];
        }
        self
    }
}

impl<W: AdditiveGroup, const D: usize> SubAssign for WideCyclotomicRing<W, D> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..D {
            self.coeffs[i] -= rhs.coeffs[i];
        }
    }
}

impl<W: AdditiveGroup, const D: usize> Neg for WideCyclotomicRing<W, D> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            coeffs: from_fn(|i| -self.coeffs[i]),
        }
    }
}

impl<W: AdditiveGroup, const D: usize> Default for WideCyclotomicRing<W, D> {
    fn default() -> Self {
        Self::zero()
    }
}
