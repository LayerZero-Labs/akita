use super::*;

/// Reduced norm coefficients, with the recoverable linear term omitted.
#[derive(Clone, Copy)]
pub(super) struct FieldNorm<E: Field, const SKIP_LINEAR: bool>([E; 3]);

impl<E: Field, const SKIP_LINEAR: bool> FieldNorm<E, SKIP_LINEAR> {
    #[inline(always)]
    pub(super) fn zero() -> Self {
        Self([E::zero(); 3])
    }

    #[inline(always)]
    pub(super) fn add(&mut self, w0: E, dw: E, e_in: E) {
        self.0[0] += e_in * (w0.square() + w0);
        if !SKIP_LINEAR {
            self.0[1] += e_in * (dw * (w0 + w0 + E::one()));
        }
        self.0[2] += e_in * dw.square();
    }

    #[inline(always)]
    pub(super) fn scaled_add(&mut self, e_out: E, inner: [E; 3]) {
        self.0[0] += e_out * inner[0];
        if !SKIP_LINEAR {
            self.0[1] += e_out * inner[1];
        }
        self.0[2] += e_out * inner[2];
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    pub(super) fn merge(&mut self, other: Self) {
        self.0[0] += other.0[0];
        if !SKIP_LINEAR {
            self.0[1] += other.0[1];
        }
        self.0[2] += other.0[2];
    }

    #[inline(always)]
    pub(super) fn totals(self) -> [E; 3] {
        self.0
    }

    #[inline(always)]
    pub(super) fn into_terms(self) -> NormRoundTerms<E> {
        NormRoundTerms::from_totals::<SKIP_LINEAR>(self.0)
    }
}

/// Signed-digit norm products kept unreduced until the equality block ends.
pub(super) struct CompactNorm<E: Field + Unreduced, const SKIP_LINEAR: bool>([E::SmallProduct; 4]);

impl<E: Field + Unreduced, const SKIP_LINEAR: bool> CompactNorm<E, SKIP_LINEAR> {
    #[inline(always)]
    pub(super) fn zero() -> Self {
        Self([E::SmallProduct::zero(); 4])
    }

    #[inline(always)]
    pub(super) fn add(&mut self, w0: i64, dw: i64, e_in: E) {
        let q0 = w0 * (w0 + 1);
        if q0 != 0 {
            self.0[0] += e_in.mul_u64_unreduced(q0 as u64);
        }
        if !SKIP_LINEAR {
            accum_small_signed::<E>(&mut self.0, 1, e_in, dw * (2 * w0 + 1));
        }
        let q2 = dw * dw;
        if q2 != 0 {
            self.0[3] += e_in.mul_u64_unreduced(q2 as u64);
        }
    }

    #[inline(always)]
    pub(super) fn reduce(self) -> [E; 3] {
        [
            E::reduce_small_product(self.0[0]),
            if SKIP_LINEAR {
                E::zero()
            } else {
                reduce_signed_accum::<E>(self.0[1], self.0[2])
            },
            E::reduce_small_product(self.0[3]),
        ]
    }
}

/// Field norm products reduced once per equality block.
pub(super) struct ProductNorm<E: Field + Unreduced, const SKIP_LINEAR: bool>([ProductSum<E>; 3]);

impl<E: Field + Unreduced, const SKIP_LINEAR: bool> ProductNorm<E, SKIP_LINEAR> {
    #[inline(always)]
    pub(super) fn zero() -> Self {
        Self([ProductSum::zero(); 3])
    }

    #[inline(always)]
    pub(super) fn add(&mut self, w0: E, dw: E, e_in: E) {
        self.0[0].add(e_in, w0.square() + w0);
        if !SKIP_LINEAR {
            self.0[1].add(e_in, dw * (w0 + w0 + E::one()));
        }
        self.0[2].add(e_in, dw.square());
    }

    #[inline(always)]
    pub(super) fn reduce(self) -> [E; 3] {
        self.0.map(ProductSum::finish)
    }
}
