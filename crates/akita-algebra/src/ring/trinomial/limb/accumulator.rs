//! Product accumulation and exact 128-term checkpoint scheduling.

use super::*;

impl TrinomialLimbAccumulator {
    pub(super) const REDUCTION_TERMS: u16 = 128;

    /// Create an empty accumulator tagged with this domain's prime.
    pub fn new(domain: &TrinomialLimbDomain) -> Self {
        Self {
            backend: domain.backend,
            arithmetic: domain.arithmetic,
            sums: [0; DEGREE],
            terms: 0,
        }
    }

    /// Add one product; reject mismatched primes before changing the sum.
    pub fn add_product(
        &mut self,
        lhs: &TrinomialLimbSlots,
        rhs: &TrinomialLimbSlots,
    ) -> Result<(), TrinomialError> {
        self.add_product_backend(lhs, rhs, self.backend)
    }

    /// Add a batch of products with the same checkpoints as sequential calls.
    ///
    /// The slices must have equal lengths and every slot must have this
    /// accumulator's prime. Invalid batches leave the accumulator unchanged.
    pub fn add_product_batch(
        &mut self,
        lhs: &[TrinomialLimbSlots],
        rhs: &[TrinomialLimbSlots],
    ) -> Result<(), TrinomialError> {
        self.add_product_batch_backend(lhs, rhs, self.backend)
    }

    pub(super) fn add_product_batch_backend(
        &mut self,
        mut lhs: &[TrinomialLimbSlots],
        mut rhs: &[TrinomialLimbSlots],
        backend: Backend,
    ) -> Result<(), TrinomialError> {
        if lhs.len() != rhs.len() {
            return Err(TrinomialError::LimbInput {
                reason: "product batch lengths do not match",
            });
        }
        if lhs
            .iter()
            .chain(rhs)
            .any(|slot| slot.prime != self.arithmetic.prime)
        {
            return Err(TrinomialError::LimbInput {
                reason: "product prime does not match the accumulator",
            });
        }
        while !lhs.is_empty() {
            let count = lhs
                .len()
                .min(usize::from(Self::REDUCTION_TERMS - self.terms));
            let (left, remaining_left) = lhs.split_at(count);
            let (right, remaining_right) = rhs.split_at(count);
            match backend {
                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                Backend::Neon => neon::accumulate_batch(&mut self.sums, left, right),
                #[cfg(target_arch = "x86_64")]
                Backend::Avx2 | Backend::Avx512VbmiBatch => {
                    // SAFETY: Selection establishes AVX2; count ends at or
                    // before the next 128-term checkpoint, bounding every sum.
                    unsafe { avx2::accumulate_batch(&mut self.sums, left, right) };
                }
                Backend::Scalar => arithmetic::accumulate_batch(&mut self.sums, left, right),
            }
            self.advance(count as u16);
            lhs = remaining_left;
            rhs = remaining_right;
        }
        Ok(())
    }

    pub(super) fn add_product_backend(
        &mut self,
        lhs: &TrinomialLimbSlots,
        rhs: &TrinomialLimbSlots,
        backend: Backend,
    ) -> Result<(), TrinomialError> {
        if lhs.prime != self.arithmetic.prime || rhs.prime != self.arithmetic.prime {
            return Err(TrinomialError::LimbInput {
                reason: "product prime does not match the accumulator",
            });
        }
        match backend {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            Backend::Neon => neon::accumulate(&mut self.sums, &lhs.values, &rhs.values),
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 | Backend::Avx512VbmiBatch => {
                // SAFETY: Selection establishes AVX2; advance bounds the sum.
                unsafe { avx2::accumulate(&mut self.sums, &lhs.values, &rhs.values) };
            }
            Backend::Scalar => arithmetic::accumulate(&mut self.sums, &lhs.values, &rhs.values),
        }
        self.advance(1);
        Ok(())
    }

    pub(super) fn advance(&mut self, terms: u16) {
        self.terms += terms;
        if self.terms == Self::REDUCTION_TERMS {
            for sum in &mut self.sums {
                *sum = i64::from(self.arithmetic.center(*sum));
            }
            self.terms = 0;
        }
    }

    /// Write the centered sum and clear it, rejecting a mismatched prime.
    pub fn finish(&mut self, out: &mut TrinomialLimbSlots) -> Result<(), TrinomialError> {
        if out.prime != self.arithmetic.prime {
            return Err(TrinomialError::LimbInput {
                reason: "output prime does not match the accumulator",
            });
        }
        for (value, &sum) in out.values.iter_mut().zip(&self.sums) {
            *value = self.arithmetic.center(sum);
        }
        self.sums.fill(0);
        self.terms = 0;
        Ok(())
    }
}
