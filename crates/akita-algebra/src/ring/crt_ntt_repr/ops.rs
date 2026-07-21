#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
use std::mem::size_of;

use crate::backend::{NttPrimeOps, ScalarBackend};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::ntt::avx::{self, AvxNttMode};
use crate::ntt::butterfly::forward_ntt;
#[cfg(target_arch = "aarch64")]
use crate::ntt::neon;
use crate::ntt::prime::{MontCoeff, NttPrime, PrimeWidth};
use crate::{AkitaError, CanonicalField, CyclotomicRing, FieldCore};

use super::{CenteredMontLut, CrtNttParamSet, CyclotomicCrtNtt, DigitMontLut};

impl<W: PrimeWidth, const K: usize, const D: usize> CyclotomicCrtNtt<W, K, D> {
    /// The additive identity (all zeros in every CRT limb).
    pub fn zero() -> Self {
        Self {
            limbs: [[MontCoeff::from_raw(W::default()); D]; K],
        }
    }

    /// Multiply a row-major prepared matrix by one signed-i16 ring vector.
    ///
    /// The caller must select this CRT profile from an exact bound covering
    /// `num_cols` and the accepted signed-input class. Shape relationships are
    /// checked before indexing so verifier callers reject malformed prepared
    /// state rather than panicking.
    pub fn mat_vec_i16<F: FieldCore + CanonicalField>(
        matrix: &[Self],
        num_rows: usize,
        num_cols: usize,
        rhs: &[[i16; D]],
        params: &CrtNttParamSet<W, K, D>,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        if rhs.len() != num_cols {
            return Err(AkitaError::InvalidProof);
        }
        let required = num_rows
            .checked_mul(num_cols)
            .ok_or(AkitaError::InvalidProof)?;
        let matrix = matrix.get(..required).ok_or_else(|| {
            AkitaError::InvalidSetup("prepared NTT matrix prefix is undersized".into())
        })?;
        if num_rows == 0 || num_cols == 0 {
            return Ok(vec![CyclotomicRing::zero(); num_rows]);
        }

        let rhs_abs_bound = rhs
            .iter()
            .flatten()
            .map(|&digit| i32::from(digit).unsigned_abs())
            .max()
            .unwrap_or(0) as i32;
        let lut = CenteredMontLut::new(params, rhs_abs_bound);
        let mut accumulators = vec![Self::zero(); num_rows];
        for (column, digits) in rhs.iter().enumerate() {
            if digits.iter().all(|&digit| digit == 0) {
                continue;
            }
            let centered = digits.map(i32::from);
            let transformed = Self::from_centered_i32_with_lut(&centered, params, &lut);
            for (accumulator, row) in accumulators.iter_mut().zip(matrix.chunks_exact(num_cols)) {
                let matrix_entry = row.get(column).ok_or_else(|| {
                    AkitaError::InvalidSetup("prepared NTT matrix row is undersized".into())
                })?;
                accumulator.add_assign_pointwise_mul_with_params(
                    matrix_entry,
                    &transformed,
                    params,
                );
            }
        }
        Ok(accumulators
            .iter()
            .map(|accumulator| accumulator.to_ring_with_params(params))
            .collect())
    }

    #[inline(always)]
    fn add_assign_pointwise_mul_limb(
        acc_limb: &mut [MontCoeff<W>; D],
        lhs_limb: &[MontCoeff<W>; D],
        rhs_limb: &[MontCoeff<W>; D],
        prime: NttPrime<W>,
    ) {
        let mut idx = 0usize;
        while idx + 4 <= D {
            for lane in 0..4 {
                let i = idx + lane;
                let prod = prime.mul(lhs_limb[i], rhs_limb[i]);
                let sum = MontCoeff::from_raw(acc_limb[i].raw().wrapping_add(prod.raw()));
                acc_limb[i] = prime.reduce_range(sum);
            }
            idx += 4;
        }

        while idx < D {
            let prod = prime.mul(lhs_limb[idx], rhs_limb[idx]);
            let sum = MontCoeff::from_raw(acc_limb[idx].raw().wrapping_add(prod.raw()));
            acc_limb[idx] = prime.reduce_range(sum);
            idx += 1;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline(always)]
    fn x86_pointwise_mode() -> Option<AvxNttMode> {
        let mode = avx::avx_ntt_mode()?;
        if size_of::<W>() == size_of::<i16>() {
            return avx::use_avx2_transform_ntt().then_some(AvxNttMode::Avx2);
        }
        (size_of::<W>() == size_of::<i32>()).then_some(mode)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline(always)]
    unsafe fn add_assign_pointwise_mul_limb_x86(
        acc_limb: &mut [MontCoeff<W>; D],
        lhs_limb: &[MontCoeff<W>; D],
        rhs_limb: &[MontCoeff<W>; D],
        prime: NttPrime<W>,
        mode: AvxNttMode,
    ) {
        // SAFETY: caller checked x86 SIMD dispatch. `MontCoeff<W>` is
        // transparent over the sealed `i16`/`i32` widths and the arrays are
        // valid for `D`.
        unsafe {
            if size_of::<W>() == size_of::<i16>() {
                avx::pointwise_mul_acc_i16(
                    acc_limb.as_mut_ptr() as *mut i16,
                    lhs_limb.as_ptr() as *const i16,
                    rhs_limb.as_ptr() as *const i16,
                    D,
                    prime.p.to_i64() as i16,
                    prime.pinv.to_i64() as i16,
                );
            } else if size_of::<W>() == size_of::<i32>() {
                match mode {
                    AvxNttMode::Avx2 => avx::pointwise_mul_acc_i32(
                        acc_limb.as_mut_ptr() as *mut i32,
                        lhs_limb.as_ptr() as *const i32,
                        rhs_limb.as_ptr() as *const i32,
                        D,
                        prime.p.to_i64() as i32,
                        prime.pinv.to_i64() as i32,
                    ),
                    AvxNttMode::Avx512 => avx::pointwise_mul_acc_i32_avx512(
                        acc_limb.as_mut_ptr() as *mut i32,
                        lhs_limb.as_ptr() as *const i32,
                        rhs_limb.as_ptr() as *const i32,
                        D,
                        prime.p.to_i64() as i32,
                        prime.pinv.to_i64() as i32,
                    ),
                }
            }
        }
    }

    /// Accumulate `lhs * rhs(digits)` into `self` while reusing caller-owned
    /// scratch storage for the digit CRT+NTT conversion.
    #[inline]
    pub fn add_assign_pointwise_mul_i8_with_lut_scratch(
        &mut self,
        lhs: &Self,
        digits: &[i8; D],
        params: &CrtNttParamSet<W, K, D>,
        lut: &DigitMontLut<W, K>,
        scratch: &mut [[MontCoeff<W>; D]; K],
    ) {
        #[cfg(target_arch = "aarch64")]
        if neon::use_neon_ntt() {
            for (k, (scratch_limb, tw)) in
                scratch.iter_mut().zip(params.twiddles.iter()).enumerate()
            {
                for (dst, &digit) in scratch_limb.iter_mut().zip(digits.iter()) {
                    *dst = lut.get(k, digit);
                }
                forward_ntt(scratch_limb, params.primes[k], tw);
            }

            for (k, rhs_limb) in scratch.iter().enumerate() {
                let prime = params.primes[k];
                unsafe {
                    if size_of::<W>() == size_of::<i32>() {
                        neon::pointwise_mul_acc_i32(
                            self.limbs[k].as_mut_ptr() as *mut i32,
                            lhs.limbs[k].as_ptr() as *const i32,
                            rhs_limb.as_ptr() as *const i32,
                            D,
                            prime.p.to_i64() as i32,
                            prime.pinv.to_i64() as i32,
                        );
                    } else {
                        neon::pointwise_mul_acc_i16(
                            self.limbs[k].as_mut_ptr() as *mut i16,
                            lhs.limbs[k].as_ptr() as *const i16,
                            rhs_limb.as_ptr() as *const i16,
                            D,
                            prime.p.to_i64() as i16,
                            prime.pinv.to_i64() as i16,
                        );
                    }
                }
            }
            return;
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let x86_mode = Self::x86_pointwise_mode();
        for (k, (scratch_limb, tw)) in scratch.iter_mut().zip(params.twiddles.iter()).enumerate() {
            for (dst, &digit) in scratch_limb.iter_mut().zip(digits.iter()) {
                *dst = lut.get(k, digit);
            }
            forward_ntt(scratch_limb, params.primes[k], tw);

            let prime = params.primes[k];
            let acc_limb = &mut self.limbs[k];
            let lhs_limb = &lhs.limbs[k];
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Some(mode) = x86_mode {
                // SAFETY: guarded by x86 runtime dispatch.
                unsafe {
                    Self::add_assign_pointwise_mul_limb_x86(
                        acc_limb,
                        lhs_limb,
                        scratch_limb,
                        prime,
                        mode,
                    );
                }
                continue;
            }
            Self::add_assign_pointwise_mul_limb(acc_limb, lhs_limb, scratch_limb, prime);
        }
    }

    /// Accumulate `mat_row * rhs(digits)` into each `accs[row]` for an arbitrary
    /// number of rows, sharing one digit CRT+NTT conversion across every row.
    ///
    /// `accs[row]` and `ntt_mat[row][col]` are the accumulator and matrix cell
    /// for output row `row`. This generalizes the former single/pair/triple
    /// `_with_lut_scratch` kernels: the rhs conversion (LUT + forward NTT) is
    /// the only shared cost, computed once per CRT limb and reused across all
    /// rows. The per-row multiply-accumulate is identical to the single-row
    /// kernel, so wider `n_a` amortizes the conversion without changing math.
    #[inline]
    pub fn add_assign_col_pointwise_mul_i8_multi_with_lut_scratch(
        accs: &mut [Self],
        ntt_mat: &[&[Self]],
        col: usize,
        digits: &[i8; D],
        params: &CrtNttParamSet<W, K, D>,
        lut: &DigitMontLut<W, K>,
        scratch: &mut [[MontCoeff<W>; D]; K],
    ) {
        debug_assert_eq!(accs.len(), ntt_mat.len());

        #[cfg(target_arch = "aarch64")]
        if neon::use_neon_ntt() {
            for (k, (scratch_limb, tw)) in
                scratch.iter_mut().zip(params.twiddles.iter()).enumerate()
            {
                for (dst, &digit) in scratch_limb.iter_mut().zip(digits.iter()) {
                    *dst = lut.get(k, digit);
                }
                forward_ntt(scratch_limb, params.primes[k], tw);
            }

            for (k, rhs_limb) in scratch.iter().enumerate() {
                let prime = params.primes[k];
                for (acc, mat_row) in accs.iter_mut().zip(ntt_mat.iter()) {
                    let lhs = &mat_row[col];
                    unsafe {
                        if size_of::<W>() == size_of::<i32>() {
                            neon::pointwise_mul_acc_i32(
                                acc.limbs[k].as_mut_ptr() as *mut i32,
                                lhs.limbs[k].as_ptr() as *const i32,
                                rhs_limb.as_ptr() as *const i32,
                                D,
                                prime.p.to_i64() as i32,
                                prime.pinv.to_i64() as i32,
                            );
                        } else {
                            neon::pointwise_mul_acc_i16(
                                acc.limbs[k].as_mut_ptr() as *mut i16,
                                lhs.limbs[k].as_ptr() as *const i16,
                                rhs_limb.as_ptr() as *const i16,
                                D,
                                prime.p.to_i64() as i16,
                                prime.pinv.to_i64() as i16,
                            );
                        }
                    }
                }
            }
            return;
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let x86_mode = Self::x86_pointwise_mode();
        for (k, (scratch_limb, tw)) in scratch.iter_mut().zip(params.twiddles.iter()).enumerate() {
            for (dst, &digit) in scratch_limb.iter_mut().zip(digits.iter()) {
                *dst = lut.get(k, digit);
            }
            forward_ntt(scratch_limb, params.primes[k], tw);

            let prime = params.primes[k];
            for (acc, mat_row) in accs.iter_mut().zip(ntt_mat.iter()) {
                let lhs = &mat_row[col];
                let acc_limb = &mut acc.limbs[k];
                let lhs_limb = &lhs.limbs[k];
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if let Some(mode) = x86_mode {
                    // SAFETY: guarded by x86 runtime dispatch.
                    unsafe {
                        Self::add_assign_pointwise_mul_limb_x86(
                            acc_limb,
                            lhs_limb,
                            scratch_limb,
                            prime,
                            mode,
                        );
                    }
                    continue;
                }
                Self::add_assign_pointwise_mul_limb(acc_limb, lhs_limb, scratch_limb, prime);
            }
        }
    }

    /// Add another CRT+NTT element and reduce each coefficient with the matching
    /// prime to maintain valid Montgomery ranges using the scalar backend.
    pub fn add_reduced(&self, rhs: &Self, primes: &[NttPrime<W>; K]) -> Self {
        self.add_reduced_with_backend::<ScalarBackend>(rhs, primes)
    }

    /// Add another CRT+NTT element and reduce using a bundled parameter set.
    pub fn add_reduced_with_params(&self, rhs: &Self, params: &CrtNttParamSet<W, K, D>) -> Self {
        self.add_reduced(rhs, &params.primes)
    }

    /// Add another CRT+NTT element and reduce each coefficient with the matching
    /// prime through an explicit backend implementation.
    pub fn add_reduced_with_backend<B: NttPrimeOps<W, D>>(
        &self,
        rhs: &Self,
        primes: &[NttPrime<W>; K],
    ) -> Self {
        let mut out = self.clone();
        for (k, (limb, rhs_limb)) in out.limbs.iter_mut().zip(rhs.limbs.iter()).enumerate() {
            let prime = primes[k];
            for (a, b) in limb.iter_mut().zip(rhs_limb.iter()) {
                let sum = MontCoeff::from_raw(a.raw().wrapping_add(b.raw()));
                *a = B::reduce_range(prime, sum);
            }
        }
        out
    }

    /// Subtract another CRT+NTT element and reduce using the scalar backend.
    pub fn sub_reduced(&self, rhs: &Self, primes: &[NttPrime<W>; K]) -> Self {
        self.sub_reduced_with_backend::<ScalarBackend>(rhs, primes)
    }

    /// Subtract another CRT+NTT element and reduce using a bundled parameter set.
    pub fn sub_reduced_with_params(&self, rhs: &Self, params: &CrtNttParamSet<W, K, D>) -> Self {
        self.sub_reduced(rhs, &params.primes)
    }

    /// Subtract another CRT+NTT element and reduce through an explicit backend.
    pub fn sub_reduced_with_backend<B: NttPrimeOps<W, D>>(
        &self,
        rhs: &Self,
        primes: &[NttPrime<W>; K],
    ) -> Self {
        let mut out = self.clone();
        for (k, (limb, rhs_limb)) in out.limbs.iter_mut().zip(rhs.limbs.iter()).enumerate() {
            let prime = primes[k];
            for (a, b) in limb.iter_mut().zip(rhs_limb.iter()) {
                let diff = MontCoeff::from_raw(a.raw().wrapping_sub(b.raw()));
                *a = B::reduce_range(prime, diff);
            }
        }
        out
    }

    /// Negate each CRT+NTT coefficient and reduce using the scalar backend.
    pub fn neg_reduced(&self, primes: &[NttPrime<W>; K]) -> Self {
        self.neg_reduced_with_backend::<ScalarBackend>(primes)
    }

    /// Negate each CRT+NTT coefficient and reduce using a bundled parameter set.
    pub fn neg_reduced_with_params(&self, params: &CrtNttParamSet<W, K, D>) -> Self {
        self.neg_reduced(&params.primes)
    }

    /// Negate each CRT+NTT coefficient and reduce through an explicit backend.
    pub fn neg_reduced_with_backend<B: NttPrimeOps<W, D>>(
        &self,
        primes: &[NttPrime<W>; K],
    ) -> Self {
        let mut out = self.clone();
        for (k, limb) in out.limbs.iter_mut().enumerate() {
            let prime = primes[k];
            for a in limb.iter_mut() {
                let neg = MontCoeff::from_raw(a.raw().wrapping_neg());
                *a = B::reduce_range(prime, neg);
            }
        }
        out
    }

    /// Pointwise multiplication in CRT+NTT domain using the scalar backend.
    pub fn pointwise_mul(&self, rhs: &Self, primes: &[NttPrime<W>; K]) -> Self {
        self.pointwise_mul_with_backend::<ScalarBackend>(rhs, primes)
    }

    /// Pointwise multiplication in CRT+NTT domain using a bundled parameter set.
    #[inline(always)]
    pub fn pointwise_mul_with_params(&self, rhs: &Self, params: &CrtNttParamSet<W, K, D>) -> Self {
        self.pointwise_mul(rhs, &params.primes)
    }

    /// Accumulate `lhs * rhs` into `self` in CRT+NTT domain.
    ///
    /// On AArch64, this uses the fused NEON pointwise-multiply-accumulate kernel
    /// when available; otherwise it falls back to the scalar loop.
    #[inline(always)]
    pub fn add_assign_pointwise_mul_with_params(
        &mut self,
        lhs: &Self,
        rhs: &Self,
        params: &CrtNttParamSet<W, K, D>,
    ) {
        #[cfg(target_arch = "aarch64")]
        if neon::use_neon_ntt() {
            for k in 0..K {
                let prime = params.primes[k];
                unsafe {
                    if size_of::<W>() == size_of::<i32>() {
                        neon::pointwise_mul_acc_i32(
                            self.limbs[k].as_mut_ptr() as *mut i32,
                            lhs.limbs[k].as_ptr() as *const i32,
                            rhs.limbs[k].as_ptr() as *const i32,
                            D,
                            prime.p.to_i64() as i32,
                            prime.pinv.to_i64() as i32,
                        );
                    } else {
                        neon::pointwise_mul_acc_i16(
                            self.limbs[k].as_mut_ptr() as *mut i16,
                            lhs.limbs[k].as_ptr() as *const i16,
                            rhs.limbs[k].as_ptr() as *const i16,
                            D,
                            prime.p.to_i64() as i16,
                            prime.pinv.to_i64() as i16,
                        );
                    }
                }
            }
            return;
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let x86_mode = Self::x86_pointwise_mode();
        for k in 0..K {
            let prime = params.primes[k];
            let acc_limb = &mut self.limbs[k];
            let lhs_limb = &lhs.limbs[k];
            let rhs_limb = &rhs.limbs[k];
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Some(mode) = x86_mode {
                // SAFETY: guarded by x86 runtime dispatch.
                unsafe {
                    Self::add_assign_pointwise_mul_limb_x86(
                        acc_limb, lhs_limb, rhs_limb, prime, mode,
                    );
                }
                continue;
            }
            Self::add_assign_pointwise_mul_limb(acc_limb, lhs_limb, rhs_limb, prime);
        }
    }

    /// Pointwise multiplication in CRT+NTT domain through an explicit backend.
    pub fn pointwise_mul_with_backend<B: NttPrimeOps<W, D>>(
        &self,
        rhs: &Self,
        primes: &[NttPrime<W>; K],
    ) -> Self {
        let mut out = [[MontCoeff::from_raw(W::default()); D]; K];
        for (k, ((o, a), b)) in out
            .iter_mut()
            .zip(self.limbs.iter())
            .zip(rhs.limbs.iter())
            .enumerate()
        {
            let prime = primes[k];
            B::pointwise_mul(prime, o, a, b);
            // Keep coefficients in a bounded range for subsequent inverse NTT.
            for c in o.iter_mut() {
                *c = B::reduce_range(prime, *c);
            }
        }
        Self { limbs: out }
    }

    /// Apply `sigma_{-1}` directly in NTT domain (`slot[j] -> slot[D-1-j]`).
    ///
    /// This is a pure index permutation per CRT limb and does not negate values.
    pub fn conjugation_automorphism_ntt(&self) -> Self {
        let limbs = std::array::from_fn(|k| {
            std::array::from_fn(|j| self.limbs[k][D.saturating_sub(1) - j])
        });
        Self { limbs }
    }
}
