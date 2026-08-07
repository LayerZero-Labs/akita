#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
use std::mem::size_of;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::mem::MaybeUninit;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::ntt::avx::{self, AvxNttMode};
use crate::ntt::butterfly::forward_ntt;
#[cfg(target_arch = "aarch64")]
use crate::ntt::neon;
use crate::ntt::prime::{MontCoeff, NttPrime, PrimeWidth, I32_LAZY_DOT_BATCH};
use crate::{AkitaError, CanonicalField, CyclotomicRing, FieldCore};

use super::convert::CenteredI16NttConverter;
use super::{CrtNttParamSet, CyclotomicCrtNtt, DigitMontLut};

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
        let accumulators = Self::mat_vec_i16_ntt(matrix, num_rows, num_cols, rhs, params)?;
        Ok(accumulators
            .iter()
            .map(|accumulator| accumulator.to_ring(params))
            .collect())
    }

    /// Multiply a prepared matrix by signed-i16 rings, retaining NTT-domain
    /// accumulators so the arithmetic core is independent of the output field.
    pub(super) fn mat_vec_i16_ntt(
        matrix: &[Self],
        num_rows: usize,
        num_cols: usize,
        rhs: &[[i16; D]],
        params: &CrtNttParamSet<W, K, D>,
    ) -> Result<Vec<Self>, AkitaError> {
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
            return Ok(vec![Self::zero(); num_rows]);
        }

        let converter = CenteredI16NttConverter::new(params, rhs);
        let mut accumulators = vec![Self::zero(); num_rows];
        if !params.uses_lazy_i32_dot() {
            for (column, digits) in rhs.iter().enumerate() {
                if digits.iter().all(|&digit| digit == 0) {
                    continue;
                }
                let transformed = converter.transform(digits);
                for (accumulator, row) in accumulators.iter_mut().zip(matrix.chunks_exact(num_cols))
                {
                    let matrix_entry = row.get(column).ok_or_else(|| {
                        AkitaError::InvalidSetup("prepared NTT matrix row is undersized".into())
                    })?;
                    accumulator.add_assign_pointwise_mul(matrix_entry, &transformed, params);
                }
            }
            return Ok(accumulators);
        }

        const BATCH: usize = I32_LAZY_DOT_BATCH;
        let mut transformed = Vec::with_capacity(BATCH);
        for batch_start in (0..num_cols).step_by(BATCH) {
            let batch_end = (batch_start + BATCH).min(num_cols);
            transformed.clear();
            for digits in &rhs[batch_start..batch_end] {
                if digits.iter().all(|&digit| digit == 0) {
                    break;
                }
                let transformed_rhs = converter.transform(digits);
                transformed.push(transformed_rhs);
            }

            if transformed.len() == batch_end - batch_start {
                for (accumulator, row) in accumulators.iter_mut().zip(matrix.chunks_exact(num_cols))
                {
                    accumulator.add_assign_pointwise_dot(
                        &row[batch_start..batch_end],
                        &transformed,
                        params,
                    );
                }
                continue;
            }
            // Preserve the zero-ring fast path when a batch is not fully dense.
            for (offset, digits) in rhs[batch_start..batch_end].iter().enumerate() {
                if digits.iter().all(|&digit| digit == 0) {
                    continue;
                }
                let transformed = converter.transform(digits);
                let column = batch_start + offset;
                for (accumulator, row) in accumulators.iter_mut().zip(matrix.chunks_exact(num_cols))
                {
                    let matrix_entry = row.get(column).ok_or_else(|| {
                        AkitaError::InvalidSetup("prepared NTT matrix row is undersized".into())
                    })?;
                    accumulator.add_assign_pointwise_mul(matrix_entry, &transformed, params);
                }
            }
        }
        Ok(accumulators)
    }

    /// Accumulate a short pointwise dot product in CRT+NTT domain.
    ///
    /// The prepared backend chooses whether to fuse the products or apply the
    /// canonical single-product primitive repeatedly.
    ///
    /// # Panics
    ///
    /// Panics if the slices differ in length or exceed the backend-independent
    /// six-product arithmetic ceiling.
    #[inline(always)]
    pub fn add_assign_pointwise_dot(
        &mut self,
        lhs: &[Self],
        rhs: &[Self],
        params: &CrtNttParamSet<W, K, D>,
    ) {
        assert_eq!(lhs.len(), rhs.len(), "pointwise dot length mismatch");
        assert!(
            lhs.len() <= I32_LAZY_DOT_BATCH,
            "pointwise dot exceeds lazy reduction bound"
        );

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if params.uses_lazy_i32_dot() {
            for k in 0..K {
                let lhs_pointers: [*const i32; I32_LAZY_DOT_BATCH] = std::array::from_fn(|index| {
                    lhs.get(index).map_or(std::ptr::null(), |entry| {
                        entry.limbs[k].as_ptr().cast::<i32>()
                    })
                });
                let rhs_pointers: [*const i32; I32_LAZY_DOT_BATCH] = std::array::from_fn(|index| {
                    rhs.get(index).map_or(std::ptr::null(), |entry| {
                        entry.limbs[k].as_ptr().cast::<i32>()
                    })
                });
                let prime = params.primes[k];
                // SAFETY: the stored plan proves AVX2 support. Pointer arrays
                // contain `lhs.len()` valid D-element limbs, and the dispatch
                // predicate also proves `W == i32`.
                unsafe {
                    avx::pointwise_dot_acc_i32(
                        self.limbs[k].as_mut_ptr().cast::<i32>(),
                        lhs_pointers.as_ptr(),
                        rhs_pointers.as_ptr(),
                        lhs.len(),
                        D,
                        prime.p.to_i64() as i32,
                        prime.pinv.to_i64() as i32,
                    )
                }
            }
            return;
        }

        for (lhs, rhs) in lhs.iter().zip(rhs) {
            self.add_assign_pointwise_mul(lhs, rhs, params);
        }
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
        if params.kernel_plan.uses_neon() {
            for (k, (scratch_limb, tw)) in
                scratch.iter_mut().zip(params.twiddles.iter()).enumerate()
            {
                for (dst, &digit) in scratch_limb.iter_mut().zip(digits.iter()) {
                    *dst = lut.get(k, digit);
                }
                forward_ntt(scratch_limb, params.primes[k], tw, params.kernel_plan);
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
        let x86_mode = params.kernel_plan.x86_pointwise_mode();
        for (k, (scratch_limb, tw)) in scratch.iter_mut().zip(params.twiddles.iter()).enumerate() {
            for (dst, &digit) in scratch_limb.iter_mut().zip(digits.iter()) {
                *dst = lut.get(k, digit);
            }
            forward_ntt(scratch_limb, params.primes[k], tw, params.kernel_plan);

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
        if params.kernel_plan.uses_neon() {
            for (k, (scratch_limb, tw)) in
                scratch.iter_mut().zip(params.twiddles.iter()).enumerate()
            {
                for (dst, &digit) in scratch_limb.iter_mut().zip(digits.iter()) {
                    *dst = lut.get(k, digit);
                }
                forward_ntt(scratch_limb, params.primes[k], tw, params.kernel_plan);
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
        let x86_mode = params.kernel_plan.x86_pointwise_mode();
        for (k, (scratch_limb, tw)) in scratch.iter_mut().zip(params.twiddles.iter()).enumerate() {
            for (dst, &digit) in scratch_limb.iter_mut().zip(digits.iter()) {
                *dst = lut.get(k, digit);
            }
            forward_ntt(scratch_limb, params.primes[k], tw, params.kernel_plan);

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
    /// prime to maintain valid Montgomery ranges.
    pub fn add_reduced(&self, rhs: &Self, params: &CrtNttParamSet<W, K, D>) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if size_of::<W>() == size_of::<i32>() && params.kernel_plan.uses_x86_transform() {
            let mut output = MaybeUninit::<Self>::uninit();
            let output_ptr = output.as_mut_ptr().cast::<i32>();
            for (k, prime) in params.primes.iter().enumerate() {
                unsafe {
                    avx::add_reduce_i32(
                        output_ptr.add(k * D),
                        self.limbs[k].as_ptr() as *const i32,
                        rhs.limbs[k].as_ptr() as *const i32,
                        D,
                        prime.p.to_i64() as i32,
                    );
                }
            }
            // SAFETY: the SIMD loop initializes every coefficient in the
            // transparent nested-array representation.
            return unsafe { output.assume_init() };
        }

        let mut output = [[MontCoeff::from_raw(W::default()); D]; K];
        for (k, ((dst_limb, lhs_limb), rhs_limb)) in output
            .iter_mut()
            .zip(self.limbs.iter())
            .zip(rhs.limbs.iter())
            .enumerate()
        {
            let prime = params.primes[k];
            for ((dst, lhs), rhs) in dst_limb.iter_mut().zip(lhs_limb).zip(rhs_limb) {
                let sum = MontCoeff::from_raw(lhs.raw().wrapping_add(rhs.raw()));
                *dst = prime.reduce_range(sum);
            }
        }
        Self { limbs: output }
    }

    /// Add another CRT+NTT element in place and reduce each coefficient.
    pub fn add_assign_reduced(&mut self, rhs: &Self, params: &CrtNttParamSet<W, K, D>) {
        #[cfg(all(target_arch = "aarch64", feature = "parallel"))]
        if params.kernel_plan.uses_neon() {
            for k in 0..K {
                let prime = params.primes[k];
                unsafe {
                    if size_of::<W>() == size_of::<i32>() {
                        neon::add_reduce_i32(
                            self.limbs[k].as_mut_ptr() as *mut i32,
                            rhs.limbs[k].as_ptr() as *const i32,
                            D,
                            prime.p.to_i64() as i32,
                        );
                    } else {
                        neon::add_reduce_i16(
                            self.limbs[k].as_mut_ptr() as *mut i16,
                            rhs.limbs[k].as_ptr() as *const i16,
                            D,
                            prime.p.to_i64() as i16,
                        );
                    }
                }
            }
            return;
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(mode) = params.kernel_plan.x86_pointwise_mode() {
            if size_of::<W>() == size_of::<i16>() {
                for k in 0..K {
                    let prime = params.primes[k];
                    unsafe {
                        avx::add_reduce_i16(
                            self.limbs[k].as_mut_ptr() as *mut i16,
                            rhs.limbs[k].as_ptr() as *const i16,
                            D,
                            prime.p.to_i64() as i16,
                        );
                    }
                }
                return;
            }
            if size_of::<W>() == size_of::<i32>() {
                for k in 0..K {
                    let prime = params.primes[k];
                    unsafe {
                        match mode {
                            AvxNttMode::Avx2 => avx::add_reduce_i32(
                                self.limbs[k].as_mut_ptr() as *mut i32,
                                self.limbs[k].as_ptr() as *const i32,
                                rhs.limbs[k].as_ptr() as *const i32,
                                D,
                                prime.p.to_i64() as i32,
                            ),
                            AvxNttMode::Avx512 => avx::add_reduce_i32_avx512(
                                self.limbs[k].as_mut_ptr() as *mut i32,
                                self.limbs[k].as_ptr() as *const i32,
                                rhs.limbs[k].as_ptr() as *const i32,
                                D,
                                prime.p.to_i64() as i32,
                            ),
                        }
                    }
                }
                return;
            }
        }

        for (k, (limb, rhs_limb)) in self.limbs.iter_mut().zip(rhs.limbs.iter()).enumerate() {
            let prime = params.primes[k];
            for (a, b) in limb.iter_mut().zip(rhs_limb.iter()) {
                let sum = MontCoeff::from_raw(a.raw().wrapping_add(b.raw()));
                *a = prime.reduce_range(sum);
            }
        }
    }

    /// Subtract another CRT+NTT element and reduce.
    pub fn sub_reduced(&self, rhs: &Self, params: &CrtNttParamSet<W, K, D>) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if size_of::<W>() == size_of::<i32>() && params.kernel_plan.uses_x86_transform() {
            let mut out = MaybeUninit::<Self>::uninit();
            let out_ptr = out.as_mut_ptr().cast::<i32>();
            for (k, prime) in params.primes.iter().enumerate() {
                let p = prime.p.to_i64() as i32;
                unsafe {
                    avx::sub_reduce_i32(
                        out_ptr.add(k * D),
                        self.limbs[k].as_ptr() as *const i32,
                        rhs.limbs[k].as_ptr() as *const i32,
                        D,
                        p,
                    )
                }
            }
            // SAFETY: the SIMD loop initializes all `D` coefficients in every
            // limb of the transparent nested-array representation.
            return unsafe { out.assume_init() };
        }
        let mut out = self.clone();
        for (k, (limb, rhs_limb)) in out.limbs.iter_mut().zip(rhs.limbs.iter()).enumerate() {
            let prime = params.primes[k];
            for (a, b) in limb.iter_mut().zip(rhs_limb.iter()) {
                let diff = MontCoeff::from_raw(a.raw().wrapping_sub(b.raw()));
                *a = prime.reduce_range(diff);
            }
        }
        out
    }

    /// Negate each CRT+NTT coefficient and reduce.
    pub fn neg_reduced(&self, params: &CrtNttParamSet<W, K, D>) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if size_of::<W>() == size_of::<i32>() {
            if let Some(mode) = params.kernel_plan.x86_pointwise_mode() {
                let mut out = MaybeUninit::<Self>::uninit();
                let out_ptr = out.as_mut_ptr().cast::<i32>();
                for (k, prime) in params.primes.iter().enumerate() {
                    let p = prime.p.to_i64() as i32;
                    unsafe {
                        match mode {
                            AvxNttMode::Avx2 => avx::neg_reduce_i32(
                                out_ptr.add(k * D),
                                self.limbs[k].as_ptr() as *const i32,
                                D,
                                p,
                            ),
                            AvxNttMode::Avx512 => avx::neg_reduce_i32_avx512(
                                out_ptr.add(k * D),
                                self.limbs[k].as_ptr() as *const i32,
                                D,
                                p,
                            ),
                        }
                    }
                }
                // SAFETY: the SIMD loop initializes all `D` coefficients in
                // every limb of the transparent nested-array representation.
                return unsafe { out.assume_init() };
            }
        }
        let mut out = self.clone();
        for (k, limb) in out.limbs.iter_mut().enumerate() {
            let prime = params.primes[k];
            for a in limb.iter_mut() {
                let neg = MontCoeff::from_raw(a.raw().wrapping_neg());
                *a = prime.reduce_range(neg);
            }
        }
        out
    }

    /// Pointwise multiplication in CRT+NTT domain.
    pub fn pointwise_mul(&self, rhs: &Self, params: &CrtNttParamSet<W, K, D>) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if size_of::<W>() == size_of::<i32>() {
            if let Some(mode) = params.kernel_plan.x86_pointwise_mode() {
                let mut out = MaybeUninit::<Self>::uninit();
                let out_ptr = out.as_mut_ptr().cast::<i32>();
                for (k, prime) in params.primes.iter().copied().enumerate() {
                    // SAFETY: `Self` and `MontCoeff<W>` are transparent over
                    // nested contiguous arrays. The width check above proves
                    // that each coefficient occupies one `i32`, and every
                    // kernel writes all `D` coefficients in its limb.
                    let out_limb = unsafe { out_ptr.add(k * D) };
                    unsafe {
                        match mode {
                            AvxNttMode::Avx2 => avx::pointwise_mul_i32(
                                out_limb,
                                self.limbs[k].as_ptr() as *const i32,
                                rhs.limbs[k].as_ptr() as *const i32,
                                D,
                                prime.p.to_i64() as i32,
                                prime.pinv.to_i64() as i32,
                            ),
                            AvxNttMode::Avx512 => avx::pointwise_mul_i32_avx512(
                                out_limb,
                                self.limbs[k].as_ptr() as *const i32,
                                rhs.limbs[k].as_ptr() as *const i32,
                                D,
                                prime.p.to_i64() as i32,
                                prime.pinv.to_i64() as i32,
                            ),
                        }
                    }
                }
                // SAFETY: the loop above initializes every coefficient in all
                // `K` limbs, which is the complete transparent representation.
                return unsafe { out.assume_init() };
            }
        }
        let mut out = [[MontCoeff::from_raw(W::default()); D]; K];
        for (k, ((output, lhs), rhs)) in out
            .iter_mut()
            .zip(self.limbs.iter())
            .zip(rhs.limbs.iter())
            .enumerate()
        {
            let prime = params.primes[k];
            prime.pointwise_mul(output, lhs, rhs);
            for coefficient in output.iter_mut() {
                *coefficient = prime.reduce_range(*coefficient);
            }
        }
        Self { limbs: out }
    }

    /// Accumulate `lhs * rhs` into `self` in CRT+NTT domain.
    ///
    /// On AArch64, this uses the fused NEON pointwise-multiply-accumulate kernel
    /// when available; otherwise it falls back to the scalar loop.
    #[inline(always)]
    pub fn add_assign_pointwise_mul(
        &mut self,
        lhs: &Self,
        rhs: &Self,
        params: &CrtNttParamSet<W, K, D>,
    ) {
        #[cfg(target_arch = "aarch64")]
        if params.kernel_plan.uses_neon() {
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
        let x86_mode = params.kernel_plan.x86_pointwise_mode();
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
