use std::sync::Arc;

use crate::ntt::binary::{forward_scalar, BinaryLimbTables};
use crate::ntt::{BinaryNttStrategy, MontCoeff, PrimeWidth};

use super::{CrtNttParamSet, CyclotomicCrtNtt, DigitMontLut};

impl<W: PrimeWidth, const K: usize> DigitMontLut<W, K> {
    pub(super) fn try_fill_binary_limb<const D: usize>(
        &self,
        k: usize,
        digits: &[i8; D],
        params: &CrtNttParamSet<W, K, D>,
        out: &mut [MontCoeff<W>; D],
        strategy: BinaryNttStrategy,
    ) -> bool {
        if D < 4
            || (strategy == BinaryNttStrategy::Split8 && D < 8)
            || (digits.iter().fold(0u8, |acc, &digit| acc | digit as u8) & !1) != 0
        {
            return false;
        }
        let tables = self.binary.get_or_init(|| {
            Arc::new(std::array::from_fn(|i| {
                BinaryLimbTables::new(params.primes[i], &params.twiddles[i])
            }))
        });
        let table = &tables[k];
        let tw = &params.twiddles[k];
        let prime = params.primes[k];
        // Digit LUTs may be reused across ring dimensions. Cached startup tables
        // additionally depend on D, the exact Montgomery prime, and the root.
        if table.factors.len() != D || table.prime != prime || table.psi != tw.psi_pows[1] {
            return false;
        }
        #[cfg(target_arch = "aarch64")]
        if params.kernel_plan().uses_neon()
            && std::mem::size_of::<W>() == 4
            && D >= 16
            && (strategy != BinaryNttStrategy::Split8 || D >= 32)
        {
            // SAFETY: sealed PrimeWidth is i16 or i32. The width check selects
            // i32; all arrays have D entries; cached geometry was checked above.
            // BinaryLimbTables itself is not reinterpreted: its typed slices are
            // passed independently to preserve Rust layout guarantees.
            unsafe {
                crate::ntt::neon::forward_binary_i32(
                    &mut *(out as *mut _ as *mut [MontCoeff<i32>; D]),
                    digits,
                    *(&prime as *const _ as *const crate::ntt::NttPrime<i32>),
                    &*(tw as *const _ as *const crate::ntt::NttTwiddles<i32, D>),
                    std::slice::from_raw_parts(table.compact.as_ptr().cast::<i32>(), 64),
                    std::slice::from_raw_parts(table.factors.as_ptr().cast::<i32>(), D),
                    std::slice::from_raw_parts(table.pairs.as_ptr().cast::<i32>(), 2 * D),
                    if strategy == BinaryNttStrategy::Positional4 {
                        Some(std::slice::from_raw_parts(
                            table.positional(prime).as_ptr().cast::<[i32; 16]>(),
                            D,
                        ))
                    } else {
                        None
                    },
                    if strategy == BinaryNttStrategy::Selected4 {
                        Some(std::slice::from_raw_parts(
                            table.selected4(prime).as_ptr().cast::<i32>(),
                            4 * D,
                        ))
                    } else {
                        None
                    },
                    std::slice::from_raw_parts(table.split_twiddles.as_ptr().cast::<i32>(), D),
                    std::slice::from_raw_parts(table.split_companions.as_ptr().cast::<i32>(), D),
                    if strategy == BinaryNttStrategy::Split8 {
                        Some(std::slice::from_raw_parts(
                            table.split8(prime).as_ptr().cast::<i32>(),
                            256,
                        ))
                    } else {
                        None
                    },
                    strategy,
                );
            }
            return true;
        }
        forward_scalar(out, digits, prime, tw, table, strategy);
        true
    }
}

impl<W: PrimeWidth, const K: usize, const D: usize> CyclotomicCrtNtt<W, K, D> {
    /// Transform binary coefficients using a selected startup strategy.
    ///
    /// Returns `None` for nonbinary input, D < 4 (D < 8 for Split8), or a LUT whose cached binary
    /// tables belong to different parameters. Prepare the LUT from `params` and
    /// reuse it across transforms to amortize table construction.
    pub fn from_binary_with_lut(
        digits: &[i8; D],
        params: &CrtNttParamSet<W, K, D>,
        lut: &DigitMontLut<W, K>,
        strategy: BinaryNttStrategy,
    ) -> Option<Self> {
        let mut limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        for (k, limb) in limbs.iter_mut().enumerate() {
            if !lut.try_fill_binary_limb(k, digits, params, limb, strategy) {
                return None;
            }
        }
        Some(Self { limbs })
    }
}
