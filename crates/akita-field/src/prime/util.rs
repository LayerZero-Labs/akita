//! Shared helpers for field arithmetic backends.

use rand_core::RngCore;

/// Sample uniformly from `[0, modulus)` with canonical byte consumption.
///
/// `modulus_bits` is the significant bit length of `modulus`. Each attempt
/// reads exactly `ceil(modulus_bits / 8)` little-endian bytes, clears unused
/// high bits, and rejects candidates greater than or equal to `modulus`.
#[inline]
pub(crate) fn sample_uniform_below<R: RngCore>(
    rng: &mut R,
    modulus: u128,
    modulus_bits: u32,
) -> u128 {
    debug_assert!(modulus > 0);
    debug_assert_eq!(modulus_bits, u128::BITS - modulus.leading_zeros());
    let byte_len = modulus_bits.div_ceil(8) as usize;
    let mask = if modulus_bits == u128::BITS {
        u128::MAX
    } else {
        (1u128 << modulus_bits) - 1
    };
    loop {
        let mut bytes = [0u8; 16];
        rng.fill_bytes(&mut bytes[..byte_len]);
        let candidate = u128::from_le_bytes(bytes) & mask;
        if candidate < modulus {
            return candidate;
        }
    }
}

#[inline(always)]
pub(crate) const fn is_pow2_u64(x: u64) -> bool {
    x != 0 && (x & (x - 1)) == 0
}

#[inline(always)]
pub(crate) const fn log2_pow2_u64(mut x: u64) -> u32 {
    let mut k = 0u32;
    while x > 1 {
        x >>= 1;
        k += 1;
    }
    k
}

/// `a * b` widening to 128 bits; returns `(lo64, hi64)`.
#[inline(always)]
pub(crate) fn mul64_wide(a: u64, b: u64) -> (u64, u64) {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    {
        unsafe { mul64_wide_bmi2(a, b) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        let prod = (a as u128) * (b as u128);
        (prod as u64, (prod >> 64) as u64)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline(always)]
unsafe fn mul64_wide_bmi2(a: u64, b: u64) -> (u64, u64) {
    let mut hi = 0;
    let lo = unsafe { std::arch::x86_64::_mulx_u64(a, b, &mut hi) };
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::sample_uniform_below;
    use crate::{
        pseudo_mersenne_modulus, CanonicalField, Prime128OffsetA7F7, Prime32Offset99,
        Prime64Offset59, PseudoMersenneField, RandomSampling,
    };
    use rand_core::{Error, RngCore};

    struct ScriptedRng {
        bytes: Vec<u8>,
        cursor: usize,
    }

    impl ScriptedRng {
        fn new(bytes: Vec<u8>) -> Self {
            Self { bytes, cursor: 0 }
        }
    }

    impl RngCore for ScriptedRng {
        fn next_u32(&mut self) -> u32 {
            let mut bytes = [0u8; 4];
            self.fill_bytes(&mut bytes);
            u32::from_le_bytes(bytes)
        }

        fn next_u64(&mut self) -> u64 {
            let mut bytes = [0u8; 8];
            self.fill_bytes(&mut bytes);
            u64::from_le_bytes(bytes)
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) {
            let end = self.cursor + dest.len();
            dest.copy_from_slice(&self.bytes[self.cursor..end]);
            self.cursor = end;
        }

        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
            self.fill_bytes(dest);
            Ok(())
        }
    }

    #[test]
    fn rejection_consumes_candidates_and_resumes_at_the_cursor() {
        let mut rng = ScriptedRng::new(vec![251, 7, 250]);
        assert_eq!(sample_uniform_below(&mut rng, 251, 8), 7);
        assert_eq!(rng.cursor, 2);
        assert_eq!(sample_uniform_below(&mut rng, 251, 8), 250);
        assert_eq!(rng.cursor, 3);
    }

    #[test]
    fn non_byte_aligned_modulus_masks_unused_high_bits() {
        // 0xff_ff_ff_ff becomes 0x3f_ff_ff_ff at a 30-bit modulus width, then
        // is rejected. The following little-endian candidate 42 is accepted.
        let mut rng = ScriptedRng::new(vec![0xff, 0xff, 0xff, 0xff, 42, 0, 0, 0]);
        assert_eq!(sample_uniform_below(&mut rng, (1u128 << 30) - 35, 30), 42);
        assert_eq!(rng.cursor, 8);
    }

    #[test]
    fn sub_word_modulus_reads_only_its_canonical_byte_width() {
        let mut rng = ScriptedRng::new(vec![42, 0, 0, 99]);
        assert_eq!(sample_uniform_below(&mut rng, (1u128 << 24) - 3, 24), 42);
        assert_eq!(rng.cursor, 3);
    }

    #[test]
    fn prime_fields_share_canonical_rejection_and_byte_consumption() {
        let fp32_modulus = pseudo_mersenne_modulus(
            Prime32Offset99::MODULUS_BITS,
            Prime32Offset99::MODULUS_OFFSET,
        )
        .unwrap();
        let mut fp32_bytes = Vec::from((fp32_modulus as u32).to_le_bytes());
        fp32_bytes.extend_from_slice(&42u32.to_le_bytes());
        let mut fp32_rng = ScriptedRng::new(fp32_bytes);
        assert_eq!(
            Prime32Offset99::random(&mut fp32_rng).to_canonical_u128(),
            42
        );
        assert_eq!(fp32_rng.cursor, 8);

        let fp64_modulus = pseudo_mersenne_modulus(
            Prime64Offset59::MODULUS_BITS,
            Prime64Offset59::MODULUS_OFFSET,
        )
        .unwrap();
        let mut fp64_bytes = Vec::from((fp64_modulus as u64).to_le_bytes());
        fp64_bytes.extend_from_slice(&42u64.to_le_bytes());
        let mut fp64_rng = ScriptedRng::new(fp64_bytes);
        assert_eq!(
            Prime64Offset59::random(&mut fp64_rng).to_canonical_u128(),
            42
        );
        assert_eq!(fp64_rng.cursor, 16);

        let fp128_modulus = pseudo_mersenne_modulus(
            Prime128OffsetA7F7::MODULUS_BITS,
            Prime128OffsetA7F7::MODULUS_OFFSET,
        )
        .unwrap();
        let mut fp128_bytes = Vec::from(fp128_modulus.to_le_bytes());
        fp128_bytes.extend_from_slice(&42u128.to_le_bytes());
        let mut fp128_rng = ScriptedRng::new(fp128_bytes);
        assert_eq!(
            Prime128OffsetA7F7::random(&mut fp128_rng).to_canonical_u128(),
            42
        );
        assert_eq!(fp128_rng.cursor, 32);
    }
}
