//! Constant-time digit decomposition, without value-dependent dispatch.

use super::*;
use akita_error::{checked, AkitaError};

impl<F: Field + CanonicalEncoding> BalancedDecomposePow2Params<F> {
    /// Bias canonical residues with no value-dependent branches or indices.
    #[inline(always)]
    fn biased_ct(&self, canonical: u128) -> [u64; 3] {
        let low = canonical as u64;
        let high = (canonical >> 64) as u64;
        // `select` is all ones exactly when `canonical > threshold`: the sign
        // of `threshold − canonical`, computed in 129-bit signed arithmetic
        // with no comparison or boolean. WARNING: a comparison here, word-wise
        // or as a `u128` borrow flag, lets LLVM turn the bias selection below
        // into a branch on x86, which leaks secret digits.
        let low_difference = u128::from(self.threshold as u64).wrapping_sub(u128::from(low));
        let low_borrow = (low_difference >> 64) & 1;
        let high_difference =
            i128::from((self.threshold >> 64) as u64) - i128::from(high) - low_borrow as i128;
        let select = (high_difference >> 64) as u64;
        let [nonnegative0, nonnegative1, nonnegative2] = self.bias_nonnegative;
        let [negative0, negative1, negative2] = self.bias_negative;
        let bias0 = nonnegative0 ^ ((nonnegative0 ^ negative0) & select);
        let bias1 = nonnegative1 ^ ((nonnegative1 ^ negative1) & select);
        let bias2 = nonnegative2 ^ ((nonnegative2 ^ negative2) & select);
        let word0 = low.wrapping_add(bias0);
        let carry0 = u64::from(word0 < low);
        let partial1 = high.wrapping_add(bias1);
        let carry1a = u64::from(partial1 < high);
        let word1 = partial1.wrapping_add(carry0);
        let carry1b = u64::from(word1 < partial1);
        [word0, word1, bias2.wrapping_add(carry1a | carry1b)]
    }
}

/// Decompose coefficients into digit-major balanced i8 digits.
///
/// Branches and memory indices depend only on public lengths and `params`.
/// Centering uses preview's arithmetic mask; no magnitude scan or digit LUT
/// is used. The guarantee assumes data-independent canonical encoding and
/// integer arithmetic on the target. The field must encode every canonical
/// coefficient in u128; encoding failure is outside the timing guarantee.
/// Invalid public output shapes or bases return `AkitaError` before writing.
pub fn balanced_decompose_coefficients_pow2_i8_into_ct<F: Field + CanonicalEncoding>(
    coefficients: &[F],
    out: &mut [i8],
    params: &BalancedDecomposePow2Params<F>,
) -> Result<(), AkitaError> {
    let expected = checked::product([coefficients.len(), params.levels])
        .ok_or_else(|| AkitaError::InvalidInput("digit output length overflow".into()))?;
    if out.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: out.len(),
        });
    }
    if params.log_basis > 8 {
        return Err(AkitaError::InvalidInput(
            "i8 decomposition requires log_basis <= 8".into(),
        ));
    }
    let mask = (1u64 << params.log_basis) - 1;
    let half = 1i16 << (params.log_basis - 1);
    for (index, coefficient) in coefficients.iter().enumerate() {
        let canonical = coefficient.to_u128_checked().ok_or_else(|| {
            AkitaError::InvalidInput(
                "constant-time decomposition requires u128 coefficients".into(),
            )
        })?;
        let words = params.biased_ct(canonical);
        for level in 0..params.levels {
            let shift = level as u32 * params.log_basis;
            let word = (shift / 64) as usize;
            let bit = shift % 64;
            let mut value = words[word] >> bit;
            if bit + params.log_basis > 64 {
                value |= words[word + 1] << (64 - bit);
            }
            out[level * coefficients.len() + index] = ((value & mask) as i16 - half) as i8;
        }
    }
    Ok(())
}
