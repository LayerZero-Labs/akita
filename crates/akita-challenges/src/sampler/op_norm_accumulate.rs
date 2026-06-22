//! Transposed frequency-row accumulation for the operator-norm predicate.
//!
//! Precomputes `cos_at[pos][k]`, `sin_at[pos][k]` so each nonzero updates all
//! `D/2` frequency accumulators from a contiguous row. On aarch64 the inner
//! row update uses NEON (`i64x2`).

use crate::sampler::op_norm::OpNormTable;

/// Build `freq_cos_at[pos * half_d + k]` and `freq_sin_at[...]` from the
/// certified base tables.
pub(super) fn build_freq_at_tables(
    d: usize,
    base_cos: &[i64],
    base_sin: &[i64],
) -> (Vec<i64>, Vec<i64>) {
    let half_d = d / 2;
    let two_d = 2 * d;
    let mut freq_cos = vec![0i64; d * half_d];
    let mut freq_sin = vec![0i64; d * half_d];
    for pos in 0..d {
        let row = pos * half_d;
        for k in 0..half_d {
            let mult = (2 * k + 1) % two_d;
            let idx = (mult * pos) % two_d;
            freq_cos[row + k] = base_cos[idx];
            freq_sin[row + k] = base_sin[idx];
        }
    }
    (freq_cos, freq_sin)
}

/// Accumulate `R_k, I_k` for `k in 0..num_freqs` via transposed rows.
#[inline]
pub(super) fn accumulate_transposed(
    table: &OpNormTable,
    positions: &[u32],
    coeffs: &[i8],
    acc_re: &mut [i64],
    acc_im: &mut [i64],
) {
    debug_assert_eq!(acc_re.len(), acc_im.len());
    #[cfg(target_arch = "aarch64")]
    {
        accumulate_transposed_neon(table, positions, coeffs, acc_re, acc_im);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        accumulate_transposed_scalar(table, positions, coeffs, acc_re, acc_im);
    }
}

#[inline]
#[cfg(any(not(target_arch = "aarch64"), test))]
pub(super) fn accumulate_transposed_scalar(
    table: &OpNormTable,
    positions: &[u32],
    coeffs: &[i8],
    acc_re: &mut [i64],
    acc_im: &mut [i64],
) {
    let half_d = acc_re.len();
    for (&pos, &coeff) in positions.iter().zip(coeffs.iter()) {
        let coeff = i64::from(coeff);
        let row = table.freq_row(pos as usize, half_d);
        let (cos_row, sin_row) = row;
        for k in 0..half_d {
            acc_re[k] += coeff * cos_row[k];
            acc_im[k] += coeff * sin_row[k];
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn accumulate_transposed_neon_inner(
    table: &OpNormTable,
    positions: &[u32],
    coeffs: &[i8],
    acc_re: &mut [i64],
    acc_im: &mut [i64],
) {
    use core::arch::aarch64::{vgetq_lane_s64, vld1q_s64};

    let half_d = acc_re.len();
    debug_assert_eq!(half_d % 2, 0, "NEON path requires an even half_d");

    for (&pos, &coeff) in positions.iter().zip(coeffs.iter()) {
        let c = i64::from(coeff);
        let row = table.freq_row(pos as usize, half_d);
        let (cos_row, sin_row) = row;
        let mut k = 0;
        while k < half_d {
            let cos_v = vld1q_s64(cos_row.as_ptr().add(k));
            let sin_v = vld1q_s64(sin_row.as_ptr().add(k));
            acc_re[k] += c * vgetq_lane_s64(cos_v, 0);
            acc_re[k + 1] += c * vgetq_lane_s64(cos_v, 1);
            acc_im[k] += c * vgetq_lane_s64(sin_v, 0);
            acc_im[k + 1] += c * vgetq_lane_s64(sin_v, 1);
            k += 2;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) fn accumulate_transposed_neon(
    table: &OpNormTable,
    positions: &[u32],
    coeffs: &[i8],
    acc_re: &mut [i64],
    acc_im: &mut [i64],
) {
    // SAFETY: aarch64 always provides NEON; inner ops are pure lane arithmetic.
    unsafe {
        accumulate_transposed_neon_inner(table, positions, coeffs, acc_re, acc_im);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::op_norm::OpNormTable;

    fn table(d: usize) -> OpNormTable {
        OpNormTable::new(d, 48, (2 * d) as u64, 64).unwrap()
    }

    #[test]
    fn transposed_scalar_matches_legacy_accumulation() {
        let d = 64;
        let t = table(d);
        let half_d = d / 2;
        let mut state = 0xC0FFEE_u64;
        for _ in 0..500 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mut positions = Vec::new();
            let mut coeffs = Vec::new();
            let weight = 8 + (state as usize % 20);
            for i in 0..weight {
                let shift = (i as u32).min(63);
                positions.push((state.wrapping_shr(shift * 3) as u32) % d as u32);
                coeffs.push(if state.wrapping_shr(shift) & 1 == 0 { 1 } else { -2 });
            }
            let mut legacy_re = vec![0i64; half_d];
            let mut legacy_im = vec![0i64; half_d];
            let two_d = 2 * d;
            for k in 0..half_d {
                let mult = (2 * k + 1) % two_d;
                for (&pos, &coeff) in positions.iter().zip(coeffs.iter()) {
                    let idx = (mult * pos as usize) % two_d;
                    let coeff = i64::from(coeff);
                    legacy_re[k] += coeff * t.base_cos_i64(idx);
                    legacy_im[k] += coeff * t.base_sin_i64(idx);
                }
            }
            let mut trans_re = vec![0i64; half_d];
            let mut trans_im = vec![0i64; half_d];
            accumulate_transposed_scalar(&t, &positions, &coeffs, &mut trans_re, &mut trans_im);
            assert_eq!(legacy_re, trans_re);
            assert_eq!(legacy_im, trans_im);
            #[cfg(target_arch = "aarch64")]
            {
                let mut neon_re = vec![0i64; half_d];
                let mut neon_im = vec![0i64; half_d];
                accumulate_transposed_neon(&t, &positions, &coeffs, &mut neon_re, &mut neon_im);
                assert_eq!(legacy_re, neon_re);
                assert_eq!(legacy_im, neon_im);
            }
        }
    }

    #[test]
    fn transposed_decision_matches_legacy_on_shell_pool() {
        use crate::sampler::exact_shell::sample_exact_shell_challenge;
        use crate::sampler::xof::XofCursor;

        let d = 64;
        let t = table(d);
        let mut cur = XofCursor::from_seed(&[0x42u8; 32]);
        for threshold in [14u64, 16, 18, 22, 30] {
            for _ in 0..200 {
                let ch = sample_exact_shell_challenge(&mut cur, d, 31, 11);
                let legacy = t
                    .decide_parts_legacy_nested_i128(
                        &ch.positions,
                        &ch.coeffs,
                        threshold,
                        d / 2,
                    )
                    .unwrap();
                let via_parts = t
                    .decide_parts(&ch.positions, &ch.coeffs, threshold, d / 2)
                    .unwrap();
                assert_eq!(legacy, via_parts, "threshold {threshold}");
            }
        }
    }
}
