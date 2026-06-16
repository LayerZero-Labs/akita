//! JL projection kernels: reference (checked `i64`) and fast (`i32`, optional SIMD).

mod reference;
mod scalar;

#[cfg(target_arch = "x86_64")]
mod simd_x86;

#[cfg(target_arch = "aarch64")]
mod simd_neon;

use akita_field::cfg_into_iter;
#[cfg(feature = "parallel")]
use akita_field::parallel::*;

pub(crate) use reference::project_row as project_row_reference;

/// Map a packed 2-bit pair to its ternary sign: `0b00 -> -1`, `0b11 -> +1`,
/// `0b01`/`0b10 -> 0`.
#[inline]
pub(crate) fn pair_to_sign(pair: u8) -> i8 {
    ((pair == 0b11) as i8) - ((pair == 0b00) as i8)
}

#[inline]
fn project_row_fast(row: &[u8], digits: &[i32], cols: usize) -> i32 {
    #[cfg(all(target_arch = "x86_64", feature = "jl-simd"))]
    {
        if simd_x86::avx512_available() {
            // SAFETY: feature detection + digit-bound contract.
            return unsafe { simd_x86::project_row_avx512(row, digits, cols) };
        }
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: feature detection + digit-bound contract.
            return unsafe { simd_x86::project_row_avx2(row, digits, cols) };
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "jl-simd"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: feature detection + digit-bound contract.
            return unsafe { simd_neon::project_row_neon(row, digits, cols) };
        }
    }

    scalar::project_row(row, digits, cols)
}

/// Project all rows with the fast `i32` kernel (scalar, NEON, AVX2, or AVX-512).
pub(crate) fn project_rows_fast(
    n_rows: usize,
    row_bytes: usize,
    packed_rows: &[u8],
    digits: &[i32],
    cols: usize,
    parallel: bool,
) -> Vec<i32> {
    if parallel {
        cfg_into_iter!(0..n_rows)
            .map(|row_idx| {
                let row_start = row_idx * row_bytes;
                let row = &packed_rows[row_start..row_start + row_bytes];
                project_row_fast(row, digits, cols)
            })
            .collect()
    } else {
        (0..n_rows)
            .map(|row_idx| {
                let row_start = row_idx * row_bytes;
                let row = &packed_rows[row_start..row_start + row_bytes];
                project_row_fast(row, digits, cols)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jl::MAX_JL_DIGIT;

    #[test]
    fn simd_dispatch_matches_scalar_kernel() {
        let cols: usize = 1027;
        let row_bytes = (cols * 2).div_ceil(8);
        let row: Vec<u8> = (0..row_bytes)
            .map(|i: usize| i.wrapping_mul(37) as u8)
            .collect();
        let digits: Vec<i32> = (0..cols).map(|i| (i % 17) as i32 - 8).collect();
        assert!(digits.iter().all(|d| d.abs() <= MAX_JL_DIGIT));

        let scalar = scalar::project_row(&row, &digits, cols);
        let dispatched = project_row_fast(&row, &digits, cols);
        assert_eq!(scalar, dispatched);
    }
}
