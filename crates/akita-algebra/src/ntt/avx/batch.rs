//! Batched-row AVX-512 NTT kernels.
//!
//! The per-polynomial transforms spend most of their butterfly stages at four
//! or fewer useful `i32` lanes because the supported CRT degrees are small
//! (`D in {32,64,128,256}`). This module transforms `BATCH_LANES = 16`
//! polynomial rows at once with the SIMD lane index equal to the row index.
//! Every butterfly then operates on a full 16-wide vector regardless of `D`,
//! and the twiddle for a given `(stage, j)` is identical across rows so it is a
//! single broadcast.
//!
//! The batch is held transposed in registers: `t[c]` holds coefficient `c` of
//! all 16 rows. The kernels read/write directly from caller memory through a
//! per-lane row stride, so the 16 rows can be the `limb[k]` arrays of 16
//! contiguous `CyclotomicCrtNtt` elements (stride `K*D`) without any copy.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::super::butterfly::NttTwiddles;
use super::super::prime::NttPrime;
use super::{mont_mul_16x_i32_avx512, reduce_range_16x_i32_avx512};

/// Number of polynomial rows processed per batch (one per AVX-512 `i32` lane).
pub(crate) const BATCH_LANES: usize = 16;

/// Lane index vector for gather/scatter: lane `r` is at `r * row_stride` i32.
#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
unsafe fn row_index_vector(row_stride: usize) -> __m512i {
    let s = row_stride as i32;
    _mm512_setr_epi32(
        0,
        s,
        2 * s,
        3 * s,
        4 * s,
        5 * s,
        6 * s,
        7 * s,
        8 * s,
        9 * s,
        10 * s,
        11 * s,
        12 * s,
        13 * s,
        14 * s,
        15 * s,
    )
}

/// Cyclic Gentleman-Sande DIF butterfly stages over a transposed batch.
#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
unsafe fn gs_dif_stages<const D: usize>(
    t: &mut [__m512i; D],
    tw: &NttTwiddles<i32, D>,
    p_v: __m512i,
    pinv_v: __m512i,
) {
    let mut len = D / 2;
    while len > 0 {
        let twiddle_base = len - 1;
        let mut start = 0usize;
        while start < D {
            for j in 0..len {
                let w = _mm512_set1_epi32(tw.fwd_twiddles[twiddle_base + j].raw());
                let u = t[start + j];
                let v = t[start + j + len];
                t[start + j] = reduce_range_16x_i32_avx512(_mm512_add_epi32(u, v), p_v);
                t[start + j + len] =
                    mont_mul_16x_i32_avx512(_mm512_sub_epi32(u, v), w, p_v, pinv_v);
            }
            start += 2 * len;
        }
        len /= 2;
    }
}

/// Load 16 rows (stride `row_stride`) into a transposed register array.
#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
unsafe fn gather_transposed<const D: usize>(base: *const i32, row_stride: usize) -> [__m512i; D] {
    let vindex = row_index_vector(row_stride);
    let mut t = [_mm512_setzero_si512(); D];
    for (c, slot) in t.iter_mut().enumerate() {
        *slot = _mm512_i32gather_epi32::<4>(vindex, base.add(c));
    }
    t
}

/// Store a transposed register array back to 16 rows (stride `row_stride`).
#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
unsafe fn scatter_transposed<const D: usize>(base: *mut i32, row_stride: usize, t: &[__m512i; D]) {
    let vindex = row_index_vector(row_stride);
    for (c, slot) in t.iter().enumerate() {
        _mm512_i32scatter_epi32::<4>(base.add(c), vindex, *slot);
    }
}

/// Batched forward negacyclic NTT for `BATCH_LANES` rows of degree `D`.
///
/// Lane `r` corresponds to the row at `base.add(r * row_stride)`. Produces, per
/// row, the same result as the scalar/per-poly
/// [`super::super::butterfly::forward_ntt`].
///
/// # Safety
///
/// AVX-512F/DQ/BW must be available. `base` must be valid for reads and writes
/// of `BATCH_LANES` rows at `row_stride` spacing, each holding `D` `i32`.
#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
pub(crate) unsafe fn batched_forward_ntt_16rows<const D: usize>(
    base: *mut i32,
    row_stride: usize,
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let mut t = gather_transposed::<D>(base as *const i32, row_stride);
    let p_v = _mm512_set1_epi32(prime.p);
    let pinv_v = _mm512_set1_epi32(prime.pinv);

    // Negacyclic twist: multiply coefficient c by psi^c (broadcast per c).
    for (c, slot) in t.iter_mut().enumerate() {
        let psi = _mm512_set1_epi32(tw.psi_pows[c].raw());
        *slot = mont_mul_16x_i32_avx512(*slot, psi, p_v, pinv_v);
    }

    gs_dif_stages::<D>(&mut t, tw, p_v, pinv_v);

    for slot in t.iter_mut() {
        *slot = reduce_range_16x_i32_avx512(*slot, p_v);
    }

    scatter_transposed::<D>(base, row_stride, &t);
}

/// Batched forward cyclic NTT for `BATCH_LANES` rows of degree `D` (no twist).
///
/// Produces, per row, the same result as the scalar/per-poly
/// [`super::super::butterfly::forward_ntt_cyclic`].
///
/// # Safety
///
/// Same contract as [`batched_forward_ntt_16rows`].
#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
pub(crate) unsafe fn batched_forward_ntt_cyclic_16rows<const D: usize>(
    base: *mut i32,
    row_stride: usize,
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let mut t = gather_transposed::<D>(base as *const i32, row_stride);
    let p_v = _mm512_set1_epi32(prime.p);
    let pinv_v = _mm512_set1_epi32(prime.pinv);

    gs_dif_stages::<D>(&mut t, tw, p_v, pinv_v);

    for slot in t.iter_mut() {
        *slot = reduce_range_16x_i32_avx512(*slot, p_v);
    }

    scatter_transposed::<D>(base, row_stride, &t);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::butterfly::{forward_ntt, forward_ntt_cyclic};
    use crate::ntt::prime::MontCoeff;

    fn random_rows<const D: usize>(
        prime: NttPrime<i32>,
        seed: u64,
    ) -> [[MontCoeff<i32>; D]; BATCH_LANES] {
        let mut state = seed;
        std::array::from_fn(|_| {
            std::array::from_fn(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let val = ((state >> 33) as i64 % prime.p as i64) as i32;
                prime.from_canonical(val)
            })
        })
    }

    fn check_forward<const D: usize>(twist: bool) {
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512bw"))
        {
            return;
        }
        let prime = NttPrime::compute(1073707009_i32);
        let tw = NttTwiddles::<i32, D>::compute(prime);
        let mut rows = random_rows::<D>(prime, 0x9e37 + D as u64);
        let reference = rows;

        let base = rows.as_mut_ptr() as *mut i32;
        // SAFETY: guarded by AVX-512 detection; rows are 16 contiguous D-arrays.
        unsafe {
            if twist {
                batched_forward_ntt_16rows::<D>(base, D, prime, &tw);
            } else {
                batched_forward_ntt_cyclic_16rows::<D>(base, D, prime, &tw);
            }
        }

        for (r, row) in rows.iter().enumerate() {
            let mut expected = reference[r];
            if twist {
                forward_ntt(&mut expected, prime, &tw);
            } else {
                forward_ntt_cyclic(&mut expected, prime, &tw);
            }
            assert_eq!(*row, expected, "row {r} mismatch (D={D}, twist={twist})");
        }
    }

    #[test]
    fn batched_forward_matches_per_poly() {
        check_forward::<32>(true);
        check_forward::<64>(true);
        check_forward::<128>(true);
        check_forward::<256>(true);
    }

    #[test]
    fn batched_forward_cyclic_matches_per_poly() {
        check_forward::<32>(false);
        check_forward::<64>(false);
        check_forward::<128>(false);
        check_forward::<256>(false);
    }
}
