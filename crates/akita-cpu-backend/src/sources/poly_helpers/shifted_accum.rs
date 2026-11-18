//! Register-blocked i16 accumulation for sparse challenges at any positions.
//!
//! Multiplying a digit plane `x` by `X^p` in `Z[X]/(X^D + 1)` sends output
//! coefficient `i` the value `x[i - p]` when `i >= p`, else `-x[i - p + D]`.
//! Both cases read `ext[i - p + D]` from the extended plane
//! `ext = (-x, x, -x, 0)`, and the negated term reads `ext[i - p + 2D]`. So
//! every output window of [`WINDOW`] coefficients is a sum of contiguous
//! windows of `ext`, one per challenge term, which stays in registers while
//! every term is added, instead of loading and storing the accumulator once
//! per term. Unlike the subring gather in `windowed_accum`, the windows need
//! not be aligned, so any challenge positions qualify.
//!
//! The source lists are [`WindowLists`], and on aarch64 `i8` sums run
//! through the NEON gather of `windowed_accum`. Every output coefficient
//! receives the same integer sum as the per-term kernels in `narrow_accum`,
//! so the result is bit-identical.

use super::windowed_accum::{WindowLists, WINDOW, WORD_OFFSETS};
use akita_challenges::SparseChallenge;

/// Per-window source lists for one challenge, with offsets into the extended
/// plane; unused slots point at its zeros.
pub(crate) struct ShiftedChallenge {
    lists: WindowLists,
    /// Whether every `i8` digit has an `i8` negation, so `i8` planes extend
    /// in `i8`; otherwise they extend widened to `i16`.
    negates_i8: bool,
}

impl ShiftedChallenge {
    /// Build the window lists, or `None` when the ring degree is not a
    /// multiple of [`WINDOW`], the extended plane outgrows the `u16` offsets,
    /// or [`WindowLists::new`] rejects the challenge. `digit_abs_bound` must
    /// bound every digit the lists are used with.
    pub(super) fn new<const D: usize>(
        challenge: &SparseChallenge,
        digit_abs_bound: u64,
    ) -> Option<Self> {
        if !D.is_multiple_of(WINDOW)
            || extended_len::<D>() > usize::from(u16::MAX)
            || challenge
                .positions
                .iter()
                .any(|&position| position as usize >= D)
        {
            return None;
        }
        let lists = WindowLists::new(
            challenge,
            D / WINDOW,
            (3 * D) as u64,
            |window, position, coefficient| {
                let sign = if coefficient < 0 { D } else { 0 };
                (D - position + window * WINDOW + sign) as u64
            },
        )?;
        Some(Self {
            lists,
            negates_i8: digit_abs_bound <= i8::MAX as u64,
        })
    }

    /// The `(once, twice)` lists of every output window of a degree-`D`
    /// ring. Panics unless the lists were built for degree `D`.
    fn windows<const D: usize>(&self) -> impl Iterator<Item = (&[u64], &[u64])> {
        assert_eq!(self.lists.num_windows(), D / WINDOW);
        self.lists.windows()
    }
}

/// Whether the shifted kernel is the faster narrow kernel here.
///
/// Only the NEON implementation has been measured to beat the term-at-a-time
/// kernel, so other targets keep the existing narrow path.
pub(super) fn shifts_windows() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        super::use_simd_decompose_fold()
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// `(-x, x, -x)` followed by one window of zeros.
const fn extended_len<const D: usize>() -> usize {
    3 * D + WINDOW
}

/// Reusable storage for the extended plane of one digit plane.
pub(crate) struct ShiftScratch<const D: usize> {
    i8_plane: Vec<i8>,
    i16_plane: Vec<i16>,
}

impl<const D: usize> ShiftScratch<D> {
    pub(crate) fn new() -> Self {
        Self {
            i8_plane: Vec::new(),
            i16_plane: Vec::new(),
        }
    }
}

/// Write `(-plane, plane, -plane, 0)` into `ext`. The caller ensures every
/// negation fits `T`.
#[inline(always)]
fn fill_extended<S: Copy, T: Copy + Default, const D: usize>(
    plane: &[S; D],
    ext: &mut Vec<T>,
    convert: impl Fn(S) -> T,
    negate: impl Fn(S) -> T,
) {
    ext.resize(extended_len::<D>(), T::default());
    let (copies, zeros) = ext.split_at_mut(3 * D);
    let (low, rest) = copies.split_at_mut(D);
    let (middle, high) = rest.split_at_mut(D);
    for (((low, middle), high), &value) in low.iter_mut().zip(middle).zip(high).zip(plane) {
        *low = negate(value);
        *middle = convert(value);
        *high = *low;
    }
    zeros.fill(T::default());
}

/// Add `challenge * plane` to `acc`.
///
/// The caller must prove that the complete partial sum stays in the `i16`
/// range, and the digit bound given to [`ShiftedChallenge::new`] must bound
/// every digit.
pub(super) fn shifted_mul_acc<const D: usize>(
    plane: &[i8; D],
    challenge: &ShiftedChallenge,
    scratch: &mut ShiftScratch<D>,
    acc: &mut [i16; D],
) {
    if challenge.negates_i8 {
        fill_extended(plane, &mut scratch.i8_plane, |x| x, |x: i8| -x);
        accumulate_i8(&scratch.i8_plane, challenge, acc);
    } else {
        fill_extended(plane, &mut scratch.i16_plane, i16::from, |x| -i16::from(x));
        accumulate_i16(&scratch.i16_plane, challenge, acc);
    }
}

/// Signed-`i16` source variant of [`shifted_mul_acc`].
pub(super) fn shifted_mul_acc_i16<const D: usize>(
    plane: &[i16; D],
    challenge: &ShiftedChallenge,
    scratch: &mut ShiftScratch<D>,
    acc: &mut [i16; D],
) {
    fill_extended(plane, &mut scratch.i16_plane, |x| x, |x: i16| -x);
    accumulate_i16(&scratch.i16_plane, challenge, acc);
}

/// Add the listed windows of the `i8` extended plane `ext` to `acc`.
#[inline(always)]
fn accumulate_i8<const D: usize>(ext: &[i8], challenge: &ShiftedChallenge, acc: &mut [i16; D]) {
    assert_eq!(ext.len(), extended_len::<D>());
    #[cfg(target_arch = "aarch64")]
    for (acc_window, (once, twice)) in acc.chunks_exact_mut(WINDOW).zip(challenge.windows::<D>()) {
        // SAFETY: `ShiftedChallenge::new::<D>` only packs offsets `o` with
        // `o + WINDOW <= 3 * D + WINDOW`, the length of `ext` checked above,
        // and `windows::<D>` ties the lists to this `D`; `acc_window` is
        // exactly one window; NEON is baseline on aarch64.
        unsafe {
            super::windowed_accum::neon::gather_window::<false>(
                ext.as_ptr(),
                once,
                twice,
                0,
                acc_window.as_mut_ptr(),
            );
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    accumulate_portable(ext, challenge, acc);
}

/// Add the listed windows of the `i16` extended plane `ext` to `acc`.
#[inline(always)]
fn accumulate_i16<const D: usize>(ext: &[i16], challenge: &ShiftedChallenge, acc: &mut [i16; D]) {
    assert_eq!(ext.len(), extended_len::<D>());
    #[cfg(target_arch = "aarch64")]
    for (acc_window, (once, twice)) in acc.chunks_exact_mut(WINDOW).zip(challenge.windows::<D>()) {
        // SAFETY: as in `accumulate_i8`, every packed offset leaves one window
        // inside `ext`, and `acc_window` is one window.
        unsafe {
            neon::gather_window_i16(ext.as_ptr(), once, twice, acc_window.as_mut_ptr());
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    accumulate_portable(ext, challenge, acc);
}

#[cfg(not(target_arch = "aarch64"))]
/// Portable sum of the listed windows, scaled by one or two.
#[inline(always)]
fn add_listed<T: Copy + Into<i16>>(ext: &[T], words: &[u64], scale: i16, sum: &mut [i16]) {
    for &word in words {
        for slot in 0..WORD_OFFSETS {
            let offset = (word >> (16 * slot)) as u16 as usize;
            for (lane, &value) in sum.iter_mut().zip(&ext[offset..offset + WINDOW]) {
                *lane = lane.wrapping_add(scale.wrapping_mul(value.into()));
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{WINDOW, WORD_OFFSETS};
    use std::arch::aarch64::*;

    /// Low and high halves of one window.
    type Halves = (int16x8_t, int16x8_t);

    /// Sum of the `i16` windows listed in `words`, in one running sum per
    /// word slot so the add chains overlap.
    ///
    /// # Safety
    ///
    /// Every offset in `words` must leave [`WINDOW`] readable values at
    /// `sources + offset`.
    #[inline(always)]
    unsafe fn sum_windows(sources: *const i16, words: &[u64]) -> Halves {
        let mut sums = [(vdupq_n_s16(0), vdupq_n_s16(0)); WORD_OFFSETS];
        for &word in words {
            for (slot, sum) in sums.iter_mut().enumerate() {
                let source = sources.add((word >> (16 * slot)) as u16 as usize);
                sum.0 = vaddq_s16(sum.0, vld1q_s16(source));
                sum.1 = vaddq_s16(sum.1, vld1q_s16(source.add(WINDOW / 2)));
            }
        }
        (
            vaddq_s16(
                vaddq_s16(sums[0].0, sums[1].0),
                vaddq_s16(sums[2].0, sums[3].0),
            ),
            vaddq_s16(
                vaddq_s16(sums[0].1, sums[1].1),
                vaddq_s16(sums[2].1, sums[3].1),
            ),
        )
    }

    /// Add the windows listed in `once` and twice those listed in `twice` to
    /// the accumulator window.
    ///
    /// # Safety
    ///
    /// Same contract as [`sum_windows`] for both lists, and `acc_window` must
    /// address [`WINDOW`] valid `i16` values.
    #[inline(always)]
    pub(super) unsafe fn gather_window_i16(
        sources: *const i16,
        once: &[u64],
        twice: &[u64],
        acc_window: *mut i16,
    ) {
        let (mut low, mut high) = sum_windows(sources, once);
        if !twice.is_empty() {
            let (twice_low, twice_high) = sum_windows(sources, twice);
            low = vaddq_s16(low, vshlq_n_s16::<1>(twice_low));
            high = vaddq_s16(high, vshlq_n_s16::<1>(twice_high));
        }
        vst1q_s16(acc_window, vaddq_s16(vld1q_s16(acc_window), low));
        let acc_high = acc_window.add(WINDOW / 2);
        vst1q_s16(acc_high, vaddq_s16(vld1q_s16(acc_high), high));
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn accumulate_portable<T: Copy + Into<i16>, const D: usize>(
    ext: &[T],
    challenge: &ShiftedChallenge,
    acc: &mut [i16; D],
) {
    for (acc_window, (once, twice)) in acc.chunks_exact_mut(WINDOW).zip(challenge.windows::<D>()) {
        let mut sum = [0i16; WINDOW];
        add_listed(ext, once, 1, &mut sum);
        add_listed(ext, twice, 2, &mut sum);
        for (lane, sum) in acc_window.iter_mut().zip(sum) {
            *lane = lane.wrapping_add(sum);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::narrow_accum::{sparse_mul_acc, sparse_mul_acc_i16};
    use super::*;

    fn challenge(seed: u32) -> SparseChallenge {
        let positions = (0..41u32)
            .map(|term| (term * 37 + seed * 11) % 512)
            .collect();
        let coeffs = (0..41)
            .map(|term| [1i8, -1, 2, -2][(term + seed as usize) % 4])
            .collect();
        SparseChallenge { positions, coeffs }
    }

    fn boxed<T: std::fmt::Debug, const D: usize>(value: impl Fn(usize) -> T) -> Box<[T; D]> {
        let values: Vec<T> = (0..D).map(value).collect();
        values.try_into().expect("D values")
    }

    /// Compare against the per-term kernels on one challenge, with `i8` digits
    /// at the bound 127, widened `i8` digits that reach -128, and `i16` digits.
    fn check_boundaries<const D: usize>(challenge: &SparseChallenge, seed: usize) {
        let mut scratch = ShiftScratch::<D>::new();
        let narrow = ShiftedChallenge::new::<D>(challenge, 127).expect("admitted challenge");
        let widened = ShiftedChallenge::new::<D>(challenge, 128).expect("admitted challenge");
        assert!(narrow.negates_i8 && !widened.negates_i8);
        let at_bound: Box<[i8; D]> = boxed(|index| [127, -127, 0, 1][(index + seed) % 4]);
        let full: Box<[i8; D]> = boxed(|index| [-128, 127, -1, 0][(index + seed) % 4]);
        let wide: Box<[i16; D]> = boxed(|index| [255, -255, 3, 0][(index + seed) % 4]);
        let start: Box<[i16; D]> = boxed(|index| (index % 17) as i16 - 8);
        for (plane, table) in [(&at_bound, &narrow), (&full, &widened)] {
            let (mut expected, mut actual) = (start.clone(), start.clone());
            sparse_mul_acc(plane, challenge, &mut expected);
            shifted_mul_acc(plane, table, &mut scratch, &mut actual);
            assert_eq!(actual, expected);
        }
        let (mut expected, mut actual) = (start.clone(), start);
        sparse_mul_acc_i16(&wide, challenge, &mut expected);
        shifted_mul_acc_i16(&wide, &narrow, &mut scratch, &mut actual);
        assert_eq!(actual, expected);
    }

    /// Mixed 41-term challenges, the table boundaries (positions 0 and
    /// `D - 1`, single-sign and magnitude-two challenges up to 64 terms), and
    /// the largest degree whose extended plane the `u16` offsets address.
    #[test]
    fn shifted_windows_match_per_term_accumulation() {
        const D: usize = 512;
        let sparse = |positions: Vec<u32>, coeffs: Vec<i8>| SparseChallenge {
            positions: positions.into(),
            coeffs: coeffs.into(),
        };
        let cases = [
            sparse(vec![0, D as u32 - 1, 255], vec![1, -2, -1]),
            sparse((0..41).map(|term| term * 12 + 5).collect(), vec![-1; 41]),
            sparse(
                (0..64).map(|term| term * 8 + 7).collect(),
                (0..64)
                    .map(|term| if term % 2 == 0 { 2 } else { -2 })
                    .collect(),
            ),
            sparse((0..64).map(|term| term * 8).collect(), vec![-2; 64]),
        ];
        for (seed, challenge) in cases.iter().enumerate() {
            check_boundaries::<D>(challenge, seed);
        }
        for seed in 0..4 {
            check_boundaries::<D>(&challenge(seed), seed as usize);
        }

        // 3 * 21,824 + 16 = 65,488 fits a u16 offset; 3 * 21,840 + 16 does not.
        const MAX_D: usize = 21_824;
        check_boundaries::<MAX_D>(
            &sparse(vec![0, MAX_D as u32 - 1, 10_000], vec![-2, 1, 2]),
            0,
        );
        assert!(ShiftedChallenge::new::<21_840>(&sparse(vec![0], vec![1]), 64).is_none());
    }

    #[test]
    fn shifted_challenge_rejects_large_coefficients_and_ragged_degrees() {
        let mut large = challenge(0);
        large.coeffs[3] = 3;
        assert!(ShiftedChallenge::new::<512>(&large, 64).is_none());
        let single = |position| SparseChallenge {
            positions: vec![position].into(),
            coeffs: vec![1].into(),
        };
        assert!(ShiftedChallenge::new::<24>(&single(1), 64).is_none());
        assert!(ShiftedChallenge::new::<16>(&single(16), 64).is_none());
    }
}
