//! Window-gathered i16 accumulation for subring-embedded sparse challenges.
//!
//! A challenge embedded from a subring only has support on multiples of the
//! embedding stride. When that stride is a multiple of [`WINDOW`], multiplying
//! by one monomial moves whole `WINDOW`-coefficient windows of the digit plane
//! without splitting any of them at the negacyclic wrap. Each output window is
//! therefore a signed sum of whole source windows, so it can stay in registers
//! while every term is added, instead of being loaded and stored once per term.
//!
//! Subtracted windows are read from the bitwise complement of the plane
//! (`!x = -x - 1`, which never leaves the range of `x`) and the missing `+1`
//! per subtracted window is added back as one precomputed constant. Every
//! output window then runs the same add-only loop with the same trip count.
//!
//! NEON widens while it adds. AVX2 and AVX-512 have no widening add, so their
//! kernel widens the plane to `i16` once and reads subtracted windows from
//! its negation.

use akita_challenges::SparseChallenge;

/// Coefficients per gathered window: one 128-bit vector of `i8` digits.
pub(super) const WINDOW: usize = 16;

/// Coefficients gathered per table entry: two neighbouring windows for the
/// AVX-512 kernel, which holds 32 `i16` sums in one register, else one.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
const GATHER: usize = 2 * WINDOW;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512bw")))]
const GATHER: usize = WINDOW;

/// Source offsets packed into one table word, 16 bits each.
const WORD_OFFSETS: usize = 4;

/// Value range of the `i8` digits a fold source hands to the gather kernel.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum I8DigitSpan {
    /// Any `i8` value.
    Full,
    /// Every digit lies in `[-64, 63]`, so two digits (or their complements)
    /// can be added in `i8` before widening.
    Half,
}

/// A digit plane, its bitwise complement and zeros, so one offset selects the
/// source and its sign, and padding offsets add nothing.
///
/// `head` repeats the first window of the plane. A two-window gather that
/// starts in the last window of `plane` or of `complement` then continues
/// into the first window with the opposite sign, which is what the negacyclic
/// wrap needs.
///
/// The x86 kernels read the same layout from [`x86::WidePlane`] instead.
#[cfg_attr(all(target_arch = "x86_64", target_feature = "avx2"), allow(dead_code))]
#[repr(C, align(16))]
struct SignedPlane<const D: usize> {
    plane: [i8; D],
    complement: [i8; D],
    head: [i8; WINDOW],
    zero: [i8; GATHER],
}

/// Per-window source lists for one challenge.
///
/// Every run of [`GATHER`] output coefficients owns `words_per_window` table
/// words: first `once_words` words listing the terms with coefficient
/// magnitude one, then the words listing the terms with magnitude two. Each
/// word packs [`WORD_OFFSETS`] [`SignedPlane`] offsets; unused slots point at
/// the zeros. `corrections[window]` is the weighted number of complemented
/// sources in that window's list.
pub(super) struct WindowedChallenge {
    words: Vec<u64>,
    // The x86 kernels read negated sources, which need no correction.
    #[cfg_attr(all(target_arch = "x86_64", target_feature = "avx2"), allow(dead_code))]
    corrections: Vec<i16>,
    once_words: usize,
    words_per_window: usize,
    // Read only by the NEON kernel.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    span: I8DigitSpan,
}

impl WindowedChallenge {
    /// Build the gather lists, or `None` when the challenge is not a sum of
    /// whole-window shifts with coefficients of magnitude one or two in a
    /// degree-`D` ring.
    pub(super) fn new<const D: usize>(
        challenge: &SparseChallenge,
        span: I8DigitSpan,
    ) -> Option<Self> {
        let num_terms = challenge.positions.len();
        if !D.is_multiple_of(GATHER)
            || 2 * D + WINDOW > usize::from(u16::MAX)
            || num_terms == 0
            || num_terms > i16::MAX as usize / 2
            || num_terms != challenge.coeffs.len()
        {
            return None;
        }
        let whole_windows = challenge
            .positions
            .iter()
            .all(|&position| (position as usize) < D && (position as usize).is_multiple_of(WINDOW));
        let small_coefficients = challenge
            .coeffs
            .iter()
            .all(|coefficient| matches!(coefficient.unsigned_abs(), 1 | 2));
        if !whole_windows || !small_coefficients {
            return None;
        }

        let num_once = challenge
            .coeffs
            .iter()
            .filter(|coefficient| coefficient.unsigned_abs() == 1)
            .count();
        let once_words = num_once.div_ceil(WORD_OFFSETS);
        let words_per_window = once_words + (num_terms - num_once).div_ceil(WORD_OFFSETS);
        let num_windows = D / WINDOW;
        let mut words = Vec::with_capacity(D / GATHER * words_per_window);
        let mut corrections = Vec::with_capacity(D / GATHER);
        let mut offsets = Vec::with_capacity(num_terms);
        for window in (0..num_windows).step_by(GATHER / WINDOW) {
            let mut correction = 0i16;
            for magnitude in [1u8, 2] {
                offsets.clear();
                for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
                    if coefficient.unsigned_abs() != magnitude {
                        continue;
                    }
                    let shift = position as usize / WINDOW;
                    // X^position moves source window `window - shift` here;
                    // the windows that wrap past X^D come back negated. A
                    // two-window gather reads on into the next source
                    // window, which `SignedPlane` keeps adjacent across the
                    // wrap as well.
                    let wrapped = window < shift;
                    let source = if wrapped {
                        window + num_windows - shift
                    } else {
                        window - shift
                    };
                    let mut offset = source * WINDOW;
                    if (coefficient < 0) != wrapped {
                        offset += D;
                        correction += i16::from(magnitude);
                    }
                    offsets.push(offset as u64);
                }
                words.extend(offsets.chunks(WORD_OFFSETS).map(|group| {
                    (0..WORD_OFFSETS).fold(0u64, |word, slot| {
                        let zeros = (2 * D + WINDOW) as u64;
                        let offset = group.get(slot).copied().unwrap_or(zeros);
                        word | (offset << (16 * slot))
                    })
                }));
            }
            corrections.push(correction);
        }
        Some(Self {
            words,
            corrections,
            once_words,
            words_per_window,
            span,
        })
    }
}

/// Whether the window-gathered kernel is the faster narrow kernel here.
///
/// Only the NEON, AVX2 and AVX-512 implementations have been measured to
/// beat the term-at-a-time kernel, so other targets keep the existing narrow
/// path.
pub(super) fn gathers_windows() -> bool {
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    ))]
    {
        super::use_simd_decompose_fold()
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    )))]
    {
        false
    }
}

/// Accumulate `challenge * digit_plane` into a proven-safe i16 partial sum.
///
/// The caller must prove that the complete partial sum stays in the i16 range
/// (the bound is the same as for the term-at-a-time narrow kernel) and that
/// the digits lie in the span `challenge` was built for.
#[inline(always)]
pub(super) fn windowed_mul_acc<const D: usize>(
    digit_plane: &[i8; D],
    challenge: &WindowedChallenge,
    acc: &mut [i16; D],
) {
    assert!(D.is_multiple_of(GATHER));
    assert_eq!(
        challenge.words.len(),
        D / GATHER * challenge.words_per_window
    );
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    // SAFETY: the length check above ties the table to this `D`, and the
    // target features the kernel needs are enabled for this build.
    unsafe {
        x86::accumulate(digit_plane, challenge, acc);
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut signed = SignedPlane {
            plane: *digit_plane,
            complement: [0; D],
            head: [0; WINDOW],
            zero: [0; GATHER],
        };
        for (complement, &digit) in signed.complement.iter_mut().zip(digit_plane) {
            *complement = !digit;
        }
        let windows = acc
            .chunks_exact_mut(WINDOW)
            .zip(challenge.words.chunks_exact(challenge.words_per_window))
            .zip(&challenge.corrections);
        for ((acc_window, words), &correction) in windows {
            let (once, twice) = words.split_at(challenge.once_words);
            #[cfg(target_arch = "aarch64")]
            // SAFETY: `WindowedChallenge::new::<D>` only packs offsets of
            // whole windows inside a `SignedPlane<D>`, and the length check
            // above ties the table to this `D`; `acc_window` is exactly one
            // window; NEON is baseline on aarch64.
            unsafe {
                let digits = (&raw const signed).cast::<i8>();
                let acc_window = acc_window.as_mut_ptr();
                match challenge.span {
                    I8DigitSpan::Full => {
                        neon::gather_window::<false>(digits, once, twice, correction, acc_window);
                    }
                    I8DigitSpan::Half => {
                        neon::gather_window::<true>(digits, once, twice, correction, acc_window);
                    }
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                for (class, scale) in [(once, 1i16), (twice, 2)] {
                    for slot in 0..class.len() * WORD_OFFSETS {
                        let word = class[slot / WORD_OFFSETS];
                        let offset = (word >> (16 * (slot % WORD_OFFSETS))) as u16 as usize;
                        let source = if offset < D {
                            &signed.plane[offset..][..WINDOW]
                        } else if offset < 2 * D {
                            &signed.complement[offset - D..][..WINDOW]
                        } else {
                            &signed.zero[..WINDOW]
                        };
                        for (sum, &digit) in acc_window.iter_mut().zip(source) {
                            *sum += scale * i16::from(digit);
                        }
                    }
                }
                for sum in acc_window {
                    *sum += correction;
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{WINDOW, WORD_OFFSETS};
    use std::arch::aarch64::*;

    /// Low and high halves of one widened window.
    type Halves = (int16x8_t, int16x8_t);

    /// # Safety
    ///
    /// Offset `slot` of `word` must leave [`WINDOW`] readable digits at
    /// `digits + offset`.
    #[inline(always)]
    unsafe fn load(digits: *const i8, word: u64, slot: usize) -> int8x16_t {
        vld1q_s8(digits.add((word >> (16 * slot)) as u16 as usize))
    }

    #[inline(always)]
    unsafe fn widen_add(sum: &mut Halves, window: int8x16_t) {
        sum.0 = vaddw_s8(sum.0, vget_low_s8(window));
        sum.1 = vaddw_high_s8(sum.1, window);
    }

    /// Four independent running sums, so the widening-add chains overlap.
    struct Sums {
        even: [Halves; 2],
        odd: [Halves; 2],
    }

    /// Add the windows of `WORDS` consecutive table words into `sums`. With
    /// `PAIRED`, neighbouring windows are added in `i8` before widening.
    ///
    /// The word count is a constant so the block is straight-line code.
    ///
    /// # Safety
    ///
    /// `words` must address `WORDS` table words, and every offset in them must
    /// leave [`WINDOW`] readable digits at `digits + offset`. With `PAIRED`,
    /// the sum of two addressed digits must fit `i8`.
    #[inline(always)]
    unsafe fn add_words<const PAIRED: bool, const WORDS: usize>(
        digits: *const i8,
        words: *const u64,
        sums: &mut Sums,
    ) {
        for index in 0..WORDS {
            let word = *words.add(index);
            let windows: [int8x16_t; WORD_OFFSETS] = [
                load(digits, word, 0),
                load(digits, word, 1),
                load(digits, word, 2),
                load(digits, word, 3),
            ];
            if PAIRED {
                let target = if index % 2 == 0 {
                    &mut sums.even
                } else {
                    &mut sums.odd
                };
                widen_add(&mut target[0], vaddq_s8(windows[0], windows[1]));
                widen_add(&mut target[1], vaddq_s8(windows[2], windows[3]));
            } else {
                widen_add(&mut sums.even[0], windows[0]);
                widen_add(&mut sums.even[1], windows[1]);
                widen_add(&mut sums.odd[0], windows[2]);
                widen_add(&mut sums.odd[1], windows[3]);
            }
        }
    }

    /// Widened sum of all windows listed in `words`.
    ///
    /// The list is split into power-of-two blocks instead of one counted
    /// loop: its length is the same for every window of a challenge, so each
    /// block test always goes the same way.
    ///
    /// # Safety
    ///
    /// Same contract as [`add_words`], for every word.
    #[inline(always)]
    unsafe fn sum_windows<const PAIRED: bool>(digits: *const i8, words: &[u64]) -> Halves {
        let zero = (vdupq_n_s16(0), vdupq_n_s16(0));
        let mut sums = Sums {
            even: [zero; 2],
            odd: [zero; 2],
        };
        let len = words.len();
        let mut words = words.as_ptr();
        for _ in 0..len / 8 {
            add_words::<PAIRED, 8>(digits, words, &mut sums);
            words = words.add(8);
        }
        if len & 4 != 0 {
            add_words::<PAIRED, 4>(digits, words, &mut sums);
            words = words.add(4);
        }
        if len & 2 != 0 {
            add_words::<PAIRED, 2>(digits, words, &mut sums);
            words = words.add(2);
        }
        if len & 1 != 0 {
            add_words::<PAIRED, 1>(digits, words, &mut sums);
        }
        (
            vaddq_s16(
                vaddq_s16(sums.even[0].0, sums.even[1].0),
                vaddq_s16(sums.odd[0].0, sums.odd[1].0),
            ),
            vaddq_s16(
                vaddq_s16(sums.even[0].1, sums.even[1].1),
                vaddq_s16(sums.odd[0].1, sums.odd[1].1),
            ),
        )
    }

    /// Add one output window's gather to its accumulator window: the windows
    /// listed in `once`, twice the windows listed in `twice`, and
    /// `correction`.
    ///
    /// # Safety
    ///
    /// Same contract as [`add_words`] for every word, and `acc_window` must
    /// address [`WINDOW`] valid i16 values.
    #[inline(always)]
    pub(super) unsafe fn gather_window<const PAIRED: bool>(
        digits: *const i8,
        once: &[u64],
        twice: &[u64],
        correction: i16,
        acc_window: *mut i16,
    ) {
        let (mut low, mut high) = sum_windows::<PAIRED>(digits, once);
        if !twice.is_empty() {
            let (twice_low, twice_high) = sum_windows::<PAIRED>(digits, twice);
            low = vaddq_s16(low, vshlq_n_s16::<1>(twice_low));
            high = vaddq_s16(high, vshlq_n_s16::<1>(twice_high));
        }
        let correction = vdupq_n_s16(correction);
        low = vaddq_s16(low, correction);
        high = vaddq_s16(high, correction);
        vst1q_s16(acc_window, vaddq_s16(vld1q_s16(acc_window), low));
        let acc_high = acc_window.add(WINDOW / 2);
        vst1q_s16(acc_high, vaddq_s16(vld1q_s16(acc_high), high));
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod x86 {
    use super::{WindowedChallenge, GATHER, WINDOW, WORD_OFFSETS};
    use std::arch::x86_64::*;

    /// A digit plane widened to `i16`, its negation and zeros, so one offset
    /// selects the source and its sign, and padding offsets add nothing.
    ///
    /// The offsets are those of [`super::SignedPlane`], counted in
    /// coefficients. Sources are already `i16`, so adding one is a single
    /// vector add with a memory operand, and negated sources need no
    /// correction.
    #[repr(C, align(64))]
    pub(super) struct WidePlane<const D: usize> {
        plane: [i16; D],
        negated: [i16; D],
        head: [i16; WINDOW],
        zero: [i16; GATHER],
    }

    /// [`GATHER`] `i16` lanes.
    #[cfg(target_feature = "avx512bw")]
    type Lanes = __m512i;
    #[cfg(not(target_feature = "avx512bw"))]
    type Lanes = __m256i;

    #[inline(always)]
    unsafe fn zero() -> Lanes {
        #[cfg(target_feature = "avx512bw")]
        return _mm512_setzero_si512();
        #[cfg(not(target_feature = "avx512bw"))]
        return _mm256_setzero_si256();
    }

    /// # Safety
    ///
    /// `source` must address [`GATHER`] readable `i16` values.
    #[inline(always)]
    unsafe fn load(source: *const i16) -> Lanes {
        #[cfg(target_feature = "avx512bw")]
        return _mm512_loadu_si512(source.cast());
        #[cfg(not(target_feature = "avx512bw"))]
        return _mm256_loadu_si256(source.cast());
    }

    #[inline(always)]
    unsafe fn add(left: Lanes, right: Lanes) -> Lanes {
        #[cfg(target_feature = "avx512bw")]
        return _mm512_add_epi16(left, right);
        #[cfg(not(target_feature = "avx512bw"))]
        return _mm256_add_epi16(left, right);
    }

    /// # Safety
    ///
    /// `target` must address [`GATHER`] writable `i16` values.
    #[inline(always)]
    unsafe fn store(target: *mut i16, lanes: Lanes) {
        #[cfg(target_feature = "avx512bw")]
        _mm512_storeu_si512(target.cast(), lanes);
        #[cfg(not(target_feature = "avx512bw"))]
        _mm256_storeu_si256(target.cast(), lanes);
    }

    /// Sum of all sources listed in `words`, in four independent running
    /// sums so the add chains overlap.
    ///
    /// # Safety
    ///
    /// Every offset in `words` must leave [`GATHER`] readable values at
    /// `sources + offset`.
    #[inline(always)]
    unsafe fn sum_sources(sources: *const i16, words: &[u64]) -> Lanes {
        let mut sums = [zero(); WORD_OFFSETS];
        for &word in words {
            for (slot, sum) in sums.iter_mut().enumerate() {
                let offset = (word >> (16 * slot)) as u16 as usize;
                *sum = add(*sum, load(sources.add(offset)));
            }
        }
        add(add(sums[0], sums[1]), add(sums[2], sums[3]))
    }

    /// Add `challenge * digit_plane` to `acc`.
    ///
    /// # Safety
    ///
    /// `challenge` must hold the table `WindowedChallenge::new::<D>` built,
    /// and the build must enable AVX2 (and AVX-512BW when [`GATHER`] is two
    /// windows).
    #[inline(always)]
    pub(super) unsafe fn accumulate<const D: usize>(
        digit_plane: &[i8; D],
        challenge: &WindowedChallenge,
        acc: &mut [i16; D],
    ) {
        let mut wide = WidePlane {
            plane: [0i16; D],
            negated: [0i16; D],
            head: [0i16; WINDOW],
            zero: [0i16; GATHER],
        };
        let widened = wide.plane.iter_mut().zip(&mut wide.negated);
        for ((plane, negated), &digit) in widened.zip(digit_plane) {
            *plane = i16::from(digit);
            *negated = -i16::from(digit);
        }
        wide.head.copy_from_slice(&wide.plane[..WINDOW]);
        let sources = (&raw const wide).cast::<i16>();
        let gathers = acc
            .chunks_exact_mut(GATHER)
            .zip(challenge.words.chunks_exact(challenge.words_per_window));
        for (acc_lanes, words) in gathers {
            let (once, twice) = words.split_at(challenge.once_words);
            let mut sum = sum_sources(sources, once);
            if !twice.is_empty() {
                let twice = sum_sources(sources, twice);
                sum = add(sum, add(twice, twice));
            }
            let acc_lanes = acc_lanes.as_mut_ptr();
            store(acc_lanes, add(load(acc_lanes), sum));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sources::poly_helpers::narrow_accum::sparse_mul_acc_terms;

    fn next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state >> 33
    }

    fn check<const D: usize>(
        stride: usize,
        num_terms: usize,
        coefficients: &[i8],
        span: I8DigitSpan,
        seed: u64,
    ) {
        let mut state = seed;
        let mut slots: Vec<u32> = (0..(D / stride) as u32).collect();
        let mut positions = Vec::new();
        let mut coeffs = Vec::new();
        for _ in 0..num_terms {
            let pick = next(&mut state) as usize % slots.len();
            positions.push(slots.swap_remove(pick) * stride as u32);
            coeffs.push(coefficients[next(&mut state) as usize % coefficients.len()]);
        }
        // Mostly extreme digits, so paired and widened sums meet both ends of
        // the span.
        let (min, max) = match span {
            I8DigitSpan::Full => (i8::MIN, i8::MAX),
            I8DigitSpan::Half => (-64, 63),
        };
        let mut plane = [0i8; D];
        for digit in &mut plane {
            *digit = match next(&mut state) % 4 {
                0 => min,
                1 => max,
                _ => (i16::from(min) + (next(&mut state) % 2 * (max as u64 + 1)) as i16) as i8,
            };
        }

        let mut expected = [7i16; D];
        sparse_mul_acc_terms(&plane, &positions, &coeffs, &mut expected);

        let challenge = SparseChallenge {
            positions: positions.into(),
            coeffs: coeffs.into(),
        };
        let windowed = WindowedChallenge::new::<D>(&challenge, span).unwrap();
        let mut actual = [7i16; D];
        windowed_mul_acc(&plane, &windowed, &mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn windowed_gather_matches_term_kernel() {
        for span in [I8DigitSpan::Full, I8DigitSpan::Half] {
            for seed in 0..8 {
                check::<64>(16, 4, &[1, -1, 2, -2], span, seed);
                check::<256>(16, 13, &[1, -1, 2, -2], span, seed);
                check::<1024>(16, 41, &[1, -1, 2, -2], span, seed);
                check::<2048>(32, 41, &[1, -1, 2, -2], span, seed);
                check::<2048>(16, 31, &[1, -1, 2, -2], span, seed);
            }
        }
    }

    /// One coefficient magnitude puts every term in one list, so 32, 33 and
    /// 64 terms give one eight-word block, a block plus a tail, and two
    /// blocks.
    #[test]
    fn windowed_gather_matches_term_kernel_on_long_lists() {
        for span in [I8DigitSpan::Full, I8DigitSpan::Half] {
            for coefficients in [&[1i8, -1], &[2, -2]] {
                for num_terms in [32, 33, 64] {
                    for seed in 0..4 {
                        check::<1024>(16, num_terms, coefficients, span, seed);
                        check::<2048>(16, num_terms, coefficients, span, seed);
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_challenges_off_the_window_grid() {
        let off_grid = SparseChallenge {
            positions: vec![0, 8].into(),
            coeffs: vec![1, -1].into(),
        };
        assert!(WindowedChallenge::new::<64>(&off_grid, I8DigitSpan::Full).is_none());
        let wide_coefficient = SparseChallenge {
            positions: vec![0, 16].into(),
            coeffs: vec![1, 3].into(),
        };
        assert!(WindowedChallenge::new::<64>(&wide_coefficient, I8DigitSpan::Full).is_none());
        let out_of_ring = SparseChallenge {
            positions: vec![64].into(),
            coeffs: vec![1].into(),
        };
        assert!(WindowedChallenge::new::<64>(&out_of_ring, I8DigitSpan::Full).is_none());
    }
}
