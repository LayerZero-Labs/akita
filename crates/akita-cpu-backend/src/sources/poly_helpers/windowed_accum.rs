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

use akita_challenges::SparseChallenge;

/// Coefficients per gathered window: one 128-bit vector of `i8` digits.
pub(super) const WINDOW: usize = 16;

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

/// A digit plane, its bitwise complement and one zero window, so one offset
/// selects the source window and its sign, and padding offsets add nothing.
#[repr(C, align(16))]
struct SignedPlane<const D: usize> {
    plane: [i8; D],
    complement: [i8; D],
    zero: [i8; WINDOW],
}

/// Per-window source lists for one challenge.
///
/// Every output window owns `words_per_window` table words: first
/// `once_words` words listing the terms with coefficient magnitude one, then
/// the words listing the terms with magnitude two. Each word packs
/// [`WORD_OFFSETS`] [`SignedPlane`] offsets; unused slots point at the zero
/// window. `corrections[window]` is the weighted number of complemented
/// sources in that window's list.
pub(super) struct WindowedChallenge {
    words: Vec<u64>,
    corrections: Vec<i16>,
    once_words: usize,
    words_per_window: usize,
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
        if !D.is_multiple_of(WINDOW)
            || 2 * D + WINDOW > usize::from(u16::MAX) + 1
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
        let mut words = Vec::with_capacity(num_windows * words_per_window);
        let mut corrections = Vec::with_capacity(num_windows);
        let mut offsets = Vec::with_capacity(num_terms);
        for window in 0..num_windows {
            let mut correction = 0i16;
            for magnitude in [1u8, 2] {
                offsets.clear();
                for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
                    if coefficient.unsigned_abs() != magnitude {
                        continue;
                    }
                    let shift = position as usize / WINDOW;
                    // X^position moves source window `window - shift` here;
                    // the windows that wrap past X^D come back negated.
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
                        let offset = group.get(slot).copied().unwrap_or(2 * D as u64);
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
/// Only the NEON implementation has been measured to beat the term-at-a-time
/// kernel, so other targets keep the existing narrow path.
pub(super) fn gathers_windows() -> bool {
    cfg!(target_arch = "aarch64") && super::use_simd_decompose_fold()
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
    assert!(D.is_multiple_of(WINDOW));
    assert_eq!(
        challenge.words.len(),
        D / WINDOW * challenge.words_per_window
    );
    let mut signed = SignedPlane {
        plane: *digit_plane,
        complement: [0; D],
        zero: [0; WINDOW],
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
        // SAFETY: `WindowedChallenge::new::<D>` only packs offsets of whole
        // windows inside a `SignedPlane<D>`, and the length check above ties
        // the table to this `D`; `acc_window` is exactly one window; NEON is
        // baseline on aarch64.
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
                        &signed.zero
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

    fn check<const D: usize>(stride: usize, num_terms: usize, span: I8DigitSpan, seed: u64) {
        let mut state = seed;
        let mut slots: Vec<u32> = (0..(D / stride) as u32).collect();
        let mut positions = Vec::new();
        let mut coeffs = Vec::new();
        for _ in 0..num_terms {
            let pick = next(&mut state) as usize % slots.len();
            positions.push(slots.swap_remove(pick) * stride as u32);
            coeffs.push([1i8, -1, 2, -2][next(&mut state) as usize % 4]);
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
                check::<64>(16, 4, span, seed);
                check::<256>(16, 13, span, seed);
                check::<1024>(16, 41, span, seed);
                check::<2048>(32, 41, span, seed);
                check::<2048>(16, 31, span, seed);
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
