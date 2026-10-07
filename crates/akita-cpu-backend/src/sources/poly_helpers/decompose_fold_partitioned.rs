//! Partitioned decompose-fold accumulation (element- and position-partitioned).

use super::narrow_accum::{
    sparse_mul_acc as sparse_mul_acc_narrow, sparse_mul_acc_i16 as sparse_mul_acc_i16_narrow,
    sparse_mul_acc_i16_terms as sparse_mul_acc_i16_narrow_terms,
    sparse_mul_acc_terms as sparse_mul_acc_narrow_terms,
};
use super::rotated_accum::{accumulate_rotated_digit_plane, should_use_rotated_challenge};
use super::windowed_accum::{gathers_windows, windowed_mul_acc, I8DigitSpan, WindowedChallenge};
use super::{
    fill_rotated_challenge, sparse_mul_acc, sparse_mul_acc_i16, sparse_mul_acc_i16_pm1,
    sparse_mul_acc_pm1, SignedDigitBasis, ValidatedSparseChallenge, ValidatedSparseChallenges,
};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_error::{checked, AkitaError};
use akita_params::SignedDigitKernel;
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field};
use std::ops::Range;

use crate::sources::packed_digits::PackedSignedDigitView;

struct PreparedPm1Challenge {
    positive: Vec<u32>,
    negative: Vec<u32>,
}

enum WidePlan<const D: usize> {
    Rotated(Box<[[i16; D]; D]>),
    Pm1(PreparedPm1Challenge),
    Generic,
}

enum ChallengePlan<const D: usize> {
    Wide(WidePlan<D>),
    /// One unchunked pass; the gather lists are present when the window
    /// kernel serves the challenge.
    NarrowFull(u64, Option<WindowedChallenge>),
    NarrowChunked(Vec<Range<usize>>),
}

struct PreparedChallenge<'a, const D: usize> {
    challenge: ValidatedSparseChallenge<'a, D>,
    plan: ChallengePlan<D>,
    block_start: usize,
}

fn prepare_wide_plan<const D: usize>(challenge: &SparseChallenge) -> ChallengePlan<D> {
    let mut positive = Vec::with_capacity(challenge.positions.len());
    let mut negative = Vec::with_capacity(challenge.positions.len());
    for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
        match coefficient {
            1 => positive.push(position),
            -1 => negative.push(position),
            _ => return ChallengePlan::Wide(WidePlan::Generic),
        }
    }
    ChallengePlan::Wide(WidePlan::Pm1(PreparedPm1Challenge { positive, negative }))
}

fn prepare_challenge<const D: usize>(
    digit_abs_bound: u64,
    window_span: Option<I8DigitSpan>,
    challenge: &SparseChallenge,
) -> ChallengePlan<D> {
    if should_use_rotated_challenge::<D>(challenge) {
        let mut rotated = Box::new([[0i16; D]; D]);
        fill_rotated_challenge::<D>(rotated.as_mut(), challenge);
        return ChallengePlan::Wide(WidePlan::Rotated(rotated));
    }
    if digit_abs_bound == 0 {
        return ChallengePlan::NarrowFull(0, None);
    }

    let max_chunk_mass = i16::MAX as u64 / digit_abs_bound;
    let mut total_mass = 0u64;
    for coefficient in &challenge.coeffs {
        let term_mass = u64::from(coefficient.unsigned_abs());
        if term_mass > max_chunk_mass {
            return prepare_wide_plan(challenge);
        }
        let Some(next_total_mass) = total_mass.checked_add(term_mass) else {
            return prepare_wide_plan(challenge);
        };
        total_mass = next_total_mass;
    }

    let Some(contribution_bound) = digit_abs_bound.checked_mul(total_mass) else {
        return prepare_wide_plan(challenge);
    };
    if contribution_bound <= i16::MAX as u64 {
        let windowed = window_span.and_then(|span| WindowedChallenge::new::<D>(challenge, span));
        return ChallengePlan::NarrowFull(contribution_bound, windowed);
    }

    let mut chunk_mass = 0u64;
    let mut chunk_start = 0usize;
    let mut term_ranges = Vec::new();
    for (term_idx, coefficient) in challenge.coeffs.iter().enumerate() {
        let term_mass = u64::from(coefficient.unsigned_abs());
        if chunk_mass + term_mass > max_chunk_mass {
            term_ranges.push(chunk_start..term_idx);
            chunk_start = term_idx;
            chunk_mass = 0;
        }
        chunk_mass += term_mass;
    }
    if chunk_start < challenge.coeffs.len() {
        term_ranges.push(chunk_start..challenge.coeffs.len());
    }
    ChallengePlan::NarrowChunked(term_ranges)
}

fn partition_thread_count(num_positions_per_block: usize) -> usize {
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;
    num_threads.min(num_positions_per_block.max(1)).max(1)
}

/// Span of stored `i8` digits known only by their absolute bound.
fn bounded_i8_digit_span(digit_abs_bound: u64) -> I8DigitSpan {
    if digit_abs_bound <= 63 {
        I8DigitSpan::Half
    } else {
        I8DigitSpan::Full
    }
}

fn position_tile_len(num_positions_per_block: usize) -> usize {
    let actual_threads = partition_thread_count(num_positions_per_block);
    if actual_threads <= 8 {
        return num_positions_per_block.div_ceil(actual_threads).max(1);
    }

    let tiles_per_thread = actual_threads.div_ceil(4).clamp(1, 4);
    num_positions_per_block
        .div_ceil(actual_threads.saturating_mul(tiles_per_thread))
        .max(1)
}

/// Source preparation is monomorphized per witness representation. The
/// associated scratch type makes the source/scratch pairing a type invariant.
trait FoldSource<const D: usize>: Sync {
    type Scratch;
    type Planes<'a>: DigitPlaneSet<D>
    where
        Self: 'a;

    fn num_rings(&self) -> usize;
    fn digit_abs_bound(&self) -> u64;
    /// Value range of the digit planes when all of them are `i8`, the storage
    /// the window-gathered narrow kernel reads.
    fn i8_digit_span(&self) -> Option<I8DigitSpan>;
    fn scratch(&self, num_digits: usize) -> Self::Scratch;
    fn digit_planes<'a>(
        &'a self,
        ring_idx: usize,
        num_digits: usize,
        scratch: &'a mut Self::Scratch,
    ) -> Self::Planes<'a>;
}

struct CachedDigits<'a, const D: usize> {
    digit_planes: &'a [[i8; D]],
    num_rings: usize,
    digit_abs_bound: u64,
}

impl<const D: usize> FoldSource<D> for CachedDigits<'_, D> {
    type Scratch = ();
    type Planes<'a>
        = DigitPlanes<'a, D>
    where
        Self: 'a;

    fn num_rings(&self) -> usize {
        self.num_rings
    }

    fn digit_abs_bound(&self) -> u64 {
        self.digit_abs_bound
    }

    fn i8_digit_span(&self) -> Option<I8DigitSpan> {
        Some(bounded_i8_digit_span(self.digit_abs_bound))
    }

    fn scratch(&self, _num_digits: usize) {}

    #[inline]
    fn digit_planes<'a>(
        &'a self,
        ring_idx: usize,
        num_digits: usize,
        _scratch: &'a mut (),
    ) -> Self::Planes<'a> {
        let start = ring_idx * num_digits;
        DigitPlanes::I8(&self.digit_planes[start..start + num_digits])
    }
}

struct LiveRings<'a, F: Field + CanonicalEncoding, const D: usize> {
    coeffs: &'a [CyclotomicRing<F, D>],
    params: &'a BalancedDecomposePow2Params<F>,
    basis: SignedDigitBasis,
}

enum LiveDigitScratch<const D: usize> {
    I8(Vec<[i8; D]>),
    I16(Vec<[i16; D]>),
}

impl<F: Field + CanonicalEncoding, const D: usize> FoldSource<D> for LiveRings<'_, F, D> {
    type Scratch = LiveDigitScratch<D>;
    type Planes<'a>
        = DigitPlanes<'a, D>
    where
        Self: 'a;

    fn num_rings(&self) -> usize {
        self.coeffs.len()
    }

    fn digit_abs_bound(&self) -> u64 {
        self.basis.abs_bound
    }

    fn i8_digit_span(&self) -> Option<I8DigitSpan> {
        // Balanced base-`2^k` digits are `[-2^(k-1), 2^(k-1) - 1]`, one short
        // of the absolute bound on the positive side.
        matches!(self.basis.kernel, SignedDigitKernel::I8).then(|| {
            if self.basis.abs_bound <= 64 {
                I8DigitSpan::Half
            } else {
                I8DigitSpan::Full
            }
        })
    }

    fn scratch(&self, num_digits: usize) -> Self::Scratch {
        match self.basis.kernel {
            SignedDigitKernel::I8 => LiveDigitScratch::I8(vec![[0i8; D]; num_digits]),
            SignedDigitKernel::I16 => LiveDigitScratch::I16(vec![[0i16; D]; num_digits]),
        }
    }

    #[inline]
    fn digit_planes<'a>(
        &'a self,
        ring_idx: usize,
        _num_digits: usize,
        scratch: &'a mut Self::Scratch,
    ) -> Self::Planes<'a> {
        match scratch {
            LiveDigitScratch::I8(planes) => {
                self.coeffs[ring_idx]
                    .balanced_decompose_pow2_i8_into_with_params(planes, self.params);
                DigitPlanes::I8(planes)
            }
            LiveDigitScratch::I16(planes) => {
                self.coeffs[ring_idx].balanced_decompose_pow2_i16_into(planes, self.params);
                DigitPlanes::I16(planes)
            }
        }
    }
}

struct PackedDigits<'a> {
    digits: PackedSignedDigitView<'a>,
    num_rings: usize,
    digit_abs_bound: u64,
}

impl<const D: usize> FoldSource<D> for PackedDigits<'_> {
    type Scratch = [i8; D];
    type Planes<'a>
        = &'a [i8; D]
    where
        Self: 'a;

    fn num_rings(&self) -> usize {
        self.num_rings
    }

    fn digit_abs_bound(&self) -> u64 {
        self.digit_abs_bound
    }

    fn i8_digit_span(&self) -> Option<I8DigitSpan> {
        Some(bounded_i8_digit_span(self.digit_abs_bound))
    }

    fn scratch(&self, _num_digits: usize) -> Self::Scratch {
        [0i8; D]
    }

    #[inline]
    fn digit_planes<'a>(
        &'a self,
        ring_idx: usize,
        num_digits: usize,
        scratch: &'a mut Self::Scratch,
    ) -> Self::Planes<'a> {
        debug_assert_eq!(num_digits, 1);
        self.digits
            .decode_range(ring_idx * D, scratch)
            .expect("validated packed recursive ring");
        scratch
    }
}

/// A ring's digit planes, independent of how the source supplied them.
enum DigitPlanes<'a, const D: usize> {
    I8(&'a [[i8; D]]),
    I16(&'a [[i16; D]]),
}

trait DigitPlaneSet<const D: usize> {
    fn accumulate_wide(
        self,
        acc: &mut [[i32; D]],
        challenge: ValidatedSparseChallenge<'_, D>,
        plan: &WidePlan<D>,
    );
    fn accumulate_narrow(
        self,
        acc: &mut [[i16; D]],
        challenge: &SparseChallenge,
        windowed: Option<&WindowedChallenge>,
    );
    fn accumulate_chunked_narrow(
        self,
        narrow: &mut [[i16; D]],
        wide: &mut [[i32; D]],
        challenge: &SparseChallenge,
        term_ranges: &[Range<usize>],
    );
}

impl<const D: usize> DigitPlaneSet<D> for &[i8; D] {
    #[inline]
    fn accumulate_wide(
        self,
        acc: &mut [[i32; D]],
        challenge: ValidatedSparseChallenge<'_, D>,
        plan: &WidePlan<D>,
    ) {
        match plan {
            WidePlan::Rotated(rotated) => {
                accumulate_rotated_digit_plane(self, rotated.as_ref(), &mut acc[0])
            }
            WidePlan::Pm1(pm1) => {
                sparse_mul_acc_pm1(self, &pm1.positive, &pm1.negative, &mut acc[0])
            }
            WidePlan::Generic => sparse_mul_acc(self, challenge, &mut acc[0]),
        }
    }

    #[inline]
    fn accumulate_narrow(
        self,
        acc: &mut [[i16; D]],
        challenge: &SparseChallenge,
        windowed: Option<&WindowedChallenge>,
    ) {
        match windowed {
            Some(windowed) => windowed_mul_acc(self, windowed, &mut acc[0]),
            None => sparse_mul_acc_narrow(self, challenge, &mut acc[0]),
        }
    }

    #[inline]
    fn accumulate_chunked_narrow(
        self,
        narrow: &mut [[i16; D]],
        wide: &mut [[i32; D]],
        challenge: &SparseChallenge,
        term_ranges: &[Range<usize>],
    ) {
        for term_range in term_ranges {
            sparse_mul_acc_narrow_terms(
                self,
                &challenge.positions[term_range.clone()],
                &challenge.coeffs[term_range.clone()],
                &mut narrow[0],
            );
            flush_narrow_accumulator(narrow, wide);
        }
    }
}

impl<const D: usize> DigitPlaneSet<D> for DigitPlanes<'_, D> {
    fn accumulate_wide(
        self,
        acc: &mut [[i32; D]],
        challenge: ValidatedSparseChallenge<'_, D>,
        plan: &WidePlan<D>,
    ) {
        match self {
            Self::I8(planes) => {
                for (plane, dst) in planes.iter().zip(acc) {
                    match plan {
                        WidePlan::Rotated(rotated) => {
                            accumulate_rotated_digit_plane(plane, rotated.as_ref(), dst)
                        }
                        WidePlan::Pm1(pm1) => {
                            sparse_mul_acc_pm1(plane, &pm1.positive, &pm1.negative, dst)
                        }
                        WidePlan::Generic => sparse_mul_acc(plane, challenge, dst),
                    }
                }
            }
            Self::I16(planes) => {
                for (plane, dst) in planes.iter().zip(acc) {
                    match plan {
                        WidePlan::Rotated(rotated) => {
                            accumulate_rotated_digit_plane(plane, rotated.as_ref(), dst)
                        }
                        WidePlan::Pm1(pm1) => {
                            sparse_mul_acc_i16_pm1(plane, &pm1.positive, &pm1.negative, dst)
                        }
                        WidePlan::Generic => sparse_mul_acc_i16(plane, challenge, dst),
                    }
                }
            }
        }
    }

    #[inline(always)]
    fn accumulate_narrow(
        self,
        acc: &mut [[i16; D]],
        challenge: &SparseChallenge,
        windowed: Option<&WindowedChallenge>,
    ) {
        match self {
            Self::I8(planes) => match windowed {
                Some(windowed) => {
                    for (plane, dst) in planes.iter().zip(acc) {
                        windowed_mul_acc(plane, windowed, dst);
                    }
                }
                None => {
                    for (plane, dst) in planes.iter().zip(acc) {
                        sparse_mul_acc_narrow(plane, challenge, dst);
                    }
                }
            },
            Self::I16(planes) => {
                for (plane, dst) in planes.iter().zip(acc) {
                    sparse_mul_acc_i16_narrow(plane, challenge, dst);
                }
            }
        }
    }

    fn accumulate_chunked_narrow(
        self,
        narrow: &mut [[i16; D]],
        wide: &mut [[i32; D]],
        challenge: &SparseChallenge,
        term_ranges: &[Range<usize>],
    ) {
        for term_range in term_ranges {
            let positions = &challenge.positions[term_range.clone()];
            let coefficients = &challenge.coeffs[term_range.clone()];
            match self {
                Self::I8(planes) => {
                    for (plane, dst) in planes.iter().zip(narrow.iter_mut()) {
                        sparse_mul_acc_narrow_terms(plane, positions, coefficients, dst);
                    }
                }
                Self::I16(planes) => {
                    for (plane, dst) in planes.iter().zip(narrow.iter_mut()) {
                        sparse_mul_acc_i16_narrow_terms(plane, positions, coefficients, dst);
                    }
                }
            }
            flush_narrow_accumulator(narrow, wide);
        }
    }
}

#[inline]
fn flush_narrow_accumulator<const D: usize>(narrow: &mut [[i16; D]], wide: &mut [[i32; D]]) {
    debug_assert_eq!(narrow.len(), wide.len());
    for (narrow_ring, wide_ring) in narrow.iter_mut().zip(wide) {
        for (narrow_coeff, wide_coeff) in narrow_ring.iter_mut().zip(wide_ring) {
            *wide_coeff += i32::from(*narrow_coeff);
            *narrow_coeff = 0;
        }
    }
}

/// Wide accumulation is total for every challenge; narrow accumulation owns
/// its scratch and flush bound together. Narrow plans are optimization hints,
/// so a wide accumulator can also execute them through the generic wide kernel.
enum FoldAccumulator<'a, const D: usize> {
    Wide(&'a mut [[i32; D]]),
    Narrow {
        wide: &'a mut [[i32; D]],
        narrow: Vec<[i16; D]>,
        bound: u64,
    },
}

impl<const D: usize> FoldAccumulator<'_, D> {
    fn accumulate<S: FoldSource<D>>(
        &mut self,
        source: &S,
        scratch: &mut S::Scratch,
        ring_range: Range<usize>,
        num_digits: usize,
        prepared: &PreparedChallenge<'_, D>,
    ) {
        let challenge = prepared.challenge.challenge();
        match (&mut *self, &prepared.plan) {
            (
                Self::Narrow {
                    wide,
                    narrow,
                    bound,
                },
                ChallengePlan::NarrowFull(contribution, windowed),
            ) => {
                if *bound + contribution > i16::MAX as u64 {
                    flush_narrow_accumulator(narrow, wide);
                    *bound = 0;
                }
                for (local, ring) in ring_range.enumerate() {
                    let planes = source.digit_planes(ring, num_digits, scratch);
                    let base = local * num_digits;
                    planes.accumulate_narrow(
                        &mut narrow[base..base + num_digits],
                        challenge,
                        windowed.as_ref(),
                    );
                }
                *bound += contribution;
            }
            (
                Self::Narrow {
                    wide,
                    narrow,
                    bound,
                },
                ChallengePlan::NarrowChunked(ranges),
            ) => {
                if *bound != 0 {
                    flush_narrow_accumulator(narrow, wide);
                    *bound = 0;
                }
                for (local, ring) in ring_range.enumerate() {
                    let planes = source.digit_planes(ring, num_digits, scratch);
                    let base = local * num_digits;
                    planes.accumulate_chunked_narrow(
                        &mut narrow[base..base + num_digits],
                        &mut wide[base..base + num_digits],
                        challenge,
                        ranges,
                    );
                }
            }
            (accumulator, plan) => {
                let wide = match accumulator {
                    Self::Wide(wide) => wide,
                    Self::Narrow {
                        wide,
                        narrow,
                        bound,
                    } => {
                        if *bound != 0 {
                            flush_narrow_accumulator(narrow, wide);
                            *bound = 0;
                        }
                        wide
                    }
                };
                let plan = match plan {
                    ChallengePlan::Wide(plan) => plan,
                    ChallengePlan::NarrowFull(..) | ChallengePlan::NarrowChunked(_) => {
                        &WidePlan::Generic
                    }
                };
                for (local, ring) in ring_range.enumerate() {
                    let planes = source.digit_planes(ring, num_digits, scratch);
                    let base = local * num_digits;
                    planes.accumulate_wide(
                        &mut wide[base..base + num_digits],
                        prepared.challenge,
                        plan,
                    );
                }
            }
        }
    }

    fn finish(self) {
        if let Self::Narrow {
            wide,
            mut narrow,
            bound,
        } = self
        {
            if bound != 0 {
                flush_narrow_accumulator(&mut narrow, wide);
            }
        }
    }
}

fn element_partitioned_decompose_fold<S: FoldSource<D>, const D: usize>(
    source: S,
    challenges: &[SparseChallenge],
    num_positions_per_block: usize,
    num_digits: usize,
) -> Result<Vec<[i32; D]>, AkitaError> {
    let inner_width = checked::product([num_positions_per_block, num_digits])
        .ok_or_else(|| AkitaError::InvalidInput("partitioned fold inner width overflow".into()))?;
    if inner_width == 0 || num_digits == 0 {
        return Ok(Vec::new());
    }

    let digit_abs_bound = source.digit_abs_bound();
    let window_span = source.i8_digit_span().filter(|_| gathers_windows());
    let challenges = ValidatedSparseChallenges::<D>::new(
        challenges,
        source.num_rings(),
        num_positions_per_block,
    )?;
    let plans = challenges
        .iter()
        .enumerate()
        .map(|(block_idx, challenge)| PreparedChallenge {
            challenge,
            plan: prepare_challenge::<D>(digit_abs_bound, window_span, challenge.challenge()),
            block_start: block_idx * num_positions_per_block,
        })
        .collect::<Vec<_>>();
    let uses_narrow_accumulation = plans.iter().any(|prepared| {
        matches!(
            prepared.plan,
            ChallengePlan::NarrowFull(..) | ChallengePlan::NarrowChunked(_)
        )
    });
    let position_tile = position_tile_len(num_positions_per_block);
    let mut out = vec![[0i32; D]; inner_width];

    cfg_chunks_mut!(out, position_tile * num_digits)
        .enumerate()
        .for_each(|(tile_idx, acc)| {
            let elem_start = tile_idx * position_tile;
            if elem_start >= num_positions_per_block {
                return;
            }
            let elems_in_chunk = acc.len() / num_digits;
            let elem_end = elem_start + elems_in_chunk;
            let mut digit_scratch = source.scratch(num_digits);
            let mut accumulator = if uses_narrow_accumulation {
                let narrow = vec![[0i16; D]; acc.len()];
                FoldAccumulator::Narrow {
                    wide: acc,
                    narrow,
                    bound: 0,
                }
            } else {
                FoldAccumulator::Wide(acc)
            };

            for prepared in &plans {
                let block_start = prepared.block_start;
                let ring_start = block_start + elem_start;
                if ring_start >= source.num_rings() {
                    continue;
                }
                let ring_end = (block_start + elem_end).min(source.num_rings());
                accumulator.accumulate(
                    &source,
                    &mut digit_scratch,
                    ring_start..ring_end,
                    num_digits,
                    prepared,
                );
            }
            accumulator.finish();
        });

    Ok(out)
}

/// Element-partitioned accumulation for predecomposed dense digit caches.
pub(crate) fn cached_digit_decompose_fold_partitioned<const D: usize>(
    digit_planes: &[[i8; D]],
    challenges: &[SparseChallenge],
    num_positions_per_block: usize,
    num_digits: usize,
    basis: SignedDigitBasis,
) -> Result<Vec<[i32; D]>, AkitaError> {
    let num_rings = digit_planes.len().checked_div(num_digits).ok_or_else(|| {
        AkitaError::InvalidInput("cached fold digit count must be nonzero".into())
    })?;
    let digit_abs_bound = basis.abs_bound.min(u64::from(i8::MIN.unsigned_abs()));
    element_partitioned_decompose_fold(
        CachedDigits {
            digit_planes,
            num_rings,
            digit_abs_bound,
        },
        challenges,
        num_positions_per_block,
        num_digits,
    )
}

/// Element-partitioned accumulation for multi-digit dense witnesses.
pub fn balanced_ring_decompose_fold_partitioned<F: Field + CanonicalEncoding, const D: usize>(
    coeffs: &[CyclotomicRing<F, D>],
    challenges: &[SparseChallenge],
    num_positions_per_block: usize,
    params: &BalancedDecomposePow2Params<F>,
) -> Result<Vec<[i32; D]>, AkitaError> {
    let basis = SignedDigitBasis::new(params.log_basis())?;
    element_partitioned_decompose_fold(
        LiveRings {
            coeffs,
            params,
            basis,
        },
        challenges,
        num_positions_per_block,
        params.levels(),
    )
}

/// Position-partitioned accumulation over a packed tight recursive witness.
pub(crate) fn packed_tight_digit_fold_partitioned<const D: usize>(
    digits: PackedSignedDigitView<'_>,
    num_rings: usize,
    challenges: &[SparseChallenge],
    num_positions_per_block: usize,
) -> Result<Vec<[i32; D]>, AkitaError> {
    let digit_abs_bound = u64::from(
        digits
            .bounds()
            .negative_abs_max()
            .max(digits.bounds().positive_max()),
    );
    element_partitioned_decompose_fold(
        PackedDigits {
            digits,
            num_rings,
            digit_abs_bound,
        },
        challenges,
        num_positions_per_block,
        1,
    )
}
