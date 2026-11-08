//! Partitioned decompose-fold accumulation (element- and position-partitioned).

use super::narrow_accum::{
    sparse_mul_acc as sparse_mul_acc_narrow, sparse_mul_acc_i16 as sparse_mul_acc_i16_narrow,
    sparse_mul_acc_i16_terms as sparse_mul_acc_i16_narrow_terms,
    sparse_mul_acc_terms as sparse_mul_acc_narrow_terms,
};
use super::rotated_accum::{accumulate_rotated_digit_plane, should_use_rotated_challenge};
use super::shifted_accum::{
    shifted_mul_acc, shifted_mul_acc_i16, shifts_windows, ShiftScratch, ShiftedChallenge,
};
use super::windowed_accum::{gathers_windows, windowed_mul_acc, I8DigitSpan, WindowedChallenge};
use super::{
    fill_rotated_challenge, sparse_mul_acc, sparse_mul_acc_i16, sparse_mul_acc_i16_pm1,
    sparse_mul_acc_pm1, SignedDigitBasis, ValidatedSparseChallenge, ValidatedSparseChallenges,
};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_error::AkitaError;
use jolt_algebra::checked;
use jolt_algebra::solinas::parallel::*;
use jolt_algebra::{CanonicalEncoding, Field};
use std::ops::Range;

use crate::sources::block_digits::{
    BlockDigits, DigitPlaneSet, DigitPlanes, DigitShape, LiveRings,
};

pub(crate) struct PreparedPm1Challenge {
    positive: Vec<u32>,
    negative: Vec<u32>,
}

pub(crate) enum WidePlan<const D: usize> {
    Rotated(Box<[[i16; D]; D]>),
    Pm1(PreparedPm1Challenge),
    Generic,
}

/// How one unchunked narrow pass multiplies a digit plane by its challenge.
pub(crate) enum NarrowKernel {
    /// Whole-window gathers for subring-embedded challenges.
    Windowed(WindowedChallenge),
    /// Register-blocked windows for `±1`/`±2` challenges at any positions.
    Shifted(ShiftedChallenge),
    /// One accumulator pass per challenge term.
    Terms,
}

enum ChallengePlan<const D: usize> {
    Wide(WidePlan<D>),
    /// One unchunked pass with the kernel that serves the challenge.
    NarrowFull(u64, NarrowKernel),
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
    shifts: bool,
    challenge: &SparseChallenge,
) -> ChallengePlan<D> {
    if should_use_rotated_challenge::<D>(challenge) {
        let mut rotated = Box::new([[0i16; D]; D]);
        fill_rotated_challenge::<D>(rotated.as_mut(), challenge);
        return ChallengePlan::Wide(WidePlan::Rotated(rotated));
    }
    if digit_abs_bound == 0 {
        return ChallengePlan::NarrowFull(0, NarrowKernel::Terms);
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
        let kernel = window_span
            .and_then(|span| WindowedChallenge::new::<D>(challenge, span))
            .map(NarrowKernel::Windowed)
            .or_else(|| {
                shifts
                    .then(|| ShiftedChallenge::new::<D>(challenge, digit_abs_bound))
                    .flatten()
                    .map(NarrowKernel::Shifted)
            })
            .unwrap_or(NarrowKernel::Terms);
        return ChallengePlan::NarrowFull(contribution_bound, kernel);
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
        kernel: &NarrowKernel,
        shift: &mut ShiftScratch<D>,
    ) {
        match kernel {
            NarrowKernel::Windowed(windowed) => windowed_mul_acc(self, windowed, &mut acc[0]),
            NarrowKernel::Shifted(shifted) => shifted_mul_acc(self, shifted, shift, &mut acc[0]),
            NarrowKernel::Terms => sparse_mul_acc_narrow(self, challenge, &mut acc[0]),
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
        kernel: &NarrowKernel,
        shift: &mut ShiftScratch<D>,
    ) {
        match (self, kernel) {
            (Self::I8(planes), NarrowKernel::Windowed(windowed)) => {
                for (plane, dst) in planes.iter().zip(acc) {
                    windowed_mul_acc(plane, windowed, dst);
                }
            }
            (Self::I8(planes), NarrowKernel::Shifted(shifted)) => {
                for (plane, dst) in planes.iter().zip(acc) {
                    shifted_mul_acc(plane, shifted, shift, dst);
                }
            }
            (Self::I8(planes), NarrowKernel::Terms) => {
                for (plane, dst) in planes.iter().zip(acc) {
                    sparse_mul_acc_narrow(plane, challenge, dst);
                }
            }
            (Self::I16(planes), NarrowKernel::Shifted(shifted)) => {
                for (plane, dst) in planes.iter().zip(acc) {
                    shifted_mul_acc_i16(plane, shifted, shift, dst);
                }
            }
            (Self::I16(planes), NarrowKernel::Windowed(_) | NarrowKernel::Terms) => {
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
        shift: ShiftScratch<D>,
    },
}

impl<const D: usize> FoldAccumulator<'_, D> {
    fn accumulate<S: BlockDigits<D>>(
        &mut self,
        source: &S,
        scratch: &mut S::Scratch,
        ring_range: Range<usize>,
        shape: DigitShape,
        prepared: &PreparedChallenge<'_, D>,
    ) -> Result<(), AkitaError> {
        let num_digits = shape.num_digits;
        let challenge = prepared.challenge.challenge();
        match (&mut *self, &prepared.plan) {
            (
                Self::Narrow {
                    wide,
                    narrow,
                    bound,
                    shift,
                },
                ChallengePlan::NarrowFull(contribution, kernel),
            ) => {
                if *bound + contribution > i16::MAX as u64 {
                    flush_narrow_accumulator(narrow, wide);
                    *bound = 0;
                }
                for (local, ring) in ring_range.enumerate() {
                    let planes = source.digit_planes(ring, shape, scratch)?;
                    let base = local * num_digits;
                    planes.accumulate_narrow(
                        &mut narrow[base..base + num_digits],
                        challenge,
                        kernel,
                        shift,
                    );
                }
                *bound += contribution;
            }
            (
                Self::Narrow {
                    wide,
                    narrow,
                    bound,
                    ..
                },
                ChallengePlan::NarrowChunked(ranges),
            ) => {
                if *bound != 0 {
                    flush_narrow_accumulator(narrow, wide);
                    *bound = 0;
                }
                for (local, ring) in ring_range.enumerate() {
                    let planes = source.digit_planes(ring, shape, scratch)?;
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
                        ..
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
                    let planes = source.digit_planes(ring, shape, scratch)?;
                    let base = local * num_digits;
                    planes.accumulate_wide(
                        &mut wide[base..base + num_digits],
                        prepared.challenge,
                        plan,
                    );
                }
            }
        }
        Ok(())
    }

    fn finish(self) {
        if let Self::Narrow {
            wide,
            mut narrow,
            bound,
            ..
        } = self
        {
            if bound != 0 {
                flush_narrow_accumulator(&mut narrow, wide);
            }
        }
    }
}

pub(crate) fn element_partitioned_decompose_fold<S: BlockDigits<D>, const D: usize>(
    source: &S,
    challenges: &[SparseChallenge],
    chunk_ranges: &[Range<usize>],
    num_positions_per_block: usize,
    shape: DigitShape,
) -> Result<Vec<Vec<[i32; D]>>, AkitaError> {
    let num_digits = shape.num_digits;
    let inner_width = checked::product([num_positions_per_block, num_digits])
        .ok_or_else(|| AkitaError::InvalidInput("partitioned fold inner width overflow".into()))?;
    let digit_abs_bound = source.digit_abs_bound(shape);
    let window_span = source.i8_digit_span(shape).filter(|_| gathers_windows());
    let shifts = shifts_windows();
    let position_tile = position_tile_len(num_positions_per_block);
    chunk_ranges
        .iter()
        .map(|range| {
            if inner_width == 0 || num_digits == 0 {
                return Ok(Vec::new());
            }
            let ring_start = (range.start * num_positions_per_block).min(source.num_rings());
            let ring_end = (range.end * num_positions_per_block).min(source.num_rings());
            let challenges = ValidatedSparseChallenges::<D>::new(
                &challenges[range.clone()],
                ring_end - ring_start,
                num_positions_per_block,
            )?;
            let plans = challenges
                .iter()
                .enumerate()
                .map(|(block_idx, challenge)| PreparedChallenge {
                    challenge,
                    plan: prepare_challenge::<D>(
                        digit_abs_bound,
                        window_span,
                        shifts,
                        challenge.challenge(),
                    ),
                    block_start: ring_start + block_idx * num_positions_per_block,
                })
                .collect::<Vec<_>>();
            let uses_narrow_accumulation = plans.iter().any(|prepared| {
                matches!(
                    prepared.plan,
                    ChallengePlan::NarrowFull(..) | ChallengePlan::NarrowChunked(_)
                )
            });
            let mut out = vec![[0i32; D]; inner_width];
            cfg_chunks_mut!(out, position_tile * num_digits)
                .enumerate()
                .try_for_each(|(tile_idx, acc)| {
                    let elem_start = tile_idx * position_tile;
                    if elem_start >= num_positions_per_block {
                        return Ok(());
                    }
                    let elem_end = elem_start + acc.len() / num_digits;
                    let mut digit_scratch = source.scratch(shape, 1);
                    let mut accumulator = if uses_narrow_accumulation {
                        let narrow = vec![[0i16; D]; acc.len()];
                        FoldAccumulator::Narrow {
                            wide: acc,
                            narrow,
                            bound: 0,
                            shift: ShiftScratch::new(),
                        }
                    } else {
                        FoldAccumulator::Wide(acc)
                    };
                    for prepared in &plans {
                        let start = prepared.block_start + elem_start;
                        if start >= ring_end {
                            continue;
                        }
                        let end = (prepared.block_start + elem_end).min(ring_end);
                        accumulator.accumulate::<S>(
                            source,
                            &mut digit_scratch,
                            start..end,
                            shape,
                            prepared,
                        )?;
                    }
                    accumulator.finish();
                    Ok::<(), AkitaError>(())
                })?;
            Ok(out)
        })
        .collect()
}

/// Element-partitioned accumulation for multi-digit dense witnesses.
pub fn balanced_ring_decompose_fold_partitioned<F: Field + CanonicalEncoding, const D: usize>(
    coeffs: &[CyclotomicRing<F, D>],
    challenges: &[SparseChallenge],
    num_positions_per_block: usize,
    params: &BalancedDecomposePow2Params<F>,
) -> Result<Vec<[i32; D]>, AkitaError> {
    let basis = SignedDigitBasis::new(params.log_basis())?;
    element_partitioned_decompose_fold::<_, D>(
        &LiveRings {
            coeffs,
            params,
            basis,
        },
        challenges,
        std::slice::from_ref(&(0..challenges.len())),
        num_positions_per_block,
        DigitShape {
            num_digits: params.levels(),
            log_basis: params.log_basis(),
        },
    )
    .map(|mut chunks| chunks.pop().unwrap_or_default())
}
