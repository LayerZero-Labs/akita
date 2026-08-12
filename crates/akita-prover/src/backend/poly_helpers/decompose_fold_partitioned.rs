//! Partitioned decompose-fold accumulation (element- and position-partitioned).

use super::narrow_accum::{
    sparse_mul_acc as sparse_mul_acc_narrow, sparse_mul_acc_i16 as sparse_mul_acc_i16_narrow,
    sparse_mul_acc_i16_terms as sparse_mul_acc_i16_narrow_terms,
    sparse_mul_acc_terms as sparse_mul_acc_narrow_terms,
};
use super::rotated_accum::{
    accumulate_rotated_digit_plane, decompose_ring_full_challenge_accumulate,
    should_use_rotated_challenge,
};
use super::{
    decompose_ring_interleaved, decompose_ring_interleaved_i16, fill_rotated_challenge,
    sparse_mul_acc, sparse_mul_acc_i16, sparse_mul_acc_i16_pm1, sparse_mul_acc_pm1,
    DecomposeParams,
};
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_field::parallel::*;
use akita_field::CanonicalField;
use akita_types::SignedDigitKernel;
use std::ops::Range;

struct PreparedPm1Challenge {
    positive: Vec<u32>,
    negative: Vec<u32>,
}

enum ChallengePlan<const D: usize> {
    Rotated(Box<[[i16; D]; D]>),
    NarrowFull(u64),
    NarrowChunked(Vec<Range<usize>>),
    WidePm1(PreparedPm1Challenge),
    WideGeneric,
}

fn prepare_wide_plan<const D: usize>(challenge: &SparseChallenge) -> ChallengePlan<D> {
    let mut positive = Vec::with_capacity(challenge.positions.len());
    let mut negative = Vec::with_capacity(challenge.positions.len());
    for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
        match coefficient {
            1 => positive.push(position),
            -1 => negative.push(position),
            _ => return ChallengePlan::WideGeneric,
        }
    }
    ChallengePlan::WidePm1(PreparedPm1Challenge { positive, negative })
}

fn prepare_challenge<const D: usize>(
    digit_abs_bound: u64,
    challenge: &SparseChallenge,
) -> ChallengePlan<D> {
    let has_valid_shape = challenge.positions.len() == challenge.coeffs.len()
        && challenge
            .positions
            .iter()
            .all(|&position| position < D as u32);
    if !has_valid_shape {
        return ChallengePlan::WideGeneric;
    }
    if should_use_rotated_challenge::<D>(challenge) {
        let mut rotated = Box::new([[0i16; D]; D]);
        fill_rotated_challenge::<D>(rotated.as_mut(), challenge);
        return ChallengePlan::Rotated(rotated);
    }
    if digit_abs_bound == 0 {
        return ChallengePlan::NarrowFull(0);
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
        return ChallengePlan::NarrowFull(contribution_bound);
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

enum ElementFoldSource<'a, F: CanonicalField, const D: usize> {
    Predecomposed {
        digit_planes: &'a [[i8; D]],
        num_rings: usize,
        digit_abs_bound: u64,
    },
    LiveRings {
        coeffs: &'a [CyclotomicRing<F, D>],
        params: &'a DecomposeParams,
    },
}

enum DigitScratch<const D: usize> {
    I8(Vec<[i8; D]>),
    I16(Vec<[i16; D]>),
}

impl<F: CanonicalField, const D: usize> ElementFoldSource<'_, F, D> {
    fn num_rings(&self) -> usize {
        match self {
            Self::Predecomposed { num_rings, .. } => *num_rings,
            Self::LiveRings { coeffs, .. } => coeffs.len(),
        }
    }

    fn digit_abs_bound(&self) -> u64 {
        match self {
            Self::Predecomposed {
                digit_abs_bound, ..
            } => *digit_abs_bound,
            Self::LiveRings { params, .. } => params.half_b as u64,
        }
    }

    fn digit_scratch(
        &self,
        plans: &[ChallengePlan<D>],
        num_digits: usize,
    ) -> Option<DigitScratch<D>> {
        match self {
            Self::LiveRings { params, .. }
                if plans
                    .iter()
                    .any(|plan| !matches!(plan, ChallengePlan::Rotated(_))) =>
            {
                Some(
                    match SignedDigitKernel::for_log_basis(params.log_basis)
                        .expect("decompose-fold parameters must use a validated signed-digit basis")
                    {
                        SignedDigitKernel::I8 => DigitScratch::I8(vec![[0i8; D]; num_digits]),
                        SignedDigitKernel::I16 => DigitScratch::I16(vec![[0i16; D]; num_digits]),
                    },
                )
            }
            _ => None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate_ring(
        &self,
        ring_idx: usize,
        local_elem_idx: usize,
        acc: &mut [[i32; D]],
        challenge: &SparseChallenge,
        plan: &ChallengePlan<D>,
        digit_scratch: Option<&mut DigitScratch<D>>,
        num_digits: usize,
    ) {
        let dst_base = local_elem_idx * num_digits;
        match (self, plan) {
            (Self::Predecomposed { digit_planes, .. }, ChallengePlan::Rotated(rotated)) => {
                let src_base = ring_idx * num_digits;
                for digit_idx in 0..num_digits {
                    accumulate_rotated_digit_plane::<D>(
                        &digit_planes[src_base + digit_idx],
                        rotated.as_ref(),
                        &mut acc[dst_base + digit_idx],
                    );
                }
            }
            (Self::Predecomposed { digit_planes, .. }, plan) => {
                let src_base = ring_idx * num_digits;
                for digit_idx in 0..num_digits {
                    let digit_plane = &digit_planes[src_base + digit_idx];
                    let digit_acc = &mut acc[dst_base + digit_idx];
                    match plan {
                        ChallengePlan::WidePm1(pm1) => {
                            sparse_mul_acc_pm1(digit_plane, &pm1.positive, &pm1.negative, digit_acc)
                        }
                        ChallengePlan::WideGeneric => {
                            sparse_mul_acc(digit_plane, challenge, digit_acc);
                        }
                        ChallengePlan::Rotated(_)
                        | ChallengePlan::NarrowFull(_)
                        | ChallengePlan::NarrowChunked(_) => unreachable!(),
                    }
                }
            }
            (Self::LiveRings { coeffs, params }, ChallengePlan::Rotated(rotated)) => {
                let base = dst_base;
                decompose_ring_full_challenge_accumulate::<F, D>(
                    &coeffs[ring_idx],
                    rotated.as_ref(),
                    &mut acc[base..base + num_digits],
                    params,
                );
            }
            (Self::LiveRings { coeffs, params }, plan) => {
                let base = dst_base;
                match digit_scratch.expect("live sparse path requires signed-digit scratch") {
                    DigitScratch::I8(digit_buf) => {
                        decompose_ring_interleaved::<F, D>(
                            &coeffs[ring_idx],
                            digit_buf,
                            num_digits,
                            params,
                        );
                        for digit in 0..num_digits {
                            match plan {
                                ChallengePlan::WidePm1(pm1) => sparse_mul_acc_pm1(
                                    &digit_buf[digit],
                                    &pm1.positive,
                                    &pm1.negative,
                                    &mut acc[base + digit],
                                ),
                                ChallengePlan::WideGeneric => sparse_mul_acc(
                                    &digit_buf[digit],
                                    challenge,
                                    &mut acc[base + digit],
                                ),
                                ChallengePlan::Rotated(_)
                                | ChallengePlan::NarrowFull(_)
                                | ChallengePlan::NarrowChunked(_) => unreachable!(),
                            }
                        }
                    }
                    DigitScratch::I16(digit_buf) => {
                        decompose_ring_interleaved_i16::<F, D>(
                            &coeffs[ring_idx],
                            digit_buf,
                            num_digits,
                            params,
                        );
                        for digit in 0..num_digits {
                            match plan {
                                ChallengePlan::WidePm1(pm1) => sparse_mul_acc_i16_pm1(
                                    &digit_buf[digit],
                                    &pm1.positive,
                                    &pm1.negative,
                                    &mut acc[base + digit],
                                ),
                                ChallengePlan::WideGeneric => sparse_mul_acc_i16(
                                    &digit_buf[digit],
                                    challenge,
                                    &mut acc[base + digit],
                                ),
                                ChallengePlan::Rotated(_)
                                | ChallengePlan::NarrowFull(_)
                                | ChallengePlan::NarrowChunked(_) => unreachable!(),
                            }
                        }
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate_ring_narrow(
        &self,
        ring_idx: usize,
        local_elem_idx: usize,
        acc: &mut [[i16; D]],
        challenge: &SparseChallenge,
        digit_scratch: Option<&mut DigitScratch<D>>,
        num_digits: usize,
    ) {
        let dst_base = local_elem_idx * num_digits;
        match self {
            Self::Predecomposed { digit_planes, .. } => {
                let src_base = ring_idx * num_digits;
                for digit_idx in 0..num_digits {
                    sparse_mul_acc_narrow(
                        &digit_planes[src_base + digit_idx],
                        challenge,
                        &mut acc[dst_base + digit_idx],
                    );
                }
            }
            Self::LiveRings { coeffs, params } => {
                match digit_scratch.expect("live narrow path requires signed-digit scratch") {
                    DigitScratch::I8(digit_buf) => {
                        decompose_ring_interleaved::<F, D>(
                            &coeffs[ring_idx],
                            digit_buf,
                            num_digits,
                            params,
                        );
                        for digit in 0..num_digits {
                            sparse_mul_acc_narrow(
                                &digit_buf[digit],
                                challenge,
                                &mut acc[dst_base + digit],
                            );
                        }
                    }
                    DigitScratch::I16(digit_buf) => {
                        decompose_ring_interleaved_i16::<F, D>(
                            &coeffs[ring_idx],
                            digit_buf,
                            num_digits,
                            params,
                        );
                        for digit in 0..num_digits {
                            sparse_mul_acc_i16_narrow(
                                &digit_buf[digit],
                                challenge,
                                &mut acc[dst_base + digit],
                            );
                        }
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate_ring_chunked_narrow(
        &self,
        ring_idx: usize,
        local_elem_idx: usize,
        narrow_acc: &mut [[i16; D]],
        wide_acc: &mut [[i32; D]],
        challenge: &SparseChallenge,
        term_ranges: &[Range<usize>],
        digit_scratch: Option<&mut DigitScratch<D>>,
        num_digits: usize,
    ) {
        let dst_base = local_elem_idx * num_digits;
        let narrow = &mut narrow_acc[dst_base..dst_base + num_digits];
        let wide = &mut wide_acc[dst_base..dst_base + num_digits];
        match self {
            Self::Predecomposed { digit_planes, .. } => {
                let src_base = ring_idx * num_digits;
                for term_range in term_ranges {
                    let positions = &challenge.positions[term_range.clone()];
                    let coefficients = &challenge.coeffs[term_range.clone()];
                    for digit_idx in 0..num_digits {
                        sparse_mul_acc_narrow_terms(
                            &digit_planes[src_base + digit_idx],
                            positions,
                            coefficients,
                            &mut narrow[digit_idx],
                        );
                    }
                    flush_narrow_accumulator(narrow, wide);
                }
            }
            Self::LiveRings { coeffs, params } => {
                match digit_scratch.expect("live chunked narrow path requires signed-digit scratch")
                {
                    DigitScratch::I8(digit_buf) => {
                        decompose_ring_interleaved::<F, D>(
                            &coeffs[ring_idx],
                            digit_buf,
                            num_digits,
                            params,
                        );
                        for term_range in term_ranges {
                            let positions = &challenge.positions[term_range.clone()];
                            let coefficients = &challenge.coeffs[term_range.clone()];
                            for digit in 0..num_digits {
                                sparse_mul_acc_narrow_terms(
                                    &digit_buf[digit],
                                    positions,
                                    coefficients,
                                    &mut narrow[digit],
                                );
                            }
                            flush_narrow_accumulator(narrow, wide);
                        }
                    }
                    DigitScratch::I16(digit_buf) => {
                        decompose_ring_interleaved_i16::<F, D>(
                            &coeffs[ring_idx],
                            digit_buf,
                            num_digits,
                            params,
                        );
                        for term_range in term_ranges {
                            let positions = &challenge.positions[term_range.clone()];
                            let coefficients = &challenge.coeffs[term_range.clone()];
                            for digit in 0..num_digits {
                                sparse_mul_acc_i16_narrow_terms(
                                    &digit_buf[digit],
                                    positions,
                                    coefficients,
                                    &mut narrow[digit],
                                );
                            }
                            flush_narrow_accumulator(narrow, wide);
                        }
                    }
                }
            }
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

fn element_partitioned_decompose_fold<F: CanonicalField, const D: usize>(
    source: ElementFoldSource<'_, F, D>,
    challenges: &[SparseChallenge],
    num_positions_per_block: usize,
    num_digits: usize,
) -> Vec<[i32; D]> {
    let inner_width = num_positions_per_block
        .checked_mul(num_digits)
        .expect("element-partitioned fold inner width overflow");
    if inner_width == 0 || num_digits == 0 {
        return Vec::new();
    }

    let digit_abs_bound = source.digit_abs_bound();
    let plans = challenges
        .iter()
        .map(|challenge| prepare_challenge::<D>(digit_abs_bound, challenge))
        .collect::<Vec<_>>();
    let uses_narrow_accumulation = plans.iter().any(|plan| {
        matches!(
            plan,
            ChallengePlan::NarrowFull(_) | ChallengePlan::NarrowChunked(_)
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
            let mut digit_scratch = source.digit_scratch(&plans, num_digits);
            let mut narrow_acc = uses_narrow_accumulation.then(|| vec![[0i16; D]; acc.len()]);
            let mut narrow_bound = 0u64;

            for (block_idx, (challenge, plan)) in challenges.iter().zip(&plans).enumerate() {
                let block_start = block_idx * num_positions_per_block;
                if block_start >= source.num_rings() {
                    break;
                }
                let ring_start = block_start + elem_start;
                if ring_start >= source.num_rings() {
                    continue;
                }
                let ring_end = (block_start + elem_end).min(source.num_rings());

                if let ChallengePlan::NarrowFull(contribution_bound) = plan {
                    let contribution_bound = *contribution_bound;
                    if narrow_bound + contribution_bound > i16::MAX as u64 {
                        flush_narrow_accumulator(
                            narrow_acc
                                .as_mut()
                                .expect("narrow fold path requires an accumulator"),
                            acc,
                        );
                        narrow_bound = 0;
                    }
                    let narrow = narrow_acc
                        .as_mut()
                        .expect("narrow fold path requires an accumulator");
                    for local_elem_idx in 0..(ring_end - ring_start) {
                        source.accumulate_ring_narrow(
                            ring_start + local_elem_idx,
                            local_elem_idx,
                            narrow,
                            challenge,
                            digit_scratch.as_mut(),
                            num_digits,
                        );
                    }
                    narrow_bound += contribution_bound;
                } else if let ChallengePlan::NarrowChunked(term_ranges) = plan {
                    if narrow_bound != 0 {
                        flush_narrow_accumulator(
                            narrow_acc
                                .as_mut()
                                .expect("narrow fold path requires an accumulator"),
                            acc,
                        );
                        narrow_bound = 0;
                    }
                    let narrow = narrow_acc
                        .as_mut()
                        .expect("narrow fold path requires an accumulator");
                    for local_elem_idx in 0..(ring_end - ring_start) {
                        source.accumulate_ring_chunked_narrow(
                            ring_start + local_elem_idx,
                            local_elem_idx,
                            narrow,
                            acc,
                            challenge,
                            term_ranges,
                            digit_scratch.as_mut(),
                            num_digits,
                        );
                    }
                } else {
                    if narrow_bound != 0 {
                        flush_narrow_accumulator(
                            narrow_acc
                                .as_mut()
                                .expect("narrow fold path requires an accumulator"),
                            acc,
                        );
                        narrow_bound = 0;
                    }
                    for local_elem_idx in 0..(ring_end - ring_start) {
                        source.accumulate_ring(
                            ring_start + local_elem_idx,
                            local_elem_idx,
                            acc,
                            challenge,
                            plan,
                            digit_scratch.as_mut(),
                            num_digits,
                        );
                    }
                }
            }
            if narrow_bound != 0 {
                flush_narrow_accumulator(
                    narrow_acc
                        .as_mut()
                        .expect("narrow fold path requires an accumulator"),
                    acc,
                );
            }
        });

    out
}

/// Element-partitioned accumulation for predecomposed dense digit caches.
pub fn cached_digit_decompose_fold_partitioned<F: CanonicalField, const D: usize>(
    digit_planes: &[[i8; D]],
    challenges: &[SparseChallenge],
    num_positions_per_block: usize,
    num_digits: usize,
    log_basis: u32,
) -> Vec<[i32; D]> {
    let num_rings = digit_planes.len() / num_digits;
    let digit_abs_bound = akita_types::balanced_signed_digit_abs_bound(log_basis)
        .expect("cached decompose-fold basis must be validated")
        .min(u64::from(i8::MIN.unsigned_abs()));
    element_partitioned_decompose_fold::<F, D>(
        ElementFoldSource::Predecomposed {
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
pub fn balanced_ring_decompose_fold_partitioned<F: CanonicalField, const D: usize>(
    coeffs: &[CyclotomicRing<F, D>],
    challenges: &[SparseChallenge],
    num_positions_per_block: usize,
    num_digits: usize,
    p: &DecomposeParams,
) -> Vec<[i32; D]> {
    element_partitioned_decompose_fold::<F, D>(
        ElementFoldSource::LiveRings { coeffs, params: p },
        challenges,
        num_positions_per_block,
        num_digits,
    )
}

/// Position-partitioned accumulation for an already-tight recursive digit witness.
pub fn balanced_tight_digit_fold_partitioned<F: CanonicalField, const D: usize>(
    coeffs: &[[i8; D]],
    challenges: &[SparseChallenge],
    num_positions_per_block: usize,
    known_balanced_log_basis: Option<u32>,
) -> Vec<[i32; D]> {
    let digit_abs_bound = known_balanced_log_basis
        .and_then(akita_types::balanced_signed_digit_abs_bound)
        .unwrap_or_else(|| u64::from(i8::MIN.unsigned_abs()))
        .min(u64::from(i8::MIN.unsigned_abs()));
    debug_assert!(coeffs
        .iter()
        .flat_map(|ring| ring.iter())
        .all(|digit| u64::from(digit.unsigned_abs()) <= digit_abs_bound));
    element_partitioned_decompose_fold::<F, D>(
        ElementFoldSource::Predecomposed {
            digit_planes: coeffs,
            num_rings: coeffs.len(),
            digit_abs_bound,
        },
        challenges,
        num_positions_per_block,
        1,
    )
}
