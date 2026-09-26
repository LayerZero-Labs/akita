//! Partitioned decompose-fold accumulation (element- and position-partitioned).

use super::narrow_accum::{
    sparse_mul_acc as sparse_mul_acc_narrow, sparse_mul_acc_i16 as sparse_mul_acc_i16_narrow,
    sparse_mul_acc_i16_terms as sparse_mul_acc_i16_narrow_terms,
    sparse_mul_acc_terms as sparse_mul_acc_narrow_terms,
};
use super::rotated_accum::{accumulate_rotated_digit_plane, should_use_rotated_challenge};
use super::{
    fill_rotated_challenge, sparse_mul_acc, sparse_mul_acc_i16, sparse_mul_acc_i16_pm1,
    sparse_mul_acc_pm1,
};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_types::SignedDigitKernel;
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field};
use std::ops::Range;

use crate::sources::packed_digits::PackedSignedDigitView;

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

enum ElementFoldSource<'a, F: Field + CanonicalEncoding, const D: usize> {
    Predecomposed {
        digit_planes: &'a [[i8; D]],
        num_rings: usize,
        digit_abs_bound: u64,
    },
    LiveRings {
        coeffs: &'a [CyclotomicRing<F, D>],
        params: &'a BalancedDecomposePow2Params<F>,
    },
    PackedTight {
        digits: PackedSignedDigitView<'a>,
        num_rings: usize,
        digit_abs_bound: u64,
    },
}

enum DigitScratch<const D: usize> {
    None,
    I8(Vec<[i8; D]>),
    I16(Vec<[i16; D]>),
    Packed([i8; D]),
}

impl<F: Field + CanonicalEncoding, const D: usize> ElementFoldSource<'_, F, D> {
    fn num_rings(&self) -> usize {
        match self {
            Self::Predecomposed { num_rings, .. } => *num_rings,
            Self::LiveRings { coeffs, .. } => coeffs.len(),
            Self::PackedTight { num_rings, .. } => *num_rings,
        }
    }

    fn digit_abs_bound(&self) -> u64 {
        match self {
            Self::Predecomposed {
                digit_abs_bound, ..
            } => *digit_abs_bound,
            Self::LiveRings { params, .. } => {
                akita_types::balanced_signed_digit_abs_bound(params.log_basis())
                    .expect("decompose-fold parameters must use a validated signed-digit basis")
            }
            Self::PackedTight {
                digit_abs_bound, ..
            } => *digit_abs_bound,
        }
    }

    fn digit_scratch(&self, num_digits: usize) -> DigitScratch<D> {
        match self {
            Self::LiveRings { params, .. } => {
                match SignedDigitKernel::for_log_basis(params.log_basis())
                    .expect("decompose-fold parameters must use a validated signed-digit basis")
                {
                    SignedDigitKernel::I8 => DigitScratch::I8(vec![[0i8; D]; num_digits]),
                    SignedDigitKernel::I16 => DigitScratch::I16(vec![[0i16; D]; num_digits]),
                }
            }
            Self::PackedTight { .. } => DigitScratch::Packed([0i8; D]),
            Self::Predecomposed { .. } => DigitScratch::None,
        }
    }

    /// Resolve the source once per ring. Cached planes are borrowed directly;
    /// live and packed sources reuse the tile's scratch across rings.
    fn digit_planes<'s>(
        &'s self,
        ring_idx: usize,
        num_digits: usize,
        scratch: &'s mut DigitScratch<D>,
    ) -> DigitPlanes<'s, D> {
        match (self, scratch) {
            (Self::Predecomposed { digit_planes, .. }, DigitScratch::None) => {
                let start = ring_idx * num_digits;
                DigitPlanes::I8(&digit_planes[start..start + num_digits])
            }
            (Self::LiveRings { coeffs, params }, DigitScratch::I8(planes)) => {
                coeffs[ring_idx].balanced_decompose_pow2_i8_into_with_params(planes, params);
                DigitPlanes::I8(planes)
            }
            (Self::LiveRings { coeffs, params }, DigitScratch::I16(planes)) => {
                coeffs[ring_idx].balanced_decompose_pow2_i16_into(planes, params);
                DigitPlanes::I16(planes)
            }
            (Self::PackedTight { digits, .. }, DigitScratch::Packed(plane)) => {
                debug_assert_eq!(num_digits, 1);
                *plane = digits
                    .decode_array::<D>(ring_idx * D)
                    .expect("validated packed recursive ring");
                DigitPlanes::I8(std::slice::from_ref(plane))
            }
            _ => unreachable!("source and tile scratch must match"),
        }
    }
}

/// A ring's digit planes, independent of how the source supplied them.
enum DigitPlanes<'a, const D: usize> {
    I8(&'a [[i8; D]]),
    I16(&'a [[i16; D]]),
}

impl<const D: usize> DigitPlanes<'_, D> {
    fn accumulate_wide(
        self,
        acc: &mut [[i32; D]],
        challenge: &SparseChallenge,
        plan: &ChallengePlan<D>,
    ) {
        match self {
            Self::I8(planes) => {
                for (plane, dst) in planes.iter().zip(acc) {
                    match plan {
                        ChallengePlan::Rotated(rotated) => {
                            accumulate_rotated_digit_plane(plane, rotated.as_ref(), dst)
                        }
                        ChallengePlan::WidePm1(pm1) => {
                            sparse_mul_acc_pm1(plane, &pm1.positive, &pm1.negative, dst)
                        }
                        ChallengePlan::WideGeneric => sparse_mul_acc(plane, challenge, dst),
                        _ => unreachable!("wide accumulation requires a wide plan"),
                    }
                }
            }
            Self::I16(planes) => {
                for (plane, dst) in planes.iter().zip(acc) {
                    match plan {
                        ChallengePlan::Rotated(rotated) => {
                            accumulate_rotated_digit_plane(plane, rotated.as_ref(), dst)
                        }
                        ChallengePlan::WidePm1(pm1) => {
                            sparse_mul_acc_i16_pm1(plane, &pm1.positive, &pm1.negative, dst)
                        }
                        ChallengePlan::WideGeneric => sparse_mul_acc_i16(plane, challenge, dst),
                        _ => unreachable!("wide accumulation requires a wide plan"),
                    }
                }
            }
        }
    }

    fn accumulate_narrow(self, acc: &mut [[i16; D]], challenge: &SparseChallenge) {
        match self {
            Self::I8(planes) => {
                for (plane, dst) in planes.iter().zip(acc) {
                    sparse_mul_acc_narrow(plane, challenge, dst);
                }
            }
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

fn element_partitioned_decompose_fold<F: Field + CanonicalEncoding, const D: usize>(
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
            let mut digit_scratch = source.digit_scratch(num_digits);
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
                        let planes = source.digit_planes(
                            ring_start + local_elem_idx,
                            num_digits,
                            &mut digit_scratch,
                        );
                        let base = local_elem_idx * num_digits;
                        planes.accumulate_narrow(&mut narrow[base..base + num_digits], challenge);
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
                        let planes = source.digit_planes(
                            ring_start + local_elem_idx,
                            num_digits,
                            &mut digit_scratch,
                        );
                        let base = local_elem_idx * num_digits;
                        planes.accumulate_chunked_narrow(
                            &mut narrow[base..base + num_digits],
                            &mut acc[base..base + num_digits],
                            challenge,
                            term_ranges,
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
                        let planes = source.digit_planes(
                            ring_start + local_elem_idx,
                            num_digits,
                            &mut digit_scratch,
                        );
                        let base = local_elem_idx * num_digits;
                        planes.accumulate_wide(&mut acc[base..base + num_digits], challenge, plan);
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
pub(crate) fn cached_digit_decompose_fold_partitioned<
    F: Field + CanonicalEncoding,
    const D: usize,
>(
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
pub fn balanced_ring_decompose_fold_partitioned<F: Field + CanonicalEncoding, const D: usize>(
    coeffs: &[CyclotomicRing<F, D>],
    challenges: &[SparseChallenge],
    num_positions_per_block: usize,
    params: &BalancedDecomposePow2Params<F>,
) -> Vec<[i32; D]> {
    element_partitioned_decompose_fold::<F, D>(
        ElementFoldSource::LiveRings { coeffs, params },
        challenges,
        num_positions_per_block,
        params.levels(),
    )
}

/// Position-partitioned accumulation over a packed tight recursive witness.
pub(crate) fn packed_tight_digit_fold_partitioned<F: Field + CanonicalEncoding, const D: usize>(
    digits: PackedSignedDigitView<'_>,
    num_rings: usize,
    challenges: &[SparseChallenge],
    num_positions_per_block: usize,
) -> Vec<[i32; D]> {
    let digit_abs_bound = u64::from(
        digits
            .bounds()
            .negative_abs_max()
            .max(digits.bounds().positive_max()),
    );
    element_partitioned_decompose_fold::<F, D>(
        ElementFoldSource::PackedTight {
            digits,
            num_rings,
            digit_abs_bound,
        },
        challenges,
        num_positions_per_block,
        1,
    )
}
