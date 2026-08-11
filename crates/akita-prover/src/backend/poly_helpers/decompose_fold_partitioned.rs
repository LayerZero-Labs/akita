//! Partitioned decompose-fold accumulation (element- and position-partitioned).

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

type RotatedTable<const D: usize> = Option<[[i16; D]; D]>;

struct PreparedPm1Challenge {
    positive: Vec<u32>,
    negative: Vec<u32>,
}

fn prepare_pm1_challenge<const D: usize>(
    challenge: &SparseChallenge,
) -> Option<PreparedPm1Challenge> {
    if challenge.positions.len() != challenge.coeffs.len() {
        return None;
    }
    let mut positive = Vec::with_capacity(challenge.positions.len());
    let mut negative = Vec::with_capacity(challenge.positions.len());
    for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
        if position >= D as u32 {
            return None;
        }
        match coefficient {
            1 => positive.push(position),
            -1 => negative.push(position),
            _ => return None,
        }
    }
    Some(PreparedPm1Challenge { positive, negative })
}

fn precompute_rotated_tables<const D: usize>(
    challenges: &[SparseChallenge],
) -> Vec<RotatedTable<D>> {
    challenges
        .iter()
        .map(|challenge| {
            should_use_rotated_challenge::<D>(challenge).then(|| {
                let mut rotated = [[0i16; D]; D];
                fill_rotated_challenge::<D>(&mut rotated, challenge);
                rotated
            })
        })
        .collect()
}

fn precompute_pm1_challenges<const D: usize>(
    challenges: &[SparseChallenge],
    rotated_tables: &[RotatedTable<D>],
) -> Vec<Option<PreparedPm1Challenge>> {
    challenges
        .iter()
        .zip(rotated_tables)
        .map(|(challenge, rotated)| {
            if rotated.is_none() {
                prepare_pm1_challenge::<D>(challenge)
            } else {
                None
            }
        })
        .collect()
}

fn partition_thread_count(num_positions_per_block: usize) -> usize {
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;
    num_threads.min(num_positions_per_block.max(1)).max(1)
}

enum ElementFoldSource<'a, F: CanonicalField, const D: usize> {
    Predecomposed {
        digit_planes: &'a [[i8; D]],
        num_rings: usize,
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

    fn digit_scratch(
        &self,
        rotated_tables: &[RotatedTable<D>],
        num_digits: usize,
    ) -> Option<DigitScratch<D>> {
        match self {
            Self::LiveRings { params, .. } if rotated_tables.iter().any(Option::is_none) => Some(
                match SignedDigitKernel::for_log_basis(params.log_basis)
                    .expect("decompose-fold parameters must use a validated signed-digit basis")
                {
                    SignedDigitKernel::I8 => DigitScratch::I8(vec![[0i8; D]; num_digits]),
                    SignedDigitKernel::I16 => DigitScratch::I16(vec![[0i16; D]; num_digits]),
                },
            ),
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
        rotated: Option<&[[i16; D]; D]>,
        pm1: Option<&PreparedPm1Challenge>,
        digit_scratch: Option<&mut DigitScratch<D>>,
        num_digits: usize,
    ) {
        let dst_base = local_elem_idx * num_digits;
        match (self, rotated) {
            (Self::Predecomposed { digit_planes, .. }, Some(rotated)) => {
                let src_base = ring_idx * num_digits;
                for digit_idx in 0..num_digits {
                    accumulate_rotated_digit_plane::<D>(
                        &digit_planes[src_base + digit_idx],
                        rotated,
                        &mut acc[dst_base + digit_idx],
                    );
                }
            }
            (Self::Predecomposed { digit_planes, .. }, None) => {
                let src_base = ring_idx * num_digits;
                for digit_idx in 0..num_digits {
                    let digit_plane = &digit_planes[src_base + digit_idx];
                    let digit_acc = &mut acc[dst_base + digit_idx];
                    if let Some(pm1) = pm1 {
                        sparse_mul_acc_pm1(digit_plane, &pm1.positive, &pm1.negative, digit_acc);
                    } else {
                        sparse_mul_acc(digit_plane, challenge, digit_acc);
                    }
                }
            }
            (Self::LiveRings { coeffs, params }, Some(rotated)) => {
                let base = dst_base;
                decompose_ring_full_challenge_accumulate::<F, D>(
                    &coeffs[ring_idx],
                    rotated,
                    &mut acc[base..base + num_digits],
                    params,
                );
            }
            (Self::LiveRings { coeffs, params }, None) => {
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
                            if let Some(pm1) = pm1 {
                                sparse_mul_acc_pm1(
                                    &digit_buf[digit],
                                    &pm1.positive,
                                    &pm1.negative,
                                    &mut acc[base + digit],
                                );
                            } else {
                                sparse_mul_acc(
                                    &digit_buf[digit],
                                    challenge,
                                    &mut acc[base + digit],
                                );
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
                            if let Some(pm1) = pm1 {
                                sparse_mul_acc_i16_pm1(
                                    &digit_buf[digit],
                                    &pm1.positive,
                                    &pm1.negative,
                                    &mut acc[base + digit],
                                );
                            } else {
                                sparse_mul_acc_i16(
                                    &digit_buf[digit],
                                    challenge,
                                    &mut acc[base + digit],
                                );
                            }
                        }
                    }
                }
            }
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

    let rotated_tables = precompute_rotated_tables::<D>(challenges);
    let pm1_challenges = precompute_pm1_challenges(challenges, &rotated_tables);
    let actual_threads = partition_thread_count(num_positions_per_block);
    let elem_chunk = num_positions_per_block.div_ceil(actual_threads);
    let mut out = vec![[0i32; D]; inner_width];

    cfg_chunks_mut!(out, elem_chunk * num_digits)
        .enumerate()
        .for_each(|(tid, acc)| {
            let elem_start = tid * elem_chunk;
            if elem_start >= num_positions_per_block {
                return;
            }
            let elems_in_chunk = acc.len() / num_digits;
            let elem_end = elem_start + elems_in_chunk;
            let mut digit_scratch = source.digit_scratch(&rotated_tables, num_digits);

            for (block_idx, challenge) in challenges.iter().enumerate() {
                let block_start = block_idx * num_positions_per_block;
                if block_start >= source.num_rings() {
                    break;
                }
                let ring_start = block_start + elem_start;
                if ring_start >= source.num_rings() {
                    continue;
                }
                let ring_end = (block_start + elem_end).min(source.num_rings());

                for local_elem_idx in 0..(ring_end - ring_start) {
                    source.accumulate_ring(
                        ring_start + local_elem_idx,
                        local_elem_idx,
                        acc,
                        challenge,
                        rotated_tables[block_idx].as_ref(),
                        pm1_challenges[block_idx].as_ref(),
                        digit_scratch.as_mut(),
                        num_digits,
                    );
                }
            }
        });

    out
}

/// Element-partitioned accumulation for predecomposed dense digit caches.
pub fn cached_digit_decompose_fold_partitioned<const D: usize>(
    digit_planes: &[[i8; D]],
    challenges: &[SparseChallenge],
    num_positions_per_block: usize,
    num_digits: usize,
) -> Vec<[i32; D]> {
    let num_rings = digit_planes.len() / num_digits;
    // `F` is unused for the predecomposed source; any `CanonicalField` instantiates the driver.
    element_partitioned_decompose_fold::<akita_field::Prime128Offset275, D>(
        ElementFoldSource::Predecomposed {
            digit_planes,
            num_rings,
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
pub fn balanced_tight_digit_fold_partitioned<const D: usize>(
    coeffs: &[[i8; D]],
    challenges: &[SparseChallenge],
    num_positions_per_block: usize,
) -> Vec<[i32; D]> {
    let num_digits = 1;
    let inner_width = num_positions_per_block;
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(inner_width).max(1);
    let pos_chunk = inner_width.div_ceil(actual_threads);
    let pm1_challenges = challenges
        .iter()
        .map(prepare_pm1_challenge::<D>)
        .collect::<Vec<_>>();

    let chunks: Vec<Vec<[i32; D]>> = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let pos_start = tid * pos_chunk;
            if pos_start >= inner_width {
                return Vec::new();
            }
            let pos_end = (pos_start + pos_chunk).min(inner_width);
            let len = pos_end - pos_start;
            let mut acc = vec![[0i32; D]; len];

            let elem_start = pos_start / num_digits;
            let elem_end = pos_end.div_ceil(num_digits);

            let lo = elem_start.min(num_positions_per_block);
            let hi = elem_end.min(num_positions_per_block);
            for col in lo..hi {
                let out_pos = col * num_digits;
                if out_pos < pos_start || out_pos >= pos_end {
                    continue;
                }

                for (block, (challenge, pm1)) in challenges.iter().zip(&pm1_challenges).enumerate()
                {
                    let Some(index) = block
                        .checked_mul(num_positions_per_block)
                        .and_then(|base| base.checked_add(col))
                    else {
                        continue;
                    };
                    let Some(coeff) = coeffs.get(index) else {
                        break;
                    };
                    if let Some(pm1) = pm1 {
                        sparse_mul_acc_pm1(
                            coeff,
                            &pm1.positive,
                            &pm1.negative,
                            &mut acc[out_pos - pos_start],
                        );
                    } else {
                        sparse_mul_acc(coeff, challenge, &mut acc[out_pos - pos_start]);
                    }
                }
            }
            acc
        })
        .collect();

    chunks.into_iter().flatten().collect()
}
