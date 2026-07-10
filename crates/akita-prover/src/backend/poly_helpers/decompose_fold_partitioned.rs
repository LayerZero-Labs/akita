//! Partitioned decompose-fold accumulation (element- and position-partitioned).

use super::rotated_accum::{
    accumulate_rotated_digit_plane, decompose_ring_full_challenge_accumulate,
    should_use_rotated_challenge,
};
use super::{decompose_ring_interleaved, fill_rotated_challenge, sparse_mul_acc, DecomposeParams};
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_field::parallel::*;
use akita_field::CanonicalField;

type RotatedTable<const D: usize> = Option<[[i16; D]; D]>;

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

fn partition_thread_count(block_len: usize) -> usize {
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;
    num_threads.min(block_len.max(1)).max(1)
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

impl<F: CanonicalField, const D: usize> ElementFoldSource<'_, F, D> {
    fn num_rings(&self) -> usize {
        match self {
            Self::Predecomposed { num_rings, .. } => *num_rings,
            Self::LiveRings { coeffs, .. } => coeffs.len(),
        }
    }

    fn needs_digit_buf(&self, rotated_tables: &[RotatedTable<D>]) -> bool {
        matches!(self, Self::LiveRings { .. }) && rotated_tables.iter().any(Option::is_none)
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate_ring(
        &self,
        ring_idx: usize,
        local_elem_idx: usize,
        acc: &mut [[i32; D]],
        challenge: &SparseChallenge,
        rotated: Option<&[[i16; D]; D]>,
        digit_buf: Option<&mut [[i8; D]]>,
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
                    sparse_mul_acc::<D>(
                        &digit_planes[src_base + digit_idx],
                        challenge,
                        &mut acc[dst_base + digit_idx],
                    );
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
                let digit_buf = digit_buf.expect("live sparse path requires a digit buffer");
                let base = dst_base;
                decompose_ring_interleaved::<F, D>(
                    &coeffs[ring_idx],
                    digit_buf,
                    num_digits,
                    params,
                );
                for digit in 0..num_digits {
                    sparse_mul_acc::<D>(&digit_buf[digit], challenge, &mut acc[base + digit]);
                }
            }
        }
    }
}

fn element_partitioned_decompose_fold<F: CanonicalField, const D: usize>(
    source: ElementFoldSource<'_, F, D>,
    challenges: &[SparseChallenge],
    block_len: usize,
    num_digits: usize,
) -> Vec<[i32; D]> {
    let inner_width = block_len
        .checked_mul(num_digits)
        .expect("element-partitioned fold inner width overflow");
    if inner_width == 0 || num_digits == 0 {
        return Vec::new();
    }

    let rotated_tables = precompute_rotated_tables::<D>(challenges);
    let actual_threads = partition_thread_count(block_len);
    let elem_chunk = block_len.div_ceil(actual_threads);
    let mut out = vec![[0i32; D]; inner_width];

    cfg_chunks_mut!(out, elem_chunk * num_digits)
        .enumerate()
        .for_each(|(tid, acc)| {
            let elem_start = tid * elem_chunk;
            if elem_start >= block_len {
                return;
            }
            let elems_in_chunk = acc.len() / num_digits;
            let elem_end = elem_start + elems_in_chunk;
            let mut digit_buf = source
                .needs_digit_buf(&rotated_tables)
                .then(|| vec![[0i8; D]; num_digits]);

            for (block_idx, challenge) in challenges.iter().enumerate() {
                let block_start = block_idx * block_len;
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
                        digit_buf.as_deref_mut(),
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
    block_len: usize,
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
        block_len,
        num_digits,
    )
}

/// Element-partitioned accumulation for multi-digit dense witnesses.
pub fn balanced_ring_decompose_fold_partitioned<F: CanonicalField, const D: usize>(
    coeffs: &[CyclotomicRing<F, D>],
    challenges: &[SparseChallenge],
    block_len: usize,
    num_digits: usize,
    p: &DecomposeParams,
) -> Vec<[i32; D]> {
    element_partitioned_decompose_fold::<F, D>(
        ElementFoldSource::LiveRings { coeffs, params: p },
        challenges,
        block_len,
        num_digits,
    )
}

/// Position-partitioned accumulation for recursive witness decompose-fold.
pub fn balanced_digit_decompose_fold_partitioned<const D: usize>(
    coeffs: &[[i8; D]],
    challenges: &[SparseChallenge],
    active_blocks: usize,
    block_len: usize,
    num_digits: usize,
    inner_width: usize,
) -> Vec<[i32; D]> {
    debug_assert_eq!(
        num_digits, 1,
        "multi-digit decomposition is not implemented for partitioned accumulation"
    );
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(inner_width).max(1);
    let pos_chunk = inner_width.div_ceil(actual_threads);

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

            let lo = elem_start.min(block_len);
            let hi = elem_end.min(block_len);
            for col in lo..hi {
                let out_pos = col * num_digits;
                if out_pos < pos_start || out_pos >= pos_end {
                    continue;
                }

                for (block, challenge) in challenges[..active_blocks].iter().enumerate() {
                    let Some(index) = block
                        .checked_mul(block_len)
                        .and_then(|base| base.checked_add(col))
                    else {
                        continue;
                    };
                    let Some(coeff) = coeffs.get(index) else {
                        break;
                    };
                    sparse_mul_acc::<D>(coeff, challenge, &mut acc[out_pos - pos_start]);
                }
            }
            acc
        })
        .collect();

    chunks.into_iter().flatten().collect()
}
