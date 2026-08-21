use super::*;

mod rotations;

use rotations::{add_rotated, prepare_rotations, PreparedRotations};

#[cfg(feature = "parallel")]
const TASKS_PER_RAYON_WORKER: usize = 4;
const DECOMPOSE_POSITION_WORKING_SET_TARGET: usize = 1 << 21;

struct DecomposeSource<'a, F: Field, I: OneHotIndex> {
    poly: &'a OneHotPoly<F, I>,
    challenge_start: usize,
    active_blocks: usize,
    ring_elems: usize,
}

#[inline]
fn accumulate_ring_range<F, I, const D: usize>(
    source: &DecomposeSource<'_, F, I>,
    ring_start: usize,
    ring_end: usize,
    block_start: usize,
    challenge_idx: usize,
    dst: &mut [[i32; D]],
    rotations: &PreparedRotations<'_, D>,
) where
    F: Field,
    I: OneHotIndex,
{
    let poly = source.poly;
    let onehot_k = poly.onehot_k;
    if onehot_k == D {
        for (ring, hot) in poly.indices[ring_start..ring_end]
            .iter()
            .copied()
            .enumerate()
        {
            if let Some(hot) = hot {
                add_rotated(
                    &mut dst[ring_start + ring - block_start],
                    rotations,
                    challenge_idx,
                    hot.as_usize(),
                );
            }
        }
    } else if onehot_k > D {
        let rings_per_chunk = onehot_k / D;
        let chunk_start = ring_start / rings_per_chunk;
        let chunk_end = ring_end.div_ceil(rings_per_chunk);
        for (chunk, hot) in poly.indices[chunk_start..chunk_end]
            .iter()
            .copied()
            .enumerate()
        {
            let Some(hot) = hot else {
                continue;
            };
            let hot = hot.as_usize();
            let ring = (chunk_start + chunk) * rings_per_chunk + hot / D;
            if ring_start <= ring && ring < ring_end {
                add_rotated(
                    &mut dst[ring - block_start],
                    rotations,
                    challenge_idx,
                    hot % D,
                );
            }
        }
    } else {
        let chunks_per_ring = D / onehot_k;
        let chunk_start = ring_start * chunks_per_ring;
        let chunk_end = ring_end * chunks_per_ring;
        for (chunk, hot) in poly.indices[chunk_start..chunk_end]
            .iter()
            .copied()
            .enumerate()
        {
            if let Some(hot) = hot {
                let local_chunk = chunk_start + chunk;
                let ring = local_chunk / chunks_per_ring;
                let lane = local_chunk % chunks_per_ring;
                add_rotated(
                    &mut dst[ring - block_start],
                    rotations,
                    challenge_idx,
                    lane * onehot_k + hot.as_usize(),
                );
            }
        }
    }
}

fn accumulate_indices<F, I, const D: usize>(
    sources: &[DecomposeSource<'_, F, I>],
    challenges: &[SparseChallenge],
    num_positions_per_block: usize,
) -> Vec<[i32; D]>
where
    F: Field,
    I: OneHotIndex,
{
    let rotations = {
        let _span = tracing::info_span!(
            "onehot_prepare_rotations",
            challenges = challenges.len(),
            ring_dimension = D,
        )
        .entered();
        prepare_rotations::<D>(challenges)
    };
    let row_alignment = sources
        .iter()
        .map(|source| (source.poly.onehot_k / D).max(1))
        .max()
        .unwrap_or(1);
    #[cfg(feature = "parallel")]
    let target_tasks = rayon::current_num_threads()
        .saturating_mul(TASKS_PER_RAYON_WORKER)
        .min(num_positions_per_block)
        .max(1);
    #[cfg(not(feature = "parallel"))]
    let target_tasks = 1usize;
    let thread_balanced_chunk = num_positions_per_block
        .div_ceil(target_tasks)
        .next_multiple_of(row_alignment);
    let cache_sized_chunk = (DECOMPOSE_POSITION_WORKING_SET_TARGET
        / std::mem::size_of::<[i32; D]>())
    .max(row_alignment)
    .next_multiple_of(row_alignment);
    let position_chunk = thread_balanced_chunk
        .min(cache_sized_chunk)
        .min(num_positions_per_block);
    let position_tasks = num_positions_per_block.div_ceil(position_chunk);
    let _span = tracing::info_span!(
        "onehot_accumulate_indices",
        sources = sources.len(),
        challenges = challenges.len(),
        ring_dimension = D,
        rotation_kind = rotations.kind(),
        position_tasks,
        position_chunk,
    )
    .entered();
    let mut compressed = vec![[0i32; D]; num_positions_per_block];
    cfg_chunks_mut!(&mut compressed, position_chunk)
        .enumerate()
        .for_each(|(position_task, dst)| {
            let position_start = position_task * position_chunk;
            let position_end = position_start + dst.len();
            for source in sources {
                for block in 0..source.active_blocks {
                    let block_base = block * num_positions_per_block;
                    let ring_start = (block_base + position_start).min(source.ring_elems);
                    let ring_end = (block_base + position_end).min(source.ring_elems);
                    if ring_start >= ring_end {
                        continue;
                    }
                    accumulate_ring_range(
                        source,
                        ring_start,
                        ring_end,
                        block_base + position_start,
                        source.challenge_start + block,
                        dst,
                        &rotations,
                    );
                }
            }
        });
    compressed
}

fn expand_onehot_accum<const D: usize>(
    compressed: Vec<[i32; D]>,
    num_digits: usize,
) -> Vec<[i32; D]> {
    if num_digits == 1 {
        return compressed;
    }

    let mut expanded = Vec::with_capacity(compressed.len().saturating_mul(num_digits));
    for coeffs in compressed {
        expanded.push(coeffs);
        for _ in 1..num_digits {
            expanded.push([0i32; D]);
        }
    }
    expanded
}

pub(super) fn finish_decompose_fold<F: Field + CanonicalEncoding, const D: usize>(
    compressed_accum: Vec<[i32; D]>,
    num_digits: usize,
) -> DecomposeFoldWitness<F> {
    let modulus = (-F::one())
        .to_u128_checked()
        .expect("Akita field element must fit in u128")
        + 1;
    let coeff_accum = {
        let _span = tracing::info_span!("onehot_expand_accum").entered();
        expand_onehot_accum(compressed_accum, num_digits)
    };
    let _span = tracing::info_span!("onehot_convert").entered();
    build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
}

impl<F: Field, I: OneHotIndex> OneHotPoly<F, I> {
    pub(super) fn decompose_fold_batched_onehot<const D: usize>(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        num_positions_per_block: usize,
        num_digits: usize,
    ) -> Option<DecomposeFoldWitness<F>>
    where
        F: Field + CanonicalEncoding,
    {
        let mut challenge_start = 0usize;
        let mut sources = Vec::with_capacity(polys.len());
        for &poly in polys {
            if challenge_start == challenges.len() {
                break;
            }
            let (ring_elems, num_blocks) = poly.view_layout(D, num_positions_per_block).ok()?;
            let active_blocks = num_blocks.min(challenges.len() - challenge_start);
            if active_blocks == 0 {
                continue;
            }
            sources.push(DecomposeSource {
                poly,
                challenge_start,
                active_blocks,
                ring_elems,
            });
            challenge_start += active_blocks;
        }
        if challenge_start == 0 {
            return None;
        }
        let compressed = accumulate_indices::<F, I, D>(
            &sources,
            &challenges[..challenge_start],
            num_positions_per_block,
        );
        Some(finish_decompose_fold(compressed, num_digits))
    }
}
