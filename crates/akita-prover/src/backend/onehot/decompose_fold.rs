use super::*;

#[cfg(feature = "parallel")]
const TASKS_PER_RAYON_WORKER: usize = 4;
const ROTATED_CHALLENGE_TABLE_BUDGET: usize = 1 << 28;
const DECOMPOSE_POSITION_WORKING_SET_TARGET: usize = 1 << 21;

#[derive(Debug)]
struct PreparedSparseClass {
    coefficient: i32,
    positions: Vec<u16>,
    wrap_cuts: Vec<u32>,
}

#[derive(Debug)]
struct PreparedSparseChallenge {
    classes: Vec<PreparedSparseClass>,
}

impl PreparedSparseChallenge {
    fn new<const D: usize>(challenge: &SparseChallenge) -> Self {
        debug_assert!(D <= usize::from(u16::MAX) + 1);
        let mut grouped = Vec::<(i8, Vec<u16>)>::new();
        for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
            let position = u16::try_from(position).expect("validated challenge position fits u16");
            if let Some((_, positions)) = grouped
                .iter_mut()
                .find(|(existing, _)| *existing == coefficient)
            {
                positions.push(position);
            } else {
                grouped.push((coefficient, vec![position]));
            }
        }
        grouped.sort_unstable_by_key(|(coefficient, _)| *coefficient);
        let classes = grouped
            .into_iter()
            .map(|(coefficient, mut positions)| {
                positions.sort_unstable();
                let wrap_cuts = (0..D)
                    .map(|shift| {
                        u32::try_from(
                            positions
                                .partition_point(|&position| usize::from(position) < D - shift),
                        )
                        .expect("sparse challenge support fits u32")
                    })
                    .collect();
                PreparedSparseClass {
                    coefficient: i32::from(coefficient),
                    positions,
                    wrap_cuts,
                }
            })
            .collect();
        Self { classes }
    }
}

#[derive(Debug)]
struct PreparedExpandedSparseClass {
    coefficient: i32,
    support: usize,
    rotated_positions: Vec<u16>,
    wrap_cuts: Vec<u32>,
}

#[derive(Debug)]
struct PreparedExpandedSparseChallenge {
    classes: Vec<PreparedExpandedSparseClass>,
}

impl PreparedExpandedSparseChallenge {
    fn new<const D: usize>(challenge: &SparseChallenge) -> Self {
        let sparse = PreparedSparseChallenge::new::<D>(challenge);
        let classes = sparse
            .classes
            .into_iter()
            .map(|class| {
                let support = class.positions.len();
                let mut rotated_positions = Vec::with_capacity(D.saturating_mul(support));
                for shift in 0..D {
                    rotated_positions.extend(class.positions.iter().map(|&position| {
                        u16::try_from((usize::from(position) + shift) % D)
                            .expect("validated rotated position fits u16")
                    }));
                }
                PreparedExpandedSparseClass {
                    coefficient: class.coefficient,
                    support,
                    rotated_positions,
                    wrap_cuts: class.wrap_cuts,
                }
            })
            .collect();
        Self { classes }
    }
}

#[derive(Debug)]
enum PreparedRotations<const D: usize> {
    Compact(Vec<[i8; D]>),
    Dense(Vec<[i16; D]>),
    ExpandedSparse(Vec<PreparedExpandedSparseChallenge>),
    Sparse(Vec<PreparedSparseChallenge>),
}

impl<const D: usize> PreparedRotations<D> {
    fn kind(&self) -> &'static str {
        match self {
            Self::Compact(_) => "compact",
            Self::Dense(_) => "dense",
            Self::ExpandedSparse(_) => "expanded_sparse",
            Self::Sparse(_) => "sparse",
        }
    }
}

fn prepare_rotations<const D: usize>(challenges: &[SparseChallenge]) -> PreparedRotations<D> {
    let dense_bytes = challenges
        .len()
        .saturating_mul(D)
        .saturating_mul(std::mem::size_of::<[i16; D]>());
    let expanded_sparse_bytes = challenges.iter().fold(
        challenges
            .len()
            .saturating_mul(std::mem::size_of::<PreparedExpandedSparseChallenge>()),
        |bytes, challenge| {
            let class_count = challenge
                .coeffs
                .iter()
                .enumerate()
                .filter(|&(idx, coefficient)| !challenge.coeffs[..idx].contains(coefficient))
                .count();
            bytes
                .saturating_add(
                    challenge
                        .positions
                        .len()
                        .saturating_mul(D)
                        .saturating_mul(std::mem::size_of::<u16>()),
                )
                .saturating_add(
                    class_count
                        .saturating_mul(D)
                        .saturating_mul(std::mem::size_of::<u32>()),
                )
                .saturating_add(
                    class_count.saturating_mul(std::mem::size_of::<PreparedExpandedSparseClass>()),
                )
        },
    );
    if D >= 128 && expanded_sparse_bytes <= ROTATED_CHALLENGE_TABLE_BUDGET {
        return PreparedRotations::ExpandedSparse(
            cfg_into_iter!(0..challenges.len())
                .map(|challenge_idx| {
                    PreparedExpandedSparseChallenge::new::<D>(&challenges[challenge_idx])
                })
                .collect(),
        );
    }
    if D == 128 {
        let compact = cfg_into_iter!(0..challenges.len())
            .map(|challenge_idx| {
                let mut dense = [0i8; D];
                let challenge = &challenges[challenge_idx];
                for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
                    dense[position as usize] = coefficient;
                }
                dense
            })
            .collect();
        return PreparedRotations::Compact(compact);
    }
    if D == 64 && dense_bytes <= ROTATED_CHALLENGE_TABLE_BUDGET {
        let mut rotated = vec![[0i16; D]; challenges.len() * D];
        cfg_chunks_mut!(&mut rotated, D)
            .enumerate()
            .for_each(|(challenge_idx, table)| {
                fill_rotated_challenge(table, &challenges[challenge_idx]);
            });
        PreparedRotations::Dense(rotated)
    } else {
        PreparedRotations::Sparse(
            cfg_into_iter!(0..challenges.len())
                .map(|challenge_idx| PreparedSparseChallenge::new::<D>(&challenges[challenge_idx]))
                .collect(),
        )
    }
}

#[inline(always)]
fn add_rotated_expanded_sparse<const D: usize>(
    dst: &mut [i32; D],
    challenge: &PreparedExpandedSparseChallenge,
    shift: usize,
) {
    for class in &challenge.classes {
        let row_start = shift * class.support;
        let positions = &class.rotated_positions[row_start..row_start + class.support];
        let cut = class.wrap_cuts[shift] as usize;
        for &position in &positions[..cut] {
            dst[usize::from(position)] += class.coefficient;
        }
        for &position in &positions[cut..] {
            dst[usize::from(position)] -= class.coefficient;
        }
    }
}

#[inline(always)]
fn add_rotated_sparse<const D: usize>(
    dst: &mut [i32; D],
    challenge: &PreparedSparseChallenge,
    shift: usize,
) {
    for class in &challenge.classes {
        let cut = class.wrap_cuts[shift] as usize;
        for &position in &class.positions[..cut] {
            dst[usize::from(position) + shift] += class.coefficient;
        }
        for &position in &class.positions[cut..] {
            dst[usize::from(position) + shift - D] -= class.coefficient;
        }
    }
}

#[inline(always)]
fn add_rotated_dense<const D: usize>(dst: &mut [i32; D], rotated: &[i16; D]) {
    for (dst, &value) in dst.iter_mut().zip(rotated) {
        *dst += i32::from(value);
    }
}

#[inline(always)]
fn add_rotated_compact<const D: usize>(dst: &mut [i32; D], dense: &[i8; D], shift: usize) {
    let split = D - shift;
    for (dst, &value) in dst[shift..].iter_mut().zip(&dense[..split]) {
        *dst += i32::from(value);
    }
    for (dst, &value) in dst[..shift].iter_mut().zip(&dense[split..]) {
        *dst -= i32::from(value);
    }
}

#[inline(always)]
fn add_rotated<const D: usize>(
    dst: &mut [i32; D],
    rotations: &PreparedRotations<D>,
    challenge_idx: usize,
    shift: usize,
) {
    match rotations {
        PreparedRotations::Compact(challenges) => {
            add_rotated_compact(dst, &challenges[challenge_idx], shift);
        }
        PreparedRotations::Dense(rotated) => {
            add_rotated_dense(dst, &rotated[challenge_idx * D + shift]);
        }
        PreparedRotations::ExpandedSparse(challenges) => {
            add_rotated_expanded_sparse(dst, &challenges[challenge_idx], shift);
        }
        PreparedRotations::Sparse(challenges) => {
            add_rotated_sparse(dst, &challenges[challenge_idx], shift);
        }
    }
}

struct DecomposeSource<'a, F: FieldCore, I: OneHotIndex> {
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
    rotations: &PreparedRotations<D>,
) where
    F: FieldCore,
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
    F: FieldCore,
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

pub(super) fn finish_decompose_fold<F: CanonicalField, const D: usize>(
    compressed_accum: Vec<[i32; D]>,
    num_digits: usize,
) -> DecomposeFoldWitness<F> {
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let coeff_accum = {
        let _span = tracing::info_span!("onehot_expand_accum").entered();
        expand_onehot_accum(compressed_accum, num_digits)
    };
    let _span = tracing::info_span!("onehot_convert").entered();
    build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
}

impl<F: FieldCore, I: OneHotIndex> OneHotPoly<F, I> {
    pub(super) fn decompose_fold_batched_onehot<const D: usize>(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        num_positions_per_block: usize,
        num_digits: usize,
    ) -> Option<DecomposeFoldWitness<F>>
    where
        F: CanonicalField,
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
