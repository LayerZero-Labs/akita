//! One-hot polynomial: sparse witness exploiting monomial structure.
//!
//! [`OneHotPoly`] implements [`HachiPolyOps`](super::HachiPolyOps) for
//! polynomials with at most one nonzero field element per chunk of size
//! `onehot_k`.  All four operations exploit sparsity, avoiding inner ring
//! multiplications during commit and decomposing only nonzero monomials.
//!
//! Also defines the [`OneHotIndex`] trait for position-index types and the
//! column-sweep Ajtai commit helpers used by the one-hot commit path.

use crate::algebra::fields::wide::HasWide;
use crate::algebra::ring::cyclotomic::WideCyclotomicRing;
use crate::algebra::ring::sparse_challenge::SparseChallenge;
use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::commitment::onehot::{
    inner_ajtai_onehot_wide, map_onehot_to_regular_blocks, map_onehot_to_sparse_blocks,
    RegularOneHotEntry, SparseBlockEntry,
};
use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::utils::flat_matrix::FlatMatrix;
use crate::protocol::commitment::utils::linear::decompose_rows_i8;
use crate::protocol::hachi_poly_ops::helpers::{
    build_decompose_fold_witness, regular_onehot_accumulate, sparse_onehot_accumulate,
};
use crate::protocol::hachi_poly_ops::{CommitInnerWitness, DecomposeFoldWitness, HachiPolyOps};
use crate::{cfg_fold_reduce, cfg_into_iter, cfg_iter, CanonicalField, FieldCore};
use std::marker::PhantomData;

/// Types usable as one-hot position indices.
///
/// Implemented for `u8`, `u16`, `u32`, and `usize`.
pub trait OneHotIndex: Copy + Send + Sync + std::fmt::Debug + 'static {
    /// Convert to `usize` for indexing.
    fn as_usize(self) -> usize;
}

impl OneHotIndex for u8 {
    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl OneHotIndex for u16 {
    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl OneHotIndex for u32 {
    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl OneHotIndex for usize {
    #[inline]
    fn as_usize(self) -> usize {
        self
    }
}

#[derive(Debug, Clone)]
pub(crate) enum OneHotBlocks {
    Regular(Vec<Vec<RegularOneHotEntry>>),
    General(Vec<Vec<SparseBlockEntry>>),
}

/// One-hot polynomial: sparse witness with at most one nonzero field element
/// per chunk of size `onehot_k`.
///
/// Exploits sparsity in all four operations, avoiding inner ring
/// multiplications during commit and decomposing only nonzero monomials.
///
/// Generic over `I`: the index type accepted at construction time. Use `u8`
/// when `onehot_k <= 256` to reduce temporary index storage before the
/// polynomial is converted into its internal sparse layout.
#[derive(Debug, Clone)]
pub struct OneHotPoly<F: FieldCore, const D: usize, I: OneHotIndex = usize> {
    pub(crate) m_vars: usize,
    pub(crate) blocks: OneHotBlocks,
    pub(crate) _marker: PhantomData<(F, I)>,
}

impl<F: FieldCore, const D: usize, I: OneHotIndex> OneHotPoly<F, D, I> {
    /// Build a one-hot polynomial from chunk size and hot-position indices.
    ///
    /// `indices[c]` is the hot position in chunk `c` (`None` for all-zero chunks).
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are inconsistent or any index is out of range.
    pub fn new(
        onehot_k: usize,
        indices: Vec<Option<I>>,
        r_vars: usize,
        m_vars: usize,
    ) -> Result<Self, HachiError> {
        let use_regular_blocks = onehot_k >= D && onehot_k % D == 0;
        let blocks = if use_regular_blocks {
            OneHotBlocks::Regular(map_onehot_to_regular_blocks(
                onehot_k, &indices, r_vars, m_vars, D,
            )?)
        } else {
            OneHotBlocks::General(map_onehot_to_sparse_blocks(
                onehot_k, &indices, r_vars, m_vars, D,
            )?)
        };
        Ok(Self {
            m_vars,
            blocks,
            _marker: PhantomData,
        })
    }

    fn num_blocks(&self) -> usize {
        match &self.blocks {
            OneHotBlocks::Regular(blocks) => blocks.len(),
            OneHotBlocks::General(blocks) => blocks.len(),
        }
    }

    fn total_ring_elems(&self) -> usize {
        self.num_blocks() * (1usize << self.m_vars)
    }

    fn decompose_fold_regular_onehot(
        &self,
        regular_blocks: &[Vec<RegularOneHotEntry>],
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
    ) -> DecomposeFoldWitness<F, D>
    where
        F: CanonicalField,
    {
        let num_blocks = challenges.len().min(regular_blocks.len());
        let modulus = (-F::one()).to_canonical_u128() + 1;

        let coeff_accum_digit0: Vec<[i32; D]> = {
            let _span = tracing::info_span!("onehot_regular_accumulate").entered();
            regular_onehot_accumulate::<D>(regular_blocks, challenges, num_blocks, block_len)
        };

        let coeff_accum = if num_digits == 1 {
            coeff_accum_digit0
        } else {
            let _span = tracing::info_span!("onehot_regular_expand").entered();
            let mut expanded = Vec::with_capacity(block_len * num_digits);
            for coeffs in coeff_accum_digit0 {
                expanded.push(coeffs);
                for _ in 1..num_digits {
                    expanded.push([0i32; D]);
                }
            }
            expanded
        };

        let _span = tracing::info_span!("onehot_regular_convert").entered();
        build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
    }

    fn decompose_fold_sparse_onehot(
        &self,
        sparse_blocks: &[Vec<SparseBlockEntry>],
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
    ) -> DecomposeFoldWitness<F, D>
    where
        F: CanonicalField,
    {
        let inner_width = block_len * num_digits;
        let num_blocks = challenges.len().min(sparse_blocks.len());
        let modulus = (-F::one()).to_canonical_u128() + 1;

        let coeff_accum = {
            let _span = tracing::info_span!("onehot_sparse_accumulate").entered();
            sparse_onehot_accumulate::<D>(
                sparse_blocks,
                challenges,
                num_blocks,
                inner_width,
                num_digits,
            )
        };

        let _span = tracing::info_span!("onehot_sparse_convert").entered();
        build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
    }
}

impl<F, const D: usize, I: OneHotIndex> HachiPolyOps<F, D> for OneHotPoly<F, D, I>
where
    F: FieldCore + CanonicalField + HasWide,
{
    type CommitCache = NttSlotCache<D>;

    fn num_ring_elems(&self) -> usize {
        self.total_ring_elems()
    }

    fn evaluate_ring(&self, scalars: &[F]) -> CyclotomicRing<F, D> {
        let block_len = 1usize << self.m_vars;
        match &self.blocks {
            OneHotBlocks::Regular(blocks) => cfg_fold_reduce!(
                0..blocks.len(),
                || CyclotomicRing::<F, D>::zero(),
                |mut acc: CyclotomicRing<F, D>, block_idx: usize| {
                    let block_offset = block_idx * block_len;
                    for entry in &blocks[block_idx] {
                        let ring_idx = block_offset + entry.pos_in_block();
                        if ring_idx < scalars.len() {
                            acc.coeffs[entry.coeff_idx()] += scalars[ring_idx];
                        }
                    }
                    acc
                },
                |a, b| a + b
            ),
            OneHotBlocks::General(blocks) => cfg_fold_reduce!(
                0..blocks.len(),
                || CyclotomicRing::<F, D>::zero(),
                |mut acc: CyclotomicRing<F, D>, block_idx: usize| {
                    let block_offset = block_idx * block_len;
                    for entry in &blocks[block_idx] {
                        let ring_idx = block_offset + entry.pos_in_block;
                        if ring_idx < scalars.len() {
                            let s = scalars[ring_idx];
                            for &ci in &entry.nonzero_coeffs {
                                acc.coeffs[ci] += s;
                            }
                        }
                    }
                    acc
                },
                |a, b| a + b
            ),
        }
    }

    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>> {
        match &self.blocks {
            OneHotBlocks::Regular(blocks) => cfg_iter!(blocks)
                .map(|entries| {
                    let mut coeffs_acc = [F::zero(); D];
                    for entry in entries {
                        let pos = entry.pos_in_block();
                        if pos < scalars.len() && pos < block_len {
                            coeffs_acc[entry.coeff_idx()] += scalars[pos];
                        }
                    }
                    CyclotomicRing::from_coefficients(coeffs_acc)
                })
                .collect(),
            OneHotBlocks::General(blocks) => cfg_iter!(blocks)
                .map(|entries| {
                    let mut coeffs_acc = [F::zero(); D];
                    for entry in entries {
                        if entry.pos_in_block < scalars.len() && entry.pos_in_block < block_len {
                            let s = scalars[entry.pos_in_block];
                            for &ci in &entry.nonzero_coeffs {
                                coeffs_acc[ci] += s;
                            }
                        }
                    }
                    CyclotomicRing::from_coefficients(coeffs_acc)
                })
                .collect(),
        }
    }

    #[tracing::instrument(skip_all, name = "OneHotPoly::decompose_fold")]
    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> DecomposeFoldWitness<F, D> {
        match &self.blocks {
            OneHotBlocks::Regular(blocks) => {
                self.decompose_fold_regular_onehot(blocks, challenges, block_len, num_digits)
            }
            OneHotBlocks::General(blocks) => {
                self.decompose_fold_sparse_onehot(blocks, challenges, block_len, num_digits)
            }
        }
    }

    #[tracing::instrument(skip_all, name = "OneHotPoly::commit_inner")]
    fn commit_inner(
        &self,
        a_matrix: &FlatMatrix<F>,
        _ntt_a: &NttSlotCache<D>,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<Vec<Vec<[i8; D]>>, HachiError> {
        let a_view = a_matrix.view::<D>();
        let n_a = a_view.num_rows();
        let num_blocks = self.num_blocks();
        let active_a_cols = num_cols_a(block_len, num_digits_commit)?;
        if active_a_cols > a_view.num_cols() {
            return Err(HachiError::InvalidSetup(format!(
                "active A width {active_a_cols} exceeds setup envelope {}",
                a_view.num_cols()
            )));
        }
        let zero_block_len = n_a.checked_mul(num_digits_open).unwrap();

        let t_all = match &self.blocks {
            OneHotBlocks::Regular(blocks) => onehot_column_sweep_ajtai_regular::<F, D>(
                &a_view,
                blocks,
                n_a,
                active_a_cols,
                num_digits_commit,
            ),
            OneHotBlocks::General(blocks) => onehot_column_sweep_ajtai::<F, D>(
                &a_view,
                blocks,
                n_a,
                active_a_cols,
                num_digits_commit,
            ),
        };

        let t_hat_all: Vec<Vec<[i8; D]>> = cfg_into_iter!(0..num_blocks)
            .map(|b| {
                if t_all[b].iter().all(|r| *r == CyclotomicRing::zero()) {
                    vec![[0i8; D]; zero_block_len]
                } else {
                    decompose_rows_i8(&t_all[b], num_digits_open, log_basis)
                }
            })
            .collect();

        Ok(t_hat_all)
    }

    #[tracing::instrument(skip_all, name = "OneHotPoly::commit_inner_witness")]
    fn commit_inner_witness(
        &self,
        a_matrix: &FlatMatrix<F>,
        _ntt_a: &NttSlotCache<D>,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<CommitInnerWitness<F, D>, HachiError> {
        let a_view = a_matrix.view::<D>();
        let n_a = a_view.num_rows();
        let active_a_cols = num_cols_a(block_len, num_digits_commit)?;
        if active_a_cols > a_view.num_cols() {
            return Err(HachiError::InvalidSetup(format!(
                "active A width {active_a_cols} exceeds setup envelope {}",
                a_view.num_cols()
            )));
        }
        let zero_block_len = n_a.checked_mul(num_digits_open).unwrap();

        let t = match &self.blocks {
            OneHotBlocks::Regular(blocks) => onehot_column_sweep_ajtai_regular::<F, D>(
                &a_view,
                blocks,
                n_a,
                active_a_cols,
                num_digits_commit,
            ),
            OneHotBlocks::General(blocks) => onehot_column_sweep_ajtai::<F, D>(
                &a_view,
                blocks,
                n_a,
                active_a_cols,
                num_digits_commit,
            ),
        };

        let t_hat: Vec<Vec<[i8; D]>> = cfg_iter!(t)
            .map(|t_i| {
                if t_i.iter().all(|r| *r == CyclotomicRing::zero()) {
                    vec![[0i8; D]; zero_block_len]
                } else {
                    decompose_rows_i8(t_i, num_digits_open, log_basis)
                }
            })
            .collect();

        Ok(CommitInnerWitness { t, t_hat })
    }
}

fn num_cols_a(block_len: usize, num_digits_commit: usize) -> Result<usize, HachiError> {
    block_len
        .checked_mul(num_digits_commit)
        .ok_or_else(|| HachiError::InvalidSetup("active A width overflow".to_string()))
}

fn inner_ajtai_regular_onehot_wide<F, const D: usize>(
    a_view: &crate::protocol::commitment::utils::flat_matrix::RingMatrixView<'_, F, D>,
    regular_entries: &[RegularOneHotEntry],
    num_digits: usize,
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: crate::AdditiveGroup + From<F> + crate::algebra::fields::wide::ReduceTo<F>,
{
    let n_a = a_view.num_rows();
    let mut t_wide = vec![WideCyclotomicRing::<F::Wide, D>::zero(); n_a];

    for entry in regular_entries {
        let col = entry.pos_in_block() * num_digits;
        let coeff_idx = entry.coeff_idx();
        for (a_idx, t_w) in t_wide.iter_mut().enumerate() {
            let a_wide = WideCyclotomicRing::from_ring(&a_view.row(a_idx)[col]);
            a_wide.shift_accumulate_into(t_w, coeff_idx);
        }
    }

    t_wide.into_iter().map(|w| w.reduce()).collect()
}

/// L2 cache budget (in bytes) for the tile of wide accumulators in the
/// column-sweep commit.  Each tile's `accums` allocation is capped to this
/// size so the scatter loop stays L2-resident.
///
/// 2 MB is a conservative middle ground: fits in Apple M-series L2
/// (~4 MB/core) and exceeds most x86 per-core L2 (~256 KB–1 MB) only
/// modestly, relying on the shared L3 backstop.
///
// TODO: benchmark column-sweep on x86 vs ARM with budget values
// (512 KB, 1 MB, 2 MB, 4 MB) at production configs to determine
// whether a smaller or arch-specific budget helps on x86.
const L2_TILE_BUDGET: usize = 1 << 21;

/// Minimum blocks-per-thread required before enabling the column-sweep kernel.
const SWEEP_THRESHOLD: usize = 32;

/// Column-sweep Ajtai commitment for regular one-hot blocks.
///
/// Two-level tiling: threads partition blocks evenly (outer, for parallelism),
/// then within each thread, blocks are processed in L2-sized tiles (inner,
/// for cache locality).  For each tile the entries are bucketed by A-column
/// so each column is loaded and widened exactly once, before scattering into
/// L2-resident block accumulators.
///
/// Falls back to the original block-by-block path when blocks_per_thread is
/// small enough that accumulators already fit in L2.
fn onehot_column_sweep_ajtai_regular<F, const D: usize>(
    a_view: &crate::protocol::commitment::utils::flat_matrix::RingMatrixView<'_, F, D>,
    regular_blocks: &[Vec<RegularOneHotEntry>],
    n_a: usize,
    active_a_cols: usize,
    num_digits_commit: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: crate::AdditiveGroup + From<F> + crate::algebra::fields::wide::ReduceTo<F>,
{
    let num_blocks = regular_blocks.len();
    debug_assert!(
        active_a_cols <= a_view.num_cols(),
        "active A width exceeds setup envelope"
    );
    let num_cols = active_a_cols;

    let accum_bytes = n_a * D * std::mem::size_of::<F::Wide>();
    let block_tile = if accum_bytes > 0 {
        (L2_TILE_BUDGET / accum_bytes).max(1)
    } else {
        num_blocks
    };

    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads().min(num_blocks).max(1);
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let blocks_per_thread = num_blocks.div_ceil(num_threads);

    if blocks_per_thread <= SWEEP_THRESHOLD {
        return cfg_iter!(regular_blocks)
            .map(|block_entries| {
                inner_ajtai_regular_onehot_wide(a_view, block_entries, num_digits_commit)
            })
            .collect();
    }

    let thread_results: Vec<Vec<Vec<CyclotomicRing<F, D>>>> = cfg_into_iter!(0..num_threads)
        .map(|tid| {
            let block_start = tid * blocks_per_thread;
            let block_end = (block_start + blocks_per_thread).min(num_blocks);
            if block_start >= block_end {
                return Vec::new();
            }
            let my_blocks = &regular_blocks[block_start..block_end];
            let my_count = my_blocks.len();

            let mut result: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(my_count);
            result.resize_with(my_count, Vec::new);

            // Reuse across tiles so that Vec capacities from earlier tiles
            // carry over, avoiding repeated heap growth.
            let mut col_entries: Vec<Vec<(u32, u16)>> = vec![Vec::new(); num_cols];

            for tile_start in (0..my_count).step_by(block_tile) {
                let tile_end = (tile_start + block_tile).min(my_count);
                let tile_blocks = &my_blocks[tile_start..tile_end];
                let tile_len = tile_blocks.len();

                for (local_b, block_entries) in tile_blocks.iter().enumerate() {
                    for entry in block_entries {
                        let col = entry.pos_in_block() * num_digits_commit;
                        col_entries[col].push((local_b as u32, entry.coeff_idx() as u16));
                    }
                }

                let mut accums: Vec<Vec<WideCyclotomicRing<F::Wide, D>>> = (0..tile_len)
                    .map(|_| vec![WideCyclotomicRing::zero(); n_a])
                    .collect();

                for a_idx in 0..n_a {
                    let a_row = a_view.row(a_idx);
                    for (col, entries) in col_entries.iter().enumerate() {
                        if entries.is_empty() {
                            continue;
                        }
                        let a_wide = WideCyclotomicRing::from_ring(&a_row[col]);
                        for &(lb, ci) in entries {
                            a_wide.shift_accumulate_into(
                                &mut accums[lb as usize][a_idx],
                                ci as usize,
                            );
                        }
                    }
                }

                for (local_b, row_accums) in accums.into_iter().enumerate() {
                    result[tile_start + local_b] =
                        row_accums.into_iter().map(|w| w.reduce()).collect();
                }

                for bucket in &mut col_entries {
                    bucket.clear();
                }
            }

            result
        })
        .collect();

    let mut out: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(num_blocks);
    for thread_blocks in thread_results {
        out.extend(thread_blocks);
    }
    out
}

/// Column-sweep Ajtai commitment for one-hot sparse blocks.
///
/// Two-level tiling: threads partition blocks evenly (outer, for parallelism),
/// then within each thread, blocks are processed in L2-sized tiles (inner,
/// for cache locality). For each tile the entries are bucketed by A-column
/// so each column is loaded and widened exactly once, before scattering into
/// L2-resident block accumulators.
///
/// Falls back to the original block-by-block path when blocks_per_thread is
/// small enough that accumulators already fit in L2.
fn onehot_column_sweep_ajtai<F, const D: usize>(
    a_view: &crate::protocol::commitment::utils::flat_matrix::RingMatrixView<'_, F, D>,
    sparse_blocks: &[Vec<SparseBlockEntry>],
    n_a: usize,
    active_a_cols: usize,
    num_digits_commit: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: crate::AdditiveGroup + From<F> + crate::algebra::fields::wide::ReduceTo<F>,
{
    let num_blocks = sparse_blocks.len();
    debug_assert!(
        active_a_cols <= a_view.num_cols(),
        "active A width exceeds setup envelope"
    );
    let num_cols = active_a_cols;

    let accum_bytes = n_a * D * std::mem::size_of::<F::Wide>();
    let block_tile = if accum_bytes > 0 {
        (L2_TILE_BUDGET / accum_bytes).max(1)
    } else {
        num_blocks
    };

    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads().min(num_blocks).max(1);
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let blocks_per_thread = num_blocks.div_ceil(num_threads);

    if blocks_per_thread <= SWEEP_THRESHOLD {
        return cfg_iter!(sparse_blocks)
            .map(|block_entries| {
                inner_ajtai_onehot_wide(a_view, block_entries, 0, num_digits_commit)
            })
            .collect();
    }

    let thread_results: Vec<Vec<Vec<CyclotomicRing<F, D>>>> = cfg_into_iter!(0..num_threads)
        .map(|tid| {
            let block_start = tid * blocks_per_thread;
            let block_end = (block_start + blocks_per_thread).min(num_blocks);
            if block_start >= block_end {
                return Vec::new();
            }
            let my_blocks = &sparse_blocks[block_start..block_end];
            let my_count = my_blocks.len();

            let mut result: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(my_count);
            result.resize_with(my_count, Vec::new);

            let mut col_entries: Vec<Vec<(u32, u8)>> = vec![Vec::new(); num_cols];

            for tile_start in (0..my_count).step_by(block_tile) {
                let tile_end = (tile_start + block_tile).min(my_count);
                let tile_blocks = &my_blocks[tile_start..tile_end];
                let tile_len = tile_blocks.len();

                for (local_b, block_entries) in tile_blocks.iter().enumerate() {
                    for entry in block_entries {
                        let col = entry.pos_in_block * num_digits_commit;
                        for &ci in &entry.nonzero_coeffs {
                            col_entries[col].push((local_b as u32, ci as u8));
                        }
                    }
                }

                let mut accums: Vec<Vec<WideCyclotomicRing<F::Wide, D>>> = (0..tile_len)
                    .map(|_| vec![WideCyclotomicRing::zero(); n_a])
                    .collect();

                for a_idx in 0..n_a {
                    let a_row = a_view.row(a_idx);
                    for (col, entries) in col_entries.iter().enumerate() {
                        if entries.is_empty() {
                            continue;
                        }
                        let a_wide = WideCyclotomicRing::from_ring(&a_row[col]);
                        for &(lb, ci) in entries {
                            a_wide.shift_accumulate_into(
                                &mut accums[lb as usize][a_idx],
                                ci as usize,
                            );
                        }
                    }
                }

                for (local_b, row_accums) in accums.into_iter().enumerate() {
                    result[tile_start + local_b] =
                        row_accums.into_iter().map(|w| w.reduce()).collect();
                }

                for bucket in &mut col_entries {
                    bucket.clear();
                }
            }

            result
        })
        .collect();

    let mut out: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(num_blocks);
    for thread_blocks in thread_results {
        out.extend(thread_blocks);
    }
    out
}
