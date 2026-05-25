//! Sparse signed ring-coefficient polynomial backend.
//!
//! This is the natural backend for Frobenius-packed one-hot tables: after
//! canonical-basis packing, each original one-hot chunk becomes a small number
//! of signed monomial coefficients inside the committed ring table.

use akita_algebra::ring::cyclotomic::WideCyclotomicRing;
use akita_algebra::CyclotomicRing;
use akita_challenges::IntegerChallenge;
use akita_field::fields::wide::{HasWide, ReduceTo};
use akita_field::parallel::*;
use akita_field::{AdditiveGroup, AkitaError, CanonicalField, FieldCore, FromPrimitiveInt};
use akita_types::{DirectWitnessProof, FlatDigitBlocks, FlatMatrix, FlatRingVec};
use std::sync::OnceLock;

use crate::backend::poly_helpers::{build_decompose_fold_witness, fill_rotated_challenge};
use crate::kernels::crt_ntt::NttSlotCache;
use crate::kernels::linear::decompose_rows_i8_into;
use crate::{AkitaPolyOps, CenteredCoeff, CommitInnerWitness, DecomposeFoldWitness};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SparseRingCoeff {
    ring_idx: u32,
    coeff_idx: u16,
    value: i8,
}

impl SparseRingCoeff {
    pub(crate) fn new(ring_idx: usize, coeff_idx: usize, value: i8) -> Result<Self, AkitaError> {
        if value == 0 {
            return Err(AkitaError::InvalidInput(
                "invalid sparse ring coefficient".to_string(),
            ));
        }
        Ok(Self {
            ring_idx: u32::try_from(ring_idx).map_err(|_| {
                AkitaError::InvalidInput("sparse ring index exceeds u32".to_string())
            })?,
            coeff_idx: u16::try_from(coeff_idx).map_err(|_| {
                AkitaError::InvalidInput("sparse coefficient index exceeds u16".to_string())
            })?,
            value,
        })
    }

    #[inline]
    fn sort_key(self) -> (u32, u16, i8) {
        (self.ring_idx, self.coeff_idx, self.value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SparseRingBlockEntry {
    pos_in_block: u32,
    coeff_idx: u16,
    value: i8,
}

impl SparseRingBlockEntry {
    #[inline]
    fn pos_in_block(self) -> usize {
        self.pos_in_block as usize
    }

    #[inline]
    fn coeff_idx(self) -> usize {
        self.coeff_idx as usize
    }
}

#[derive(Debug, Clone)]
struct SparseRingBlocks {
    entries: Vec<SparseRingBlockEntry>,
    offsets: Vec<u32>,
}

impl SparseRingBlocks {
    fn from_coeffs(
        coeffs: &[SparseRingCoeff],
        total_ring_elems: usize,
        block_len: usize,
    ) -> Result<Self, AkitaError> {
        if block_len == 0 || !block_len.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "block_len={block_len} must be a nonzero power of two"
            )));
        }
        if !total_ring_elems.is_multiple_of(block_len) {
            return Err(AkitaError::InvalidSize {
                expected: total_ring_elems,
                actual: block_len,
            });
        }
        if u32::try_from(block_len).is_err() {
            return Err(AkitaError::InvalidInput(format!(
                "block_len={block_len} exceeds u32::MAX"
            )));
        }
        let num_blocks = total_ring_elems / block_len;
        let mut offsets = Vec::with_capacity(num_blocks + 1);
        let mut entries = Vec::with_capacity(coeffs.len());
        offsets.push(0);
        let mut current_block = 0usize;
        for coeff in coeffs {
            let ring_idx = coeff.ring_idx as usize;
            if ring_idx >= total_ring_elems {
                return Err(AkitaError::InvalidInput(
                    "sparse ring coefficient index out of range".to_string(),
                ));
            }
            let block_idx = ring_idx / block_len;
            while current_block < block_idx {
                offsets.push(entries.len() as u32);
                current_block += 1;
            }
            entries.push(SparseRingBlockEntry {
                pos_in_block: (ring_idx % block_len) as u32,
                coeff_idx: coeff.coeff_idx,
                value: coeff.value,
            });
        }
        while current_block < num_blocks {
            offsets.push(entries.len() as u32);
            current_block += 1;
        }
        Ok(Self { entries, offsets })
    }

    #[inline]
    fn num_blocks(&self) -> usize {
        self.offsets.len() - 1
    }

    #[inline]
    fn block(&self, idx: usize) -> &[SparseRingBlockEntry] {
        let lo = self.offsets[idx] as usize;
        let hi = self.offsets[idx + 1] as usize;
        &self.entries[lo..hi]
    }
}

/// Sparse polynomial whose ring coefficients are small signed monomials.
#[derive(Debug, Clone)]
pub struct SparseRingPoly<F: FieldCore, const D: usize> {
    num_vars: usize,
    total_ring_elems: usize,
    coeffs: Vec<SparseRingCoeff>,
    block_cache: OnceLock<(usize, SparseRingBlocks)>,
    _marker: core::marker::PhantomData<F>,
}

impl<F: FieldCore, const D: usize> SparseRingPoly<F, D> {
    /// Build from `(ring_idx, coeff_idx, value)` triples.
    ///
    /// # Errors
    ///
    /// Returns an error when `D` cannot be represented by the sparse block
    /// format, the expected ring-element count does not match `num_vars`, or a
    /// supplied coefficient triple is out of range.
    pub fn from_signed_coeffs(
        num_vars: usize,
        total_ring_elems: usize,
        coeffs: Vec<(usize, usize, i8)>,
    ) -> Result<Self, AkitaError> {
        Self::from_signed_coeffs_with_order(num_vars, total_ring_elems, coeffs, false)
    }

    /// Build from `(ring_idx, coeff_idx, value)` triples already sorted by
    /// `(ring_idx, coeff_idx, value)`.
    ///
    /// # Errors
    ///
    /// Returns an error for the same malformed inputs as
    /// [`Self::from_signed_coeffs`], and also when the supplied triples are not
    /// sorted.
    pub fn from_sorted_signed_coeffs(
        num_vars: usize,
        total_ring_elems: usize,
        coeffs: Vec<(usize, usize, i8)>,
    ) -> Result<Self, AkitaError> {
        Self::from_signed_coeffs_with_order(num_vars, total_ring_elems, coeffs, true)
    }

    /// Build from compact sparse coefficient triples.
    ///
    /// # Errors
    ///
    /// Returns an error for the same malformed inputs as
    /// [`Self::from_signed_coeffs`].
    pub(crate) fn from_packed_coeffs(
        num_vars: usize,
        total_ring_elems: usize,
        coeffs: Vec<SparseRingCoeff>,
    ) -> Result<Self, AkitaError> {
        Self::from_packed_coeffs_with_order(num_vars, total_ring_elems, coeffs, false)
    }

    /// Build from compact sparse coefficient triples already sorted by
    /// `(ring_idx, coeff_idx, value)`.
    ///
    /// # Errors
    ///
    /// Returns an error for the same malformed inputs as
    /// [`Self::from_sorted_signed_coeffs`].
    pub(crate) fn from_sorted_packed_coeffs(
        num_vars: usize,
        total_ring_elems: usize,
        coeffs: Vec<SparseRingCoeff>,
    ) -> Result<Self, AkitaError> {
        Self::from_packed_coeffs_with_order(num_vars, total_ring_elems, coeffs, true)
    }

    fn from_signed_coeffs_with_order(
        num_vars: usize,
        total_ring_elems: usize,
        coeffs: Vec<(usize, usize, i8)>,
        already_sorted: bool,
    ) -> Result<Self, AkitaError> {
        let mut packed = Vec::with_capacity(coeffs.len());
        for (ring_idx, coeff_idx, value) in coeffs {
            packed.push(SparseRingCoeff::new(ring_idx, coeff_idx, value)?);
        }
        Self::from_packed_coeffs_with_order(num_vars, total_ring_elems, packed, already_sorted)
    }

    fn from_packed_coeffs_with_order(
        num_vars: usize,
        total_ring_elems: usize,
        mut packed: Vec<SparseRingCoeff>,
        already_sorted: bool,
    ) -> Result<Self, AkitaError> {
        if D > usize::from(u16::MAX) + 1 {
            return Err(AkitaError::InvalidInput(format!(
                "D={D} exceeds sparse coefficient index capacity"
            )));
        }
        let expected_ring_elems = 1usize
            .checked_shl(num_vars as u32)
            .ok_or_else(|| AkitaError::InvalidInput("sparse arity overflow".to_string()))?
            .checked_div(D)
            .ok_or_else(|| AkitaError::InvalidInput("D must be nonzero".to_string()))?;
        if expected_ring_elems != total_ring_elems {
            return Err(AkitaError::InvalidSize {
                expected: expected_ring_elems,
                actual: total_ring_elems,
            });
        }
        let mut previous_key = None;
        for entry in &packed {
            if entry.ring_idx as usize >= total_ring_elems
                || entry.coeff_idx as usize >= D
                || entry.value == 0
            {
                return Err(AkitaError::InvalidInput(
                    "invalid sparse ring coefficient".to_string(),
                ));
            }
            let key = entry.sort_key();
            if already_sorted && previous_key.is_some_and(|previous| key < previous) {
                return Err(AkitaError::InvalidInput(
                    "sorted sparse ring constructor received unsorted coefficients".to_string(),
                ));
            }
            previous_key = Some(key);
        }
        if !already_sorted {
            packed.sort_unstable_by_key(|entry| entry.sort_key());
        }
        Ok(Self {
            num_vars,
            total_ring_elems,
            coeffs: packed,
            block_cache: OnceLock::new(),
            _marker: core::marker::PhantomData,
        })
    }

    fn blocks_for(&self, block_len: usize) -> Result<&SparseRingBlocks, AkitaError> {
        if let Some((cached_len, blocks)) = self.block_cache.get() {
            if *cached_len == block_len {
                return Ok(blocks);
            }
            return Err(AkitaError::InvalidInput(format!(
                "SparseRingPoly was first used with block_len={cached_len} but is now used with block_len={block_len}"
            )));
        }
        let (_, blocks) = self.block_cache.get_or_init(|| {
            let blocks =
                SparseRingBlocks::from_coeffs(&self.coeffs, self.total_ring_elems, block_len)
                    .expect("block_len validation is deterministic");
            (block_len, blocks)
        });
        Ok(blocks)
    }
}

impl<F, const D: usize> AkitaPolyOps<F, D> for SparseRingPoly<F, D>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    type CommitCache = NttSlotCache<D>;

    fn num_ring_elems(&self) -> usize {
        self.total_ring_elems
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>> {
        let blocks = self
            .blocks_for(block_len)
            .expect("SparseRingPoly::fold_blocks: invalid block_len");
        cfg_into_iter!(0..blocks.num_blocks())
            .map(|block_idx| fold_sparse_block(blocks.block(block_idx), scalars, block_len))
            .collect()
    }

    fn fold_blocks_ring(
        &self,
        scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let blocks = self
            .blocks_for(block_len)
            .expect("SparseRingPoly::fold_blocks_ring: invalid block_len");
        cfg_into_iter!(0..blocks.num_blocks())
            .map(|block_idx| fold_sparse_block_ring(blocks.block(block_idx), scalars, block_len))
            .collect()
    }

    fn evaluate_and_fold_ring(
        &self,
        eval_outer_scalars: &[CyclotomicRing<F, D>],
        fold_scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        let folded = self.fold_blocks_ring(fold_scalars, block_len);
        let mut eval = CyclotomicRing::<F, D>::zero();
        for (f_i, s_i) in folded.iter().zip(eval_outer_scalars.iter()) {
            f_i.mul_accumulate_sparse_rhs_into(s_i, &mut eval);
        }
        (eval, folded)
    }

    #[tracing::instrument(skip_all, name = "SparseRingPoly::decompose_fold")]
    fn decompose_fold(
        &self,
        challenges: &[IntegerChallenge],
        block_len: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> DecomposeFoldWitness<F, D> {
        let blocks = self
            .blocks_for(block_len)
            .expect("SparseRingPoly::decompose_fold: invalid block_len");
        let num_blocks = challenges.len().min(blocks.num_blocks());
        let inner_width = block_len * num_digits;
        let coeff_accum =
            sparse_accumulate::<D>(blocks, challenges, num_blocks, inner_width, num_digits);
        let modulus = (-F::one()).to_canonical_u128() + 1;
        build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
    }

    #[tracing::instrument(skip_all, name = "SparseRingPoly::commit_inner")]
    fn commit_inner(
        &self,
        a_matrix: &FlatMatrix<F>,
        _ntt_a: &NttSlotCache<D>,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
        matrix_stride: usize,
    ) -> Result<FlatDigitBlocks<D>, AkitaError> {
        let t =
            self.commit_inner_rows(a_matrix, n_a, block_len, num_digits_commit, matrix_stride)?;
        decompose_commit_rows::<F, D>(&t, n_a, num_digits_open, log_basis)
    }

    #[tracing::instrument(skip_all, name = "SparseRingPoly::commit_inner_witness")]
    fn commit_inner_witness(
        &self,
        a_matrix: &FlatMatrix<F>,
        _ntt_a: &NttSlotCache<D>,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
        matrix_stride: usize,
    ) -> Result<CommitInnerWitness<F, D>, AkitaError> {
        let t =
            self.commit_inner_rows(a_matrix, n_a, block_len, num_digits_commit, matrix_stride)?;
        let decomposed_inner_rows =
            decompose_commit_rows::<F, D>(&t, n_a, num_digits_open, log_basis)?;
        Ok(CommitInnerWitness {
            recomposed_inner_rows: t,
            decomposed_inner_rows,
        })
    }

    fn direct_root_witness(&self) -> Result<DirectWitnessProof<F>, AkitaError> {
        let total_coeffs = self.total_ring_elems.checked_mul(D).ok_or_else(|| {
            AkitaError::InvalidInput("sparse direct witness length overflow".to_string())
        })?;
        let mut coeffs = vec![F::zero(); total_coeffs];
        for entry in &self.coeffs {
            let idx = (entry.ring_idx as usize)
                .checked_mul(D)
                .and_then(|base| base.checked_add(entry.coeff_idx as usize))
                .ok_or_else(|| {
                    AkitaError::InvalidInput("sparse direct witness index overflow".to_string())
                })?;
            coeffs[idx] += F::from_i8(entry.value);
        }
        Ok(DirectWitnessProof::FieldElements(FlatRingVec::from_coeffs(
            coeffs,
        )))
    }
}

impl<F, const D: usize> SparseRingPoly<F, D>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    fn commit_inner_rows(
        &self,
        a_matrix: &FlatMatrix<F>,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
        matrix_stride: usize,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        let blocks = self.blocks_for(block_len)?;
        let a_view = a_matrix.ring_view::<D>(n_a, matrix_stride)?;
        let active_a_cols = block_len
            .checked_mul(num_digits_commit)
            .ok_or_else(|| AkitaError::InvalidSetup("active A width overflow".to_string()))?;
        if active_a_cols > a_view.num_cols() {
            return Err(AkitaError::InvalidSetup(format!(
                "active A width {active_a_cols} exceeds setup envelope {}",
                a_view.num_cols()
            )));
        }
        let block_views = (0..blocks.num_blocks())
            .map(|idx| blocks.block(idx))
            .collect::<Vec<_>>();
        let a_rows = (0..n_a)
            .map(|idx| a_view.row(idx))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(column_sweep_sparse(
            &a_rows,
            &block_views,
            n_a,
            block_len,
            num_digits_commit,
        ))
    }
}

fn fold_sparse_block<F, const D: usize>(
    entries: &[SparseRingBlockEntry],
    scalars: &[F],
    block_len: usize,
) -> CyclotomicRing<F, D>
where
    F: FieldCore + FromPrimitiveInt,
{
    let mut coeffs = [F::zero(); D];
    for entry in entries {
        let pos = entry.pos_in_block();
        if pos < scalars.len() && pos < block_len {
            coeffs[entry.coeff_idx()] += scalars[pos] * F::from_i8(entry.value);
        }
    }
    CyclotomicRing::from_coefficients(coeffs)
}

fn fold_sparse_block_ring<F, const D: usize>(
    entries: &[SparseRingBlockEntry],
    scalars: &[CyclotomicRing<F, D>],
    block_len: usize,
) -> CyclotomicRing<F, D>
where
    F: FieldCore + FromPrimitiveInt,
{
    let mut acc = CyclotomicRing::<F, D>::zero();
    for entry in entries {
        let pos = entry.pos_in_block();
        if pos < scalars.len() && pos < block_len {
            match entry.value {
                1 => scalars[pos].shift_accumulate_into(&mut acc, entry.coeff_idx()),
                -1 => scalars[pos].shift_sub_into(&mut acc, entry.coeff_idx()),
                value => scalars[pos].shift_scale_accumulate_into(
                    &mut acc,
                    entry.coeff_idx(),
                    F::from_i8(value),
                ),
            }
        }
    }
    acc
}

fn sparse_accumulate<const D: usize>(
    blocks: &SparseRingBlocks,
    challenges: &[IntegerChallenge],
    num_blocks: usize,
    inner_width: usize,
    num_digits: usize,
) -> Vec<[CenteredCoeff; D]> {
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(inner_width.max(1));
    let pos_chunk = inner_width.div_ceil(actual_threads);
    let chunks: Vec<Vec<[CenteredCoeff; D]>> = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let pos_start = tid * pos_chunk;
            if pos_start >= inner_width {
                return Vec::new();
            }
            let pos_end = (pos_start + pos_chunk).min(inner_width);
            let mut acc = vec![[0 as CenteredCoeff; D]; pos_end - pos_start];
            let mut rotated = vec![[0 as CenteredCoeff; D]; D];

            for (block_idx, challenge) in challenges.iter().enumerate().take(num_blocks) {
                let entries = blocks.block(block_idx);
                let lo = entries.partition_point(|e| e.pos_in_block() * num_digits < pos_start);
                let hi = entries.partition_point(|e| e.pos_in_block() * num_digits < pos_end);
                if lo >= hi {
                    continue;
                }
                fill_rotated_challenge::<D>(&mut rotated, challenge);
                for entry in &entries[lo..hi] {
                    let local_pos = entry.pos_in_block() * num_digits - pos_start;
                    let rot = &rotated[entry.coeff_idx()];
                    let dst = &mut acc[local_pos];
                    let weight = CenteredCoeff::from(entry.value);
                    for k in 0..D {
                        dst[k] += weight * rot[k];
                    }
                }
            }
            acc
        })
        .collect();
    chunks.into_iter().flatten().collect()
}

type WeightedColEntry = (usize, u32, u16, i8);
type WeightedPosEntry = (u32, u16, i8);
const L2_TILE_BUDGET: usize = 1 << 21;

#[inline]
fn shift_signed_unit_into<W, const D: usize>(
    src: &WideCyclotomicRing<W, D>,
    dst: &mut WideCyclotomicRing<W, D>,
    coeff_idx: u16,
    value: i8,
) where
    W: AdditiveGroup,
{
    match value {
        1 => src.shift_accumulate_into(dst, coeff_idx as usize),
        -1 => src.shift_sub_into(dst, coeff_idx as usize),
        _ => unreachable!("sparse Frobenius coefficients are signed units"),
    }
}

fn column_sweep_sparse<F, const D: usize>(
    a_rows: &[&[CyclotomicRing<F, D>]],
    blocks: &[&[SparseRingBlockEntry]],
    n_a: usize,
    block_len: usize,
    num_digits_commit: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    let num_blocks = blocks.len();
    let accum_bytes = n_a * D * std::mem::size_of::<F::Wide>();
    let block_tile = L2_TILE_BUDGET
        .checked_div(accum_bytes)
        .map_or(num_blocks, |tile| tile.max(1));

    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads().min(num_blocks).max(1);
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;
    let blocks_per_thread = num_blocks.div_ceil(num_threads);

    let thread_results: Vec<Vec<Vec<CyclotomicRing<F, D>>>> = cfg_into_iter!(0..num_threads)
        .map(|tid| {
            let block_start = tid * blocks_per_thread;
            let block_end = (block_start + blocks_per_thread).min(num_blocks);
            if block_start >= block_end {
                return Vec::new();
            }
            let my_count = block_end - block_start;
            let mut result = Vec::with_capacity(my_count);
            result.resize_with(my_count, Vec::new);
            let mut col_entries: Vec<WeightedColEntry> = Vec::new();
            let mut pos_offsets: Vec<usize> = Vec::new();
            let mut pos_cursor: Vec<usize> = Vec::new();
            let mut pos_entries: Vec<WeightedPosEntry> = Vec::new();

            for tile_start in (0..my_count).step_by(block_tile) {
                let tile_end = (tile_start + block_tile).min(my_count);
                let tile_len = tile_end - tile_start;
                let mut accums: Vec<Vec<WideCyclotomicRing<F::Wide, D>>> = (0..tile_len)
                    .map(|_| vec![WideCyclotomicRing::zero(); n_a])
                    .collect();

                let tile_blocks = &blocks[(block_start + tile_start)..(block_start + tile_end)];
                let entry_count = tile_blocks
                    .iter()
                    .map(|entries| entries.len())
                    .sum::<usize>();
                // Dense tiles are cheaper to bucket by block position than to
                // comparison-sort by A-column.
                if entry_count >= block_len {
                    pos_offsets.clear();
                    pos_offsets.resize(block_len + 1, 0);
                    for block_entries in tile_blocks {
                        for entry in *block_entries {
                            pos_offsets[entry.pos_in_block() + 1] += 1;
                        }
                    }
                    for pos in 1..=block_len {
                        pos_offsets[pos] += pos_offsets[pos - 1];
                    }

                    pos_entries.clear();
                    pos_entries.resize(entry_count, (0, 0, 0));
                    pos_cursor.clear();
                    pos_cursor.extend_from_slice(&pos_offsets[..block_len]);
                    for (local_b, block_entries) in tile_blocks.iter().enumerate() {
                        for entry in *block_entries {
                            let pos = entry.pos_in_block();
                            let dst = pos_cursor[pos];
                            pos_cursor[pos] += 1;
                            pos_entries[dst] = (local_b as u32, entry.coeff_idx, entry.value);
                        }
                    }

                    for (a_idx, a_row) in a_rows.iter().take(n_a).enumerate() {
                        for pos in 0..block_len {
                            let start = pos_offsets[pos];
                            let end = pos_offsets[pos + 1];
                            if start == end {
                                continue;
                            }
                            let a_wide =
                                WideCyclotomicRing::from_ring(&a_row[pos * num_digits_commit]);
                            for &(local_b, coeff_idx, value) in &pos_entries[start..end] {
                                shift_signed_unit_into(
                                    &a_wide,
                                    &mut accums[local_b as usize][a_idx],
                                    coeff_idx,
                                    value,
                                );
                            }
                        }
                    }
                } else {
                    col_entries.clear();
                    for local_b in 0..tile_len {
                        for entry in blocks[block_start + tile_start + local_b] {
                            col_entries.push((
                                entry.pos_in_block() * num_digits_commit,
                                local_b as u32,
                                entry.coeff_idx,
                                entry.value,
                            ));
                        }
                    }
                    col_entries.sort_unstable_by_key(|&(col, _, _, _)| col);

                    for (a_idx, a_row) in a_rows.iter().take(n_a).enumerate() {
                        let mut idx = 0usize;
                        while idx < col_entries.len() {
                            let col = col_entries[idx].0;
                            let a_wide = WideCyclotomicRing::from_ring(&a_row[col]);
                            while idx < col_entries.len() && col_entries[idx].0 == col {
                                let (_, local_b, coeff_idx, value) = col_entries[idx];
                                shift_signed_unit_into(
                                    &a_wide,
                                    &mut accums[local_b as usize][a_idx],
                                    coeff_idx,
                                    value,
                                );
                                idx += 1;
                            }
                        }
                    }
                }
                for (local_b, row_accums) in accums.into_iter().enumerate() {
                    result[tile_start + local_b] =
                        row_accums.into_iter().map(|w| w.reduce()).collect();
                }
            }
            result
        })
        .collect();

    let mut out = Vec::with_capacity(num_blocks);
    for thread_blocks in thread_results {
        out.extend(thread_blocks);
    }
    out
}

fn decompose_commit_rows<F, const D: usize>(
    rows: &[Vec<CyclotomicRing<F, D>>],
    n_a: usize,
    num_digits_open: usize,
    log_basis: u32,
) -> Result<FlatDigitBlocks<D>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let zero_block_len = n_a.checked_mul(num_digits_open).ok_or_else(|| {
        AkitaError::InvalidSetup("commit witness digit block length overflow".to_string())
    })?;
    let mut out = FlatDigitBlocks::zeroed(vec![zero_block_len; rows.len()])?;
    let dst_blocks = out.split_blocks_mut();
    #[cfg(feature = "parallel")]
    cfg_into_iter!(dst_blocks)
        .zip(cfg_iter!(rows))
        .for_each(|(dst, row)| {
            if !row.iter().all(|r| *r == CyclotomicRing::zero()) {
                decompose_rows_i8_into(row, dst, num_digits_open, log_basis);
            }
        });
    #[cfg(not(feature = "parallel"))]
    dst_blocks
        .into_iter()
        .zip(rows.iter())
        .for_each(|(dst, row)| {
            if !row.iter().all(|r| *r == CyclotomicRing::zero()) {
                decompose_rows_i8_into(row, dst, num_digits_open, log_basis);
            }
        });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DensePoly;
    use akita_field::Prime128OffsetA7F7 as F;

    #[test]
    fn sparse_ring_fold_matches_dense_reference() {
        const D: usize = 8;
        let sparse = SparseRingPoly::<F, D>::from_signed_coeffs(
            5,
            4,
            vec![(0, 1, 1), (1, 3, -1), (3, 2, 1)],
        )
        .unwrap();
        let mut dense_coeffs = vec![CyclotomicRing::<F, D>::zero(); 4];
        dense_coeffs[0].coeffs[1] += F::one();
        dense_coeffs[1].coeffs[3] -= F::one();
        dense_coeffs[3].coeffs[2] += F::one();
        let dense = DensePoly::<F, D>::from_ring_coeffs(dense_coeffs);
        let scalars = (0..2)
            .map(|idx| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                    F::from_u64(10 + idx * 10 + k as u64)
                }))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            sparse.fold_blocks_ring(&scalars, 2),
            dense.fold_blocks_ring(&scalars, 2)
        );
    }

    #[test]
    fn sorted_sparse_ring_constructor_rejects_unsorted_coeffs() {
        const D: usize = 8;
        let sorted =
            SparseRingPoly::<F, D>::from_sorted_signed_coeffs(5, 4, vec![(0, 1, 1), (2, 3, -1)])
                .unwrap();
        assert_eq!(sorted.num_ring_elems(), 4);

        assert!(SparseRingPoly::<F, D>::from_sorted_signed_coeffs(
            5,
            4,
            vec![(2, 3, -1), (0, 1, 1)],
        )
        .is_err());
    }

    #[test]
    fn packed_sparse_ring_constructor_matches_tuple_constructor() {
        const D: usize = 8;
        let tuples = vec![(0, 1, 1), (1, 3, -1), (3, 2, 1)];
        let packed = tuples
            .iter()
            .copied()
            .map(|(ring_idx, coeff_idx, value)| {
                SparseRingCoeff::new(ring_idx, coeff_idx, value).unwrap()
            })
            .collect::<Vec<_>>();
        let from_tuples = SparseRingPoly::<F, D>::from_signed_coeffs(5, 4, tuples).unwrap();
        let from_packed = SparseRingPoly::<F, D>::from_packed_coeffs(5, 4, packed).unwrap();

        let scalars = (0..2)
            .map(|idx| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                    F::from_u64(20 + idx * 10 + k as u64)
                }))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            from_packed.fold_blocks_ring(&scalars, 2),
            from_tuples.fold_blocks_ring(&scalars, 2)
        );
    }
}
