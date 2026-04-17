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
use crate::protocol::commitment::utils::linear::decompose_rows_i8_into;
use crate::protocol::hachi_poly_ops::helpers::{
    build_decompose_fold_witness, regular_onehot_accumulate, regular_onehot_accumulate_slices,
    sparse_onehot_accumulate, sparse_onehot_accumulate_slices,
};
use crate::protocol::hachi_poly_ops::{CommitInnerWitness, DecomposeFoldWitness, HachiPolyOps};
use crate::protocol::proof::{DirectWitnessProof, FlatDigitBlocks, FlatRingVec};
use crate::{CanonicalField, FieldCore};
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
    pub(crate) num_vars: usize,
    pub(crate) onehot_k: usize,
    pub(crate) indices: Vec<Option<usize>>,
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
        let total_field_elems = indices.len().checked_mul(onehot_k).ok_or_else(|| {
            HachiError::InvalidInput("onehot total field element count overflow".to_string())
        })?;
        if !total_field_elems.is_power_of_two() {
            return Err(HachiError::InvalidInput(format!(
                "onehot total field elements {total_field_elems} is not a power of two"
            )));
        }
        let indices_usize: Vec<Option<usize>> = indices
            .iter()
            .map(|idx| idx.map(OneHotIndex::as_usize))
            .collect();
        let use_regular_blocks = onehot_k >= D && onehot_k.is_multiple_of(D);
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
            num_vars: total_field_elems.trailing_zeros() as usize,
            onehot_k,
            indices: indices_usize,
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

    fn decompose_fold_batched_regular_onehot(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
    ) -> Option<DecomposeFoldWitness<F, D>>
    where
        F: CanonicalField,
    {
        let total_blocks = challenges.len();
        let mut flat_blocks = Vec::with_capacity(total_blocks);
        for poly in polys {
            let OneHotBlocks::Regular(blocks) = &poly.blocks else {
                return None;
            };
            flat_blocks.extend(blocks.iter().map(Vec::as_slice));
        }
        if flat_blocks.is_empty() {
            return None;
        }
        let active_blocks = flat_blocks.len().min(total_blocks);
        let modulus = (-F::one()).to_canonical_u128() + 1;

        let coeff_accum_digit0 = {
            let _span = tracing::info_span!("onehot_regular_accumulate_batched").entered();
            regular_onehot_accumulate_slices::<D>(
                &flat_blocks,
                challenges,
                active_blocks,
                block_len,
            )
        };

        let coeff_accum = if num_digits == 1 {
            coeff_accum_digit0
        } else {
            let _span = tracing::info_span!("onehot_regular_expand_batched").entered();
            let mut expanded = Vec::with_capacity(block_len * num_digits);
            for coeffs in coeff_accum_digit0 {
                expanded.push(coeffs);
                for _ in 1..num_digits {
                    expanded.push([0i32; D]);
                }
            }
            expanded
        };

        let _span = tracing::info_span!("onehot_regular_convert_batched").entered();
        Some(build_decompose_fold_witness::<F, D>(coeff_accum, modulus))
    }

    fn decompose_fold_batched_sparse_onehot(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
    ) -> Option<DecomposeFoldWitness<F, D>>
    where
        F: CanonicalField,
    {
        let total_blocks = challenges.len();
        let mut flat_blocks = Vec::with_capacity(total_blocks);
        for poly in polys {
            let OneHotBlocks::General(blocks) = &poly.blocks else {
                return None;
            };
            flat_blocks.extend(blocks.iter().map(Vec::as_slice));
        }
        if flat_blocks.is_empty() {
            return None;
        }
        let active_blocks = flat_blocks.len().min(total_blocks);
        let modulus = (-F::one()).to_canonical_u128() + 1;
        let inner_width = block_len * num_digits;

        let coeff_accum = {
            let _span = tracing::info_span!("onehot_sparse_accumulate_batched").entered();
            sparse_onehot_accumulate_slices::<D>(
                &flat_blocks,
                challenges,
                active_blocks,
                inner_width,
                num_digits,
            )
        };

        let _span = tracing::info_span!("onehot_sparse_convert_batched").entered();
        Some(build_decompose_fold_witness::<F, D>(coeff_accum, modulus))
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

    fn num_vars(&self) -> usize {
        self.num_vars
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
        assert_eq!(
            block_len,
            1usize << self.m_vars,
            "OneHotPoly::fold_blocks requires block_len to match the poly's internal block \
             size (2^m_vars); onehot polys bake the block split in at construction time. \
             Rebuild the poly with the caller-expected (r_vars, m_vars) split, or use \
             hachi_batched_root_layout(nv, max_num_batched_polys) if you plan to commit in \
             a batched context."
        );
        match &self.blocks {
            OneHotBlocks::Regular(blocks) => cfg_iter!(blocks)
                .map(|entries| fold_regular_onehot_block(entries, scalars, block_len))
                .collect(),
            OneHotBlocks::General(blocks) => cfg_iter!(blocks)
                .map(|entries| fold_sparse_onehot_block(entries, scalars, block_len))
                .collect(),
        }
    }

    fn evaluate_and_fold(
        &self,
        eval_outer_scalars: &[F],
        fold_scalars: &[F],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        assert_eq!(
            block_len,
            1usize << self.m_vars,
            "OneHotPoly::evaluate_and_fold requires block_len to match the poly's internal \
             block size (2^m_vars); onehot polys bake the block split in at construction \
             time. Rebuild the poly with the caller-expected (r_vars, m_vars) split, or use \
             hachi_batched_root_layout(nv, max_num_batched_polys) if you plan to commit in \
             a batched context."
        );
        let folded: Vec<CyclotomicRing<F, D>> = match &self.blocks {
            OneHotBlocks::Regular(blocks) => cfg_iter!(blocks)
                .map(|entries| fold_regular_onehot_block(entries, fold_scalars, block_len))
                .collect(),
            OneHotBlocks::General(blocks) => cfg_iter!(blocks)
                .map(|entries| fold_sparse_onehot_block(entries, fold_scalars, block_len))
                .collect(),
        };
        let eval = folded
            .iter()
            .zip(eval_outer_scalars.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
                acc + f_i.scale(s_i)
            });
        (eval, folded)
    }

    #[tracing::instrument(skip_all, name = "OneHotPoly::decompose_fold")]
    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> DecomposeFoldWitness<F, D> {
        assert_eq!(
            block_len,
            1usize << self.m_vars,
            "OneHotPoly::decompose_fold requires block_len to match the poly's internal \
             block size (2^m_vars); rebuild the poly with the caller-expected (r_vars, \
             m_vars) split if you are committing in a batched context."
        );
        match &self.blocks {
            OneHotBlocks::Regular(blocks) => {
                self.decompose_fold_regular_onehot(blocks, challenges, block_len, num_digits)
            }
            OneHotBlocks::General(blocks) => {
                self.decompose_fold_sparse_onehot(blocks, challenges, block_len, num_digits)
            }
        }
    }

    #[tracing::instrument(skip_all, name = "OneHotPoly::decompose_fold_batched")]
    fn decompose_fold_batched(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> Option<DecomposeFoldWitness<F, D>> {
        if let Some(first) = polys.first() {
            assert_eq!(
                block_len,
                1usize << first.m_vars,
                "OneHotPoly::decompose_fold_batched requires block_len to match every \
                 poly's internal block size (2^m_vars); rebuild the polys with the \
                 caller-expected (r_vars, m_vars) split if you are committing in a \
                 batched context."
            );
            for poly in polys {
                assert_eq!(
                    poly.m_vars, first.m_vars,
                    "OneHotPoly::decompose_fold_batched requires every poly in the batch \
                     to share the same (r_vars, m_vars) split."
                );
            }
        }
        match &polys.first()?.blocks {
            OneHotBlocks::Regular(_) => Self::decompose_fold_batched_regular_onehot(
                polys, challenges, block_len, num_digits,
            ),
            OneHotBlocks::General(_) => {
                Self::decompose_fold_batched_sparse_onehot(polys, challenges, block_len, num_digits)
            }
        }
    }

    #[tracing::instrument(skip_all, name = "OneHotPoly::commit_inner")]
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
    ) -> Result<FlatDigitBlocks<D>, HachiError> {
        if block_len != (1usize << self.m_vars) {
            return Err(HachiError::InvalidInput(format!(
                "OneHotPoly::commit_inner: block_len={block_len} does not match the poly's \
                 internal block size 2^m_vars={}. Onehot polys bake the (r_vars, m_vars) \
                 split in at construction time; if you commit in a batched context, build \
                 the poly with the layout returned by hachi_batched_root_layout(nv, \
                 max_num_batched_polys).",
                1usize << self.m_vars
            )));
        }
        let a_view = a_matrix.ring_view::<D>(n_a, matrix_stride);
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
            OneHotBlocks::Regular(blocks) => onehot_column_sweep_ajtai_regular::<_, F, D>(
                &a_view,
                blocks,
                n_a,
                active_a_cols,
                num_digits_commit,
            ),
            OneHotBlocks::General(blocks) => onehot_column_sweep_ajtai::<_, F, D>(
                &a_view,
                blocks,
                n_a,
                active_a_cols,
                num_digits_commit,
            ),
        };

        let mut t_hat = FlatDigitBlocks::zeroed(vec![zero_block_len; num_blocks])?;
        let dst_blocks = t_hat.split_blocks_mut();
        #[cfg(feature = "parallel")]
        cfg_into_iter!(dst_blocks)
            .zip(cfg_iter!(t_all))
            .for_each(|(dst, t_i)| {
                if !t_i.iter().all(|r| *r == CyclotomicRing::zero()) {
                    decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis);
                }
            });
        #[cfg(not(feature = "parallel"))]
        dst_blocks
            .into_iter()
            .zip(t_all.iter())
            .for_each(|(dst, t_i)| {
                if !t_i.iter().all(|r| *r == CyclotomicRing::zero()) {
                    decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis);
                }
            });

        Ok(t_hat)
    }

    #[tracing::instrument(skip_all, name = "OneHotPoly::commit_inner_witness")]
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
    ) -> Result<CommitInnerWitness<F, D>, HachiError> {
        let a_view = a_matrix.ring_view::<D>(n_a, matrix_stride);
        let active_a_cols = num_cols_a(block_len, num_digits_commit)?;
        if active_a_cols > a_view.num_cols() {
            return Err(HachiError::InvalidSetup(format!(
                "active A width {active_a_cols} exceeds setup envelope {}",
                a_view.num_cols()
            )));
        }
        let zero_block_len = n_a.checked_mul(num_digits_open).unwrap();

        let t = match &self.blocks {
            OneHotBlocks::Regular(blocks) => onehot_column_sweep_ajtai_regular::<_, F, D>(
                &a_view,
                blocks,
                n_a,
                active_a_cols,
                num_digits_commit,
            ),
            OneHotBlocks::General(blocks) => onehot_column_sweep_ajtai::<_, F, D>(
                &a_view,
                blocks,
                n_a,
                active_a_cols,
                num_digits_commit,
            ),
        };

        let mut t_hat = FlatDigitBlocks::zeroed(vec![zero_block_len; t.len()])?;
        let dst_blocks = t_hat.split_blocks_mut();
        #[cfg(feature = "parallel")]
        cfg_into_iter!(dst_blocks)
            .zip(cfg_iter!(t))
            .for_each(|(dst, t_i)| {
                if !t_i.iter().all(|r| *r == CyclotomicRing::zero()) {
                    decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis);
                }
            });
        #[cfg(not(feature = "parallel"))]
        dst_blocks.into_iter().zip(t.iter()).for_each(|(dst, t_i)| {
            if !t_i.iter().all(|r| *r == CyclotomicRing::zero()) {
                decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis);
            }
        });

        Ok(CommitInnerWitness { t, t_hat })
    }

    fn direct_root_witness(&self) -> Result<DirectWitnessProof<F>, HachiError> {
        let total_evals = 1usize.checked_shl(self.num_vars as u32).ok_or_else(|| {
            HachiError::InvalidInput(format!("2^{} does not fit usize", self.num_vars))
        })?;
        let mut evals = vec![F::zero(); total_evals];
        for (chunk_idx, hot_pos) in self.indices.iter().enumerate() {
            let Some(hot_pos) = hot_pos else {
                continue;
            };
            let field_pos = chunk_idx
                .checked_mul(self.onehot_k)
                .and_then(|base| base.checked_add(*hot_pos))
                .ok_or_else(|| {
                    HachiError::InvalidInput("onehot direct witness index overflow".to_string())
                })?;
            if field_pos >= evals.len() {
                return Err(HachiError::InvalidInput(format!(
                    "onehot direct witness index {field_pos} out of range for {} evals",
                    evals.len()
                )));
            }
            evals[field_pos] = F::one();
        }
        Ok(DirectWitnessProof::FieldElements(FlatRingVec::from_coeffs(
            evals,
        )))
    }
}

fn num_cols_a(block_len: usize, num_digits_commit: usize) -> Result<usize, HachiError> {
    block_len
        .checked_mul(num_digits_commit)
        .ok_or_else(|| HachiError::InvalidSetup("active A width overflow".to_string()))
}

fn fold_regular_onehot_block<F: FieldCore, const D: usize>(
    entries: &[RegularOneHotEntry],
    scalars: &[F],
    block_len: usize,
) -> CyclotomicRing<F, D> {
    let mut coeffs_acc = [F::zero(); D];
    for entry in entries {
        let pos = entry.pos_in_block();
        if pos < scalars.len() && pos < block_len {
            coeffs_acc[entry.coeff_idx()] += scalars[pos];
        }
    }
    CyclotomicRing::from_coefficients(coeffs_acc)
}

fn fold_sparse_onehot_block<F: FieldCore, const D: usize>(
    entries: &[SparseBlockEntry],
    scalars: &[F],
    block_len: usize,
) -> CyclotomicRing<F, D> {
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
fn inner_ajtai_regular_onehot_chunked<F, const D: usize>(
    a_view: &crate::protocol::commitment::utils::flat_matrix::RingMatrixView<'_, F, D>,
    regular_entries: &[RegularOneHotEntry],
    num_digits: usize,
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: crate::AdditiveGroup + From<F> + crate::algebra::fields::wide::ReduceTo<F>,
{
    let n_a = a_view.num_rows();
    let mut t = vec![CyclotomicRing::<F, D>::zero(); n_a];

    for chunk in regular_entries.chunks(MAX_WIDE_SHIFT_ACCUMULATIONS) {
        let partial = inner_ajtai_regular_onehot_wide(a_view, chunk, num_digits);
        for (dst, src) in t.iter_mut().zip(partial.iter()) {
            *dst += *src;
        }
    }

    t
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

/// Wide accumulators use 16-bit chunks in `i32` limbs, so they can safely
/// absorb at most 32,768 unit-scale additions before overflow.
const MAX_WIDE_SHIFT_ACCUMULATIONS: usize = 1 << 15;

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
fn onehot_column_sweep_ajtai_regular<B, F, const D: usize>(
    a_view: &crate::protocol::commitment::utils::flat_matrix::RingMatrixView<'_, F, D>,
    regular_blocks: &[B],
    n_a: usize,
    active_a_cols: usize,
    num_digits_commit: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    B: AsRef<[RegularOneHotEntry]> + Sync,
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: crate::AdditiveGroup + From<F> + crate::algebra::fields::wide::ReduceTo<F>,
{
    let num_blocks = regular_blocks.len();
    debug_assert!(
        active_a_cols <= a_view.num_cols(),
        "active A width exceeds setup envelope"
    );
    if regular_blocks
        .iter()
        .any(|block_entries| block_entries.as_ref().len() > MAX_WIDE_SHIFT_ACCUMULATIONS)
    {
        return cfg_iter!(regular_blocks)
            .map(|block_entries| {
                inner_ajtai_regular_onehot_chunked(
                    a_view,
                    block_entries.as_ref(),
                    num_digits_commit,
                )
            })
            .collect();
    }
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
                inner_ajtai_regular_onehot_wide(a_view, block_entries.as_ref(), num_digits_commit)
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

            // Reuse across tiles so earlier capacity carries over, but only
            // allocate buckets for columns that are actually touched.
            let mut col_entries: Vec<(usize, u32, u16)> = Vec::new();

            for tile_start in (0..my_count).step_by(block_tile) {
                let tile_end = (tile_start + block_tile).min(my_count);
                let tile_blocks = &my_blocks[tile_start..tile_end];
                let tile_len = tile_blocks.len();

                col_entries.clear();
                for (local_b, block_entries) in tile_blocks.iter().enumerate() {
                    for entry in block_entries.as_ref() {
                        let col = entry.pos_in_block() * num_digits_commit;
                        col_entries.push((col, local_b as u32, entry.coeff_idx() as u16));
                    }
                }
                col_entries.sort_unstable_by_key(|&(col, _, _)| col);

                let mut accums: Vec<Vec<WideCyclotomicRing<F::Wide, D>>> = (0..tile_len)
                    .map(|_| vec![WideCyclotomicRing::zero(); n_a])
                    .collect();

                for a_idx in 0..n_a {
                    let a_row = a_view.row(a_idx);
                    let mut idx = 0usize;
                    while idx < col_entries.len() {
                        let col = col_entries[idx].0;
                        let a_wide = WideCyclotomicRing::from_ring(&a_row[col]);
                        while idx < col_entries.len() && col_entries[idx].0 == col {
                            let (_, lb, ci) = col_entries[idx];
                            a_wide.shift_accumulate_into(
                                &mut accums[lb as usize][a_idx],
                                ci as usize,
                            );
                            idx += 1;
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
fn onehot_column_sweep_ajtai<B, F, const D: usize>(
    a_view: &crate::protocol::commitment::utils::flat_matrix::RingMatrixView<'_, F, D>,
    sparse_blocks: &[B],
    n_a: usize,
    active_a_cols: usize,
    num_digits_commit: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    B: AsRef<[SparseBlockEntry]> + Sync,
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: crate::AdditiveGroup + From<F> + crate::algebra::fields::wide::ReduceTo<F>,
{
    let num_blocks = sparse_blocks.len();
    debug_assert!(
        active_a_cols <= a_view.num_cols(),
        "active A width exceeds setup envelope"
    );
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
                inner_ajtai_onehot_wide(a_view, block_entries.as_ref(), 0, num_digits_commit)
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

            let mut col_entries: Vec<(usize, u32, u8)> = Vec::new();

            for tile_start in (0..my_count).step_by(block_tile) {
                let tile_end = (tile_start + block_tile).min(my_count);
                let tile_blocks = &my_blocks[tile_start..tile_end];
                let tile_len = tile_blocks.len();

                col_entries.clear();
                for (local_b, block_entries) in tile_blocks.iter().enumerate() {
                    for entry in block_entries.as_ref() {
                        let col = entry.pos_in_block * num_digits_commit;
                        for &ci in &entry.nonzero_coeffs {
                            col_entries.push((col, local_b as u32, ci as u8));
                        }
                    }
                }
                col_entries.sort_unstable_by_key(|&(col, _, _)| col);

                let mut accums: Vec<Vec<WideCyclotomicRing<F::Wide, D>>> = (0..tile_len)
                    .map(|_| vec![WideCyclotomicRing::zero(); n_a])
                    .collect();

                for a_idx in 0..n_a {
                    let a_row = a_view.row(a_idx);
                    let mut idx = 0usize;
                    while idx < col_entries.len() {
                        let col = col_entries[idx].0;
                        let a_wide = WideCyclotomicRing::from_ring(&a_row[col]);
                        while idx < col_entries.len() && col_entries[idx].0 == col {
                            let (_, lb, ci) = col_entries[idx];
                            a_wide.shift_accumulate_into(
                                &mut accums[lb as usize][a_idx],
                                ci as usize,
                            );
                            idx += 1;
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

    let mut out: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(num_blocks);
    for thread_blocks in thread_results {
        out.extend(thread_blocks);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Pow2Offset24Field;
    use crate::protocol::commitment::utils::flat_matrix::FlatMatrix;
    use crate::FromSmallInt;

    fn aggregate_witnesses<F: FieldCore, const D: usize>(
        witnesses: &[DecomposeFoldWitness<F, D>],
    ) -> DecomposeFoldWitness<F, D> {
        let mut acc = witnesses[0].clone();
        for witness in &witnesses[1..] {
            for (dst, src) in acc.z_pre.iter_mut().zip(witness.z_pre.iter()) {
                *dst += *src;
            }
            for (dst, src) in acc
                .centered_coeffs
                .iter_mut()
                .zip(witness.centered_coeffs.iter())
            {
                for k in 0..D {
                    dst[k] += src[k];
                }
            }
        }
        acc.centered_inf_norm = acc
            .centered_coeffs
            .iter()
            .flat_map(|coeffs| coeffs.iter())
            .map(|coeff| coeff.unsigned_abs())
            .max()
            .unwrap_or(0);
        acc
    }

    #[test]
    fn regular_onehot_large_block_uses_safe_accumulator_path() {
        type F = Pow2Offset24Field;
        const D: usize = 64;

        let block_len = MAX_WIDE_SHIFT_ACCUMULATIONS + 1;
        let max_coeff = F::from_canonical_u128_reduced((1u128 << 24) - 4);
        let dense_ring = CyclotomicRing::from_coefficients([max_coeff; D]);
        let a_matrix = [vec![dense_ring; block_len]];
        let regular_blocks = vec![{
            let mut entries = Vec::with_capacity(block_len);
            for pos in 0..block_len {
                entries.push(RegularOneHotEntry::new(pos, pos % D).unwrap());
            }
            entries
        }];

        let a_flat = FlatMatrix::from_ring_slice(&a_matrix[0]);
        let a_view = a_flat.ring_view::<D>(1, block_len);

        let got =
            onehot_column_sweep_ajtai_regular::<_, F, D>(&a_view, &regular_blocks, 1, block_len, 1);
        let expected = inner_ajtai_regular_onehot_chunked::<F, D>(&a_view, &regular_blocks[0], 1);

        assert_eq!(got.len(), 1);
        assert_eq!(got[0], expected);
    }

    #[test]
    fn batched_regular_onehot_decompose_fold_matches_individual_aggregation() {
        type F = Pow2Offset24Field;
        const D: usize = 64;

        let block_len = 64;
        let mut indices0 = vec![None; 128];
        indices0[0] = Some(1usize);
        indices0[17] = Some(5usize);
        indices0[64] = Some(9usize);
        indices0[91] = Some(33usize);
        let mut indices1 = vec![None; 128];
        indices1[3] = Some(7usize);
        indices1[29] = Some(11usize);
        indices1[64] = Some(19usize);
        indices1[100] = Some(21usize);
        let polys = [
            OneHotPoly::<F, D>::new(block_len, indices0, 1, 6).unwrap(),
            OneHotPoly::<F, D>::new(block_len, indices1, 1, 6).unwrap(),
        ];
        let challenges = vec![
            SparseChallenge {
                positions: vec![0, 5],
                coeffs: vec![1, -1],
            },
            SparseChallenge {
                positions: vec![2, 7],
                coeffs: vec![1, 1],
            },
            SparseChallenge {
                positions: vec![4, 11],
                coeffs: vec![-1, 2],
            },
            SparseChallenge {
                positions: vec![8, 13],
                coeffs: vec![1, -2],
            },
        ];

        let expected = aggregate_witnesses(
            &polys
                .iter()
                .zip(challenges.chunks(2))
                .map(|(poly, poly_challenges)| {
                    poly.decompose_fold(poly_challenges, block_len, 1, 0)
                })
                .collect::<Vec<_>>(),
        );
        let poly_refs: Vec<&OneHotPoly<F, D>> = polys.iter().collect();
        let got = <OneHotPoly<F, D> as HachiPolyOps<F, D>>::decompose_fold_batched(
            &poly_refs,
            &challenges,
            block_len,
            1,
            0,
        )
        .expect("onehot batched path should apply");

        assert_eq!(got, expected);
    }

    #[test]
    fn regular_onehot_evaluate_and_fold_matches_factorized_eval() {
        type F = Pow2Offset24Field;
        const D: usize = 64;

        let poly = OneHotPoly::<F, D>::new(
            64,
            vec![Some(1usize), None, Some(9usize), Some(17usize)],
            1,
            1,
        )
        .unwrap();
        let block_len = 1usize << poly.m_vars;
        let fold_scalars = vec![F::from_u64(3), F::from_u64(5)];
        let eval_outer_scalars = vec![F::from_u64(7), F::from_u64(11)];

        let (eval, folded) = poly.evaluate_and_fold(&eval_outer_scalars, &fold_scalars, block_len);
        let expected_folded = poly.fold_blocks(&fold_scalars, block_len);
        assert_eq!(folded, expected_folded);

        let full_scalars: Vec<F> = eval_outer_scalars
            .iter()
            .flat_map(|outer| fold_scalars.iter().map(move |inner| *outer * *inner))
            .collect();
        let expected_eval = poly.evaluate_ring(&full_scalars);
        assert_eq!(eval, expected_eval);
    }

    #[test]
    fn sparse_onehot_evaluate_and_fold_matches_factorized_eval() {
        type F = Pow2Offset24Field;
        const D: usize = 64;

        let poly = OneHotPoly::<F, D>::new(
            32,
            vec![
                Some(1usize),
                None,
                Some(7usize),
                Some(12usize),
                None,
                Some(3usize),
                None,
                Some(15usize),
            ],
            1,
            1,
        )
        .unwrap();
        let block_len = 1usize << poly.m_vars;
        let fold_scalars = vec![F::from_u64(2), F::from_u64(4)];
        let eval_outer_scalars = vec![F::from_u64(3), F::from_u64(5)];

        let (eval, folded) = poly.evaluate_and_fold(&eval_outer_scalars, &fold_scalars, block_len);
        let expected_folded = poly.fold_blocks(&fold_scalars, block_len);
        assert_eq!(folded, expected_folded);

        let full_scalars: Vec<F> = eval_outer_scalars
            .iter()
            .flat_map(|outer| fold_scalars.iter().map(move |inner| *outer * *inner))
            .collect();
        let expected_eval = poly.evaluate_ring(&full_scalars);
        assert_eq!(eval, expected_eval);
    }
}
