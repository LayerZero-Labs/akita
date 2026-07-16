//! Sparse signed ring-coefficient polynomial backend.
//!
//! This is the natural backend for Frobenius-packed one-hot tables: after
//! canonical-basis packing, each original one-hot chunk becomes a small number
//! of signed monomial coefficients inside the committed ring table.

use akita_algebra::ring::cyclotomic::WideCyclotomicRing;
use akita_algebra::CyclotomicRing;
use akita_challenges::{SparseChallenge, TensorChallenges as TensorChallengeSet};
use akita_field::parallel::*;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AdditiveGroup, AkitaError, CanonicalField, FieldCore, FromPrimitiveInt};
use akita_types::embed_ring_subfield_vector;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::backend::poly_helpers::{build_decompose_fold_witness, fill_rotated_challenge};
use crate::compute::{
    CommitInnerPlan, CommitmentComputeBackend, FlatBlockTable, SparseRingCommitRowsPlan,
};
use crate::kernels::linear::decompose_commit_blocks_into;
use crate::{CommitInnerWitness, DecomposeFoldWitness};

mod ops;

pub use ops::{SparseRingBatchView, SparseRingView};

mod tensor_fold;

type SparseLayoutCacheKey = (usize, usize);
type SparseBlockCache = Arc<Mutex<HashMap<SparseLayoutCacheKey, Arc<SparseRingBlocks>>>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SparseRingCoeff {
    /// Flat field-coefficient position: `ring_idx * ring_d + coeff_idx` at the
    /// ring dimension the coefficient was constructed with. The ring dimension
    /// is a view selected at kernel entry, not a property of the stored data.
    flat_idx: u64,
    value: i8,
}

impl SparseRingCoeff {
    pub(crate) fn new(flat_idx: usize, value: i8) -> Result<Self, AkitaError> {
        if !matches!(value, -1 | 1) {
            return Err(AkitaError::InvalidInput(
                "sparse ring coefficients must be signed units".to_string(),
            ));
        }
        Ok(Self {
            flat_idx: u64::try_from(flat_idx).map_err(|_| {
                AkitaError::InvalidInput("sparse flat coefficient index exceeds u64".to_string())
            })?,
            value,
        })
    }

    /// Pack `(ring_idx, coeff_idx)` at ring dimension `ring_d` into a flat
    /// field-coefficient position.
    pub(crate) fn from_ring_coords(
        ring_idx: usize,
        coeff_idx: usize,
        ring_d: usize,
        value: i8,
    ) -> Result<Self, AkitaError> {
        let flat_idx = ring_idx
            .checked_mul(ring_d)
            .and_then(|base| base.checked_add(coeff_idx))
            .ok_or_else(|| {
                AkitaError::InvalidInput("sparse flat coefficient index overflow".to_string())
            })?;
        Self::new(flat_idx, value)
    }

    #[inline]
    fn ring_idx(self, ring_d: usize) -> usize {
        (self.flat_idx as usize) / ring_d
    }

    #[inline]
    fn coeff_idx(self, ring_d: usize) -> usize {
        (self.flat_idx as usize) % ring_d
    }

    #[inline]
    fn sort_key(self) -> (u64, i8) {
        // `flat_idx = ring_idx * ring_d + coeff_idx` is order-equivalent to
        // the previous `(ring_idx, coeff_idx, value)` lexicographic key.
        (self.flat_idx, self.value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseRingBlockEntry {
    pos_in_block: u32,
    coeff_idx: u16,
    value: i8,
}

impl SparseRingBlockEntry {
    #[inline]
    pub fn pos_in_block(self) -> usize {
        self.pos_in_block as usize
    }

    #[inline]
    pub fn coeff_idx(self) -> usize {
        self.coeff_idx as usize
    }

    #[inline]
    pub fn value(self) -> i8 {
        self.value
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SparseRingBlocks {
    entries: Vec<SparseRingBlockEntry>,
    offsets: Vec<u32>,
}

impl SparseRingBlocks {
    fn from_coeffs(
        coeffs: &[SparseRingCoeff],
        ring_d: usize,
        total_ring_elems: usize,
        block_len: usize,
    ) -> Result<Self, AkitaError> {
        if ring_d == 0 {
            return Err(AkitaError::InvalidInput(
                "ring_d must be nonzero".to_string(),
            ));
        }
        if block_len == 0 || !block_len.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "block_len={block_len} must be a nonzero power of two"
            )));
        }
        if u32::try_from(block_len).is_err() {
            return Err(AkitaError::InvalidInput(format!(
                "block_len={block_len} exceeds u32::MAX"
            )));
        }
        let num_blocks = total_ring_elems.div_ceil(block_len);
        let mut offsets = Vec::with_capacity(num_blocks + 1);
        let mut entries = Vec::with_capacity(coeffs.len());
        offsets.push(0);
        let mut current_block = 0usize;
        for coeff in coeffs {
            let ring_idx = coeff.ring_idx(ring_d);
            if ring_idx >= total_ring_elems {
                return Err(AkitaError::InvalidInput(
                    "sparse ring coefficient index out of range".to_string(),
                ));
            }
            // Block entries pack the in-ring coefficient index as `u16`.
            // Supported ring dimensions are <= 256 so this always holds; reject
            // (rather than truncate or panic) if it ever does not.
            let coeff_idx = u16::try_from(coeff.coeff_idx(ring_d)).map_err(|_| {
                AkitaError::InvalidInput(
                    "sparse coefficient index exceeds u16 block-entry capacity".to_string(),
                )
            })?;
            let block_idx = ring_idx / block_len;
            while current_block < block_idx {
                offsets.push(entries.len() as u32);
                current_block += 1;
            }
            entries.push(SparseRingBlockEntry {
                pos_in_block: (ring_idx % block_len) as u32,
                coeff_idx,
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
    pub(crate) fn num_blocks(&self) -> usize {
        self.offsets.len() - 1
    }

    #[inline]
    pub(crate) fn block(&self, idx: usize) -> &[SparseRingBlockEntry] {
        let lo = self.offsets[idx] as usize;
        let hi = self.offsets[idx + 1] as usize;
        &self.entries[lo..hi]
    }

    #[inline]
    fn table(&self) -> FlatBlockTable<'_, SparseRingBlockEntry> {
        FlatBlockTable::new(&self.entries, &self.offsets)
    }
}

/// Sparse polynomial whose ring coefficients are signed monomials.
///
/// Storage is D-free: coefficients record flat field-coefficient positions,
/// and the ring dimension is a view selected at kernel entry (each ring-shaped
/// method takes it as a const generic).
#[derive(Debug, Clone)]
pub struct SparseRingPoly<F: FieldCore> {
    num_vars: usize,
    /// Ring-element count at the CONSTRUCTION dimension; metadata, not
    /// authority — kernels validate at their own dimension.
    total_ring_elems: usize,
    coeffs: Vec<SparseRingCoeff>,
    /// Cached per-block layouts keyed by `(ring_d, block_len)`.
    block_cache: SparseBlockCache,
    _marker: core::marker::PhantomData<F>,
}

impl<F: FieldCore> SparseRingPoly<F> {
    /// Build from `(ring_idx, coeff_idx, value)` triples interpreted at ring
    /// dimension `ring_d`.
    ///
    /// # Errors
    ///
    /// Returns an error when `ring_d` is zero, the expected ring-element count
    /// does not match `num_vars`, or a supplied coefficient triple is out of
    /// range or has value other than `-1` or `1`.
    pub fn from_signed_coeffs(
        num_vars: usize,
        ring_d: usize,
        total_ring_elems: usize,
        coeffs: Vec<(usize, usize, i8)>,
    ) -> Result<Self, AkitaError> {
        Self::from_signed_coeffs_with_order(num_vars, ring_d, total_ring_elems, coeffs, false)
    }

    /// Build from `(ring_idx, coeff_idx, value)` triples interpreted at ring
    /// dimension `ring_d`, already sorted by `(ring_idx, coeff_idx, value)`.
    ///
    /// # Errors
    ///
    /// Returns an error for the same malformed inputs as
    /// [`Self::from_signed_coeffs`], and also when the supplied triples are not
    /// sorted.
    pub fn from_sorted_signed_coeffs(
        num_vars: usize,
        ring_d: usize,
        total_ring_elems: usize,
        coeffs: Vec<(usize, usize, i8)>,
    ) -> Result<Self, AkitaError> {
        Self::from_signed_coeffs_with_order(num_vars, ring_d, total_ring_elems, coeffs, true)
    }

    /// Build from compact sparse coefficients whose flat positions were packed
    /// at ring dimension `ring_d`.
    ///
    /// # Errors
    ///
    /// Returns an error for the same malformed inputs as
    /// [`Self::from_signed_coeffs`].
    pub(crate) fn from_packed_coeffs(
        num_vars: usize,
        ring_d: usize,
        total_ring_elems: usize,
        coeffs: Vec<SparseRingCoeff>,
    ) -> Result<Self, AkitaError> {
        Self::from_packed_coeffs_with_order(num_vars, ring_d, total_ring_elems, coeffs, false)
    }

    /// Build from compact sparse coefficients whose flat positions were packed
    /// at ring dimension `ring_d`, already sorted by `(flat_idx, value)`
    /// (equivalently, `(ring_idx, coeff_idx, value)` at `ring_d`).
    ///
    /// # Errors
    ///
    /// Returns an error for the same malformed inputs as
    /// [`Self::from_sorted_signed_coeffs`].
    pub(crate) fn from_sorted_packed_coeffs(
        num_vars: usize,
        ring_d: usize,
        total_ring_elems: usize,
        coeffs: Vec<SparseRingCoeff>,
    ) -> Result<Self, AkitaError> {
        Self::from_packed_coeffs_with_order(num_vars, ring_d, total_ring_elems, coeffs, true)
    }

    fn from_signed_coeffs_with_order(
        num_vars: usize,
        ring_d: usize,
        total_ring_elems: usize,
        coeffs: Vec<(usize, usize, i8)>,
        already_sorted: bool,
    ) -> Result<Self, AkitaError> {
        let mut packed = Vec::with_capacity(coeffs.len());
        for (ring_idx, coeff_idx, value) in coeffs {
            if ring_d != 0 && (coeff_idx >= ring_d || ring_idx >= total_ring_elems) {
                return Err(AkitaError::InvalidInput(
                    "invalid sparse ring coefficient".to_string(),
                ));
            }
            packed.push(SparseRingCoeff::from_ring_coords(
                ring_idx, coeff_idx, ring_d, value,
            )?);
        }
        Self::from_packed_coeffs_with_order(
            num_vars,
            ring_d,
            total_ring_elems,
            packed,
            already_sorted,
        )
    }

    fn from_packed_coeffs_with_order(
        num_vars: usize,
        ring_d: usize,
        total_ring_elems: usize,
        mut packed: Vec<SparseRingCoeff>,
        already_sorted: bool,
    ) -> Result<Self, AkitaError> {
        let field_len = 1usize
            .checked_shl(num_vars as u32)
            .ok_or_else(|| AkitaError::InvalidInput("sparse arity overflow".to_string()))?;
        if ring_d == 0 {
            return Err(AkitaError::InvalidInput(
                "ring_d must be nonzero".to_string(),
            ));
        }
        let expected_ring_elems = field_len.div_ceil(ring_d);
        if expected_ring_elems != total_ring_elems {
            return Err(AkitaError::InvalidSize {
                expected: expected_ring_elems,
                actual: total_ring_elems,
            });
        }
        let mut previous_key = None;
        for entry in &packed {
            if entry.ring_idx(ring_d) >= total_ring_elems || !matches!(entry.value, -1 | 1) {
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
            block_cache: Arc::new(Mutex::new(HashMap::new())),
            _marker: core::marker::PhantomData,
        })
    }

    fn blocks_for(
        &self,
        ring_d: usize,
        block_len: usize,
    ) -> Result<Arc<SparseRingBlocks>, AkitaError> {
        let key = (ring_d, block_len);
        if let Some(blocks) = self
            .block_cache
            .lock()
            .map_err(|_| AkitaError::InvalidSetup("sparse block cache lock poisoned".into()))?
            .get(&key)
        {
            return Ok(Arc::clone(blocks));
        }
        let field_len = 1usize
            .checked_shl(self.num_vars as u32)
            .ok_or_else(|| AkitaError::InvalidInput("sparse arity overflow".to_string()))?;
        if ring_d == 0 {
            return Err(AkitaError::InvalidInput(
                "ring_d must be nonzero".to_string(),
            ));
        }
        let ring_elems_at_d = field_len.div_ceil(ring_d);
        let built =
            SparseRingBlocks::from_coeffs(&self.coeffs, ring_d, ring_elems_at_d, block_len)?;
        let mut cache = self
            .block_cache
            .lock()
            .map_err(|_| AkitaError::InvalidSetup("sparse block cache lock poisoned".into()))?;
        Ok(Arc::clone(
            cache.entry(key).or_insert_with(|| Arc::new(built)),
        ))
    }

    /// Total number of variables (`log2(total field evaluation slots)`).
    #[inline]
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Total number of ring elements at the construction dimension.
    #[inline]
    pub fn num_ring_elems(&self) -> usize {
        self.total_ring_elems
    }
}

impl<F> SparseRingPoly<F>
where
    F: FieldCore + FromPrimitiveInt,
{
    /// Materialize the dense field-evaluation table directly from the flat
    /// coefficient positions.
    ///
    /// This is the D-free field-materialization shared by the tensor helpers
    /// and the [`DirectRootWitnessSource`] impl (which wraps it in a
    /// [`akita_types::CleartextWitnessProof::FieldElements`] payload).
    ///
    /// # Errors
    ///
    /// Returns an error when the evaluation-table length overflows `usize`.
    pub(crate) fn direct_field_evals(&self) -> Result<Vec<F>, AkitaError> {
        let total_coeffs = 1usize.checked_shl(self.num_vars as u32).ok_or_else(|| {
            AkitaError::InvalidInput("sparse direct witness length overflow".to_string())
        })?;
        let mut coeffs = vec![F::zero(); total_coeffs];
        for entry in &self.coeffs {
            let idx = usize::try_from(entry.flat_idx).map_err(|_| {
                AkitaError::InvalidInput("sparse direct witness index overflow".to_string())
            })?;
            coeffs[idx] += F::from_i8(entry.value);
        }
        Ok(coeffs)
    }
}

impl<F> SparseRingPoly<F>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    pub(crate) fn fold_blocks<const D: usize>(
        &self,
        scalars: &[F],
        block_len: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let blocks = self
            .blocks_for(D, block_len)
            .expect("SparseRingPoly::fold_blocks: invalid block_len");
        cfg_into_iter!(0..blocks.num_blocks())
            .map(|block_idx| fold_sparse_block(blocks.block(block_idx), scalars, block_len))
            .collect()
    }

    pub(crate) fn fold_blocks_ring<const D: usize>(
        &self,
        scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let blocks = self
            .blocks_for(D, block_len)
            .expect("SparseRingPoly::fold_blocks_ring: invalid block_len");
        cfg_into_iter!(0..blocks.num_blocks())
            .map(|block_idx| fold_sparse_block_ring(blocks.block(block_idx), scalars, block_len))
            .collect()
    }

    pub(crate) fn evaluate_and_fold<const D: usize>(
        &self,
        block_weights: &[F],
        position_weights: &[F],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        let folded = self.fold_blocks::<D>(position_weights, block_len);
        let eval = folded
            .iter()
            .zip(block_weights.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
                acc + f_i.scale(s_i)
            });
        (eval, folded)
    }

    pub(crate) fn evaluate_and_fold_ring<const D: usize>(
        &self,
        block_weights: &[CyclotomicRing<F, D>],
        position_weights: &[CyclotomicRing<F, D>],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        let folded = self.fold_blocks_ring::<D>(position_weights, block_len);
        let mut eval = CyclotomicRing::<F, D>::zero();
        for (f_i, s_i) in folded.iter().zip(block_weights.iter()) {
            f_i.mul_accumulate_sparse_rhs_into(s_i, &mut eval);
        }
        (eval, folded)
    }

    #[tracing::instrument(skip_all, name = "SparseRingPoly::decompose_fold")]
    pub(crate) fn decompose_fold<const D: usize>(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> DecomposeFoldWitness<F> {
        let blocks = self
            .blocks_for(D, block_len)
            .expect("SparseRingPoly::decompose_fold: invalid block_len");
        let num_blocks = challenges.len().min(blocks.num_blocks());
        let inner_width = block_len * num_digits;
        let coeff_accum =
            sparse_accumulate::<D>(&blocks, challenges, num_blocks, inner_width, num_digits);
        let modulus = (-F::one()).to_canonical_u128() + 1;
        build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
    }

    #[tracing::instrument(skip_all, name = "SparseRingPoly::decompose_fold_tensor_batched")]
    pub(crate) fn decompose_fold_tensor_batched<const D: usize>(
        polys: &[&Self],
        tensor: &TensorChallengeSet,
        block_len: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> Result<Option<DecomposeFoldWitness<F>>, AkitaError> {
        Ok(Some(tensor_fold::decompose_fold_batched_tensor_sparse::<
            F,
            D,
        >(polys, tensor, block_len, num_digits)?))
    }

    #[tracing::instrument(skip_all, name = "SparseRingPoly::commit_inner")]
    pub(crate) fn commit_inner<B, const D: usize>(
        &self,
        backend: &B,
        prepared: &B::PreparedSetup,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F>, AkitaError>
    where
        B: CommitmentComputeBackend<F>,
    {
        let t = self.commit_inner_rows::<B, D>(
            backend,
            prepared,
            plan.n_a,
            plan.block_len,
            plan.num_digits_commit,
        )?;
        let decomposed_inner_rows =
            decompose_commit_blocks_into::<F, D>(&t, plan.num_digits_open, plan.log_basis)?;
        CommitInnerWitness::from_parts(t, decomposed_inner_rows)
    }

    pub(crate) fn tensor_extension_column_partials<E>(
        &self,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: akita_field::MulBaseUnreduced<F>,
    {
        let num_vars = self.num_vars();
        if logical_point.len() != num_vars {
            return Err(AkitaError::InvalidPointDimension {
                expected: num_vars,
                actual: logical_point.len(),
            });
        }
        let field_elems = self.direct_field_evals()?;
        akita_types::tensor_column_partials_from_base_evals::<F, E>(
            num_vars,
            &field_elems,
            logical_point,
        )
    }

    pub(crate) fn tensor_packed_extension_evals<E>(&self) -> Result<Vec<E>, AkitaError>
    where
        E: akita_field::ExtField<F>,
    {
        let num_vars = self.num_vars();
        let field_elems = self.direct_field_evals()?;
        akita_types::tensor_packed_witness_evals::<F, E>(num_vars, &field_elems)
    }

    pub(crate) fn tensor_packed_extension_sparse_evals<E>(
        &self,
    ) -> Result<
        Option<crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness<E>>,
        AkitaError,
    >
    where
        E: akita_field::ExtField<F>,
    {
        Ok(None)
    }

    pub(crate) fn tensor_packed_extension_poly<E, const D: usize>(
        &self,
    ) -> Result<crate::backend::dense::DensePoly<F>, AkitaError>
    where
        F: CanonicalField + FromPrimitiveInt,
        E: akita_types::FpExtEncoding<F>,
    {
        let evals = self.tensor_packed_extension_evals::<E>()?;
        let packed_len = D / E::EXT_DEGREE;
        if packed_len == 0 {
            return Err(AkitaError::InvalidInput(
                "extension degree exceeds root ring dimension".to_string(),
            ));
        }
        let mut rings = Vec::with_capacity(evals.len().div_ceil(packed_len));
        for chunk in evals.chunks(packed_len) {
            let mut values = chunk.to_vec();
            values.resize(packed_len, E::zero());
            rings.push(embed_ring_subfield_vector::<F, E, D>(
                &values,
                AkitaError::InvalidInput(
                    "root transformed witness does not encode in the ring-subfield basis"
                        .to_string(),
                ),
            )?);
        }
        Ok(crate::backend::dense::DensePoly::from_ring_coeffs::<D>(
            rings,
        ))
    }

    fn commit_inner_rows<B, const D: usize>(
        &self,
        backend: &B,
        prepared: &B::PreparedSetup,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        B: CommitmentComputeBackend<F>,
    {
        let blocks = self.blocks_for(D, block_len)?;
        backend.sparse_ring_commit_rows(
            prepared,
            SparseRingCommitRowsPlan {
                n_a,
                block_len,
                num_digits_commit,
                blocks: blocks.table(),
            },
        )
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
    challenges: &[SparseChallenge],
    num_blocks: usize,
    inner_width: usize,
    num_digits: usize,
) -> Vec<[i32; D]> {
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(inner_width.max(1));
    let pos_chunk = inner_width.div_ceil(actual_threads);
    let chunks: Vec<Vec<[i32; D]>> = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let pos_start = tid * pos_chunk;
            if pos_start >= inner_width {
                return Vec::new();
            }
            let pos_end = (pos_start + pos_chunk).min(inner_width);
            let mut acc = vec![[0i32; D]; pos_end - pos_start];
            let mut rotated = vec![[0i16; D]; D];

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
                    let weight = entry.value as i32;
                    for k in 0..D {
                        dst[k] += weight * i32::from(rot[k]);
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

pub(crate) fn column_sweep_sparse<F, const D: usize>(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::test_support::{
        aggregate_witnesses, negacyclic_tensor_product_challenges_i8, tensor_oracle_challenges,
    };
    use crate::DensePoly;
    use akita_field::Prime128OffsetA7F7 as F;

    #[test]
    fn sparse_ring_fold_matches_dense_reference() {
        const D: usize = 8;
        let sparse = SparseRingPoly::<F>::from_signed_coeffs(
            5,
            D,
            4,
            vec![(0, 1, 1), (1, 3, -1), (3, 2, 1)],
        )
        .unwrap();
        let mut dense_coeffs = vec![CyclotomicRing::<F, D>::zero(); 4];
        dense_coeffs[0].coeffs[1] += F::one();
        dense_coeffs[1].coeffs[3] -= F::one();
        dense_coeffs[3].coeffs[2] += F::one();
        let dense = DensePoly::from_ring_coeffs(dense_coeffs);
        let scalars = (0..2)
            .map(|idx| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                    F::from_u64(10 + idx * 10 + k as u64)
                }))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            sparse.fold_blocks_ring::<D>(&scalars, 2),
            dense.fold_blocks_ring::<D>(&scalars, 2)
        );
    }

    #[test]
    fn sparse_ring_fold_matches_dense_for_partial_final_slice() {
        const D: usize = 8;
        let sparse = SparseRingPoly::<F>::from_signed_coeffs(
            5,
            D,
            4,
            vec![(0, 1, 1), (1, 3, -1), (3, 2, 1)],
        )
        .unwrap();
        let mut dense_coeffs = vec![CyclotomicRing::<F, D>::zero(); 4];
        dense_coeffs[0].coeffs[1] += F::one();
        dense_coeffs[1].coeffs[3] -= F::one();
        dense_coeffs[3].coeffs[2] += F::one();
        let dense = DensePoly::from_ring_coeffs(dense_coeffs);
        let block_len = 8usize;
        let position_weights = (0..block_len)
            .map(|idx| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                    F::from_u64(10 + idx as u64 * 10 + k as u64)
                }))
            })
            .collect::<Vec<_>>();

        assert_eq!(
            sparse.fold_blocks_ring::<D>(&position_weights, block_len),
            dense.fold_blocks_ring::<D>(&position_weights, block_len)
        );
    }

    #[test]
    fn sparse_ring_tensor_decompose_fold_matches_negacyclic_product_reference() {
        const D: usize = 8;
        let block_len = 2;
        let num_digits = 1;
        let tensor = tensor_oracle_challenges::<D>();
        let polys = [
            SparseRingPoly::<F>::from_signed_coeffs(
                6,
                D,
                8,
                vec![(0, 1, 1), (1, 3, -1), (3, 2, 1), (6, 5, -1)],
            )
            .unwrap(),
            SparseRingPoly::<F>::from_signed_coeffs(
                6,
                D,
                8,
                vec![(0, 0, -1), (2, 4, 1), (5, 7, 1), (7, 2, -1)],
            )
            .unwrap(),
        ];
        let product_challenges = negacyclic_tensor_product_challenges_i8::<D>(&tensor).unwrap();

        let expected = aggregate_witnesses::<F, D>(
            &polys
                .iter()
                .zip(product_challenges.chunks(4))
                .map(|(poly, challenges)| {
                    poly.decompose_fold::<D>(challenges, block_len, num_digits, 0)
                })
                .collect::<Vec<_>>(),
        );
        let poly_refs = polys.iter().collect::<Vec<_>>();
        let got = SparseRingPoly::<F>::decompose_fold_tensor_batched::<D>(
            &poly_refs, &tensor, block_len, num_digits, 0,
        )
        .unwrap()
        .unwrap();

        assert_eq!(got, expected);
    }

    #[test]
    fn sparse_ring_tensor_decompose_fold_supports_partial_final_low_row() {
        const D: usize = 8;
        let block_len = 2;
        let num_digits = 1;
        let base_tensor = tensor_oracle_challenges::<D>();
        let tensor = TensorChallengeSet {
            fold_high: vec![
                base_tensor.fold_high[0].clone(),
                base_tensor.fold_high[2].clone(),
            ],
            fold_low: (0..2)
                .flat_map(|claim| {
                    (0..8).map({
                        let base_tensor = &base_tensor;
                        move |low| base_tensor.fold_low[claim * 2 + low % 2].clone()
                    })
                })
                .collect(),
            live_blocks_per_claim: 4,
            fold_low_len: 8,
            num_claims: 2,
        };
        let polys = [
            SparseRingPoly::<F>::from_signed_coeffs(
                6,
                D,
                8,
                vec![(0, 1, 1), (1, 3, -1), (3, 2, 1), (6, 5, -1)],
            )
            .unwrap(),
            SparseRingPoly::<F>::from_signed_coeffs(
                6,
                D,
                8,
                vec![(0, 0, -1), (2, 4, 1), (5, 7, 1), (7, 2, -1)],
            )
            .unwrap(),
        ];
        let product_challenges = negacyclic_tensor_product_challenges_i8::<D>(&tensor).unwrap();

        let expected = aggregate_witnesses::<F, D>(
            &polys
                .iter()
                .zip(product_challenges.chunks(4))
                .map(|(poly, challenges)| {
                    poly.decompose_fold::<D>(challenges, block_len, num_digits, 0)
                })
                .collect::<Vec<_>>(),
        );
        let poly_refs = polys.iter().collect::<Vec<_>>();
        let got = SparseRingPoly::<F>::decompose_fold_tensor_batched::<D>(
            &poly_refs, &tensor, block_len, num_digits, 0,
        )
        .unwrap()
        .unwrap();

        assert_eq!(got, expected);
    }

    #[test]
    fn sparse_ring_poly_caches_multiple_runtime_layouts() {
        let sparse = SparseRingPoly::<F>::from_signed_coeffs(
            8,
            32,
            8,
            vec![(0, 1, 1), (1, 3, -1), (7, 31, 1)],
        )
        .unwrap();

        let d32_blocks = sparse.blocks_for(32, 4).unwrap();
        let d64_blocks = sparse.blocks_for(64, 2).unwrap();

        assert_eq!(d32_blocks.num_blocks(), 2);
        assert_eq!(d64_blocks.num_blocks(), 2);
        assert_eq!(sparse.block_cache.lock().unwrap().len(), 2);
    }

    #[test]
    fn sorted_sparse_ring_constructor_rejects_unsorted_coeffs() {
        const D: usize = 8;
        let sorted =
            SparseRingPoly::<F>::from_sorted_signed_coeffs(5, D, 4, vec![(0, 1, 1), (2, 3, -1)])
                .unwrap();
        assert_eq!(sorted.num_ring_elems(), 4);

        assert!(SparseRingPoly::<F>::from_sorted_signed_coeffs(
            5,
            D,
            4,
            vec![(2, 3, -1), (0, 1, 1)],
        )
        .is_err());
    }

    #[test]
    fn sparse_ring_constructor_rejects_non_signed_unit_coefficients() {
        const D: usize = 8;
        for value in [-2, 0, 2] {
            assert!(matches!(
                SparseRingPoly::<F>::from_signed_coeffs(5, D, 4, vec![(0, 1, value)]),
                Err(AkitaError::InvalidInput(_))
            ));
        }
    }

    #[test]
    fn packed_sparse_ring_constructor_matches_tuple_constructor() {
        const D: usize = 8;
        let tuples = vec![(0, 1, 1), (1, 3, -1), (3, 2, 1)];
        let packed = tuples
            .iter()
            .copied()
            .map(|(ring_idx, coeff_idx, value)| {
                SparseRingCoeff::from_ring_coords(ring_idx, coeff_idx, D, value).unwrap()
            })
            .collect::<Vec<_>>();
        let from_tuples = SparseRingPoly::<F>::from_signed_coeffs(5, D, 4, tuples).unwrap();
        let from_packed = SparseRingPoly::<F>::from_packed_coeffs(5, D, 4, packed).unwrap();

        let scalars = (0..2)
            .map(|idx| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                    F::from_u64(20 + idx * 10 + k as u64)
                }))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            from_packed.fold_blocks_ring::<D>(&scalars, 2),
            from_tuples.fold_blocks_ring::<D>(&scalars, 2)
        );
    }
}
