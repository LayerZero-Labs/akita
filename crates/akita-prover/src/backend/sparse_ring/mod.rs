//! Sparse signed ring-coefficient polynomial backend.
//!
//! This is the natural backend for Frobenius-packed one-hot tables: after
//! canonical-basis packing, each original one-hot chunk becomes a small number
//! of signed monomial coefficients inside the committed ring table.

use akita_algebra::ring::cyclotomic::WideCyclotomicRing;
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_field::parallel::*;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AdditiveGroup, AkitaError, CanonicalField, FieldCore, FromPrimitiveInt};
use akita_types::{embed_ring_subfield_vector, RingMatrixView};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::backend::flat_blocks::FlatBlocks;
use crate::backend::poly_helpers::build_decompose_fold_witness;
use crate::DecomposeFoldWitness;

mod ops;

pub use ops::{SparseRingBatchView, SparseRingView};

type SparseLayoutCacheKey = (usize, usize);
type SparseBlockCache =
    Arc<Mutex<HashMap<SparseLayoutCacheKey, Arc<FlatBlocks<SparseRingBlockEntry>>>>>;

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
    pub(crate) fn new(pos_in_block: u32, coeff_idx: u16, value: i8) -> Self {
        Self {
            pos_in_block,
            coeff_idx,
            value,
        }
    }

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

impl FlatBlocks<SparseRingBlockEntry> {
    fn from_coeffs(
        coeffs: &[SparseRingCoeff],
        ring_d: usize,
        total_ring_elems: usize,
        num_positions_per_block: usize,
    ) -> Result<Self, AkitaError> {
        if ring_d == 0 {
            return Err(AkitaError::InvalidInput(
                "ring_d must be nonzero".to_string(),
            ));
        }
        if num_positions_per_block == 0 || !num_positions_per_block.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "num_positions_per_block={num_positions_per_block} must be a nonzero power of two"
            )));
        }
        if u32::try_from(num_positions_per_block).is_err() {
            return Err(AkitaError::InvalidInput(format!(
                "num_positions_per_block={num_positions_per_block} exceeds u32::MAX"
            )));
        }
        let num_live_blocks = total_ring_elems.div_ceil(num_positions_per_block);
        let mut blocks = Self::with_capacity(num_live_blocks, coeffs.len());
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
            let block_idx = ring_idx / num_positions_per_block;
            let pos_in_block = u32::try_from(ring_idx % num_positions_per_block).map_err(|_| {
                AkitaError::InvalidInput("sparse ring block position exceeds u32".to_string())
            })?;
            blocks.push_entry(
                &mut current_block,
                block_idx,
                num_live_blocks,
                SparseRingBlockEntry::new(pos_in_block, coeff_idx, coeff.value),
            )?;
        }
        blocks.finish_build(current_block, num_live_blocks)
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
    /// Cached per-block layouts keyed by `(ring_d, num_positions_per_block)`.
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
        num_positions_per_block: usize,
    ) -> Result<Arc<FlatBlocks<SparseRingBlockEntry>>, AkitaError> {
        let key = (ring_d, num_positions_per_block);
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
        let built = FlatBlocks::<SparseRingBlockEntry>::from_coeffs(
            &self.coeffs,
            ring_d,
            ring_elems_at_d,
            num_positions_per_block,
        )?;
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
    /// This is the D-free field materialization used by the tensor helpers.
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
        num_positions_per_block: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let blocks = self
            .blocks_for(D, num_positions_per_block)
            .expect("SparseRingPoly::fold_blocks: invalid num_positions_per_block");
        cfg_into_iter!(0..blocks.num_live_blocks())
            .map(|block_idx| {
                fold_sparse_block(blocks.block(block_idx), scalars, num_positions_per_block)
            })
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn fold_blocks_ring<const D: usize>(
        &self,
        scalars: &[CyclotomicRing<F, D>],
        num_positions_per_block: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let blocks = self
            .blocks_for(D, num_positions_per_block)
            .expect("SparseRingPoly::fold_blocks_ring: invalid num_positions_per_block");
        cfg_into_iter!(0..blocks.num_live_blocks())
            .map(|block_idx| {
                fold_sparse_block_ring(blocks.block(block_idx), scalars, num_positions_per_block)
            })
            .collect()
    }

    pub(crate) fn fold_blocks_subfield<const D: usize>(
        &self,
        multipliers: &akita_types::SubfieldMultiplierOpeningPoint<F>,
        num_positions_per_block: usize,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        let blocks = self.blocks_for(D, num_positions_per_block)?;
        cfg_into_iter!(0..blocks.num_live_blocks())
            .map(|block_idx| {
                fold_sparse_block_subfield(
                    blocks.block(block_idx),
                    multipliers,
                    num_positions_per_block,
                )
            })
            .collect()
    }

    pub(crate) fn evaluate_and_fold<const D: usize>(
        &self,
        live_block_weights: &[F],
        position_weights: &[F],
        num_positions_per_block: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        let folded = self.fold_blocks::<D>(position_weights, num_positions_per_block);
        crate::backend::poly_helpers::fused_evaluate_and_fold_base(folded, live_block_weights)
    }

    pub(crate) fn evaluate_and_fold_subfield<const D: usize>(
        &self,
        multipliers: &akita_types::SubfieldMultiplierOpeningPoint<F>,
        num_positions_per_block: usize,
    ) -> Result<(CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>), AkitaError> {
        crate::backend::poly_helpers::fused_evaluate_and_fold_subfield(
            self.fold_blocks_subfield::<D>(multipliers, num_positions_per_block)?,
            multipliers,
        )
    }

    #[tracing::instrument(skip_all, name = "SparseRingPoly::decompose_fold")]
    pub(crate) fn decompose_fold<const D: usize>(
        &self,
        challenges: &[SparseChallenge],
        num_positions_per_block: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> DecomposeFoldWitness<F> {
        let blocks = self
            .blocks_for(D, num_positions_per_block)
            .expect("SparseRingPoly::decompose_fold: invalid num_positions_per_block");
        let num_live_blocks = challenges.len().min(blocks.num_live_blocks());
        let inner_width = num_positions_per_block * num_digits;
        let coeff_accum = sparse_accumulate::<D>(
            &blocks,
            challenges,
            num_live_blocks,
            inner_width,
            num_digits,
        );
        let modulus = (-F::one()).to_canonical_u128() + 1;
        build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
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
}

fn fold_sparse_block<F, const D: usize>(
    entries: &[SparseRingBlockEntry],
    scalars: &[F],
    num_positions_per_block: usize,
) -> CyclotomicRing<F, D>
where
    F: FieldCore + FromPrimitiveInt,
{
    let mut coeffs = [F::zero(); D];
    for entry in entries {
        let pos = entry.pos_in_block();
        if pos < scalars.len() && pos < num_positions_per_block {
            coeffs[entry.coeff_idx()] += scalars[pos] * F::from_i8(entry.value);
        }
    }
    CyclotomicRing::from_coefficients(coeffs)
}

#[cfg(test)]
fn fold_sparse_block_ring<F, const D: usize>(
    entries: &[SparseRingBlockEntry],
    scalars: &[CyclotomicRing<F, D>],
    num_positions_per_block: usize,
) -> CyclotomicRing<F, D>
where
    F: FieldCore + FromPrimitiveInt,
{
    let mut acc = CyclotomicRing::<F, D>::zero();
    for entry in entries {
        let pos = entry.pos_in_block();
        if pos < scalars.len() && pos < num_positions_per_block {
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

fn fold_sparse_block_subfield<F, const D: usize>(
    entries: &[SparseRingBlockEntry],
    multipliers: &akita_types::SubfieldMultiplierOpeningPoint<F>,
    num_positions_per_block: usize,
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
{
    let mut acc = CyclotomicRing::<F, D>::zero();
    for entry in entries {
        let position = entry.pos_in_block();
        if position < num_positions_per_block {
            multipliers.accumulate_position_monomial(
                position,
                entry.coeff_idx(),
                F::from_i8(entry.value),
                &mut acc,
            )?;
        }
    }
    Ok(acc)
}

fn sparse_accumulate<const D: usize>(
    blocks: &FlatBlocks<SparseRingBlockEntry>,
    challenges: &[SparseChallenge],
    num_live_blocks: usize,
    inner_width: usize,
    num_digits: usize,
) -> Vec<[i32; D]> {
    if inner_width == 0 {
        return Vec::new();
    }

    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let row_chunk = inner_width.div_ceil(num_threads.min(inner_width.max(1)));
    let mut centered = vec![[0i32; D]; inner_width];
    cfg_chunks_mut!(&mut centered, row_chunk)
        .enumerate()
        .for_each(|(chunk_idx, rows)| {
            let row_start = chunk_idx * row_chunk;
            let row_end = row_start + rows.len();
            for (block_idx, challenge) in challenges.iter().enumerate().take(num_live_blocks) {
                let entries = blocks.block(block_idx);
                let lo = entries.partition_point(|e| e.pos_in_block() * num_digits < row_start);
                let hi = entries.partition_point(|e| e.pos_in_block() * num_digits < row_end);
                for entry in &entries[lo..hi] {
                    let local_row = entry.pos_in_block() * num_digits - row_start;
                    let dst = &mut rows[local_row];
                    let source_coeff = i32::from(entry.value);
                    for (&challenge_pos, &challenge_coeff) in
                        challenge.positions.iter().zip(&challenge.coeffs)
                    {
                        let target = entry.coeff_idx() + challenge_pos as usize;
                        let value = source_coeff * i32::from(challenge_coeff);
                        if target < D {
                            dst[target] += value;
                        } else {
                            dst[target - D] -= value;
                        }
                    }
                }
            }
        });
    centered
}

type WeightedColEntry = (usize, u32, u16, i8);
type WeightedPosEntry = (u32, u16, i8);

fn sparse_block_tile_for_scratch<F, const D: usize>(
    blocks: &[&[SparseRingBlockEntry]],
    n_a: usize,
    num_positions_per_block: usize,
    scratch_bytes_per_worker: usize,
) -> Result<usize, AkitaError>
where
    F: FieldCore + HasWide,
{
    let wide_accums = n_a
        .checked_mul(D)
        .and_then(|count| count.checked_mul(std::mem::size_of::<F::Wide>()))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("sparse commitment accumulator size overflow".into())
        })?;
    let max_entries = blocks
        .iter()
        .map(|entries| entries.len())
        .max()
        .unwrap_or(0);
    // Both index vectors retain their allocation across tiles, even though
    // only one is populated for a given sweep.
    let entry_indexes = max_entries
        .checked_mul(
            std::mem::size_of::<WeightedColEntry>() + std::mem::size_of::<WeightedPosEntry>(),
        )
        .ok_or_else(|| {
            AkitaError::InvalidSetup("sparse commitment entry scratch overflow".into())
        })?;
    let per_block = wide_accums.checked_add(entry_indexes).ok_or_else(|| {
        AkitaError::InvalidSetup("sparse commitment tile scratch overflow".into())
    })?;
    let fixed = num_positions_per_block
        .checked_add(1)
        .and_then(|count| count.checked_mul(2 * std::mem::size_of::<usize>()))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("sparse commitment index scratch overflow".into())
        })?;
    let minimum = fixed.checked_add(per_block).ok_or_else(|| {
        AkitaError::InvalidSetup("sparse commitment minimum scratch overflow".into())
    })?;
    if minimum > scratch_bytes_per_worker {
        return Err(AkitaError::InvalidSetup(format!(
            "sparse commitment geometry needs at least {minimum} scratch bytes per worker but the CPU backend allows {scratch_bytes_per_worker}"
        )));
    }
    Ok(((scratch_bytes_per_worker - fixed) / per_block.max(1))
        .max(1)
        .min(blocks.len().max(1)))
}

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
    a_view: &RingMatrixView<'_, F, D>,
    blocks: &[&[SparseRingBlockEntry]],
    n_a: usize,
    num_positions_per_block: usize,
    num_digits_inner: usize,
    scratch_bytes_per_worker: usize,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    let num_live_blocks = blocks.len();
    let block_tile = sparse_block_tile_for_scratch::<F, D>(
        blocks,
        n_a,
        num_positions_per_block,
        scratch_bytes_per_worker,
    )?;

    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads().min(num_live_blocks).max(1);
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;
    let blocks_per_thread = num_live_blocks.div_ceil(num_threads);

    let thread_results: Vec<Vec<Vec<CyclotomicRing<F, D>>>> = cfg_into_iter!(0..num_threads)
        .map(|tid| {
            let block_start = tid * blocks_per_thread;
            let block_end = (block_start + blocks_per_thread).min(num_live_blocks);
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
                if entry_count >= num_positions_per_block {
                    pos_offsets.clear();
                    pos_offsets.resize(num_positions_per_block + 1, 0);
                    for block_entries in tile_blocks {
                        for entry in *block_entries {
                            pos_offsets[entry.pos_in_block() + 1] += 1;
                        }
                    }
                    for pos in 1..=num_positions_per_block {
                        pos_offsets[pos] += pos_offsets[pos - 1];
                    }

                    pos_entries.clear();
                    pos_entries.resize(entry_count, (0, 0, 0));
                    pos_cursor.clear();
                    pos_cursor.extend_from_slice(&pos_offsets[..num_positions_per_block]);
                    for (local_b, block_entries) in tile_blocks.iter().enumerate() {
                        for entry in *block_entries {
                            let pos = entry.pos_in_block();
                            let dst = pos_cursor[pos];
                            pos_cursor[pos] += 1;
                            pos_entries[dst] = (local_b as u32, entry.coeff_idx, entry.value);
                        }
                    }

                    for (a_idx, a_row) in a_view.rows().take(n_a).enumerate() {
                        for pos in 0..num_positions_per_block {
                            let start = pos_offsets[pos];
                            let end = pos_offsets[pos + 1];
                            if start == end {
                                continue;
                            }
                            let a_wide =
                                WideCyclotomicRing::from_ring(&a_row[pos * num_digits_inner]);
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
                                entry.pos_in_block() * num_digits_inner,
                                local_b as u32,
                                entry.coeff_idx,
                                entry.value,
                            ));
                        }
                    }
                    col_entries.sort_unstable_by_key(|&(col, _, _, _)| col);

                    for (a_idx, a_row) in a_view.rows().take(n_a).enumerate() {
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

    let mut out = Vec::with_capacity(num_live_blocks);
    for thread_blocks in thread_results {
        out.extend(thread_blocks);
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
