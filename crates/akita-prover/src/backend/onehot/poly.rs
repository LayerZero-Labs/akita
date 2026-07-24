use super::*;

type LayoutCacheKey = (usize, usize);
type OneHotBlockCache = Arc<Mutex<HashMap<LayoutCacheKey, Arc<OneHotBlocks>>>>;
type TensorRootCache<F> = Arc<Mutex<HashMap<LayoutCacheKey, Arc<SparseRingPoly<F>>>>>;

/// One-hot polynomial: sparse witness with at most one nonzero field element
/// per chunk of size `onehot_k`.
///
/// The polynomial is stored layout-agnostically as the flat list of hot
/// indices supplied at construction. Each op takes `num_positions_per_block` at call time
/// and the per-block bucketing is materialized lazily per `(ring_d, num_positions_per_block)`.
/// That mirrors how [`DensePoly`](crate::DensePoly) accepts `num_positions_per_block` per op
/// and keeps `OneHotPoly` free of the commit-layout parameters it used to bake
/// in at construction.
///
/// Storage is D-free: the per-chunk hot indices are flat logical data, and
/// the ring dimension is a view selected at kernel entry (each ring-shaped
/// method takes it as a const generic).
///
/// Generic over `I`: the index type accepted and stored per chunk. Use `u8`
/// when `onehot_k <= 256` to reduce index storage footprint.
#[derive(Debug, Clone)]
pub struct OneHotPoly<F: FieldCore, I: OneHotIndex = usize> {
    pub(crate) num_vars: usize,
    pub(crate) onehot_k: usize,
    /// Per-chunk hot-position indices. `None` denotes an all-zero chunk.
    pub(crate) indices: Vec<Option<I>>,
    /// Ring-element count at the CONSTRUCTION dimension; metadata, not
    /// authority — kernels validate at their own dimension.
    pub(crate) total_ring_elems: usize,
    /// Cached per-block layouts keyed by `(ring_d, num_positions_per_block)`.
    pub(crate) block_cache: OneHotBlockCache,
    /// Cached tensor-projected sparse root polynomials keyed by `(ring_d, width)`.
    pub(crate) tensor_root_cache: TensorRootCache<F>,
    pub(crate) _marker: PhantomData<(F, I)>,
}

impl<F: FieldCore, I: OneHotIndex> OneHotPoly<F, I> {
    /// Build a one-hot polynomial from chunk size and hot-position indices.
    ///
    /// `indices[c]` is the hot position in chunk `c` (`None` for all-zero chunks).
    ///
    /// `ring_d` is the caller's configured ring dimension. It is recorded as
    /// construction metadata (the [`crate::compute::RootPolyMeta`] ring-element
    /// count) only; the storage itself is D-free and kernels select their view
    /// dimension at entry.
    ///
    /// The commit-layout split (how blocks are tiled within the polynomial)
    /// is no longer baked in at construction. Each op receives `num_positions_per_block`
    /// from the caller and the per-block representation is materialized on
    /// demand.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are inconsistent, any index is out of
    /// range, or `onehot_k` and `ring_d` are not nicely matched.
    pub fn new(
        onehot_k: usize,
        ring_d: usize,
        indices: Vec<Option<I>>,
    ) -> Result<Self, AkitaError> {
        if onehot_k == 0 {
            return Err(AkitaError::InvalidInput(
                "onehot_k must be nonzero".to_string(),
            ));
        }
        if ring_d == 0 {
            return Err(AkitaError::InvalidInput(
                "ring_d must be nonzero".to_string(),
            ));
        }
        if !(onehot_k.is_multiple_of(ring_d) || ring_d.is_multiple_of(onehot_k)) {
            return Err(AkitaError::InvalidInput(format!(
                "onehot_k={onehot_k} and D={ring_d} must be nicely matched (one divides the other)"
            )));
        }
        let total_field_elems = indices.len().checked_mul(onehot_k).ok_or_else(|| {
            AkitaError::InvalidInput("onehot total field element count overflow".to_string())
        })?;
        if !total_field_elems.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "onehot total field elements {total_field_elems} is not a power of two"
            )));
        }
        if !total_field_elems.is_multiple_of(ring_d) {
            return Err(AkitaError::InvalidInput(format!(
                "total field elements {total_field_elems} is not divisible by D={ring_d}"
            )));
        }
        let total_ring_elems = total_field_elems / ring_d;
        for (chunk_idx, opt) in indices.iter().copied().enumerate() {
            if let Some(raw) = opt {
                let idx = raw.as_usize();
                if idx >= onehot_k {
                    return Err(AkitaError::InvalidInput(format!(
                        "index {idx} out of range for chunk size K={onehot_k} at position {chunk_idx}"
                    )));
                }
            }
        }
        Ok(Self {
            num_vars: total_field_elems.trailing_zeros() as usize,
            onehot_k,
            indices,
            total_ring_elems,
            block_cache: Arc::new(Mutex::new(HashMap::new())),
            tensor_root_cache: Arc::new(Mutex::new(HashMap::new())),
            _marker: PhantomData,
        })
    }

    /// Number of field-evaluation slots in each compact one-hot chunk.
    #[inline]
    pub fn onehot_k(&self) -> usize {
        self.onehot_k
    }

    /// Per-chunk hot-position indices. `None` denotes an all-zero chunk.
    #[inline]
    pub fn indices(&self) -> &[Option<I>] {
        &self.indices
    }

    /// Materialize the dense field-evaluation table directly from the flat
    /// hot-index positions.
    ///
    /// This is the D-free field materialization used by the tensor helpers.
    ///
    /// # Errors
    ///
    /// Returns an error when the evaluation-table length overflows `usize` or
    /// a hot position falls outside the table.
    pub(crate) fn direct_field_evals(&self) -> Result<Vec<F>, AkitaError> {
        let total_evals = 1usize.checked_shl(self.num_vars as u32).ok_or_else(|| {
            AkitaError::InvalidInput(format!("2^{} does not fit usize", self.num_vars))
        })?;
        let mut evals = vec![F::zero(); total_evals];
        for (chunk_idx, opt) in self.indices.iter().copied().enumerate() {
            let Some(raw) = opt else {
                continue;
            };
            let field_pos = chunk_idx
                .checked_mul(self.onehot_k)
                .and_then(|base| base.checked_add(raw.as_usize()))
                .ok_or_else(|| {
                    AkitaError::InvalidInput("onehot direct witness index overflow".to_string())
                })?;
            if field_pos >= evals.len() {
                return Err(AkitaError::InvalidInput(format!(
                    "onehot direct witness index {field_pos} out of range for {} evals",
                    evals.len()
                )));
            }
            evals[field_pos] = F::one();
        }
        Ok(evals)
    }

    /// Return cached per-block storage, building it on first call for the
    /// requested `(ring_d, num_positions_per_block)` view.
    pub(super) fn blocks_for(
        &self,
        ring_d: usize,
        num_positions_per_block: usize,
    ) -> Result<Arc<OneHotBlocks>, AkitaError> {
        let key = (ring_d, num_positions_per_block);
        if let Some(blocks) = self
            .block_cache
            .lock()
            .map_err(|_| AkitaError::InvalidSetup("onehot block cache lock poisoned".into()))?
            .get(&key)
        {
            return Ok(Arc::clone(blocks));
        }
        // Slow path: build blocks and install them. Validate `ring_d` and
        // `num_positions_per_block` *before* building so the error path is cheap.
        if num_positions_per_block == 0 || !num_positions_per_block.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "num_positions_per_block={num_positions_per_block} must be a nonzero power of two"
            )));
        }
        let field_len = 1usize
            .checked_shl(self.num_vars as u32)
            .ok_or_else(|| AkitaError::InvalidInput("onehot arity overflow".to_string()))?;
        if ring_d == 0 {
            return Err(AkitaError::InvalidInput(
                "ring_d must be nonzero".to_string(),
            ));
        }
        let ring_elems_at_d = field_len.div_ceil(ring_d);
        // Kernel-entry view validation: the layout invariants `OneHotPoly::new`
        // pinned at the construction dimension must also hold at the view
        // dimension the blocks are built for.
        if !(self.onehot_k.is_multiple_of(ring_d) || ring_d.is_multiple_of(self.onehot_k)) {
            return Err(AkitaError::InvalidInput(format!(
                "onehot_k={} and D={ring_d} must be nicely matched (one divides the other)",
                self.onehot_k
            )));
        }
        let built = {
            let _span =
                tracing::debug_span!("OneHotPoly::build_blocks", ring_d, num_positions_per_block)
                    .entered();
            self.build_blocks_inner(ring_d, num_positions_per_block, ring_elems_at_d)?
        };
        let mut cache = self
            .block_cache
            .lock()
            .map_err(|_| AkitaError::InvalidSetup("onehot block cache lock poisoned".into()))?;
        Ok(Arc::clone(
            cache.entry(key).or_insert_with(|| Arc::new(built)),
        ))
    }

    /// Sparse fast path for `tensor_extension_column_partials_batch`.
    /// (the `split_bits <= low_vars`, power-of-two `onehot_k`, shared-shape
    /// case). Byte-identical to the dense column partials but exploits the
    /// one-hot structure to replace the per-chunk extension *multiply* of the
    /// dense path with a per-chunk extension *add*.
    ///
    /// The caller supplies the opening point already split into Lagrange
    /// factor tables:
    /// * `low_tail_weights = eq(point[split_bits..low_vars])`
    /// * the high `hi_vars = num_vars - low_vars` coordinates factored as
    ///   `low_eq = eq(point[low_vars..low_vars + inner_bits])` and
    ///   `high_eq = eq(point[low_vars + inner_bits..])`, so that the high
    ///   Lagrange weight of chunk `c = (j << inner_bits) | i` is exactly
    ///   `high_eq[j] * low_eq[i]` (the standard little-endian tensor split of
    ///   the `eq` table). We therefore never materialize the full
    ///   `2^hi_vars`-entry weight table.
    ///
    /// Each chunk carries a single hot position `raw in 0..onehot_k`. We:
    /// 1. scatter `low_eq[i]` into a `raw`-indexed scratch table using *adds
    ///    only* (one add per nonzero chunk),
    /// 2. fold the scratch into a running per-`raw` bucket with one multiply by
    ///    `high_eq[j]` per touched `raw` (cheap: at most `onehot_k` per outer
    ///    block), and
    /// 3. collapse the `onehot_k` buckets into the `width` column partials via
    ///    `partials[raw & (width - 1)] += bucket[raw] * low_tail_weights[raw >> split_bits]`.
    ///
    /// Field addition/multiplication are exactly associative, commutative, and
    /// distributive, so the bucket regrouping and the parallel block split both
    /// yield the identical field element the dense path produces.
    pub(super) fn tensor_column_partials_from_shared_eq<E>(
        &self,
        split_bits: usize,
        width: usize,
        inner_bits: usize,
        low_eq: &[E],
        high_eq: &[E],
        low_tail_weights: &[E],
    ) -> Vec<E>
    where
        E: ExtField<F>,
    {
        let onehot_k = self.onehot_k;
        let head_mask = width - 1;
        let inner_len = low_eq.len();
        let num_live_blocks = high_eq.len();
        let zero = E::zero();
        debug_assert_eq!(inner_len, 1usize << inner_bits);
        debug_assert_eq!(self.indices.len(), num_live_blocks * inner_len);

        // Partition the outer blocks into contiguous ranges so the heavy
        // scatter is parallel; each range accumulates an independent per-`raw`
        // bucket which we then reduce (addition is associative, so the result
        // is independent of the range split).
        #[cfg(feature = "parallel")]
        let target_ranges = rayon::current_num_threads().max(1) * 4;
        #[cfg(not(feature = "parallel"))]
        let target_ranges = 1usize;
        let range_len = num_live_blocks.div_ceil(target_ranges.max(1)).max(1);
        let ranges = (0..num_live_blocks)
            .step_by(range_len)
            .map(|start| (start, (start + range_len).min(num_live_blocks)))
            .collect::<Vec<_>>();

        let partial_buckets = cfg_into_iter!(ranges)
            .map(|(jstart, jend)| {
                let mut bucket = vec![zero; onehot_k];
                let mut scratch = vec![zero; onehot_k];
                let mut touched = vec![false; onehot_k];
                let mut touched_raws = Vec::with_capacity(inner_len.min(onehot_k));
                for (jrel, &hj) in high_eq[jstart..jend].iter().enumerate() {
                    let base = (jstart + jrel) << inner_bits;
                    let block = &self.indices[base..base + inner_len];
                    for (hot, &le) in block.iter().copied().zip(low_eq.iter()) {
                        if let Some(raw) = hot {
                            let raw = raw.as_usize();
                            if !touched[raw] {
                                touched[raw] = true;
                                touched_raws.push(raw);
                            }
                            scratch[raw] += le;
                        }
                    }
                    for raw in touched_raws.drain(..) {
                        let slot = &mut scratch[raw];
                        bucket[raw] += hj * *slot;
                        *slot = zero;
                        touched[raw] = false;
                    }
                }
                bucket
            })
            .collect::<Vec<_>>();

        let mut bucket = vec![zero; onehot_k];
        for partial in &partial_buckets {
            for (acc, part) in bucket.iter_mut().zip(partial.iter()) {
                *acc += *part;
            }
        }

        let mut partials = vec![zero; width];
        for (raw, &value) in bucket.iter().enumerate() {
            if value != zero {
                partials[raw & head_mask] += value * low_tail_weights[raw >> split_bits];
            }
        }
        partials
    }

    pub(super) fn tensor_packed_sparse_witness<E>(
        &self,
    ) -> Result<SparseExtensionOpeningWitness<E>, AkitaError>
    where
        E: ExtField<F>,
    {
        let (width, total_evals) = self.tensor_packing_shape::<E>()?;
        let table_len = total_evals / width;
        let _span = tracing::info_span!(
            "OneHotPoly::tensor_packed_sparse_witness",
            width,
            table_len,
            chunks = self.indices.len()
        )
        .entered();
        let mut entries = Vec::with_capacity(self.indices.len());
        for (chunk_idx, opt) in self.indices.iter().copied().enumerate() {
            let Some(raw) = opt else {
                continue;
            };
            let field_pos = self.hot_field_position(chunk_idx, raw, "tensor-packed witness")?;
            let tail = field_pos / width;
            let head = field_pos % width;
            let mut coords = vec![F::zero(); width];
            coords[head] = F::one();
            entries.push((tail, E::from_base_slice(&coords)));
        }
        SparseExtensionOpeningWitness::new(table_len, entries)
    }

    pub(super) fn tensor_packed_sparse_ring_poly<E, const D: usize>(
        &self,
    ) -> Result<Arc<SparseRingPoly<F>>, AkitaError>
    where
        F: FromPrimitiveInt,
        E: FpExtEncoding<F>,
    {
        let (width, total_evals) = self.tensor_packing_shape::<E>()?;
        let _span = tracing::info_span!(
            "OneHotPoly::tensor_packed_sparse_ring_poly",
            width,
            total_evals,
            chunks = self.indices.len()
        )
        .entered();
        if !D.is_multiple_of(width) {
            return Err(AkitaError::InvalidInput(
                "tensor width must divide root ring dimension".to_string(),
            ));
        }
        let double_width = width.checked_mul(2).ok_or_else(|| {
            AkitaError::InvalidInput(
                "tensor width is too large for root ring projection".to_string(),
            )
        })?;
        if D < double_width {
            return Err(AkitaError::InvalidInput(
                "root ring dimension must be at least twice the tensor width".to_string(),
            ));
        }
        let packed_len = D / width;
        let half = D / double_width;
        let step = D / double_width;
        let total_ring_elems = total_evals / D;
        let key = (D, width);
        if let Some(poly) = self
            .tensor_root_cache
            .lock()
            .map_err(|_| AkitaError::InvalidSetup("onehot tensor cache lock poisoned".into()))?
            .get(&key)
        {
            return Ok(Arc::clone(poly));
        }
        let mut coeffs = Vec::with_capacity(self.indices.len() * width.min(2));

        for (chunk_idx, opt) in self.indices.iter().copied().enumerate() {
            let Some(raw) = opt else {
                continue;
            };
            let field_pos = self.hot_field_position(chunk_idx, raw, "tensor-projected ring")?;
            let tail = field_pos / width;
            let coord = field_pos % width;
            let ring_idx = tail / packed_len;
            let slot_idx = tail % packed_len;
            if slot_idx < half {
                let shift = slot_idx;
                if coord == 0 {
                    coeffs.push(SparseRingCoeff::from_ring_coords(ring_idx, shift, D, 1)?);
                } else {
                    let pos_offset = coord * step;
                    coeffs.push(SparseRingCoeff::from_ring_coords(
                        ring_idx,
                        shift + pos_offset,
                        D,
                        1,
                    )?);
                    coeffs.push(SparseRingCoeff::from_ring_coords(
                        ring_idx,
                        shift + D - pos_offset,
                        D,
                        -1,
                    )?);
                }
            } else {
                let shift = slot_idx - half + D / 2;
                if coord == 0 {
                    coeffs.push(SparseRingCoeff::from_ring_coords(ring_idx, shift, D, 1)?);
                } else {
                    let pos_offset = coord * step;
                    coeffs.push(SparseRingCoeff::from_ring_coords(
                        ring_idx,
                        shift - pos_offset,
                        D,
                        1,
                    )?);
                    coeffs.push(SparseRingCoeff::from_ring_coords(
                        ring_idx,
                        shift + pos_offset,
                        D,
                        1,
                    )?);
                }
            }
        }

        let poly = if self.onehot_k >= D {
            SparseRingPoly::<F>::from_sorted_packed_coeffs(
                self.num_vars,
                D,
                total_ring_elems,
                coeffs,
            )
        } else {
            SparseRingPoly::<F>::from_packed_coeffs(self.num_vars, D, total_ring_elems, coeffs)
        }?;
        let poly = Arc::new(poly);
        let mut cache = self
            .tensor_root_cache
            .lock()
            .map_err(|_| AkitaError::InvalidSetup("onehot tensor cache lock poisoned".into()))?;
        Ok(Arc::clone(cache.entry(key).or_insert(poly)))
    }

    pub(super) fn tensor_packing_shape<E>(&self) -> Result<(usize, usize), AkitaError>
    where
        E: ExtField<F>,
    {
        let (split_bits, width) = akita_types::tensor_opening_split::<F, E>()?;
        if split_bits > self.num_vars {
            return Err(AkitaError::InvalidInput(
                "extension-opening tensor split exceeds polynomial arity".to_string(),
            ));
        }
        let total_evals = 1usize.checked_shl(self.num_vars as u32).ok_or_else(|| {
            AkitaError::InvalidInput(format!("2^{} does not fit usize", self.num_vars))
        })?;
        Ok((width, total_evals))
    }

    pub(super) fn hot_field_position(
        &self,
        chunk_idx: usize,
        raw: I,
        context: &'static str,
    ) -> Result<usize, AkitaError> {
        chunk_idx
            .checked_mul(self.onehot_k)
            .and_then(|base| base.checked_add(raw.as_usize()))
            .ok_or_else(|| AkitaError::InvalidInput(format!("onehot {context} index overflow")))
    }

    pub(super) fn next_tensor_packed_sparse_position(
        &self,
        cursor: &mut usize,
        width: usize,
    ) -> Result<Option<(usize, usize)>, AkitaError> {
        while *cursor < self.indices.len() {
            let chunk_idx = *cursor;
            *cursor += 1;
            let Some(raw) = self.indices[chunk_idx] else {
                continue;
            };
            let field_pos =
                self.hot_field_position(chunk_idx, raw, "tensor-packed witness batch")?;
            return Ok(Some((field_pos / width, field_pos % width)));
        }
        Ok(None)
    }

    pub(super) fn build_blocks_inner(
        &self,
        ring_d: usize,
        num_positions_per_block: usize,
        ring_elems_at_d: usize,
    ) -> Result<OneHotBlocks, AkitaError> {
        // `blocks_for` has already validated that `num_positions_per_block` is a nonzero
        // power of two and that
        // K and `ring_d` are nicely matched; `OneHotPoly::new` has validated
        // that every per-chunk index is in range. Here we only need to
        // compute `num_live_blocks` for the flat-layout offsets array and check
        // that `num_positions_per_block` and `ring_d` fit in the packed entry field widths.
        if u32::try_from(num_positions_per_block).is_err() {
            return Err(AkitaError::InvalidInput(format!(
                "num_positions_per_block={num_positions_per_block} exceeds u32::MAX and cannot be packed into an entry"
            )));
        }
        // Coefficient indices inside a ring element are `< ring_d` and get
        // packed as `u16` in the entry types below (see
        // `SingleChunkEntry::coeff_idx` and `MultiChunkEntry::nonzero_coeffs`).
        // Reject out-of-range `ring_d` here rather than silently truncating below.
        if ring_d > usize::from(u16::MAX) + 1 {
            return Err(AkitaError::InvalidInput(format!(
                "D={ring_d} exceeds 65536 and cannot be packed into SingleChunkEntry::coeff_idx / MultiChunkEntry::nonzero_coeffs (both `u16`)"
            )));
        }
        let num_live_blocks = ring_elems_at_d.div_ceil(num_positions_per_block);

        // The single-chunk (one-hot-chunk-per-ring-element) layout
        // applies when K >= D && D | K; otherwise fall back to the
        // multi-chunk layout.
        if self.onehot_k >= ring_d && self.onehot_k.is_multiple_of(ring_d) {
            Ok(OneHotBlocks::SingleChunk(
                FlatBlocks::<SingleChunkEntry>::from_indices(
                    self.onehot_k,
                    &self.indices,
                    num_positions_per_block,
                    ring_d,
                    num_live_blocks,
                )?,
            ))
        } else {
            Ok(OneHotBlocks::MultiChunk(
                FlatBlocks::<MultiChunkEntry>::from_indices(
                    self.onehot_k,
                    &self.indices,
                    num_positions_per_block,
                    ring_d,
                    num_live_blocks,
                )?,
            ))
        }
    }
}
