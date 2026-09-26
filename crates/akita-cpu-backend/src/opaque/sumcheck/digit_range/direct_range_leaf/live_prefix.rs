use super::*;
use crate::opaque::sumcheck::single_row_tile_pairs;
use jolt_poly::OmittedConstantPoly;

/// Accumulate the round coefficients of pairs `tile` of a flat live-prefix table.
///
/// `pair` yields the left and right entries of one pair. Blocks follow the split-eq
/// inner table, so each block pays one outer-weight multiplication per coefficient.
#[inline]
fn accumulate_live_prefix_tile<E: Field + Ring + Unreduced>(
    polynomial_precomputation: &RangePolynomialPrecomputation,
    e_first: &[E],
    e_second: &[E],
    block_size: usize,
    tile: Range<usize>,
    mut pair: impl FnMut(usize) -> (E, E),
) -> [E::Product; MAX_DIRECT_RANGE_COEFFICIENTS] {
    let num_coeffs_q = polynomial_precomputation.num_coefficients();
    debug_assert!(num_coeffs_q <= MAX_DIRECT_RANGE_COEFFICIENTS);
    let num_first = e_first.len();
    let first_bits = num_first.trailing_zeros();
    let mut tile_accumulator = [E::Product::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];
    let mut batch_out = [[E::zero(); MAX_DIRECT_RANGE_COEFFICIENTS]; 4];
    let mut entry_buf = [E::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];
    let mut block_start = tile.start;
    while block_start < tile.end {
        let block_end = (block_start + block_size).min(tile.end);
        let mut block_accumulator = [E::Product::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];
        let complete_quartets = (block_end - block_start) / 4;

        for quartet in 0..complete_quartets {
            let pair_base = block_start + quartet * 4;
            let pairs: [(E, E); 4] = std::array::from_fn(|slot| pair(pair_base + slot));
            compute_entry_coefficients_x4(
                &mut batch_out,
                polynomial_precomputation,
                pairs.map(|(left, _)| left),
                pairs.map(|(left, right)| right - left),
            );
            for (slot, entry) in batch_out.iter().enumerate() {
                accumulate_dense_entry_coeffs(
                    &mut block_accumulator[..num_coeffs_q],
                    &entry[..num_coeffs_q],
                    e_first[(pair_base + slot) & (num_first - 1)],
                );
            }
        }

        for pair_index in block_start + complete_quartets * 4..block_end {
            let (left, right) = pair(pair_index);
            compute_entry_coefficients(
                &mut entry_buf,
                polynomial_precomputation,
                left,
                right - left,
            );
            accumulate_dense_entry_coeffs(
                &mut block_accumulator[..num_coeffs_q],
                &entry_buf[..num_coeffs_q],
                e_first[pair_index & (num_first - 1)],
            );
        }

        let equality_suffix = e_second[block_start >> first_bits];
        for (total, block) in tile_accumulator
            .iter_mut()
            .zip(block_accumulator)
            .take(num_coeffs_q)
        {
            *total += equality_suffix.mul_unreduced(E::reduce_product(block));
        }
        block_start = block_end;
    }
    tile_accumulator
}

impl<E: Field + Ring + Unreduced> LowBasisRangeCheckProver<E> {
    fn live_prefix_round_poly(
        &self,
        tile_accumulators: Vec<[E::Product; MAX_DIRECT_RANGE_COEFFICIENTS]>,
    ) -> OmittedConstantPoly<E> {
        let mut accumulated = [E::Product::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];
        for tile_accumulator in tile_accumulators {
            for (total, term) in accumulated.iter_mut().zip(tile_accumulator) {
                *total += term;
            }
        }
        let num_coeffs_q = self.polynomial_precomputation.num_coefficients();
        OmittedConstantPoly::new(
            accumulated[..num_coeffs_q]
                .iter()
                .copied()
                .map(E::reduce_product)
                .collect(),
        )
    }

    /// Compute the current round over a flat live-prefix table.
    ///
    /// Sparse ring rounds and live-prefix column rounds share this layout: the live
    /// entries are a prefix of the flat table in binding order, and every later
    /// entry is zero, which the range polynomial maps to zero.
    #[tracing::instrument(skip_all, name = "LowBasisRangeCheckProver::compute_round_live_prefix")]
    pub(super) fn compute_round_live_prefix(&self, range_image: &[E]) -> OmittedConstantPoly<E> {
        debug_assert!(self.rounds_completed < self.num_vars);
        let live_pairs = range_image.len().div_ceil(2);
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let block_size = e_first.len().min(live_pairs);
        let tile_pairs = single_row_tile_pairs(block_size);
        let tile_accumulators = cfg_into_iter!(0..live_pairs.div_ceil(tile_pairs))
            .map(|tile| {
                let tile_start = tile * tile_pairs;
                let tile_end = (tile_start + tile_pairs).min(live_pairs);
                accumulate_live_prefix_tile(
                    &self.polynomial_precomputation,
                    e_first,
                    e_second,
                    block_size,
                    tile_start..tile_end,
                    |pair| {
                        let left = 2 * pair;
                        (
                            range_image[left],
                            range_image.get(left + 1).copied().unwrap_or_else(E::zero),
                        )
                    },
                )
            })
            .collect();
        self.live_prefix_round_poly(tile_accumulators)
    }

    /// Fold a flat live-prefix table at `r` and compute the next round from the
    /// folded table in the same pass.
    #[tracing::instrument(
        skip_all,
        name = "LowBasisRangeCheckProver::fuse_live_prefix_and_compute_round"
    )]
    pub(super) fn fuse_live_prefix_and_compute_round(
        &self,
        range_image: &[E],
        r: E,
    ) -> (Vec<E>, OmittedConstantPoly<E>) {
        debug_assert!(self.next_round_uses_live_prefix());
        let next_live = range_image.len().div_ceil(2);
        let live_pairs = next_live.div_ceil(2);
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let block_size = e_first.len().min(live_pairs);
        let tile_pairs = single_row_tile_pairs(block_size);
        let mut out = vec![E::zero(); next_live];
        let tile_accumulators = cfg_chunks_mut!(out, 2 * tile_pairs)
            .enumerate()
            .map(|(tile, tile_out)| {
                let tile_start = tile * tile_pairs;
                let tile_end = tile_start + tile_out.len().div_ceil(2);
                accumulate_live_prefix_tile(
                    &self.polynomial_precomputation,
                    e_first,
                    e_second,
                    block_size,
                    tile_start..tile_end,
                    |pair| {
                        let local_left = 2 * (pair - tile_start);
                        let left = fold_prefix_pair_with_zero_padding(range_image, 4 * pair, r);
                        tile_out[local_left] = left;
                        let right = if local_left + 1 < tile_out.len() {
                            let right =
                                fold_prefix_pair_with_zero_padding(range_image, 4 * pair + 2, r);
                            tile_out[local_left + 1] = right;
                            right
                        } else {
                            E::zero()
                        };
                        (left, right)
                    },
                )
            })
            .collect();
        (out, self.live_prefix_round_poly(tile_accumulators))
    }

    #[inline]
    #[tracing::instrument(
        skip_all,
        name = "LowBasisRangeCheckProver::compute_round_compact_prefix_x"
    )]
    pub(super) fn compute_round_compact_prefix_x<S: CompactRangeImageSource + ?Sized>(
        &self,
        compact_range_image: &S,
    ) -> OmittedConstantPoly<E> {
        debug_assert!(self.rounds_completed < self.num_vars);
        debug_assert_eq!(
            compact_range_image.len(),
            self.live_x_cols * (1usize << (self.num_vars - self.col_bits))
        );

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let current_x_half = 1usize << (self.current_x_width() - 1);
        let live_pairs = self.live_x_cols.div_ceil(2);
        let block_size = num_first.min(live_pairs);

        let polynomial_precomputation = &self.polynomial_precomputation;
        let full_num_coeffs_q = polynomial_precomputation.num_coefficients();
        let num_coeffs_q = full_num_coeffs_q;
        let q_coeffs = cfg_fold_reduce!(
            0..(1usize << (self.num_vars - self.col_bits)),
            || vec![E::Product::zero(); num_coeffs_q],
            |mut outer_accum, y| {
                let row_start = y * self.live_x_cols;
                let j_base = y * current_x_half;

                let mut blk = 0usize;
                while blk < live_pairs {
                    let blk_end = (blk + block_size).min(live_pairs);
                    let j_high = (j_base + blk) >> first_bits;
                    let mut inner_pos = [E::SmallProduct::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];
                    let mut inner_neg = [E::SmallProduct::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];

                    for pair_x in blk..blk_end {
                        let j_low = (j_base + pair_x) & (num_first - 1);
                        let e_in = e_first[j_low];
                        let left = 2 * pair_x;
                        let left_range_image_integer =
                            compact_range_image.range_image_value(row_start + left);
                        let right_range_image_integer = if left + 1 < self.live_x_cols {
                            compact_range_image.range_image_value(row_start + left + 1)
                        } else {
                            0
                        };
                        let coeffs = polynomial_precomputation.compact_coeffs_lut(
                            left_range_image_integer,
                            right_range_image_integer,
                        );
                        accumulate_compact_coeffs(
                            &mut inner_pos[..num_coeffs_q],
                            &mut inner_neg[..num_coeffs_q],
                            e_in,
                            coeffs,
                        );
                    }

                    let e_out = e_second[j_high];
                    for k in 0..num_coeffs_q {
                        let inner_reduced = reduce_small_coeff_accum(inner_pos[k], inner_neg[k]);
                        outer_accum[k] += e_out.mul_unreduced(inner_reduced);
                    }
                    blk = blk_end;
                }
                outer_accum
            },
            |mut a, b_vec| {
                for (ai, bi) in a.iter_mut().zip(b_vec.iter()) {
                    *ai += *bi;
                }
                a
            }
        )
        .into_iter()
        .map(E::reduce_product)
        .collect();

        OmittedConstantPoly::new(q_coeffs)
    }

    #[tracing::instrument(
        skip_all,
        name = "LowBasisRangeCheckProver::fold_compact_range_image_prefix_x"
    )]
    pub(super) fn fold_compact_range_image_prefix_x<S: CompactRangeImageSource + ?Sized>(
        compact_range_image: &S,
        live_x_cols: usize,
        y_len: usize,
        fold_lut: &CompactPairFoldLut<E>,
    ) -> Vec<E> {
        let next_live_x_cols = live_x_cols.div_ceil(2);
        let mut out = vec![E::zero(); y_len * next_live_x_cols];

        cfg_chunks_mut!(out, next_live_x_cols)
            .enumerate()
            .for_each(|(y, row_out)| {
                let row_start = y * live_x_cols;
                for (pair_x, dst) in row_out.iter_mut().enumerate() {
                    let left = 2 * pair_x;
                    let right_range_image = if left + 1 < live_x_cols {
                        compact_range_image.range_image_value(row_start + left + 1)
                    } else {
                        0
                    };
                    *dst = fold_lut.fold(
                        compact_range_image.range_image_value(row_start + left),
                        right_range_image,
                    );
                }
            });

        out
    }

    /// Fold a flat live-prefix table at `r`, zero-padding an odd tail.
    #[tracing::instrument(skip_all, name = "LowBasisRangeCheckProver::fold_live_prefix")]
    pub(super) fn fold_live_prefix(range_image: &[E], r: E) -> Vec<E> {
        cfg_into_iter!(0..range_image.len().div_ceil(2))
            .map(|pair| fold_prefix_pair_with_zero_padding(range_image, 2 * pair, r))
            .collect()
    }
}
