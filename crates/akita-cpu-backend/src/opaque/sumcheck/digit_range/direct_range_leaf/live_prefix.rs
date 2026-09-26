use super::*;
use crate::opaque::sumcheck::{single_row_tile_pairs, sum_partials};
use jolt_poly::OmittedConstantPoly;

/// Accumulate the weighted sums of pairs `tile` of a flat live-prefix table.
///
/// `entry(pair, weight, sums)` adds `weight` times the sums of one pair. Blocks
/// follow the split-eq inner table, so each block pays one outer-weight
/// multiplication per sum.
#[inline]
fn accumulate_live_prefix_tile<E: Field + Unreduced>(
    num_coeffs_q: usize,
    e_first: &[E],
    e_second: &[E],
    block_size: usize,
    tile: Range<usize>,
    mut entry: impl FnMut(usize, E, &mut TaylorSums<E>),
) -> TaylorSums<E> {
    debug_assert!(num_coeffs_q <= MAX_DIRECT_RANGE_COEFFICIENTS);
    let num_first = e_first.len();
    let first_bits = num_first.trailing_zeros();
    let mut tile_accumulator = [E::Product::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];
    let mut block_start = tile.start;
    while block_start < tile.end {
        let block_end = (block_start + block_size).min(tile.end);
        let mut block_accumulator = [E::Product::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];
        for pair_index in block_start..block_end {
            entry(
                pair_index,
                e_first[pair_index & (num_first - 1)],
                &mut block_accumulator,
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
    /// Accumulate the current round's weighted pair sums over a flat
    /// live-prefix table with `live_pairs` live pairs.
    ///
    /// Sparse ring rounds and live-prefix column rounds share this layout: the live
    /// entries are a prefix of the flat table in binding order, and every later
    /// entry is zero, which the range polynomial maps to zero. `entries_for_tile`
    /// receives a tile's pair range and returns the accumulator of its weighted
    /// per-pair sums.
    #[tracing::instrument(skip_all, name = "LowBasisRangeCheckProver::compute_round_live_prefix")]
    pub(super) fn compute_round_live_prefix<W: FnMut(usize, E, &mut TaylorSums<E>)>(
        &self,
        live_pairs: usize,
        entries_for_tile: impl Fn(Range<usize>) -> W + Sync,
    ) -> TaylorSums<E> {
        debug_assert!(self.rounds_completed < self.num_vars);
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let block_size = e_first.len().min(live_pairs);
        let tile_pairs = single_row_tile_pairs(block_size);
        let num_coeffs_q = self.polynomial_precomputation.num_coefficients();
        let tile_accumulators: Vec<_> = cfg_into_iter!(0..live_pairs.div_ceil(tile_pairs))
            .map(|tile| {
                let tile_start = tile * tile_pairs;
                let tile_end = (tile_start + tile_pairs).min(live_pairs);
                accumulate_live_prefix_tile(
                    num_coeffs_q,
                    e_first,
                    e_second,
                    block_size,
                    tile_start..tile_end,
                    entries_for_tile(tile_start..tile_end),
                )
            })
            .collect();
        sum_partials(E::Product::zero(), tile_accumulators)
    }

    /// Fold the current flat live-prefix table into a table of `next_live`
    /// entries and compute the next round from it in the same pass.
    ///
    /// `folds_for_tile` receives a tile's range of folded entries and returns
    /// the function computing each of them.
    #[tracing::instrument(
        skip_all,
        name = "LowBasisRangeCheckProver::fuse_live_prefix_and_compute_round"
    )]
    pub(super) fn fuse_live_prefix_and_compute_round<F: FnMut(usize) -> E>(
        &self,
        next_live: usize,
        folds_for_tile: impl Fn(Range<usize>) -> F + Sync,
    ) -> (Vec<E>, OmittedConstantPoly<E>) {
        debug_assert!(self.rounds_completed + 1 < self.num_vars);
        let live_pairs = next_live.div_ceil(2);
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let block_size = e_first.len().min(live_pairs);
        let tile_pairs = single_row_tile_pairs(block_size);
        let precomputation = &self.polynomial_precomputation;
        let num_coeffs_q = precomputation.num_coefficients();
        let mut out = vec![E::zero(); next_live];
        let tile_accumulators: Vec<_> = cfg_chunks_mut!(out, 2 * tile_pairs)
            .enumerate()
            .map(|(tile, tile_out)| {
                let tile_start = tile * tile_pairs;
                let tile_end = tile_start + tile_out.len().div_ceil(2);
                let entry_start = 2 * tile_start;
                let mut fold = folds_for_tile(entry_start..entry_start + tile_out.len());
                accumulate_live_prefix_tile(
                    num_coeffs_q,
                    e_first,
                    e_second,
                    block_size,
                    tile_start..tile_end,
                    |pair, weight, sums| {
                        let local_left = 2 * (pair - tile_start);
                        let left = fold(entry_start + local_left);
                        tile_out[local_left] = left;
                        let right = if local_left + 1 < tile_out.len() {
                            let right = fold(entry_start + local_left + 1);
                            tile_out[local_left + 1] = right;
                            right
                        } else {
                            E::zero()
                        };
                        accumulate_entry_terms(sums, precomputation, left, right - left, weight);
                    },
                )
            })
            .collect();
        (
            out,
            precomputation.round_poly_from_sums(
                &sum_partials(E::Product::zero(), tile_accumulators),
                LinearSum::Taylor,
            ),
        )
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
