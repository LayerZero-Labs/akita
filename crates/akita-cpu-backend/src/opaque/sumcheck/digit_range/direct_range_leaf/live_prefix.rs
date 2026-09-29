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
        let num_coeffs_q = self.range_poly.num_coefficients();
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
        let precomputation = &self.range_poly;
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
                        precomputation.accumulate_entry_terms(sums, left, right - left, weight);
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

    /// Fold a flat live-prefix table at `r`, zero-padding an odd tail.
    #[tracing::instrument(skip_all, name = "LowBasisRangeCheckProver::fold_live_prefix")]
    pub(super) fn fold_live_prefix(range_image: &[E], r: E) -> Vec<E> {
        cfg_into_iter!(0..range_image.len().div_ceil(2))
            .map(|pair| fold_prefix_pair_with_zero_padding(range_image, 2 * pair, r))
            .collect()
    }
}
