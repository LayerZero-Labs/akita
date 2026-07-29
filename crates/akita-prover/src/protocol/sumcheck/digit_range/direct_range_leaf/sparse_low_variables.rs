use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> LowBasisRangeCheckProver<E> {
    #[inline]
    pub(super) fn use_sparse_x_y_round(&self) -> bool {
        !self.in_x_phase() && self.live_x_cols < (1usize << self.col_bits)
    }

    #[tracing::instrument(
        skip_all,
        name = "LowBasisRangeCheckProver::compute_round_compact_sparse_x_y"
    )]
    pub(super) fn compute_round_compact_sparse_x_y<V: CompactRangeImageValue>(
        &self,
        compact_range_image: &[V],
    ) -> EqFactoredUniPoly<E> {
        debug_assert!(self.use_sparse_x_y_round());
        let y_len = compact_range_image.len() / self.live_x_cols;
        let y_pairs = y_len / 2;
        compute_range_round_polynomial_from_compact_image_pairs(
            &self.split_eq,
            &self.polynomial_precomputation,
            |j| {
                let x = j / y_pairs;
                if x >= self.live_x_cols {
                    return (0, 0);
                }
                let y_pair = j % y_pairs;
                let top = x * y_len + 2 * y_pair;
                (
                    compact_range_image[top].range_image_value(),
                    compact_range_image[top + 1].range_image_value(),
                )
            },
        )
    }

    #[tracing::instrument(
        skip_all,
        name = "LowBasisRangeCheckProver::compute_round_materialized_sparse_x_y"
    )]
    pub(super) fn compute_round_materialized_sparse_x_y(
        &self,
        range_image: &[E],
    ) -> EqFactoredUniPoly<E> {
        debug_assert!(self.use_sparse_x_y_round());
        let y_len = range_image.len() / self.live_x_cols;
        let y_pairs = y_len / 2;
        compute_range_round_polynomial_from_range_image(
            &self.split_eq,
            &self.polynomial_precomputation,
            |j| {
                let x = j / y_pairs;
                if x >= self.live_x_cols {
                    return (E::zero(), E::zero());
                }
                let y_pair = j % y_pairs;
                let top = x * y_len + 2 * y_pair;
                (range_image[top], range_image[top + 1])
            },
        )
    }

    #[tracing::instrument(
        skip_all,
        name = "LowBasisRangeCheckProver::fuse_materialized_sparse_x_y_and_compute_round"
    )]
    pub(super) fn fuse_materialized_sparse_x_y_and_compute_round(
        &self,
        range_image: &[E],
        r: E,
    ) -> (Vec<E>, EqFactoredUniPoly<E>) {
        debug_assert!(self.use_sparse_x_y_round());
        debug_assert!(self.next_use_sparse_x_y_round_after_current());
        let live_x_cols = self.live_x_cols;
        let y_len = range_image.len() / live_x_cols;
        debug_assert_eq!(y_len % 4, 0);
        let next_y_len = y_len / 2;
        let live_pairs = next_y_len / 2;
        let current_y_half = next_y_len / 2;
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let block_size = num_first.min(live_pairs);
        let polynomial_precomputation = &self.polynomial_precomputation;
        let full_num_coeffs_q = polynomial_precomputation.degree_q + 1;
        let num_coeffs_q = full_num_coeffs_q;
        let mut out = vec![E::zero(); live_x_cols * next_y_len];

        let process_column = |(x, col_out): (usize, &mut [E])| {
            debug_assert!(full_num_coeffs_q <= MAX_DIRECT_RANGE_COEFFICIENTS);
            let col = &range_image[x * y_len..(x + 1) * y_len];
            let j_base = x * current_y_half;
            let mut outer_accum = vec![E::ProductAccum::zero(); num_coeffs_q];
            let mut batch_out = [[E::zero(); MAX_DIRECT_RANGE_COEFFICIENTS]; 4];
            let mut entry_buf = [E::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];

            let mut blk = 0usize;
            while blk < live_pairs {
                let blk_end = (blk + block_size).min(live_pairs);
                let j_high = (j_base + blk) >> first_bits;
                let mut inner_accum = [E::ProductAccum::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];
                let blk_len = blk_end - blk;
                let full_chunks = blk_len / 4;

                for chunk in 0..full_chunks {
                    let pair_base = blk + chunk * 4;
                    let mut pairs = [(E::zero(), E::zero()); 4];
                    for (slot, pair_y) in (pair_base..pair_base + 4).enumerate() {
                        let top_y = 2 * pair_y;
                        let top = 4 * pair_y;
                        let left_range_image = col[top] + r * (col[top + 1] - col[top]);
                        let right_range_image = col[top + 2] + r * (col[top + 3] - col[top + 2]);
                        col_out[top_y] = left_range_image;
                        col_out[top_y + 1] = right_range_image;
                        pairs[slot] = (left_range_image, right_range_image);
                    }

                    compute_entry_coefficients_x4(
                        &mut batch_out,
                        polynomial_precomputation,
                        [pairs[0].0, pairs[1].0, pairs[2].0, pairs[3].0],
                        [
                            pairs[0].1 - pairs[0].0,
                            pairs[1].1 - pairs[1].0,
                            pairs[2].1 - pairs[2].0,
                            pairs[3].1 - pairs[3].0,
                        ],
                    );

                    for (slot, _) in pairs.iter().enumerate() {
                        let pair_y = pair_base + slot;
                        let j_low = (j_base + pair_y) & (num_first - 1);
                        let e_in = e_first[j_low];
                        accumulate_dense_entry_coeffs(
                            &mut inner_accum[..num_coeffs_q],
                            &batch_out[slot][..full_num_coeffs_q],
                            e_in,
                        );
                    }
                }

                for pair_y in blk + full_chunks * 4..blk_end {
                    let top_y = 2 * pair_y;
                    let top = 4 * pair_y;
                    let left_range_image = col[top] + r * (col[top + 1] - col[top]);
                    let right_range_image = col[top + 2] + r * (col[top + 3] - col[top + 2]);
                    col_out[top_y] = left_range_image;
                    col_out[top_y + 1] = right_range_image;
                    compute_entry_coefficients(
                        &mut entry_buf,
                        polynomial_precomputation,
                        left_range_image,
                        right_range_image - left_range_image,
                    );
                    let j_low = (j_base + pair_y) & (num_first - 1);
                    let e_in = e_first[j_low];
                    accumulate_dense_entry_coeffs(
                        &mut inner_accum[..num_coeffs_q],
                        &entry_buf[..full_num_coeffs_q],
                        e_in,
                    );
                }

                let e_out = e_second[j_high];
                for k in 0..num_coeffs_q {
                    let inner_reduced = E::reduce_product_accum(inner_accum[k]);
                    outer_accum[k] += e_out.mul_to_product_accum(inner_reduced);
                }
                blk = blk_end;
            }

            outer_accum
        };
        let merge_accumulators = |mut left: Vec<E::ProductAccum>, right: Vec<E::ProductAccum>| {
            for (left_coefficient, right_coefficient) in left.iter_mut().zip(right) {
                *left_coefficient += right_coefficient;
            }
            left
        };

        #[cfg(feature = "parallel")]
        let accumulated = cfg_chunks_mut!(out, next_y_len)
            .enumerate()
            .map(process_column)
            .reduce(
                || vec![E::ProductAccum::zero(); num_coeffs_q],
                merge_accumulators,
            );
        #[cfg(not(feature = "parallel"))]
        let accumulated = cfg_chunks_mut!(out, next_y_len)
            .enumerate()
            .map(process_column)
            .fold(
                vec![E::ProductAccum::zero(); num_coeffs_q],
                merge_accumulators,
            );
        let q_coeffs = accumulated
            .into_iter()
            .map(E::reduce_product_accum)
            .collect();

        let poly = EqFactoredUniPoly::from_q_coeffs(q_coeffs);
        (out, poly)
    }

    #[tracing::instrument(
        skip_all,
        name = "LowBasisRangeCheckProver::fold_range_image_sparse_x_y"
    )]
    pub(super) fn fold_range_image_sparse_x_y(
        range_image: &[E],
        live_x_cols: usize,
        y_len: usize,
        r: E,
    ) -> Vec<E> {
        debug_assert_eq!(y_len % 2, 0);
        let next_y_len = y_len / 2;
        let mut out = vec![E::zero(); live_x_cols * next_y_len];

        cfg_chunks_mut!(out, next_y_len)
            .enumerate()
            .for_each(|(x, col_out)| {
                let col = &range_image[x * y_len..(x + 1) * y_len];
                for (pair_y, dst) in col_out.iter_mut().enumerate() {
                    let top = 2 * pair_y;
                    let left_range_image = col[top];
                    let right_range_image = col[top + 1];
                    *dst = left_range_image + r * (right_range_image - left_range_image);
                }
            });

        out
    }
}
