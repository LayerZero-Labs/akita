use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> LowBasisRangeCheckProver<E> {
    #[tracing::instrument(
        skip_all,
        name = "LowBasisRangeCheckProver::fuse_full_prefix_x_and_compute_round"
    )]
    pub(super) fn fuse_full_prefix_x_and_compute_round(
        &self,
        range_image: &[E],
        r: E,
    ) -> (Vec<E>, EqFactoredUniPoly<E>) {
        debug_assert!(self.next_use_prefix_x_round_after_current());
        debug_assert!(self.current_x_width() >= 2);

        let old_live_x_cols = self.live_x_cols;
        let next_live_x_cols = old_live_x_cols.div_ceil(2);
        let y_len = range_image.len() / old_live_x_cols;
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let next_current_x_half = 1usize << (self.current_x_width() - 2);
        let live_pairs = next_live_x_cols.div_ceil(2);
        let block_size = num_first.min(live_pairs);

        let polynomial_precomputation = &self.polynomial_precomputation;
        let full_num_coeffs_q = polynomial_precomputation.degree_q + 1;
        let num_coeffs_q = full_num_coeffs_q;
        let mut out = vec![E::zero(); y_len * next_live_x_cols];

        let process_row = |(y, row_out): (usize, &mut [E])| {
            debug_assert!(full_num_coeffs_q <= MAX_DIRECT_RANGE_COEFFICIENTS);
            let row = &range_image[y * old_live_x_cols..(y + 1) * old_live_x_cols];
            let j_base = y * next_current_x_half;
            let mut outer_accum = vec![E::ProductAccum::zero(); num_coeffs_q];
            let mut batch_out = [[E::zero(); MAX_DIRECT_RANGE_COEFFICIENTS]; 4];
            let mut entry_buf = [E::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];

            let mut block_start = 0usize;
            while block_start < live_pairs {
                let block_end = (block_start + block_size).min(live_pairs);
                let equality_suffix_index = (j_base + block_start) >> first_bits;
                let mut block_accumulator =
                    [E::ProductAccum::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];
                let complete_quartets = (block_end - block_start) / 4;

                for quartet in 0..complete_quartets {
                    let pair_base = block_start + quartet * 4;
                    let mut pairs = [(E::zero(), E::zero()); 4];
                    for (slot, pair_x) in (pair_base..pair_base + 4).enumerate() {
                        let left_next = 2 * pair_x;
                        let left_old = 4 * pair_x;
                        let left_range_image = fold_full_prefix_pair(row, left_old, r);
                        row_out[left_next] = left_range_image;
                        let right_range_image = if left_next + 1 < next_live_x_cols {
                            let right_range_image = fold_full_prefix_pair(row, left_old + 2, r);
                            row_out[left_next + 1] = right_range_image;
                            right_range_image
                        } else {
                            E::zero()
                        };
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
                        let pair_x = pair_base + slot;
                        let equality_prefix_index = (j_base + pair_x) & (num_first - 1);
                        accumulate_dense_entry_coeffs(
                            &mut block_accumulator[..num_coeffs_q],
                            &batch_out[slot][..full_num_coeffs_q],
                            e_first[equality_prefix_index],
                        );
                    }
                }

                for pair_x in block_start + complete_quartets * 4..block_end {
                    let left_next = 2 * pair_x;
                    let left_old = 4 * pair_x;
                    let left_range_image = fold_full_prefix_pair(row, left_old, r);
                    row_out[left_next] = left_range_image;
                    let right_range_image = if left_next + 1 < next_live_x_cols {
                        let right_range_image = fold_full_prefix_pair(row, left_old + 2, r);
                        row_out[left_next + 1] = right_range_image;
                        right_range_image
                    } else {
                        E::zero()
                    };
                    compute_entry_coefficients(
                        &mut entry_buf,
                        polynomial_precomputation,
                        left_range_image,
                        right_range_image - left_range_image,
                    );
                    let equality_prefix_index = (j_base + pair_x) & (num_first - 1);
                    accumulate_dense_entry_coeffs(
                        &mut block_accumulator[..num_coeffs_q],
                        &entry_buf[..full_num_coeffs_q],
                        e_first[equality_prefix_index],
                    );
                }

                let equality_suffix = e_second[equality_suffix_index];
                for coefficient_index in 0..num_coeffs_q {
                    let reduced = E::reduce_product_accum(block_accumulator[coefficient_index]);
                    outer_accum[coefficient_index] += equality_suffix.mul_to_product_accum(reduced);
                }
                block_start = block_end;
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
        let accumulated = cfg_chunks_mut!(out, next_live_x_cols)
            .enumerate()
            .map(process_row)
            .reduce(
                || vec![E::ProductAccum::zero(); num_coeffs_q],
                merge_accumulators,
            );

        #[cfg(not(feature = "parallel"))]
        let accumulated = cfg_chunks_mut!(out, next_live_x_cols)
            .enumerate()
            .map(process_row)
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

    #[inline]
    #[tracing::instrument(
        skip_all,
        name = "LowBasisRangeCheckProver::compute_round_compact_prefix_x"
    )]
    pub(super) fn compute_round_compact_prefix_x<V: CompactRangeImageValue>(
        &self,
        compact_range_image: &[V],
    ) -> EqFactoredUniPoly<E> {
        debug_assert!(self.rounds_completed < self.col_bits);
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
        let full_num_coeffs_q = polynomial_precomputation.degree_q + 1;
        let num_coeffs_q = full_num_coeffs_q;
        let q_coeffs = cfg_fold_reduce!(
            0..(1usize << (self.num_vars - self.col_bits)),
            || vec![E::ProductAccum::zero(); num_coeffs_q],
            |mut outer_accum, y| {
                let row_start = y * self.live_x_cols;
                let row = &compact_range_image[row_start..row_start + self.live_x_cols];
                let j_base = y * current_x_half;

                let mut blk = 0usize;
                while blk < live_pairs {
                    let blk_end = (blk + block_size).min(live_pairs);
                    let j_high = (j_base + blk) >> first_bits;
                    let mut inner_pos = [E::MulU64Accum::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];
                    let mut inner_neg = [E::MulU64Accum::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];

                    for pair_x in blk..blk_end {
                        let j_low = (j_base + pair_x) & (num_first - 1);
                        let e_in = e_first[j_low];
                        let left = 2 * pair_x;
                        let left_range_image_integer = row[left].range_image_value();
                        let right_range_image_integer = if left + 1 < self.live_x_cols {
                            row[left + 1].range_image_value()
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
                        outer_accum[k] += e_out.mul_to_product_accum(inner_reduced);
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
        .map(E::reduce_product_accum)
        .collect();

        EqFactoredUniPoly::from_q_coeffs(q_coeffs)
    }

    #[tracing::instrument(
        skip_all,
        name = "LowBasisRangeCheckProver::compute_round_full_prefix_x"
    )]
    pub(super) fn compute_round_full_prefix_x(&self, range_image: &[E]) -> EqFactoredUniPoly<E> {
        debug_assert!(self.rounds_completed < self.col_bits);
        let y_len = range_image.len() / self.live_x_cols;
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let current_x_half = 1usize << (self.current_x_width() - 1);
        let live_pairs = self.live_x_cols.div_ceil(2);
        let block_size = num_first.min(live_pairs);

        let polynomial_precomputation = &self.polynomial_precomputation;
        let full_num_coeffs_q = polynomial_precomputation.degree_q + 1;
        let num_coeffs_q = full_num_coeffs_q;
        let q_coeffs = cfg_fold_reduce!(
            0..y_len,
            || vec![E::ProductAccum::zero(); num_coeffs_q],
            |mut outer_accum, y| {
                debug_assert!(full_num_coeffs_q <= MAX_DIRECT_RANGE_COEFFICIENTS);
                let row_start = y * self.live_x_cols;
                let row = &range_image[row_start..row_start + self.live_x_cols];
                let j_base = y * current_x_half;
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
                        for (slot, pair_x) in (pair_base..pair_base + 4).enumerate() {
                            let left = 2 * pair_x;
                            let left_range_image = row[left];
                            let right_range_image = if left + 1 < self.live_x_cols {
                                row[left + 1]
                            } else {
                                E::zero()
                            };
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
                            let pair_x = pair_base + slot;
                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
                            accumulate_dense_entry_coeffs(
                                &mut inner_accum[..num_coeffs_q],
                                &batch_out[slot][..full_num_coeffs_q],
                                e_in,
                            );
                        }
                    }

                    for pair_x in blk + full_chunks * 4..blk_end {
                        let left = 2 * pair_x;
                        let left_range_image = row[left];
                        let right_range_image = if left + 1 < self.live_x_cols {
                            row[left + 1]
                        } else {
                            E::zero()
                        };
                        compute_entry_coefficients(
                            &mut entry_buf,
                            polynomial_precomputation,
                            left_range_image,
                            right_range_image - left_range_image,
                        );
                        let j_low = (j_base + pair_x) & (num_first - 1);
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
            },
            |mut ca, cb| {
                for (ai, bi) in ca.iter_mut().zip(cb.iter()) {
                    *ai += *bi;
                }
                ca
            }
        );

        let q_coeffs: Vec<E> = q_coeffs.into_iter().map(E::reduce_product_accum).collect();
        EqFactoredUniPoly::from_q_coeffs(q_coeffs)
    }

    #[tracing::instrument(
        skip_all,
        name = "LowBasisRangeCheckProver::fold_compact_range_image_prefix_x"
    )]
    pub(super) fn fold_compact_range_image_prefix_x<V: CompactRangeImageValue>(
        compact_range_image: &[V],
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
                let row = &compact_range_image[row_start..row_start + live_x_cols];
                for (pair_x, dst) in row_out.iter_mut().enumerate() {
                    let left = 2 * pair_x;
                    let right_range_image = if left + 1 < live_x_cols {
                        row[left + 1].range_image_value()
                    } else {
                        0
                    };
                    *dst = fold_lut.fold(row[left].range_image_value(), right_range_image);
                }
            });

        out
    }

    #[tracing::instrument(skip_all, name = "LowBasisRangeCheckProver::fold_range_image_prefix_x")]
    pub(super) fn fold_range_image_prefix_x(
        range_image: &[E],
        live_x_cols: usize,
        y_len: usize,
        r: E,
    ) -> Vec<E> {
        let next_live_x_cols = live_x_cols.div_ceil(2);
        let mut out = vec![E::zero(); y_len * next_live_x_cols];

        cfg_chunks_mut!(out, next_live_x_cols)
            .enumerate()
            .for_each(|(y, row_out)| {
                let row_start = y * live_x_cols;
                let row = &range_image[row_start..row_start + live_x_cols];
                for (pair_x, dst) in row_out.iter_mut().enumerate() {
                    let left = 2 * pair_x;
                    let left_range_image = row[left];
                    let right_range_image = if left + 1 < live_x_cols {
                        row[left + 1]
                    } else {
                        E::zero()
                    };
                    *dst = left_range_image + r * (right_range_image - left_range_image);
                }
            });

        out
    }
}
