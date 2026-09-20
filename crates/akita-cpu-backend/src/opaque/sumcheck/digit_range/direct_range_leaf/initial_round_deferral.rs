use super::*;

impl<E: Field + Ring + Unreduced> LowBasisRangeCheckProver<E> {
    #[inline]
    pub(super) fn direct_fold_range_image_quad_to_round2(
        range_image_00: i16,
        range_image_10: i16,
        range_image_01: i16,
        range_image_11: i16,
        r0: E,
        r1: E,
    ) -> E {
        let range_image_00 = E::from_i64(i64::from(range_image_00));
        let range_image_10 = E::from_i64(i64::from(range_image_10));
        let range_image_01 = E::from_i64(i64::from(range_image_01));
        let range_image_11 = E::from_i64(i64::from(range_image_11));
        let first_fold = range_image_00 + r0 * (range_image_10 - range_image_00);
        let second_fold = range_image_01 + r0 * (range_image_11 - range_image_01);
        first_fold + r1 * (second_fold - first_fold)
    }

    #[inline(always)]
    pub(super) fn stage1_b4_quad_lookup_index_from_row<S: CompactRangeImageSource + ?Sized>(
        row: &S,
        base: usize,
    ) -> usize {
        let d0 = stage1_b4_digit_from_compact_range_image(row.range_image_value(base));
        let d1 = stage1_b4_digit_from_compact_range_image(row.range_image_value(base + 1));
        let d2 = stage1_b4_digit_from_compact_range_image(row.range_image_value(base + 2));
        let d3 = stage1_b4_digit_from_compact_range_image(row.range_image_value(base + 3));
        d0 | (d1 << 1) | (d2 << 2) | (d3 << 3)
    }

    pub(super) fn build_round2_range_image_lookup_b4(r0: E, r1: E) -> Vec<E> {
        const RANGE_IMAGE_VALUES: [i16; 2] = [0, 2];
        (0..16usize)
            .map(|idx| {
                let d0 = idx & 0b1;
                let d1 = (idx >> 1) & 0b1;
                let d2 = (idx >> 2) & 0b1;
                let d3 = (idx >> 3) & 0b1;
                Self::direct_fold_range_image_quad_to_round2(
                    RANGE_IMAGE_VALUES[d0],
                    RANGE_IMAGE_VALUES[d1],
                    RANGE_IMAGE_VALUES[d2],
                    RANGE_IMAGE_VALUES[d3],
                    r0,
                    r1,
                )
            })
            .collect()
    }

    /// Build `Q(left + X(right-left))` for every binary range-image octet.
    ///
    /// After the first two challenges, an octet has two folded endpoints for
    /// round three. Each original range-image entry is either zero or two, so
    /// the complete challenge-dependent table has only `2^8` rows and three
    /// coefficients per row.
    fn build_binary_range_image_third_round_coefficient_table(r0: E, r1: E) -> [[E; 3]; 256] {
        let folded_quads = Self::build_round2_range_image_lookup_b4(r0, r1);
        std::array::from_fn(|octet_index| {
            let left = folded_quads[octet_index & 0x0f];
            let delta = folded_quads[octet_index >> 4] - left;
            [
                left * (left - E::from_u64(2)),
                E::from_u64(2) * delta * (left - E::one()),
                delta * delta,
            ]
        })
    }

    #[tracing::instrument(
        skip_all,
        name = "LowBasisRangeCheckProver::compute_binary_range_image_third_round_from_compact_octets"
    )]
    pub(super) fn compute_binary_range_image_third_round_from_compact_octets<
        S: CompactRangeImageSource + ?Sized,
    >(
        &self,
        compact_range_image: &S,
        r0: E,
        r1: E,
    ) -> EqFactoredUniPoly<E> {
        debug_assert!(self.defers_compact_range_image_through_third_round());
        debug_assert_eq!(self.rounds_completed, 1);
        let y_len = compact_range_image.len() / self.live_x_cols;
        let octets_per_column = y_len / 8;
        let coefficient_table =
            Self::build_binary_range_image_third_round_coefficient_table(r0, r1);
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let live_octets = self.live_x_cols * octets_per_column;

        let accumulated = cfg_fold_reduce!(
            0..e_second.len(),
            || [E::Product::zero(); 3],
            |mut outer_accum, j_high| {
                let mut inner_accum = [E::Product::zero(); 3];
                let base_j = j_high * num_first;
                let live_end = (base_j + num_first).min(live_octets);
                let source_start = (8 * base_j).min(compact_range_image.len());
                let mut quads = compact_range_image.quad_lookup_iter(source_start..8 * live_end, 4);
                for (j_low, &inner_equality_weight) in e_first.iter().enumerate() {
                    let pair_index = base_j + j_low;
                    if pair_index >= live_octets {
                        continue;
                    }
                    let left = quads.next().expect("compact octet left quad");
                    let right = quads.next().expect("compact octet right quad");
                    let table_index = left | (right << 4);
                    for (accumulator, &coefficient) in inner_accum
                        .iter_mut()
                        .zip(coefficient_table[table_index].iter())
                    {
                        *accumulator += inner_equality_weight.mul_unreduced(coefficient);
                    }
                }
                let outer_equality_weight = e_second[j_high];
                for (accumulator, inner) in outer_accum.iter_mut().zip(inner_accum) {
                    *accumulator += outer_equality_weight.mul_unreduced(E::reduce_product(inner));
                }
                outer_accum
            },
            |mut left, right| {
                for (left_coefficient, right_coefficient) in left.iter_mut().zip(right) {
                    *left_coefficient += right_coefficient;
                }
                left
            }
        );

        EqFactoredUniPoly::from_q_coeffs(accumulated.into_iter().map(E::reduce_product).collect())
    }

    /// Fold every binary range-image octet through all three initial challenges.
    fn build_binary_range_image_octet_fold_table(r0: E, r1: E, r2: E) -> [E; 256] {
        let folded_quads = Self::build_round2_range_image_lookup_b4(r0, r1);
        std::array::from_fn(|octet_index| {
            let left = folded_quads[octet_index & 0x0f];
            let right = folded_quads[octet_index >> 4];
            left + r2 * (right - left)
        })
    }

    #[tracing::instrument(
        skip_all,
        name = "LowBasisRangeCheckProver::materialize_binary_range_image_after_third_round"
    )]
    pub(super) fn materialize_binary_range_image_after_third_round<
        S: CompactRangeImageSource + ?Sized,
    >(
        compact_range_image: &S,
        live_x_cols: usize,
        y_len: usize,
        r0: E,
        r1: E,
        r2: E,
    ) -> Vec<E> {
        debug_assert_eq!(y_len % 8, 0);
        let next_y_len = y_len / 8;
        let fold_table = Self::build_binary_range_image_octet_fold_table(r0, r1, r2);
        let mut output = vec![E::zero(); live_x_cols * next_y_len];
        cfg_chunks_mut!(output, next_y_len)
            .enumerate()
            .for_each(|(x, column_output)| {
                let column_start = x * y_len;
                let mut quads =
                    compact_range_image.quad_lookup_iter(column_start..column_start + y_len, 4);
                for value in column_output {
                    let left = quads.next().expect("compact octet left quad");
                    let right = quads.next().expect("compact octet right quad");
                    let table_index = left | (right << 4);
                    *value = fold_table[table_index];
                }
            });
        output
    }

    #[inline(always)]
    pub(super) fn stage1_b8_quad_lookup_index_from_row<S: CompactRangeImageSource + ?Sized>(
        row: &S,
        base: usize,
    ) -> usize {
        let d0 = stage1_b8_digit_from_compact_range_image(row.range_image_value(base));
        let d1 = stage1_b8_digit_from_compact_range_image(row.range_image_value(base + 1));
        let d2 = stage1_b8_digit_from_compact_range_image(row.range_image_value(base + 2));
        let d3 = stage1_b8_digit_from_compact_range_image(row.range_image_value(base + 3));
        d0 | (d1 << 2) | (d2 << 4) | (d3 << 6)
    }

    pub(super) fn build_round2_range_image_lookup_b8(r0: E, r1: E) -> Vec<E> {
        const RANGE_IMAGE_VALUES: [i16; 4] = [0, 2, 6, 12];
        (0..256usize)
            .map(|idx| {
                let d0 = idx & 0b11;
                let d1 = (idx >> 2) & 0b11;
                let d2 = (idx >> 4) & 0b11;
                let d3 = (idx >> 6) & 0b11;
                Self::direct_fold_range_image_quad_to_round2(
                    RANGE_IMAGE_VALUES[d0],
                    RANGE_IMAGE_VALUES[d1],
                    RANGE_IMAGE_VALUES[d2],
                    RANGE_IMAGE_VALUES[d3],
                    r0,
                    r1,
                )
            })
            .collect()
    }

    /// Cache `[Q(a), Q'(a), Q''(a)/2, Q'''(a)/6]` for every folded quad.
    ///
    /// Then `Q(a + dX)` needs only the powers of `d` and three coefficient
    /// multiplications per octet; the leading coefficient of `Q` is one.
    fn build_quartic_taylor_coefficient_table(folded_quads: &[E]) -> [[E; 4]; 256] {
        std::array::from_fn(|index| {
            let left = folded_quads[index];
            let twice_left = left + left;
            let four_times_left = twice_left + twice_left;
            let eight_times_left = four_times_left + four_times_left;
            let sixteen_times_left = eight_times_left + eight_times_left;
            let thirty_two_times_left = sixteen_times_left + sixteen_times_left;
            let sixty_four_times_left = thirty_two_times_left + thirty_two_times_left;
            let left_squared = left * left;
            let twice_left_squared = left_squared + left_squared;
            let four_times_left_squared = twice_left_squared + twice_left_squared;
            let six_times_left_squared = four_times_left_squared + twice_left_squared;
            let first_quadratic = left_squared - twice_left;
            let second_quadratic =
                left_squared - (sixteen_times_left + twice_left) + E::from_u64(72);
            let value = first_quadratic * second_quadratic;
            let first_derivative = first_quadratic * (twice_left - E::from_u64(18))
                + second_quadratic * (twice_left - E::from_u64(2));
            let second_derivative_over_two = six_times_left_squared
                - (sixty_four_times_left - four_times_left)
                + E::from_u64(108);
            let third_derivative_over_six = four_times_left - E::from_u64(20);
            [
                value,
                first_derivative,
                second_derivative_over_two,
                third_derivative_over_six,
            ]
        })
    }

    #[inline]
    fn quartic_affine_coefficients_from_octet_class(
        coefficients: &mut [E; MAX_DIRECT_RANGE_COEFFICIENTS],
        octet_class: usize,
        folded_quads: &[E],
        taylor_coefficients: &[[E; 4]; 256],
    ) {
        let left_index = octet_class >> 8;
        let right_index = octet_class & 0xff;
        let range_image_delta = folded_quads[right_index] - folded_quads[left_index];
        let delta_squared = range_image_delta * range_image_delta;
        let delta_cubed = delta_squared * range_image_delta;
        let taylor_row = taylor_coefficients[left_index];
        coefficients[0] = taylor_row[0];
        coefficients[1] = taylor_row[1] * range_image_delta;
        coefficients[2] = taylor_row[2] * delta_squared;
        coefficients[3] = taylor_row[3] * delta_cubed;
        coefficients[4] = delta_squared * delta_squared;
    }

    #[tracing::instrument(
        skip_all,
        name = "LowBasisRangeCheckProver::compute_quartic_range_image_third_round_from_compact_octets"
    )]
    pub(super) fn compute_quartic_range_image_third_round_from_compact_octets<
        S: CompactRangeImageSource + ?Sized,
    >(
        &self,
        compact_range_image: &S,
        r0: E,
        r1: E,
    ) -> EqFactoredUniPoly<E> {
        debug_assert_eq!(self.basis, 8);
        debug_assert_eq!(self.rounds_completed, 1);
        let y_len = compact_range_image.len() / self.live_x_cols;
        let octets_per_column = y_len / 8;
        let folded_quads = Self::build_round2_range_image_lookup_b8(r0, r1);
        let taylor_coefficients = Self::build_quartic_taylor_coefficient_table(&folded_quads);
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let live_octets = self.live_x_cols * octets_per_column;
        let accumulated = cfg_fold_reduce!(
            0..e_second.len(),
            || [E::Product::zero(); 5],
            |mut outer_accum, j_high| {
                let mut inner_accum = [E::Product::zero(); 5];
                let mut coefficients = [E::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];
                let base_j = j_high * num_first;
                let live_end = (base_j + num_first).min(live_octets);
                let source_start = (8 * base_j).min(compact_range_image.len());
                let mut quads = compact_range_image.quad_lookup_iter(source_start..8 * live_end, 8);
                for (j_low, &inner_equality_weight) in e_first.iter().enumerate() {
                    let octet_index = base_j + j_low;
                    if octet_index >= live_octets {
                        continue;
                    }
                    let left = quads.next().expect("compact octet left quad");
                    let right = quads.next().expect("compact octet right quad");
                    let class = (left << 8) | right;
                    Self::quartic_affine_coefficients_from_octet_class(
                        &mut coefficients,
                        class,
                        &folded_quads,
                        &taylor_coefficients,
                    );
                    accumulate_dense_entry_coeffs(
                        &mut inner_accum,
                        &coefficients,
                        inner_equality_weight,
                    );
                }
                let outer_equality_weight = e_second[j_high];
                for (accumulator, inner) in outer_accum.iter_mut().zip(inner_accum) {
                    *accumulator += outer_equality_weight.mul_unreduced(E::reduce_product(inner));
                }
                outer_accum
            },
            |mut left, right| {
                for (left_coefficient, right_coefficient) in left.iter_mut().zip(right) {
                    *left_coefficient += right_coefficient;
                }
                left
            }
        );

        EqFactoredUniPoly::from_q_coeffs(accumulated.into_iter().map(E::reduce_product).collect())
    }

    #[tracing::instrument(
        skip_all,
        name = "LowBasisRangeCheckProver::materialize_quartic_range_image_after_third_round"
    )]
    pub(super) fn materialize_quartic_range_image_after_third_round<
        S: CompactRangeImageSource + ?Sized,
    >(
        compact_range_image: &S,
        live_x_cols: usize,
        y_len: usize,
        r0: E,
        r1: E,
        r2: E,
    ) -> Vec<E> {
        debug_assert_eq!(y_len % 8, 0);
        let next_y_len = y_len / 8;
        let folded_quads = Self::build_round2_range_image_lookup_b8(r0, r1);
        let mut output = vec![E::zero(); live_x_cols * next_y_len];
        cfg_chunks_mut!(output, next_y_len)
            .enumerate()
            .for_each(|(x, column_output)| {
                let column_start = x * y_len;
                let mut quads =
                    compact_range_image.quad_lookup_iter(column_start..column_start + y_len, 8);
                for value in column_output {
                    let left = folded_quads[quads.next().expect("compact octet left quad")];
                    let right = folded_quads[quads.next().expect("compact octet right quad")];
                    *value = left + r2 * (right - left);
                }
            });
        output
    }

    #[tracing::instrument(
        skip_all,
        name = "LowBasisRangeCheckProver::fold_compact_range_image_to_round2"
    )]
    pub(super) fn fold_compact_range_image_to_round2<S: CompactRangeImageSource + ?Sized>(
        compact_range_image: &S,
        live_x_cols: usize,
        y_len: usize,
        r0: E,
        r1: E,
    ) -> Vec<E> {
        debug_assert_eq!(y_len % 4, 0);
        let next_y_len = y_len / 4;
        let mut out = vec![E::zero(); live_x_cols * next_y_len];
        for (x, col_out) in out.chunks_mut(next_y_len).enumerate() {
            let col_start = x * y_len;
            for (quad_y, dst) in col_out.iter_mut().enumerate() {
                let base = 4 * quad_y;
                *dst = Self::direct_fold_range_image_quad_to_round2(
                    compact_range_image.range_image_value(col_start + base),
                    compact_range_image.range_image_value(col_start + base + 1),
                    compact_range_image.range_image_value(col_start + base + 2),
                    compact_range_image.range_image_value(col_start + base + 3),
                    r0,
                    r1,
                );
            }
        }
        out
    }

    #[tracing::instrument(
        skip_all,
        name = "LowBasisRangeCheckProver::fuse_compact_to_round2_and_compute_round"
    )]
    pub(super) fn fuse_compact_to_round2_and_compute_round<S: CompactRangeImageSource + ?Sized>(
        &self,
        compact_range_image: &S,
        r0: E,
        r1: E,
    ) -> (Vec<E>, EqFactoredUniPoly<E>) {
        debug_assert!(self.ring_bits() > 2);
        let live_x_cols = self.live_x_cols;
        let y_len = compact_range_image.len() / live_x_cols;
        debug_assert_eq!(y_len % 4, 0);
        let next_y_len = y_len / 4;
        let live_pairs = next_y_len / 2;
        let current_y_half = next_y_len / 2;
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let block_size = num_first.min(live_pairs);
        let quad_fold_lut = match self.basis {
            4 => Self::build_round2_range_image_lookup_b4(r0, r1),
            _ => Self::build_round2_range_image_lookup_b8(r0, r1),
        };

        let polynomial_precomputation = &self.polynomial_precomputation;
        let full_num_coeffs_q = polynomial_precomputation.degree_q + 1;
        let num_coeffs_q = full_num_coeffs_q;
        let mut out = vec![E::zero(); live_x_cols * next_y_len];

        let process_column = |(x, col_out): (usize, &mut [E])| {
            let col_start = x * y_len;
            let j_base = x * current_y_half;
            let mut outer_accum = vec![E::Product::zero(); num_coeffs_q];
            let mut entry_buf = [E::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];

            let mut block_start = 0usize;
            while block_start < live_pairs {
                let block_end = (block_start + block_size).min(live_pairs);
                let outer_equality_index = (j_base + block_start) >> first_bits;
                let mut inner_accum = [E::Product::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];

                for pair_y in block_start..block_end {
                    let inner_equality_index = (j_base + pair_y) & (num_first - 1);
                    let inner_equality_weight = e_first[inner_equality_index];
                    let output_offset = 2 * pair_y;
                    let input_offset = 8 * pair_y;
                    let left_range_image = quad_fold_lut[match self.basis {
                        4 => Self::stage1_b4_quad_lookup_index_from_row(
                            compact_range_image,
                            col_start + input_offset,
                        ),
                        _ => Self::stage1_b8_quad_lookup_index_from_row(
                            compact_range_image,
                            col_start + input_offset,
                        ),
                    }];
                    let right_range_image = quad_fold_lut[match self.basis {
                        4 => Self::stage1_b4_quad_lookup_index_from_row(
                            compact_range_image,
                            col_start + input_offset + 4,
                        ),
                        _ => Self::stage1_b8_quad_lookup_index_from_row(
                            compact_range_image,
                            col_start + input_offset + 4,
                        ),
                    }];
                    col_out[output_offset] = left_range_image;
                    col_out[output_offset + 1] = right_range_image;
                    compute_entry_coefficients(
                        &mut entry_buf,
                        polynomial_precomputation,
                        left_range_image,
                        right_range_image - left_range_image,
                    );
                    accumulate_dense_entry_coeffs(
                        &mut inner_accum[..num_coeffs_q],
                        &entry_buf[..full_num_coeffs_q],
                        inner_equality_weight,
                    );
                }

                let outer_equality_weight = e_second[outer_equality_index];
                for coefficient_index in 0..num_coeffs_q {
                    let inner_reduced = E::reduce_product(inner_accum[coefficient_index]);
                    outer_accum[coefficient_index] +=
                        outer_equality_weight.mul_unreduced(inner_reduced);
                }
                block_start = block_end;
            }
            outer_accum
        };
        let merge_accumulators = |mut left: Vec<E::Product>, right: Vec<E::Product>| {
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
                || vec![E::Product::zero(); num_coeffs_q],
                merge_accumulators,
            );
        #[cfg(not(feature = "parallel"))]
        let accumulated = cfg_chunks_mut!(out, next_y_len)
            .enumerate()
            .map(process_column)
            .fold(vec![E::Product::zero(); num_coeffs_q], merge_accumulators);
        let q_coeffs = accumulated.into_iter().map(E::reduce_product).collect();

        let poly = EqFactoredUniPoly::from_q_coeffs(q_coeffs);
        (out, poly)
    }
}
