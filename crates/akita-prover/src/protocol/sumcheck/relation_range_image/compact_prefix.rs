use super::*;

#[allow(clippy::too_many_arguments)]
fn materialize_compact_lane_and_compute_next<
    E: FieldCore + FromPrimitiveInt + HasUnreducedOps,
    A: ProductSumAccumulator<E>,
    const SKIP_LINEAR: bool,
>(
    lane: usize,
    lane_values: &[i8],
    lane_out: &mut [E],
    alpha_round2: &[E],
    trace_round2: &PreparedProverEvaluationTrace<E>,
    lane_weight: E,
    e_first: &[E],
    e_second: &[E],
    first_bits: usize,
    block_size: usize,
    quad_fold_lut: &[E],
    quad_index_fn: fn(&[i8], usize) -> usize,
) -> ([E; 3], [E; 3]) {
    let next_coeff_count = lane_out.len();
    let current_coefficient_half = next_coeff_count / 2;
    let equality_address_base = lane * current_coefficient_half;
    let mut virt: [A; 3] = std::array::from_fn(|_| A::zero());
    let mut alpha_rel: [A; 3] = std::array::from_fn(|_| A::zero());
    let mut blk = 0usize;

    while blk < current_coefficient_half {
        let (j_high, blk_end) = stage2_eq_block(
            equality_address_base,
            blk,
            e_first.len(),
            first_bits,
            block_size,
            current_coefficient_half,
        );
        let mut inner_virt: [A; 3] = std::array::from_fn(|_| A::zero());

        for coefficient_pair in blk..blk_end {
            let j_low = (equality_address_base + coefficient_pair) & (e_first.len() - 1);
            let e_in = e_first[j_low];
            let left = 2 * coefficient_pair;
            let base = 8 * coefficient_pair;
            let w0 = quad_fold_lut[quad_index_fn(lane_values, base)];
            let w1 = quad_fold_lut[quad_index_fn(lane_values, base + 4)];
            lane_out[left] = w0;
            lane_out[left + 1] = w1;
            let dw = w1 - w0;

            inner_virt[0].add_product(e_in, w0 * (w0 + E::one()));
            if !SKIP_LINEAR {
                inner_virt[1].add_product(e_in, dw * (w0 + w0 + E::one()));
            }
            inner_virt[2].add_product(e_in, dw * dw);

            let alpha0 = alpha_round2[left];
            let alpha_delta = alpha_round2[left + 1] - alpha0;
            accumulate_relation_products(&mut alpha_rel, w0, dw, alpha0, alpha_delta);
        }

        let e_out = e_second[j_high];
        let [inner_constant, inner_linear, inner_quadratic] = inner_virt;
        virt[0].add_product(e_out, inner_constant.finish());
        if !SKIP_LINEAR {
            virt[1].add_product(e_out, inner_linear.finish());
        }
        virt[2].add_product(e_out, inner_quadratic.finish());
        blk = blk_end;
    }

    let mut rel = alpha_rel.map(|accum| lane_weight * accum.finish());
    trace_round2.for_each_source_in_lane(lane, |factor, source_values| {
        let mut source_rel: [A; 3] = std::array::from_fn(|_| A::zero());
        for coefficient_pair in 0..current_coefficient_half {
            let left = 2 * coefficient_pair;
            let w0 = lane_out[left];
            let dw = lane_out[left + 1] - w0;
            let source0 = source_values[left];
            let source_delta = source_values[left + 1] - source0;
            accumulate_relation_products(&mut source_rel, w0, dw, source0, source_delta);
        }
        for (coefficient, source) in rel.iter_mut().zip(source_rel) {
            *coefficient += factor * source.finish();
        }
    });

    (virt.map(A::finish), rel)
}

fn add_compact_round_terms<E: FieldCore>(left: &mut ([E; 3], [E; 3]), right: ([E; 3], [E; 3])) {
    for (left_term, right_term) in left.0.iter_mut().zip(right.0) {
        *left_term += right_term;
    }
    for (left_term, right_term) in left.1.iter_mut().zip(right.1) {
        *left_term += right_term;
    }
}

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> RelationRangeImageProver<E> {
    #[inline]
    pub(super) fn direct_fold_w_quad_two_rounds(
        w00: i8,
        w10: i8,
        w01: i8,
        w11: i8,
        r0: E,
        r1: E,
    ) -> E {
        let w00 = E::from_i64(w00 as i64);
        let w10 = E::from_i64(w10 as i64);
        let w01 = E::from_i64(w01 as i64);
        let w11 = E::from_i64(w11 as i64);
        fold_two_round_quad(w00, w10, w01, w11, r0, r1)
    }

    #[inline(always)]
    pub(super) fn stage2_b4_quad_lookup_index_from_column(
        lane_values: &[i8],
        base: usize,
    ) -> usize {
        let d0 = stage2_b4_w_digit(lane_values[base]);
        let d1 = stage2_b4_w_digit(lane_values[base + 1]);
        let d2 = stage2_b4_w_digit(lane_values[base + 2]);
        let d3 = stage2_b4_w_digit(lane_values[base + 3]);
        d0 | (d1 << 2) | (d2 << 4) | (d3 << 6)
    }

    pub(super) fn build_round2_w_lookup_b4(r0: E, r1: E) -> Vec<E> {
        const W_VALUES: [i8; 4] = [-2, -1, 0, 1];
        (0..256usize)
            .map(|idx| {
                let d0 = idx & 0b11;
                let d1 = (idx >> 2) & 0b11;
                let d2 = (idx >> 4) & 0b11;
                let d3 = (idx >> 6) & 0b11;
                Self::direct_fold_w_quad_two_rounds(
                    W_VALUES[d0],
                    W_VALUES[d1],
                    W_VALUES[d2],
                    W_VALUES[d3],
                    r0,
                    r1,
                )
            })
            .collect()
    }

    #[inline(always)]
    pub(super) fn stage2_b8_quad_lookup_index_from_column(
        lane_values: &[i8],
        base: usize,
    ) -> usize {
        let d0 = stage2_b8_w_digit(lane_values[base]);
        let d1 = stage2_b8_w_digit(lane_values[base + 1]);
        let d2 = stage2_b8_w_digit(lane_values[base + 2]);
        let d3 = stage2_b8_w_digit(lane_values[base + 3]);
        d0 | (d1 << 3) | (d2 << 6) | (d3 << 9)
    }

    pub(super) fn build_round2_w_lookup_b8(r0: E, r1: E) -> Vec<E> {
        const W_VALUES: [i8; 8] = [-4, -3, -2, -1, 0, 1, 2, 3];
        (0..4096usize)
            .map(|idx| {
                let d0 = idx & 0b111;
                let d1 = (idx >> 3) & 0b111;
                let d2 = (idx >> 6) & 0b111;
                let d3 = (idx >> 9) & 0b111;
                Self::direct_fold_w_quad_two_rounds(
                    W_VALUES[d0],
                    W_VALUES[d1],
                    W_VALUES[d2],
                    W_VALUES[d3],
                    r0,
                    r1,
                )
            })
            .collect()
    }

    #[tracing::instrument(
        skip_all,
        name = "RelationRangeImageProver::materialize_two_round_compact_prefix"
    )]
    pub(super) fn materialize_two_round_compact_prefix(
        compact_witness: &[i8],
        live_lane_count: usize,
        coeff_count: usize,
        r0: E,
        r1: E,
    ) -> Vec<E> {
        debug_assert!(coeff_count.is_power_of_two());
        debug_assert!(coeff_count >= 4);
        let next_coeff_count = coeff_count >> 2;
        let mut out = vec![E::zero(); live_lane_count * next_coeff_count];
        for lane in 0..live_lane_count {
            let src_start = lane * coeff_count;
            let dst_start = lane * next_coeff_count;
            let lane_values = &compact_witness[src_start..src_start + coeff_count];
            for (quad_y, dst) in out[dst_start..dst_start + next_coeff_count]
                .iter_mut()
                .enumerate()
            {
                let base = 4 * quad_y;
                *dst = Self::direct_fold_w_quad_two_rounds(
                    lane_values[base],
                    lane_values[base + 1],
                    lane_values[base + 2],
                    lane_values[base + 3],
                    r0,
                    r1,
                );
            }
        }
        out
    }

    #[tracing::instrument(skip_all, name = "RelationRangeImageProver::fold_alpha_two_rounds")]
    pub(super) fn fold_alpha_two_rounds(common_alpha_factor: &[E], r0: E, r1: E) -> Vec<E> {
        debug_assert!(common_alpha_factor.len().is_power_of_two());
        debug_assert!(common_alpha_factor.len() >= 4);
        let next_coeff_count = common_alpha_factor.len() >> 2;
        let mut out = vec![E::zero(); next_coeff_count];
        for (quad_y, dst) in out.iter_mut().enumerate() {
            let base = 4 * quad_y;
            *dst = fold_two_round_quad(
                common_alpha_factor[base],
                common_alpha_factor[base + 1],
                common_alpha_factor[base + 2],
                common_alpha_factor[base + 3],
                r0,
                r1,
            );
        }
        out
    }

    #[tracing::instrument(
        skip_all,
        name = "RelationRangeImageProver::materialize_two_round_compact_prefix_and_compute_next_round"
    )]
    pub(super) fn materialize_two_round_compact_prefix_and_compute_next_round(
        &self,
        compact_witness: &[i8],
        alpha_round2: &[E],
        trace_round2: &PreparedProverEvaluationTrace<E>,
        r0: E,
        r1: E,
    ) -> (Vec<E>, NormRoundTerms<E>, [E; 3]) {
        match (
            E::DELAYED_PRODUCT_SUM_IS_EXACT,
            self.can_skip_norm_linear_coeff(),
        ) {
            (true, true) => self
                .materialize_compact_prefix_and_compute_next::<DelayedProductSum<E>, true>(
                    compact_witness,
                    alpha_round2,
                    trace_round2,
                    r0,
                    r1,
                ),
            (true, false) => self
                .materialize_compact_prefix_and_compute_next::<DelayedProductSum<E>, false>(
                    compact_witness,
                    alpha_round2,
                    trace_round2,
                    r0,
                    r1,
                ),
            (false, true) => self
                .materialize_compact_prefix_and_compute_next::<DirectProductSum<E>, true>(
                    compact_witness,
                    alpha_round2,
                    trace_round2,
                    r0,
                    r1,
                ),
            (false, false) => self
                .materialize_compact_prefix_and_compute_next::<DirectProductSum<E>, false>(
                    compact_witness,
                    alpha_round2,
                    trace_round2,
                    r0,
                    r1,
                ),
        }
    }

    fn materialize_compact_prefix_and_compute_next<
        A: ProductSumAccumulator<E>,
        const SKIP_LINEAR: bool,
    >(
        &self,
        compact_witness: &[i8],
        alpha_round2: &[E],
        trace_round2: &PreparedProverEvaluationTrace<E>,
        r0: E,
        r1: E,
    ) -> (Vec<E>, NormRoundTerms<E>, [E; 3]) {
        debug_assert!(self.coefficient_bits() > 2);
        let coeff_count = self.common_alpha_factor.len();
        debug_assert_eq!(compact_witness.len(), self.live_lane_count * coeff_count);
        debug_assert_eq!(alpha_round2.len(), coeff_count >> 2);

        let next_coeff_count = coeff_count >> 2;
        let current_coefficient_half = next_coeff_count >> 1;
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros() as usize;
        let block_size = num_first.min(current_coefficient_half);
        let relation_lane_weights = &self.relation_lane_weights;
        let quad_fold_lut = match self.b {
            4 => Self::build_round2_w_lookup_b4(r0, r1),
            8 => Self::build_round2_w_lookup_b8(r0, r1),
            _ => unreachable!("unsupported stage-2 two-round prefix basis"),
        };
        let quad_index_fn: fn(&[i8], usize) -> usize = match self.b {
            4 => Self::stage2_b4_quad_lookup_index_from_column,
            8 => Self::stage2_b8_quad_lookup_index_from_column,
            _ => unreachable!("unsupported stage-2 two-round prefix basis"),
        };
        let mut out = vec![E::zero(); self.live_lane_count * next_coeff_count];

        let compute_lane = |(lane, lane_out): (usize, &mut [E])| {
            let lane_start = lane * coeff_count;
            materialize_compact_lane_and_compute_next::<E, A, SKIP_LINEAR>(
                lane,
                &compact_witness[lane_start..lane_start + coeff_count],
                lane_out,
                alpha_round2,
                trace_round2,
                relation_lane_weights[lane],
                e_first,
                e_second,
                first_bits,
                block_size,
                &quad_fold_lut,
                quad_index_fn,
            )
        };

        #[cfg(feature = "parallel")]
        let totals = out
            .par_chunks_mut(next_coeff_count)
            .enumerate()
            .map(compute_lane)
            .reduce(
                || ([E::zero(); 3], [E::zero(); 3]),
                |mut left, right| {
                    add_compact_round_terms(&mut left, right);
                    left
                },
            );

        #[cfg(not(feature = "parallel"))]
        let totals = out
            .chunks_mut(next_coeff_count)
            .enumerate()
            .map(compute_lane)
            .fold(([E::zero(); 3], [E::zero(); 3]), |mut left, right| {
                add_compact_round_terms(&mut left, right);
                left
            });

        let virt_terms = if SKIP_LINEAR {
            NormRoundTerms::SkipLinear([totals.0[0], totals.0[2]])
        } else {
            NormRoundTerms::Full(totals.0)
        };
        (out, virt_terms, totals.1)
    }
}
