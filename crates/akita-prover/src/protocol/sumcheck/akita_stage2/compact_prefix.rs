use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage2Prover<E> {
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
        let x0 = w00 + r0 * (w10 - w00);
        let x1 = w01 + r0 * (w11 - w01);
        x0 + r1 * (x1 - x0)
    }

    #[inline]
    pub(super) fn direct_fold_e_quad_two_rounds(e00: E, e10: E, e01: E, e11: E, r0: E, r1: E) -> E {
        let x0 = e00 + r0 * (e10 - e00);
        let x1 = e01 + r0 * (e11 - e01);
        x0 + r1 * (x1 - x0)
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
        name = "AkitaStage2Prover::materialize_two_round_compact_prefix"
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

    #[tracing::instrument(skip_all, name = "AkitaStage2Prover::fold_alpha_two_rounds")]
    pub(super) fn fold_alpha_two_rounds(common_alpha_factor: &[E], r0: E, r1: E) -> Vec<E> {
        debug_assert!(common_alpha_factor.len().is_power_of_two());
        debug_assert!(common_alpha_factor.len() >= 4);
        let next_coeff_count = common_alpha_factor.len() >> 2;
        let mut out = vec![E::zero(); next_coeff_count];
        for (quad_y, dst) in out.iter_mut().enumerate() {
            let base = 4 * quad_y;
            *dst = Self::direct_fold_e_quad_two_rounds(
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

    #[inline]
    fn add_trace_pair_to_relation_factor(
        evaluation_trace: &PreparedProverEvaluationTrace<E>,
        lane: usize,
        left: usize,
        coeff_count: usize,
        p0: &mut E,
        p1: &mut E,
    ) {
        *p0 += evaluation_trace.get(lane, left, coeff_count);
        *p1 += evaluation_trace.get(lane, left + 1, coeff_count);
    }

    #[tracing::instrument(
        skip_all,
        name = "AkitaStage2Prover::materialize_two_round_compact_prefix_and_compute_next_round"
    )]
    pub(super) fn materialize_two_round_compact_prefix_and_compute_next_round(
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

        if self.can_skip_norm_linear_coeff() {
            #[cfg(feature = "parallel")]
            let (virt_coeffs, rel_coeffs) = out
                .par_chunks_mut(next_coeff_count)
                .enumerate()
                .map(|(lane, lane_out)| {
                    let lane_start = lane * coeff_count;
                    let lane_values = &compact_witness[lane_start..lane_start + coeff_count];
                    let lane_weight = relation_lane_weights[lane];
                    let equality_address_base = lane * current_coefficient_half;
                    let mut virt = [E::zero(); 2];
                    let mut rel = [E::zero(); 3];
                    let mut blk = 0usize;

                    while blk < current_coefficient_half {
                        let (j_high, blk_end) = stage2_eq_block(
                            equality_address_base,
                            blk,
                            num_first,
                            first_bits,
                            block_size,
                            current_coefficient_half,
                        );
                        let mut inner_virt = [E::zero(); 2];

                        for coefficient_pair in blk..blk_end {
                            let j_low =
                                (equality_address_base + coefficient_pair) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * coefficient_pair;
                            let base = 8 * coefficient_pair;
                            let w0 = quad_fold_lut[quad_index_fn(lane_values, base)];
                            let w1 = quad_fold_lut[quad_index_fn(lane_values, base + 4)];
                            lane_out[left] = w0;
                            lane_out[left + 1] = w1;
                            let dw = w1 - w0;

                            inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                            inner_virt[1] += e_in * (dw * dw);

                            let mut p0 = alpha_round2[left] * lane_weight;
                            let mut p1 = alpha_round2[left + 1] * lane_weight;
                            Self::add_trace_pair_to_relation_factor(
                                trace_round2,
                                lane,
                                left,
                                next_coeff_count,
                                &mut p0,
                                &mut p1,
                            );
                            accumulate_relation_coeffs(&mut rel, w0, dw, p0, p1);
                        }

                        let e_out = e_second[j_high];
                        virt[0] += e_out * inner_virt[0];
                        virt[1] += e_out * inner_virt[1];
                        blk = blk_end;
                    }

                    (virt, rel)
                })
                .reduce(
                    || ([E::zero(); 2], [E::zero(); 3]),
                    |(mut va, mut ra), (vb, rb)| {
                        for (ai, bi) in va.iter_mut().zip(vb.iter()) {
                            *ai += *bi;
                        }
                        for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                            *ai += *bi;
                        }
                        (va, ra)
                    },
                );

            #[cfg(not(feature = "parallel"))]
            let (virt_coeffs, rel_coeffs) = {
                let mut virt = [E::zero(); 2];
                let mut rel = [E::zero(); 3];
                for (lane, lane_out) in out.chunks_mut(next_coeff_count).enumerate() {
                    let lane_start = lane * coeff_count;
                    let lane_values = &compact_witness[lane_start..lane_start + coeff_count];
                    let lane_weight = relation_lane_weights[lane];
                    let equality_address_base = lane * current_coefficient_half;
                    let mut blk = 0usize;

                    while blk < current_coefficient_half {
                        let (j_high, blk_end) = stage2_eq_block(
                            equality_address_base,
                            blk,
                            num_first,
                            first_bits,
                            block_size,
                            current_coefficient_half,
                        );
                        let mut inner_virt = [E::zero(); 2];

                        for coefficient_pair in blk..blk_end {
                            let j_low =
                                (equality_address_base + coefficient_pair) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * coefficient_pair;
                            let base = 8 * coefficient_pair;
                            let w0 = quad_fold_lut[quad_index_fn(lane_values, base)];
                            let w1 = quad_fold_lut[quad_index_fn(lane_values, base + 4)];
                            lane_out[left] = w0;
                            lane_out[left + 1] = w1;
                            let dw = w1 - w0;

                            inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                            inner_virt[1] += e_in * (dw * dw);

                            let mut p0 = alpha_round2[left] * lane_weight;
                            let mut p1 = alpha_round2[left + 1] * lane_weight;
                            Self::add_trace_pair_to_relation_factor(
                                trace_round2,
                                lane,
                                left,
                                next_coeff_count,
                                &mut p0,
                                &mut p1,
                            );
                            accumulate_relation_coeffs(&mut rel, w0, dw, p0, p1);
                        }

                        let e_out = e_second[j_high];
                        virt[0] += e_out * inner_virt[0];
                        virt[1] += e_out * inner_virt[1];
                        blk = blk_end;
                    }
                }
                (virt, rel)
            };

            (out, NormRoundTerms::SkipLinear(virt_coeffs), rel_coeffs)
        } else {
            #[cfg(feature = "parallel")]
            let (virt_coeffs, rel_coeffs) = out
                .par_chunks_mut(next_coeff_count)
                .enumerate()
                .map(|(lane, lane_out)| {
                    let lane_start = lane * coeff_count;
                    let lane_values = &compact_witness[lane_start..lane_start + coeff_count];
                    let lane_weight = relation_lane_weights[lane];
                    let equality_address_base = lane * current_coefficient_half;
                    let mut virt = [E::zero(); 3];
                    let mut rel = [E::zero(); 3];
                    let mut blk = 0usize;

                    while blk < current_coefficient_half {
                        let (j_high, blk_end) = stage2_eq_block(
                            equality_address_base,
                            blk,
                            num_first,
                            first_bits,
                            block_size,
                            current_coefficient_half,
                        );
                        let mut inner_virt = [E::zero(); 3];

                        for coefficient_pair in blk..blk_end {
                            let j_low =
                                (equality_address_base + coefficient_pair) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * coefficient_pair;
                            let base = 8 * coefficient_pair;
                            let w0 = quad_fold_lut[quad_index_fn(lane_values, base)];
                            let w1 = quad_fold_lut[quad_index_fn(lane_values, base + 4)];
                            lane_out[left] = w0;
                            lane_out[left + 1] = w1;
                            let dw = w1 - w0;
                            let two_w0_plus_one = w0 + w0 + E::one();

                            inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                            inner_virt[1] += e_in * (dw * two_w0_plus_one);
                            inner_virt[2] += e_in * (dw * dw);

                            let mut p0 = alpha_round2[left] * lane_weight;
                            let mut p1 = alpha_round2[left + 1] * lane_weight;
                            Self::add_trace_pair_to_relation_factor(
                                trace_round2,
                                lane,
                                left,
                                next_coeff_count,
                                &mut p0,
                                &mut p1,
                            );
                            accumulate_relation_coeffs(&mut rel, w0, dw, p0, p1);
                        }

                        let e_out = e_second[j_high];
                        virt[0] += e_out * inner_virt[0];
                        virt[1] += e_out * inner_virt[1];
                        virt[2] += e_out * inner_virt[2];
                        blk = blk_end;
                    }

                    (virt, rel)
                })
                .reduce(
                    || ([E::zero(); 3], [E::zero(); 3]),
                    |(mut va, mut ra), (vb, rb)| {
                        for (ai, bi) in va.iter_mut().zip(vb.iter()) {
                            *ai += *bi;
                        }
                        for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                            *ai += *bi;
                        }
                        (va, ra)
                    },
                );

            #[cfg(not(feature = "parallel"))]
            let (virt_coeffs, rel_coeffs) = {
                let mut virt = [E::zero(); 3];
                let mut rel = [E::zero(); 3];
                for (lane, lane_out) in out.chunks_mut(next_coeff_count).enumerate() {
                    let lane_start = lane * coeff_count;
                    let lane_values = &compact_witness[lane_start..lane_start + coeff_count];
                    let lane_weight = relation_lane_weights[lane];
                    let equality_address_base = lane * current_coefficient_half;
                    let mut blk = 0usize;

                    while blk < current_coefficient_half {
                        let (j_high, blk_end) = stage2_eq_block(
                            equality_address_base,
                            blk,
                            num_first,
                            first_bits,
                            block_size,
                            current_coefficient_half,
                        );
                        let mut inner_virt = [E::zero(); 3];

                        for coefficient_pair in blk..blk_end {
                            let j_low =
                                (equality_address_base + coefficient_pair) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * coefficient_pair;
                            let base = 8 * coefficient_pair;
                            let w0 = quad_fold_lut[quad_index_fn(lane_values, base)];
                            let w1 = quad_fold_lut[quad_index_fn(lane_values, base + 4)];
                            lane_out[left] = w0;
                            lane_out[left + 1] = w1;
                            let dw = w1 - w0;
                            let two_w0_plus_one = w0 + w0 + E::one();

                            inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                            inner_virt[1] += e_in * (dw * two_w0_plus_one);
                            inner_virt[2] += e_in * (dw * dw);

                            let mut p0 = alpha_round2[left] * lane_weight;
                            let mut p1 = alpha_round2[left + 1] * lane_weight;
                            Self::add_trace_pair_to_relation_factor(
                                trace_round2,
                                lane,
                                left,
                                next_coeff_count,
                                &mut p0,
                                &mut p1,
                            );
                            accumulate_relation_coeffs(&mut rel, w0, dw, p0, p1);
                        }

                        let e_out = e_second[j_high];
                        virt[0] += e_out * inner_virt[0];
                        virt[1] += e_out * inner_virt[1];
                        virt[2] += e_out * inner_virt[2];
                        blk = blk_end;
                    }
                }
                (virt, rel)
            };

            (out, NormRoundTerms::Full(virt_coeffs), rel_coeffs)
        }
    }
}
