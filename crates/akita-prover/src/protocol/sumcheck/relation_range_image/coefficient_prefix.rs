use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> RelationRangeImageProver<E> {
    #[tracing::instrument(
        skip_all,
        name = "RelationRangeImageProver::compute_compact_partial_lane_coefficient_round_terms"
    )]
    pub(super) fn compute_compact_partial_lane_coefficient_round_terms(
        &self,
        compact_witness: &[i8],
    ) -> (NormRoundTerms<E>, [E; 3]) {
        debug_assert!(self.in_coefficient_round());
        debug_assert_eq!(
            compact_witness.len(),
            self.live_lane_count * self.common_alpha_factor.len()
        );

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros() as usize;
        let current_coefficient_half = 1usize << (self.current_coefficient_width() - 1);
        let block_size = num_first.min(current_coefficient_half);
        let common_alpha_factor = &self.common_alpha_factor;
        let relation_lane_weights = &self.relation_lane_weights;
        debug_assert_eq!(relation_lane_weights.len(), self.current_lane_capacity());

        if self.can_skip_norm_linear_coeff() {
            let (virt_coeffs, rel_accum) = cfg_fold_reduce!(
                0..self.live_lane_count,
                || ([E::zero(); 2], [E::MulU64Accum::zero(); 6]),
                |(mut virt, mut rel), lane| {
                    let lane_start = lane * common_alpha_factor.len();
                    let lane_values =
                        &compact_witness[lane_start..lane_start + common_alpha_factor.len()];
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
                        let mut inner_virt = [E::MulU64Accum::zero(); 2];

                        for coefficient_pair in blk..blk_end {
                            let j_low =
                                (equality_address_base + coefficient_pair) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * coefficient_pair;
                            let w0 = lane_values[left] as i32;
                            let w1 = lane_values[left + 1] as i32;
                            let dw = w1 - w0;
                            let w0_i64 = w0 as i64;
                            let dw_i64 = dw as i64;

                            let q0 = w0_i64 * (w0_i64 + 1);
                            if q0 != 0 {
                                inner_virt[0] += e_in.mul_u64_unreduced(q0 as u64);
                            }
                            let q2 = dw_i64 * dw_i64;
                            if q2 != 0 {
                                inner_virt[1] += e_in.mul_u64_unreduced(q2 as u64);
                            }

                            let p0 = common_alpha_factor[left] * lane_weight;
                            let p1 = common_alpha_factor[left + 1] * lane_weight;
                            self.accumulate_fused_relation_trace_signed(
                                &mut rel,
                                w0_i64,
                                dw_i64,
                                lane_start + left,
                                p0,
                                p1,
                            );
                        }

                        let reduced_inner: [E; 2] = reduce_compact_virt_skip_linear(inner_virt);
                        let e_out = e_second[j_high];
                        virt[0] += e_out * reduced_inner[0];
                        virt[1] += e_out * reduced_inner[1];
                        blk = blk_end;
                    }

                    (virt, rel)
                },
                |(mut va, mut ra), (vb, rb)| {
                    for (ai, bi) in va.iter_mut().zip(vb.iter()) {
                        *ai += *bi;
                    }
                    for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                        *ai += *bi;
                    }
                    (va, ra)
                }
            );

            (
                NormRoundTerms::SkipLinear(virt_coeffs),
                reduce_compact_rel(rel_accum),
            )
        } else {
            let (virt_coeffs, rel_accum) = cfg_fold_reduce!(
                0..self.live_lane_count,
                || ([E::zero(); 3], [E::MulU64Accum::zero(); 6]),
                |(mut virt, mut rel), lane| {
                    let lane_start = lane * common_alpha_factor.len();
                    let lane_values =
                        &compact_witness[lane_start..lane_start + common_alpha_factor.len()];
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
                        let mut inner_virt = [E::MulU64Accum::zero(); 4];

                        for coefficient_pair in blk..blk_end {
                            let j_low =
                                (equality_address_base + coefficient_pair) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * coefficient_pair;
                            let w0 = lane_values[left] as i32;
                            let w1 = lane_values[left + 1] as i32;
                            let dw = w1 - w0;
                            let w0_i64 = w0 as i64;
                            let dw_i64 = dw as i64;

                            let q0 = w0_i64 * (w0_i64 + 1);
                            if q0 != 0 {
                                inner_virt[0] += e_in.mul_u64_unreduced(q0 as u64);
                            }
                            let q1 = dw_i64 * (2 * w0_i64 + 1);
                            accum_small_signed::<E>(&mut inner_virt, 1, e_in, q1);
                            let q2 = dw_i64 * dw_i64;
                            if q2 != 0 {
                                inner_virt[3] += e_in.mul_u64_unreduced(q2 as u64);
                            }

                            let p0 = common_alpha_factor[left] * lane_weight;
                            let p1 = common_alpha_factor[left + 1] * lane_weight;
                            self.accumulate_fused_relation_trace_signed(
                                &mut rel,
                                w0_i64,
                                dw_i64,
                                lane_start + left,
                                p0,
                                p1,
                            );
                        }

                        let reduced_inner: [E; 3] = reduce_compact_virt(inner_virt);
                        let e_out = e_second[j_high];
                        virt[0] += e_out * reduced_inner[0];
                        virt[1] += e_out * reduced_inner[1];
                        virt[2] += e_out * reduced_inner[2];
                        blk = blk_end;
                    }

                    (virt, rel)
                },
                |(mut va, mut ra), (vb, rb)| {
                    for (ai, bi) in va.iter_mut().zip(vb.iter()) {
                        *ai += *bi;
                    }
                    for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                        *ai += *bi;
                    }
                    (va, ra)
                }
            );

            (
                NormRoundTerms::Full(virt_coeffs),
                reduce_compact_rel(rel_accum),
            )
        }
    }

    #[tracing::instrument(
        skip_all,
        name = "RelationRangeImageProver::compute_folded_partial_lane_coefficient_round_terms"
    )]
    pub(super) fn compute_folded_partial_lane_coefficient_round_terms(
        &self,
        folded_witness: &[E],
    ) -> (NormRoundTerms<E>, [E; 3]) {
        debug_assert!(self.in_coefficient_round());
        debug_assert_eq!(
            folded_witness.len(),
            self.live_lane_count * self.common_alpha_factor.len()
        );

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros() as usize;
        let current_coefficient_half = 1usize << (self.current_coefficient_width() - 1);
        let block_size = num_first.min(current_coefficient_half);
        let common_alpha_factor = &self.common_alpha_factor;
        let relation_lane_weights = &self.relation_lane_weights;
        debug_assert_eq!(relation_lane_weights.len(), self.current_lane_capacity());

        if self.can_skip_norm_linear_coeff() {
            let (virt_coeffs, rel_coeffs) = cfg_fold_reduce!(
                0..self.live_lane_count,
                || ([E::zero(); 2], [E::zero(); 3]),
                |(mut virt, mut rel), lane| {
                    let lane_start = lane * common_alpha_factor.len();
                    let lane_values =
                        &folded_witness[lane_start..lane_start + common_alpha_factor.len()];
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
                            let w0 = lane_values[left];
                            let w1 = lane_values[left + 1];
                            let dw = w1 - w0;

                            inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                            inner_virt[1] += e_in * (dw * dw);

                            let p0 = common_alpha_factor[left] * lane_weight;
                            let p1 = common_alpha_factor[left + 1] * lane_weight;
                            self.accumulate_fused_relation_trace(
                                &mut rel,
                                w0,
                                dw,
                                lane_start + left,
                                p0,
                                p1,
                            );
                        }

                        let e_out = e_second[j_high];
                        virt[0] += e_out * inner_virt[0];
                        virt[1] += e_out * inner_virt[1];
                        blk = blk_end;
                    }

                    (virt, rel)
                },
                |(mut va, mut ra), (vb, rb)| {
                    for (ai, bi) in va.iter_mut().zip(vb.iter()) {
                        *ai += *bi;
                    }
                    for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                        *ai += *bi;
                    }
                    (va, ra)
                }
            );
            (NormRoundTerms::SkipLinear(virt_coeffs), rel_coeffs)
        } else {
            let (virt_coeffs, rel_coeffs) = cfg_fold_reduce!(
                0..self.live_lane_count,
                || ([E::zero(); 3], [E::zero(); 3]),
                |(mut virt, mut rel), lane| {
                    let lane_start = lane * common_alpha_factor.len();
                    let lane_values =
                        &folded_witness[lane_start..lane_start + common_alpha_factor.len()];
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
                            let w0 = lane_values[left];
                            let w1 = lane_values[left + 1];
                            let dw = w1 - w0;
                            let two_w0_plus_one = w0 + w0 + E::one();

                            inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                            inner_virt[1] += e_in * (dw * two_w0_plus_one);
                            inner_virt[2] += e_in * (dw * dw);

                            let p0 = common_alpha_factor[left] * lane_weight;
                            let p1 = common_alpha_factor[left + 1] * lane_weight;
                            self.accumulate_fused_relation_trace(
                                &mut rel,
                                w0,
                                dw,
                                lane_start + left,
                                p0,
                                p1,
                            );
                        }

                        let e_out = e_second[j_high];
                        virt[0] += e_out * inner_virt[0];
                        virt[1] += e_out * inner_virt[1];
                        virt[2] += e_out * inner_virt[2];
                        blk = blk_end;
                    }

                    (virt, rel)
                },
                |(mut va, mut ra), (vb, rb)| {
                    for (ai, bi) in va.iter_mut().zip(vb.iter()) {
                        *ai += *bi;
                    }
                    for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                        *ai += *bi;
                    }
                    (va, ra)
                }
            );
            (NormRoundTerms::Full(virt_coeffs), rel_coeffs)
        }
    }

    pub(super) fn fold_folded_coefficients(
        folded_witness: &[E],
        live_lane_count: usize,
        coeff_count: usize,
        r: E,
    ) -> Vec<E> {
        debug_assert!(coeff_count.is_power_of_two());
        debug_assert!(coeff_count >= 2);
        let next_coeff_count = coeff_count >> 1;
        let mut out = vec![E::zero(); live_lane_count * next_coeff_count];

        cfg_chunks_mut!(out, next_coeff_count)
            .enumerate()
            .for_each(|(lane, lane_out)| {
                let lane_start = lane * coeff_count;
                let lane_values = &folded_witness[lane_start..lane_start + coeff_count];
                for (coefficient_pair, dst) in lane_out.iter_mut().enumerate() {
                    let left = 2 * coefficient_pair;
                    let w0 = lane_values[left];
                    let w1 = lane_values[left + 1];
                    *dst = w0 + r * (w1 - w0);
                }
            });

        out
    }
}
