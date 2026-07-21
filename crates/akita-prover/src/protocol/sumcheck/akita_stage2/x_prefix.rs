use super::*;

#[inline]
#[allow(clippy::too_many_arguments)]
fn accumulate_fused_prefix_x_relation<E: FieldCore>(
    evaluation_trace: &PreparedProverEvaluationTrace<E>,
    table_coeff_count: usize,
    rel: &mut [E; 3],
    w0: E,
    dw: E,
    p0: E,
    p1: E,
    y: usize,
    left_next: usize,
    next_live_lane_count: usize,
) {
    accumulate_relation_coeffs(rel, w0, dw, p0, p1);
    let (t0, t1) = if left_next + 1 < next_live_lane_count {
        evaluation_trace.pair_at_columns(left_next, left_next + 1, y, table_coeff_count)
    } else {
        (
            evaluation_trace.get(left_next, y, table_coeff_count),
            E::zero(),
        )
    };
    accumulate_relation_coeffs(rel, w0, dw, t0, t1);
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn accumulate_fused_prefix_x_relation_signed<E: FieldCore + HasUnreducedOps>(
    evaluation_trace: &PreparedProverEvaluationTrace<E>,
    table_coeff_count: usize,
    rel: &mut [E::MulU64Accum; 6],
    w0: i64,
    dw: i64,
    p0: E,
    p1: E,
    y: usize,
    left: usize,
    live_lane_count: usize,
) {
    accumulate_relation_coeffs_signed(rel, w0, dw, p0, p1);
    let (t0, t1) = if left + 1 < live_lane_count {
        evaluation_trace.pair_at_columns(left, left + 1, y, table_coeff_count)
    } else {
        (evaluation_trace.get(left, y, table_coeff_count), E::zero())
    };
    accumulate_relation_coeffs_signed(rel, w0, dw, t0, t1);
}

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage2Prover<E> {
    #[tracing::instrument(
        skip_all,
        name = "AkitaStage2Prover::fuse_full_prefix_x_and_compute_round"
    )]
    pub(super) fn fuse_full_prefix_x_and_compute_round(
        &self,
        w_full: &[E],
        r: E,
    ) -> (Vec<E>, Vec<E>, NormRoundTerms<E>, [E; 3]) {
        debug_assert!(self.next_uses_partial_lane_round());
        debug_assert!(self.current_lane_width() >= 2);

        let old_live_lane_count = self.live_lane_count;
        let next_live_lane_count = old_live_lane_count.div_ceil(2);
        let coeff_count = self.common_alpha_factor.len();
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros() as usize;
        let next_current_lane_half = 1usize << (self.current_lane_width() - 2);
        let live_pairs = next_live_lane_count.div_ceil(2);
        let block_size = num_first.min(live_pairs);
        let common_alpha_factor = &self.common_alpha_factor;
        let next_relation_lane_weights = Self::fold_m_prefix(&self.relation_lane_weights, r);
        let mut out = vec![E::zero(); coeff_count * next_live_lane_count];
        let evaluation_trace = &self.evaluation_trace;

        if self.can_skip_norm_linear_coeff() {
            #[cfg(feature = "parallel")]
            let (virt_coeffs, rel_coeffs) = out
                .par_chunks_mut(next_live_lane_count)
                .enumerate()
                .map(|(y, row_out)| {
                    let row = &w_full[y * old_live_lane_count..(y + 1) * old_live_lane_count];
                    let alpha = common_alpha_factor[y];
                    let j_base = y * next_current_lane_half;
                    let mut virt = [E::zero(); 2];
                    let mut rel = [E::zero(); 3];

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let (j_high, blk_end) = stage2_eq_block(
                            j_base, blk, num_first, first_bits, block_size, live_pairs,
                        );
                        let mut inner_virt = [E::zero(); 2];

                        for pair_x in blk..blk_end {
                            let left_next = 2 * pair_x;
                            let left_old = 4 * pair_x;
                            let w0 = fold_full_prefix_pair(row, left_old, r);
                            row_out[left_next] = w0;
                            let w1 = if left_next + 1 < next_live_lane_count {
                                let w1 = fold_full_prefix_pair(row, left_old + 2, r);
                                row_out[left_next + 1] = w1;
                                w1
                            } else {
                                E::zero()
                            };
                            let dw = w1 - w0;

                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
                            inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                            inner_virt[1] += e_in * (dw * dw);

                            let m0 = next_relation_lane_weights[left_next];
                            let m1 = next_relation_lane_weights[left_next + 1];
                            let p0 = alpha * m0;
                            let p1 = alpha * m1;
                            accumulate_fused_prefix_x_relation(
                                evaluation_trace,
                                coeff_count,
                                &mut rel,
                                w0,
                                dw,
                                p0,
                                p1,
                                y,
                                left_next,
                                next_live_lane_count,
                            );
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
                for (y, row_out) in out.chunks_mut(next_live_lane_count).enumerate() {
                    let row = &w_full[y * old_live_lane_count..(y + 1) * old_live_lane_count];
                    let alpha = common_alpha_factor[y];
                    let j_base = y * next_current_lane_half;
                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let (j_high, blk_end) = stage2_eq_block(
                            j_base, blk, num_first, first_bits, block_size, live_pairs,
                        );
                        let mut inner_virt = [E::zero(); 2];

                        for pair_x in blk..blk_end {
                            let left_next = 2 * pair_x;
                            let left_old = 4 * pair_x;
                            let w0 = fold_full_prefix_pair(row, left_old, r);
                            row_out[left_next] = w0;
                            let w1 = if left_next + 1 < next_live_lane_count {
                                let w1 = fold_full_prefix_pair(row, left_old + 2, r);
                                row_out[left_next + 1] = w1;
                                w1
                            } else {
                                E::zero()
                            };
                            let dw = w1 - w0;

                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
                            inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                            inner_virt[1] += e_in * (dw * dw);

                            let m0 = next_relation_lane_weights[left_next];
                            let m1 = next_relation_lane_weights[left_next + 1];
                            let p0 = alpha * m0;
                            let p1 = alpha * m1;
                            accumulate_fused_prefix_x_relation(
                                evaluation_trace,
                                coeff_count,
                                &mut rel,
                                w0,
                                dw,
                                p0,
                                p1,
                                y,
                                left_next,
                                next_live_lane_count,
                            );
                        }

                        let e_out = e_second[j_high];
                        virt[0] += e_out * inner_virt[0];
                        virt[1] += e_out * inner_virt[1];
                        blk = blk_end;
                    }
                }
                (virt, rel)
            };

            (
                out,
                next_relation_lane_weights,
                NormRoundTerms::SkipLinear(virt_coeffs),
                rel_coeffs,
            )
        } else {
            #[cfg(feature = "parallel")]
            let (virt_coeffs, rel_coeffs) = out
                .par_chunks_mut(next_live_lane_count)
                .enumerate()
                .map(|(y, row_out)| {
                    let row = &w_full[y * old_live_lane_count..(y + 1) * old_live_lane_count];
                    let alpha = common_alpha_factor[y];
                    let j_base = y * next_current_lane_half;
                    let mut virt = [E::zero(); 3];
                    let mut rel = [E::zero(); 3];

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let (j_high, blk_end) = stage2_eq_block(
                            j_base, blk, num_first, first_bits, block_size, live_pairs,
                        );
                        let mut inner_virt = [E::zero(); 3];

                        for pair_x in blk..blk_end {
                            let left_next = 2 * pair_x;
                            let left_old = 4 * pair_x;
                            let w0 = fold_full_prefix_pair(row, left_old, r);
                            row_out[left_next] = w0;
                            let w1 = if left_next + 1 < next_live_lane_count {
                                let w1 = fold_full_prefix_pair(row, left_old + 2, r);
                                row_out[left_next + 1] = w1;
                                w1
                            } else {
                                E::zero()
                            };
                            let dw = w1 - w0;
                            let two_w0_plus_one = w0 + w0 + E::one();

                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
                            inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                            inner_virt[1] += e_in * (dw * two_w0_plus_one);
                            inner_virt[2] += e_in * (dw * dw);

                            let m0 = next_relation_lane_weights[left_next];
                            let m1 = next_relation_lane_weights[left_next + 1];
                            let p0 = alpha * m0;
                            let p1 = alpha * m1;
                            accumulate_fused_prefix_x_relation(
                                evaluation_trace,
                                coeff_count,
                                &mut rel,
                                w0,
                                dw,
                                p0,
                                p1,
                                y,
                                left_next,
                                next_live_lane_count,
                            );
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
                for (y, row_out) in out.chunks_mut(next_live_lane_count).enumerate() {
                    let row = &w_full[y * old_live_lane_count..(y + 1) * old_live_lane_count];
                    let alpha = common_alpha_factor[y];
                    let j_base = y * next_current_lane_half;
                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let (j_high, blk_end) = stage2_eq_block(
                            j_base, blk, num_first, first_bits, block_size, live_pairs,
                        );
                        let mut inner_virt = [E::zero(); 3];

                        for pair_x in blk..blk_end {
                            let left_next = 2 * pair_x;
                            let left_old = 4 * pair_x;
                            let w0 = fold_full_prefix_pair(row, left_old, r);
                            row_out[left_next] = w0;
                            let w1 = if left_next + 1 < next_live_lane_count {
                                let w1 = fold_full_prefix_pair(row, left_old + 2, r);
                                row_out[left_next + 1] = w1;
                                w1
                            } else {
                                E::zero()
                            };
                            let dw = w1 - w0;
                            let two_w0_plus_one = w0 + w0 + E::one();

                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
                            inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                            inner_virt[1] += e_in * (dw * two_w0_plus_one);
                            inner_virt[2] += e_in * (dw * dw);

                            let m0 = next_relation_lane_weights[left_next];
                            let m1 = next_relation_lane_weights[left_next + 1];
                            let p0 = alpha * m0;
                            let p1 = alpha * m1;
                            accumulate_fused_prefix_x_relation(
                                evaluation_trace,
                                coeff_count,
                                &mut rel,
                                w0,
                                dw,
                                p0,
                                p1,
                                y,
                                left_next,
                                next_live_lane_count,
                            );
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

            (
                out,
                next_relation_lane_weights,
                NormRoundTerms::Full(virt_coeffs),
                rel_coeffs,
            )
        }
    }

    #[tracing::instrument(
        skip_all,
        name = "AkitaStage2Prover::compute_round_compact_prefix_x_terms"
    )]
    pub(super) fn compute_round_compact_prefix_x_terms(
        &self,
        w_compact: &[i8],
    ) -> (NormRoundTerms<E>, [E; 3]) {
        debug_assert!(self.rounds_completed >= self.coefficient_bits());
        debug_assert!(self.lane_rounds_completed() < self.lane_bits);
        debug_assert_eq!(
            w_compact.len(),
            self.live_lane_count * self.common_alpha_factor.len()
        );

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros() as usize;
        let current_lane_half = 1usize << (self.current_lane_width() - 1);
        let live_pairs = self.live_lane_count.div_ceil(2);
        let block_size = num_first.min(live_pairs);
        let common_alpha_factor = &self.common_alpha_factor;
        let relation_lane_weights = &self.relation_lane_weights;
        let evaluation_trace = &self.evaluation_trace;
        let coeff_count = common_alpha_factor.len();
        debug_assert_eq!(relation_lane_weights.len(), self.current_lane_capacity());

        if self.can_skip_norm_linear_coeff() {
            let (virt_coeffs, rel_accum) = cfg_fold_reduce!(
                0..common_alpha_factor.len(),
                || ([E::zero(); 2], [E::MulU64Accum::zero(); 6]),
                |(mut virt, mut rel), y| {
                    let row_start = y * self.live_lane_count;
                    let row = &w_compact[row_start..row_start + self.live_lane_count];
                    let alpha = common_alpha_factor[y];
                    let j_base = y * current_lane_half;

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let (j_high, blk_end) = stage2_eq_block(
                            j_base, blk, num_first, first_bits, block_size, live_pairs,
                        );
                        let mut inner_virt = [E::MulU64Accum::zero(); 2];

                        for pair_x in blk..blk_end {
                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * pair_x;
                            let w0 = row[left] as i32;
                            let w1 = if left + 1 < self.live_lane_count {
                                row[left + 1] as i32
                            } else {
                                0
                            };
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

                            let m0 = relation_lane_weights[left];
                            let m1 = relation_lane_weights[left + 1];
                            let p0 = alpha * m0;
                            let p1 = alpha * m1;
                            accumulate_fused_prefix_x_relation_signed(
                                evaluation_trace,
                                coeff_count,
                                &mut rel,
                                w0_i64,
                                dw_i64,
                                p0,
                                p1,
                                y,
                                left,
                                self.live_lane_count,
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
                0..common_alpha_factor.len(),
                || ([E::zero(); 3], [E::MulU64Accum::zero(); 6]),
                |(mut virt, mut rel), y| {
                    let row_start = y * self.live_lane_count;
                    let row = &w_compact[row_start..row_start + self.live_lane_count];
                    let alpha = common_alpha_factor[y];
                    let j_base = y * current_lane_half;

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let (j_high, blk_end) = stage2_eq_block(
                            j_base, blk, num_first, first_bits, block_size, live_pairs,
                        );
                        let mut inner_virt = [E::MulU64Accum::zero(); 4];

                        for pair_x in blk..blk_end {
                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * pair_x;
                            let w0 = row[left] as i32;
                            let w1 = if left + 1 < self.live_lane_count {
                                row[left + 1] as i32
                            } else {
                                0
                            };
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

                            let m0 = relation_lane_weights[left];
                            let m1 = relation_lane_weights[left + 1];
                            let p0 = alpha * m0;
                            let p1 = alpha * m1;
                            accumulate_fused_prefix_x_relation_signed(
                                evaluation_trace,
                                coeff_count,
                                &mut rel,
                                w0_i64,
                                dw_i64,
                                p0,
                                p1,
                                y,
                                left,
                                self.live_lane_count,
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
        name = "AkitaStage2Prover::compute_round_full_prefix_x_terms"
    )]
    pub(super) fn compute_round_full_prefix_x_terms(
        &self,
        w_full: &[E],
    ) -> (NormRoundTerms<E>, [E; 3]) {
        debug_assert!(self.rounds_completed >= self.coefficient_bits());
        debug_assert!(self.lane_rounds_completed() < self.lane_bits);
        debug_assert_eq!(
            w_full.len(),
            self.live_lane_count * self.common_alpha_factor.len()
        );

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros() as usize;
        let current_lane_half = 1usize << (self.current_lane_width() - 1);
        let live_pairs = self.live_lane_count.div_ceil(2);
        let block_size = num_first.min(live_pairs);
        let common_alpha_factor = &self.common_alpha_factor;
        let relation_lane_weights = &self.relation_lane_weights;
        let evaluation_trace = &self.evaluation_trace;
        let coeff_count = common_alpha_factor.len();
        debug_assert_eq!(relation_lane_weights.len(), self.current_lane_capacity());

        if self.can_skip_norm_linear_coeff() {
            let (virt_coeffs, rel_coeffs) = cfg_fold_reduce!(
                0..common_alpha_factor.len(),
                || ([E::zero(); 2], [E::zero(); 3]),
                |(mut virt, mut rel), y| {
                    let row_start = y * self.live_lane_count;
                    let row = &w_full[row_start..row_start + self.live_lane_count];
                    let alpha = common_alpha_factor[y];
                    let j_base = y * current_lane_half;

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let (j_high, blk_end) = stage2_eq_block(
                            j_base, blk, num_first, first_bits, block_size, live_pairs,
                        );
                        let mut inner_virt = [E::zero(); 2];

                        for pair_x in blk..blk_end {
                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * pair_x;
                            let w0 = row[left];
                            let w1 = if left + 1 < self.live_lane_count {
                                row[left + 1]
                            } else {
                                E::zero()
                            };
                            let dw = w1 - w0;

                            inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                            inner_virt[1] += e_in * (dw * dw);

                            let m0 = relation_lane_weights[left];
                            let m1 = relation_lane_weights[left + 1];
                            let p0 = alpha * m0;
                            let p1 = alpha * m1;
                            accumulate_fused_prefix_x_relation(
                                evaluation_trace,
                                coeff_count,
                                &mut rel,
                                w0,
                                dw,
                                p0,
                                p1,
                                y,
                                left,
                                self.live_lane_count,
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
                0..common_alpha_factor.len(),
                || ([E::zero(); 3], [E::zero(); 3]),
                |(mut virt, mut rel), y| {
                    let row_start = y * self.live_lane_count;
                    let row = &w_full[row_start..row_start + self.live_lane_count];
                    let alpha = common_alpha_factor[y];
                    let j_base = y * current_lane_half;

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let (j_high, blk_end) = stage2_eq_block(
                            j_base, blk, num_first, first_bits, block_size, live_pairs,
                        );
                        let mut inner_virt = [E::zero(); 3];

                        for pair_x in blk..blk_end {
                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * pair_x;
                            let w0 = row[left];
                            let w1 = if left + 1 < self.live_lane_count {
                                row[left + 1]
                            } else {
                                E::zero()
                            };
                            let dw = w1 - w0;
                            let two_w0_plus_one = w0 + w0 + E::one();

                            inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                            inner_virt[1] += e_in * (dw * two_w0_plus_one);
                            inner_virt[2] += e_in * (dw * dw);

                            let m0 = relation_lane_weights[left];
                            let m1 = relation_lane_weights[left + 1];
                            let p0 = alpha * m0;
                            let p1 = alpha * m1;
                            accumulate_fused_prefix_x_relation(
                                evaluation_trace,
                                coeff_count,
                                &mut rel,
                                w0,
                                dw,
                                p0,
                                p1,
                                y,
                                left,
                                self.live_lane_count,
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

    pub(super) fn fold_compact_prefix_x(
        w_compact: &[i8],
        live_lane_count: usize,
        coeff_count: usize,
        fold_lut: &CompactPairFoldLut<E>,
    ) -> Vec<E> {
        let next_live_lane_count = live_lane_count.div_ceil(2);
        let mut out = vec![E::zero(); coeff_count * next_live_lane_count];

        #[cfg(feature = "parallel")]
        out.par_chunks_mut(next_live_lane_count)
            .enumerate()
            .for_each(|(y, row_out)| {
                let row_start = y * live_lane_count;
                let row = &w_compact[row_start..row_start + live_lane_count];
                for (pair_x, dst) in row_out.iter_mut().enumerate() {
                    let left = 2 * pair_x;
                    let w_1 = if left + 1 < live_lane_count {
                        i16::from(row[left + 1])
                    } else {
                        0
                    };
                    *dst = fold_lut.fold(i16::from(row[left]), w_1);
                }
            });

        #[cfg(not(feature = "parallel"))]
        for (y, row_out) in out.chunks_mut(next_live_lane_count).enumerate() {
            let row_start = y * live_lane_count;
            let row = &w_compact[row_start..row_start + live_lane_count];
            for (pair_x, dst) in row_out.iter_mut().enumerate() {
                let left = 2 * pair_x;
                let w_1 = if left + 1 < live_lane_count {
                    i16::from(row[left + 1])
                } else {
                    0
                };
                *dst = fold_lut.fold(i16::from(row[left]), w_1);
            }
        }

        out
    }

    pub(super) fn fold_full_prefix_x(
        w_full: &[E],
        live_lane_count: usize,
        coeff_count: usize,
        r: E,
    ) -> Vec<E> {
        let next_live_lane_count = live_lane_count.div_ceil(2);
        let mut out = vec![E::zero(); coeff_count * next_live_lane_count];

        #[cfg(feature = "parallel")]
        out.par_chunks_mut(next_live_lane_count)
            .enumerate()
            .for_each(|(y, row_out)| {
                let row_start = y * live_lane_count;
                let row = &w_full[row_start..row_start + live_lane_count];
                for (pair_x, dst) in row_out.iter_mut().enumerate() {
                    let left = 2 * pair_x;
                    let w_0 = row[left];
                    let w_1 = if left + 1 < live_lane_count {
                        row[left + 1]
                    } else {
                        E::zero()
                    };
                    *dst = w_0 + r * (w_1 - w_0);
                }
            });

        #[cfg(not(feature = "parallel"))]
        for (y, row_out) in out.chunks_mut(next_live_lane_count).enumerate() {
            let row_start = y * live_lane_count;
            let row = &w_full[row_start..row_start + live_lane_count];
            for (pair_x, dst) in row_out.iter_mut().enumerate() {
                let left = 2 * pair_x;
                let w_0 = row[left];
                let w_1 = if left + 1 < live_lane_count {
                    row[left + 1]
                } else {
                    E::zero()
                };
                *dst = w_0 + r * (w_1 - w_0);
            }
        }

        out
    }

    pub(super) fn fold_m_prefix(relation_lane_weights: &[E], r: E) -> Vec<E> {
        debug_assert!(relation_lane_weights.len().is_power_of_two());
        debug_assert!(relation_lane_weights.len() >= 2);
        let next_lane_capacity = relation_lane_weights.len() >> 1;
        cfg_into_iter!(0..next_lane_capacity)
            .map(|pair_x| {
                let left = 2 * pair_x;
                let m_0 = relation_lane_weights[left];
                let m_1 = relation_lane_weights[left + 1];
                m_0 + r * (m_1 - m_0)
            })
            .collect()
    }
}
