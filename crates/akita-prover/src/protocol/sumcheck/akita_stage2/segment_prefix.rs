use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage2Prover<E> {
    #[tracing::instrument(
        skip_all,
        name = "AkitaStage2Prover::fused_fold_scan_segment_axis"
    )]
    pub(super) fn fused_fold_scan_segment_axis(
        &self,
        w_full: &[E],
        r: E,
    ) -> (Vec<E>, NormRoundTerms<E>, [E; 3]) {
        debug_assert!(self.next_use_segment_prefix_round_after_current());
        debug_assert!(self.current_segment_width() >= 2);

        let old_live_segments = self.live_segments;
        let next_live_segments = old_live_segments.div_ceil(2);
        let coeff_len = self.relation_weight_coeff_len();
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros() as usize;
        let next_current_x_half = 1usize << (self.current_segment_width() - 2);
        let live_pairs = next_live_segments.div_ceil(2);
        let block_size = num_first.min(live_pairs);
        let mut out = vec![E::zero(); coeff_len * next_live_segments];

        if self.can_skip_norm_linear_coeff() {
            #[cfg(feature = "parallel")]
            let (virt_coeffs, rel_coeffs) = out
                .par_chunks_mut(next_live_segments)
                .enumerate()
                .map(|(y, row_out)| {
                    let row = &w_full[y * old_live_segments..(y + 1) * old_live_segments];
                    let j_base = y * next_current_x_half;
                    let mut virt = [E::zero(); 2];
                    let mut rel = [E::zero(); 3];

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let (j_high, blk_end) = stage2_eq_block(
                            j_base, blk, num_first, first_bits, block_size, live_pairs,
                        );
                        let mut inner_virt = [E::zero(); 2];

                        for pair_segment in blk..blk_end {
                            let left_next = 2 * pair_segment;
                            let left_old = 4 * pair_segment;
                            let w0 = fold_full_prefix_pair(row, left_old, r);
                            row_out[left_next] = w0;
                            let w1 = if left_next + 1 < next_live_segments {
                                let w1 = fold_full_prefix_pair(row, left_old + 2, r);
                                row_out[left_next + 1] = w1;
                                w1
                            } else {
                                E::zero()
                            };
                            let dw = w1 - w0;

                            let j_low = (j_base + pair_segment) & (num_first - 1);
                            let e_in = e_first[j_low];
                            inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                            inner_virt[1] += e_in * (dw * dw);

                            let (p0, p1) =
                                self.relation_weight_pair_tiles(left_next, left_next + 1, y);
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
                for (y, row_out) in out.chunks_mut(next_live_segments).enumerate() {
                    let row = &w_full[y * old_live_segments..(y + 1) * old_live_segments];
                    let j_base = y * next_current_x_half;
                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let (j_high, blk_end) = stage2_eq_block(
                            j_base, blk, num_first, first_bits, block_size, live_pairs,
                        );
                        let mut inner_virt = [E::zero(); 2];

                        for pair_segment in blk..blk_end {
                            let left_next = 2 * pair_segment;
                            let left_old = 4 * pair_segment;
                            let w0 = fold_full_prefix_pair(row, left_old, r);
                            row_out[left_next] = w0;
                            let w1 = if left_next + 1 < next_live_segments {
                                let w1 = fold_full_prefix_pair(row, left_old + 2, r);
                                row_out[left_next + 1] = w1;
                                w1
                            } else {
                                E::zero()
                            };
                            let dw = w1 - w0;

                            let j_low = (j_base + pair_segment) & (num_first - 1);
                            let e_in = e_first[j_low];
                            inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                            inner_virt[1] += e_in * (dw * dw);

                            let (p0, p1) =
                                self.relation_weight_pair_tiles(left_next, left_next + 1, y);
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
                .par_chunks_mut(next_live_segments)
                .enumerate()
                .map(|(y, row_out)| {
                    let row = &w_full[y * old_live_segments..(y + 1) * old_live_segments];
                    let j_base = y * next_current_x_half;
                    let mut virt = [E::zero(); 3];
                    let mut rel = [E::zero(); 3];

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let (j_high, blk_end) = stage2_eq_block(
                            j_base, blk, num_first, first_bits, block_size, live_pairs,
                        );
                        let mut inner_virt = [E::zero(); 3];

                        for pair_segment in blk..blk_end {
                            let left_next = 2 * pair_segment;
                            let left_old = 4 * pair_segment;
                            let w0 = fold_full_prefix_pair(row, left_old, r);
                            row_out[left_next] = w0;
                            let w1 = if left_next + 1 < next_live_segments {
                                let w1 = fold_full_prefix_pair(row, left_old + 2, r);
                                row_out[left_next + 1] = w1;
                                w1
                            } else {
                                E::zero()
                            };
                            let dw = w1 - w0;
                            let two_w0_plus_one = w0 + w0 + E::one();

                            let j_low = (j_base + pair_segment) & (num_first - 1);
                            let e_in = e_first[j_low];
                            inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                            inner_virt[1] += e_in * (dw * two_w0_plus_one);
                            inner_virt[2] += e_in * (dw * dw);

                            let (p0, p1) =
                                self.relation_weight_pair_tiles(left_next, left_next + 1, y);
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
                for (y, row_out) in out.chunks_mut(next_live_segments).enumerate() {
                    let row = &w_full[y * old_live_segments..(y + 1) * old_live_segments];
                    let j_base = y * next_current_x_half;
                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let (j_high, blk_end) = stage2_eq_block(
                            j_base, blk, num_first, first_bits, block_size, live_pairs,
                        );
                        let mut inner_virt = [E::zero(); 3];

                        for pair_segment in blk..blk_end {
                            let left_next = 2 * pair_segment;
                            let left_old = 4 * pair_segment;
                            let w0 = fold_full_prefix_pair(row, left_old, r);
                            row_out[left_next] = w0;
                            let w1 = if left_next + 1 < next_live_segments {
                                let w1 = fold_full_prefix_pair(row, left_old + 2, r);
                                row_out[left_next + 1] = w1;
                                w1
                            } else {
                                E::zero()
                            };
                            let dw = w1 - w0;
                            let two_w0_plus_one = w0 + w0 + E::one();

                            let j_low = (j_base + pair_segment) & (num_first - 1);
                            let e_in = e_first[j_low];
                            inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                            inner_virt[1] += e_in * (dw * two_w0_plus_one);
                            inner_virt[2] += e_in * (dw * dw);

                            let (p0, p1) =
                                self.relation_weight_pair_tiles(left_next, left_next + 1, y);
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

    #[tracing::instrument(
        skip_all,
        name = "AkitaStage2Prover::scan_embedded_segment_compact"
    )]
    pub(super) fn scan_embedded_segment_compact(
        &self,
        w_compact: &[i8],
    ) -> (NormRoundTerms<E>, [E; 3]) {
        debug_assert!(self.rounds_completed >= self.coeff_bits());
        debug_assert!(self.segment_rounds_completed() < self.segment_bits);
        debug_assert_eq!(
            w_compact.len(),
            self.live_segments * self.relation_weight_coeff_len()
        );

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros() as usize;
        let current_x_half = 1usize << (self.current_segment_width() - 1);
        let live_pairs = self.live_segments.div_ceil(2);
        let block_size = num_first.min(live_pairs);

        if self.can_skip_norm_linear_coeff() {
            let (virt_coeffs, rel_accum) = cfg_fold_reduce!(
                0..self.relation_weight_coeff_len(),
                || ([E::zero(); 2], [E::MulU64Accum::zero(); 6]),
                |(mut virt, mut rel), y| {
                    let row_start = y * self.live_segments;
                    let row = &w_compact[row_start..row_start + self.live_segments];
                    let j_base = y * current_x_half;

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let (j_high, blk_end) = stage2_eq_block(
                            j_base, blk, num_first, first_bits, block_size, live_pairs,
                        );
                        let mut inner_virt = [E::MulU64Accum::zero(); 2];

                        for pair_segment in blk..blk_end {
                            let j_low = (j_base + pair_segment) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * pair_segment;
                            let w0 = row[left] as i32;
                            let w1 = if left + 1 < self.live_segments {
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

                            let (p0, p1) = self.relation_weight_pair_tiles(left, left + 1, y);

                            accumulate_relation_coeffs_signed(&mut rel, w0_i64, dw_i64, p0, p1);
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
                0..self.relation_weight_coeff_len(),
                || ([E::zero(); 3], [E::MulU64Accum::zero(); 6]),
                |(mut virt, mut rel), y| {
                    let row_start = y * self.live_segments;
                    let row = &w_compact[row_start..row_start + self.live_segments];
                    let j_base = y * current_x_half;

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let (j_high, blk_end) = stage2_eq_block(
                            j_base, blk, num_first, first_bits, block_size, live_pairs,
                        );
                        let mut inner_virt = [E::MulU64Accum::zero(); 4];

                        for pair_segment in blk..blk_end {
                            let j_low = (j_base + pair_segment) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * pair_segment;
                            let w0 = row[left] as i32;
                            let w1 = if left + 1 < self.live_segments {
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

                            let (p0, p1) = self.relation_weight_pair_tiles(left, left + 1, y);

                            accumulate_relation_coeffs_signed(&mut rel, w0_i64, dw_i64, p0, p1);
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
        name = "AkitaStage2Prover::scan_embedded_segment_full"
    )]
    pub(super) fn scan_embedded_segment_full(&self, w_full: &[E]) -> (NormRoundTerms<E>, [E; 3]) {
        debug_assert!(self.rounds_completed >= self.coeff_bits());
        debug_assert!(self.segment_rounds_completed() < self.segment_bits);
        debug_assert_eq!(
            w_full.len(),
            self.live_segments * self.relation_weight_coeff_len()
        );

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros() as usize;
        let current_x_half = 1usize << (self.current_segment_width() - 1);
        let live_pairs = self.live_segments.div_ceil(2);
        let block_size = num_first.min(live_pairs);

        if self.can_skip_norm_linear_coeff() {
            let (virt_coeffs, rel_coeffs) = cfg_fold_reduce!(
                0..self.relation_weight_coeff_len(),
                || ([E::zero(); 2], [E::zero(); 3]),
                |(mut virt, mut rel), y| {
                    let row_start = y * self.live_segments;
                    let row = &w_full[row_start..row_start + self.live_segments];
                    let j_base = y * current_x_half;

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let (j_high, blk_end) = stage2_eq_block(
                            j_base, blk, num_first, first_bits, block_size, live_pairs,
                        );
                        let mut inner_virt = [E::zero(); 2];

                        for pair_segment in blk..blk_end {
                            let j_low = (j_base + pair_segment) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * pair_segment;
                            let w0 = row[left];
                            let w1 = if left + 1 < self.live_segments {
                                row[left + 1]
                            } else {
                                E::zero()
                            };
                            let dw = w1 - w0;

                            inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                            inner_virt[1] += e_in * (dw * dw);

                            let (p0, p1) = self.relation_weight_pair_tiles(left, left + 1, y);

                            accumulate_relation_coeffs(&mut rel, w0, dw, p0, p1);
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
                0..self.relation_weight_coeff_len(),
                || ([E::zero(); 3], [E::zero(); 3]),
                |(mut virt, mut rel), y| {
                    let row_start = y * self.live_segments;
                    let row = &w_full[row_start..row_start + self.live_segments];
                    let j_base = y * current_x_half;

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let (j_high, blk_end) = stage2_eq_block(
                            j_base, blk, num_first, first_bits, block_size, live_pairs,
                        );
                        let mut inner_virt = [E::zero(); 3];

                        for pair_segment in blk..blk_end {
                            let j_low = (j_base + pair_segment) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * pair_segment;
                            let w0 = row[left];
                            let w1 = if left + 1 < self.live_segments {
                                row[left + 1]
                            } else {
                                E::zero()
                            };
                            let dw = w1 - w0;
                            let two_w0_plus_one = w0 + w0 + E::one();

                            inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                            inner_virt[1] += e_in * (dw * two_w0_plus_one);
                            inner_virt[2] += e_in * (dw * dw);

                            let (p0, p1) = self.relation_weight_pair_tiles(left, left + 1, y);

                            accumulate_relation_coeffs(&mut rel, w0, dw, p0, p1);
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

    pub(super) fn fold_witness_embedded_segment_compact(
        w_compact: &[i8],
        live_segments: usize,
        coeff_len: usize,
        fold_lut: &CompactPairFoldLut<E>,
    ) -> Vec<E> {
        let next_live_segments = live_segments.div_ceil(2);
        let mut out = vec![E::zero(); coeff_len * next_live_segments];

        #[cfg(feature = "parallel")]
        out.par_chunks_mut(next_live_segments)
            .enumerate()
            .for_each(|(y, row_out)| {
                let row_start = y * live_segments;
                let row = &w_compact[row_start..row_start + live_segments];
                for (pair_segment, dst) in row_out.iter_mut().enumerate() {
                    let left = 2 * pair_segment;
                    let w_1 = if left + 1 < live_segments {
                        i16::from(row[left + 1])
                    } else {
                        0
                    };
                    *dst = fold_lut.fold(i16::from(row[left]), w_1);
                }
            });

        #[cfg(not(feature = "parallel"))]
        for (y, row_out) in out.chunks_mut(next_live_segments).enumerate() {
            let row_start = y * live_segments;
            let row = &w_compact[row_start..row_start + live_segments];
            for (pair_segment, dst) in row_out.iter_mut().enumerate() {
                let left = 2 * pair_segment;
                let w_1 = if left + 1 < live_segments {
                    i16::from(row[left + 1])
                } else {
                    0
                };
                *dst = fold_lut.fold(i16::from(row[left]), w_1);
            }
        }

        out
    }

    pub(super) fn fold_witness_embedded_segment_full(
        w_full: &[E],
        live_segments: usize,
        coeff_len: usize,
        r: E,
    ) -> Vec<E> {
        let next_live_segments = live_segments.div_ceil(2);
        let mut out = vec![E::zero(); coeff_len * next_live_segments];

        #[cfg(feature = "parallel")]
        out.par_chunks_mut(next_live_segments)
            .enumerate()
            .for_each(|(y, row_out)| {
                let row_start = y * live_segments;
                let row = &w_full[row_start..row_start + live_segments];
                for (pair_segment, dst) in row_out.iter_mut().enumerate() {
                    let left = 2 * pair_segment;
                    let w_0 = row[left];
                    let w_1 = if left + 1 < live_segments {
                        row[left + 1]
                    } else {
                        E::zero()
                    };
                    *dst = w_0 + r * (w_1 - w_0);
                }
            });

        #[cfg(not(feature = "parallel"))]
        for (y, row_out) in out.chunks_mut(next_live_segments).enumerate() {
            let row_start = y * live_segments;
            let row = &w_full[row_start..row_start + live_segments];
            for (pair_segment, dst) in row_out.iter_mut().enumerate() {
                let left = 2 * pair_segment;
                let w_0 = row[left];
                let w_1 = if left + 1 < live_segments {
                    row[left + 1]
                } else {
                    E::zero()
                };
                *dst = w_0 + r * (w_1 - w_0);
            }
        }

        out
    }

    pub(super) fn fold_relation_weight_embedded_segment(
        evals: &[E],
        live_segments: usize,
        coeff_len: usize,
        r: E,
    ) -> Vec<E> {
        let next_live_segments = live_segments.div_ceil(2);
        let mut out = vec![E::zero(); next_live_segments * coeff_len];
        for pair_segment in 0..next_live_segments {
            let left = 2 * pair_segment;
            let dst_start = pair_segment * coeff_len;
            let left_start = left * coeff_len;
            let right_start = (left + 1) * coeff_len;
            for y in 0..coeff_len {
                let a = evals[left_start + y];
                let b = if left + 1 < live_segments {
                    evals[right_start + y]
                } else {
                    E::zero()
                };
                out[dst_start + y] = a + r * (b - a);
            }
        }
        out
    }

    pub(super) fn fold_relation_weight_embedded_coefficient(
        evals: &[E],
        live_segments: usize,
        coeff_len: usize,
        r: E,
    ) -> Vec<E> {
        let next_coeff_len = coeff_len / 2;
        let mut out = vec![E::zero(); live_segments * next_coeff_len];
        for x in 0..live_segments {
            let col_start = x * coeff_len;
            let col_out_start = x * next_coeff_len;
            for j in 0..next_coeff_len {
                let left = 2 * j;
                let e0 = evals[col_start + left];
                let e1 = evals[col_start + left + 1];
                out[col_out_start + j] = e0 + r * (e1 - e0);
            }
        }
        out
    }

    pub(super) fn fold_relation_weight_initial_batch(
        evals: &[E],
        live_segments: usize,
        coeff_len: usize,
        r0: E,
        r1: E,
    ) -> Vec<E> {
        debug_assert!(coeff_len.is_power_of_two());
        debug_assert!(coeff_len >= 4);
        let next_coeff_len = coeff_len >> 2;
        let mut out = vec![E::zero(); live_segments * next_coeff_len];
        for x in 0..live_segments {
            let src_start = x * coeff_len;
            let dst_start = x * next_coeff_len;
            let column = &evals[src_start..src_start + coeff_len];
            for (quad_y, dst) in out[dst_start..dst_start + next_coeff_len]
                .iter_mut()
                .enumerate()
            {
                let base = 4 * quad_y;
                *dst = Self::direct_fold_e_quad_to_round2(
                    column[base],
                    column[base + 1],
                    column[base + 2],
                    column[base + 3],
                    r0,
                    r1,
                );
            }
        }
        out
    }
}
