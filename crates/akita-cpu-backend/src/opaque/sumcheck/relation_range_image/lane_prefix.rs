use super::*;

#[inline]
#[allow(clippy::too_many_arguments)]
fn accumulate_fused_partial_lane_relation<E: Field>(
    linear_terms: &PreparedProverLinearTerms<E>,
    linear_coeff_count: usize,
    rel: &mut [E; 3],
    w0: E,
    dw: E,
    p0: E,
    p1: E,
    coefficient: usize,
    next_left_lane: usize,
    next_live_lane_count: usize,
) {
    let (t0, t1) = if next_left_lane + 1 < next_live_lane_count {
        linear_terms.pair_at_lanes(
            next_left_lane,
            next_left_lane + 1,
            coefficient,
            linear_coeff_count,
        )
    } else {
        (
            linear_terms.get(next_left_lane, coefficient, linear_coeff_count),
            E::zero(),
        )
    };
    accumulate_relation_coeffs(rel, w0, dw, p0 + t0, p1 + t1);
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn accumulate_fused_partial_lane_relation_signed<E: Field + Unreduced>(
    linear_terms: &PreparedProverLinearTerms<E>,
    linear_coeff_count: usize,
    rel: &mut [E::SmallProduct; 6],
    w0: i64,
    dw: i64,
    p0: E,
    p1: E,
    coefficient: usize,
    left: usize,
    live_lane_count: usize,
) {
    let (t0, t1) = if left + 1 < live_lane_count {
        linear_terms.pair_at_lanes(left, left + 1, coefficient, linear_coeff_count)
    } else {
        (
            linear_terms.get(left, coefficient, linear_coeff_count),
            E::zero(),
        )
    };
    accumulate_relation_coeffs_signed(rel, w0, dw, p0 + t0, p1 + t1);
}

#[allow(clippy::too_many_arguments)]
fn fold_lane_tile_and_compute_next_round<E: Field + Ring + Unreduced, const SKIP_LINEAR: bool>(
    linear_terms: &PreparedProverLinearTerms<E>,
    folded_witness: &[E],
    target: &mut [E],
    first_pair: usize,
    next_live_lane_count: usize,
    alpha_factor: E,
    next_relation_lane_weights: &[E],
    challenge: E,
    e_first: &[E],
    e_second: &[E],
    block_size: usize,
) -> ([E; 3], [E; 3]) {
    let num_first = e_first.len();
    let first_bits = num_first.trailing_zeros() as usize;
    let end_pair = first_pair + target.len().div_ceil(2);
    let mut virt = [E::zero(); 3];
    let mut rel = [E::zero(); 3];
    let mut blk = first_pair;

    while blk < end_pair {
        let (j_high, blk_end) =
            stage2_eq_block(0, blk, num_first, first_bits, block_size, end_pair);
        let mut inner_virt = [E::zero(); 3];

        for lane_pair in blk..blk_end {
            let next_left_lane = 2 * lane_pair;
            let local_left = next_left_lane - 2 * first_pair;
            let w0 = fold_folded_lane_pair(folded_witness, 2 * next_left_lane, challenge);
            target[local_left] = w0;
            let w1 = if next_left_lane + 1 < next_live_lane_count {
                let w1 = fold_folded_lane_pair(folded_witness, 2 * next_left_lane + 2, challenge);
                target[local_left + 1] = w1;
                w1
            } else {
                E::zero()
            };
            let dw = w1 - w0;

            let e_in = e_first[lane_pair & (num_first - 1)];
            inner_virt[0] += e_in * (w0 * (w0 + E::one()));
            if !SKIP_LINEAR {
                inner_virt[1] += e_in * (dw * (w0 + w0 + E::one()));
            }
            inner_virt[2] += e_in * (dw * dw);

            let p0 = alpha_factor * next_relation_lane_weights[next_left_lane];
            let p1 = alpha_factor * next_relation_lane_weights[next_left_lane + 1];
            accumulate_fused_partial_lane_relation(
                linear_terms,
                1,
                &mut rel,
                w0,
                dw,
                p0,
                p1,
                0,
                next_left_lane,
                next_live_lane_count,
            );
        }

        let e_out = e_second[j_high];
        virt[0] += e_out * inner_virt[0];
        if !SKIP_LINEAR {
            virt[1] += e_out * inner_virt[1];
        }
        virt[2] += e_out * inner_virt[2];
        blk = blk_end;
    }

    (virt, rel)
}

impl<E: Field + Ring + Unreduced> RelationRangeImageProver<E> {
    #[tracing::instrument(
        skip_all,
        name = "RelationRangeImageProver::fuse_folded_partial_lane_and_compute_next_round"
    )]
    pub(super) fn fuse_folded_partial_lane_and_compute_next_round(
        &self,
        folded_witness: &[E],
        weights: &RelationWeightFactorization<E>,
        r: E,
    ) -> (Vec<E>, Vec<E>, NormRoundTerms<E>, [E; 3]) {
        debug_assert!(self.next_uses_partial_lane_round());
        debug_assert!(self.current_lane_width() >= 2);
        // Lane rounds follow every coefficient round, so one coefficient row remains.
        debug_assert_eq!(weights.common_alpha_factor().len(), 1);
        debug_assert_eq!(folded_witness.len(), self.live_lane_count);

        let next_live_lane_count = self.live_lane_count.div_ceil(2);
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let block_size = e_first.len().min(next_live_lane_count.div_ceil(2));
        let tile_pairs = crate::opaque::sumcheck::single_row_tile_pairs(block_size);
        let alpha_factor = weights.common_alpha_factor()[0];
        let next_relation_lane_weights =
            Self::fold_relation_lane_weights(weights.relation_lane_weights(), r);
        let skip_linear = self.can_skip_norm_linear_coeff();
        let mut out = vec![E::zero(); next_live_lane_count];

        let tile_terms: Vec<([E; 3], [E; 3])> = cfg_chunks_mut!(out, 2 * tile_pairs)
            .enumerate()
            .map(|(tile, target)| {
                if skip_linear {
                    fold_lane_tile_and_compute_next_round::<E, true>(
                        &self.linear_terms,
                        folded_witness,
                        target,
                        tile * tile_pairs,
                        next_live_lane_count,
                        alpha_factor,
                        &next_relation_lane_weights,
                        r,
                        e_first,
                        e_second,
                        block_size,
                    )
                } else {
                    fold_lane_tile_and_compute_next_round::<E, false>(
                        &self.linear_terms,
                        folded_witness,
                        target,
                        tile * tile_pairs,
                        next_live_lane_count,
                        alpha_factor,
                        &next_relation_lane_weights,
                        r,
                        e_first,
                        e_second,
                        block_size,
                    )
                }
            })
            .collect();
        let (virt, rel) =
            tile_terms
                .into_iter()
                .fold(([E::zero(); 3], [E::zero(); 3]), |mut totals, terms| {
                    add_round_terms(&mut totals, terms);
                    totals
                });

        let virt_terms = if skip_linear {
            NormRoundTerms::SkipLinear([virt[0], virt[2]])
        } else {
            NormRoundTerms::Full(virt)
        };
        (out, next_relation_lane_weights, virt_terms, rel)
    }

    #[tracing::instrument(
        skip_all,
        name = "RelationRangeImageProver::compute_compact_partial_lane_round_terms"
    )]
    pub(super) fn compute_compact_partial_lane_round_terms(
        &self,
        compact_witness: PackedSignedDigitView<'_>,
        weights: &RelationWeightFactorization<E>,
    ) -> (NormRoundTerms<E>, [E; 3]) {
        debug_assert!(self.rounds_completed >= self.coefficient_bits());
        debug_assert!(self.lane_rounds_completed() < self.lane_bits);
        debug_assert_eq!(
            compact_witness.len(),
            self.live_lane_count * weights.common_alpha_factor().len()
        );

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros() as usize;
        let current_lane_half = 1usize << (self.current_lane_width() - 1);
        let live_pairs = self.live_lane_count.div_ceil(2);
        let block_size = num_first.min(live_pairs);
        let common_alpha_factor = weights.common_alpha_factor();
        let relation_lane_weights = weights.relation_lane_weights();
        let linear_terms = &self.linear_terms;
        let coeff_count = common_alpha_factor.len();
        debug_assert_eq!(relation_lane_weights.len(), self.current_lane_capacity());

        if self.can_skip_norm_linear_coeff() {
            let (virt_coeffs, rel_accum) = cfg_fold_reduce!(
                0..common_alpha_factor.len(),
                || ([E::zero(); 2], [E::SmallProduct::zero(); 6]),
                |(mut virt, mut rel), coefficient| {
                    let coefficient_start = coefficient * self.live_lane_count;
                    let alpha_factor = common_alpha_factor[coefficient];
                    let equality_address_base = coefficient * current_lane_half;

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let (j_high, blk_end) = stage2_eq_block(
                            equality_address_base,
                            blk,
                            num_first,
                            first_bits,
                            block_size,
                            live_pairs,
                        );
                        let mut inner_virt = [E::SmallProduct::zero(); 2];

                        for lane_pair in blk..blk_end {
                            let j_low = (equality_address_base + lane_pair) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * lane_pair;
                            let w0 = i32::from(compact_witness.at(coefficient_start + left));
                            let w1 = if left + 1 < self.live_lane_count {
                                i32::from(compact_witness.at(coefficient_start + left + 1))
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

                            let lane_weight0 = relation_lane_weights[left];
                            let lane_weight1 = relation_lane_weights[left + 1];
                            let p0 = alpha_factor * lane_weight0;
                            let p1 = alpha_factor * lane_weight1;
                            accumulate_fused_partial_lane_relation_signed(
                                linear_terms,
                                coeff_count,
                                &mut rel,
                                w0_i64,
                                dw_i64,
                                p0,
                                p1,
                                coefficient,
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
                || ([E::zero(); 3], [E::SmallProduct::zero(); 6]),
                |(mut virt, mut rel), coefficient| {
                    let coefficient_start = coefficient * self.live_lane_count;
                    let alpha_factor = common_alpha_factor[coefficient];
                    let equality_address_base = coefficient * current_lane_half;

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let (j_high, blk_end) = stage2_eq_block(
                            equality_address_base,
                            blk,
                            num_first,
                            first_bits,
                            block_size,
                            live_pairs,
                        );
                        let mut inner_virt = [E::SmallProduct::zero(); 4];

                        for lane_pair in blk..blk_end {
                            let j_low = (equality_address_base + lane_pair) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * lane_pair;
                            let w0 = i32::from(compact_witness.at(coefficient_start + left));
                            let w1 = if left + 1 < self.live_lane_count {
                                i32::from(compact_witness.at(coefficient_start + left + 1))
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

                            let lane_weight0 = relation_lane_weights[left];
                            let lane_weight1 = relation_lane_weights[left + 1];
                            let p0 = alpha_factor * lane_weight0;
                            let p1 = alpha_factor * lane_weight1;
                            accumulate_fused_partial_lane_relation_signed(
                                linear_terms,
                                coeff_count,
                                &mut rel,
                                w0_i64,
                                dw_i64,
                                p0,
                                p1,
                                coefficient,
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
        name = "RelationRangeImageProver::compute_folded_partial_lane_round_terms"
    )]
    pub(super) fn compute_folded_partial_lane_round_terms(
        &self,
        folded_witness: &[E],
        weights: &RelationWeightFactorization<E>,
    ) -> (NormRoundTerms<E>, [E; 3]) {
        debug_assert!(self.rounds_completed >= self.coefficient_bits());
        debug_assert!(self.lane_rounds_completed() < self.lane_bits);
        debug_assert_eq!(
            folded_witness.len(),
            self.live_lane_count * weights.common_alpha_factor().len()
        );

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros() as usize;
        let current_lane_half = 1usize << (self.current_lane_width() - 1);
        let live_pairs = self.live_lane_count.div_ceil(2);
        let block_size = num_first.min(live_pairs);
        let common_alpha_factor = weights.common_alpha_factor();
        let relation_lane_weights = weights.relation_lane_weights();
        let linear_terms = &self.linear_terms;
        let coeff_count = common_alpha_factor.len();
        let tiles =
            crate::opaque::sumcheck::ReductionTiles::new(coeff_count, live_pairs, block_size);
        debug_assert_eq!(relation_lane_weights.len(), self.current_lane_capacity());

        if self.can_skip_norm_linear_coeff() {
            let (virt_coeffs, rel_coeffs) = cfg_fold_reduce!(
                tiles.work_items(),
                || ([E::zero(); 2], [E::zero(); 3]),
                |(mut virt, mut rel), work_item| {
                    let tile = tiles.decode(work_item);
                    let coefficient = tile.outer;
                    let blk = tile.inner.start;
                    let coefficient_start = coefficient * self.live_lane_count;
                    let coefficient_values = &folded_witness
                        [coefficient_start..coefficient_start + self.live_lane_count];
                    let alpha_factor = common_alpha_factor[coefficient];
                    let equality_address_base = coefficient * current_lane_half;
                    let (j_high, blk_end) = stage2_eq_block(
                        equality_address_base,
                        blk,
                        num_first,
                        first_bits,
                        tile.inner.len(),
                        live_pairs,
                    );
                    let mut inner_virt = [E::zero(); 2];

                    for lane_pair in blk..blk_end {
                        let j_low = (equality_address_base + lane_pair) & (num_first - 1);
                        let e_in = e_first[j_low];
                        let left = 2 * lane_pair;
                        let w0 = coefficient_values[left];
                        let w1 = if left + 1 < self.live_lane_count {
                            coefficient_values[left + 1]
                        } else {
                            E::zero()
                        };
                        let dw = w1 - w0;

                        inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                        inner_virt[1] += e_in * (dw * dw);

                        let lane_weight0 = relation_lane_weights[left];
                        let lane_weight1 = relation_lane_weights[left + 1];
                        let p0 = alpha_factor * lane_weight0;
                        let p1 = alpha_factor * lane_weight1;
                        accumulate_fused_partial_lane_relation(
                            linear_terms,
                            coeff_count,
                            &mut rel,
                            w0,
                            dw,
                            p0,
                            p1,
                            coefficient,
                            left,
                            self.live_lane_count,
                        );
                    }

                    let e_out = e_second[j_high];
                    virt[0] += e_out * inner_virt[0];
                    virt[1] += e_out * inner_virt[1];
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
                tiles.work_items(),
                || ([E::zero(); 3], [E::zero(); 3]),
                |(mut virt, mut rel), work_item| {
                    let tile = tiles.decode(work_item);
                    let coefficient = tile.outer;
                    let blk = tile.inner.start;
                    let coefficient_start = coefficient * self.live_lane_count;
                    let coefficient_values = &folded_witness
                        [coefficient_start..coefficient_start + self.live_lane_count];
                    let alpha_factor = common_alpha_factor[coefficient];
                    let equality_address_base = coefficient * current_lane_half;
                    let (j_high, blk_end) = stage2_eq_block(
                        equality_address_base,
                        blk,
                        num_first,
                        first_bits,
                        tile.inner.len(),
                        live_pairs,
                    );
                    let mut inner_virt = [E::zero(); 3];

                    for lane_pair in blk..blk_end {
                        let j_low = (equality_address_base + lane_pair) & (num_first - 1);
                        let e_in = e_first[j_low];
                        let left = 2 * lane_pair;
                        let w0 = coefficient_values[left];
                        let w1 = if left + 1 < self.live_lane_count {
                            coefficient_values[left + 1]
                        } else {
                            E::zero()
                        };
                        let dw = w1 - w0;
                        let two_w0_plus_one = w0 + w0 + E::one();

                        inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                        inner_virt[1] += e_in * (dw * two_w0_plus_one);
                        inner_virt[2] += e_in * (dw * dw);

                        let lane_weight0 = relation_lane_weights[left];
                        let lane_weight1 = relation_lane_weights[left + 1];
                        let p0 = alpha_factor * lane_weight0;
                        let p1 = alpha_factor * lane_weight1;
                        accumulate_fused_partial_lane_relation(
                            linear_terms,
                            coeff_count,
                            &mut rel,
                            w0,
                            dw,
                            p0,
                            p1,
                            coefficient,
                            left,
                            self.live_lane_count,
                        );
                    }

                    let e_out = e_second[j_high];
                    virt[0] += e_out * inner_virt[0];
                    virt[1] += e_out * inner_virt[1];
                    virt[2] += e_out * inner_virt[2];
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

    pub(super) fn fold_compact_partial_lanes(
        compact_witness: PackedSignedDigitView<'_>,
        live_lane_count: usize,
        coeff_count: usize,
        fold_lut: &CompactPairFoldLut<E>,
    ) -> Vec<E> {
        let next_live_lane_count = live_lane_count.div_ceil(2);
        let mut out = vec![E::zero(); coeff_count * next_live_lane_count];

        cfg_chunks_mut!(out, next_live_lane_count)
            .enumerate()
            .for_each(|(coefficient, coefficient_out)| {
                let coefficient_start = coefficient * live_lane_count;
                for (lane_pair, dst) in coefficient_out.iter_mut().enumerate() {
                    let left = 2 * lane_pair;
                    let w_1 = if left + 1 < live_lane_count {
                        i16::from(compact_witness.at(coefficient_start + left + 1))
                    } else {
                        0
                    };
                    *dst =
                        fold_lut.fold(i16::from(compact_witness.at(coefficient_start + left)), w_1);
                }
            });

        out
    }

    pub(super) fn fold_folded_partial_lanes(
        folded_witness: &[E],
        live_lane_count: usize,
        coeff_count: usize,
        r: E,
    ) -> Vec<E> {
        let next_live_lane_count = live_lane_count.div_ceil(2);
        let mut out = vec![E::zero(); coeff_count * next_live_lane_count];

        cfg_chunks_mut!(out, next_live_lane_count)
            .enumerate()
            .for_each(|(coefficient, coefficient_out)| {
                let coefficient_start = coefficient * live_lane_count;
                let coefficient_values =
                    &folded_witness[coefficient_start..coefficient_start + live_lane_count];
                for (lane_pair, dst) in coefficient_out.iter_mut().enumerate() {
                    let left = 2 * lane_pair;
                    let w_0 = coefficient_values[left];
                    let w_1 = if left + 1 < live_lane_count {
                        coefficient_values[left + 1]
                    } else {
                        E::zero()
                    };
                    *dst = w_0 + r * (w_1 - w_0);
                }
            });

        out
    }

    pub(super) fn fold_relation_lane_weights(relation_lane_weights: &[E], r: E) -> Vec<E> {
        debug_assert!(relation_lane_weights.len().is_power_of_two());
        debug_assert!(relation_lane_weights.len() >= 2);
        let next_lane_capacity = relation_lane_weights.len() >> 1;
        cfg_into_iter!(0..next_lane_capacity)
            .map(|lane_pair| {
                let left = 2 * lane_pair;
                let m_0 = relation_lane_weights[left];
                let m_1 = relation_lane_weights[left + 1];
                m_0 + r * (m_1 - m_0)
            })
            .collect()
    }
}
