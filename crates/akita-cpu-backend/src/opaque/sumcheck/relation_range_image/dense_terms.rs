use super::*;

impl<E: Field + Ring + Unreduced> RelationRangeImageProver<E> {
    #[tracing::instrument(
        skip_all,
        name = "RelationRangeImageProver::compute_round_compact_dense_terms"
    )]
    fn compute_round_compact_dense_terms_with(
        &self,
        compact_witness: PackedSignedDigitView<'_>,
        live_pairs: usize,
        relation_pair: impl Fn(usize) -> (E, E) + Sync,
    ) -> (NormRoundTerms<E>, [E; 3]) {
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let num_second = e_second.len();
        debug_assert!(live_pairs <= num_first * num_second);

        if self.can_skip_norm_linear_coeff() {
            let (virt_coeffs, rel_accum) = cfg_fold_reduce!(
                0..num_second,
                || ([E::zero(); 2], [E::SmallProduct::zero(); 6]),
                |(mut virt, mut rel), j_high| {
                    let mut inner_virt = [E::SmallProduct::zero(); 2];
                    let base = j_high * num_first;
                    for (j_low, &e_in) in e_first.iter().enumerate() {
                        let j = base + j_low;
                        if j >= live_pairs {
                            break;
                        }
                        let w0 = compact_witness.get(2 * j).map_or(0, i32::from);
                        let w1 = compact_witness.get(2 * j + 1).map_or(0, i32::from);
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

                        let (p0, p1) = relation_pair(2 * j);
                        self.accumulate_fused_relation_linear_signed(
                            &mut rel,
                            w0_i64,
                            dw_i64,
                            2 * j,
                            p0,
                            p1,
                        );
                    }

                    let reduced_inner: [E; 2] = reduce_compact_virt_skip_linear(inner_virt);
                    let e_out = e_second[j_high];
                    virt[0] += e_out * reduced_inner[0];
                    virt[1] += e_out * reduced_inner[1];

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
                0..num_second,
                || ([E::zero(); 3], [E::SmallProduct::zero(); 6]),
                |(mut virt, mut rel), j_high| {
                    let mut inner_virt = [E::SmallProduct::zero(); 4];
                    let base = j_high * num_first;
                    for (j_low, &e_in) in e_first.iter().enumerate() {
                        let j = base + j_low;
                        if j >= live_pairs {
                            break;
                        }
                        let w0 = compact_witness.get(2 * j).map_or(0, i32::from);
                        let w1 = compact_witness.get(2 * j + 1).map_or(0, i32::from);
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

                        let (p0, p1) = relation_pair(2 * j);
                        self.accumulate_fused_relation_linear_signed(
                            &mut rel,
                            w0_i64,
                            dw_i64,
                            2 * j,
                            p0,
                            p1,
                        );
                    }

                    let reduced_inner: [E; 3] = reduce_compact_virt(inner_virt);
                    let e_out = e_second[j_high];
                    virt[0] += e_out * reduced_inner[0];
                    virt[1] += e_out * reduced_inner[1];
                    virt[2] += e_out * reduced_inner[2];

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

    pub(super) fn compute_round_compact_dense_terms(
        &self,
        compact_witness: PackedSignedDigitView<'_>,
        weights: &RelationWeightFactorization<E>,
    ) -> (NormRoundTerms<E>, [E; 3]) {
        let folding_coefficient_round = self.in_coefficient_round();
        let current_lane_width = self.current_lane_width();
        let current_lane_mask = (1usize << current_lane_width).wrapping_sub(1);
        let current_coefficient_width = self.current_coefficient_width();
        let current_coefficient_mask = (1usize << current_coefficient_width).wrapping_sub(1);
        let common_alpha_factor = weights.common_alpha_factor();
        let relation_lane_weights = weights.relation_lane_weights();
        let relation_pair = |left: usize| {
            let right = left + 1;
            if folding_coefficient_round {
                (
                    common_alpha_factor[left & current_coefficient_mask]
                        * relation_lane_weights[left >> current_coefficient_width],
                    common_alpha_factor[right & current_coefficient_mask]
                        * relation_lane_weights[right >> current_coefficient_width],
                )
            } else {
                (
                    common_alpha_factor[left >> current_lane_width]
                        * relation_lane_weights[left & current_lane_mask],
                    common_alpha_factor[right >> current_lane_width]
                        * relation_lane_weights[right & current_lane_mask],
                )
            }
        };
        self.compute_round_compact_dense_terms_with(
            compact_witness,
            compact_witness.len().div_ceil(2),
            relation_pair,
        )
    }

    pub(super) fn compute_round_compact_reduced_dense_terms(
        &self,
        compact_witness: PackedSignedDigitView<'_>,
        dense: &[E],
    ) -> (NormRoundTerms<E>, [E; 3]) {
        self.compute_round_compact_dense_terms_with(
            compact_witness,
            compact_witness.len().div_ceil(2),
            |left| (dense[left], dense[left + 1]),
        )
    }

    #[tracing::instrument(
        skip_all,
        name = "RelationRangeImageProver::compute_folded_dense_round_terms"
    )]
    fn compute_folded_dense_round_terms_with(
        &self,
        folded_witness: &[E],
        live_pairs: usize,
        relation_pair: impl Fn(usize) -> (E, E) + Sync,
    ) -> (NormRoundTerms<E>, [E; 3]) {
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let num_second = e_second.len();
        debug_assert!(live_pairs <= num_first * num_second);

        if self.can_skip_norm_linear_coeff() {
            let (virt_coeffs, rel_coeffs) = cfg_fold_reduce!(
                0..num_second,
                || ([E::zero(); 2], [E::zero(); 3]),
                |(mut virt, mut rel), j_high| {
                    let mut inner_virt = [E::zero(); 2];
                    let base = j_high * num_first;

                    for (j_low, &e_in) in e_first.iter().enumerate() {
                        let j = base + j_low;
                        if j >= live_pairs {
                            break;
                        }
                        let w0 = folded_witness.get(2 * j).copied().unwrap_or_else(E::zero);
                        let w1 = folded_witness
                            .get(2 * j + 1)
                            .copied()
                            .unwrap_or_else(E::zero);
                        let dw = w1 - w0;

                        inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                        inner_virt[1] += e_in * (dw * dw);

                        let (p0, p1) = relation_pair(2 * j);
                        self.accumulate_fused_relation_linear(&mut rel, w0, dw, 2 * j, p0, p1);
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
                0..num_second,
                || ([E::zero(); 3], [E::zero(); 3]),
                |(mut virt, mut rel), j_high| {
                    let mut inner_virt = [E::zero(); 3];
                    let base = j_high * num_first;

                    for (j_low, &e_in) in e_first.iter().enumerate() {
                        let j = base + j_low;
                        if j >= live_pairs {
                            break;
                        }
                        let w0 = folded_witness.get(2 * j).copied().unwrap_or_else(E::zero);
                        let w1 = folded_witness
                            .get(2 * j + 1)
                            .copied()
                            .unwrap_or_else(E::zero);
                        let dw = w1 - w0;
                        let two_w0_plus_one = w0 + w0 + E::one();

                        inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                        inner_virt[1] += e_in * (dw * two_w0_plus_one);
                        inner_virt[2] += e_in * (dw * dw);

                        let (p0, p1) = relation_pair(2 * j);
                        self.accumulate_fused_relation_linear(&mut rel, w0, dw, 2 * j, p0, p1);
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

    pub(super) fn compute_folded_dense_round_terms(
        &self,
        folded_witness: &[E],
        weights: &RelationWeightFactorization<E>,
    ) -> (NormRoundTerms<E>, [E; 3]) {
        let folding_coefficient_round = self.in_coefficient_round();
        let current_lane_width = self.current_lane_width();
        let current_lane_mask = (1usize << current_lane_width).wrapping_sub(1);
        let current_coefficient_width = self.current_coefficient_width();
        let current_coefficient_mask = (1usize << current_coefficient_width).wrapping_sub(1);
        let common_alpha_factor = weights.common_alpha_factor();
        let relation_lane_weights = weights.relation_lane_weights();
        let relation_pair = |left: usize| {
            let right = left + 1;
            if folding_coefficient_round {
                (
                    common_alpha_factor[left & current_coefficient_mask]
                        * relation_lane_weights[left >> current_coefficient_width],
                    common_alpha_factor[right & current_coefficient_mask]
                        * relation_lane_weights[right >> current_coefficient_width],
                )
            } else {
                (
                    common_alpha_factor[left >> current_lane_width]
                        * relation_lane_weights[left & current_lane_mask],
                    common_alpha_factor[right >> current_lane_width]
                        * relation_lane_weights[right & current_lane_mask],
                )
            }
        };
        self.compute_folded_dense_round_terms_with(
            folded_witness,
            folded_witness.len().div_ceil(2),
            relation_pair,
        )
    }

    pub(super) fn compute_folded_reduced_dense_round_terms(
        &self,
        folded_witness: &[E],
        dense: &[E],
    ) -> (NormRoundTerms<E>, [E; 3]) {
        self.compute_folded_dense_round_terms_with(
            folded_witness,
            folded_witness.len().div_ceil(2),
            |left| (dense[left], dense[left + 1]),
        )
    }

    #[cfg(test)]
    pub(super) fn compute_round_compact_dense_polys(
        &self,
        compact_witness: PackedSignedDigitView<'_>,
    ) -> (UniPoly<E>, UniPoly<E>) {
        let weights = self
            .quotient_weights()
            .expect("factored dense test helper requires quotient weights");
        let (virt_q_coeffs, rel_coeffs) =
            self.compute_round_compact_dense_terms(compact_witness, weights);
        self.polys_from_terms(virt_q_coeffs, rel_coeffs)
    }
}
