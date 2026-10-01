use super::*;
use crate::opaque::sumcheck::par_fold_by_grain;

impl<E: Field + Ring + Unreduced> RelationRoundState<E> {
    #[tracing::instrument(
        skip_all,
        name = "RelationRangeImageProver::compute_compact_partial_lane_coefficient_round_terms"
    )]
    pub(super) fn compute_compact_partial_lane_coefficient_round_terms(
        &self,
        compact_witness: PackedSignedDigitView<'_>,
        weights: &RelationWeightFactorization<E>,
    ) -> (NormRoundTerms<'_, E>, RoundMessage<E>) {
        debug_assert!(self.in_coefficient_round());
        debug_assert_eq!(
            compact_witness.len(),
            self.live_lane_count * weights.common_alpha_factor().len()
        );

        let recovery = self.split_eq.prepare_linear_q_recovery();
        let (norm, relation) = if recovery.is_some() {
            self.compute_compact_partial_lane_coefficient_round_terms_skip_linear::<true, false>(
                compact_witness,
                weights,
            )
        } else {
            self.compute_compact_partial_lane_coefficient_round_terms_skip_linear::<false, false>(
                compact_witness,
                weights,
            )
        };
        (NormRoundTerms::from_totals(norm, recovery), relation)
    }

    pub(super) fn compute_compact_partial_lane_coefficient_round_norm_terms(
        &self,
        compact_witness: PackedSignedDigitView<'_>,
        weights: &RelationWeightFactorization<E>,
    ) -> NormRoundTerms<'_, E> {
        let recovery = self.split_eq.prepare_linear_q_recovery();
        let (norm, _) = if recovery.is_some() {
            self.compute_compact_partial_lane_coefficient_round_terms_skip_linear::<true, true>(
                compact_witness,
                weights,
            )
        } else {
            self.compute_compact_partial_lane_coefficient_round_terms_skip_linear::<false, true>(
                compact_witness,
                weights,
            )
        };
        NormRoundTerms::from_totals(norm, recovery)
    }

    fn compute_compact_partial_lane_coefficient_round_terms_skip_linear<
        const SKIP_LINEAR: bool,
        const SKIP_RELATION: bool,
    >(
        &self,
        compact_witness: PackedSignedDigitView<'_>,
        weights: &RelationWeightFactorization<E>,
    ) -> ([E; 3], RoundMessage<E>) {
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros() as usize;
        let current_coefficient_half = 1usize << (self.current_coefficient_width() - 1);
        let block_size = num_first.min(current_coefficient_half);
        let common_alpha_factor = weights.common_alpha_factor();
        let relation_lane_weights = weights.relation_lane_weights();
        debug_assert_eq!(relation_lane_weights.len(), 1usize << self.lane_bits);

        let identity = || {
            (
                FieldNorm::<E, SKIP_LINEAR>::zero(),
                [E::SmallProduct::zero(); 4],
            )
        };
        let fold = |(mut virt, mut rel): (FieldNorm<E, SKIP_LINEAR>, CompactRelAccum<E>),
                    lane: usize| {
            let lane_start = lane * common_alpha_factor.len();
            let lane_weight = if SKIP_RELATION {
                E::zero()
            } else {
                relation_lane_weights[lane]
            };
            let linear_lane = if SKIP_RELATION {
                PreparedLinearLane::zero()
            } else {
                self.linear_terms.resolve_lane(lane)
            };
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
                let mut inner_virt = CompactNorm::<E, SKIP_LINEAR>::zero();

                for coefficient_pair in blk..blk_end {
                    let j_low = (equality_address_base + coefficient_pair) & (num_first - 1);
                    let e_in = e_first[j_low];
                    let left = 2 * coefficient_pair;
                    let w0 = i32::from(compact_witness.at(lane_start + left));
                    let w1 = i32::from(compact_witness.at(lane_start + left + 1));
                    let dw = w1 - w0;
                    let w0_i64 = w0 as i64;
                    let dw_i64 = dw as i64;

                    inner_virt.add(w0_i64, dw_i64, e_in);

                    if !SKIP_RELATION {
                        let p0 = common_alpha_factor[left] * lane_weight;
                        let p1 = common_alpha_factor[left + 1] * lane_weight;
                        let (t0, t1) = linear_lane.pair(left);
                        accumulate_relation_eval_coeffs_signed(
                            &mut rel,
                            w0_i64,
                            dw_i64,
                            p0 + t0,
                            p1 + t1,
                        );
                    }
                }

                virt.scaled_add(e_second[j_high], inner_virt.reduce());
                blk = blk_end;
            }

            (virt, rel)
        };
        let (virt_coeffs, rel_accum) = par_fold_by_grain(
            cfg_into_iter!(0..self.live_lane_count),
            current_coefficient_half,
            identity,
            fold,
            |(mut va, mut ra), (vb, rb)| {
                va.merge(vb);
                for (left, right) in ra.iter_mut().zip(rb) {
                    *left += right;
                }
                (va, ra)
            },
        );

        (virt_coeffs.totals(), reduce_compact_rel(rel_accum))
    }

    #[tracing::instrument(
        skip_all,
        name = "RelationRangeImageProver::compute_folded_partial_lane_coefficient_round_terms"
    )]
    pub(super) fn compute_folded_partial_lane_coefficient_round_terms(
        &self,
        folded_witness: &[E],
        weights: &RelationWeightFactorization<E>,
    ) -> (NormRoundTerms<'_, E>, RoundMessage<E>) {
        debug_assert!(self.in_coefficient_round());
        debug_assert_eq!(
            folded_witness.len(),
            self.live_lane_count * weights.common_alpha_factor().len()
        );

        let recovery = self.split_eq.prepare_linear_q_recovery();
        let (norm, relation) = if recovery.is_some() {
            self.compute_folded_partial_lane_coefficient_round_terms_skip_linear::<true, false>(
                folded_witness,
                weights,
            )
        } else {
            self.compute_folded_partial_lane_coefficient_round_terms_skip_linear::<false, false>(
                folded_witness,
                weights,
            )
        };
        (NormRoundTerms::from_totals(norm, recovery), relation)
    }

    pub(super) fn compute_folded_partial_lane_coefficient_round_norm_terms(
        &self,
        folded_witness: &[E],
        weights: &RelationWeightFactorization<E>,
    ) -> NormRoundTerms<'_, E> {
        let recovery = self.split_eq.prepare_linear_q_recovery();
        let (norm, _) = if recovery.is_some() {
            self.compute_folded_partial_lane_coefficient_round_terms_skip_linear::<true, true>(
                folded_witness,
                weights,
            )
        } else {
            self.compute_folded_partial_lane_coefficient_round_terms_skip_linear::<false, true>(
                folded_witness,
                weights,
            )
        };
        NormRoundTerms::from_totals(norm, recovery)
    }

    fn compute_folded_partial_lane_coefficient_round_terms_skip_linear<
        const SKIP_LINEAR: bool,
        const SKIP_RELATION: bool,
    >(
        &self,
        folded_witness: &[E],
        weights: &RelationWeightFactorization<E>,
    ) -> ([E; 3], RoundMessage<E>) {
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros() as usize;
        let current_coefficient_half = 1usize << (self.current_coefficient_width() - 1);
        let block_size = num_first.min(current_coefficient_half);
        let common_alpha_factor = weights.common_alpha_factor();
        let relation_lane_weights = weights.relation_lane_weights();
        debug_assert_eq!(relation_lane_weights.len(), 1usize << self.lane_bits);

        let identity = || (FieldNorm::<E, SKIP_LINEAR>::zero(), RoundMessage::zero());
        let fold = |(mut virt, mut rel): (FieldNorm<E, SKIP_LINEAR>, RoundMessage<E>),
                    lane: usize| {
            let lane_start = lane * common_alpha_factor.len();
            let lane_values = &folded_witness[lane_start..lane_start + common_alpha_factor.len()];
            let equality_address_base = lane * current_coefficient_half;
            let mut lane_rel = RelationPairAccumulator::<E>::zero();
            let lane_weight = if SKIP_RELATION {
                E::zero()
            } else {
                relation_lane_weights[lane]
            };
            let linear_lane = if SKIP_RELATION {
                PreparedLinearLane::zero()
            } else {
                self.linear_terms.resolve_lane(lane)
            };
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
                let mut inner_virt = ProductNorm::<E, SKIP_LINEAR>::zero();

                for coefficient_pair in blk..blk_end {
                    let j_low = (equality_address_base + coefficient_pair) & (num_first - 1);
                    let e_in = e_first[j_low];
                    let left = 2 * coefficient_pair;
                    let w0 = lane_values[left];
                    let w1 = lane_values[left + 1];
                    let dw = w1 - w0;

                    inner_virt.add(w0, dw, e_in);

                    if !SKIP_RELATION {
                        let p0 = common_alpha_factor[left] * lane_weight;
                        let p1 = common_alpha_factor[left + 1] * lane_weight;
                        let (t0, t1) = linear_lane.pair(left);
                        let q0 = p0 + t0;
                        let q1 = p1 + t1;
                        lane_rel.add_pair(w1, dw, q0, q1);
                    }
                }

                virt.scaled_add(e_second[j_high], inner_virt.reduce());
                blk = blk_end;
            }

            if !SKIP_RELATION {
                rel.add_assign(lane_rel.finish());
            }
            (virt, rel)
        };
        let (virt_coeffs, rel_coeffs) = par_fold_by_grain(
            cfg_into_iter!(0..self.live_lane_count),
            current_coefficient_half,
            identity,
            fold,
            |(mut va, mut ra), (vb, rb)| {
                va.merge(vb);
                ra.add_assign(rb);
                (va, ra)
            },
        );
        (virt_coeffs.totals(), rel_coeffs)
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
        let fold_lane = |(lane, lane_out): (usize, &mut [E])| {
            let lane_start = lane * coeff_count;
            let lane_values = &folded_witness[lane_start..lane_start + coeff_count];
            for (coefficient_pair, dst) in lane_out.iter_mut().enumerate() {
                let left = 2 * coefficient_pair;
                let w0 = lane_values[left];
                let w1 = lane_values[left + 1];
                *dst = w0 + r * (w1 - w0);
            }
        };
        par_fold_by_grain(
            cfg_chunks_mut!(out, next_coeff_count).enumerate(),
            next_coeff_count,
            || (),
            |(), item| fold_lane(item),
            |(), ()| (),
        );

        out
    }
}
