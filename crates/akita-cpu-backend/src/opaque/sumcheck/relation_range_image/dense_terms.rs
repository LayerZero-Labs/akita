use super::*;
use crate::opaque::sumcheck::par_fold_by_grain;

impl<E: Field + Ring + Unreduced> RelationRangeImageProver<E> {
    /// Independently recompute the dense round constant before claim recovery.
    #[cfg(debug_assertions)]
    pub(super) fn debug_dense_round_at_zero(&self) -> Option<E> {
        let Phase::Coefficient {
            witness, relation, ..
        } = self.phase.as_ref()?
        else {
            return None;
        };
        let witness_at = |index| match witness {
            WitnessState::CompactPrefix(witness) => witness
                .get(index)
                .map_or_else(E::zero, |w| E::from_i64(i64::from(w))),
            WitnessState::FoldedSuffix(witness) => {
                witness.get(index).copied().unwrap_or_else(E::zero)
            }
        };
        let witness_len = match witness {
            WitnessState::CompactPrefix(witness) => witness.len(),
            WitnessState::FoldedSuffix(witness) => witness.len(),
        };
        let mut at_zero = E::zero();
        for left in (0..witness_len).step_by(2) {
            let p0 = match relation {
                CoefficientRelation::Factored(weights) => {
                    self.factored_relation_pair(weights)(left).0
                }
                CoefficientRelation::ReducedDense(weights) => weights.evaluations()[left],
            };
            let t0 = self.linear_terms.pair_from_flat_index(left).0;
            at_zero += witness_at(left) * (p0 + t0);
        }
        at_zero += self.prev_norm_poly.as_ref()?.evaluate(E::zero());
        if let Some(additional) = &self.additional_relation_terms {
            at_zero += additional.debug_round_at_zero(witness_at);
        }
        Some(at_zero)
    }

    #[tracing::instrument(
        skip_all,
        name = "RelationRangeImageProver::compute_round_compact_dense_terms"
    )]
    fn compute_round_compact_dense_terms_with(
        &self,
        compact_witness: PackedSignedDigitView<'_>,
        live_pairs: usize,
        relation_pair: impl Fn(usize) -> (E, E) + Sync,
    ) -> (NormRoundTerms<E>, RoundMessage<E>) {
        if self.can_skip_norm_linear_coeff() {
            self.compute_round_compact_dense_terms_with_skip_linear::<true>(
                compact_witness,
                live_pairs,
                relation_pair,
            )
        } else {
            self.compute_round_compact_dense_terms_with_skip_linear::<false>(
                compact_witness,
                live_pairs,
                relation_pair,
            )
        }
    }

    fn compute_round_compact_dense_terms_with_skip_linear<const SKIP_LINEAR: bool>(
        &self,
        compact_witness: PackedSignedDigitView<'_>,
        live_pairs: usize,
        relation_pair: impl Fn(usize) -> (E, E) + Sync,
    ) -> (NormRoundTerms<E>, RoundMessage<E>) {
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let num_second = e_second.len();
        debug_assert!(live_pairs <= num_first * num_second);

        let live_rows = live_pairs.div_ceil(num_first);
        let identity = || {
            (
                FieldNorm::<E, SKIP_LINEAR>::zero(),
                [E::SmallProduct::zero(); 4],
            )
        };
        let fold = |(mut virt, mut rel): (FieldNorm<E, SKIP_LINEAR>, CompactRelAccum<E>),
                    j_high: usize| {
            let mut inner_virt = CompactNorm::<E, SKIP_LINEAR>::zero();
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

                inner_virt.add(w0_i64, dw_i64, e_in);

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

            let reduced_inner = inner_virt.reduce();
            let e_out = e_second[j_high];
            virt.scaled_add(e_out, reduced_inner);

            (virt, rel)
        };
        let (virt_coeffs, rel_accum) = par_fold_by_grain(
            cfg_into_iter!(0..live_rows),
            num_first,
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

        (virt_coeffs.into_terms(), reduce_compact_rel(rel_accum))
    }

    /// `(p(left), p(left + 1))` for the factored relation weight
    /// `p = alpha(coefficient) * lw(lane)` in a coefficient round.
    fn factored_relation_pair<'a>(
        &self,
        weights: &'a RelationWeightFactorization<E>,
    ) -> impl Fn(usize) -> (E, E) + Sync + 'a {
        debug_assert!(self.in_coefficient_round());
        let coefficient_width = self.current_coefficient_width();
        let coefficient_mask = (1usize << coefficient_width) - 1;
        let common_alpha_factor = weights.common_alpha_factor();
        let relation_lane_weights = weights.relation_lane_weights();
        let weight = move |index: usize| {
            common_alpha_factor[index & coefficient_mask]
                * relation_lane_weights[index >> coefficient_width]
        };
        move |left| (weight(left), weight(left + 1))
    }

    pub(super) fn compute_round_compact_dense_terms(
        &self,
        compact_witness: PackedSignedDigitView<'_>,
        weights: &RelationWeightFactorization<E>,
    ) -> (NormRoundTerms<E>, RoundMessage<E>) {
        let relation_pair = self.factored_relation_pair(weights);
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
    ) -> (NormRoundTerms<E>, RoundMessage<E>) {
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
    ) -> (NormRoundTerms<E>, RoundMessage<E>) {
        if self.can_skip_norm_linear_coeff() {
            self.compute_folded_dense_round_terms_with_skip_linear::<true>(
                folded_witness,
                live_pairs,
                relation_pair,
            )
        } else {
            self.compute_folded_dense_round_terms_with_skip_linear::<false>(
                folded_witness,
                live_pairs,
                relation_pair,
            )
        }
    }

    fn compute_folded_dense_round_terms_with_skip_linear<const SKIP_LINEAR: bool>(
        &self,
        folded_witness: &[E],
        live_pairs: usize,
        relation_pair: impl Fn(usize) -> (E, E) + Sync,
    ) -> (NormRoundTerms<E>, RoundMessage<E>) {
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let num_second = e_second.len();
        debug_assert!(live_pairs <= num_first * num_second);

        let live_rows = live_pairs.div_ceil(num_first);
        let identity = || (FieldNorm::<E, SKIP_LINEAR>::zero(), RoundMessage::zero());
        let fold = |(mut virt, mut rel): (FieldNorm<E, SKIP_LINEAR>, RoundMessage<E>),
                    j_high: usize| {
            let mut inner_virt = ProductNorm::<E, SKIP_LINEAR>::zero();
            let mut inner_rel = RelationPairAccumulator::<E>::zero();
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

                inner_virt.add(w0, dw, e_in);

                let (p0, p1) = relation_pair(2 * j);
                let (t0, t1) = self.linear_terms.pair_from_flat_index(2 * j);
                let q0 = p0 + t0;
                let q1 = p1 + t1;
                inner_rel.add_pair(w1, dw, q0, q1);
            }

            virt.scaled_add(e_second[j_high], inner_virt.reduce());
            rel.add_assign(inner_rel.finish());

            (virt, rel)
        };
        let (virt_coeffs, rel_coeffs) = par_fold_by_grain(
            cfg_into_iter!(0..live_rows),
            num_first,
            identity,
            fold,
            |(mut va, mut ra), (vb, rb)| {
                va.merge(vb);
                ra.add_assign(rb);
                (va, ra)
            },
        );
        (virt_coeffs.into_terms(), rel_coeffs)
    }

    pub(super) fn compute_folded_dense_round_terms(
        &self,
        folded_witness: &[E],
        weights: &RelationWeightFactorization<E>,
    ) -> (NormRoundTerms<E>, RoundMessage<E>) {
        let relation_pair = self.factored_relation_pair(weights);
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
    ) -> (NormRoundTerms<E>, RoundMessage<E>) {
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
    ) -> (UnivariatePoly<E>, UnivariatePoly<E>) {
        let weights = self
            .quotient_weights()
            .expect("factored dense test helper requires quotient weights");
        let (virt_terms, relation_message) =
            self.compute_round_compact_dense_terms(compact_witness, weights);
        let relation_pair = self.factored_relation_pair(weights);
        let mut relation_claim = E::zero();
        for left in (0..compact_witness.len()).step_by(2) {
            let (p0, p1) = relation_pair(left);
            let (t0, t1) = self.linear_terms.pair_from_flat_index(left);
            let w0 = compact_witness.get(left).map_or(0, i8::from);
            let w1 = compact_witness.get(left + 1).map_or(0, i8::from);
            relation_claim +=
                E::from_i64(i64::from(w0)) * (p0 + t0) + E::from_i64(i64::from(w1)) * (p1 + t1);
        }
        (
            self.norm_poly_from_terms(virt_terms),
            relation_message.into_polynomial(relation_claim),
        )
    }
}
