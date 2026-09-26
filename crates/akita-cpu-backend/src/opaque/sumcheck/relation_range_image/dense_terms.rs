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
    ) -> (NormRoundTerms<E>, [E; 3]) {
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let num_second = e_second.len();
        debug_assert!(live_pairs <= num_first * num_second);

        let (virt_coeffs, rel_accum) = cfg_fold_reduce!(
            0..num_second,
            || (
                FieldNorm::<E, SKIP_LINEAR>::zero(),
                [E::SmallProduct::zero(); 6]
            ),
            |(mut virt, mut rel), j_high| {
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
            },
            |(mut va, mut ra), (vb, rb)| {
                va.merge(vb);
                add_assign_all(&mut ra, &rb);
                (va, ra)
            }
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
    ) -> (NormRoundTerms<E>, [E; 3]) {
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
    ) -> (NormRoundTerms<E>, [E; 3]) {
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let num_second = e_second.len();
        debug_assert!(live_pairs <= num_first * num_second);

        let (virt_coeffs, rel_coeffs) = cfg_fold_reduce!(
            0..num_second,
            || (FieldNorm::<E, SKIP_LINEAR>::zero(), [E::zero(); 3]),
            |(mut virt, mut rel), j_high| {
                let mut inner_virt = FieldNorm::<E, SKIP_LINEAR>::zero();
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
                    self.accumulate_fused_relation_linear(&mut rel, w0, dw, 2 * j, p0, p1);
                }

                virt.scaled_add(e_second[j_high], inner_virt.totals());

                (virt, rel)
            },
            |(mut va, mut ra), (vb, rb)| {
                va.merge(vb);
                add_assign_all(&mut ra, &rb);
                (va, ra)
            }
        );
        (virt_coeffs.into_terms(), rel_coeffs)
    }

    pub(super) fn compute_folded_dense_round_terms(
        &self,
        folded_witness: &[E],
        weights: &RelationWeightFactorization<E>,
    ) -> (NormRoundTerms<E>, [E; 3]) {
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
    ) -> (UnivariatePoly<E>, UnivariatePoly<E>) {
        let weights = self
            .quotient_weights()
            .expect("factored dense test helper requires quotient weights");
        let (virt_q_coeffs, rel_coeffs) =
            self.compute_round_compact_dense_terms(compact_witness, weights);
        self.polys_from_terms(virt_q_coeffs, rel_coeffs)
    }
}
