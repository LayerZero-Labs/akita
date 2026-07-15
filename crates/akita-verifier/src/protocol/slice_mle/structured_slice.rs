#[cfg(test)]
use crate::protocol::ring_switch::PreparedChallengeEvals;
use crate::protocol::ring_switch::RelationMatrixEvaluator;
use akita_algebra::offset_eq::{eq_eval_at_index, MAX_COMPACT_STRIDE_TERMS};
#[cfg(test)]
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};

/// Number of carry buckets per outer index produced by
/// [`StructuredSliceMleEvaluator::compute_inner_sum`].
///
/// **Note:** This module is only tested and intended for the
/// `POSSIBLE_CARRIES = 2` case. Anything other than `2` would require the
/// outer-sum algebra to be reworked; do not change this constant.
#[cfg(test)]
pub(super) const POSSIBLE_CARRIES: usize = 2;

/// Inner-sum slot for the no-carry bucket (`carry = 0`).
#[cfg(test)]
pub(super) const CARRY0: usize = 0;

/// Inner-sum slot for the one-carry bucket (`carry = 1`).
#[cfg(test)]
pub(super) const CARRY1: usize = 1;

/// Peeled-block MLE evaluator for one structured slice of `M`. See
/// `book/src/how/verifying/matrix_evaluation.md` for the full derivation.
#[cfg(test)]
pub(crate) trait StructuredSliceMleEvaluator<F: FieldCore>: Sync {
    /// Number of outer-loop indices.
    fn num_outer_indices(&self) -> usize;

    /// Precomputed high-bit equality table, indexed *relative* to the slice's
    /// high offset: `table[k] == eq_high(offset_high + k)`.
    fn high_eq_table(&self) -> &[F];

    /// Compute the inner sum at `outer_index`: this evaluator's contribution
    /// to each carry bucket ([`CARRY0`], [`CARRY1`]) for that outer index.
    fn compute_inner_sum(&self, outer_index: usize) -> [F; POSSIBLE_CARRIES];

    /// Whether [`Self::evaluate`] should iterate the outer dimension in
    /// parallel when collecting carry terms.
    ///
    /// Default `false` (sequential). Override to `true` for evaluators with
    /// non-trivial per-outer-index work.
    #[inline]
    fn parallelize_outer(&self) -> bool {
        false
    }

    /// Combine the per-outer-index carry terms with the precomputed high-bit
    /// equality table:
    ///
    /// ```text
    /// Σ_q  carry_terms[q][CARRY0] · table[q]
    ///    + carry_terms[q][CARRY1] · table[q + 1]
    /// ```
    ///
    /// **Note:** Both this default impl and the algebra it implements are
    /// only tested and intended for [`POSSIBLE_CARRIES`] = 2. The two carry
    /// buckets [`CARRY0`] and [`CARRY1`] are the only ones that arise from
    /// the peeled-block split.
    #[inline]
    fn compute_outer_sum(&self, carry_terms: &[[F; POSSIBLE_CARRIES]]) -> F {
        let table = self.high_eq_table();
        carry_terms
            .iter()
            .enumerate()
            .fold(F::zero(), |acc, (q, terms)| {
                let acc = if terms[CARRY0].is_zero() {
                    acc
                } else {
                    acc + terms[CARRY0] * table[q]
                };
                if terms[CARRY1].is_zero() {
                    acc
                } else {
                    acc + terms[CARRY1] * table[q + 1]
                }
            })
    }

    /// Evaluate this slice's multilinear extension at the slice's
    /// randomness.
    #[inline]
    fn evaluate(&self) -> F {
        let n = self.num_outer_indices();
        let carry_terms: Vec<[F; POSSIBLE_CARRIES]> = if self.parallelize_outer() {
            cfg_into_iter!(0..n)
                .map(|outer_index| self.compute_inner_sum(outer_index))
                .collect()
        } else {
            (0..n)
                .map(|outer_index| self.compute_inner_sum(outer_index))
                .collect()
        };
        self.compute_outer_sum(&carry_terms)
    }
}

/// E-hat segment slice evaluator.
#[cfg(test)]
pub(crate) struct EStructuredSlicesEvaluator<'a, F, E> {
    /// Gadget vector for the digit decomposition of `e`. Length =
    /// `num_digits`.
    pub gadget_vector: &'a [F],
    /// Per-claim carry summary of `c_alpha`. Length = `num_claims`.
    pub challenge_block_summaries: &'a [[E; 2]],
    /// `tau1` equality weight for the consistency-challenge row of `M`.
    pub challenge_weight: E,
    /// Precomputed high-eq table relative to the slice's high offset.
    pub high_eq_table: &'a [E],
}

#[cfg(test)]
impl<F, E> StructuredSliceMleEvaluator<E> for EStructuredSlicesEvaluator<'_, F, E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    #[inline]
    fn num_outer_indices(&self) -> usize {
        self.challenge_block_summaries.len() * self.gadget_vector.len()
    }

    #[inline]
    fn high_eq_table(&self) -> &[E] {
        self.high_eq_table
    }

    #[inline]
    fn compute_inner_sum(&self, outer_index: usize) -> [E; POSSIBLE_CARRIES] {
        let num_claims = self.challenge_block_summaries.len();
        let digit = outer_index / num_claims;
        let claim_idx = outer_index % num_claims;

        let [aggregated_challenge_carry0, aggregated_challenge_carry1] =
            self.challenge_block_summaries[claim_idx];

        [
            (self.challenge_weight * aggregated_challenge_carry0)
                .mul_base(self.gadget_vector[digit]),
            (self.challenge_weight * aggregated_challenge_carry1)
                .mul_base(self.gadget_vector[digit]),
        ]
    }
}

/// T-segment slice evaluator.
#[cfg(test)]
pub(crate) struct TStructuredSlicesEvaluator<'a, F, E> {
    /// Gadget vector for the digit decomposition of `w`. Length =
    /// `num_digits`.
    pub gadget_vector: &'a [F],
    /// Per-claim carry summary of `c_alpha`. Length = `num_claims`.
    pub challenge_block_summaries: &'a [[E; 2]],
    /// `tau1` equality weight for each `A`-row of `M`. Length =
    /// number of `A` rows.
    pub a_row_weights: &'a [E],
    /// Precomputed high-eq table relative to the slice's high offset.
    pub high_eq_table: &'a [E],
}

#[cfg(test)]
impl<F, E> StructuredSliceMleEvaluator<E> for TStructuredSlicesEvaluator<'_, F, E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    #[inline]
    fn num_outer_indices(&self) -> usize {
        self.challenge_block_summaries.len() * self.gadget_vector.len() * self.a_row_weights.len()
    }

    #[inline]
    fn high_eq_table(&self) -> &[E] {
        self.high_eq_table
    }

    #[inline]
    fn compute_inner_sum(&self, outer_index: usize) -> [E; POSSIBLE_CARRIES] {
        let num_claims = self.challenge_block_summaries.len();
        let num_digits = self.gadget_vector.len();
        let claim_idx = outer_index % num_claims;
        let compound = outer_index / num_claims;
        let digit = compound % num_digits;
        let a_row_idx = compound / num_digits;
        let [aggregated_challenge_carry0, aggregated_challenge_carry1] =
            self.challenge_block_summaries[claim_idx];
        [
            self.a_row_weights[a_row_idx].mul_base(self.gadget_vector[digit])
                * aggregated_challenge_carry0,
            self.a_row_weights[a_row_idx].mul_base(self.gadget_vector[digit])
                * aggregated_challenge_carry1,
        ]
    }
}

/// Compute the `r`-tail contribution.
///
/// Physical `r` addresses are mapped into the canonical opening domain before
/// applying their equality weights.
pub(crate) fn compute_r_contribution<F, E>(
    prepared: &RelationMatrixEvaluator<E>,
    full_vec_randomness: &[E],
    offset_r: usize,
    denom: E,
    r_gadget: &[F],
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    let levels = r_gadget.len();
    let rows = prepared.setup_contribution_inputs.rows;
    let terms = rows.checked_mul(levels).ok_or(AkitaError::InvalidProof)?;
    if terms > MAX_COMPACT_STRIDE_TERMS {
        return Err(AkitaError::InvalidSize {
            expected: MAX_COMPACT_STRIDE_TERMS,
            actual: terms,
        });
    }
    let mut contribution = E::zero();
    for row_idx in 0..rows {
        for (level_idx, &gadget) in r_gadget.iter().enumerate() {
            let physical_index = offset_r
                .checked_add(row_idx * levels + level_idx)
                .ok_or(AkitaError::InvalidProof)?;
            let opening_index = akita_types::checked_opening_source_index(
                prepared.setup_contribution_layout.opening_source_len(),
                physical_index,
            )?;
            contribution -= eq_eval_at_index(full_vec_randomness, opening_index)
                * prepared.setup_contribution_inputs.eq_tau1[row_idx]
                * E::lift_base(gadget)
                * denom;
        }
    }
    Ok(contribution)
}

#[cfg(test)]
mod tests {
    use super::*;

    use akita_algebra::eq_poly::EqPolynomial;
    use akita_algebra::offset_eq::{eq_eval_at_index, summarize_pow2_block_carries};
    use akita_algebra::ring::scalar_powers;
    use akita_algebra::CyclotomicRing;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Prime128OffsetA7F7;
    use akita_types::{
        gadget_row_scalars, r_decomp_levels, LevelParams, OpeningClaimsLayout,
        RelationMatrixRowLayout, RingMultiplierOpeningPoint, RingOpeningPoint,
        RingRelationInstance, SetupContributionPlan, SetupContributionPlanInputs,
        SisModulusProfileId, WitnessLayout,
    };

    use crate::protocol::ring_switch::{
        build_setup_contribution_layout, RelationMatrixGroupEvaluator,
    };

    type F = Prime128OffsetA7F7;
    const D: usize = 32;

    struct StructuredFixture {
        prepared: RelationMatrixEvaluator<F>,
        full_vec_randomness: Vec<F>,
        offset_e: usize,
        offset_t: usize,
        offset_r: usize,
        g1_open: Vec<F>,
        r_gadget: Vec<F>,
    }

    fn f(value: u128) -> F {
        F::from_canonical_u128_reduced(value)
    }

    fn fixture_lp() -> LevelParams {
        LevelParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            D,
            5,
            2,
            2,
            2,
            SparseChallengeConfig::pm1_only(1),
        )
        .with_decomp(512, 512 * 8, 1, 26)
        .expect("structured slice fixture lp")
    }

    fn ring_relation_segment_layout_for_opening_shape(
        lp: &LevelParams,
        relation_matrix_row_layout: RelationMatrixRowLayout,
        num_polys: usize,
    ) -> Result<WitnessLayout, AkitaError> {
        let opening_batch = OpeningClaimsLayout::new(32, num_polys)?;
        let opening_point = RingOpeningPoint {
            position_weights: vec![F::zero(); lp.fold_position_count],
            fold_weights: vec![F::zero(); lp.live_fold_count],
        };
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&opening_point);
        let num_claims = opening_batch.num_total_polynomials();
        let challenges = akita_challenges::Challenges::Sparse {
            challenges: Vec::new(),
            live_folds_per_claim: lp.live_fold_count,
            num_claims,
        };
        let v = vec![CyclotomicRing::<F, D>::zero(); lp.d_key.row_len()];
        let commitment_rows = vec![CyclotomicRing::<F, D>::zero(); lp.b_key.row_len()];
        let row_coefficient_rings = vec![CyclotomicRing::<F, D>::zero(); num_claims];
        let relation_rhs_layout =
            akita_types::relation_rhs_layout_for(lp, &opening_batch, relation_matrix_row_layout)?;
        let relation_rhs = akita_types::assemble_relation_rhs::<F>(
            lp.role_dims(),
            &relation_rhs_layout,
            &akita_types::RingVec::from_ring_elems(&v),
            &akita_types::RingVec::from_ring_elems(&commitment_rows),
        )?;
        let instance = RingRelationInstance::<F>::new(
            relation_matrix_row_layout,
            vec![challenges],
            vec![opening_point],
            vec![ring_multiplier_point],
            opening_batch,
            vec![F::zero(); num_claims],
            akita_types::RingVec::from_ring_elems(&row_coefficient_rings),
            relation_rhs,
            akita_types::RingVec::from_ring_elems(&v),
            lp.role_dims(),
        )?;
        instance.segment_layout(lp, None)
    }

    fn fixture() -> StructuredFixture {
        // `nv = 32` in `fp128_d32_onehot.rs` includes repeated compact
        // recursive levels with this real D=32 shape.
        let live_fold_count = 8usize;
        let fold_position_count = 512usize;
        let log_basis = 5u32;
        let depth_open = 26usize;
        let depth_commit = 1usize;
        let depth_fold = 4usize;
        let n_a = 2usize;
        let n_b = 2usize;
        let n_d = 2usize;
        let num_claims = 3usize;
        let num_points = 1usize;
        let total_blocks = live_fold_count * num_claims;
        let rows = 1 + n_a + n_b * num_points + n_d;
        let inner_width = fold_position_count * depth_commit;

        let levels = r_decomp_levels::<F>(log_basis);
        let lp = fixture_lp();
        let layout = ring_relation_segment_layout_for_opening_shape(
            &lp,
            RelationMatrixRowLayout::WithDBlock,
            num_claims,
        )
        .expect("witness segment layout");
        let unit0 = &layout.units()[0];
        let offset_e = unit0.e_range().start;
        let offset_t = unit0.t_range().start;
        let offset_r = layout.r_range().start;
        let total_len = offset_r + rows * levels;
        let bits = total_len.next_power_of_two().trailing_zeros() as usize;

        let opening_a_evals = (0..fold_position_count)
            .map(|idx| f(1_000 + idx as u128))
            .collect();
        let setup_contribution_inputs = SetupContributionPlanInputs {
            relation_matrix_row_layout: RelationMatrixRowLayout::WithDBlock,
            rows,
            n_a,
            n_b,
            n_d,
            num_groups: 1,
            num_polys_per_group: vec![num_claims],
            num_t_vectors: num_claims,
            num_claims,
            live_fold_count,
            fold_position_count,
            depth_open,
            depth_commit,
            depth_fold,
            inner_width,
            eq_tau1: (0..rows.next_power_of_two())
                .map(|idx| f(4_000 + idx as u128))
                .collect(),
        };
        let groups = vec![RelationMatrixGroupEvaluator {
            c_alphas: PreparedChallengeEvals::Flat(
                (0..total_blocks)
                    .map(|idx| f(3_000 + idx as u128))
                    .collect(),
            ),
            opening_a_evals,
            group_id: 0,
            num_claims,
            live_fold_count,
            fold_position_count,
            depth_open,
            depth_commit,
            depth_fold,
            log_basis,
            n_a,
            n_b,
            t_cols_per_vector: n_a * depth_open * live_fold_count,
            a_row_start: 1,
            b_row_start: 1 + n_a,
        }];
        let opening_source_len = layout.total_len();
        let layout = std::sync::Arc::new(layout);
        let setup_contribution_layout =
            build_setup_contribution_layout(layout.clone(), opening_source_len, &groups).unwrap();
        let setup_contribution_static = SetupContributionPlan::prepare_static(
            &setup_contribution_inputs,
            &setup_contribution_layout,
        )
        .unwrap();
        let prepared = RelationMatrixEvaluator {
            role_dims: lp.role_dims(),
            groups,
            log_basis,
            setup_contribution_layout,
            setup_contribution_inputs,
            setup_contribution_static,
            flat_context: None,
        };
        let full_vec_randomness = (0..bits).map(|idx| f(6_000 + idx as u128)).collect();
        let g1_open = gadget_row_scalars::<F>(depth_open, log_basis);
        let r_gadget = gadget_row_scalars::<F>(levels, log_basis);

        StructuredFixture {
            prepared,
            full_vec_randomness,
            offset_e,
            offset_t,
            offset_r,
            g1_open,
            r_gadget,
        }
    }

    fn eq_evals(_total_len: usize, full_vec_randomness: &[F]) -> Vec<F> {
        (0..(1usize << full_vec_randomness.len()))
            .map(|idx| eq_eval_at_index(full_vec_randomness, idx))
            .collect()
    }

    #[test]
    fn e_structured_matches_materialized_range_inner_product() {
        let fx = fixture();
        let p = &fx.prepared;
        let g = &p.groups[0];
        let total_blocks = g.live_fold_count * g.num_claims;
        let e_len = g.depth_open * total_blocks;
        let eq = eq_evals(fx.offset_e + e_len, &fx.full_vec_randomness);
        let offset_low_bits = g.live_fold_count.trailing_zeros() as usize;
        let eq_low = EqPolynomial::evals(&fx.full_vec_randomness[..offset_low_bits]).unwrap();
        let block_offset_low = fx.offset_e & (g.live_fold_count - 1);

        let challenge_block_summaries: Vec<[F; 2]> = (0..g.num_claims)
            .map(|claim_idx| {
                let start = claim_idx * g.live_fold_count;
                summarize_pow2_block_carries(
                    &eq_low,
                    block_offset_low,
                    &g.c_alphas.as_flat().unwrap()[start..(start + g.live_fold_count)],
                )
            })
            .collect::<Result<_, _>>()
            .unwrap();
        let c_alphas = g.c_alphas.as_flat().unwrap();
        let e_high = &fx.full_vec_randomness[offset_low_bits..];
        let e_offset_high = fx.offset_e >> offset_low_bits;
        let e_outer = challenge_block_summaries.len() * fx.g1_open.len();
        let eq_hi_e: Vec<F> = (0..=e_outer)
            .map(|k| eq_eval_at_index(e_high, e_offset_high + k))
            .collect();
        let got = EStructuredSlicesEvaluator {
            gadget_vector: &fx.g1_open,
            challenge_block_summaries: &challenge_block_summaries,
            challenge_weight: p.setup_contribution_inputs.eq_tau1[0],
            high_eq_table: &eq_hi_e,
        }
        .evaluate();

        let mut expected = F::zero();
        for x in 0..e_len {
            let dig = x / total_blocks;
            let blk = x % total_blocks;
            let entry = p.setup_contribution_inputs.eq_tau1[0] * c_alphas[blk] * fx.g1_open[dig];
            expected += entry * eq[fx.offset_e + x];
        }
        assert_eq!(got, expected);
    }

    #[test]
    fn t_structured_matches_materialized_range_inner_product() {
        let fx = fixture();
        let p = &fx.prepared;
        let g = &p.groups[0];
        let total_blocks = g.live_fold_count * g.num_claims;
        let t_len = g.depth_open * g.n_a * total_blocks;
        let eq = eq_evals(fx.offset_t + t_len, &fx.full_vec_randomness);
        let offset_low_bits = g.live_fold_count.trailing_zeros() as usize;
        let eq_low = EqPolynomial::evals(&fx.full_vec_randomness[..offset_low_bits]).unwrap();
        let block_offset_low = fx.offset_t & (g.live_fold_count - 1);

        let challenge_block_summaries: Vec<[F; 2]> = (0..g.num_claims)
            .map(|claim_idx| {
                let start = claim_idx * g.live_fold_count;
                summarize_pow2_block_carries(
                    &eq_low,
                    block_offset_low,
                    &g.c_alphas.as_flat().unwrap()[start..(start + g.live_fold_count)],
                )
            })
            .collect::<Result<_, _>>()
            .unwrap();
        let c_alphas = g.c_alphas.as_flat().unwrap();
        let a_start = 1;
        let t_high = &fx.full_vec_randomness[offset_low_bits..];
        let t_offset_high = fx.offset_t >> offset_low_bits;
        let t_outer = challenge_block_summaries.len() * fx.g1_open.len() * g.n_a;
        let eq_hi_t: Vec<F> = (0..=t_outer)
            .map(|k| eq_eval_at_index(t_high, t_offset_high + k))
            .collect();
        let got = TStructuredSlicesEvaluator {
            gadget_vector: &fx.g1_open,
            challenge_block_summaries: &challenge_block_summaries,
            a_row_weights: &p.setup_contribution_inputs.eq_tau1[a_start..(a_start + g.n_a)],
            high_eq_table: &eq_hi_t,
        }
        .evaluate();

        let mut expected = F::zero();
        for x in 0..t_len {
            let compound_dig = x / total_blocks;
            let blk = x % total_blocks;
            let a_idx = compound_dig / g.depth_open;
            let digit_idx = compound_dig % g.depth_open;
            let entry = p.setup_contribution_inputs.eq_tau1[a_start + a_idx]
                * c_alphas[blk]
                * fx.g1_open[digit_idx];
            expected += entry * eq[fx.offset_t + x];
        }
        assert_eq!(got, expected);
    }

    #[test]
    fn r_tail_matches_materialized_range_inner_product() {
        let fx = fixture();
        let p = &fx.prepared;
        let alpha = f(7_000);
        let alpha_pows = scalar_powers(alpha, D);
        let denom = alpha_pows[D - 1] * alpha + F::one();
        let r_len = p.setup_contribution_inputs.rows * fx.r_gadget.len();
        let eq = eq_evals(fx.offset_r + r_len, &fx.full_vec_randomness);

        let got = compute_r_contribution::<F, F>(
            p,
            &fx.full_vec_randomness,
            fx.offset_r,
            denom,
            &fx.r_gadget,
        )
        .unwrap();
        let mut expected = F::zero();
        for idx in 0..r_len {
            let row_idx = idx / fx.r_gadget.len();
            let level_idx = idx % fx.r_gadget.len();
            let entry =
                -(p.setup_contribution_inputs.eq_tau1[row_idx] * denom * fx.r_gadget[level_idx]);
            expected += entry * eq[fx.offset_r + idx];
        }
        assert_eq!(got, expected);
    }
}
