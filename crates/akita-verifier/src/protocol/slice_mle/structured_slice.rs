#[cfg(test)]
use crate::protocol::ring_switch::PreparedChallengeEvals;
use crate::protocol::ring_switch::RingSwitchDeferredRowEval;
use akita_algebra::offset_eq::{eq_eval_at_index, eval_offset_eq_tensor};
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};

/// Number of carry buckets per outer index produced by
/// [`StructuredSliceMleEvaluator::compute_inner_sum`].
///
/// **Note:** This module is only tested and intended for the
/// `POSSIBLE_CARRIES = 2` case. Anything other than `2` would require the
/// outer-sum algebra to be reworked; do not change this constant.
pub(super) const POSSIBLE_CARRIES: usize = 2;

/// Inner-sum slot for the no-carry bucket (`carry = 0`).
pub(super) const CARRY0: usize = 0;

/// Inner-sum slot for the one-carry bucket (`carry = 1`).
pub(super) const CARRY1: usize = 1;

/// Build `table[k] = eq_high(offset_high + k)` for `k ∈ [0, hi_len]`.
pub(crate) fn high_eq_window<E: FieldCore>(
    high_challenges: &[E],
    offset_high: usize,
    hi_len: usize,
) -> Vec<E> {
    (0..=hi_len)
        .map(|k| eq_eval_at_index(high_challenges, offset_high + k))
        .collect()
}

/// Peeled-block MLE evaluator for one structured slice of `M`. See
/// `book/src/how/verifying/matrix_evaluation.md` for the full derivation.
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

/// Pow2 Z-segment slice evaluator.
pub(crate) struct ZStructuredPow2SlicesEvaluator<'a, F: FieldCore, E> {
    /// Commit-side gadget. Length = `depth_commit`.
    pub g1_commit: &'a [F],
    /// Fold-side gadget. Length = `depth_fold`.
    pub fold_gadget: &'a [F],
    /// Per-opening-point carry summary of `opening_point.a[..block_len]`.
    pub a_block_summary: &'a [[E; 2]],
    /// `tau1` equality weight for the consistency-challenge row of `M`.
    pub consistency_weight: E,
    /// Precomputed high-eq table relative to the slice's high offset.
    pub high_eq_table: &'a [E],
}

impl<F, E> StructuredSliceMleEvaluator<E> for ZStructuredPow2SlicesEvaluator<'_, F, E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    #[inline]
    fn num_outer_indices(&self) -> usize {
        self.a_block_summary.len() * self.fold_gadget.len() * self.g1_commit.len()
    }

    #[inline]
    fn high_eq_table(&self) -> &[E] {
        self.high_eq_table
    }

    #[inline]
    fn compute_inner_sum(&self, outer_index: usize) -> [E; POSSIBLE_CARRIES] {
        let num_points = self.a_block_summary.len();
        let depth_fold = self.fold_gadget.len();
        let pt = outer_index % num_points;
        let q1 = outer_index / num_points;
        let df = q1 % depth_fold;
        let dc = q1 / depth_fold;

        let [a_carry0, a_carry1] = self.a_block_summary[pt];
        let scale = (-self.consistency_weight)
            .mul_base(self.g1_commit[dc])
            .mul_base(self.fold_gadget[df]);
        [scale * a_carry0, scale * a_carry1]
    }
}

/// Dense fallback for non-pow2 Z segments. This path materializes the Z slice
/// and evaluates it through the generic offset-equality tensor helper.
pub(crate) struct ZDenseSlicesEvaluator<'a, F: FieldCore, E> {
    /// Commit-side gadget. Length = `depth_commit`.
    pub g1_commit: &'a [F],
    /// Fold-side gadget. Length = `depth_fold`.
    pub fold_gadget: &'a [F],
    /// `tau1` equality weight for the consistency-challenge row of `M`.
    pub consistency_weight: E,
    /// Per-point alpha-evaluated ring-multiplier `a` values.
    pub a_evals_by_point: &'a [Vec<E>],
    /// Full multilinear evaluation point.
    pub full_vec_randomness: &'a [E],
    /// Start-of-slice offset of `z` inside `M`.
    pub offset_z: usize,
    /// Inner block size of the `z` segment.
    pub block_len: usize,
}

impl<F, E> ZDenseSlicesEvaluator<'_, F, E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    /// Evaluate the dense materialized Z segment.
    pub(crate) fn evaluate(&self) -> Result<E, AkitaError> {
        let z_total_blocks = self.a_evals_by_point.len() * self.block_len;
        let z_len = self.fold_gadget.len() * self.g1_commit.len() * z_total_blocks;
        let z_segment_struct: Vec<E> = cfg_into_iter!(0..z_len)
            .map(|x| {
                let compound_dig = x / z_total_blocks;
                let global_blk = x % z_total_blocks;
                let dc_idx = compound_dig / self.fold_gadget.len();
                let df = compound_dig % self.fold_gadget.len();
                let point_idx = global_blk / self.block_len;
                let blk = global_blk % self.block_len;
                -self.consistency_weight
                    * self.a_evals_by_point[point_idx][blk]
                        .mul_base(self.g1_commit[dc_idx])
                        .mul_base(self.fold_gadget[df])
            })
            .collect();
        eval_offset_eq_tensor(
            self.full_vec_randomness,
            self.offset_z,
            E::one(),
            &[z_segment_struct.as_slice()],
        )
    }
}

/// Compute the `r`-tail contribution. Power-of-two `levels` uses a
/// multi-factor `eval_offset_eq_tensor`; otherwise materialises the
/// `r`-tail vector and falls back to the single-factor path.
pub(crate) fn compute_r_contribution<F, E>(
    prepared: &RingSwitchDeferredRowEval<E>,
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
    if levels.is_power_of_two() {
        let _span = tracing::info_span!("r_structured").entered();
        let r_gadget_ext: Vec<E> = r_gadget.iter().copied().map(E::lift_base).collect();
        eval_offset_eq_tensor(
            full_vec_randomness,
            offset_r,
            -denom,
            &[&r_gadget_ext, &prepared.eq_tau1[..prepared.setup_contribution_inputs.rows]],
        )
    } else {
        let _span = tracing::info_span!("r_dense").entered();
        let r_tail: Vec<E> = cfg_into_iter!(0..prepared.setup_contribution_inputs.rows * levels)
            .map(|idx| {
                let row_idx = idx / levels;
                let level_idx = idx % levels;
                -(prepared.eq_tau1[row_idx] * denom).mul_base(r_gadget[level_idx])
            })
            .collect();
        eval_offset_eq_tensor(
            full_vec_randomness,
            offset_r,
            E::one(),
            &[r_tail.as_slice()],
        )
    }
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
        gadget_row_scalars, r_decomp_levels, LevelParams, MRowLayout, OpeningClaimsLayout,
        RingMultiplierOpeningPoint, RingOpeningPoint, RingRelationInstance, SetupContributionPlanInputs,
        SisModulusFamily, WitnessLayout,
    };

    use crate::protocol::ring_switch::summarize_pow2_block_carries_base;

    type F = Prime128OffsetA7F7;
    const D: usize = 32;

    struct StructuredFixture {
        prepared: RingSwitchDeferredRowEval<F>,
        opening_point: RingOpeningPoint<F>,
        full_vec_randomness: Vec<F>,
        offset_e: usize,
        offset_t: usize,
        offset_z: usize,
        offset_r: usize,
        g1_open: Vec<F>,
        g1_commit: Vec<F>,
        fold_gadget: Vec<F>,
        r_gadget: Vec<F>,
    }

    fn f(value: u128) -> F {
        F::from_canonical_u128_reduced(value)
    }

    fn fixture_lp() -> LevelParams {
        LevelParams::params_only(
            SisModulusFamily::Q128,
            D,
            5,
            2,
            2,
            2,
            SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![1],
            },
        )
        .with_decomp(2, 3, 1, 26, 512 * 8)
        .expect("structured slice fixture lp")
    }

    fn ring_relation_segment_layout_for_opening_shape(
        lp: &LevelParams,
        m_row_layout: MRowLayout,
        num_polys: usize,
    ) -> Result<WitnessLayout, AkitaError> {
        let opening_batch = OpeningClaimsLayout::new(32, num_polys)?;
        let opening_point = RingOpeningPoint {
            a: vec![F::zero(); lp.block_len],
            b: vec![F::zero(); lp.num_blocks],
        };
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&opening_point);
        let num_claims = opening_batch.num_total_polynomials();
        let challenges = akita_challenges::Challenges::Sparse {
            challenges: Vec::new(),
            num_blocks_per_claim: lp.num_blocks,
            num_claims,
        };
        let v = vec![CyclotomicRing::<F, D>::zero(); lp.d_key.row_len()];
        let commitment_rows = vec![CyclotomicRing::<F, D>::zero(); lp.b_key.row_len()];
        let row_coefficient_rings = vec![CyclotomicRing::<F, D>::zero(); num_claims];
        let y = akita_types::assemble_relation_y::<F>(
            lp.role_dims(),
            akita_types::RelationYLayout {
                n_d: lp.d_key.row_len(),
                commit_rows_per_group: lp.b_key.row_len(),
                b_inner_rows_per_group: 0,
                n_a: lp.a_key.row_len(),
            },
            &akita_types::RingVec::from_ring_elems(&v),
            &akita_types::RingVec::from_ring_elems(&commitment_rows),
        )?;
        let instance = RingRelationInstance::<F>::new(
            m_row_layout,
            challenges,
            opening_point,
            ring_multiplier_point,
            opening_batch,
            vec![F::zero(); num_claims],
            akita_types::RingVec::from_ring_elems(&row_coefficient_rings),
            y,
            akita_types::RingVec::from_ring_elems(&v),
            lp.role_dims(),
        )?;
        instance.segment_layout(lp, None)
    }

    fn fixture() -> StructuredFixture {
        // `nv = 32` in `fp128_d32_onehot.rs` includes repeated compact
        // recursive levels with this real D=32 shape.
        let num_blocks = 8usize;
        let block_len = 512usize;
        let log_basis = 5u32;
        let depth_open = 26usize;
        let depth_commit = 1usize;
        let depth_fold = 4usize;
        let n_a = 2usize;
        let n_b = 2usize;
        let n_d = 2usize;
        let num_claims = 3usize;
        let num_points = 1usize;
        let total_blocks = num_blocks * num_claims;
        let rows = 1 + n_a + n_b * num_points + n_d;
        let inner_width = block_len * depth_commit;

        let levels = r_decomp_levels::<F>(log_basis);
        let lp = fixture_lp();
        let chunk_layout =
            ring_relation_segment_layout_for_opening_shape(&lp, MRowLayout::WithDBlock, num_claims)
                .expect("witness segment layout");
        let chunk0 = chunk_layout.chunks[0];
        let offset_e = chunk0.offset_e;
        let offset_t = chunk0.offset_t;
        let offset_z = chunk0.offset_z;
        let offset_r = chunk0.offset_r.expect("single chunk carries r-tail");
        let total_len = offset_r + rows * levels;
        let bits = total_len.next_power_of_two().trailing_zeros() as usize;

        let opening_point = RingOpeningPoint {
            a: (0..block_len).map(|idx| f(1_000 + idx as u128)).collect(),
            b: (0..num_blocks).map(|idx| f(2_000 + idx as u128)).collect(),
        };
        let prepared = RingSwitchDeferredRowEval {
            c_alphas: PreparedChallengeEvals::Flat(
                (0..total_blocks)
                    .map(|idx| f(3_000 + idx as u128))
                    .collect(),
            ),
            eq_tau1: (0..rows.next_power_of_two())
                .map(|idx| f(4_000 + idx as u128))
                .collect(),
            num_t_vectors: num_claims,
            num_blocks,
            num_claims,
            depth_open,
            depth_commit,
            depth_fold,
            block_len,
            log_basis,
            n_a,
            chunk_layout,
            setup_contribution_inputs: SetupContributionPlanInputs {
                eq_tau1: (0..rows.next_power_of_two())
                    .map(|idx| f(4_000 + idx as u128))
                    .collect(),
                num_t_vectors: num_claims,
                num_blocks,
                num_claims,
                depth_open,
                depth_commit,
                depth_fold,
                block_len,
                inner_width,
                n_a,
                n_d,
                m_row_layout: MRowLayout::WithDBlock,
                n_b,
                num_segments: 1,
                rows,
                num_polys_per_segment: vec![num_claims],
            },
        };
        let full_vec_randomness = (0..bits).map(|idx| f(6_000 + idx as u128)).collect();
        let g1_open = gadget_row_scalars::<F>(depth_open, log_basis);
        let g1_commit = gadget_row_scalars::<F>(depth_commit, log_basis);
        let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
        let r_gadget = gadget_row_scalars::<F>(levels, log_basis);

        StructuredFixture {
            prepared,
            opening_point,
            full_vec_randomness,
            offset_e,
            offset_t,
            offset_z,
            offset_r,
            g1_open,
            g1_commit,
            fold_gadget,
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
        let e_len = p.depth_open * p.total_blocks();
        let eq = eq_evals(fx.offset_e + e_len, &fx.full_vec_randomness);
        let offset_low_bits = p.num_blocks.trailing_zeros() as usize;
        let eq_low = EqPolynomial::evals(&fx.full_vec_randomness[..offset_low_bits]).unwrap();
        let block_offset_low = fx.offset_e & (p.num_blocks - 1);

        let challenge_block_summaries: Vec<[F; 2]> = (0..p.num_claims)
            .map(|claim_idx| {
                let start = claim_idx * p.num_blocks;
                summarize_pow2_block_carries(
                    &eq_low,
                    block_offset_low,
                    &p.c_alphas.as_flat().unwrap()[start..(start + p.num_blocks)],
                )
            })
            .collect::<Result<_, _>>()
            .unwrap();
        let c_alphas = p.c_alphas.as_flat().unwrap();
        let e_high = &fx.full_vec_randomness[offset_low_bits..];
        let e_offset_high = fx.offset_e >> offset_low_bits;
        let e_outer = challenge_block_summaries.len() * fx.g1_open.len();
        let eq_hi_e: Vec<F> = (0..=e_outer)
            .map(|k| eq_eval_at_index(e_high, e_offset_high + k))
            .collect();
        let got = EStructuredSlicesEvaluator {
            gadget_vector: &fx.g1_open,
            challenge_block_summaries: &challenge_block_summaries,
            challenge_weight: p.eq_tau1[0],
            high_eq_table: &eq_hi_e,
        }
        .evaluate();

        let mut expected = F::zero();
        for x in 0..e_len {
            let total_blocks = p.total_blocks();
            let dig = x / total_blocks;
            let blk = x % total_blocks;
            let entry = p.eq_tau1[0] * c_alphas[blk] * fx.g1_open[dig];
            expected += entry * eq[fx.offset_e + x];
        }
        assert_eq!(got, expected);
    }

    #[test]
    fn t_structured_matches_materialized_range_inner_product() {
        let fx = fixture();
        let p = &fx.prepared;
        let t_len = p.depth_open * p.n_a * p.total_blocks();
        let eq = eq_evals(fx.offset_t + t_len, &fx.full_vec_randomness);
        let offset_low_bits = p.num_blocks.trailing_zeros() as usize;
        let eq_low = EqPolynomial::evals(&fx.full_vec_randomness[..offset_low_bits]).unwrap();
        let block_offset_low = fx.offset_t & (p.num_blocks - 1);

        let challenge_block_summaries: Vec<[F; 2]> = (0..p.num_claims)
            .map(|claim_idx| {
                let start = claim_idx * p.num_blocks;
                summarize_pow2_block_carries(
                    &eq_low,
                    block_offset_low,
                    &p.c_alphas.as_flat().unwrap()[start..(start + p.num_blocks)],
                )
            })
            .collect::<Result<_, _>>()
            .unwrap();
        let c_alphas = p.c_alphas.as_flat().unwrap();
        let a_start = 1;
        let t_high = &fx.full_vec_randomness[offset_low_bits..];
        let t_offset_high = fx.offset_t >> offset_low_bits;
        let t_outer = challenge_block_summaries.len() * fx.g1_open.len() * p.n_a;
        let eq_hi_t: Vec<F> = (0..=t_outer)
            .map(|k| eq_eval_at_index(t_high, t_offset_high + k))
            .collect();
        let got = TStructuredSlicesEvaluator {
            gadget_vector: &fx.g1_open,
            challenge_block_summaries: &challenge_block_summaries,
            a_row_weights: &p.eq_tau1[a_start..(a_start + p.n_a)],
            high_eq_table: &eq_hi_t,
        }
        .evaluate();

        let mut expected = F::zero();
        for x in 0..t_len {
            let total_blocks = p.total_blocks();
            let compound_dig = x / total_blocks;
            let blk = x % total_blocks;
            let a_idx = compound_dig / p.depth_open;
            let digit_idx = compound_dig % p.depth_open;
            let entry = p.eq_tau1[a_start + a_idx] * c_alphas[blk] * fx.g1_open[digit_idx];
            expected += entry * eq[fx.offset_t + x];
        }
        assert_eq!(got, expected);
    }

    #[test]
    fn z_structured_matches_materialized_range_inner_product() {
        let fx = fixture();
        let p = &fx.prepared;
        let z_len = p.depth_fold * p.depth_commit * p.block_len;
        let eq = eq_evals(fx.offset_z + z_len, &fx.full_vec_randomness);
        let z_offset_low_bits = p.block_len.trailing_zeros() as usize;
        let z_block_low_eq =
            EqPolynomial::evals(&fx.full_vec_randomness[..z_offset_low_bits]).unwrap();
        let z_offset_low = fx.offset_z & (p.block_len - 1);

        let a_block_summary = vec![summarize_pow2_block_carries_base::<F, F>(
            &z_block_low_eq,
            z_offset_low,
            &fx.opening_point.a[..p.block_len],
        )
        .unwrap()];
        let z_high = &fx.full_vec_randomness[z_offset_low_bits..];
        let z_offset_high = fx.offset_z >> z_offset_low_bits;
        let z_outer = a_block_summary.len() * fx.fold_gadget.len() * fx.g1_commit.len();
        let eq_hi_z: Vec<F> = (0..=z_outer)
            .map(|k| eq_eval_at_index(z_high, z_offset_high + k))
            .collect();
        let got = ZStructuredPow2SlicesEvaluator {
            g1_commit: &fx.g1_commit,
            fold_gadget: &fx.fold_gadget,
            a_block_summary: &a_block_summary,
            consistency_weight: p.eq_tau1[0],
            high_eq_table: &eq_hi_z,
        }
        .evaluate();

        let mut expected = F::zero();
        let z_total_blocks = p.block_len;
        for x in 0..z_len {
            let compound_dig = x / z_total_blocks;
            let global_blk = x % z_total_blocks;
            let dc = compound_dig / p.depth_fold;
            let df = compound_dig % p.depth_fold;
            let blk = global_blk % p.block_len;
            let entry =
                -(p.eq_tau1[0] * fx.opening_point.a[blk] * fx.g1_commit[dc] * fx.fold_gadget[df]);
            expected += entry * eq[fx.offset_z + x];
        }
        assert_eq!(got, expected);
    }

    #[test]
    fn z_dense_matches_materialized_range_inner_product() {
        let mut fx = fixture();
        fx.prepared.block_len = 510;
        fx.prepared.setup_contribution_inputs.inner_width =
            fx.prepared.block_len * fx.prepared.depth_commit;
        let p = &fx.prepared;
        assert!(!p.block_len.is_power_of_two());

        let z_len = p.depth_fold * p.depth_commit * p.block_len;
        let eq = eq_evals(fx.offset_z + z_len, &fx.full_vec_randomness);
        let a_evals_by_point = vec![fx.opening_point.a[..p.block_len].to_vec()];
        let got = ZDenseSlicesEvaluator {
            g1_commit: &fx.g1_commit,
            fold_gadget: &fx.fold_gadget,
            consistency_weight: p.eq_tau1[0],
            a_evals_by_point: &a_evals_by_point,
            full_vec_randomness: &fx.full_vec_randomness,
            offset_z: fx.offset_z,
            block_len: p.block_len,
        }
        .evaluate()
        .unwrap();

        let mut expected = F::zero();
        let z_total_blocks = p.block_len;
        for x in 0..z_len {
            let compound_dig = x / z_total_blocks;
            let global_blk = x % z_total_blocks;
            let dc = compound_dig / p.depth_fold;
            let df = compound_dig % p.depth_fold;
            let blk = global_blk % p.block_len;
            let entry =
                -(p.eq_tau1[0] * fx.opening_point.a[blk] * fx.g1_commit[dc] * fx.fold_gadget[df]);
            expected += entry * eq[fx.offset_z + x];
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
            let entry = -(p.eq_tau1[row_idx] * denom * fx.r_gadget[level_idx]);
            expected += entry * eq[fx.offset_r + idx];
        }
        assert_eq!(got, expected);
    }
}
