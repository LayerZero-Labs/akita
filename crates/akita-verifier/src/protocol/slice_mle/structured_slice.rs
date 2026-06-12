#[cfg(test)]
use crate::protocol::ring_switch::PreparedChallengeEvals;
use crate::protocol::ring_switch::RingSwitchDeferredRowEval;
use akita_algebra::offset_eq::{eq_eval_at_index, eval_offset_eq_tensor};
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};

#[inline]
fn evaluate_structured_slice<F, InnerSum>(
    num_outer_indices: usize,
    high_challenges: &[F],
    offset_high: usize,
    mut inner_sum: InnerSum,
) -> F
where
    F: FieldCore,
    InnerSum: FnMut(usize) -> [F; 2],
{
    (0..num_outer_indices).fold(F::zero(), |acc, q| {
        let [carry0, carry1] = inner_sum(q);
        let acc = if carry0.is_zero() {
            acc
        } else {
            acc + carry0 * eq_eval_at_index(high_challenges, offset_high + q)
        };
        if carry1.is_zero() {
            acc
        } else {
            acc + carry1 * eq_eval_at_index(high_challenges, offset_high + q + 1)
        }
    })
}

/// E-hat segment slice evaluator. See `specs/optimized_verifier.md`.
pub(crate) struct EStructuredSlicesEvaluator<'a, F, E> {
    pub high_challenges: &'a [E],
    pub offset_high: usize,
    pub gadget_vector: &'a [F],
    pub public_block_summaries: &'a [[E; 2]],
    pub challenge_block_summaries: &'a [[E; 2]],
    pub public_row_weights_by_claim: &'a [E],
    pub challenge_weight: E,
}

impl<F, E> EStructuredSlicesEvaluator<'_, F, E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    #[inline]
    pub(crate) fn evaluate(&self) -> E {
        evaluate_structured_slice(
            self.public_block_summaries.len() * self.gadget_vector.len(),
            self.high_challenges,
            self.offset_high,
            |outer_index| {
                let num_claims = self.public_block_summaries.len();
                let digit = outer_index / num_claims;
                let claim_idx = outer_index % num_claims;

                let [aggregated_opening_carry0, aggregated_opening_carry1] =
                    self.public_block_summaries[claim_idx];
                let [aggregated_challenge_carry0, aggregated_challenge_carry1] =
                    self.challenge_block_summaries[claim_idx];

                [
                    (self.public_row_weights_by_claim[claim_idx] * aggregated_opening_carry0
                        + self.challenge_weight * aggregated_challenge_carry0)
                        .mul_base(self.gadget_vector[digit]),
                    (self.public_row_weights_by_claim[claim_idx] * aggregated_opening_carry1
                        + self.challenge_weight * aggregated_challenge_carry1)
                        .mul_base(self.gadget_vector[digit]),
                ]
            },
        )
    }
}

/// T-segment slice evaluator. See `specs/optimized_verifier.md`.
pub(crate) struct TStructuredSlicesEvaluator<'a, F, E> {
    pub high_challenges: &'a [E],
    pub offset_high: usize,
    pub gadget_vector: &'a [F],
    pub challenge_block_summaries: &'a [[E; 2]],
    pub a_row_weights: &'a [E],
}

impl<F, E> TStructuredSlicesEvaluator<'_, F, E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    #[inline]
    pub(crate) fn evaluate(&self) -> E {
        evaluate_structured_slice(
            self.challenge_block_summaries.len()
                * self.gadget_vector.len()
                * self.a_row_weights.len(),
            self.high_challenges,
            self.offset_high,
            |outer_index| {
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
            },
        )
    }
}

/// Pow2 Z-segment slice evaluator. See `specs/optimized_verifier.md`.
pub(crate) struct ZStructuredPow2SlicesEvaluator<'a, F: FieldCore, E> {
    pub high_challenges: &'a [E],
    pub offset_high: usize,
    pub g1_commit: &'a [F],
    pub fold_gadget: &'a [F],
    pub a_block_summary: &'a [[E; 2]],
    pub consistency_weight: E,
}

impl<F, E> ZStructuredPow2SlicesEvaluator<'_, F, E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    #[inline]
    pub(crate) fn evaluate(&self) -> E {
        evaluate_structured_slice(
            self.a_block_summary.len() * self.fold_gadget.len() * self.g1_commit.len(),
            self.high_challenges,
            self.offset_high,
            |outer_index| {
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
            },
        )
    }
}

/// Dense fallback for non-pow2 Z segments. This path materializes the Z slice
/// and evaluates it through the generic offset-equality tensor helper.
pub(crate) struct ZDenseSlicesEvaluator<'a, F: FieldCore, E> {
    pub g1_commit: &'a [F],
    pub fold_gadget: &'a [F],
    pub consistency_weight: E,
    pub a_evals_by_point: &'a [Vec<E>],
    pub full_vec_randomness: &'a [E],
    pub offset_z: usize,
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
            &[&r_gadget_ext, &prepared.eq_tau1[..prepared.rows]],
        )
    } else {
        let _span = tracing::info_span!("r_dense").entered();
        let r_tail: Vec<E> = cfg_into_iter!(0..prepared.rows * levels)
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
    use akita_algebra::offset_eq::summarize_pow2_block_carries;
    use akita_algebra::ring::scalar_powers;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Prime128OffsetA7F7;
    use akita_types::{
        gadget_row_scalars, r_decomp_levels, ring_relation_segment_layout_for_opening_shape,
        LevelParams, MRowLayout, RingOpeningPoint, SisModulusFamily,
    };

    type F = Prime128OffsetA7F7;
    const D: usize = 32;

    struct StructuredFixture {
        prepared: RingSwitchDeferredRowEval<F>,
        opening_points: Vec<RingOpeningPoint<F>>,
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
        let num_polys_per_point = vec![2usize, 1usize];
        let num_points = num_polys_per_point.len();
        let num_claims = 3usize;
        let num_public_rows = num_points;
        let total_blocks = num_blocks * num_claims;
        let rows = 1 + num_public_rows + n_d + n_b * num_points + n_a;
        let inner_width = block_len * depth_commit;

        let levels = r_decomp_levels::<F>(log_basis);
        let lp = fixture_lp();
        let witness_segment_layout = ring_relation_segment_layout_for_opening_shape::<F, D>(
            &lp,
            MRowLayout::WithDBlock,
            &num_polys_per_point,
        )
        .expect("witness segment layout");
        let offset_e = witness_segment_layout.offset_e;
        let offset_t = witness_segment_layout.offset_t;
        let offset_z = witness_segment_layout.offset_z;
        let offset_r = witness_segment_layout.offset_r;
        let total_len = offset_r + rows * levels;
        let bits = total_len.next_power_of_two().trailing_zeros() as usize;

        let opening_points = (0..num_points)
            .map(|pt| RingOpeningPoint {
                a: (0..block_len)
                    .map(|idx| f(1_000 + (pt * block_len + idx) as u128))
                    .collect(),
                b: (0..num_blocks)
                    .map(|idx| f(2_000 + (pt * num_blocks + idx) as u128))
                    .collect(),
            })
            .collect();
        let prepared = RingSwitchDeferredRowEval {
            c_alphas: PreparedChallengeEvals::Flat {
                evals: (0..total_blocks)
                    .map(|idx| f(3_000 + idx as u128))
                    .collect(),
                num_claims,
                num_blocks,
            },
            eq_tau1: (0..rows.next_power_of_two())
                .map(|idx| f(4_000 + idx as u128))
                .collect(),
            num_t_vectors: num_polys_per_point.iter().sum(),
            num_blocks,
            num_claims,
            depth_open,
            depth_commit,
            depth_fold,
            #[cfg(feature = "zk")]
            d_blinding_segment_len: 0,
            #[cfg(feature = "zk")]
            b_blinding_digit_planes_per_point: 0,
            #[cfg(feature = "zk")]
            b_blinding_segment_len: 0,
            block_len,
            inner_width,
            log_basis,
            n_a,
            n_d,
            m_row_layout: MRowLayout::WithDBlock,
            n_b,
            tier_split: 1,
            n_f: 0,
            num_points,
            rows,
            claim_to_t_vector: vec![1, 2, 0],
            num_polys_per_commitment_group: num_polys_per_point,
            num_public_rows,
            gamma: (0..num_claims).map(|idx| f(5_000 + idx as u128)).collect(),
            claim_to_point: vec![1, 0, 1],
            witness_segment_layout,
        };
        let full_vec_randomness = (0..bits).map(|idx| f(6_000 + idx as u128)).collect();
        let g1_open = gadget_row_scalars::<F>(depth_open, log_basis);
        let g1_commit = gadget_row_scalars::<F>(depth_commit, log_basis);
        let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
        let r_gadget = gadget_row_scalars::<F>(levels, log_basis);

        StructuredFixture {
            prepared,
            opening_points,
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

    fn eq_evals(full_vec_randomness: &[F]) -> Vec<F> {
        (0..(1usize << full_vec_randomness.len()))
            .map(|idx| eq_eval_at_index(full_vec_randomness, idx))
            .collect()
    }

    #[test]
    fn e_structured_matches_materialized_range_inner_product() {
        let fx = fixture();
        let p = &fx.prepared;
        let total_blocks = p.num_blocks * p.num_claims;
        let e_len = p.depth_open * total_blocks;
        let eq = eq_evals(&fx.full_vec_randomness);
        let offset_low_bits = p.num_blocks.trailing_zeros() as usize;
        let eq_low = EqPolynomial::evals(&fx.full_vec_randomness[..offset_low_bits]).unwrap();
        let block_offset_low = fx.offset_e & (p.num_blocks - 1);

        let public_block_summaries: Vec<[F; 2]> = (0..p.num_claims)
            .map(|claim_idx| {
                let point_idx = p.claim_to_point[claim_idx];
                let mut summary = summarize_pow2_block_carries(
                    &eq_low,
                    block_offset_low,
                    &fx.opening_points[point_idx].b,
                )?;
                summary[0] *= p.gamma[claim_idx];
                summary[1] *= p.gamma[claim_idx];
                Ok::<[F; 2], AkitaError>(summary)
            })
            .collect::<Result<_, _>>()
            .unwrap();
        let public_row_weights_by_claim: Vec<F> = p
            .claim_to_point
            .iter()
            .map(|&point_idx| p.eq_tau1[1 + point_idx])
            .collect();
        let PreparedChallengeEvals::Flat {
            evals: c_alphas, ..
        } = &p.c_alphas
        else {
            unreachable!("structured slice fixture uses flat challenges");
        };
        let challenge_block_summaries: Vec<[F; 2]> = (0..p.num_claims)
            .map(|claim_idx| {
                let start = claim_idx * p.num_blocks;
                summarize_pow2_block_carries(
                    &eq_low,
                    block_offset_low,
                    &c_alphas[start..(start + p.num_blocks)],
                )
            })
            .collect::<Result<_, _>>()
            .unwrap();
        let got = EStructuredSlicesEvaluator {
            high_challenges: &fx.full_vec_randomness[offset_low_bits..],
            offset_high: fx.offset_e >> offset_low_bits,
            gadget_vector: &fx.g1_open,
            public_block_summaries: &public_block_summaries,
            challenge_block_summaries: &challenge_block_summaries,
            public_row_weights_by_claim: &public_row_weights_by_claim,
            challenge_weight: p.eq_tau1[0],
        }
        .evaluate();

        let mut expected = F::zero();
        for x in 0..e_len {
            let dig = x / total_blocks;
            let blk = x % total_blocks;
            let claim_idx = blk / p.num_blocks;
            let block_idx = blk % p.num_blocks;
            let point_idx = p.claim_to_point[claim_idx];
            let entry = (p.eq_tau1[1 + point_idx]
                * p.gamma[claim_idx]
                * fx.opening_points[point_idx].b[block_idx]
                + p.eq_tau1[0] * c_alphas[blk])
                * fx.g1_open[dig];
            expected += entry * eq[fx.offset_e + x];
        }
        assert_eq!(got, expected);
    }

    #[test]
    fn t_structured_matches_materialized_range_inner_product() {
        let fx = fixture();
        let p = &fx.prepared;
        let total_blocks = p.num_blocks * p.num_claims;
        let t_len = p.depth_open * p.n_a * total_blocks;
        let eq = eq_evals(&fx.full_vec_randomness);
        let offset_low_bits = p.num_blocks.trailing_zeros() as usize;
        let eq_low = EqPolynomial::evals(&fx.full_vec_randomness[..offset_low_bits]).unwrap();
        let block_offset_low = fx.offset_t & (p.num_blocks - 1);

        let PreparedChallengeEvals::Flat {
            evals: c_alphas, ..
        } = &p.c_alphas
        else {
            unreachable!("structured slice fixture uses flat challenges");
        };
        let challenge_block_summaries: Vec<[F; 2]> = (0..p.num_claims)
            .map(|claim_idx| {
                let start = claim_idx * p.num_blocks;
                summarize_pow2_block_carries(
                    &eq_low,
                    block_offset_low,
                    &c_alphas[start..(start + p.num_blocks)],
                )
            })
            .collect::<Result<_, _>>()
            .unwrap();
        let a_start = 1 + p.num_public_rows + p.n_d_active() + p.n_b * p.num_points;
        let got = TStructuredSlicesEvaluator {
            high_challenges: &fx.full_vec_randomness[offset_low_bits..],
            offset_high: fx.offset_t >> offset_low_bits,
            gadget_vector: &fx.g1_open,
            challenge_block_summaries: &challenge_block_summaries,
            a_row_weights: &p.eq_tau1[a_start..p.rows],
        }
        .evaluate();

        let mut expected = F::zero();
        for x in 0..t_len {
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
        let z_len = p.depth_fold * p.depth_commit * p.num_points * p.block_len;
        let eq = eq_evals(&fx.full_vec_randomness);
        let z_offset_low_bits = p.block_len.trailing_zeros() as usize;
        let z_block_low_eq =
            EqPolynomial::evals(&fx.full_vec_randomness[..z_offset_low_bits]).unwrap();
        let z_offset_low = fx.offset_z & (p.block_len - 1);

        let a_block_summary: Vec<[F; 2]> = fx
            .opening_points
            .iter()
            .map(|point| {
                summarize_pow2_block_carries(&z_block_low_eq, z_offset_low, &point.a[..p.block_len])
            })
            .collect::<Result<_, _>>()
            .unwrap();
        let got = ZStructuredPow2SlicesEvaluator {
            high_challenges: &fx.full_vec_randomness[z_offset_low_bits..],
            offset_high: fx.offset_z >> z_offset_low_bits,
            g1_commit: &fx.g1_commit,
            fold_gadget: &fx.fold_gadget,
            a_block_summary: &a_block_summary,
            consistency_weight: p.eq_tau1[0],
        }
        .evaluate();

        let mut expected = F::zero();
        let z_total_blocks = p.num_points * p.block_len;
        for x in 0..z_len {
            let compound_dig = x / z_total_blocks;
            let global_blk = x % z_total_blocks;
            let dc = compound_dig / p.depth_fold;
            let df = compound_dig % p.depth_fold;
            let point_idx = global_blk / p.block_len;
            let blk = global_blk % p.block_len;
            let entry = -(p.eq_tau1[0]
                * fx.opening_points[point_idx].a[blk]
                * fx.g1_commit[dc]
                * fx.fold_gadget[df]);
            expected += entry * eq[fx.offset_z + x];
        }
        assert_eq!(got, expected);
    }

    #[test]
    fn z_dense_matches_materialized_range_inner_product() {
        let mut fx = fixture();
        fx.prepared.block_len = 510;
        fx.prepared.inner_width = fx.prepared.block_len * fx.prepared.depth_commit;
        let p = &fx.prepared;
        assert!(!p.block_len.is_power_of_two());

        let z_len = p.depth_fold * p.depth_commit * p.num_points * p.block_len;
        let eq = eq_evals(&fx.full_vec_randomness);
        let a_evals_by_point: Vec<Vec<F>> = fx
            .opening_points
            .iter()
            .map(|point| point.a[..p.block_len].to_vec())
            .collect();
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
        let z_total_blocks = p.num_points * p.block_len;
        for x in 0..z_len {
            let compound_dig = x / z_total_blocks;
            let global_blk = x % z_total_blocks;
            let dc = compound_dig / p.depth_fold;
            let df = compound_dig % p.depth_fold;
            let point_idx = global_blk / p.block_len;
            let blk = global_blk % p.block_len;
            let entry = -(p.eq_tau1[0]
                * fx.opening_points[point_idx].a[blk]
                * fx.g1_commit[dc]
                * fx.fold_gadget[df]);
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
        let r_len = p.rows * fx.r_gadget.len();
        let eq = eq_evals(&fx.full_vec_randomness);

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
