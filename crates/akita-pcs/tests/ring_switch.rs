//! Ring-switch integration regressions.

use akita_algebra::CyclotomicRing;
#[cfg(all(test, feature = "parallel"))]
use akita_field::parallel::*;
use akita_field::AkitaError;
use akita_pcs::{CanonicalField, FieldCore};
use std::array::from_fn;

fn compute_r_via_poly_division<F: FieldCore + CanonicalField, const D: usize>(
    m: &[Vec<CyclotomicRing<F, D>>],
    z: &[CyclotomicRing<F, D>],
    y: &[CyclotomicRing<F, D>],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    let poly_len = 2 * D - 1;
    let out = m
        .iter()
        .zip(y.iter())
        .map(|(row, y_i)| {
            let column_contribution =
                |m_ij: &CyclotomicRing<F, D>, z_j: &CyclotomicRing<F, D>| -> Vec<F> {
                    let mut local = vec![F::zero(); poly_len];
                    if m_ij.is_zero() {
                        return local;
                    }
                    let a = m_ij.coefficients();
                    let b = z_j.coefficients();
                    let is_scalar = a[1..].iter().all(|c| c.is_zero());
                    if is_scalar {
                        let scalar = a[0];
                        for s in 0..D {
                            local[s] = scalar * b[s];
                        }
                    } else {
                        for t in 0..D {
                            for s in 0..D {
                                local[t + s] += a[t] * b[s];
                            }
                        }
                    }
                    local
                };

            let pointwise_add = |mut a: Vec<F>, b: Vec<F>| -> Vec<F> {
                for (ai, bi) in a.iter_mut().zip(b.iter()) {
                    *ai += *bi;
                }
                a
            };

            #[cfg(feature = "parallel")]
            let mut poly = row
                .par_iter()
                .zip(z.par_iter())
                .fold(
                    || vec![F::zero(); poly_len],
                    |acc, (m_ij, z_j)| pointwise_add(acc, column_contribution(m_ij, z_j)),
                )
                .reduce(|| vec![F::zero(); poly_len], pointwise_add);

            #[cfg(not(feature = "parallel"))]
            let mut poly = row
                .iter()
                .zip(z.iter())
                .fold(vec![F::zero(); poly_len], |acc, (m_ij, z_j)| {
                    pointwise_add(acc, column_contribution(m_ij, z_j))
                });
            let y_coeffs = y_i.coefficients();
            for k in 0..D {
                poly[k] -= y_coeffs[k];
            }
            let mut quotient = vec![F::zero(); D];
            for k in (D..poly_len).rev() {
                let q = poly[k];
                quotient[k - D] = q;
                poly[k - D] -= q;
            }
            let coeffs: [F; D] = from_fn(|k| quotient[k]);
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::compute_r_via_poly_division;
    use akita_algebra::ring::scalar_powers;
    use akita_algebra::CyclotomicRing;
    use akita_config::proof_optimized::fp128;
    use akita_config::{BareCfg, CommitmentConfig};
    use akita_pcs::AkitaCommitmentScheme;
    use akita_pcs::{CanonicalField, CommitmentProver, Transcript};
    use akita_prover::protocol::ring_switch::{
        build_w_evals_compact, compute_m_evals_x, ring_switch_build_w,
    };
    use akita_prover::{AkitaPolyOps, DensePoly, QuadraticEquation};
    use akita_transcript::labels::{ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS};
    use akita_transcript::Blake2bTranscript;
    use akita_types::relation_claim_from_rows;
    use akita_types::AppendToTranscript;
    use akita_types::{ring_opening_point_from_field, BasisMode, BlockOrder};
    use akita_verifier::prepare_m_eval;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::array::from_fn;

    use akita_pcs::{FieldCore, FromPrimitiveInt, RandomSampling};

    fn compute_r_schoolbook<F: FieldCore, const D: usize>(
        m: &[Vec<CyclotomicRing<F, D>>],
        z: &[CyclotomicRing<F, D>],
        y: &[CyclotomicRing<F, D>],
    ) -> Vec<CyclotomicRing<F, D>> {
        let poly_len = 2 * D - 1;
        m.iter()
            .zip(y.iter())
            .map(|(row, y_i)| {
                let mut poly = vec![F::zero(); poly_len];
                for (m_ij, z_j) in row.iter().zip(z.iter()) {
                    if m_ij.is_zero() {
                        continue;
                    }
                    let a = m_ij.coefficients();
                    let b = z_j.coefficients();
                    let is_scalar = a[1..].iter().all(|c| c.is_zero());
                    if is_scalar {
                        let scalar = a[0];
                        for s in 0..D {
                            poly[s] += scalar * b[s];
                        }
                    } else {
                        for t in 0..D {
                            for s in 0..D {
                                poly[t + s] += a[t] * b[s];
                            }
                        }
                    }
                }
                let y_coeffs = y_i.coefficients();
                for k in 0..D {
                    poly[k] -= y_coeffs[k];
                }
                let mut quotient = vec![F::zero(); D];
                for k in (D..poly_len).rev() {
                    let q = poly[k];
                    quotient[k - D] = q;
                    poly[k - D] -= q;
                }
                let coeffs: [F; D] = from_fn(|k| quotient[k]);
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect()
    }

    #[test]
    fn compute_r_matches_schoolbook_reference() {
        type F = fp128::Field;
        const D: usize = 64;

        let m: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
            .map(|i| {
                (0..4)
                    .map(|j| {
                        if (i + j) % 3 == 0 {
                            let mut coeffs = [F::zero(); D];
                            coeffs[0] = F::from_u64((i * 5 + j + 1) as u64);
                            CyclotomicRing::from_coefficients(coeffs)
                        } else {
                            let coeffs = from_fn(|k| {
                                F::from_u64((i as u64 * 1000 + j as u64 * 100 + k as u64 + 1) % 97)
                            });
                            CyclotomicRing::from_coefficients(coeffs)
                        }
                    })
                    .collect()
            })
            .collect();
        let z: Vec<CyclotomicRing<F, D>> = (0..4)
            .map(|j| {
                let coeffs = from_fn(|k| F::from_u64((j as u64 * 37 + k as u64 + 5) % 89));
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect();
        let y: Vec<CyclotomicRing<F, D>> = (0..3)
            .map(|i| {
                let coeffs = from_fn(|k| F::from_u64((i as u64 * 29 + k as u64 + 7) % 83));
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect();

        let expected = compute_r_schoolbook(&m, &z, &y);
        let got = compute_r_via_poly_division::<F, D>(&m, &z, &y)
            .expect("ring-switch CRT+NTT path should dispatch for D=64");
        assert_eq!(got, expected);
    }

    fn direct_relation_claim<F: FieldCore + FromPrimitiveInt>(
        w_compact: &[i8],
        alpha_evals_y: &[F],
        m_evals_x: &[F],
        live_x_cols: usize,
    ) -> F {
        (0..live_x_cols).fold(F::zero(), |acc_x, x| {
            let column_start = x * alpha_evals_y.len();
            let y_eval = alpha_evals_y
                .iter()
                .enumerate()
                .fold(F::zero(), |acc_y, (y, &alpha)| {
                    acc_y + F::from_i64(w_compact[column_start + y] as i64) * alpha
                });
            acc_x + y_eval * m_evals_x[x]
        })
    }

    #[test]
    fn full_root_rows_match_direct_relation_claim() {
        type F = fp128::Field;
        // `D128Full` defaults to CR-on with tier_shrink=2 per audit B-1;
        // the cascade is infeasible at NV=12 (too few variables for the
        // shrink chain). This test exercises the M-relation row layout
        // and does not need a CR-on cfg, so route through `BareCfg` to
        // pick up the un-cascaded schedule.
        type Cfg = BareCfg<fp128::D128Full>;
        const D: usize = fp128::D128Full::D;
        const NV: usize = 12;

        let lp = Cfg::commitment_layout(NV).expect("lp");

        let mut rng = StdRng::seed_from_u64(0x5eed_cafe);
        let evals: Vec<F> = (0..(1usize << NV))
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(NV, &evals).expect("dense poly");
        let point: Vec<F> = (0..NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let (commitment, batched_hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<
            F,
            D,
        >>::commit(&[poly.clone()], &setup)
        .expect("commitment");

        let alpha_bits = D.trailing_zeros() as usize;
        let outer_point = &point[alpha_bits..];
        let ring_opening_point = ring_opening_point_from_field(
            outer_point,
            lp.r_vars,
            lp.m_vars,
            BasisMode::Lagrange,
            BlockOrder::RowMajor,
        )
        .expect("ring opening point");
        let (y_ring, w_folded) =
            poly.evaluate_and_fold(&ring_opening_point.b, &ring_opening_point.a, lp.block_len);

        let mut transcript = Blake2bTranscript::<F>::new(b"ring-switch-row-regression");
        commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);

        let mut quad_eq = QuadraticEquation::<F, D>::new_prover(
            &setup.ntt_shared,
            vec![ring_opening_point],
            vec![0usize],
            &[&poly],
            vec![w_folded],
            &[1usize],
            lp.clone(),
            vec![batched_hint],
            &mut transcript,
            std::slice::from_ref(&commitment),
            std::slice::from_ref(&y_ring),
            vec![F::one()],
            setup.expanded.seed.max_stride,
        )
        .expect("quadratic equation");

        let w = ring_switch_build_w::<F, D>(&mut quad_eq, &setup.expanded, &setup.ntt_shared, &lp)
            .expect("ring-switch witness");
        let (w_compact, _col_bits, ring_bits) =
            build_w_evals_compact(w.as_i8_digits(), D).expect("compact witness");
        let live_x_cols = w_compact.len() >> ring_bits;

        let alpha = F::from_u64(17);
        let alpha_evals_y = scalar_powers(alpha, D);
        let rows = lp.m_row_count(1, 1);
        let num_i = rows.next_power_of_two().trailing_zeros() as usize;

        for row in 0..rows {
            let tau1: Vec<F> = (0..num_i)
                .map(|bit| {
                    if (row >> bit) & 1 == 1 {
                        F::one()
                    } else {
                        F::zero()
                    }
                })
                .collect();
            let m_evals_x = compute_m_evals_x::<F, D>(
                &setup.expanded,
                &[quad_eq.opening_point().clone()],
                &[0usize],
                &quad_eq.challenges,
                alpha,
                &alpha_evals_y,
                &lp,
                &tau1,
                &[1usize],
                &[F::one()],
                1,
            )
            .expect("m evals");
            let got = direct_relation_claim(&w_compact, &alpha_evals_y, &m_evals_x, live_x_cols);
            let expected = relation_claim_from_rows::<F, D>(
                &tau1,
                alpha,
                &quad_eq.v,
                &commitment.u,
                std::slice::from_ref(&y_ring),
            );
            assert_eq!(got, expected, "row {row} mismatch");
        }
    }

    #[test]
    fn asymmetric_centering_decompose_roundtrip() {
        use akita_types::layout::digit_math::compute_num_digits_full_field;

        type F = fp128::Field;
        const D: usize = 64;

        let mut rng = rand::thread_rng();

        for log_basis in [2u32, 3, 4, 5, 6] {
            let field_bits = 128u32;
            let num_digits = compute_num_digits_full_field(field_bits, log_basis);

            let ring: CyclotomicRing<F, D> = RandomSampling::random(&mut rng);

            let mut digits = vec![CyclotomicRing::<F, D>::zero(); num_digits];
            ring.balanced_decompose_pow2_into(&mut digits, log_basis);
            let recomposed = CyclotomicRing::gadget_recompose_pow2(&digits, log_basis);
            assert_eq!(
                ring, recomposed,
                "field-element roundtrip failed for log_basis={log_basis}, num_digits={num_digits}"
            );

            let mut i8_digits = vec![[0i8; D]; num_digits];
            ring.balanced_decompose_pow2_i8_into(&mut i8_digits, log_basis);
            let recomposed_i8 = CyclotomicRing::gadget_recompose_pow2_i8(&i8_digits, log_basis);
            assert_eq!(
                ring, recomposed_i8,
                "i8 roundtrip failed for log_basis={log_basis}, num_digits={num_digits}"
            );
        }
    }

    fn assert_prepared_m_eval_matches_materialized(
        level_params: akita_types::LevelParams,
        num_claims: usize,
    ) {
        use akita_sumcheck::multilinear_eval;

        type F = fp128::Field;
        // `D128Full` defaults to CR-on with tier_shrink=2 per audit B-1;
        // the cascade is infeasible at NV=12. This helper only exercises
        // the M-eval matching invariant, so route through `BareCfg`.
        type Cfg = BareCfg<fp128::D128Full>;
        const D: usize = fp128::D128Full::D;
        const NV: usize = 12;

        assert!(num_claims > 0);
        let mut rng = StdRng::seed_from_u64(0xdead_beef);
        let polys: Vec<DensePoly<F, D>> = (0..num_claims)
            .map(|claim_idx| {
                let evals: Vec<F> = (0..(1usize << NV))
                    .map(|_| {
                        let raw = rng.gen::<u128>() ^ ((claim_idx as u128) << 96);
                        F::from_canonical_u128_reduced(raw)
                    })
                    .collect();
                DensePoly::<F, D>::from_field_evals(NV, &evals).expect("dense poly")
            })
            .collect();
        let poly_refs: Vec<&DensePoly<F, D>> = polys.iter().collect();
        let point: Vec<F> = (0..NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let setup = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(
            NV, num_claims, 1,
        );
        let (commitment, batched_hint) =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(&poly_refs, &setup)
                .expect("commitment");

        let alpha_bits = D.trailing_zeros() as usize;
        let outer_point = &point[alpha_bits..];
        let ring_opening_point = ring_opening_point_from_field(
            outer_point,
            level_params.r_vars,
            level_params.m_vars,
            BasisMode::Lagrange,
            BlockOrder::RowMajor,
        )
        .expect("ring opening point");
        let gamma: Vec<F> = (0..num_claims)
            .map(|idx| F::from_u64(11 + 7 * idx as u64))
            .collect();
        let mut combined_y = CyclotomicRing::<F, D>::zero();
        let mut w_folded_by_poly = Vec::with_capacity(num_claims);
        for (poly, &gamma_i) in polys.iter().zip(gamma.iter()) {
            let (y_ring, w_folded) = poly.evaluate_and_fold(
                &ring_opening_point.b,
                &ring_opening_point.a,
                level_params.block_len,
            );
            combined_y += y_ring.scale(&gamma_i);
            w_folded_by_poly.push(w_folded);
        }
        let claim_to_point = vec![0usize; num_claims];
        let claim_group_sizes = [num_claims];

        let mut transcript = Blake2bTranscript::<F>::new(b"prepared-m-eval-test");
        commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &combined_y);

        let mut quad_eq = QuadraticEquation::<F, D>::new_prover(
            &setup.ntt_shared,
            vec![ring_opening_point.clone()],
            claim_to_point.clone(),
            &poly_refs,
            w_folded_by_poly,
            &claim_group_sizes,
            level_params.clone(),
            vec![batched_hint],
            &mut transcript,
            std::slice::from_ref(&commitment),
            std::slice::from_ref(&combined_y),
            gamma.clone(),
            setup.expanded.seed.max_stride,
        )
        .expect("quadratic equation");

        ring_switch_build_w::<F, D>(
            &mut quad_eq,
            &setup.expanded,
            &setup.ntt_shared,
            &level_params,
        )
        .expect("ring-switch witness");

        let alpha = F::from_u64(42);
        let alpha_evals_y = scalar_powers(alpha, D);
        let rows = level_params.m_row_count(1, 1);
        let num_i = rows.next_power_of_two().trailing_zeros() as usize;
        let tau1: Vec<F> = (0..num_i)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let m_evals_x = compute_m_evals_x::<F, D>(
            &setup.expanded,
            &[ring_opening_point.clone()],
            &claim_to_point,
            &quad_eq.challenges,
            alpha,
            &alpha_evals_y,
            &level_params,
            &tau1,
            &claim_group_sizes,
            &gamma,
            1,
        )
        .expect("m evals (materialized)");

        let x_challenges: Vec<F> = (0..m_evals_x.len().trailing_zeros() as usize)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let expected = multilinear_eval(&m_evals_x, &x_challenges).expect("multilinear_eval");

        let prepared = prepare_m_eval::<F, D>(
            &quad_eq.challenges,
            alpha,
            &level_params,
            &tau1,
            &claim_group_sizes,
            &gamma,
            1,
            1,
            &claim_to_point,
        )
        .expect("prepare_m_eval");

        let got = prepared
            .eval_at_point::<D>(
                &x_challenges,
                &setup.expanded,
                std::slice::from_ref(&ring_opening_point),
                alpha,
            )
            .expect("eval_at_point");
        let split = prepared
            .eval_split_at_point::<D>(
                &x_challenges,
                &setup.expanded,
                std::slice::from_ref(&ring_opening_point),
                alpha,
            )
            .expect("eval_split_at_point");

        assert_eq!(
            got, expected,
            "PreparedMEval::eval_at_point must match materialized multilinear_eval"
        );
        assert_eq!(
            split.combined(),
            expected,
            "PreparedMEval split terms must recombine to materialized multilinear_eval"
        );

        let split_table = prepared
            .split_eval_table::<D>(
                &setup.expanded,
                std::slice::from_ref(&ring_opening_point),
                alpha,
            )
            .expect("split table");
        let combined_table: Vec<F> = split_table.iter().map(|split| split.combined()).collect();
        assert_eq!(
            combined_table, m_evals_x,
            "PreparedMEval split table must recombine to materialized M-eval table"
        );

        {
            use akita_sumcheck::{
                prove_sumcheck, verify_sumcheck, EqWeightedTableProver, EqWeightedTableVerifier,
                SumcheckInstanceProver, WeightedTableProver, WeightedTableVerifier,
            };

            let setup_weights = prepared
                .setup_weight_table_at_point::<D>(&x_challenges, &setup.expanded, alpha)
                .expect("setup weights");
            let row_count = level_params
                .a_key
                .row_len()
                .max(level_params.b_key.row_len())
                .max(level_params.d_key.row_len())
                .max(1);
            let col_count = setup.expanded.seed.max_stride.max(1);
            let setup_view = setup
                .expanded
                .shared_matrix
                .setup_polynomial_view::<D>(row_count, col_count);
            let row_bits = setup_view.row_bits();
            let col_bits = setup_view.col_bits();
            let setup_table: Vec<F> = (0..setup_weights.len())
                .map(|idx| {
                    let row = idx & ((1usize << row_bits) - 1);
                    let col = (idx >> row_bits) & ((1usize << col_bits) - 1);
                    let coeff = idx >> (row_bits + col_bits);
                    setup_view.coeff(row, col, coeff)
                })
                .collect();
            let setup_weight_claim = setup_table
                .iter()
                .zip(setup_weights.iter())
                .fold(F::zero(), |acc, (&setup, &weight)| acc + setup * weight);
            assert_eq!(
                setup_weight_claim, split.setup,
                "setup-variable weights must reproduce split setup contribution"
            );
            let mut weighted_prover =
                WeightedTableProver::new(setup_table.clone(), setup_weights.clone())
                    .expect("weighted setup prover");
            let weighted_claim = weighted_prover.input_claim();
            let mut weighted_prover_transcript =
                Blake2bTranscript::<F>::new(b"prepared-m-eval-weighted-setup-claim");
            let (weighted_proof, weighted_prover_challenges, _) = prove_sumcheck::<F, _, F, _, _>(
                &mut weighted_prover,
                &mut weighted_prover_transcript,
                |tr| tr.challenge_scalar(akita_transcript::labels::CHALLENGE_SUMCHECK_ROUND),
            )
            .expect("prove weighted setup claim");
            let weighted_verifier =
                WeightedTableVerifier::new(setup_table, setup_weights, weighted_claim)
                    .expect("weighted setup verifier");
            let mut weighted_verifier_transcript =
                Blake2bTranscript::<F>::new(b"prepared-m-eval-weighted-setup-claim");
            let weighted_verifier_challenges = verify_sumcheck::<F, _, F, _, _>(
                &weighted_proof,
                &weighted_verifier,
                &mut weighted_verifier_transcript,
                |tr| tr.challenge_scalar(akita_transcript::labels::CHALLENGE_SUMCHECK_ROUND),
            )
            .expect("verify weighted setup claim");
            assert_eq!(weighted_verifier_challenges, weighted_prover_challenges);

            let setup_table: Vec<F> = split_table.iter().map(|split| split.setup).collect();
            let scale = F::from_u64(19);
            let mut setup_prover =
                EqWeightedTableProver::new(setup_table.clone(), &x_challenges, scale)
                    .expect("setup claim prover");
            let setup_claim = setup_prover.input_claim();
            assert_eq!(
                setup_claim,
                scale * split.setup,
                "setup claim must match the split setup contribution at x"
            );
            let mut prover_transcript = Blake2bTranscript::<F>::new(b"prepared-m-eval-setup-claim");
            let (setup_proof, prover_challenges, _) =
                prove_sumcheck::<F, _, F, _, _>(&mut setup_prover, &mut prover_transcript, |tr| {
                    tr.challenge_scalar(akita_transcript::labels::CHALLENGE_SUMCHECK_ROUND)
                })
                .expect("prove setup claim");

            let setup_verifier =
                EqWeightedTableVerifier::new(setup_table, x_challenges.clone(), setup_claim, scale)
                    .expect("setup claim verifier");
            let mut verifier_transcript =
                Blake2bTranscript::<F>::new(b"prepared-m-eval-setup-claim");
            let verifier_challenges = verify_sumcheck::<F, _, F, _, _>(
                &setup_proof,
                &setup_verifier,
                &mut verifier_transcript,
                |tr| tr.challenge_scalar(akita_transcript::labels::CHALLENGE_SUMCHECK_ROUND),
            )
            .expect("verify setup claim");
            assert_eq!(verifier_challenges, prover_challenges);
        }
    }

    #[test]
    fn prepared_m_eval_matches_materialized() {
        // `D128Full` defaults to CR-on with tier_shrink=2 per audit B-1;
        // the cascade is infeasible at NV=12. Use `BareCfg` to bypass
        // the cascade: this test only exercises the M-eval matching
        // invariant, which is independent of CR routing.
        type Cfg = BareCfg<fp128::D128Full>;
        const NV: usize = 12;
        let level_params = Cfg::commitment_layout(NV).expect("commitment layout");
        assert_prepared_m_eval_matches_materialized(level_params, 1);
    }

    #[test]
    fn prepared_m_eval_batched_same_point_matches_materialized() {
        type Cfg = BareCfg<fp128::D128Full>;
        const NV: usize = 12;
        const NUM_CLAIMS: usize = 4;
        let level_params =
            Cfg::get_params_for_commitment(NV, NUM_CLAIMS).expect("batched commitment layout");
        assert_prepared_m_eval_matches_materialized(level_params, NUM_CLAIMS);
    }

    #[test]
    fn prepared_m_eval_tensor_matches_materialized() {
        type Cfg = BareCfg<fp128::D128Full>;
        const NV: usize = 12;
        let level_params = Cfg::commitment_layout(NV)
            .expect("commitment layout")
            .with_tensor_stage1_challenges();
        assert_prepared_m_eval_matches_materialized(level_params, 1);
    }

    fn assert_setup_claim_reduction_roundtrip(level_params: akita_types::LevelParams) {
        type F = fp128::Field;
        // `D128Full` defaults to CR-on with tier_shrink=2 per audit B-1;
        // the cascade is infeasible at NV=12. The CR sumcheck under test
        // is constructed directly from `level_params`/setup matrix, so
        // route through `BareCfg` to pick up the un-cascaded schedule.
        type Cfg = BareCfg<fp128::D128Full>;
        const D: usize = fp128::D128Full::D;
        const NV: usize = 12;

        let mut rng = StdRng::seed_from_u64(0xc1a1_de5e);
        let evals: Vec<F> = (0..(1usize << NV))
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(NV, &evals).expect("dense poly");
        let point: Vec<F> = (0..NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let (commitment, batched_hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<
            F,
            D,
        >>::commit(&[poly.clone()], &setup)
        .expect("commitment");

        let alpha_bits = D.trailing_zeros() as usize;
        let outer_point = &point[alpha_bits..];
        let ring_opening_point = ring_opening_point_from_field(
            outer_point,
            level_params.r_vars,
            level_params.m_vars,
            BasisMode::Lagrange,
            BlockOrder::RowMajor,
        )
        .expect("ring opening point");
        let (y_ring, w_folded) = poly.evaluate_and_fold(
            &ring_opening_point.b,
            &ring_opening_point.a,
            level_params.block_len,
        );

        let mut transcript = Blake2bTranscript::<F>::new(b"setup-claim-reduction-roundtrip");
        commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);

        let mut quad_eq = QuadraticEquation::<F, D>::new_prover(
            &setup.ntt_shared,
            vec![ring_opening_point.clone()],
            vec![0usize],
            &[&poly],
            vec![w_folded],
            &[1usize],
            level_params.clone(),
            vec![batched_hint],
            &mut transcript,
            std::slice::from_ref(&commitment),
            std::slice::from_ref(&y_ring),
            vec![F::one()],
            setup.expanded.seed.max_stride,
        )
        .expect("quadratic equation");

        ring_switch_build_w::<F, D>(
            &mut quad_eq,
            &setup.expanded,
            &setup.ntt_shared,
            &level_params,
        )
        .expect("ring-switch witness");

        let alpha = F::from_u64(99);
        let rows = level_params.m_row_count(1, 1);
        let num_i = rows.next_power_of_two().trailing_zeros() as usize;
        let tau1: Vec<F> = (0..num_i)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let alpha_evals_y = scalar_powers(alpha, D);
        let m_evals_x = compute_m_evals_x::<F, D>(
            &setup.expanded,
            &[ring_opening_point.clone()],
            &[0usize],
            &quad_eq.challenges,
            alpha,
            &alpha_evals_y,
            &level_params,
            &tau1,
            &[1usize],
            &[F::one()],
            1,
        )
        .expect("m evals (materialized)");

        let prepared = prepare_m_eval::<F, D>(
            &quad_eq.challenges,
            alpha,
            &level_params,
            &tau1,
            &[1usize],
            &[F::one()],
            1,
            1,
            &[],
        )
        .expect("prepare_m_eval");

        let x_challenges: Vec<F> = (0..m_evals_x.len().trailing_zeros() as usize)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let claim_scale = F::from_u64(17);

        let mut prover_tr = Blake2bTranscript::<F>::new(b"setup-claim-reduction-rt");
        let prover_out = akita_prover::protocol::prove_setup_claim_reduction::<F, _, D>(
            &prepared,
            &setup.expanded,
            &x_challenges,
            alpha,
            claim_scale,
            &mut prover_tr,
        )
        .expect("prove setup claim reduction");

        let payload = akita_types::SetupClaimReductionPayload {
            m_setup_eval: prover_out.input_claim,
            s_opening_value: prover_out.s_opening_value,
            sumcheck: prover_out.proof.clone(),
        };
        let mut verifier_tr = Blake2bTranscript::<F>::new(b"setup-claim-reduction-rt");
        let (verifier_challenges, verifier_s_opening_value) =
            akita_verifier::verify_setup_claim_reduction::<F, _, D>(
                &prepared,
                &setup.expanded,
                &x_challenges,
                alpha,
                claim_scale,
                &payload,
                &mut verifier_tr,
                false,
            )
            .expect("verify setup claim reduction");
        assert_eq!(verifier_challenges, prover_out.challenges);
        assert_eq!(verifier_s_opening_value, prover_out.s_opening_value);
    }

    #[test]
    fn setup_claim_reduction_roundtrip_flat() {
        type Cfg = BareCfg<fp128::D128Full>;
        const NV: usize = 12;
        let level_params = Cfg::commitment_layout(NV).expect("commitment layout");
        assert_setup_claim_reduction_roundtrip(level_params);
    }

    #[test]
    fn setup_claim_reduction_roundtrip_tensor() {
        type Cfg = BareCfg<fp128::D128Full>;
        const NV: usize = 12;
        let level_params = Cfg::commitment_layout(NV)
            .expect("commitment layout")
            .with_tensor_stage1_challenges();
        assert_setup_claim_reduction_roundtrip(level_params);
    }

    #[test]
    #[ignore = "diagnostic measurement; run explicitly with --ignored --nocapture"]
    fn measure_setup_claim_reduction_row_coeff_vs_raw_full_table() {
        use akita_sumcheck::multilinear_eval;
        use akita_verifier::materialize_setup_claim_tables;
        use std::time::Instant;

        type F = fp128::Field;
        type Cfg = BareCfg<fp128::D128Full>;
        const D: usize = fp128::D128Full::D;
        const NV: usize = 12;

        let level_params = Cfg::commitment_layout(NV)
            .expect("commitment layout")
            .with_tensor_stage1_challenges();
        let mut rng = StdRng::seed_from_u64(0xc1a1_4d35);
        let evals: Vec<F> = (0..(1usize << NV))
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(NV, &evals).expect("dense poly");
        let point: Vec<F> = (0..NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let (commitment, batched_hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<
            F,
            D,
        >>::commit(&[poly.clone()], &setup)
        .expect("commitment");

        let alpha_bits = D.trailing_zeros() as usize;
        let ring_opening_point = ring_opening_point_from_field(
            &point[alpha_bits..],
            level_params.r_vars,
            level_params.m_vars,
            BasisMode::Lagrange,
            BlockOrder::RowMajor,
        )
        .expect("ring opening point");
        let (y_ring, w_folded) = poly.evaluate_and_fold(
            &ring_opening_point.b,
            &ring_opening_point.a,
            level_params.block_len,
        );
        let mut transcript = Blake2bTranscript::<F>::new(b"setup-claim-reduction-measure");
        commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);
        let mut quad_eq = QuadraticEquation::<F, D>::new_prover(
            &setup.ntt_shared,
            vec![ring_opening_point.clone()],
            vec![0usize],
            &[&poly],
            vec![w_folded],
            &[1usize],
            level_params.clone(),
            vec![batched_hint],
            &mut transcript,
            std::slice::from_ref(&commitment),
            std::slice::from_ref(&y_ring),
            vec![F::one()],
            setup.expanded.seed.max_stride,
        )
        .expect("quadratic equation");
        ring_switch_build_w::<F, D>(
            &mut quad_eq,
            &setup.expanded,
            &setup.ntt_shared,
            &level_params,
        )
        .expect("ring-switch witness");

        let alpha = F::from_u64(99);
        let rows = level_params.m_row_count(1, 1);
        let tau1: Vec<F> = (0..rows.next_power_of_two().trailing_zeros() as usize)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let alpha_evals_y = scalar_powers(alpha, D);
        let m_evals_x = compute_m_evals_x::<F, D>(
            &setup.expanded,
            &[ring_opening_point],
            &[0usize],
            &quad_eq.challenges,
            alpha,
            &alpha_evals_y,
            &level_params,
            &tau1,
            &[1usize],
            &[F::one()],
            1,
        )
        .expect("m evals");
        let prepared = prepare_m_eval::<F, D>(
            &quad_eq.challenges,
            alpha,
            &level_params,
            &tau1,
            &[1usize],
            &[F::one()],
            1,
            1,
            &[],
        )
        .expect("prepare_m_eval");
        let x_challenges: Vec<F> = (0..m_evals_x.len().trailing_zeros() as usize)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let claim_scale = F::from_u64(17);

        let start = Instant::now();
        let raw_weights = prepared
            .setup_weight_table_at_point::<D>(&x_challenges, &setup.expanded, alpha)
            .expect("raw weights");
        let raw_weight_elapsed = start.elapsed();

        let (raw_rows, raw_cols, raw_coeffs) =
            prepared.setup_polynomial_padded_dims(setup.expanded.seed.max_stride);
        let raw_table_start = Instant::now();
        let raw_table = setup
            .expanded
            .shared_matrix
            .setup_polynomial_view_with_stride::<D>(
                prepared.setup_polynomial_row_count(),
                1usize << raw_cols,
                setup.expanded.seed.max_stride.max(1usize << raw_cols),
            )
            .materialize_table();
        let raw_table_elapsed = raw_table_start.elapsed();
        let r_raw: Vec<F> = (0..raw_rows + raw_cols + raw_coeffs)
            .map(|i| F::from_u64(401 + i as u64))
            .collect();
        let raw_eval_start = Instant::now();
        let raw_eval = multilinear_eval(&raw_weights, &r_raw).expect("raw mle");
        let raw_eval_elapsed = raw_eval_start.elapsed();

        let new_start = Instant::now();
        let (new_weights, new_table) = materialize_setup_claim_tables::<F, D>(
            &prepared,
            &x_challenges,
            &setup.expanded,
            alpha,
            claim_scale,
        )
        .expect("new row|coeff tables");
        let new_elapsed = new_start.elapsed();
        let (new_row_bits, new_coeff_bits) = prepared.setup_claim_reduction_dims();
        let r_new: Vec<F> = (0..new_row_bits + new_coeff_bits)
            .map(|i| F::from_u64(401 + i as u64))
            .collect();
        let new_eval_start = Instant::now();
        let new_eval = multilinear_eval(&new_weights, &r_new).expect("new mle");
        let new_eval_elapsed = new_eval_start.elapsed();

        eprintln!(
            "setup CR measurement: raw_dims=row_bits {raw_rows}, col_bits {raw_cols}, coeff_bits {raw_coeffs}; \
             new_dims=row_bits {new_row_bits}, coeff_bits {new_coeff_bits}; raw_len={} new_len={} len_ratio={:.2}x",
            raw_weights.len(),
            new_weights.len(),
            raw_weights.len() as f64 / new_weights.len() as f64
        );
        eprintln!(
            "setup CR measurement: raw_weight={raw_weight_elapsed:?}, raw_table={raw_table_elapsed:?}, \
             raw_eval={raw_eval_elapsed:?}; new_tables={new_elapsed:?}, new_eval={new_eval_elapsed:?}"
        );
        assert_eq!(raw_weights.len(), raw_table.len());
        assert_eq!(new_weights.len(), new_table.len());
        assert!(!raw_eval.is_zero() || !new_eval.is_zero());
    }
}
