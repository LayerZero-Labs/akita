//! Ring-switch integration regressions.

#![cfg(not(feature = "zk"))]

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
    use akita_config::CommitmentConfig;
    use akita_pcs::AkitaCommitmentScheme;
    use akita_pcs::{CanonicalField, CommitmentProver, Transcript};
    use akita_prover::protocol::ring_switch::{
        build_w_evals_compact, compute_m_evals_x, ring_switch_build_w,
    };
    use akita_prover::{AkitaPolyOps, DensePoly, QuadraticEquation};
    use akita_transcript::labels::{ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS};
    use akita_transcript::AkitaTranscript;
    use akita_types::relation_claim_from_rows;
    use akita_types::AppendToTranscript;
    use akita_types::{
        ring_opening_point_from_field, BasisMode, BlockOrder, ClaimIncidenceSummary, MRowLayout,
        RingMultiplierOpeningPoint,
    };
    use akita_verifier::prepare_ring_switch_row_eval;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::array::from_fn;

    use akita_pcs::{FieldCore, FromPrimitiveInt, RandomSampling};

    fn single_point_group_incidence(
        num_vars: usize,
        group_poly_count: usize,
    ) -> ClaimIncidenceSummary {
        ClaimIncidenceSummary::from_point_polys(num_vars, vec![group_poly_count])
            .expect("valid single-point incidence")
    }

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

    fn nonconstant_ring_multiplier_point<F, const D: usize>(
        block_len: usize,
        num_blocks: usize,
    ) -> RingMultiplierOpeningPoint<F, D>
    where
        F: FieldCore + FromPrimitiveInt,
    {
        let a = (0..block_len)
            .map(|idx| {
                CyclotomicRing::from_coefficients(from_fn(|k| {
                    if k % 17 == idx % 17 {
                        F::from_u64(((idx + 3 * k + 5) % 11 + 1) as u64)
                    } else {
                        F::zero()
                    }
                }))
            })
            .collect();
        let b = (0..num_blocks)
            .map(|idx| {
                CyclotomicRing::from_coefficients(from_fn(|k| {
                    if k % 19 == idx % 19 {
                        F::from_u64(((2 * idx + k + 7) % 13 + 1) as u64)
                    } else {
                        F::zero()
                    }
                }))
            })
            .collect();
        RingMultiplierOpeningPoint::from_ring(a, b)
    }

    #[test]
    fn ring_multiplier_root_rows_match_direct_relation_claim() {
        type F = fp128::Field;
        type Cfg = fp128::D128Full;
        const D: usize = Cfg::D;
        const NV: usize = 12;

        let lp = Cfg::commitment_layout(NV).expect("lp");

        let mut rng = StdRng::seed_from_u64(0x5151_5eed);
        let evals: Vec<F> = (0..(1usize << NV))
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(NV, &evals).expect("dense poly");
        let point = vec![F::zero(); NV];

        let setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let (commitment, batched_hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<
            F,
            D,
        >>::commit(std::slice::from_ref(&poly), &setup)
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
        let ring_multiplier_point =
            nonconstant_ring_multiplier_point::<F, D>(lp.block_len, lp.num_blocks);
        let (y_ring, w_folded) = poly.evaluate_and_fold_ring(
            ring_multiplier_point
                .b_rings()
                .expect("nonconstant test point has ring b weights"),
            ring_multiplier_point
                .a_rings()
                .expect("nonconstant test point has ring a weights"),
            lp.block_len,
        );

        let mut transcript = AkitaTranscript::<F>::new(b"ring-switch-ring-multiplier-regression");
        commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);
        let incidence_summary = single_point_group_incidence(NV, 1);

        let mut quad_eq = QuadraticEquation::<F, D>::new_prover(
            &setup.ntt_shared,
            vec![ring_opening_point],
            vec![ring_multiplier_point.clone()],
            vec![0usize],
            &[&poly],
            vec![w_folded],
            &incidence_summary,
            lp.clone(),
            vec![batched_hint],
            &mut transcript,
            std::slice::from_ref(&commitment),
            std::slice::from_ref(&y_ring),
            vec![CyclotomicRing::<F, D>::one()],
            setup.expanded.seed.max_stride,
            MRowLayout::Intermediate,
        )
        .expect("quadratic equation");

        let w = ring_switch_build_w::<F, D>(&mut quad_eq, &setup.expanded, &setup.ntt_shared, &lp)
            .expect("ring-switch witness");
        let (w_compact, _col_bits, ring_bits) =
            build_w_evals_compact(w.as_i8_digits(), D, 1).expect("compact witness");
        let live_x_cols = w_compact.len() >> ring_bits;

        let alpha = F::from_u64(29);
        let alpha_evals_y = scalar_powers(alpha, D);
        let rows = lp.m_row_count(1, 1).expect("valid row count");
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
            let m_evals_x = compute_m_evals_x::<F, F, _, D>(
                &setup.expanded,
                &[quad_eq.opening_point().clone()],
                std::slice::from_ref(&ring_multiplier_point),
                &[0usize],
                &quad_eq.challenges,
                alpha,
                &alpha_evals_y,
                &lp,
                &tau1,
                &[1usize],
                &[0usize],
                &[0usize],
                &[F::one()],
                1,
                MRowLayout::Intermediate,
            )
            .expect("m evals");
            let got = direct_relation_claim(&w_compact, &alpha_evals_y, &m_evals_x, live_x_cols);
            let expected = relation_claim_from_rows::<F, D>(
                &tau1,
                alpha,
                &quad_eq.v,
                &commitment.u,
                std::slice::from_ref(&y_ring),
            )
            .expect("relation claim");
            assert_eq!(got, expected, "ring-multiplier row {row} mismatch");
        }
    }

    #[test]
    fn full_root_rows_match_direct_relation_claim() {
        type F = fp128::Field;
        type Cfg = fp128::D128Full;
        const D: usize = Cfg::D;
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
        >>::commit(std::slice::from_ref(&poly), &setup)
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
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&ring_opening_point);
        let (y_ring, w_folded) =
            poly.evaluate_and_fold(&ring_opening_point.b, &ring_opening_point.a, lp.block_len);

        let mut transcript = AkitaTranscript::<F>::new(b"ring-switch-row-regression");
        commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);
        let incidence_summary = single_point_group_incidence(NV, 1);

        let mut quad_eq = QuadraticEquation::<F, D>::new_prover(
            &setup.ntt_shared,
            vec![ring_opening_point],
            vec![ring_multiplier_point.clone()],
            vec![0usize],
            &[&poly],
            vec![w_folded],
            &incidence_summary,
            lp.clone(),
            vec![batched_hint],
            &mut transcript,
            std::slice::from_ref(&commitment),
            std::slice::from_ref(&y_ring),
            vec![CyclotomicRing::<F, D>::one()],
            setup.expanded.seed.max_stride,
            MRowLayout::Intermediate,
        )
        .expect("quadratic equation");

        let w = ring_switch_build_w::<F, D>(&mut quad_eq, &setup.expanded, &setup.ntt_shared, &lp)
            .expect("ring-switch witness");
        let (w_compact, _col_bits, ring_bits) =
            build_w_evals_compact(w.as_i8_digits(), D, 1).expect("compact witness");
        let live_x_cols = w_compact.len() >> ring_bits;

        let alpha = F::from_u64(17);
        let alpha_evals_y = scalar_powers(alpha, D);
        let rows = lp.m_row_count(1, 1).unwrap();
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
            let m_evals_x = compute_m_evals_x::<F, F, _, D>(
                &setup.expanded,
                &[quad_eq.opening_point().clone()],
                std::slice::from_ref(&ring_multiplier_point),
                &[0usize],
                &quad_eq.challenges,
                alpha,
                &alpha_evals_y,
                &lp,
                &tau1,
                &[1usize],
                &[0usize],
                &[0usize],
                &[F::one()],
                1,
                MRowLayout::Intermediate,
            )
            .expect("m evals");
            let got = direct_relation_claim(&w_compact, &alpha_evals_y, &m_evals_x, live_x_cols);
            let expected = relation_claim_from_rows::<F, D>(
                &tau1,
                alpha,
                &quad_eq.v,
                &commitment.u,
                std::slice::from_ref(&y_ring),
            )
            .unwrap();
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

    #[test]
    fn prepared_row_eval_matches_materialized() {
        use akita_sumcheck::multilinear_eval;

        type F = fp128::Field;
        type Cfg = fp128::D128Full;
        const D: usize = Cfg::D;
        const NV: usize = 12;

        let level_params = Cfg::commitment_layout(NV).expect("commitment layout");

        let mut rng = StdRng::seed_from_u64(0xdead_beef);
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
        >>::commit(std::slice::from_ref(&poly), &setup)
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
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&ring_opening_point);
        let (y_ring, w_folded) = poly.evaluate_and_fold(
            &ring_opening_point.b,
            &ring_opening_point.a,
            level_params.block_len,
        );

        let mut transcript = AkitaTranscript::<F>::new(b"prepared-m-eval-test");
        commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);
        let incidence_summary = single_point_group_incidence(NV, 1);

        let mut quad_eq = QuadraticEquation::<F, D>::new_prover(
            &setup.ntt_shared,
            vec![ring_opening_point.clone()],
            vec![ring_multiplier_point.clone()],
            vec![0usize],
            &[&poly],
            vec![w_folded],
            &incidence_summary,
            level_params.clone(),
            vec![batched_hint],
            &mut transcript,
            std::slice::from_ref(&commitment),
            std::slice::from_ref(&y_ring),
            vec![CyclotomicRing::<F, D>::one()],
            setup.expanded.seed.max_stride,
            MRowLayout::Intermediate,
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
        let rows = level_params.m_row_count(1, 1).unwrap();
        let num_i = rows.next_power_of_two().trailing_zeros() as usize;
        let tau1: Vec<F> = (0..num_i)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let m_evals_x = compute_m_evals_x::<F, F, _, D>(
            &setup.expanded,
            std::slice::from_ref(&ring_opening_point),
            std::slice::from_ref(&ring_multiplier_point),
            &[0usize],
            &quad_eq.challenges,
            alpha,
            &alpha_evals_y,
            &level_params,
            &tau1,
            &[1usize],
            &[0usize],
            &[0usize],
            &[F::one()],
            1,
            MRowLayout::Intermediate,
        )
        .expect("m evals (materialized)");

        let x_challenges: Vec<F> = (0..m_evals_x.len().trailing_zeros() as usize)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let expected = multilinear_eval(&m_evals_x, &x_challenges).expect("multilinear_eval");

        let prepared = prepare_ring_switch_row_eval::<F, F, D>(
            &quad_eq.challenges,
            alpha,
            &level_params,
            &tau1,
            &[1usize],
            &[0usize],
            &[0usize],
            &[F::one()],
            1,
            MRowLayout::Intermediate,
            1,
            std::slice::from_ref(&ring_multiplier_point),
            &[0usize],
        )
        .expect("prepare_ring_switch_row_eval");

        let got = prepared
            .eval_at_point::<F, D>(
                &x_challenges,
                &setup.expanded,
                std::slice::from_ref(&ring_opening_point),
                std::slice::from_ref(&ring_multiplier_point),
                alpha,
            )
            .expect("eval_at_point");

        assert_eq!(
            got, expected,
            "RingSwitchDeferredRowEval::eval_at_point must match materialized multilinear_eval"
        );
    }
}
