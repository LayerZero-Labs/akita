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
    use akita_config::CommitmentConfig;
    use akita_pcs::AkitaCommitmentScheme;
    use akita_pcs::{CanonicalField, CommitmentProver, Transcript};
    use akita_prover::backend::DenseView;
    use akita_prover::compute::{OpeningFoldKernel, OpeningFoldPlan, RootOpeningSource};
    use akita_prover::protocol::ring_switch::{
        build_w_evals_compact, compute_m_evals_x, ring_switch_build_w,
    };
    use akita_prover::{
        ComputeBackendSetup, CpuBackend, DensePoly, ProverCommitmentGroup, ProverOpeningBatch,
        RingRelationProver,
    };
    use akita_transcript::labels::{ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS};
    use akita_transcript::AkitaTranscript;
    use akita_types::relation_claim_from_rows;
    use akita_types::AppendToTranscript;
    use akita_types::{
        ring_opening_point_from_field, AkitaCommitmentHint, BasisMode, BlockOrder, MRowLayout,
        PointVariableSelection, RingCommitment, RingMultiplierOpeningPoint,
    };
    use akita_verifier::{prepare_ring_switch_row_eval, RingSwitchReplay};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::array::from_fn;

    use akita_pcs::{FieldCore, FromPrimitiveInt, RandomSampling};

    fn prover_fold_claims<'a, F: FieldCore + Clone, P, const D: usize>(
        point: &'a [F],
        polynomials: &'a [&'a P],
        commitment: &'a RingCommitment<F, D>,
        hint: AkitaCommitmentHint<F, D>,
    ) -> ProverOpeningBatch<'a, F, P, F, D> {
        ProverOpeningBatch {
            point: point.into(),
            groups: vec![ProverCommitmentGroup {
                point_vars: PointVariableSelection::prefix(point.len(), point.len())
                    .expect("full-point prover group"),
                polynomials,
                commitment: (commitment.clone(), hint),
            }],
        }
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

        let lp = Cfg::get_params_for_batched_commitment(
            &akita_types::OpeningBatchShape::new(NV, 1).expect("singleton opening batch"),
        )
        .expect("lp");

        let mut rng = StdRng::seed_from_u64(0x5151_5eed);
        let evals: Vec<F> = (0..(1usize << NV))
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(NV, &evals).expect("dense poly");
        let point = vec![F::zero(); NV];

        let setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(NV, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let (commitment, batched_hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<
            F,
            D,
        >>::commit(
            &setup, std::slice::from_ref(&poly), &stack
        )
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
        let opening = OpeningFoldKernel::<DenseView<'_, F, D>, F, D>::evaluate_and_fold(
            &CpuBackend,
            None,
            poly.opening_view().expect("opening view"),
            OpeningFoldPlan::Ring {
                eval_outer_scalars: ring_multiplier_point
                    .b_rings()
                    .expect("nonconstant test point has ring b weights"),
                fold_scalars: ring_multiplier_point
                    .a_rings()
                    .expect("nonconstant test point has ring a weights"),
                block_len: lp.block_len,
            },
        )
        .expect("evaluate_and_fold_ring");
        let e_folded = opening.folded;

        let mut transcript = AkitaTranscript::<F>::new(b"ring-switch-ring-multiplier-regression");
        commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        let op_ctx =
            akita_prover::OperationCtx::new(&CpuBackend, &prepared, setup.expanded.as_ref())
                .expect("operation ctx");
        let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
        let fold_claims = prover_fold_claims(&point, &poly_refs, &commitment, batched_hint);
        let (instance, witness) =
            RingRelationProver::new::<F, F, D, _, DensePoly<F, D>, CpuBackend, CpuBackend>(
                &op_ctx,
                &op_ctx,
                ring_opening_point,
                ring_multiplier_point.clone(),
                fold_claims,
                vec![e_folded],
                lp.clone(),
                &mut transcript,
                vec![CyclotomicRing::<F, D>::one()],
                MRowLayout::WithDBlock,
                None,
            )
            .expect("ring relation");

        let build_output =
            ring_switch_build_w::<F, CpuBackend, D>(&instance, witness, &op_ctx, &lp, false)
                .expect("ring-switch witness");
        let (w_compact, _col_bits, ring_bits) =
            build_w_evals_compact(build_output.w.as_i8_digits(), D, 1).expect("compact witness");
        let live_x_cols = w_compact.len() >> ring_bits;

        let alpha = F::from_u64(29);
        let alpha_evals_y = scalar_powers(alpha, D);
        let rows = lp
            .m_row_count_for(1, MRowLayout::WithDBlock)
            .expect("valid row count");
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
            let m_evals_x = compute_m_evals_x::<F, F, D>(
                &setup.expanded,
                instance.opening_point(),
                &ring_multiplier_point,
                &instance.challenges,
                alpha,
                &alpha_evals_y,
                &lp,
                &tau1,
                1,
                &[F::one()],
                MRowLayout::WithDBlock,
            )
            .expect("m evals");
            let got = direct_relation_claim(&w_compact, &alpha_evals_y, &m_evals_x, live_x_cols);
            let expected = relation_claim_from_rows::<F, D>(
                &tau1,
                alpha,
                lp.a_key.row_len(),
                &instance.v,
                &commitment.u,
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

        let lp = Cfg::get_params_for_batched_commitment(
            &akita_types::OpeningBatchShape::new(NV, 1).expect("singleton opening batch"),
        )
        .expect("lp");

        let mut rng = StdRng::seed_from_u64(0x5eed_cafe);
        let evals: Vec<F> = (0..(1usize << NV))
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(NV, &evals).expect("dense poly");
        let point: Vec<F> = (0..NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(NV, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let (commitment, batched_hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<
            F,
            D,
        >>::commit(
            &setup, std::slice::from_ref(&poly), &stack
        )
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
        let opening = OpeningFoldKernel::<DenseView<'_, F, D>, F, D>::evaluate_and_fold(
            &CpuBackend,
            None,
            poly.opening_view().expect("opening view"),
            OpeningFoldPlan::Base {
                eval_outer_scalars: &ring_opening_point.b,
                fold_scalars: &ring_opening_point.a,
                block_len: lp.block_len,
            },
        )
        .expect("evaluate_and_fold");
        let e_folded = opening.folded;

        let mut transcript = AkitaTranscript::<F>::new(b"ring-switch-row-regression");
        commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        let op_ctx =
            akita_prover::OperationCtx::new(&CpuBackend, &prepared, setup.expanded.as_ref())
                .expect("operation ctx");
        let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
        let fold_claims = prover_fold_claims(&point, &poly_refs, &commitment, batched_hint);
        let (instance, witness) =
            RingRelationProver::new::<F, F, D, _, DensePoly<F, D>, CpuBackend, CpuBackend>(
                &op_ctx,
                &op_ctx,
                ring_opening_point,
                ring_multiplier_point.clone(),
                fold_claims,
                vec![e_folded],
                lp.clone(),
                &mut transcript,
                vec![CyclotomicRing::<F, D>::one()],
                MRowLayout::WithDBlock,
                None,
            )
            .expect("ring relation");

        let build_output =
            ring_switch_build_w::<F, CpuBackend, D>(&instance, witness, &op_ctx, &lp, false)
                .expect("ring-switch witness");
        let (w_compact, _col_bits, ring_bits) =
            build_w_evals_compact(build_output.w.as_i8_digits(), D, 1).expect("compact witness");
        let live_x_cols = w_compact.len() >> ring_bits;

        let alpha = F::from_u64(17);
        let alpha_evals_y = scalar_powers(alpha, D);
        let rows = lp.m_row_count_for(1, MRowLayout::WithDBlock).unwrap();
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
            let m_evals_x = compute_m_evals_x::<F, F, D>(
                &setup.expanded,
                instance.opening_point(),
                &ring_multiplier_point,
                &instance.challenges,
                alpha,
                &alpha_evals_y,
                &lp,
                &tau1,
                1,
                &[F::one()],
                MRowLayout::WithDBlock,
            )
            .expect("m evals");
            let got = direct_relation_claim(&w_compact, &alpha_evals_y, &m_evals_x, live_x_cols);
            let expected = relation_claim_from_rows::<F, D>(
                &tau1,
                alpha,
                lp.a_key.row_len(),
                &instance.v,
                &commitment.u,
            )
            .unwrap();
            assert_eq!(got, expected, "row {row} mismatch");
        }
    }

    #[test]
    fn asymmetric_centering_decompose_roundtrip() {
        use akita_types::sis::compute_num_digits_full_field;
        use rand::SeedableRng;

        type F = fp128::Field;
        const D: usize = 64;

        let mut rng = rand::rngs::StdRng::seed_from_u64(0xA11A_D15C_A11A_D15C);

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
        use akita_types::{r_decomp_levels, ChunkedWitnessCfg};

        type F = fp128::Field;
        type Cfg = fp128::D128Full;
        const D: usize = Cfg::D;
        const NV: usize = 12;

        let level_params = Cfg::get_params_for_batched_commitment(
            &akita_types::OpeningBatchShape::new(NV, 1).expect("singleton opening batch"),
        )
        .expect("commitment layout");

        let mut rng = StdRng::seed_from_u64(0xdead_beef);
        let evals: Vec<F> = (0..(1usize << NV))
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(NV, &evals).expect("dense poly");
        let point: Vec<F> = (0..NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(NV, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let (commitment, batched_hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<
            F,
            D,
        >>::commit(
            &setup, std::slice::from_ref(&poly), &stack
        )
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
        let opening = OpeningFoldKernel::<DenseView<'_, F, D>, F, D>::evaluate_and_fold(
            &CpuBackend,
            None,
            poly.opening_view().expect("opening view"),
            OpeningFoldPlan::Base {
                eval_outer_scalars: &ring_opening_point.b,
                fold_scalars: &ring_opening_point.a,
                block_len: level_params.block_len,
            },
        )
        .expect("evaluate_and_fold");
        let e_folded = opening.folded;

        let mut transcript = AkitaTranscript::<F>::new(b"prepared-m-eval-test");
        commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        let op_ctx =
            akita_prover::OperationCtx::new(&CpuBackend, &prepared, setup.expanded.as_ref())
                .expect("operation ctx");
        let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
        let fold_claims = prover_fold_claims(&point, &poly_refs, &commitment, batched_hint);
        let (instance, witness) =
            RingRelationProver::new::<F, F, D, _, DensePoly<F, D>, CpuBackend, CpuBackend>(
                &op_ctx,
                &op_ctx,
                ring_opening_point.clone(),
                ring_multiplier_point.clone(),
                fold_claims,
                vec![e_folded],
                level_params.clone(),
                &mut transcript,
                vec![CyclotomicRing::<F, D>::one()],
                MRowLayout::WithDBlock,
                None,
            )
            .expect("ring relation");

        ring_switch_build_w::<F, CpuBackend, D>(&instance, witness, &op_ctx, &level_params, false)
            .expect("ring-switch witness");

        let alpha = F::from_u64(42);
        let alpha_evals_y = scalar_powers(alpha, D);
        let rows = level_params
            .m_row_count_for(1, MRowLayout::WithDBlock)
            .unwrap();
        let num_i = rows.next_power_of_two().trailing_zeros() as usize;
        let tau1: Vec<F> = (0..num_i)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let m_evals_x = compute_m_evals_x::<F, F, D>(
            &setup.expanded,
            &ring_opening_point,
            &ring_multiplier_point,
            &instance.challenges,
            alpha,
            &alpha_evals_y,
            &level_params,
            &tau1,
            1,
            &[F::one()],
            MRowLayout::WithDBlock,
        )
        .expect("m evals (materialized)");

        let x_challenges: Vec<F> = (0..m_evals_x.len().trailing_zeros() as usize)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let expected = multilinear_eval(&m_evals_x, &x_challenges).expect("multilinear_eval");

        let gamma = [F::one()];
        let replay = RingSwitchReplay {
            relation: &instance,
            row_coefficients: &gamma,
            lp: &level_params,
        };
        let prepared = prepare_ring_switch_row_eval::<F, F, D>(&replay, alpha, &tau1, None)
            .expect("prepare_ring_switch_row_eval");

        let got = prepared
            .eval_at_point::<F, D>(
                &x_challenges,
                &setup.expanded,
                &ring_opening_point,
                &ring_multiplier_point,
                alpha,
                None,
            )
            .expect("eval_at_point");

        assert_eq!(
            got, expected,
            "RingSwitchDeferredRowEval::eval_at_point must match materialized multilinear_eval"
        );

        // ----- Chunked layout ground truth (W ∈ powers of two | num_blocks) --
        // The chunked relation's column-MLE has the *same per-cell values* as
        // the single-chunk `m_evals_x`, only repositioned into the
        // `[z|e_j|t_j]…[r]` layout (z replicated, e/t partitioned by global
        // block). Rearranging the trusted single-chunk vector therefore yields
        // an independent reference for the verifier's chunked `eval_at_point`.
        let num_blocks = level_params.num_blocks;
        let block_len = level_params.block_len;
        let depth_open = level_params.num_digits_open;
        let depth_commit = level_params.num_digits_commit;
        let depth_fold = level_params
            .num_digits_fold(1, <F as akita_field::CanonicalField>::modulus_bits())
            .unwrap();
        let n_a = level_params.a_key.row_len();
        let num_claims = 1usize;
        let num_t_vectors = 1usize;
        let z_len = depth_fold * depth_commit * block_len;
        let e_len = depth_open * num_claims * num_blocks;
        let t_len = depth_open * n_a * num_t_vectors * num_blocks;
        let r_tail_len = rows * r_decomp_levels::<F>(level_params.log_basis);

        // Single-chunk segments (z ‖ e ‖ t ‖ r; no tiered u in this fixture).
        let z_seg = m_evals_x[0..z_len].to_vec();
        let e_seg = m_evals_x[z_len..z_len + e_len].to_vec();
        let t_seg = m_evals_x[z_len + e_len..z_len + e_len + t_len].to_vec();
        let r_off1 = z_len + e_len + t_len;
        let r_seg = m_evals_x[r_off1..r_off1 + r_tail_len].to_vec();

        let chunk_counts: Vec<usize> = (0..)
            .map(|k| 1usize << k)
            .take_while(|&w| w <= num_blocks)
            .filter(|&w| num_blocks % w == 0)
            .collect();
        assert!(
            chunk_counts.iter().any(|&w| w > 1),
            "fixture must have num_blocks > 1 to exercise chunking (num_blocks={num_blocks})"
        );
        for w in chunk_counts.into_iter().filter(|&w| w > 1) {
            let bpc = num_blocks / w;
            let mut chunked: Vec<F> = Vec::new();
            for j in 0..w {
                // z_j: replicated full single-chunk fold response.
                chunked.extend_from_slice(&z_seg);
                // e_j: window of the block axis, order (digit, claim, block).
                for dig in 0..depth_open {
                    for claim in 0..num_claims {
                        for bl in 0..bpc {
                            let gb = j * bpc + bl;
                            let src = (dig * num_claims + claim) * num_blocks + gb;
                            chunked.push(e_seg[src]);
                        }
                    }
                }
                // t_j: window of the block axis, order (a_row, digit, t_vector, block).
                for a_idx in 0..n_a {
                    for digit in 0..depth_open {
                        for tvec in 0..num_t_vectors {
                            for bl in 0..bpc {
                                let compound = a_idx * depth_open + digit;
                                let gb = j * bpc + bl;
                                let src = compound * (num_t_vectors * num_blocks)
                                    + tvec * num_blocks
                                    + gb;
                                chunked.push(t_seg[src]);
                            }
                        }
                    }
                }
            }
            // Single shared r̂ tail after the last chunk.
            chunked.extend_from_slice(&r_seg);
            let x_len_w = chunked.len().next_power_of_two();
            chunked.resize(x_len_w, F::zero());

            let x_challenges_w: Vec<F> = (0..x_len_w.trailing_zeros() as usize)
                .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
                .collect();
            let expected_w = multilinear_eval(&chunked, &x_challenges_w).expect("multilinear_eval");

            let mut lp_w = level_params.clone();
            lp_w.witness_chunk = ChunkedWitnessCfg {
                num_chunks: w,
                num_activated_levels: 1,
            };
            let replay_w = RingSwitchReplay {
                relation: &instance,
                row_coefficients: &gamma,
                lp: &lp_w,
            };
            let prepared_w = prepare_ring_switch_row_eval::<F, F, D>(&replay_w, alpha, &tau1, None)
                .expect("prepare chunked row eval");
            let got_w = prepared_w
                .eval_at_point::<F, D>(
                    &x_challenges_w,
                    &setup.expanded,
                    &ring_opening_point,
                    &ring_multiplier_point,
                    alpha,
                    None,
                )
                .expect("chunked eval_at_point");
            assert_eq!(
                got_w, expected_w,
                "chunked eval_at_point must match materialized chunked row for W={w}"
            );

            // Prover-side cross-check: the chunked `compute_m_evals_x` must emit
            // exactly the rearranged column layout, and its multilinear eval must
            // match the verifier's chunked row eval.
            let prover_chunked = compute_m_evals_x::<F, F, D>(
                &setup.expanded,
                &ring_opening_point,
                &ring_multiplier_point,
                &instance.challenges,
                alpha,
                &alpha_evals_y,
                &lp_w,
                &tau1,
                1,
                &[F::one()],
                0,
                MRowLayout::WithDBlock,
            )
            .expect("chunked m evals (prover)");
            assert_eq!(
                prover_chunked, chunked,
                "prover chunked compute_m_evals_x must equal the rearranged column layout for W={w}"
            );
            let prover_eval =
                multilinear_eval(&prover_chunked, &x_challenges_w).expect("multilinear_eval");
            assert_eq!(
                prover_eval, got_w,
                "prover chunked relation MLE must match verifier chunked row eval for W={w}"
            );
        }
    }

    #[test]
    fn segment_typed_expand_matches_logical_w() {
        use akita_types::{
            build_segment_typed_witness, expand_segment_typed_to_i8_digits,
            ring_opening_point_from_field, BasisMode, BlockOrder,
        };

        type F = fp128::Field;
        type Cfg = fp128::D128Full;
        const D: usize = Cfg::D;
        const NV: usize = 12;

        let level_params = Cfg::get_params_for_batched_commitment(
            &akita_types::OpeningBatchShape::new(NV, 1).expect("singleton opening batch"),
        )
        .expect("commitment layout");

        let mut rng = StdRng::seed_from_u64(0x5E6E_7E8E);
        let evals: Vec<F> = (0..(1usize << NV))
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(NV, &evals).expect("dense poly");
        let point: Vec<F> = (0..NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(NV, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let (commitment, batched_hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<
            F,
            D,
        >>::commit(
            &setup, std::slice::from_ref(&poly), &stack
        )
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
        let opening = OpeningFoldKernel::<DenseView<'_, F, D>, F, D>::evaluate_and_fold(
            &CpuBackend,
            None,
            poly.opening_view().expect("opening view"),
            OpeningFoldPlan::Base {
                eval_outer_scalars: &ring_opening_point.b,
                fold_scalars: &ring_opening_point.a,
                block_len: level_params.block_len,
            },
        )
        .expect("evaluate_and_fold");
        let e_folded = opening.folded;

        let mut transcript = AkitaTranscript::<F>::new(b"segment-typed-expand-test");
        commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        let op_ctx =
            akita_prover::OperationCtx::new(&CpuBackend, &prepared, setup.expanded.as_ref())
                .expect("operation ctx");
        let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
        let fold_claims = prover_fold_claims(&point, &poly_refs, &commitment, batched_hint);
        let (instance, witness) =
            RingRelationProver::new::<F, F, D, _, DensePoly<F, D>, CpuBackend, CpuBackend>(
                &op_ctx,
                &op_ctx,
                ring_opening_point,
                ring_multiplier_point,
                fold_claims,
                vec![e_folded],
                level_params.clone(),
                &mut transcript,
                vec![CyclotomicRing::<F, D>::one()],
                MRowLayout::WithoutDBlock,
                None,
            )
            .expect("ring relation");

        let build_output = ring_switch_build_w::<F, CpuBackend, D>(
            &instance,
            witness,
            &op_ctx,
            &level_params,
            true,
        )
        .expect("ring-switch witness");
        let logical_digits = build_output.w.as_i8_digits();
        let artifacts = build_output
            .terminal_artifacts
            .expect("terminal artifacts retained");
        let segment = build_segment_typed_witness::<D, F>(
            &artifacts.e_folded,
            &artifacts.recomposed_inner_rows,
            &artifacts.z_folded_centered,
            &artifacts.r,
            &level_params,
            1,
            1,
            1,
            1,
        )
        .expect("segment witness");
        let expanded = expand_segment_typed_to_i8_digits::<D, F>(&segment, &level_params, 1)
            .expect("expand segment typed");
        assert_eq!(
            expanded, logical_digits,
            "segment-typed expand must match ring_switch_build_w digit stream"
        );
    }
}
