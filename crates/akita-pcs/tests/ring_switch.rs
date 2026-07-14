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
    use akita_pcs::{CanonicalField, Transcript};
    use akita_prover::backend::DenseView;
    use akita_prover::compute::{OpeningFoldKernel, OpeningFoldPlan, RootOpeningSource};
    use akita_prover::protocol::ring_switch::{
        build_w_evals_compact, compute_relation_weight_evals, ring_switch_build_w,
    };
    use akita_prover::{
        ComputeBackendSetup, CpuBackend, DensePoly, ProverOpeningData, RingRelationProver,
    };
    use akita_transcript::labels::{ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS};
    use akita_transcript::AkitaTranscript;
    use akita_types::relation_claim_from_rows;
    use akita_types::witness::ChunkedWitnessCfg;
    use akita_types::{
        ring_opening_point_from_field, AkitaCommitmentHint, BasisMode, Commitment,
        OpeningBlockLayout, OpeningClaims, PointVariableSelection, PolynomialGroupClaims,
        RelationMatrixRowLayout, RingMultiplierOpeningPoint, RingVec, SemanticGroupId,
    };
    use akita_verifier::{prepare_relation_matrix_evaluator, RingSwitchReplay};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::array::from_fn;

    use akita_pcs::{FieldCore, FromPrimitiveInt, RandomSampling};

    fn prover_fold_claims<'a, F: FieldCore + Clone, P>(
        point: &'a [F],
        polynomials: &'a [&'a P],
        commitment: &'a Commitment<F>,
        hint: AkitaCommitmentHint<F>,
    ) -> ProverOpeningData<'a, F, P, F> {
        let group = PolynomialGroupClaims::new(
            PointVariableSelection::prefix(point.len(), point.len())
                .expect("full-point prover group"),
            vec![F::zero(); polynomials.len()],
            commitment.clone(),
        )
        .expect("valid prover claims group");
        let opening_claims =
            OpeningClaims::from_groups(point.to_vec(), vec![group]).expect("valid prover claims");
        ProverOpeningData::new(opening_claims, vec![hint], vec![polynomials])
            .expect("valid prover opening data")
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
        relation_weight_evals: &[F],
    ) -> F {
        w_compact
            .iter()
            .zip(relation_weight_evals)
            .fold(F::zero(), |acc, (&w, &weight)| {
                acc + F::from_i64(i64::from(w)) * weight
            })
    }

    fn nonconstant_ring_multiplier_point<F, const D: usize>(
        fold_position_count: usize,
        live_fold_count: usize,
    ) -> RingMultiplierOpeningPoint<F>
    where
        F: FieldCore + FromPrimitiveInt,
    {
        let a = (0..fold_position_count)
            .map(|idx| {
                CyclotomicRing::<F, D>::from_coefficients(from_fn(|k| {
                    if k % 17 == idx % 17 {
                        F::from_u64(((idx + 3 * k + 5) % 11 + 1) as u64)
                    } else {
                        F::zero()
                    }
                }))
            })
            .collect();
        let b = (0..live_fold_count)
            .map(|idx| {
                CyclotomicRing::<F, D>::from_coefficients(from_fn(|k| {
                    if k % 19 == idx % 19 {
                        F::from_u64(((2 * idx + k + 7) % 13 + 1) as u64)
                    } else {
                        F::zero()
                    }
                }))
            })
            .collect();
        RingMultiplierOpeningPoint::from_ring::<D>(a, b)
    }

    #[test]
    fn ring_multiplier_root_rows_match_direct_relation_claim() {
        type F = fp128::Field;
        type Cfg = fp128::D128Full;
        const D: usize = Cfg::D;
        const NV: usize = 12;

        let opening_batch =
            akita_types::OpeningClaimsLayout::new(NV, 1).expect("singleton opening batch");
        let lp = Cfg::get_params_for_batched_commitment(&opening_batch).expect("lp");

        let mut rng = StdRng::seed_from_u64(0x5151_5eed);
        let evals: Vec<F> = (0..(1usize << NV))
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F>::from_field_evals(NV, D, &evals).expect("dense poly");
        let point = vec![F::zero(); NV];

        let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(NV, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let (commitment, batched_hint) = AkitaCommitmentScheme::<Cfg>::commit::<_, _>(
            &setup,
            std::slice::from_ref(&poly),
            &stack,
        )
        .expect("commitment");

        let alpha_bits = D.trailing_zeros() as usize;
        let outer_point = &point[alpha_bits..];
        let ring_opening_point = ring_opening_point_from_field(
            outer_point,
            OpeningBlockLayout::new(lp.live_fold_count, lp.fold_position_count).unwrap(),
            BasisMode::Lagrange,
        )
        .expect("ring opening point");
        let ring_multiplier_point =
            nonconstant_ring_multiplier_point::<F, D>(lp.fold_position_count, lp.live_fold_count);
        let opening = OpeningFoldKernel::<DenseView<'_, F, D>, F, D>::evaluate_and_fold(
            &CpuBackend,
            None,
            poly.opening_view().expect("opening view"),
            OpeningFoldPlan::Ring {
                eval_outer_scalars: ring_multiplier_point
                    .b_rings_trusted::<D>()
                    .expect("nonconstant test point has ring b weights")
                    .expect("ring b weights"),
                fold_scalars: ring_multiplier_point
                    .a_rings_trusted::<D>()
                    .expect("nonconstant test point has ring a weights")
                    .expect("ring a weights"),
                fold_position_count: lp.fold_position_count,
            },
        )
        .expect("evaluate_and_fold_ring");
        let e_folded = opening.folded;

        let mut transcript = AkitaTranscript::<F>::new(b"ring-switch-ring-multiplier-regression");
        commitment
            .append_to_transcript(ABSORB_COMMITMENT, D, &mut transcript)
            .expect("commitment transcript");
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        let op_ctx =
            akita_prover::OperationCtx::new(&CpuBackend, &prepared, setup.expanded.as_ref())
                .expect("operation ctx");
        let poly_refs: [&DensePoly<F>; 1] = [&poly];
        let fold_claims = prover_fold_claims(&point, &poly_refs, &commitment, batched_hint);
        let (instance, witness) =
            RingRelationProver::new::<F, F, _, DensePoly<F>, CpuBackend, CpuBackend>(
                &op_ctx,
                &op_ctx,
                ring_opening_point,
                ring_multiplier_point.clone(),
                fold_claims,
                vec![RingVec::from_ring_elems(&e_folded)],
                lp.clone(),
                &mut transcript,
                RingVec::from_single(&CyclotomicRing::<F, D>::one()),
                RelationMatrixRowLayout::WithDBlock,
                None,
            )
            .expect("ring relation");

        let build_output =
            ring_switch_build_w::<F, CpuBackend>(&instance, witness, &op_ctx, &lp, false)
                .expect("ring-switch witness");
        let opening_layout =
            OpeningBlockLayout::new(1, build_output.w.len() / D).expect("opening layout");
        let (w_compact, _col_bits, _ring_bits) =
            build_w_evals_compact(build_output.w.as_i8_digits(), D, 1, opening_layout)
                .expect("compact witness");

        let alpha = F::from_u64(29);
        let alpha_evals_y = scalar_powers(alpha, D);
        let rows = lp
            .relation_matrix_row_count_for(1, RelationMatrixRowLayout::WithDBlock)
            .expect("valid row count");
        let num_i = lp
            .relation_row_index_num_vars_for_layout(
                RelationMatrixRowLayout::WithDBlock,
                &opening_batch,
            )
            .expect("tau1 vars");

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
            let relation_matrix_col_evals = compute_relation_weight_evals::<F, F>(
                &setup.expanded,
                &instance,
                alpha,
                &alpha_evals_y,
                lp.role_dims(),
                &lp,
                &tau1,
                &[F::one()],
                RelationMatrixRowLayout::WithDBlock,
                opening_layout,
                D,
            )
            .expect("m evals");
            let got = direct_relation_claim(&w_compact, &relation_matrix_col_evals);
            let expected = relation_claim_from_rows::<F, D>(
                &tau1,
                alpha,
                lp.a_key.row_len(),
                instance.v_trusted::<D>().expect("v"),
                &commitment
                    .rows()
                    .try_to_vec::<D>()
                    .expect("commitment rows"),
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

        let opening_batch =
            akita_types::OpeningClaimsLayout::new(NV, 1).expect("singleton opening batch");
        let lp = Cfg::get_params_for_batched_commitment(&opening_batch).expect("lp");

        let mut rng = StdRng::seed_from_u64(0x5eed_cafe);
        let evals: Vec<F> = (0..(1usize << NV))
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F>::from_field_evals(NV, D, &evals).expect("dense poly");
        let point: Vec<F> = (0..NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(NV, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let (commitment, batched_hint) = AkitaCommitmentScheme::<Cfg>::commit::<_, _>(
            &setup,
            std::slice::from_ref(&poly),
            &stack,
        )
        .expect("commitment");

        let alpha_bits = D.trailing_zeros() as usize;
        let outer_point = &point[alpha_bits..];
        let ring_opening_point = ring_opening_point_from_field(
            outer_point,
            OpeningBlockLayout::new(lp.live_fold_count, lp.fold_position_count).unwrap(),
            BasisMode::Lagrange,
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
                fold_position_count: lp.fold_position_count,
            },
        )
        .expect("evaluate_and_fold");
        let e_folded = opening.folded;

        let mut transcript = AkitaTranscript::<F>::new(b"ring-switch-row-regression");
        commitment
            .append_to_transcript(ABSORB_COMMITMENT, D, &mut transcript)
            .expect("commitment transcript");
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        let op_ctx =
            akita_prover::OperationCtx::new(&CpuBackend, &prepared, setup.expanded.as_ref())
                .expect("operation ctx");
        let poly_refs: [&DensePoly<F>; 1] = [&poly];
        let fold_claims = prover_fold_claims(&point, &poly_refs, &commitment, batched_hint);
        let (instance, witness) =
            RingRelationProver::new::<F, F, _, DensePoly<F>, CpuBackend, CpuBackend>(
                &op_ctx,
                &op_ctx,
                ring_opening_point,
                ring_multiplier_point.clone(),
                fold_claims,
                vec![RingVec::from_ring_elems(&e_folded)],
                lp.clone(),
                &mut transcript,
                RingVec::from_single(&CyclotomicRing::<F, D>::one()),
                RelationMatrixRowLayout::WithDBlock,
                None,
            )
            .expect("ring relation");

        let build_output =
            ring_switch_build_w::<F, CpuBackend>(&instance, witness, &op_ctx, &lp, false)
                .expect("ring-switch witness");
        let opening_layout =
            OpeningBlockLayout::new(1, build_output.w.len() / D).expect("opening layout");
        let (w_compact, _col_bits, _ring_bits) =
            build_w_evals_compact(build_output.w.as_i8_digits(), D, 1, opening_layout)
                .expect("compact witness");

        let alpha = F::from_u64(17);
        let alpha_evals_y = scalar_powers(alpha, D);
        let rows = lp
            .relation_matrix_row_count_for(1, RelationMatrixRowLayout::WithDBlock)
            .unwrap();
        let num_i = lp
            .relation_row_index_num_vars_for_layout(
                RelationMatrixRowLayout::WithDBlock,
                &opening_batch,
            )
            .expect("tau1 vars");

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
            let relation_matrix_col_evals = compute_relation_weight_evals::<F, F>(
                &setup.expanded,
                &instance,
                alpha,
                &alpha_evals_y,
                lp.role_dims(),
                &lp,
                &tau1,
                &[F::one()],
                RelationMatrixRowLayout::WithDBlock,
                opening_layout,
                D,
            )
            .expect("m evals");
            let got = direct_relation_claim(&w_compact, &relation_matrix_col_evals);
            let expected = relation_claim_from_rows::<F, D>(
                &tau1,
                alpha,
                lp.a_key.row_len(),
                instance.v_trusted::<D>().expect("v"),
                &commitment
                    .rows()
                    .try_to_vec::<D>()
                    .expect("commitment rows"),
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
    fn relation_matrix_evaluator_matches_materialized() {
        use akita_sumcheck::multilinear_eval;

        type F = fp128::Field;
        type Cfg = fp128::D128Full;
        const D: usize = Cfg::D;
        const NV: usize = 12;

        let opening_batch =
            akita_types::OpeningClaimsLayout::new(NV, 1).expect("singleton opening batch");
        let level_params =
            Cfg::get_params_for_batched_commitment(&opening_batch).expect("commitment layout");

        let mut rng = StdRng::seed_from_u64(0xdead_beef);
        let evals: Vec<F> = (0..(1usize << NV))
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F>::from_field_evals(NV, D, &evals).expect("dense poly");
        let point: Vec<F> = (0..NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(NV, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let (commitment, batched_hint) = AkitaCommitmentScheme::<Cfg>::commit::<_, _>(
            &setup,
            std::slice::from_ref(&poly),
            &stack,
        )
        .expect("commitment");

        let alpha_bits = D.trailing_zeros() as usize;
        let outer_point = &point[alpha_bits..];
        let ring_opening_point = ring_opening_point_from_field(
            outer_point,
            OpeningBlockLayout::new(
                level_params.live_fold_count,
                level_params.fold_position_count,
            )
            .unwrap(),
            BasisMode::Lagrange,
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
                fold_position_count: level_params.fold_position_count,
            },
        )
        .expect("evaluate_and_fold");
        let e_folded = opening.folded;

        let mut transcript = AkitaTranscript::<F>::new(b"prepared-m-eval-test");
        commitment
            .append_to_transcript(ABSORB_COMMITMENT, D, &mut transcript)
            .expect("commitment transcript");
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        let op_ctx =
            akita_prover::OperationCtx::new(&CpuBackend, &prepared, setup.expanded.as_ref())
                .expect("operation ctx");
        let poly_refs: [&DensePoly<F>; 1] = [&poly];
        let fold_claims = prover_fold_claims(&point, &poly_refs, &commitment, batched_hint);
        let (instance, witness) =
            RingRelationProver::new::<F, F, _, DensePoly<F>, CpuBackend, CpuBackend>(
                &op_ctx,
                &op_ctx,
                ring_opening_point.clone(),
                ring_multiplier_point.clone(),
                fold_claims,
                vec![RingVec::from_ring_elems(&e_folded)],
                level_params.clone(),
                &mut transcript,
                RingVec::from_single(&CyclotomicRing::<F, D>::one()),
                RelationMatrixRowLayout::WithDBlock,
                None,
            )
            .expect("ring relation");

        ring_switch_build_w::<F, CpuBackend>(&instance, witness, &op_ctx, &level_params, false)
            .expect("ring-switch witness");

        let alpha = F::from_u64(42);
        let alpha_evals_y = scalar_powers(alpha, D);
        let rows = level_params
            .relation_matrix_row_count_for(1, RelationMatrixRowLayout::WithDBlock)
            .unwrap();
        let num_i = level_params
            .relation_row_index_num_vars_for_layout(
                RelationMatrixRowLayout::WithDBlock,
                &opening_batch,
            )
            .expect("tau1 vars");
        let tau1: Vec<F> = (0..num_i)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let witness_layout = instance.segment_layout(&level_params, None).unwrap();
        let opening_layout = OpeningBlockLayout::new(1, witness_layout.total_len()).unwrap();

        let relation_weight_evals = compute_relation_weight_evals::<F, F>(
            &setup.expanded,
            &instance,
            alpha,
            &alpha_evals_y,
            level_params.role_dims(),
            &level_params,
            &tau1,
            &[F::one()],
            RelationMatrixRowLayout::WithDBlock,
            opening_layout,
            D,
        )
        .expect("relation weight evals (materialized)");
        let relation_matrix_col_evals: Vec<F> = relation_weight_evals
            .chunks_exact(D)
            .map(|coefficients| coefficients[0])
            .collect();

        let x_challenges: Vec<F> = (0..relation_matrix_col_evals.len().trailing_zeros() as usize)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let expected =
            multilinear_eval(&relation_matrix_col_evals, &x_challenges).expect("multilinear_eval");

        let gamma = [F::one()];
        let replay = RingSwitchReplay {
            setup: &setup.expanded,
            relation: &instance,
            row_coefficients: &gamma,
            lp: &level_params,
            opening_layout,
            opening_ring_dim: D,
        };
        let prepared = prepare_relation_matrix_evaluator::<F, F, D>(&replay, alpha, &tau1, None)
            .expect("prepare_relation_matrix_evaluator");

        let got = prepared
            .eval_at_point::<F, D>(&x_challenges, &setup.expanded, alpha, None)
            .expect("eval_at_point");

        assert_eq!(
            got, expected,
            "RelationMatrixEvaluator::eval_at_point must match materialized multilinear_eval"
        );

        // ----- Chunked layout ground truth (W ∈ powers of two | live_fold_count) --
        // The chunked relation's column-MLE has the same per-cell values as the
        // single-unit vector, repositioned by the canonical witness descriptor.
        // Each ownership unit receives its e/t block window and one replicated z
        // segment; the r tail remains shared.
        let live_fold_count = level_params.live_fold_count;
        let group_id = SemanticGroupId(0);
        let single_layout = instance
            .segment_layout(&level_params, None)
            .expect("single-unit witness layout");
        let single_unit = single_layout
            .units_for_group(group_id)
            .expect("single-unit group")[0]
            .clone();
        let group = single_layout.group(group_id).expect("single group").clone();

        let chunk_counts: Vec<usize> = (0..)
            .map(|k| 1usize << k)
            .take_while(|&w| w <= live_fold_count)
            .filter(|&w| live_fold_count % w == 0)
            .collect();
        assert!(
            chunk_counts.iter().any(|&w| w > 1),
            "fixture must have live_fold_count > 1 to exercise chunking (live_fold_count={live_fold_count})"
        );
        for w in chunk_counts.into_iter().filter(|&w| w > 1) {
            let mut lp_w = level_params.clone();
            lp_w.witness_chunk = ChunkedWitnessCfg {
                num_chunks: w,
                num_activated_levels: 1,
            };
            let chunk_layout = instance
                .segment_layout(&lp_w, None)
                .expect("chunked witness layout");
            let mut chunked = vec![F::zero(); chunk_layout.total_len().next_power_of_two()];
            for unit in chunk_layout
                .units_for_group(group_id)
                .expect("chunked group")
            {
                for position in 0..group.fold_position_count {
                    for commit_digit in 0..group.depth_commit {
                        for fold_digit in 0..group.depth_fold {
                            let source = single_layout
                                .z_index(&single_unit, position, commit_digit, fold_digit)
                                .expect("single z index");
                            let target = chunk_layout
                                .z_index(unit, position, commit_digit, fold_digit)
                                .expect("chunked z index");
                            chunked[target] = relation_matrix_col_evals[source];
                        }
                    }
                }
                let block_end = unit.global_block_base + unit.blocks;
                for claim in 0..group.num_claims {
                    for block in unit.global_block_base..block_end {
                        for digit in 0..group.depth_open {
                            let source = single_layout
                                .e_index(&single_unit, claim, block, digit)
                                .expect("single e index");
                            let target = chunk_layout
                                .e_index(unit, claim, block, digit)
                                .expect("chunked e index");
                            chunked[target] = relation_matrix_col_evals[source];
                            for a_row in 0..group.n_a {
                                let source = single_layout
                                    .t_index(&single_unit, claim, block, a_row, digit)
                                    .expect("single t index");
                                let target = chunk_layout
                                    .t_index(unit, claim, block, a_row, digit)
                                    .expect("chunked t index");
                                chunked[target] = relation_matrix_col_evals[source];
                            }
                        }
                    }
                }
            }
            for row in 0..rows {
                for digit in 0..single_layout.quotient_depth {
                    let source = single_layout.r_index(row, digit).expect("single r index");
                    let target = chunk_layout.r_index(row, digit).expect("chunked r index");
                    chunked[target] = relation_matrix_col_evals[source];
                }
            }
            let x_len_w = chunked.len();
            let opening_layout_w = OpeningBlockLayout::new(1, chunk_layout.total_len()).unwrap();

            let x_challenges_w: Vec<F> = (0..x_len_w.trailing_zeros() as usize)
                .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
                .collect();
            let expected_w = multilinear_eval(&chunked, &x_challenges_w).expect("multilinear_eval");

            let replay_w = RingSwitchReplay {
                setup: &setup.expanded,
                relation: &instance,
                row_coefficients: &gamma,
                lp: &lp_w,
                opening_layout: opening_layout_w,
                opening_ring_dim: D,
            };
            let prepared_w =
                prepare_relation_matrix_evaluator::<F, F, D>(&replay_w, alpha, &tau1, None)
                    .expect("prepare chunked row eval");
            let got_w = prepared_w
                .eval_at_point::<F, D>(&x_challenges_w, &setup.expanded, alpha, None)
                .expect("chunked eval_at_point");
            assert_eq!(
                got_w, expected_w,
                "chunked eval_at_point must match materialized chunked row for W={w}"
            );

            // Prover-side cross-check: the chunked grouped M evaluator must emit
            // exactly the rearranged column layout, and its multilinear eval must
            // match the verifier's chunked row eval.
            let prover_chunked_weights = compute_relation_weight_evals::<F, F>(
                &setup.expanded,
                &instance,
                alpha,
                &alpha_evals_y,
                lp_w.role_dims(),
                &lp_w,
                &tau1,
                &[F::one()],
                RelationMatrixRowLayout::WithDBlock,
                opening_layout_w,
                D,
            )
            .expect("chunked relation weight evals (prover)");
            let prover_chunked: Vec<F> = prover_chunked_weights
                .chunks_exact(D)
                .map(|coefficients| coefficients[0])
                .collect();
            assert_eq!(
                prover_chunked, chunked,
                "prover chunked grouped M evals must equal the rearranged column layout for W={w}"
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
            ring_opening_point_from_field, BasisMode,
        };

        type F = fp128::Field;
        type Cfg = fp128::D128Full;
        const D: usize = Cfg::D;
        const NV: usize = 12;

        let level_params = Cfg::get_params_for_batched_commitment(
            &akita_types::OpeningClaimsLayout::new(NV, 1).expect("singleton opening batch"),
        )
        .expect("commitment layout");

        let mut rng = StdRng::seed_from_u64(0x5E6E_7E8E);
        let evals: Vec<F> = (0..(1usize << NV))
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F>::from_field_evals(NV, D, &evals).expect("dense poly");
        let point: Vec<F> = (0..NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(NV, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let (commitment, batched_hint) = AkitaCommitmentScheme::<Cfg>::commit::<_, _>(
            &setup,
            std::slice::from_ref(&poly),
            &stack,
        )
        .expect("commitment");

        let alpha_bits = D.trailing_zeros() as usize;
        let outer_point = &point[alpha_bits..];
        let ring_opening_point = ring_opening_point_from_field(
            outer_point,
            OpeningBlockLayout::new(
                level_params.live_fold_count,
                level_params.fold_position_count,
            )
            .unwrap(),
            BasisMode::Lagrange,
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
                fold_position_count: level_params.fold_position_count,
            },
        )
        .expect("evaluate_and_fold");
        let e_folded = opening.folded;

        let mut transcript = AkitaTranscript::<F>::new(b"segment-typed-expand-test");
        commitment
            .append_to_transcript(ABSORB_COMMITMENT, D, &mut transcript)
            .expect("commitment transcript");
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        let op_ctx =
            akita_prover::OperationCtx::new(&CpuBackend, &prepared, setup.expanded.as_ref())
                .expect("operation ctx");
        let poly_refs: [&DensePoly<F>; 1] = [&poly];
        let fold_claims = prover_fold_claims(&point, &poly_refs, &commitment, batched_hint);
        let (instance, witness) =
            RingRelationProver::new::<F, F, _, DensePoly<F>, CpuBackend, CpuBackend>(
                &op_ctx,
                &op_ctx,
                ring_opening_point,
                ring_multiplier_point,
                fold_claims,
                vec![RingVec::from_ring_elems(&e_folded)],
                level_params.clone(),
                &mut transcript,
                RingVec::from_single(&CyclotomicRing::<F, D>::one()),
                RelationMatrixRowLayout::WithoutDBlock,
                None,
            )
            .expect("ring relation");

        let build_output =
            ring_switch_build_w::<F, CpuBackend>(&instance, witness, &op_ctx, &level_params, true)
                .expect("ring-switch witness");
        let logical_digits = build_output.w.as_i8_digits();
        let artifacts = build_output
            .terminal_artifacts
            .expect("terminal artifacts retained");
        artifacts.ensure_ring_dim::<D>().expect("ring dim");
        let segment = build_segment_typed_witness::<F>(
            artifacts.ring_dim(),
            &artifacts.e_folded,
            &artifacts.recomposed_inner_rows,
            artifacts.z_folded_centered_flat(),
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
