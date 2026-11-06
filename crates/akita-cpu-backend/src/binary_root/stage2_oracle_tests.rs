//! Stage 2 on block profiles against Stage 2 on the materialized root table.

use super::*;
use akita_types::binary_source::{root_weight_fixtures, RootFixtureField as E};
use jolt_field::{One, Zero};

/// `eq(point, x)` for every address `x`, with bit `i` of `x` against `point[i]`.
fn equality_table<E: Field>(point: &[E]) -> Vec<E> {
    let mut table = vec![E::one()];
    for &r in point {
        let low = table.iter().map(|&value| value * (E::one() - r));
        let high = table.iter().map(|&value| value * r);
        table = low.chain(high).collect::<Vec<_>>();
    }
    table
}

#[test]
fn profiled_root_stage2_matches_the_dense_table_round_by_round() {
    for fixture in root_weight_fixtures() {
        let dense = fixture.weights.materialize().unwrap();
        let domain_len = dense.len();
        let num_vars = domain_len.trailing_zeros() as usize;
        let block_bits = num_vars.min(ROOT_PROFILE_BLOCK_BITS);
        // Negative-binary addresses hold -1 or 0, so their `w(w+1)` terms vanish.
        let mut witness = (0..domain_len)
            .map(|index| {
                if index < fixture.logical_len {
                    ((index * 2_654_435_761) >> 7 & 7) as i8 - 4
                } else {
                    0
                }
            })
            .collect::<Vec<_>>();
        for &(address, _) in &fixture.negative_binary_weights {
            witness[address] = -((address & 1) as i8);
        }
        let point = (0..num_vars as u64)
            .map(|bit| E::from_u64(1_000_033 + 17 * bit))
            .collect::<Vec<_>>();
        let range_image = equality_table(&point).into_iter().zip(&witness).fold(
            E::zero(),
            |sum, (equality, &digit)| {
                let digit = E::from_i64(i64::from(digit));
                sum + equality * digit * (digit + E::one())
            },
        );
        let claim = dense
            .iter()
            .zip(&witness)
            .fold(E::zero(), |sum, (&weight, &digit)| {
                sum + weight * E::from_i64(i64::from(digit))
            });
        let batching = E::from_u64(1_000_211);
        let mut expected = CpuStage2SessionHandle::from_dense_relation(
            witness.clone(),
            dense,
            claim,
            &point,
            range_image,
            batching,
        )
        .unwrap()
        .with_negative_binary_weights(fixture.negative_binary_weights.clone())
        .unwrap();
        let live_len = fixture.logical_len.next_multiple_of(1 << block_bits);
        let mut actual = CpuStage2SessionHandle::from_structured_relation(
            PackedSignedDigits::from_i8_digits_auto(witness[..live_len].to_vec()),
            Box::new(fixture.weights.profiles(block_bits).unwrap()),
            block_bits,
            claim,
            &point,
            range_image,
            batching,
        )
        .unwrap()
        .with_negative_binary_weights(fixture.negative_binary_weights)
        .unwrap();
        assert_eq!(actual.input_claim(), expected.input_claim());
        assert_eq!(actual.num_rounds(), num_vars);
        let mut previous = expected.input_claim();
        for round in 0..num_vars {
            let polynomial = expected.round_polynomial(round, previous).unwrap();
            assert_eq!(
                actual.round_polynomial(round, previous).unwrap(),
                polynomial,
                "{} round {round}",
                fixture.name
            );
            let challenge = E::from_u64(2_000_003 + 101 * round as u64);
            previous = polynomial.evaluate(challenge);
            expected.bind_challenge(round, challenge).unwrap();
            actual.bind_challenge(round, challenge).unwrap();
        }
        let expected = expected.finish().unwrap();
        let actual = actual.finish().unwrap();
        assert_eq!(actual.witness_evaluation(), expected.witness_evaluation());
        assert_eq!(actual.final_claim(), expected.final_claim());
    }
}

mod p64 {
    use super::{
        equality_table, CpuStage2SessionHandle, PackedSignedDigits, ROOT_PROFILE_BLOCK_BITS,
    };
    use crate::binary_root::{binary_row_partials, BinaryEvaluationWitness, BinaryRootRelation};
    use crate::opaque::sumcheck::digit_range::{DigitRangeProver, DigitRangeSession};
    use crate::opaque::{Stage1PublicTransition, Stage1Step};
    use akita_algebra::ring::trinomial::{image::Q24Image, signed::signed_trinomial_product};
    use akita_algebra::MinusTrinomial;
    use akita_challenges::{Challenges, FoldChallengeDistribution, SparseChallenge};
    use akita_params::binary_setup::BinaryFieldProfile;
    use akita_params::{DigitRangePlan, FlatBooleanDomain};
    use akita_types::binary_source::{
        binary_public_digit_commitment, binary_root_matrix, BinaryAOperand,
        BinaryCommittedGroupManifest, BinaryIndexMap, BinaryRootPublic, BinarySourceDescriptor,
        BinaryTerminalClaim, PrimePlaneManifest, BINARY_SOURCE_Q,
    };
    use jolt_field::{Ext2, ExtField, One, Prime64Offset59, Ring, Zero};
    type F = Prime64Offset59;
    type E = Ext2<F>;

    fn extension(index: usize) -> E {
        E::from_base_fn(|coordinate| F::from_u64((17 + index * 31 + coordinate * 43) as u64))
    }

    fn carrier() -> akita_params::GroupCommitPhaseParams {
        use akita_params::sis::{
            sis_table_key_for_linf_bound, InnerCommitMatrixParams, OuterCommitMatrixParams,
            SisMatrixRole, SisTableDigest, DEFAULT_SIS_SECURITY_POLICY,
        };
        use akita_params::{
            BlockGeometry, CommitmentSliceCount, CommitmentSliceGeometry, GadgetDigits,
            GroupCommitPhaseParams, PolynomialGroupLayout, RoleParams, SisModulusProfileId,
        };
        let key = |role, degree| {
            sis_table_key_for_linf_bound(
                DEFAULT_SIS_SECURITY_POLICY,
                SisTableDigest::CURRENT,
                SisModulusProfileId::Q64Offset59,
                role,
                degree,
                8,
            )
            .unwrap()
        };
        let inner =
            InnerCommitMatrixParams::try_new_with_min_rank(key(SisMatrixRole::Inner, 128), 2)
                .unwrap();
        let geometry = CommitmentSliceGeometry::try_new(
            CommitmentSliceCount::ONE,
            4,
            1,
            inner.output_rank(),
            22,
            128,
            64,
        )
        .unwrap();
        let outer = OuterCommitMatrixParams::try_new_with_min_rank(
            key(SisMatrixRole::Outer, 64),
            geometry.physical_input_width(),
        )
        .unwrap();
        GroupCommitPhaseParams {
            version: GroupCommitPhaseParams::VERSION,
            group: PolynomialGroupLayout::singleton(10),
            blocks: BlockGeometry::new(8, 2, 4),
            outer_slice_count: CommitmentSliceCount::ONE,
            inner: RoleParams::new(GadgetDigits::new(3, 1), inner),
            outer: RoleParams::new(GadgetDigits::new(3, 22), outer),
        }
    }

    #[test]
    fn binary_root_p64_k_is_range_checked_and_authenticated_in_one_cubic_stage2() {
        let seed = [41; 32];
        let index = BinaryIndexMap::new(128, 8, 8, 2).unwrap();
        let profiles =
            BinaryCommittedGroupManifest::profiles_for_field(&index, 1, 0, BinaryFieldProfile::P64)
                .unwrap();
        let descriptor = BinarySourceDescriptor::new(index, vec![[1; 32]], profiles).unwrap();
        let manifest = BinaryCommittedGroupManifest::with_field_profile(
            descriptor.clone(),
            0,
            &carrier(),
            BinaryFieldProfile::P64,
        )
        .unwrap();
        let matrix = binary_root_matrix(&seed, &manifest).unwrap();
        let mut sources = vec![[0i128; 648]; 2];
        for (column, source) in sources.iter_mut().enumerate() {
            for (index, coefficient) in source.iter_mut().enumerate().take(512) {
                if (index + column * 7) % 31 == 0 {
                    *coefficient = if index % 2 == 0 { 1 } else { -1 };
                }
            }
        }
        let images = sources
            .iter()
            .flat_map(|source| {
                matrix.iter().map(|row| {
                    signed_trinomial_product::<648, MinusTrinomial>(&row[0], source)
                        .unwrap()
                        .map(|value| {
                            Q24Image::from_residue(
                                value.rem_euclid(i128::from(BINARY_SOURCE_Q)) as u32
                            )
                            .unwrap()
                        })
                })
            })
            .collect::<Vec<_>>();
        let challenges = Challenges::from_sparse(
            (0..2)
                .map(|column| SparseChallenge {
                    positions: (0..46)
                        .map(|index| ((index * 3 + column) % 162) as u32)
                        .collect::<Vec<_>>()
                        .into(),
                    coeffs: vec![1; 46].into(),
                })
                .collect(),
            2,
            1,
        )
        .unwrap();
        let relation = BinaryRootRelation::build(
            &descriptor,
            &matrix,
            &sources,
            &images,
            &challenges,
            FoldChallengeDistribution::BinaryRoot,
        )
        .unwrap();
        let residuals = BinaryRootRelation::remainders(
            &descriptor,
            &matrix,
            &relation.response,
            &images,
            &challenges,
            FoldChallengeDistribution::BinaryRoot,
        )
        .unwrap();
        assert_eq!(
            residuals,
            relation
                .rows
                .iter()
                .map(|row| row.remainder)
                .collect::<Vec<_>>()
        );
        // Zero row and column points select scalar row zero, column zero;
        // its packed binary value is read directly from that source component.
        let mut value = [0u8; 21];
        for coefficient in 0..128 {
            value[coefficient / 8] |=
                ((sources[0][4 * coefficient] & 1) as u8) << (coefficient % 8);
        }
        let claim =
            BinaryTerminalClaim::new(&descriptor, [1; 32], vec![[0; 21]; 2], vec![[0; 21]], value)
                .unwrap();
        let setup = akita_types::AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: akita_types::MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
            setup_seed: seed.into(),
        };
        let prime = PrimePlaneManifest::new(&descriptor, None).unwrap();
        let public =
            BinaryRootPublic::new(&manifest, Some(&claim), &prime, &setup, 1 << 17, 64, None)
                .unwrap();
        let plan = public.plan();
        let quotient = plan.quotient_segment().unwrap();
        let bounds = plan.projection_bounds().quotient().unwrap();
        let partials = binary_row_partials(&descriptor, &claim, &sources).unwrap();
        let evaluation = BinaryEvaluationWitness::build(
            &descriptor,
            &claim,
            &partials,
            &relation.response,
            &challenges,
            BinaryFieldProfile::P64.modulus(),
        )
        .unwrap();
        let natural = [
            BinaryRootRelation::response_digits(&relation.response).unwrap(),
            BinaryRootRelation::image_digits(&images).unwrap(),
            partials,
            evaluation.parity_digits,
            evaluation.value_digits,
            BinaryRootRelation::quotient_digits(&residuals, bounds).unwrap(),
        ];
        assert!(natural[quotient].iter().any(|digit| *digit < 0));
        let witness = plan
            .repacking()
            .repack(&natural.iter().map(Vec::as_slice).collect::<Vec<_>>())
            .unwrap();
        let table = witness
            .iter()
            .map(|&digit| E::from_i64(i64::from(digit)))
            .collect::<Vec<_>>();
        let b =
            binary_public_digit_commitment::<F>(&seed, manifest.b_label(), 0, &natural[1]).unwrap();
        let d = binary_public_digit_commitment::<F>(
            &seed,
            manifest.d_label(true, 0).unwrap(),
            0,
            &natural[2],
        )
        .unwrap();
        crate::opaque::sumcheck::stage3::p64_root_transcript_tests::word_root_round_trip(
            &manifest,
            &public,
            &matrix,
            &challenges,
            &witness,
            &b,
            &d,
        );
        let lift = |values: &[F]| values.iter().copied().map(E::lift_base).collect::<Vec<_>>();
        let batches = (0..public.batch_count()).map(extension).collect::<Vec<_>>();
        let rho = batches[..plan.coefficient_batch_count().unwrap()]
            .chunks_exact(648)
            .map(|chunk| <[E; 648]>::try_from(chunk).unwrap())
            .collect::<Vec<_>>();
        let (weights, negative) = public
            .relation(
                &challenges,
                extension(3),
                &batches,
                &[],
                &[],
                &lift(&b),
                &lift(&d),
                None,
                Some(BinaryAOperand::new::<F>(&matrix)),
            )
            .unwrap();
        assert!(negative.is_empty());
        weights.check_table(&table).unwrap();
        let dense = weights.materialize().unwrap();
        // The independent reference reads R and K directly; it uses no A/T
        // adjoint and gives exactly the defect of the compiled common batch.
        let reference = |table: &[E]| {
            residuals
                .iter()
                .zip(&rho)
                .enumerate()
                .fold(E::zero(), |sum, (row, (residual, rho))| {
                    sum + residual.iter().zip(rho).enumerate().fold(
                        E::zero(),
                        |sum, (coefficient, (&residual, &rho))| {
                            sum + rho
                                * (E::from_i128(residual)
                                    - E::from_u64(u64::from(BINARY_SOURCE_Q))
                                        * plan
                                            .repacking()
                                            .recompose_coefficient(
                                                table,
                                                quotient,
                                                row,
                                                coefficient,
                                            )
                                            .unwrap())
                        },
                    )
                })
        };
        assert_eq!(reference(&table), E::zero());
        let variables = plan.repacking().domain_len().trailing_zeros() as usize;
        let tau = (0..variables).map(extension).collect::<Vec<_>>();
        let domain = FlatBooleanDomain::new(witness.len(), variables).unwrap();
        let range = DigitRangeProver::from_packed_digits(
            PackedSignedDigits::from_i8_digits_auto(witness.clone()),
            DigitRangePlan::new(8).unwrap(),
            domain,
            akita_types::DigitRangeEqualityPoint::from_column_then_ring_challenges(
                &tau, variables, 0,
            )
            .unwrap(),
        )
        .unwrap();
        let mut stage1 = DigitRangeSession::new(range, None).unwrap();
        for round in 0..variables {
            stage1
                .round_polynomial(Stage1Step::RangeLeaf, round, stage1.current_claim())
                .unwrap();
            stage1
                .bind_challenge(Stage1Step::RangeLeaf, round, extension(100 + round))
                .unwrap();
        }
        let Stage1PublicTransition::Final {
            range_image_evaluation,
            ..
        } = stage1.public_transition(Stage1Step::RangeLeaf).unwrap()
        else {
            panic!("expected range terminal");
        };
        let final_range = stage1.finish().unwrap();
        let point = final_range.point();
        let expected_range = equality_table(point)
            .iter()
            .zip(&table)
            .fold(E::zero(), |sum, (&eq, &digit)| {
                sum + eq * digit * (digit + E::one())
            });
        assert_eq!(range_image_evaluation, expected_range);
        let image = range_image_evaluation;
        let range_polynomial =
            image * (image - E::from_u64(2)) * (image - E::from_u64(6)) * (image - E::from_u64(12));
        // Equality-factored replay removes the equality factors round by
        // round, so the verifier's terminal check is Q(range_image), alone.
        assert_eq!(final_range.final_claim(), range_polynomial);
        let gamma = extension(400);
        let mut dense_session = CpuStage2SessionHandle::from_dense_relation(
            witness.clone(),
            dense.clone(),
            weights.claim(),
            point,
            range_image_evaluation,
            gamma,
        )
        .unwrap();
        let mut session = CpuStage2SessionHandle::from_structured_relation(
            PackedSignedDigits::from_i8_digits_auto(witness.clone()),
            Box::new(weights.profiles(ROOT_PROFILE_BLOCK_BITS).unwrap()),
            ROOT_PROFILE_BLOCK_BITS,
            weights.claim(),
            point,
            range_image_evaluation,
            gamma,
        )
        .unwrap();
        let mut previous = weights.claim() + gamma * range_image_evaluation;
        for round in 0..variables {
            let polynomial = session.round_polynomial(round, previous).unwrap();
            assert!(polynomial.coefficients().len() <= 4);
            assert_eq!(
                polynomial.evaluate(E::zero()) + polynomial.evaluate(E::one()),
                previous
            );
            assert_eq!(
                dense_session.round_polynomial(round, previous).unwrap(),
                polynomial
            );
            let challenge = extension(500 + round);
            previous = polynomial.evaluate(challenge);
            session.bind_challenge(round, challenge).unwrap();
            dense_session.bind_challenge(round, challenge).unwrap();
        }
        let terminal = session.finish().unwrap();
        let dense_terminal = dense_session.finish().unwrap();
        assert_eq!(terminal.final_claim(), dense_terminal.final_claim());
        assert_eq!(
            terminal.witness_evaluation(),
            dense_terminal.witness_evaluation()
        );
        // The same dense coefficient coordinates feed Stage 3 over the
        // selected base setup; no integer representative of E64 is used.
        let typed =
            akita_types::binary_source::BinaryTypedPrefix::new(&manifest, true, 0, 64).unwrap();
        let stage2_point = (0..variables)
            .map(|round| extension(500 + round))
            .collect::<Vec<_>>();
        let (stage3_rho, _) = public.coefficient_inputs(&batches, &[], &[]).unwrap();
        assert_eq!(stage3_rho.as_ref(), rho);
        let a = akita_types::binary_source::BinaryAProduct::new(
            plan,
            &stage3_rho,
            &stage2_point,
            typed.domain_len,
        )
        .unwrap();
        let authentication = public.authentication_batches(&batches).unwrap();
        let typed_weights = typed
            .prover_weights(
                &a,
                &stage2_point,
                extension(3),
                authentication,
                typed.domain_len,
            )
            .unwrap();
        let setup = typed
            .setup_coefficients::<F>(&seed, &manifest, typed.domain_len)
            .unwrap();
        let stage3_claim = setup
            .iter()
            .zip(typed_weights)
            .fold(E::zero(), |sum, (&value, weight)| {
                sum + E::lift_base(value) * weight
            });
        let (online, _) = public
            .relation(
                &challenges,
                extension(3),
                &batches,
                &[],
                &[],
                &lift(&b),
                &lift(&d),
                None,
                None,
            )
            .unwrap();
        let online_weight = online.evaluate(&stage2_point).unwrap();
        assert_eq!(
            online_weight + stage3_claim,
            weights.evaluate(&stage2_point).unwrap()
        );
        assert_ne!(stage3_claim.base_coefficient(1), F::zero());
        // Keep Z, T, U and all carries fixed. Only K changes, within the range.
        for (row, digit, coefficient) in [(0, 0, 0), (2, bounds.digits - 1, 647)] {
            let address = plan
                .repacking()
                .coefficient_address(quotient, row, digit, coefficient)
                .unwrap();
            let mut altered = table.clone();
            altered[address] += if witness[address] == 3 {
                -E::one()
            } else {
                E::one()
            };
            plan.repacking().validate_digit_table(&altered).unwrap();
            assert_ne!(reference(&altered), E::zero());
            let compiled = dense
                .iter()
                .zip(&altered)
                .fold(E::zero(), |sum, (&weight, &value)| sum + weight * value)
                - weights.claim();
            assert_eq!(compiled, reference(&altered));
            assert!(weights.check_table(&altered).is_err());
            let mut altered_digits = witness.clone();
            altered_digits[address] += if witness[address] == 3 { -1 } else { 1 };
            let altered_range = equality_table(point)
                .iter()
                .zip(&altered)
                .fold(E::zero(), |sum, (&eq, &value)| {
                    sum + eq * value * (value + E::one())
                });
            let rejected = CpuStage2SessionHandle::from_structured_relation(
                PackedSignedDigits::from_i8_digits_auto(altered_digits),
                Box::new(weights.profiles(ROOT_PROFILE_BLOCK_BITS).unwrap()),
                ROOT_PROFILE_BLOCK_BITS,
                weights.claim(),
                point,
                altered_range,
                gamma,
            );
            // Debug construction checks the input claim with a full scan.
            // Release proving omits that scan; its terminal identity must
            // reject the same false claim after the sumcheck rounds.
            if let Ok(mut rejected) = rejected {
                let mut claim = weights.claim() + gamma * altered_range;
                for round in 0..variables {
                    let polynomial = rejected.round_polynomial(round, claim).unwrap();
                    let challenge = extension(500 + round);
                    claim = polynomial.evaluate(challenge);
                    rejected.bind_challenge(round, challenge).unwrap();
                }
                assert!(rejected.finish().is_err());
            }
        }
        let address = plan
            .repacking()
            .coefficient_address(quotient, 0, 0, 0)
            .unwrap();
        let mut outside = witness;
        outside[address] = 4;
        let outside = outside
            .into_iter()
            .map(|digit| E::from_i64(i64::from(digit)))
            .collect::<Vec<_>>();
        assert!(plan.repacking().validate_digit_table(&outside).is_err());
        let image = E::from_u64(4 * 5);
        assert_ne!(
            image * (image - E::from_u64(2)) * (image - E::from_u64(6)) * (image - E::from_u64(12)),
            E::zero()
        );
    }
}
