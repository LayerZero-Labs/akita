use super::*;
use crate::RecursiveWitnessFlat;
use akita_config::proof_optimized::fp128::OneHot;
use akita_transcript::AkitaTranscript;
use akita_types::{AkitaScheduleLookupKey, OpeningClaimsLayout, PolynomialGroupLayout};
use jolt_field::{Fp32, FpExt2, One, TwoNr, Zero};

type F = Fp32<251>;
type E = FpExt2<F, TwoNr>;

fn eor_test_plan(rounds: usize, batches_claims: bool) -> akita_types::GrindingPlan {
    let mut runs = vec![akita_types::GrindingRun::proof_of_work(
        akita_types::GrindingSite::ExtensionOpeningPoint { level: 1 },
        1,
        128,
    )
    .unwrap()];
    if batches_claims {
        runs.push(
            akita_types::GrindingRun::proof_of_work(
                akita_types::GrindingSite::ExtensionOpeningClaimBatch { level: 1 },
                1,
                128,
            )
            .unwrap(),
        );
    }
    for round in 0..rounds {
        runs.push(
            akita_types::GrindingRun::proof_of_work(
                akita_types::GrindingSite::SumcheckRound {
                    protocol: akita_types::SumcheckProtocol::ExtensionOpeningReduction,
                    level: 1,
                    stage: 0,
                    round: u32::try_from(round).unwrap(),
                },
                1,
                128,
            )
            .unwrap(),
        );
    }
    akita_types::GrindingPlan::new(runs, 128).unwrap()
}

#[test]
fn coefficient_packing_bypasses_eor_while_evaluation_trace_uses_it() {
    let packing = akita_types::OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: 64,
    };
    assert!(!packing.requires_extension_opening_reduction(2));
    assert!(!packing.requires_extension_opening_reduction(<E as ExtField<F>>::DEGREE));
    assert!(akita_types::OpeningMethod::EvaluationTrace
        .requires_extension_opening_reduction(<E as ExtField<F>>::DEGREE));
}

#[test]
fn recursive_extension_opening_reduction_pads_to_opening_cube() {
    let mut digits = vec![0; 3 * 64];
    digits[..6].copy_from_slice(&[1, -1, 2, 0, 3, -2]);
    let logical_w = RecursiveWitnessFlat::from_i8_digits(digits);
    let point = [
        E::new(F::from_u64(2), F::from_u64(3)),
        E::new(F::from_u64(5), F::from_u64(7)),
        E::new(F::from_u64(11), F::from_u64(13)),
        E::new(F::from_u64(17), F::from_u64(19)),
        E::new(F::from_u64(23), F::from_u64(29)),
        E::new(F::from_u64(31), F::from_u64(37)),
        E::new(F::from_u64(41), F::from_u64(43)),
        E::new(F::from_u64(47), F::from_u64(53)),
    ];
    let logical_polys = [&logical_w];
    let logical_group = PreparedProverGroup::from_refs(&logical_polys).expect("logical group");

    let mut transcript =
        AkitaTranscript::<F>::new(b"test/recursive-extension-opening-reduction-padding");
    let groups = vec![ExtensionOpeningGroupInput {
        group: &logical_group,
        point: &point,
        ring_dimension: 64,
    }];
    let plan = eor_test_plan(point.len() - 1, false);
    let mut transcript =
        akita_types::ProverGrindingTranscript::<_>::new(&mut transcript, &plan).unwrap();
    let proved = prove_extension_opening_reduction::<F, E, _, _, _>(
        &crate::compute::CpuBackend::DEFAULT,
        None,
        &groups,
        &mut transcript,
        1,
        "recursive",
    )
    .expect("padded logical witnesses should reduce over the opening cube");
    transcript.finish().unwrap();

    assert_eq!(
        proved.reduction.proof.partials.len(),
        <E as ExtField<F>>::DEGREE
    );
    assert_eq!(proved.reduction.proof.num_rounds(), point.len() - 1);
}

#[test]
fn extension_opening_reduction_shares_challenges_across_groups() {
    let short_witness = RecursiveWitnessFlat::from_i8_digits(vec![1; 64]);
    let mut long_digits = vec![0; 3 * 64];
    long_digits[..6].copy_from_slice(&[1, -1, 2, 0, 3, -2]);
    let long_witness = RecursiveWitnessFlat::from_i8_digits(long_digits);
    let short_point = (0..6)
        .map(|index| E::new(F::from_u64(index + 2), F::from_u64(index + 11)))
        .collect::<Vec<_>>();
    let long_point = (0..8)
        .map(|index| E::new(F::from_u64(index + 3), F::from_u64(index + 17)))
        .collect::<Vec<_>>();
    let polys = [&short_witness, &long_witness];
    let prepared_groups = [
        PreparedProverGroup::from_ref_vec(vec![polys[0]]).expect("short group"),
        PreparedProverGroup::from_ref_vec(vec![polys[1]]).expect("long group"),
    ];
    let groups = vec![
        ExtensionOpeningGroupInput {
            group: &prepared_groups[0],
            point: &short_point,
            ring_dimension: 64,
        },
        ExtensionOpeningGroupInput {
            group: &prepared_groups[1],
            point: &long_point,
            ring_dimension: 64,
        },
    ];
    let mut transcript = AkitaTranscript::<F>::new(b"test/grouped-extension-opening-reduction");
    let plan = eor_test_plan(long_point.len() - 1, true);
    let mut transcript =
        akita_types::ProverGrindingTranscript::<_>::new(&mut transcript, &plan).unwrap();

    let proved = prove_extension_opening_reduction::<F, E, _, _, _>(
        &crate::compute::CpuBackend::DEFAULT,
        None,
        &groups,
        &mut transcript,
        1,
        "recursive",
    )
    .expect("all groups should reduce through one shared challenge sequence");
    transcript.finish().unwrap();

    assert_eq!(proved.protocol_points.len(), 2);
    assert_eq!(proved.reduction.final_factors.len(), 2);
    assert_eq!(proved.reduction.proof.final_claims.len(), 2);
    assert_eq!(proved.reduction.proof.num_rounds(), long_point.len() - 1);
}

fn direct_eq_at_boolean<E: Field>(point: &[E], index: usize) -> E {
    point
        .iter()
        .enumerate()
        .fold(E::one(), |acc, (bit, &coordinate)| {
            acc * if (index >> bit) & 1 == 0 {
                E::one() - coordinate
            } else {
                coordinate
            }
        })
}

fn direct_lifted_mle<B, E>(base_evals: &[B], point: &[E]) -> E
where
    B: Field,
    E: ExtField<B>,
{
    base_evals
        .iter()
        .enumerate()
        .fold(E::zero(), |acc, (index, &value)| {
            acc + E::lift_base(value) * direct_eq_at_boolean(point, index)
        })
}

fn direct_tensor_tables<B, E>(base_evals: &[B], point: &[E], eta: E) -> (Vec<E>, Vec<E>)
where
    B: Field,
    E: ExtField<B>,
{
    assert_eq!(E::DEGREE, 2);
    assert_eq!(base_evals.len(), 1usize << point.len());
    let tail_point = &point[1..];
    let tail_len = 1usize << tail_point.len();
    let packed = (0..tail_len)
        .map(|tail| E::from_base_slice(&base_evals[2 * tail..2 * tail + 2]))
        .collect::<Vec<_>>();
    let factor = (0..tail_len)
        .map(|tail| {
            let equality = direct_eq_at_boolean(tail_point, tail);
            (E::one() - eta) * E::lift_base(equality.base_coefficient(0))
                + eta * E::lift_base(equality.base_coefficient(1))
        })
        .collect::<Vec<_>>();
    (packed, factor)
}

fn direct_column_partials<B, E>(base_evals: &[B], point: &[E]) -> Vec<E>
where
    B: Field,
    E: ExtField<B>,
{
    assert_eq!(E::DEGREE, 2);
    assert_eq!(base_evals.len(), 1usize << point.len());
    let tail_point = &point[1..];
    let tail_len = 1usize << tail_point.len();
    (0..2)
        .map(|head| {
            (0..tail_len).fold(E::zero(), |acc, tail| {
                acc + E::lift_base(base_evals[2 * tail + head])
                    * direct_eq_at_boolean(tail_point, tail)
            })
        })
        .collect()
}

#[test]
fn mixed_setup_prefix_and_suffix_eor_matches_independent_dense_oracle() {
    use akita_transcript::labels::ABSORB_SUMCHECK_CLAIM;
    use akita_types::{
        sample_akita_setup_seed, AkitaCommitmentHint, AkitaSetupDescriptor, CommittedGroupParams,
        FlatMatrix, GroupCommitPhaseParams, InnerCommitMatrixParams, OuterCommitMatrixParams,
        SetupPrefixPublicCommitment, SetupPrefixSlot, SetupPrefixSlotId, SisModulusProfileId,
        EXTENSION_OPENING_REDUCTION_DEGREE,
    };
    use jolt_field::{Ext2, Prime128OffsetA7F7};

    type Base = Prime128OffsetA7F7;
    type Extension = Ext2<Base>;
    const D: usize = 128;

    let mut params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        D,
        2,
        1,
        1,
        1,
        akita_challenges::SparseChallengeConfig::production_for_ring_dim(D).unwrap(),
    )
    .with_decomp(4, 4, 2, 2, 2)
    .unwrap();
    let inner = &params.inner().matrix;
    let inner_bound =
        *akita_types::sis::inner_coeff_linf_bounds(inner.sis_modulus_profile(), D as u32)
            .first()
            .expect("audited setup-prefix A bound");
    params.own_group_mut().profile.inner.matrix = InnerCommitMatrixParams::new_unchecked(
        inner.security_policy(),
        inner.sis_table_key().unwrap().table_digest,
        inner.sis_modulus_profile(),
        inner.output_rank(),
        inner.input_width(),
        inner_bound,
        D,
    );
    let outer = &params.outer().matrix;
    params.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width(),
        3,
        D,
    );
    let profile =
        GroupCommitPhaseParams::try_from_params(PolynomialGroupLayout::singleton(9), &params)
            .unwrap();
    let setup_evals = (0..512)
        .map(|index| Base::from_i64((index % 17) as i64 - 8))
        .collect::<Vec<_>>();
    let expanded = Arc::new(
        AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupDescriptor {
                max_num_vars: 9,
                max_num_batched_polys: 1,
                num_field_elements: setup_evals.len(),
                setup_seed: sample_akita_setup_seed(),
            },
            FlatMatrix::from_flat_data(setup_evals.clone()),
        ),
    );
    let slot = Arc::new(SetupPrefixSlot {
        id: SetupPrefixSlotId {
            natural_len: 400,
            commitment_profile: profile,
        },
        commitment: SetupPrefixPublicCommitment { rows: Vec::new() },
        hint: AkitaCommitmentHint::new(1, Vec::new()).unwrap(),
    });

    let long_digits = (0..512)
        .map(|index| (index % 11) as i8 - 5)
        .collect::<Vec<_>>();
    let short_digits = (0..128)
        .map(|index| (index % 7) as i8 - 3)
        .collect::<Vec<_>>();
    let long_evals = long_digits
        .iter()
        .copied()
        .map(Base::from_i8)
        .collect::<Vec<_>>();
    let short_evals = short_digits
        .iter()
        .copied()
        .map(Base::from_i8)
        .collect::<Vec<_>>();
    let setup_source = crate::RecursiveFoldSource::setup_prefix(expanded, slot);
    let long_source = crate::RecursiveFoldSource::witness(Arc::new(
        RecursiveWitnessFlat::from_i8_digits(long_digits),
    ));
    let short_source = crate::RecursiveFoldSource::witness(Arc::new(
        RecursiveWitnessFlat::from_i8_digits(short_digits),
    ));
    let mixed_refs = [&setup_source, &long_source];
    let short_refs = [&short_source];
    let mixed_group = PreparedProverGroup::from_refs(&mixed_refs).unwrap();
    let short_group = PreparedProverGroup::from_refs(&short_refs).unwrap();
    let long_point = (0..9)
        .map(|index| {
            Extension::new(
                Base::from_u64((index + 2) as u64),
                Base::from_u64((2 * index + 3) as u64),
            )
        })
        .collect::<Vec<_>>();
    let short_point = (0..7)
        .map(|index| {
            Extension::new(
                Base::from_u64((index + 5) as u64),
                Base::from_u64((3 * index + 7) as u64),
            )
        })
        .collect::<Vec<_>>();
    let inputs = [
        ExtensionOpeningGroupInput {
            group: &mixed_group,
            point: &long_point,
            ring_dimension: D,
        },
        ExtensionOpeningGroupInput {
            group: &short_group,
            point: &short_point,
            ring_dimension: D,
        },
    ];
    let mut prover_transcript =
        AkitaTranscript::<Base>::new(b"test/mixed-setup-prefix-suffix-dense-oracle");
    let grinding_plan = eor_test_plan(long_point.len() - 1, true);
    let mut prover_transcript =
        akita_types::ProverGrindingTranscript::new(&mut prover_transcript, &grinding_plan).unwrap();
    let proved = prove_extension_opening_reduction::<Base, Extension, _, _, _>(
        &crate::compute::CpuBackend::DEFAULT,
        None,
        &inputs,
        &mut prover_transcript,
        1,
        "recursive",
    )
    .unwrap();
    let nonce_stream = prover_transcript.finish().unwrap();

    let opening_layout = OpeningClaimsLayout::from_groups(vec![
        PolynomialGroupLayout::new(9, 2),
        PolynomialGroupLayout::new(7, 1),
    ])
    .unwrap();
    let openings = vec![
        direct_lifted_mle::<Base, Extension>(&setup_evals, &long_point),
        direct_lifted_mle::<Base, Extension>(&long_evals, &long_point),
        direct_lifted_mle::<Base, Extension>(&short_evals, &short_point),
    ];
    let expected_partials = [
        direct_column_partials::<Base, Extension>(&setup_evals, &long_point),
        direct_column_partials::<Base, Extension>(&long_evals, &long_point),
        direct_column_partials::<Base, Extension>(&short_evals, &short_point),
    ]
    .concat();
    assert_eq!(proved.reduction.proof.partials, expected_partials);
    let mut replay = AkitaTranscript::<Base>::new(b"test/mixed-setup-prefix-suffix-dense-oracle");
    let mut replay =
        akita_types::VerifierGrindingTranscript::new(&mut replay, &nonce_stream, &grinding_plan)
            .unwrap();
    append_claim_values_to_transcript::<Base, Extension, _>(&openings, &mut replay);
    for partial in &proved.reduction.proof.partials {
        append_ext_field::<Base, Extension, _>(&mut replay, ABSORB_EVALUATION_CLAIMS, partial);
    }
    akita_types::TranscriptGrinding::grind_query(
        &mut replay,
        akita_types::GrindingSite::ExtensionOpeningPoint { level: 1 },
    )
    .unwrap();
    let eta = sample_ext_challenge::<Base, Extension, _>(&mut replay, CHALLENGE_SUMCHECK_BATCH);
    let claim_coefficients = akita_types::sample_row_coefficients::<Base, Extension, _>(
        &opening_layout,
        akita_types::GrindingSite::ExtensionOpeningClaimBatch { level: 1 },
        &mut replay,
    )
    .unwrap();

    let (setup_packed, long_factor) =
        direct_tensor_tables::<Base, Extension>(&setup_evals, &long_point, eta);
    let (long_packed, same_long_factor) =
        direct_tensor_tables::<Base, Extension>(&long_evals, &long_point, eta);
    let (short_packed, short_factor) =
        direct_tensor_tables::<Base, Extension>(&short_evals, &short_point, eta);
    assert_eq!(long_factor, same_long_factor);
    let direct_input_claims = [
        setup_packed
            .iter()
            .zip(&long_factor)
            .fold(Extension::zero(), |acc, (&witness, &factor)| {
                acc + witness * factor
            }),
        long_packed
            .iter()
            .zip(&long_factor)
            .fold(Extension::zero(), |acc, (&witness, &factor)| {
                acc + witness * factor
            }),
        short_packed
            .iter()
            .zip(&short_factor)
            .fold(Extension::zero(), |acc, (&witness, &factor)| {
                acc + witness * factor
            }),
    ];
    let input_claim = direct_input_claims
        .iter()
        .zip(&claim_coefficients)
        .fold(Extension::zero(), |acc, (&claim, &coefficient)| {
            acc + coefficient * claim
        });
    replay.append_serde(ABSORB_SUMCHECK_CLAIM, &input_claim);
    let mut round = 0u32;
    let (batched_final_claim, rho) =
        proved
            .reduction
            .proof
            .sumcheck
            .verify::<Base, _, _>(
                input_claim,
                8,
                EXTENSION_OPENING_REDUCTION_DEGREE,
                &mut replay,
                |transcript| {
                    let challenge =
                        akita_types::sample_grinded_sumcheck_challenge::<Base, Extension, _>(
                            transcript,
                            akita_types::SumcheckProtocol::ExtensionOpeningReduction,
                            1,
                            0,
                            round,
                        )?;
                    round = round.checked_add(1).expect("EOR test round count fits u32");
                    Ok(challenge)
                },
            )
            .unwrap();
    replay.finish().unwrap();

    let long_final_factor = akita_sumcheck::multilinear_eval(&long_factor, &rho[..8]).unwrap();
    let short_final_factor = akita_sumcheck::multilinear_eval(&short_factor, &rho[..6]).unwrap()
        * rho[6..].iter().fold(Extension::one(), |acc, &challenge| {
            acc * (Extension::one() - challenge)
        });
    let expected_final_claims = vec![
        akita_sumcheck::multilinear_eval(&setup_packed, &rho[..8]).unwrap() * long_final_factor,
        akita_sumcheck::multilinear_eval(&long_packed, &rho[..8]).unwrap() * long_final_factor,
        akita_sumcheck::multilinear_eval(&short_packed, &rho[..6]).unwrap() * short_final_factor,
    ];
    let expected_batched_final = expected_final_claims
        .iter()
        .zip(&claim_coefficients)
        .fold(Extension::zero(), |acc, (&claim, &coefficient)| {
            acc + coefficient * claim
        });

    assert_eq!(
        proved.reduction.final_factors,
        vec![long_final_factor, short_final_factor]
    );
    assert_eq!(proved.reduction.proof.final_claims, expected_final_claims);
    assert_eq!(batched_final_claim, expected_batched_final);
}

#[test]
fn proof_schedule_from_layout_includes_entire_batch() {
    let catalog = akita_config::test_support::workspace_schedule_catalog::<OneHot>()
        .expect("workspace schedule catalog");
    let batch = OpeningClaimsLayout::from_groups(vec![
        PolynomialGroupLayout::new(16, 1),
        PolynomialGroupLayout::new(16, 1),
        PolynomialGroupLayout::new(32, 2),
    ])
    .expect("multi-group shape");
    assert_eq!(batch.num_groups(), 3);
    let precommitted = catalog
        .resolve_key(&AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(
            16, 1,
        )))
        .expect("independent row")
        .profiles()
        .final_group;
    let schedule = catalog
        .resolve_key(&AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(32, 2),
            precommitteds: vec![precommitted, precommitted],
        })
        .expect("multi-group schedule")
        .schedule()
        .clone();
    let root_params = schedule.root.params.clone();
    assert_eq!(root_params.precommitted_groups().len(), 2);
    for precommitted in root_params.precommitted_groups() {
        assert_eq!(
            precommitted.profile.group,
            PolynomialGroupLayout::new(16, 1)
        );
    }
}
