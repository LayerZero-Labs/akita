use super::*;

use akita_algebra::CyclotomicRing;
use akita_config::proof_optimized::fp32;
use akita_types::{basis_weights, AkitaScheduleLookupKey, OpeningMethod, PolynomialGroupLayout};
use jolt_field::ExtField;

type PackingCfg = crate::test_support::RootCoefficientPackingConfig<fp32::Dense>;
type PackingField = <PackingCfg as CommitmentConfig>::Field;
type PackingExt = <PackingCfg as CommitmentConfig>::ExtField;
type PackingScheme = AkitaCommitmentScheme<PackingCfg>;
type RootEvaluationTraceCfg = crate::test_support::EarlyEvaluationTraceConfig<fp32::Dense, 0>;
type RecursiveEvaluationTraceCfg = crate::test_support::EarlyEvaluationTraceConfig<fp32::Dense, 1>;

#[test]
fn synthetic_packing_row_is_derived_from_one_checked_authority() {
    let catalog = akita_config::test_support::workspace_schedule_catalog::<PackingCfg>()
        .expect("workspace schedule catalog");
    let key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::singleton(20),
        precommitteds: Vec::new(),
    };
    let first = PackingCfg::derive_catalog_row(&catalog, &key, 64).unwrap();
    let second = PackingCfg::derive_catalog_row(&catalog, &key, 64).unwrap();
    assert_eq!(first.selection(), second.selection());
    assert_eq!(first.schedule(), second.schedule());

    let schedule = first.schedule();
    let root = &schedule.root.params;
    assert_eq!(PackingCfg::EXT_DEGREE, 4);
    let OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension,
    } = root.opening_method()
    else {
        panic!("synthetic root must use coefficient packing");
    };
    assert_eq!(challenge_subring_dimension, 64);
    let geometry = akita_types::SubringCoefficientPackingGeometry::try_new(
        PackingCfg::EXT_DEGREE,
        root.d_a(),
        challenge_subring_dimension,
    )
    .unwrap();
    assert!(
        geometry.packing_factor() > 1,
        "synthetic packing row must reduce the physical opening width"
    );
    assert_eq!(
        root.d_a(),
        geometry.extension_degree()
            * geometry.challenge_subring_dimension()
            * geometry.packing_factor(),
    );
    assert_eq!(
        geometry.partial_base_field_width() / root.role_dims().d_d(),
        2,
    );
    assert_eq!(
        root.source_encoding,
        akita_types::CommittedSourceEncoding::CanonicalCoefficientTable,
    );

    let successor = &schedule.recursive_folds[0];
    assert_eq!(
        schedule.root.output_witness_len,
        successor.input_witness_len
    );
    assert!(matches!(
        successor.params.opening_method(),
        OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: 64
        }
    ));
    assert_eq!(
        successor.params.source_encoding,
        akita_types::CommittedSourceEncoding::CanonicalCoefficientTable,
    );
    let prefix = successor
        .params
        .setup_prefix()
        .expect("synthetic successor must consume the root setup prefix");
    assert_eq!(
        prefix.source_encoding(),
        akita_types::CommittedSourceEncoding::CanonicalCoefficientTable,
    );
    assert!(matches!(
        prefix.opening.opening_method,
        OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: 64
        }
    ));
    assert_eq!(
        successor.output_witness_len,
        schedule.terminal.input_witness_len
    );
    schedule.validate_structure().unwrap();

    PackingCfg::derive_catalog_row(&catalog, &key, 96)
        .expect_err("a non-production challenge subring must reject");
}

#[test]
fn fixed_root_packing_rejects_a_stale_successor_length() {
    let catalog = akita_config::test_support::workspace_schedule_catalog::<PackingCfg>()
        .expect("workspace schedule catalog");
    let opening_batch = OpeningClaimsLayout::new(20, 1).unwrap();
    let key = AkitaScheduleLookupKey::single(opening_batch.root_final_group_layout().unwrap());
    let row = PackingCfg::derive_catalog_row(&catalog, &key, 64).unwrap();
    let mut schedule = row.schedule().clone();
    schedule.terminal.input_witness_len += 1;
    schedule
        .validate_structure()
        .expect_err("a successor length stale against the packed root must reject");
}

#[test]
fn packing_setup_prefix_dispatch_rejects_an_unsupported_dimension() {
    let result = akita_types::dispatch_for_field!(
        akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
        PackingField,
        96,
        |D_SETUP| Ok::<usize, akita_error::AkitaError>(D_SETUP)
    );
    assert!(result.is_err());
}

#[test]
fn fixed_root_packing_round_trips_in_both_bases() {
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(|| {
            let catalog = akita_config::test_support::workspace_schedule_catalog::<PackingCfg>()
                .expect("workspace schedule catalog");
            let num_vars = 20;
            let opening_batch = OpeningClaimsLayout::new(num_vars, 1).unwrap();
            let key =
                AkitaScheduleLookupKey::single(opening_batch.root_final_group_layout().unwrap());
            let row = PackingCfg::derive_catalog_row(&catalog, &key, 64).unwrap();
            let schedules = akita_config::TrustedScheduleCatalog::try_new(
                PackingCfg::schedule_family_name(),
                [(row.profiles().clone(), row.schedule().clone())],
                &akita_config::policy_of::<PackingCfg>(),
                PackingCfg::ring_challenge_config,
            )
            .expect("packing schedule catalog");
            let scheme = PackingScheme::new(schedules).expect("packing scheme");
            let root = &row.schedule().root.params;
            let OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension,
            } = root.opening_method()
            else {
                panic!("test catalog did not select coefficient packing");
            };
            assert!(
                PackingCfg::EXT_DEGREE * challenge_subring_dimension < root.d_a(),
                "the authenticated fixture must exercise reduced packing width"
            );
            assert_eq!(
                PackingCfg::EXT_DEGREE * challenge_subring_dimension / root.role_dims().d_d(),
                2,
                "the authenticated fixture must exercise two physical D subcolumns"
            );
            assert_eq!(row.schedule().recursive_folds.len(), 1);
            assert!(row.schedule().recursive_folds[0]
                .params
                .setup_prefix()
                .is_some());
            assert!(
                root.open().matrix.input_width()
                    < root.open().digits.num_digits
                        * root.blocks().live_blocks
                        * root.d_a().div_ceil(root.role_dims().d_d()),
                "the fixed row must shrink the shared D input"
            );

            let source_len = 1usize << num_vars;
            let evaluations = (0..source_len)
                .map(|index| PackingField::from_i64((index % 7) as i64 - 3))
                .collect::<Vec<_>>();
            let polynomial =
                akita_prover::DensePoly::from_field_evals(num_vars, &evaluations).unwrap();
            let polynomial =
                akita_prover::MultilinearPolynomial::<PackingField, usize>::dense(polynomial);

            let mut setup = scheme.setup_prover(num_vars, 1).unwrap();
            let setup_prefix = row.schedule().recursive_folds[0]
                .params
                .setup_prefix()
                .as_ref()
                .unwrap()
                .slot_id()
                .expect("setup prefix group");
            assert_eq!(
                setup_prefix.d_setup(),
                setup_prefix
                    .commitment_profile
                    .inner
                    .matrix
                    .ring_dimension(),
                "the prefix dispatcher must use its frozen A-ring dimension"
            );
            let prefix_prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
            let prefix_slot = akita_types::dispatch_for_field!(
                akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
                PackingField,
                setup_prefix.d_setup(),
                |D_SETUP| {
                    akita_prover::commit_setup_prefix::<PackingField, D_SETUP, _>(
                        setup.expanded.as_ref(),
                        &CpuBackend::DEFAULT,
                        &prefix_prepared,
                        &setup_prefix.commitment_profile,
                        setup_prefix.n_prefix().unwrap(),
                        setup_prefix.natural_len,
                    )
                }
            )
            .unwrap();
            setup.prefix_slots.insert(prefix_slot).unwrap();
            let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
            let stack = akita_prover::UniformProverStack::uniform(
                &CpuBackend::DEFAULT,
                &prepared,
                setup.expanded.as_ref(),
            )
            .unwrap();
            let verifier_setup = scheme.setup_verifier(&setup).unwrap();
            let akita_prover::CommitOutput {
                committed_group,
                hint,
            } = scheme
                .commit::<_, _>(
                    &setup,
                    std::slice::from_ref(&polynomial),
                    &stack,
                    akita_prover::GroupContext::scheduler_without_precommitted_groups(),
                )
                .unwrap();
            assert_eq!(committed_group.profile(), &row.profiles().final_group);
            akita_types::dispatch_for_field!(
                akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
                PackingField,
                root.d_a(),
                |D_A| {
                    let a_matrix = setup
                        .expanded
                        .shared_matrix()
                        .ring_view::<D_A>(
                            root.inner().matrix.output_rank(),
                            root.inner().matrix.input_width(),
                        )
                        .unwrap();
                    let mut source_digits = Vec::new();
                    for coefficients in evaluations
                        .chunks_exact(D_A)
                        .take(root.blocks().positions_per_block)
                    {
                        source_digits.extend(
                            CyclotomicRing::<PackingField, D_A>::from_coefficients(
                                coefficients.try_into().unwrap(),
                            )
                            .balanced_decompose_pow2_i8(
                                root.inner().digits.num_digits,
                                root.inner().digits.log_basis,
                            ),
                        );
                    }
                    let hint_rows = hint.inner_rows()[0].as_ring_slice::<D_A>().unwrap();
                    let output_rank = root.inner().matrix.output_rank();
                    assert_eq!(hint_rows.len(), output_rank * root.blocks().live_blocks);
                    for (row, actual) in hint_rows.iter().take(output_rank).enumerate() {
                        let expected = a_matrix.row(row).unwrap().iter().zip(&source_digits).fold(
                            CyclotomicRing::zero(),
                            |sum, (matrix, digits)| {
                                sum + *matrix
                                    * CyclotomicRing::from_coefficients(std::array::from_fn(
                                        |index| PackingField::from_i8(digits[index]),
                                    ))
                            },
                        );
                        assert_eq!(*actual, expected, "A hint row {row} mismatch");
                    }
                    Ok::<(), akita_error::AkitaError>(())
                }
            )
            .unwrap();
            let polynomial_refs = [&polynomial];
            let point = (0..num_vars)
                .map(|index| PackingExt::from_u64((index as u64).wrapping_mul(3).wrapping_add(1)))
                .collect::<Vec<_>>();

            for basis in [BasisMode::Lagrange, BasisMode::Monomial] {
                let weights = basis_weights(&point, basis).unwrap();
                let expected = evaluations.iter().zip(&weights).fold(
                    PackingExt::zero(),
                    |sum, (&coefficient, &weight)| {
                        sum + weight * PackingExt::lift_base(coefficient)
                    },
                );
                let prover_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                    point.clone(),
                    vec![PackingExt::zero()],
                    committed_group.clone(),
                )
                .unwrap()])
                .unwrap();
                let prover_data = SelectedProverOpeningData::from_committed_claims::<PackingCfg>(
                    prover_claims,
                    vec![hint.clone()],
                    vec![&polynomial_refs],
                    scheme.schedules(),
                )
                .unwrap();
                let selection = prover_data.selection();
                let label = match basis {
                    BasisMode::Lagrange => b"packing/root/lagrange".as_slice(),
                    BasisMode::Monomial => b"packing/root/monomial".as_slice(),
                };
                let mut prover_transcript = AkitaTranscript::<PackingField>::new(label);
                let proof = scheme
                    .batched_prove::<_, _, _>(
                        &setup,
                        prover_data,
                        &stack,
                        &mut prover_transcript,
                        basis,
                    )
                    .unwrap();
                assert!(
                    proof.root.stage3_sumcheck_proof().is_some(),
                    "packing root must offload its setup contribution through Stage 3"
                );

                let shape = proof.shape();
                let mut encoded = Vec::new();
                proof.serialize_uncompressed(&mut encoded).unwrap();
                let proof =
                    AkitaBatchedProof::<PackingField, PackingExt>::deserialize_uncompressed(
                        encoded.as_slice(),
                        &shape,
                    )
                    .unwrap();
                let verifier_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                    point.clone(),
                    vec![expected],
                    &committed_group,
                )
                .unwrap()])
                .unwrap();
                let statement = GroupBatchStatement::new(selection, verifier_claims).unwrap();
                let mut verifier_transcript = AkitaTranscript::<PackingField>::new(label);
                scheme
                    .batched_verify(
                        &proof,
                        &verifier_setup,
                        &mut verifier_transcript,
                        statement,
                        basis,
                    )
                    .unwrap();

                if basis == BasisMode::Lagrange {
                    let mut malformed = proof.clone();
                    malformed.root.extension_opening_reduction =
                        Some(ExtensionOpeningReductionProof {
                            partials: vec![PackingExt::zero()],
                            sumcheck: akita_sumcheck::SumcheckProof {
                                round_polys: Vec::new(),
                            },
                            final_claims: Vec::new(),
                        });
                    let verifier_claims =
                        OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                            point.clone(),
                            vec![expected],
                            &committed_group,
                        )
                        .unwrap()])
                        .unwrap();
                    let statement = GroupBatchStatement::new(selection, verifier_claims).unwrap();
                    #[cfg(feature = "logging-transcript")]
                    let mut transcript = akita_transcript::LoggingTranscript::wrap(
                        AkitaTranscript::<PackingField>::new(label),
                    );
                    #[cfg(not(feature = "logging-transcript"))]
                    let mut transcript = AkitaTranscript::<PackingField>::new(label);
                    assert!(scheme
                        .batched_verify(
                            &malformed,
                            &verifier_setup,
                            &mut transcript,
                            statement,
                            basis,
                        )
                        .is_err());
                    #[cfg(feature = "logging-transcript")]
                    assert!(
                        transcript.events().is_empty(),
                        "unexpected packing EOR must reject before transcript replay"
                    );

                    macro_rules! assert_early_evaluation_trace_rejects_at_catalog_boundary {
                        ($config:ty, $context:literal) => {{
                            let result = <$config>::derive_row(&catalog, &key)
                                .and_then(|row| {
                                    akita_config::TrustedScheduleCatalog::try_new(
                                        <$config>::schedule_family_name(),
                                        [(row.profiles().clone(), row.schedule().clone())],
                                        &akita_config::policy_of::<$config>(),
                                        <$config>::ring_challenge_config,
                                    )
                                })
                                .and_then(AkitaCommitmentScheme::<$config>::new);
                            assert!(
                                result.is_err(),
                                concat!($context, " must reject at the trusted catalog boundary")
                            );
                        }};
                    }

                    assert_early_evaluation_trace_rejects_at_catalog_boundary!(
                        RootEvaluationTraceCfg,
                        "root EvaluationTrace"
                    );
                    assert_early_evaluation_trace_rejects_at_catalog_boundary!(
                        RecursiveEvaluationTraceCfg,
                        "level-1 EvaluationTrace"
                    );
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
