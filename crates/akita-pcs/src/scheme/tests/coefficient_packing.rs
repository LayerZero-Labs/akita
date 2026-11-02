use super::*;

use akita_config::proof_optimized::fp32;
use akita_params::ScheduleLookupKey;
use akita_params::{basis_weights, OpeningMethod, PolynomialGroupLayout};
use jolt_field::ExtField;

type PackingCfg = crate::test_support::RootCoefficientPackingConfig<fp32::Dense>;
type PackingField = <PackingCfg as CommitmentConfig>::Field;
type PackingExt = <PackingCfg as CommitmentConfig>::ExtField;
type PackingScheme = AkitaCommitmentScheme<PackingCfg>;
type RootEvaluationTraceCfg = crate::test_support::UnsupportedEvaluationTraceConfig<fp32::Dense, 0>;
type RecursiveEvaluationTraceCfg =
    crate::test_support::UnsupportedEvaluationTraceConfig<fp32::Dense, 1>;

#[test]
fn synthetic_packing_row_is_derived_from_one_checked_authority() {
    let catalog = akita_config::test_support::workspace_schedule_catalog::<PackingCfg>(
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .expect("workspace schedule catalog");
    let key = ScheduleLookupKey {
        final_group: PolynomialGroupLayout::singleton(20),
        precommitteds: Vec::new(),
    };
    let first = PackingCfg::derive_catalog_row(&catalog, &key, 64, 1).unwrap();
    let second = PackingCfg::derive_catalog_row(&catalog, &key, 64, 1).unwrap();
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
    let geometry = akita_params::SubringCoefficientPackingGeometry::try_new(
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
        akita_params::CommittedSourceEncoding::CanonicalCoefficientTable,
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
        akita_params::CommittedSourceEncoding::CanonicalCoefficientTable,
    );
    let prefix = successor
        .params
        .setup_prefix()
        .expect("synthetic successor must consume the root setup prefix");
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

    PackingCfg::derive_catalog_row(&catalog, &key, 96, 1)
        .expect_err("a non-production challenge subring must reject");
}

#[test]
fn fixed_root_packing_rejects_a_stale_successor_length() {
    let catalog = akita_config::test_support::workspace_schedule_catalog::<PackingCfg>(
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .expect("workspace schedule catalog");
    let opening_batch = OpeningClaimsLayout::new(20, 1).unwrap();
    let key = ScheduleLookupKey::single(opening_batch.root_final_group_layout().unwrap());
    let row = PackingCfg::derive_catalog_row(&catalog, &key, 64, 1).unwrap();
    let mut schedule = row.schedule().clone();
    schedule.terminal.input_witness_len += 1;
    schedule
        .validate_structure()
        .expect_err("a successor length stale against the packed root must reject");
}

#[test]
fn packing_setup_prefix_dispatch_rejects_an_unsupported_dimension() {
    let result = akita_params::dispatch_for_field!(
        akita_params::ProtocolDispatchSlot::Role(akita_params::RingRole::Inner),
        PackingField,
        96,
        |D_SETUP| Ok::<usize, akita_error::AkitaError>(D_SETUP)
    );
    assert!(result.is_err());
}

#[test]
fn fixed_root_packing_round_trips_in_both_bases() {
    packing_round_trip(1);
}

#[test]
fn level_two_packing_with_setup_prefix_round_trips_in_both_bases() {
    packing_round_trip(2);
}

fn packing_round_trip(prefixed_fold_count: usize) {
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(move || {
            let catalog = akita_config::test_support::workspace_schedule_catalog::<PackingCfg>(
                akita_params::ChunkedWitnessCfg::default_non_chunked(),
            )
            .expect("workspace schedule catalog");
            let num_vars = 20;
            let opening_batch = OpeningClaimsLayout::new(num_vars, 1).unwrap();
            let key = ScheduleLookupKey::single(opening_batch.root_final_group_layout().unwrap());
            let row =
                PackingCfg::derive_catalog_row(&catalog, &key, 64, prefixed_fold_count).unwrap();
            let schedules = akita_config::ValidatedScheduleCatalog::try_new(
                PackingCfg::schedule_family_name(),
                [(row.profiles().clone(), row.schedule().clone())],
                &akita_config::policy_of::<PackingCfg>(
                    akita_params::ChunkedWitnessCfg::default_non_chunked(),
                ),
                PackingCfg::ring_challenge_config,
            )
            .and_then(|catalog| {
                akita_config::TrustedScheduleCatalog::<PackingCfg>::new(
                    catalog,
                    akita_params::ChunkedWitnessCfg::default_non_chunked(),
                )
            })
            .expect("packing schedule catalog");
            let scheme = PackingScheme::new(schedules);
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
            assert_eq!(row.schedule().recursive_folds.len(), prefixed_fold_count);
            for (index, consumer) in row.schedule().recursive_folds.iter().enumerate() {
                assert!(
                    consumer.params.setup_prefix().is_some(),
                    "L{} consumes a prefix",
                    index + 1
                );
                assert!(matches!(
                    consumer.params.opening_method(),
                    OpeningMethod::SubringCoefficientPacking { .. }
                ));
                assert_eq!(consumer.params.witness_chunk.num_chunks, 1);
                assert_eq!(
                    consumer.params.ring_relation_mode,
                    akita_params::RingRelationMode::QuotientLift
                );
                assert_eq!(
                    consumer.predecessor_setup_contribution_mode(),
                    akita_params::SetupContributionMode::Recursive
                );
            }
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
                akita_cpu_backend::DensePoly::from_field_evals(num_vars, &evaluations).unwrap();

            let mut setup = scheme.setup_prover(num_vars, 1).unwrap();
            let setup_prefixes = row
                .schedule()
                .recursive_folds
                .iter()
                .map(|fold| {
                    fold.params
                        .setup_prefix()
                        .unwrap()
                        .slot_id()
                        .expect("setup prefix group")
                })
                .collect::<Vec<_>>();
            setup.export_setup_prefixes(&setup_prefixes).unwrap();
            let stack = CpuBackend::new(setup.expanded().clone()).expect("backend");
            let verifier = self_derived_verifier(&scheme, &setup);
            let akita_cpu_backend::CommitOutput {
                committed_group,
                private_handle: hint,
            } = stack
                .commit(
                    scheme.schedules(),
                    &stack
                        .import_source(vec![polynomial.clone()])
                        .expect("source"),
                    akita_cpu_backend::GroupContext::scheduler_without_precommitted_groups(),
                )
                .unwrap();
            assert_eq!(committed_group.profile(), &row.profiles().final_group);

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
                    vec![expected],
                    committed_group.clone(),
                )
                .unwrap()])
                .unwrap();
                let prover_data = SelectedProverOpeningData::from_committed_claims::<PackingCfg>(
                    prover_claims,
                    vec![hint.clone()],
                    scheme.schedules(),
                )
                .unwrap();
                let selection = prover_data.selection();
                let label = match basis {
                    BasisMode::Lagrange => b"packing/root/lagrange".as_slice(),
                    BasisMode::Monomial => b"packing/root/monomial".as_slice(),
                };
                let proof = scheme
                    .batched_prove(&setup, prover_data, &stack, label, basis)
                    .unwrap();
                assert!(!proof.is_empty());
                let verifier_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                    point.clone(),
                    vec![expected],
                    &committed_group,
                )
                .unwrap()])
                .unwrap();
                let statement = GroupBatchStatement::new(selection, verifier_claims).unwrap();
                verifier
                    .batched_verify(proof.as_slice(), label, statement, basis)
                    .unwrap();

                if basis == BasisMode::Lagrange {
                    let mut malformed = proof.clone();
                    malformed.truncate(malformed.len().saturating_sub(1));
                    let verifier_claims =
                        OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                            point.clone(),
                            vec![expected],
                            &committed_group,
                        )
                        .unwrap()])
                        .unwrap();
                    let statement = GroupBatchStatement::new(selection, verifier_claims).unwrap();
                    assert!(verifier
                        .batched_verify(malformed.as_slice(), label, statement, basis)
                        .is_err());

                    macro_rules! unsupported_evaluation_trace_admission_error {
                        ($config:ty, $context:literal, $expected:literal) => {{
                            let row = <$config>::derive_row(&catalog, &key)
                                .expect(concat!($context, " test row must derive"));
                            row.1
                                .validate_structure()
                                .expect(concat!($context, " test row must be structurally valid"));
                            let error = akita_config::ResolvedScheduleRow::try_new(
                                row.0,
                                row.1,
                                catalog.policy(),
                            )
                            .expect_err(concat!(
                                $context,
                                " must reject at the executable row boundary"
                            ));
                            assert!(
                                matches!(
                                    &error,
                                    akita_error::AkitaError::InvalidSetup(message)
                                        if message == $expected
                                ),
                                "unexpected admission error: {error}"
                            );
                        }};
                    }

                    unsupported_evaluation_trace_admission_error!(
                        RootEvaluationTraceCfg,
                        "root EvaluationTrace",
                        "nonterminal level 0 requires subring coefficient packing"
                    );
                    unsupported_evaluation_trace_admission_error!(
                        RecursiveEvaluationTraceCfg,
                        "multi-chunk level-1 EvaluationTrace",
                        "nonterminal level 1 requires subring coefficient packing"
                    );
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
