//! One binary source in the canonical grouped opening and caller transcript.
#![cfg(feature = "labinius-trinomial")]

mod common;
#[path = "binary_source_end_to_end/fault_backend.rs"]
mod fault_backend;
#[path = "binary_source_end_to_end/field_switch.rs"]
mod field_switch;
#[path = "binary_source_end_to_end/reference.rs"]
mod reference;
#[path = "binary_source_end_to_end/statement_binding.rs"]
mod statement_binding;

use akita_algebra::binary::{BinaryField128, BinaryField162};
use akita_challenges::FoldChallengeFamily;
use akita_config::{
    proof_optimized::fp128::DenseBounded, CommitmentConfig, TrustedScheduleCatalog,
};
use akita_cpu_backend::{CpuBackend, DensePoly, GroupContext};
use akita_error::AkitaError;
use akita_params::{
    sis::CommittedSourceClass, BasisMode, ChunkedWitnessCfg, DecompositionParams,
    PolynomialGroupLayout, ScheduleLookupKey, SisModulusProfileId,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::{new_prover_channel, new_verifier_channel};
use akita_types::binary_source::{
    BinaryCommittedGroupManifest, BinaryIndexMap, BinarySourceDescriptor, BinaryWordClaim,
};
use akita_types::{CommittedGroup, OpeningClaims, PolynomialGroupClaims};
use common::{init_rayon_pool, run_on_large_stack, selected_prover_data, selected_statement};
use fault_backend::{Fault, FaultBackend};
use jolt_field::{One, Prime128Offset275, Ring, Zero};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

type F = Prime128Offset275;

/// Existing ordinary suffix catalog, explicitly selected only by this test.
/// The small binary source's rank-three A and rank-two B have no security claim.
#[derive(Clone)]
struct BinaryTestSuffix;

impl CommitmentConfig for BinaryTestSuffix {
    type Field = F;
    type ExtField = F;

    const RING_DIMENSION_SCHEDULE: akita_schedules::RingDimensionSchedule =
        DenseBounded::RING_DIMENSION_SCHEDULE;

    fn decomposition() -> DecompositionParams {
        DecompositionParams {
            log_basis: DenseBounded::decomposition().log_basis,
            log_commit_bound: 5,
            log_open_bound: Some(128),
        }
    }

    fn ring_challenge_config(dimension: usize) -> Result<FoldChallengeFamily, AkitaError> {
        DenseBounded::ring_challenge_config(dimension)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        SisModulusProfileId::Q128Offset275
    }

    fn opening_basis_range() -> (u32, u32) {
        DenseBounded::opening_basis_range()
    }

    fn inner_basis_range() -> (u32, u32) {
        DenseBounded::inner_basis_range()
    }

    fn committed_source_class() -> CommittedSourceClass {
        DenseBounded::committed_source_class()
    }

    fn schedule_family_name() -> &'static str {
        "labinius_fp128_digits_b4"
    }
}

fn test_catalog() -> TrustedScheduleCatalog<BinaryTestSuffix> {
    // Match the workspace tests' runtime artifact boundary, with the explicit
    // experimental catalog directory and its dev-protocol counterpart.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../artifacts")
        .join(if akita_params::DEV_PROTOCOL {
            "schedules-labinius-dev"
        } else {
            "schedules-labinius"
        })
        .join(format!("{}.aks", BinaryTestSuffix::schedule_family_name()));
    let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    TrustedScheduleCatalog::<BinaryTestSuffix>::from_artifact_bytes(
        &bytes,
        ChunkedWitnessCfg::default_non_chunked(),
    )
    .unwrap()
}

fn scalar(low: u64, middle: u64, high: u64) -> BinaryField162 {
    BinaryField162::from_words([low, middle, high]).unwrap()
}

/// Reference word MLE is independent of the field-switch arithmetic.
fn honest_claim(descriptor: &BinarySourceDescriptor, words: &[u128]) -> BinaryWordClaim {
    let split = descriptor.index_map().claim_variable_split();
    let point = (0..split.intra_carrier + split.carrier_address)
        .map(|axis| reference::host(31 + axis as u128 * 17))
        .collect::<Vec<_>>();
    reference::claim(descriptor, words, &point)
}

#[test]
fn binary_group_opens_once_inside_the_callers_transcript() {
    init_rayon_pool();
    run_on_large_stack(|| run_fixture(false, None, 128));
}

#[test]
fn binary_and_ordinary_groups_share_one_opening() {
    init_rayon_pool();
    run_on_large_stack(|| run_fixture(true, None, 128));
}

#[test]
fn field_switch_binary_root_rejects_a_foreign_source_commitment() {
    init_rayon_pool();
    run_on_large_stack(|| run_fixture(false, Some(Fault::ForeignCommitment), 128));
}

#[test]
fn binary_root_rejects_a_foreign_partial_commitment() {
    init_rayon_pool();
    run_on_large_stack(|| run_fixture(false, Some(Fault::ForeignPartials), 128));
}

#[test]
fn field_switch_narrow_source_mutations_authenticate() {
    init_rayon_pool();
    run_on_large_stack(|| run_fixture(false, None, 8));
}

fn run_fixture(mixed: bool, authentication_fault: Option<Fault>, word_width: usize) {
    let catalog = test_catalog();
    let scheme = AkitaCommitmentScheme::<BinaryTestSuffix>::new(catalog);
    let singleton_carrier = scheme
        .schedules()
        .resolve_key(&ScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(if mixed { 23 } else { 18 }),
        ))
        .unwrap()
        .profiles()
        .final_group;
    let selected_profiles = if mixed {
        scheme
            .schedules()
            .resolve_key(&ScheduleLookupKey {
                final_group: PolynomialGroupLayout::singleton(18),
                precommitteds: vec![singleton_carrier],
            })
            .unwrap()
            .profiles()
            .clone()
    } else {
        scheme
            .schedules()
            .resolve_key(&ScheduleLookupKey::single(
                PolynomialGroupLayout::singleton(18),
            ))
            .unwrap()
            .profiles()
            .clone()
    };
    let carrier = if mixed {
        *selected_profiles.precommitteds.first().unwrap()
    } else {
        selected_profiles.final_group
    };
    let setup = scheme
        .setup_prover(if mixed { 23 } else { 18 }, if mixed { 2 } else { 1 })
        .unwrap();
    let backend = Arc::new(CpuBackend::new(setup.expanded.clone()).unwrap());
    let index = BinaryIndexMap::new(word_width, 8 * (128 / word_width), 8, 2).unwrap();
    let profiles = BinaryCommittedGroupManifest::commitment_profiles(&index, 1, 0).unwrap();
    let descriptor = BinarySourceDescriptor::new(index, vec![[29; 32]], profiles).unwrap();
    let manifest = BinaryCommittedGroupManifest::new(descriptor.clone(), 0, &carrier).unwrap();
    let words = vec![
        0x0123_4567_89ab_cdef_fedc_ba98_7654_3210,
        0x8000_0000_0000_0000_0000_0000_0000_0001,
        0xaaaa_5555_aaaa_5555_aaaa_5555_aaaa_5555,
        0x3333_cccc_3333_cccc_3333_cccc_3333_cccc,
        0x1212_3434_5656_7878_9090_abab_cdcd_efef,
        0xffff_0000_ffff_0000_ffff_0000_ffff_0000,
        0x0101_0202_0303_0404_0505_0606_0707_0808,
        0x1357_9bdf_2468_ace0_1357_9bdf_2468_ace0,
    ];
    let claim = honest_claim(&descriptor, &words);
    let logical_words = reference::logical_words(&descriptor, &words);
    let source = backend
        .import_binary_source(manifest.clone(), carrier, vec![logical_words.clone()])
        .unwrap();
    let committed = backend
        .commit(
            scheme.schedules(),
            &source,
            GroupContext::explicit(&carrier),
        )
        .unwrap();
    let binary_handle = committed.private_handle.clone();
    let original_commitment = if matches!(authentication_fault, Some(Fault::ForeignCommitment)) {
        let mut foreign_words = words.clone();
        foreign_words[0] ^= 1;
        assert_ne!(
            honest_claim(&descriptor, &foreign_words).value(),
            claim.value()
        );
        let foreign = backend
            .import_binary_source(
                manifest.clone(),
                carrier,
                vec![reference::logical_words(&descriptor, &foreign_words)],
            )
            .unwrap();
        let foreign = backend
            .commit(
                scheme.schedules(),
                &foreign,
                GroupContext::explicit(&carrier),
            )
            .unwrap()
            .committed_group;
        assert_ne!(foreign.commitment(), committed.committed_group.commitment());
        foreign
    } else {
        committed.committed_group
    };
    let mut commitment_bytes = Vec::new();
    original_commitment
        .serialize_compressed(&mut commitment_bytes)
        .unwrap();
    let commitment =
        CommittedGroup::<F>::deserialize_compressed(commitment_bytes.as_slice(), &()).unwrap();
    assert_eq!(commitment, original_commitment);
    // A binary word claim cannot be attached to an ordinary group kind, even if
    // its payload and carrier metadata came from the same binary commitment.
    let wrong_kind = CommittedGroup::new(carrier, commitment.commitment().clone());
    assert!(PolynomialGroupClaims::<F, _>::new_binary(wrong_kind, claim.clone()).is_err());
    let duplicate = OpeningClaims::from_groups(vec![
        PolynomialGroupClaims::<F, _>::new_binary(commitment.clone(), claim.clone()).unwrap(),
        PolynomialGroupClaims::<F, _>::new_binary(commitment.clone(), claim.clone()).unwrap(),
    ]);
    assert!(matches!(duplicate, Err(AkitaError::InvalidInput(_))));
    let (ordinary_commitment, ordinary_handle, ordinary_point, ordinary_value) = if mixed {
        let coefficients = (0..1usize << 18)
            .map(|index| F::from_u64(((index ^ (index >> 3) ^ (index >> 11)) & 1) as u64))
            .collect::<Vec<_>>();
        let point = (0..18)
            .map(|index| F::from_u64(31 + index * 17))
            .collect::<Vec<_>>();
        let mut values = coefficients.clone();
        for &coordinate in &point {
            values = values
                .chunks_exact(2)
                .map(|pair| (F::one() - coordinate) * pair[0] + coordinate * pair[1])
                .collect();
        }
        let value = values[0];
        let ordinary_source = backend
            .import_source(vec![DensePoly::from_field_evals(18, &coefficients).unwrap()])
            .unwrap();
        let ordinary = backend
            .commit(
                scheme.schedules(),
                &ordinary_source,
                GroupContext::explicit(&selected_profiles.final_group),
            )
            .unwrap();
        (
            Some(ordinary.committed_group),
            Some(ordinary.private_handle),
            point,
            value,
        )
    } else {
        (None, None, Vec::new(), F::zero())
    };
    let mut groups =
        vec![PolynomialGroupClaims::<F, _>::new_binary(commitment.clone(), claim.clone()).unwrap()];
    let mut handles = vec![committed.private_handle];
    if let (Some(ordinary), Some(handle)) = (&ordinary_commitment, ordinary_handle) {
        groups.push(
            PolynomialGroupClaims::new(
                ordinary_point.clone(),
                vec![ordinary_value],
                ordinary.clone(),
            )
            .unwrap(),
        );
        handles.push(handle);
    }
    let claims = OpeningClaims::from_groups(groups).unwrap();
    let data = selected_prover_data::<BinaryTestSuffix>(claims, handles, scheme.schedules());
    let statement = |claim: &BinaryWordClaim, point: &[F], value: F| {
        let owned =
            PolynomialGroupClaims::<F, _>::new_binary(commitment.clone(), claim.clone()).unwrap();
        let mut groups = vec![owned.map_commitment(|_| &commitment)];
        if let Some(ordinary) = &ordinary_commitment {
            groups.push(PolynomialGroupClaims::new(point.to_vec(), vec![value], ordinary).unwrap());
        }
        selected_statement::<BinaryTestSuffix>(
            OpeningClaims::from_groups(groups).unwrap(),
            scheme.schedules(),
        )
    };

    let prove_fault = |fault: Fault| {
        let verifier = scheme
            .verifier(scheme.setup_verifier(&setup).unwrap())
            .unwrap();
        let attacker = FaultBackend::new(
            backend.clone(),
            setup.expanded.descriptor().setup_seed.seed,
            logical_words.clone(),
            fault,
        );
        let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::<F, _>::new_binary(
            commitment.clone(),
            claim.clone(),
        )
        .unwrap()])
        .unwrap();
        let data = selected_prover_data::<BinaryTestSuffix>(
            claims,
            vec![binary_handle.clone()],
            scheme.schedules(),
        );
        let mut prover = new_prover_channel(b"binary-source-end-to-end-v1", b"caller").unwrap();
        prover.public_message(b"application-prefix");
        let resolved = scheme
            .schedules()
            .resolve_selection(data.selection())
            .unwrap();
        let required = akita_config::required_setup_prefix_slot_ids_for_schedule(
            resolved.schedule(),
            data.opening_layout(),
        )
        .unwrap();
        let prefixes = backend
            .import_setup_prefixes(&setup.prefix_slots, &required)
            .unwrap();
        let mut registry = akita_prover::BackendRegistry::<BinaryTestSuffix>::new().unwrap();
        let root = registry.register(&attacker, &prefixes).unwrap();
        let mut route =
            akita_prover::FixedFoldRoute::new(vec![root; resolved.schedule().num_fold_levels()]);
        akita_prover::batched_prove_in_session::<BinaryTestSuffix, _>(
            setup.expanded.descriptor(),
            scheme.schedules(),
            &registry,
            data,
            &mut prover,
            BasisMode::Lagrange,
            &mut route,
        )
        .unwrap_or_else(|error| panic!("adversarial {fault:?} proof did not finish: {error}"));
        let malformed = prover.narg_string().to_vec();
        eprintln!(
            "completed adversarial {fault:?} proof: {} bytes",
            malformed.len()
        );
        let rejected = catch_unwind(AssertUnwindSafe(|| {
            let mut replay =
                new_verifier_channel(b"binary-source-end-to-end-v1", b"caller", &malformed)
                    .unwrap();
            replay.public_message(b"application-prefix");
            verifier.batched_verify_in_session(
                &mut replay,
                statement(&claim, &ordinary_point, ordinary_value),
                BasisMode::Lagrange,
            )
        }));
        assert!(rejected.is_ok(), "{fault:?} caused a verifier panic");
        assert!(rejected.unwrap().is_err(), "{fault:?} was accepted");
    };
    if let Some(fault) = authentication_fault {
        prove_fault(fault);
        return;
    }
    let mut prover = new_prover_channel(b"binary-source-end-to-end-v1", b"caller").unwrap();
    prover.public_message(b"application-prefix");
    scheme
        .batched_prove_in_session(
            &setup,
            data,
            backend.as_ref(),
            &mut prover,
            BasisMode::Lagrange,
        )
        .unwrap();
    let proof = prover.narg_string().to_vec();
    eprintln!(
        "{} ordinary grouped proof (word width {}): {} bytes",
        if mixed {
            "mixed binary/ordinary"
        } else {
            "binary-only"
        },
        descriptor.index_map().word_width(),
        proof.len()
    );
    prover.public_message(b"application-suffix");
    let continuation: [u8; 32] = prover.verifier_message();
    let verifier = scheme
        .verifier(scheme.setup_verifier(&setup).unwrap())
        .unwrap();
    let mismatch = catch_unwind(AssertUnwindSafe(|| {
        let mut groups = vec![PolynomialGroupClaims::new(
            vec![F::zero(); carrier.group.num_vars()],
            vec![F::zero()],
            &commitment,
        )
        .unwrap()];
        if let Some(ordinary) = &ordinary_commitment {
            groups.push(
                PolynomialGroupClaims::new(ordinary_point.clone(), vec![ordinary_value], ordinary)
                    .unwrap(),
            );
        }
        let statement = selected_statement::<BinaryTestSuffix>(
            OpeningClaims::from_groups(groups).unwrap(),
            scheme.schedules(),
        );
        let mut replay =
            new_verifier_channel(b"binary-source-end-to-end-v1", b"caller", &proof).unwrap();
        replay.public_message(b"application-prefix");
        verifier.batched_verify_in_session(&mut replay, statement, BasisMode::Lagrange)
    }));
    assert!(mismatch.is_ok(), "group-kind mismatch panicked");
    assert!(
        mismatch.unwrap().is_err(),
        "ordinary claim accepted a binary commitment kind"
    );
    let unsupported_basis = catch_unwind(AssertUnwindSafe(|| {
        let mut replay =
            new_verifier_channel(b"binary-source-end-to-end-v1", b"caller", &proof).unwrap();
        replay.public_message(b"application-prefix");
        verifier.batched_verify_in_session(
            &mut replay,
            statement(&claim, &ordinary_point, ordinary_value),
            BasisMode::Monomial,
        )
    }));
    assert!(unsupported_basis.is_ok(), "unsupported basis panicked");
    assert!(
        unsupported_basis.unwrap().is_err(),
        "binary root accepted an unsupported basis"
    );
    let mut replay =
        new_verifier_channel(b"binary-source-end-to-end-v1", b"caller", &proof).unwrap();
    replay.public_message(b"application-prefix");
    verifier
        .batched_verify_in_session(
            &mut replay,
            statement(&claim, &ordinary_point, ordinary_value),
            BasisMode::Lagrange,
        )
        .unwrap();
    replay.public_message(b"application-suffix");
    assert_eq!(replay.verifier_message::<[u8; 32]>().unwrap(), continuation);
    replay.check_eof().unwrap();

    let rejects = |bytes: &[u8], claim: &BinaryWordClaim, point: &[F], value: F| {
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let mut replay =
                new_verifier_channel(b"binary-source-end-to-end-v1", b"caller", bytes).unwrap();
            replay.public_message(b"application-prefix");
            verifier
                .batched_verify_in_session(
                    &mut replay,
                    statement(claim, point, value),
                    BasisMode::Lagrange,
                )
                .and_then(|_| replay.check_eof())
        }));
        assert!(outcome.is_ok(), "malformed opening caused a verifier panic");
        assert!(outcome.unwrap().is_err(), "malformed opening was accepted");
    };
    let mut value = *claim.value();
    value[0] ^= 1;
    let altered = BinaryWordClaim::new(
        &descriptor,
        *claim.polynomial(),
        claim.point().to_vec(),
        value,
    )
    .unwrap();
    rejects(&proof, &altered, &ordinary_point, ordinary_value);
    for coordinate in 0..claim.point().len() {
        let mut point = claim.point().to_vec();
        point[coordinate][0] ^= 1;
        let altered =
            BinaryWordClaim::new(&descriptor, *claim.polynomial(), point, *claim.value()).unwrap();
        rejects(&proof, &altered, &ordinary_point, ordinary_value);
    }
    // Preserve the original U commitment mutation after the frontend prefix;
    // also mutate a partial and the mandatory terminal in the full opening.
    for offset in [0, 128 * 16 + 3 * 42, 128 * 16 + 3 * 42 + 21] {
        let mut altered = proof.clone();
        altered[offset] ^= 1;
        rejects(&altered, &claim, &ordinary_point, ordinary_value);
    }
    for offset in [proof.len() / 2, proof.len() - 1] {
        let mut altered = proof.clone();
        altered[offset] ^= 1;
        rejects(&altered, &claim, &ordinary_point, ordinary_value);
    }
    rejects(
        &proof[..proof.len() - 1],
        &claim,
        &ordinary_point,
        ordinary_value,
    );
    let mut trailing = proof.clone();
    trailing.push(0);
    rejects(&trailing, &claim, &ordinary_point, ordinary_value);
    if mixed {
        rejects(&proof, &claim, &ordinary_point, ordinary_value + F::one());
        let mut point = ordinary_point.clone();
        point[0] += F::one();
        rejects(&proof, &claim, &point, ordinary_value);
    } else {
        // These adversarial backends commit a changed private digit table and
        // finish an entire proof using a deliberately incorrect Stage-2 public
        // oracle. Rejection therefore exercises the verifier, rather than only
        // the honest prover's construction checks.
        for fault in Fault::ALL {
            prove_fault(fault);
        }
    }
}

/// Scan the two root orchestration modules; helpers are outside this guard.
#[test]
fn binary_root_orchestrators_do_not_invoke_a_nested_driver() {
    for source in [
        include_str!("../../akita-prover/src/protocol/prove/binary_root.rs"),
        include_str!("../../akita-verifier/src/fold/binary_root.rs"),
    ] {
        for forbidden in [
            "akita_labinius_pcs::",
            "akita_labinius_prover::",
            "akita_labinius_verifier::",
            "new_prover_channel(",
            "new_verifier_channel(",
            "batched_prove(",
            "batched_verify(",
        ] {
            assert!(
                !source.contains(forbidden),
                "nested root invocation: {forbidden}"
            );
        }
    }
}
