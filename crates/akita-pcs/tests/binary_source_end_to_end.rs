//! One binary source in the canonical grouped opening and caller transcript.
#![cfg(feature = "labinius-trinomial")]

mod common;
#[path = "binary_source_end_to_end/fault_backend.rs"]
mod fault_backend;

use akita_algebra::binary::BinaryField162;
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
    BinaryCommittedGroupManifest, BinaryIndexMap, BinarySourceDescriptor, BinaryTerminalClaim,
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

fn scalar(low: u64, middle: u64, high: u64) -> BinaryField162 {
    BinaryField162::from_words([low, middle, high]).unwrap()
}

/// Independent source evaluation: collapse the low carrier bit for columns,
/// followed by the two row bits. It does not use the root's public-weight code.
fn honest_claim(descriptor: &BinarySourceDescriptor, words: &[u128]) -> BinaryTerminalClaim {
    let row = [scalar(3, 5, 1), scalar(7, 11, 2)];
    let column = [scalar(13, 17, 3)];
    let mut values = words
        .iter()
        .map(|&word| scalar(word as u64, (word >> 64) as u64, 0))
        .collect::<Vec<_>>();
    for coordinate in column.iter().chain(&row) {
        values = values
            .chunks_exact(2)
            .map(|pair| (BinaryField162::ONE + *coordinate) * pair[0] + *coordinate * pair[1])
            .collect();
    }
    assert_eq!(values.len(), 1);
    BinaryTerminalClaim::new(
        descriptor,
        descriptor.polynomials()[0],
        row.iter().map(|value| value.to_bytes()).collect(),
        column.iter().map(|value| value.to_bytes()).collect(),
        values[0].to_bytes(),
    )
    .unwrap()
}

#[test]
fn binary_group_opens_once_inside_the_callers_transcript() {
    init_rayon_pool();
    run_on_large_stack(|| run_fixture(false));
}

#[test]
fn binary_and_ordinary_groups_share_one_opening() {
    init_rayon_pool();
    run_on_large_stack(|| run_fixture(true));
}

fn run_fixture(mixed: bool) {
    let catalog = TrustedScheduleCatalog::<BinaryTestSuffix>::from_artifact_bytes(
        if akita_params::DEV_PROTOCOL {
            include_bytes!("../../../artifacts/schedules-labinius-dev/labinius_fp128_digits_b4.aks")
        } else {
            include_bytes!("../../../artifacts/schedules-labinius/labinius_fp128_digits_b4.aks")
        },
        ChunkedWitnessCfg::default_non_chunked(),
    )
    .unwrap();
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
    let index = BinaryIndexMap::new(128, 8, 8, 2).unwrap();
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
    let source = backend
        .import_binary_source(manifest, carrier, vec![words.clone()])
        .unwrap();
    let committed = backend
        .commit(
            scheme.schedules(),
            &source,
            GroupContext::explicit(&carrier),
        )
        .unwrap();
    let binary_handle = committed.private_handle.clone();
    let original_commitment = committed.committed_group;
    let mut commitment_bytes = Vec::new();
    original_commitment
        .serialize_compressed(&mut commitment_bytes)
        .unwrap();
    let commitment =
        CommittedGroup::<F>::deserialize_compressed(commitment_bytes.as_slice(), &()).unwrap();
    assert_eq!(commitment, original_commitment);
    // A binary terminal cannot be attached to an ordinary group kind, even if
    // its payload and carrier metadata came from the same binary commitment.
    let wrong_kind = CommittedGroup::new(carrier, commitment.commitment().clone());
    assert!(PolynomialGroupClaims::<F, _>::new_binary(wrong_kind, claim.clone()).is_err());
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
    let statement = |claim: &BinaryTerminalClaim, point: &[F], value: F| {
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
        "{} ordinary grouped proof: {} bytes",
        if mixed {
            "mixed binary/ordinary"
        } else {
            "binary-only"
        },
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

    let rejects = |bytes: &[u8], claim: &BinaryTerminalClaim, point: &[F], value: F| {
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
    let mut row = claim.rho_row().to_vec();
    row[0][0] ^= 1;
    let mut column = claim.rho_col().to_vec();
    column[0][0] ^= 1;
    for (row, column, value) in [
        (claim.rho_row().to_vec(), claim.rho_col().to_vec(), value),
        (row, claim.rho_col().to_vec(), *claim.value()),
        (claim.rho_row().to_vec(), column, *claim.value()),
    ] {
        let altered =
            BinaryTerminalClaim::new(&descriptor, *claim.polynomial(), row, column, value).unwrap();
        rejects(&proof, &altered, &ordinary_point, ordinary_value);
    }
    // U's D payload is the first private message in the restricted root order.
    let mut altered_d = proof.clone();
    altered_d[0] ^= 1;
    rejects(&altered_d, &claim, &ordinary_point, ordinary_value);
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
            let attacker = FaultBackend::new(
                backend.clone(),
                setup.expanded.descriptor().setup_seed.seed,
                words.clone(),
                fault,
            );
            let claims =
                OpeningClaims::from_groups(vec![PolynomialGroupClaims::<F, _>::new_binary(
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
                akita_prover::FixedFoldRoute::new(vec![
                    root;
                    resolved.schedule().num_fold_levels()
                ]);
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
        }
    }
}

#[test]
fn binary_root_uses_the_existing_caller_channel_without_a_nested_driver() {
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
