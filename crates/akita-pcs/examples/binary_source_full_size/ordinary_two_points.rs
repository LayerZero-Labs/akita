//! Both E64 readings of each source share one public opening and suffix.
#![cfg(feature = "binary-source-measurements")]

#[path = "../../tests/common/mod.rs"]
mod common;
#[path = "ordinary_two_points_transcript.rs"]
mod transcript;

use super::{claims, config::P64 as OrdinaryPoints, native_catalog};
use crate::p64::fault_backend;
use akita_cpu_backend::{CpuBackend, DensePoly, GroupContext};
use akita_error::AkitaError;
use akita_params::binary_setup::BinaryFieldProfile;
use akita_params::BasisMode;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::SelectedProverOpeningData;
use akita_transcript::{new_prover_channel, new_verifier_channel};
use akita_types::binary_source::{
    BinaryCommittedGroupManifest, BinaryTypedPrefix, PrimeClaimValue, PrimeGroupClaim,
    PrimePlaneClaim, PrimePlaneManifest,
};
use akita_types::{GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use akita_verifier::SetupPolicy;
use fault_backend::{Fault, FaultBackend};
use jolt_field::{CanonicalEncoding, Ext2, ExtField, One, Prime64Offset59, Ring};
use jolt_transcript::BytesMsg;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

type F = Prime64Offset59;
type E = Ext2<F>;
const PROTOCOL: &[u8] = b"ordinary-two-point-public-opening";

/// The measurement fixtures' size for the binary source and the ordinary group.
/// The selected planner finds no ordinary suffix for the native root transition
/// at log 3 and no multi-group root schedule for an 18-variable ordinary group.
const LOG_CARRIERS: usize = 22;

/// Two E64 readings of the single binary owner. The first eight axes select
/// the bit within a word and later axes fold the word index, low bit first.
fn binary_claims(words: &[u128]) -> Vec<PrimeGroupClaim> {
    (0..2)
        .map(|copy| {
            let point = (0..LOG_CARRIERS as u64 + 8)
                .map(|axis| [19 + axis * 7, 3 + copy * 11 + axis])
                .collect::<Vec<[u64; 2]>>();
            let extension = point
                .iter()
                .map(|x| E::from_base_fn(|i| F::from_u64(x[i])))
                .collect::<Vec<_>>();
            let value = claims::prime_value(words, &extension).unwrap();
            PrimeGroupClaim {
                point: point
                    .into_iter()
                    .map(|p| p.map(u128::from).into())
                    .collect(),
                evaluations: vec![PrimePlaneClaim {
                    polynomial: 0,
                    value: PrimeClaimValue::Extension(std::array::from_fn(|i| {
                        value.base_coefficient(i).to_u128_checked().unwrap()
                    })),
                }],
            }
        })
        .collect()
}

#[test]
fn binary_and_ordinary_two_point_readings_share_one_public_opening() {
    common::init_rayon_pool();
    common::run_on_large_stack(run);
}

fn run() {
    let words = claims::words(1 << LOG_CARRIERS);
    let prime_claims = binary_claims(&words);
    let directory =
        std::env::temp_dir().join(format!("akita-ordinary-two-points-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let (catalog, key) = native_catalog::plan::<OrdinaryPoints>(
        LOG_CARRIERS,
        "prime-extension-two-plus-ordinary",
        &directory,
    )
    .unwrap();
    std::fs::remove_dir_all(directory).unwrap();
    let scheme = AkitaCommitmentScheme::<OrdinaryPoints>::new(catalog);
    let resolved = scheme.schedules().resolve_key(&key).unwrap();
    let schedule = resolved.schedule();
    let profiles = resolved.profiles();
    let carrier = profiles.precommitteds[0];
    let ordinary_profile = profiles.final_group;
    let mut capacity = akita_params::SetupMatrixCapacity::minimum();
    for row in scheme.schedules().catalog().rows() {
        capacity.num_field_elements = capacity.num_field_elements.max(
            akita_params::setup_matrix_capacity_for_schedule(row.schedule())
                .unwrap()
                .num_field_elements,
        );
    }
    let mut setup = akita_cpu_backend::AkitaProverSetup::generate_with_capacity(
        carrier
            .group
            .num_vars()
            .max(ordinary_profile.group.num_vars()),
        4,
        capacity,
    )
    .unwrap();
    let backend = Arc::new(CpuBackend::new(setup.expanded.clone()).unwrap());
    let manifest = BinaryCommittedGroupManifest::with_field_profile(
        claims::descriptor::<OrdinaryPoints>(LOG_CARRIERS).unwrap(),
        0,
        &carrier,
        BinaryFieldProfile::P64,
    )
    .unwrap();
    let binary_source = backend
        .import_binary_source(manifest.clone(), carrier, vec![words])
        .unwrap();
    let binary = backend
        .commit(
            scheme.schedules(),
            &binary_source,
            GroupContext::explicit(&carrier),
        )
        .unwrap();
    let n = ordinary_profile.group.num_vars();
    // The independent MLE X_0 gives an exact dense-source reading oracle.
    let table = (0..1usize << n)
        .map(|i| F::from_u64((i & 1) as u64))
        .collect::<Vec<_>>();
    let ordinary_source = backend
        .import_source(vec![DensePoly::from_field_evals(n, table).unwrap()])
        .unwrap();
    let ordinary = backend
        .commit(
            scheme.schedules(),
            &ordinary_source,
            GroupContext::explicit(&ordinary_profile),
        )
        .unwrap();
    let points: [Vec<E>; 2] = std::array::from_fn(|copy| {
        (0..n)
            .map(|axis| {
                E::from_base_fn(|limb| F::from_u64((7 + axis * 3 + copy * 11 + limb * 19) as u64))
            })
            .collect()
    });
    let values: [Vec<E>; 2] = std::array::from_fn(|copy| points[copy][..1].to_vec());
    let claims = PolynomialGroupClaims::new_two_points(
        points.clone(),
        values.clone(),
        ordinary.committed_group.clone(),
    )
    .unwrap();
    assert_eq!(claims.num_evaluations(), 1);
    assert_eq!(claims.num_points(), 2);
    assert_eq!(claims.point_evaluations().count(), 2);
    assert!(OpeningClaims::from_groups(vec![claims.clone()]).is_err());
    for (bad_points, bad_values) in [
        ([points[0].clone(), points[0].clone()], values.clone()),
        (points.clone(), [values[0].clone(), vec![values[1][0]; 2]]),
        (points.clone(), [Vec::new(), values[1].clone()]),
        (points.clone(), [values[0].clone(), Vec::new()]),
        ([points[0][1..].to_vec(), points[1].clone()], values.clone()),
    ] {
        assert!(PolynomialGroupClaims::new_two_points(
            bad_points,
            bad_values,
            ordinary.committed_group.clone(),
        )
        .is_err());
    }
    let binary_claim = PolynomialGroupClaims::<E, _>::new_binary_planes(
        binary.committed_group.clone(),
        None,
        prime_claims.clone(),
    )
    .unwrap();
    let groups = OpeningClaims::from_groups(vec![binary_claim.clone(), claims.clone()]).unwrap();
    let data = SelectedProverOpeningData::from_committed_claims::<OrdinaryPoints>(
        groups,
        vec![
            binary.private_handle.clone(),
            ordinary.private_handle.clone(),
        ],
        scheme.schedules(),
    )
    .unwrap();
    let first = schedule.recursive_folds[0]
        .params
        .setup_prefix()
        .unwrap()
        .slot_id()
        .unwrap();
    let later = schedule
        .recursive_folds
        .iter()
        .skip(1)
        .filter_map(|fold| {
            fold.params
                .setup_prefix()
                .and_then(|prefix| prefix.slot_id())
        })
        .collect::<Vec<_>>();
    let plane_count = PrimePlaneManifest::with_field_profile(
        manifest.descriptor(),
        prime_claims.clone(),
        BinaryFieldProfile::P64,
    )
    .unwrap()
    .planes()
    .len();
    let typed = BinaryTypedPrefix::new(
        &manifest,
        false,
        plane_count,
        schedule.recursive_folds[0].params.d_a(),
    )
    .unwrap();
    let prefixes = backend
        .prepare_binary_typed_prefixes(&manifest, &typed, &first, &later)
        .unwrap();
    let ordinary_ids = later
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    setup.prefix_slots = backend.export_setup_prefixes(&ordinary_ids).unwrap();
    let verifier_setup = scheme
        .setup_verifier_for_schedule(&setup, schedule, &profiles.opening_layout().unwrap())
        .unwrap();
    let verifier_setup = backend
        .prepare_binary_typed_verifier_setup(
            &verifier_setup,
            &manifest,
            &typed,
            &first,
            &prefixes.get(&first).unwrap().public,
        )
        .unwrap();
    let mut prover = new_prover_channel(PROTOCOL);
    prover.public_message(&BytesMsg(b"application-prefix"));
    let mut registry = akita_prover::BackendRegistry::<OrdinaryPoints>::new().unwrap();
    let root = registry.register(backend.as_ref(), &prefixes).unwrap();
    let mut route = akita_prover::FixedFoldRoute::new(vec![root; schedule.num_fold_levels()]);
    akita_prover::batched_prove_embedded::<OrdinaryPoints, _>(
        setup.expanded.descriptor(),
        scheme.schedules(),
        &registry,
        data,
        &mut prover,
        BasisMode::Lagrange,
        &mut route,
    )
    .unwrap();
    let proof = prover.narg_string().to_vec();
    prover.public_message(&BytesMsg(b"application-suffix"));
    let continuation: [u8; 32] = prover.verifier_message();
    let verifier = scheme.verifier(verifier_setup).unwrap();
    let verify =
        |proof: &[u8],
         continuation: [u8; 32],
         binary_claim: &PolynomialGroupClaims<'_, E, akita_types::CommittedGroup<F>>,
         claim: &PolynomialGroupClaims<'_, E, akita_types::CommittedGroup<F>>| {
            let groups = OpeningClaims::from_groups(vec![
                binary_claim.map_commitment(|_| &binary.committed_group),
                claim.map_commitment(|_| &ordinary.committed_group),
            ])
            .unwrap();
            let statement = GroupBatchStatement::new(resolved.selection(), groups).unwrap();
            let mut replay = new_verifier_channel(PROTOCOL, proof);
            replay.public_message(&BytesMsg(b"application-prefix"));
            let pending = verifier.batched_verify_embedded(
                &mut replay,
                statement,
                BasisMode::Lagrange,
                SetupPolicy::Check(None),
            )?;
            replay.public_message(&BytesMsg(b"application-suffix"));
            assert_eq!(replay.verifier_message::<[u8; 32]>(), continuation);
            pending.check_eof(replay)
        };
    verify(&proof, continuation, &binary_claim, &claims).unwrap();
    // Keep the honest proof fixed: these exercise statement binding.
    let rejects = |claim| {
        let result = catch_unwind(AssertUnwindSafe(|| {
            verify(&proof, continuation, &binary_claim, &claim)
        }));
        assert!(result.is_ok(), "ordinary mutation panicked");
        assert!(result.unwrap().is_err(), "ordinary mutation verified");
    };
    for copy in 0..2 {
        let mut altered = values.clone();
        altered[copy][0] += E::one();
        rejects(
            PolynomialGroupClaims::new_two_points(
                points.clone(),
                altered,
                ordinary.committed_group.clone(),
            )
            .unwrap(),
        );
    }
    rejects(
        PolynomialGroupClaims::new_two_points(
            [points[1].clone(), points[0].clone()],
            values.clone(),
            ordinary.committed_group.clone(),
        )
        .unwrap(),
    );
    // Even swapping the matching values changes the bound record order.
    rejects(
        PolynomialGroupClaims::new_two_points(
            [points[1].clone(), points[0].clone()],
            [values[1].clone(), values[0].clone()],
            ordinary.committed_group.clone(),
        )
        .unwrap(),
    );
    rejects(
        PolynomialGroupClaims::new(
            points[0].clone(),
            values[0].clone(),
            ordinary.committed_group.clone(),
        )
        .unwrap(),
    );
    // Complete forged proofs from the adversarial backend: one committed digit
    // in each binary coordinate plane, or a false prime/ordinary value with an
    // honest witness.
    // Each must pass every earlier check and fail the final root relation.
    let planes = (0..2).flat_map(|point| {
        (0..2).map(move |coordinate| Fault::PrimePlane {
            owner: 0,
            point,
            coordinate,
        })
    });
    let values_faults = (0..2)
        .flat_map(|point| (0..2).map(move |coordinate| Fault::PrimeValue { point, coordinate }));
    for fault in planes.chain(values_faults).chain([Fault::OrdinaryValue]) {
        let mut altered_prime = prime_claims.clone();
        if let Fault::PrimeValue { point, coordinate } = fault {
            let PrimeClaimValue::Extension(value) = &mut altered_prime[point].evaluations[0].value
            else {
                unreachable!();
            };
            let value = &mut value[coordinate];
            *value = (F::from_u128(*value) + F::one()).to_u128_checked().unwrap();
        }
        let binary_claim = PolynomialGroupClaims::<E, _>::new_binary_planes(
            binary.committed_group.clone(),
            None,
            altered_prime,
        )
        .unwrap();
        let mut altered = values.clone();
        if matches!(fault, Fault::OrdinaryValue) {
            // The source and its committed witness stay honest.
            altered[1][0] += E::one();
        }
        let ordinary_claim = PolynomialGroupClaims::new_two_points(
            points.clone(),
            altered,
            ordinary.committed_group.clone(),
        )
        .unwrap();
        let groups = vec![binary_claim.clone(), ordinary_claim.clone()];
        let data = SelectedProverOpeningData::from_committed_claims::<OrdinaryPoints>(
            OpeningClaims::from_groups(groups).unwrap(),
            vec![
                binary.private_handle.clone(),
                ordinary.private_handle.clone(),
            ],
            scheme.schedules(),
        )
        .unwrap();
        let attacker = FaultBackend::new(
            backend.clone(),
            setup.expanded.descriptor().setup_seed.seed,
            Vec::new(),
            fault,
        );
        let mut prover = new_prover_channel(PROTOCOL);
        prover.public_message(&BytesMsg(b"application-prefix"));
        attacker
            .prove_prime_fault(&setup, scheme.schedules(), &prefixes, data, &mut prover)
            .unwrap();
        let forged = prover.narg_string().to_vec();
        prover.public_message(&BytesMsg(b"application-suffix"));
        let continuation: [u8; 32] = prover.verifier_message();
        let (result, rejections) = crate::p64::root_relation_rejections(|| {
            verify(&forged, continuation, &binary_claim, &ordinary_claim)
        });
        assert!(
            matches!(result, Err(AkitaError::InvalidProof)),
            "forged relation {fault:?}"
        );
        assert_eq!(
            rejections, 1,
            "forged relation {fault:?} must reach the final root equation"
        );
    }
}
