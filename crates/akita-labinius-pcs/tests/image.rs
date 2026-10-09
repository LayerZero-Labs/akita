#![cfg(feature = "labinius")]

mod common;

use akita_algebra::{
    binary::{BinaryField128, BinaryField192},
    MinusTrinomial,
};
use akita_config::policy_of;
use akita_cpu_backend::{CpuBackend, DensePoly, GroupContext};
use akita_labinius_pcs::{ImageConfig, F};
use akita_labinius_prover::{commit_binary_clear, lowered::flatten_image};
use akita_labinius_verifier::{channel::ClearChannel, lowered::LoweredRootLayout};
use akita_params::{
    sis::labinius::LabiniusRootShape, BasisMode, PolynomialGroupLayout, ScheduleLookupKey,
};
use akita_prover::SelectedProverOpeningData;
use akita_types::{GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use common::{fixture, Case, Event, Host, Recording, BASES, INNER_DOMAIN, PREFIX, PROFILE};
use jolt_field::Zero;

fn complete<H: Host>(fold: u32) {
    let case = Case::<H>::new(fold);
    let expected =
        commit_binary_clear::<H, F, 648, MinusTrinomial>(case.root.setup(), &case.source).unwrap();
    assert_eq!(case.output.image.images, expected.images);
    for base in BASES {
        let layout = LoweredRootLayout::new(case.root.setup(), case.root.shape(), base).unwrap();
        assert_eq!(flatten_image(&layout, &expected).unwrap(), case.table);
        let mut addressed = vec![F::zero(); layout.image_len()];
        for (entry, ring) in expected.images.iter().enumerate() {
            for (coefficient, &value) in ring.coefficients().iter().enumerate() {
                addressed[layout.image_address(entry, coefficient).unwrap()] = value;
            }
            for coefficient in 648..layout.padded_coefficients() {
                assert_eq!(
                    case.table[layout.image_address(entry, coefficient).unwrap()],
                    F::zero()
                );
            }
        }
        assert_eq!(addressed, case.table);
    }
    assert_eq!(
        akita_algebra::poly::multilinear_eval(&case.table, &case.point).unwrap(),
        case.value
    );
    let (proof, seed, after) = case.prove(PREFIX);
    assert_eq!(case.verify(&proof, PREFIX).unwrap(), after);
    assert!(case.verify(&proof, b"different prefix").is_err());
    plain_api_agreement(&case, &proof, seed);
}

fn plain_api_agreement<H: Host>(case: &Case<H>, framed: &[u8], seed: [u8; 32]) {
    let f = fixture(case.layout.image_log_len());
    let backend = CpuBackend::<F, F>::new(f.prover_setup.expanded.clone()).unwrap();
    let polynomial =
        DensePoly::from_field_evals(case.layout.image_log_len(), case.table.clone()).unwrap();
    let source = backend.import_source(vec![polynomial]).unwrap();
    let output = backend
        .commit(
            f.scheme.schedules(),
            &source,
            GroupContext::scheduler_without_precommitted_groups(),
        )
        .unwrap();
    assert_eq!(case.output.committed_group, output.committed_group);
    let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        &case.point[..],
        vec![case.value],
        output.committed_group.clone(),
    )
    .unwrap()])
    .unwrap();
    let opening = SelectedProverOpeningData::from_committed_claims::<ImageConfig>(
        claims,
        vec![output.private_handle],
        f.scheme.schedules(),
    )
    .unwrap();
    let selection = opening.selection();
    let mut session = INNER_DOMAIN.to_vec();
    session.extend_from_slice(&seed);
    let plain = f
        .scheme
        .batched_prove(
            &f.prover_setup,
            opening,
            &backend,
            &session,
            BasisMode::Lagrange,
        )
        .unwrap();
    let inner_len = u64::from_le_bytes(framed[..8].try_into().unwrap()) as usize;
    assert_eq!(inner_len, framed.len() - 8);
    assert_eq!(plain, framed[8..]);
    let key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(
        case.layout.image_log_len(),
    ));
    let row = f.scheme.schedules().resolve_key(&key).unwrap();
    let bound = akita_schedules::expanded_schedule_proof_bound(
        &key,
        row.schedule(),
        &policy_of::<ImageConfig>(),
    )
    .unwrap();
    assert!(inner_len <= bound, "proof {inner_len}, bound {bound}");
    let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        &case.point[..],
        vec![case.value],
        &case.output.committed_group,
    )
    .unwrap()])
    .unwrap();
    let statement = GroupBatchStatement::new(selection, claims).unwrap();
    f.scheme
        .verifier(f.verifier_setup.clone())
        .unwrap()
        .batched_verify(&plain, &session, statement, BasisMode::Lagrange)
        .unwrap();
}

#[test]
fn both_host_fields_and_both_small_geometries() {
    complete::<BinaryField128>(0);
    complete::<BinaryField192>(0);
    complete::<BinaryField128>(1);
    complete::<BinaryField192>(1);
}

#[test]
fn first_profile_sizes_use_canonical_derivation_without_image_action() {
    let shape = LabiniusRootShape::derive(PROFILE, 22, 8, 128).unwrap();
    assert_eq!(shape.num_cells(), 1 << 22);
    assert_eq!(shape.fold_width(), 256);
    assert_eq!(shape.ring_elements_per_column(), 4096);
    assert_eq!(shape.rank_a(), 1);
    for base in BASES {
        let encoding = shape.derive_encoding(base).unwrap();
        assert_eq!(encoding.image_table_log_len(), 18);
        assert_eq!(encoding.padded_coefficient_len(), 1024);
    }
}

#[test]
fn two_openings_compose_in_order_and_absorb_inner_payload() {
    let case = Case::<BinaryField128>::new(0);
    let state = akita_transcript::new_prover_channel(b"compose/v1", b"").unwrap();
    let mut prover = Recording::new(state);
    prover.public(PREFIX).unwrap();
    case.prover
        .open_on_channel::<BinaryField128, _>(
            &case.root,
            &case.output,
            case.evaluation(),
            &mut prover,
        )
        .unwrap();
    let between = prover.challenge_block().unwrap();
    let first_len = prover.inner.narg_string().len();
    case.prover
        .open_on_channel::<BinaryField128, _>(
            &case.root,
            &case.output,
            case.evaluation(),
            &mut prover,
        )
        .unwrap();
    let after = prover.challenge_block().unwrap();
    let proof = prover.inner.narg_string().to_vec();
    let mut verifier = akita_transcript::new_verifier_channel(b"compose/v1", b"", &proof).unwrap();
    verifier.public(PREFIX).unwrap();
    case.verifier
        .verify_on_channel::<BinaryField128, _>(
            &case.root,
            &case.output.committed_group,
            case.evaluation(),
            &mut verifier,
        )
        .unwrap();
    assert_eq!(verifier.challenge_block().unwrap(), between);
    case.verifier
        .verify_on_channel::<BinaryField128, _>(
            &case.root,
            &case.output.committed_group,
            case.evaluation(),
            &mut verifier,
        )
        .unwrap();
    assert_eq!(verifier.challenge_block().unwrap(), after);
    verifier.check_eof().unwrap();
    let mut misplaced =
        akita_transcript::new_verifier_channel(b"compose/v1", b"", &proof[first_len..]).unwrap();
    misplaced.public(PREFIX).unwrap();
    assert!(case
        .verifier
        .verify_on_channel::<BinaryField128, _>(
            &case.root,
            &case.output.committed_group,
            case.evaluation(),
            &mut misplaced
        )
        .is_err());
    // Replay every parent event in order, omitting only the two inner payloads.
    // Length frames and all earlier challenge draws stay in their original places.
    let mut exact = akita_transcript::new_prover_channel(b"compose/v1", b"").unwrap();
    let mut without_payload = akita_transcript::new_prover_channel(b"compose/v1", b"").unwrap();
    let mut message_index = 0;
    let mut omitted = 0;
    let mut counterfactual_after = None;
    for event in &prover.events {
        match event {
            Event::Public(bytes) => {
                exact.public(bytes).unwrap();
                without_payload.public(bytes).unwrap();
            }
            Event::Message(bytes) => {
                exact.message(&mut bytes.clone()).unwrap();
                if message_index % 2 == 0 {
                    without_payload.message(&mut bytes.clone()).unwrap();
                } else {
                    omitted += 1;
                }
                message_index += 1;
            }
            Event::Challenge(expected) => {
                assert_eq!(exact.challenge_block().unwrap(), *expected);
                counterfactual_after = Some(without_payload.challenge_block().unwrap());
            }
        }
    }
    assert_eq!(omitted, 2);
    assert_ne!(counterfactual_after.unwrap(), after);
}
