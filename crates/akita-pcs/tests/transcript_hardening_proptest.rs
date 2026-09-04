#![allow(missing_docs)]
#![cfg(feature = "logging-transcript")]

mod common;

use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_transcript::{labels, AkitaTranscript, LoggingTranscript};
use akita_types::OpeningClaimsLayout;
use common::*;
use proptest::prelude::*;

fn batch_case(index: usize) -> (usize, usize) {
    // Keep fuzz inputs on exact generated rows so failures exercise transcript
    // replay rather than missing-schedule rejection.
    match index {
        0 => (14, 1),
        1 => (15, 2),
        2 => (17, 4),
        // Keep one recursive row for terminal-window coverage without turning
        // this transcript-semantic test into a large-prover benchmark.
        _ => (16, 1),
    }
}

fn logged_dense_round_trip(shape_index: usize, basis_mode: BasisMode, seed: u64) {
    init_rayon_pool();
    let scheme = load_workspace_scheme::<DenseCfg>().expect("workspace schedule catalog");

    let (num_vars, total_claims) = batch_case(shape_index);
    let opening_batch =
        OpeningClaimsLayout::new(num_vars, total_claims).expect("valid opening batch");
    let layout = scheme
        .schedules()
        .resolve_key(&akita_types::AkitaScheduleLookupKey::single(
            opening_batch
                .root_final_group_layout()
                .expect("batched group layout"),
        ))
        .map(|row| row.schedule().root.params.final_group())
        .expect("batched commit layout");

    let polys: Vec<DensePoly<F>> = (0..total_claims)
        .map(|poly_idx| make_dense_poly(num_vars, seed.wrapping_add(poly_idx as u64)))
        .collect();
    let opening_point = random_point(num_vars, seed.wrapping_add(0x9e37_0000));
    let poly_refs: Vec<&DensePoly<F>> = polys.iter().collect();
    let openings: Vec<F> = poly_refs
        .iter()
        .map(|poly| opening_from_poly_for_layout(*poly, &opening_point, &layout, basis_mode))
        .collect();

    let setup = scheme.setup_prover(num_vars, total_claims).unwrap();
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");
    let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");

    let akita_prover::CommitOutput {
        committed_group: commitment,
        hint,
    } = scheme
        .commit(
            &setup,
            &polys,
            &stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("commit");
    let mut prover_transcript =
        LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"hardening/proptest"));
    let proof = scheme
        .batched_prove(
            &setup,
            prove_input::<DenseCfg, _>(
                &opening_point,
                &poly_refs,
                &commitment,
                hint,
                scheme.schedules(),
            ),
            &stack,
            &mut prover_transcript,
            basis_mode,
        )
        .expect("prove");

    let mut verifier_transcript =
        LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"hardening/proptest"));
    scheme
        .batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input::<DenseCfg>(&opening_point, &openings, &commitment, scheme.schedules()),
            basis_mode,
        )
        .expect("verify");

    prover_transcript.assert_smell_checks();
    verifier_transcript.assert_smell_checks();
    let prover_public = public_transcript_events(prover_transcript.events());
    let verifier_public = public_transcript_events(verifier_transcript.events());
    assert_eq!(prover_public, verifier_public);
    let batching_squeezes = assert_claim_batching_follows_opening_payload(&prover_public);
    if total_claims > 1 {
        assert!(
            batching_squeezes > 0,
            "multi-claim root must exercise public claim batching"
        );
    }
    let terminal_e_hat = assert_terminal_event_order_if_present(&prover_public);
    if shape_index == 3 {
        let terminal_e_hat =
            terminal_e_hat.expect("recursive corpus case must include a terminal fold");
        let tau0 = first_label_index(&prover_public, labels::CHALLENGE_TAU0)
            .expect("recursive corpus case must include non-terminal tau0");
        assert!(
            tau0 < terminal_e_hat,
            "recursive tau0 must occur before the terminal transcript window"
        );
    }
}

#[test]
fn seed_corpus_covers_nv_basis_and_batch_shapes() {
    run_on_large_stack(|| {
        for (shape_index, basis_mode, seed) in [
            (0, BasisMode::Lagrange, 0x1001),
            (1, BasisMode::Lagrange, 0x1002),
            (2, BasisMode::Lagrange, 0x1004),
            (2, BasisMode::Monomial, 0x1005),
            (3, BasisMode::Lagrange, 0x1006),
        ] {
            logged_dense_round_trip(shape_index, basis_mode, seed);
        }
    });
}

proptest! {
    // Full proof construction can make a rare grind-heavy input take hours.
    // Keep CI's semantic corpus reproducible; the fixed seed also makes any
    // future runtime regression locally replayable.
    #![proptest_config(ProptestConfig {
        cases: 4,
        rng_seed: proptest::test_runner::RngSeed::Fixed(1),
        ..ProptestConfig::default()
    })]

    #[test]
    fn event_stream_equality_fuzzes_batch_shapes(shape_index in 0usize..4, seed in any::<u64>()) {
        run_on_large_stack(move || logged_dense_round_trip(shape_index, BasisMode::Lagrange, seed));
    }
}
