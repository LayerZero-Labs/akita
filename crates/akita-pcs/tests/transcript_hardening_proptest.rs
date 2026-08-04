#![allow(missing_docs)]
#![cfg(feature = "logging-transcript")]

mod common;

use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_transcript::{labels, AkitaTranscript, LoggingTranscript};
use akita_types::OpeningClaimsLayout;
use common::*;
use proptest::prelude::*;

type Scheme = AkitaCommitmentScheme<DenseCfg>;

fn batch_shape(index: usize) -> usize {
    match index {
        0 => 1,
        1 => 2,
        _ => 3,
    }
}

fn logged_dense_round_trip(num_vars: usize, shape_index: usize, basis_mode: BasisMode, seed: u64) {
    init_rayon_pool();

    let total_claims = batch_shape(shape_index);
    let opening_batch =
        OpeningClaimsLayout::new(num_vars, total_claims).expect("valid opening batch");
    let layout =
        DenseCfg::get_params_for_batched_commitment(&opening_batch).expect("batched commit layout");

    let polys: Vec<DensePoly<F>> = (0..total_claims)
        .map(|poly_idx| make_dense_poly(num_vars, seed.wrapping_add(poly_idx as u64)))
        .collect();
    let opening_point = random_point(num_vars, seed.wrapping_add(0x9e37_0000));
    let poly_refs: Vec<&DensePoly<F>> = polys.iter().collect();
    let openings: Vec<F> = poly_refs
        .iter()
        .map(|poly| {
            opening_from_poly_with_basis::<DENSE_D, _>(*poly, &opening_point, &layout, basis_mode)
        })
        .collect();

    let setup = Scheme::setup_prover(num_vars, total_claims).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = Scheme::setup_verifier(&setup).expect("verifier setup");

    let (commitment, hint) =
        Scheme::batched_commit(&setup, &polys, &stack).expect("batched commit");
    let mut prover_transcript =
        LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"hardening/proptest"));
    let proof = Scheme::batched_prove(
        &setup,
        prove_input(&opening_point, &poly_refs, &commitment, hint),
        &stack,
        &mut prover_transcript,
        basis_mode,
    )
    .expect("prove");

    let mut verifier_transcript =
        LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"hardening/proptest"));
    Scheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verify_input(&opening_point, &openings, &commitment),
        basis_mode,
    )
    .expect("verify");

    prover_transcript.assert_smell_checks();
    verifier_transcript.assert_smell_checks();
    let prover_public = public_transcript_events(prover_transcript.events());
    let verifier_public = public_transcript_events(verifier_transcript.events());
    assert_eq!(prover_public, verifier_public);
    let terminal_e_hat = assert_terminal_event_order_if_present(&prover_public);
    if num_vars >= 20 {
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
        for (num_vars, shape_index, basis_mode, seed) in [
            (15, 0, BasisMode::Lagrange, 0x1001),
            (15, 1, BasisMode::Lagrange, 0x1002),
            (20, 0, BasisMode::Lagrange, 0x1003),
            (15, 2, BasisMode::Lagrange, 0x1004),
            (15, 3, BasisMode::Monomial, 0x1005),
        ] {
            logged_dense_round_trip(num_vars, shape_index, basis_mode, seed);
        }
    });
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4))]

    #[test]
    fn event_stream_equality_fuzzes_batch_shapes(shape_index in 0usize..4, seed in any::<u64>()) {
        run_on_large_stack(move || logged_dense_round_trip(15, shape_index, BasisMode::Lagrange, seed));
    }
}
