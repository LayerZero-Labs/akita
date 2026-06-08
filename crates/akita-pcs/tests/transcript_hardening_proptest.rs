#![allow(missing_docs)]
#![cfg(all(feature = "logging-transcript", not(feature = "zk")))]

mod common;

use akita_pcs::AkitaCommitmentScheme;
use akita_prover::CommitmentProver;
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_transcript::{labels, AkitaTranscript, LoggingTranscript};
use akita_types::ClaimIncidenceSummary;
use akita_verifier::CommitmentVerifier;
use common::*;
use proptest::prelude::*;

type Scheme = AkitaCommitmentScheme<DENSE_D, DenseCfg>;

fn batch_shape(index: usize) -> Vec<usize> {
    match index {
        0 => vec![1],
        1 => vec![2],
        2 => vec![1, 2],
        _ => vec![2, 1],
    }
}

fn logged_dense_round_trip(num_vars: usize, shape_index: usize, basis_mode: BasisMode, seed: u64) {
    init_rayon_pool();

    let num_polys_per_point = batch_shape(shape_index);
    let total_claims: usize = num_polys_per_point.iter().sum();
    let incidence = ClaimIncidenceSummary::from_point_polys(num_vars, num_polys_per_point.clone())
        .expect("valid incidence");
    let layout =
        DenseCfg::get_params_for_batched_commitment(&incidence).expect("batched commit layout");

    let polys_per_point: Vec<Vec<DensePoly<F, DENSE_D>>> = num_polys_per_point
        .iter()
        .enumerate()
        .map(|(point_idx, &count)| {
            (0..count)
                .map(|poly_idx| {
                    make_dense_poly(
                        num_vars,
                        seed.wrapping_add((point_idx as u64) << 16)
                            .wrapping_add(poly_idx as u64),
                    )
                })
                .collect()
        })
        .collect();
    let opening_points_owned: Vec<Vec<F>> = (0..num_polys_per_point.len())
        .map(|point_idx| random_point(num_vars, seed.wrapping_add(0x9e37_0000 + point_idx as u64)))
        .collect();
    let openings_per_point: Vec<Vec<F>> = polys_per_point
        .iter()
        .zip(opening_points_owned.iter())
        .map(|(polys, point)| {
            polys
                .iter()
                .map(|poly| opening_from_poly_with_basis(poly, point, &layout, basis_mode))
                .collect()
        })
        .collect();

    let polys_per_point_refs: Vec<&[DensePoly<F, DENSE_D>]> =
        polys_per_point.iter().map(Vec::as_slice).collect();
    let openings_per_point_refs: Vec<&[F]> = openings_per_point.iter().map(Vec::as_slice).collect();
    let opening_points: Vec<&[F]> = opening_points_owned.iter().map(Vec::as_slice).collect();

    let setup = <Scheme as CommitmentProver<F, DENSE_D>>::setup_prover(
        num_vars,
        total_claims,
        num_polys_per_point.len(),
    )
    .unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <Scheme as CommitmentProver<F, DENSE_D>>::setup_verifier(&setup);

    let commit_outputs = <Scheme as CommitmentProver<F, DENSE_D>>::batched_commit(
        &setup,
        &polys_per_point_refs,
        &CpuBackend,
        &prepared,
    )
    .expect("batched commit");
    let (commitments, hints): (Vec<_>, Vec<_>) = commit_outputs.into_iter().unzip();

    let mut prover_transcript =
        LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"hardening/proptest"));
    let proof = <Scheme as CommitmentProver<F, DENSE_D>>::batched_prove(
        &setup,
        &CpuBackend,
        &prepared,
        prove_inputs_from_groups(&opening_points, &polys_per_point_refs, &commitments, hints),
        &mut prover_transcript,
        basis_mode,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("prove");

    let mut verifier_transcript =
        LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"hardening/proptest"));
    <Scheme as CommitmentVerifier<F, DENSE_D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verify_inputs_from_groups(&opening_points, &openings_per_point_refs, &commitments),
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
            (8, 0, BasisMode::Lagrange, 0x1001),
            (10, 1, BasisMode::Lagrange, 0x1002),
            (20, 0, BasisMode::Lagrange, 0x1003),
            (10, 2, BasisMode::Lagrange, 0x1004),
            (10, 3, BasisMode::Monomial, 0x1005),
        ] {
            logged_dense_round_trip(num_vars, shape_index, basis_mode, seed);
        }
    });
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4))]

    #[test]
    fn event_stream_equality_fuzzes_batch_shapes(shape_index in 0usize..4, seed in any::<u64>()) {
        run_on_large_stack(move || logged_dense_round_trip(10, shape_index, BasisMode::Lagrange, seed));
    }
}
