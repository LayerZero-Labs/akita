#![allow(missing_docs)]
#![cfg(feature = "logging-transcript")]

//! Complete-fold F1 wire-preservation oracle epoch, captured from the literal #311 head
//! `bc959ef34572aee143ba0114094b0b4212b4e111`.

mod common;

use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_serialization::AkitaSerialize;
use akita_transcript::{AkitaTranscript, LoggingTranscript};
use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout};
use common::*;

type Scheme = AkitaCommitmentScheme<OneHotCfg>;

struct Stage1LevelEpoch {
    basis: usize,
    payload_len: usize,
    payload_digest: &'static str,
}

struct FoldOracleEpoch {
    name: &'static str,
    num_vars: usize,
    witness_seed: u64,
    transcript_domain: &'static [u8],
    proof_len: usize,
    proof_digest: &'static str,
    event_count: usize,
    event_digest: &'static str,
    terminal_len: usize,
    terminal_digest: &'static str,
    stage1_levels: &'static [Stage1LevelEpoch],
}

const PR311_FOLD_ORACLE_EPOCH: &[FoldOracleEpoch] = &[
    FoldOracleEpoch {
        name: "direct-to-terminal",
        num_vars: 12,
        witness_seed: 0xf1_311_001,
        transcript_domain: b"akita/f1/direct-to-terminal",
        proof_len: 57_250,
        proof_digest: "69a998cf92c7af0479803bd23da368a0",
        event_count: 164,
        event_digest: "e7082502c6830cdcb549121978752b9c",
        terminal_len: 54_286,
        terminal_digest: "2fffaaaefe3aa58f9e5a972bfd29750e",
        stage1_levels: &[Stage1LevelEpoch {
            basis: 8,
            payload_len: 1_104,
            payload_digest: "7e868b3d1026fc2ef97f68fa7782bd6a",
        }],
    },
    FoldOracleEpoch {
        name: "recursive-nonterminal",
        num_vars: 20,
        witness_seed: 0xf1_311_002,
        transcript_domain: b"akita/f1/recursive-nonterminal",
        proof_len: 74_246,
        proof_digest: "1ad8feff3b79b8a8cbe5fb2ea458e6f8",
        event_count: 677,
        event_digest: "f6ce402e34096ea9751b09637bcb5835",
        terminal_len: 57_722,
        terminal_digest: "8b2dced0ca215f7588e7c88261a3b76f",
        stage1_levels: &[
            Stage1LevelEpoch {
                basis: 64,
                payload_len: 3_056,
                payload_digest: "41c4852f6f4405c48578ed00b7c71745",
            },
            Stage1LevelEpoch {
                basis: 64,
                payload_len: 2_896,
                payload_digest: "99cde8d12d9d716a761430e90b2c73cb",
            },
            Stage1LevelEpoch {
                basis: 64,
                payload_len: 2_896,
                payload_digest: "999fa85658303c9243b8b0a12e4297bd",
            },
        ],
    },
];

fn assert_fold_epoch(expected: &FoldOracleEpoch) {
    let layout = OneHotCfg::get_params_for_batched_commitment(
        &akita_types::OpeningClaimsLayout::new(expected.num_vars, 1)
            .expect("singleton opening batch"),
    )
    .expect("layout");
    let poly = make_onehot_poly(&layout, expected.witness_seed);
    let point = random_point(expected.num_vars, expected.witness_seed.wrapping_add(1));
    let opening = opening_from_poly::<ONEHOT_D, _>(&poly, &point, &layout);

    let setup = Scheme::setup_prover(expected.num_vars, 1).expect("setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepare setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = Scheme::setup_verifier(&setup).expect("verifier setup");
    let (commitment, hint) =
        Scheme::commit(&setup, std::slice::from_ref(&poly), &stack).expect("commit");

    let mut prover_transcript =
        LoggingTranscript::wrap(AkitaTranscript::<F>::new(expected.transcript_domain));
    let proof = Scheme::batched_prove(
        &setup,
        prove_input(&point, &[&poly], &commitment, hint),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("prove");

    let mut verifier_transcript =
        LoggingTranscript::wrap(AkitaTranscript::<F>::new(expected.transcript_domain));
    verifier_transcript.expect_wire_label(akita_transcript::labels::ABSORB_TERMINAL_E_HAT);
    verifier_transcript.expect_wire_label(akita_transcript::labels::ABSORB_TERMINAL_W_REMAINDER);
    Scheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verify_input(&point, &[opening], &commitment),
        BasisMode::Lagrange,
    )
    .expect("verify");

    let prover_events = public_transcript_events(prover_transcript.events());
    let verifier_events = public_transcript_events(verifier_transcript.events());
    assert_eq!(
        prover_events, verifier_events,
        "{} transcript replay",
        expected.name
    );

    let schedule = OneHotCfg::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::new(expected.num_vars, 1),
    ))
    .expect("generated schedule");
    assert_eq!(
        schedule.folds.len(),
        expected.stage1_levels.len() + 1,
        "{} schedule must end in exactly one terminal fold",
        expected.name
    );
    assert_eq!(
        proof.nonterminal_folds().count(),
        expected.stage1_levels.len(),
        "{} non-terminal level count",
        expected.name
    );
    for ((level, scheduled), level_expected) in proof
        .nonterminal_folds()
        .zip(schedule.folds.iter())
        .zip(expected.stage1_levels)
    {
        let bytes = serialize_stage1_payload(level.stage1());
        assert_eq!(
            1usize << scheduled.params.log_basis,
            level_expected.basis,
            "{} scheduled range basis",
            expected.name
        );
        assert_eq!(
            bytes.len(),
            level_expected.payload_len,
            "{} Stage 1 payload length",
            expected.name
        );
        assert_eq!(
            f1_oracle_digest::<F>(&bytes),
            level_expected.payload_digest,
            "{} Stage 1 payload changed",
            expected.name
        );
    }

    let mut proof_bytes = Vec::new();
    proof
        .serialize_compressed(&mut proof_bytes)
        .expect("serialize complete proof");
    let mut terminal_bytes = Vec::new();
    proof
        .terminal
        .serialize_compressed(&mut terminal_bytes)
        .expect("serialize terminal proof");
    let event_bytes = serialize_transcript_events(&prover_events);

    assert_eq!(
        proof_bytes.len(),
        expected.proof_len,
        "{} proof",
        expected.name
    );
    assert_eq!(
        f1_oracle_digest::<F>(&proof_bytes),
        expected.proof_digest,
        "{} complete proof changed",
        expected.name
    );
    assert_eq!(
        prover_events.len(),
        expected.event_count,
        "{} transcript event count",
        expected.name
    );
    assert_eq!(
        f1_oracle_digest::<F>(&event_bytes),
        expected.event_digest,
        "{} transcript events changed",
        expected.name
    );
    assert_eq!(
        terminal_bytes.len(),
        expected.terminal_len,
        "{} terminal payload length",
        expected.name
    );
    assert_eq!(
        f1_oracle_digest::<F>(&terminal_bytes),
        expected.terminal_digest,
        "{} terminal payload changed",
        expected.name
    );
}

#[test]
fn folds_match_direct_terminal_and_recursive_nonterminal_f1_epochs() {
    init_rayon_pool();
    run_on_large_stack(|| {
        for expected in PR311_FOLD_ORACLE_EPOCH {
            assert_fold_epoch(expected);
        }
    });
}
