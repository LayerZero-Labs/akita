#![allow(missing_docs)]
#![cfg(feature = "logging-transcript")]

//! Complete-fold wire fixture for the in-development descriptor v1: typed fold
//! topology plus the direct terminal response.

mod common;

use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_serialization::AkitaSerialize;
use akita_transcript::{AkitaTranscript, LoggingTranscript};
use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout};
use common::*;

type Scheme = AkitaCommitmentScheme<OneHotCfg>;

struct DigitRangeLevelEpoch {
    basis: usize,
    payload_len: usize,
    payload_digest: &'static str,
}

struct FoldProtocolEpoch {
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
    digit_range_levels: &'static [DigitRangeLevelEpoch],
}

const FOLD_PROTOCOL_EPOCH: &[FoldProtocolEpoch] = &[
    FoldProtocolEpoch {
        name: "direct-to-terminal",
        num_vars: 12,
        witness_seed: 0xd1_613_001,
        transcript_domain: b"akita/protocol-epoch/direct-to-terminal",
        proof_len: 49_056,
        proof_digest: "e0850f401a4a562cc6f737551e909e86",
        event_count: 165,
        event_digest: "2975386a0020d9e508622a17e1965df4",
        terminal_len: 46_092,
        terminal_digest: "8d2b162d8d28cc823b1d3fe6d6075a25",
        digit_range_levels: &[DigitRangeLevelEpoch {
            basis: 8,
            payload_len: 1_104,
            payload_digest: "ab4205d9be75e95cc1b6df9fe005a454",
        }],
    },
    FoldProtocolEpoch {
        name: "recursive-nonterminal",
        num_vars: 20,
        witness_seed: 0xd1_613_002,
        transcript_domain: b"akita/protocol-epoch/recursive-nonterminal",
        proof_len: 73_593,
        proof_digest: "c6f45ff5ac3a7ffe1b1b61544947af5c",
        event_count: 480,
        event_digest: "439bc0da9679e4107475af478541c4b9",
        terminal_len: 62_849,
        terminal_digest: "aae790d57fc562b8feeb5d337e9738de",
        digit_range_levels: &[
            DigitRangeLevelEpoch {
                basis: 64,
                payload_len: 3_056,
                payload_digest: "fff9798bcf551d8d8e11906861949cd2",
            },
            DigitRangeLevelEpoch {
                basis: 64,
                payload_len: 2_896,
                payload_digest: "b59ca00aae8eec837a26fe1614c74579",
            },
        ],
    },
];

fn assert_fold_protocol_epoch(expected: &FoldProtocolEpoch) {
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
        schedule.num_fold_levels(),
        expected.digit_range_levels.len() + 1,
        "{} schedule must end in exactly one terminal fold",
        expected.name
    );
    assert_eq!(
        proof.nonterminal_folds().count(),
        expected.digit_range_levels.len(),
        "{} non-terminal level count",
        expected.name
    );
    let scheduled_nonterminal = std::iter::once(&schedule.root.params.final_group.commitment)
        .chain(
            schedule
                .recursive_folds
                .iter()
                .map(|step| &step.params.witness),
        );
    let mut stage1_digests = Vec::with_capacity(expected.digit_range_levels.len());
    for ((level, scheduled), level_expected) in proof
        .nonterminal_folds()
        .zip(scheduled_nonterminal)
        .zip(expected.digit_range_levels)
    {
        let bytes = serialize_stage1_payload(level.stage1());
        assert_eq!(
            1usize << scheduled.log_basis_open,
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
        stage1_digests.push(protocol_epoch_digest::<F>(&bytes));
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
        prover_events.len(),
        expected.event_count,
        "{} transcript event count",
        expected.name
    );
    assert_eq!(
        terminal_bytes.len(),
        expected.terminal_len,
        "{} terminal payload length",
        expected.name
    );
    let expected_stage1_digests = expected
        .digit_range_levels
        .iter()
        .map(|level| level.payload_digest.to_string())
        .collect::<Vec<_>>();
    assert_eq!(
        (
            stage1_digests,
            protocol_epoch_digest::<F>(&proof_bytes),
            protocol_epoch_digest::<F>(&event_bytes),
            protocol_epoch_digest::<F>(&terminal_bytes),
        ),
        (
            expected_stage1_digests,
            expected.proof_digest.to_string(),
            expected.event_digest.to_string(),
            expected.terminal_digest.to_string(),
        ),
        "{} protocol digests changed",
        expected.name,
    );
}

#[test]
fn folds_match_direct_terminal_and_recursive_nonterminal_protocol_epoch() {
    init_rayon_pool();
    run_on_large_stack(|| {
        for expected in FOLD_PROTOCOL_EPOCH {
            assert_fold_protocol_epoch(expected);
        }
    });
}
