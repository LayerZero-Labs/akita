#![allow(missing_docs)]
#![cfg(feature = "logging-transcript")]

//! Complete-fold wire fixture for descriptor v1: typed fold
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
        proof_digest: "bdac1c6cbe6116ef870b26c162e51014",
        event_count: 139,
        event_digest: "35f3cecd6d723e8e1ed030099d525cd1",
        terminal_len: 46_092,
        terminal_digest: "f713bb0f417be18c018152a6a30d670c",
        digit_range_levels: &[DigitRangeLevelEpoch {
            basis: 8,
            payload_len: 1_104,
            payload_digest: "b5979b40defa71ef43ffa4ce89632a8f",
        }],
    },
    FoldProtocolEpoch {
        name: "recursive-nonterminal",
        num_vars: 20,
        witness_seed: 0xd1_613_002,
        transcript_domain: b"akita/protocol-epoch/recursive-nonterminal",
        proof_len: 78_443,
        proof_digest: "f334ffc92cba4b260c92848a868d964e",
        event_count: 929,
        event_digest: "74731249b2484544e06b4a1007dc3a39",
        terminal_len: 52_391,
        terminal_digest: "310445db0b54ee9b5c012479d08e311f",
        digit_range_levels: &[
            DigitRangeLevelEpoch {
                basis: 8,
                payload_len: 1_232,
                payload_digest: "3d9e2acbccbec1223b17c75b34cf4395",
            },
            DigitRangeLevelEpoch {
                basis: 32,
                payload_len: 2_384,
                payload_digest: "f369104ac997101470181e401c4be940",
            },
            DigitRangeLevelEpoch {
                basis: 64,
                payload_len: 3_056,
                payload_digest: "76c7384487bfbeddcccb239b48db2739",
            },
            DigitRangeLevelEpoch {
                basis: 64,
                payload_len: 2_896,
                payload_digest: "d61b3f47eb64ce68b2c3a21fdbb2b990",
            },
            DigitRangeLevelEpoch {
                basis: 64,
                payload_len: 2_896,
                payload_digest: "889873517078f66a4543dce1770b7c6e",
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
        prove_input::<OneHotCfg, _>(&point, &[&poly], &commitment, hint),
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
        verify_input::<OneHotCfg>(&point, &[opening], &commitment),
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
    let point_run_start = first_label_index(
        &prover_events,
        akita_transcript::labels::ABSORB_EVALUATION_CLAIMS,
    )
    .expect("root opening-point transcript run");
    let point_run_end = prover_events[point_run_start..]
        .iter()
        .position(|event| {
            event_label(event) != Some(akita_transcript::labels::ABSORB_EVALUATION_CLAIMS)
        })
        .map_or(prover_events.len(), |offset| point_run_start + offset);
    assert_eq!(
        point_run_end - point_run_start,
        point.len(),
        "{} must absorb each root point coordinate exactly once",
        expected.name
    );
    assert_eq!(
        event_label(
            prover_events
                .get(point_run_end)
                .expect("opening value after root point")
        ),
        Some(akita_transcript::labels::ABSORB_EVAL_OPENINGS_FIELD),
        "{} root point/opening order",
        expected.name
    );

    let schedule = OneHotCfg::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::new(expected.num_vars, 1),
        OneHotCfg::group_source(),
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
