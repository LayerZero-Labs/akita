use super::*;
use akita_transcript::{
    grinding_predicate_accepts, preview_grinding_predicate, search_grinding_nonce, AkitaTranscript,
    Transcript,
};
use jolt_field::Prime128Offset275;
use std::num::NonZeroU8;

fn stream_test_plan() -> GrindingPlan {
    GrindingPlan::new(
        vec![
            GrindingRun::proof_of_work(GrindingSite::RingSwitchAlpha { level: 0 }, 2, 128).unwrap(),
            GrindingRun::fold_response(0),
            GrindingRun::proof_of_work(GrindingSite::Tau0Point { level: 0 }, 4, 128).unwrap(),
            GrindingRun::fold_response(1),
        ],
        128,
    )
    .unwrap()
}

fn test_fold_response_nonce(site: GrindingSite) -> u32 {
    site.canonical_bytes()
        .into_iter()
        .fold(0u32, |value, byte| value.wrapping_add(u32::from(byte)))
        % FOLD_RESPONSE_ATTEMPTS
}

fn test_nonce(site: GrindingSite, width: u8) -> u32 {
    let mask = (1u32 << width) - 1;
    site.canonical_bytes()
        .into_iter()
        .fold(u32::default(), |value, byte| {
            value.rotate_left(5) ^ u32::from(byte)
        })
        & mask
}

#[test]
fn one_proof_of_work_entry_searches_packs_and_replays() {
    let site = GrindingSite::Tau0Point { level: 0 };
    let run = GrindingRun::proof_of_work(site, 1, 127).unwrap();
    let plan = GrindingPlan::new(vec![run], 127).unwrap();
    assert_eq!(run.grind_bits(), 1);
    assert_eq!(run.nonce_bits(), 8);

    let mut prover =
        AkitaTranscript::<Prime128Offset275>::prover(b"grinding-wire-test", b"instance");
    let nonce = search_grinding_nonce(&prover, run.grind_bits(), run.nonce_bits()).unwrap();
    let preview =
        preview_grinding_predicate(&prover, run.grind_bits(), run.nonce_bits(), nonce).unwrap();

    let mut writer = TranscriptNonceWriter::new(&plan).unwrap();
    writer.write(site, nonce).unwrap();
    let stream = writer.finish().unwrap();
    let wire = stream.as_bytes().to_vec();
    let decoded = TranscriptNonceStream::from_bytes(wire, plan.total_nonce_bits()).unwrap();
    let mut reader = decoded.reader(&plan).unwrap();
    let decoded_nonce = reader.read(site).unwrap();
    reader.finish().unwrap();
    assert_eq!(decoded_nonce, nonce);

    let prover_predicate = Transcript::grinding_predicate(
        &mut prover,
        akita_transcript::labels::CHALLENGE_TAU0,
        run.grind_bits(),
        run.nonce_bits(),
        nonce,
    )
    .unwrap();
    let mut verifier =
        AkitaTranscript::<Prime128Offset275>::verifier(b"grinding-wire-test", b"instance");
    let verifier_predicate = Transcript::grinding_predicate(
        &mut verifier,
        akita_transcript::labels::CHALLENGE_TAU0,
        run.grind_bits(),
        run.nonce_bits(),
        decoded_nonce,
    )
    .unwrap();
    assert_eq!(preview, prover_predicate);
    assert_eq!(prover_predicate, verifier_predicate);
    assert!(grinding_predicate_accepts(
        &verifier_predicate,
        NonZeroU8::new(run.grind_bits()).unwrap()
    ));
}

#[test]
fn verifier_replay_accepts_exactly_the_public_predicate() {
    let site = GrindingSite::Tau0Point { level: 0 };
    let run = GrindingRun::proof_of_work(site, 1, 127).unwrap();
    let plan = GrindingPlan::new(vec![run], 127).unwrap();
    let transcript =
        AkitaTranscript::<Prime128Offset275>::prover(b"grinding-predicate-test", b"instance");
    let grind_bits = NonZeroU8::new(run.grind_bits()).unwrap();
    let candidate_limit = 1u64 << run.nonce_bits();
    let mut accepted = None;
    let mut rejected = None;
    for candidate in 0..candidate_limit {
        let candidate = u32::try_from(candidate).unwrap();
        let predicate =
            preview_grinding_predicate(&transcript, run.grind_bits(), run.nonce_bits(), candidate)
                .unwrap();
        if grinding_predicate_accepts(&predicate, grind_bits) {
            accepted.get_or_insert(candidate);
        } else {
            rejected.get_or_insert(candidate);
        }
        if accepted.is_some() && rejected.is_some() {
            break;
        }
    }
    let accepted = accepted.expect("one-bit predicate must have an accepted nonce");
    let rejected = rejected.expect("one-bit predicate must have a rejected nonce");

    let stream_for = |nonce| {
        let mut writer = TranscriptNonceWriter::new(&plan).unwrap();
        writer.write(site, nonce).unwrap();
        writer.finish().unwrap()
    };

    let accepted_stream = stream_for(accepted);
    let mut accepted_transcript =
        AkitaTranscript::<Prime128Offset275>::verifier(b"grinding-predicate-test", b"instance");
    let mut accepted_replay =
        VerifierGrindingTranscript::new(&mut accepted_transcript, &accepted_stream, &plan).unwrap();
    accepted_replay.grind_query(site).unwrap();
    let _: Prime128Offset275 =
        accepted_replay.challenge_scalar(site.proof_of_work_label().unwrap());
    accepted_replay.finish().unwrap();

    let rejected_stream = stream_for(rejected);
    let mut rejected_transcript =
        AkitaTranscript::<Prime128Offset275>::verifier(b"grinding-predicate-test", b"instance");
    let mut rejected_replay =
        VerifierGrindingTranscript::new(&mut rejected_transcript, &rejected_stream, &plan).unwrap();
    assert!(rejected_replay.grind_query(site).is_err());
}

#[test]
fn nonce_stream_is_little_endian_and_crosses_byte_boundaries() {
    let plan = stream_test_plan();
    let alpha = test_nonce(GrindingSite::RingSwitchAlpha { level: 0 }, 8);
    let first_fold = test_fold_response_nonce(GrindingSite::FoldResponse { level: 0 });
    let tau0 = test_nonce(GrindingSite::Tau0Point { level: 0 }, 9);
    let second_fold = test_fold_response_nonce(GrindingSite::FoldResponse { level: 1 });
    let mut writer = TranscriptNonceWriter::new(&plan).unwrap();
    writer
        .write(GrindingSite::RingSwitchAlpha { level: 0 }, alpha)
        .unwrap();
    writer
        .write(GrindingSite::FoldResponse { level: 0 }, first_fold)
        .unwrap();
    writer
        .write(GrindingSite::Tau0Point { level: 0 }, tau0)
        .unwrap();
    writer
        .write(GrindingSite::FoldResponse { level: 1 }, second_fold)
        .unwrap();
    let stream = writer.finish().unwrap();
    assert_eq!(stream.bit_len(), 41);
    assert_eq!(stream.as_bytes(), &[0x00, 0x04, 0x00, 0xa0, 0x00, 0x00]);

    let mut reader = stream.reader(&plan).unwrap();
    assert_eq!(
        reader
            .read(GrindingSite::RingSwitchAlpha { level: 0 })
            .unwrap(),
        alpha
    );
    assert_eq!(
        reader
            .read(GrindingSite::FoldResponse { level: 0 })
            .unwrap(),
        first_fold
    );
    assert_eq!(
        reader.read(GrindingSite::Tau0Point { level: 0 }).unwrap(),
        tau0
    );
    assert_eq!(
        reader
            .read(GrindingSite::FoldResponse { level: 1 })
            .unwrap(),
        second_fold
    );
    reader.finish().unwrap();
}

#[test]
fn exact_cursor_rejects_omitted_entries_and_checks_fold_width() {
    let plan = stream_test_plan();
    let first_fold_nonce = test_fold_response_nonce(GrindingSite::FoldResponse { level: 0 });
    let second_fold_nonce = test_fold_response_nonce(GrindingSite::FoldResponse { level: 1 });
    let mut writer = TranscriptNonceWriter::new(&plan).unwrap();
    assert!(writer
        .write_fold_response(GrindingSite::FoldResponse { level: 0 }, first_fold_nonce,)
        .is_err());

    let mut writer = TranscriptNonceWriter::new(&plan).unwrap();
    writer
        .write(GrindingSite::RingSwitchAlpha { level: 0 }, u32::default())
        .unwrap();
    writer
        .write_fold_response(GrindingSite::FoldResponse { level: 0 }, first_fold_nonce)
        .unwrap();
    writer
        .write(GrindingSite::Tau0Point { level: 0 }, u32::default())
        .unwrap();
    assert!(writer
        .write_fold_response(
            GrindingSite::FoldResponse { level: 1 },
            FOLD_RESPONSE_ATTEMPTS,
        )
        .is_err());

    let mut writer = TranscriptNonceWriter::new(&plan).unwrap();
    writer
        .write(GrindingSite::RingSwitchAlpha { level: 0 }, u32::default())
        .unwrap();
    writer
        .write_fold_response(GrindingSite::FoldResponse { level: 0 }, first_fold_nonce)
        .unwrap();
    writer
        .write(GrindingSite::Tau0Point { level: 0 }, u32::default())
        .unwrap();
    writer
        .write_fold_response(GrindingSite::FoldResponse { level: 1 }, second_fold_nonce)
        .unwrap();
    let stream = writer.finish().unwrap();

    let mut reader = stream.reader(&plan).unwrap();
    assert!(reader
        .read_fold_response(GrindingSite::FoldResponse { level: 0 })
        .is_err());

    let mut reader = stream.reader(&plan).unwrap();
    assert_eq!(
        reader
            .read(GrindingSite::RingSwitchAlpha { level: 0 })
            .unwrap(),
        0
    );
    assert_eq!(
        reader
            .read_fold_response(GrindingSite::FoldResponse { level: 0 })
            .unwrap(),
        first_fold_nonce
    );
    assert_eq!(
        reader.read(GrindingSite::Tau0Point { level: 0 }).unwrap(),
        0
    );
    assert_eq!(
        reader
            .read_fold_response(GrindingSite::FoldResponse { level: 1 })
            .unwrap(),
        second_fold_nonce
    );
    reader.finish().unwrap();
}

#[test]
fn nonce_stream_packs_max_width_after_an_unaligned_entry() {
    let fold_site = GrindingSite::FoldResponse { level: 0 };
    let pow_site = GrindingSite::RingSwitchAlpha { level: 0 };
    let plan = GrindingPlan::new(
        vec![
            GrindingRun::fold_response(0),
            GrindingRun::proof_of_work(pow_site, 1u64 << MAX_GRINDING_BITS, 128).unwrap(),
        ],
        128,
    )
    .unwrap();
    assert_eq!(plan.runs()[1].nonce_bits(), 32);

    let mut writer = TranscriptNonceWriter::new(&plan).unwrap();
    writer.write_fold_response(fold_site, 0xabc).unwrap();
    writer.write(pow_site, u32::MAX).unwrap();
    let stream = writer.finish().unwrap();
    assert_eq!(stream.bit_len(), 44);
    assert_eq!(stream.as_bytes(), &[0xbc, 0xfa, 0xff, 0xff, 0xff, 0x0f]);

    let mut reader = stream.reader(&plan).unwrap();
    assert_eq!(reader.read_fold_response(fold_site).unwrap(), 0xabc);
    assert_eq!(reader.read(pow_site).unwrap(), u32::MAX);
    reader.finish().unwrap();
}

#[test]
fn nonce_stream_rejects_wrong_length_and_nonzero_padding() {
    assert!(TranscriptNonceStream::from_bytes(vec![0], 9).is_err());
    assert!(TranscriptNonceStream::from_bytes(vec![0, 0x80], 9).is_err());
    assert!(TranscriptNonceStream::from_bytes(vec![0, 1], 9).is_ok());
}

#[test]
fn current_capacity_prices_exact_nominal_loss_bits() {
    for (loss, expected) in [(1, 0), (2, 1), (3, 2), (4, 2), (5, 3), (u64::MAX, 64)] {
        let actual = if expected > u32::from(MAX_GRINDING_BITS) {
            grind_bits_for_loss(loss, 128).expect_err("oversized target")
        } else {
            let actual = grind_bits_for_loss(loss, 128).expect("supported target");
            assert_eq!(u32::from(actual), expected);
            continue;
        };
        assert!(matches!(actual, AkitaError::InvalidSetup(_)));
    }
}

#[test]
fn nominal_security_inequality_holds_for_every_supported_target() {
    let losses = [
        1,
        2,
        3,
        4,
        5,
        (1u64 << (MAX_GRINDING_BITS - 1)) - 1,
        1u64 << (MAX_GRINDING_BITS - 1),
        (1u64 << MAX_GRINDING_BITS) - 1,
        1u64 << MAX_GRINDING_BITS,
    ];
    for loss in losses {
        let grind = grind_bits_for_loss(loss, 128).expect("supported loss");
        assert!(u128::from(loss) <= (1u128 << grind));
    }
}

#[test]
fn nonce_slack_provisions_exactly_128_expected_trials() {
    for grind in 1..=MAX_GRINDING_BITS {
        let nonce_bits = grind + GRINDING_NONCE_SLACK_BITS;
        assert_eq!((1u64 << nonce_bits) / (1u64 << grind), 128);
        let failure = (1.0 - 2f64.powi(-i32::from(grind))).powf(2f64.powi(i32::from(nonce_bits)));
        assert!(failure <= (-128f64).exp());
    }
}

#[test]
fn plan_encoding_covers_every_discriminator() {
    let capacity = 128;
    let sites = [
        GrindingSite::EvaluationBatch { level: 0 },
        GrindingSite::ExtensionOpeningPoint { level: 0 },
        GrindingSite::ExtensionOpeningClaimBatch { level: 0 },
        GrindingSite::SumcheckRound {
            protocol: SumcheckProtocol::ExtensionOpeningReduction,
            level: 0,
            stage: 1,
            round: 2,
        },
        GrindingSite::SumcheckRound {
            protocol: SumcheckProtocol::Stage1,
            level: 3,
            stage: 4,
            round: 5,
        },
        GrindingSite::SumcheckRound {
            protocol: SumcheckProtocol::PhysicalL2,
            level: 6,
            stage: 7,
            round: 8,
        },
        GrindingSite::SumcheckRound {
            protocol: SumcheckProtocol::Stage2,
            level: 9,
            stage: 10,
            round: 11,
        },
        GrindingSite::SumcheckRound {
            protocol: SumcheckProtocol::Stage3,
            level: 12,
            stage: 13,
            round: 14,
        },
        GrindingSite::RingSwitchAlpha { level: 1 },
        GrindingSite::Tau0Point { level: 1 },
        GrindingSite::Tau1Point { level: 1 },
        GrindingSite::Stage1InterstageBatch { level: 1, stage: 2 },
        GrindingSite::L2SubclaimBatch { level: 1 },
        GrindingSite::L2NormMerge { level: 1 },
        GrindingSite::L2VirtualBatch { level: 1 },
        GrindingSite::CompressionBinary { level: 1 },
        GrindingSite::Stage2Batch { level: 1 },
    ];
    let mut runs = sites
        .into_iter()
        .map(|site| GrindingRun::proof_of_work(site, 3, capacity).unwrap())
        .collect::<Vec<_>>();
    runs.push(GrindingRun::fold_response(2));
    runs.push(GrindingRun::fold_challenge_group(2, 3, 4).unwrap());
    let plan = GrindingPlan::new(runs, capacity).unwrap();
    let bytes = plan.canonical_bytes().unwrap();
    assert!(bytes.starts_with(GRINDING_PLAN_DOMAIN));
    assert_eq!(plan.expanded_query_count(), 23);
    assert_eq!(plan.total_nonce_bits(), 17 * 9 + 12);
    assert_eq!(
        plan.digest().unwrap(),
        [
            39, 122, 171, 62, 218, 90, 138, 181, 88, 68, 8, 137, 172, 57, 59, 185, 161, 221, 184,
            197, 193, 194, 6, 57, 56, 205, 203, 0, 209, 126, 16, 38,
        ]
    );
}

#[test]
fn ring_switch_loss_uses_the_opening_polynomial_dimension() {
    assert_eq!(
        ring_switch_alpha_loss_factor(OpeningMethod::EvaluationTrace, 64).unwrap(),
        127
    );
    assert_eq!(
        ring_switch_alpha_loss_factor(
            OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension: 16,
            },
            64,
        )
        .unwrap(),
        31
    );
}

#[test]
fn special_proof_of_work_site_and_reserved_sentinel_are_rejected() {
    assert!(GrindingRun::proof_of_work(GrindingSite::FoldResponse { level: 0 }, 1, 128).is_err());

    let mut underpriced =
        GrindingRun::proof_of_work(GrindingSite::RingSwitchAlpha { level: 0 }, 3, 128).unwrap();
    underpriced.grind_bits = 1;
    underpriced.nonce_bits = 8;
    assert!(GrindingPlan::new(vec![underpriced], 128).is_err());

    let reserved = GrindingRun::proof_of_work(
        GrindingSite::SumcheckRound {
            protocol: SumcheckProtocol::Stage2,
            level: u32::MAX,
            stage: 0,
            round: 0,
        },
        3,
        128,
    )
    .unwrap();
    assert!(GrindingPlan::new(vec![reserved], 128).is_err());
}

#[test]
fn public_plan_rejects_query_limit_without_expanding_runs() {
    let accepted =
        GrindingRun::fold_challenge_group(0, 0, TRANSCRIPT_GRINDING_QUERY_LIMIT - 2).unwrap();
    assert_eq!(
        GrindingPlan::new(vec![accepted], 128)
            .unwrap()
            .expanded_query_count(),
        TRANSCRIPT_GRINDING_QUERY_LIMIT - 1
    );

    let excessive =
        GrindingRun::fold_challenge_group(0, 0, TRANSCRIPT_GRINDING_QUERY_LIMIT - 1).unwrap();
    assert!(matches!(
        GrindingPlan::new(vec![excessive], 128),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn planner_accumulator_prices_an_oversized_edge_without_plan_validation() {
    let run = GrindingRun::fold_challenge_group(0, 0, TRANSCRIPT_GRINDING_QUERY_LIMIT - 1).unwrap();
    let mut accumulator = GrindingPlanAccumulator::new(128).unwrap();
    accumulator.push(run).unwrap();
    assert_eq!(
        accumulator.cost(),
        TranscriptGrindingCost {
            total_nonce_bits: 0,
            expanded_query_count: TRANSCRIPT_GRINDING_QUERY_LIMIT,
        }
    );
}
