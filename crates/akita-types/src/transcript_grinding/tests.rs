use super::*;
use akita_field::Prime128Offset275;
use akita_transcript::{
    grinding_predicate_accepts, preview_grinding_predicate, search_grinding_nonce, AkitaTranscript,
    Transcript,
};
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
fn nonce_stream_is_little_endian_and_crosses_byte_boundaries() {
    let plan = stream_test_plan();
    let mut writer = TranscriptNonceWriter::new(&plan).unwrap();
    writer
        .write(GrindingSite::RingSwitchAlpha { level: 0 }, 0xa5)
        .unwrap();
    writer
        .write(GrindingSite::FoldResponse { level: 0 }, 0xabc)
        .unwrap();
    writer
        .write(GrindingSite::Tau0Point { level: 0 }, 0x101)
        .unwrap();
    writer
        .write(GrindingSite::FoldResponse { level: 1 }, 0x123)
        .unwrap();
    let stream = writer.finish().unwrap();
    assert_eq!(stream.bit_len(), 41);
    assert_eq!(stream.as_bytes(), &[0xa5, 0xbc, 0x1a, 0x70, 0x24, 0x00]);

    let mut reader = stream.reader(&plan).unwrap();
    assert_eq!(
        reader
            .read(GrindingSite::RingSwitchAlpha { level: 0 })
            .unwrap(),
        0xa5
    );
    assert_eq!(
        reader
            .read(GrindingSite::FoldResponse { level: 0 })
            .unwrap(),
        0xabc
    );
    assert_eq!(
        reader.read(GrindingSite::Tau0Point { level: 0 }).unwrap(),
        0x101
    );
    assert_eq!(
        reader
            .read(GrindingSite::FoldResponse { level: 1 })
            .unwrap(),
        0x123
    );
    reader.finish().unwrap();
}

#[test]
fn exact_cursor_rejects_omitted_entries_and_checks_fold_width() {
    let plan = stream_test_plan();
    let mut writer = TranscriptNonceWriter::new(&plan).unwrap();
    assert!(writer
        .write_fold_response(GrindingSite::FoldResponse { level: 0 }, 17)
        .is_err());

    let mut writer = TranscriptNonceWriter::new(&plan).unwrap();
    writer
        .write(GrindingSite::RingSwitchAlpha { level: 0 }, 0)
        .unwrap();
    writer
        .write_fold_response(GrindingSite::FoldResponse { level: 0 }, 17)
        .unwrap();
    writer
        .write(GrindingSite::Tau0Point { level: 0 }, 0)
        .unwrap();
    assert!(writer
        .write_fold_response(
            GrindingSite::FoldResponse { level: 1 },
            FOLD_RESPONSE_ATTEMPTS,
        )
        .is_err());

    let mut writer = TranscriptNonceWriter::new(&plan).unwrap();
    writer
        .write(GrindingSite::RingSwitchAlpha { level: 0 }, 0)
        .unwrap();
    writer
        .write_fold_response(GrindingSite::FoldResponse { level: 0 }, 17)
        .unwrap();
    writer
        .write(GrindingSite::Tau0Point { level: 0 }, 0)
        .unwrap();
    writer
        .write_fold_response(GrindingSite::FoldResponse { level: 1 }, 23)
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
        17
    );
    assert_eq!(
        reader.read(GrindingSite::Tau0Point { level: 0 }).unwrap(),
        0
    );
    assert_eq!(
        reader
            .read_fold_response(GrindingSite::FoldResponse { level: 1 })
            .unwrap(),
        23
    );
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
        GrindingSite::EvaluationBatch,
        GrindingSite::ExtensionOpeningPoint,
        GrindingSite::ExtensionOpeningClaimBatch,
        GrindingSite::SumcheckRound {
            protocol: SumcheckProtocol::ExtensionOpeningReduction,
            level: u32::MAX,
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
    runs.push(GrindingRun::fold_challenge_root(2, 3));
    runs.push(GrindingRun::fold_challenge_coordinates(2, 3, 4));
    let plan = GrindingPlan::new(runs, capacity).unwrap();
    let bytes = plan.canonical_bytes().unwrap();
    assert!(bytes.starts_with(GRINDING_PLAN_DOMAIN));
    assert_eq!(plan.expanded_query_count(), 23);
    assert_eq!(plan.total_nonce_bits(), 17 * 9 + 12);
    assert_eq!(
        plan.digest().unwrap(),
        [
            201, 71, 193, 56, 131, 65, 105, 160, 79, 152, 66, 44, 189, 232, 205, 168, 168, 208, 84,
            23, 96, 48, 174, 168, 14, 112, 165, 199, 177, 190, 157, 156,
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
