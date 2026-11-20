use super::*;
use crate::{FoldChallengeDrawDomain, FoldChallengeFamily, FoldDraw};

struct Draw<'a>(&'a [u8]);
impl FoldDraw for Draw<'_> {
    fn draw_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<T>, AkitaError>,
    ) -> Result<Option<T>, AkitaError> {
        assert_eq!(len, self.0.len());
        decode(self.0)
    }
}

/// Independent producer: decode without advice, then serialize positions in
/// the order a serial verifier examines candidates, before predicate rejection.
fn position_tape(
    stream: &SparseChallengeStream,
    message: &[u8],
    expected: &[SparseChallenge],
) -> Vec<u8> {
    let binomials = BinomialTable::new(
        stream.shell.ring_d,
        stream
            .shell
            .config
            .count_pm1
            .max(stream.shell.config.count_pm2),
    )
    .unwrap();
    let decoder = CandidateDecoder {
        position_hints: false,
        stream,
        message,
        binomials: &binomials,
    };
    let (mut accepted, mut rejected, mut outside) = (0, 0, 0);
    let mut tape = Vec::new();
    for index in 0..stream.candidates() {
        let Some(candidate) = decoder.candidate(index).unwrap() else {
            outside += 1;
            continue;
        };
        let (pm1, pm2) = candidate.positions.split_at(stream.shell.config.count_pm1);
        let positions = pm1.iter().copied().chain(
            pm2.iter()
                .map(|&p| p - pm1.iter().filter(|&&q| q < p).count() as u32),
        );
        let mut words = vec![0u64; stream.shell.config.weight().div_ceil(4)];
        for (i, position) in positions.enumerate() {
            words[i / 4] |= u64::from(position) << (16 * (i % 4));
        }
        tape.extend(words.into_iter().flat_map(u64::to_le_bytes));
        if candidate == expected[accepted] {
            accepted += 1;
            if accepted == expected.len() {
                break;
            }
        } else {
            rejected += 1;
        }
    }
    assert_eq!(accepted, expected.len());
    if stream.shell.rejection.is_some() {
        assert!(rejected > 0, "must include predicate-rejected candidates");
        assert!(outside > 0, "must skip out-of-range candidates");
    }
    tape
}

#[test]
fn versioned_frame_round_trips_mixed_fold_routes() {
    let mut rng = SplitMix(91);
    let cases = [
        (
            64,
            SparseChallengeConfig::production_for_ring_dim(64).unwrap(),
            None,
            1023,
        ),
        (
            128,
            SparseChallengeConfig::production_for_ring_dim(128).unwrap(),
            None,
            600,
        ),
        (
            64,
            D64_SELECTIVE_L2_CHALLENGE_CONFIG,
            Some(OperatorNormRejection::D64_SELECTIVE_L2),
            600,
        ),
    ];
    let mut draws = Vec::new();
    let quartets = include_bytes!("../../../testdata/checked_hint_codes.bin");
    let mut frame = b"AKITAH02".to_vec();
    frame.extend_from_slice(&(quartets.len() as u64).to_le_bytes());
    frame.extend_from_slice(quartets);
    for (d, config, rejection, count) in cases {
        let family = FoldChallengeFamily::new(
            FoldChallengeDrawDomain::EvaluationTrace,
            d,
            config,
            rejection,
        )
        .unwrap();
        let stream = family.stream(count).unwrap();
        assert!(!stream.uses_checked_quartets());
        let message = rng.message(stream.message_len());
        let expected = stream.decode(&message).unwrap().unwrap();
        frame.extend(position_tape(&stream, &message, &expected));
        draws.push((family, stream, message, expected));
    }
    crate::install_checked_hints(&frame, frame.len()).unwrap();
    for (family, stream, message, expected) in draws {
        // Ordinary decode ignores the installed tape, even above the parallel threshold.
        assert_eq!(stream.decode(&message).unwrap().unwrap(), expected);
        #[cfg(feature = "parallel")]
        assert_eq!(stream.parallel(), !capture_enabled());
        let hinted = Draw(&message)
            .draw_fold_challenges(&family, stream.count, 1)
            .unwrap()
            .unwrap();
        assert_eq!(hinted.as_slice(), expected);
    }
    let family = FoldChallengeFamily::new(
        FoldChallengeDrawDomain::EvaluationTrace,
        64,
        SparseChallengeConfig::production_for_ring_dim(64).unwrap(),
        None,
    )
    .unwrap();
    let stream = family.stream(1024).unwrap();
    assert!(stream.uses_checked_quartets());
    let message = include_bytes!("../../../testdata/checked_hint_message.bin");
    let expected = stream.decode(message).unwrap().unwrap();
    let hinted = Draw(message)
        .draw_fold_challenges(&family, 1024, 1)
        .unwrap()
        .unwrap();
    assert_eq!(hinted.as_slice(), expected);
    crate::finish_checked_hints().unwrap();
}

#[test]
fn malformed_position_records_reject_through_fold_drawing() {
    let config = SparseChallengeConfig::production_for_ring_dim(128).unwrap();
    let family =
        FoldChallengeFamily::new(FoldChallengeDrawDomain::EvaluationTrace, 128, config, None)
            .unwrap();
    let stream = family.stream(3).unwrap();
    let message = SplitMix(92).message(stream.message_len());
    let expected = stream.decode(&message).unwrap().unwrap();
    let mut frame = b"AKITAH02".to_vec();
    frame.extend_from_slice(&0u64.to_le_bytes());
    frame.extend(position_tape(&stream, &message, &expected));
    let mut changed_position = frame.clone();
    changed_position[16] ^= 1;
    let mut padding = frame.clone();
    padding[16 + 31 * 2] = 1;
    let truncated = frame[..frame.len() - 8].to_vec();
    for invalid in [changed_position, padding, truncated] {
        crate::install_checked_hints(&invalid, frame.len()).unwrap();
        assert!(Draw(&message).draw_fold_challenges(&family, 3, 1).is_err());
        crate::abort_checked_hints().unwrap();
    }
    frame.extend_from_slice(&0u64.to_le_bytes());
    crate::install_checked_hints(&frame, frame.len()).unwrap();
    assert!(Draw(&message)
        .draw_fold_challenges(&family, 3, 1)
        .unwrap()
        .is_some());
    assert!(crate::finish_checked_hints().is_err());
}

#[test]
fn compact_quartets_bind_magnitudes_positions_and_signs() {
    let config = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
    let stream = layout(64, &config, None, 1024).unwrap();
    let mut words = [0u64; 4];
    // Positions 0..31 are -1, 31..41 are -2: both subset ranks are zero.
    for q in 0..16 {
        let mut code = 0;
        for i in (0..4).rev() {
            let position = 4 * q + i;
            let digit = if position < 31 {
                1
            } else if position < 41 {
                0
            } else {
                2
            };
            code = 5 * code + digit;
        }
        words[q / 4] |= code << (10 * (q % 4));
    }
    let rank_space = stream.shell.pm1_rank_space;
    let signs = (1u64 << 41) - 1;
    crate::checked_hint::validate(&words, 0, rank_space, signs).unwrap();
    for bit in 0..41 {
        assert!(crate::checked_hint::validate(&words, 0, rank_space, signs ^ (1 << bit)).is_err());
    }
    assert!(crate::checked_hint::validate(&words, 1, rank_space, signs).is_err());
    // A zero-metadata fallback would preserve the otherwise zero final quartet.
    words[3] = (words[3] & !(1023u64 << 30)) | (937u64 << 30);
    assert!(crate::checked_hint::validate(&words, 0, rank_space, signs).is_err());
}

#[test]
fn quartet_eligibility_matches_the_family_and_threshold() {
    for (d, config, rejection, count, eligible) in [
        (
            64,
            SparseChallengeConfig::production_for_ring_dim(64).unwrap(),
            None,
            1023,
            false,
        ),
        (
            64,
            SparseChallengeConfig::production_for_ring_dim(64).unwrap(),
            None,
            1024,
            true,
        ),
        (
            128,
            SparseChallengeConfig::production_for_ring_dim(128).unwrap(),
            None,
            1024,
            false,
        ),
        (
            64,
            D64_SELECTIVE_L2_CHALLENGE_CONFIG,
            Some(OperatorNormRejection::D64_SELECTIVE_L2),
            1024,
            false,
        ),
    ] {
        assert_eq!(
            layout(d, &config, rejection, count)
                .unwrap()
                .uses_checked_quartets(),
            eligible
        );
    }
}
