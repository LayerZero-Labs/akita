use crate::{FoldChallengeDrawDomain, FoldChallengeFamily, FoldDraw, SparseChallengeConfig};
use akita_error::AkitaError;
use jolt_field::{Prime128OffsetA7F7 as F, Ring};
struct Message(Vec<u8>);
impl FoldDraw for Message {
    fn draw_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<T>, AkitaError>,
    ) -> Result<Option<T>, AkitaError> {
        assert_eq!(len, self.0.len() - 1);
        decode(&self.0[1..])
    }
}
fn family() -> FoldChallengeFamily {
    FoldChallengeFamily::new(
        FoldChallengeDrawDomain::SubringCoefficientPacking {
            challenge_subring_dimension: 64,
        },
        64,
        SparseChallengeConfig {
            count_pm1: 31,
            count_pm2: 10,
        },
        None,
    )
    .unwrap()
}
fn message() -> Message {
    let mut bytes = vec![0];
    bytes.extend_from_slice(MESSAGE);
    Message(bytes)
}
#[test]
fn checked_hints_match_frozen_field_values_with_unaligned_ingress() {
    let mut hints = vec![0];
    hints.extend_from_slice(HINTS);
    crate::install_checked_hints(&hints[1..], HINTS.len()).unwrap();
    let sampled = message()
        .draw_fold_challenges(&family(), 1024, 1)
        .unwrap()
        .unwrap();
    crate::finish_checked_hints().unwrap();
    for (i, stride) in [1, 4].into_iter().enumerate() {
        let batch = sampled
            .embed_subring_positions(64, stride, 64 * stride)
            .unwrap();
        let mut p = F::from_u64(1);
        let powers = (0..64 * stride)
            .map(|_| {
                let v = p;
                p *= F::from_u64(37);
                v
            })
            .collect::<Vec<_>>();
        let values = batch.evals_at_pows::<F, F>(&powers).unwrap();
        assert_eq!(values[0], F::from_u128(EXPECTED[i][0]));
        assert_eq!(values[1023], F::from_u128(EXPECTED[i][1]));
        assert_eq!(
            values.iter().copied().fold(F::from_u64(0), |a, b| a + b),
            F::from_u128(EXPECTED[i][2])
        );
    }
}
#[test]
fn checked_hints_reject_another_valid_challenge_at_the_wrong_transcript_slot() {
    let mut hints = HINTS.to_vec();
    let first = hints[..32].to_vec();
    hints.copy_within(32..64, 0);
    hints[32..64].copy_from_slice(&first);
    crate::install_checked_hints(&hints, HINTS.len()).unwrap();
    assert!(message().draw_fold_challenges(&family(), 1024, 1).is_err());
    crate::abort_checked_hints().unwrap();
}

#[test]
fn checked_hint_sessions_reuse_the_same_thread_after_success_and_failure() {
    let family = family();
    let draw = || message().draw_fold_challenges(&family, 1024, 1);
    for _ in 0..2 {
        crate::install_checked_hints(HINTS, HINTS.len()).unwrap();
        assert!(draw().unwrap().is_some());
        crate::finish_checked_hints().unwrap();
        assert!(!crate::checked_hint::active());
    }

    // A malformed draw must stay rejected; abort only releases its session.
    let mut malformed = HINTS.to_vec();
    malformed[..8].fill(0xff);
    crate::install_checked_hints(&malformed, HINTS.len()).unwrap();
    assert!(draw().is_err());
    crate::abort_checked_hints().unwrap();
    assert!(!crate::checked_hint::active());

    crate::install_checked_hints(HINTS, HINTS.len()).unwrap();
    assert!(draw().unwrap().is_some());
    crate::finish_checked_hints().unwrap();
    // Unhinted decoding must no longer select an exhausted hint tape.
    assert!(draw().unwrap().is_some());
}

#[test]
fn finishing_an_incomplete_hint_session_rejects_and_clears_it() {
    crate::install_checked_hints(HINTS, HINTS.len()).unwrap();
    assert!(crate::finish_checked_hints().is_err());
    assert!(!crate::checked_hint::active());
    crate::install_checked_hints(HINTS, HINTS.len()).unwrap();
    assert!(message()
        .draw_fold_challenges(&family(), 1024, 1)
        .unwrap()
        .is_some());
    crate::finish_checked_hints().unwrap();
    assert!(crate::finish_checked_hints().is_err());
}
const MESSAGE: &[u8] = include_bytes!("../testdata/checked_hint_message.bin");
const HINTS: &[u8] = include_bytes!("../testdata/checked_hint_codes.bin");
const EXPECTED: [[u128; 3]; 2] = [
    [
        21047448056705363178069556910279857643u128,
        144914228926386039077961049912358002305u128,
        245957718946870999808281516405739222931u128,
    ],
    [
        133644585919095421225984225732961269912u128,
        18391955382529900038727915850102110357u128,
        326177195468316913605597546064481269453u128,
    ],
];

#[test]
fn checked_hints_match_for_aligned_and_unaligned_messages_and_tapes() {
    #[repr(align(8))]
    struct Aligned<const N: usize>([u8; N]);
    let mut message = Aligned([0u8; MESSAGE.len() + 8]);
    let mut hints = Aligned([0u8; HINTS.len() + 8]);
    let family = family();
    let stream = family.stream(1024).unwrap();
    let expected = stream.decode(MESSAGE).unwrap().unwrap();
    for offset in [0, 1] {
        message.0[offset..offset + MESSAGE.len()].copy_from_slice(MESSAGE);
        hints.0[offset..offset + HINTS.len()].copy_from_slice(HINTS);
        let bytes = &message.0[offset..offset + MESSAGE.len()];
        assert_eq!(
            bytes.as_ptr().align_offset(8),
            if offset == 0 { 0 } else { 7 }
        );
        crate::install_checked_hints(&hints.0[offset..offset + HINTS.len()], HINTS.len()).unwrap();
        let packed = stream.decode_checked_hints(bytes).unwrap();
        let actual = crate::Challenges::from_checked_quartets(packed, 1024, 1).unwrap();
        assert_eq!(actual.as_slice(), expected);
        crate::finish_checked_hints().unwrap();
    }
}
