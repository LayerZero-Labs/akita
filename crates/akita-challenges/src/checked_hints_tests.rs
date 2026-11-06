use crate::{
    Challenges, FoldChallengeDrawDomain, FoldChallengeFamily, FoldDraw, SparseChallengeConfig,
};
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
    crate::install_checked_hints(&hints[1..]).unwrap();
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
    crate::install_checked_hints(&hints).unwrap();
    assert!(message().draw_fold_challenges(&family(), 1024, 1).is_err());
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
