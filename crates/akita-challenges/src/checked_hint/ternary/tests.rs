use super::*;
use crate::checked_hint::frame;
use crate::{
    FoldChallengeDrawDomain, FoldChallengeFamily, FoldDraw, SparseChallenge, SparseChallengeConfig,
};
use jolt_field::{One, Prime128Offset6A23801F as F, Ring, Zero};

struct Message(Vec<u8>);
impl FoldDraw for Message {
    fn draw_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<T>, AkitaError>,
    ) -> Result<Option<T>, AkitaError> {
        assert_eq!(len, self.0.len());
        decode(&self.0)
    }
}
fn family(n: usize, a: usize, b: usize) -> FoldChallengeFamily {
    FoldChallengeFamily::new(
        FoldChallengeDrawDomain::SubringCoefficientPacking {
            challenge_subring_dimension: n,
        },
        n,
        SparseChallengeConfig {
            count_pm1: a,
            count_pm2: b,
        },
        None,
    )
    .unwrap()
}
fn message(family: &FoldChallengeFamily, count: usize) -> Message {
    let mut state = 0x123456789abcdefu64;
    Message(
        (0..family.stream(count).unwrap().message_len())
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect(),
    )
}
fn choose(n: usize, k: usize) -> u128 {
    if k > n {
        return 0;
    }
    (1..=k).fold(1u128, |value, i| value * (n + 1 - i) as u128 / i as u128)
}
fn pack(challenge: &SparseChallenge) -> [u64; 4] {
    let mut codes = [ZERO_CODE as i32; GROUPS];
    for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
        let p = position as usize;
        codes[p / 6] += i32::from(coefficient) * 3i32.pow((p % 6) as u32);
    }
    let mut words = [0; 4];
    for (i, code) in codes.into_iter().enumerate() {
        words[i / 6] |= (code as u64) << (10 * (i % 6));
    }
    words
}

#[test]
fn combined_rank_tables_match_independent_binomial_sums() {
    for (group, block) in RANKS.iter().enumerate() {
        for (prefix, row) in block.iter().enumerate() {
            for (mask, &value) in row.iter().enumerate() {
                if prefix + mask.count_ones() as usize > 31 || (group == 21 && mask & !3 != 0) {
                    continue;
                }
                let mut count = prefix;
                let expected = (0..6)
                    .filter(|i| mask & (1 << i) != 0)
                    .map(|i| {
                        count += 1;
                        choose(group * 6 + i, count)
                    })
                    .sum::<u128>();
                assert_eq!(value, expected, "group={group} prefix={prefix} mask={mask}");
            }
        }
    }
}

#[test]
fn packed_advice_matches_transcript_and_sparse_evaluation_after_embedding() {
    let family = family(128, 31, 0);
    let mut source = message(&family, 1024);
    let expected = source
        .draw_fold_challenges(&family, 1024, 1)
        .unwrap()
        .unwrap();
    let packed = expected.as_slice().iter().map(pack).collect::<Vec<_>>();
    let tape = packed
        .iter()
        .flat_map(|words| words.iter().flat_map(|w| w.to_le_bytes()))
        .collect::<Vec<_>>();
    let bytes = frame(&[], &tape, &[]);
    let unaligned = [vec![0], bytes.clone()].concat();
    crate::install_checked_hints(&unaligned[1..], bytes.len()).unwrap();
    let actual = source
        .draw_fold_challenges(&family, 1024, 1)
        .unwrap()
        .unwrap();
    crate::finish_checked_hints().unwrap();
    for alpha in [F::zero(), F::one(), -F::one(), F::from_u64(37)] {
        for stride in [1, 4] {
            let mut p = F::one();
            let powers = (0..128 * stride)
                .map(|_| {
                    let result = p;
                    p *= alpha;
                    result
                })
                .collect::<Vec<_>>();
            let a = actual
                .embed_subring_positions(128, stride, 128 * stride)
                .unwrap();
            let b = expected
                .embed_subring_positions(128, stride, 128 * stride)
                .unwrap();
            assert_eq!(
                a.evals_at_pows::<F, F>(&powers).unwrap(),
                b.evals_at_pows::<F, F>(&powers).unwrap()
            );
            assert_eq!(a.as_slice(), b.as_slice());
        }
    }
    for challenge in expected.as_slice().iter().take(32) {
        let words = pack(challenge);
        let rank = challenge
            .positions
            .iter()
            .enumerate()
            .map(|(i, &p)| choose(p as usize, i + 1))
            .sum();
        let signs = challenge
            .coeffs
            .iter()
            .enumerate()
            .fold(0u64, |v, (i, &c)| v | (u64::from(c < 0) << i));
        validate(&words, rank, signs).unwrap();
        for bit in 0..256 {
            let mut bad = words;
            bad[bit / 64] ^= 1u64 << (bit % 64);
            assert!(validate(&bad, rank, signs).is_err(), "accepted bit {bit}");
        }
    }
}

#[test]
fn frame_rejects_missing_excess_and_reordered_advice() {
    let f = family(128, 31, 0);
    let mut source = message(&f, 1024);
    let reference = source.draw_fold_challenges(&f, 1024, 1).unwrap().unwrap();
    let mut tape = reference
        .as_slice()
        .iter()
        .map(pack)
        .flat_map(|words| words.into_iter().flat_map(|w| w.to_le_bytes()))
        .collect::<Vec<_>>();
    for length in 0..16 {
        let header = frame(&[], &tape, &[]);
        assert!(crate::install_checked_hints(&header[..length], header.len()).is_err());
    }
    let first = tape[..32].to_vec();
    tape.copy_within(32..64, 0);
    tape[32..64].copy_from_slice(&first);
    let reordered = frame(&[], &tape, &[]);
    crate::install_checked_hints(&reordered, reordered.len()).unwrap();
    assert!(source.draw_fold_challenges(&f, 1024, 1).is_err());
    crate::abort_checked_hints().unwrap();
    tape.copy_within(0..32, 32);
    tape[..32].copy_from_slice(&first);
    let missing = frame(&[], &tape[..tape.len() - 8], &[]);
    crate::install_checked_hints(&missing, missing.len()).unwrap();
    assert!(source.draw_fold_challenges(&f, 1024, 1).is_err());
    crate::abort_checked_hints().unwrap();
    tape.extend([0; 32]);
    let excess = frame(&[], &tape, &[]);
    crate::install_checked_hints(&excess, excess.len()).unwrap();
    source.draw_fold_challenges(&f, 1024, 1).unwrap().unwrap();
    assert!(crate::finish_checked_hints().is_err());
    assert!(crate::install_checked_hints(&excess, excess.len() - 1).is_err());
}

#[test]
fn frame_keeps_three_interleaved_tapes_independent() {
    let f = family(128, 31, 0);
    let mut source = message(&f, 1024);
    let expected = source.draw_fold_challenges(&f, 1024, 1).unwrap().unwrap();
    let ternary = expected
        .as_slice()
        .iter()
        .map(pack)
        .flat_map(|words| words.into_iter().flat_map(u64::to_le_bytes))
        .collect::<Vec<_>>();
    let mut small = message(&f, 2);
    let small_expected = small.draw_fold_challenges(&f, 2, 1).unwrap().unwrap();
    let mut positions = Vec::new();
    for challenge in small_expected.as_slice() {
        let mut words = [0u64; 8];
        for (i, &position) in challenge.positions.iter().enumerate() {
            words[i / 4] |= u64::from(position) << (16 * (i % 4));
        }
        positions.extend(words.into_iter().flat_map(u64::to_le_bytes));
    }
    let q = include_bytes!("../../../testdata/checked_hint_codes.bin");
    let frame = frame(q, &ternary, &positions);
    crate::install_checked_hints(&frame, frame.len()).unwrap();
    assert_eq!(
        small
            .draw_fold_challenges(&f, 2, 1)
            .unwrap()
            .unwrap()
            .as_slice(),
        small_expected.as_slice()
    );
    assert_eq!(
        source
            .draw_fold_challenges(&f, 1024, 1)
            .unwrap()
            .unwrap()
            .as_slice(),
        expected.as_slice()
    );
    let mut quartet =
        Message(include_bytes!("../../../testdata/checked_hint_message.bin").to_vec());
    let quartet_family = family(64, 31, 10);
    quartet
        .draw_fold_challenges(&quartet_family, 1024, 1)
        .unwrap()
        .unwrap();
    crate::finish_checked_hints().unwrap();
}
