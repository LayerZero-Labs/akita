//! Byte-for-byte fixtures generated from the original preview implementations.
//! Regenerate with `scripts/gen_indexed_preview.py`; the fixture records source
//! revision and SHA-256 hashes. No preview crate is a test dependency.

use akita_challenges::{
    sample_challenge_ct_counted, sample_indexed_challenges_from_seed, OperatorNormRejection,
    SparseChallenge, SparseChallengeConfig,
};

fn encoded(challenge: &SparseChallenge) -> String {
    if challenge.positions.is_empty() {
        return "-".into();
    }
    challenge
        .positions
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .chain(challenge.coeffs.iter().map(|&x| x as u8))
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[test]
fn fixed_seeds_match_preview_output_bytes() {
    let mut rows = 0;
    let mut norm_redraws = 0;
    for line in include_str!("fixtures/indexed_preview.txt").lines() {
        if line.starts_with('#') {
            continue;
        }
        let columns: Vec<_> = line.split_whitespace().collect();
        assert_eq!(columns.len(), 8, "fixture row {line}");
        let numbers: Vec<usize> = columns[..6].iter().map(|s| s.parse().unwrap()).collect();
        let [tag, ring_d, count_pm1, count_pm2, norm, coordinate] = numbers[..] else {
            unreachable!()
        };
        let seed: [u8; 32] = std::array::from_fn(|i| if tag == 256 { i as u8 } else { tag as u8 });
        let cfg = SparseChallengeConfig {
            count_pm1,
            count_pm2,
        };
        let rejection = match (norm, ring_d) {
            (0, _) => None,
            (1, 64) => Some(OperatorNormRejection::D64_SELECTIVE_L2),
            (1, 128) => Some(OperatorNormRejection::D128_SELECTIVE_L2),
            _ => panic!("invalid fixture policy"),
        };
        let challenges =
            sample_indexed_challenges_from_seed(&seed, ring_d, coordinate + 1, &cfg, rejection)
                .unwrap();
        assert_eq!(
            encoded(&challenges[coordinate]),
            columns[6],
            "fixture row {line}"
        );
        challenges[coordinate].validate_dyn(ring_d).unwrap();
        let attempts: usize = columns[7].parse().unwrap();
        if attempts != 0 {
            let (ct, actual_attempts) = sample_challenge_ct_counted(&seed).unwrap();
            assert_eq!(encoded(&ct), columns[6], "CT fixture row {line}");
            assert_eq!(actual_attempts, attempts);
            assert!(!ct.positions.spilled());
            assert!(!ct.coeffs.spilled());
            norm_redraws += usize::from(attempts > 1);
        }
        rows += 1;
    }
    assert_eq!(rows, 198);
    assert!(norm_redraws > 0);
}
