use super::*;
use crate::{D128_SELECTIVE_L2_CHALLENGE_CONFIG, D64_SELECTIVE_L2_CHALLENGE_CONFIG};

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn message(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next() as u8).collect()
    }
}

/// Lay out `count` challenges of one family.
fn layout(
    ring_d: usize,
    config: &SparseChallengeConfig,
    rejection: Option<OperatorNormRejection>,
    count: usize,
) -> Result<SparseChallengeStream, AkitaError> {
    SparseChallengeShell::new(ring_d, config, rejection)?.stream(count)
}

fn production_families() -> Vec<(usize, SparseChallengeConfig, u32)> {
    [
        (64, 128),
        (128, 129),
        (256, 131),
        (512, 132),
        (1024, 131),
        (2048, 131),
    ]
    .into_iter()
    .map(|(d, bits)| {
        (
            d,
            SparseChallengeConfig::production_for_ring_dim(d).unwrap(),
            bits,
        )
    })
    .collect()
}

fn operator_norm_families() -> [(usize, SparseChallengeConfig, OperatorNormRejection); 2] {
    [
        (
            64,
            D64_SELECTIVE_L2_CHALLENGE_CONFIG,
            OperatorNormRejection::D64_SELECTIVE_L2,
        ),
        (
            128,
            D128_SELECTIVE_L2_CHALLENGE_CONFIG,
            OperatorNormRejection::D128_SELECTIVE_L2,
        ),
    ]
}

/// Colexicographic rank of an increasing subset.
fn colex_rank(subset: &[u32]) -> u128 {
    subset
        .iter()
        .enumerate()
        .map(|(i, &c)| binomial(c as usize, i + 1).unwrap())
        .sum()
}

/// Re-encode a decoded challenge as its candidate integer `(signs, rank)`.
fn encode(stream: &SparseChallengeStream, challenge: &SparseChallenge) -> (u128, u128) {
    let a = stream.shell.config.count_pm1;
    let (pm1, pm2) = challenge.positions.split_at(a);
    assert!(pm1.windows(2).all(|w| w[0] < w[1]));
    assert!(pm2.windows(2).all(|w| w[0] < w[1]));
    let pm2_slots = pm2
        .iter()
        .map(|&p| {
            assert!(!pm1.contains(&p));
            p - pm1.iter().filter(|&&q| q < p).count() as u32
        })
        .collect::<Vec<_>>();
    let rank = colex_rank(pm1) + stream.shell.pm1_rank_space * colex_rank(&pm2_slots);
    let signs = challenge
        .coeffs
        .iter()
        .enumerate()
        .map(|(t, &c)| {
            assert_eq!(c.unsigned_abs(), if t < a { 1 } else { 2 });
            u128::from(c < 0) << t
        })
        .sum();
    (signs, rank)
}

fn assert_round_trip(
    stream: &SparseChallengeStream,
    message: &[u8],
    challenges: &[SparseChallenge],
) {
    let weight = stream.shell.config.weight();
    for (index, challenge) in challenges.iter().enumerate() {
        challenge.validate_dyn(stream.shell.ring_d).unwrap();
        let start = index * stream.shell.candidate_bits;
        let (signs, rank) = encode(stream, challenge);
        assert_eq!(signs, read_bits(message, start, weight as u32).unwrap());
        assert_eq!(
            rank,
            read_bits(message, start + weight, stream.shell.rank_bits).unwrap()
        );
    }
}

#[test]
fn unfiltered_widths_match_the_production_ladder() {
    for (d, config, bits) in production_families() {
        let stream = layout(d, &config, None, 3).unwrap();
        assert_eq!(stream.set_bits(), bits, "d = {d}");
        assert_eq!(stream.candidate_bits(), bits as usize, "d = {d}");
        assert_eq!(stream.candidates(), 3);
        assert_eq!(stream.message_len(), (3 * bits as usize).div_ceil(8));
    }
}

#[test]
fn operator_norm_pools_match_the_specification() {
    let [(d64, c64, r64), (d128, c128, r128)] = operator_norm_families();
    for (count, pool, bytes) in [
        (4, 297, 4864),
        (6, 327, 5355),
        (8, 357, 5846),
        (64, 942, 15426),
    ] {
        let stream = layout(d64, &c64, Some(r64), count).unwrap();
        assert_eq!(stream.candidate_bits(), 131);
        assert_eq!(stream.set_bits(), 128);
        assert_eq!(stream.candidates(), pool, "D64 count {count}");
        assert_eq!(stream.message_len(), bytes, "D64 count {count}");
    }
    for (count, pool, bytes) in [(8, 172, 2795), (64, 567, 9214)] {
        let stream = layout(d128, &c128, Some(r128), count).unwrap();
        assert_eq!(stream.candidate_bits(), 130);
        assert_eq!(stream.set_bits(), 128);
        assert_eq!(stream.candidates(), pool, "D128 count {count}");
        assert_eq!(stream.message_len(), bytes, "D128 count {count}");
    }
}

#[test]
fn unfiltered_decode_inverts_the_rank() {
    let mut rng = SplitMix(1);
    for (d, config, _) in production_families() {
        let stream = layout(d, &config, None, 37).unwrap();
        for _ in 0..8 {
            let message = rng.message(stream.message_len());
            let challenges = stream.decode(&message).unwrap().unwrap();
            assert_eq!(challenges.len(), 37);
            assert_round_trip(&stream, &message, &challenges);
        }
    }
}

#[test]
fn unfiltered_extreme_messages_decode() {
    for (d, config, _) in production_families() {
        let stream = layout(d, &config, None, 5).unwrap();
        for fill in [0x00, 0xff] {
            let message = vec![fill; stream.message_len()];
            let challenges = stream.decode(&message).unwrap().unwrap();
            assert_round_trip(&stream, &message, &challenges);
        }
    }
}

#[test]
fn decode_rejects_a_wrong_length() {
    let config = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
    let stream = layout(64, &config, None, 4).unwrap();
    assert!(stream.decode(&vec![0; stream.message_len() - 1]).is_err());
    assert!(stream.decode(&vec![0; stream.message_len() + 1]).is_err());
}

#[test]
fn replacing_one_candidate_changes_only_its_challenge() {
    let config = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
    let stream = layout(64, &config, None, 12).unwrap();
    let mut message = SplitMix(2).message(stream.message_len());
    let before = stream.decode(&message).unwrap().unwrap();
    // Candidate 7 occupies bits [7 * 128, 8 * 128), bytes 112..128.
    for byte in &mut message[112..128] {
        *byte ^= 0x5a;
    }
    let after = stream.decode(&message).unwrap().unwrap();
    for index in 0..12 {
        assert_eq!(before[index] == after[index], index != 7, "index {index}");
    }
}

#[test]
fn operator_norm_decode_accepts_only_predicate_challenges() {
    let mut rng = SplitMix(3);
    // 512 challenges take the chunked parallel path when it is enabled.
    for ((d, config, rejection), count) in operator_norm_families()
        .into_iter()
        .flat_map(|family| [(family, 6), (family, 512)])
    {
        let stream = layout(d, &config, Some(rejection), count).unwrap();
        let ChallengePolicy::OperatorNorm { threshold, table } =
            ChallengePolicy::new(d, &config, Some(rejection)).unwrap()
        else {
            panic!("operator-norm policy expected");
        };
        for _ in 0..4 {
            let message = rng.message(stream.message_len());
            let challenges = stream.decode(&message).unwrap().unwrap();
            assert_eq!(challenges.len(), count);
            let binomials = BinomialTable::new(d, config.count_pm1.max(config.count_pm2)).unwrap();
            let decoder = CandidateDecoder {
                stream: &stream,
                message: &message,
                binomials: &binomials,
            };
            let expected = (0..stream.candidates())
                .filter_map(|index| decoder.candidate(index).unwrap())
                .filter(|c| {
                    table
                        .accept_strict_parts(&c.positions, &c.coeffs, u64::from(threshold))
                        .unwrap()
                })
                .take(count)
                .collect::<Vec<_>>();
            assert_eq!(challenges, expected);
        }
    }
}

#[test]
fn operator_norm_pool_runs_out_closed() {
    for (d, config, rejection) in operator_norm_families() {
        let stream = layout(d, &config, Some(rejection), 4).unwrap();
        // Every rank field is all ones, above the shell's rank space.
        let message = vec![0xff; stream.message_len()];
        assert_eq!(stream.decode(&message).unwrap(), None);
    }
}

#[test]
fn large_unfiltered_decode_matches_per_candidate_decode() {
    let config = SparseChallengeConfig::production_for_ring_dim(128).unwrap();
    let stream = layout(128, &config, None, 2048).unwrap();
    let message = SplitMix(4).message(stream.message_len());
    let challenges = stream.decode(&message).unwrap().unwrap();
    let binomials = BinomialTable::new(128, 31).unwrap();
    let decoder = CandidateDecoder {
        stream: &stream,
        message: &message,
        binomials: &binomials,
    };
    for (index, challenge) in challenges.iter().enumerate() {
        assert_eq!(Some(challenge.clone()), decoder.candidate(index).unwrap());
    }
}

#[test]
fn unsupported_families_are_setup_errors() {
    let config = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
    assert!(layout(64, &config, None, 0).is_err());
    assert!(layout(4096, &SparseChallengeConfig::pm1_only(8), None, 1).is_err());
    assert!(layout(
        64,
        &config,
        Some(OperatorNormRejection::D64_SELECTIVE_L2),
        1
    )
    .is_err());
}

/// The pool search the closed form replaced: doubling and bisection for the
/// least Chernoff pool, then a linear scan for the least union-bound pool.
fn pool_len_by_scan(count: usize, rates: &PoolRates) -> Result<usize, AkitaError> {
    let overflow = || AkitaError::InvalidSetup("operator-norm pool size overflow".into());
    let (u, v) = rates.accept;
    let (r_num, r_den) = rates.fail_bits;
    let w = u128::try_from(count).map_err(|_| overflow())?;
    let chernoff = |b: u128| -> Option<bool> {
        let bu = b.checked_mul(u)?;
        let wv = w.checked_mul(v)?;
        if bu <= wv {
            return Some(false);
        }
        let gap = bu - wv;
        Some(
            gap.checked_mul(gap)?
                >= POOL_CHERNOFF_SCALE
                    .checked_mul(b)?
                    .checked_mul(u)?
                    .checked_mul(v)?,
        )
    };
    let mut high = w.max(1);
    while !chernoff(high).ok_or_else(overflow)? {
        high = high
            .checked_mul(2)
            .filter(|&b| b <= MAX_POOL_CANDIDATES)
            .ok_or_else(overflow)?;
    }
    let mut low = w;
    while low < high {
        let mid = low + (high - low) / 2;
        if chernoff(mid).ok_or_else(overflow)? {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    let union = |b: u128| -> Option<bool> {
        let tail = b.checked_sub(w)?.checked_add(1)?;
        let bitlen = u128::from(u128::BITS - b.leading_zeros());
        let rhs = w
            .checked_sub(1)?
            .checked_mul(bitlen)?
            .checked_add(POOL_FAILURE_BITS)?;
        Some(tail.checked_mul(r_num)? >= rhs.checked_mul(r_den)?)
    };
    let pool = (w..high)
        .find(|&b| union(b).unwrap_or(false))
        .unwrap_or(high);
    usize::try_from(pool).map_err(|_| overflow())
}

fn production_pool_rates() -> [PoolRates; 2] {
    operator_norm_families().map(|(_, _, rejection)| operator_norm_rates(rejection).unwrap().0)
}

fn assert_pool_matches_scan(rates: &PoolRates, count: usize) {
    assert_eq!(
        pool_len(count, rates).ok(),
        pool_len_by_scan(count, rates).ok(),
        "count {count}"
    );
}

#[test]
fn closed_form_pool_matches_the_scan() {
    for rates in production_pool_rates() {
        for count in 0..=4096 {
            assert_pool_matches_scan(&rates, count);
        }
        for bits in 12..=20 {
            for count in [(1 << bits) - 1, 1 << bits, (1 << bits) + 1, 3 << (bits - 2)] {
                assert_pool_matches_scan(&rates, count);
            }
        }
        // Both searches reject a count whose doubled pool passes the cap.
        let (mut largest, mut rejected) = (1usize, 1usize << 32);
        while rejected - largest > 1 {
            let mid = largest + (rejected - largest) / 2;
            if pool_len(mid, &rates).is_ok() {
                largest = mid;
            } else {
                rejected = mid;
            }
        }
        assert!(pool_len(largest, &rates).unwrap() as u128 <= MAX_POOL_CANDIDATES);
        for count in [largest + 1, largest + 2, 1 << 32, usize::MAX] {
            assert!(pool_len(count, &rates).is_err(), "count {count}");
            assert_pool_matches_scan(&rates, count);
        }
    }
}

#[test]
#[ignore = "exhaustive over 2^16 counts; about 16 s in release"]
fn closed_form_pool_matches_the_scan_exhaustively() {
    for rates in production_pool_rates() {
        for count in 0..=1 << 16 {
            assert_pool_matches_scan(&rates, count);
        }
    }
}
