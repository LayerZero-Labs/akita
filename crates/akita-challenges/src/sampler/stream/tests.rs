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

/// Production unfiltered families with their priced set exponent and
/// candidate width `w + min(ceil(log2 R) + 16, 128)`.
fn production_families() -> Vec<(usize, SparseChallengeConfig, u32, usize)> {
    [
        (64, 128, 145),
        (128, 129, 146),
        (256, 131, 148),
        (512, 132, 147),
        (1024, 131, 144),
        (2048, 131, 142),
    ]
    .into_iter()
    .map(|(d, bits, width)| {
        (
            d,
            SparseChallengeConfig::production_for_ring_dim(d).unwrap(),
            bits,
            width,
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

/// Re-encode a decoded challenge as its `(signs, rank)` pair.
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

/// The `(signs, rank)` pairs of the message's candidates whose rank words are
/// accepted, in message order, read candidate by candidate. The rank map is
/// the production `lemire_rank`; `lemire_rank_matches_its_definition` checks
/// it independently at full width.
fn accepted_draws(stream: &SparseChallengeStream, message: &[u8]) -> Vec<(u128, u128)> {
    let shell = &stream.shell;
    let weight = shell.config.weight();
    (0..stream.candidates())
        .filter_map(|index| {
            let start = index * shell.candidate_bits;
            let signs = read_bits(message, start, weight as u32).unwrap();
            let word = read_bits(message, start + weight, shell.rank_bits).unwrap();
            lemire_rank(word, shell.rank_space, shell.rank_bits, shell.reject_below)
                .map(|rank| (signs, rank))
        })
        .collect()
}

fn assert_round_trip(
    stream: &SparseChallengeStream,
    message: &[u8],
    challenges: &[SparseChallenge],
) {
    let draws = accepted_draws(stream, message);
    assert!(draws.len() >= challenges.len());
    for (challenge, draw) in challenges.iter().zip(draws) {
        challenge.validate_dyn(stream.shell.ring_d).unwrap();
        assert_eq!(encode(stream, challenge), draw);
    }
}

/// Write the low `width` bits of `value` at bit `offset`.
fn write_bits(message: &mut [u8], offset: usize, width: u32, value: u128) {
    for t in 0..width as usize {
        let i = offset + t;
        let bit = ((value >> t) & 1) as u8;
        message[i / 8] = (message[i / 8] & !(1 << (i % 8))) | (bit << (i % 8));
    }
}

#[test]
fn unfiltered_widths_match_the_production_ladder() {
    for (d, config, bits, width) in production_families() {
        let shell = SparseChallengeShell::new(d, &config, None).unwrap();
        assert_eq!(shell.set_bits(), bits, "d = {d}");
        assert_eq!(shell.candidate_bits, width, "d = {d}");
        assert!(shell.reject_below > 0, "d = {d}");
        assert!(shell.rank_bits <= u128::BITS);
    }
    // Pools from the specification's exact rule.
    let config = SparseChallengeConfig::production_for_ring_dim(128).unwrap();
    for (count, pool, bytes) in [(3, 13, 240), (2048, 2088, 38112)] {
        let stream = layout(128, &config, None, count).unwrap();
        assert_eq!(stream.candidates(), pool, "D128 count {count}");
        assert_eq!(stream.message_len(), bytes, "D128 count {count}");
    }
    let config = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
    for (count, pool, bytes) in [
        (9, 20, 368),
        (30, 43, 784),
        (79, 93, 1688),
        (150, 166, 3016),
        (256, 273, 4952),
    ] {
        let stream = layout(64, &config, None, count).unwrap();
        assert_eq!(stream.candidates(), pool, "D64 count {count}");
        assert_eq!(stream.message_len(), bytes, "D64 count {count}");
    }
}

#[test]
fn operator_norm_pools_match_the_specification() {
    let [(d64, c64, r64), (d128, c128, r128)] = operator_norm_families();
    for (count, pool, bytes) in [
        (4, 124, 2280),
        (6, 139, 2560),
        (8, 151, 2776),
        (64, 524, 9632),
    ] {
        let stream = layout(d64, &c64, Some(r64), count).unwrap();
        assert_eq!(stream.candidate_bits(), 147);
        assert_eq!(stream.set_bits(), 128);
        assert_eq!(stream.candidates(), pool, "D64 count {count}");
        assert_eq!(stream.message_len(), bytes, "D64 count {count}");
    }
    for (count, pool, bytes) in [(8, 104, 1904), (64, 397, 7248)] {
        let stream = layout(d128, &c128, Some(r128), count).unwrap();
        assert_eq!(stream.candidate_bits(), 146);
        assert_eq!(stream.set_bits(), 128);
        assert_eq!(stream.candidates(), pool, "D128 count {count}");
        assert_eq!(stream.message_len(), bytes, "D128 count {count}");
    }
}

#[test]
fn lemire_map_is_exactly_uniform_on_small_spaces() {
    for rank_space in [1u128, 2, 3, 5, 7, 12, 100, 255, 256, 1000] {
        let ceil = u128::BITS - (rank_space - 1).leading_zeros();
        for m in ceil.max(1)..=(ceil + 5).min(14) {
            let t = (1u128 << m) % rank_space;
            let mut hits = vec![0u128; rank_space as usize];
            let mut rejected = 0u128;
            for x in 0..1u128 << m {
                match lemire_rank(x, rank_space, m, t) {
                    Some(rank) => {
                        assert_eq!(rank, (x * rank_space) >> m);
                        hits[rank as usize] += 1;
                    }
                    None => rejected += 1,
                }
            }
            assert_eq!(rejected, t, "R = {rank_space}, m = {m}");
            assert!(
                hits.iter().all(|&h| h == (1u128 << m) / rank_space),
                "R = {rank_space}, m = {m}"
            );
        }
    }
}

#[test]
fn widening_multiply_is_exact() {
    assert_eq!(widening_mul(u128::MAX, u128::MAX), (u128::MAX - 1, 1));
    assert_eq!(widening_mul(u128::MAX, 1), (0, u128::MAX));
    assert_eq!(widening_mul(1 << 64, 1 << 64), (1, 0));
    let mut rng = SplitMix(9);
    let primes = [18_446_744_073_709_551_557u128, 18_446_744_073_709_551_533];
    for _ in 0..10_000 {
        let x = u128::from(rng.next()) << 64 | u128::from(rng.next());
        let y = u128::from(rng.next()) << (rng.next() % 65) | u128::from(rng.next());
        let (high, low) = widening_mul(x, y);
        assert_eq!(low, x.wrapping_mul(y));
        for p in primes {
            let two_128 = (u128::MAX % p + 1) % p;
            let lhs = ((high % p) * two_128 % p + low % p) % p;
            assert_eq!(lhs, (x % p) * (y % p) % p);
        }
    }
}

#[test]
fn lemire_rank_matches_its_definition() {
    // `x R = hi 2^m + lo`: `lo` is `x R mod 2^m`, computed here by wrapping
    // multiplication, and `hi` is checked by its congruence modulo two primes
    // near 2^64, independently of `widening_mul` and the shifts.
    let primes = [18_446_744_073_709_551_557u128, 18_446_744_073_709_551_533];
    let mut rng = SplitMix(11);
    let mut spaces: Vec<(u128, u32)> = production_families()
        .into_iter()
        .map(|(d, config, _, _)| {
            let shell = layout(d, &config, None, 1).unwrap().shell;
            (shell.rank_space, shell.rank_bits)
        })
        .collect();
    for m in [63, 64, 65, 127, 128] {
        let top = if m == u128::BITS {
            u128::MAX
        } else {
            (1u128 << m) - 1
        };
        spaces.push((top >> 9 | 1, m));
        spaces.push((top / 3 + 7, m));
    }
    for (rank_space, m) in spaces {
        let mask = if m == u128::BITS {
            u128::MAX
        } else {
            (1u128 << m) - 1
        };
        let threshold = if m == u128::BITS {
            (u128::MAX % rank_space + 1) % rank_space
        } else {
            (1u128 << m) % rank_space
        };
        let mut words = vec![0, 1, mask, mask - 1, mask >> 1];
        words.extend(
            (0..2_000).map(|_| (u128::from(rng.next()) << 64 | u128::from(rng.next())) & mask),
        );
        for x in words {
            let low = x.wrapping_mul(rank_space) & mask;
            let got = lemire_rank(x, rank_space, m, threshold);
            assert_eq!(
                got.is_some(),
                low >= threshold,
                "R={rank_space} m={m} x={x}"
            );
            // Rejected words have no rank; recompute the quotient for them too.
            let high = match got {
                Some(high) => high,
                None => lemire_rank(x, rank_space, m, 0).unwrap(),
            };
            assert!(high < rank_space);
            for p in primes {
                let two_m = if m == u128::BITS {
                    (u128::MAX % p + 1) % p
                } else {
                    (1u128 << m) % p
                };
                let lhs = ((high % p) * two_m % p + low % p) % p;
                assert_eq!(
                    lhs,
                    (x % p) * (rank_space % p) % p,
                    "R={rank_space} m={m} x={x}"
                );
            }
        }
    }
}

#[test]
fn unfiltered_decode_inverts_the_rank() {
    let mut rng = SplitMix(1);
    for (d, config, _, _) in production_families() {
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
fn edge_ranks_decode() {
    for (d, config, _, _) in production_families() {
        let stream = layout(d, &config, None, 2).unwrap();
        let shell = stream.shell;
        let weight = shell.config.weight();
        let mut message = SplitMix(7).message(stream.message_len());
        // Word 1 maps to rank 0 and the all-ones word to rank R - 1.
        let top = if shell.rank_bits == u128::BITS {
            u128::MAX
        } else {
            (1u128 << shell.rank_bits) - 1
        };
        write_bits(&mut message, weight, shell.rank_bits, 1);
        write_bits(
            &mut message,
            shell.candidate_bits + weight,
            shell.rank_bits,
            top,
        );
        let challenges = stream.decode(&message).unwrap().unwrap();
        assert_eq!(encode(&stream, &challenges[0]).1, 0, "d = {d}");
        assert_eq!(
            encode(&stream, &challenges[1]).1,
            shell.rank_space - 1,
            "d = {d}"
        );
        assert_round_trip(&stream, &message, &challenges);
    }
}

#[test]
fn unfiltered_decode_reaches_the_whole_shell() {
    // A `2^K` prefix of the colexicographic shell never sets the top
    // positions: D128 never used X^127 and D512 never used X^499 or above.
    for (d, count, top) in [(128, 2048, 127), (512, 256, 499)] {
        let config = SparseChallengeConfig::production_for_ring_dim(d).unwrap();
        let stream = layout(d, &config, None, count).unwrap();
        let message = SplitMix(8).message(stream.message_len());
        let challenges = stream.decode(&message).unwrap().unwrap();
        assert!(
            challenges
                .iter()
                .any(|c| c.positions.iter().any(|&p| p >= top)),
            "d = {d}"
        );
    }
}

#[test]
fn rejected_rank_words_run_the_pool_out_closed() {
    // A zero rank word has `lo = 0 < 2^m mod R`, so it is always rejected.
    let unfiltered = production_families()
        .into_iter()
        .map(|(d, config, _, _)| (d, config, None));
    let operator_norm = operator_norm_families()
        .into_iter()
        .map(|(d, config, rejection)| (d, config, Some(rejection)));
    for (d, config, rejection) in unfiltered.chain(operator_norm) {
        for count in [1, 4, 600] {
            let stream = layout(d, &config, rejection, count).unwrap();
            assert!(stream.shell.reject_below > 0);
            let message = vec![0; stream.message_len()];
            assert_eq!(stream.decode(&message).unwrap(), None, "d = {d}");
        }
    }
}

#[test]
fn word_reader_matches_the_little_endian_bit_stream() {
    let message = SplitMix(5).message(40);
    let bit = |i: usize| u128::from((message[i / 8] >> (i % 8)) & 1);
    for offset in 0..=(320 - 128) {
        for width in [0u32, 1, 7, 63, 64, 65, 127, 128] {
            let expected: u128 = (0..width as usize).map(|t| bit(offset + t) << t).sum();
            assert_eq!(
                read_bits(&message, offset, width).unwrap(),
                expected,
                "offset {offset}, width {width}"
            );
        }
    }
    assert!(read_bits(&message, 320 - 64, 65).is_err());
    assert!(read_bits(&message, 0, 129).is_err());
}

#[test]
fn trailing_word_padding_is_ignored() {
    for (d, config, _, _) in production_families() {
        let stream = layout(d, &config, None, 3).unwrap();
        let mut message = SplitMix(6).message(stream.message_len());
        let before = stream.decode(&message).unwrap();
        for i in stream.candidates() * stream.candidate_bits()..8 * message.len() {
            message[i / 8] ^= 1 << (i % 8);
        }
        assert_eq!(stream.decode(&message).unwrap(), before, "d = {d}");
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
    let width = stream.candidate_bits();
    let accepts_first_twelve = |message: &[u8]| {
        let decoder = CandidateDecoder {
            stream: &stream,
            message,
            binomials: &BinomialTable::new(64, 31).unwrap(),
        };
        (0..12).all(|index| decoder.candidate(index).unwrap().is_some())
    };
    let flip = |message: &mut Vec<u8>| {
        for i in 7 * width..8 * width {
            message[i / 8] ^= u8::from(i % 3 == 0) << (i % 8);
        }
    };
    // The first seed whose candidates 0..12 are accepted before and after.
    let (message, flipped) = (0..)
        .map(|seed| {
            let message = SplitMix(seed).message(stream.message_len());
            let mut flipped = message.clone();
            flip(&mut flipped);
            (message, flipped)
        })
        .find(|(m, f)| accepts_first_twelve(m) && accepts_first_twelve(f))
        .unwrap();
    let before = stream.decode(&message).unwrap().unwrap();
    let after = stream.decode(&flipped).unwrap().unwrap();
    for index in 0..12 {
        assert_eq!(before[index] == after[index], index != 7, "index {index}");
    }
}

#[test]
fn operator_norm_decode_accepts_only_predicate_challenges() {
    let mut rng = SplitMix(3);
    for ((d, config, rejection), count) in operator_norm_families()
        .into_iter()
        .flat_map(|family| [(family, 6), (family, 600)])
    {
        let stream = layout(d, &config, Some(rejection), count).unwrap();
        #[cfg(feature = "parallel")]
        assert_eq!(stream.parallel(), count == 600);
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
            let draws = accepted_draws(&stream, &message);
            for challenge in &challenges {
                assert!(draws.contains(&encode(&stream, challenge)));
            }
        }
    }
}

#[test]
fn large_unfiltered_decode_matches_per_candidate_decode() {
    let config = SparseChallengeConfig::production_for_ring_dim(128).unwrap();
    let stream = layout(128, &config, None, 2048).unwrap();
    #[cfg(feature = "parallel")]
    assert!(stream.parallel());
    let message = SplitMix(4).message(stream.message_len());
    let challenges = stream.decode(&message).unwrap().unwrap();
    let binomials = BinomialTable::new(128, 31).unwrap();
    let decoder = CandidateDecoder {
        stream: &stream,
        message: &message,
        binomials: &binomials,
    };
    let expected = (0..stream.candidates())
        .filter_map(|index| decoder.candidate(index).unwrap())
        .take(2048)
        .collect::<Vec<_>>();
    assert_eq!(challenges, expected);
    assert_round_trip(&stream, &message, &challenges);
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
/// least Chernoff pool, then a linear scan for the least union-bound pool
/// `(B - W + 1) r >= 161 + min(W - 1, B - W + 1) bitlen(B)`.
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
            .min(tail)
            .checked_mul(bitlen)?
            .checked_add(POOL_FAILURE_BITS)?;
        Some(tail.checked_mul(r_num)? >= rhs.checked_mul(r_den)?)
    };
    let pool = (w..high)
        .find(|&b| union(b).unwrap_or(false))
        .unwrap_or(high);
    usize::try_from(pool).map_err(|_| overflow())
}

/// Pool rates of every production family: the two operator-norm sizing rates
/// and the exact rank-word rates of the unfiltered ladder.
fn production_pool_rates() -> Vec<PoolRates> {
    operator_norm_families()
        .map(|(_, _, rejection)| operator_norm_rates(rejection).unwrap().0)
        .into_iter()
        .chain(production_families().into_iter().map(|(d, config, _, _)| {
            SparseChallengeShell::new(d, &config, None)
                .unwrap()
                .pool_rates()
                .unwrap()
        }))
        .collect()
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
