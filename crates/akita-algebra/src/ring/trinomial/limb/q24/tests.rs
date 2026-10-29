use super::super::backend::Backend;
use super::super::{TrinomialLimbAccumulator, TrinomialLimbDomain, TrinomialLimbSlots, DEGREE};
use super::*;
use crate::ring::trinomial::image::{IMAGE_MAX, IMAGE_MIN, Q24_CENTERED_BOUND};

const P: i64 = Q24 as i64;
const H: i64 = Q24_CENTERED_BOUND as i64;
const BLOCK: usize = Q24_REDUCTION_TERMS as usize;
// Computed independently in Python with exact integers for the vector in
// `q24_known_answer_vector`: slots 0, 1, 647 and sum of (i + 1) * slot i.
const KNOWN_ANSWER: (i32, i32, i32, i128) = (-1_895_504, 1_636_398, -262_794, 11_283_659_138);

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn center(value: i128) -> i32 {
    let residue = value.rem_euclid(i128::from(P));
    (if residue > i128::from(H) {
        residue - i128::from(P)
    } else {
        residue
    }) as i32
}

fn domain(backend: Option<Backend>) -> TrinomialLimbDomain {
    let mut domain = TrinomialLimbDomain::new(Q24).unwrap();
    if let Some(backend) = backend {
        domain.backend = backend;
    }
    domain
}

fn slots(values: [i32; DEGREE]) -> TrinomialLimbSlots {
    TrinomialLimbSlots { values, prime: Q24 }
}

fn pack(slots: &TrinomialLimbSlots) -> TrinomialQ24PackedSlots {
    TrinomialQ24PackedSlots::from_slots(slots).unwrap()
}

/// One q24 accumulator per available kernel, each forced to that kernel.
fn accumulators() -> Vec<TrinomialQ24Accumulator> {
    let domain = domain(None);
    available_q24_backends()
        .into_iter()
        .map(|backend| {
            let mut accumulator = TrinomialQ24Accumulator::new(&domain).unwrap();
            accumulator.backend = backend;
            accumulator
        })
        .collect()
}

/// The existing trinomial limb accumulator on its portable backend.
fn reference() -> TrinomialLimbAccumulator {
    TrinomialLimbAccumulator::new(&domain(Some(Backend::Scalar)))
}

fn finish_q24(accumulator: &TrinomialQ24Accumulator) -> [i32; DEGREE] {
    let mut out = slots([0; DEGREE]);
    accumulator.clone().finish(&mut out).unwrap();
    out.values
}

fn finish_reference(accumulator: &TrinomialLimbAccumulator) -> [i32; DEGREE] {
    let mut out = slots([0; DEGREE]);
    accumulator.clone().finish(&mut out).unwrap();
    out.values
}

fn assert_lane_bound(accumulator: &TrinomialQ24Accumulator) {
    let bound = (P as u128 - 1) + accumulator.terms as u128 * MAX_PRODUCT;
    assert!(bound < 1 << 63);
    assert!(accumulator.sums.iter().all(|&sum| u128::from(sum) <= bound));
}

#[test]
fn q24_kernels_follow_existing_backend_detection() {
    let backends = available_q24_backends();
    eprintln!("q24 kernels available: {backends:?}");
    let scalar = TrinomialQ24Accumulator::new(&domain(Some(Backend::Scalar))).unwrap();
    assert_eq!(scalar.backend, Q24Backend::Scalar);
    let selected = TrinomialQ24Accumulator::new(&domain(None)).unwrap();
    eprintln!("q24 kernel selected: {:?}", selected.backend);
    assert!(backends.contains(&selected.backend));
    assert_eq!(selected.backend, q24_backend(domain(None).backend));
}

#[test]
fn q24_pack_uses_centered_image_words_and_round_trips() {
    let mut seed = 0x51d7_348e_2c6a_90b1;
    let p = P as i32;
    let h = H as i32;
    let edges = [0, 1, -1, h, -h, h + 1, -h - 1, p - 1, 1 - p, p - 2, 2 - p];
    for round in 0..64 {
        let values: [i32; DEGREE] = std::array::from_fn(|i| {
            if round == 0 {
                edges[i % edges.len()]
            } else {
                (next(&mut seed) % (2 * P as u64 - 1)) as i32 - (p - 1)
            }
        });
        let packed = pack(&slots(values));
        for (i, chunk) in packed.as_bytes().chunks_exact(3).enumerate() {
            let expected = Q24Image::from_centered(center(i128::from(values[i]))).unwrap();
            assert_eq!(chunk, expected.bytes().as_slice());
        }
        let mut out = slots([7; DEGREE]);
        packed.to_slots(&mut out).unwrap();
        let canonical: [i32; DEGREE] = std::array::from_fn(|i| center(i128::from(values[i])));
        assert_eq!(out.values, canonical);
        assert_eq!(pack(&out), packed);
    }
}

#[test]
fn q24_rejects_other_primes_and_bad_batches_without_changes() {
    let other = TrinomialLimbDomain::new(33_568_993).unwrap();
    assert!(TrinomialQ24Accumulator::new(&other).is_err());
    let foreign = other.zero_slots();
    assert!(TrinomialQ24PackedSlots::from_slots(&foreign).is_err());
    let packed = pack(&slots([3; DEGREE]));
    let mut foreign_out = other.zero_slots();
    assert!(packed.to_slots(&mut foreign_out).is_err());
    assert_eq!(foreign_out.values, [0; DEGREE]);
    for mut accumulator in accumulators() {
        accumulator.add_product(&packed, &packed);
        let before = (accumulator.sums, accumulator.terms);
        let one = std::slice::from_ref(&packed);
        assert!(accumulator
            .add_product_batch(&[packed.clone(), packed.clone()], one)
            .is_err());
        assert_eq!((accumulator.sums, accumulator.terms), before);
        assert!(accumulator.finish(&mut foreign_out).is_err());
        assert_eq!(foreign_out.values, [0; DEGREE]);
        assert_eq!((accumulator.sums, accumulator.terms), before);
        let mut out = slots([0; DEGREE]);
        accumulator.finish(&mut out).unwrap();
        assert_eq!(out.values, [9; DEGREE]);
        assert_eq!((accumulator.sums, accumulator.terms), ([0; DEGREE], 0));
    }
}

/// Fixed transformed vectors from both forward paths, cycled through the
/// reduction boundaries and compared with the existing limb accumulator.
#[test]
fn q24_fixed_vectors_match_trinomial_scalar_accumulator() {
    const POOL: usize = 1024;
    const MILESTONES: [usize; 9] = [1, 7, 128, 129, 32767, 32768, 32769, 65536, 65537];
    let mut seed = 0x0a7a_24c0_ffee_2026;
    let forward = [domain(Some(Backend::Scalar)), domain(None)];
    let mut pairs = Vec::new();
    for index in 0..8 {
        let operand = |seed: &mut u64| {
            let coefficients: [i32; DEGREE] =
                std::array::from_fn(|_| center(i128::from(next(seed))));
            let mut out = forward[index % 2].zero_slots();
            forward[index % 2]
                .forward_centered(&coefficients, &mut out)
                .unwrap();
            out
        };
        pairs.push((operand(&mut seed), operand(&mut seed)));
    }
    let lhs: Vec<_> = (0..POOL).map(|t| pairs[t % 8].0.clone()).collect();
    let rhs: Vec<_> = (0..POOL).map(|t| pairs[t % 8].1.clone()).collect();
    let packed_lhs: Vec<_> = lhs.iter().map(pack).collect();
    let packed_rhs: Vec<_> = rhs.iter().map(pack).collect();
    let chunks = [1, 3, 1000, 517, 1024, 77, 2, 640];
    let mut expected = reference();
    let mut actual = accumulators();
    let mut done = 0;
    let mut step = 0;
    for milestone in MILESTONES {
        while done < milestone {
            let offset = done % POOL;
            let count = chunks[step % chunks.len()]
                .min(POOL - offset)
                .min(milestone - done);
            step += 1;
            let range = offset..offset + count;
            let (a, b) = (&lhs[range.clone()], &rhs[range.clone()]);
            let (pa, pb) = (&packed_lhs[range.clone()], &packed_rhs[range]);
            expected.add_product_batch(a, b).unwrap();
            for accumulator in &mut actual {
                if count == 1 {
                    accumulator.add_product(&pa[0], &pb[0]);
                } else {
                    accumulator.add_product_batch(pa, pb).unwrap();
                }
            }
            done += count;
        }
        let want = finish_reference(&expected);
        for accumulator in &actual {
            assert_eq!(accumulator.terms as usize, done % BLOCK);
            assert_lane_bound(accumulator);
            assert_eq!(
                finish_q24(accumulator),
                want,
                "{:?} after {done} products",
                accumulator.backend
            );
        }
    }
}

/// Canonical residue q - 1 maximizes every product; mixed signs check the
/// centered output on both sides of zero.
#[test]
fn q24_extremal_residues_cross_reduction_boundaries() {
    let h = H as i32;
    let lhs_cases = [-1, h, -h, 0, 1, -1];
    let rhs_cases = [-1, -h, h, 1, -1, h];
    let lhs = slots(std::array::from_fn(|i| lhs_cases[i % lhs_cases.len()]));
    let rhs = slots(std::array::from_fn(|i| rhs_cases[i % rhs_cases.len()]));
    let (packed_lhs, packed_rhs) = (pack(&lhs), pack(&rhs));
    let checks = [1, BLOCK - 1, BLOCK, BLOCK + 1, 2 * BLOCK, 2 * BLOCK + 1];
    let mut expected = reference();
    let mut actual = accumulators();
    for terms in 1..=2 * BLOCK + 1 {
        expected.add_product(&lhs, &rhs).unwrap();
        for accumulator in &mut actual {
            accumulator.add_product(&packed_lhs, &packed_rhs);
        }
        if !checks.contains(&terms) {
            continue;
        }
        let exact: [i32; DEGREE] = std::array::from_fn(|i| {
            center(i128::from(lhs.values[i]) * i128::from(rhs.values[i]) * terms as i128)
        });
        assert_eq!(finish_reference(&expected), exact);
        for accumulator in &actual {
            assert_eq!(accumulator.terms as usize, terms % BLOCK);
            assert_lane_bound(accumulator);
            if terms == BLOCK - 1 {
                // The largest raw lane just before a reduction.
                assert_eq!(
                    u128::from(accumulator.sums[0]),
                    (BLOCK as u128 - 1) * MAX_PRODUCT
                );
            }
            assert_eq!(finish_q24(accumulator), exact, "{:?}", accumulator.backend);
        }
    }
}

/// Arbitrary offset words, including the image interval endpoints, decode to
/// noncanonical lazy slots; both accumulators agree on them.
#[test]
fn q24_arbitrary_image_words_match_decoded_slots() {
    let mut seed = 0x9e37_79b9_7f4a_7c15;
    let mut packed = Vec::new();
    for index in 0..4 {
        let bytes: [u8; Q24_PACKED_SLOT_BYTES] = std::array::from_fn(|i| match (index, i % 6) {
            (0, 0..=2) => 0x00,
            (0, _) => 0xff,
            _ => next(&mut seed) as u8,
        });
        packed.push(TrinomialQ24PackedSlots::from_bytes(bytes));
    }
    let decoded: Vec<_> = packed
        .iter()
        .map(|packed| {
            let mut out = slots([0; DEGREE]);
            packed.to_slots(&mut out).unwrap();
            assert!(out
                .values
                .iter()
                .all(|v| (IMAGE_MIN..=IMAGE_MAX).contains(v)));
            out
        })
        .collect();
    assert_eq!(decoded[0].values[0], IMAGE_MIN);
    assert_eq!(decoded[0].values[1], IMAGE_MAX);
    let mut expected = reference();
    let mut actual = accumulators();
    for terms in 1..=BLOCK + 233 {
        let (a, b) = (terms % 4, (terms * 3 + 1) % 4);
        expected.add_product(&decoded[a], &decoded[b]).unwrap();
        for accumulator in &mut actual {
            accumulator.add_product(&packed[a], &packed[b]);
        }
    }
    let want = finish_reference(&expected);
    for accumulator in &actual {
        assert_eq!(finish_q24(accumulator), want, "{:?}", accumulator.backend);
    }
}

/// A known answer computed outside Rust, shared by the existing accumulator
/// and every q24 kernel. Slot `i` of product `t` multiplies
/// `((7919 i^2 + 104729 t + 13) mod q) - (q-1)/2` by
/// `(q-1)/2 - ((65537 i + 31 t^2 + 7) mod q)`, for `t < 40000`.
#[test]
fn q24_known_answer_vector() {
    const PRODUCTS: i64 = 40_000;
    let mut expected = reference();
    let mut actual = accumulators();
    for t in 0..PRODUCTS {
        let lhs = slots(std::array::from_fn(|i| {
            let i = i as i64;
            ((7919 * i * i + 104_729 * t + 13) % P - H) as i32
        }));
        let rhs = slots(std::array::from_fn(|i| {
            let i = i as i64;
            (H - (65_537 * i + 31 * t * t + 7) % P) as i32
        }));
        expected.add_product(&lhs, &rhs).unwrap();
        let (lhs, rhs) = (pack(&lhs), pack(&rhs));
        for accumulator in &mut actual {
            accumulator.add_product(&lhs, &rhs);
        }
    }
    let want = finish_reference(&expected);
    let digest: i128 = want
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as i128 + 1) * i128::from(v))
        .sum();
    assert_eq!(
        (want[0], want[1], want[DEGREE - 1], digest),
        KNOWN_ANSWER,
        "existing scalar accumulator"
    );
    for accumulator in &actual {
        assert_eq!(finish_q24(accumulator), want, "{:?}", accumulator.backend);
    }
}

/// The numbers behind the `2^15` reduction interval, stated in the module
/// documentation: the largest product, the lane bound and its `2^63` margin.
#[test]
fn q24_reduction_interval_bound_is_pinned() {
    // The module's const assertions check the same inequalities at compile time;
    // `black_box` keeps these runtime checks from being constant-folded away.
    let (max_product, lane_bound) = std::hint::black_box((MAX_PRODUCT, LANE_BOUND));
    let low = std::hint::black_box(P as u128 - 1);
    assert_eq!(Q24, 14_253_409);
    assert_eq!(Q24_REDUCTION_TERMS, 1 << 15);
    assert_eq!(max_product, 203_159_639_614_464);
    assert!(max_product < 1 << 48);
    assert_eq!(lane_bound, 6_657_135_070_901_009_760);
    assert_eq!(lane_bound, low + (1 << 15) * max_product);
    assert!(lane_bound < 1 << 63);
    // One more interval of the same length would cross 2^63.
    assert_eq!(low + (1 << 16) * max_product, 13_314_270_141_787_766_112);
    assert!(low + (1 << 16) * max_product >= 1 << 63);
    // The largest exact interval below 2^63.
    assert!(low + 45_399 * max_product < 1 << 63);
    assert!(low + 45_400 * max_product >= 1 << 63);
}

/// Batch lengths around the IFMA (8 slots) and NEON (4 slots) vector widths,
/// including the empty batch, on zero, canonical extremes and `q - 1`.
#[test]
fn q24_batch_lengths_and_extremes_match_reference() {
    const LENGTHS: [usize; 16] = [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64];
    let h = H as i32;
    // Centered -1 is the canonical residue q - 1.
    let values = [0, 1, -1, h, -h, h - 1, 1 - h];
    let mut seed = 0x3b1d_0f24_8c55_e017;
    let pool: Vec<TrinomialLimbSlots> = (0..64)
        .map(|index| {
            slots(std::array::from_fn(|i| match index % 4 {
                0 => 0,
                1 => values[i % values.len()],
                2 => values[(i + index) % values.len()],
                _ => center(i128::from(next(&mut seed))),
            }))
        })
        .collect();
    let packed: Vec<_> = pool.iter().map(pack).collect();
    for length in LENGTHS {
        let (lhs, rhs) = (&pool[..length], &pool[64 - length..]);
        let (packed_lhs, packed_rhs) = (&packed[..length], &packed[64 - length..]);
        let mut expected = reference();
        expected.add_product_batch(lhs, rhs).unwrap();
        let want = finish_reference(&expected);
        for mut accumulator in accumulators() {
            accumulator
                .add_product_batch(packed_lhs, packed_rhs)
                .unwrap();
            assert_eq!(accumulator.terms as usize, length);
            if length == 0 {
                assert_eq!(accumulator.sums, [0; DEGREE]);
            }
            assert_eq!(
                finish_q24(&accumulator),
                want,
                "{:?} length {length}",
                accumulator.backend
            );
        }
    }
}
