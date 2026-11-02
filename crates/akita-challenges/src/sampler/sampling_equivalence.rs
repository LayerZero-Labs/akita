//! Independent, deliberately simple byte-stream oracle for sampler refactors.
use super::{signed_sparse, IndexedXofPrefix, SignedSparseScratch, XofCursor};
use crate::{SparseChallenge, SparseChallengeCoefficients, SparseChallengePositions};

// Preserve the base sampler's bits-based rejection independently of the
// optimized cursor helpers, including the exact number of bytes per attempt.
fn reference_usize_mod(cursor: &mut XofCursor, modulus: usize) -> usize {
    assert!(modulus > 0);
    if modulus == 1 {
        return 0;
    }
    let bits = usize::BITS - (modulus - 1).leading_zeros();
    let width = if bits <= 8 {
        1
    } else if bits <= 16 {
        2
    } else {
        4
    };
    let mask = (1usize << bits) - 1;
    loop {
        let mut bytes = [0u8; 4];
        cursor.fill_bytes(&mut bytes[..width]);
        let value = (u32::from_le_bytes(bytes) as usize) & mask;
        if value < modulus {
            return value;
        }
    }
}

fn reference(cursor: &mut XofCursor, d: usize, pm1: usize, weight: usize) -> SparseChallenge {
    let mut permutation: Vec<u32> = (0..d as u32).collect();
    let mut positions = SparseChallengePositions::new();
    for i in 0..weight {
        let j = i + reference_usize_mod(cursor, d - i);
        permutation.swap(i, j);
        positions.push(permutation[i]);
    }
    let mut signs = vec![0; weight];
    cursor.fill_bytes(&mut signs);
    let coeffs: SparseChallengeCoefficients = signs
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let magnitude = if i < pm1 { 1 } else { 2 };
            if b & 1 == 0 {
                magnitude
            } else {
                -magnitude
            }
        })
        .collect();
    SparseChallenge { positions, coeffs }
}

fn compare(d: usize, pm1: usize, weight: usize, draw: impl Fn(&mut XofCursor) -> SparseChallenge) {
    for seed in 0..8u8 {
        let prefix = IndexedXofPrefix::new(&[seed; 32]).unwrap();
        for coordinate in [0, 1, 7, 127, u64::MAX] {
            let mut expected = XofCursor::from_indexed_prefix(&prefix, coordinate);
            let mut actual = XofCursor::from_indexed_prefix(&prefix, coordinate);
            let result = draw(&mut actual);
            assert_eq!(
                result,
                reference(&mut expected, d, pm1, weight),
                "D={d} w={weight}"
            );
            result.validate_dyn(d).unwrap();
            let (mut a, mut b) = ([0; 300], [0; 300]);
            actual.fill_bytes(&mut a);
            expected.fill_bytes(&mut b);
            assert_eq!(a, b, "XOF suffix D={d} w={weight}");
        }
    }
}

#[test]
fn dynamic_sampler_matches_reference_at_mask_sign_and_sparse_boundaries() {
    for d in [
        8usize, 16, 32, 63, 64, 65, 127, 128, 129, 255, 256, 257, 512, 1024, 2047, 2048,
    ] {
        let mut weights = vec![0, 1, d / 64, d / 64 + 1, d / 2, d - 1, d];
        weights.extend(
            [31, 32, 41, 42, 48, 49, 63, 64, 65]
                .into_iter()
                .filter(|&w| w <= d),
        );
        weights.sort_unstable();
        weights.dedup();
        for weight in weights {
            for pm1 in [0, weight / 2, weight] {
                compare(d, pm1, weight, |cursor| {
                    let mut scratch = SignedSparseScratch::new(pm1, weight - pm1);
                    scratch.sample(cursor, d, pm1, weight - pm1).unwrap();
                    scratch.take_challenge()
                });
            }
        }
    }
}

#[test]
fn fixed_sampler_matches_reference_across_dimensions_and_weights() {
    macro_rules! cases {
        ($(($d:literal, $pm1:expr, $weight:expr)),+ $(,)?) => { $(
            compare($d, $pm1, $weight, |cursor| {
                signed_sparse::sample_fixed::<$d, { $pm1 }, { $weight }>(cursor)
            });
        )+ };
    }
    cases! {
        (64, 0, 0), (64, 1, 1), (64, 31, 31), (64, 16, 32),
        (64, crate::D64_PRODUCTION_PM1_COUNT, crate::D64_PRODUCTION_PM1_COUNT + crate::D64_PRODUCTION_PM2_COUNT),
        (64, 31, 42), (64, 24, 48), (64, 24, 49), (64, 32, 64),
        (128, 31, 31), (256, 23, 23), (512, 19, 19), (1024, 16, 16),
        (2048, 14, 14), (2048, 15, 31), (2048, 16, 32), (2048, 1024, 2048),
    }
}

#[test]
fn rejection_draws_match_the_reference_candidate_stream() {
    use super::{IndexedChallengeWorker, IndexedSamplingPolicy, MAX_OP_NORM_ATTEMPTS};
    use crate::{
        OperatorNormRejection, D128_SELECTIVE_L2_CHALLENGE_CONFIG,
        D64_SELECTIVE_L2_CHALLENGE_CONFIG,
    };
    for (d, config, rejection) in [
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
    ] {
        let policy = IndexedSamplingPolicy::new(d, &config, Some(rejection)).unwrap();
        let IndexedSamplingPolicy::OperatorNorm { threshold, table } = &policy else {
            unreachable!()
        };
        for seed in 0..8u8 {
            let prefix = IndexedXofPrefix::new(&[seed; 32]).unwrap();
            for index in [0, 1, 7, 127] {
                let mut worker = IndexedChallengeWorker::new(&prefix, &config, &policy);
                let actual = worker.sample(index, d, &config).unwrap();
                let mut cursor = XofCursor::from_indexed_prefix(&prefix, index);
                let mut accepted = None;
                for _ in 0..MAX_OP_NORM_ATTEMPTS {
                    let candidate = reference(&mut cursor, d, config.count_pm1, config.weight());
                    if table
                        .accept_strict_parts(
                            &candidate.positions,
                            &candidate.coeffs,
                            u64::from(*threshold),
                        )
                        .unwrap()
                    {
                        accepted = Some(candidate);
                        break;
                    }
                }
                assert_eq!(Some(actual), accepted);
                let (mut a, mut b) = ([0; 300], [0; 300]);
                worker.cursor.fill_bytes(&mut a);
                cursor.fill_bytes(&mut b);
                assert_eq!(a, b, "rejection stream D={d} seed={seed} index={index}");
            }
        }
    }
}
