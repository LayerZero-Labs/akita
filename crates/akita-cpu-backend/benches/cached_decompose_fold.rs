#![allow(missing_docs)]

use akita_challenges::{SparseChallenge, SparseChallengeConfig};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use jolt_field::solinas::parallel::*;
use std::hint::black_box;

#[allow(dead_code)]
#[path = "../src/sources/poly_helpers/narrow_accum.rs"]
mod narrow_accum;

// Source-included unit tests are not run by Criterion's harness-free binary.
#[allow(dead_code, unused_imports)]
#[path = "../src/sources/poly_helpers/cached_narrow_accum.rs"]
mod cached_narrow_accum;

fn cached_case<const D: usize>(c: &mut Criterion) {
    const POSITIONS: usize = 512;
    const BLOCKS: usize = 32;
    let config = SparseChallengeConfig::production_for_ring_dim(D).unwrap();
    let challenges: Vec<_> = (0..BLOCKS)
        .map(|block| SparseChallenge {
            positions: (0..config.weight())
                .map(|term| ((term * 37 + block * 13) % D) as u32)
                .collect(),
            coeffs: (0..config.weight())
                .map(|term| {
                    let magnitude = if term < config.count_pm1 { 1 } else { 2 };
                    if (term + block).is_multiple_of(2) {
                        magnitude
                    } else {
                        -magnitude
                    }
                })
                .collect(),
        })
        .collect();
    // This entire sequence fits the production narrow accumulator's bound.
    assert!(
        challenges
            .iter()
            .flat_map(|c| &c.coeffs)
            .map(|v| u64::from(v.unsigned_abs()) * 4)
            .sum::<u64>()
            <= i16::MAX as u64
    );
    let planes: Vec<[i8; D]> = (0..POSITIONS * BLOCKS)
        .map(|ring| std::array::from_fn(|i| ((ring * 11 + i * 7) % 8) as i8 - 4))
        .collect();
    let mut output = vec![[0i16; D]; POSITIONS];
    let mut group = c.benchmark_group("cached_decompose_fold");
    group.throughput(Throughput::Elements((planes.len() * D) as u64));
    group.bench_function(format!("d{D}_positions{POSITIONS}_blocks{BLOCKS}"), |b| {
        b.iter(|| {
            output.fill([0; D]);
            cfg_chunks_mut!(output, 16)
                .enumerate()
                .for_each(|(tile, out)| {
                    let mut scratch = [[0i16; D]; 2];
                    for (block, challenge) in challenges.iter().enumerate() {
                        for (local, acc) in out.iter_mut().enumerate() {
                            cached_narrow_accum::sparse_mul_acc(
                                &planes[block * POSITIONS + tile * 16 + local],
                                challenge,
                                acc,
                                &mut scratch,
                            );
                        }
                    }
                });
            black_box(&output);
        })
    });
    group.finish();
}

fn bench_cached(c: &mut Criterion) {
    cached_case::<64>(c);
    cached_case::<128>(c);
    cached_case::<256>(c);
}

criterion_group!(cached, bench_cached);
criterion_main!(cached);
