//! Offline microbenchmarks for the opt-in binary challenge families.

use akita_challenges::{
    BinaryChallengeProfile, BinaryChallengeSampler, BinaryScalarRing, FoldDraw,
};
use akita_error::AkitaError;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use shake::digest::{ExtendableOutput, Update, XofReader};
use shake::Shake256;
use std::hint::black_box;

struct BenchmarkDraw;

impl FoldDraw for BenchmarkDraw {
    fn absorb_and_squeeze(&mut self, payload: &[u8]) -> Result<[u8; 32], AkitaError> {
        let mut xof = Shake256::default();
        xof.update(payload);
        let mut root = [0u8; 32];
        xof.finalize_xof().read(&mut root);
        Ok(root)
    }
}

fn bench_binary_challenge(c: &mut Criterion) {
    let mut samplers = [
        (
            "d162_fixed_w47",
            BinaryChallengeSampler::new(
                BinaryChallengeProfile::fixed_weight(BinaryScalarRing::Cyclotomic243, 47).unwrap(),
            ),
        ),
        (
            "d162_bounded_w46",
            BinaryChallengeSampler::new(
                BinaryChallengeProfile::bounded_weight(BinaryScalarRing::Cyclotomic243, 46)
                    .unwrap(),
            ),
        ),
        (
            "d486_fixed_w25",
            BinaryChallengeSampler::new(
                BinaryChallengeProfile::fixed_weight(BinaryScalarRing::Cyclotomic729, 25).unwrap(),
            ),
        ),
    ];
    let mut group = c.benchmark_group("labinius_binary_challenge_batch");
    for count in [1usize, 64, 4096] {
        group.throughput(Throughput::Elements(count as u64));
        for (name, sampler) in &mut samplers {
            group.bench_with_input(BenchmarkId::new(*name, count), &count, |b, &count| {
                b.iter(|| {
                    black_box(
                        sampler
                            .sample_challenges(&mut BenchmarkDraw, b"bench/binary-batch", count)
                            .unwrap(),
                    )
                });
            });
        }
    }
    group.finish();
}

criterion_group!(binary_challenge, bench_binary_challenge);
criterion_main!(binary_challenge);
