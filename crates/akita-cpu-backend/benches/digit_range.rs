#![allow(missing_docs)]

//! Ordinary Stage 1 digit-range session for the `binary-and-prime-claims` first
//! suffix fold: 17,269,248 balanced base-64 digits on a 2^25 domain.

use akita_cpu_backend::standalone::digit_range_session;
use criterion::{criterion_group, criterion_main, Criterion};
use jolt_algebra::Prime128Offset275 as F;
use std::hash::{Hash, Hasher};
use std::hint::black_box;

const LIVE_DIGITS: usize = 17_269_248;
const NUM_VARS: usize = 25;
const RING_VARIABLES: usize = 9;

fn digits(basis: usize) -> Vec<i8> {
    let half = (basis / 2) as u64;
    (0..LIVE_DIGITS as u64)
        .map(|index| {
            let mixed = index.wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(29)
                ^ index.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            ((mixed >> 17) % (2 * half)) as i8 - half as i8
        })
        .collect()
}

fn bench_digit_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("digit_range_session");
    group.sample_size(10);
    for basis in [64, 16] {
        let digits = digits(basis);
        // A digest of every round polynomial and claim, to compare two builds.
        let output = digit_range_session::<F>(digits.clone(), basis, NUM_VARS, RING_VARIABLES, 7)
            .expect("session");
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        format!("{output:?}").hash(&mut hasher);
        println!(
            "digit_range basis{basis} values={} digest={:016x}",
            output.len(),
            hasher.finish()
        );
        group.bench_function(format!("basis{basis}_live{LIVE_DIGITS}"), |b| {
            b.iter(|| {
                black_box(
                    digit_range_session::<F>(digits.clone(), basis, NUM_VARS, RING_VARIABLES, 7)
                        .expect("session"),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_digit_range);
criterion_main!(benches);
