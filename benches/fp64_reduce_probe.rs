#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

const P40: u64 = hachi_pcs::algebra::fields::pseudo_mersenne::POW2_OFFSET_MODULUS_40;
const P64: u64 = hachi_pcs::algebra::fields::pseudo_mersenne::POW2_OFFSET_MODULUS_64;
const C40: u64 = (1u64 << 40) - P40; // 195
const C64: u64 = 0u64.wrapping_sub(P64); // 59
const MASK40: u64 = (1u64 << 40) - 1;
const MASK64_U128: u128 = u64::MAX as u128;

#[inline(always)]
fn mul_c40_split(x: u64) -> u64 {
    let c = C40 as u32;
    let x_lo = x as u32;
    let x_hi = (x >> 32) as u32;
    (c as u64 * x_lo as u64).wrapping_add((c as u64 * x_hi as u64) << 32)
}

#[inline(always)]
fn reduce40_split(lo: u64, hi: u64) -> u64 {
    let high = (lo >> 40) | (hi << 24);
    let f1 = (lo & MASK40).wrapping_add(mul_c40_split(high));
    let f2 = (f1 & MASK40).wrapping_add(mul_c40_split(f1 >> 40));
    let reduced = f2.wrapping_sub(P40);
    let borrow = reduced >> 63;
    reduced.wrapping_add(borrow.wrapping_neg() & P40)
}

#[inline(always)]
fn reduce40_direct(lo: u64, hi: u64) -> u64 {
    let high = (lo >> 40) | (hi << 24);
    let f1 = (lo & MASK40).wrapping_add(C40.wrapping_mul(high));
    let f2 = (f1 & MASK40).wrapping_add(C40.wrapping_mul(f1 >> 40));
    let reduced = f2.wrapping_sub(P40);
    let borrow = reduced >> 63;
    reduced.wrapping_add(borrow.wrapping_neg() & P40)
}

#[inline(always)]
fn reduce64(lo: u64, hi: u64) -> u64 {
    let f1 = (lo as u128) + (C64 as u128) * (hi as u128);
    let f2 = (f1 & MASK64_U128) + (C64 as u128) * ((f1 >> 64) as u64 as u128);
    let reduced = f2.wrapping_sub(P64 as u128);
    let borrow = reduced >> 127;
    reduced.wrapping_add(borrow.wrapping_neg() & (P64 as u128)) as u64
}

#[inline(always)]
fn next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn bench_fp64_reduce_probe(c: &mut Criterion) {
    let n = 1 << 13;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;

    let mut pairs40 = Vec::with_capacity(n);
    let mut pairs64 = Vec::with_capacity(n);
    for _ in 0..n {
        let a40 = next_u64(&mut seed) % P40;
        let b40 = next_u64(&mut seed) % P40;
        let x40 = (a40 as u128) * (b40 as u128);
        pairs40.push((x40 as u64, (x40 >> 64) as u64));

        let a64 = next_u64(&mut seed);
        let b64 = next_u64(&mut seed);
        let x64 = (a64 as u128) * (b64 as u128);
        pairs64.push((x64 as u64, (x64 >> 64) as u64));
    }

    for &(lo, hi) in &pairs40 {
        assert_eq!(reduce40_split(lo, hi), reduce40_direct(lo, hi));
    }

    let mut group = c.benchmark_group("fp64_reduce_probe");
    group.throughput(Throughput::Elements(n as u64));

    group.bench_function("reduce40_split", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for &(lo, hi) in black_box(&pairs40) {
                acc ^= reduce40_split(lo, hi);
            }
            black_box(acc)
        })
    });

    group.bench_function("reduce40_direct", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for &(lo, hi) in black_box(&pairs40) {
                acc ^= reduce40_direct(lo, hi);
            }
            black_box(acc)
        })
    });

    group.bench_function("reduce64", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for &(lo, hi) in black_box(&pairs64) {
                acc ^= reduce64(lo, hi);
            }
            black_box(acc)
        })
    });

    group.finish();
}

criterion_group!(fp64_reduce_probe, bench_fp64_reduce_probe);
criterion_main!(fp64_reduce_probe);
