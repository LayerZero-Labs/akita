#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hachi_pcs::algebra::fields::fft::{
    field_pow, field_pow_u128, primitive_root_of_unity, rs_extend_fft, SmoothDomain,
};
use hachi_pcs::algebra::{Prime128Offset2355, Prime128OffsetA7F7};
use hachi_pcs::{FieldCore, FieldSampling, FromSmallInt, Invertible};
use rand::{rngs::StdRng, SeedableRng};
use std::fmt::Debug;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

type F = Prime128Offset2355;

const P: u128 = 0xfffffffffffffffffffffffffffff6cd;
const P_MINUS_1: u128 = P - 1;

// p = 2^128 − 2^32 + 22537 (Prime128OffsetA7F7).
const P_B: u128 = 0xffffffffffffffffffffffff00005809;
const P_B_MINUS_1: u128 = P_B - 1;

fn generator() -> F {
    F::from_canonical_u128(2)
}

/// Find an `n`-th root of unity in `F` by scanning small integer bases.
///
/// `g^((p-1)/n)` always satisfies `x^n = 1`, but its exact order can be a
/// proper divisor of `n` if `g` is missing a prime-power factor of `n` in its
/// multiplicative order. This helper tries bases `2, 3, 5, 7, ...` until one
/// lands on an element of exact order `n`.
fn find_nth_root<Fld: FieldCore + FromSmallInt + Invertible + Debug>(
    p_minus_1: u128,
    n: usize,
) -> Fld {
    assert_eq!(
        p_minus_1 % (n as u128),
        0,
        "n={n} must divide p-1={p_minus_1}"
    );
    let exp = p_minus_1 / (n as u128);
    for g_val in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] {
        let candidate = field_pow_u128(Fld::from_u64(g_val), exp);
        if field_pow(candidate, n as u64) != Fld::one() {
            continue;
        }
        let mut ok = true;
        for &q in &[2u64, 3, 5, 7, 11, 13, 17, 19] {
            if (n as u64) % q == 0 && field_pow(candidate, (n as u64) / q) == Fld::one() {
                ok = false;
                break;
            }
        }
        if ok {
            return candidate;
        }
    }
    panic!("no n-th root of unity found for n={n}");
}

fn bench_forward(c: &mut Criterion) {
    let g = generator();
    let mut group = c.benchmark_group("fft_forward");

    // Prime128Offset2355 (mixed radix): N | p − 1 = 2^2 · 3 · 5^2 · 7^2 · …
    for &n in &[300, 1470, 2940, 7350, 14700] {
        if P_MINUS_1 % (n as u128) != 0 {
            continue;
        }
        let omega = primitive_root_of_unity(g, P_MINUS_1, n);
        let domain = SmoothDomain::new(omega, n);
        let mut rng = StdRng::seed_from_u64(0xff00 + n as u64);
        let input: Vec<F> = (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();

        let label = format!("pA/N={n}");
        group.bench_with_input(BenchmarkId::from_parameter(&label), &label, |b, _| {
            b.iter(|| black_box(domain.forward(black_box(&input))))
        });
    }

    // Prime128OffsetA7F7 (radix-3 / mixed radix 2^a · 3^b): smooth part
    // 2^3 · 3^7 = 17 496. Sizes cover the pure radix-3 ladder (243, 729,
    // 2187), the radix-2-on-top mixes (1458 = 2·3^6, 4374 = 2·3^7,
    // 8748 = 2^2·3^7), and the full smooth subgroup (17 496 = 2^3·3^7).
    for &n in &[243usize, 729, 1458, 2187, 4374, 8748, 17496] {
        if P_B_MINUS_1 % (n as u128) != 0 {
            continue;
        }
        let omega = find_nth_root::<Prime128OffsetA7F7>(P_B_MINUS_1, n);
        let domain = SmoothDomain::new(omega, n);
        let mut rng = StdRng::seed_from_u64(0xfe00 + n as u64);
        let input: Vec<Prime128OffsetA7F7> =
            (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();

        let label = format!("pB/N={n}");
        group.bench_with_input(BenchmarkId::from_parameter(&label), &label, |b, _| {
            b.iter(|| black_box(domain.forward(black_box(&input))))
        });
    }

    group.finish();
}

fn bench_inverse(c: &mut Criterion) {
    let g = generator();
    let mut group = c.benchmark_group("fft_inverse");

    for &n in &[300, 1470, 2940, 7350, 14700] {
        if P_MINUS_1 % (n as u128) != 0 {
            continue;
        }
        let omega = primitive_root_of_unity(g, P_MINUS_1, n);
        let domain = SmoothDomain::new(omega, n);
        let mut rng = StdRng::seed_from_u64(0xff01 + n as u64);
        let input: Vec<F> = (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();
        let transformed = domain.forward(&input);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| black_box(domain.inverse(black_box(&transformed))))
        });
    }

    group.finish();
}

fn bench_rs_extend(c: &mut Criterion) {
    let g = generator();
    let mut group = c.benchmark_group("fft_rs_extend");

    for &(k, blowup) in &[(300, 7), (1470, 5), (1470, 10), (2100, 7)] {
        let n = k * blowup;
        if P_MINUS_1 % (n as u128) != 0 {
            continue;
        }
        let omega_n = primitive_root_of_unity(g, P_MINUS_1, n);
        let omega_k = field_pow(omega_n, blowup as u64);
        let domain_k = SmoothDomain::new(omega_k, k);
        let mut rng = StdRng::seed_from_u64(0xff02 + k as u64);
        let evals: Vec<F> = (0..k).map(|_| FieldSampling::sample(&mut rng)).collect();

        let label = format!("{k}x{blowup}");
        group.bench_with_input(BenchmarkId::from_parameter(&label), &label, |b, _| {
            b.iter(|| black_box(rs_extend_fft(black_box(&evals), &domain_k, omega_n, blowup)))
        });
    }

    group.finish();
}

fn bench_rs_expand_256_to_1024(c: &mut Criterion) {
    let g = generator();
    let domain_size = 1470usize;
    let k = 256usize;
    let omega = primitive_root_of_unity(g, P_MINUS_1, domain_size);
    let domain = SmoothDomain::new(omega, domain_size);

    let mut rng = StdRng::seed_from_u64(0xff03);
    let base_evals: Vec<F> = (0..k).map(|_| FieldSampling::sample(&mut rng)).collect();

    // Zero-pad the 256 evaluations to the 1470-point domain and IFFT to
    // get a coefficient vector. This is a synthetic benchmark payload, not
    // a true RS interpolation (the zero-padding does not correspond to
    // evaluations at the remaining domain points).
    let mut padded_evals = vec![F::zero(); domain_size];
    padded_evals[..k].copy_from_slice(&base_evals);
    let coeffs = domain.inverse(&padded_evals);

    c.bench_function("fft_rs_expand/256_to_1024_via_1470", |b| {
        b.iter(|| {
            let all_evals = domain.coset_forward(black_box(&coeffs), F::one());
            black_box(&all_evals[k..k + 1024]);
            all_evals
        })
    });
}

#[cfg(feature = "parallel")]
fn run_rs_expand_256_par<Fld>(
    c: &mut Criterion,
    label: &str,
    omega: Fld,
    domain_size: usize,
    seed_tag: u64,
) where
    Fld: FieldCore + FromSmallInt + Invertible + FieldSampling + Debug + Send + Sync,
{
    let k = 256usize;
    let count = 32_768usize;
    let domain = SmoothDomain::new(omega, domain_size);

    let coeffs_batch: Vec<Vec<Fld>> = (0..count)
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(seed_tag.wrapping_add(i as u64));
            let base: Vec<Fld> = (0..k).map(|_| FieldSampling::sample(&mut rng)).collect();
            let mut padded = vec![Fld::zero(); domain_size];
            padded[..k].copy_from_slice(&base);
            domain.inverse(&padded)
        })
        .collect();

    c.bench_function(label, |b| {
        b.iter(|| {
            let results: Vec<Vec<Fld>> = coeffs_batch
                .par_iter()
                .map(|coeffs| domain.coset_forward(coeffs, Fld::one()))
                .collect();
            black_box(&results);
        })
    });
}

#[cfg(feature = "parallel")]
fn bench_rs_expand_256_to_1024_par(c: &mut Criterion) {
    // Prime128Offset2355 (p = 2^128 − 2355): smooth subgroup of order
    // 14 700 = 2²·3·5²·7². Uses the `1470 = 2·3·5·7²` subgroup, matching
    // the original benchmark.
    {
        let g = generator();
        let domain_size = 1470usize;
        let omega = primitive_root_of_unity(g, P_MINUS_1, domain_size);
        run_rs_expand_256_par::<F>(
            c,
            "fft_rs_expand/256_to_1024_via_1470_x32768_par_primeA",
            omega,
            domain_size,
            0xff04,
        );
    }

    // Prime128OffsetA7F7 (p = 2^128 − 2^32 + 22537): smooth part of order
    // 2^3·3^7 = 17 496. There is no subgroup of size 1470; the closest
    // analogue to `2·3·5·7² = 1470` that still covers the 256 + 1024 RS
    // extend budget is `1458 = 2·3^6`. `g = 2` is a QR modulo this prime,
    // so the helper scans small bases until it finds one of exact order
    // 1458.
    {
        let domain_size = 1458usize;
        let omega = find_nth_root::<Prime128OffsetA7F7>(P_B_MINUS_1, domain_size);
        run_rs_expand_256_par::<Prime128OffsetA7F7>(
            c,
            "fft_rs_expand/256_to_1024_via_1458_x32768_par_primeB",
            omega,
            domain_size,
            0xff05,
        );
    }
}

#[cfg(not(feature = "parallel"))]
fn bench_rs_expand_256_to_1024_par(_c: &mut Criterion) {}

criterion_group!(
    fft_smooth,
    bench_forward,
    bench_inverse,
    bench_rs_extend,
    bench_rs_expand_256_to_1024,
    bench_rs_expand_256_to_1024_par,
);
criterion_main!(fft_smooth);
