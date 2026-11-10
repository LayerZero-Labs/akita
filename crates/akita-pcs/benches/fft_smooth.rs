#![allow(missing_docs)]

use akita_algebra::fft::SmoothFftField;
use akita_algebra::fft::{field_pow, primitive_nth_root, rs_extend_fft, SmoothDomain};
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use jolt_field::Prime128Offset6A23801F;
use jolt_field::{Field, One, Zero};
use rand::{rngs::StdRng, SeedableRng};

#[cfg(feature = "parallel")]
use jolt_field::Ring;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "parallel")]
use std::fmt::Debug;

type F = Prime128Offset6A23801F;

fn bench_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_forward");

    // Prime128Offset6A23801F: smooth subgroup `337_500_000 = 2⁵·3³·5⁸`.
    for &n in &[250, 864, 1000, 2000, 4000, 10000, 20000] {
        if F::SMOOTH_SUBGROUP_ORDER % n != 0 {
            continue;
        }
        let omega = primitive_nth_root::<F>(n);
        let domain = SmoothDomain::new(omega, n);
        let mut rng = StdRng::seed_from_u64(0xff00 + n as u64);
        let input: Vec<F> = (0..n).map(|_| Field::random(&mut rng)).collect();

        let label = format!("6A23801F/N={n}");
        group.bench_with_input(BenchmarkId::from_parameter(&label), &label, |b, _| {
            b.iter(|| black_box(domain.forward(black_box(&input))))
        });
    }

    group.finish();
}

fn bench_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_inverse");

    for &n in &[250, 864, 1000, 2000, 4000, 10000, 20000] {
        if F::SMOOTH_SUBGROUP_ORDER % n != 0 {
            continue;
        }
        let omega = primitive_nth_root::<F>(n);
        let domain = SmoothDomain::new(omega, n);
        let mut rng = StdRng::seed_from_u64(0xff01 + n as u64);
        let input: Vec<F> = (0..n).map(|_| Field::random(&mut rng)).collect();
        let transformed = domain.forward(&input);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| black_box(domain.inverse(black_box(&transformed))))
        });
    }

    group.finish();
}

fn bench_rs_extend(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_rs_extend");

    for &(k, blowup) in &[(250, 4), (250, 8), (800, 5), (1000, 4)] {
        let n = k * blowup;
        if F::SMOOTH_SUBGROUP_ORDER % n != 0 {
            continue;
        }
        let omega_n = primitive_nth_root::<F>(n);
        let omega_k = field_pow(omega_n, blowup as u64);
        let domain_k = SmoothDomain::new(omega_k, k);
        let mut rng = StdRng::seed_from_u64(0xff02 + k as u64);
        let evals: Vec<F> = (0..k).map(|_| Field::random(&mut rng)).collect();

        let label = format!("{k}x{blowup}");
        group.bench_with_input(BenchmarkId::from_parameter(&label), &label, |b, _| {
            b.iter(|| black_box(rs_extend_fft(black_box(&evals), &domain_k, omega_n, blowup)))
        });
    }

    group.finish();
}

fn bench_rs_expand_256_to_1024(c: &mut Criterion) {
    let domain_size = 1350usize;
    let k = 256usize;
    let omega = primitive_nth_root::<F>(domain_size);
    let domain = SmoothDomain::new(omega, domain_size);

    let mut rng = StdRng::seed_from_u64(0xff03);
    let base_evals: Vec<F> = (0..k).map(|_| Field::random(&mut rng)).collect();

    // Zero-pad the 256 evaluations to the 1350-point domain and IFFT to
    // get a coefficient vector. This is a synthetic benchmark payload, not
    // a true RS interpolation (the zero-padding does not correspond to
    // evaluations at the remaining domain points).
    let mut padded_evals = vec![F::zero(); domain_size];
    padded_evals[..k].copy_from_slice(&base_evals);
    let coeffs = domain.inverse(&padded_evals);

    c.bench_function("fft_rs_expand/256_to_1024_via_1350", |b| {
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
    Fld: Field + Ring + Debug + Send + Sync,
{
    let k = 256usize;
    let count = 32_768usize;
    let domain = SmoothDomain::new(omega, domain_size);

    let coeffs_batch: Vec<Vec<Fld>> = (0..count)
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(seed_tag.wrapping_add(i as u64));
            let base: Vec<Fld> = (0..k).map(|_| Field::random(&mut rng)).collect();
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
    // Prime128Offset6A23801F: smooth subgroup of order `2⁵·3³·5⁸`. The smallest
    // divisor covering the 256+1024 RS-extend budget is `1350 = 2·3³·5²`.
    {
        let domain_size = 1350usize;
        let omega = primitive_nth_root::<F>(domain_size);
        run_rs_expand_256_par::<F>(
            c,
            "fft_rs_expand/256_to_1024_via_1350_x32768_par_6A23801F",
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
