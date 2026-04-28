#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hachi_pcs::algebra::fields::fft::{field_pow, primitive_nth_root, rs_extend_fft, SmoothDomain};
use hachi_pcs::algebra::{Prime128Offset2355, Prime128OffsetA7F7};
use hachi_pcs::{FieldCore, FieldSampling, FromSmallInt, Invertible, SmoothFftField};
use rand::{rngs::StdRng, SeedableRng};
use std::fmt::Debug;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

type F = Prime128Offset2355;

fn bench_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_forward");

    // Prime128Offset2355: smooth subgroup `14_700 = 2²·3·5²·7²`.
    for &n in &[300, 1470, 2940, 7350, 14700] {
        if F::SMOOTH_SUBGROUP_ORDER % n != 0 {
            continue;
        }
        let omega = primitive_nth_root::<F>(n);
        let domain = SmoothDomain::new(omega, n);
        let mut rng = StdRng::seed_from_u64(0xff00 + n as u64);
        let input: Vec<F> = (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();

        let label = format!("pA/N={n}");
        group.bench_with_input(BenchmarkId::from_parameter(&label), &label, |b, _| {
            b.iter(|| black_box(domain.forward(black_box(&input))))
        });
    }

    // Prime128OffsetA7F7: smooth subgroup `17_496 = 2³·3⁷`. Sizes cover the
    // pure radix-3 ladder (243, 729, 2187), the radix-2-on-top mixes
    // (1458 = 2·3⁶, 4374 = 2·3⁷, 8748 = 2²·3⁷), and the full subgroup.
    type FB = Prime128OffsetA7F7;
    for &n in &[243usize, 729, 1458, 2187, 4374, 8748, 17496] {
        if FB::SMOOTH_SUBGROUP_ORDER % n != 0 {
            continue;
        }
        let omega = primitive_nth_root::<FB>(n);
        let domain = SmoothDomain::new(omega, n);
        let mut rng = StdRng::seed_from_u64(0xfe00 + n as u64);
        let input: Vec<FB> = (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();

        let label = format!("pB/N={n}");
        group.bench_with_input(BenchmarkId::from_parameter(&label), &label, |b, _| {
            b.iter(|| black_box(domain.forward(black_box(&input))))
        });
    }

    group.finish();
}

fn bench_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_inverse");

    for &n in &[300, 1470, 2940, 7350, 14700] {
        if F::SMOOTH_SUBGROUP_ORDER % n != 0 {
            continue;
        }
        let omega = primitive_nth_root::<F>(n);
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
    let mut group = c.benchmark_group("fft_rs_extend");

    for &(k, blowup) in &[(300, 7), (1470, 5), (1470, 10), (2100, 7)] {
        let n = k * blowup;
        if F::SMOOTH_SUBGROUP_ORDER % n != 0 {
            continue;
        }
        let omega_n = primitive_nth_root::<F>(n);
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
    let domain_size = 1470usize;
    let k = 256usize;
    let omega = primitive_nth_root::<F>(domain_size);
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
    // Prime128Offset2355: uses the `1470 = 2·3·5·7²` subgroup of the
    // 14_700-order smooth subgroup.
    {
        let domain_size = 1470usize;
        let omega = primitive_nth_root::<F>(domain_size);
        run_rs_expand_256_par::<F>(
            c,
            "fft_rs_expand/256_to_1024_via_1470_x32768_par_primeA",
            omega,
            domain_size,
            0xff04,
        );
    }

    // Prime128OffsetA7F7: smooth subgroup of order `2³·3⁷ = 17_496`.
    // There is no 1470-divisor; the closest analogue to `2·3·5·7² = 1470`
    // that still covers the 256+1024 RS-extend budget is `1458 = 2·3⁶`.
    {
        let domain_size = 1458usize;
        let omega = primitive_nth_root::<Prime128OffsetA7F7>(domain_size);
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
