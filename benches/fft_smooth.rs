#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hachi_pcs::algebra::fields::fft::{
    field_pow, primitive_root_of_unity, rs_extend_fft, SmoothDomain,
};
use hachi_pcs::algebra::Prime128Offset2355;
use hachi_pcs::{FieldCore, FieldSampling};
use rand::{rngs::StdRng, SeedableRng};

type F = Prime128Offset2355;

const P: u128 = 0xfffffffffffffffffffffffffffff6cd;
const P_MINUS_1: u128 = P - 1;

fn generator() -> F {
    F::from_canonical_u128(2)
}

fn bench_forward(c: &mut Criterion) {
    let g = generator();
    let mut group = c.benchmark_group("fft_forward");

    for &n in &[300, 1470, 2940, 7350, 14700] {
        if P_MINUS_1 % (n as u128) != 0 {
            continue;
        }
        let omega = primitive_root_of_unity(g, P_MINUS_1, n);
        let domain = SmoothDomain::new(omega, n);
        let mut rng = StdRng::seed_from_u64(0xff00 + n as u64);
        let input: Vec<F> = (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
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

    // Interpolate: inverse FFT on the 1470-domain (zero-padded) recovers
    // the degree-255 polynomial from its evaluations at ω^0 .. ω^255.
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

criterion_group!(
    fft_smooth,
    bench_forward,
    bench_inverse,
    bench_rs_extend,
    bench_rs_expand_256_to_1024,
);
criterion_main!(fft_smooth);
