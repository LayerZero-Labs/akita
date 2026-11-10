//! Baselines for the Akita eq APIs that predate the Jolt cutover.
//! Linux instruction count: valgrind --tool=callgrind --collect-atstart=no
//! --toggle-collect=akita_eq_vars_16_le <bench-executable> --instruction-count

use akita_algebra::eq_poly::{EqPolynomial, SplitEqEvals};
use akita_algebra::split_eq::GruenSplitEq;
use criterion::{BatchSize, BenchmarkId, Criterion};
use jolt_field::{
    Field, FpExt4, Fr, Prime128Offset275, Prime128Offset6A23801F, Prime128OffsetA7F7,
    Prime32Offset99, Ring,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::hint::black_box;

/// Akita's production 128-bit field (`proof_optimized::fp128::Field`).
type F = Prime128Offset6A23801F;

fn point<E: Field>(n: usize, seed: u64) -> Vec<E> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..n).map(|_| E::random(&mut rng)).collect()
}

fn orders<E: Field>(c: &mut Criterion, field: &str) {
    let mut group = c.benchmark_group(format!("eq_orders/{field}"));
    for n in [12, 14, 16, 17, 20, 22] {
        let r = point::<E>(n, 300 + n as u64);
        group.bench_function(BenchmarkId::new("le", n), |b| {
            b.iter(|| EqPolynomial::evals(black_box(&r)).unwrap());
        });
    }
    for n in [10, 14] {
        let mut r = point::<E>(n, 600 + n as u64);
        // Jolt passes this fixture MSB-first; Akita accepts LSB-first points.
        r.reverse();
        // Akita appends each new LSB, matching Jolt's cached_msb recurrence.
        group.bench_function(BenchmarkId::new("cached_msb", n), |b| {
            b.iter(|| EqPolynomial::evals_cached(black_box(&r)).unwrap());
        });
        group.bench_function(BenchmarkId::new("tensor", n), |b| {
            b.iter(|| SplitEqEvals::new(black_box(&r)).unwrap());
        });
    }
    group.finish();
}

fn builders<E: Field>(c: &mut Criterion, field: &str) {
    let mut group = c.benchmark_group(format!("eq_builders/{field}"));
    for n in 12..=18 {
        let r = point::<E>(n, 0xe12 + n as u64);
        group.bench_function(BenchmarkId::new("evals", n), |b| {
            b.iter(|| EqPolynomial::evals(black_box(&r)).unwrap());
        });
        group.bench_function(BenchmarkId::new("mapped", n), |b| {
            b.iter(|| EqPolynomial::evals_mapped(black_box(&r), |x| x * x).unwrap());
        });
        group.bench_function(BenchmarkId::new("evals_then_map", n), |b| {
            b.iter(|| {
                let mut table = EqPolynomial::evals(black_box(&r)).unwrap();
                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    table.par_iter_mut().for_each(|x| *x *= *x);
                }
                #[cfg(not(feature = "parallel"))]
                for x in &mut table {
                    *x *= *x;
                }
                table
            });
        });
        for len in [1, (1 << n) / 3, 1 << n] {
            group.bench_function(BenchmarkId::new(format!("prefix_{len}"), n), |b| {
                b.iter(|| EqPolynomial::evals_prefix(black_box(&r), len).unwrap());
            });
        }
    }
    group.finish();
}

fn gruen(c: &mut Criterion) {
    let mut group = c.benchmark_group("gruen");
    for n in [16, 20] {
        let mut rng = ChaCha20Rng::seed_from_u64(0xe13 + n as u64);
        let mut r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let challenges: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        r.reverse();
        let label = format!("LowToHigh_{n}");
        group.bench_function(BenchmarkId::new("construct", &label), |b| {
            b.iter(|| GruenSplitEq::new(black_box(&r)).unwrap());
        });
        // Build fresh state outside the timed sweep; Akita does not implement Clone.
        group.bench_function(BenchmarkId::new("bind", &label), |b| {
            b.iter_batched(
                || GruenSplitEq::new(&r).unwrap(),
                |mut state| {
                    for &challenge in &challenges {
                        state.bind(black_box(challenge));
                    }
                    state
                },
                BatchSize::SmallInput,
            );
        });
        let split = GruenSplitEq::new(&r).unwrap();
        let (l0, l1) = split.linear_factor_evals();
        let hint = l0 * Fr::from_u64(3) + l1 * Fr::from_u64(12);
        group.bench_function(BenchmarkId::new("degree_3", &label), |b| {
            b.iter(|| {
                split
                    .try_gruen_poly_deg_3(black_box(Fr::from_u64(3)), Fr::from_u64(2), hint)
                    .unwrap()
            });
        });
    }
    group.finish();
}

// Keep a stable symbol so Callgrind can exclude process startup and fixture creation.
#[no_mangle]
#[inline(never)]
pub fn akita_eq_vars_16_le(r: &[F]) -> Vec<F> {
    EqPolynomial::evals(black_box(r)).unwrap()
}

fn main() {
    if std::env::args().any(|arg| arg == "--instruction-count") {
        if cfg!(feature = "parallel") {
            eprintln!("build the instruction case without the parallel feature");
            std::process::exit(2);
        }
        let r = point::<F>(16, 316);
        assert_eq!(akita_eq_vars_16_le(&r), EqPolynomial::evals(&r).unwrap());
        return;
    }
    let mut c = Criterion::default().configure_from_args();
    orders::<Fr>(&mut c, "Fr");
    orders::<Prime128OffsetA7F7>(&mut c, "Prime128OffsetA7F7");
    orders::<F>(&mut c, "Prime128Offset6A23801F");
    orders::<FpExt4<Prime32Offset99>>(&mut c, "FpExt4");
    builders::<Fr>(&mut c, "Fr");
    builders::<Prime128Offset275>(&mut c, "Prime128Offset275");
    builders::<FpExt4<Prime32Offset99>>(&mut c, "FpExt4");
    gruen(&mut c);
    c.final_summary();
}
