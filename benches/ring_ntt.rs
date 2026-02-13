#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hachi_pcs::algebra::ntt::butterfly::{forward_ntt, inverse_ntt, NttTwiddles};
use hachi_pcs::algebra::tables::{Q32_DATA, Q32_MODULUS, Q32_NUM_PRIMES, Q32_PRIMES};
use hachi_pcs::algebra::{CyclotomicNtt, CyclotomicRing, Fp64, MontCoeff};
use hachi_pcs::Field;

type F = Fp64<{ Q32_MODULUS }>;
type R = CyclotomicRing<F, 64>;
type N = CyclotomicNtt<Q32_NUM_PRIMES, 64>;

fn sample_ring(seed: u64) -> R {
    let coeffs = std::array::from_fn(|i| {
        let x = seed
            .wrapping_mul(31)
            .wrapping_add((i as u64).wrapping_mul(17));
        F::from_u64(x % Q32_MODULUS)
    });
    R::from_coefficients(coeffs)
}

fn bench_ring_schoolbook_mul(c: &mut Criterion) {
    let lhs = sample_ring(3);
    let rhs = sample_ring(11);
    c.bench_function("ring_schoolbook_mul_d64", |b| {
        b.iter(|| black_box(lhs) * black_box(rhs))
    });
}

fn bench_ntt_single_prime_round_trip(c: &mut Criterion) {
    let prime = Q32_PRIMES[0];
    let tw = NttTwiddles::<64>::compute(prime);
    let base: [MontCoeff; 64] =
        std::array::from_fn(|i| prime.from_canonical(((i * 5 + 7) as i16) % prime.p));

    c.bench_function("ntt_single_prime_forward_inverse_d64", |b| {
        b.iter(|| {
            let mut a = base;
            forward_ntt(&mut a, prime, &tw);
            inverse_ntt(&mut a, prime, &tw);
            black_box(a)
        })
    });
}

fn bench_crt_round_trip(c: &mut Criterion) {
    let ring = sample_ring(19);
    let twiddles: [NttTwiddles<64>; Q32_NUM_PRIMES] =
        std::array::from_fn(|k| NttTwiddles::<64>::compute(Q32_PRIMES[k]));

    c.bench_function("ring_ntt_crt_round_trip_d64_k6", |b| {
        b.iter(|| {
            let ntt = N::from_ring(black_box(&ring), &Q32_PRIMES, &twiddles);
            let back: R = ntt.to_ring(&Q32_PRIMES, &twiddles, &Q32_DATA);
            black_box(back)
        })
    });
}

criterion_group!(
    ring_ntt,
    bench_ring_schoolbook_mul,
    bench_ntt_single_prime_round_trip,
    bench_crt_round_trip
);
criterion_main!(ring_ntt);
