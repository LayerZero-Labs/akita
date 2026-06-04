use akita_algebra::backend::{CrtReconstruct, NttPrimeOps};
use akita_algebra::ntt::butterfly::{forward_ntt, inverse_ntt, NttTwiddles};
use akita_algebra::poly::Poly;
use akita_algebra::tables::{
    q128_garner, q128_primes, q32_garner, q64_garner, Q128_MODULUS, Q128_NUM_PRIMES, Q32_MODULUS,
    Q32_NUM_PRIMES, Q32_PRIMES, Q64_MODULUS, Q64_NUM_PRIMES, Q64_PRIMES,
};
use akita_algebra::NttPrime;
use akita_algebra::{
    CenteredMontLut, CrtNttParamSet, CyclotomicCrtNtt, CyclotomicRing, DigitMontLut, LimbQ,
    MontCoeff, PackedPartialSplitEval16, PartialSplitEval16, PartialSplitNtt16, ScalarBackend,
};
use akita_field::{Fp128, Fp32, Fp64, HasPacking, Prime128Offset159, Prime128Offset275};

#[test]
fn limbq_from_to_u128_round_trip() {
    for &val in &[0u128, 1, 12345, 123456789, (1u128 << 28) - 1] {
        let limb: LimbQ<3> = LimbQ::from(val);
        assert_eq!(
            u128::try_from(limb).unwrap(),
            val,
            "round-trip failed for {val}"
        );
    }
}

#[test]
fn limbq_add_sub_inverse() {
    let a: LimbQ<3> = LimbQ::from(12345u128);
    let b: LimbQ<3> = LimbQ::from(6789u128);
    let sum = a + b;
    let diff = sum - b;
    assert_eq!(diff, a);
}

#[test]
fn limbq_ordering() {
    let a: LimbQ<3> = LimbQ::from(100u128);
    let b: LimbQ<3> = LimbQ::from(200u128);
    assert!(a < b);
    assert!(b > a);
    assert_eq!(a, a);
}

#[test]
fn ntt_normalize_in_range() {
    for prime in &Q32_PRIMES {
        for &a in &[0i32, 1, -1, 100, -100, prime.p - 1, -(prime.p - 1)] {
            let n = prime.normalize(MontCoeff::from_raw(a));
            assert!(
                n.raw() >= 0 && n.raw() < prime.p,
                "normalize({a}) = {} for p={}",
                n.raw(),
                prime.p
            );
        }
    }
}

#[test]
fn csubp_widened_handles_large_negative_i16() {
    let prime = NttPrime::compute(15361_i16);
    let p = prime.p;
    // Values in (-2p, -(2^15 - p)) that previously overflowed in narrow i16
    for &raw in &[-20000i16, -(p + p / 2), -(p + 1000)] {
        if raw <= -2 * p || raw >= 0 {
            continue;
        }
        let a = MontCoeff::from_raw(raw);
        let reduced = prime.reduce_range(a);
        let r = reduced.raw();
        assert!(
            r > -p && r < p,
            "reduce_range({raw}) = {r} not in (-{p}, {p}) for p={p}"
        );

        let norm = prime.normalize(reduced);
        let n = norm.raw();
        assert!(
            n >= 0 && n < p,
            "normalize(reduce_range({raw})) = {n} not in [0, {p}) for p={p}"
        );
    }
}

#[test]
fn ntt_mul_commutative() {
    let prime = Q32_PRIMES[0];
    let a = MontCoeff::from_raw(1234);
    let b = MontCoeff::from_raw(5678);
    assert_eq!(prime.mul(a, b), prime.mul(b, a));
}

#[test]
fn mont_coeff_round_trip() {
    for prime in &Q32_PRIMES {
        for &val in &[0i32, 1, 2, 100, prime.p - 1] {
            let mont = prime.from_canonical(val);
            let back = prime.to_canonical(mont);
            assert_eq!(back, val, "round-trip failed for val={val}, p={}", prime.p);
        }
    }
}

#[test]
fn digit_lut_covers_log_basis_six_balanced_range() {
    let params = CrtNttParamSet::<i32, Q32_NUM_PRIMES, 64>::new(Q32_PRIMES);
    let lut = DigitMontLut::<_, Q32_NUM_PRIMES>::new(&params);

    for (k, prime) in params.primes.iter().enumerate() {
        for raw in -32i8..=31 {
            assert_eq!(lut.get(k, raw), prime.from_canonical(i32::from(raw)));
        }
    }
}

#[test]
fn digit_lut_can_cover_active_small_balanced_range() {
    let params = CrtNttParamSet::<i32, Q32_NUM_PRIMES, 64>::new(Q32_PRIMES);
    let lut = DigitMontLut::<_, Q32_NUM_PRIMES>::new_with_digit_bound(&params, 2);

    for (k, prime) in params.primes.iter().enumerate() {
        for raw in -2i8..=1 {
            assert_eq!(lut.get(k, raw), prime.from_canonical(i32::from(raw)));
        }
    }
}

#[test]
fn centered_lut_understated_bound_falls_back_exactly() {
    const D: usize = 64;
    let params = CrtNttParamSet::<i32, Q32_NUM_PRIMES, D>::new(Q32_PRIMES);
    let lut = CenteredMontLut::new(&params, 1);
    let coeffs = std::array::from_fn(|i| {
        if i % 2 == 0 {
            20 + i as i32
        } else {
            -20 - i as i32
        }
    });

    let with_lut = CyclotomicCrtNtt::from_centered_i32_pair_with_lut(&coeffs, &params, &lut);
    let direct = CyclotomicCrtNtt::<i32, Q32_NUM_PRIMES, D>::from_centered_i32_pair_with_params(
        &coeffs, &params,
    );

    assert_eq!(with_lut, direct);
}

#[test]
fn poly_add_sub_neg() {
    type F = Fp32<251>;
    let a = Poly::<F, 3>([F::from_u64(1), F::from_u64(2), F::from_u64(3)]);
    let b = Poly::<F, 3>([F::from_u64(10), F::from_u64(20), F::from_u64(30)]);

    let sum = a + b;
    assert_eq!(sum.0[0], F::from_u64(11));
    assert_eq!(sum.0[1], F::from_u64(22));
    assert_eq!(sum.0[2], F::from_u64(33));

    let diff = b - a;
    assert_eq!(diff.0[0], F::from_u64(9));

    let neg_a = -a;
    assert_eq!(a + neg_a, Poly::zero());
}

#[test]
fn ntt_forward_inverse_round_trip() {
    let prime = Q32_PRIMES[0];
    let tw = NttTwiddles::<i32, 64>::compute(prime);

    let original: [MontCoeff<i32>; 64] =
        std::array::from_fn(|i| prime.from_canonical((i as i32) % prime.p));

    let mut a = original;
    forward_ntt(&mut a, prime, &tw);
    inverse_ntt(&mut a, prime, &tw);

    for (i, (got, expected)) in a.iter().zip(original.iter()).enumerate() {
        let got_canon = prime.to_canonical(prime.normalize(*got));
        let exp_canon = prime.to_canonical(prime.normalize(*expected));
        assert_eq!(
            got_canon, exp_canon,
            "NTT round-trip mismatch at index {i}: got {got_canon}, expected {exp_canon}"
        );
    }
}

#[test]
fn ntt_forward_inverse_all_primes() {
    for (pi, prime) in Q32_PRIMES.iter().enumerate() {
        let tw = NttTwiddles::<i32, 64>::compute(*prime);

        let original: [_; 64] =
            std::array::from_fn(|i| prime.from_canonical(((i * (pi + 1)) as i32) % prime.p));

        let mut a = original;
        forward_ntt(&mut a, *prime, &tw);
        inverse_ntt(&mut a, *prime, &tw);

        for (i, (got, expected)) in a.iter().zip(original.iter()).enumerate() {
            let g = prime.to_canonical(prime.normalize(*got));
            let e = prime.to_canonical(prime.normalize(*expected));
            assert_eq!(
                g, e,
                "prime[{pi}] p={}: round-trip mismatch at index {i}",
                prime.p
            );
        }
    }
}

#[test]
fn negacyclic_ntt_mul_matches_schoolbook_single_prime_d8() {
    const D: usize = 8;
    let prime = Q32_PRIMES[0];
    let tw = NttTwiddles::<i32, D>::compute(prime);

    let a_canon: [i32; D] = std::array::from_fn(|i| ((i as i32 * 7) + 3) % prime.p);
    let b_canon: [i32; D] = std::array::from_fn(|i| ((i as i32 * 5) + 11) % prime.p);

    // Schoolbook negacyclic convolution mod p: X^D = -1.
    let mut school = [0i32; D];
    for (i, &ai) in a_canon.iter().enumerate() {
        for (j, &bj) in b_canon.iter().enumerate() {
            let prod = (ai as i64 * bj as i64) % (prime.p as i64);
            let idx = i + j;
            if idx < D {
                school[idx] = ((school[idx] as i64 + prod) % (prime.p as i64)) as i32;
            } else {
                let k = idx - D;
                school[k] = ((school[k] as i64 - prod) % (prime.p as i64)) as i32;
            }
        }
    }
    for x in &mut school {
        if *x < 0 {
            *x = (*x as i64 + prime.p as i64) as i32;
        }
    }

    let mut a = std::array::from_fn(|i| prime.from_canonical(a_canon[i]));
    let mut b = std::array::from_fn(|i| prime.from_canonical(b_canon[i]));
    forward_ntt(&mut a, prime, &tw);
    forward_ntt(&mut b, prime, &tw);

    let mut c: [_; D] = std::array::from_fn(|i| prime.mul(a[i], b[i]));
    inverse_ntt(&mut c, prime, &tw);

    let got: [i32; D] = std::array::from_fn(|i| prime.to_canonical(prime.normalize(c[i])));
    assert_eq!(got, school);
}

#[test]
fn negacyclic_ntt_forward_matches_manual_evals_d8() {
    const D: usize = 8;
    let prime = Q32_PRIMES[0];
    let tw = NttTwiddles::<i32, D>::compute(prime);
    let p = prime.p as i64;

    fn pow_mod(mut base: i64, mut exp: i64, modulus: i64) -> i64 {
        let mut acc = 1i64;
        base %= modulus;
        while exp > 0 {
            if exp & 1 == 1 {
                acc = (acc * base) % modulus;
            }
            base = (base * base) % modulus;
            exp >>= 1;
        }
        acc
    }

    // Compute canonical psi (primitive 2D-th root) directly.
    let half = (p - 1) / 2;
    let exp = (p - 1) / (2 * D as i64);
    let mut psi = None;
    for a in 2..p {
        if pow_mod(a, half, p) == p - 1 {
            let cand = pow_mod(a, exp, p);
            if pow_mod(cand, D as i64, p) == p - 1 {
                psi = Some(cand);
                break;
            }
        }
    }
    let psi = psi.expect("psi should exist");
    let a_canon: [i32; D] = std::array::from_fn(|i| ((i as i32 * 7) + 3) % prime.p);

    let mut expected = Vec::with_capacity(D);
    for k in 0..D {
        let alpha = pow_mod(psi, (2 * k + 1) as i64, p);
        let mut acc = 0i64;
        let mut power = 1i64;
        for &ai in &a_canon {
            acc = (acc + (ai as i64) * power) % p;
            power = (power * alpha) % p;
        }
        expected.push(acc as i32);
    }
    expected.sort_unstable();

    let mut a = std::array::from_fn(|i| prime.from_canonical(a_canon[i]));
    forward_ntt(&mut a, prime, &tw);
    let mut got: Vec<i32> = a
        .iter()
        .map(|x| prime.to_canonical(prime.normalize(*x)))
        .collect();
    got.sort_unstable();

    assert_eq!(got, expected);
}

#[test]
fn negacyclic_ntt_mul_matches_schoolbook_single_prime_d64() {
    const D: usize = 64;
    let prime = Q32_PRIMES[0];
    let tw = NttTwiddles::<i32, D>::compute(prime);
    let p = prime.p as i64;

    let a_canon: [i32; D] = std::array::from_fn(|i| ((i as i32 * 7) + 3) % prime.p);
    let b_canon: [i32; D] = std::array::from_fn(|i| ((i as i32 * 5) + 11) % prime.p);

    let mut school = [0i32; D];
    for (i, &ai) in a_canon.iter().enumerate() {
        for (j, &bj) in b_canon.iter().enumerate() {
            let prod = (ai as i64 * bj as i64) % p;
            let idx = i + j;
            if idx < D {
                school[idx] = ((school[idx] as i64 + prod) % p) as i32;
            } else {
                let k = idx - D;
                school[k] = ((school[k] as i64 - prod) % p) as i32;
            }
        }
    }
    for x in &mut school {
        if *x < 0 {
            *x = (*x as i64 + p) as i32;
        }
    }

    let mut a = std::array::from_fn(|i| prime.from_canonical(a_canon[i]));
    let mut b = std::array::from_fn(|i| prime.from_canonical(b_canon[i]));
    forward_ntt(&mut a, prime, &tw);
    forward_ntt(&mut b, prime, &tw);

    let mut c: [_; D] = std::array::from_fn(|i| prime.reduce_range(prime.mul(a[i], b[i])));
    inverse_ntt(&mut c, prime, &tw);

    let got: [i32; D] = std::array::from_fn(|i| prime.to_canonical(prime.normalize(c[i])));
    assert_eq!(got, school);
}

#[test]
fn negacyclic_ntt_mul_matches_schoolbook_all_q32_primes_d64() {
    const D: usize = 64;
    let a_canon: [i32; D] = std::array::from_fn(|i| i as i32 * 7 + 3);
    let b_canon: [i32; D] = std::array::from_fn(|i| i as i32 * 5 + 11);

    for (pi, &prime) in Q32_PRIMES.iter().enumerate() {
        let tw = NttTwiddles::<i32, D>::compute(prime);
        let p = prime.p as i64;

        let a_mod: [i32; D] = std::array::from_fn(|i| ((a_canon[i] as i64).rem_euclid(p)) as i32);
        let b_mod: [i32; D] = std::array::from_fn(|i| ((b_canon[i] as i64).rem_euclid(p)) as i32);

        let mut school = [0i32; D];
        for (i, &ai) in a_mod.iter().enumerate() {
            for (j, &bj) in b_mod.iter().enumerate() {
                let prod = (ai as i64 * bj as i64) % p;
                let idx = i + j;
                if idx < D {
                    school[idx] = ((school[idx] as i64 + prod) % p) as i32;
                } else {
                    let k = idx - D;
                    school[k] = ((school[k] as i64 - prod) % p) as i32;
                }
            }
        }
        for x in &mut school {
            if *x < 0 {
                *x = (*x as i64 + p) as i32;
            }
        }

        let mut a = std::array::from_fn(|i| prime.from_canonical(a_mod[i]));
        let mut b = std::array::from_fn(|i| prime.from_canonical(b_mod[i]));
        forward_ntt(&mut a, prime, &tw);
        forward_ntt(&mut b, prime, &tw);

        let mut c = [MontCoeff::from_raw(0i32); D];
        for i in 0..D {
            c[i] = prime.reduce_range(prime.mul(a[i], b[i]));
        }
        inverse_ntt(&mut c, prime, &tw);

        let got: [i32; D] = std::array::from_fn(|i| prime.to_canonical(prime.normalize(c[i])));
        assert_eq!(got, school, "prime[{pi}] p={} mismatch", prime.p);
    }
}

#[test]
fn cyclotomic_ntt_crt_round_trip_q32() {
    type F = Fp64<{ Q32_MODULUS }>;
    type R = CyclotomicRing<F, 64>;
    type N = CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, 64>;

    let primes = Q32_PRIMES;
    let twiddles: [NttTwiddles<i32, 64>; Q32_NUM_PRIMES] =
        std::array::from_fn(|k| NttTwiddles::compute(primes[k]));

    let coeffs: [F; 64] = std::array::from_fn(|i| F::from_u64(((i as u64 * 17) + 5) % Q32_MODULUS));
    let ring = R::from_coefficients(coeffs);
    let ntt = N::from_ring(&ring, &primes, &twiddles);
    let garner = q32_garner();
    let round_trip = ntt.to_ring(&primes, &twiddles, &garner);

    assert_eq!(ring, round_trip);
}

const SYNTHETIC_I16_NUM_PRIMES: usize = 3;

fn synthetic_i16_primes() -> [NttPrime<i16>; SYNTHETIC_I16_NUM_PRIMES] {
    [
        NttPrime::compute(15361_i16),
        NttPrime::compute(13313_i16),
        NttPrime::compute(12289_i16),
    ]
}

fn assert_synthetic_i16_ntt_round_trip<const D: usize>() {
    type F = Fp64<{ Q32_MODULUS }>;
    type R<const D: usize> = CyclotomicRing<F, D>;
    type N<const D: usize> = CyclotomicCrtNtt<i16, SYNTHETIC_I16_NUM_PRIMES, D>;

    let params = CrtNttParamSet::<i16, SYNTHETIC_I16_NUM_PRIMES, D>::new(synthetic_i16_primes());
    let coeffs: [F; D] = std::array::from_fn(|i| F::from_u64(((i as u64 * 17) + 5) % Q32_MODULUS));
    let ring = R::<D>::from_coefficients(coeffs);
    let ntt = N::<D>::from_ring_with_params(&ring, &params);
    let round_trip = ntt.to_ring_with_params(&params);

    assert_eq!(ring, round_trip);
}

fn assert_q32_ntt_round_trip<const D: usize>() {
    type F = Fp64<{ Q32_MODULUS }>;
    type R<const D: usize> = CyclotomicRing<F, D>;
    type N<const D: usize> = CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, D>;

    let params = CrtNttParamSet::<i32, Q32_NUM_PRIMES, D>::new(Q32_PRIMES);
    let coeffs: [F; D] = std::array::from_fn(|i| F::from_u64(((i as u64 * 17) + 5) % Q32_MODULUS));
    let ring = R::<D>::from_coefficients(coeffs);
    let ntt = N::<D>::from_ring_with_params(&ring, &params);
    let round_trip = ntt.to_ring_with_params(&params);

    assert_eq!(ring, round_trip);
}

fn assert_q64_ntt_round_trip<const D: usize>() {
    type F = Fp64<{ Q64_MODULUS }>;
    type R<const D: usize> = CyclotomicRing<F, D>;
    type N<const D: usize> = CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, D>;

    let params = CrtNttParamSet::<i32, Q64_NUM_PRIMES, D>::new(Q64_PRIMES);
    let coeffs: [F; D] = std::array::from_fn(|i| F::from_u64(((i as u64 * 19) + 3) % Q64_MODULUS));
    let ring = R::<D>::from_coefficients(coeffs);
    let ntt = N::<D>::from_ring_with_params(&ring, &params);
    let round_trip = ntt.to_ring_with_params(&params);

    assert_eq!(ring, round_trip);
}

#[test]
fn reduced_q32_ntt_round_trips_across_supported_ring_dims() {
    assert_q32_ntt_round_trip::<32>();
    assert_q32_ntt_round_trip::<64>();
    assert_q32_ntt_round_trip::<128>();
    assert_q32_ntt_round_trip::<256>();
}

#[test]
fn synthetic_i16_ntt_round_trips_across_supported_ring_dims() {
    assert_synthetic_i16_ntt_round_trip::<32>();
    assert_synthetic_i16_ntt_round_trip::<64>();
    assert_synthetic_i16_ntt_round_trip::<128>();
    assert_synthetic_i16_ntt_round_trip::<256>();
}

#[test]
fn reduced_q64_ntt_round_trips_across_supported_ring_dims() {
    assert_q64_ntt_round_trip::<32>();
    assert_q64_ntt_round_trip::<64>();
    assert_q64_ntt_round_trip::<128>();
    assert_q64_ntt_round_trip::<256>();
}

#[test]
fn cyclotomic_ntt_reduced_ops_are_stable() {
    type F = Fp64<{ Q32_MODULUS }>;
    type R = CyclotomicRing<F, 64>;
    type N = CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, 64>;

    let primes = Q32_PRIMES;
    let twiddles: [NttTwiddles<i32, 64>; Q32_NUM_PRIMES] =
        std::array::from_fn(|k| NttTwiddles::compute(primes[k]));

    let a = R::from_coefficients(std::array::from_fn(|i| {
        F::from_u64(((i as u64 * 3) + 1) % Q32_MODULUS)
    }));
    let b = R::from_coefficients(std::array::from_fn(|i| {
        F::from_u64(((i as u64 * 11) + 7) % Q32_MODULUS)
    }));

    let ntt_a = N::from_ring(&a, &primes, &twiddles);
    let ntt_b = N::from_ring(&b, &primes, &twiddles);

    let sum = ntt_a.add_reduced(&ntt_b, &primes);
    let back = sum.sub_reduced(&ntt_b, &primes);
    assert_eq!(back, ntt_a);

    let garner = q32_garner();
    let zero_ntt = ntt_a.add_reduced(&ntt_a.neg_reduced(&primes), &primes);
    let zero_ring = zero_ntt.to_ring(&primes, &twiddles, &garner);
    assert_eq!(zero_ring, R::zero());
}

#[test]
fn backend_path_matches_default_scalar_path() {
    type F = Fp64<{ Q32_MODULUS }>;
    type R = CyclotomicRing<F, 64>;
    type N = CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, 64>;

    let primes = Q32_PRIMES;
    let twiddles: [NttTwiddles<i32, 64>; Q32_NUM_PRIMES] =
        std::array::from_fn(|k| NttTwiddles::compute(primes[k]));
    let ring = R::from_coefficients(std::array::from_fn(|i| {
        F::from_u64(((i as u64 * 13) + 9) % Q32_MODULUS)
    }));

    let default_ntt = N::from_ring(&ring, &primes, &twiddles);
    let backend_ntt = N::from_ring_with_backend::<F, ScalarBackend>(&ring, &primes, &twiddles);
    assert_eq!(default_ntt, backend_ntt);

    let garner = q32_garner();
    let default_back = default_ntt.to_ring(&primes, &twiddles, &garner);
    let backend_back =
        backend_ntt.to_ring_with_backend::<F, ScalarBackend>(&primes, &twiddles, &garner);
    assert_eq!(default_back, backend_back);
}

#[test]
fn crt_ntt_mul_matches_schoolbook_q32() {
    type F = Fp64<{ Q32_MODULUS }>;
    type R = CyclotomicRing<F, 64>;
    type N = CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, 64>;

    let primes = Q32_PRIMES;
    let twiddles: [NttTwiddles<i32, 64>; Q32_NUM_PRIMES] =
        std::array::from_fn(|k| NttTwiddles::compute(primes[k]));
    let garner = q32_garner();

    let a = R::from_coefficients(std::array::from_fn(|i| {
        F::from_u64(((i as u64 * 7) + 3) % Q32_MODULUS)
    }));
    let b = R::from_coefficients(std::array::from_fn(|i| {
        F::from_u64(((i as u64 * 5) + 11) % Q32_MODULUS)
    }));

    let schoolbook = a * b;

    let ntt_a = N::from_ring(&a, &primes, &twiddles);
    let ntt_b = N::from_ring(&b, &primes, &twiddles);
    let ntt_prod = ntt_a.pointwise_mul(&ntt_b, &primes);
    let ntt_result: R = ntt_prod.to_ring(&primes, &twiddles, &garner);

    assert_eq!(schoolbook, ntt_result);
}

#[test]
fn q128_garner_reconstruct_matches_coeffs_no_ntt() {
    type F = Fp128<{ Q128_MODULUS }>;

    let primes = q128_primes();
    let garner = q128_garner();

    let coeffs: [F; 64] = std::array::from_fn(|i| {
        if i < 8 {
            F::from_u64((i as u64 * 31) + 7)
        } else {
            F::zero()
        }
    });

    let mut canonical = [[0i32; 64]; Q128_NUM_PRIMES];
    for (k, prime) in primes.iter().enumerate() {
        let p = prime.p as u32 as u128;
        for (i, c) in coeffs.iter().enumerate() {
            canonical[k][i] = (c.to_canonical_u128() % p) as i32;
        }
    }

    let reconstructed: [F; 64] =
        <ScalarBackend as CrtReconstruct<i32, Q128_NUM_PRIMES, 64>>::reconstruct(
            &primes, &canonical, &garner,
        );

    assert_eq!(reconstructed, coeffs);
}

#[test]
fn q128_prime_ntt_round_trip_per_prime() {
    let primes = q128_primes();
    let twiddles: [NttTwiddles<i32, 64>; Q128_NUM_PRIMES] =
        std::array::from_fn(|k| NttTwiddles::compute(primes[k]));

    // Use the same sparse coefficient pattern as q128_ntt_round_trip, but test
    // the per-prime NTT+Montgomery machinery in isolation (no Garner/Fp128).
    let residues: [u32; 64] = std::array::from_fn(|i| if i < 8 { (i as u32 * 31) + 7 } else { 0 });

    for k in 0..Q128_NUM_PRIMES {
        let prime = primes[k];
        let mut limb = [MontCoeff::from_raw(0i32); 64];
        for (i, r) in residues.iter().enumerate() {
            let reduced = (*r as i64 % (prime.p as i64)) as i32;
            limb[i] = <ScalarBackend as NttPrimeOps<i32, 64>>::from_canonical(prime, reduced);
        }

        forward_ntt(&mut limb, prime, &twiddles[k]);
        inverse_ntt(&mut limb, prime, &twiddles[k]);

        for (i, r) in residues.iter().enumerate() {
            let expected = (*r as i64 % (prime.p as i64)) as i32;
            let got = <ScalarBackend as NttPrimeOps<i32, 64>>::to_canonical(prime, limb[i]);
            assert_eq!(got, expected, "prime idx={k} coeff idx={i}");
        }
    }
}

#[test]
fn q128_ntt_round_trip() {
    type F = Fp128<{ Q128_MODULUS }>;
    type R = CyclotomicRing<F, 64>;
    type N = CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, 64>;

    let primes = q128_primes();
    let twiddles: [NttTwiddles<i32, 64>; Q128_NUM_PRIMES] =
        std::array::from_fn(|k| NttTwiddles::compute(primes[k]));
    let garner = q128_garner();

    let coeffs: [F; 64] = std::array::from_fn(|i| {
        if i < 8 {
            F::from_u64((i as u64 * 31) + 7)
        } else {
            F::zero()
        }
    });
    let ring = R::from_coefficients(coeffs);
    let ntt = N::from_ring(&ring, &primes, &twiddles);
    let round_trip: R = ntt.to_ring(&primes, &twiddles, &garner);

    assert_eq!(ring, round_trip);
}

#[test]
fn crt_ntt_mul_matches_schoolbook_q128() {
    type F = Fp128<{ Q128_MODULUS }>;
    type R = CyclotomicRing<F, 64>;
    type N = CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, 64>;

    let primes = q128_primes();
    let twiddles: [NttTwiddles<i32, 64>; Q128_NUM_PRIMES] =
        std::array::from_fn(|k| NttTwiddles::compute(primes[k]));
    let garner = q128_garner();

    let a = R::from_coefficients(std::array::from_fn(|i| {
        if i < 8 {
            F::from_u64((i as u64 * 7) + 3)
        } else {
            F::zero()
        }
    }));
    let b = R::from_coefficients(std::array::from_fn(|i| {
        if i < 8 {
            F::from_u64((i as u64 * 9) + 11)
        } else {
            F::zero()
        }
    }));

    let schoolbook = a * b;

    let ntt_a = N::from_ring(&a, &primes, &twiddles);
    let ntt_b = N::from_ring(&b, &primes, &twiddles);
    let ntt_prod = ntt_a.pointwise_mul(&ntt_b, &primes);
    let ntt_result: R = ntt_prod.to_ring(&primes, &twiddles, &garner);

    assert_eq!(schoolbook, ntt_result);
}

#[test]
fn partial_split_forward_matches_direct_eval_q128m159() {
    type F = Prime128Offset159;

    fn eval_poly(coeffs: &[F; 16], x: F) -> F {
        coeffs
            .iter()
            .rev()
            .fold(F::zero(), |acc, coeff| acc * x + *coeff)
    }

    let split = PartialSplitNtt16::<F>::compute();
    let coeffs: [F; 16] = std::array::from_fn(|i| {
        let centered = ((i as i64 * 19) % 29) - 14;
        F::from_i64(centered)
    });

    let mut got = coeffs;
    split.forward_class(&mut got);

    let expected: [F; 16] = std::array::from_fn(|i| eval_poly(&coeffs, split.eval_roots()[i]));
    assert_eq!(got, expected);
}

#[test]
fn partial_split_mul_matches_schoolbook_q128m159() {
    type F = Prime128Offset159;
    type R = CyclotomicRing<F, 32>;

    let split = PartialSplitNtt16::<F>::compute();
    let a = R::from_coefficients(std::array::from_fn(|i| {
        let centered = ((i as i64 * 7 + 3) % 41) - 20;
        F::from_i64(centered)
    }));
    let b = R::from_coefficients(std::array::from_fn(|i| {
        let centered = ((i as i64 * 11 + 5) % 37) - 18;
        F::from_i64(centered)
    }));

    let schoolbook = a * b;
    let split_result = split.multiply_d32(&a, &b);

    assert_eq!(schoolbook, split_result);
}

#[test]
fn partial_split_matches_crt_mul_q128m159() {
    type F = Prime128Offset159;
    type R = CyclotomicRing<F, 32>;
    type N = CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, 32>;

    let split = PartialSplitNtt16::<F>::compute();
    let params = CrtNttParamSet::new(q128_primes());

    let a = R::from_coefficients(std::array::from_fn(|i| {
        let centered = ((i as i64 * 13 + 1) % 33) - 16;
        F::from_i64(centered)
    }));
    let b = R::from_coefficients(std::array::from_fn(|i| {
        let centered = ((i as i64 * 9 + 7) % 35) - 17;
        F::from_i64(centered)
    }));

    let split_result = split.multiply_d32(&a, &b);

    let ntt_a = N::from_ring_with_params(&a, &params);
    let ntt_b = N::from_ring_with_params(&b, &params);
    let crt_result: R = ntt_a
        .pointwise_mul_with_params(&ntt_b, &params)
        .to_ring_with_params(&params);

    assert_eq!(split_result, crt_result);
}

#[test]
fn partial_split_mul_centered_i8_matches_schoolbook_q128m159() {
    type F = Prime128Offset159;
    type R = CyclotomicRing<F, 32>;

    let split = PartialSplitNtt16::<F>::compute();
    let lhs = R::from_coefficients(std::array::from_fn(|i| {
        let centered = ((i as i64 * 7 + 3) % 41) - 20;
        F::from_i64(centered)
    }));
    let rhs_i8: [i8; 32] = std::array::from_fn(|i| (((i * 23 + 11) % 256) as i32 - 128) as i8);
    let rhs = R::from_coefficients(std::array::from_fn(|i| F::from_i8(rhs_i8[i])));

    let schoolbook = lhs * rhs;
    let split_result = split.multiply_d32_rhs_i8(&lhs, &rhs_i8);

    assert_eq!(schoolbook, split_result);
}

#[test]
fn partial_split_mul_centered_i8_matches_crt_q128m159() {
    type F = Prime128Offset159;
    type R = CyclotomicRing<F, 32>;
    type N = CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, 32>;

    let split = PartialSplitNtt16::<F>::compute();
    let params = CrtNttParamSet::new(q128_primes());
    let lhs = R::from_coefficients(std::array::from_fn(|i| {
        let centered = ((i as i64 * 13 + 1) % 33) - 16;
        F::from_i64(centered)
    }));
    let rhs_i8: [i8; 32] = std::array::from_fn(|i| (((i * 19 + 5) % 256) as i32 - 128) as i8);

    let split_result = split.multiply_d32_rhs_i8(&lhs, &rhs_i8);
    let crt_result: R = N::from_ring_with_params(&lhs, &params)
        .pointwise_mul_with_params(&N::from_i8_with_params(&rhs_i8, &params), &params)
        .to_ring_with_params(&params);

    assert_eq!(split_result, crt_result);
}

#[test]
fn partial_split_repr_round_trip_q128m159() {
    type F = Prime128Offset159;
    type R = CyclotomicRing<F, 32>;

    let split = PartialSplitNtt16::<F>::compute();
    let ring = R::from_coefficients(std::array::from_fn(|i| {
        let centered = ((i as i64 * 17 + 9) % 53) - 26;
        F::from_i64(centered)
    }));

    let eval = PartialSplitEval16::from_ring(&split, &ring);
    let back = eval.to_ring(&split);

    assert_eq!(ring, back);
}

#[test]
fn partial_split_repr_cached_product_matches_schoolbook_q128m159() {
    type F = Prime128Offset159;
    type R = CyclotomicRing<F, 32>;

    let split = PartialSplitNtt16::<F>::compute();
    let lhs = R::from_coefficients(std::array::from_fn(|i| {
        let centered = ((i as i64 * 5 + 7) % 47) - 23;
        F::from_i64(centered)
    }));
    let rhs = R::from_coefficients(std::array::from_fn(|i| {
        let centered = ((i as i64 * 11 + 3) % 43) - 21;
        F::from_i64(centered)
    }));

    let lhs_eval = PartialSplitEval16::from_ring(&split, &lhs);
    let rhs_eval = PartialSplitEval16::from_ring(&split, &rhs);
    let cached = lhs_eval.pointwise_mul(&rhs_eval, &split).to_ring(&split);

    assert_eq!(cached, lhs * rhs);
}

#[test]
fn partial_split_cyclic_mul_matches_schoolbook_q128m159() {
    type F = Prime128Offset159;
    type R = CyclotomicRing<F, 32>;

    let split = PartialSplitNtt16::<F>::compute();
    let lhs = R::from_coefficients(std::array::from_fn(|i| {
        let centered = ((i as i64 * 5 + 7) % 47) - 23;
        F::from_i64(centered)
    }));
    let rhs = R::from_coefficients(std::array::from_fn(|i| {
        let centered = ((i as i64 * 11 + 3) % 43) - 21;
        F::from_i64(centered)
    }));

    let mut school = [F::zero(); 32];
    for i in 0..32 {
        for j in 0..32 {
            school[(i + j) % 32] += lhs.coefficients()[i] * rhs.coefficients()[j];
        }
    }

    assert_eq!(split.multiply_cyclic_d32(&lhs, &rhs), school);
}

#[test]
fn partial_split_quotient_matches_schoolbook_high_half_q128m159() {
    type F = Prime128Offset159;
    type R = CyclotomicRing<F, 32>;

    let split = PartialSplitNtt16::<F>::compute();
    let lhs = R::from_coefficients(std::array::from_fn(|i| {
        let centered = ((i as i64 * 7 + 5) % 41) - 20;
        F::from_i64(centered)
    }));
    let rhs = R::from_coefficients(std::array::from_fn(|i| {
        let centered = ((i as i64 * 9 + 1) % 39) - 19;
        F::from_i64(centered)
    }));

    let mut high = [F::zero(); 32];
    for i in 0..32 {
        for j in 0..32 {
            let idx = i + j;
            if idx >= 32 {
                high[idx - 32] += lhs.coefficients()[i] * rhs.coefficients()[j];
            }
        }
    }

    let quotient = split.unreduced_quotient_d32(&lhs, &rhs);
    assert_eq!(quotient.coefficients(), &high);
}

#[test]
fn partial_split_cached_matvec_matches_schoolbook_q128m159() {
    type F = Prime128Offset159;
    type R = CyclotomicRing<F, 32>;

    const ROWS: usize = 3;
    const COLS: usize = 5;

    let split = PartialSplitNtt16::<F>::compute();
    let matrix: Vec<Vec<R>> = (0..ROWS)
        .map(|r| {
            (0..COLS)
                .map(|c| {
                    R::from_coefficients(std::array::from_fn(|i| {
                        let centered = ((i as i64 * 7 + (11 * r + 5 * c) as i64) % 37) - 18;
                        F::from_i64(centered)
                    }))
                })
                .collect()
        })
        .collect();
    let vector: Vec<R> = (0..COLS)
        .map(|c| {
            R::from_coefficients(std::array::from_fn(|i| {
                let centered = ((i as i64 * 13 + (9 * c) as i64) % 41) - 20;
                F::from_i64(centered)
            }))
        })
        .collect();

    let matrix_eval: Vec<Vec<PartialSplitEval16<F>>> = matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|ring| PartialSplitEval16::from_ring(&split, ring))
                .collect()
        })
        .collect();
    let vector_eval: Vec<PartialSplitEval16<F>> = vector
        .iter()
        .map(|ring| PartialSplitEval16::from_ring(&split, ring))
        .collect();

    let got: Vec<R> = (0..ROWS)
        .map(|r| {
            let mut acc = PartialSplitEval16::zero();
            for (mat_entry, vec_entry) in matrix_eval[r].iter().zip(vector_eval.iter()) {
                acc.add_mul_assign(mat_entry, vec_entry, &split);
            }
            acc.to_ring(&split)
        })
        .collect();

    let expected: Vec<R> = (0..ROWS)
        .map(|r| {
            let mut acc = R::zero();
            for (mat_entry, vec_entry) in matrix[r].iter().zip(vector.iter()) {
                acc += *mat_entry * *vec_entry;
            }
            acc
        })
        .collect();

    assert_eq!(got, expected);
}

#[test]
fn partial_split_packed_cached_matvec_matches_scalar_q128m159() {
    type F = Prime128Offset159;
    type PF = <F as HasPacking>::Packing;
    type R = CyclotomicRing<F, 32>;

    let rows = PackedPartialSplitEval16::<PF>::WIDTH + 3;
    let cols = 5usize;

    let split = PartialSplitNtt16::<F>::compute();
    let packed = split.packed::<PF>();
    let matrix: Vec<Vec<R>> = (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| {
                    R::from_coefficients(std::array::from_fn(|i| {
                        let centered = ((i as i64 * 7 + (11 * r + 5 * c) as i64) % 37) - 18;
                        F::from_i64(centered)
                    }))
                })
                .collect()
        })
        .collect();
    let vector: Vec<R> = (0..cols)
        .map(|c| {
            R::from_coefficients(std::array::from_fn(|i| {
                let centered = ((i as i64 * 13 + (9 * c) as i64) % 41) - 20;
                F::from_i64(centered)
            }))
        })
        .collect();

    let matrix_eval: Vec<Vec<PartialSplitEval16<F>>> = matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|ring| PartialSplitEval16::from_ring(&split, ring))
                .collect()
        })
        .collect();
    let vector_eval: Vec<PartialSplitEval16<F>> = vector
        .iter()
        .map(|ring| PartialSplitEval16::from_ring(&split, ring))
        .collect();
    let vector_packed: Vec<PackedPartialSplitEval16<PF>> = vector_eval
        .iter()
        .map(PackedPartialSplitEval16::<PF>::broadcast)
        .collect();

    let mut got = Vec::with_capacity(rows);
    let mut row_chunks = matrix_eval.chunks_exact(PackedPartialSplitEval16::<PF>::WIDTH);
    for row_chunk in row_chunks.by_ref() {
        let packed_row: Vec<PackedPartialSplitEval16<PF>> = (0..cols)
            .map(|c| PackedPartialSplitEval16::<PF>::from_fn(|lane| row_chunk[lane][c]))
            .collect();
        let mut acc = PackedPartialSplitEval16::<PF>::zero();
        for (mat_entry, vec_entry) in packed_row.iter().zip(vector_packed.iter()) {
            packed.add_mul_assign(&mut acc, mat_entry, vec_entry);
        }
        packed.append_rings(&acc, &mut got);
    }
    for row in row_chunks.remainder() {
        let mut acc = PartialSplitEval16::zero();
        for (mat_entry, vec_entry) in row.iter().zip(vector_eval.iter()) {
            acc.add_mul_assign(mat_entry, vec_entry, &split);
        }
        got.push(acc.to_ring(&split));
    }

    let expected: Vec<R> = (0..rows)
        .map(|r| {
            let mut acc = R::zero();
            for (mat_entry, vec_entry) in matrix[r].iter().zip(vector.iter()) {
                acc += *mat_entry * *vec_entry;
            }
            acc
        })
        .collect();

    assert_eq!(got, expected);
}

#[test]
fn crt_add_assign_pointwise_mul_matches_scalar_q128m275() {
    type F = Prime128Offset275;
    type R = CyclotomicRing<F, 64>;
    type N = CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, 64>;

    let params = CrtNttParamSet::new(q128_primes());
    let acc0 = N::from_ring_with_params(
        &R::from_coefficients(std::array::from_fn(|i| {
            F::from_i64(((i as i64 * 5 + 1) % 31) - 15)
        })),
        &params,
    );
    let lhs = N::from_ring_with_params(
        &R::from_coefficients(std::array::from_fn(|i| {
            F::from_i64(((i as i64 * 7 + 3) % 37) - 18)
        })),
        &params,
    );
    let rhs = N::from_ring_with_params(
        &R::from_coefficients(std::array::from_fn(|i| {
            F::from_i64(((i as i64 * 11 + 9) % 41) - 20)
        })),
        &params,
    );

    let mut got = acc0.clone();
    got.add_assign_pointwise_mul_with_params(&lhs, &rhs, &params);

    let expected =
        acc0.add_reduced_with_params(&lhs.pointwise_mul_with_params(&rhs, &params), &params);

    assert_eq!(got, expected);
}

#[test]
fn q64_ntt_round_trip() {
    type F = Fp64<{ Q64_MODULUS }>;
    type R = CyclotomicRing<F, 64>;
    type N = CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, 64>;

    let primes = Q64_PRIMES;
    let twiddles: [NttTwiddles<i32, 64>; Q64_NUM_PRIMES] =
        std::array::from_fn(|k| NttTwiddles::compute(primes[k]));
    let garner = q64_garner();

    let coeffs: [F; 64] = std::array::from_fn(|i| F::from_u64(((i as u64 * 19) + 3) % Q64_MODULUS));
    let ring = R::from_coefficients(coeffs);
    let ntt = N::from_ring(&ring, &primes, &twiddles);
    let round_trip: R = ntt.to_ring(&primes, &twiddles, &garner);

    assert_eq!(ring, round_trip);
}

#[test]
fn crt_ntt_mul_matches_schoolbook_q64() {
    type F = Fp64<{ Q64_MODULUS }>;
    type R = CyclotomicRing<F, 64>;
    type N = CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, 64>;

    let primes = Q64_PRIMES;
    let twiddles: [NttTwiddles<i32, 64>; Q64_NUM_PRIMES] =
        std::array::from_fn(|k| NttTwiddles::compute(primes[k]));
    let garner = q64_garner();

    let a = R::from_coefficients(std::array::from_fn(|i| {
        F::from_u64(((i as u64 * 5) + 9) % Q64_MODULUS)
    }));
    let b = R::from_coefficients(std::array::from_fn(|i| {
        F::from_u64(((i as u64 * 17) + 13) % Q64_MODULUS)
    }));

    let schoolbook = a * b;

    let ntt_a = N::from_ring(&a, &primes, &twiddles);
    let ntt_b = N::from_ring(&b, &primes, &twiddles);
    let ntt_prod = ntt_a.pointwise_mul(&ntt_b, &primes);
    let ntt_result: R = ntt_prod.to_ring(&primes, &twiddles, &garner);

    assert_eq!(schoolbook, ntt_result);
}
