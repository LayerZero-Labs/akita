use rand::{rngs::StdRng, SeedableRng};

use akita_algebra::tables::Q32_MODULUS;
use akita_algebra::CyclotomicRing;
use akita_field::{Fp32, Fp64};

#[test]
fn cyclotomic_ring_negacyclic_property() {
    type F = Fp32<251>;
    type R = CyclotomicRing<F, 4>;

    // X in the ring: [0, 1, 0, 0]
    let x = R::x();

    // X^2
    let x2 = x * x;
    let expected_x2 = R::from_coefficients([F::zero(), F::zero(), F::one(), F::zero()]);
    assert_eq!(x2, expected_x2);

    // X^4 should equal -1 (because X^4 + 1 = 0 in the ring)
    let x4 = x2 * x2;
    assert_eq!(x4, -R::one(), "X^D should equal -1 in Z_q[X]/(X^D + 1)");
}

#[test]
fn cyclotomic_ring_mul_identity() {
    type F = Fp32<251>;
    type R = CyclotomicRing<F, 4>;

    let a = R::from_coefficients([
        F::from_u64(3),
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(42),
    ]);
    assert_eq!(a * R::one(), a);
    assert_eq!(R::one() * a, a);
}

#[test]
fn cyclotomic_ring_mul_zero() {
    type F = Fp32<251>;
    type R = CyclotomicRing<F, 4>;

    let a = R::from_coefficients([
        F::from_u64(3),
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(42),
    ]);
    assert_eq!(a * R::zero(), R::zero());
}

#[test]
fn cyclotomic_ring_commutativity() {
    type F = Fp32<251>;
    type R = CyclotomicRing<F, 4>;

    let a = R::from_coefficients([
        F::from_u64(3),
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(42),
    ]);
    let b = R::from_coefficients([
        F::from_u64(5),
        F::from_u64(13),
        F::from_u64(99),
        F::from_u64(1),
    ]);
    assert_eq!(a * b, b * a);
}

#[test]
fn cyclotomic_ring_distributivity() {
    type F = Fp32<251>;
    type R = CyclotomicRing<F, 4>;

    let a = R::from_coefficients([
        F::from_u64(3),
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(42),
    ]);
    let b = R::from_coefficients([
        F::from_u64(5),
        F::from_u64(13),
        F::from_u64(99),
        F::from_u64(1),
    ]);
    let c = R::from_coefficients([
        F::from_u64(2),
        F::from_u64(9),
        F::from_u64(50),
        F::from_u64(77),
    ]);
    assert_eq!(a * (b + c), a * b + a * c);
}

#[test]
fn cyclotomic_ring_associativity() {
    type F = Fp32<251>;
    type R = CyclotomicRing<F, 4>;

    let a = R::from_coefficients([
        F::from_u64(3),
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(42),
    ]);
    let b = R::from_coefficients([
        F::from_u64(5),
        F::from_u64(13),
        F::from_u64(99),
        F::from_u64(1),
    ]);
    let c = R::from_coefficients([
        F::from_u64(2),
        F::from_u64(9),
        F::from_u64(50),
        F::from_u64(77),
    ]);
    assert_eq!((a * b) * c, a * (b * c));
}

#[test]
fn cyclotomic_ring_additive_inverse() {
    type F = Fp32<251>;
    type R = CyclotomicRing<F, 4>;

    let a = R::from_coefficients([
        F::from_u64(3),
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(42),
    ]);
    assert_eq!(a + (-a), R::zero());
}

#[test]
fn cyclotomic_ring_degree_64() {
    type F = Fp64<4294967197>;
    type R = CyclotomicRing<F, 64>;

    // X^64 = -1 in Z_q[X]/(X^64 + 1)
    let x = R::x();
    let mut power = R::one();
    for _ in 0..64 {
        power *= x;
    }
    assert_eq!(power, -R::one(), "X^64 should equal -1");
}

#[test]
fn cyclotomic_sigma_is_ring_automorphism() {
    type F = Fp32<251>;
    type R = CyclotomicRing<F, 8>;
    let a = R::from_coefficients(std::array::from_fn(|i| F::from_u64((3 * i + 1) as u64)));
    let b = R::from_coefficients(std::array::from_fn(|i| F::from_u64((5 * i + 2) as u64)));

    let k1 = 3usize;
    let k2 = 5usize;
    let two_d = 16usize;

    assert_eq!(a.sigma(1), a);
    assert_eq!(a.sigma_m1().sigma_m1(), a);
    assert_eq!(a.sigma(k1).sigma(k2), a.sigma((k1 * k2) % two_d));
    assert_eq!((a * b).sigma(k1), a.sigma(k1) * b.sigma(k1));
}

#[test]
fn cyclotomic_balanced_pow2_decompose_recompose_round_trip() {
    type F = Fp64<{ Q32_MODULUS }>;
    type R = CyclotomicRing<F, 64>;

    let ring = R::from_coefficients(std::array::from_fn(|i| {
        F::from_u64(((i as u64 * 73) + 17) % Q32_MODULUS)
    }));

    // Q32 balanced base-16: 9 levels absorb the carry-out near q/2.
    let digits = ring.balanced_decompose_pow2(9, 4);
    let round_trip = R::gadget_recompose_pow2(&digits, 4);
    assert_eq!(round_trip, ring);
}

#[test]
fn sparse_pm1_challenge_has_expected_weight() {
    type F = Fp32<251>;
    type R = CyclotomicRing<F, 64>;

    let mut rng = StdRng::seed_from_u64(123);
    let challenge = R::sample_sparse_pm1(&mut rng, 11);
    assert_eq!(challenge.hamming_weight(), 11);

    for c in challenge.coefficients() {
        let x = c.to_canonical_u32();
        if x != 0 {
            assert!(x == 1 || x == 250, "nonzero coefficient must be +/-1");
        }
    }
}

#[test]
fn negacyclic_shift_equals_mul_by_monomial() {
    type F = Fp32<251>;
    type R = CyclotomicRing<F, 8>;

    let a = R::from_coefficients(std::array::from_fn(|i| F::from_u64((3 * i + 1) as u64)));

    for k in 0..32 {
        let k = k % 16;
        let mut monomial_coeffs = [F::zero(); 8];
        monomial_coeffs[k % 8] = if k >= 8 { -F::one() } else { F::one() };
        let monomial = R::from_coefficients(monomial_coeffs);
        assert_eq!(
            a.negacyclic_shift(k),
            a * monomial,
            "negacyclic_shift({k}) != mul by X^{k}"
        );
    }

    assert_eq!(a.negacyclic_shift(0), a);
    assert_eq!(
        a.negacyclic_shift(8),
        -a,
        "shift by D should negate (X^D = -1)"
    );
}

#[test]
fn negacyclic_shift_degree_64() {
    type F = Fp64<4294967197>;
    type R = CyclotomicRing<F, 64>;

    let a = R::from_coefficients(std::array::from_fn(|i| F::from_u64((7 * i + 3) as u64)));
    let x = R::x();
    let mut x_pow = R::one();
    for k in 0..64 {
        assert_eq!(
            a.negacyclic_shift(k),
            a * x_pow,
            "negacyclic_shift({k}) mismatch at D=64"
        );
        x_pow *= x;
    }
}

#[test]
fn mul_by_monomial_sum_matches_ring_mul() {
    type F = Fp32<251>;
    type R = CyclotomicRing<F, 8>;

    let a = R::from_coefficients(std::array::from_fn(|i| F::from_u64((5 * i + 2) as u64)));

    // Sum of X^1 + X^3 + X^5
    let positions = [1, 3, 5];
    let mut sparse = [F::zero(); 8];
    for &p in &positions {
        sparse[p] = F::one();
    }
    let sparse_ring = R::from_coefficients(sparse);

    assert_eq!(
        a.mul_by_monomial_sum(&positions),
        a * sparse_ring,
        "mul_by_monomial_sum should equal ring mul by sparse element"
    );
}

#[test]
fn mul_by_monomial_sum_single_position_equals_shift() {
    type F = Fp32<251>;
    type R = CyclotomicRing<F, 8>;

    let a = R::from_coefficients(std::array::from_fn(|i| F::from_u64((i + 1) as u64)));
    for k in 0..8 {
        assert_eq!(
            a.mul_by_monomial_sum(&[k]),
            a.negacyclic_shift(k),
            "single-position monomial_sum should equal negacyclic_shift"
        );
    }
}

#[test]
fn mul_by_monomial_sum_empty_is_zero() {
    type F = Fp32<251>;
    type R = CyclotomicRing<F, 8>;

    let a = R::from_coefficients(std::array::from_fn(|i| F::from_u64((i + 1) as u64)));
    assert_eq!(a.mul_by_monomial_sum(&[]), R::zero());
}

#[test]
fn is_zero_detects_zero_and_nonzero() {
    type F = Fp32<251>;
    type R = CyclotomicRing<F, 8>;

    assert!(R::zero().is_zero());
    assert!(!R::one().is_zero());

    let a = R::from_coefficients(std::array::from_fn(|i| F::from_u64(i as u64)));
    assert!(!a.is_zero());
}

#[test]
fn kron_scalars_matches_kron_row_constant_rings() {
    type F = Fp64<4294967197>;
    type R = CyclotomicRing<F, 16>;

    let scalars_a: Vec<F> = (0..4).map(|i| F::from_u64(i * 3 + 1)).collect();
    let scalars_b: Vec<F> = (0..3).map(|i| F::from_u64(i * 7 + 2)).collect();

    let rings_a: Vec<R> = scalars_a
        .iter()
        .map(|&s| {
            let mut c = [F::zero(); 16];
            c[0] = s;
            R::from_coefficients(c)
        })
        .collect();
    let rings_b: Vec<R> = scalars_b
        .iter()
        .map(|&s| {
            let mut c = [F::zero(); 16];
            c[0] = s;
            R::from_coefficients(c)
        })
        .collect();

    let via_ring: Vec<R> = rings_a
        .iter()
        .flat_map(|l| rings_b.iter().map(move |r| *l * *r))
        .collect();

    let via_scalar: Vec<R> = scalars_a
        .iter()
        .flat_map(|&l| {
            scalars_b.iter().map(move |&r| {
                let mut c = [F::zero(); 16];
                c[0] = l * r;
                R::from_coefficients(c)
            })
        })
        .collect();

    assert_eq!(via_ring, via_scalar);
}
