use akita_algebra::CyclotomicRing;
use jolt_field::{Fp32, Fp64, One, Ring, Zero};

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
    for k in 0..128 {
        assert_eq!(
            a.negacyclic_shift(k),
            a * x_pow,
            "negacyclic_shift({k}) mismatch at D=64"
        );
        x_pow *= x;
    }
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
