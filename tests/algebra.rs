#![allow(missing_docs)]

use hachi_pcs::algebra::{Fp128, Fp2, Fp2Config, Fp32, Fp4, Fp4Config, Fp64, VectorModule};
use hachi_pcs::Field;
use hachi_pcs::Module;

#[test]
fn fp32_basic_arith() {
    type F = Fp32<103>;
    let a = F::from_u64(17);
    let b = F::from_u64(99);
    assert_eq!((a + b).to_canonical_u32(), (17 + 99) % 103);
    assert_eq!((a * b).to_canonical_u32(), (17 * 99) % 103);

    let inv = a.inv().unwrap();
    assert_eq!(a * inv, F::one());
}

#[test]
fn fp64_hachi_q_inv() {
    type F = Fp64<4294967197>;
    let two = F::from_u64(2);
    let inv2 = two.inv().unwrap();
    assert_eq!(two * inv2, F::one());
}

#[test]
fn fp128_basic_arith() {
    // 2^128 - 159 (commonly used 128-bit prime)
    const P: u128 = 340282366920938463463374607431768211297u128;
    type F = Fp128<P>;

    let a = F::from_u64(123);
    let b = F::from_u64(456);
    let c = a * b + a - b;
    let inv = c.inv().unwrap();
    assert_eq!(c * inv, F::one());
}

struct NR;
impl Fp2Config<Fp32<103>> for NR {
    fn non_residue() -> Fp32<103> {
        -Fp32::<103>::one()
    }
}

struct NR4;
impl Fp4Config<Fp32<103>, NR> for NR4 {
    fn non_residue() -> Fp2<Fp32<103>, NR> {
        // v^2 = u
        Fp2::new(Fp32::<103>::zero(), Fp32::<103>::one())
    }
}

#[test]
fn fp2_fp4_inversion_smoke() {
    type F = Fp32<103>;
    type F2 = Fp2<F, NR>;
    type F4 = Fp4<F, NR, NR4>;

    let x = F2::new(F::from_u64(3), F::from_u64(7));
    let inv = x.inv().unwrap();
    assert!((x * inv) == F2::one());

    let y = F4::new(
        F2::new(F::from_u64(5), F::from_u64(1)),
        F2::new(F::from_u64(2), F::from_u64(9)),
    );
    let invy = y.inv().unwrap();
    assert!((y * invy) == F4::one());
}

#[test]
fn vector_module_ops() {
    type F = Fp32<103>;

    let a = VectorModule::<F, 3>([F::from_u64(1), F::from_u64(2), F::from_u64(3)]);
    let b = VectorModule::<F, 3>([F::from_u64(3), F::from_u64(4), F::from_u64(5)]);

    let c = a + b;
    assert_eq!(c.0[0], F::from_u64(4));

    let d = a.scale(&F::from_u64(7));
    assert_eq!(d.0[1], F::from_u64(14));
}

#[test]
fn inv_zero_returns_none() {
    assert!(Fp32::<103>::zero().inv().is_none());
    assert!(Fp64::<4294967197>::zero().inv().is_none());
    const P128: u128 = 340282366920938463463374607431768211297;
    assert!(Fp128::<P128>::zero().inv().is_none());
}

#[test]
fn field_identities_fp32() {
    type F = Fp32<103>;
    let a = F::from_u64(42);
    let b = F::from_u64(73);
    let c = F::from_u64(11);

    // Additive identity
    assert_eq!(a + F::zero(), a);
    // Multiplicative identity
    assert_eq!(a * F::one(), a);
    // Additive inverse
    assert_eq!(a + (-a), F::zero());
    // Distributivity
    assert_eq!(a * (b + c), a * b + a * c);
    // Commutativity
    assert_eq!(a * b, b * a);
    assert_eq!(a + b, b + a);
}

#[test]
fn field_identities_fp64() {
    type F = Fp64<4294967197>;
    let a = F::from_u64(123456);
    let b = F::from_u64(789012);
    let c = F::from_u64(345678);

    assert_eq!(a + F::zero(), a);
    assert_eq!(a * F::one(), a);
    assert_eq!(a + (-a), F::zero());
    assert_eq!(a * (b + c), a * b + a * c);
}

#[test]
fn field_identities_fp128() {
    const P: u128 = 340282366920938463463374607431768211297;
    type F = Fp128<P>;
    let a = F::from_u64(999999);
    let b = F::from_u64(888888);
    let c = F::from_u64(777777);

    assert_eq!(a + F::zero(), a);
    assert_eq!(a * F::one(), a);
    assert_eq!(a + (-a), F::zero());
    assert_eq!(a * (b + c), a * b + a * c);
}

#[test]
fn serialization_round_trip_fp32() {
    use hachi_pcs::{HachiDeserialize, HachiSerialize};
    type F = Fp32<103>;
    let val = F::from_u64(42);
    let mut buf = Vec::new();
    val.serialize_compressed(&mut buf).unwrap();
    let restored = F::deserialize_compressed(&buf[..]).unwrap();
    assert_eq!(val, restored);
}

#[test]
fn serialization_round_trip_fp64() {
    use hachi_pcs::{HachiDeserialize, HachiSerialize};
    type F = Fp64<4294967197>;
    let val = F::from_u64(123456789);
    let mut buf = Vec::new();
    val.serialize_compressed(&mut buf).unwrap();
    let restored = F::deserialize_compressed(&buf[..]).unwrap();
    assert_eq!(val, restored);
}

#[test]
fn serialization_round_trip_fp128() {
    use hachi_pcs::{HachiDeserialize, HachiSerialize};
    const P: u128 = 340282366920938463463374607431768211297;
    type F = Fp128<P>;
    let val = F::from_u64(999999999);
    let mut buf = Vec::new();
    val.serialize_compressed(&mut buf).unwrap();
    let restored = F::deserialize_compressed(&buf[..]).unwrap();
    assert_eq!(val, restored);
}

#[test]
fn serialization_round_trip_ext() {
    use hachi_pcs::{HachiDeserialize, HachiSerialize};
    type F = Fp32<103>;
    type F2 = Fp2<F, NR>;
    let val = F2::new(F::from_u64(3), F::from_u64(7));
    let mut buf = Vec::new();
    val.serialize_compressed(&mut buf).unwrap();
    let restored = F2::deserialize_compressed(&buf[..]).unwrap();
    assert!(val == restored);
}

#[test]
fn serialization_round_trip_vector_module() {
    use hachi_pcs::{HachiDeserialize, HachiSerialize};
    type F = Fp32<103>;
    let val = VectorModule::<F, 3>([F::from_u64(1), F::from_u64(2), F::from_u64(3)]);
    let mut buf = Vec::new();
    val.serialize_compressed(&mut buf).unwrap();
    let restored = VectorModule::<F, 3>::deserialize_compressed(&buf[..]).unwrap();
    assert_eq!(val, restored);
}

#[test]
fn fp2_conjugate_and_norm() {
    type F = Fp32<103>;
    type F2 = Fp2<F, NR>;
    let x = F2::new(F::from_u64(3), F::from_u64(7));
    let conj = x.conjugate();
    assert!(conj == F2::new(F::from_u64(3), -F::from_u64(7)));
    // For Fp2 with u^2 = -1: norm = c0^2 + c1^2 = 9 + 49 = 58
    assert_eq!(x.norm(), F::from_u64(58));
    // x * conjugate(x) should embed the norm into Fp2
    let prod = x * conj;
    assert!(prod == F2::new(F::from_u64(58), F::zero()));
}

#[test]
fn fp2_distributivity() {
    type F = Fp32<103>;
    type F2 = Fp2<F, NR>;
    let a = F2::new(F::from_u64(3), F::from_u64(7));
    let b = F2::new(F::from_u64(11), F::from_u64(5));
    let c = F2::new(F::from_u64(2), F::from_u64(9));
    assert!(a * (b + c) == a * b + a * c);
}

#[test]
fn u256_mul_known_values() {
    use hachi_pcs::algebra::U256;
    // Small values: 3 * 7 = 21
    let r = U256::mul_u128(3, 7);
    assert_eq!(r, U256::new(0, 21));

    // 2^64 * 2^64 = 2^128
    let r = U256::mul_u128(1u128 << 64, 1u128 << 64);
    assert_eq!(r, U256::new(1, 0));

    // max * 2 = 2^129 - 2
    let max = u128::MAX;
    let r = U256::mul_u128(max, 2);
    assert_eq!(r, U256::new(1, max - 1));
}

#[test]
fn u256_bit_access() {
    use hachi_pcs::algebra::U256;
    let v = U256::new(0, 1);
    assert!(v.bit(0));
    assert!(!v.bit(1));

    let v = U256::new(1, 0);
    assert!(v.bit(128));
    assert!(!v.bit(127));
    assert!(!v.bit(129));
}

#[test]
fn limbq_from_to_u128_round_trip() {
    use hachi_pcs::algebra::LimbQ;
    for &val in &[0u128, 1, 12345, 123456789, (1u128 << 28) - 1] {
        let limb: LimbQ<3> = LimbQ::from_u128(val);
        assert_eq!(limb.to_u128(), Some(val), "round-trip failed for {val}");
    }
}

#[test]
fn limbq_add_sub_inverse() {
    use hachi_pcs::algebra::LimbQ;
    let a: LimbQ<3> = LimbQ::from_u128(12345);
    let b: LimbQ<3> = LimbQ::from_u128(6789);
    let sum = a.add_limbs(b);
    let diff = sum.sub_limbs(b);
    assert_eq!(diff, a);
}

#[test]
fn qdata_q_matches_const() {
    use hachi_pcs::algebra::tables::{labrador32_q_u64, LABRADOR32_QDATA};
    let q_from_data = LABRADOR32_QDATA.q_u128().unwrap();
    assert_eq!(q_from_data, labrador32_q_u64() as u128);
}

#[test]
fn ntt_normalize_in_range() {
    use hachi_pcs::algebra::tables::LABRADOR32_PRIMES;
    for prime in &LABRADOR32_PRIMES {
        for &a in &[0i16, 1, -1, 100, -100, prime.p - 1, -(prime.p - 1)] {
            let n = prime.normalize(a);
            assert!(
                n >= 0 && n < prime.p,
                "normalize({a}) = {n} for p={}",
                prime.p
            );
        }
    }
}

#[test]
fn ntt_fpmul_commutative() {
    use hachi_pcs::algebra::tables::LABRADOR32_PRIMES;
    let prime = LABRADOR32_PRIMES[0];
    assert_eq!(prime.fpmul(1234, 5678), prime.fpmul(5678, 1234));
}

#[test]
fn poly_add_sub_neg() {
    use hachi_pcs::algebra::poly::Poly;
    type F = Fp32<103>;
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
