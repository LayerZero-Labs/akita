use rand::{rngs::StdRng, SeedableRng};

use akita_algebra::{Module, VectorModule};
use jolt_field::{
    pseudo_mersenne_modulus, Fp32, Fp64, FpExt2, FpExt4, Invertible, Prime128Offset159,
    Prime128Offset2355, Prime128Offset275, Prime128OffsetA7F7, PrimeOffsetSpec,
    PseudoMersenneField, RandomSampling, PRIME_OFFSET_MAX, PRIME_OFFSET_SPECS,
};

use super::fixtures::{check_solinas_prime, NR};

#[test]
fn fp32_basic_arith() {
    type F = Fp32<251>;
    let a = F::from_u64(17);
    let b = F::from_u64(99);
    assert_eq!((a + b).to_canonical_u32(), 116);
    assert_eq!((a * b).to_canonical_u32(), (17u32 * 99) % 251);

    let inv = a.inverse().unwrap();
    assert_eq!(a * inv, F::one());
}

#[test]
fn fp64_akita_q_inv() {
    type F = Fp64<4294967197>;
    let two = F::from_u64(2);
    let inv2 = two.inverse().unwrap();
    assert_eq!(two * inv2, F::one());
}

#[test]
fn fp128_basic_arith() {
    type F = Prime128Offset275;

    let a = F::from_u64(123);
    let b = F::from_u64(456);
    let c = a * b + a - b;
    let inv = c.inverse().unwrap();
    assert_eq!(c * inv, F::one());
}

#[test]
fn fp128_primes_match_biguint_oracle() {
    const P159: u128 = 0xffffffffffffffffffffffffffffff61u128;
    const P275: u128 = 0xfffffffffffffffffffffffffffffeedu128;
    const P2355: u128 = 0xfffffffffffffffffffffffffffff6cdu128;
    // p_A7F7 = 2^128 - 2^32 + 22537 = 2^128 - 0xFFFFA7F7.
    const P_A7F7: u128 = u128::MAX - 0xFFFFA7F7u128 + 1;
    check_solinas_prime::<Prime128Offset159>(P159, 2_000, 159);
    check_solinas_prime::<Prime128Offset275>(P275, 2_000, 275);
    check_solinas_prime::<Prime128Offset2355>(P2355, 2_000, 2355);
    check_solinas_prime::<Prime128OffsetA7F7>(P_A7F7, 2_000, 0xA7F7);
}

#[test]
fn fp_ext2_fp_ext4_inversion_smoke() {
    type F = Fp32<251>;
    type F2 = FpExt2<F, NR>;
    type F4 = FpExt4<F>;

    let x = F2::new(F::from_u64(3), F::from_u64(7));
    let inv = x.inverse().unwrap();
    assert!((x * inv) == F2::one());

    let y = F4::new([
        F::from_u64(5),
        F::from_u64(1),
        F::from_u64(2),
        F::from_u64(9),
    ]);
    let invy = y.inverse().unwrap();
    assert!((y * invy) == F4::one());
}

#[test]
fn vector_module_ops() {
    type F = Fp32<251>;

    let a = VectorModule::<F, 3>([F::from_u64(1), F::from_u64(2), F::from_u64(3)]);
    let b = VectorModule::<F, 3>([F::from_u64(3), F::from_u64(4), F::from_u64(5)]);

    let c = a + b;
    assert_eq!(c.0[0], F::from_u64(4));

    let d = a.scale(&F::from_u64(7));
    assert_eq!(d.0[1], F::from_u64(14));
}

#[test]
fn inv_zero_returns_none() {
    assert!(Fp32::<251>::zero().inverse().is_none());
    assert!(Fp64::<4294967197>::zero().inverse().is_none());
    assert!(Prime128Offset275::zero().inverse().is_none());
}

#[test]
fn inv_or_zero_behavior_for_prime_fields() {
    type F32 = Fp32<251>;
    assert_eq!(F32::zero().inv_or_zero(), F32::zero());
    let x32 = F32::from_u64(17);
    let inv32 = x32.inv_or_zero();
    assert_eq!(x32 * inv32, F32::one());

    type F64 = Fp64<4294967197>;
    assert_eq!(F64::zero().inv_or_zero(), F64::zero());
    let x64 = F64::from_u64(2);
    let inv64 = x64.inv_or_zero();
    assert_eq!(x64 * inv64, F64::one());

    type F128 = Prime128Offset275;
    assert_eq!(F128::zero().inv_or_zero(), F128::zero());
    let x128 = F128::from_u64(12345);
    let inv128 = x128.inv_or_zero();
    assert_eq!(x128 * inv128, F128::one());
}

#[test]
fn field_identities_fp32() {
    type F = Fp32<251>;
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
    type F = Prime128Offset275;
    let a = F::from_u64(999999);
    let b = F::from_u64(888888);
    let c = F::from_u64(777777);

    assert_eq!(a + F::zero(), a);
    assert_eq!(a * F::one(), a);
    assert_eq!(a + (-a), F::zero());
    assert_eq!(a * (b + c), a * b + a * c);
}

#[test]
fn fp_ext2_conjugate_and_norm() {
    type F = Fp32<251>;
    type F2 = FpExt2<F, NR>;
    let x = F2::new(F::from_u64(3), F::from_u64(7));
    let conj = x.conjugate();
    assert!(conj == F2::new(F::from_u64(3), -F::from_u64(7)));
    // For FpExt2 with u^2 = -1: norm = c0^2 + c1^2 = 9 + 49 = 58
    assert_eq!(x.norm(), F::from_u64(58));
    // x * conjugate(x) should embed the norm into FpExt2
    let prod = x * conj;
    assert!(prod == F2::new(F::from_u64(58), F::zero()));
}

#[test]
fn fp_ext2_distributivity() {
    type F = Fp32<251>;
    type F2 = FpExt2<F, NR>;
    let a = F2::new(F::from_u64(3), F::from_u64(7));
    let b = F2::new(F::from_u64(11), F::from_u64(5));
    let c = F2::new(F::from_u64(2), F::from_u64(9));
    assert!(a * (b + c) == a * b + a * c);
}

#[test]
fn field_sampling_respects_modulus() {
    type F = Fp32<251>;
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..1024 {
        let x = F::random(&mut rng);
        assert!(x.to_canonical_u32() < 251);
    }
}

#[test]
fn prime_offset_registry_is_consistent() {
    fn assert_is_pseudo_mersenne<F: PseudoMersenneField>() {}
    assert_is_pseudo_mersenne::<Prime128Offset275>();

    for PrimeOffsetSpec {
        bits,
        offset,
        modulus,
        ..
    } in PRIME_OFFSET_SPECS
    {
        assert!((offset as u128) <= PRIME_OFFSET_MAX);
        assert_eq!(
            Some(modulus),
            pseudo_mersenne_modulus(bits, offset as u128),
            "2^k-offset modulus mismatch for k={bits}, offset={offset}"
        );
        if bits < 128 {
            assert_eq!(modulus % 8, 5);
        }
    }

    let x = Prime128Offset275::from_u64(1234567);
    let inv = x.inverse().unwrap();
    assert_eq!(x * inv, Prime128Offset275::one());
}
