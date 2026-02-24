#![allow(missing_docs)]

#[cfg(test)]
mod tests {
    use hachi_pcs::algebra::{Fp128, Fp2, Fp2Config, Fp32, Fp4, Fp4Config, Fp64, VectorModule};
    use hachi_pcs::Module;
    use hachi_pcs::{CanonicalField, FieldCore, Invertible};

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
    fn inv_or_zero_behavior_for_prime_fields() {
        type F32 = Fp32<103>;
        assert_eq!(F32::zero().inv_or_zero(), F32::zero());
        let x32 = F32::from_u64(17);
        let inv32 = x32.inv_or_zero();
        assert_eq!(x32 * inv32, F32::one());

        type F64 = Fp64<4294967197>;
        assert_eq!(F64::zero().inv_or_zero(), F64::zero());
        let x64 = F64::from_u64(2);
        let inv64 = x64.inv_or_zero();
        assert_eq!(x64 * inv64, F64::one());

        const P128: u128 = 340282366920938463463374607431768211297;
        type F128 = Fp128<P128>;
        assert_eq!(F128::zero().inv_or_zero(), F128::zero());
        let x128 = F128::from_u64(12345);
        let inv128 = x128.inv_or_zero();
        assert_eq!(x128 * inv128, F128::one());
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
    fn serialization_round_trip_fp4() {
        use hachi_pcs::{HachiDeserialize, HachiSerialize};
        type F = Fp32<103>;
        type F2 = Fp2<F, NR>;
        type F4 = Fp4<F, NR, NR4>;

        let val = F4::new(
            F2::new(F::from_u64(5), F::from_u64(1)),
            F2::new(F::from_u64(2), F::from_u64(9)),
        );
        let mut buf = Vec::new();
        val.serialize_compressed(&mut buf).unwrap();
        let restored = F4::deserialize_compressed(&buf[..]).unwrap();
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
    fn serialization_round_trip_poly() {
        use hachi_pcs::algebra::poly::Poly;
        use hachi_pcs::{HachiDeserialize, HachiSerialize};
        type F = Fp32<103>;

        let val = Poly::<F, 4>([
            F::from_u64(7),
            F::from_u64(11),
            F::from_u64(13),
            F::from_u64(29),
        ]);
        let mut buf = Vec::new();
        val.serialize_compressed(&mut buf).unwrap();
        let restored = Poly::<F, 4>::deserialize_compressed(&buf[..]).unwrap();
        assert_eq!(val, restored);
    }

    #[test]
    fn deserialize_checked_rejects_non_canonical_field_elements() {
        use hachi_pcs::primitives::serialization::SerializationError;
        use hachi_pcs::HachiDeserialize;

        type F32 = Fp32<103>;
        let bad32 = 103u32.to_le_bytes();
        let err32 = F32::deserialize_compressed(&bad32[..]).unwrap_err();
        assert!(matches!(err32, SerializationError::InvalidData(_)));
        let unchecked32 = F32::deserialize_compressed_unchecked(&bad32[..]).unwrap();
        assert_eq!(unchecked32, F32::zero());

        type F64 = Fp64<4294967197>;
        let bad64 = 4294967197u64.to_le_bytes();
        let err64 = F64::deserialize_compressed(&bad64[..]).unwrap_err();
        assert!(matches!(err64, SerializationError::InvalidData(_)));
        let unchecked64 = F64::deserialize_compressed_unchecked(&bad64[..]).unwrap();
        assert_eq!(unchecked64, F64::zero());

        const P128: u128 = 340282366920938463463374607431768211297u128;
        type F128 = Fp128<P128>;
        let bad128 = P128.to_le_bytes();
        let err128 = F128::deserialize_compressed(&bad128[..]).unwrap_err();
        assert!(matches!(err128, SerializationError::InvalidData(_)));
        let unchecked128 = F128::deserialize_compressed_unchecked(&bad128[..]).unwrap();
        assert_eq!(unchecked128, F128::zero());
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
        use hachi_pcs::algebra::LimbQ;
        let a: LimbQ<3> = LimbQ::from(12345u128);
        let b: LimbQ<3> = LimbQ::from(6789u128);
        let sum = a + b;
        let diff = sum - b;
        assert_eq!(diff, a);
    }

    #[test]
    fn limbq_ordering() {
        use hachi_pcs::algebra::LimbQ;
        let a: LimbQ<3> = LimbQ::from(100u128);
        let b: LimbQ<3> = LimbQ::from(200u128);
        assert!(a < b);
        assert!(b > a);
        assert_eq!(a, a);
    }

    #[test]
    fn qdata_q_matches_const() {
        use hachi_pcs::algebra::tables::{Q32_DATA, Q32_MODULUS};
        let q_from_data = Q32_DATA.q_u128().unwrap();
        assert_eq!(q_from_data, Q32_MODULUS as u128);
    }

    #[test]
    fn ntt_normalize_in_range() {
        use hachi_pcs::algebra::tables::Q32_PRIMES;
        use hachi_pcs::algebra::MontCoeff;
        for prime in &Q32_PRIMES {
            for &a in &[0i16, 1, -1, 100, -100, prime.p - 1, -(prime.p - 1)] {
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
    fn ntt_mul_commutative() {
        use hachi_pcs::algebra::tables::Q32_PRIMES;
        use hachi_pcs::algebra::MontCoeff;
        let prime = Q32_PRIMES[0];
        let a = MontCoeff::from_raw(1234);
        let b = MontCoeff::from_raw(5678);
        assert_eq!(prime.mul(a, b), prime.mul(b, a));
    }

    #[test]
    fn mont_coeff_round_trip() {
        use hachi_pcs::algebra::tables::Q32_PRIMES;
        for prime in &Q32_PRIMES {
            for &val in &[0i16, 1, 2, 100, prime.p - 1] {
                let mont = prime.from_canonical(val);
                let back = prime.to_canonical(mont);
                assert_eq!(back, val, "round-trip failed for val={val}, p={}", prime.p);
            }
        }
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

    #[test]
    fn cyclotomic_ring_negacyclic_property() {
        use hachi_pcs::algebra::CyclotomicRing;
        type F = Fp32<103>;
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
        use hachi_pcs::algebra::CyclotomicRing;
        type F = Fp32<103>;
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
        use hachi_pcs::algebra::CyclotomicRing;
        type F = Fp32<103>;
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
        use hachi_pcs::algebra::CyclotomicRing;
        type F = Fp32<103>;
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
        use hachi_pcs::algebra::CyclotomicRing;
        type F = Fp32<103>;
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
        use hachi_pcs::algebra::CyclotomicRing;
        type F = Fp32<103>;
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
        use hachi_pcs::algebra::CyclotomicRing;
        type F = Fp32<103>;
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
    fn cyclotomic_ring_serialization_round_trip() {
        use hachi_pcs::algebra::CyclotomicRing;
        use hachi_pcs::{HachiDeserialize, HachiSerialize};
        type F = Fp32<103>;
        type R = CyclotomicRing<F, 4>;

        let a = R::from_coefficients([
            F::from_u64(3),
            F::from_u64(7),
            F::from_u64(11),
            F::from_u64(42),
        ]);
        let mut buf = Vec::new();
        a.serialize_compressed(&mut buf).unwrap();
        let restored = R::deserialize_compressed(&buf[..]).unwrap();
        assert_eq!(a, restored);
    }

    #[test]
    fn cyclotomic_ring_degree_64() {
        use hachi_pcs::algebra::CyclotomicRing;
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
    fn ntt_forward_inverse_round_trip() {
        use hachi_pcs::algebra::ntt::butterfly::{forward_ntt, inverse_ntt, NttTwiddles};
        use hachi_pcs::algebra::tables::Q32_PRIMES;
        use hachi_pcs::algebra::MontCoeff;

        let prime = Q32_PRIMES[0];
        let tw = NttTwiddles::<64>::compute(prime);

        // Create a test polynomial in Montgomery form.
        let original: [MontCoeff; 64] =
            std::array::from_fn(|i| prime.from_canonical((i as i16) % prime.p));

        // Forward then inverse should give back the original.
        let mut a = original;
        forward_ntt(&mut a, prime, &tw);
        inverse_ntt(&mut a, prime, &tw);

        // Normalize and compare.
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
        use hachi_pcs::algebra::ntt::butterfly::{forward_ntt, inverse_ntt, NttTwiddles};
        use hachi_pcs::algebra::tables::Q32_PRIMES;

        for (pi, prime) in Q32_PRIMES.iter().enumerate() {
            let tw = NttTwiddles::<64>::compute(*prime);

            let original: [_; 64] =
                std::array::from_fn(|i| prime.from_canonical(((i * (pi + 1)) as i16) % prime.p));

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
    fn ntt_mul_matches_schoolbook() {
        use hachi_pcs::algebra::ntt::butterfly::{forward_ntt, inverse_ntt, NttTwiddles};
        use hachi_pcs::algebra::tables::Q32_PRIMES;
        use hachi_pcs::algebra::{CyclotomicRing, MontCoeff};

        type F = Fp32<{ Q32_PRIMES[0].p as u32 }>;

        let prime = Q32_PRIMES[0];
        let tw = NttTwiddles::<64>::compute(prime);

        // Two test polynomials in the ring Z_p[X]/(X^64 + 1).
        let a_coeffs: [F; 64] =
            std::array::from_fn(|i| F::from_u64((i as u64 + 1) % prime.p as u64));
        let b_coeffs: [F; 64] =
            std::array::from_fn(|i| F::from_u64(((i * 3 + 7) as u64) % prime.p as u64));

        // Schoolbook multiplication.
        let ring_a = CyclotomicRing::<F, 64>::from_coefficients(a_coeffs);
        let ring_b = CyclotomicRing::<F, 64>::from_coefficients(b_coeffs);
        let schoolbook = ring_a * ring_b;

        // NTT multiplication (single prime, not full CRT).
        let mut ntt_a: [MontCoeff; 64] =
            std::array::from_fn(|i| prime.from_canonical(a_coeffs[i].to_canonical_u32() as i16));
        let mut ntt_b: [MontCoeff; 64] =
            std::array::from_fn(|i| prime.from_canonical(b_coeffs[i].to_canonical_u32() as i16));
        forward_ntt(&mut ntt_a, prime, &tw);
        forward_ntt(&mut ntt_b, prime, &tw);

        // Pointwise multiply.
        let mut ntt_c = [MontCoeff::from_raw(0); 64];
        prime.pointwise_mul(&mut ntt_c, &ntt_a, &ntt_b);

        // Inverse NTT.
        inverse_ntt(&mut ntt_c, prime, &tw);

        // Compare.
        for (i, c) in ntt_c.iter().enumerate() {
            let ntt_val = prime.to_canonical(prime.normalize(*c));
            let school_val = schoolbook.coefficients()[i].to_canonical_u32() as i16;
            assert_eq!(
                ntt_val, school_val,
                "NTT vs schoolbook mismatch at index {i}: NTT={ntt_val}, school={school_val}"
            );
        }
    }

    #[test]
    fn cyclotomic_ntt_crt_round_trip_q32() {
        use hachi_pcs::algebra::ntt::butterfly::NttTwiddles;
        use hachi_pcs::algebra::tables::{Q32_DATA, Q32_MODULUS, Q32_NUM_PRIMES, Q32_PRIMES};
        use hachi_pcs::algebra::{CyclotomicCrtNtt, CyclotomicRing};

        type F = Fp64<{ Q32_MODULUS }>;
        type R = CyclotomicRing<F, 64>;
        type N = CyclotomicCrtNtt<Q32_NUM_PRIMES, 64>;

        let twiddles: [NttTwiddles<64>; Q32_NUM_PRIMES] =
            std::array::from_fn(|k| NttTwiddles::<64>::compute(Q32_PRIMES[k]));

        let coeffs: [F; 64] =
            std::array::from_fn(|i| F::from_u64(((i as u64 * 17) + 5) % Q32_MODULUS));
        let ring = R::from_coefficients(coeffs);
        let ntt = N::from_ring(&ring, &Q32_PRIMES, &twiddles);
        let round_trip = ntt.to_ring(&Q32_PRIMES, &twiddles, &Q32_DATA);

        assert_eq!(ring, round_trip);
    }

    #[test]
    fn cyclotomic_ntt_reduced_ops_are_stable() {
        use hachi_pcs::algebra::ntt::butterfly::NttTwiddles;
        use hachi_pcs::algebra::tables::{Q32_DATA, Q32_MODULUS, Q32_NUM_PRIMES, Q32_PRIMES};
        use hachi_pcs::algebra::{CyclotomicCrtNtt, CyclotomicRing};

        type F = Fp64<{ Q32_MODULUS }>;
        type R = CyclotomicRing<F, 64>;
        type N = CyclotomicCrtNtt<Q32_NUM_PRIMES, 64>;

        let twiddles: [NttTwiddles<64>; Q32_NUM_PRIMES] =
            std::array::from_fn(|k| NttTwiddles::<64>::compute(Q32_PRIMES[k]));

        let a = R::from_coefficients(std::array::from_fn(|i| {
            F::from_u64(((i as u64 * 3) + 1) % Q32_MODULUS)
        }));
        let b = R::from_coefficients(std::array::from_fn(|i| {
            F::from_u64(((i as u64 * 11) + 7) % Q32_MODULUS)
        }));

        let ntt_a = N::from_ring(&a, &Q32_PRIMES, &twiddles);
        let ntt_b = N::from_ring(&b, &Q32_PRIMES, &twiddles);

        let sum = ntt_a.add_reduced(&ntt_b, &Q32_PRIMES);
        let back = sum.sub_reduced(&ntt_b, &Q32_PRIMES);
        assert_eq!(back, ntt_a);

        let zero_ntt = ntt_a.add_reduced(&ntt_a.neg_reduced(&Q32_PRIMES), &Q32_PRIMES);
        let zero_ring = zero_ntt.to_ring(&Q32_PRIMES, &twiddles, &Q32_DATA);
        assert_eq!(zero_ring, R::zero());
    }

    #[test]
    fn backend_path_matches_default_scalar_path() {
        use hachi_pcs::algebra::ntt::butterfly::NttTwiddles;
        use hachi_pcs::algebra::tables::{Q32_DATA, Q32_MODULUS, Q32_NUM_PRIMES, Q32_PRIMES};
        use hachi_pcs::algebra::{CyclotomicCrtNtt, CyclotomicRing, ScalarBackend};

        type F = Fp64<{ Q32_MODULUS }>;
        type R = CyclotomicRing<F, 64>;
        type N = CyclotomicCrtNtt<Q32_NUM_PRIMES, 64>;

        let twiddles: [NttTwiddles<64>; Q32_NUM_PRIMES] =
            std::array::from_fn(|k| NttTwiddles::<64>::compute(Q32_PRIMES[k]));
        let ring = R::from_coefficients(std::array::from_fn(|i| {
            F::from_u64(((i as u64 * 13) + 9) % Q32_MODULUS)
        }));

        let default_ntt = N::from_ring(&ring, &Q32_PRIMES, &twiddles);
        let backend_ntt =
            N::from_ring_with_backend::<F, ScalarBackend>(&ring, &Q32_PRIMES, &twiddles);
        assert_eq!(default_ntt, backend_ntt);

        let default_back = default_ntt.to_ring(&Q32_PRIMES, &twiddles, &Q32_DATA);
        let backend_back = backend_ntt.to_ring_with_backend::<F, ScalarBackend, 3>(
            &Q32_PRIMES,
            &twiddles,
            &Q32_DATA,
        );
        assert_eq!(default_back, backend_back);
    }

    #[test]
    fn field_sampling_respects_modulus() {
        use hachi_pcs::FieldSampling;
        use rand::{rngs::StdRng, SeedableRng};

        type F = Fp32<103>;
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..1024 {
            let x = F::sample(&mut rng);
            assert!(x.to_canonical_u32() < 103);
        }
    }

    #[test]
    fn pow2_offset_registry_is_consistent() {
        use hachi_pcs::algebra::{
            pseudo_mersenne_modulus, Pow2Offset128Field, Pow2OffsetPrimeSpec, POW2_OFFSET_MAX,
            POW2_OFFSET_PRIMES, POW2_OFFSET_TABLE,
        };
        use hachi_pcs::{CanonicalField, PseudoMersenneField};

        fn assert_is_pseudo_mersenne<F: PseudoMersenneField>() {}
        assert_is_pseudo_mersenne::<Pow2Offset128Field>();

        for Pow2OffsetPrimeSpec {
            bits,
            offset,
            modulus,
            ..
        } in POW2_OFFSET_PRIMES
        {
            assert!((offset as u128) <= POW2_OFFSET_MAX);
            assert_eq!(POW2_OFFSET_TABLE[bits as usize], offset as i16);
            assert_eq!(
                Some(modulus),
                pseudo_mersenne_modulus(bits, offset as u128),
                "2^k-offset modulus mismatch for k={bits}, offset={offset}"
            );
            assert_eq!(modulus % 8, 5);
        }

        let x = Pow2Offset128Field::from_u64(1234567);
        let inv = x.inv().unwrap();
        assert_eq!(x * inv, Pow2Offset128Field::one());
    }

    #[test]
    fn cyclotomic_sigma_is_ring_automorphism() {
        use hachi_pcs::algebra::CyclotomicRing;

        type F = Fp32<103>;
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
        use hachi_pcs::algebra::tables::Q32_MODULUS;
        use hachi_pcs::algebra::CyclotomicRing;

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
        use hachi_pcs::algebra::CyclotomicRing;
        use rand::{rngs::StdRng, SeedableRng};

        type F = Fp32<103>;
        type R = CyclotomicRing<F, 64>;

        let mut rng = StdRng::seed_from_u64(123);
        let challenge = R::sample_sparse_pm1(&mut rng, 11);
        assert_eq!(challenge.hamming_weight(), 11);

        for c in challenge.coefficients() {
            let x = c.to_canonical_u32();
            if x != 0 {
                assert!(x == 1 || x == 102, "nonzero coefficient must be +/-1");
            }
        }
    }
}
