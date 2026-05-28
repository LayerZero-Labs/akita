#![allow(missing_docs)]
#![cfg(not(feature = "zk"))]

#[cfg(test)]
mod tests {
    use num_bigint::BigUint;
    use rand::{rngs::StdRng, SeedableRng};

    use akita_algebra::backend::{CrtReconstruct, NttPrimeOps};
    use akita_algebra::ntt::butterfly::{forward_ntt, inverse_ntt, NttTwiddles};
    use akita_algebra::poly::Poly;
    use akita_algebra::tables::{
        q128_garner, q128_primes, q32_garner, q64_garner, q64_primes, Q128_MODULUS,
        Q128_NUM_PRIMES, Q32_MODULUS, Q32_NUM_PRIMES, Q32_PRIMES, Q64_MODULUS, Q64_NUM_PRIMES,
    };
    use akita_algebra::{
        CrtNttParamSet, CyclotomicCrtNtt, CyclotomicRing, LimbQ, Module, MontCoeff,
        PackedPartialSplitEval16, PartialSplitEval16, PartialSplitNtt16, ScalarBackend,
        VectorModule,
    };
    use akita_field::{
        pseudo_mersenne_modulus, CanonicalField, FieldCore, Fp128, Fp2, Fp2Config, Fp32, Fp64,
        HasPacking, Invertible, Prime128Offset159, Prime128Offset2355, Prime128Offset275,
        Prime128OffsetA7F7, PrimeOffsetSpec, PseudoMersenneField, RandomSampling, TowerBasisFp4,
        TowerBasisFp4Config, PRIME_OFFSET_MAX, PRIME_OFFSET_SPECS,
    };
    use akita_serialization::SerializationError;
    use akita_serialization::{AkitaDeserialize, AkitaSerialize};

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

    fn rand_u128<R: rand_core::RngCore>(rng: &mut R) -> u128 {
        let lo = rng.next_u64() as u128;
        let hi = rng.next_u64() as u128;
        lo | (hi << 64)
    }

    fn biguint_to_u128(x: &num_bigint::BigUint) -> u128 {
        let mut bytes = x.to_bytes_le();
        bytes.resize(16, 0);
        let mut arr = [0u8; 16];
        arr.copy_from_slice(&bytes[..16]);
        u128::from_le_bytes(arr)
    }

    fn big_mul_mod_u128(a: u128, b: u128, p: u128) -> u128 {
        let n = BigUint::from(a) * BigUint::from(b);
        let r = n % BigUint::from(p);
        biguint_to_u128(&r)
    }

    fn check_solinas_prime<
        S: CanonicalField + FieldCore + Invertible + PseudoMersenneField + std::fmt::Debug,
    >(
        p: u128,
        iters: usize,
        seed: u64,
    ) {
        assert_eq!(<S as PseudoMersenneField>::MODULUS_BITS, 128);
        assert_eq!(
            <S as PseudoMersenneField>::MODULUS_OFFSET,
            0u128.wrapping_sub(p)
        );
        assert_eq!(std::mem::size_of::<S>(), 16);

        let mut rng = StdRng::seed_from_u64(seed);

        for _ in 0..iters {
            let a_raw = rand_u128(&mut rng);
            let b_raw = rand_u128(&mut rng);

            let a = S::from_canonical_u128_reduced(a_raw);
            let b = S::from_canonical_u128_reduced(b_raw);

            // Canonical range invariant.
            assert!(a.to_canonical_u128() < p);
            assert!(b.to_canonical_u128() < p);

            // Add/sub/neg identities.
            assert_eq!(a + S::zero(), a);
            assert_eq!(a - S::zero(), a);
            assert_eq!(a + (-a), S::zero());

            // Multiplicative identity.
            assert_eq!(a * S::one(), a);

            // BigUint oracle for mul and sqr (exercises reduction).
            let aa = a.to_canonical_u128();
            let bb = b.to_canonical_u128();
            let got_mul = (a * b).to_canonical_u128();
            let exp_mul = big_mul_mod_u128(aa, bb, p);
            assert_eq!(got_mul, exp_mul);

            let got_sqr = (a * a).to_canonical_u128();
            let exp_sqr = big_mul_mod_u128(aa, aa, p);
            assert_eq!(got_sqr, exp_sqr);

            // Inversion checks (skip explicit inv on zero).
            let inv = a.inv_or_zero();
            if a.is_zero() {
                assert_eq!(inv, S::zero());
            } else {
                assert_eq!(a * inv, S::one());
                assert_eq!(a.inverse().unwrap(), inv);
            }
        }
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

    struct NR;
    impl Fp2Config<Fp32<251>> for NR {
        fn non_residue() -> Fp32<251> {
            -Fp32::<251>::one()
        }
    }

    struct NR4;
    impl TowerBasisFp4Config<Fp32<251>, NR> for NR4 {
        fn non_residue() -> Fp2<Fp32<251>, NR> {
            Fp2::new(Fp32::<251>::zero(), Fp32::<251>::one())
        }
    }

    #[test]
    fn fp2_fp4_inversion_smoke() {
        type F = Fp32<251>;
        type F2 = Fp2<F, NR>;
        type F4 = TowerBasisFp4<F, NR, NR4>;

        let x = F2::new(F::from_u64(3), F::from_u64(7));
        let inv = x.inverse().unwrap();
        assert!((x * inv) == F2::one());

        let y = F4::new(
            F2::new(F::from_u64(5), F::from_u64(1)),
            F2::new(F::from_u64(2), F::from_u64(9)),
        );
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
    fn serialization_round_trip_fp32() {
        type F = Fp32<251>;
        let val = F::from_u64(42);
        let mut buf = Vec::new();
        val.serialize_compressed(&mut buf).unwrap();
        let restored = F::deserialize_compressed(&buf[..], &()).unwrap();
        assert_eq!(val, restored);
    }

    #[test]
    fn serialization_round_trip_fp64() {
        type F = Fp64<4294967197>;
        let val = F::from_u64(123456789);
        let mut buf = Vec::new();
        val.serialize_compressed(&mut buf).unwrap();
        let restored = F::deserialize_compressed(&buf[..], &()).unwrap();
        assert_eq!(val, restored);
    }

    #[test]
    fn serialization_round_trip_fp128() {
        type F = Prime128Offset275;
        let val = F::from_u64(999999999);
        let mut buf = Vec::new();
        val.serialize_compressed(&mut buf).unwrap();
        let restored = F::deserialize_compressed(&buf[..], &()).unwrap();
        assert_eq!(val, restored);
    }

    #[test]
    fn serialization_round_trip_ext() {
        type F = Fp32<251>;
        type F2 = Fp2<F, NR>;
        let val = F2::new(F::from_u64(3), F::from_u64(7));
        let mut buf = Vec::new();
        val.serialize_compressed(&mut buf).unwrap();
        let restored = F2::deserialize_compressed(&buf[..], &()).unwrap();
        assert!(val == restored);
    }

    #[test]
    fn serialization_round_trip_fp4() {
        type F = Fp32<251>;
        type F2 = Fp2<F, NR>;
        type F4 = TowerBasisFp4<F, NR, NR4>;

        let val = F4::new(
            F2::new(F::from_u64(5), F::from_u64(1)),
            F2::new(F::from_u64(2), F::from_u64(9)),
        );
        let mut buf = Vec::new();
        val.serialize_compressed(&mut buf).unwrap();
        let restored = F4::deserialize_compressed(&buf[..], &()).unwrap();
        assert!(val == restored);
    }

    #[test]
    fn serialization_round_trip_vector_module() {
        type F = Fp32<251>;
        let val = VectorModule::<F, 3>([F::from_u64(1), F::from_u64(2), F::from_u64(3)]);
        let mut buf = Vec::new();
        val.serialize_compressed(&mut buf).unwrap();
        let restored = VectorModule::<F, 3>::deserialize_compressed(&buf[..], &()).unwrap();
        assert_eq!(val, restored);
    }

    #[test]
    fn serialization_round_trip_poly() {
        type F = Fp32<251>;

        let val = Poly::<F, 4>([
            F::from_u64(7),
            F::from_u64(11),
            F::from_u64(13),
            F::from_u64(29),
        ]);
        let mut buf = Vec::new();
        val.serialize_compressed(&mut buf).unwrap();
        let restored = Poly::<F, 4>::deserialize_compressed(&buf[..], &()).unwrap();
        assert_eq!(val, restored);
    }

    #[test]
    fn deserialize_checked_rejects_non_canonical_field_elements() {
        type F32 = Fp32<251>;
        let bad32 = 251u32.to_le_bytes();
        let err32 = F32::deserialize_compressed(&bad32[..], &()).unwrap_err();
        assert!(matches!(err32, SerializationError::InvalidData(_)));
        let unchecked32 = F32::deserialize_compressed_unchecked(&bad32[..], &()).unwrap();
        assert_eq!(unchecked32, F32::zero());

        type F64 = Fp64<4294967197>;
        let bad64 = 4294967197u64.to_le_bytes();
        let err64 = F64::deserialize_compressed(&bad64[..], &()).unwrap_err();
        assert!(matches!(err64, SerializationError::InvalidData(_)));
        let unchecked64 = F64::deserialize_compressed_unchecked(&bad64[..], &()).unwrap();
        assert_eq!(unchecked64, F64::zero());

        type F128 = Prime128Offset275;
        const P275: u128 = 0xfffffffffffffffffffffffffffffeedu128;
        let bad128 = P275.to_le_bytes();
        let err128 = F128::deserialize_compressed(&bad128[..], &()).unwrap_err();
        assert!(matches!(err128, SerializationError::InvalidData(_)));
        let unchecked128 = F128::deserialize_compressed_unchecked(&bad128[..], &()).unwrap();
        assert_eq!(unchecked128, F128::zero());

        type F128b = Prime128Offset275;
        const P275B: u128 = 0xfffffffffffffffffffffffffffffeedu128;
        let bad275 = P275B.to_le_bytes();
        let err275 = F128b::deserialize_compressed(&bad275[..], &()).unwrap_err();
        assert!(matches!(err275, SerializationError::InvalidData(_)));
        let unchecked275 = F128b::deserialize_compressed_unchecked(&bad275[..], &()).unwrap();
        assert_eq!(unchecked275, F128b::zero());
    }

    #[test]
    fn fp2_conjugate_and_norm() {
        type F = Fp32<251>;
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
        type F = Fp32<251>;
        type F2 = Fp2<F, NR>;
        let a = F2::new(F::from_u64(3), F::from_u64(7));
        let b = F2::new(F::from_u64(11), F::from_u64(5));
        let c = F2::new(F::from_u64(2), F::from_u64(9));
        assert!(a * (b + c) == a * b + a * c);
    }

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
    fn csubp_widened_handles_large_negative_i16() {
        for &prime in &Q32_PRIMES {
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
            for &val in &[0i16, 1, 2, 100, prime.p - 1] {
                let mont = prime.from_canonical(val);
                let back = prime.to_canonical(mont);
                assert_eq!(back, val, "round-trip failed for val={val}, p={}", prime.p);
            }
        }
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
    fn cyclotomic_ring_serialization_round_trip() {
        type F = Fp32<251>;
        type R = CyclotomicRing<F, 4>;

        let a = R::from_coefficients([
            F::from_u64(3),
            F::from_u64(7),
            F::from_u64(11),
            F::from_u64(42),
        ]);
        let mut buf = Vec::new();
        a.serialize_compressed(&mut buf).unwrap();
        let restored = R::deserialize_compressed(&buf[..], &()).unwrap();
        assert_eq!(a, restored);
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
    fn ntt_forward_inverse_round_trip() {
        let prime = Q32_PRIMES[0];
        let tw = NttTwiddles::<i16, 64>::compute(prime);

        let original: [MontCoeff<i16>; 64] =
            std::array::from_fn(|i| prime.from_canonical((i as i16) % prime.p));

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
            let tw = NttTwiddles::<i16, 64>::compute(*prime);

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
    fn negacyclic_ntt_mul_matches_schoolbook_single_prime_d8() {
        const D: usize = 8;
        let prime = Q32_PRIMES[0];
        let tw = NttTwiddles::<i16, D>::compute(prime);

        let a_canon: [i16; D] = std::array::from_fn(|i| ((i as i16 * 7) + 3) % prime.p);
        let b_canon: [i16; D] = std::array::from_fn(|i| ((i as i16 * 5) + 11) % prime.p);

        // Schoolbook negacyclic convolution mod p: X^D = -1.
        let mut school = [0i16; D];
        for (i, &ai) in a_canon.iter().enumerate() {
            for (j, &bj) in b_canon.iter().enumerate() {
                let prod = (ai as i64 * bj as i64) % (prime.p as i64);
                let idx = i + j;
                if idx < D {
                    school[idx] = ((school[idx] as i64 + prod) % (prime.p as i64)) as i16;
                } else {
                    let k = idx - D;
                    school[k] = ((school[k] as i64 - prod) % (prime.p as i64)) as i16;
                }
            }
        }
        for x in &mut school {
            if *x < 0 {
                *x = (*x as i64 + prime.p as i64) as i16;
            }
        }

        let mut a = std::array::from_fn(|i| prime.from_canonical(a_canon[i]));
        let mut b = std::array::from_fn(|i| prime.from_canonical(b_canon[i]));
        forward_ntt(&mut a, prime, &tw);
        forward_ntt(&mut b, prime, &tw);

        let mut c: [_; D] = std::array::from_fn(|i| prime.mul(a[i], b[i]));
        inverse_ntt(&mut c, prime, &tw);

        let got: [i16; D] = std::array::from_fn(|i| prime.to_canonical(prime.normalize(c[i])));
        assert_eq!(got, school);
    }

    #[test]
    fn negacyclic_ntt_forward_matches_manual_evals_d8() {
        const D: usize = 8;
        let prime = Q32_PRIMES[0];
        let tw = NttTwiddles::<i16, D>::compute(prime);
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
        let a_canon: [i16; D] = std::array::from_fn(|i| ((i as i16 * 7) + 3) % prime.p);

        let mut expected = Vec::with_capacity(D);
        for k in 0..D {
            let alpha = pow_mod(psi, (2 * k + 1) as i64, p);
            let mut acc = 0i64;
            let mut power = 1i64;
            for &ai in &a_canon {
                acc = (acc + (ai as i64) * power) % p;
                power = (power * alpha) % p;
            }
            expected.push(acc as i16);
        }
        expected.sort_unstable();

        let mut a = std::array::from_fn(|i| prime.from_canonical(a_canon[i]));
        forward_ntt(&mut a, prime, &tw);
        let mut got: Vec<i16> = a
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
        let tw = NttTwiddles::<i16, D>::compute(prime);
        let p = prime.p as i64;

        let a_canon: [i16; D] = std::array::from_fn(|i| ((i as i16 * 7) + 3) % prime.p);
        let b_canon: [i16; D] = std::array::from_fn(|i| ((i as i16 * 5) + 11) % prime.p);

        let mut school = [0i16; D];
        for (i, &ai) in a_canon.iter().enumerate() {
            for (j, &bj) in b_canon.iter().enumerate() {
                let prod = (ai as i64 * bj as i64) % p;
                let idx = i + j;
                if idx < D {
                    school[idx] = ((school[idx] as i64 + prod) % p) as i16;
                } else {
                    let k = idx - D;
                    school[k] = ((school[k] as i64 - prod) % p) as i16;
                }
            }
        }
        for x in &mut school {
            if *x < 0 {
                *x = (*x as i64 + p) as i16;
            }
        }

        let mut a = std::array::from_fn(|i| prime.from_canonical(a_canon[i]));
        let mut b = std::array::from_fn(|i| prime.from_canonical(b_canon[i]));
        forward_ntt(&mut a, prime, &tw);
        forward_ntt(&mut b, prime, &tw);

        let mut c: [_; D] = std::array::from_fn(|i| prime.reduce_range(prime.mul(a[i], b[i])));
        inverse_ntt(&mut c, prime, &tw);

        let got: [i16; D] = std::array::from_fn(|i| prime.to_canonical(prime.normalize(c[i])));
        assert_eq!(got, school);
    }

    #[test]
    fn negacyclic_ntt_mul_matches_schoolbook_all_q32_primes_d64() {
        const D: usize = 64;
        let a_canon: [i16; D] = std::array::from_fn(|i| i as i16 * 7 + 3);
        let b_canon: [i16; D] = std::array::from_fn(|i| i as i16 * 5 + 11);

        for (pi, &prime) in Q32_PRIMES.iter().enumerate() {
            let tw = NttTwiddles::<i16, D>::compute(prime);
            let p = prime.p as i64;

            let a_mod: [i16; D] =
                std::array::from_fn(|i| ((a_canon[i] as i64).rem_euclid(p)) as i16);
            let b_mod: [i16; D] =
                std::array::from_fn(|i| ((b_canon[i] as i64).rem_euclid(p)) as i16);

            let mut school = [0i16; D];
            for (i, &ai) in a_mod.iter().enumerate() {
                for (j, &bj) in b_mod.iter().enumerate() {
                    let prod = (ai as i64 * bj as i64) % p;
                    let idx = i + j;
                    if idx < D {
                        school[idx] = ((school[idx] as i64 + prod) % p) as i16;
                    } else {
                        let k = idx - D;
                        school[k] = ((school[k] as i64 - prod) % p) as i16;
                    }
                }
            }
            for x in &mut school {
                if *x < 0 {
                    *x = (*x as i64 + p) as i16;
                }
            }

            let mut a = std::array::from_fn(|i| prime.from_canonical(a_mod[i]));
            let mut b = std::array::from_fn(|i| prime.from_canonical(b_mod[i]));
            forward_ntt(&mut a, prime, &tw);
            forward_ntt(&mut b, prime, &tw);

            let mut c = [MontCoeff::from_raw(0i16); D];
            for i in 0..D {
                c[i] = prime.reduce_range(prime.mul(a[i], b[i]));
            }
            inverse_ntt(&mut c, prime, &tw);

            let got: [i16; D] = std::array::from_fn(|i| prime.to_canonical(prime.normalize(c[i])));
            assert_eq!(got, school, "prime[{pi}] p={} mismatch", prime.p);
        }
    }

    #[test]
    fn cyclotomic_ntt_crt_round_trip_q32() {
        type F = Fp64<{ Q32_MODULUS }>;
        type R = CyclotomicRing<F, 64>;
        type N = CyclotomicCrtNtt<i16, Q32_NUM_PRIMES, 64>;

        let twiddles: [NttTwiddles<i16, 64>; Q32_NUM_PRIMES] =
            std::array::from_fn(|k| NttTwiddles::compute(Q32_PRIMES[k]));

        let coeffs: [F; 64] =
            std::array::from_fn(|i| F::from_u64(((i as u64 * 17) + 5) % Q32_MODULUS));
        let ring = R::from_coefficients(coeffs);
        let ntt = N::from_ring(&ring, &Q32_PRIMES, &twiddles);
        let garner = q32_garner();
        let round_trip = ntt.to_ring(&Q32_PRIMES, &twiddles, &garner);

        assert_eq!(ring, round_trip);
    }

    #[test]
    fn cyclotomic_ntt_reduced_ops_are_stable() {
        type F = Fp64<{ Q32_MODULUS }>;
        type R = CyclotomicRing<F, 64>;
        type N = CyclotomicCrtNtt<i16, Q32_NUM_PRIMES, 64>;

        let twiddles: [NttTwiddles<i16, 64>; Q32_NUM_PRIMES] =
            std::array::from_fn(|k| NttTwiddles::compute(Q32_PRIMES[k]));

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

        let garner = q32_garner();
        let zero_ntt = ntt_a.add_reduced(&ntt_a.neg_reduced(&Q32_PRIMES), &Q32_PRIMES);
        let zero_ring = zero_ntt.to_ring(&Q32_PRIMES, &twiddles, &garner);
        assert_eq!(zero_ring, R::zero());
    }

    #[test]
    fn backend_path_matches_default_scalar_path() {
        type F = Fp64<{ Q32_MODULUS }>;
        type R = CyclotomicRing<F, 64>;
        type N = CyclotomicCrtNtt<i16, Q32_NUM_PRIMES, 64>;

        let twiddles: [NttTwiddles<i16, 64>; Q32_NUM_PRIMES] =
            std::array::from_fn(|k| NttTwiddles::compute(Q32_PRIMES[k]));
        let ring = R::from_coefficients(std::array::from_fn(|i| {
            F::from_u64(((i as u64 * 13) + 9) % Q32_MODULUS)
        }));

        let default_ntt = N::from_ring(&ring, &Q32_PRIMES, &twiddles);
        let backend_ntt =
            N::from_ring_with_backend::<F, ScalarBackend>(&ring, &Q32_PRIMES, &twiddles);
        assert_eq!(default_ntt, backend_ntt);

        let garner = q32_garner();
        let default_back = default_ntt.to_ring(&Q32_PRIMES, &twiddles, &garner);
        let backend_back =
            backend_ntt.to_ring_with_backend::<F, ScalarBackend>(&Q32_PRIMES, &twiddles, &garner);
        assert_eq!(default_back, backend_back);
    }

    #[test]
    fn crt_ntt_mul_matches_schoolbook_q32() {
        type F = Fp64<{ Q32_MODULUS }>;
        type R = CyclotomicRing<F, 64>;
        type N = CyclotomicCrtNtt<i16, Q32_NUM_PRIMES, 64>;

        let twiddles: [NttTwiddles<i16, 64>; Q32_NUM_PRIMES] =
            std::array::from_fn(|k| NttTwiddles::compute(Q32_PRIMES[k]));
        let garner = q32_garner();

        let a = R::from_coefficients(std::array::from_fn(|i| {
            F::from_u64(((i as u64 * 7) + 3) % Q32_MODULUS)
        }));
        let b = R::from_coefficients(std::array::from_fn(|i| {
            F::from_u64(((i as u64 * 5) + 11) % Q32_MODULUS)
        }));

        let schoolbook = a * b;

        let ntt_a = N::from_ring(&a, &Q32_PRIMES, &twiddles);
        let ntt_b = N::from_ring(&b, &Q32_PRIMES, &twiddles);
        let ntt_prod = ntt_a.pointwise_mul(&ntt_b, &Q32_PRIMES);
        let ntt_result: R = ntt_prod.to_ring(&Q32_PRIMES, &twiddles, &garner);

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
        let residues: [u32; 64] =
            std::array::from_fn(|i| if i < 8 { (i as u32 * 31) + 7 } else { 0 });

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
        let rhs_i8: [i8; 32] = std::array::from_fn(|i| (((i * 23 + 11) % 256) as i16 - 128) as i8);
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
        let rhs_i8: [i8; 32] = std::array::from_fn(|i| (((i * 19 + 5) % 256) as i16 - 128) as i8);

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

        let primes = q64_primes();
        let twiddles: [NttTwiddles<i32, 64>; Q64_NUM_PRIMES] =
            std::array::from_fn(|k| NttTwiddles::compute(primes[k]));
        let garner = q64_garner();

        let coeffs: [F; 64] =
            std::array::from_fn(|i| F::from_u64(((i as u64 * 19) + 3) % Q64_MODULUS));
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

        let primes = q64_primes();
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

        for k in 0..8 {
            let mut monomial_coeffs = [F::zero(); 8];
            monomial_coeffs[k] = F::one();
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
            a,
            "shift by D should be identity mod D"
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
}
