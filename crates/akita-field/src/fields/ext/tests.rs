use super::*;
use crate::fields::lift::{
    canonical_frobenius_thetas, solve_frobenius_moore, validate_canonical_frobenius_thetas,
    ExtField, FrobeniusExtField,
};
use crate::{Fp64, Prime16Offset99};
use crate::{FromPrimitiveInt, Invertible};
use rand::rngs::StdRng;
use rand::SeedableRng;

type F = Fp64<4294967197>;
type E2 = Ext2<F>;
type E4 = TowerBasisFp4<F, TwoNr, UnitNr>;
type P4 = PowerBasisFp4<F, TwoNr>;
type R4 = RingSubfieldFp4<F>;
type R8 = RingSubfieldFp8<F>;
type R8Fp16 = RingSubfieldFp8<Prime16Offset99>;

#[test]
fn fp2_add_sub_identity() {
    let a = E2::new(F::from_u64(3), F::from_u64(5));
    let b = E2::new(F::from_u64(7), F::from_u64(11));
    let c = a + b;
    assert_eq!(c - b, a);
    assert_eq!(c - a, b);
}

#[test]
fn fp2_mul_one() {
    let a = E2::new(F::from_u64(42), F::from_u64(13));
    assert_eq!(a * E2::one(), a);
    assert_eq!(E2::one() * a, a);
}

#[test]
fn fp2_mul_commutativity() {
    let mut rng = StdRng::seed_from_u64(1234);
    let a = E2::random(&mut rng);
    let b = E2::random(&mut rng);
    assert_eq!(a * b, b * a);
}

#[test]
fn fp2_karatsuba_matches_schoolbook() {
    let mut rng = StdRng::seed_from_u64(5678);
    for _ in 0..100 {
        let a = E2::random(&mut rng);
        let b = E2::random(&mut rng);
        let nr = <TwoNr as Fp2Config<F>>::non_residue();
        let expected = E2::new(
            (a.coeffs[0] * b.coeffs[0]) + (nr * (a.coeffs[1] * b.coeffs[1])),
            (a.coeffs[0] * b.coeffs[1]) + (a.coeffs[1] * b.coeffs[0]),
        );
        assert_eq!(a * b, expected);
    }
}

#[test]
fn fp2_square_matches_mul() {
    let mut rng = StdRng::seed_from_u64(9012);
    for _ in 0..100 {
        let a = E2::random(&mut rng);
        assert_eq!(a.square(), a * a, "square mismatch for {a:?}");
    }
}

#[test]
fn fp2_inv() {
    let mut rng = StdRng::seed_from_u64(3456);
    for _ in 0..50 {
        let a = E2::random(&mut rng);
        if !a.is_zero() {
            let inv = a.inverse().unwrap();
            assert_eq!(a * inv, E2::one());
        }
    }
}

#[test]
fn fp4_mul_commutativity() {
    let mut rng = StdRng::seed_from_u64(7890);
    let a = E4::random(&mut rng);
    let b = E4::random(&mut rng);
    assert_eq!(a * b, b * a);
}

#[test]
fn fp4_square_matches_mul() {
    let mut rng = StdRng::seed_from_u64(1111);
    for _ in 0..50 {
        let a = E4::random(&mut rng);
        assert_eq!(a.square(), a * a);
    }
}

#[test]
fn fp4_inv() {
    let mut rng = StdRng::seed_from_u64(2222);
    for _ in 0..50 {
        let a = E4::random(&mut rng);
        if !a.is_zero() {
            let inv = a.inverse().unwrap();
            assert_eq!(a * inv, E4::one());
        }
    }
}

#[test]
fn power_basis_fp4_square_matches_mul() {
    let mut rng = StdRng::seed_from_u64(3333);
    for _ in 0..50 {
        let a = P4::random(&mut rng);
        assert_eq!(a.square(), a * a);
    }
}

#[test]
fn power_basis_fp4_inv() {
    let mut rng = StdRng::seed_from_u64(4444);
    for _ in 0..50 {
        let a = P4::random(&mut rng);
        if !a.is_zero() {
            let inv = a.inverse().unwrap();
            assert_eq!(a * inv, P4::one());
        }
    }
}

#[test]
fn ring_subfield_fp4_multiplication_table() {
    let two = F::from_u64(2);
    let e1 = R4::new([F::zero(), F::one(), F::zero(), F::zero()]);
    let e2 = R4::new([F::zero(), F::zero(), F::one(), F::zero()]);
    let e3 = R4::new([F::zero(), F::zero(), F::zero(), F::one()]);
    let two_const = R4::new([two, F::zero(), F::zero(), F::zero()]);

    assert_eq!(e1 * e1, two_const + e2);
    assert_eq!(e1 * e2, e1 + e3);
    assert_eq!(e1 * e3, e2);
    assert_eq!(e2 * e2, two_const);
    assert_eq!(e2 * e3, e1 - e3);
    assert_eq!(e3 * e3, two_const - e2);
}

#[test]
fn ring_subfield_fp4_square_matches_mul() {
    let mut rng = StdRng::seed_from_u64(5555);
    for _ in 0..50 {
        let a = R4::random(&mut rng);
        assert_eq!(a.square(), a * a);
    }
}

#[test]
fn ring_subfield_fp4_inv() {
    let mut rng = StdRng::seed_from_u64(6666);
    for _ in 0..50 {
        let a = R4::random(&mut rng);
        if !a.is_zero() {
            let inv = a.inverse().unwrap();
            assert_eq!(a * inv, R4::one());
        }
    }
}

#[test]
fn ring_subfield_fp8_multiplication_table_spot_checks() {
    let two = F::from_u64(2);
    let e = |idx: usize| {
        R8::new(std::array::from_fn(|i| {
            if i == idx {
                F::one()
            } else {
                F::zero()
            }
        }))
    };
    let two_const = R8::new([
        two,
        F::zero(),
        F::zero(),
        F::zero(),
        F::zero(),
        F::zero(),
        F::zero(),
        F::zero(),
    ]);

    assert_eq!(e(1) * e(1), two_const + e(2));
    assert_eq!(e(2) * e(2), two_const + e(4));
    assert_eq!(e(4) * e(4), two_const);
    assert_eq!(e(7) * e(7), two_const - e(2));
    assert_eq!(e(5) * e(7), e(2) - e(4));
}

#[test]
fn ring_subfield_fp8_square_matches_mul() {
    let mut rng = StdRng::seed_from_u64(7777);
    for _ in 0..50 {
        let a = R8::random(&mut rng);
        assert_eq!(a.square(), a * a);
    }
}

#[test]
fn ring_subfield_fp8_inv() {
    let mut rng = StdRng::seed_from_u64(8888);
    for _ in 0..50 {
        let a = R8::random(&mut rng);
        if !a.is_zero() {
            let inv = a.inverse().unwrap();
            assert_eq!(a * inv, R8::one());
        }
    }
}

#[test]
fn ring_subfield_fp8_fp16_serialization_is_coeff_ordered() {
    let x = R8Fp16::new(std::array::from_fn(|i| {
        Prime16Offset99::from_u64(i as u64 + 1)
    }));
    let mut bytes = Vec::new();
    x.serialize_with_mode(&mut bytes, Compress::No).unwrap();
    assert_eq!(x.serialized_size(Compress::No), 16);
    assert_eq!(bytes, vec![1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0]);

    let decoded =
        R8Fp16::deserialize_with_mode(&bytes[..], Compress::No, Validate::Yes, &()).unwrap();
    assert_eq!(decoded, x);
}

#[test]
fn frobenius_fp2_is_conjugation() {
    let x = E2::new(F::from_u64(13), F::from_u64(21));
    assert_eq!(<E2 as FrobeniusExtField<F>>::frobenius_pow(x, 0), x);
    assert_eq!(
        <E2 as FrobeniusExtField<F>>::frobenius_pow(x, 1),
        x.conjugate()
    );
    assert_eq!(<E2 as FrobeniusExtField<F>>::frobenius_pow(x, 2), x);
    assert_eq!(
        <E2 as FrobeniusExtField<F>>::frobenius_inv_pow(x, 1),
        x.conjugate()
    );
}

#[test]
fn canonical_moore_thetas_solve_fp2() {
    validate_canonical_frobenius_thetas::<F, E2>(2).unwrap();
    let thetas = canonical_frobenius_thetas::<F, E2>(2).unwrap();
    let z = [
        E2::new(F::from_u64(3), F::from_u64(5)),
        E2::new(F::from_u64(7), F::from_u64(11)),
    ];
    let r = (0..2)
        .map(|row| {
            thetas
                .iter()
                .zip(z.iter())
                .fold(E2::zero(), |acc, (&theta, &z_h)| {
                    acc + <E2 as FrobeniusExtField<F>>::frobenius_inv_pow(theta, row) * z_h
                })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        solve_frobenius_moore::<F, E2>(&thetas, &r).unwrap(),
        z.to_vec()
    );
}

#[test]
fn canonical_ring_subfield_thetas_are_the_packing_basis() {
    let thetas = canonical_frobenius_thetas::<F, R4>(4).unwrap();
    assert_eq!(
        thetas[0],
        R4::new([F::one(), F::zero(), F::zero(), F::zero()])
    );
    assert_eq!(
        thetas[1],
        R4::new([F::zero(), F::one(), F::zero(), F::zero()])
    );
    assert_eq!(
        thetas[2],
        R4::new([F::zero(), F::zero(), F::one(), F::zero()])
    );
    assert_eq!(
        thetas[3],
        R4::new([F::zero(), F::zero(), F::zero(), F::one()])
    );
    validate_canonical_frobenius_thetas::<F, R4>(4).unwrap();
}

#[test]
fn canonical_ring_subfield_fp8_thetas_are_the_packing_basis() {
    let thetas = canonical_frobenius_thetas::<F, R8>(8).unwrap();
    for (idx, theta) in thetas.iter().enumerate().take(8) {
        assert_eq!(
            *theta,
            R8::new(std::array::from_fn(|i| {
                if i == idx {
                    F::one()
                } else {
                    F::zero()
                }
            }))
        );
    }
    validate_canonical_frobenius_thetas::<F, R8>(8).unwrap();
}

#[test]
fn duplicate_moore_theta_rejects() {
    let theta = E2::one();
    let err = solve_frobenius_moore::<F, E2>(&[theta, theta], &[E2::one(), E2::one()])
        .expect_err("duplicate theta should be singular");
    assert!(format!("{err}").contains("singular"));
}

#[test]
fn from_small_int_fp2() {
    let a = E2::from_u64(42);
    assert_eq!(a, E2::new(F::from_u64(42), F::zero()));

    let b = E2::from_i64(-3);
    assert_eq!(b, E2::new(F::from_i64(-3), F::zero()));

    let c = E2::from_u8(7);
    assert_eq!(c, E2::from_u64(7));

    let d = E2::from_u32(100_000);
    assert_eq!(d, E2::from_u64(100_000));
}

#[test]
fn from_small_int_fp4() {
    let a = E4::from_u64(42);
    assert_eq!(a, E4::new(E2::from_u64(42), E2::zero()));

    let b = E4::from_i64(-7);
    assert_eq!(b, E4::new(E2::from_i64(-7), E2::zero()));
}

#[test]
fn ext_field_degree() {
    assert_eq!(<F as ExtField<F>>::EXT_DEGREE, 1);
    assert_eq!(<E2 as ExtField<F>>::EXT_DEGREE, 2);
    assert_eq!(<E4 as ExtField<F>>::EXT_DEGREE, 4);
    assert_eq!(<R4 as ExtField<F>>::EXT_DEGREE, 4);
    assert_eq!(<R8 as ExtField<F>>::EXT_DEGREE, 8);
}

#[test]
fn ext_field_from_base_slice() {
    let c0 = F::from_u64(3);
    let c1 = F::from_u64(5);
    let e2 = E2::from_base_slice(&[c0, c1]);
    assert_eq!(e2, E2::new(c0, c1));

    let c2 = F::from_u64(7);
    let c3 = F::from_u64(11);
    let e4 = E4::from_base_slice(&[c0, c1, c2, c3]);
    assert_eq!(e4, E4::new(E2::new(c0, c2), E2::new(c1, c3)));

    let p4 = P4::from_base_slice(&[c0, c1, c2, c3]);
    assert_eq!(p4, P4::new([c0, c1, c2, c3]));

    let r4 = R4::from_base_slice(&[c0, c1, c2, c3]);
    assert_eq!(r4, R4::new([c0, c1, c2, c3]));

    let c4 = F::from_u64(13);
    let c5 = F::from_u64(17);
    let c6 = F::from_u64(19);
    let c7 = F::from_u64(23);
    let r8 = R8::from_base_slice(&[c0, c1, c2, c3, c4, c5, c6, c7]);
    assert_eq!(r8, R8::new([c0, c1, c2, c3, c4, c5, c6, c7]));
}

#[test]
fn tower_and_power_basis_fp4_multiplication_agree() {
    let x_p = P4::new([
        F::from_u64(1),
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(4),
    ]);
    let y_p = P4::new([
        F::from_u64(5),
        F::from_u64(6),
        F::from_u64(7),
        F::from_u64(8),
    ]);
    let x_t: E4 = x_p.into();
    let y_t: E4 = y_p.into();

    let got: P4 = (x_t * y_t).into();
    assert_eq!(got, x_p * y_p);
}

#[test]
fn power_basis_fp4_transcript_limb_order_is_univariate() {
    let x = P4::new([
        F::from_u64(1),
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(4),
    ]);
    assert_eq!(
        <P4 as ExtField<F>>::to_base_vec(&x),
        vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4)
        ]
    );
}

#[test]
fn tower_basis_fp4_transcript_limb_order_is_univariate() {
    let x = E4::new(
        E2::new(F::from_u64(1), F::from_u64(3)),
        E2::new(F::from_u64(2), F::from_u64(4)),
    );
    assert_eq!(
        <E4 as ExtField<F>>::to_base_vec(&x),
        vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4)
        ]
    );
}

#[test]
fn extension_fields_are_array_layouts() {
    assert_eq!(core::mem::size_of::<E2>(), core::mem::size_of::<[F; 2]>());
    assert_eq!(core::mem::align_of::<E2>(), core::mem::align_of::<[F; 2]>());
    assert_eq!(core::mem::size_of::<P4>(), core::mem::size_of::<[F; 4]>());
    assert_eq!(core::mem::align_of::<P4>(), core::mem::align_of::<[F; 4]>());
    assert_eq!(core::mem::size_of::<R4>(), core::mem::size_of::<[F; 4]>());
    assert_eq!(core::mem::align_of::<R4>(), core::mem::align_of::<[F; 4]>());
    assert_eq!(core::mem::size_of::<E4>(), core::mem::size_of::<[E2; 2]>());
    assert_eq!(
        core::mem::align_of::<E4>(),
        core::mem::align_of::<[E2; 2]>()
    );
}

#[test]
fn eq_impl() {
    let a = E2::new(F::from_u64(1), F::from_u64(2));
    let b = E2::new(F::from_u64(1), F::from_u64(2));
    let c = E2::new(F::from_u64(1), F::from_u64(3));
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn ring_subfield_fp4_fp32_product_accum_matches_direct_mul() {
    use super::ring_subfield_fp4::ring_subfield_fp4_mul_to_accum_fp32;
    use crate::fields::wide::RingSubfieldFp4Fp32ProductAccum;
    use crate::Fp32;
    use num_traits::Zero;

    type Fp = Fp32<251>;
    type R4Fp32 = RingSubfieldFp4<Fp>;

    let mut rng = StdRng::seed_from_u64(0xACC0);
    for _ in 0..200 {
        let a = R4Fp32::random(&mut rng);
        let b = R4Fp32::random(&mut rng);
        let direct = a * b;
        let accum = ring_subfield_fp4_mul_to_accum_fp32(a.coeffs, b.coeffs);
        let reduced = R4Fp32::new(accum.reduce::<251>());
        assert_eq!(direct, reduced, "accum mismatch for a={a:?} b={b:?}");
    }

    let zero_accum = RingSubfieldFp4Fp32ProductAccum::ZERO;
    assert!(zero_accum.is_zero());
    let reduced_zero = R4Fp32::new(zero_accum.reduce::<251>());
    assert_eq!(reduced_zero, R4Fp32::zero());
}

#[test]
fn ring_subfield_fp4_fp32_accum_summation() {
    use crate::Fp32;
    use num_traits::Zero;

    type Fp = Fp32<251>;
    type R4Fp32 = RingSubfieldFp4<Fp>;

    let mut rng = StdRng::seed_from_u64(0xACC1);
    let n = 1024;
    let pairs: Vec<(R4Fp32, R4Fp32)> = (0..n)
        .map(|_| (R4Fp32::random(&mut rng), R4Fp32::random(&mut rng)))
        .collect();

    let direct_sum: R4Fp32 = pairs
        .iter()
        .map(|(a, b)| *a * *b)
        .fold(R4Fp32::zero(), |s, p| s + p);

    let accum_sum = pairs.iter().fold(
        <R4Fp32 as HasUnreducedOps>::ProductAccum::zero(),
        |s, (a, b)| s + a.mul_to_product_accum(*b),
    );
    let reduced = R4Fp32::reduce_product_accum(accum_sum);

    assert_eq!(
        direct_sum, reduced,
        "accumulated sum of {n} products mismatched"
    );
}

// Regression guard for the `Fp2<Fp64>` delayed-reduction accumulator. The earlier
// bug dropped the carry into bit 128 because each Fp2 coefficient (c0 up to ~2^130,
// c1 up to ~2^129) was formed in a single `u128`. It only surfaces with near-`p`
// operands -- products around 2^128 -- which the small-modulus tests never reach,
// so these use the real 2^64-59 prime and cover both Fp2 configs.
#[test]
fn fp2_fp64_product_accum_matches_direct_mul_large_operands() {
    use crate::Prime64Offset59;

    let mut rng = StdRng::seed_from_u64(0xF64A);
    for _ in 0..256 {
        // TwoNr (IS_NEG_ONE = false): c0 = p00 + 2*p11.
        let a = Ext2::<Prime64Offset59>::random(&mut rng);
        let b = Ext2::<Prime64Offset59>::random(&mut rng);
        assert_eq!(
            a * b,
            Ext2::<Prime64Offset59>::reduce_product_accum(a.mul_to_product_accum(b)),
            "TwoNr accum mismatch a={a:?} b={b:?}"
        );

        // NegOneNr (IS_NEG_ONE = true): c0 = p00 + p^2 - p11.
        let c = Fp2::<Prime64Offset59, NegOneNr>::random(&mut rng);
        let d = Fp2::<Prime64Offset59, NegOneNr>::random(&mut rng);
        assert_eq!(
            c * d,
            Fp2::<Prime64Offset59, NegOneNr>::reduce_product_accum(c.mul_to_product_accum(d)),
            "NegOneNr accum mismatch c={c:?} d={d:?}"
        );
    }
}

#[test]
fn fp2_fp64_accum_summation_large_operands() {
    use crate::Prime64Offset59;
    use num_traits::Zero;

    type E = Ext2<Prime64Offset59>;

    let mut rng = StdRng::seed_from_u64(0xF64C);
    let n = 1024;
    let pairs: Vec<(E, E)> = (0..n)
        .map(|_| (E::random(&mut rng), E::random(&mut rng)))
        .collect();

    let direct_sum: E = pairs
        .iter()
        .map(|(a, b)| *a * *b)
        .fold(E::zero(), |s, p| s + p);

    let accum_sum = pairs
        .iter()
        .fold(<E as HasUnreducedOps>::ProductAccum::zero(), |s, (a, b)| {
            s + a.mul_to_product_accum(*b)
        });

    assert_eq!(
        direct_sum,
        E::reduce_product_accum(accum_sum),
        "fp2<fp64> accumulated sum of {n} products mismatched"
    );
}

#[test]
fn ring_subfield_fp8_fp16_product_accum_matches_direct_mul() {
    use super::ring_subfield_fp8::ring_subfield_fp8_mul_to_accum_fp16;
    use crate::fields::wide::RingSubfieldFp8Fp16ProductAccum;
    use num_traits::Zero;

    // Prime16Offset99 = Fp16<65_437> (2^16 - 99).
    const P: u32 = 65_437;

    let mut rng = StdRng::seed_from_u64(0x8ACC0);
    for _ in 0..400 {
        let a = R8Fp16::random(&mut rng);
        let b = R8Fp16::random(&mut rng);
        let direct = a * b;
        let accum = ring_subfield_fp8_mul_to_accum_fp16(a.coeffs, b.coeffs);
        let reduced = R8Fp16::new(accum.reduce::<P>());
        assert_eq!(direct, reduced, "accum mismatch for a={a:?} b={b:?}");
    }

    let zero_accum = RingSubfieldFp8Fp16ProductAccum::ZERO;
    assert!(zero_accum.is_zero());
    let reduced_zero = R8Fp16::new(zero_accum.reduce::<P>());
    assert_eq!(reduced_zero, R8Fp16::zero());
}

#[test]
fn ring_subfield_fp8_fp16_accum_summation() {
    use num_traits::Zero;

    let mut rng = StdRng::seed_from_u64(0x8ACC1);
    let n = 4096;
    let pairs: Vec<(R8Fp16, R8Fp16)> = (0..n)
        .map(|_| (R8Fp16::random(&mut rng), R8Fp16::random(&mut rng)))
        .collect();

    let direct_sum: R8Fp16 = pairs
        .iter()
        .map(|(a, b)| *a * *b)
        .fold(R8Fp16::zero(), |s, p| s + p);

    let accum_sum = pairs.iter().fold(
        <R8Fp16 as HasUnreducedOps>::ProductAccum::zero(),
        |s, (a, b)| s + a.mul_to_product_accum(*b),
    );
    let reduced = R8Fp16::reduce_product_accum(accum_sum);

    assert_eq!(
        direct_sum, reduced,
        "accumulated sum of {n} products mismatched"
    );
}

#[test]
fn ring_subfield_fp8_fp16_fold_matrix_matches_generic() {
    // The 8×8 fold matrix must reproduce the generic ring-multiply fold
    // `even + r·(odd - even)` byte-for-byte, otherwise the EOR fold diverges.
    let mut rng = StdRng::seed_from_u64(0x8F01D);
    for _ in 0..400 {
        let r = R8Fp16::random(&mut rng);
        let even = R8Fp16::random(&mut rng);
        let odd = R8Fp16::random(&mut rng);
        let ctx = <R8Fp16 as HasOptimizedFold>::precompute_fold(r);
        let matrix = <R8Fp16 as HasOptimizedFold>::fold_one(&ctx, even, odd);
        let generic = even + r * (odd - even);
        assert_eq!(
            matrix, generic,
            "fold mismatch r={r:?} even={even:?} odd={odd:?}"
        );
    }
}
