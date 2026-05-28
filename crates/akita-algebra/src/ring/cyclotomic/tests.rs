use super::*;
use akita_field::fields::{Fp128x8i32, Fp64, Fp64x4i32, Prime128Offset275};
use rand::rngs::StdRng;
use rand::SeedableRng;

type F64 = Fp64<4294967197>;
type F128 = Prime128Offset275;
const D: usize = 8;

#[test]
fn cyclotomic_ring_satisfies_jolt_ring_core() {
    fn assert_ring_core<R: RingCore>() {}
    assert_ring_core::<CyclotomicRing<F64, D>>();

    let x = CyclotomicRing::<F64, D>::x();
    assert_eq!(x.square(), x * x);
    assert_eq!(
        [x, CyclotomicRing::one()]
            .into_iter()
            .product::<CyclotomicRing<F64, D>>(),
        x
    );
}

#[test]
fn shift_accumulate_into_matches_negacyclic_shift() {
    let mut rng = StdRng::seed_from_u64(0x1234);
    let a = CyclotomicRing::<F64, D>::random(&mut rng);
    let dst = CyclotomicRing::<F64, D>::random(&mut rng);

    for k in 0..32 {
        let expected = dst + a.negacyclic_shift(k);
        let mut actual = dst;
        a.shift_accumulate_into(&mut actual, k);
        assert_eq!(actual, expected, "shift_accumulate_into k={k}");
    }
}

#[test]
fn shift_sub_into_matches_negacyclic_shift() {
    let mut rng = StdRng::seed_from_u64(0x1234);
    let a = CyclotomicRing::<F64, D>::random(&mut rng);
    let dst = CyclotomicRing::<F64, D>::random(&mut rng);

    for k in 0..32 {
        let expected = dst - a.negacyclic_shift(k);
        let mut actual = dst;
        a.shift_sub_into(&mut actual, k);
        assert_eq!(actual, expected, "shift_sub_into k={k}");
    }
}

#[test]
fn shift_scale_accumulate_into_matches_scaled_negacyclic_shift() {
    let mut rng = StdRng::seed_from_u64(0x2468);
    let a = CyclotomicRing::<F64, D>::random(&mut rng);
    let dst = CyclotomicRing::<F64, D>::random(&mut rng);
    let scales = [
        F64::zero(),
        F64::one(),
        -F64::one(),
        F64::from_u64(7),
        F64::from_u64(4294967196),
    ];

    for k in 0..32 {
        for &scale in &scales {
            let mut actual = dst;
            a.shift_scale_accumulate_into(&mut actual, k, scale);

            let expected = dst + a.scale(&scale).negacyclic_shift(k);
            assert_eq!(
                actual, expected,
                "shift_scale_accumulate_into k={k} scale={scale:?}"
            );
        }
    }
}

#[test]
fn wide_shift_accumulate_matches_narrow_fp64() {
    let mut rng = StdRng::seed_from_u64(0x1234);
    let src = CyclotomicRing::<F64, D>::random(&mut rng);
    let initial = CyclotomicRing::<F64, D>::random(&mut rng);

    for k in [0, 1, 7, 31, 63, 64, 67] {
        let mut narrow = initial;
        src.shift_accumulate_into(&mut narrow, k);

        let wide_src = WideCyclotomicRing::<Fp64x4i32, D>::from_ring(&src);
        let mut wide_dst = WideCyclotomicRing::<Fp64x4i32, D>::from_ring(&initial);
        wide_src.shift_accumulate_into(&mut wide_dst, k);
        let wide_reduced: CyclotomicRing<F64, D> = wide_dst.reduce();

        assert_eq!(narrow, wide_reduced, "shift_accumulate k={k}");
    }
}

#[test]
fn wide_shift_sub_matches_narrow_fp64() {
    let mut rng = StdRng::seed_from_u64(0x5678);
    let src = CyclotomicRing::<F64, D>::random(&mut rng);
    let initial = CyclotomicRing::<F64, D>::random(&mut rng);

    for k in [0, 1, 15, 32, 63, 64, 67] {
        let mut narrow = initial;
        src.shift_sub_into(&mut narrow, k);

        let wide_src = WideCyclotomicRing::<Fp64x4i32, D>::from_ring(&src);
        let mut wide_dst = WideCyclotomicRing::<Fp64x4i32, D>::from_ring(&initial);
        wide_src.shift_sub_into(&mut wide_dst, k);
        let wide_reduced: CyclotomicRing<F64, D> = wide_dst.reduce();

        assert_eq!(narrow, wide_reduced, "shift_sub k={k}");
    }
}

#[test]
fn wide_mul_by_monomial_sum_matches_narrow_fp64() {
    let mut rng = StdRng::seed_from_u64(0xabcd);
    let src = CyclotomicRing::<F64, D>::random(&mut rng);
    let positions = vec![0, 5, 17, 42, 63];

    let mut narrow = CyclotomicRing::<F64, D>::zero();
    src.mul_by_monomial_sum_into(&mut narrow, &positions);

    let wide_src = WideCyclotomicRing::<Fp64x4i32, D>::from_ring(&src);
    let mut wide_dst = WideCyclotomicRing::<Fp64x4i32, D>::zero();
    wide_src.mul_by_monomial_sum_into(&mut wide_dst, &positions);
    let wide_reduced: CyclotomicRing<F64, D> = wide_dst.reduce();

    assert_eq!(narrow, wide_reduced);
}

#[test]
fn wide_many_accumulations_fp128() {
    let mut rng = StdRng::seed_from_u64(0xbeef);
    let src = CyclotomicRing::<F128, D>::random(&mut rng);

    let mut narrow = CyclotomicRing::<F128, D>::zero();
    let wide_src = WideCyclotomicRing::<Fp128x8i32, D>::from_ring(&src);
    let mut wide_dst = WideCyclotomicRing::<Fp128x8i32, D>::zero();

    for k in 0..50 {
        src.shift_accumulate_into(&mut narrow, k % D);
        wide_src.shift_accumulate_into(&mut wide_dst, k % D);
    }
    for k in 0..30 {
        src.shift_sub_into(&mut narrow, k % D);
        wide_src.shift_sub_into(&mut wide_dst, k % D);
    }

    let wide_reduced: CyclotomicRing<F128, D> = wide_dst.reduce();
    assert_eq!(narrow, wide_reduced);
}

#[test]
fn center_for_decomposition_hits_fp128_overflow_boundaries() {
    let q = (-F128::one()).to_canonical_u128() + 1;
    let i128_max = i128::MAX as u128;

    for &(levels, log_basis) in &[(64usize, 2u32), (32usize, 4u32)] {
        let threshold = decompose_centering_threshold(levels, log_basis, q);
        let cases = [
            (threshold, false),
            (threshold + 1, true),
            (q - i128_max - 1, true),
            (q - i128_max, false),
            (q - 1, false),
        ];

        for (canonical, expect_overflow) in cases {
            let (_, first_digit) = center_for_decomposition(canonical, q, threshold, log_basis);
            assert_eq!(
                first_digit.is_some(),
                expect_overflow,
                "unexpected overflow classification for levels={levels}, log_basis={log_basis}, canonical={canonical}"
            );
        }
    }
}

#[test]
fn asymmetric_centering_boundary_roundtrip_fp128() {
    let q = (-F128::one()).to_canonical_u128() + 1;
    let i128_max = i128::MAX as u128;

    for &(log_basis, levels) in &[(2u32, 64usize), (4u32, 32usize)] {
        let threshold = decompose_centering_threshold(levels, log_basis, q);
        let boundary_values = [
            0,
            1,
            threshold.saturating_sub(1),
            threshold,
            threshold + 1,
            q - i128_max - 1,
            q - i128_max,
            q - 2,
            q - 1,
        ];
        let ring = CyclotomicRing::<F128, D>::from_coefficients(from_fn(|i| {
            F128::from_canonical_u128_reduced(boundary_values[i % boundary_values.len()])
        }));

        let mut digits = vec![CyclotomicRing::<F128, D>::zero(); levels];
        ring.balanced_decompose_pow2_into(&mut digits, log_basis);
        let recomposed = CyclotomicRing::gadget_recompose_pow2(&digits, log_basis);
        assert_eq!(
            ring, recomposed,
            "field roundtrip failed for log_basis={log_basis}, levels={levels}"
        );

        let mut i8_digits = vec![[0i8; D]; levels];
        ring.balanced_decompose_pow2_i8_into(&mut i8_digits, log_basis);
        let recomposed_i8 = CyclotomicRing::gadget_recompose_pow2_i8(&i8_digits, log_basis);
        assert_eq!(
            ring, recomposed_i8,
            "i8 roundtrip failed for log_basis={log_basis}, levels={levels}"
        );
    }
}
