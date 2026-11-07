use super::unit_intervals;
use super::{basis_eval_at_index, EqPairSeed};
use jolt_field::{One, Prime128OffsetA7F7 as F, Ring, Zero};

fn differential<const LM: bool, const RM: bool, const STATES: usize>() {
    let mut state = 0xb12b_abc7_45d6_9e01u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for case in 0..120 {
        let lb = (next() % 8) as usize;
        let rb = (next() % 8) as usize;
        let left = (0..lb)
            .map(|_| F::from_u64(next() % if case % 3 == 0 { 2 } else { 101 }))
            .collect::<Vec<_>>();
        let right = (0..rb)
            .map(|_| F::from_u64(next() % if case % 3 == 0 { 2 } else { 101 }))
            .collect::<Vec<_>>();
        let len = (next() % 150 + 1) as usize;
        let seeds = (0..9)
            .map(|_| EqPairSeed {
                left_offset: (next() % (2u64 << lb)) as usize,
                right_offset: (next() % (2u64 << rb)) as usize,
                weight: F::from_u64(next() % 127),
            })
            .collect::<Vec<_>>();
        let mut expected = F::zero();
        for seed in &seeds {
            for i in 0..len {
                expected += seed.weight
                    * basis_eval_at_index::<F, LM>(&left, seed.left_offset + i)
                    * basis_eval_at_index::<F, RM>(&right, seed.right_offset + i * (STATES - 1));
            }
        }
        let actual =
            unit_intervals::eval::<F, LM, RM, STATES>(&left, &right, len, &seeds, &mut 0).unwrap();
        assert_eq!(
            actual, expected,
            "case {case}, left bits {lb}, right bits {rb}, len {len}"
        );
    }
}
#[test]
fn cached_unit_intervals_match_dense_in_all_bases() {
    differential::<false, false, 2>();
    differential::<false, true, 2>();
    differential::<true, false, 2>();
    differential::<true, true, 2>();
    differential::<false, false, 4>();
    differential::<false, true, 4>();
    differential::<true, false, 4>();
    differential::<true, true, 4>();
    differential::<false, false, 6>();
    differential::<false, true, 6>();
    differential::<true, false, 6>();
    differential::<true, true, 6>();
    differential::<false, false, 8>();
    differential::<false, true, 8>();
    differential::<true, false, 8>();
    differential::<true, true, 8>();
}
#[test]
fn cached_unit_intervals_enforce_address_and_work_bounds() {
    let seed = EqPairSeed {
        left_offset: usize::MAX,
        right_offset: 0,
        weight: F::one(),
    };
    assert!(unit_intervals::eval::<F, false, false, 2>(
        &[F::one()],
        &[F::one()],
        2,
        &[seed],
        &mut 0
    )
    .is_err());
    let seed = EqPairSeed {
        left_offset: 0,
        right_offset: 0,
        weight: F::one(),
    };
    let mut work = crate::offset_eq::MAX_COMPACT_STRIDE_TERMS;
    assert!(unit_intervals::eval::<F, false, false, 2>(
        &[F::one()],
        &[F::one()],
        2,
        &[seed],
        &mut work
    )
    .is_err());
}

#[test]
fn power_two_right_intervals_match_dense() {
    let left = (0..9).map(|i| F::from_u64(i + 2)).collect::<Vec<_>>();
    let right = (0..10).map(|i| F::from_u64(i + 11)).collect::<Vec<_>>();
    for stride in [2, 4, 8, 16] {
        let seeds = (0..37)
            .map(|i| EqPairSeed {
                left_offset: i * 7,
                right_offset: i * stride + 3,
                weight: F::from_u64(i as u64 + 7),
            })
            .collect::<Vec<_>>();
        let expected = seeds.iter().fold(F::zero(), |sum, seed| {
            sum + seed.weight
                * (0..123).fold(F::zero(), |v, i| {
                    v + basis_eval_at_index::<F, false>(&left, seed.left_offset + i)
                        * basis_eval_at_index::<F, false>(&right, seed.right_offset + stride * i)
                })
        });
        assert_eq!(
            unit_intervals::eval_scaled_right::<F, false, false, 2>(
                &left, &right, stride, 123, &seeds, &mut 0
            )
            .unwrap(),
            expected
        );
    }
}

fn scaled_differential<const LM: bool, const RM: bool>() {
    let left = (0..9).map(|i| F::from_u64(i + 2)).collect::<Vec<_>>();
    let right = (0..10).map(|i| F::from_u64(i + 11)).collect::<Vec<_>>();
    for stride in [7, 14, 28, 56] {
        let seeds = (0..37)
            .map(|i| EqPairSeed {
                left_offset: i * 7,
                right_offset: i * stride + 3,
                weight: F::from_u64(i as u64 + 7),
            })
            .collect::<Vec<_>>();
        let expected = seeds.iter().fold(F::zero(), |sum, seed| {
            sum + seed.weight
                * (0..123).fold(F::zero(), |v, i| {
                    v + basis_eval_at_index::<F, LM>(&left, seed.left_offset + i)
                        * basis_eval_at_index::<F, RM>(&right, seed.right_offset + stride * i)
                })
        });
        assert_eq!(
            unit_intervals::eval_scaled_right::<F, LM, RM, 8>(
                &left, &right, stride, 123, &seeds, &mut 0,
            )
            .unwrap(),
            expected
        );
    }
}

#[test]
fn scaled_odd_stride_intervals_preserve_fixed_low_bits_and_clipping() {
    scaled_differential::<false, false>();
    scaled_differential::<false, true>();
    scaled_differential::<true, false>();
    scaled_differential::<true, true>();
}

#[test]
fn scaled_intervals_validate_carry_bounds_and_work() {
    let seed = EqPairSeed {
        left_offset: 0,
        right_offset: 0,
        weight: F::one(),
    };
    for stride in [0, 12, 60] {
        assert!(unit_intervals::eval_scaled_right::<F, false, false, 8>(
            &[F::one(); 6],
            &[F::one(); 6],
            stride,
            3,
            &[seed],
            &mut 0,
        )
        .is_err());
    }
    let mut work = crate::offset_eq::MAX_COMPACT_STRIDE_TERMS;
    assert!(unit_intervals::eval_scaled_right::<F, false, false, 8>(
        &[F::one(); 6],
        &[F::one(); 6],
        28,
        3,
        &[seed],
        &mut work,
    )
    .is_err());
}
