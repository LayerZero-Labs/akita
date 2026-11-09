use super::unit_intervals;
use super::{basis_eval_at_index, EqPairSeed};
use jolt_field::{One, Prime128Offset6A23801F as F, Ring, Zero};

fn differential<const LM: bool, const RM: bool>() {
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
                    * basis_eval_at_index::<F, RM>(&right, seed.right_offset + i);
            }
        }
        let actual = unit_intervals::eval::<F, LM, RM>(&left, &right, len, &seeds, &mut 0).unwrap();
        assert_eq!(
            actual, expected,
            "case {case}, left bits {lb}, right bits {rb}, len {len}"
        );
    }
}
#[test]
fn cached_unit_intervals_match_dense_in_all_bases() {
    differential::<false, false>();
    differential::<false, true>();
    differential::<true, false>();
    differential::<true, true>();
}
#[test]
fn cached_unit_intervals_enforce_address_and_work_bounds() {
    let seed = EqPairSeed {
        left_offset: usize::MAX,
        right_offset: 0,
        weight: F::one(),
    };
    assert!(
        unit_intervals::eval::<F, false, false>(&[F::one()], &[F::one()], 2, &[seed], &mut 0)
            .is_err()
    );
    let seed = EqPairSeed {
        left_offset: 0,
        right_offset: 0,
        weight: F::one(),
    };
    let mut work = crate::offset_eq::MAX_COMPACT_STRIDE_TERMS;
    assert!(unit_intervals::eval::<F, false, false>(
        &[F::one()],
        &[F::one()],
        2,
        &[seed],
        &mut work
    )
    .is_err());
}
