use crate::binary_source::{root_weight_scenarios, RootFixtureField as F};
use akita_algebra::poly::fold_evals_in_place;
use jolt_field::Ring;

/// Every block width, including widths that send the 256-stride lanes
/// through the per-address fallback, folds like the materialized table.
#[test]
fn profiles_fold_like_the_materialized_table() {
    for scenario in root_weight_scenarios() {
        let fixture = scenario.compile_stage2();
        let dense = fixture.weights.materialize().unwrap();
        let num_vars = dense.len().trailing_zeros() as usize;
        for block_bits in [0, 3, 8, 10].into_iter().filter(|&bits| bits <= num_vars) {
            let mut profiles = fixture.weights.profiles(block_bits).unwrap();
            assert_eq!(profiles.len(), dense.len());
            assert_eq!(profiles.clone().take_dense(), dense, "{}", fixture.name);
            let mut folded = dense.clone();
            for round in 0..block_bits {
                for left in (0..folded.len()).step_by(2 * 37 + 2) {
                    assert_eq!(
                        profiles.pair(left),
                        (folded[left], folded[left + 1]),
                        "{} bits {block_bits} round {round}",
                        fixture.name
                    );
                }
                let challenge = F::from_u64(1_000_003 + 7 * round as u64);
                profiles.bind(challenge);
                fold_evals_in_place(&mut folded, challenge);
                assert_eq!(profiles.len(), folded.len());
            }
            assert_eq!(
                profiles.take_dense(),
                folded,
                "{} bits {block_bits}",
                fixture.name
            );
            assert!(profiles.is_empty());
        }
    }
}

#[test]
fn profiles_reject_blocks_wider_than_the_table() {
    let weights = crate::native_root_relation::NativeRootWeights::<F>::new(
        8,
        vec![(0, F::from_u64(1))],
        F::from_u64(0),
    )
    .unwrap();
    assert!(weights.profiles(4).is_err());
}

/// Lane runs fill the same table as per-address terms, across several
/// parallel tasks, a partial last block per lane and blocks that already
/// hold rows.
#[test]
fn lanes_add_like_per_address_terms() {
    use super::ProfileBuilder;
    let (degree, stride) = (13, 16);
    let bases = (0..70).map(|lane| lane * stride).collect::<Vec<_>>();
    let value = |lane: usize, offset: usize| F::from_u64((lane * 1_000 + offset + 1) as u64);
    let mut lanes = ProfileBuilder::<F>::new(1 << 12, 3).unwrap();
    let mut terms = ProfileBuilder::<F>::new(1 << 12, 3).unwrap();
    for builder in [&mut lanes, &mut terms] {
        builder.add(5 * stride + 2, F::from_u64(9)).unwrap();
    }
    lanes
        .add_lanes(&bases, degree, |run, rows| {
            let lane_rows = rows.len() / run.len();
            for (lane, rows) in run.zip(rows.chunks_exact_mut(lane_rows)) {
                for (offset, slot) in rows
                    .iter_mut()
                    .flat_map(|row| row.iter_mut())
                    .enumerate()
                    .take(degree)
                {
                    *slot += value(lane, offset);
                }
            }
            Ok(())
        })
        .unwrap();
    for (lane, &base) in bases.iter().enumerate() {
        for offset in 0..degree {
            terms.add(base + offset, value(lane, offset)).unwrap();
        }
    }
    assert_eq!(lanes.finish().take_dense(), terms.finish().take_dense());
}

/// Interval runs fill the same table as per-address terms, with overlapping
/// and unaligned intervals, intervals that cross task boundaries, an empty
/// interval and blocks that already hold rows.
#[test]
fn intervals_add_like_per_address_terms() {
    use super::ProfileBuilder;
    let intervals = [
        (5, 3),
        (0, 0),
        (510, 1_100),
        (512, 9),
        (1_000, 600),
        (4_000, 96),
        (4_090, 6),
    ];
    let value =
        |interval: usize, offset: usize| F::from_u64((interval * 10_000 + offset + 1) as u64);
    let mut runs = ProfileBuilder::<F>::new(1 << 12, 3).unwrap();
    let mut terms = ProfileBuilder::<F>::new(1 << 12, 3).unwrap();
    for builder in [&mut runs, &mut terms] {
        builder.add(1_001, F::from_u64(9)).unwrap();
    }
    runs.add_intervals(&intervals, |interval, first, row| {
        for (offset, slot) in (first..).zip(row.iter_mut()) {
            *slot += value(interval, offset);
        }
        Ok(())
    })
    .unwrap();
    for (interval, &(start, len)) in intervals.iter().enumerate() {
        for offset in 0..len {
            terms.add(start + offset, value(interval, offset)).unwrap();
        }
    }
    assert_eq!(runs.finish().take_dense(), terms.finish().take_dense());
    let mut outside = ProfileBuilder::<F>::new(1 << 12, 3).unwrap();
    assert!(outside
        .add_intervals(&[(4_090, 7)], |_, _, _| Ok(()))
        .is_err());
}
