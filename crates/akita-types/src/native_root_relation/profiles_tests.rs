use crate::binary_source::{root_weight_fixtures, RootFixtureField as F};
use akita_algebra::poly::fold_evals_in_place;
use jolt_field::{Ring, Zero};

/// Every block width, including widths that send the 256-stride lanes
/// through the per-address fallback, folds like the materialized table.
#[test]
fn profiles_fold_like_the_materialized_table() {
    for fixture in root_weight_fixtures() {
        let dense = fixture.weights.materialize().unwrap();
        let num_vars = dense.len().trailing_zeros() as usize;
        for block_bits in [0, 3, 8, 10].into_iter().filter(|&bits| bits <= num_vars) {
            let mut profiles = fixture.weights.profiles(block_bits).unwrap();
            let embedded = super::NativeRootWeights::concatenate(
                4 * dense.len(),
                [
                    (dense.len(), fixture.weights.clone()),
                    (2 * dense.len(), fixture.weights.clone()),
                ],
            )
            .unwrap();
            let mut embedded_profiles = embedded.profiles(block_bits).unwrap();
            // A zero explicit term selects additive emission. It must still
            // give the same table as the independent embedded builders.
            let mut additive = embedded.clone();
            additive.terms.push((0, F::zero()));
            assert_eq!(
                additive.profiles(block_bits).unwrap().take_dense(),
                embedded_profiles.clone().take_dense()
            );
            // Embedding keeps both sets of local rows and only one zero row;
            // a per-address visitor can be numerically right and fail this.
            assert_eq!(
                embedded_profiles.rows.len(),
                2 * profiles.rows.len() - (1 << block_bits)
            );
            let mut embedded_dense = vec![F::zero(); 4 * dense.len()];
            embedded_dense[dense.len()..2 * dense.len()].copy_from_slice(&dense);
            embedded_dense[2 * dense.len()..3 * dense.len()].copy_from_slice(&dense);
            assert_eq!(embedded_profiles.clone().take_dense(), embedded_dense);
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
                let challenge = F::from_u64(match round % 3 {
                    0 => 0,
                    1 => 1,
                    _ => 1_000_003 + 7 * round as u64,
                });
                profiles.bind(challenge);
                embedded_profiles.bind(challenge);
                fold_evals_in_place(&mut embedded_dense, challenge);
                fold_evals_in_place(&mut folded, challenge);
                assert_eq!(profiles.len(), folded.len());
                assert_eq!(embedded_profiles.clone().take_dense(), embedded_dense);
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
    let fixture = root_weight_fixtures().swap_remove(0);
    let num_vars = fixture.weights.table_len().trailing_zeros() as usize;
    assert!(fixture.weights.profiles(num_vars + 1).is_err());
    let mut builder = super::ProfileBuilder::<F>::new(16, 2).unwrap();
    let local = super::NativeRootWeights::new(4, vec![(0, F::from_u64(3))], F::zero()).unwrap();
    assert!(builder
        .append_profiles(1, local.profiles(2).unwrap())
        .is_err());
    assert!(builder
        .append_profiles(16, local.profiles(2).unwrap())
        .is_err());
    assert!(builder
        .append_profiles(0, local.profiles(1).unwrap())
        .is_err());
    builder
        .append_profiles(4, local.profiles(2).unwrap())
        .unwrap();
    assert!(builder
        .append_profiles(4, local.profiles(2).unwrap())
        .is_err());
}

/// Lane runs fill the same table as per-address terms, across several
/// parallel tasks, a partial last block per lane and blocks that already
/// hold rows.
#[test]
fn lanes_add_like_per_address_terms() {
    use super::ProfileBuilder;
    let (degree, stride) = (13, 16);
    let bases = (0..1_100).map(|lane| lane * stride).collect::<Vec<_>>();
    let value = |lane: usize, offset: usize| F::from_u64((lane * 1_000 + offset + 1) as u64);
    let mut lanes = ProfileBuilder::<F>::new(1 << 15, 3).unwrap();
    let mut terms = ProfileBuilder::<F>::new(1 << 15, 3).unwrap();
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
        (5_003, 20_003),
    ];
    let value =
        |interval: usize, offset: usize| F::from_u64((interval * 10_000 + offset + 1) as u64);
    let mut runs = ProfileBuilder::<F>::new(1 << 15, 3).unwrap();
    let mut terms = ProfileBuilder::<F>::new(1 << 15, 3).unwrap();
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
    let mut outside = ProfileBuilder::<F>::new(1 << 15, 3).unwrap();
    assert!(outside
        .add_intervals(&[(32_760, 9)], |_, _, _| Ok(()))
        .is_err());
}
