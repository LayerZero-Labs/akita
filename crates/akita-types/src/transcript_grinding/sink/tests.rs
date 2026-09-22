use super::*;

fn batch(rounds: usize) -> SumcheckRoundBatch {
    SumcheckRoundBatch {
        capacity: 128,
        protocol: SumcheckProtocol::Stage1,
        level: 2,
        stage: 1,
        rounds,
        degree: 5,
    }
}

// Preserve the original per-round construction as an independent batch oracle.
fn old_runs(batch: SumcheckRoundBatch) -> Result<Vec<GrindingRun>, AkitaError> {
    (0..batch.rounds)
        .map(|round| {
            GrindingRun::proof_of_work(
                GrindingSite::SumcheckRound {
                    protocol: batch.protocol,
                    level: batch.level,
                    stage: batch.stage,
                    round: crate::narrowing::usize_to_u32(round, "sumcheck grinding round")?,
                },
                polynomial_identity_loss_factor(batch.degree)?,
                batch.capacity,
            )
        })
        .collect()
}

#[test]
fn aggregated_rounds_match_old_loop_cost_and_exact_sites() {
    for capacity in [128, 256] {
        for protocol in [
            SumcheckProtocol::Stage1,
            SumcheckProtocol::PhysicalL2,
            SumcheckProtocol::Stage2,
            SumcheckProtocol::Stage3,
            SumcheckProtocol::ExtensionOpeningReduction,
        ] {
            for rounds in [0, 1, 7, 32] {
                for degree in [2, 3, 5, 9] {
                    let batch = SumcheckRoundBatch {
                        capacity,
                        protocol,
                        degree,
                        ..batch(rounds)
                    };
                    let expected = old_runs(batch).unwrap();
                    let mut materialized = Vec::new();
                    (|run| {
                        materialized.push(run);
                        Ok(())
                    })
                    .sumcheck_rounds(batch)
                    .unwrap();
                    assert_eq!(materialized, expected);
                    let mut old = GrindingPlanAccumulator::new(capacity).unwrap();
                    let mut aggregated = GrindingPlanAccumulator::new(capacity).unwrap();
                    // Include neighboring distinct sites to check accumulation.
                    old.push(GrindingRun::fold_response(1)).unwrap();
                    aggregated.push(GrindingRun::fold_response(1)).unwrap();
                    for run in expected {
                        old.push(run).unwrap();
                    }
                    aggregated.sumcheck_rounds(batch).unwrap();
                    assert_eq!(aggregated.cost(), old.cost());
                    assert_eq!(aggregated.run_count, old.run_count);
                }
            }
        }
    }
}

#[test]
fn empty_rounds_skip_invalid_metadata_but_nonempty_rounds_reject_it() {
    let invalids = [
        SumcheckRoundBatch {
            capacity: 0,
            ..batch(1)
        },
        SumcheckRoundBatch {
            level: u32::MAX,
            ..batch(1)
        },
        SumcheckRoundBatch {
            stage: u32::MAX,
            ..batch(1)
        },
        SumcheckRoundBatch {
            degree: usize::MAX,
            ..batch(1)
        },
    ];
    for invalid in invalids {
        let mut old = GrindingPlanAccumulator::new(128).unwrap();
        let expected = old_runs(invalid).and_then(|runs| {
            for run in runs {
                old.push(run)?;
            }
            Ok(())
        });
        let mut aggregated = GrindingPlanAccumulator::new(128).unwrap();
        assert!(expected.is_err());
        assert!(aggregated.sumcheck_rounds(invalid).is_err());
        let mut empty = GrindingPlanAccumulator::new(128).unwrap();
        empty
            .sumcheck_rounds(SumcheckRoundBatch {
                rounds: 0,
                ..invalid
            })
            .unwrap();
        assert_eq!(empty.run_count, 0);
        assert_eq!(empty.cost().expanded_query_count, 0);
        assert_eq!(empty.cost().total_nonce_bits, 0);
    }
    // A target constructed for a different capacity must still be rejected.
    let mut accumulator = GrindingPlanAccumulator::new(128).unwrap();
    assert!(accumulator
        .sumcheck_rounds(SumcheckRoundBatch {
            capacity: 256,
            ..batch(1)
        })
        .is_err());
}

#[test]
fn batching_preserves_run_query_and_nonce_limits() {
    let mut count = GrindingPlanAccumulator::new(128).unwrap();
    count.run_count = u32::MAX - 3;
    count.sumcheck_rounds(batch(3)).unwrap();
    assert_eq!(count.run_count, u32::MAX);
    assert!(count.sumcheck_rounds(batch(1)).is_err());

    // Edge pricing reports the count; only whole-plan construction applies
    // the strict u32::MAX query-budget cap.
    let mut edge = GrindingPlanAccumulator::new(128).unwrap();
    edge.expanded_query_count = u64::from(u32::MAX) - 2;
    edge.sumcheck_rounds(batch(3)).unwrap();
    assert_eq!(edge.cost().expanded_query_count, u64::from(u32::MAX) + 1);
    let mut query = GrindingPlanAccumulator::new(128).unwrap();
    query.expanded_query_count = u64::MAX - 2;
    assert!(query.sumcheck_rounds(batch(3)).is_err());

    let run = old_runs(batch(1)).unwrap()[0];
    let bits = usize::from(run.nonce_bits());
    assert!(bits > 0);
    let mut nonce = GrindingPlanAccumulator::new(128).unwrap();
    nonce.total_nonce_bits = usize::MAX - 3 * bits;
    nonce.sumcheck_rounds(batch(3)).unwrap();
    assert_eq!(nonce.cost().total_nonce_bits, usize::MAX);
    assert!(nonce.sumcheck_rounds(batch(1)).is_err());
}

#[test]
fn last_round_and_run_count_have_distinct_reserved_boundaries() {
    // The final valid site index is MAX-1, giving MAX distinct zero-bit runs.
    let mut accumulator = GrindingPlanAccumulator::new(256).unwrap();
    accumulator
        .sumcheck_rounds(SumcheckRoundBatch {
            capacity: 256,
            ..batch(u32::MAX as usize)
        })
        .unwrap();
    assert_eq!(accumulator.run_count, u32::MAX);
    assert_eq!(accumulator.cost().expanded_query_count, u64::from(u32::MAX));
    assert!(accumulator
        .sumcheck_rounds(SumcheckRoundBatch {
            capacity: 256,
            ..batch(1)
        })
        .is_err());
    #[cfg(target_pointer_width = "64")]
    {
        let mut sentinel = GrindingPlanAccumulator::new(256).unwrap();
        assert!(sentinel
            .sumcheck_rounds(SumcheckRoundBatch {
                capacity: 256,
                ..batch(u32::MAX as usize + 1)
            })
            .is_err());
        assert!(sentinel
            .sumcheck_rounds(SumcheckRoundBatch {
                capacity: 256,
                ..batch(u32::MAX as usize + 2)
            })
            .is_err());
    }
    #[cfg(target_pointer_width = "32")]
    {
        let mut nonce_product = GrindingPlanAccumulator::new(128).unwrap();
        assert!(nonce_product
            .sumcheck_rounds(batch(u32::MAX as usize))
            .is_err());
    }
}
