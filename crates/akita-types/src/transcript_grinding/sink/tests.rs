use super::{GrindingPlanSink, SumcheckRoundBatch};
use crate::polynomial_identity_loss_factor;
use crate::transcript_grinding::{
    GrindingPlanAccumulator, GrindingRun, GrindingSite, SumcheckProtocol,
};
use crate::ChallengeFieldOrder;
use akita_error::AkitaError;

fn batch(rounds: usize) -> SumcheckRoundBatch {
    SumcheckRoundBatch {
        challenge_order: ChallengeFieldOrder::from_full_capacity(128).unwrap(),
        protocol: SumcheckProtocol::Stage1,
        level: 2,
        stage: 1,
        rounds,
        degree: 5,
    }
}

// Materialize each scheduled round to check the aggregated batch result.
fn explicit_runs(batch: SumcheckRoundBatch) -> Result<Vec<GrindingRun>, AkitaError> {
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
                batch.challenge_order,
            )
        })
        .collect()
}

#[test]
fn aggregated_rounds_match_per_round_cost_and_exact_sites() {
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
                        challenge_order: ChallengeFieldOrder::from_full_capacity(capacity).unwrap(),
                        protocol,
                        degree,
                        ..batch(rounds)
                    };
                    let per_round_runs = explicit_runs(batch).unwrap();
                    let mut materialized = Vec::new();
                    (|run| {
                        materialized.push(run);
                        Ok(())
                    })
                    .sumcheck_rounds(batch)
                    .unwrap();
                    assert_eq!(materialized, per_round_runs);
                    let mut per_round = GrindingPlanAccumulator::new(
                        ChallengeFieldOrder::from_full_capacity(capacity).unwrap(),
                    );
                    let mut aggregated = GrindingPlanAccumulator::new(
                        ChallengeFieldOrder::from_full_capacity(capacity).unwrap(),
                    );
                    // Include neighboring distinct sites to check accumulation.
                    per_round.push(GrindingRun::fold_response(1)).unwrap();
                    aggregated.push(GrindingRun::fold_response(1)).unwrap();
                    for run in per_round_runs {
                        per_round.push(run).unwrap();
                    }
                    aggregated.sumcheck_rounds(batch).unwrap();
                    assert_eq!(aggregated.cost(), per_round.cost());
                    assert_eq!(aggregated.run_count, per_round.run_count);
                }
            }
        }
    }
}

#[test]
fn empty_rounds_skip_invalid_metadata_but_nonempty_rounds_reject_it() {
    let order = ChallengeFieldOrder::from_full_capacity(128).unwrap();
    let invalids = [
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
    assert!(ChallengeFieldOrder::from_full_capacity(0).is_err());
    for invalid in invalids {
        let mut per_round = GrindingPlanAccumulator::new(order);
        let expected = explicit_runs(invalid).and_then(|runs| {
            for run in runs {
                per_round.push(run)?;
            }
            Ok(())
        });
        let mut aggregated = GrindingPlanAccumulator::new(order);
        assert!(expected.is_err());
        assert_eq!(per_round.cost().total_nonce_bits, 0);
        assert!(aggregated.sumcheck_rounds(invalid).is_err());
        let mut empty = GrindingPlanAccumulator::new(order);
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
    // A batch priced for a different challenge order must still be rejected.
    let mut accumulator = GrindingPlanAccumulator::new(order);
    assert!(accumulator
        .sumcheck_rounds(SumcheckRoundBatch {
            challenge_order: ChallengeFieldOrder::from_full_capacity(256).unwrap(),
            ..batch(1)
        })
        .is_err());
}

#[test]
fn batching_preserves_run_query_and_nonce_limits() {
    let order = ChallengeFieldOrder::from_full_capacity(128).unwrap();
    let mut count = GrindingPlanAccumulator::new(order);
    count.run_count = u32::MAX - 3;
    count.sumcheck_rounds(batch(3)).unwrap();
    assert_eq!(count.run_count, u32::MAX);
    assert!(count.sumcheck_rounds(batch(1)).is_err());

    // Edge pricing reports the count; only whole-plan construction applies
    // the strict u32::MAX query-budget cap.
    let mut edge = GrindingPlanAccumulator::new(order);
    edge.expanded_query_count = u64::from(u32::MAX) - 2;
    edge.sumcheck_rounds(batch(3)).unwrap();
    assert_eq!(edge.cost().expanded_query_count, u64::from(u32::MAX) + 1);
    let mut query = GrindingPlanAccumulator::new(order);
    query.expanded_query_count = u64::MAX - 2;
    assert!(query.sumcheck_rounds(batch(3)).is_err());

    let run = explicit_runs(batch(1)).unwrap()[0];
    let bits = usize::from(run.nonce_bits());
    assert!(bits > 0);
    let mut nonce = GrindingPlanAccumulator::new(order);
    nonce.total_nonce_bits = usize::MAX - 3 * bits;
    nonce.sumcheck_rounds(batch(3)).unwrap();
    assert_eq!(nonce.cost().total_nonce_bits, usize::MAX);
    assert!(nonce.sumcheck_rounds(batch(1)).is_err());
}

#[test]
fn last_round_and_run_count_have_distinct_reserved_boundaries() {
    let wide_order = ChallengeFieldOrder::from_full_capacity(256).unwrap();
    // The final valid site index is MAX-1, giving MAX distinct zero-bit runs.
    let mut accumulator = GrindingPlanAccumulator::new(wide_order);
    accumulator
        .sumcheck_rounds(SumcheckRoundBatch {
            challenge_order: wide_order,
            ..batch(u32::MAX as usize)
        })
        .unwrap();
    assert_eq!(accumulator.run_count, u32::MAX);
    assert_eq!(accumulator.cost().expanded_query_count, u64::from(u32::MAX));
    assert!(accumulator
        .sumcheck_rounds(SumcheckRoundBatch {
            challenge_order: wide_order,
            ..batch(1)
        })
        .is_err());
    #[cfg(target_pointer_width = "64")]
    {
        let mut sentinel = GrindingPlanAccumulator::new(wide_order);
        assert!(sentinel
            .sumcheck_rounds(SumcheckRoundBatch {
                challenge_order: wide_order,
                ..batch(u32::MAX as usize + 1)
            })
            .is_err());
        assert!(sentinel
            .sumcheck_rounds(SumcheckRoundBatch {
                challenge_order: wide_order,
                ..batch(u32::MAX as usize + 2)
            })
            .is_err());
    }
    #[cfg(target_pointer_width = "32")]
    {
        let mut nonce_product =
            GrindingPlanAccumulator::new(ChallengeFieldOrder::from_full_capacity(128).unwrap());
        assert!(nonce_product
            .sumcheck_rounds(batch(u32::MAX as usize))
            .is_err());
    }
}
