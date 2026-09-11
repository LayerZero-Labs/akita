//! Typed adapter for canonical transcript-grinding plan derivation.

use crate::CommitmentConfig;
use akita_error::AkitaError;
use akita_types::{FoldSchedule, GrindingPlan, OpeningClaimsLayout};
use jolt_field::{CanonicalEncoding, ExtField};

/// Derive the only accepted grinding plan for one effective schedule and call.
pub fn derive_transcript_grinding_plan<Cfg: CommitmentConfig>(
    schedule: &FoldSchedule,
    root_layout: &OpeningClaimsLayout,
) -> Result<GrindingPlan, AkitaError>
where
    Cfg::Field: CanonicalEncoding,
    Cfg::ExtField: ExtField<Cfg::Field>,
{
    let extension_degree = <Cfg::ExtField as ExtField<Cfg::Field>>::DEGREE;
    if Cfg::EXT_DEGREE != extension_degree {
        return Err(AkitaError::InvalidSetup(
            "grinding plan extension degree does not match the field tower".into(),
        ));
    }
    akita_types::derive_transcript_grinding_plan_from_public_shape(
        schedule,
        root_layout,
        <Cfg::Field as CanonicalEncoding>::MODULUS_BITS,
        extension_degree,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof_optimized::fp128;
    use akita_types::{GrindingQueryKind, GrindingSite, GRINDING_NONCE_SLACK_BITS};
    use jolt_field::PseudoMersenne;

    #[test]
    fn production_onehot_plan_is_canonical_and_fully_priced() {
        let catalog = crate::test_support::workspace_schedule_catalog::<fp128::OneHot>()
            .expect("one-hot schedule catalog");
        let layout = OpeningClaimsLayout::new(14, 1).expect("opening layout");
        let row = catalog
            .resolve_key(&akita_types::AkitaScheduleLookupKey::single(
                layout.root_final_group_layout().expect("root group"),
            ))
            .expect("generated production row");
        let plan = derive_transcript_grinding_plan::<fp128::OneHot>(row.schedule(), &layout)
            .expect("grinding plan");

        assert_eq!(
            plan.runs().first().unwrap().site(),
            GrindingSite::FoldResponse { level: 0 }
        );
        assert_eq!(
            plan.runs()
                .iter()
                .filter(|run| matches!(run.site(), GrindingSite::EvaluationBatch { .. }))
                .count(),
            0,
            "singleton row batching draws no challenge and has no plan entry"
        );
        for run in plan.runs() {
            if run.kind() == GrindingQueryKind::ProofOfWork {
                assert!(u128::from(run.loss_factor()) <= (1u128 << run.grind_bits()));
                assert_eq!(
                    run.nonce_bits(),
                    if run.grind_bits() == 0 {
                        0
                    } else {
                        run.grind_bits() + GRINDING_NONCE_SLACK_BITS
                    }
                );
            }
        }
        assert_eq!(
            (
                plan.runs().len(),
                plan.expanded_query_count(),
                plan.total_nonce_bits(),
                plan.digest().unwrap(),
            ),
            (
                43,
                50,
                383,
                [
                    236, 232, 157, 232, 43, 58, 62, 68, 118, 58, 218, 127, 36, 83, 166, 123, 31,
                    133, 157, 222, 197, 92, 67, 6, 62, 148, 191, 98, 57, 29, 78, 210,
                ],
            )
        );
    }

    #[test]
    fn stage1_prices_the_full_eq_factored_round_degree() {
        let catalog = crate::test_support::workspace_schedule_catalog::<fp128::OneHot>()
            .expect("one-hot schedule catalog");
        let layout = OpeningClaimsLayout::new(14, 1).expect("opening layout");
        let row = catalog
            .resolve_key(&akita_types::AkitaScheduleLookupKey::single(
                layout.root_final_group_layout().expect("root group"),
            ))
            .expect("generated production row");
        let plan = derive_transcript_grinding_plan::<fp128::OneHot>(row.schedule(), &layout)
            .expect("grinding plan");
        let root = &row.schedule().root;
        let rounds = akita_types::sumcheck_rounds(root.params.d_a(), root.output_witness_len);
        let basis = 1usize
            .checked_shl(root.params.open().digits.log_basis)
            .expect("digit range basis");
        let range = akita_types::DigitRangePlan::new(basis).expect("digit range plan");
        let (stages, _) = range
            .proof_shapes_for_route(rounds, root.params.inner().matrix.security_route())
            .expect("Stage 1 shapes");

        for run in plan.runs() {
            let GrindingSite::SumcheckRound {
                protocol: akita_types::SumcheckProtocol::Stage1,
                level: 0,
                stage,
                ..
            } = run.site()
            else {
                continue;
            };
            let q_degree = stages[usize::try_from(stage).expect("stage index")]
                .sumcheck_proof
                .1;
            assert_eq!(run.loss_factor(), u64::try_from(q_degree + 1).unwrap());
        }
    }

    #[test]
    fn exact_field_orders_report_the_pseudo_mersenne_deficit_without_repricing() {
        fn exact_order<F: PseudoMersenne>(extension_degree: usize) -> (u32, u128, usize) {
            (F::MODULUS_BITS, F::OFFSET, extension_degree)
        }

        for (bits, _, degree) in [
            exact_order::<fp128::Field>(1),
            exact_order::<crate::proof_optimized::fp64::Field>(2),
            exact_order::<crate::proof_optimized::fp32::Field>(4),
        ] {
            assert_eq!(
                akita_types::nominal_challenge_capacity_bits(bits, degree).unwrap(),
                128
            );
            assert_eq!(akita_types::grind_bits_for_loss(3, 128).unwrap(), 2);
        }
    }

    #[test]
    fn every_generated_production_row_derives_a_complete_plan() {
        use crate::proof_optimized::{fp32, fp64};

        fn audit<Cfg: CommitmentConfig>()
        where
            Cfg::Field: CanonicalEncoding,
            Cfg::ExtField: ExtField<Cfg::Field>,
        {
            let catalog = crate::test_support::workspace_schedule_catalog::<Cfg>()
                .expect("workspace schedule catalog");
            for row in catalog.rows() {
                let layout = row.profiles().opening_layout().expect("opening layout");
                let plan = derive_transcript_grinding_plan::<Cfg>(row.schedule(), &layout)
                    .expect("complete grinding plan");
                let count = |kind| plan.runs().iter().filter(|run| run.kind() == kind).count();
                assert_eq!(
                    count(GrindingQueryKind::FoldResponse),
                    row.schedule().num_fold_levels()
                );
                let expected_evaluation_batch_levels =
                    std::iter::once(layout.requires_row_batch_challenge().then_some(0))
                        .flatten()
                        .chain(
                            row.schedule()
                                .recursive_folds
                                .iter()
                                .enumerate()
                                .filter_map(|(index, step)| {
                                    step.params.setup_prefix().map(|_| {
                                        u32::try_from(index + 1)
                                            .expect("recursive fold level fits u32")
                                    })
                                }),
                        )
                        .collect::<Vec<_>>();
                let evaluation_batch_levels = plan
                    .runs()
                    .iter()
                    .filter_map(|run| match run.site() {
                        GrindingSite::EvaluationBatch { level } => Some(level),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    evaluation_batch_levels, expected_evaluation_batch_levels,
                    "evaluation batching belongs only to nonterminal multi-polynomial layouts"
                );
                assert!(count(GrindingQueryKind::FoldChallengeGroup) > 0);
                assert!(plan.expanded_query_count() >= plan.runs().len() as u64);

                // A level's Stage 3 rounds are induced by its *successor's*
                // setup prefix, and their count is the prefix group's own
                // committed width. Pin that against the stored layout so no
                // second derivation of the prefix width can reappear.
                for (index, step) in row.schedule().recursive_folds.iter().enumerate() {
                    let level = u32::try_from(index).expect("grinding level fits u32");
                    let expected = step
                        .params
                        .setup_prefix()
                        .map_or(0, |prefix| prefix.profile.group.num_vars());
                    let observed = plan
                        .runs()
                        .iter()
                        .filter(|run| {
                            matches!(
                                run.site(),
                                GrindingSite::SumcheckRound {
                                    protocol: akita_types::SumcheckProtocol::Stage3,
                                    level: run_level,
                                    ..
                                } if run_level == level
                            )
                        })
                        .count();
                    assert_eq!(
                        observed, expected,
                        "level {level} Stage 3 rounds must equal the successor prefix group width"
                    );
                }
            }
        }

        audit::<fp128::Dense>();
        audit::<fp128::DenseBounded>();
        audit::<fp128::DenseMultiChunk>();
        audit::<fp128::OneHot>();
        audit::<fp128::OneHotMultiChunk>();
        audit::<fp128::OneHotMultiChunkW2R2>();
        audit::<fp128::OneHotMultiChunkW4R2>();
        audit::<fp64::Dense>();
        audit::<fp64::OneHot>();
        audit::<fp32::Dense>();
        audit::<fp32::OneHot>();
        audit::<crate::RecursiveCommitmentConfig<fp128::OneHot>>();
        audit::<crate::RecursiveCommitmentConfig<fp128::OneHotMultiChunk>>();
    }
}
