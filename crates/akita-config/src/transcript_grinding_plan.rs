//! Typed adapter for canonical transcript-grinding plan derivation.

use crate::CommitmentConfig;
use akita_error::AkitaError;
use akita_field::{CanonicalField, ExtField};
use akita_types::{FoldSchedule, GrindingPlan, OpeningClaimsLayout};

/// Derive the only accepted grinding plan for one effective schedule and call.
pub fn derive_transcript_grinding_plan<Cfg: CommitmentConfig>(
    schedule: &FoldSchedule,
    root_layout: &OpeningClaimsLayout,
) -> Result<GrindingPlan, AkitaError>
where
    Cfg::Field: CanonicalField,
    Cfg::ExtField: ExtField<Cfg::Field>,
{
    let extension_degree = <Cfg::ExtField as ExtField<Cfg::Field>>::EXT_DEGREE;
    if Cfg::EXT_DEGREE != extension_degree {
        return Err(AkitaError::InvalidSetup(
            "grinding plan extension degree does not match the field tower".into(),
        ));
    }
    akita_types::derive_transcript_grinding_plan_from_public_shape(
        schedule,
        root_layout,
        <Cfg::Field as CanonicalField>::modulus_bits(),
        extension_degree,
    )
}

#[cfg(test)]
mod tests {
    #[cfg(any(feature = "all-schedules", feature = "schedules-default"))]
    use super::*;
    use crate::proof_optimized::fp128;
    use akita_field::PseudoMersenneField;
    #[cfg(any(feature = "all-schedules", feature = "schedules-default"))]
    use akita_types::{GrindingQueryKind, GrindingSite, GRINDING_NONCE_SLACK_BITS};

    #[cfg(feature = "schedules-default")]
    #[test]
    fn production_onehot_plan_is_canonical_and_fully_priced() {
        let layout = OpeningClaimsLayout::new(14, 1).expect("opening layout");
        let row = fp128::OneHot::resolve_catalog_row_for_opening(&layout)
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

    #[cfg(feature = "schedules-default")]
    #[test]
    fn stage1_prices_the_full_eq_factored_round_degree() {
        let layout = OpeningClaimsLayout::new(14, 1).expect("opening layout");
        let row = fp128::OneHot::resolve_catalog_row_for_opening(&layout)
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
        fn exact_order<F: PseudoMersenneField>(extension_degree: usize) -> (u32, u128, usize) {
            (F::MODULUS_BITS, F::MODULUS_OFFSET, extension_degree)
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

    #[cfg(feature = "all-schedules")]
    #[test]
    fn every_generated_production_row_derives_a_complete_plan() {
        use crate::proof_optimized::{fp32, fp64};

        fn audit<Cfg: CommitmentConfig>()
        where
            Cfg::Field: CanonicalField,
            Cfg::ExtField: ExtField<Cfg::Field>,
        {
            for entry in Cfg::schedule_catalog().expect("production catalog").entries {
                let row = Cfg::resolve_catalog_row_for_key(&entry.to_runtime_lookup_key())
                    .expect("admitted production row");
                let layout = row.profiles().opening_layout().expect("opening layout");
                let plan = derive_transcript_grinding_plan::<Cfg>(row.schedule(), &layout)
                    .expect("complete grinding plan");
                let count = |kind| plan.runs().iter().filter(|run| run.kind() == kind).count();
                assert_eq!(
                    count(GrindingQueryKind::FoldResponse),
                    row.schedule().num_fold_levels()
                );
                assert_eq!(
                    plan.runs()
                        .iter()
                        .filter(|run| matches!(run.site(), GrindingSite::EvaluationBatch { .. }))
                        .count(),
                    row.schedule().num_fold_levels()
                );
                assert!(count(GrindingQueryKind::FoldChallengeGroup) > 0);
                assert!(plan.expanded_query_count() >= plan.runs().len() as u64);
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
