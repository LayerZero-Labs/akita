//! Typed adapter for canonical transcript-grinding plan derivation.

use crate::CommitmentConfig;
use akita_error::AkitaError;
use akita_field::{CanonicalField, ExtField};
use akita_types::{BasisMode, FoldSchedule, GrindingPlan, OpeningClaimsLayout};

/// Derive the only accepted grinding plan for one effective schedule and call.
pub fn derive_transcript_grinding_plan<Cfg: CommitmentConfig>(
    schedule: &FoldSchedule,
    root_layout: &OpeningClaimsLayout,
    _basis: BasisMode,
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
        let plan = derive_transcript_grinding_plan::<fp128::OneHot>(
            row.schedule(),
            &layout,
            BasisMode::Lagrange,
        )
        .expect("grinding plan");

        assert_eq!(
            plan.runs().first().unwrap().site(),
            GrindingSite::EvaluationBatch
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
                46,
                51,
                366,
                [
                    51, 160, 87, 176, 119, 237, 86, 152, 107, 131, 142, 46, 249, 143, 16, 50, 190,
                    49, 242, 207, 136, 30, 42, 249, 19, 22, 15, 17, 91, 211, 133, 102,
                ],
            )
        );
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
                let plan = derive_transcript_grinding_plan::<Cfg>(
                    row.schedule(),
                    &layout,
                    BasisMode::Lagrange,
                )
                .expect("complete grinding plan");
                let count = |kind| plan.runs().iter().filter(|run| run.kind() == kind).count();
                assert_eq!(
                    count(GrindingQueryKind::FoldResponse),
                    row.schedule().num_fold_levels()
                );
                assert_eq!(
                    count(GrindingQueryKind::FoldChallengeRoot),
                    count(GrindingQueryKind::FoldChallengeCoordinates)
                );
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
