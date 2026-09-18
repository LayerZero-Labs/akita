use super::*;
use crate::RecursiveWitnessFlat;
use akita_config::proof_optimized::fp128::OneHot;
use akita_types::{AkitaScheduleLookupKey, OpeningClaimsLayout, PolynomialGroupLayout};
use jolt_field::{Fp32, FpExt2, One, TwoNr, Zero};

type F = Fp32<251>;
type E = FpExt2<F, TwoNr>;

fn eor_test_plan(rounds: usize, batches_claims: bool) -> akita_types::GrindingPlan {
    let mut runs = vec![akita_types::GrindingRun::proof_of_work(
        akita_types::GrindingSite::ExtensionOpeningPoint { level: 1 },
        1,
        128,
    )
    .unwrap()];
    if batches_claims {
        runs.push(
            akita_types::GrindingRun::proof_of_work(
                akita_types::GrindingSite::ExtensionOpeningClaimBatch { level: 1 },
                1,
                128,
            )
            .unwrap(),
        );
    }
    for round in 0..rounds {
        runs.push(
            akita_types::GrindingRun::proof_of_work(
                akita_types::GrindingSite::SumcheckRound {
                    protocol: akita_types::SumcheckProtocol::ExtensionOpeningReduction,
                    level: 1,
                    stage: 0,
                    round: u32::try_from(round).unwrap(),
                },
                1,
                128,
            )
            .unwrap(),
        );
    }
    akita_types::GrindingPlan::new(runs, 128).unwrap()
}

#[test]
fn coefficient_packing_bypasses_eor_while_evaluation_trace_uses_it() {
    let packing = akita_types::OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: 64,
    };
    assert!(!packing.requires_extension_opening_reduction(2));
    assert!(!packing.requires_extension_opening_reduction(<E as ExtField<F>>::DEGREE));
    assert!(akita_types::OpeningMethod::EvaluationTrace
        .requires_extension_opening_reduction(<E as ExtField<F>>::DEGREE));
}

#[test]
fn native_extension_opening_reduction_streams_without_a_proof_object() {
    let logical_w = RecursiveWitnessFlat::from_i8_digits(vec![1; 3 * 64]);
    let point = (0..8)
        .map(|index| E::new(F::from_u64(index + 2), F::from_u64(index + 17)))
        .collect::<Vec<_>>();
    let logical_polys = [&logical_w];
    let logical_group = PreparedProverGroup::from_refs(&logical_polys).expect("logical group");
    let groups = vec![ExtensionOpeningGroupInput {
        group: &logical_group,
        point: &point,
        ring_dimension: 64,
    }];
    let plan = eor_test_plan(point.len() - 1, false);
    let state = akita_transcript::new_native_prover(b"native-eor-prover", b"fixture").unwrap();
    let mut grinding = akita_types::NativeProverGrinding::new(state, &plan);

    let proved = prove_extension_opening_reduction_native::<F, E, _, _>(
        &crate::compute::CpuBackend::DEFAULT,
        None,
        &groups,
        &mut grinding,
        1,
        "recursive",
    )
    .expect("native EOR proving should stream every proof value");
    let proof = grinding.finish().unwrap();

    assert!(!proof.is_empty());
    assert_eq!(proved.protocol_points.len(), 1);
    assert_eq!(proved.reduction.final_claims.len(), 1);
    assert_eq!(proved.reduction.final_factors.len(), 1);
}

#[test]
fn proof_schedule_from_layout_includes_entire_batch() {
    let catalog = akita_config::test_support::workspace_schedule_catalog::<OneHot>()
        .expect("workspace schedule catalog");
    let batch = OpeningClaimsLayout::from_groups(vec![
        PolynomialGroupLayout::new(16, 1),
        PolynomialGroupLayout::new(16, 1),
        PolynomialGroupLayout::new(32, 2),
    ])
    .expect("multi-group shape");
    assert_eq!(batch.num_groups(), 3);
    let precommitted = catalog
        .resolve_key(&AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(
            16, 1,
        )))
        .expect("independent row")
        .profiles()
        .final_group;
    let schedule = catalog
        .resolve_key(&AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(32, 2),
            precommitteds: vec![precommitted, precommitted],
        })
        .expect("multi-group schedule")
        .schedule()
        .clone();
    let root_params = schedule.root.params.clone();
    assert_eq!(root_params.precommitted_groups().len(), 2);
    for precommitted in root_params.precommitted_groups() {
        assert_eq!(
            precommitted.profile.group,
            PolynomialGroupLayout::new(16, 1)
        );
    }
}
