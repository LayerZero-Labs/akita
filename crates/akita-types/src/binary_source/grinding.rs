//! One grinding plan for a native coefficient root and its ordinary suffix.

use akita_error::AkitaError;
use akita_params::binary_setup::BinaryFieldProfile;
use akita_params::{
    multilinear_point_loss_factor, polynomial_identity_loss_factor, powers_batch_loss_factor,
    DigitRangePlan, GrindingPlan, GrindingRun, GrindingSite, SumcheckProtocol,
};

/// Replace the ordinary root queries while preserving every ordinary suffix query.
///
/// This is an experimental execution plan, not catalog or security admission.
/// It is selected explicitly by a native root, never by feature enablement.
/// The resulting single cursor covers response rejection, both root sumchecks,
/// the root Stage 3, and all subsequent folds.
pub fn binary_root_grinding_plan(
    ordinary: &GrindingPlan,
    field_profile: BinaryFieldProfile,
    fold_coordinates: &[usize],
    domain_len: usize,
    maximum_native_degree: usize,
) -> Result<GrindingPlan, AkitaError> {
    if fold_coordinates.is_empty()
        || fold_coordinates.contains(&0)
        || !domain_len.is_power_of_two()
        || maximum_native_degree < 648
    {
        return Err(AkitaError::InvalidSetup(
            "invalid native root grinding geometry".into(),
        ));
    }
    // The selected pair owns the exact order. Column work still uses the
    // unchanged 136-bit family, independently of this field order.
    let order = field_profile.challenge_order()?;
    let rounds = domain_len.trailing_zeros() as usize;
    let challenges = fold_coordinates
        .iter()
        .try_fold(0u64, |sum, &count| {
            sum.checked_add(u64::try_from(count).ok()?)
        })
        .ok_or(AkitaError::InvalidProof)?;
    let mut runs = vec![
        GrindingRun::fold_response(0),
        GrindingRun::fold_challenge_work(
            0,
            challenges,
            super::root_transcript::binary_root_family()?.set_bits(),
        )?,
    ];
    for (group, &count) in fold_coordinates.iter().enumerate() {
        runs.push(GrindingRun::fold_challenge_group(
            0,
            u32::try_from(group).map_err(|_| AkitaError::InvalidProof)?,
            u64::try_from(count).map_err(|_| AkitaError::InvalidProof)?,
        )?);
    }
    for (site, loss) in [
        (
            GrindingSite::RingSwitchAlpha { level: 0 },
            polynomial_identity_loss_factor(maximum_native_degree)?,
        ),
        (
            GrindingSite::Tau0Point { level: 0 },
            multilinear_point_loss_factor(rounds)?,
        ),
        // Independent row coefficients: projections, parity/value, B/D,
        // ordinary rows, negative bits and prime planes. No padding row is
        // batched. Loss 1 rests on the checkpoint's candidate-comparison
        // premise (specs/binary-source-checkpoint.md, section 8): two differing
        // candidates under one checkpoint input yield a collision, so the
        // residual vector is fixed before these draws. A coordinate-wise tree
        // extractor would price the vector at its length instead; see
        // specs/binary-source-zero-padding.md, section 3.5.
        (GrindingSite::Tau1Point { level: 0 }, 1),
    ] {
        runs.push(GrindingRun::proof_of_work(site, loss, order)?);
    }
    for (stage, shape) in DigitRangePlan::new(8)?
        .stage_shapes(rounds)
        .into_iter()
        .enumerate()
    {
        let stage = u32::try_from(stage).map_err(|_| AkitaError::InvalidProof)?;
        let degree = shape
            .sumcheck_proof
            .1
            .checked_add(1)
            .ok_or(AkitaError::InvalidProof)?;
        for round in 0..shape.sumcheck_proof.0 {
            runs.push(GrindingRun::proof_of_work(
                GrindingSite::SumcheckRound {
                    protocol: SumcheckProtocol::Stage1,
                    level: 0,
                    stage,
                    round: u32::try_from(round).map_err(|_| AkitaError::InvalidProof)?,
                },
                polynomial_identity_loss_factor(degree)?,
                order,
            )?);
        }
        if shape.child_claims > 0 {
            runs.push(GrindingRun::proof_of_work(
                GrindingSite::Stage1InterstageBatch { level: 0, stage },
                powers_batch_loss_factor(shape.child_claims)?,
                order,
            )?);
        }
    }
    runs.push(GrindingRun::proof_of_work(
        GrindingSite::Stage2Batch { level: 0 },
        1,
        order,
    )?);
    for round in 0..rounds {
        runs.push(GrindingRun::proof_of_work(
            GrindingSite::SumcheckRound {
                protocol: SumcheckProtocol::Stage2,
                level: 0,
                stage: 0,
                round: u32::try_from(round).map_err(|_| AkitaError::InvalidProof)?,
            },
            polynomial_identity_loss_factor(3)?,
            order,
        )?);
    }
    runs.extend(ordinary.runs().iter().copied().filter(|run| {
        run.site().level() != 0
            || matches!(
                run.site(),
                GrindingSite::SumcheckRound {
                    protocol: SumcheckProtocol::Stage3,
                    ..
                }
            )
    }));
    GrindingPlan::new(runs, order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p64_keeps_column_work_and_suffix_with_exact_field_pricing() {
        for profile in [BinaryFieldProfile::P128, BinaryFieldProfile::P64] {
            let order = profile.challenge_order().unwrap();
            let suffix = GrindingRun::proof_of_work(
                GrindingSite::SumcheckRound {
                    protocol: SumcheckProtocol::Stage3,
                    level: 0,
                    stage: 0,
                    round: 0,
                },
                2,
                order,
            )
            .unwrap();
            let ordinary = GrindingPlan::new(vec![suffix], order).unwrap();
            for (coordinates, bits) in [(256, 0), (257, 1), (512, 1), (1024, 2), (1_048_576, 12)] {
                let plan =
                    binary_root_grinding_plan(&ordinary, profile, &[coordinates], 1 << 17, 648)
                        .unwrap();
                assert_eq!(plan.runs()[0], GrindingRun::fold_response(0));
                let work = plan.runs()[1];
                assert_eq!(work.site(), GrindingSite::FoldChallengeWork { level: 0 });
                assert_eq!(work.challenge_set_bits(), 136);
                assert_eq!(work.loss_factor(), coordinates as u64);
                assert_eq!(work.grind_bits(), bits);
                assert_eq!(plan.runs().last(), Some(&suffix));
                let batch = plan
                    .runs()
                    .iter()
                    .find(|run| run.site() == GrindingSite::Tau1Point { level: 0 })
                    .unwrap();
                assert_eq!(batch.loss_factor(), 1);
                assert_eq!(batch.grind_bits(), order.grind_bits(1).unwrap());
            }
        }
    }
}
