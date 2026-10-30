//! One grinding plan for a native coefficient root and its ordinary suffix.

use akita_error::AkitaError;
use akita_params::binary_setup::BINARY_SETUP_P;
use akita_params::{
    multilinear_point_loss_factor, polynomial_identity_loss_factor, powers_batch_loss_factor,
    ChallengeFieldOrder, DigitRangePlan, GrindingPlan, GrindingRun, GrindingSite, SumcheckProtocol,
};

/// Replace the ordinary root queries while preserving every ordinary suffix query.
///
/// This is an experimental execution plan, not catalog or security admission.
/// It is selected explicitly by a native root, never by feature enablement.
/// The resulting single cursor covers response rejection, both root sumchecks,
/// and all subsequent folds. Direct setup evaluation has no root Stage 3 query.
pub fn binary_root_grinding_plan(
    ordinary: &GrindingPlan,
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
    let order = ChallengeFieldOrder::from_field(128, 1, BINARY_SETUP_P)?;
    let rounds = domain_len.trailing_zeros() as usize;
    let mut runs = vec![GrindingRun::fold_response(0)];
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
        // Independent row coefficients, including parity/value and padding.
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
    runs.extend(
        ordinary
            .runs()
            .iter()
            .copied()
            .filter(|run| run.site().level() != 0),
    );
    GrindingPlan::new(runs, order)
}
