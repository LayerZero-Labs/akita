//! Proof-stream atoms emitted after stage-1 sumcheck challenges.

use crate::GrindingReplay;
use akita_error::AkitaError;
use akita_transcript::{exchange_extension_group, ProtocolSiteId, SITE_FAMILY_STAGE1};
use core::slice;
use jolt_field::{CanonicalEncoding, ExtField, Field};

const ROLE_CHILD_CLAIMS: u32 = 1;
const ROLE_RANGE_IMAGE: u32 = 2;

fn stage1_site(level: u32, stage: u32, role: u32) -> ProtocolSiteId {
    ProtocolSiteId {
        family: SITE_FAMILY_STAGE1,
        level,
        stage,
        detail: role,
        ..ProtocolSiteId::default()
    }
}

/// Exchange one product stage's schedule-fixed child claims after its rounds.
///
/// The verifier passes `claims` sized from the schedule and receives them in
/// place.
pub fn stage1_child_claims<F, E, G>(
    grinding: &mut G,
    level: u32,
    stage: u32,
    claims: &mut [E],
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    G: GrindingReplay,
{
    exchange_extension_group::<F, E, _>(
        grinding.state_mut(),
        stage1_site(level, stage, ROLE_CHILD_CLAIMS),
        claims,
    )
}

/// Exchange the final range-image evaluation after leaf sumcheck challenges.
pub fn stage1_range_image<F, E, G>(
    grinding: &mut G,
    level: u32,
    stage: u32,
    mut evaluation: E,
) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    G: GrindingReplay,
{
    exchange_extension_group::<F, E, _>(
        grinding.state_mut(),
        stage1_site(level, stage, ROLE_RANGE_IMAGE),
        slice::from_mut(&mut evaluation),
    )?;
    Ok(evaluation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ProverGrinding, VerifierGrinding};
    use akita_params::{ChallengeFieldOrder, GrindingPlan, GrindingRun, GrindingSite};
    use akita_transcript::{new_prover_channel, new_verifier_channel};
    use jolt_field::{FpExt4, Prime32Offset99, Ring, Zero};

    type F = Prime32Offset99;
    type E = FpExt4<F>;

    #[test]
    fn stage1_late_claims_precede_the_interstage_challenge() {
        let level = 2;
        let stage = 1;
        let plan = GrindingPlan::new(
            vec![GrindingRun::proof_of_work(
                GrindingSite::Stage1InterstageBatch { level, stage },
                1,
                ChallengeFieldOrder::from_full_capacity(128).unwrap(),
            )
            .unwrap()],
            ChallengeFieldOrder::from_full_capacity(128).unwrap(),
        )
        .unwrap();
        let mut claims = [E::from_u64(3), E::from_u64(5), E::from_u64(8)];
        let range_image = E::from_u64(13);
        let state = new_prover_channel(b"native-stage1", b"fixture").unwrap();
        let mut prover = ProverGrinding::new(state, &plan);
        stage1_child_claims::<F, E, _>(&mut prover, level, stage, &mut claims).unwrap();
        let prover_gamma = prover
            .grinded_ext_challenge::<F, E>(GrindingSite::Stage1InterstageBatch { level, stage })
            .unwrap();
        stage1_range_image::<F, E, _>(&mut prover, level, stage + 1, range_image).unwrap();
        let proof = prover.finish().unwrap();

        let state = new_verifier_channel(b"native-stage1", b"fixture", &proof).unwrap();
        let mut verifier = VerifierGrinding::new(state, &plan);
        let mut received = [E::zero(); 3];
        stage1_child_claims::<F, E, _>(&mut verifier, level, stage, &mut received).unwrap();
        assert_eq!(received, claims);
        let verifier_gamma = verifier
            .grinded_ext_challenge::<F, E>(GrindingSite::Stage1InterstageBatch { level, stage })
            .unwrap();
        assert_eq!(verifier_gamma, prover_gamma);
        assert_eq!(
            stage1_range_image::<F, E, _>(&mut verifier, level, stage + 1, E::zero()).unwrap(),
            range_image
        );
        verifier.finish().unwrap();
    }
}
