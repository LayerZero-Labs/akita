//! Proof-stream atoms emitted after stage-1 sumcheck challenges.

use crate::transcript::{ProtocolSiteId, SITE_FAMILY_STAGE1};
use crate::GrindingReplay;
use akita_error::AkitaError;
use jolt_field::{CanonicalDecode, CanonicalEncoding, ExtField, Field};
use jolt_transcript::Channel;

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
    E: ExtField<F> + CanonicalDecode,
    G: GrindingReplay,
{
    let state = grinding.state_mut();
    state.site(stage1_site(level, stage, ROLE_CHILD_CLAIMS).into());
    Ok(state.exchange_all(claims)?)
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
    E: ExtField<F> + CanonicalDecode,
    G: GrindingReplay,
{
    let state = grinding.state_mut();
    state.site(stage1_site(level, stage, ROLE_RANGE_IMAGE).into());
    state.exchange(&mut evaluation)?;
    Ok(evaluation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::test_transcripts::{prover as new_prover, verifier as new_verifier};
    use crate::{
        ChallengeFieldOrder, GrindingPlan, GrindingRun, GrindingSite, ProverGrinding,
        VerifierGrinding,
    };
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
        let mut transcript = new_prover(b"native-stage1");
        let mut prover = ProverGrinding::new(&mut transcript, &plan);
        stage1_child_claims::<F, E, _>(&mut prover, level, stage, &mut claims).unwrap();
        let prover_gamma = prover
            .grinded_ext_challenge::<F, E>(GrindingSite::Stage1InterstageBatch { level, stage })
            .unwrap();
        stage1_range_image::<F, E, _>(&mut prover, level, stage + 1, range_image).unwrap();
        prover.finish().unwrap();
        let proof = transcript.finish();

        let mut transcript = new_verifier(b"native-stage1", &proof);
        let mut verifier = VerifierGrinding::new(&mut transcript, &plan);
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
        transcript.finish().unwrap();
    }
}
