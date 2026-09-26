//! Native proof-stream atoms emitted after stage-1 sumcheck challenges.

use crate::{NativeProverGrinding, NativeVerifierGrinding};
use akita_error::AkitaError;
use akita_transcript::{
    receive_native_extension_group, send_native_extension_group, ProtocolSiteId, SITE_FAMILY_STAGE1,
};
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

/// Emit one product stage's schedule-fixed child claims after its rounds.
pub fn native_stage1_prover_child_claims<F, E>(
    grinding: &mut NativeProverGrinding<'_>,
    level: u32,
    stage: u32,
    claims: &[E],
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    send_native_extension_group::<F, E>(
        grinding.state_mut(),
        stage1_site(level, stage, ROLE_CHILD_CLAIMS),
        claims,
    )
    .map_err(|_| AkitaError::InvalidProof)
}

/// Receive one product stage's schedule-fixed child claims after its rounds.
pub fn native_stage1_verifier_child_claims<F, E>(
    grinding: &mut NativeVerifierGrinding<'_, '_>,
    level: u32,
    stage: u32,
    count: usize,
) -> Result<Vec<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    receive_native_extension_group::<F, E>(
        grinding.state_mut(),
        stage1_site(level, stage, ROLE_CHILD_CLAIMS),
        count,
    )
    .map_err(|_| AkitaError::InvalidProof)
}

/// Emit the final range-image evaluation after leaf sumcheck challenges.
pub fn native_stage1_prover_range_image<F, E>(
    grinding: &mut NativeProverGrinding<'_>,
    level: u32,
    stage: u32,
    evaluation: E,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    send_native_extension_group::<F, E>(
        grinding.state_mut(),
        stage1_site(level, stage, ROLE_RANGE_IMAGE),
        &[evaluation],
    )
    .map_err(|_| AkitaError::InvalidProof)
}

/// Receive the final range-image evaluation after leaf sumcheck challenges.
pub fn native_stage1_verifier_range_image<F, E>(
    grinding: &mut NativeVerifierGrinding<'_, '_>,
    level: u32,
    stage: u32,
) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let mut values = receive_native_extension_group::<F, E>(
        grinding.state_mut(),
        stage1_site(level, stage, ROLE_RANGE_IMAGE),
        1,
    )
    .map_err(|_| AkitaError::InvalidProof)?;
    values.pop().ok_or(AkitaError::InvalidProof)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GrindingPlan, GrindingRun, GrindingSite};
    use akita_transcript::{new_native_prover, new_native_verifier};
    use jolt_field::{FpExt4, Prime32Offset99, Ring};

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
                128,
            )
            .unwrap()],
            128,
        )
        .unwrap();
        let claims = [E::from_u64(3), E::from_u64(5), E::from_u64(8)];
        let range_image = E::from_u64(13);
        let state = new_native_prover(b"native-stage1", b"fixture").unwrap();
        let mut prover = NativeProverGrinding::new(state, &plan);
        native_stage1_prover_child_claims::<F, E>(&mut prover, level, stage, &claims).unwrap();
        let prover_gamma = prover
            .grinded_ext_challenge::<F, E>(GrindingSite::Stage1InterstageBatch { level, stage })
            .unwrap();
        native_stage1_prover_range_image::<F, E>(&mut prover, level, stage + 1, range_image)
            .unwrap();
        let proof = prover.finish().unwrap();

        let state = new_native_verifier(b"native-stage1", b"fixture", &proof).unwrap();
        let mut verifier = NativeVerifierGrinding::new(state, &plan);
        assert_eq!(
            native_stage1_verifier_child_claims::<F, E>(&mut verifier, level, stage, claims.len(),)
                .unwrap(),
            claims
        );
        let verifier_gamma = verifier
            .grinded_ext_challenge::<F, E>(GrindingSite::Stage1InterstageBatch { level, stage })
            .unwrap();
        assert_eq!(verifier_gamma, prover_gamma);
        assert_eq!(
            native_stage1_verifier_range_image::<F, E>(&mut verifier, level, stage + 1).unwrap(),
            range_image
        );
        verifier.finish().unwrap();
    }
}
