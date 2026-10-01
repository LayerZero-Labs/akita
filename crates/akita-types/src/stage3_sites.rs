//! Proof-stream grammar for recursive setup-product stage 3.

use crate::GrindingReplay;
use akita_error::AkitaError;
use akita_transcript::{
    exchange_extension_group, public_bytes, ProtocolSiteId, SITE_FAMILY_STAGE3,
};
use core::slice;
use jolt_field::{CanonicalEncoding, ExtField, Field};

const ROLE_SETUP_SLOT: u32 = 1;
const ROLE_INPUT_CLAIM: u32 = 2;
const ROLE_PREFIX_EVAL: u32 = 3;

fn site(level: u32, role: u32) -> ProtocolSiteId {
    ProtocolSiteId {
        family: SITE_FAMILY_STAGE3,
        level,
        detail: role,
        ..ProtocolSiteId::default()
    }
}

/// Bind the selected public setup-prefix slot.
pub fn stage3_public_slot<G: GrindingReplay>(
    grinding: &mut G,
    level: u32,
    encoded_slot: &[u8],
) -> Result<(), AkitaError> {
    public_bytes(
        grinding.state_mut(),
        site(level, ROLE_SETUP_SLOT),
        encoded_slot,
    )
}

/// Exchange the setup-product input claim before sumcheck challenges.
pub fn stage3_claim<F, E, G>(grinding: &mut G, level: u32, claim: E) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    G: GrindingReplay,
{
    exchange_one::<F, E, G>(grinding, site(level, ROLE_INPUT_CLAIM), claim)
}

/// Exchange the setup-prefix evaluation after sumcheck challenges.
pub fn stage3_prefix_eval<F, E, G>(
    grinding: &mut G,
    level: u32,
    evaluation: E,
) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    G: GrindingReplay,
{
    exchange_one::<F, E, G>(grinding, site(level, ROLE_PREFIX_EVAL), evaluation)
}

fn exchange_one<F, E, G>(
    grinding: &mut G,
    site: ProtocolSiteId,
    mut value: E,
) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    G: GrindingReplay,
{
    exchange_extension_group::<F, E, _>(grinding.state_mut(), site, slice::from_mut(&mut value))?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ProverGrinding, VerifierGrinding};
    use akita_params::{ChallengeFieldOrder, GrindingPlan};
    use akita_transcript::{new_prover_channel, new_verifier_channel};
    use jolt_field::{FpExt4, Prime32Offset99, Ring, Zero};

    type F = Prime32Offset99;
    type E = FpExt4<F>;

    #[test]
    fn stage3_public_slot_and_late_prefix_eval_roundtrip() {
        let plan = GrindingPlan::new(
            Vec::new(),
            ChallengeFieldOrder::from_full_capacity(128).unwrap(),
        )
        .unwrap();
        let slot = b"canonical setup slot";
        let claim = E::from_u64(17);
        let prefix_eval = E::from_u64(29);
        let state = new_prover_channel(b"native-stage3", b"fixture").unwrap();
        let mut prover = ProverGrinding::new(state, &plan);
        stage3_public_slot(&mut prover, 3, slot).unwrap();
        stage3_claim::<F, E, _>(&mut prover, 3, claim).unwrap();
        stage3_prefix_eval::<F, E, _>(&mut prover, 3, prefix_eval).unwrap();
        let proof = prover.finish().unwrap();

        let state = new_verifier_channel(b"native-stage3", b"fixture", &proof).unwrap();
        let mut verifier = VerifierGrinding::new(state, &plan);
        stage3_public_slot(&mut verifier, 3, slot).unwrap();
        assert_eq!(
            stage3_claim::<F, E, _>(&mut verifier, 3, E::zero()).unwrap(),
            claim
        );
        assert_eq!(
            stage3_prefix_eval::<F, E, _>(&mut verifier, 3, E::zero()).unwrap(),
            prefix_eval
        );
        verifier.finish().unwrap();
    }
}
