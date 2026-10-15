//! Proof-stream grammar for recursive setup-product stage 3.

use crate::GrindingReplay;
use akita_error::AkitaError;
use akita_params::transcript_site::{ProtocolSiteId, SITE_FAMILY_STAGE3};
use jolt_field::{CanonicalDecode, CanonicalEncoding, ExtField, Field};
use jolt_transcript::Channel;

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
pub fn stage3_public_slot<G: GrindingReplay>(grinding: &mut G, level: u32, encoded_slot: &[u8]) {
    let state = grinding.state_mut();
    state.site(site(level, ROLE_SETUP_SLOT).into());
    state.public_bytes(encoded_slot);
}

/// Exchange the setup-product input claim before sumcheck challenges.
pub fn stage3_claim<F, E, G>(grinding: &mut G, level: u32, claim: E) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
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
    E: ExtField<F> + CanonicalDecode,
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
    E: ExtField<F> + CanonicalDecode,
    G: GrindingReplay,
{
    let state = grinding.state_mut();
    state.site(site.into());
    state.exchange(&mut value)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::test_transcripts::{prover as new_prover, verifier as new_verifier};
    use crate::{ProverGrinding, VerifierGrinding};
    use akita_params::{ChallengeFieldOrder, GrindingPlan};
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
        let mut transcript = new_prover(b"native-stage3");
        let mut prover = ProverGrinding::new(&mut transcript, &plan);
        stage3_public_slot(&mut prover, 3, slot);
        stage3_claim::<F, E, _>(&mut prover, 3, claim).unwrap();
        stage3_prefix_eval::<F, E, _>(&mut prover, 3, prefix_eval).unwrap();
        prover.finish().unwrap();
        let proof = transcript.finish();

        let mut transcript = new_verifier(b"native-stage3", &proof);
        let mut verifier = VerifierGrinding::new(&mut transcript, &plan);
        stage3_public_slot(&mut verifier, 3, slot);
        assert_eq!(
            stage3_claim::<F, E, _>(&mut verifier, 3, E::zero()).unwrap(),
            claim
        );
        assert_eq!(
            stage3_prefix_eval::<F, E, _>(&mut verifier, 3, E::zero()).unwrap(),
            prefix_eval
        );
        verifier.finish().unwrap();
        transcript.finish().unwrap();
    }
}
