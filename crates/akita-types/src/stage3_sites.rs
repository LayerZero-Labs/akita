//! Proof-stream grammar for recursive setup-product stage 3.

use crate::preprocessing::{prefix_commitment_digest, StreamIdentity};
use crate::{AkitaSetupSeed, GrindingReplay, SetupPrefixVerifierSlot};
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
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

/// Bind the selected setup-prefix slot:
/// `serialize_compressed(slot id) || prefix_commitment_digest`, the digest
/// recomputed from `slot` in every proof.
pub fn stage3_public_slot<F, G>(
    grinding: &mut G,
    level: u32,
    setup_seed: &AkitaSetupSeed,
    slot: &SetupPrefixVerifierSlot<F>,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize,
    G: GrindingReplay,
{
    let mut bound = Vec::with_capacity(slot.id.compressed_size() + 32);
    slot.id
        .serialize_compressed(&mut bound)
        .map_err(|error| AkitaError::InvalidSetup(format!("setup-prefix slot id: {error}")))?;
    let stream = StreamIdentity::for_field::<F>(setup_seed)?;
    bound.extend_from_slice(
        prefix_commitment_digest(&stream, &slot.id.commitment_profile, &slot.commitment)?
            .as_bytes(),
    );
    public_bytes(grinding.state_mut(), site(level, ROLE_SETUP_SLOT), &bound)
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

#[cfg(all(test, feature = "prover"))]
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
        let id = akita_params::scheduled_setup_prefix(
            64,
            crate::proof::setup::tests::prefix_commitment_params(64, 64),
        )
        .slot_id()
        .unwrap();
        let slot = SetupPrefixVerifierSlot {
            id,
            commitment: crate::SetupPrefixPublicCommitment {
                rows: vec![crate::RingVec::from_coeffs(vec![F::from_u64(3); 16])],
            },
        };
        let seed = AkitaSetupSeed::shake256_paged_v1([5; 32]);
        let claim = E::from_u64(17);
        let prefix_eval = E::from_u64(29);
        let mut state = new_prover_channel(b"native-stage3");
        let mut prover = ProverGrinding::new(&mut state, &plan);
        stage3_public_slot(&mut prover, 3, &seed, &slot).unwrap();
        stage3_claim::<F, E, _>(&mut prover, 3, claim).unwrap();
        stage3_prefix_eval::<F, E, _>(&mut prover, 3, prefix_eval).unwrap();
        prover.finish().unwrap();
        let proof = state.narg_string().to_vec();

        let mut state = new_verifier_channel(b"native-stage3", &proof);
        let mut verifier = VerifierGrinding::new(&mut state, &plan);
        stage3_public_slot(&mut verifier, 3, &seed, &slot).unwrap();
        assert_eq!(
            stage3_claim::<F, E, _>(&mut verifier, 3, E::zero()).unwrap(),
            claim
        );
        assert_eq!(
            stage3_prefix_eval::<F, E, _>(&mut verifier, 3, E::zero()).unwrap(),
            prefix_eval
        );
        verifier.finish().unwrap().check_eof(state).unwrap();
    }
}
