//! Native proof-stream grammar for recursive setup-product stage 3.

use crate::{NativeProverGrinding, NativeVerifierGrinding};
use akita_error::AkitaError;
use akita_transcript::{
    public_native_bytes_prover, public_native_bytes_verifier, receive_native_extension_group,
    send_native_extension_group, ProtocolSiteId, SITE_FAMILY_STAGE3,
};
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

/// Bind the selected public setup-prefix slot on the prover side.
pub fn native_stage3_public_slot_prover(
    grinding: &mut NativeProverGrinding<'_>,
    level: u32,
    encoded_slot: &[u8],
) -> Result<(), AkitaError> {
    public_native_bytes_prover(
        grinding.state_mut(),
        site(level, ROLE_SETUP_SLOT),
        encoded_slot,
    )
    .map_err(|_| AkitaError::InvalidProof)
}

/// Bind the selected public setup-prefix slot on the verifier side.
pub fn native_stage3_public_slot_verifier(
    grinding: &mut NativeVerifierGrinding<'_, '_>,
    level: u32,
    encoded_slot: &[u8],
) -> Result<(), AkitaError> {
    public_native_bytes_verifier(
        grinding.state_mut(),
        site(level, ROLE_SETUP_SLOT),
        encoded_slot,
    )
    .map_err(|_| AkitaError::InvalidProof)
}

/// Emit the setup-product input claim before sumcheck challenges.
pub fn native_stage3_prover_claim<F, E>(
    grinding: &mut NativeProverGrinding<'_>,
    level: u32,
    claim: E,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    send_native_extension_group::<F, E>(
        grinding.state_mut(),
        site(level, ROLE_INPUT_CLAIM),
        &[claim],
    )
    .map_err(|_| AkitaError::InvalidProof)
}

/// Receive the setup-product input claim before sumcheck challenges.
pub fn native_stage3_verifier_claim<F, E>(
    grinding: &mut NativeVerifierGrinding<'_, '_>,
    level: u32,
) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    receive_one::<F, E>(grinding, site(level, ROLE_INPUT_CLAIM))
}

/// Emit the setup-prefix evaluation after sumcheck challenges.
pub fn native_stage3_prover_prefix_eval<F, E>(
    grinding: &mut NativeProverGrinding<'_>,
    level: u32,
    evaluation: E,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    send_native_extension_group::<F, E>(
        grinding.state_mut(),
        site(level, ROLE_PREFIX_EVAL),
        &[evaluation],
    )
    .map_err(|_| AkitaError::InvalidProof)
}

/// Receive the setup-prefix evaluation after sumcheck challenges.
pub fn native_stage3_verifier_prefix_eval<F, E>(
    grinding: &mut NativeVerifierGrinding<'_, '_>,
    level: u32,
) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    receive_one::<F, E>(grinding, site(level, ROLE_PREFIX_EVAL))
}

fn receive_one<F, E>(
    grinding: &mut NativeVerifierGrinding<'_, '_>,
    site: ProtocolSiteId,
) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let mut values = receive_native_extension_group::<F, E>(grinding.state_mut(), site, 1)
        .map_err(|_| AkitaError::InvalidProof)?;
    values.pop().ok_or(AkitaError::InvalidProof)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GrindingPlan;
    use akita_transcript::{new_native_prover, new_native_verifier};
    use jolt_field::{FpExt4, Prime32Offset99, Ring};

    type F = Prime32Offset99;
    type E = FpExt4<F>;

    #[test]
    fn stage3_public_slot_and_late_prefix_eval_roundtrip() {
        let plan = GrindingPlan::new(Vec::new(), 128).unwrap();
        let slot = b"canonical setup slot";
        let claim = E::from_u64(17);
        let prefix_eval = E::from_u64(29);
        let state = new_native_prover(b"native-stage3", b"fixture").unwrap();
        let mut prover = NativeProverGrinding::new(state, &plan);
        native_stage3_public_slot_prover(&mut prover, 3, slot).unwrap();
        native_stage3_prover_claim::<F, E>(&mut prover, 3, claim).unwrap();
        native_stage3_prover_prefix_eval::<F, E>(&mut prover, 3, prefix_eval).unwrap();
        let proof = prover.finish().unwrap();

        let state = new_native_verifier(b"native-stage3", b"fixture", &proof).unwrap();
        let mut verifier = NativeVerifierGrinding::new(state, &plan);
        native_stage3_public_slot_verifier(&mut verifier, 3, slot).unwrap();
        assert_eq!(
            native_stage3_verifier_claim::<F, E>(&mut verifier, 3).unwrap(),
            claim
        );
        assert_eq!(
            native_stage3_verifier_prefix_eval::<F, E>(&mut verifier, 3).unwrap(),
            prefix_eval
        );
        verifier.finish().unwrap();
    }
}
