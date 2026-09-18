//! Native proof-stream atoms emitted after stage-2 sumcheck challenges.

use crate::{NativeProverGrinding, NativeVerifierGrinding};
use akita_error::AkitaError;
use akita_transcript::{
    receive_native_extension_group, send_native_extension_group, ProtocolSiteId, SITE_FAMILY_STAGE2,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};

fn witness_evaluation_site(level: u32) -> ProtocolSiteId {
    ProtocolSiteId {
        family: SITE_FAMILY_STAGE2,
        level,
        stage: 1,
        ..ProtocolSiteId::default()
    }
}

/// Emit the stage-2 witness evaluation after all sumcheck challenges.
pub fn native_stage2_prover_w_eval<F, E>(
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
        witness_evaluation_site(level),
        &[evaluation],
    )
    .map_err(|_| AkitaError::InvalidProof)
}

/// Receive the stage-2 witness evaluation after replaying all sumcheck rounds.
pub fn native_stage2_verifier_w_eval<F, E>(
    grinding: &mut NativeVerifierGrinding<'_, '_>,
    level: u32,
) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let mut values = receive_native_extension_group::<F, E>(
        grinding.state_mut(),
        witness_evaluation_site(level),
        1,
    )
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
    fn stage2_witness_evaluation_is_one_native_extension_atom() {
        let plan = GrindingPlan::new(Vec::new(), 128).unwrap();
        let state = new_native_prover(b"native-stage2", b"fixture").unwrap();
        let mut prover = NativeProverGrinding::new(state, &plan);
        let evaluation = E::from_u64(42);
        native_stage2_prover_w_eval::<F, E>(&mut prover, 7, evaluation).unwrap();
        let proof = prover.finish().unwrap();
        assert_eq!(proof.len(), E::DEGREE * F::NUM_BYTES);

        let state = new_native_verifier(b"native-stage2", b"fixture", &proof).unwrap();
        let mut verifier = NativeVerifierGrinding::new(state, &plan);
        assert_eq!(
            native_stage2_verifier_w_eval::<F, E>(&mut verifier, 7).unwrap(),
            evaluation
        );
        verifier.finish().unwrap();
    }
}
