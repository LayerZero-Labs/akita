//! Native proof-stream atoms emitted after stage-2 sumcheck challenges.

use crate::NativeGrinding;
use akita_error::AkitaError;
use akita_transcript::{exchange_native_extension_group, ProtocolSiteId, SITE_FAMILY_STAGE2};
use core::slice;
use jolt_field::{CanonicalEncoding, ExtField, Field};

fn witness_evaluation_site(level: u32) -> ProtocolSiteId {
    ProtocolSiteId {
        family: SITE_FAMILY_STAGE2,
        level,
        stage: 1,
        ..ProtocolSiteId::default()
    }
}

/// Exchange the stage-2 witness evaluation after all sumcheck challenges.
///
/// The prover emits `evaluation`; the verifier passes a placeholder and gets
/// the received value back.
pub fn native_stage2_w_eval<F, E, G>(
    grinding: &mut G,
    level: u32,
    mut evaluation: E,
) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    G: NativeGrinding,
{
    exchange_native_extension_group::<F, E, _>(
        grinding.state_mut(),
        witness_evaluation_site(level),
        slice::from_mut(&mut evaluation),
    )?;
    Ok(evaluation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChallengeFieldOrder, GrindingPlan, NativeProverGrinding, NativeVerifierGrinding};
    use akita_transcript::{new_native_prover, new_native_verifier};
    use jolt_field::{CanonicalBytes, FpExt4, Prime32Offset99, Ring, Zero};

    type F = Prime32Offset99;
    type E = FpExt4<F>;

    #[test]
    fn stage2_witness_evaluation_is_one_native_extension_atom() {
        let plan = GrindingPlan::new(
            Vec::new(),
            ChallengeFieldOrder::from_full_capacity(128).unwrap(),
        )
        .unwrap();
        let state = new_native_prover(b"native-stage2", b"fixture").unwrap();
        let mut prover = NativeProverGrinding::new(state, &plan);
        let evaluation = E::from_u64(42);
        native_stage2_w_eval::<F, E, _>(&mut prover, 7, evaluation).unwrap();
        let proof = prover.finish().unwrap();
        assert_eq!(proof.len(), E::DEGREE * F::NUM_BYTES);

        let state = new_native_verifier(b"native-stage2", b"fixture", &proof).unwrap();
        let mut verifier = NativeVerifierGrinding::new(state, &plan);
        assert_eq!(
            native_stage2_w_eval::<F, E, _>(&mut verifier, 7, E::zero()).unwrap(),
            evaluation
        );
        verifier.finish().unwrap();
    }
}
