//! Proof-stream atoms emitted after stage-2 sumcheck challenges.

use crate::transcript::{ProtocolSiteId, SITE_FAMILY_STAGE2};
use crate::GrindingReplay;
use akita_error::AkitaError;
use jolt_field::{CanonicalDecode, CanonicalEncoding, ExtField, Field};
use jolt_transcript::Channel;

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
pub fn stage2_w_eval<F, E, G>(
    grinding: &mut G,
    level: u32,
    mut evaluation: E,
) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
    G: GrindingReplay,
{
    let state = grinding.state_mut();
    state.site(witness_evaluation_site(level).into());
    state.exchange(&mut evaluation)?;
    Ok(evaluation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::test_transcripts::{prover as new_prover, verifier as new_verifier};
    use crate::{ChallengeFieldOrder, GrindingPlan, ProverGrinding, VerifierGrinding};
    use jolt_field::{CanonicalBytes, FpExt4, Prime32Offset99, Ring, Zero};

    type F = Prime32Offset99;
    type E = FpExt4<F>;

    #[test]
    fn stage2_witness_evaluation_is_one_extension_atom() {
        let plan = GrindingPlan::new(
            Vec::new(),
            ChallengeFieldOrder::from_full_capacity(128).unwrap(),
        )
        .unwrap();
        let mut transcript = new_prover(b"native-stage2");
        let mut prover = ProverGrinding::new(&mut transcript, &plan);
        let evaluation = E::from_u64(42);
        stage2_w_eval::<F, E, _>(&mut prover, 7, evaluation).unwrap();
        prover.finish().unwrap();
        let proof = transcript.finish();
        assert_eq!(proof.len(), E::DEGREE * F::NUM_BYTES);

        let mut transcript = new_verifier(b"native-stage2", &proof);
        let mut verifier = VerifierGrinding::new(&mut transcript, &plan);
        assert_eq!(
            stage2_w_eval::<F, E, _>(&mut verifier, 7, E::zero()).unwrap(),
            evaluation
        );
        verifier.finish().unwrap();
        transcript.finish().unwrap();
    }
}
