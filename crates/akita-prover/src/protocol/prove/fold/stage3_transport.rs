use crate::protocol::prove::Stage3ProveOutput;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_transcript::ChallengeCodec;
use jolt_algebra::{CanonicalEncoding, ExtField, Field};
use jolt_poly::UnivariatePoly;

/// Extension seam: run Stage 3 transport for a backend session whose public
/// prefix and input claim the caller has already bound, through the terminal
/// prefix evaluation. <https://github.com/LayerZero-Labs/akita/issues/285>.
pub(in crate::protocol::prove) fn prove_stage3_session<F, E, B, C>(
    backend: &B,
    mut session: B::Stage3SessionHandle,
    claim: E,
    rounds: usize,
    grinding: &mut akita_types::ProverGrinding<'_, C>,
    level: u32,
) -> Result<Stage3ProveOutput<E>, AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize,
    E: ExtField<F> + AkitaSerialize + ChallengeCodec,
    B: crate::backend::OpaqueStage3Kernel<F, E>,
    C: akita_types::ProverGrindingChannel + akita_transcript::PrivateProofChannel<F>,
{
    let mut kernel = Stage3Sumcheck {
        backend,
        session: &mut session,
        claim,
        rounds,
        next_round: 0,
        field: std::marker::PhantomData,
    };
    // Stage 3's rounds are public in both modes: the setup product depends
    // only on the public setup and public challenges.
    let mut channel = akita_types::GrindingSumcheckProver::<F, E, _>::new_public(
        grinding,
        akita_params::SumcheckProtocol::Stage3,
        level,
        0,
    );
    let (setup_prefix_point, _) = akita_sumcheck::prove_sumcheck::<F, E, _, _>(
        &mut kernel,
        &mut channel,
        akita_sumcheck::SumcheckShape::new(rounds, akita_params::SETUP_SUMCHECK_DEGREE)?,
        0,
    )?;
    let setup_prefix_eval = backend.finish_stage3(session)?;
    akita_types::stage3_prefix_eval::<F, E, _>(grinding, level, setup_prefix_eval)?;
    Ok(Stage3ProveOutput {
        setup_prefix_eval,
        setup_prefix_point,
    })
}

struct Stage3Sumcheck<'a, F: Field, E: Field, B: crate::backend::OpaqueStage3Kernel<F, E>> {
    backend: &'a B,
    session: &'a mut B::Stage3SessionHandle,
    claim: E,
    rounds: usize,
    next_round: usize,
    field: std::marker::PhantomData<F>,
}

impl<F: Field, E: Field, B: crate::backend::OpaqueStage3Kernel<F, E>>
    akita_sumcheck::SumcheckKernel<E> for Stage3Sumcheck<'_, F, E, B>
{
    fn num_rounds(&self) -> usize {
        self.rounds
    }
    fn degree_bound(&self) -> usize {
        akita_params::SETUP_SUMCHECK_DEGREE
    }
    fn input_claim(&self) -> E {
        self.claim
    }
    fn round_polynomial(
        &mut self,
        round: usize,
        claim: E,
    ) -> Result<UnivariatePoly<E>, AkitaError> {
        self.backend
            .stage3_round_polynomial(self.session, round, claim)
    }
    fn bind_challenge(&mut self, round: usize, challenge: E) -> Result<(), AkitaError> {
        self.backend
            .bind_stage3_challenge(self.session, round, challenge)?;
        self.next_round += 1;
        Ok(())
    }
    fn finish(&mut self) -> Result<(), AkitaError> {
        if self.next_round != self.rounds {
            return Err(AkitaError::Internal(
                "stage 3 sumcheck finished before all rounds were bound".into(),
            ));
        }
        Ok(())
    }
}
