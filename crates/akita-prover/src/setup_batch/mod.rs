//! External setup batching over backend-owned factored arithmetic.
use crate::backend::{SetupBatchBackend, SetupBatchKernel};
use akita_error::AkitaError;
use akita_sumcheck::{prove_sumcheck, SumcheckShape};
use akita_transcript::ChallengeCodec;
use akita_types::setup_claims::{
    transcript::{
        batch_coefficients, setup_evaluation, standalone_grinding_plan, standalone_prover_channel,
    },
    SetupBatchProof, SetupBatchStatement,
};
use akita_types::{GrindingReplay, GrindingSumcheckProver, ProverGrinding};
use jolt_field::{CanonicalEncoding, ExtField, Field};

/// Reduce all obligations to one setup evaluation, reusing native advice preparation.
pub fn prove_setup_batch<F, E, B>(
    backend: &B,
    statement: &SetupBatchStatement<E>,
    session: &[u8],
    prepared: Vec<B::PreparedQuery>,
) -> Result<SetupBatchProof, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + ChallengeCodec,
    B: SetupBatchBackend<F, E>,
{
    let plan = standalone_grinding_plan::<F, E>(statement)?;
    let mut channel = standalone_prover_channel::<F, E>(session, statement, &plan)?;
    let mut grinding = ProverGrinding::new(&mut channel, &plan);
    let coefficients = batch_coefficients::<F, E, _>(&mut grinding, 0, statement.claims().len())?;
    let mut kernel = backend.begin_setup_batch(statement, prepared, &coefficients)?;
    prove_sumcheck::<F, E, _, _>(
        &mut kernel,
        &mut GrindingSumcheckProver::<F, E>::new(
            &mut grinding,
            akita_params::SumcheckProtocol::SetupBatch,
            0,
            0,
        ),
        SumcheckShape::new(
            statement.polynomial().num_variables(),
            akita_params::SETUP_SUMCHECK_DEGREE,
        )?,
        0,
    )?;
    setup_evaluation::<F, E, _>(grinding.state_mut(), kernel.setup_evaluation()?)?;
    grinding.finish()?;
    Ok(SetupBatchProof::from_argument(
        channel.narg_string().to_vec(),
    ))
}
