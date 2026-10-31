//! External setup batching over backend-owned factored arithmetic.
use crate::backend::{SetupBatchBackend, SetupBatchKernel};
use akita_error::AkitaError;
use akita_sumcheck::{prove_sumcheck, SumcheckShape};
use akita_types::setup_claims::{
    transcript::{batch_coefficients, setup_evaluation, SetupBatchProverChannel},
    SetupBatchProof, SetupBatchStatement,
};
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
    E: ExtField<F>,
    B: SetupBatchBackend<F, E>,
{
    let mut channel = SetupBatchProverChannel::<F, E>::new(session, statement)?;
    let coefficients = batch_coefficients::<F, E, _>(&mut channel.state, statement.claims().len())?;
    let mut kernel = backend.begin_setup_batch(statement, prepared, &coefficients)?;
    prove_sumcheck::<F, E, _, _>(
        &mut kernel,
        &mut channel,
        SumcheckShape::new(statement.polynomial().num_variables(), 2)?,
        0,
    )?;
    setup_evaluation::<F, E, _>(&mut channel.state, kernel.setup_evaluation()?)?;
    Ok(SetupBatchProof::from_argument(
        channel.state.narg_string().to_vec(),
    ))
}
