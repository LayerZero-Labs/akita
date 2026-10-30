//! Integrated outer setup checks and optional standalone reductions.
use akita_algebra::eq_poly::{EqPolynomial, SplitEqEvals};
use akita_error::{checked, AkitaError};
use akita_sumcheck::{verify_sumcheck_rounds, SumcheckShape};
use akita_types::{
    setup_claims::{
        transcript::{batch_coefficients, setup_evaluation, SetupBatchVerifierChannel},
        SetupAccumulator, SetupBatchProof, SetupBatchStatement, SetupPolynomialId,
    },
    AkitaExpandedSetup, FpExtEncoding,
};
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};

/// Verify a reduction, leaving exactly the returned setup evaluation unresolved.
pub fn verify_setup_batch_deferred<F, E>(
    expected: &SetupPolynomialId,
    statement: &SetupBatchStatement<E>,
    proof: &SetupBatchProof,
    session: &[u8],
) -> Result<SetupAccumulator<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + ExtField<F>,
{
    if statement.polynomial() != expected {
        return Err(AkitaError::InvalidInput(
            "unexpected setup batch polynomial".into(),
        ));
    }
    let mut channel = SetupBatchVerifierChannel::<F, E>::new(session, statement, proof.argument())?;
    let coefficients = batch_coefficients::<F, E, _>(&mut channel.state, statement.claims().len())?;
    let input_claim = statement.input_claim(&coefficients)?;
    let replay = verify_sumcheck_rounds::<F, E, _>(
        &mut channel,
        0,
        input_claim,
        SumcheckShape::new(expected.num_variables(), 2)?,
    )?;
    let value = setup_evaluation::<F, E, _>(&mut channel.state, E::zero())?;
    let weight = statement.evaluate_weight(&coefficients, &replay.challenges)?;
    if replay.output_claim != value * weight {
        return Err(AkitaError::InvalidProof);
    }
    channel.state.check_eof()?;
    SetupAccumulator::new(expected.clone(), replay.challenges, value)
}

/// Discharge a setup accumulator by scanning the authenticated base-field prefix once.
/// Only small coefficient and split index equality vectors are allocated.
pub fn decide_setup_accumulator<F, E>(
    setup: &AkitaExpandedSetup<F>,
    accumulator: &SetupAccumulator<E>,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + ExtField<F> + Ring + MulBaseUnreduced<F>,
{
    accumulator.polynomial().validate::<F, E>(setup)?;
    if accumulator.point().len() != accumulator.polynomial().num_variables() {
        return Err(AkitaError::InvalidProof);
    }
    let (low, high) = accumulator
        .point()
        .split_at_checked(accumulator.point().len().min(6))
        .ok_or(AkitaError::InvalidProof)?;
    let coefficient = EqPolynomial::evals(low)?;
    let index = SplitEqEvals::new(high)?;
    let source = setup
        .shared_matrix()
        .as_field_slice()
        .get(..accumulator.polynomial().field_len())
        .ok_or(AkitaError::InvalidProof)?;
    let value = cfg_fold_reduce!(
        0..index.len(),
        || Ok(E::zero()),
        |sum: Result<E, AkitaError>, row| {
            let range = checked::range(
                checked::product([row, coefficient.len()]).ok_or(AkitaError::InvalidProof)?,
                coefficient.len(),
            )
            .ok_or(AkitaError::InvalidProof)?;
            let coefficients = source.get(range).ok_or(AkitaError::InvalidProof)?;
            Ok(sum? + index.eval_at(row)? * E::dot_base(&coefficient, coefficients))
        },
        |left: Result<E, AkitaError>, right: Result<E, AkitaError>| Ok(left? + right?)
    )?;
    if value != accumulator.value() {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

mod outer;
pub(crate) use outer::{decide_setup_batch, prepare_outer_setup, PreparedOuterSetup};
