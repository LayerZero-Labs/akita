use super::aborts::response_within_bound;
use super::commitment::CommitmentBackend;
use super::proof::{QuadraticMask, ZkSigmaProof};
use super::statement::{ZkSigmaStatement, ZkSigmaWitness};
use super::transcript::{append_first_message, append_response, append_statement};
use crate::error::HachiError;
use crate::protocol::transcript::labels;
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore};

/// Sampler for Sigma mask vectors.
pub trait MaskSampler<F: FieldCore> {
    /// Sample one mask vector for a retry attempt.
    ///
    /// # Errors
    ///
    /// Returns an error if the sampler cannot produce a vector of `len`.
    fn sample_mask(&mut self, attempt: u32, len: usize) -> Result<Vec<F>, HachiError>;
}

/// Prove the standalone Sigma statement.
///
/// The prover retries internally until the response satisfies the optional
/// abort bound or `max_attempts` is exhausted.
///
/// # Errors
///
/// Returns an error for malformed statements, invalid witnesses, sampler
/// failures, or if all attempts abort.
pub fn prove<F, T, S>(
    statement: &ZkSigmaStatement<F>,
    witness: &ZkSigmaWitness<F>,
    transcript: &mut T,
    sampler: &mut S,
    max_attempts: u32,
) -> Result<ZkSigmaProof<F>, HachiError>
where
    F: CanonicalField + 'static,
    T: Transcript<F>,
    S: MaskSampler<F>,
{
    if max_attempts == 0 {
        return Err(HachiError::InvalidInput(
            "max_attempts must be positive".into(),
        ));
    }
    statement.check_shapes()?;
    if witness.values.len() != statement.commitment_key.witness_len() {
        return Err(HachiError::InvalidSize {
            expected: statement.commitment_key.witness_len(),
            actual: witness.values.len(),
        });
    }
    let expected_commitment = statement.commitment_key.commit(&witness.values)?;
    if expected_commitment != statement.commitment {
        return Err(HachiError::InvalidInput(
            "witness does not match statement commitment".into(),
        ));
    }

    let mut base_transcript = transcript.clone();
    append_statement(&mut base_transcript, statement);

    for attempt in 0..max_attempts {
        let mask = sampler.sample_mask(attempt, witness.values.len())?;
        if mask.len() != witness.values.len() {
            return Err(HachiError::InvalidSize {
                expected: witness.values.len(),
                actual: mask.len(),
            });
        }

        let mask_commitment = statement.commitment_key.commit(&mask)?;
        let linear_masks = evaluate_linear_masks(statement, &mask)?;
        let quadratic_masks = evaluate_quadratic_masks(statement, &mask)?;
        let mut attempt_transcript = base_transcript.clone();
        append_first_message(
            &mut attempt_transcript,
            attempt,
            &mask_commitment,
            &linear_masks,
            &quadratic_masks,
        );
        let challenge = attempt_transcript.challenge_scalar(labels::CHALLENGE_ZK_SIGMA);
        let response = response_vector(challenge, &witness.values, &mask);
        if response_within_bound(statement.response_linf_bound, &response) {
            append_response(&mut attempt_transcript, &response);
            *transcript = attempt_transcript;
            return Ok(ZkSigmaProof {
                attempt,
                mask_commitment,
                linear_masks,
                quadratic_masks,
                response,
            });
        }
    }

    Err(HachiError::InvalidInput(
        "Sigma prover exhausted all abort attempts".into(),
    ))
}

fn response_vector<F: FieldCore>(challenge: F, witness: &[F], mask: &[F]) -> Vec<F> {
    witness
        .iter()
        .zip(mask)
        .map(|(&w, &y)| challenge * w + y)
        .collect()
}

fn evaluate_linear_masks<F: FieldCore>(
    statement: &ZkSigmaStatement<F>,
    mask: &[F],
) -> Result<Vec<F>, HachiError> {
    statement
        .linear_relations
        .iter()
        .map(|relation| relation.expression.evaluate(mask))
        .collect()
}

fn evaluate_quadratic_masks<F: FieldCore>(
    statement: &ZkSigmaStatement<F>,
    mask: &[F],
) -> Result<Vec<QuadraticMask<F>>, HachiError> {
    statement
        .quadratic_relations
        .iter()
        .map(|relation| {
            Ok(QuadraticMask {
                left: relation.left.evaluate(mask)?,
                right: relation.right.evaluate(mask)?,
                output: relation.output.evaluate(mask)?,
            })
        })
        .collect()
}
