use super::aborts::response_within_bound;
use super::commitment::CommitmentBackend;
use super::proof::ZkSigmaProof;
use super::statement::ZkSigmaStatement;
use super::transcript::{append_first_message, append_response, append_statement};
use crate::error::HachiError;
use crate::protocol::transcript::labels;
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore};

/// Verify a standalone Sigma proof.
///
/// # Errors
///
/// Returns an error for malformed statements or proof shapes.
pub fn verify<F, T>(
    statement: &ZkSigmaStatement<F>,
    proof: &ZkSigmaProof<F>,
    transcript: &mut T,
) -> Result<bool, HachiError>
where
    F: CanonicalField + 'static,
    T: Transcript<F>,
{
    statement.check_shapes()?;
    proof.check_against_statement(statement)?;

    append_statement(transcript, statement);
    append_first_message(
        transcript,
        proof.attempt,
        &proof.mask_commitment,
        &proof.linear_masks,
        &proof.quadratic_masks,
    );
    let challenge = transcript.challenge_scalar(labels::CHALLENGE_ZK_SIGMA);
    append_response(transcript, &proof.response);

    if !response_within_bound(statement.response_linf_bound, &proof.response) {
        return Ok(false);
    }

    let actual_commitment = statement.commitment_key.commit(&proof.response)?;
    let expected_commitment = statement.commitment_key.combine_commitments(
        challenge,
        &statement.commitment,
        &proof.mask_commitment,
    )?;
    if actual_commitment != expected_commitment {
        return Ok(false);
    }

    Ok(verify_linear_relations(statement, proof, challenge)?
        && verify_quadratic_relations(statement, proof, challenge)?)
}

fn verify_linear_relations<F: FieldCore>(
    statement: &ZkSigmaStatement<F>,
    proof: &ZkSigmaProof<F>,
    challenge: F,
) -> Result<bool, HachiError> {
    for (relation, &mask_eval) in statement.linear_relations.iter().zip(&proof.linear_masks) {
        let lhs = relation.expression.evaluate(&proof.response)?;
        let rhs = challenge * relation.target + mask_eval;
        if lhs != rhs {
            return Ok(false);
        }
    }
    Ok(true)
}

fn verify_quadratic_relations<F: FieldCore>(
    statement: &ZkSigmaStatement<F>,
    proof: &ZkSigmaProof<F>,
    challenge: F,
) -> Result<bool, HachiError> {
    for (relation, mask) in statement
        .quadratic_relations
        .iter()
        .zip(&proof.quadratic_masks)
    {
        let left_z = relation.left.evaluate(&proof.response)?;
        let right_z = relation.right.evaluate(&proof.response)?;
        let output_z = relation.output.evaluate(&proof.response)?;
        let lhs =
            (left_z - mask.left) * (right_z - mask.right) - challenge * (output_z - mask.output);
        let rhs = challenge.square() * relation.target;
        if lhs != rhs {
            return Ok(false);
        }
    }
    Ok(true)
}
