#![allow(missing_docs)]

use hachi_pcs::algebra::Fp64;
use hachi_pcs::protocol::transcript::labels;
use hachi_pcs::protocol::{
    prove_zk_sigma, verify_zk_sigma, Blake2bTranscript, CommitmentBackend, LinearExpression,
    LinearRelation, MaskSampler, MatrixCommitmentKey, QuadraticRelation, Transcript, ZkSigmaProof,
    ZkSigmaStatement, ZkSigmaWitness,
};
use hachi_pcs::{FieldCore, FromSmallInt, HachiDeserialize, HachiError, HachiSerialize};

type F = Fp64<4294967197>;

fn f(value: i64) -> F {
    F::from_i64(value)
}

#[derive(Clone)]
struct ScriptedSampler {
    masks: Vec<Vec<F>>,
}

impl MaskSampler<F> for ScriptedSampler {
    fn sample_mask(&mut self, attempt: u32, len: usize) -> Result<Vec<F>, HachiError> {
        let mask = self
            .masks
            .get(attempt as usize)
            .cloned()
            .ok_or_else(|| HachiError::InvalidInput("missing scripted mask".into()))?;
        assert_eq!(mask.len(), len);
        Ok(mask)
    }
}

fn expr(coeffs: &[i64]) -> LinearExpression<F> {
    LinearExpression {
        coeffs: coeffs.iter().copied().map(f).collect(),
        constant: F::zero(),
    }
}

fn sample_statement_and_witness() -> (ZkSigmaStatement<F>, ZkSigmaWitness<F>) {
    let commitment_key =
        MatrixCommitmentKey::new(2, 3, vec![f(1), f(2), f(0), f(0), f(1), f(1)]).unwrap();
    let witness = ZkSigmaWitness {
        values: vec![f(2), f(3), f(4)],
    };
    let commitment = commitment_key.commit(&witness.values).unwrap();
    let linear_relations = vec![LinearRelation {
        expression: expr(&[1, 1, 1]),
        target: f(9),
    }];
    let quadratic_relations = vec![QuadraticRelation {
        left: expr(&[1, 0, 0]),
        right: expr(&[0, 1, 0]),
        output: expr(&[0, 0, 1]),
        target: f(2),
    }];
    let statement = ZkSigmaStatement {
        commitment_key,
        commitment,
        linear_relations,
        quadratic_relations,
        response_linf_bound: None,
    };
    (statement, witness)
}

#[test]
fn zk_sigma_accepts_linear_and_quadratic_relations() {
    let (statement, witness) = sample_statement_and_witness();
    let mut sampler = ScriptedSampler {
        masks: vec![vec![f(5), f(7), f(11)]],
    };
    let mut prover_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    let proof = prove_zk_sigma(
        &statement,
        &witness,
        &mut prover_transcript,
        &mut sampler,
        1,
    )
    .unwrap();

    let mut verifier_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    assert!(
        verify_zk_sigma(&statement, &proof, &mut verifier_transcript).unwrap(),
        "valid Sigma proof should verify"
    );

    let prover_tail = prover_transcript.challenge_scalar(b"zk-sigma/test-tail");
    let verifier_tail = verifier_transcript.challenge_scalar(b"zk-sigma/test-tail");
    assert_eq!(prover_tail, verifier_tail);
}

#[test]
fn zk_sigma_rejects_tampered_linear_target() {
    let (mut statement, witness) = sample_statement_and_witness();
    let mut sampler = ScriptedSampler {
        masks: vec![vec![f(5), f(7), f(11)]],
    };
    let mut prover_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    let proof = prove_zk_sigma(
        &statement,
        &witness,
        &mut prover_transcript,
        &mut sampler,
        1,
    )
    .unwrap();
    statement.linear_relations[0].target += f(1);

    let mut verifier_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    assert!(!verify_zk_sigma(&statement, &proof, &mut verifier_transcript).unwrap());
}

#[test]
fn zk_sigma_rejects_tampered_response() {
    let (statement, witness) = sample_statement_and_witness();
    let mut sampler = ScriptedSampler {
        masks: vec![vec![f(5), f(7), f(11)]],
    };
    let mut prover_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    let mut proof = prove_zk_sigma(
        &statement,
        &witness,
        &mut prover_transcript,
        &mut sampler,
        1,
    )
    .unwrap();
    proof.response[0] += f(1);

    let mut verifier_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    assert!(!verify_zk_sigma(&statement, &proof, &mut verifier_transcript).unwrap());
}

#[test]
fn zk_sigma_aborts_until_response_bound_passes() {
    let commitment_key = MatrixCommitmentKey::new(1, 3, vec![f(1), f(0), f(0)]).unwrap();
    let witness = ZkSigmaWitness {
        values: vec![F::zero(), F::zero(), F::zero()],
    };
    let statement = ZkSigmaStatement {
        commitment: commitment_key.commit(&witness.values).unwrap(),
        commitment_key,
        linear_relations: vec![LinearRelation {
            expression: expr(&[1, 0, 0]),
            target: F::zero(),
        }],
        quadratic_relations: vec![],
        response_linf_bound: Some(3),
    };
    let mut sampler = ScriptedSampler {
        masks: vec![vec![f(20), f(0), f(0)], vec![f(1), f(-2), f(3)]],
    };
    let mut prover_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    let proof = prove_zk_sigma(
        &statement,
        &witness,
        &mut prover_transcript,
        &mut sampler,
        2,
    )
    .unwrap();
    assert_eq!(proof.attempt, 1);

    let mut verifier_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    assert!(verify_zk_sigma(&statement, &proof, &mut verifier_transcript).unwrap());
}

#[test]
fn zk_sigma_is_deterministic_for_identical_transcripts_and_masks() {
    let (statement, witness) = sample_statement_and_witness();
    let sampler = ScriptedSampler {
        masks: vec![vec![f(5), f(7), f(11)]],
    };

    let mut sampler_a = sampler.clone();
    let mut transcript_a = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    let proof_a =
        prove_zk_sigma(&statement, &witness, &mut transcript_a, &mut sampler_a, 1).unwrap();

    let mut sampler_b = sampler;
    let mut transcript_b = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    let proof_b =
        prove_zk_sigma(&statement, &witness, &mut transcript_b, &mut sampler_b, 1).unwrap();

    assert_eq!(proof_a, proof_b);
}

#[test]
fn zk_sigma_proof_serializes_roundtrip() {
    let (statement, witness) = sample_statement_and_witness();
    let mut sampler = ScriptedSampler {
        masks: vec![vec![f(5), f(7), f(11)]],
    };
    let mut transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    let proof = prove_zk_sigma(&statement, &witness, &mut transcript, &mut sampler, 1).unwrap();

    let mut bytes = Vec::new();
    proof.serialize_uncompressed(&mut bytes).unwrap();
    let decoded = ZkSigmaProof::<F>::deserialize_uncompressed(&bytes[..], &()).unwrap();
    assert_eq!(proof, decoded);
}

#[test]
fn zk_sigma_statement_serializes_roundtrip() {
    let (statement, _) = sample_statement_and_witness();
    let mut bytes = Vec::new();
    statement.serialize_uncompressed(&mut bytes).unwrap();
    let decoded = ZkSigmaStatement::<F>::deserialize_uncompressed(&bytes[..], &()).unwrap();
    assert_eq!(statement, decoded);
}
