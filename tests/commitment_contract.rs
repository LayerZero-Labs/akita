#![allow(missing_docs)]

use hachi_pcs::algebra::Fp64;
use hachi_pcs::protocol::commitment::{DummyProof, HachiCommitment};
use hachi_pcs::protocol::transcript::labels;
use hachi_pcs::protocol::{
    AppendToTranscript, Blake2bTranscript, CommitmentScheme, StreamingCommitmentScheme, Transcript,
};
use hachi_pcs::{CanonicalField, FieldCore, HachiError, Polynomial};

type F = Fp64<4294967197>;

#[derive(Clone)]
struct SimplePoly {
    coeffs: Vec<F>,
}

impl Polynomial<F> for SimplePoly {
    fn num_vars(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    fn evaluate(&self, point: &[F]) -> F {
        assert_eq!(point.len(), self.num_vars());
        let mut acc = self.coeffs[0];
        for (i, r_i) in point.iter().enumerate() {
            acc = acc + self.coeffs[i + 1] * *r_i;
        }
        acc
    }

    fn coeffs(&self) -> Vec<F> {
        self.coeffs.clone()
    }
}

#[derive(Clone)]
struct DummySetup {
    _max_num_vars: usize,
}

#[derive(Clone)]
struct DummyScheme;

impl CommitmentScheme<F> for DummyScheme {
    type ProverSetup = DummySetup;
    type VerifierSetup = DummySetup;
    type Commitment = HachiCommitment;
    type Proof = DummyProof;
    type OpeningProofHint = HachiCommitment;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        DummySetup {
            _max_num_vars: max_num_vars,
        }
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        setup.clone()
    }

    fn commit<P: Polynomial<F>>(
        poly: &P,
        _setup: &Self::ProverSetup,
    ) -> Result<(Self::Commitment, Self::OpeningProofHint), HachiError> {
        let zero = vec![F::zero(); poly.num_vars()];
        let c = HachiCommitment(poly.evaluate(&zero).to_canonical_u128());
        Ok((c, c))
    }

    fn prove<T: Transcript<F>, P: Polynomial<F>>(
        setup: &Self::ProverSetup,
        poly: &P,
        opening_point: &[F],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut T,
    ) -> Result<Self::Proof, HachiError> {
        if opening_point.len() != poly.num_vars() {
            return Err(HachiError::InvalidPointDimension {
                expected: poly.num_vars(),
                actual: opening_point.len(),
            });
        }

        let absorb_commitment = if let Some(h) = hint {
            h
        } else {
            Self::commit(poly, setup)?.0
        };
        absorb_commitment.append_to_transcript(labels::ABSORB_COMMITMENT, transcript);

        let q = transcript.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
        let opening = poly.evaluate(opening_point);
        Ok(DummyProof(
            opening.to_canonical_u128() ^ q.to_canonical_u128(),
        ))
    }

    fn verify<T: Transcript<F>>(
        proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        transcript: &mut T,
        _opening_point: &[F],
        opening: &F,
        commitment: &Self::Commitment,
    ) -> Result<(), HachiError> {
        commitment.append_to_transcript(labels::ABSORB_COMMITMENT, transcript);
        let q = transcript.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
        let expected = opening.to_canonical_u128() ^ q.to_canonical_u128();
        if proof.0 == expected {
            Ok(())
        } else {
            Err(HachiError::InvalidProof)
        }
    }

    fn combine_commitments(commitments: &[Self::Commitment], coeffs: &[F]) -> Self::Commitment {
        let acc = commitments
            .iter()
            .zip(coeffs.iter())
            .fold(0u128, |sum, (c, coeff)| {
                sum.wrapping_add(c.0.wrapping_mul(coeff.to_canonical_u128()))
            });
        HachiCommitment(acc)
    }

    fn combine_hints(hints: Vec<Self::OpeningProofHint>, coeffs: &[F]) -> Self::OpeningProofHint {
        let acc = hints
            .iter()
            .zip(coeffs.iter())
            .fold(0u128, |sum, (h, coeff)| {
                sum.wrapping_add(h.0.wrapping_mul(coeff.to_canonical_u128()))
            });
        HachiCommitment(acc)
    }

    fn protocol_name() -> &'static [u8] {
        b"HachiDummy"
    }
}

impl StreamingCommitmentScheme<F> for DummyScheme {
    type ChunkState = HachiCommitment;

    fn process_chunk(_setup: &Self::ProverSetup, chunk: &[F]) -> Self::ChunkState {
        let sum = chunk
            .iter()
            .fold(0u128, |acc, x| acc.wrapping_add(x.to_canonical_u128()));
        HachiCommitment(sum)
    }

    fn process_chunk_onehot(
        _setup: &Self::ProverSetup,
        onehot_k: usize,
        chunk: &[Option<usize>],
    ) -> Self::ChunkState {
        let sum = chunk.iter().fold(0u128, |acc, x| {
            let v = x.unwrap_or(0) as u128;
            acc.wrapping_add(v)
        });
        HachiCommitment(sum.wrapping_add(onehot_k as u128))
    }

    fn aggregate_chunks(
        _setup: &Self::ProverSetup,
        _onehot_k: Option<usize>,
        chunks: &[Self::ChunkState],
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        let sum = chunks.iter().fold(0u128, |acc, c| acc.wrapping_add(c.0));
        let c = HachiCommitment(sum);
        (c, c)
    }
}

#[test]
fn commitment_scheme_round_trip() {
    let poly = SimplePoly {
        coeffs: vec![F::from_u64(3), F::from_u64(5), F::from_u64(7)],
    };
    let opening_point = [F::from_u64(11), F::from_u64(13)];
    let opening = poly.evaluate(&opening_point);

    let psetup = DummyScheme::setup_prover(poly.num_vars());
    let vsetup = DummyScheme::setup_verifier(&psetup);

    let (commitment, hint) = DummyScheme::commit(&poly, &psetup).unwrap();

    let mut prover_t = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    let proof =
        DummyScheme::prove(&psetup, &poly, &opening_point, Some(hint), &mut prover_t).unwrap();

    let mut verifier_t = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    DummyScheme::verify(
        &proof,
        &vsetup,
        &mut verifier_t,
        &opening_point,
        &opening,
        &commitment,
    )
    .unwrap();
}

#[test]
fn combine_commitments_and_hints_are_consistent() {
    let c1 = HachiCommitment(10);
    let c2 = HachiCommitment(20);
    let coeffs = [F::from_u64(3), F::from_u64(7)];

    let combined_c = DummyScheme::combine_commitments(&[c1, c2], &coeffs);
    let combined_h = DummyScheme::combine_hints(vec![c1, c2], &coeffs);

    let expected = 10u128
        .wrapping_mul(coeffs[0].to_canonical_u128())
        .wrapping_add(20u128.wrapping_mul(coeffs[1].to_canonical_u128()));
    assert_eq!(combined_c.0, expected);
    assert_eq!(combined_h.0, expected);
}

#[test]
fn streaming_chunk_path_aggregates() {
    let setup = DummyScheme::setup_prover(4);
    let c1 = DummyScheme::process_chunk(&setup, &[F::from_u64(1), F::from_u64(2)]);
    let c2 = DummyScheme::process_chunk_onehot(&setup, 8, &[Some(3), None, Some(5)]);

    let (commitment, hint) = DummyScheme::aggregate_chunks(&setup, Some(8), &[c1, c2]);
    assert_eq!(commitment, hint);
}
