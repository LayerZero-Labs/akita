#![allow(missing_docs)]

use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_field::Fp64;
use akita_field::{AkitaError, CanonicalField};
use akita_prover::kernels::crt_ntt::NttSlotCache;
use akita_prover::{AkitaPolyOps, CommitmentProver, DecomposeFoldWitness};
use akita_transcript::{labels, Blake2bTranscript, Transcript};
use akita_types::FlatMatrix;
use akita_types::{
    AkitaCommitment, AppendToTranscript, BasisMode, DummyProof, FlatDigitBlocks, OpeningStatement,
    PointToPolynomialMap,
};
use akita_verifier::CommitmentVerifier;

type F = Fp64<4294967197>;

/// Trivial polynomial wrapper that implements `AkitaPolyOps<F, 1>`.
#[derive(Debug, Clone)]
struct DummyPoly {
    coeffs: Vec<F>,
}

impl DummyPoly {
    fn evaluate(&self, point: &[F]) -> F {
        assert_eq!(point.len(), self.num_vars());
        let mut acc = self.coeffs[0];
        for (i, r_i) in point.iter().enumerate() {
            acc += self.coeffs[i + 1] * *r_i;
        }
        acc
    }

    fn num_vars(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }
}

impl AkitaPolyOps<F, 1> for DummyPoly {
    type CommitCache = NttSlotCache<1>;

    fn num_ring_elems(&self) -> usize {
        self.coeffs.len()
    }

    fn fold_blocks(&self, _scalars: &[F], _block_len: usize) -> Vec<CyclotomicRing<F, 1>> {
        vec![]
    }

    fn decompose_fold(
        &self,
        _challenges: &[SparseChallenge],
        _block_len: usize,
        _num_digits: usize,
        _log_basis: u32,
    ) -> DecomposeFoldWitness<F, 1> {
        DecomposeFoldWitness {
            z_pre: vec![],
            centered_coeffs: vec![],
            centered_inf_norm: 0,
        }
    }

    fn commit_inner(
        &self,
        _a_matrix: &FlatMatrix<F>,
        _ntt_a: &NttSlotCache<1>,
        _n_a: usize,
        _block_len: usize,
        _num_digits_commit: usize,
        _num_digits_open: usize,
        _log_basis: u32,
        _matrix_stride: usize,
    ) -> Result<FlatDigitBlocks<1>, AkitaError> {
        Ok(FlatDigitBlocks::from_blocks(vec![]))
    }
}

#[derive(Clone)]
struct DummySetup {
    _max_num_vars: usize,
}

#[derive(Clone)]
struct DummyScheme;

impl CommitmentVerifier<F, 1> for DummyScheme {
    type VerifierSetup = DummySetup;
    type Commitment = AkitaCommitment;
    type BatchedProof = DummyProof;
    type Claims<'a>
        = OpeningStatement<'a, F, Self::Commitment>
    where
        F: 'a,
        Self: 'a;

    fn batched_verify<'a, T: Transcript<F>>(
        proof: &Self::BatchedProof,
        _setup: &Self::VerifierSetup,
        transcript: &mut T,
        claims: Self::Claims<'a>,
        _basis: BasisMode,
    ) -> Result<(), AkitaError> {
        for commitment in claims.commitments() {
            commitment.append_to_transcript(labels::ABSORB_COMMITMENT, transcript);
        }
        let q = transcript.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
        if proof.0 == q.to_canonical_u128() {
            Ok(())
        } else {
            Err(AkitaError::InvalidProof)
        }
    }

    fn protocol_name() -> &'static [u8] {
        b"AkitaDummy"
    }
}

impl CommitmentProver<F, 1> for DummyScheme {
    type ProverSetup = DummySetup;
    type VerifierSetup = DummySetup;
    type Commitment = AkitaCommitment;
    type CommitHint = AkitaCommitment;
    type BatchedProof = DummyProof;

    fn setup_prover(
        max_num_vars: usize,
        _max_num_batched_polys: usize,
        _max_num_points: usize,
    ) -> Self::ProverSetup {
        DummySetup {
            _max_num_vars: max_num_vars,
        }
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        setup.clone()
    }

    fn commit<P: AkitaPolyOps<F, 1, CommitCache = NttSlotCache<1>>>(
        _polys: &[P],
        _setup: &Self::ProverSetup,
    ) -> Result<(Self::Commitment, Self::CommitHint), AkitaError> {
        let c = AkitaCommitment(0);
        Ok((c, c))
    }

    fn batched_prove<'a, T: Transcript<F>, P: AkitaPolyOps<F, 1, CommitCache = NttSlotCache<1>>>(
        _setup: &Self::ProverSetup,
        statement: OpeningStatement<'a, F, Self::Commitment>,
        _polynomials: Vec<&'a P>,
        _hints: Vec<Self::CommitHint>,
        transcript: &mut T,
        _basis: BasisMode,
    ) -> Result<Self::BatchedProof, AkitaError> {
        for commitment in statement.commitments() {
            commitment.append_to_transcript(labels::ABSORB_COMMITMENT, transcript);
        }
        let q = transcript.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
        Ok(DummyProof(q.to_canonical_u128()))
    }
}

#[test]
fn commitment_scheme_round_trip() {
    let poly = DummyPoly {
        coeffs: vec![F::from_u64(3), F::from_u64(5), F::from_u64(7)],
    };
    let opening_point = [F::from_u64(11), F::from_u64(13)];

    let psetup = DummyScheme::setup_prover(poly.num_vars(), 1, 1);
    let vsetup = DummyScheme::setup_verifier(&psetup);

    let (commitment, hint) = DummyScheme::commit(std::slice::from_ref(&poly), &psetup).unwrap();
    let opening = poly.evaluate(&opening_point);

    let poly_refs: [&DummyPoly; 1] = [&poly];
    let commitments = [commitment];
    let openings = [opening];
    let statement = OpeningStatement::new(
        vec![&opening_point[..]],
        commitments.to_vec(),
        openings.to_vec(),
        vec![vec![PointToPolynomialMap {
            point_idx: 0,
            polynomial_idx: 0,
        }]],
    )
    .unwrap();

    let mut prover_t = Blake2bTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let proof = DummyScheme::batched_prove(
        &psetup,
        statement.clone(),
        poly_refs.to_vec(),
        vec![hint],
        &mut prover_t,
        BasisMode::Lagrange,
    )
    .unwrap();

    let mut verifier_t = Blake2bTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    DummyScheme::batched_verify(
        &proof,
        &vsetup,
        &mut verifier_t,
        statement,
        BasisMode::Lagrange,
    )
    .unwrap();
}
