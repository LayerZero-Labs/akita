#![allow(missing_docs)]

use hachi_pcs::algebra::CyclotomicRing;
use hachi_pcs::algebra::Fp64;
use hachi_pcs::algebra::SparseChallenge;
use hachi_pcs::protocol::commitment::utils::crt_ntt::NttSlotCache;
use hachi_pcs::protocol::commitment::utils::flat_matrix::FlatMatrix;
use hachi_pcs::protocol::commitment::{DummyProof, HachiCommitment};
use hachi_pcs::protocol::hachi_poly_ops::{DecomposeFoldWitness, HachiPolyOps};
use hachi_pcs::protocol::proof::FlatDigitBlocks;
use hachi_pcs::protocol::transcript::labels;
use hachi_pcs::protocol::{
    AppendToTranscript, BasisMode, Blake2bTranscript, CommitmentScheme, Transcript,
};
use hachi_pcs::{CanonicalField, FromSmallInt, HachiError};

type F = Fp64<4294967197>;

/// Trivial polynomial wrapper that implements `HachiPolyOps<F, 1>`.
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

impl HachiPolyOps<F, 1> for DummyPoly {
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
    ) -> Result<FlatDigitBlocks<1>, HachiError> {
        Ok(FlatDigitBlocks::from_blocks(vec![]))
    }
}

#[derive(Clone)]
struct DummySetup {
    _max_num_vars: usize,
}

#[derive(Clone)]
struct DummyScheme;

impl CommitmentScheme<F, 1> for DummyScheme {
    type ProverSetup = DummySetup;
    type VerifierSetup = DummySetup;
    type Commitment = HachiCommitment;
    type BatchedProof = DummyProof;
    type CommitHint = HachiCommitment;
    type BatchedCommitHint = Vec<HachiCommitment>;

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

    fn commit<P: HachiPolyOps<F, 1>>(
        _polys: &[P],
        _setup: &Self::ProverSetup,
    ) -> Result<(Self::Commitment, Self::CommitHint), HachiError> {
        let c = HachiCommitment(0);
        Ok((c, c))
    }

    fn batched_prove<T: Transcript<F>, P: HachiPolyOps<F, 1>>(
        _setup: &Self::ProverSetup,
        _poly_groups_by_point: &[&[&[P]]],
        _opening_points: &[&[F]],
        _hints_by_point: Vec<Self::BatchedCommitHint>,
        transcript: &mut T,
        commitments_by_point: &[&[Self::Commitment]],
        _basis: BasisMode,
    ) -> Result<Self::BatchedProof, HachiError> {
        for commitments in commitments_by_point {
            for commitment in *commitments {
                commitment.append_to_transcript(labels::ABSORB_COMMITMENT, transcript);
            }
        }
        let q = transcript.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
        Ok(DummyProof(q.to_canonical_u128()))
    }

    fn batched_verify<T: Transcript<F>>(
        proof: &Self::BatchedProof,
        _setup: &Self::VerifierSetup,
        transcript: &mut T,
        _opening_points: &[&[F]],
        _opening_groups_by_point: &[&[&[F]]],
        commitments_by_point: &[&[Self::Commitment]],
        _basis: BasisMode,
    ) -> Result<(), HachiError> {
        for commitments in commitments_by_point {
            for commitment in *commitments {
                commitment.append_to_transcript(labels::ABSORB_COMMITMENT, transcript);
            }
        }
        let q = transcript.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
        if proof.0 == q.to_canonical_u128() {
            Ok(())
        } else {
            Err(HachiError::InvalidProof)
        }
    }

    fn protocol_name() -> &'static [u8] {
        b"HachiDummy"
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
    let poly_groups = [&poly_refs[..]];
    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];

    let mut prover_t = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    let proof = DummyScheme::batched_prove(
        &psetup,
        &[&poly_groups[..]],
        &[&opening_point[..]],
        vec![vec![hint]],
        &mut prover_t,
        &[&commitments[..]],
        BasisMode::Lagrange,
    )
    .unwrap();

    let mut verifier_t = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    DummyScheme::batched_verify(
        &proof,
        &vsetup,
        &mut verifier_t,
        &[&opening_point[..]],
        &[&opening_groups[..]],
        &[&commitments[..]],
        BasisMode::Lagrange,
    )
    .unwrap();
}
