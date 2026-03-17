#![allow(missing_docs)]

use hachi_pcs::algebra::CyclotomicRing;
use hachi_pcs::algebra::Fp64;
use hachi_pcs::algebra::SparseChallenge;
use hachi_pcs::primitives::{Compress, SerializationError, Valid, Validate};
use hachi_pcs::protocol::commitment::utils::crt_ntt::NttSlotCache;
use hachi_pcs::protocol::commitment::utils::flat_matrix::FlatMatrix;
use hachi_pcs::protocol::hachi_poly_ops::{DecomposeFoldWitness, HachiPolyOps};
use hachi_pcs::protocol::transcript::labels;
use hachi_pcs::protocol::{
    BasisMode, Blake2bTranscript, CommitmentScheme, HachiCommitmentLayout, Transcript,
};
use hachi_pcs::{
    CanonicalField, FieldCore, FromSmallInt, HachiDeserialize, HachiError, HachiSerialize,
};
use std::io::{Read, Write};

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

    fn evaluate_ring(&self, scalars: &[F]) -> CyclotomicRing<F, 1> {
        let mut acc = F::zero();
        for (c, &s) in self.coeffs.iter().zip(scalars.iter()) {
            acc += *c * s;
        }
        CyclotomicRing::from_coefficients([acc])
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
        _block_len: usize,
        _num_digits_commit: usize,
        _num_digits_open: usize,
        _log_basis: u32,
    ) -> Result<Vec<Vec<[i8; 1]>>, HachiError> {
        Ok(vec![])
    }
}

#[derive(Clone)]
struct DummySetup {
    _max_num_vars: usize,
}

#[derive(Clone)]
struct DummyScheme;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct TestCommitment(u128);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct TestProof(u128);

impl Valid for TestCommitment {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl HachiSerialize for TestCommitment {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(&mut writer, Compress::No)
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        16
    }
}

impl HachiDeserialize for TestCommitment {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let value = u128::deserialize_with_mode(&mut reader, Compress::No, validate)?;
        Ok(Self(value))
    }
}

impl Valid for TestProof {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl HachiSerialize for TestProof {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(&mut writer, Compress::No)
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        16
    }
}

impl HachiDeserialize for TestProof {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let value = u128::deserialize_with_mode(&mut reader, Compress::No, validate)?;
        Ok(Self(value))
    }
}

impl CommitmentScheme<F, 1> for DummyScheme {
    type ProverSetup = DummySetup;
    type VerifierSetup = DummySetup;
    type Commitment = TestCommitment;
    type Proof = TestProof;
    type CommitHint = TestCommitment;

    fn setup_prover(max_num_vars: usize) -> Result<Self::ProverSetup, HachiError> {
        Ok(DummySetup {
            _max_num_vars: max_num_vars,
        })
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        setup.clone()
    }

    fn commit<P: HachiPolyOps<F, 1>>(
        _poly: &P,
        _setup: &Self::ProverSetup,
        _layout: &HachiCommitmentLayout,
    ) -> Result<(Self::Commitment, Self::CommitHint), HachiError> {
        let c = TestCommitment(0);
        Ok((c, c))
    }

    fn prove<T: Transcript<F>, P: HachiPolyOps<F, 1>>(
        _setup: &Self::ProverSetup,
        _poly: &P,
        _opening_point: &[F],
        _hint: Self::CommitHint,
        transcript: &mut T,
        commitment: &Self::Commitment,
        _basis: BasisMode,
        _layout: &HachiCommitmentLayout,
    ) -> Result<Self::Proof, HachiError> {
        transcript.append_serde(labels::ABSORB_COMMITMENT, commitment);
        let q = transcript.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
        Ok(TestProof(q.to_canonical_u128()))
    }

    fn verify<T: Transcript<F>>(
        proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        transcript: &mut T,
        _opening_point: &[F],
        _opening: &F,
        commitment: &Self::Commitment,
        _basis: BasisMode,
        _layout: &HachiCommitmentLayout,
    ) -> Result<(), HachiError> {
        transcript.append_serde(labels::ABSORB_COMMITMENT, commitment);
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

    let psetup = DummyScheme::setup_prover(poly.num_vars()).unwrap();
    let vsetup = DummyScheme::setup_verifier(&psetup);

    let layout = HachiCommitmentLayout {
        m_vars: 0,
        r_vars: 0,
        block_len: 1,
        num_blocks: 1,
        num_digits_commit: 1,
        num_digits_open: 1,
        num_digits_fold: 1,
        inner_width: 1,
        outer_width: 1,
        d_matrix_width: 1,
        log_basis: 1,
    };
    let (commitment, hint) = DummyScheme::commit(&poly, &psetup, &layout).unwrap();
    let opening = poly.evaluate(&opening_point);

    let mut prover_t = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    let proof = DummyScheme::prove(
        &psetup,
        &poly,
        &opening_point,
        hint,
        &mut prover_t,
        &commitment,
        BasisMode::Lagrange,
        &layout,
    )
    .unwrap();

    let mut verifier_t = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    DummyScheme::verify(
        &proof,
        &vsetup,
        &mut verifier_t,
        &opening_point,
        &opening,
        &commitment,
        BasisMode::Lagrange,
        &layout,
    )
    .unwrap();
}
