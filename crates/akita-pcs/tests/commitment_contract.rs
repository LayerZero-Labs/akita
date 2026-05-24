#![allow(missing_docs)]
#![cfg(not(feature = "zk"))]

use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_field::fields::wide::{HasWide, ReduceTo};
use akita_field::Fp64;
use akita_field::{AdditiveGroup, AkitaError, CanonicalField, HalvingField};
use akita_pcs::{
    AkitaPolyOps, CommitmentComputeBackend, CommitmentProver, CommittedPolynomials,
    ComputeBackendSetup, DecomposeFoldWitness, DenseCommitRowsPlan, LinearComputeBackend,
    OneHotCommitBlocks, OneHotCommitRowsPlan, ProverClaims, ProverComputeBackend,
    RecursiveWitnessCommitRowsPlan, RingSwitchComputeBackend, SparseRingCommitRowsPlan,
};
use akita_transcript::{labels, AkitaTranscript, Transcript};
use akita_types::AkitaExpandedSetup;
use akita_types::{AkitaCommitment, AppendToTranscript, BasisMode, DummyProof, FlatDigitBlocks};
use akita_verifier::{CommitmentVerifier, CommittedOpenings, VerifierClaims};
use std::sync::Arc;

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

    fn commit_inner<B>(
        &self,
        _backend: &B,
        _prepared: &B::PreparedSetup<1>,
        _n_a: usize,
        _block_len: usize,
        _num_digits_commit: usize,
        _num_digits_open: usize,
        _log_basis: u32,
    ) -> Result<FlatDigitBlocks<1>, AkitaError>
    where
        B: CommitmentComputeBackend<F>,
    {
        Ok(FlatDigitBlocks::from_blocks(vec![]))
    }
}

#[derive(Clone)]
struct DummySetup {
    _max_num_vars: usize,
}

#[derive(Clone, Copy)]
struct DummyBackend;

impl ComputeBackendSetup<F> for DummyBackend {
    type PreparedSetup<const D: usize> = DummySetup;

    fn prepare_expanded<const D: usize>(
        &self,
        _expanded: Arc<AkitaExpandedSetup<F>>,
    ) -> Result<Self::PreparedSetup<D>, AkitaError> {
        Ok(DummySetup { _max_num_vars: 0 })
    }

    fn expanded<'a, const D: usize>(
        &self,
        _prepared: &'a Self::PreparedSetup<D>,
    ) -> &'a Arc<AkitaExpandedSetup<F>> {
        unreachable!("dummy backend does not expose expanded setup")
    }
}

impl CommitmentComputeBackend<F> for DummyBackend {
    fn dense_commit_rows<const D: usize>(
        &self,
        _prepared: &Self::PreparedSetup<D>,
        _plan: DenseCommitRowsPlan<'_, F, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        unreachable!("dummy backend is not used for compute")
    }

    fn onehot_commit_rows<const D: usize>(
        &self,
        _prepared: &Self::PreparedSetup<D>,
        plan: OneHotCommitRowsPlan<'_>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        <F as HasWide>::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    {
        let _readable_by_external_backend: usize = match plan.blocks() {
            OneHotCommitBlocks::SingleChunk(blocks) => blocks
                .iter()
                .flat_map(|block| block.iter())
                .map(|entry| entry.pos_in_block() + entry.coeff_idx())
                .sum(),
            OneHotCommitBlocks::MultiChunk(blocks) => blocks
                .iter()
                .flat_map(|block| block.iter())
                .map(|entry| {
                    entry.pos_in_block()
                        + entry
                            .nonzero_coeffs()
                            .iter()
                            .map(|&coeff| usize::from(coeff))
                            .sum::<usize>()
                })
                .sum(),
        };
        unreachable!("dummy backend is not used for compute")
    }

    fn sparse_ring_commit_rows<const D: usize>(
        &self,
        _prepared: &Self::PreparedSetup<D>,
        plan: SparseRingCommitRowsPlan<'_>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        <F as HasWide>::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    {
        let _readable_by_external_backend: i64 = plan
            .blocks()
            .iter()
            .flat_map(|block| block.iter())
            .map(|entry| {
                entry.pos_in_block() as i64 + entry.coeff_idx() as i64 + i64::from(entry.value())
            })
            .sum();
        unreachable!("dummy backend is not used for compute")
    }

    fn recursive_witness_commit_rows<const D: usize>(
        &self,
        _prepared: &Self::PreparedSetup<D>,
        _plan: RecursiveWitnessCommitRowsPlan<'_, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        unreachable!("dummy backend is not used for compute")
    }
}

impl LinearComputeBackend<F> for DummyBackend {
    fn digit_rows<const D: usize>(
        &self,
        _prepared: &Self::PreparedSetup<D>,
        _row_len: usize,
        _digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        unreachable!("dummy backend is not used for compute")
    }

    fn cyclic_digit_rows<const D: usize>(
        &self,
        _prepared: &Self::PreparedSetup<D>,
        _row_len: usize,
        _digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        unreachable!("dummy backend is not used for compute")
    }
}

impl RingSwitchComputeBackend<F> for DummyBackend {
    fn ring_switch_relation_rows<const D: usize>(
        &self,
        _prepared: &Self::PreparedSetup<D>,
        _n_d: usize,
        _n_b: usize,
        _n_a: usize,
        _w_hat: &[[i8; D]],
        _t_hat: &[[i8; D]],
        _z_segment: &[[i32; D]],
        _z_pre_centered_inf_norm: u32,
    ) -> Result<
        (
            Vec<CyclotomicRing<F, D>>,
            Vec<CyclotomicRing<F, D>>,
            Vec<CyclotomicRing<F, D>>,
        ),
        AkitaError,
    >
    where
        F: HalvingField,
    {
        unreachable!("dummy backend is not used for compute")
    }
}

#[derive(Clone)]
struct DummyScheme;

impl CommitmentVerifier<F, 1> for DummyScheme {
    type ClaimField = F;
    type VerifierSetup = DummySetup;
    type Commitment = AkitaCommitment;
    type BatchedProof = DummyProof;

    fn batched_verify<'a, T: Transcript<F>>(
        proof: &Self::BatchedProof,
        _setup: &Self::VerifierSetup,
        transcript: &mut T,
        claims: VerifierClaims<'a, F, Self::Commitment>,
        _basis: BasisMode,
    ) -> Result<(), AkitaError> {
        for (_, payload) in claims {
            payload
                .commitment
                .append_to_transcript(labels::ABSORB_COMMITMENT, transcript);
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
    type ClaimField = F;
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

    fn commit<P, B>(
        _backend: &B,
        _prepared: &B::PreparedSetup<1>,
        _polys: &[P],
    ) -> Result<(Self::Commitment, Self::CommitHint), AkitaError>
    where
        P: AkitaPolyOps<F, 1>,
        B: CommitmentComputeBackend<F>,
    {
        let c = AkitaCommitment(0);
        Ok((c, c))
    }

    fn batched_prove<'a, T, P, B>(
        _backend: &B,
        _prepared: &B::PreparedSetup<1>,
        claims: ProverClaims<'a, F, P, Self::Commitment, Self::CommitHint>,
        transcript: &mut T,
        _basis: BasisMode,
    ) -> Result<Self::BatchedProof, AkitaError>
    where
        T: Transcript<F>,
        P: AkitaPolyOps<F, 1>,
        B: ProverComputeBackend<F>,
    {
        for (_, payload) in claims {
            payload
                .commitment
                .append_to_transcript(labels::ABSORB_COMMITMENT, transcript);
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
    let prepared = DummySetup {
        _max_num_vars: poly.num_vars(),
    };
    let vsetup = DummyScheme::setup_verifier(&psetup);

    let (commitment, hint) =
        DummyScheme::commit(&DummyBackend, &prepared, std::slice::from_ref(&poly)).unwrap();
    let opening = poly.evaluate(&opening_point);

    let poly_refs: [&DummyPoly; 1] = [&poly];
    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];

    let mut prover_t = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let prove_inputs = vec![(
        &opening_point[..],
        CommittedPolynomials {
            polynomials: &poly_refs[..],
            commitment: &commitments[0],
            hint,
        },
    )];
    let proof = DummyScheme::batched_prove(
        &DummyBackend,
        &prepared,
        prove_inputs,
        &mut prover_t,
        BasisMode::Lagrange,
    )
    .unwrap();

    let mut verifier_t = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let verify_inputs = vec![(
        &opening_point[..],
        CommittedOpenings {
            openings: opening_groups[0],
            commitment: &commitments[0],
        },
    )];
    DummyScheme::batched_verify(
        &proof,
        &vsetup,
        &mut verifier_t,
        verify_inputs,
        BasisMode::Lagrange,
    )
    .unwrap();
}
