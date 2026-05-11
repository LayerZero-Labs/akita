#![allow(missing_docs)]
#![cfg(feature = "zk")]

use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_field::CanonicalField;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{AkitaPolyOps, CommitmentProver, CommittedPolynomials, DensePoly, ProverClaims};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::Blake2bTranscript;
use akita_types::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, AkitaBatchedProof,
    AkitaVerifierSetup, BasisMode, BlockOrder, LevelParams, RingCommitment,
};
use akita_verifier::{CommitmentVerifier, CommittedOpenings, VerifierClaims};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Once;

type F = fp128::Field;
const STACK_SIZE: usize = 256 * 1024 * 1024;

static INIT_RAYON: Once = Once::new();

fn init_rayon_pool() {
    INIT_RAYON.call_once(|| {
        #[cfg(feature = "parallel")]
        rayon::ThreadPoolBuilder::new()
            .stack_size(STACK_SIZE)
            .build_global()
            .ok();
    });
}

fn run_on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(f)
        .expect("failed to spawn test thread")
        .join()
        .expect("test thread panicked");
}

fn random_point(nv: usize, seed: u64) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect()
}

fn opening_from_poly<const D: usize>(
    poly: &DensePoly<F, D>,
    point: &[F],
    layout: &LevelParams,
) -> F {
    let alpha_bits = D.trailing_zeros() as usize;
    assert_eq!(point.len(), alpha_bits + layout.m_vars + layout.r_vars);

    let inner_point = &point[..alpha_bits];
    let reduced_point = &point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field(
        reduced_point,
        layout.r_vars,
        layout.m_vars,
        BasisMode::Lagrange,
        BlockOrder::RowMajor,
    )
    .expect("opening point shape should match layout");

    let (y_ring, _) = poly.evaluate_and_fold(
        &ring_opening_point.b,
        &ring_opening_point.a,
        layout.block_len,
    );
    let v = reduce_inner_opening_to_ring_element::<F, D>(inner_point, BasisMode::Lagrange)
        .expect("inner opening point should match ring dimension");
    (y_ring * v.sigma_m1()).coefficients()[0]
}

fn prove_input<'a, P, C, H>(
    point: &'a [F],
    polynomials: &'a [P],
    commitment: &'a C,
    hint: H,
) -> ProverClaims<'a, F, P, C, H> {
    vec![(
        point,
        vec![CommittedPolynomials {
            polynomials,
            commitment,
            hint,
        }],
    )]
}

fn verify_input<'a, C>(
    point: &'a [F],
    openings: &'a [F],
    commitment: &'a C,
) -> VerifierClaims<'a, F, C> {
    vec![(
        point,
        vec![CommittedOpenings {
            openings,
            commitment,
        }],
    )]
}

fn run_zk_dense_e2e<const D: usize, Cfg>(nv: usize, label: &'static [u8])
where
    Cfg: CommitmentConfig<Field = F, ClaimField = F, ChallengeField = F>,
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
            F,
            D,
            ClaimField = F,
            VerifierSetup = AkitaVerifierSetup<F>,
            Commitment = RingCommitment<F, D>,
            BatchedProof = AkitaBatchedProof<F, F>,
        > + CommitmentVerifier<
            F,
            D,
            ClaimField = F,
            VerifierSetup = AkitaVerifierSetup<F>,
            Commitment = RingCommitment<F, D>,
            BatchedProof = AkitaBatchedProof<F, F>,
        >,
{
    assert_eq!(Cfg::D, D);
    init_rayon_pool();
    run_on_large_stack(move || {
        let layout = Cfg::commitment_layout(nv).expect("zk layout");
        let mut rng = StdRng::seed_from_u64(0x5eed_5eed_0000 + D as u64 + nv as u64);
        let evals: Vec<F> = (0..1usize << nv)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(nv, &evals).expect("dense poly");
        let point = random_point(nv, 0x0bad_f00d_0000 + D as u64 + nv as u64);
        let expected_opening = opening_from_poly(&poly, &point, &layout);

        type Scheme<const DD: usize, Config> = AkitaCommitmentScheme<DD, Config>;
        let setup = <Scheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, 1, 1);
        let verifier_setup = <Scheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);

        let commit_input = std::slice::from_ref(&poly);
        let (commitment, hint) =
            <Scheme<D, Cfg> as CommitmentProver<F, D>>::commit(commit_input, &setup)
                .expect("first zk commit");
        let (rerandomized_commitment, _) =
            <Scheme<D, Cfg> as CommitmentProver<F, D>>::commit(commit_input, &setup)
                .expect("second zk commit");
        assert_ne!(
            commitment, rerandomized_commitment,
            "ZK commitment should re-randomize for the same polynomial at D={D}, nv={nv}"
        );

        let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];
        let hints = vec![hint];

        let mut prover_transcript = Blake2bTranscript::<F>::new(label);
        let proof = <Scheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(
                &point,
                &poly_refs,
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("zk prove");

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize zk proof");
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(serialized),
            &proof_shape,
        )
        .expect("deserialize zk proof");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(label);
        <Scheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&point, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("zk verify");
    });
}

#[test]
fn zk_dense_d32_commitments_rerandomize_and_verify() {
    run_zk_dense_e2e::<32, fp128::D32Full>(14, b"zk_dense_d32");
}

#[test]
fn zk_dense_d64_commitments_rerandomize_and_verify() {
    run_zk_dense_e2e::<64, fp128::D64Full>(15, b"zk_dense_d64");
}

#[test]
fn zk_dense_d128_commitments_rerandomize_and_verify() {
    run_zk_dense_e2e::<128, fp128::D128Full>(16, b"zk_dense_d128");
}
