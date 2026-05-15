//! End-to-end tests for the book §5.4 tiered setup commitment path.
//!
//! These tests exercise the routed tiered fourth-root verifier of book
//! Figure 12 and §5.4: the prover splits the shared setup polynomial
//! `S` into `k = f²` chunks under shared `D_chunk/B_chunk` matrices,
//! binds the per-chunk B-side commitments via a tier-3 meta commit,
//! and routes both into the next fold level's joint multi-claim
//! recursive open. The verifier mirrors by deriving the per-chunk and
//! meta commitments from the public setup matrix, populating
//! [`AkitaVerifierSetup::tiered_s_cache`] on first use.
//!
//! The wrapper config is [`akita_config::ClaimReductionCfg<Base, F>`]
//! at the chosen tiered shrink factor. The small tests at `f = 2,
//! k = 4` exercise the 10 stage-2 check groups end-to-end at the
//! tightest schedule that still runs under typical CI memory; the
//! `tiered_production_prove_verify` test exercises the book sweet spot
//! `f = 8, k = 64`.

#![allow(missing_docs)]

mod common;

use akita_algebra::CyclotomicRing;
use akita_config::{ClaimReductionCfg, CommitmentConfig, TieredClaimReductionCfg};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::CommitmentProver;
use akita_transcript::Blake2bTranscript;
use akita_types::{TieredSetupCommitments, TieredSetupParams};
use akita_verifier::CommitmentVerifier;
use common::*;
use std::sync::Mutex;

static E2E_TEST_LOCK: Mutex<()> = Mutex::new(());

type TieredDenseSmallCfg = ClaimReductionCfg<DenseCfg, 2>;
type TieredOneHotSmallCfg = ClaimReductionCfg<OneHotCfg, 2>;
type TieredDenseProductionCfg = TieredClaimReductionCfg<DenseCfg>;

#[test]
fn tiered_dense_prove_verify_small() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 20;
        const D: usize = DENSE_D;
        type Scheme = AkitaCommitmentScheme<D, TieredDenseSmallCfg>;

        let layout = TieredDenseSmallCfg::commitment_layout(NV).expect("layout");
        assert!(layout.use_setup_claim_reduction);
        let poly = make_dense_poly(NV, 0x715e_d001);
        let pt = random_point(NV, 0x715e_d002);
        let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"tiered_setup_e2e/dense_small");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("tiered dense prove");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"tiered_setup_e2e/dense_small");
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("tiered dense verify");
    });
}

#[test]
fn tiered_onehot_prove_verify_small() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 20;
        const D: usize = ONEHOT_D;
        type Scheme = AkitaCommitmentScheme<D, TieredOneHotSmallCfg>;

        let layout = TieredOneHotSmallCfg::commitment_layout(NV).expect("layout");
        assert!(layout.use_setup_claim_reduction);
        let poly = make_onehot_poly(&layout, 0x715e_0001);
        let pt = random_point(NV, 0x715e_0002);
        let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"tiered_setup_e2e/onehot_small");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("tiered onehot prove");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"tiered_setup_e2e/onehot_small");
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("tiered onehot verify");
    });
}

#[test]
fn tiered_production_prove_verify() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 32;
        const D: usize = DENSE_D;
        type Scheme = AkitaCommitmentScheme<D, TieredDenseProductionCfg>;

        let layout = TieredDenseProductionCfg::commitment_layout(NV).expect("layout");
        assert!(layout.use_setup_claim_reduction);
        let poly = make_dense_poly(NV, 0x715e_f801);
        let pt = random_point(NV, 0x715e_f802);
        let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"tiered_setup_e2e/production");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("tiered production prove");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"tiered_setup_e2e/production");
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("tiered production verify");
    });
}

#[test]
fn tiered_rejects_tampered_s_opening_value() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 20;
        const D: usize = DENSE_D;
        type Scheme = AkitaCommitmentScheme<D, TieredDenseSmallCfg>;

        let layout = TieredDenseSmallCfg::commitment_layout(NV).expect("layout");
        let poly = make_dense_poly(NV, 0x715e_5001);
        let pt = random_point(NV, 0x715e_5002);
        let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"tiered_setup_e2e/tamper_s_opening");
        let mut proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("tiered tamper prove");

        let fold_root = proof
            .root
            .as_fold_mut()
            .expect("tiered test must exercise root fold");
        let payload = fold_root
            .stage2
            .setup_claim_reduction
            .as_mut()
            .expect("tiered proof should carry setup claim reduction");
        payload.s_opening_value += F::from_u64(1);

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"tiered_setup_e2e/tamper_s_opening");
        let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(result.is_err(), "tampered s_opening_value must reject");
    });
}

#[test]
fn tiered_rejects_tampered_meta_material() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 20;
        const D: usize = DENSE_D;
        type Scheme = AkitaCommitmentScheme<D, TieredDenseSmallCfg>;

        let layout = TieredDenseSmallCfg::commitment_layout(NV).expect("layout");
        let poly = make_dense_poly(NV, 0x715e_6001);
        let pt = random_point(NV, 0x715e_6002);
        let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"tiered_setup_e2e/tamper_meta");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("tiered meta tamper prove");
        let fold_root = proof
            .root
            .as_fold()
            .expect("meta tamper test must exercise root fold");
        assert!(
            fold_root.stage2.setup_claim_reduction.is_some(),
            "meta tamper test must include root setup claim reduction"
        );

        let tier = TieredSetupParams::new(2).expect("f=2 tier");
        let wrong = TieredSetupCommitments::<F, D> {
            chunk_b_commitments: vec![vec![CyclotomicRing::<F, D>::zero()]; tier.num_chunks],
            meta_b_commitment: vec![CyclotomicRing::<F, D>::zero()],
            params: tier,
        };
        verifier_setup
            .tiered_s_cache
            .set(Box::new(wrong) as Box<dyn std::any::Any + Send + Sync>)
            .expect("test cache should be empty before verify");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"tiered_setup_e2e/tamper_meta");
        let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(result.is_err(), "tampered meta-tier cache must reject");
    });
}
