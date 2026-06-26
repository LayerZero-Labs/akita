//! Slice-0 tripwire: transparent proof bytes for fixed folded instances are pinned
//! against accidental transparent-path regressions (`specs/akita-zk-strip-for-audit.md`, I1).
//! Re-pin digests when an intentional wire-format change (e.g. terminal Golomb encoding) lands.
//!
//! Cases exercise the main shipped presets on non-root-direct schedules:
//! - `fp128::D64Full` at nv = 15
//! - `fp128::D64OneHot` at nv = 20

#![allow(missing_docs)]
#![cfg(not(feature = "zk"))]

mod common;

use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_field::CanonicalField;
use akita_field::FieldCore;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{CommitmentProver, ComputeBackendSetup, CpuBackend, DensePoly, OneHotPoly};
use akita_serialization::AkitaSerialize;
use akita_transcript::AkitaTranscript;
use akita_types::{AkitaBatchedProof, AkitaScheduleLookupKey, BasisMode};
use akita_verifier::CommitmentVerifier;
use common::{dense_field_evals, opening_from_poly, OneHotCfg, F, ONEHOT_D, ONEHOT_K};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};

const GOLDEN_D64_FULL_NV15_SHA256: &str =
    "c99fcc1867742d10ac4b9c1bc0aa62081085ef5abd37a6f813ec59563b767072";
const GOLDEN_D64_ONEHOT_NV20_SHA256: &str =
    "4849bef9b51c9327e39044960abb6c477756ff5a2fa8f0e3d00a8db1cd0daf1b";

fn fixed_opening_point(nv: usize, seed: u64) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect()
}

fn batched_total_fold_levels<FF: CanonicalField, L: FieldCore>(
    proof: &AkitaBatchedProof<FF, L>,
) -> usize {
    use akita_types::{AkitaBatchedRootProof, AkitaLevelProof};
    let root_fold = match proof.root {
        AkitaBatchedRootProof::Fold(_) | AkitaBatchedRootProof::Terminal(_) => 1,
        AkitaBatchedRootProof::ZeroFold { .. } => 0,
    };
    let suffix_fold = proof
        .steps
        .iter()
        .filter(|step| {
            matches!(
                step,
                AkitaLevelProof::Intermediate { .. } | AkitaLevelProof::Terminal { .. }
            )
        })
        .count();
    root_fold + suffix_fold
}

fn assert_folded_not_root_direct<Cfg: CommitmentConfig>(
    nv: usize,
    proof: &AkitaBatchedProof<F, F>,
) {
    let plan = Cfg::runtime_schedule(AkitaScheduleLookupKey::singleton(nv)).expect("schedule plan");
    assert!(
        plan.num_fold_levels() > 0,
        "nv={nv}: schedule must use fold levels (not root-direct)"
    );
    assert!(
        !proof.is_root_direct(),
        "nv={nv}: proof must not be root-direct"
    );
    let folds = batched_total_fold_levels(proof);
    assert!(folds > 0, "nv={nv}: proof must have fold levels");
    assert_eq!(
        folds,
        plan.num_fold_levels(),
        "nv={nv}: proof fold count must match planner"
    );
}

fn prove_on_large_stack(build: impl FnOnce() -> Vec<u8> + Send + 'static) -> Vec<u8> {
    common::init_rayon_pool();
    let (tx, rx) = std::sync::mpsc::channel();
    common::run_on_large_stack(move || {
        tx.send(build()).expect("send golden bytes");
    });
    rx.recv().expect("receive golden bytes")
}

fn build_d64_full_nv15_proof_bytes() -> Vec<u8> {
    type Cfg = fp128::D64Full;
    const D: usize = Cfg::D;
    const NV: usize = 15;
    const POLY_SEED: u64 = 0xface_feed_000f;
    const POINT_SEED: u64 = 0xbabe_000f;
    const TRANSCRIPT_LABEL: &[u8] = b"transparent_proof_golden/d64-full-nv15";

    prove_on_large_stack(move || {
        let layout = Cfg::get_params_for_batched_commitment(
            &akita_types::OpeningBatchShape::new(NV, 1).expect("singleton opening batch"),
        )
        .expect("layout");

        let evals = dense_field_evals(NV, POLY_SEED);
        let poly = DensePoly::<F, D>::from_field_evals(NV, &evals).expect("dense poly");
        let opening_point = fixed_opening_point(NV, POINT_SEED);
        let opening = opening_from_poly(&poly, &opening_point, &layout);

        let setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(NV, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let (commitment, hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
            &setup,
            std::slice::from_ref(&poly),
            &stack,
        )
        .unwrap();
        let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
        let commitments = [commitment];

        let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
        let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
            &setup,
            common::prove_input(&opening_point[..], &poly_refs[..], &commitments[0], hint),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .unwrap();

        assert_folded_not_root_direct::<Cfg>(NV, &proof);

        // Verify-side golden (spec I1/4a): the transparent verifier must accept
        // the pinned proof. Read-only, so the serialized bytes are unaffected.
        let verifier_setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
        let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
        <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            common::verify_input(
                &opening_point[..],
                std::slice::from_ref(&opening),
                &commitments[0],
            ),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .expect("transparent verifier must accept the d64-full nv15 golden proof");

        let mut out = Vec::new();
        proof
            .serialize_compressed(&mut out)
            .expect("serialize golden proof");
        out
    })
}

fn build_d64_onehot_nv20_proof_bytes() -> Vec<u8> {
    const NV: usize = 20;
    const POLY_SEED: u64 = 0xdead_beef_0000 + NV as u64;
    const POINT_SEED: u64 = 0xcafe_0000 + NV as u64;
    const TRANSCRIPT_LABEL: &[u8] = b"transparent_proof_golden/d64-onehot-nv20";

    prove_on_large_stack(move || {
        let layout = OneHotCfg::get_params_for_batched_commitment(
            &akita_types::OpeningBatchShape::new(NV, 1).expect("singleton opening batch"),
        )
        .expect("layout");
        let total_field = layout.num_blocks * layout.block_len * ONEHOT_D;
        assert_eq!(total_field, 1usize << NV);
        let total_chunks = total_field / ONEHOT_K;

        let mut rng = StdRng::seed_from_u64(POLY_SEED);
        let indices: Vec<Option<u8>> = (0..total_chunks)
            .map(|_| Some(rng.gen_range(0..ONEHOT_K) as u8))
            .collect();
        let poly = OneHotPoly::<F, ONEHOT_D, u8>::new(ONEHOT_K, indices).expect("onehot poly");
        let opening_point = fixed_opening_point(NV, POINT_SEED);
        let opening = opening_from_poly(&poly, &opening_point, &layout);

        let setup = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<F, ONEHOT_D>>::setup_prover(NV, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let (commitment, hint) = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::commit(&setup, std::slice::from_ref(&poly), &stack)
        .unwrap();
        let poly_refs: [&OneHotPoly<F, ONEHOT_D, u8>; 1] = [&poly];
        let commitments = [commitment];

        let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
        let proof = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::batched_prove(
            &setup,
            common::prove_input(
                &opening_point[..],
                &poly_refs[..],
                &commitments[0],
                hint,
            ),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .unwrap();

        assert_folded_not_root_direct::<OneHotCfg>(NV, &proof);

        // Verify-side golden (spec I1/4a): read-only, bytes unaffected.
        let verifier_setup = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::setup_verifier(&setup);
        let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
        <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentVerifier<F, ONEHOT_D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            common::verify_input(&opening_point[..], std::slice::from_ref(&opening), &commitments[0]),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .expect("transparent verifier must accept the d64-onehot nv20 golden proof");

        let mut out = Vec::new();
        proof
            .serialize_compressed(&mut out)
            .expect("serialize golden proof");
        out
    })
}

#[test]
fn transparent_proof_golden_d64_full_nv15_digest() {
    let bytes = build_d64_full_nv15_proof_bytes();
    let digest = hex::encode(Sha256::digest(&bytes));
    if GOLDEN_D64_FULL_NV15_SHA256 == "PLACEHOLDER" {
        panic!(
            "pin GOLDEN_D64_FULL_NV15_SHA256: {digest} ({} bytes)",
            bytes.len()
        );
    }
    assert_eq!(
        digest, GOLDEN_D64_FULL_NV15_SHA256,
        "fp128 D64Full nv15 proof bytes changed — re-pin after intentional wire-format updates"
    );
}

#[test]
fn transparent_proof_golden_d64_onehot_nv20_digest() {
    let bytes = build_d64_onehot_nv20_proof_bytes();
    let digest = hex::encode(Sha256::digest(&bytes));
    if GOLDEN_D64_ONEHOT_NV20_SHA256 == "PLACEHOLDER" {
        panic!(
            "pin GOLDEN_D64_ONEHOT_NV20_SHA256: {digest} ({} bytes)",
            bytes.len()
        );
    }
    assert_eq!(
        digest, GOLDEN_D64_ONEHOT_NV20_SHA256,
        "fp128 D64OneHot nv20 proof bytes changed — re-pin after intentional wire-format updates"
    );
}

#[test]
fn transparent_proof_bytes_are_deterministic() {
    let full_a = build_d64_full_nv15_proof_bytes();
    let full_b = build_d64_full_nv15_proof_bytes();
    assert_eq!(
        full_a, full_b,
        "D64Full nv15 serialization must be deterministic"
    );

    let onehot_a = build_d64_onehot_nv20_proof_bytes();
    let onehot_b = build_d64_onehot_nv20_proof_bytes();
    assert_eq!(
        onehot_a, onehot_b,
        "D64OneHot nv20 serialization must be deterministic"
    );
}
