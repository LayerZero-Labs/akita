//! Slice-0 tripwire: transparent proof bytes for a fixed instance must not change
//! across the ZK strip cutover (`specs/akita-zk-strip-for-audit.md`, invariant I1).
//!
//! Instance copied from `full_d32_tiny_root_direct_roundtrip_and_serialization` in
//! `akita_e2e.rs` with all RNG inputs pinned.

#![allow(missing_docs)]
#![cfg(not(feature = "zk"))]

mod common;

use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_field::CanonicalField;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{CommitmentProver, ComputeBackendSetup, CpuBackend, DensePoly};
use akita_serialization::AkitaSerialize;
use akita_transcript::AkitaTranscript;
use akita_types::{AkitaScheduleLookupKey, BasisMode};
use common::{opening_from_poly, F};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};

type Cfg = fp128::D32Full;
const D: usize = Cfg::D;
const NV: usize = 4;
const POLY_SEED: u64 = 0x0ddc_0ffe_e123_4567;
const OPENING_POINT_SEED: u64 = 0x600d_f00d_0000_0004;
const TRANSCRIPT_LABEL: &[u8] = b"akita_e2e/full-d32-direct-root";

const GOLDEN_PROOF_SHA256: &str =
    "b903f409d3772b08df53268848ebfac5eec7262db3dacb98bf5cc2538f0c393e";

fn singleton_layout(num_vars: usize) -> akita_types::LevelParams {
    let opening_batch =
        akita_types::OpeningBatchShape::new(num_vars, 1).expect("singleton opening batch");
    Cfg::get_params_for_batched_commitment(&opening_batch).expect("singleton commitment layout")
}

fn fixed_opening_point(nv: usize) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(OPENING_POINT_SEED);
    (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect()
}

fn build_golden_proof_bytes() -> Vec<u8> {
    common::init_rayon_pool();
    let (tx, rx) = std::sync::mpsc::channel();
    common::run_on_large_stack(move || {
        let plan =
            Cfg::runtime_schedule(AkitaScheduleLookupKey::singleton(NV)).expect("schedule plan");
        assert_eq!(
            plan.num_fold_levels(),
            0,
            "tiny roots should use direct mode"
        );

        let layout = singleton_layout(NV);

        let mut rng = StdRng::seed_from_u64(POLY_SEED);
        let evals: Vec<F> = (0..1usize << NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(NV, &evals).unwrap();
        let opening_point = fixed_opening_point(NV);
        let _opening = opening_from_poly(&poly, &opening_point, &layout);

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

        assert!(proof.is_root_direct());
        assert_eq!(proof.size(), plan.total_bytes);

        let mut out = Vec::new();
        proof
            .serialize_compressed(&mut out)
            .expect("serialize golden proof");
        tx.send(out).expect("send golden bytes");
    });
    rx.recv().expect("receive golden bytes")
}

#[test]
fn transparent_proof_golden_digest() {
    let bytes = build_golden_proof_bytes();
    let digest = hex::encode(Sha256::digest(&bytes));
    if GOLDEN_PROOF_SHA256 == "PLACEHOLDER" {
        panic!(
            "pin GOLDEN_PROOF_SHA256 in transparent_proof_golden.rs: {digest} ({} bytes)",
            bytes.len()
        );
    }
    assert_eq!(
        digest, GOLDEN_PROOF_SHA256,
        "transparent proof bytes changed — ZK strip violated invariant I1"
    );
}

#[test]
fn transparent_proof_bytes_are_deterministic() {
    let a = build_golden_proof_bytes();
    let b = build_golden_proof_bytes();
    assert_eq!(
        a, b,
        "fixed-instance proof serialization must be deterministic"
    );
}
