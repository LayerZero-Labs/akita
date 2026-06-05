//! End-to-end test for the **tiered commitment** path (`fp128::D64OneHotTiered`).
//!
//! Commits a same-point batch of one-hot polynomials large enough that the
//! planner tiers the root (the first-tier `B` would exceed `A`, so it is reused
//! across `f` slices and the partial images are committed with the second-tier
//! `F`). Produces an opening proof, round-trips it through serialization, and
//! verifies it. The batch size is chosen so the root layout actually carries an
//! `f_key` (asserted below).

#![allow(missing_docs)]
#![cfg(not(feature = "zk"))]

use akita_prover::{ComputeBackendSetup, CpuBackend};

mod common;

use akita_pcs::AkitaCommitmentScheme;
use akita_prover::CommitmentProver;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::AkitaBatchedProof;
use akita_verifier::CommitmentVerifier;
use common::*;

type TieredCfg = fp128::D64OneHotTiered;
const TIERED_D: usize = TieredCfg::D;

fn run_tiered_batch(nv: usize, num_polys: usize) {
    init_rayon_pool();
    run_on_large_stack(move || {
        let incidence = akita_types::ClaimIncidenceSummary::same_point(nv, num_polys)
            .expect("same-point incidence");
        let layout = TieredCfg::get_params_for_batched_commitment(&incidence).expect("layout");
        assert!(
            layout.f_key.is_some(),
            "expected a tiered root layout (f_key) for nv={nv} batch={num_polys}"
        );

        let polys: Vec<OneHotPoly<F, TIERED_D, u8>> = (0..num_polys)
            .map(|i| make_onehot_poly(&layout, 0x7000_0000 + i as u64))
            .collect();

        let pt = random_point(nv, 0x7115_0000 + nv as u64);
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| opening_from_poly(poly, &pt, &layout))
            .collect();

        let setup = <AkitaCommitmentScheme<TIERED_D, TieredCfg> as CommitmentProver<F, TIERED_D>>::setup_prover(nv, num_polys, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let verifier_setup = <AkitaCommitmentScheme<TIERED_D, TieredCfg> as CommitmentProver<
            F,
            TIERED_D,
        >>::setup_verifier(&setup);

        let (commitment, hint) = <AkitaCommitmentScheme<TIERED_D, TieredCfg> as CommitmentProver<
            F,
            TIERED_D,
        >>::commit(&setup, &CpuBackend, &prepared, &polys)
        .expect("commit");

        let poly_refs: Vec<&OneHotPoly<F, TIERED_D, u8>> = polys.iter().collect();

        let mut prover_transcript = AkitaTranscript::<F>::new(b"tiered_e2e");
        let proof = <AkitaCommitmentScheme<TIERED_D, TieredCfg> as CommitmentProver<F, TIERED_D>>::batched_prove(
            &setup,
            &CpuBackend,
            &prepared,
            prove_input(&pt[..], &poly_refs[..], &commitment, hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .expect("prove");

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize");
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(serialized),
            &proof_shape,
        )
        .expect("deserialize");

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"tiered_e2e");
        let result = <AkitaCommitmentScheme<TIERED_D, TieredCfg> as CommitmentVerifier<
            F,
            TIERED_D,
        >>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], &openings[..], &commitment),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        );
        assert!(
            result.is_ok(),
            "tiered nv={nv} batch={num_polys} verification failed: {:?}",
            result.err()
        );
    });
}

#[test]
fn tiered_onehot_batch_nv14() {
    // Smallest natural tiering instance for fp128::D64OneHotTiered.
    run_tiered_batch(14, 16);
}
