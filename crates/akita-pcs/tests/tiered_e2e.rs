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
use akita_types::{
    AkitaBatchedProof, AkitaBatchedRootProof, AkitaLevelProof, SetupContributionMode,
};
use akita_verifier::CommitmentVerifier;
use common::*;

type TieredCfg = fp128::D64OneHotTiered;
const TIERED_D: usize = TieredCfg::D;

/// Count of **non-terminal** fold levels — the levels that carry the recursive
/// setup-product sumcheck under [`SetupContributionMode::Recursive`]. The root
/// fold level is tiered (`f_key` present), so a positive count means the tiered
/// prover-side setup-contribution path (`create_setup_contribution_inputs`) is
/// genuinely exercised, not just the Direct scan.
fn setup_sumcheck_levels(proof: &AkitaBatchedProof<F, F>) -> usize {
    let root_fold = match proof.root {
        AkitaBatchedRootProof::Fold(_) => 1,
        AkitaBatchedRootProof::Terminal(_) | AkitaBatchedRootProof::ZeroFold { .. } => 0,
    };
    let suffix_fold = proof
        .steps
        .iter()
        .filter(|step| matches!(step, AkitaLevelProof::Intermediate { .. }))
        .count();
    root_fold + suffix_fold
}

fn run_tiered_singleton(nv: usize, mode: SetupContributionMode) {
    init_rayon_pool();
    run_on_large_stack(move || {
        let opening_batch = akita_types::OpeningBatch::new(nv, 1).expect("singleton opening batch");
        let layout = TieredCfg::get_params_for_batched_commitment(&opening_batch).expect("layout");
        assert!(
            layout.f_key.is_some(),
            "expected a tiered root layout (f_key) for nv={nv} singleton"
        );

        let poly = make_onehot_poly(&layout, 0x7000_0000);
        let pt = random_point(nv, 0x7115_0000 + nv as u64);
        let opening = opening_from_poly(&poly, &pt, &layout);

        let setup = <AkitaCommitmentScheme<TIERED_D, TieredCfg> as CommitmentProver<F, TIERED_D>>::setup_prover(nv, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let verifier_setup = <AkitaCommitmentScheme<TIERED_D, TieredCfg> as CommitmentProver<
            F,
            TIERED_D,
        >>::setup_verifier(&setup);

        let (commitment, hint) = <AkitaCommitmentScheme<TIERED_D, TieredCfg> as CommitmentProver<
            F,
            TIERED_D,
        >>::commit(
            &setup, &CpuBackend, &prepared, std::slice::from_ref(&poly)
        )
        .expect("commit");
        assert_eq!(
            commitment.u.len(),
            layout.effective_commit_rows(),
            "sent commitment must match F row count when tiered"
        );

        let mut prover_transcript = AkitaTranscript::<F>::new(b"tiered_e2e");
        let proof = <AkitaCommitmentScheme<TIERED_D, TieredCfg> as CommitmentProver<F, TIERED_D>>::batched_prove(
            &setup,
            &CpuBackend,
            &prepared,
            prove_input(&pt[..], std::slice::from_ref(&poly), &commitment, hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
            mode,
        )
        .expect("prove");

        if mode == SetupContributionMode::Recursive {
            assert!(
                setup_sumcheck_levels(&proof) > 0,
                "recursive tiered nv={nv} singleton must fold so the (tiered) \
                 setup-product sumcheck runs on at least one level"
            );
        }

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
            verify_input(&pt[..], std::slice::from_ref(&opening), &commitment),
            BasisMode::Lagrange,
            mode,
        );
        assert!(
            result.is_ok(),
            "tiered nv={nv} singleton ({mode:?}) verification failed: {:?}",
            result.err()
        );
    });
}

fn run_tiered_batch(nv: usize, num_polys: usize, mode: SetupContributionMode) {
    init_rayon_pool();
    run_on_large_stack(move || {
        let opening_batch =
            akita_types::OpeningBatch::new(nv, num_polys).expect("same-point opening_batch");
        let layout = TieredCfg::get_params_for_batched_commitment(&opening_batch).expect("layout");
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

        let setup = <AkitaCommitmentScheme<TIERED_D, TieredCfg> as CommitmentProver<F, TIERED_D>>::setup_prover(nv, num_polys).unwrap();
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
            mode,
        )
        .expect("prove");

        if mode == SetupContributionMode::Recursive {
            assert!(
                setup_sumcheck_levels(&proof) > 0,
                "recursive tiered nv={nv} batch={num_polys} must fold so the \
                 (tiered) setup-product sumcheck runs on at least one level"
            );
        }

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
            mode,
        );
        assert!(
            result.is_ok(),
            "tiered nv={nv} batch={num_polys} ({mode:?}) verification failed: {:?}",
            result.err()
        );
    });
}

#[test]
fn tiered_onehot_batch_nv14() {
    // Smallest natural tiering instance for fp128::D64OneHotTiered.
    run_tiered_batch(14, 16, SetupContributionMode::Direct);
}

#[test]
fn tiered_onehot_singleton_nv27() {
    // Smallest singleton whose root fold both tiers (f_key present) and folds
    // (so the recursive variant exercises the tiered stage-3 setup sumcheck)
    // under the trace-internalized schedules. Tiering vs nv is non-monotonic
    // (it also holds at nv=12, but nv=12 is too small to fold); the next
    // tiering points are nv=31, 33, 35-38.
    run_tiered_singleton(27, SetupContributionMode::Direct);
}

/// Same tiered instances under [`SetupContributionMode::Recursive`]: the root
/// fold level is tiered (`f_key`), so the stage-3 setup-product sumcheck runs on
/// the tiered level and exercises the prover-side `create_setup_contribution_inputs`
/// tiered path (which must size the `B'` width by `tier_split`, not the full B
/// width). Guards against the recursive setup mode rejecting a valid tiered
/// layout.
#[test]
fn tiered_onehot_batch_nv14_recursive() {
    run_tiered_batch(14, 16, SetupContributionMode::Recursive);
}

#[test]
fn tiered_onehot_singleton_nv27_recursive() {
    run_tiered_singleton(27, SetupContributionMode::Recursive);
}
