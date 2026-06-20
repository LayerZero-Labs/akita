#![allow(missing_docs)]
#![cfg(feature = "zk")]

mod common;

use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{CommitmentProver, ComputeBackendSetup, CpuBackend};
use akita_transcript::AkitaTranscript;
use akita_types::{
    sis::{FoldWitnessLinfCapPolicy, MAX_FOLD_GRIND_ATTEMPTS},
    AkitaBatchedRootProof, FoldLinfProtocolBinding, FOLD_GRIND_PROBE_ORDER_TRANSCRIPT_SHUFFLE,
};
use akita_verifier::CommitmentVerifier;
use common::*;

type Scheme = AkitaCommitmentScheme<ONEHOT_D, OneHotCfg>;

#[test]
fn zk_tail_bound_with_grind_onehot_roundtrip() {
    init_rayon_pool();
    run_on_large_stack(|| {
        assert_eq!(
            FoldLinfProtocolBinding::CURRENT.grind_probe_order,
            FOLD_GRIND_PROBE_ORDER_TRANSCRIPT_SHUFFLE
        );

        let num_vars = 28usize;
        let layout = OneHotCfg::get_params_for_batched_commitment(
            &akita_types::OpeningBatch::same_point(num_vars, 1).expect("singleton opening batch"),
        )
        .expect("layout");
        assert_eq!(
            layout.fold_witness_linf_cap_policy(),
            FoldWitnessLinfCapPolicy::TailBoundWithGrind
        );

        let poly = make_onehot_poly(&layout, 0x5151_2001);
        let point = random_point(num_vars, 0x5151_2002);
        let opening = opening_from_poly(&poly, &point, &layout);

        let setup =
            <Scheme as CommitmentProver<F, ONEHOT_D>>::setup_prover(num_vars, 1).expect("setup");
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepare setup");
        let verifier_setup = <Scheme as CommitmentProver<F, ONEHOT_D>>::setup_verifier(&setup);
        let (commitment, hint) = <Scheme as CommitmentProver<F, ONEHOT_D>>::commit(
            &setup,
            std::slice::from_ref(&poly),
            &CpuBackend,
            &prepared,
        )
        .expect("commit");

        let mut prover_transcript = AkitaTranscript::<F>::new(b"fold-linf/zk-onehot");
        let proof = <Scheme as CommitmentProver<F, ONEHOT_D>>::batched_prove(
            &setup,
            prove_input(&point, &[&poly], &commitment, hint),
            &CpuBackend,
            &prepared,
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .expect("prove");

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"fold-linf/zk-onehot");
        <Scheme as CommitmentVerifier<F, ONEHOT_D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&point, &[opening], &commitment),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .expect("verify");

        assert!(matches!(proof.root, AkitaBatchedRootProof::Fold(_)));
        for step in proof.fold_levels() {
            assert!(
                step.fold_grind_nonce() < MAX_FOLD_GRIND_ATTEMPTS,
                "grind nonce must stay within cap"
            );
        }
    });
}
