//! Wave 0 mixed-D per-level fixture and cutover regression oracle.
//!
//! Uses `fp128::D128Full` setup (`gen_ring_dim = 128`) with a hand-built schedule:
//! fold levels 0–1 at `D = 128`, level 2+ at `D = 64`. Pins proof wire bytes and
//! the effective schedule digest. Regenerate the oracle fixture with
//! `AKITA_CAPTURE_MIXED_ORACLE=1 cargo test -p akita-pcs --test mixed_d_per_level_e2e
//! mixed_d_per_level_prove -- --nocapture`.

mod common;

use akita_config::proof_optimized::fp128;
use akita_config::test_support::mixed_d_per_level_schedule;
use akita_config::{bind_transcript_instance_descriptor, CommitmentConfig};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{prove, ComputeBackendSetup, CpuBackend, DensePoly, UniformProverStack};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::{
    digest_effective_schedule, validate_ring_dim_plan_at_entry, validate_schedule_context_at_entry,
    AkitaBatchedProof, BasisMode, SetupContributionMode,
};
use akita_verifier::test_support::batched_verify_with_schedule;
use common::*;

type Cfg = fp128::D128Full;
type SuffixCfg = fp128::D64Full;
type F = fp128::Field;
type Scheme = AkitaCommitmentScheme<Cfg>;
const D: usize = Cfg::D;

/// Fold levels `[0, MIXED_D_SWITCH_FOLD)` use D128; `[MIXED_D_SWITCH_FOLD, …)` use D64.
const MIXED_D_SWITCH_FOLD: usize = 2;
const NUM_VARS: usize = 16;

const ORACLE_EFFECTIVE_SCHEDULE_DIGEST: [u8; 32] = [
    0x71, 0x98, 0xb5, 0x11, 0x4a, 0x6c, 0x21, 0x18, 0x94, 0x6c, 0x69, 0x15, 0x61, 0xe9, 0xdc, 0xb1,
    0x56, 0xe1, 0xe1, 0xef, 0xcb, 0xd2, 0x36, 0xa5, 0x81, 0x4b, 0xba, 0x9f, 0xbd, 0x61, 0xea, 0x4b,
];
const ORACLE_PROOF_BYTES: &[u8] =
    include_bytes!("fixtures/wave0_uniform_handbuilt_d128_nv16.proof.bin");

// Populated once the mixed-D fixture proves on the legacy suffix path.
const MIXED_ORACLE_EFFECTIVE_SCHEDULE_DIGEST: [u8; 32] = [
    0x52, 0xfd, 0x22, 0xb4, 0x4e, 0xa0, 0x04, 0xdc, 0xf3, 0x86, 0x92, 0xed, 0x14, 0x08, 0xb0, 0xab,
    0x2b, 0xbd, 0xe4, 0x27, 0xbb, 0xb9, 0xe1, 0x44, 0x75, 0x1e, 0x38, 0x09, 0xd6, 0x8c, 0xa7, 0x22,
];
const MIXED_ORACLE_PROOF_BYTES: &[u8] = include_bytes!("fixtures/wave0_mixed_d_nv16.proof.bin");

fn assert_mixed_d_fixture_schedule(schedule: &akita_types::Schedule) {
    let folds: Vec<_> = schedule.fold_steps().collect();
    assert!(
        folds.len() > MIXED_D_SWITCH_FOLD,
        "fixture must reach suffix levels at D=64"
    );
    for (level, fold) in folds.iter().enumerate() {
        let expected_d = if level < MIXED_D_SWITCH_FOLD { 128 } else { 64 };
        assert_eq!(
            fold.params.ring_dimension, expected_d,
            "fold level {level} ring_dimension"
        );
    }
}

#[test]
fn hand_built_schedule_uniform_d128_oracle_baseline() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let layout = Cfg::get_params_for_batched_commitment(
            &akita_types::OpeningBatchShape::new(NUM_VARS, 1).expect("opening batch"),
        )
        .expect("commit layout");
        let poly = make_dense_poly(NUM_VARS, 0xcede_0001);
        let point = random_point(NUM_VARS, 0xcede_0002);
        let opening = opening_from_poly(&poly, &point, &layout);

        let setup = Scheme::setup_prover(NUM_VARS, 1).expect("setup");
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
        let stack = UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
        let verifier_setup = setup.verifier_setup().expect("verifier setup");
        let (commitment, hint) = <Scheme as akita_prover::TypedCommitmentProver<F, D>>::commit(
            &setup,
            std::slice::from_ref(&poly),
            &stack,
        )
        .expect("commit");

        // switch past last fold => identical to shipped D128Full table schedule
        let schedule = mixed_d_per_level_schedule::<Cfg, SuffixCfg>(NUM_VARS, 1, 4)
            .expect("uniform hand-built schedule");

        let opening_batch =
            akita_types::OpeningBatchShape::new(NUM_VARS, 1).expect("opening batch");
        let poly_refs = [&poly];
        let claims = typed_prove_input(&point, &poly_refs, &commitment, hint);

        let mut prover_transcript = AkitaTranscript::<F>::new(b"test/mixed_d_uniform_baseline");
        bind_transcript_instance_descriptor::<F, _, Cfg>(
            setup.expanded.as_ref(),
            &opening_batch,
            &schedule,
            BasisMode::Lagrange,
            &mut prover_transcript,
        )
        .expect("bind descriptor");
        let schedule_ctx =
            validate_schedule_context_at_entry(&schedule, setup.expanded.seed()).expect("plan");
        let proof = prove::<Cfg, _, DensePoly<F, D>, _, _, _, _, D>(
            &setup.expanded,
            &setup.prefix_slots,
            &stack,
            &mut prover_transcript,
            claims,
            &schedule,
            &schedule_ctx,
            BasisMode::Lagrange,
            SetupContributionMode::Direct,
        )
        .map(|(proof, _levels)| proof)
        .expect("prove");

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize proof");
        let schedule_digest = digest_effective_schedule(&schedule);
        if std::env::var_os("AKITA_CAPTURE_UNIFORM_ORACLE").is_some() {
            std::fs::write(
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/fixtures/wave0_uniform_handbuilt_d128_nv16.proof.bin"
                ),
                &serialized,
            )
            .expect("write uniform oracle fixture");
            eprintln!("UNIFORM_ORACLE_EFFECTIVE_SCHEDULE_DIGEST={schedule_digest:?}");
            eprintln!("UNIFORM_ORACLE_PROOF_BYTES_LEN={}", serialized.len());
            return;
        }
        assert_eq!(schedule_digest, ORACLE_EFFECTIVE_SCHEDULE_DIGEST);
        assert_eq!(serialized.as_slice(), ORACLE_PROOF_BYTES);

        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(&serialized),
            &proof_shape,
        )
        .expect("deserialize proof");

        let openings = [opening];
        let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/mixed_d_uniform_baseline");
        batched_verify_with_schedule::<Cfg, _, D>(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&point, &openings, &commitment),
            &schedule,
            BasisMode::Lagrange,
            SetupContributionMode::Direct,
        )
        .expect("verify");
    });
}

#[test]
fn mixed_d_per_level_prove_verify_and_transcript_replay() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let layout = Cfg::get_params_for_batched_commitment(
            &akita_types::OpeningBatchShape::new(NUM_VARS, 1).expect("opening batch"),
        )
        .expect("commit layout");
        let poly = make_dense_poly(NUM_VARS, 0xcede_0001);
        let point = random_point(NUM_VARS, 0xcede_0002);
        let opening = opening_from_poly(&poly, &point, &layout);

        let setup = Scheme::setup_prover(NUM_VARS, 1).expect("setup");
        assert_eq!(setup.expanded.seed().gen_ring_dim, 128);
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
        let stack = UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
        let verifier_setup = setup.verifier_setup().expect("verifier setup");
        let commit_input = std::slice::from_ref(&poly);
        let (commitment, hint) = <Scheme as akita_prover::TypedCommitmentProver<F, D>>::commit(
            &setup,
            commit_input,
            &stack,
        )
        .expect("commit");

        let schedule =
            mixed_d_per_level_schedule::<Cfg, SuffixCfg>(NUM_VARS, 1, MIXED_D_SWITCH_FOLD)
                .expect("mixed-D schedule");
        assert_mixed_d_fixture_schedule(&schedule);

        let plan = validate_ring_dim_plan_at_entry(&schedule, setup.expanded.seed())
            .expect("ring dim plan");
        assert_eq!(plan.dim_at(0).expect("d0"), 128);
        assert_eq!(plan.dim_at(1).expect("d1"), 128);
        assert_eq!(plan.dim_at(2).expect("d2"), 64);
        assert_eq!(plan.unique_dims(), vec![64, 128]);

        let opening_batch =
            akita_types::OpeningBatchShape::new(NUM_VARS, 1).expect("opening batch");
        let poly_refs = [&poly];
        let claims = typed_prove_input(&point, &poly_refs, &commitment, hint);

        let mut prover_transcript = AkitaTranscript::<F>::new(b"test/mixed_d_per_level_e2e");
        bind_transcript_instance_descriptor::<F, _, Cfg>(
            setup.expanded.as_ref(),
            &opening_batch,
            &schedule,
            BasisMode::Lagrange,
            &mut prover_transcript,
        )
        .expect("bind descriptor");
        let schedule_ctx =
            validate_schedule_context_at_entry(&schedule, setup.expanded.seed()).expect("plan");
        let proof = prove::<Cfg, _, DensePoly<F, D>, _, _, _, _, D>(
            &setup.expanded,
            &setup.prefix_slots,
            &stack,
            &mut prover_transcript,
            claims,
            &schedule,
            &schedule_ctx,
            BasisMode::Lagrange,
            SetupContributionMode::Direct,
        )
        .map(|(proof, _levels)| proof)
        .expect("mixed-D prove");

        assert!(
            !proof.is_root_direct(),
            "mixed-D fixture must exercise folded recursive prove"
        );
        assert_eq!(proof.steps.len() + 1, schedule.num_fold_levels());

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize proof");

        let schedule_digest = digest_effective_schedule(&schedule);
        assert_eq!(
            schedule_digest, MIXED_ORACLE_EFFECTIVE_SCHEDULE_DIGEST,
            "effective schedule digest oracle (Wave 0 mixed-D fixture)"
        );
        assert_eq!(
            serialized.as_slice(),
            MIXED_ORACLE_PROOF_BYTES,
            "proof wire bytes oracle (Wave 0 mixed-D fixture)"
        );

        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(&serialized),
            &proof_shape,
        )
        .expect("deserialize proof");

        let openings = [opening];
        let verify_claims = verify_input(&point, &openings, &commitment);

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/mixed_d_per_level_e2e");
        batched_verify_with_schedule::<Cfg, _, D>(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_claims,
            &schedule,
            BasisMode::Lagrange,
            SetupContributionMode::Direct,
        )
        .expect("verify");

        let mut replay_transcript = AkitaTranscript::<F>::new(b"test/mixed_d_per_level_e2e");
        batched_verify_with_schedule::<Cfg, _, D>(
            &proof,
            &verifier_setup,
            &mut replay_transcript,
            verify_input(&point, &openings, &commitment),
            &schedule,
            BasisMode::Lagrange,
            SetupContributionMode::Direct,
        )
        .expect("transcript replay verify");
    });
}
