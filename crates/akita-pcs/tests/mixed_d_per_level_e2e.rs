//! Mixed ring-dimension-per-level E2E acceptance test for the runtime ring
//! cutover (specs/runtime-ring-cutover.md §Acceptance / §Testing Strategy).
//!
//! Uses the fp128 `D128Full` setup (`gen_ring_dim = 128`) with a hand-built
//! schedule: fold levels `[0, MIXED_D_SWITCH_FOLD)` at `D = 128`, levels
//! `[MIXED_D_SWITCH_FOLD, …)` at `D = 64` (stitched from the shipped
//! `D64Full` table by `akita_config::test_support::mixed_d_per_level_schedule`).
//!
//! The proof is produced and checked exclusively through the **normal public
//! PCS API** — `AkitaCommitmentScheme::{commit, batched_prove,
//! batched_verify}` — by routing the mixed schedule through a test
//! `CommitmentConfig` whose `get_params_for_prove` returns the hand-built
//! schedule (the same hook shipped presets use for their catalogs). No
//! test-only typed path is involved.

#![allow(missing_docs)]

mod common;

use akita_config::proof_optimized::fp128;
use akita_config::test_support::mixed_d_per_level_schedule;
use akita_field::AkitaError;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::{
    AkitaBatchedProof, AkitaBatchedRootProof, AkitaLevelProof, AkitaStage2Proof,
    CleartextWitnessProof, OpeningBatchShape, RingDimPlan, RingVec, Schedule,
    SetupContributionMode,
};
use common::*;

/// Envelope preset: root levels at `D = 128`, generation ring dimension 128.
type Envelope = fp128::D128Full;
/// Suffix preset: recursive levels at `D = 64`.
type Suffix = fp128::D64Full;

/// Fold levels `[0, MIXED_D_SWITCH_FOLD)` run at `D = 128`; levels
/// `[MIXED_D_SWITCH_FOLD, …)` run at `D = 64`.
const MIXED_D_SWITCH_FOLD: usize = 2;
const NUM_VARS: usize = 16;
const ENVELOPE_D: usize = 128;
const SUFFIX_D: usize = 64;

const TRANSCRIPT_LABEL: &[u8] = b"test/mixed_d_per_level_e2e";

/// Test preset identical to [`Envelope`] except that its prove/verify
/// schedule is the hand-built mixed-D-per-level schedule. Both
/// `batched_prove` and `batched_verify` resolve their schedule through
/// `effective_batched_schedule::<Cfg>` → `Cfg::get_params_for_prove`, so this
/// override is the normal public plumbing, not a test-only side door.
#[derive(Clone, Copy, Debug, Default)]
struct MixedD128To64;

impl akita_config::CommitmentConfig for MixedD128To64 {
    type Field = <Envelope as akita_config::CommitmentConfig>::Field;
    type ExtField = <Envelope as akita_config::CommitmentConfig>::ExtField;

    const D: usize = <Envelope as akita_config::CommitmentConfig>::D;

    fn decomposition() -> akita_types::DecompositionParams {
        Envelope::decomposition()
    }

    fn ring_challenge_config(
        d: usize,
    ) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
        Envelope::ring_challenge_config(d)
    }

    fn sis_modulus_family() -> akita_types::SisModulusFamily {
        Envelope::sis_modulus_family()
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<akita_types::SetupMatrixEnvelope, AkitaError> {
        Envelope::max_setup_matrix_size(max_num_vars, max_num_batched_polys)
    }

    fn basis_range() -> (u32, u32) {
        Envelope::basis_range()
    }

    fn get_params_for_prove(opening_batch: &OpeningBatchShape) -> Result<Schedule, AkitaError> {
        let key = akita_types::AkitaScheduleLookupKey::new_from_opening_batch(opening_batch)?;
        mixed_d_per_level_schedule::<Envelope, Suffix>(
            key.num_vars,
            key.num_polynomials,
            MIXED_D_SWITCH_FOLD,
        )
    }
}

/// Like [`MixedD128To64`], but one suffix fold level advertises a ring
/// dimension that does not divide the setup's `gen_ring_dim`. Entry
/// validation (`RingDimPlan`) must reject it with an error, never a panic.
#[derive(Clone, Copy, Debug, Default)]
struct MixedDBadLevelDim;

impl akita_config::CommitmentConfig for MixedDBadLevelDim {
    type Field = <Envelope as akita_config::CommitmentConfig>::Field;
    type ExtField = <Envelope as akita_config::CommitmentConfig>::ExtField;

    const D: usize = <Envelope as akita_config::CommitmentConfig>::D;

    fn decomposition() -> akita_types::DecompositionParams {
        Envelope::decomposition()
    }

    fn ring_challenge_config(
        d: usize,
    ) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
        Envelope::ring_challenge_config(d)
    }

    fn sis_modulus_family() -> akita_types::SisModulusFamily {
        Envelope::sis_modulus_family()
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<akita_types::SetupMatrixEnvelope, AkitaError> {
        Envelope::max_setup_matrix_size(max_num_vars, max_num_batched_polys)
    }

    fn basis_range() -> (u32, u32) {
        Envelope::basis_range()
    }

    fn get_params_for_prove(opening_batch: &OpeningBatchShape) -> Result<Schedule, AkitaError> {
        let key = akita_types::AkitaScheduleLookupKey::new_from_opening_batch(opening_batch)?;
        let mut schedule = mixed_d_per_level_schedule::<Envelope, Suffix>(
            key.num_vars,
            key.num_polynomials,
            MIXED_D_SWITCH_FOLD,
        )?;
        // Corrupt the first suffix fold level: 96 does not divide the
        // setup's gen_ring_dim (128) and is not a power of two.
        if let Some(akita_types::Step::Fold(fold)) = schedule.steps.get_mut(MIXED_D_SWITCH_FOLD) {
            fold.params.ring_dimension = 96;
        }
        Ok(schedule)
    }
}

type Scheme = AkitaCommitmentScheme<MixedD128To64>;

fn mixed_schedule() -> Schedule {
    mixed_d_per_level_schedule::<Envelope, Suffix>(NUM_VARS, 1, MIXED_D_SWITCH_FOLD)
        .expect("mixed-D schedule")
}

fn assert_mixed_d_fixture_schedule(schedule: &Schedule) {
    let folds: Vec<_> = schedule.fold_steps().collect();
    assert!(
        folds.len() > MIXED_D_SWITCH_FOLD,
        "fixture must reach suffix levels at D={SUFFIX_D}"
    );
    for (level, fold) in folds.iter().enumerate() {
        let expected_d = if level < MIXED_D_SWITCH_FOLD {
            ENVELOPE_D
        } else {
            SUFFIX_D
        };
        assert_eq!(
            fold.params.ring_dimension, expected_d,
            "fold level {level} ring_dimension"
        );
    }
}

struct MixedDFixture {
    point: Vec<F>,
    openings: [F; 1],
    commitment: akita_types::Commitment<F>,
    verifier_setup: akita_types::AkitaVerifierSetup<F>,
    proof: AkitaBatchedProof<F, F>,
    serialized: Vec<u8>,
}

/// Commit + prove the mixed-D fixture once through the public PCS API.
fn prove_mixed_fixture() -> MixedDFixture {
    let opening_batch = OpeningBatchShape::new(NUM_VARS, 1).expect("opening batch");
    let layout =
        <MixedD128To64 as akita_config::CommitmentConfig>::get_params_for_batched_commitment(
            &opening_batch,
        )
        .expect("commit layout");

    let poly = make_dense_poly(NUM_VARS, 0xcede_0001);
    let point = random_point(NUM_VARS, 0xcede_0002);
    let opening = opening_from_poly::<DENSE_D, _>(&poly, &point, &layout);

    let setup = Scheme::setup_prover(NUM_VARS, 1).expect("setup");
    assert_eq!(setup.expanded.seed().gen_ring_dim, ENVELOPE_D);
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = Scheme::setup_verifier(&setup);
    let (commitment, hint) =
        Scheme::commit(&setup, std::slice::from_ref(&poly), &stack).expect("commit");

    let poly_refs = [&poly];
    let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
    let proof = Scheme::batched_prove(
        &setup,
        prove_input(&point, &poly_refs, &commitment, hint),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        SetupContributionMode::Direct,
    )
    .expect("mixed-D prove");

    let mut serialized = Vec::new();
    proof
        .serialize_compressed(&mut serialized)
        .expect("serialize proof");

    MixedDFixture {
        point,
        openings: [opening],
        commitment,
        verifier_setup,
        proof,
        serialized,
    }
}

fn verify_mixed(
    fixture: &MixedDFixture,
    proof: &AkitaBatchedProof<F, F>,
    commitment: &akita_types::Commitment<F>,
) -> Result<(), AkitaError> {
    let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
    Scheme::batched_verify(
        proof,
        &fixture.verifier_setup,
        &mut verifier_transcript,
        verify_input(&fixture.point, &fixture.openings, commitment),
        BasisMode::Lagrange,
        SetupContributionMode::Direct,
    )
}

/// Level index (0 = root) → ring dimension expected by the fixture.
fn expected_dim(level: usize) -> usize {
    if level < MIXED_D_SWITCH_FOLD {
        ENVELOPE_D
    } else {
        SUFFIX_D
    }
}

fn truncate_ring_vec(rv: &mut RingVec<F>, new_len: usize) {
    let mut coeffs = rv.coeffs().to_vec();
    assert!(
        new_len < coeffs.len(),
        "tamper must shrink the buffer ({new_len} >= {})",
        coeffs.len()
    );
    coeffs.truncate(new_len);
    *rv = RingVec::from_coeffs(coeffs);
}

#[test]
fn mixed_d_schedule_shape_and_ring_dim_plan() {
    let schedule = mixed_schedule();
    assert_mixed_d_fixture_schedule(&schedule);
    assert_eq!(schedule.num_fold_levels(), 4);

    // RingDimPlan admits the schedule under a gen_ring_dim = 128 seed and
    // reports the per-level dims.
    init_rayon_pool();
    run_on_large_stack(|| {
        let setup = Scheme::setup_prover(NUM_VARS, 1).expect("setup");
        let schedule = mixed_schedule();
        let plan =
            RingDimPlan::from_schedule(&schedule, setup.expanded.seed()).expect("ring dim plan");
        assert_eq!(plan.dim_at(0).expect("d0"), ENVELOPE_D);
        assert_eq!(plan.dim_at(1).expect("d1"), ENVELOPE_D);
        assert_eq!(plan.dim_at(2).expect("d2"), SUFFIX_D);
        assert_eq!(plan.dim_at(3).expect("d3"), SUFFIX_D);
        assert_eq!(plan.unique_dims(), vec![SUFFIX_D, ENVELOPE_D]);
    });
}

#[test]
fn mixed_d_per_level_prove_verify_replay_and_malformed_rejections() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let fixture = prove_mixed_fixture();

        // The proof must exercise the folded recursive path across both ring
        // dimensions: root fold + 3 recursive steps.
        assert!(
            matches!(fixture.proof.root, AkitaBatchedRootProof::Fold(_)),
            "mixed-D fixture must exercise the folded recursive prove path"
        );
        assert_eq!(
            fixture.proof.steps.len() + 1,
            mixed_schedule().num_fold_levels(),
            "proof must carry one step per scheduled fold level"
        );

        // Verify the in-memory proof object through the public API.
        verify_mixed(&fixture, &fixture.proof, &fixture.commitment)
            .expect("verify in-memory proof");

        // Serialization roundtrip, then verify the decoded proof against a
        // fresh transcript (transcript replay).
        let proof_shape = fixture.proof.shape();
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(&fixture.serialized),
            &proof_shape,
        )
        .expect("deserialize proof");
        assert_eq!(
            decoded, fixture.proof,
            "serialization roundtrip must preserve the mixed-D proof"
        );
        verify_mixed(&fixture, &decoded, &fixture.commitment).expect("verify decoded proof");

        // Wire tamper: flipping any single proof byte must be rejected —
        // either at deserialization or at verification — and never panic.
        for offset in [
            0usize,
            fixture.serialized.len() / 3,
            fixture.serialized.len() / 2,
            fixture.serialized.len() - 1,
        ] {
            let mut tampered = fixture.serialized.clone();
            tampered[offset] ^= 0x01;
            let rejected = match AkitaBatchedProof::<F, F>::deserialize_compressed(
                &mut std::io::Cursor::new(&tampered),
                &proof_shape,
            ) {
                Err(_) => true,
                Ok(tampered_proof) => {
                    verify_mixed(&fixture, &tampered_proof, &fixture.commitment).is_err()
                }
            };
            assert!(rejected, "byte flip at offset {offset} must be rejected");
        }

        // Root commitment length: shrink the claims-side commitment to the
        // suffix level's dim-sized footprint (wrong level's dim).
        {
            let mut commitment = fixture.commitment.clone();
            let len = commitment.rows().coeffs().len();
            truncate_ring_vec(&mut commitment.0, len / (ENVELOPE_D / SUFFIX_D));
            let err = verify_mixed(&fixture, &fixture.proof, &commitment)
                .expect_err("wrong-dim root commitment must be rejected");
            let _: AkitaError = err;
        }

        // Root fold `next_w_commitment` length: size it at the wrong level's
        // ring dimension footprint.
        {
            let mut proof = fixture.proof.clone();
            let AkitaBatchedRootProof::Fold(root) = &mut proof.root else {
                panic!("fixture root must be a fold proof");
            };
            let stage2 = root
                .stage2
                .as_intermediate_mut()
                .expect("root fold stage2 must be intermediate");
            let len = stage2.next_w_commitment.coeffs().len();
            truncate_ring_vec(&mut stage2.next_w_commitment, len / (ENVELOPE_D / SUFFIX_D));
            verify_mixed(&fixture, &proof, &fixture.commitment)
                .expect_err("wrong-dim root next_w_commitment must be rejected");
        }

        // Recursive fold commitment length at every intermediate suffix
        // level: a commitment sized at the OTHER level's dim must be
        // rejected (this is the mixed-D-specific length confusion).
        for (idx, step) in fixture.proof.steps.iter().enumerate() {
            let level = idx + 1;
            if !matches!(step, AkitaLevelProof::Intermediate { .. }) {
                continue;
            }
            let mut proof = fixture.proof.clone();
            let AkitaLevelProof::Intermediate { stage2, .. } = &mut proof.steps[idx] else {
                unreachable!();
            };
            let AkitaStage2Proof::Intermediate(inner) = stage2 else {
                panic!("intermediate level {level} must carry intermediate stage2");
            };
            let len = inner.next_w_commitment.coeffs().len();
            // Rescale the commitment as if it had been produced at the wrong
            // level's ring dimension.
            let wrong_len = len * expected_dim(level.saturating_sub(1)) / expected_dim(level + 1);
            let new_len = if wrong_len == len { len / 2 } else { wrong_len };
            if new_len >= len {
                let mut coeffs = inner.next_w_commitment.coeffs().to_vec();
                coeffs.resize(new_len, F::zero());
                inner.next_w_commitment = RingVec::from_coeffs(coeffs);
            } else {
                truncate_ring_vec(&mut inner.next_w_commitment, new_len);
            }
            verify_mixed(&fixture, &proof, &fixture.commitment).expect_err(
                "recursive fold commitment sized at the wrong level's dim must be rejected",
            );
        }

        // Fold `v` vector length (D · ŵ at the level's own dim).
        {
            let mut proof = fixture.proof.clone();
            let AkitaLevelProof::Intermediate { v, .. } = &mut proof.steps[0] else {
                panic!("first recursive step must be intermediate");
            };
            let len = v.coeffs().len();
            truncate_ring_vec(v, len / 2);
            verify_mixed(&fixture, &proof, &fixture.commitment)
                .expect_err("wrong-length fold v vector must be rejected");
        }

        // Terminal/direct witness length: drop payload bytes / digit fields
        // from the cleartext terminal witness (which lives at D = 64 here).
        {
            let mut proof = fixture.proof.clone();
            let terminal = proof
                .steps
                .last_mut()
                .and_then(AkitaLevelProof::as_terminal_mut)
                .expect("fixture must end in a terminal step");
            let witness = terminal
                .stage2_mut()
                .final_witness_mut()
                .expect("terminal step must carry final witness");
            match witness {
                CleartextWitnessProof::SegmentTyped(segment) => {
                    segment.z_payload.pop();
                }
                CleartextWitnessProof::FieldElements(elems) => {
                    let len = elems.coeffs().len();
                    truncate_ring_vec(elems, len.saturating_sub(1));
                }
            }
            verify_mixed(&fixture, &proof, &fixture.commitment)
                .expect_err("wrong-length terminal witness must be rejected");
        }

        // Terminal witness digit-field (e_fields) length.
        {
            let mut proof = fixture.proof.clone();
            let terminal = proof
                .steps
                .last_mut()
                .and_then(AkitaLevelProof::as_terminal_mut)
                .expect("fixture must end in a terminal step");
            let witness = terminal
                .stage2_mut()
                .final_witness_mut()
                .expect("terminal step must carry final witness");
            if let CleartextWitnessProof::SegmentTyped(segment) = witness {
                let len = segment.e_fields.coeffs().len();
                truncate_ring_vec(&mut segment.e_fields, len.saturating_sub(1));
                verify_mixed(&fixture, &proof, &fixture.commitment)
                    .expect_err("wrong-length terminal e_fields must be rejected");
            }
        }
    });
}

#[test]
fn mixed_d_malformed_hint_digit_length_rejected() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let poly = make_dense_poly(NUM_VARS, 0xcede_0001);
        let point = random_point(NUM_VARS, 0xcede_0002);

        let setup = Scheme::setup_prover(NUM_VARS, 1).expect("setup");
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let (commitment, _hint) =
            Scheme::commit(&setup, std::slice::from_ref(&poly), &stack).expect("commit");

        let poly_refs = [&poly];

        // Hint with no per-polynomial digit streams at all.
        let empty_hint = AkitaCommitmentHint::<F>::new(Vec::new());
        let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
        Scheme::batched_prove(
            &setup,
            prove_input(&point, &poly_refs, &commitment, empty_hint),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
            SetupContributionMode::Direct,
        )
        .expect_err("prove must reject a hint with a missing digit stream");

        // Hint whose digit stream is sized at the wrong level's ring
        // dimension (D=64 stride for the D=128 root) with a wrong length.
        let wrong_dim_hint = AkitaCommitmentHint::<F>::singleton(
            akita_types::DigitBlocks::zeroed(vec![1], SUFFIX_D).expect("digit blocks"),
        );
        let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
        Scheme::batched_prove(
            &setup,
            prove_input(&point, &poly_refs, &commitment, wrong_dim_hint),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
            SetupContributionMode::Direct,
        )
        .expect_err("prove must reject a hint digit stream sized at the wrong level's dim");
    });
}

#[test]
fn mixed_d_schedule_with_non_dividing_level_dim_is_rejected() {
    init_rayon_pool();
    run_on_large_stack(|| {
        type BadScheme = AkitaCommitmentScheme<MixedDBadLevelDim>;

        let opening_batch = OpeningBatchShape::new(NUM_VARS, 1).expect("opening batch");
        let layout = <MixedDBadLevelDim as akita_config::CommitmentConfig>::
            get_params_for_batched_commitment(&opening_batch)
        .expect("commit layout (root level params are untouched)");

        let poly = make_dense_poly(NUM_VARS, 0xcede_0001);
        let point = random_point(NUM_VARS, 0xcede_0002);
        let opening = opening_from_poly::<DENSE_D, _>(&poly, &point, &layout);

        let setup = BadScheme::setup_prover(NUM_VARS, 1).expect("setup");
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup = BadScheme::setup_verifier(&setup);
        let (commitment, hint) =
            BadScheme::commit(&setup, std::slice::from_ref(&poly), &stack).expect("commit");

        // Prover entry must reject the schedule (level dim 96 does not
        // divide gen_ring_dim 128) with an error, not a panic.
        let poly_refs = [&poly];
        let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
        BadScheme::batched_prove(
            &setup,
            prove_input(&point, &poly_refs, &commitment, hint),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
            SetupContributionMode::Direct,
        )
        .expect_err("prove must reject a level dim that does not divide gen_ring_dim");

        // Verifier entry must reject the same schedule for any proof bytes.
        let good = prove_mixed_fixture();
        let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
        let openings = [opening];
        BadScheme::batched_verify(
            &good.proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&point, &openings, &commitment),
            BasisMode::Lagrange,
            SetupContributionMode::Direct,
        )
        .expect_err("verify must reject a level dim that does not divide gen_ring_dim");
    });
}
