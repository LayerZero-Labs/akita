//! Mixed ring-dimension-per-level E2E acceptance test for the runtime ring
//! cutover (specs/runtime-ring-cutover.md §Acceptance / §Testing Strategy).
//!
//! Uses a flat fp128 public setup with a hand-built schedule: fold levels
//! `[0, MIXED_D_SWITCH_FOLD)` at `D = 128`, levels
//! `[MIXED_D_SWITCH_FOLD, …)` at `D = 64` (stitched from the shipped
//! `D64Dense` table by [`mixed_d_per_level_fixture::mixed_d_per_level_schedule`]).
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
use akita_field::AkitaError;
use akita_pcs::test_support::{mixed_d_per_level_schedule, MixedDConfig};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::{
    setup_matrix_capacity_for_schedule, validate_schedule_ring_dims, AkitaBatchedProof,
    AkitaScheduleLookupKey, FoldSchedule, GroupBatchStatement, NextWitnessBinding,
    OpeningClaimsLayout, OpeningScheduleSelection, PolynomialGroupLayout, RingVec,
};
use common::*;

/// Root preset: leading levels at `D = 128`.
type Envelope = fp128::D128Dense;
/// Suffix preset: recursive levels at `D = 64`.
type Suffix = fp128::D64Dense;

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
type MixedD128To64 = MixedDConfig<Envelope, Suffix, MIXED_D_SWITCH_FOLD>;

/// Like [`MixedD128To64`], but one suffix fold advertises unsupported ring
/// dimension 96. Entry validation (`validate_schedule_ring_dims`) must reject
/// it with an error, never a panic.
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

    fn sis_modulus_profile() -> akita_types::SisModulusProfileId {
        Envelope::sis_modulus_profile()
    }

    fn setup_matrix_capacity(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<akita_types::SetupMatrixCapacity, AkitaError> {
        Envelope::setup_matrix_capacity(max_num_vars, max_num_batched_polys)
    }

    fn basis_range() -> (u32, u32) {
        Envelope::basis_range()
    }

    fn root_honest_fold_policy() -> akita_types::sis::HonestFoldPolicySpec {
        Envelope::root_honest_fold_policy()
    }

    fn get_params_for_prove(
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        let mut schedule = mixed_d_per_level_schedule::<Envelope, Suffix>(
            opening_batch.max_num_vars(),
            opening_batch.num_total_polynomials(),
            MIXED_D_SWITCH_FOLD,
        )?;
        // Corrupt the first suffix fold level with unsupported dimension 96.
        if let Some(fold) = schedule.recursive_folds.get_mut(MIXED_D_SWITCH_FOLD - 1) {
            let matrix = &fold.params.witness.inner_commit_matrix;
            fold.params.witness.inner_commit_matrix =
                akita_types::InnerCommitMatrixParams::new_unchecked(
                    matrix.security_policy(),
                    matrix.sis_table_key().table_digest,
                    matrix.sis_modulus_profile(),
                    matrix.output_rank(),
                    matrix.input_width(),
                    matrix.coeff_linf_bound(),
                    96,
                );
        }
        Ok(schedule)
    }
}

type Scheme = AkitaCommitmentScheme<MixedD128To64>;

fn make_envelope_dense_poly(nv: usize, seed: u64) -> DensePoly<F> {
    let evals = dense_field_evals(nv, seed);
    DensePoly::<F>::from_field_evals(nv, ENVELOPE_D, &evals).expect("dense poly")
}

fn mixed_schedule() -> FoldSchedule {
    mixed_d_per_level_schedule::<Envelope, Suffix>(NUM_VARS, 1, MIXED_D_SWITCH_FOLD)
        .expect("mixed-D schedule")
}

fn assert_mixed_d_fixture_schedule(schedule: &FoldSchedule) {
    let dims = std::iter::once(schedule.root.params.final_group.commitment.d_a())
        .chain(
            schedule
                .recursive_folds
                .iter()
                .map(|step| step.params.witness.d_a()),
        )
        .chain(std::iter::once(schedule.terminal.params.witness.d_a()))
        .collect::<Vec<_>>();
    assert!(
        dims.len() > MIXED_D_SWITCH_FOLD,
        "fixture must reach suffix levels at D={SUFFIX_D}"
    );
    for (level, actual_d) in dims.into_iter().enumerate() {
        let expected_d = if level < MIXED_D_SWITCH_FOLD {
            ENVELOPE_D
        } else {
            SUFFIX_D
        };
        assert_eq!(actual_d, expected_d, "fold level {level} ring_dimension");
    }
}

struct MixedDFixture {
    point: Vec<F>,
    openings: [F; 1],
    commitment: akita_types::CommittedGroup<F>,
    verifier_setup: akita_types::AkitaVerifierSetup<F>,
    proof: AkitaBatchedProof<F, F>,
    serialized: Vec<u8>,
}

/// Commit + prove the mixed-D fixture once through the public PCS API.
fn prove_mixed_fixture() -> MixedDFixture {
    let opening_batch = OpeningClaimsLayout::new(NUM_VARS, 1).expect("opening batch");
    let layout =
        <MixedD128To64 as akita_config::CommitmentConfig>::get_params_for_batched_commitment(
            &opening_batch,
        )
        .expect("commit layout");

    let poly = make_envelope_dense_poly(NUM_VARS, 0xcede_0001);
    let point = random_point(NUM_VARS, 0xcede_0002);
    let opening = opening_from_poly::<ENVELOPE_D, _>(&poly, &point, &layout);

    let setup = Scheme::setup_prover(NUM_VARS, 1).expect("setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = Scheme::setup_verifier(&setup).expect("verifier setup");
    let (commitment, hint) =
        Scheme::commit(&setup, std::slice::from_ref(&poly), &stack).expect("commit");

    let poly_refs = [&poly];
    let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
    let proof = Scheme::batched_prove(
        &setup,
        prove_input::<MixedD128To64, _>(&point, &poly_refs, &commitment, hint),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
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
    commitment: &akita_types::CommittedGroup<F>,
) -> Result<(), AkitaError> {
    let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
    Scheme::batched_verify(
        proof,
        &fixture.verifier_setup,
        &mut verifier_transcript,
        verify_input::<MixedD128To64>(&fixture.point, &fixture.openings, commitment),
        BasisMode::Lagrange,
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
fn mixed_d_schedule_shape_and_ring_dim_validation() {
    let schedule = mixed_schedule();
    assert_mixed_d_fixture_schedule(&schedule);

    init_rayon_pool();
    run_on_large_stack(|| {
        let setup = Scheme::setup_prover(NUM_VARS, 1).expect("setup");
        let schedule = mixed_schedule();
        validate_schedule_ring_dims(&schedule, setup.expanded.seed()).expect("ring dims valid");
        assert_mixed_d_fixture_schedule(&schedule);
        let mut unique = std::collections::BTreeSet::new();
        let root_dims = schedule.root.params.final_group.commitment.role_dims();
        unique.insert(root_dims.inner);
        unique.insert(root_dims.outer);
        unique.insert(root_dims.opening);
        for step in &schedule.recursive_folds {
            let dims = step.params.witness.role_dims();
            unique.insert(dims.inner);
            unique.insert(dims.outer);
            unique.insert(dims.opening);
        }
        unique.insert(schedule.terminal.params.witness.d_a());
        assert_eq!(
            unique.into_iter().collect::<Vec<_>>(),
            vec![SUFFIX_D, ENVELOPE_D]
        );
    });
}

#[test]
fn tableless_mixed_d_setup_uses_the_synthetic_schedule_envelope() {
    type TablelessMixedD = MixedDConfig<fp128::D256OneHot, fp128::D64OneHot, 1>;
    type TablelessScheme = AkitaCommitmentScheme<TablelessMixedD>;
    const TABLELESS_NUM_VARS: usize = 20;

    let schedule = TablelessMixedD::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(TABLELESS_NUM_VARS),
    ))
    .expect("tableless mixed-D schedule");
    let required =
        setup_matrix_capacity_for_schedule(&schedule).expect("synthetic schedule envelope");
    let configured = TablelessMixedD::setup_matrix_capacity(TABLELESS_NUM_VARS, 1)
        .expect("mixed-D setup capacity");
    assert!(configured.num_field_elements >= required.num_field_elements);

    let setup = TablelessScheme::setup_prover(TABLELESS_NUM_VARS, 1)
        .expect("tableless mixed-D prover setup");
    assert!(setup.expanded.seed().num_field_elements >= required.num_field_elements);
}

#[test]
fn mixed_d_per_level_prove_verify_replay_and_malformed_rejections() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let fixture = prove_mixed_fixture();

        // The proof must exercise the folded recursive path across both ring
        // dimensions: root fold + 3 recursive steps.
        assert_eq!(
            fixture.proof.num_fold_levels(),
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
            truncate_ring_vec(&mut commitment.commitment.0, len / (ENVELOPE_D / SUFFIX_D));
            let err = verify_mixed(&fixture, &fixture.proof, &commitment)
                .expect_err("wrong-dim root commitment must be rejected");
            let _: AkitaError = err;
        }

        // A value-level root commitment-row tamper must fail root replay.
        {
            let mut commitment = fixture.commitment.clone();
            let mut coeffs = commitment.commitment.0.coeffs().to_vec();
            coeffs[0] += F::one();
            commitment.commitment.0 = RingVec::from_coeffs(coeffs);
            verify_mixed(&fixture, &fixture.proof, &commitment)
                .expect_err("tampered commitment row must be rejected");
        }

        // Root fold `next_w_commitment` length: size it at the wrong level's
        // ring dimension footprint.
        {
            let mut proof = fixture.proof.clone();
            let stage2 = &mut proof.root.stage2;
            let NextWitnessBinding::OuterCommitment(next_w_commitment) =
                &mut stage2.next_witness_binding
            else {
                panic!("mixed-D fixture root must carry an outer commitment");
            };
            let len = next_w_commitment.coeffs().len();
            truncate_ring_vec(next_w_commitment, len / (ENVELOPE_D / SUFFIX_D));
            verify_mixed(&fixture, &proof, &fixture.commitment)
                .expect_err("wrong-dim root next_w_commitment must be rejected");
        }

        // Recursive fold commitment length at every intermediate suffix
        // level: a commitment sized at the OTHER level's dim must be
        // rejected (this is the mixed-D-specific length confusion).
        for (idx, _) in fixture.proof.recursive_folds.iter().enumerate() {
            let level = idx + 1;
            let mut proof = fixture.proof.clone();
            let inner = &mut proof.recursive_folds[idx].stage2;
            let NextWitnessBinding::OuterCommitment(next_w_commitment) =
                &mut inner.next_witness_binding
            else {
                continue;
            };
            let len = next_w_commitment.coeffs().len();
            // Rescale the commitment as if it had been produced at the wrong
            // level's ring dimension.
            let wrong_len = len * expected_dim(level.saturating_sub(1)) / expected_dim(level + 1);
            let new_len = if wrong_len == len { len / 2 } else { wrong_len };
            if new_len >= len {
                let mut coeffs = next_w_commitment.coeffs().to_vec();
                coeffs.resize(new_len, F::zero());
                *next_w_commitment = RingVec::from_coeffs(coeffs);
            } else {
                truncate_ring_vec(next_w_commitment, new_len);
            }
            verify_mixed(&fixture, &proof, &fixture.commitment).expect_err(
                "recursive fold commitment sized at the wrong level's dim must be rejected",
            );
        }

        // Fold `v` vector length (D · ŵ at the level's own dim).
        {
            let mut proof = fixture.proof.clone();
            let v = &mut proof.recursive_folds[0].v;
            let len = v.coeffs().len();
            truncate_ring_vec(v, len / 2);
            verify_mixed(&fixture, &proof, &fixture.commitment)
                .expect_err("wrong-length fold v vector must be rejected");
        }

        // Terminal/direct witness length: drop payload bytes / digit fields
        // from the cleartext terminal witness (which lives at D = 64 here).
        {
            let mut proof = fixture.proof.clone();
            let witness = proof.terminal.terminal_response_mut();
            witness.z_payloads[0].pop();
            verify_mixed(&fixture, &proof, &fixture.commitment)
                .expect_err("wrong-length terminal witness must be rejected");
        }

        // Terminal witness digit-field (e_fields) length.
        {
            let mut proof = fixture.proof.clone();
            let witness = proof.terminal.terminal_response_mut();
            let len = witness.e_fields.coeffs().len();
            truncate_ring_vec(&mut witness.e_fields, len.saturating_sub(1));
            verify_mixed(&fixture, &proof, &fixture.commitment)
                .expect_err("wrong-length terminal e_fields must be rejected");
        }

        // Terminal t segment is the predecessor-bound inner state and must be
        // linked to the response by the direct A relation.
        {
            let mut proof = fixture.proof.clone();
            let witness = proof.terminal.terminal_response_mut();
            let mut coeffs = witness.t_fields.coeffs().to_vec();
            coeffs[0] += F::one();
            witness.t_fields = RingVec::from_coeffs(coeffs);
            verify_mixed(&fixture, &proof, &fixture.commitment)
                .expect_err("tampered terminal t_fields must be rejected");
        }
    });
}

#[test]
fn mixed_d_malformed_hint_inner_rows_rejected() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let poly = make_envelope_dense_poly(NUM_VARS, 0xcede_0001);
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

        // Hint with no per-polynomial A rows at all.
        let empty_hint = AkitaCommitmentHint::<F>::new(ENVELOPE_D, Vec::new()).expect("empty hint");
        let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
        Scheme::batched_prove(
            &setup,
            prove_input::<MixedD128To64, _>(&point, &poly_refs, &commitment, empty_hint),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect_err("prove must reject a hint with missing inner rows");

        // Hint whose semantic rows declare the suffix dimension instead of
        // the root A dimension.
        let wrong_dim_hint = AkitaCommitmentHint::<F>::singleton(
            RingVec::from_coeffs_with_ring_dim(vec![F::zero(); SUFFIX_D], SUFFIX_D)
                .expect("inner rows"),
        )
        .expect("wrong-dimension hint");
        let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
        Scheme::batched_prove(
            &setup,
            prove_input::<MixedD128To64, _>(&point, &poly_refs, &commitment, wrong_dim_hint),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect_err("prove must reject hint rows at the wrong A dimension");
    });
}

#[test]
fn mixed_d_schedule_with_non_dividing_level_dim_is_rejected() {
    init_rayon_pool();
    run_on_large_stack(|| {
        type BadScheme = AkitaCommitmentScheme<MixedDBadLevelDim>;

        let opening_batch = OpeningClaimsLayout::new(NUM_VARS, 1).expect("opening batch");
        let layout = <MixedDBadLevelDim as akita_config::CommitmentConfig>::
            get_params_for_batched_commitment(&opening_batch)
        .expect("commit layout (root level params are untouched)");

        let poly = make_envelope_dense_poly(NUM_VARS, 0xcede_0001);
        let point = random_point(NUM_VARS, 0xcede_0002);
        let opening = opening_from_poly::<ENVELOPE_D, _>(&poly, &point, &layout);

        let setup = BadScheme::setup_prover(NUM_VARS, 1).expect("setup");
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup = BadScheme::setup_verifier(&setup).expect("verifier setup");
        let (commitment, hint) =
            BadScheme::commit(&setup, std::slice::from_ref(&poly), &stack).expect("commit");

        let malformed_schedule =
            MixedDBadLevelDim::get_params_for_prove(&opening_batch).expect("malformed schedule");
        validate_schedule_ring_dims(&malformed_schedule, setup.expanded.seed())
            .expect_err("unsupported level dimension 96 must reject");

        // An invalid schedule cannot be materialized as an audited resolved
        // row. Exercise both public entries with an unresolved selection and
        // require an error rather than a panic.
        let unresolved_selection = OpeningScheduleSelection::default();
        let poly_refs = [&poly];
        let prover_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point.clone(),
            vec![opening],
            commitment.clone(),
        )
        .expect("prover claims group")])
        .expect("prover claims");
        let prover_data = (
            unresolved_selection,
            ProverOpeningData::new(prover_claims, vec![hint], vec![&poly_refs])
                .expect("prover opening data"),
        );
        let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
        BadScheme::batched_prove(
            &setup,
            prover_data,
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect_err("prove must reject an unresolved malformed schedule");

        // Verifier entry must reject the same unresolved selection for any
        // proof bytes.
        let good = prove_mixed_fixture();
        let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
        let openings = [opening];
        let verifier_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point.clone(),
            openings.to_vec(),
            &commitment,
        )
        .expect("verifier claims group")])
        .expect("verifier claims");
        let statement = GroupBatchStatement::new(unresolved_selection, verifier_claims)
            .expect("verifier statement");
        BadScheme::batched_verify(
            &good.proof,
            &verifier_setup,
            &mut verifier_transcript,
            statement,
            BasisMode::Lagrange,
        )
        .expect_err("verify must reject an unresolved malformed schedule");
    });
}
