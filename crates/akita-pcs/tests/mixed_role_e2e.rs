//! Direct-setup E2E coverage for mixed commitment-role ring dimensions.
//!
//! The root fold uses `d_a/d_b/d_d = 128/64/32`; later folds retain the
//! shipped D128 schedule. Recursive setup offload intentionally requires
//! uniform predecessor role dimensions, so this fixture exercises the
//! supported mixed-role path: direct root setup contraction followed by the
//! ordinary recursive fold suffix. The proof is committed, produced,
//! serialized, and verified only through the public PCS API.

#![allow(missing_docs)]

mod common;

use akita_config::{proof_optimized::fp128, CommitmentConfig};
use akita_field::AkitaError;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::sis::SisTableKey;
use akita_types::{
    intermediate_w_ring_element_count_with_counts_bits, validate_schedule_ring_dims,
    AkitaBatchedProof, AkitaScheduleLookupKey, CommitmentRingDims, FoldSchedule,
    OpenCommitMatrixParams, OpeningClaimsLayout, OuterCommitMatrixParams, PolynomialGroupLayout,
};
use common::*;

type Envelope = fp128::D128Full;
type Scheme = AkitaCommitmentScheme<MixedRoleRoot>;

const NUM_VARS: usize = 16;
const D: usize = Envelope::D;
const ROLE_DIMS: CommitmentRingDims = CommitmentRingDims {
    inner: 128,
    outer: 64,
    opening: 32,
};
const TRANSCRIPT_LABEL: &[u8] = b"test/mixed_role_e2e";

#[derive(Clone, Copy, Debug, Default)]
struct MixedRoleRoot;

fn retarget_matrix_shape(
    key: SisTableKey,
    input_width: usize,
    ring_dimension: usize,
) -> Result<(SisTableKey, usize), AkitaError> {
    if ring_dimension == 0 || !D.is_multiple_of(ring_dimension) {
        return Err(AkitaError::InvalidSetup(
            "mixed-role matrix dimension must divide the setup envelope".into(),
        ));
    }
    let column_scale = D.checked_div(ring_dimension).ok_or_else(|| {
        AkitaError::InvalidSetup("mixed-role matrix dimension must divide the envelope".into())
    })?;
    let input_width = input_width
        .checked_mul(column_scale)
        .ok_or_else(|| AkitaError::InvalidSetup("mixed-role matrix width overflow".into()))?;
    Ok((
        SisTableKey {
            ring_dimension: ring_dimension as u32,
            ..key
        },
        input_width,
    ))
}

fn retarget_outer_matrix(
    matrix: &OuterCommitMatrixParams,
    ring_dimension: usize,
) -> Result<OuterCommitMatrixParams, AkitaError> {
    let (key, input_width) =
        retarget_matrix_shape(matrix.sis_table_key(), matrix.input_width(), ring_dimension)?;
    OuterCommitMatrixParams::try_new_with_min_rank(key, input_width)
}

fn retarget_open_matrix(
    matrix: &OpenCommitMatrixParams,
    ring_dimension: usize,
) -> Result<OpenCommitMatrixParams, AkitaError> {
    let (key, input_width) =
        retarget_matrix_shape(matrix.sis_table_key(), matrix.input_width(), ring_dimension)?;
    OpenCommitMatrixParams::try_new_with_min_rank(key, input_width)
}

fn mixed_role_schedule(
    num_vars: usize,
    num_polynomials: usize,
) -> Result<FoldSchedule, AkitaError> {
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(num_vars, num_polynomials));
    let mut schedule = Envelope::runtime_schedule(key)?;
    let field_bits = Envelope::decomposition().field_bits();

    {
        let root = &mut schedule.root.params.final_group.commitment;
        root.outer_commit_matrix =
            retarget_outer_matrix(&root.outer_commit_matrix, ROLE_DIMS.d_b())?;
        root.open_commit_matrix = retarget_open_matrix(&root.open_commit_matrix, ROLE_DIMS.d_d())?;
        schedule.root.params.open_commit_matrix = root.open_commit_matrix.clone();
    }

    let root = &schedule.root.params.final_group.commitment;
    let successor_d = schedule
        .recursive_folds
        .first()
        .map_or(schedule.terminal.params.witness.d_a(), |step| {
            step.params.witness.d_a()
        });
    let next_w_len =
        intermediate_w_ring_element_count_with_counts_bits(field_bits, root, num_polynomials, 1)?
            .checked_mul(successor_d)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("mixed-role next witness length overflow".into())
            })?;
    schedule.root.output_witness_len = next_w_len;
    if let Some(successor) = schedule.recursive_folds.first_mut() {
        successor.input_witness_len = next_w_len;
    } else {
        schedule.terminal.input_witness_len = next_w_len;
    }
    schedule.validate_structure()?;
    Ok(schedule)
}

impl CommitmentConfig for MixedRoleRoot {
    type Field = <Envelope as CommitmentConfig>::Field;
    type ExtField = <Envelope as CommitmentConfig>::ExtField;

    const D: usize = Envelope::D;

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

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<akita_types::SetupMatrixEnvelope, AkitaError> {
        Envelope::max_setup_matrix_size(max_num_vars, max_num_batched_polys)
    }

    fn basis_range() -> (u32, u32) {
        Envelope::basis_range()
    }

    fn get_params_for_prove(
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        mixed_role_schedule(
            opening_batch.max_num_vars(),
            opening_batch.num_total_polynomials(),
        )
    }
}

fn verify_proof(
    proof: &AkitaBatchedProof<F, F>,
    verifier_setup: &akita_types::AkitaVerifierSetup<F>,
    point: &[F],
    opening: F,
    commitment: &akita_types::Commitment<F>,
) -> Result<(), AkitaError> {
    let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
    Scheme::batched_verify(
        proof,
        verifier_setup,
        &mut verifier_transcript,
        verify_input(point, &[opening], commitment),
        BasisMode::Lagrange,
    )
}

#[test]
fn direct_setup_mixed_role_root_proves_serializes_and_verifies() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let opening_batch = OpeningClaimsLayout::new(NUM_VARS, 1).expect("opening batch");
        let schedule = mixed_role_schedule(NUM_VARS, 1).expect("mixed-role schedule");
        let root = &schedule.root.params.final_group.commitment;
        assert_eq!(root.role_dims(), ROLE_DIMS);
        assert_eq!(
            schedule
                .recursive_folds
                .first()
                .map_or(akita_types::SetupContributionMode::Direct, |successor| {
                    successor.params.predecessor_setup_contribution_mode()
                }),
            akita_types::SetupContributionMode::Direct,
        );
        assert_eq!(ROLE_DIMS.common_relation_coeff_count(), 32);
        assert_eq!(
            ROLE_DIMS.common_relation_witness_coeff_count(
                schedule
                    .recursive_folds
                    .first()
                    .map_or(schedule.terminal.params.witness.d_a(), |step| {
                        step.params.witness.d_a()
                    })
            ),
            32
        );

        let layout = MixedRoleRoot::get_params_for_batched_commitment(&opening_batch)
            .expect("commitment layout");
        let evals = dense_field_evals(NUM_VARS, 0x6d69_7865_645f_726f);
        let poly = DensePoly::<F>::from_field_evals(NUM_VARS, D, &evals).expect("dense poly");
        let point = random_point(NUM_VARS, 0x6d69_7865_645f_7074);
        let opening = opening_from_poly::<D, _>(&poly, &point, &layout);

        let setup = Scheme::setup_prover(NUM_VARS, 1).expect("setup");
        validate_schedule_ring_dims(&schedule, setup.expanded.seed()).expect("valid role dims");
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("prover stack");
        let verifier_setup = Scheme::setup_verifier(&setup).expect("verifier setup");
        let (commitment, hint) =
            Scheme::commit(&setup, std::slice::from_ref(&poly), &stack).expect("commit");

        let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
        let proof = Scheme::batched_prove(
            &setup,
            prove_input(&point, &[&poly], &commitment, hint),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("mixed-role prove");
        verify_proof(&proof, &verifier_setup, &point, opening, &commitment)
            .expect("verify in-memory mixed-role proof");

        let mut bytes = Vec::new();
        proof
            .serialize_compressed(&mut bytes)
            .expect("serialize mixed-role proof");
        let decoded = AkitaBatchedProof::deserialize_compressed(
            &mut std::io::Cursor::new(&bytes),
            &proof.shape(),
        )
        .expect("deserialize mixed-role proof");
        assert_eq!(decoded, proof, "mixed-role proof serialization roundtrip");
        verify_proof(&decoded, &verifier_setup, &point, opening, &commitment)
            .expect("verify decoded mixed-role proof");
    });
}
