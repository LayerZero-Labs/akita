// MERGE-FLAG(#314<->317): disabled during the Stage-2=#314 / schedule=317 merge.
// This relation/Stage-2 eval test targets an API that no longer exists after the
// merge (317's compute_relation_weight_evals / eval_at_point, or #314's Schedule/folds).
// Re-derive against the merged relation_range_image Stage-2 API before re-enabling.
#![cfg(any())]

//! Direct-setup E2E coverage for mixed commitment-role ring dimensions.
//!
//! The root fold uses `d_a/d_b/d_d = 128/64/32`; later folds retain the
//! shipped D128 schedule. The proof is committed, produced, and verified only
//! through the public PCS API.

#![allow(missing_docs)]

mod common;

use akita_config::{policy_of, proof_optimized::fp128, CommitmentConfig};
use akita_field::AkitaError;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_transcript::AkitaTranscript;
use akita_types::sis::{min_secure_rank, SisTableKey};
use akita_types::{
    intermediate_w_ring_element_count_with_counts_bits, level_proof_bytes,
    validate_schedule_ring_dims, AjtaiKeyParams, AkitaScheduleLookupKey, CommitmentRingDims,
    NextWitnessBindingPolicy, OpeningClaimsLayout, PolynomialGroupLayout, RelationMatrixRowLayout,
    Schedule,
};
use common::*;

type Envelope = fp128::D128Dense;
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

fn retarget_key(key: &AjtaiKeyParams, ring_dimension: usize) -> Result<AjtaiKeyParams, AkitaError> {
    let column_scale = D.checked_div(ring_dimension).ok_or_else(|| {
        AkitaError::InvalidSetup("mixed-role key dimension must divide the envelope".into())
    })?;
    let col_len = key
        .col_len()
        .checked_mul(column_scale)
        .ok_or_else(|| AkitaError::InvalidSetup("mixed-role key width overflow".into()))?;
    let table_key = SisTableKey {
        ring_dimension: ring_dimension as u32,
        ..key.sis_table_key()
    };
    let row_len = min_secure_rank(table_key, col_len as u64).ok_or_else(|| {
        AkitaError::InvalidSetup("mixed-role key is outside the audited SIS table".into())
    })?;
    AjtaiKeyParams::try_new(
        key.security_policy(),
        table_key.table_digest,
        key.sis_modulus_profile(),
        table_key.role,
        row_len,
        col_len,
        key.coeff_linf_bound(),
        ring_dimension,
    )
}

fn mixed_role_schedule(num_vars: usize, num_polynomials: usize) -> Result<Schedule, AkitaError> {
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(num_vars, num_polynomials));
    let mut schedule = Envelope::runtime_schedule(key)?;
    let root = schedule
        .folds
        .first_mut()
        .ok_or_else(|| AkitaError::InvalidSetup("mixed-role fixture needs a root fold".into()))?;
    root.params.b_key = retarget_key(&root.params.b_key, ROLE_DIMS.d_b())?;
    root.params.d_key = retarget_key(&root.params.d_key, ROLE_DIMS.d_d())?;
    root.params.stamp_role_dims_from_keys();

    let field_bits = Envelope::decomposition().field_bits();
    let next_w_len = intermediate_w_ring_element_count_with_counts_bits(
        field_bits,
        &schedule.folds[0].params,
        num_polynomials,
        1,
    )?
    .checked_mul(schedule.folds[1].params.d_a())
    .ok_or_else(|| AkitaError::InvalidSetup("mixed-role next witness length overflow".into()))?;
    schedule.folds[0].next_w_len = next_w_len;
    schedule.folds[1].current_w_len = next_w_len;

    let root = &schedule.folds[0];
    let next = schedule.folds.get(1).ok_or_else(|| {
        AkitaError::InvalidSetup("mixed-role fixture needs a recursive successor".into())
    })?;
    let binding = if schedule.folds.len() == 2 {
        NextWitnessBindingPolicy::TerminalInnerState
    } else {
        NextWitnessBindingPolicy::OuterCommitment
    };
    let challenge_field_bits = field_bits * policy_of::<Envelope>().chal_ext_degree as u32;
    schedule.folds[0].level_bytes = level_proof_bytes(
        field_bits,
        challenge_field_bits,
        &root.params,
        Some(&next.params),
        root.next_w_len,
        RelationMatrixRowLayout::WithDBlock,
        Some(binding),
    )?;
    schedule.total_bytes = schedule
        .folds
        .iter()
        .map(|fold| fold.level_bytes)
        .try_fold(0usize, |total, bytes| total.checked_add(bytes))
        .and_then(|total| total.checked_add(schedule.terminal.terminal_bytes))
        .ok_or_else(|| AkitaError::InvalidSetup("mixed-role proof size overflow".into()))?;
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

    fn get_params_for_prove(opening_batch: &OpeningClaimsLayout) -> Result<Schedule, AkitaError> {
        mixed_role_schedule(
            opening_batch.max_num_vars(),
            opening_batch.num_total_polynomials(),
        )
    }
}

#[test]
fn direct_setup_mixed_role_root_proves_and_verifies() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let opening_batch = OpeningClaimsLayout::new(NUM_VARS, 1).expect("opening batch");
        let schedule = mixed_role_schedule(NUM_VARS, 1).expect("mixed-role schedule");
        let root = &schedule.folds[0].params;
        assert_eq!(root.role_dims(), ROLE_DIMS);
        assert_eq!(
            root.setup_contribution_mode,
            akita_types::SetupContributionMode::Direct
        );
        assert_eq!(ROLE_DIMS.common_relation_coeff_count(), 32);
        assert_eq!(
            ROLE_DIMS.common_relation_witness_coeff_count(schedule.folds[1].params.d_a()),
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

        let poly_refs = [&poly];
        let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
        let proof = Scheme::batched_prove(
            &setup,
            prove_input(&point, &poly_refs, &commitment, hint),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("mixed-role prove");

        let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
        Scheme::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&point, &[opening], &commitment),
            BasisMode::Lagrange,
        )
        .expect("mixed-role verify");
    });
}
