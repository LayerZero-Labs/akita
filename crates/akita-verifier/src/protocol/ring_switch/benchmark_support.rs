//! Feature-gated fixtures for benchmarking the production relation evaluator.

use super::{FlatRelationContext, RelationMatrixEvaluator, RelationMatrixGroupEvaluator};
use akita_algebra::eq_poly::EqPolynomial;
use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_types::{
    gadget_row_scalars, r_decomp_levels, AkitaExpandedSetup, AkitaSetupDescriptor,
    CommitmentRingDims, CommittedGroupParams, FlatMatrix, InnerCommitMatrixParams,
    OpenCommitMatrixParams, OpeningClaimsLayout, OuterCommitMatrixParams, PreparedRelationAddress,
    RelationAddressGeometry, SetupContributionPlan, SisModulusProfileId, WitnessLayout,
};
use jolt_field::{CanonicalEncoding, Prime128OffsetA7F7};
use std::sync::Arc;

/// Inputs for one exact production relation-evaluator benchmark cell.
pub struct RelationEvaluatorBenchmarkCase {
    /// Prepared verifier evaluator.
    pub evaluator: RelationMatrixEvaluator<Prime128OffsetA7F7>,
    /// Complete coefficient/lane/column evaluation point.
    pub point: Vec<Prime128OffsetA7F7>,
    /// Expanded public setup scanned by direct evaluation.
    pub setup: AkitaExpandedSetup<Prime128OffsetA7F7>,
    /// Ring-switch alpha challenge.
    pub alpha: Prime128OffsetA7F7,
}

/// Build one U/L/M benchmark cell with identical semantic workload dimensions.
///
/// # Errors
///
/// Returns an error if the requested role or outgoing geometry is invalid.
pub fn relation_evaluator_benchmark_case(
    role_dims: CommitmentRingDims,
    outgoing_ring_dimension: usize,
) -> Result<RelationEvaluatorBenchmarkCase, AkitaError> {
    relation_evaluator_benchmark_case_with_chunks(role_dims, outgoing_ring_dimension, 1)
}

/// Build one U/L/M benchmark cell with a selected physical chunk count.
///
/// # Errors
///
/// Returns an error if the requested role, outgoing, or chunk geometry is
/// invalid.
pub fn relation_evaluator_benchmark_case_with_chunks(
    role_dims: CommitmentRingDims,
    outgoing_ring_dimension: usize,
    witness_chunks: usize,
) -> Result<RelationEvaluatorBenchmarkCase, AkitaError> {
    type F = Prime128OffsetA7F7;
    const A_D: usize = 128;
    const NUM_CLAIMS: usize = 2;
    const NUM_LIVE_BLOCKS: usize = 64;
    const NUM_POSITIONS_PER_BLOCK: usize = 8;
    const N_A: usize = 2;
    const N_B: usize = 2;
    const N_D: usize = 2;
    const DEPTH_COMMIT: usize = 2;
    const DEPTH_OPEN: usize = 2;
    const LOG_BASIS: u32 = 4;

    if role_dims.d_a() != A_D {
        return Err(AkitaError::InvalidSetup(
            "relation benchmark requires A dimension 128".into(),
        ));
    }
    let mut level_params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        A_D,
        LOG_BASIS,
        N_A,
        N_B,
        N_D,
        SparseChallengeConfig::production_for_ring_dim(A_D)
            .ok_or_else(|| AkitaError::InvalidSetup("missing benchmark fold challenge".into()))?,
    )
    .with_decomp(
        NUM_POSITIONS_PER_BLOCK,
        NUM_LIVE_BLOCKS * NUM_POSITIONS_PER_BLOCK,
        DEPTH_COMMIT,
        DEPTH_OPEN,
        DEPTH_OPEN,
    )?;
    let inner = &level_params.inner.matrix;
    level_params.inner.matrix = InnerCommitMatrixParams::new_unchecked(
        inner.security_policy(),
        inner.sis_table_key().table_digest,
        inner.sis_modulus_profile(),
        N_A,
        NUM_POSITIONS_PER_BLOCK * DEPTH_COMMIT,
        inner.coeff_linf_bound().max(1),
        role_dims.d_a(),
    );
    let outer = &level_params.outer.matrix;
    level_params.outer.matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        N_B,
        NUM_CLAIMS * N_A * DEPTH_COMMIT * NUM_LIVE_BLOCKS * (role_dims.d_a() / role_dims.d_b()),
        outer.coeff_linf_bound().max(1),
        role_dims.d_b(),
    );
    let open = &level_params.open.matrix;
    level_params.open.matrix = OpenCommitMatrixParams::new_unchecked(
        open.security_policy(),
        open.sis_table_key().table_digest,
        open.sis_modulus_profile(),
        N_D,
        NUM_CLAIMS * DEPTH_OPEN * NUM_LIVE_BLOCKS * (role_dims.d_a() / role_dims.d_d()),
        open.coeff_linf_bound().max(1),
        role_dims.d_d(),
    );

    let opening_batch = OpeningClaimsLayout::new(0, NUM_CLAIMS)?;
    let rows = level_params.relation_matrix_row_count(opening_batch.num_groups())?;
    let quotient_depth = r_decomp_levels::<F>(LOG_BASIS);
    let witness_layout = WitnessLayout::new(
        &level_params,
        &opening_batch,
        witness_chunks,
        quotient_depth,
    )?;
    let relation_address_geometry = RelationAddressGeometry::new(
        role_dims,
        outgoing_ring_dimension,
        witness_layout.live_coeff_len(),
    )?;
    let alpha = scalar(3);
    let point = (0..relation_address_geometry.relation_point_variable_count())
        .map(|index| scalar(101 + index as u128))
        .collect::<Vec<_>>();
    let tau1_bits = rows.next_power_of_two().trailing_zeros() as usize;
    let tau1 = (0..tau1_bits)
        .map(|index| scalar(211 + index as u128))
        .collect::<Vec<_>>();
    let eq_tau1: Arc<[F]> = EqPolynomial::evals_prefix(&tau1, rows)?.into();
    let depth_fold = level_params.num_digits_fold();
    let evaluator = RelationMatrixEvaluator {
        relation_address_geometry,
        groups: vec![RelationMatrixGroupEvaluator {
            c_alphas: (0..NUM_CLAIMS * NUM_LIVE_BLOCKS)
                .map(|index| scalar(307 + index as u128))
                .collect(),
            opening_a_evals: (0..NUM_POSITIONS_PER_BLOCK)
                .map(|index| scalar(401 + index as u128))
                .collect(),
            group_id: 0,
            num_claims: NUM_CLAIMS,
            depth_fold,
            a_row_start: 1,
            b_row_start: 1 + N_A,
        }],
        log_basis: LOG_BASIS,
        eq_tau1,
        flat_context: Some(FlatRelationContext {
            level_params,
            opening_batch,
            witness_layout: Arc::new(witness_layout),
            extension_degree: 1,
        }),
        setup_plan_cache: Default::default(),
    };
    let coefficient_bits = relation_address_geometry.relation_coefficient_variable_count();
    let relation_address = PreparedRelationAddress::new(
        point
            .get(coefficient_bits..)
            .ok_or(AkitaError::InvalidProof)?,
    )?;
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, LOG_BASIS);
    let mut plan: SetupContributionPlan<F> =
        evaluator.setup_contribution_plan::<F>(relation_address, Some(&fold_gadget))?;
    plan.materialize_direct_scan(alpha)?;
    let setup_field_elements = plan.projection_geometry().natural_field_len();
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: NUM_CLAIMS,
            num_field_elements: setup_field_elements,
            setup_seed: [7; 32].into(),
        },
        FlatMatrix::from_flat_data(
            (0..setup_field_elements)
                .map(|index| scalar(503 + index as u128))
                .collect(),
        ),
    );

    Ok(RelationEvaluatorBenchmarkCase {
        evaluator,
        point,
        setup,
        alpha,
    })
}

fn scalar(value: u128) -> Prime128OffsetA7F7 {
    Prime128OffsetA7F7::from_u128_checked(value).expect("benchmark scalar must be canonical")
}
