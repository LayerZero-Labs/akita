//! Feature-gated fixtures for benchmarking the production relation evaluator.

use super::{prepare_relation_matrix_evaluator, RelationMatrixEvaluator, RingSwitchReplay};
use akita_challenges::{Challenges, SparseChallenge, SparseChallengeConfig};
use akita_error::AkitaError;
use akita_types::{
    relation_rhs_coeff_len, AkitaExpandedSetup, AkitaSetupDescriptor, ChunkedWitnessCfg,
    CommitmentRingDims, CommittedGroupParams, FlatMatrix, InnerCommitMatrixParams,
    OpenCommitMatrixParams, OpeningClaimsLayout, OuterCommitMatrixParams, RelationWitnessGeometry,
    RingMultiplierOpeningPoint, RingOpeningPoint, RingRelationGroupOpening, RingRelationInstance,
    RingRelationMode, RingVec, SisModulusProfileId,
};
use jolt_field::{CanonicalEncoding, Prime128OffsetA7F7};

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

/// One production-prepared relation evaluation used for isolated phase timing.
pub struct PreparedRelationEvaluatorBenchmark<'a> {
    prepared: super::relation_evaluation::PreparedDirectRelation<'a, Prime128OffsetA7F7>,
}

impl RelationEvaluatorBenchmarkCase {
    /// Prepare the production relation point and setup-contribution plan.
    pub fn prepare(&self) -> Result<PreparedRelationEvaluatorBenchmark<'_>, AkitaError> {
        let mut prepared = super::relation_evaluation::PreparedDirectRelation::prepare::<
            Prime128OffsetA7F7,
        >(&self.evaluator, &self.point, self.alpha)?;
        prepared.materialize_setup()?;
        Ok(PreparedRelationEvaluatorBenchmark { prepared })
    }
}

impl PreparedRelationEvaluatorBenchmark<'_> {
    /// Materialize setup weights and scan the active setup exactly once.
    pub fn setup_scan(
        self,
        setup: &AkitaExpandedSetup<Prime128OffsetA7F7>,
    ) -> Result<Prime128OffsetA7F7, AkitaError> {
        self.prepared.evaluate_setup::<Prime128OffsetA7F7>(setup)
    }

    /// Materialize setup weights, then evaluate only the non-setup relation weight.
    pub fn relation_weight(self) -> Result<Prime128OffsetA7F7, AkitaError> {
        self.prepared
            .evaluate_relation_weight::<Prime128OffsetA7F7>()
    }

    /// Evaluate the structured group contribution only.
    pub fn structured_groups(self) -> Result<Prime128OffsetA7F7, AkitaError> {
        self.prepared.evaluate_structured::<Prime128OffsetA7F7>()
    }

    /// Evaluate the quotient-tail contribution, identically zero in reduced mode.
    pub fn quotient_tail(self) -> Result<Prime128OffsetA7F7, AkitaError> {
        self.prepared.evaluate_quotient_tail::<Prime128OffsetA7F7>()
    }
}

/// Build one U/L/M benchmark cell with identical semantic workload dimensions.
///
/// # Errors
///
/// Returns an error if the requested role or outgoing geometry is invalid.
pub fn relation_evaluator_benchmark_case(
    mode: RingRelationMode,
    role_dims: CommitmentRingDims,
    outgoing_ring_dimension: usize,
) -> Result<RelationEvaluatorBenchmarkCase, AkitaError> {
    relation_evaluator_benchmark_case_with_chunks(mode, role_dims, outgoing_ring_dimension, 1)
}

/// Build one U/L/M benchmark cell with a selected physical chunk count.
///
/// # Errors
///
/// Returns an error if the requested role, outgoing, or chunk geometry is
/// invalid.
pub fn relation_evaluator_benchmark_case_with_chunks(
    mode: RingRelationMode,
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
    level_params.ring_relation_mode = mode;
    let inner = level_params.inner().matrix;
    let inner_table_digest = inner
        .sis_table_key()
        .ok_or_else(|| AkitaError::InvalidSetup("missing benchmark inner SIS key".into()))?
        .table_digest;
    level_params.own_group_mut().profile.inner.matrix = InnerCommitMatrixParams::new_unchecked(
        inner.security_policy(),
        inner_table_digest,
        inner.sis_modulus_profile(),
        N_A,
        NUM_POSITIONS_PER_BLOCK * DEPTH_COMMIT,
        inner.coeff_linf_bound().unwrap_or(1).max(1),
        role_dims.d_a(),
    );
    let outer = level_params.outer().matrix;
    level_params.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        N_B,
        NUM_CLAIMS * N_A * DEPTH_COMMIT * NUM_LIVE_BLOCKS * (role_dims.d_a() / role_dims.d_b()),
        outer.coeff_linf_bound().max(1),
        role_dims.d_b(),
    );
    let open = level_params.open().matrix;
    level_params.open_matrix = OpenCommitMatrixParams::new_unchecked(
        open.security_policy(),
        open.sis_table_key().table_digest,
        open.sis_modulus_profile(),
        N_D,
        NUM_CLAIMS * DEPTH_OPEN * NUM_LIVE_BLOCKS * (role_dims.d_a() / role_dims.d_d()),
        open.coeff_linf_bound().max(1),
        role_dims.d_d(),
    );

    level_params.witness_chunk = if witness_chunks == 1 {
        ChunkedWitnessCfg::default_non_chunked()
    } else {
        ChunkedWitnessCfg {
            num_chunks: witness_chunks,
            num_activated_levels: 1,
        }
    };
    level_params.witness_chunk.validate()?;

    let opening_batch = OpeningClaimsLayout::new(0, NUM_CLAIMS)?;
    let challenges = Challenges::from_sparse(
        (0..NUM_CLAIMS * NUM_LIVE_BLOCKS)
            .map(|index| SparseChallenge {
                positions: vec![(index % A_D) as u32].into(),
                coeffs: vec![1].into(),
            })
            .collect(),
        NUM_LIVE_BLOCKS,
        NUM_CLAIMS,
    )?;
    let opening = RingMultiplierOpeningPoint::from_base(&RingOpeningPoint {
        position_weights: (0..NUM_POSITIONS_PER_BLOCK)
            .map(|index| scalar(401 + index as u128))
            .collect(),
        live_block_weights: vec![F::default(); NUM_LIVE_BLOCKS],
    });
    let gamma = (0..NUM_CLAIMS)
        .map(|index| scalar(307 + index as u128))
        .collect::<Vec<_>>();
    let row_coefficient_rings = gamma
        .iter()
        .flat_map(|coefficient| {
            let mut ring = vec![F::default(); A_D];
            ring[0] = *coefficient;
            ring
        })
        .collect();
    let relation_geometry =
        RelationWitnessGeometry::for_evaluation_trace_execution(&level_params, &opening_batch)?;
    let relation = RingRelationInstance::new(
        vec![RingRelationGroupOpening::evaluation_trace(
            challenges, opening,
        )],
        1,
        opening_batch,
        gamma,
        RingVec::from_coeffs_with_ring_dim(row_coefficient_rings, A_D)?,
        RingVec::from_coeffs(vec![
            F::default();
            relation_rhs_coeff_len(relation_geometry.rhs_layout())?
        ]),
        RingVec::from_coeffs(Vec::new()),
        role_dims,
    )?;
    let witness_layout = relation.segment_layout(&level_params, None)?;
    if outgoing_ring_dimension == 0 {
        return Err(AkitaError::InvalidSetup(
            "relation benchmark outgoing ring dimension must be nonzero".into(),
        ));
    }
    let opening_source_len = witness_layout
        .live_coeff_len()
        .div_ceil(outgoing_ring_dimension);
    let placeholder_setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: NUM_CLAIMS,
            num_field_elements: 1,
            setup_seed: [7; 32].into(),
        },
        FlatMatrix::from_flat_data(vec![F::default()]),
    );
    let row_coefficients = (0..NUM_CLAIMS)
        .map(|index| scalar(601 + index as u128))
        .collect::<Vec<_>>();
    let rows = level_params.relation_matrix_row_count(relation.opening_batch().num_groups())?;
    let tau1 = (0..rows.next_power_of_two().trailing_zeros() as usize)
        .map(|index| scalar(211 + index as u128))
        .collect::<Vec<_>>();
    let alpha = scalar(3);
    let replay = RingSwitchReplay {
        setup: &placeholder_setup,
        relation: &relation,
        row_coefficients: &row_coefficients,
        lp: &level_params,
        opening_source_len,
        opening_ring_dim: outgoing_ring_dimension,
    };
    let evaluator = prepare_relation_matrix_evaluator::<F, F>(&replay, alpha, &tau1, None)?;
    let point = (0..evaluator
        .relation_address_geometry
        .relation_point_variable_count())
        .map(|index| scalar(101 + index as u128))
        .collect::<Vec<_>>();
    let mut prepared = super::relation_evaluation::PreparedDirectRelation::prepare::<F>(
        &evaluator, &point, alpha,
    )?;
    prepared.materialize_setup()?;
    let setup_field_elements = prepared.setup_field_len();
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
