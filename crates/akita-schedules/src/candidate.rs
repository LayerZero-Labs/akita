//! Canonical scalar root candidate construction shared by offline planning and
//! standalone precommit selection.

use crate::runtime::{optimize_fold_challenge_shape, PlannerPolicy};
use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::sis::{
    decomposed_s_block_ring_count, decomposed_t_ring_count, decomposed_w_ring_count,
    min_secure_rank, num_digits_inner, num_digits_open, projected_role_ring_count,
    rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm, HonestFoldPolicy,
    HonestFoldPolicySpec, HonestFoldSizingQuery, InnerCommitMatrixParams, OpenCommitMatrixParams,
    OuterCommitMatrixParams, SisTableKey,
};
use akita_types::{CommitmentRingDims, CommittedGroupParams, DecompositionParams, SisMatrixRole};

/// Ring-dimension choice for one planner candidate.
#[derive(Clone, Copy, Debug)]
pub enum RingDimensionCandidate<'a> {
    /// Use one exact A/B/D tuple.
    Fixed(CommitmentRingDims),
    /// Search has fixed A; derive B/D by minimum rank within the supplied
    /// role domains and the previous level's monotonic ceiling.
    Adaptive {
        inner: usize,
        outer_dimensions: &'a [usize],
        opening_dimensions: &'a [usize],
        ceiling: CommitmentRingDims,
    },
}

impl RingDimensionCandidate<'_> {
    /// Return the already-selected A dimension.
    pub fn inner(self) -> usize {
        match self {
            Self::Fixed(dimensions) => dimensions.d_a(),
            Self::Adaptive { inner, .. } => inner,
        }
    }

    fn validate(self) -> Result<(), AkitaError> {
        match self {
            Self::Fixed(dimensions) => dimensions.validate_role_projection(),
            Self::Adaptive { inner, ceiling, .. } => {
                ceiling.validate_role_projection()?;
                if inner == 0 || !inner.is_power_of_two() || inner > ceiling.d_a() {
                    return Err(AkitaError::InvalidSetup(format!(
                        "adaptive A dimension D{inner} is invalid under D{} ceiling",
                        ceiling.d_a()
                    )));
                }
                Ok(())
            }
        }
    }

    /// Select the B or D dimension for this candidate.
    pub fn collision_role_price(
        self,
        policy: &PlannerPolicy,
        role: SisMatrixRole,
        native_width: usize,
        log_basis: u32,
    ) -> Option<(SisTableKey, usize)> {
        let carrier_dimension = self.inner();
        match self {
            Self::Fixed(dimensions) => {
                let role_dimension = match role {
                    SisMatrixRole::Outer => dimensions.d_b(),
                    SisMatrixRole::Open => dimensions.d_d(),
                    SisMatrixRole::Inner => return None,
                };
                projected_collision_role_price(
                    policy,
                    role,
                    carrier_dimension,
                    role_dimension,
                    native_width,
                    log_basis,
                )
            }
            Self::Adaptive {
                outer_dimensions,
                opening_dimensions,
                ceiling,
                ..
            } => {
                let (dimensions, role_ceiling) = match role {
                    SisMatrixRole::Outer => (outer_dimensions, ceiling.d_b()),
                    SisMatrixRole::Open => (opening_dimensions, ceiling.d_d()),
                    SisMatrixRole::Inner => return None,
                };
                let mut best: Option<(usize, usize, SisTableKey, usize)> = None;
                for &role_dimension in dimensions {
                    if role_dimension > carrier_dimension || role_dimension > role_ceiling {
                        continue;
                    }
                    let Some((key, width)) = projected_collision_role_price(
                        policy,
                        role,
                        carrier_dimension,
                        role_dimension,
                        native_width,
                        log_basis,
                    ) else {
                        continue;
                    };
                    let Ok(width_u64) = u64::try_from(width) else {
                        continue;
                    };
                    let Some(rank) = min_secure_rank(key, width_u64) else {
                        continue;
                    };
                    if best.as_ref().is_none_or(|(best_rank, best_d, _, _)| {
                        (rank, role_dimension) < (*best_rank, *best_d)
                    }) {
                        best = Some((rank, role_dimension, key, width));
                    }
                    if rank == 1 {
                        break;
                    }
                }
                best.map(|(_, _, key, width)| (key, width))
            }
        }
    }
}

/// Build one scalar root-fold candidate for an explicit basis and split.
///
/// `Ok(None)` is the canonical candidate-infeasibility signal used by schedule
/// optimization, setup-capacity certification, and standalone precommit
/// selection.
#[allow(clippy::too_many_arguments)]
pub fn scalar_root_fold_level_params_candidate(
    policy: &PlannerPolicy,
    ring_challenge_cfg: &SparseChallengeConfig,
    dimensions: RingDimensionCandidate<'_>,
    num_vars: usize,
    num_claims: usize,
    log_basis: u32,
    block_index_bits: usize,
    requested_fold_shape: TensorChallengeShape,
    honest_fold_policy: HonestFoldPolicySpec,
) -> Result<Option<CommittedGroupParams>, AkitaError> {
    dimensions.validate()?;
    let d_a = dimensions.inner();
    let alpha = (d_a as u32).trailing_zeros() as usize;
    let reduced_vars = num_vars.saturating_sub(alpha);
    if reduced_vars == 0 || block_index_bits >= reduced_vars {
        return Ok(None);
    }
    let num_live_blocks = 1usize.checked_shl(block_index_bits as u32).ok_or_else(|| {
        AkitaError::InvalidSetup("root candidate num_live_blocks overflow".to_string())
    })?;
    let root_num_chunks = policy.chunks_at_level(0);
    if num_live_blocks < root_num_chunks {
        return Ok(None);
    }
    let fold_challenge_shape =
        optimize_fold_challenge_shape(requested_fold_shape, num_live_blocks)?;
    let position_index_bits = reduced_vars - block_index_bits;
    let num_positions_per_block =
        1usize
            .checked_shl(position_index_bits as u32)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("root candidate position count overflow".to_string())
            })?;
    let num_live_ring_elements_per_claim = num_live_blocks
        .checked_mul(num_positions_per_block)
        .ok_or_else(|| AkitaError::InvalidSetup("root candidate source length overflow".into()))?;
    let level_decomp = DecompositionParams {
        log_basis,
        ..policy.decomposition
    };
    let witness_decomp = DecompositionParams {
        log_basis,
        ..policy.decomposition
    };
    let num_digits_inner = num_digits_inner(witness_decomp, true);
    let num_digits_open = num_digits_open(level_decomp);
    let Some(width_s) = decomposed_s_block_ring_count(num_positions_per_block, num_digits_inner)
    else {
        return Ok(None);
    };
    let Some(num_fold_coeffs) = width_s
        .checked_mul(d_a)
        .and_then(|count| count.checked_mul(root_num_chunks))
    else {
        return Ok(None);
    };
    let Ok(num_digits_fold) = honest_fold_policy.num_digits_fold(HonestFoldSizingQuery {
        ring_dimension: d_a,
        num_claims,
        num_live_blocks,
        num_chunks: root_num_chunks,
        num_fold_coeffs,
        log_basis,
        challenge_config: ring_challenge_cfg,
        challenge_shape: fold_challenge_shape,
    }) else {
        return Ok(None);
    };
    let Some(norm_s) = rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_table_digest,
        policy.sis_modulus_profile,
        d_a,
        log_basis,
        ring_challenge_cfg,
        fold_challenge_shape,
        num_digits_fold,
        policy.ring_subfield_norm_bound,
    ) else {
        return Ok(None);
    };
    let Ok(inner_commit_matrix) = InnerCommitMatrixParams::try_new_with_min_rank(
        sis_key_at_dimension(policy, SisMatrixRole::Inner, d_a, norm_s),
        width_s,
    ) else {
        return Ok(None);
    };
    let Some(native_width_t) = decomposed_t_ring_count(
        inner_commit_matrix.output_rank(),
        num_digits_open,
        num_live_blocks,
        num_claims,
    ) else {
        return Ok(None);
    };
    let Some((outer_key, width_t)) =
        dimensions.collision_role_price(policy, SisMatrixRole::Outer, native_width_t, log_basis)
    else {
        return Ok(None);
    };
    let Ok(outer_commit_matrix) =
        OuterCommitMatrixParams::try_new_with_min_rank(outer_key, width_t)
    else {
        return Ok(None);
    };
    let Some(native_width_w) =
        decomposed_w_ring_count(num_digits_open, num_live_blocks, num_claims)
    else {
        return Ok(None);
    };
    let Some((open_key, width_w)) =
        dimensions.collision_role_price(policy, SisMatrixRole::Open, native_width_w, log_basis)
    else {
        return Ok(None);
    };
    let Ok(open_commit_matrix) = OpenCommitMatrixParams::try_new_with_min_rank(open_key, width_w)
    else {
        return Ok(None);
    };
    Ok(Some(CommittedGroupParams {
        log_basis_inner: witness_decomp.log_basis,
        log_basis_outer: log_basis,
        log_basis_open: log_basis,
        inner_commit_matrix,
        outer_commit_matrix,
        open_commit_matrix,
        num_live_ring_elements_per_claim,
        num_positions_per_block,
        num_live_blocks,
        fold_challenge_config: *ring_challenge_cfg,
        fold_challenge_shape,
        num_digits_inner,
        num_digits_outer: num_digits_open,
        num_digits_open,
        num_digits_fold,
        witness_chunk: policy.witness_chunk_for_level(0),
        precommitted_groups: Vec::new(),
        setup_prefix: None,
    }))
}

/// Construct the canonical SIS-table key for one role and ring dimension.
pub fn sis_key_at_dimension(
    policy: &PlannerPolicy,
    role: SisMatrixRole,
    ring_dimension: usize,
    coeff_linf_bound: u128,
) -> SisTableKey {
    SisTableKey {
        policy: policy.sis_security_policy,
        table_digest: policy.sis_table_digest,
        modulus_profile: policy.sis_modulus_profile,
        role,
        ring_dimension: ring_dimension as u32,
        coeff_linf_bound,
    }
}

/// Price one projected B/D collision role using canonical physical width and
/// coefficient bounds.
pub fn projected_collision_role_price(
    policy: &PlannerPolicy,
    role: SisMatrixRole,
    carrier_dimension: usize,
    role_dimension: usize,
    native_width: usize,
    log_basis: u32,
) -> Option<(SisTableKey, usize)> {
    if role == SisMatrixRole::Inner
        || role_dimension == 0
        || !carrier_dimension.is_multiple_of(role_dimension)
    {
        return None;
    }
    let coeff_linf_bound = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        role,
        role_dimension,
        log_basis,
    )?;
    let physical_width =
        projected_role_ring_count(carrier_dimension, role_dimension, native_width)?;
    Some((
        sis_key_at_dimension(policy, role, role_dimension, coeff_linf_bound),
        physical_width,
    ))
}
