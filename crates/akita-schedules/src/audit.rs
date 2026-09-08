//! Canonical security audit for one fully expanded schedule row.

use akita_error::AkitaError;
use akita_types::sis::{
    num_digits_inner, num_digits_open, rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm,
    InnerCommitMatrixParams, InnerCommitSecurityRoute, OpenCommitMatrixParams,
    OuterCommitMatrixParams, SisMatrixRole, SisTableKey,
};
#[cfg(test)]
use akita_types::TerminalResponseShape;
use akita_types::{
    shared_d_digit_log_basis, validate_role_dims, CommitmentSliceGeometry,
    CommittedGroupBatchProfile, CommittedGroupParams, DecompositionParams, FoldSchedule,
    GroupOpenPhaseParams, TerminalFoldParams,
};

use crate::candidate::{selective_l2_inner_matrix, SelectiveL2CandidateGeometry};
use crate::runtime::validate_policy;
use crate::traversal::{visit_schedule_groups, ScheduleGroup, ScheduleGroupPosition};
use crate::PlannerPolicy;

fn invalid(label: &str, detail: &str) -> AkitaError {
    AkitaError::InvalidSetup(format!("{label}: {detail}"))
}

fn audit_sis_key(
    label: &str,
    key: SisTableKey,
    expected_role: SisMatrixRole,
    policy: &PlannerPolicy,
) -> Result<(), AkitaError> {
    if key.policy != policy.sis_security_policy
        || key.table_digest != policy.sis_table_digest
        || key.modulus_profile != policy.sis_modulus_profile
        || key.role != expected_role
    {
        return Err(invalid(
            label,
            "matrix SIS policy, table, modulus profile, or role disagrees with the catalog policy",
        ));
    }
    Ok(())
}

fn audit_inner_matrix(
    label: &str,
    matrix: &InnerCommitMatrixParams,
    policy: &PlannerPolicy,
) -> Result<(), AkitaError> {
    matrix.validate()?;
    match matrix.security_route() {
        InnerCommitSecurityRoute::Linf(key) => {
            audit_sis_key(label, key, SisMatrixRole::Inner, policy)
        }
        InnerCommitSecurityRoute::L2 { table_key, .. } => {
            if table_key.policy != policy.sis_security_policy
                || table_key.table_digest != policy.sis_l2_table_digest
                || table_key.modulus_profile != policy.sis_modulus_profile
            {
                return Err(invalid(
                    label,
                    "A matrix L2 policy, table, or modulus profile disagrees with catalog policy",
                ));
            }
            Ok(())
        }
    }
}

fn audit_outer_matrix(
    label: &str,
    matrix: &OuterCommitMatrixParams,
    policy: &PlannerPolicy,
) -> Result<(), AkitaError> {
    matrix.validate()?;
    audit_sis_key(label, matrix.sis_table_key(), SisMatrixRole::Outer, policy)
}

fn audit_open_matrix(
    label: &str,
    matrix: &OpenCommitMatrixParams,
    policy: &PlannerPolicy,
) -> Result<(), AkitaError> {
    matrix.validate()?;
    audit_sis_key(label, matrix.sis_table_key(), SisMatrixRole::Open, policy)
}

fn audit_bound(label: &str, declared: u128, required: Option<u128>) -> Result<(), AkitaError> {
    let required = required.ok_or_else(|| invalid(label, "accepted envelope has no SIS row"))?;
    if declared < required {
        return Err(invalid(
            label,
            &format!("declared coefficient bound {declared} is below required bound {required}"),
        ));
    }
    Ok(())
}

fn audit_frozen_group(
    label: &str,
    params: &GroupOpenPhaseParams,
    num_response_chunks: usize,
    policy: &PlannerPolicy,
) -> Result<(), AkitaError> {
    params.validate()?;
    audit_inner_matrix(label, &params.profile.inner.matrix, policy)?;
    audit_outer_matrix(label, &params.profile.outer.matrix, policy)?;

    let expected_open_digits = num_digits_open(DecompositionParams {
        log_basis: params.opening.log_basis_open,
        ..policy.decomposition
    });
    if params.opening.num_digits_open != expected_open_digits {
        return Err(invalid(
            label,
            "opening digit depth is not canonical for the field and basis",
        ));
    }

    let declared_a_bound = params
        .profile
        .inner
        .matrix
        .coeff_linf_bound()
        .ok_or_else(|| invalid(label, "frozen groups cannot use an L2 A security route"))?;
    audit_bound(
        label,
        declared_a_bound,
        rounded_up_role_a_inf_norm(
            policy.sis_security_policy,
            policy.sis_table_digest,
            policy.sis_modulus_profile,
            params.profile.inner.matrix.ring_dimension(),
            params.opening.log_basis_open,
            &params.opening.fold_challenge_config,
            params.opening.num_digits_fold,
            num_response_chunks,
        ),
    )?;
    audit_bound(
        label,
        params.profile.outer.matrix.coeff_linf_bound(),
        rounded_up_collision_inf_norm(
            policy.sis_security_policy,
            policy.sis_modulus_profile,
            SisMatrixRole::Outer,
            params.profile.outer.matrix.ring_dimension(),
            params.opening.log_basis_open,
        ),
    )
}

fn expected_d_width(
    label: &str,
    params: &CommittedGroupParams,
    num_claims: usize,
    extension_degree: usize,
) -> Result<usize, AkitaError> {
    let dims = params.role_dims();
    let mut width = akita_types::opening_d_segment_width(
        params.opening_method(),
        extension_degree,
        dims.d_a(),
        dims.d_d(),
        params.open().digits.num_digits,
        params.blocks().live_blocks,
        num_claims,
    )
    .map_err(|_| invalid(label, "main D width is incompatible with opening geometry"))?;

    for group in params.precommitted_groups() {
        width = width
            .checked_add(group.d_segment_width(extension_degree, dims.d_d())?)
            .ok_or_else(|| invalid(label, "precommitted D width overflow"))?;
    }
    if let Some(prefix) = params.setup_prefix() {
        width = width
            .checked_add(prefix.d_segment_width(extension_degree, dims.d_d())?)
            .ok_or_else(|| invalid(label, "setup-prefix D width overflow"))?;
    }
    Ok(width)
}

fn audit_committed_params(
    label: &str,
    params: &CommittedGroupParams,
    num_claims: usize,
    fold_level: usize,
    policy: &PlannerPolicy,
) -> Result<(), AkitaError> {
    if num_claims == 0 {
        return Err(invalid(label, "fold claim count must be positive"));
    }
    params.validate_block_geometry()?;
    params.witness_chunk.validate()?;
    validate_role_dims(params.role_dims())?;
    params
        .fold_challenge_config()
        .validate_for_ring_dim(params.d_a())
        .map_err(|message| invalid(label, message))?;
    audit_inner_matrix(label, &params.inner().matrix, policy)?;
    audit_outer_matrix(label, &params.outer().matrix, policy)?;
    audit_open_matrix(label, &params.open().matrix, policy)?;

    let expected_outer_digits = num_digits_open(DecompositionParams {
        log_basis: params.outer().digits.log_basis,
        ..policy.decomposition
    });
    let expected_open_digits = num_digits_open(DecompositionParams {
        log_basis: params.open().digits.log_basis,
        ..policy.decomposition
    });
    // An artifact stores its own `num_digits_inner` verbatim, so pin it to the
    // depth the declared committed-source bound
    // demands at this level's selected A basis. At the root that bound is
    // `log_commit_bound` — the one place a bounded source differs from a
    // full-field one — and at a recursive level it collapses to the level's own
    // `log_basis`. Precommitted groups are audited separately: they are frozen
    // under a possibly different producer bound and are not covered here.
    let expected_inner_digits = num_digits_inner(
        DecompositionParams {
            log_basis: params.inner().digits.log_basis,
            ..policy.decomposition
        },
        fold_level == 0,
    );
    if params.inner().digits.num_digits != expected_inner_digits
        || params.num_digits_fold() == 0
        || params.outer().digits.num_digits != expected_outer_digits
        || params.open().digits.num_digits != expected_open_digits
    {
        return Err(invalid(label, "digit depths are missing or noncanonical"));
    }

    let dims = params.role_dims();
    let expected_a_width = params
        .blocks()
        .positions_per_block
        .checked_mul(params.inner().digits.num_digits)
        .ok_or_else(|| invalid(label, "A width overflow"))?;
    let expected_b_width = CommitmentSliceGeometry::try_new(
        params.outer_slice_count(),
        params.blocks().live_blocks,
        num_claims,
        params.inner().matrix.output_rank(),
        params.outer().digits.num_digits,
        dims.d_a(),
        dims.d_b(),
    )?
    .physical_input_width();
    let expected_d_width = expected_d_width(label, params, num_claims, policy.claim_ext_degree)?;
    if params.inner().matrix.input_width() != expected_a_width
        || params.outer().matrix.input_width() != expected_b_width
        || params.open().matrix.input_width() != expected_d_width
    {
        return Err(invalid(
            label,
            "A, B, or D width disagrees with the accepted digit geometry",
        ));
    }

    match params.inner().matrix.security_route() {
        InnerCommitSecurityRoute::Linf(key) => audit_bound(
            label,
            key.coeff_linf_bound,
            rounded_up_role_a_inf_norm(
                policy.sis_security_policy,
                policy.sis_table_digest,
                policy.sis_modulus_profile,
                dims.d_a(),
                params.open().digits.log_basis,
                &params.fold_challenge_config(),
                params.num_digits_fold(),
                params.witness_chunk.num_chunks,
            ),
        )?,
        InnerCommitSecurityRoute::L2 {
            response_l2_sq_cap, ..
        } => {
            let fold_basis = 1usize
                .checked_shl(params.open().digits.log_basis)
                .ok_or_else(|| invalid(label, "L2 balanced digit basis overflow"))?;
            let expected = selective_l2_inner_matrix(
                policy,
                SelectiveL2CandidateGeometry {
                    fold_level,
                    num_claims,
                    num_chunks: params.witness_chunk.num_chunks,
                    inner_width: expected_a_width,
                    ring_dimension: dims.d_a(),
                    fold_basis,
                    fold_digit_count: params.num_digits_fold(),
                    fold_challenge_config: &params.fold_challenge_config(),
                    response_l2_sq_cap: Some(response_l2_sq_cap),
                    norm_proof_shape: None,
                },
            )?
            .ok_or_else(|| {
                invalid(
                    label,
                    "L2 route is not admitted by the frozen suffix response cap",
                )
            })?;
            if params.inner().matrix != expected {
                return Err(invalid(
                    label,
                    "L2 A matrix disagrees with canonical cap, proof shape, table, or rank",
                ));
            }
        }
    }
    audit_bound(
        label,
        params.outer().matrix.coeff_linf_bound(),
        rounded_up_collision_inf_norm(
            policy.sis_security_policy,
            policy.sis_modulus_profile,
            SisMatrixRole::Outer,
            dims.d_b(),
            params.outer().digits.log_basis,
        ),
    )?;
    audit_bound(
        label,
        params.open().matrix.coeff_linf_bound(),
        rounded_up_collision_inf_norm(
            policy.sis_security_policy,
            policy.sis_modulus_profile,
            SisMatrixRole::Open,
            dims.d_d(),
            shared_d_digit_log_basis(params.open().digits.log_basis, params.precommitted_groups()),
        ),
    )
}

#[derive(Clone, Copy)]
struct TerminalL2ModelState {
    fold_level: usize,
}

fn audit_terminal(
    params: &TerminalFoldParams,
    model_state: TerminalL2ModelState,
    policy: &PlannerPolicy,
) -> Result<(), AkitaError> {
    let label = "terminal fold";
    let sparse = &params.fold_challenge_config;
    let response_shape = &params.response_shape;
    audit_inner_matrix(label, &params.inner.matrix, policy)?;
    sparse
        .validate_for_ring_dim(params.d_a())
        .map_err(|message| invalid(label, message))?;
    if params.fold.log_basis == 0
        || params.fold.num_digits == 0
        || params.blocks.live_ring_elements_per_claim == 0
        || params.blocks.positions_per_block == 0
        || !params.blocks.positions_per_block.is_power_of_two()
        || params.blocks.live_blocks
            != params
                .blocks
                .live_ring_elements_per_claim
                .div_ceil(params.blocks.positions_per_block)
    {
        return Err(invalid(label, "invalid terminal fold or block geometry"));
    }

    let expected_digits = num_digits_inner(
        DecompositionParams {
            log_basis: params.inner.digits.log_basis,
            ..policy.decomposition
        },
        false,
    );
    let expected_width = params
        .blocks
        .positions_per_block
        .checked_mul(expected_digits)
        .ok_or_else(|| invalid(label, "A width overflow"))?;
    if params.inner.digits.num_digits != expected_digits
        || params.inner.matrix.input_width() != expected_width
    {
        return Err(invalid(
            label,
            "terminal digits or A width are not canonical",
        ));
    }

    let group = response_shape
        .layout
        .groups
        .first()
        .ok_or_else(|| invalid(label, "terminal response shape is missing its group"))?;
    let d = params.d_a();
    let expected_z_coords = params
        .inner_width()
        .checked_mul(d)
        .ok_or_else(|| invalid(label, "terminal z coordinates overflow"))?;
    let expected_e_field_elems = params
        .blocks
        .live_blocks
        .checked_mul(d)
        .ok_or_else(|| invalid(label, "terminal e coordinates overflow"))?;
    let expected_t_field_elems = params
        .blocks
        .live_blocks
        .checked_mul(params.inner.matrix.output_rank())
        .and_then(|value| value.checked_mul(d))
        .ok_or_else(|| invalid(label, "terminal t coordinates overflow"))?;
    let expected_logical_num_elems = expected_z_coords
        .checked_add(expected_e_field_elems)
        .and_then(|value| value.checked_add(expected_t_field_elems))
        .ok_or_else(|| invalid(label, "terminal response coordinates overflow"))?;
    if matches!(
        params.inner.matrix.security_route(),
        akita_types::InnerCommitSecurityRoute::L2 { .. }
    ) {
        if akita_challenges::selective_l2_operator_norm_rejection(d, sparse).is_none() {
            return Err(invalid(label, "terminal L2 challenge is not certified"));
        }
        let fold_basis = 1usize
            .checked_shl(params.fold.log_basis)
            .ok_or_else(|| invalid(label, "L2 balanced digit basis overflow"))?;
        let expected = selective_l2_inner_matrix(
            policy,
            SelectiveL2CandidateGeometry {
                fold_level: model_state.fold_level,
                num_claims: 1,
                num_chunks: 1,
                inner_width: expected_width,
                ring_dimension: d,
                fold_basis,
                fold_digit_count: params.fold.num_digits,
                fold_challenge_config: sparse,
                response_l2_sq_cap: params.response_l2_sq_cap(),
                norm_proof_shape: Some(akita_types::PhysicalL2NormProofShape::Direct {
                    physical_response_len: expected_z_coords,
                }),
            },
        )?
        .ok_or_else(|| {
            invalid(
                label,
                "L2 route is not admitted by the frozen terminal response cap",
            )
        })?;
        if params.inner.matrix != expected {
            return Err(invalid(
                label,
                "L2 A matrix disagrees with canonical cap, proof shape, table, or rank",
            ));
        }
    }
    if response_shape.layout.groups.len() != 1
        || response_shape.layout.ring_dimension != d
        || group.z_coords != expected_z_coords
        || group.e_field_elems != expected_e_field_elems
        || group.t_field_elems != expected_t_field_elems
        || response_shape.layout.logical_num_elems != expected_logical_num_elems
    {
        return Err(invalid(
            label,
            "terminal response shape disagrees with the committed witness geometry",
        ));
    }
    if params.validate_terminal_linf_cap(group.z_linf_cap).is_err()
        || group.z_rice_low_bits >= 64
        || group.z_payload_bytes == 0
    {
        return Err(invalid(
            label,
            "terminal response shape has invalid wire parameters or exceeds the matrix-certified cap",
        ));
    }
    Ok(())
}

/// Re-audit one complete expanded row against the policy the verifier trusts.
pub(crate) fn audit_resolved_schedule(
    profiles: &CommittedGroupBatchProfile,
    schedule: &FoldSchedule,
    policy: &PlannerPolicy,
) -> Result<(), AkitaError> {
    validate_policy(policy)?;
    profiles.validate(policy.decomposition.field_bits())?;
    schedule.validate_structure()?;

    let final_params = &schedule.root.params;
    // Four of the five comparisons that used to live here compared a field with
    // a copy of itself and are deleted with the merge: the shared D matrix, the
    // precommitted-group count (twice over), and each group's `descriptor`
    // against its own `commitment.profile`. What survives is the only one that
    // relates two independent sources: the ordered profiles from the lookup key
    // against the expanded row.
    if profiles.final_group != final_params.own_group().profile
        || profiles.precommitteds.len() != final_params.precommitted_groups().len()
    {
        return Err(invalid(
            "root fold",
            "ordered profiles disagree with the expanded row",
        ));
    }

    for (index, (profile, group)) in profiles
        .precommitteds
        .iter()
        .zip(final_params.precommitted_groups())
        .enumerate()
    {
        if profile != &group.profile {
            return Err(invalid(
                &format!("root precommitted group {index}"),
                "profile and consuming parameters disagree",
            ));
        }
    }

    visit_schedule_groups(schedule, |group| {
        let label = group.position().to_string();
        match group {
            ScheduleGroup::Frozen {
                params,
                num_response_chunks,
                ..
            } => audit_frozen_group(&label, params, num_response_chunks, policy),
            ScheduleGroup::Final {
                position,
                params,
                num_claims,
                fold_level,
            } => {
                if position == ScheduleGroupPosition::RootFinal
                    && !matches!(
                        params.inner().matrix.security_route(),
                        InnerCommitSecurityRoute::Linf(_)
                    )
                {
                    return Err(invalid(&label, "root cannot use an L2 A security route"));
                }
                audit_committed_params(&label, params, num_claims, fold_level, policy)
            }
            ScheduleGroup::Terminal {
                params, fold_level, ..
            } => audit_terminal(params, TerminalL2ModelState { fold_level }, policy),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PlannerCostModelId, RingDimensionScheduleMode, SelectionPolicyId};
    use akita_types::{
        ChunkedWitnessCfg, GadgetDigits, InnerRoleParams, SisL2TableDigest, SisModulusProfileId,
        SisSecurityPolicyId, SisTableDigest, TailSegmentLayout,
    };

    const INNER_WIDTH: usize = 16;
    const RESPONSE_CAP: u128 = 500_000_000;

    fn policy() -> PlannerPolicy {
        PlannerPolicy {
            cost_model: PlannerCostModelId::ExactPayloadAndSetupEnvelope,
            selective_l2_response_model: crate::SelectiveL2ResponseModelId::Disabled,
            selection_policy: SelectionPolicyId::MinEstimatedProofPayloadV2,
            recursive_split_search_policy: crate::RecursiveSplitSearchPolicy::Exhaustive,
            recursive_setup_search_policy: crate::RecursiveSetupSearchPolicy::Exhaustive,
            setup_field_budget: None,
            min_offloaded_witness_contraction: 3,
            ring_dimension_schedule_mode: RingDimensionScheduleMode::UniformDimension {
                ring_dimension: 64,
            },
            decomposition: DecompositionParams {
                log_basis: 4,
                log_commit_bound: 1,
                log_open_bound: Some(128),
            },
            sis_modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
            sis_security_policy: SisSecurityPolicyId::Quantum128BitADPS16,
            sis_table_digest: SisTableDigest::CURRENT,
            sis_l2_table_digest: SisL2TableDigest::CURRENT,
            claim_ext_degree: 1,
            chal_ext_degree: 1,
            inner_basis_range: (3, 16),
            opening_basis_range: (3, 6),
            witness_chunk: ChunkedWitnessCfg::default(),
            recursive_setup_planning: false,
        }
    }

    #[test]
    fn terminal_audit_rejects_a_stale_l2_bucket() {
        let policy = policy();
        let sparse = akita_challenges::selective_l2_challenge_config(64)
            .expect("certified D64 L2 challenge");
        let expected = selective_l2_inner_matrix(
            &policy,
            SelectiveL2CandidateGeometry {
                fold_level: 3,
                num_claims: 1,
                num_chunks: 1,
                inner_width: INNER_WIDTH,
                ring_dimension: 64,
                fold_basis: 16,
                fold_digit_count: 3,
                fold_challenge_config: &sparse,
                response_l2_sq_cap: Some(RESPONSE_CAP),
                norm_proof_shape: Some(akita_types::PhysicalL2NormProofShape::Direct {
                    physical_response_len: INNER_WIDTH * 64,
                }),
            },
        )
        .expect("candidate construction")
        .expect("exact terminal calibration");
        let mut terminal = TerminalFoldParams {
            blocks: akita_types::BlockGeometry::new(16, 16, 1),
            inner: InnerRoleParams::new(GadgetDigits::new(4, 1), expected),
            fold: GadgetDigits::new(4, 3),
            fold_challenge_config: sparse,
            response_shape: TerminalResponseShape {
                layout: TailSegmentLayout {
                    ring_dimension: 64,
                    groups: Vec::new(),
                    logical_num_elems: 0,
                },
            },
            input_witness_len: 1_024,
        };
        terminal.response_shape =
            TerminalResponseShape::derive(&terminal, 10).expect("valid terminal response shape");
        audit_terminal(&terminal, TerminalL2ModelState { fold_level: 3 }, &policy)
            .expect("canonical terminal matrix");

        let (table_key, norm_proof_shape) = match terminal.inner.matrix.security_route() {
            InnerCommitSecurityRoute::L2 {
                table_key,
                norm_proof_shape,
                ..
            } => (table_key, norm_proof_shape),
            InnerCommitSecurityRoute::Linf(_) => panic!("expected L2 terminal"),
        };
        terminal.inner.matrix = InnerCommitMatrixParams::try_new_l2_with_min_rank(
            table_key,
            INNER_WIDTH,
            RESPONSE_CAP * 16,
            norm_proof_shape,
        )
        .expect("locally well-formed matrix with a stale table bucket");
        assert!(
            audit_terminal(&terminal, TerminalL2ModelState { fold_level: 3 }, &policy,).is_err()
        );
    }
}
