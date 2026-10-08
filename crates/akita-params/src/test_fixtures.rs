//! Shared parameter fixtures for unit and cross-crate tests.

use crate::{
    CommittedGroupParams, GroupCommitPhaseParams, GroupOpenPhaseParams, InnerCommitMatrixParams,
    OpenCommitMatrixParams, OpeningClaimsLayout, OuterCommitMatrixParams, PolynomialGroupLayout,
    SisModulusProfileId,
};
use akita_challenges::SparseChallengeConfig;

pub fn sample_level_params() -> CommittedGroupParams {
    CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        64,
        3,
        3,
        3,
        2,
        SparseChallengeConfig::pm1_only(3),
    )
    .with_decomp(4, 3, 2, 2, 2)
    .expect("sample level params")
}

pub fn prefix_eligible_level_params() -> CommittedGroupParams {
    let field_element_digits = crate::sis::compute_num_digits_field_width(
        SisModulusProfileId::Q32Offset99.field_bits(),
        3,
    );
    CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        64,
        2,
        2,
        3,
        2,
        SparseChallengeConfig::pm1_only(3),
    )
    .with_decomp(2, 3, field_element_digits, 2, 2)
    .expect("prefix eligible level params")
}

pub fn retarget_group_role_dims(
    params: &mut CommittedGroupParams,
    inner_ring_dimension: usize,
    outer_ring_dimension: usize,
) {
    params.own_group_mut().opening.fold_challenge_config =
        SparseChallengeConfig::production_for_ring_dim(inner_ring_dimension)
            .expect("production challenge");
    let a_bound = *crate::sis::inner_coeff_linf_bounds(
        params.inner().matrix.sis_modulus_profile(),
        u32::try_from(inner_ring_dimension).expect("test ring dimension"),
    )
    .first()
    .expect("retargeted exact A bounds");
    let inner = params.inner().matrix;
    params.own_group_mut().profile.inner.matrix =
        crate::InnerCommitMatrixParams::try_new_with_min_rank(
            crate::SisTableKey {
                policy: inner.security_policy(),
                table_digest: inner
                    .sis_table_key()
                    .expect("L infinity test matrix")
                    .table_digest,
                modulus_profile: inner.sis_modulus_profile(),
                role: crate::sis::SisMatrixRole::Inner,
                ring_dimension: u32::try_from(inner_ring_dimension).expect("test ring dimension"),
                coeff_linf_bound: a_bound,
            },
            inner.input_width(),
        )
        .expect("audited retargeted A matrix");
    let inner_output_rank = params.inner().matrix.output_rank();
    let projected_outer_width = inner_output_rank
        .checked_mul(params.outer().digits.num_digits)
        .and_then(|width| width.checked_mul(params.blocks().live_blocks))
        .and_then(|width| width.checked_mul(inner_ring_dimension / outer_ring_dimension))
        .expect("retargeted B width");
    let outer = params.outer().matrix;
    params.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::try_new_with_min_rank(
        crate::SisTableKey {
            policy: outer.security_policy(),
            table_digest: outer.sis_table_key().table_digest,
            modulus_profile: outer.sis_modulus_profile(),
            role: crate::sis::SisMatrixRole::Outer,
            ring_dimension: u32::try_from(outer_ring_dimension).expect("test ring dimension"),
            coeff_linf_bound: 3,
        },
        projected_outer_width,
    )
    .expect("audited retargeted B matrix");
}

pub fn retarget_group_role_dims_wide(
    params: &mut CommittedGroupParams,
    inner_ring_dimension: usize,
    outer_ring_dimension: usize,
    min_input_width: usize,
) {
    retarget_group_role_dims(params, inner_ring_dimension, outer_ring_dimension);
    let inner = params.inner().matrix;
    params.own_group_mut().profile.inner.matrix =
        crate::InnerCommitMatrixParams::try_new_with_min_rank(
            inner.sis_table_key().expect("L infinity test matrix"),
            inner.input_width().max(min_input_width),
        )
        .expect("wide audited A matrix");
    let inner_output_rank = params.inner().matrix.output_rank();
    let projected_outer_width = inner_output_rank
        .checked_mul(params.outer().digits.num_digits)
        .and_then(|width| width.checked_mul(params.blocks().live_blocks))
        .and_then(|width| width.checked_mul(inner_ring_dimension / outer_ring_dimension))
        .expect("wide retargeted B width")
        .max(min_input_width);
    let outer = params.outer().matrix;
    params.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::try_new_with_min_rank(
        crate::SisTableKey {
            policy: outer.security_policy(),
            table_digest: outer.sis_table_key().table_digest,
            modulus_profile: outer.sis_modulus_profile(),
            role: crate::sis::SisMatrixRole::Outer,
            ring_dimension: u32::try_from(outer_ring_dimension).expect("test ring dimension"),
            coeff_linf_bound: 3,
        },
        projected_outer_width,
    )
    .expect("wide audited B matrix");
}

pub fn sample_params_only() -> CommittedGroupParams {
    CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        64,
        3,
        2,
        4,
        3,
        SparseChallengeConfig::pm1_only(3),
    )
}

pub fn sample_layout_lp() -> CommittedGroupParams {
    sample_params_only().with_decomp(16, 64, 2, 2, 2).unwrap()
}

fn certify_test_sis_bounds(lp: &mut CommittedGroupParams) {
    const OUTER_BOUND: u128 = 3;
    let inner_bound = crate::sis::rounded_up_role_a_inf_norm(
        lp.inner().matrix.security_policy(),
        lp.inner()
            .matrix
            .sis_table_key()
            .expect("L infinity test matrix")
            .table_digest,
        lp.inner().matrix.sis_modulus_profile(),
        lp.d_a(),
        lp.open().digits.log_basis,
        &lp.fold_challenge_config(),
        lp.num_digits_fold(),
        lp.witness_chunk.num_chunks,
    )
    .expect("exact A-role test bound");
    lp.own_group_mut().profile.inner.matrix = InnerCommitMatrixParams::new_unchecked(
        lp.inner().matrix.security_policy(),
        lp.inner()
            .matrix
            .sis_table_key()
            .expect("L infinity test matrix")
            .table_digest,
        lp.inner().matrix.sis_modulus_profile(),
        lp.inner().matrix.output_rank(),
        lp.inner().matrix.input_width(),
        inner_bound,
        lp.d_a(),
    );
    lp.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
        lp.outer().matrix.security_policy(),
        lp.outer().matrix.sis_table_key().table_digest,
        lp.outer().matrix.sis_modulus_profile(),
        lp.outer().matrix.output_rank(),
        lp.outer().matrix.input_width(),
        OUTER_BOUND,
        lp.d_a(),
    );
}

pub fn sample_multi_group_root_params() -> (CommittedGroupParams, OpeningClaimsLayout) {
    use crate::schedule::GroupCommitPhaseParams;
    let mut lp = sample_params_only()
        .with_layout(&sample_layout_lp())
        .unwrap();
    lp.own_group_mut().opening.fold_challenge_config =
        SparseChallengeConfig::production_for_ring_dim(lp.d_a()).expect("test challenge");
    let mut precommit_lp = sample_params_only()
        .with_layout(&sample_layout_lp())
        .unwrap();
    precommit_lp.own_group_mut().opening.fold_challenge_config =
        SparseChallengeConfig::production_for_ring_dim(precommit_lp.d_a())
            .expect("precommit test challenge");
    certify_test_sis_bounds(&mut precommit_lp);
    let outer_commit_matrix = OuterCommitMatrixParams::new_unchecked(
        precommit_lp.outer().matrix.security_policy(),
        precommit_lp.outer().matrix.sis_table_key().table_digest,
        precommit_lp.outer().matrix.sis_modulus_profile(),
        5,
        precommit_lp.outer().matrix.input_width(),
        precommit_lp.outer().matrix.coeff_linf_bound(),
        precommit_lp.d_a(),
    );
    let mut layout = GroupCommitPhaseParams::from_params_unchecked_for_test(
        PolynomialGroupLayout::new(4, 1),
        &precommit_lp,
    );
    layout.outer.matrix = outer_commit_matrix;
    let precommit = GroupOpenPhaseParams {
        setup_natural_len: None,
        profile: layout,
        opening: crate::GroupOpeningPlan::evaluation_trace(
            precommit_lp.fold_challenge_config(),
            precommit_lp.open().digits.log_basis,
            precommit_lp.open().digits.num_digits,
            precommit_lp.num_digits_fold(),
        ),
    };
    let mut grouped = lp;
    grouped.set_precommitted_groups(vec![precommit]).unwrap();
    let batch = OpeningClaimsLayout::from_group_sizes(4, &[1, 1]).expect("layout");
    (grouped, batch)
}

fn configure_test_role_dims(lp: &mut CommittedGroupParams, d_b: usize, d_d: usize) {
    let d_a = lp.d_a();
    assert!(d_a.is_multiple_of(d_b));
    assert!(d_a.is_multiple_of(d_d));
    let outer = &lp.outer().matrix;
    lp.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width() * (d_a / d_b),
        outer.coeff_linf_bound(),
        d_b,
    );
    let open = &lp.open().matrix;
    lp.open_matrix = OpenCommitMatrixParams::new_unchecked(
        open.security_policy(),
        open.sis_table_key().table_digest,
        open.sis_modulus_profile(),
        open.output_rank(),
        open.input_width() * (d_a / d_d),
        open.coeff_linf_bound(),
        d_d,
    );
}

pub(crate) fn address_oracle_group_params(
    d_a: usize,
    d_b: usize,
    d_d: usize,
    blocks: usize,
) -> CommittedGroupParams {
    let mut lp = CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        d_a,
        3,
        3,
        2,
        3,
        SparseChallengeConfig::production_for_ring_dim(d_a).expect("test challenge"),
    )
    .with_decomp(4, blocks * 4, 2, 2, 2)
    .expect("address-oracle params");
    configure_test_role_dims(&mut lp, d_b, d_d);
    lp
}

pub(crate) fn address_oracle_precommit(
    d_a: usize,
    d_b: usize,
    d_d: usize,
    blocks: usize,
    claims: usize,
) -> GroupOpenPhaseParams {
    let mut lp = address_oracle_group_params(d_a, d_b, d_d, blocks);
    certify_test_sis_bounds(&mut lp);
    let outer = &lp.outer().matrix;
    lp.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width() * claims,
        outer.coeff_linf_bound(),
        d_b,
    );
    let layout = GroupCommitPhaseParams::from_params_unchecked_for_test(
        PolynomialGroupLayout::new(4, claims),
        &lp,
    );
    GroupOpenPhaseParams {
        setup_natural_len: None,
        profile: layout,
        opening: crate::GroupOpeningPlan::evaluation_trace(
            lp.fold_challenge_config(),
            lp.open().digits.log_basis,
            lp.open().digits.num_digits,
            lp.num_digits_fold(),
        ),
    }
}

pub fn address_oracle_fixture(group_count: usize) -> (CommittedGroupParams, OpeningClaimsLayout) {
    let (final_dims, precommitted) = match group_count {
        1 => ((64, 64, 64, 8, 2), Vec::new()),
        2 => ((64, 64, 64, 8, 2), vec![(128, 64, 64, 16, 1)]),
        3 => (
            (64, 64, 64, 8, 2),
            vec![(128, 64, 64, 16, 1), (64, 64, 64, 8, 3)],
        ),
        _ => panic!("address-oracle fixture supports one to three groups"),
    };
    let (d_a, d_b, d_d, blocks, final_claims) = final_dims;
    let mut lp = address_oracle_group_params(d_a, d_b, d_d, blocks);
    lp.set_precommitted_groups(
        precommitted
            .iter()
            .map(|&(a, b, d, blocks, claims)| address_oracle_precommit(a, b, d, blocks, claims))
            .collect(),
    )
    .unwrap();
    let precommitted_layouts = lp
        .precommitted_groups()
        .iter()
        .map(|group| group.profile.group)
        .collect::<Vec<_>>();
    let batch = OpeningClaimsLayout::from_root_groups(
        &precommitted_layouts,
        PolynomialGroupLayout::new(4, final_claims),
    )
    .expect("address-oracle opening layout");
    (lp, batch)
}

pub fn committed_params(ring_dimension: usize) -> CommittedGroupParams {
    committed_params_with_geometry(ring_dimension, 4, 4)
}

pub fn committed_params_with_geometry(
    ring_dimension: usize,
    num_positions_per_block: usize,
    num_live_ring_elements_per_claim: usize,
) -> CommittedGroupParams {
    let mut params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        ring_dimension,
        3,
        2,
        2,
        2,
        SparseChallengeConfig::production_for_ring_dim(ring_dimension)
            .expect("production test challenge"),
    )
    .with_decomp(
        num_positions_per_block,
        num_live_ring_elements_per_claim,
        2,
        2,
        2,
    )
    .expect("schedule validation params");
    let a_bound = exact_test_a_bound(&params);
    let inner = params.inner().matrix;
    params.own_group_mut().profile.inner.matrix = crate::InnerCommitMatrixParams::try_new(
        inner.security_policy(),
        inner
            .sis_table_key()
            .expect("L infinity test matrix")
            .table_digest,
        inner.sis_modulus_profile(),
        inner.output_rank(),
        inner.input_width(),
        a_bound,
        inner.ring_dimension(),
    )
    .expect("audited schedule A matrix");
    let outer = params.outer().matrix;
    params.own_group_mut().profile.outer.matrix = crate::OuterCommitMatrixParams::try_new(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width(),
        3,
        outer.ring_dimension(),
    )
    .expect("audited schedule B matrix");
    let source_len = num_live_ring_elements_per_claim * ring_dimension;
    assert!(source_len.is_power_of_two());
    params.own_group_mut().profile.group =
        PolynomialGroupLayout::new(source_len.trailing_zeros() as usize, 1);
    params
}

pub fn exact_test_a_bound(params: &CommittedGroupParams) -> u128 {
    crate::sis::rounded_up_role_a_inf_norm(
        params.inner().matrix.security_policy(),
        params
            .inner()
            .matrix
            .sis_table_key()
            .expect("L infinity test matrix")
            .table_digest,
        params.inner().matrix.sis_modulus_profile(),
        params.d_a(),
        params.open().digits.log_basis,
        &params.fold_challenge_config(),
        params.num_digits_fold(),
        params.witness_chunk.num_chunks,
    )
    .expect("exact test A bound")
}

pub fn test_lp() -> CommittedGroupParams {
    let mut params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        64,
        3,
        2,
        3,
        2,
        SparseChallengeConfig::pm1_only(3),
    )
    .with_decomp(8, 32, 2, 3, 3)
    .expect("tail segment test params");
    let key = crate::sis::SisTableKey {
        policy: params.inner().matrix.security_policy(),
        table_digest: params
            .inner()
            .matrix
            .sis_table_key()
            .expect("L infinity test matrix")
            .table_digest,
        modulus_profile: params.inner().matrix.sis_modulus_profile(),
        role: crate::sis::SisMatrixRole::Inner,
        ring_dimension: 64,
        coeff_linf_bound: crate::sis::sis_role_cells()
            .into_iter()
            .filter(|cell| {
                cell.role == crate::sis::SisMatrixRole::Inner
                    && cell.modulus_profile == params.inner().matrix.sis_modulus_profile()
                    && cell.ring_dimension == 64
            })
            .map(|cell| cell.coeff_linf_bound)
            .max()
            .expect("nonempty SIS A bounds"),
    };
    params.own_group_mut().profile.inner.matrix =
        crate::sis::InnerCommitMatrixParams::try_new_with_min_rank(
            key,
            params.inner().matrix.input_width(),
        )
        .expect("secure terminal test matrix");
    params
}
