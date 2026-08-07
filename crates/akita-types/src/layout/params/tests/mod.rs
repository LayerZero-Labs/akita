use super::*;
use crate::PolynomialGroupLayout;

fn sample_params_only() -> CommittedGroupParams {
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

fn sample_layout_lp() -> CommittedGroupParams {
    sample_params_only().with_decomp(16, 64, 2, 2, 2).unwrap()
}

#[test]
fn distinct_semantic_depths_size_a_b_and_d_independently() {
    let mut params = sample_params_only();
    params.log_basis_inner = 2;
    params.log_basis_outer = 3;
    params.log_basis_open = 4;
    let params = params
        .with_decomp(8, 17, 5, 4, 3)
        .expect("distinct semantic decomposition");
    let blocks = 17usize.div_ceil(8);
    assert_eq!(
        params.inner_commit_matrix.input_width(),
        8 * 5,
        "A uses inner depth"
    );
    assert_eq!(
        params.outer_commit_matrix.input_width(),
        params.inner_commit_matrix.output_rank() * 4 * blocks,
        "B uses outer depth"
    );
    assert_eq!(
        params.open_commit_matrix.input_width(),
        3 * blocks,
        "D uses open depth"
    );
    assert_eq!(
        (
            params.log_basis_inner,
            params.log_basis_outer,
            params.log_basis_open,
        ),
        (2, 3, 4)
    );
}

fn laid_out_sample_lp() -> CommittedGroupParams {
    sample_params_only()
        .with_layout(&sample_layout_lp())
        .unwrap()
}

fn certify_test_sis_bounds(lp: &mut CommittedGroupParams) {
    const INNER_BOUND: u128 = 2;
    const OUTER_BOUND: u128 = 3;
    lp.inner_commit_matrix = InnerCommitMatrixParams::new_unchecked(
        lp.inner_commit_matrix.security_policy(),
        lp.inner_commit_matrix.sis_table_key().table_digest,
        lp.inner_commit_matrix.sis_modulus_profile(),
        lp.inner_commit_matrix.output_rank(),
        lp.inner_commit_matrix.input_width(),
        INNER_BOUND,
        lp.d_a(),
    );
    lp.outer_commit_matrix = OuterCommitMatrixParams::new_unchecked(
        lp.outer_commit_matrix.security_policy(),
        lp.outer_commit_matrix.sis_table_key().table_digest,
        lp.outer_commit_matrix.sis_modulus_profile(),
        lp.outer_commit_matrix.output_rank(),
        lp.outer_commit_matrix.input_width(),
        OUTER_BOUND,
        lp.d_a(),
    );
}

fn sample_multi_group_root_params() -> (CommittedGroupParams, OpeningClaimsLayout) {
    use crate::schedule::CommittedGroupProfile;
    let mut lp = sample_params_only()
        .with_layout(&sample_layout_lp())
        .unwrap();
    lp.fold_challenge_config =
        SparseChallengeConfig::production_for_ring_dim(lp.d_a()).expect("test challenge");
    let mut precommit_lp = sample_params_only()
        .with_layout(&sample_layout_lp())
        .unwrap();
    precommit_lp.fold_challenge_config =
        SparseChallengeConfig::production_for_ring_dim(precommit_lp.d_a())
            .expect("precommit test challenge");
    certify_test_sis_bounds(&mut precommit_lp);
    let outer_commit_matrix = OuterCommitMatrixParams::new_unchecked(
        precommit_lp.outer_commit_matrix.security_policy(),
        precommit_lp
            .outer_commit_matrix
            .sis_table_key()
            .table_digest,
        precommit_lp.outer_commit_matrix.sis_modulus_profile(),
        5,
        precommit_lp.outer_commit_matrix.input_width(),
        precommit_lp.outer_commit_matrix.coeff_linf_bound(),
        precommit_lp.d_a(),
    );
    let mut layout =
        CommittedGroupProfile::from_params(PolynomialGroupLayout::new(4, 1), &precommit_lp);
    layout.outer_commit_matrix = outer_commit_matrix;
    let precommit = PrecommittedLevelParams {
        layout,
        log_basis_open: precommit_lp.log_basis_open,
        fold_challenge_config: precommit_lp.fold_challenge_config,
        num_digits_open: precommit_lp.num_digits_open,
        num_digits_fold: precommit_lp.num_digits_fold,
    };
    let mut grouped = lp;
    grouped.precommitted_groups = vec![precommit];
    let batch = OpeningClaimsLayout::from_group_sizes(4, &[1, 1]).expect("layout");
    (grouped, batch)
}

#[test]
fn fold_nonce_accepts_only_the_global_attempt_range() {
    let (params, opening_batch) = sample_multi_group_root_params();
    let attempts = crate::FoldLinfProtocolBinding::CURRENT.max_grind_attempts;

    params
        .validate_fold_grind_nonce(&opening_batch, attempts - 1)
        .expect("last in-range nonce");
    assert!(params
        .validate_fold_grind_nonce(&opening_batch, attempts)
        .is_err());
}

#[test]
fn shared_d_digit_basis_uses_root_opening_basis() {
    let (mut grouped, _) = sample_multi_group_root_params();
    grouped.log_basis_open = 3;
    grouped.precommitted_groups[0].layout.log_basis_outer = 6;

    assert_eq!(grouped.shared_d_digit_log_basis(), 3);
    assert_eq!(shared_d_digit_log_basis(5, &[]), 5);
}

#[test]
fn with_decomp_derives_exact_live_block_geometry() {
    let lp = sample_params_only().with_decomp(8, 17, 2, 2, 2).unwrap();

    assert_eq!(lp.num_live_ring_elements_per_claim, 17);
    assert_eq!(lp.num_positions_per_block, 8);
    assert_eq!(lp.num_live_blocks, 3);
    assert_eq!(lp.position_index_bits(), 3);
    assert_eq!(lp.block_index_bits(), 2);
    assert_eq!(lp.block_index_domain_size().unwrap(), 4);
    assert_eq!(lp.n_ring_elems().unwrap(), 17);

    assert!(sample_params_only().with_decomp(3, 17, 2, 2, 2).is_err());
}

#[test]
fn with_layout_keeps_self_ranks() {
    let params = sample_params_only();
    let layout_lp = sample_layout_lp();

    let lp = params.with_layout(&layout_lp).unwrap();

    assert_eq!(lp.d_a(), 64);
    assert_eq!(lp.log_basis_inner, layout_lp.log_basis_inner);
    assert_eq!(lp.log_basis_outer, layout_lp.log_basis_outer);
    assert_eq!(lp.log_basis_open, layout_lp.log_basis_open);
    assert_eq!(lp.inner_commit_matrix.output_rank(), 2);
    assert_eq!(lp.outer_commit_matrix.output_rank(), 4);
    assert_eq!(lp.open_commit_matrix.output_rank(), 3);
    assert_eq!(lp.num_live_blocks, layout_lp.num_live_blocks);
    assert_eq!(
        lp.num_positions_per_block,
        layout_lp.num_positions_per_block
    );
    assert_eq!(lp.challenge_l1_mass(), 3);
    assert_eq!(lp.num_digits_inner, layout_lp.num_digits_inner);
    assert_eq!(lp.num_digits_outer, layout_lp.num_digits_outer);
    assert_eq!(lp.num_digits_open, layout_lp.num_digits_open);
}

#[test]
fn derived_widths_match_ajtai_col_len() {
    let lp = sample_params_only()
        .with_layout(&sample_layout_lp())
        .unwrap();

    assert_eq!(lp.inner_width(), lp.inner_commit_matrix.input_width());
    assert_eq!(lp.outer_width(), lp.outer_commit_matrix.input_width());
    assert_eq!(lp.d_matrix_width(), lp.open_commit_matrix.input_width());
}

#[test]
fn derived_log_values() {
    let layout_lp = sample_layout_lp();
    let lp = sample_params_only().with_layout(&layout_lp).unwrap();

    assert_eq!(lp.block_index_bits(), layout_lp.block_index_bits());
    assert_eq!(lp.position_index_bits(), layout_lp.position_index_bits());
    assert_eq!(
        lp.outer_vars(),
        layout_lp.position_index_bits() + layout_lp.block_index_bits()
    );
}

#[test]
fn relation_matrix_row_count_values() {
    let lp = sample_params_only()
        .with_layout(&sample_layout_lp())
        .unwrap();

    assert_eq!(lp.relation_matrix_row_count(1).unwrap(), 1 + 3 + 4 + 2 + 4);
    assert_eq!(
        lp.relation_matrix_row_count(2).unwrap(),
        1 + 3 + 4 * 2 + 2 + 6
    );
    assert_eq!(
        lp.relation_matrix_row_count(4).unwrap(),
        1 + 3 + 4 * 4 + 2 + 10
    );
}

#[test]
fn canonical_row_offsets_match_open_coded_layout() {
    let lp = sample_params_only()
        .with_layout(&sample_layout_lp())
        .unwrap();
    let n_a = lp.inner_commit_matrix.output_rank();
    let n_b = lp.outer_commit_matrix.output_rank();
    let n_d = lp.open_commit_matrix.output_rank();

    for nc in [1usize, 2, 4] {
        let n_d_active = n_d;
        let a_start = 1;
        let b_start = a_start + n_a;
        let d_start = b_start + n_b * nc;

        assert_eq!(lp.a_start(), a_start);
        assert_eq!(lp.b_start().unwrap(), b_start);
        assert_eq!(lp.d_start(nc).unwrap(), d_start);
        assert_eq!(
            lp.relation_matrix_row_count(nc).unwrap(),
            d_start + n_d_active + 2 * (nc + 1)
        );
    }
}

#[path = "params_precommitted_group_tests.rs"]
mod precommitted_group_tests;
#[path = "params_relation_row_tests.rs"]
mod relation_row_tests;
