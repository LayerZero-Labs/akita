use super::*;
use crate::PolynomialGroupLayout;

fn sample_params_only() -> LevelParams {
    LevelParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        64,
        3,
        2,
        4,
        3,
        SparseChallengeConfig::pm1_only(3),
    )
}

fn sample_layout_lp() -> LevelParams {
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
    assert_eq!(params.a_key.col_len(), 8 * 5, "A uses inner depth");
    assert_eq!(
        params.b_key.col_len(),
        params.a_key.row_len() * 4 * blocks,
        "B uses outer depth"
    );
    assert_eq!(params.d_key.col_len(), 3 * blocks, "D uses open depth");
    assert_eq!(
        (
            params.log_basis_inner,
            params.log_basis_outer,
            params.log_basis_open,
        ),
        (2, 3, 4)
    );
}

fn laid_out_sample_lp() -> LevelParams {
    sample_params_only()
        .with_layout(&sample_layout_lp(), 128)
        .unwrap()
}

fn certify_test_sis_bounds(lp: &mut LevelParams) {
    const BOUND: u128 = 1;
    lp.a_key = AjtaiKeyParams::new_unchecked(
        lp.a_key.security_policy(),
        lp.a_key.sis_table_key().table_digest,
        lp.a_key.sis_modulus_profile(),
        crate::sis::SisMatrixRole::A,
        lp.a_key.row_len(),
        lp.a_key.col_len(),
        BOUND,
        lp.ring_dimension,
    );
    lp.b_key = AjtaiKeyParams::new_unchecked(
        lp.b_key.security_policy(),
        lp.b_key.sis_table_key().table_digest,
        lp.b_key.sis_modulus_profile(),
        crate::sis::SisMatrixRole::B,
        lp.b_key.row_len(),
        lp.b_key.col_len(),
        BOUND,
        lp.ring_dimension,
    );
}

fn sample_multi_group_root_params() -> (LevelParams, OpeningClaimsLayout) {
    use crate::schedule::PrecommittedGroupParams;
    let lp = sample_params_only()
        .with_layout(&sample_layout_lp(), 128)
        .unwrap();
    let mut precommit_lp = sample_params_only()
        .with_layout(&sample_layout_lp(), 128)
        .unwrap();
    certify_test_sis_bounds(&mut precommit_lp);
    let a_key = precommit_lp.a_key.clone();
    let b_key = AjtaiKeyParams::new_unchecked(
        precommit_lp.b_key.security_policy(),
        precommit_lp.b_key.sis_table_key().table_digest,
        precommit_lp.b_key.sis_modulus_profile(),
        precommit_lp.b_key.sis_table_key().role,
        5,
        precommit_lp.b_key.col_len(),
        precommit_lp.b_key.coeff_linf_bound(),
        precommit_lp.ring_dimension,
    );
    let mut layout =
        PrecommittedGroupParams::from_params(PolynomialGroupLayout::new(4, 1), &precommit_lp);
    layout.n_b = b_key.row_len();
    layout.b_coeff_linf_bound = b_key.coeff_linf_bound();
    let precommit = PrecommittedLevelParams {
        layout,
        a_key,
        b_key,
        log_basis_open: precommit_lp.log_basis_open,
        num_digits_inner: precommit_lp.num_digits_inner,
        num_digits_outer: precommit_lp.num_digits_outer,
        num_digits_open: precommit_lp.num_digits_open,
        num_digits_fold_one: precommit_lp.num_digits_fold_one,
    };
    let mut grouped = lp;
    grouped.precommitted_groups = vec![precommit];
    let batch = OpeningClaimsLayout::from_group_sizes(4, &[1, 1]).expect("layout");
    (grouped, batch)
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
fn root_group_fold_linf_config_uses_group_local_tensor_shape() {
    let (mut lp, batch) = sample_multi_group_root_params();
    lp.precommitted_groups[0].layout.fold_challenge_shape =
        TensorChallengeShape::Tensor { fold_low_len: 2 };

    let precommitted = lp.group_params(&batch, 0).unwrap();
    let final_group = lp.group_params(&batch, 1).unwrap();
    let precommitted_config = lp
        .fold_witness_linf_cap_config_for_params(precommitted)
        .unwrap();
    let final_config = lp
        .fold_witness_linf_cap_config_for_params(final_group)
        .unwrap();

    assert_eq!(precommitted_config.tensor_fold_low_len, 2);
    assert_eq!(final_config.tensor_fold_low_len, 0);
}

#[test]
fn with_layout_keeps_self_ranks() {
    let params = sample_params_only();
    let layout_lp = sample_layout_lp();

    let lp = params.with_layout(&layout_lp, 128).unwrap();

    assert_eq!(lp.ring_dimension, 64);
    assert_eq!(lp.log_basis_inner, layout_lp.log_basis_inner);
    assert_eq!(lp.log_basis_outer, layout_lp.log_basis_outer);
    assert_eq!(lp.log_basis_open, layout_lp.log_basis_open);
    assert_eq!(lp.a_key.row_len(), 2);
    assert_eq!(lp.b_key.row_len(), 4);
    assert_eq!(lp.d_key.row_len(), 3);
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
        .with_layout(&sample_layout_lp(), 128)
        .unwrap();

    assert_eq!(lp.inner_width(), lp.a_key.col_len());
    assert_eq!(lp.outer_width(), lp.b_key.col_len());
    assert_eq!(lp.d_matrix_width(), lp.d_key.col_len());
}

#[test]
fn with_fold_linf_cap_config_propagates_fold_digit_errors() {
    let mut lp = sample_layout_lp();
    lp.fold_challenge_config = SparseChallengeConfig::pm1_only(0);

    let err = lp
        .with_fold_linf_cap_config(128, 1)
        .expect_err("zero challenge mass must reject");

    assert!(matches!(err, AkitaError::InvalidSetup(message) if message.contains("β = 0")));
}

#[test]
fn derived_log_values() {
    let layout_lp = sample_layout_lp();
    let lp = sample_params_only().with_layout(&layout_lp, 128).unwrap();

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
        .with_layout(&sample_layout_lp(), 128)
        .unwrap();

    assert_eq!(
        lp.relation_matrix_row_count_for(1, RelationMatrixRowLayout::WithDBlock)
            .unwrap(),
        1 + 3 + 4 + 2
    );
    assert_eq!(
        lp.relation_matrix_row_count_for(2, RelationMatrixRowLayout::WithDBlock)
            .unwrap(),
        1 + 3 + 4 * 2 + 2
    );
    assert_eq!(
        lp.relation_matrix_row_count_for(4, RelationMatrixRowLayout::WithDBlock)
            .unwrap(),
        1 + 3 + 4 * 4 + 2
    );
    assert_eq!(
        lp.relation_matrix_row_count_for(2, RelationMatrixRowLayout::WithoutDBlock)
            .unwrap(),
        1 + 4 * 2 + 2
    );
}

#[test]
fn canonical_row_offsets_match_open_coded_layout() {
    let lp = sample_params_only()
        .with_layout(&sample_layout_lp(), 128)
        .unwrap();
    let n_a = lp.a_key.row_len();
    let n_b = lp.b_key.row_len();
    let n_d = lp.d_key.row_len();

    for nc in [1usize, 2, 4] {
        for layout in [
            RelationMatrixRowLayout::WithDBlock,
            RelationMatrixRowLayout::WithoutDBlock,
        ] {
            let n_d_active = match layout {
                RelationMatrixRowLayout::WithDBlock => n_d,
                RelationMatrixRowLayout::WithoutDBlock => 0,
            };
            let a_start = 1;
            let b_start = a_start + n_a;
            let d_start = b_start + n_b * nc;

            assert_eq!(lp.a_start(), a_start);
            assert_eq!(lp.b_start().unwrap(), b_start);
            assert_eq!(lp.d_start(nc).unwrap(), d_start);
            assert_eq!(
                lp.relation_matrix_row_count_for(nc, layout).unwrap(),
                d_start + n_d_active
            );
        }
    }
}

#[path = "params_precommitted_group_tests.rs"]
mod precommitted_group_tests;
#[path = "params_relation_row_tests.rs"]
mod relation_row_tests;
