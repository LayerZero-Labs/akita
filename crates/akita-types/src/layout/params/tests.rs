use super::*;
use crate::PolynomialGroupLayout;

fn sample_params_only() -> LevelParams {
    LevelParams::params_only(
        SisModulusFamily::Q128,
        64,
        3,
        2,
        4,
        3,
        SparseChallengeConfig::pm1_only(3),
    )
}

fn sample_layout_lp() -> LevelParams {
    sample_params_only().with_decomp(4, 2, 2, 2, 0).unwrap()
}

fn laid_out_sample_lp() -> LevelParams {
    sample_params_only()
        .with_layout(&sample_layout_lp(), 128)
        .unwrap()
}

fn sample_multi_group_root_params() -> (LevelParams, OpeningClaimsLayout) {
    use crate::schedule::PrecommittedGroupParams;
    let lp = sample_params_only()
        .with_layout(&sample_layout_lp(), 128)
        .unwrap();
    let precommit_lp = sample_params_only()
        .with_layout(&sample_layout_lp(), 128)
        .unwrap();
    let precommit = PrecommittedLevelParams {
        layout: PrecommittedGroupParams::from_params(
            PolynomialGroupLayout::new(4, 1),
            &precommit_lp,
        ),
        a_key: precommit_lp.a_key.clone(),
        b_key: AjtaiKeyParams::new_unchecked(
            precommit_lp.b_key.min_security_bits(),
            precommit_lp.b_key.sis_family(),
            5,
            precommit_lp.b_key.col_len(),
            precommit_lp.b_key.coeff_linf_bound(),
            precommit_lp.ring_dimension,
        ),
        num_blocks: precommit_lp.num_blocks,
        block_len: precommit_lp.block_len,
        num_digits_commit: precommit_lp.num_digits_commit,
        num_digits_open: precommit_lp.num_digits_open,
        num_digits_fold_one: precommit_lp.num_digits_fold_one,
    };
    let mut grouped = lp;
    grouped.precommitted_groups = vec![precommit];
    let batch = OpeningClaimsLayout::from_group_sizes(4, &[1, 1]).expect("layout");
    (grouped, batch)
}

#[test]
fn with_layout_keeps_self_ranks() {
    let params = sample_params_only();
    let layout_lp = sample_layout_lp();

    let lp = params.with_layout(&layout_lp, 128).unwrap();

    assert_eq!(lp.ring_dimension, 64);
    assert_eq!(lp.log_basis, layout_lp.log_basis);
    assert_eq!(lp.a_key.row_len(), 2);
    assert_eq!(lp.b_key.row_len(), 4);
    assert_eq!(lp.d_key.row_len(), 3);
    assert_eq!(lp.num_blocks, layout_lp.num_blocks);
    assert_eq!(lp.block_len, layout_lp.block_len);
    assert_eq!(lp.challenge_l1_mass(), 3);
    assert_eq!(lp.num_digits_commit, layout_lp.num_digits_commit);
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

    assert_eq!(lp.log_num_blocks(), layout_lp.r_vars);
    assert_eq!(lp.log_block_len(), layout_lp.m_vars);
    assert_eq!(lp.outer_vars(), layout_lp.m_vars + layout_lp.r_vars);
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

#[path = "tests/params_precommitted_group_tests.rs"]
mod precommitted_group_tests;
#[path = "tests/params_relation_row_tests.rs"]
mod relation_row_tests;
