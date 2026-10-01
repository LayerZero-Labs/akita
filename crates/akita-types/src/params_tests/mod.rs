use akita_challenges::SparseChallengeConfig;
use akita_params::layout::params::*;
use akita_params::test_fixtures::{
    sample_layout_lp, sample_multi_group_root_params, sample_params_only,
};
use akita_params::PolynomialGroupLayout;
use akita_params::{CommitmentRingDims, OpeningClaimsLayout};

#[test]
fn relation_mode_is_bound_immediately_after_payload_mode() {
    let quotient = sample_params_only();
    let mut reduced = quotient.clone();
    reduced.ring_relation_mode = akita_params::RingRelationMode::ReducedEvaluation;

    let quotient_descriptor = quotient.canonical_descriptor_bytes();
    let reduced_descriptor = reduced.canonical_descriptor_bytes();
    assert_eq!(quotient_descriptor[0], quotient.payload_mode.tag());
    assert_eq!(
        quotient_descriptor[1],
        akita_params::RingRelationMode::QuotientLift.tag()
    );
    assert_eq!(
        reduced_descriptor[1],
        akita_params::RingRelationMode::ReducedEvaluation.tag()
    );
    assert_ne!(quotient_descriptor, reduced_descriptor);
}

#[test]
fn distinct_semantic_depths_size_a_b_and_d_independently() {
    let mut params = sample_params_only();
    params.own_group_mut().profile.inner.digits.log_basis = 2;
    params.own_group_mut().profile.outer.digits.log_basis = 3;
    params.own_group_mut().opening.log_basis_open = 4;
    let params = params
        .with_decomp(8, 17, 5, 4, 3)
        .expect("distinct semantic decomposition");
    let blocks = 17usize.div_ceil(8);
    assert_eq!(
        params.inner().matrix.input_width(),
        8 * 5,
        "A uses inner depth"
    );
    assert_eq!(
        params.outer().matrix.input_width(),
        params.inner().matrix.output_rank() * 4 * blocks,
        "B uses outer depth"
    );
    assert_eq!(
        params.open().matrix.input_width(),
        3 * blocks,
        "D uses open depth"
    );
    assert_eq!(
        (
            params.inner().digits.log_basis,
            params.outer().digits.log_basis,
            params.open().digits.log_basis,
        ),
        (2, 3, 4)
    );
}

fn laid_out_sample_lp() -> CommittedGroupParams {
    sample_params_only()
        .with_layout(&sample_layout_lp())
        .unwrap()
}

#[test]
fn precommitted_challenge_l1_mass_counts_magnitude_two_coefficients_twice() {
    let (params, _) = sample_multi_group_root_params();
    let precommitted = &params.precommitted_groups()[0];

    assert_eq!(precommitted.opening.fold_challenge_config.weight(), 41);
    assert_eq!(precommitted.challenge_l1_mass(), 51);
}

#[test]
fn shared_d_digit_basis_uses_root_opening_basis() {
    let (mut grouped, _) = sample_multi_group_root_params();
    grouped.own_group_mut().opening.log_basis_open = 3;
    grouped
        .preceding_group_mut_for_test(0)
        .unwrap()
        .profile
        .outer
        .digits
        .log_basis = 6;

    assert_eq!(grouped.shared_d_digit_log_basis(), 3);
    assert_eq!(shared_d_digit_log_basis(5, &[]), 5);
}

#[test]
fn with_decomp_derives_exact_live_block_geometry() {
    let lp = sample_params_only().with_decomp(8, 17, 2, 2, 2).unwrap();

    assert_eq!(lp.blocks().live_ring_elements_per_claim, 17);
    assert_eq!(lp.blocks().positions_per_block, 8);
    assert_eq!(lp.blocks().live_blocks, 3);
    assert_eq!(lp.position_index_bits(), 3);
    assert_eq!(lp.block_index_bits(), 2);
    assert_eq!(lp.block_index_domain_size().unwrap(), 4);

    assert!(sample_params_only().with_decomp(3, 17, 2, 2, 2).is_err());
}

#[test]
fn with_layout_keeps_self_ranks() {
    let params = sample_params_only();
    let layout_lp = sample_layout_lp();

    let lp = params.with_layout(&layout_lp).unwrap();

    assert_eq!(lp.d_a(), 64);
    assert_eq!(
        lp.inner().digits.log_basis,
        layout_lp.inner().digits.log_basis
    );
    assert_eq!(
        lp.outer().digits.log_basis,
        layout_lp.outer().digits.log_basis
    );
    assert_eq!(
        lp.open().digits.log_basis,
        layout_lp.open().digits.log_basis
    );
    assert_eq!(lp.inner().matrix.output_rank(), 2);
    assert_eq!(lp.outer().matrix.output_rank(), 4);
    assert_eq!(lp.open().matrix.output_rank(), 3);
    assert_eq!(lp.blocks().live_blocks, layout_lp.blocks().live_blocks);
    assert_eq!(
        lp.blocks().positions_per_block,
        layout_lp.blocks().positions_per_block
    );
    assert_eq!(lp.challenge_l1_mass(), 3);
    assert_eq!(
        lp.inner().digits.num_digits,
        layout_lp.inner().digits.num_digits
    );
    assert_eq!(
        lp.outer().digits.num_digits,
        layout_lp.outer().digits.num_digits
    );
    assert_eq!(
        lp.open().digits.num_digits,
        layout_lp.open().digits.num_digits
    );
}

#[test]
fn derived_widths_match_ajtai_col_len() {
    let lp = sample_params_only()
        .with_layout(&sample_layout_lp())
        .unwrap();

    assert_eq!(lp.inner_width(), lp.inner().matrix.input_width());
    assert_eq!(lp.outer_width(), lp.outer().matrix.input_width());
    assert_eq!(lp.d_matrix_width(), lp.open().matrix.input_width());
}

#[test]
fn derived_log_values() {
    let layout_lp = sample_layout_lp();
    let lp = sample_params_only().with_layout(&layout_lp).unwrap();

    assert_eq!(lp.block_index_bits(), layout_lp.block_index_bits());
    assert_eq!(lp.position_index_bits(), layout_lp.position_index_bits());
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
    let n_a = lp.inner().matrix.output_rank();
    let n_b = lp.outer().matrix.output_rank();
    let n_d = lp.open().matrix.output_rank();

    for nc in [1usize, 2, 4] {
        let n_d_active = n_d;
        let a_start = 1;
        let b_start = a_start + n_a;
        let d_start = b_start + n_b * nc;

        assert_eq!(lp.a_start(), a_start);
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
