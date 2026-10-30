use super::*;
use crate::test_fixtures::committed_params;

#[test]
fn accepts_group_local_packing_subring_dimensions() {
    let packing_group = |s| {
        let mut params = committed_params(128);
        params.own_group_mut().opening.opening_method =
            crate::OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension: s,
            };
        params.own_group_mut().opening.fold_challenge_config =
            akita_challenges::FoldChallengeFamily::production_for_ring_dim(s).unwrap();
        params.source_encoding = crate::CommittedSourceEncoding::CanonicalCoefficientTable;
        params
    };
    let s64 = packing_group(64);
    let s128 = packing_group(128);
    let groups = [
        OpeningExecutionGroup {
            opening_method: s64.opening_method(),
            inner_commit_matrix: &s64.inner().matrix,
            fold_challenge_config: s64.fold_challenge_config(),
            source_encoding: s64.source_encoding,
            expected_source_encoding: None,
        },
        OpeningExecutionGroup {
            opening_method: s128.opening_method(),
            inner_commit_matrix: &s128.inner().matrix,
            fold_challenge_config: s128.fold_challenge_config(),
            source_encoding: s128.source_encoding,
            expected_source_encoding: None,
        },
    ];
    validate_level_opening_execution(0, 2, 1, &groups)
        .expect("each root group owns its challenge subring dimension");
    validate_level_opening_execution(2, 8, 1, &groups)
        .expect("the third multi-chunk fold must admit subring packing");
    for level in [0, 1, 2, 3] {
        validate_level_opening_execution(level, 1, 1, &groups)
            .expect("single-chunk folds also admit subring packing");
    }
}

#[test]
fn rejects_evaluation_trace_at_the_third_multi_chunk_fold() {
    let params = committed_params(128);
    assert_eq!(params.opening_method(), OpeningMethod::EvaluationTrace);
    let groups = [OpeningExecutionGroup {
        opening_method: params.opening_method(),
        inner_commit_matrix: &params.inner().matrix,
        fold_challenge_config: params.fold_challenge_config(),
        source_encoding: params.source_encoding,
        expected_source_encoding: None,
    }];
    for level in [0, 1] {
        validate_level_opening_execution(level, 1, 1, &groups)
            .expect_err("the first two folds require packing even without chunking");
    }
    validate_level_opening_execution(2, 1, 1, &groups)
        .expect("the single-chunk suffix uses evaluation trace");
    let error = validate_level_opening_execution(0, 1, 4, &groups)
        .expect_err("canonical extension-field roots require packing");
    assert!(error
        .to_string()
        .contains("requires subring coefficient packing"));
    let error = validate_level_opening_execution(2, 8, 1, &groups)
        .expect_err("multi-chunk evaluation trace must be rejected");
    assert!(error
        .to_string()
        .contains("requires subring coefficient packing"));
    validate_level_opening_execution(3, 1, 1, &groups)
        .expect("the suffix after three multi-chunk folds uses evaluation trace");
}

#[test]
fn packing_requirement_covers_the_first_two_folds_and_all_multi_chunk_folds() {
    for level in 0..5 {
        assert_eq!(fold_requires_coefficient_packing(1, level), level < 2);
        assert!(fold_requires_coefficient_packing(8, level));
    }
}
