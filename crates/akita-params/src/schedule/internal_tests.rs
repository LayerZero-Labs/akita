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
            akita_challenges::SparseChallengeConfig::production_for_ring_dim(s).unwrap();
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
    validate_level_opening_execution(0, 1, &groups)
        .expect("each root group owns its challenge subring dimension");
}
