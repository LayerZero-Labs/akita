use super::*;

fn use_expected_producer_encodings(schedule: &mut FoldSchedule, extension_degree: usize) {
    let root = &mut schedule.root.params.final_group.commitment;
    root.source_encoding = crate::CommittedSourceEncoding::for_producer(
        root.opening_method,
        extension_degree,
        root.d_a(),
        schedule.root.input_witness_len.trailing_zeros() as usize,
        true,
    );
    for step in &mut schedule.recursive_folds {
        let witness = &mut step.params.witness;
        witness.source_encoding = crate::CommittedSourceEncoding::for_producer(
            witness.opening_method,
            extension_degree,
            witness.d_a(),
            0,
            false,
        );
    }
}

#[test]
fn accepts_packing_geometry_for_every_extension_degree() {
    for (extension_degree, d_a) in [(1, 64), (2, 128), (4, 256)] {
        let mut schedule = recursive_schedule(d_a, d_a, false);
        let root = &mut schedule.root.params.final_group.commitment;
        root.opening_method = crate::OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: 64,
        };
        root.fold_challenge_config =
            akita_challenges::SparseChallengeConfig::production_for_ring_dim(64).unwrap();
        use_expected_producer_encodings(&mut schedule, extension_degree);
        schedule
            .validate_nonterminal_opening_execution(extension_degree)
            .expect("packing geometry should be executable");
    }
}

#[test]
fn accepts_group_local_packing_subring_dimensions() {
    let packing_group = |s| {
        let mut params = committed_params(128);
        params.opening_method = crate::OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: s,
        };
        params.fold_challenge_config =
            akita_challenges::SparseChallengeConfig::production_for_ring_dim(s).unwrap();
        params.source_encoding = crate::CommittedSourceEncoding::CanonicalCoefficientTable;
        params
    };
    let s64 = packing_group(64);
    let s128 = packing_group(128);
    let groups = [
        OpeningExecutionGroup {
            params: &s64,
            expected_source_encoding: None,
        },
        OpeningExecutionGroup {
            params: &s128,
            expected_source_encoding: None,
        },
    ];
    validate_level_opening_execution(0, 1, &groups)
        .expect("each root group owns its challenge subring dimension");
}

#[test]
fn accepts_level_one_packing() {
    let mut schedule = recursive_schedule(128, 128, false);
    let recursive = &mut schedule.recursive_folds[0].params.witness;
    recursive.opening_method = crate::OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: 64,
    };
    recursive.fold_challenge_config =
        akita_challenges::SparseChallengeConfig::production_for_ring_dim(64).unwrap();
    use_expected_producer_encodings(&mut schedule, 2);
    schedule
        .validate_nonterminal_opening_execution(2)
        .expect("absolute level one packing should be executable");
}

#[test]
fn rejects_level_two_packing() {
    let mut schedule = recursive_schedule(128, 128, false);
    append_recursive_fold(&mut schedule);
    let recursive = &mut schedule.recursive_folds[1].params.witness;
    recursive.opening_method = crate::OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: 64,
    };
    recursive.fold_challenge_config =
        akita_challenges::SparseChallengeConfig::production_for_ring_dim(64).unwrap();
    use_expected_producer_encodings(&mut schedule, 2);
    assert!(matches!(
        schedule.validate_nonterminal_opening_execution(2),
        Err(AkitaError::InvalidSetup(message)) if message.contains("levels 0 and 1")
    ));
}

#[test]
fn rejects_subring_that_ignores_extension_degree() {
    let mut schedule = recursive_schedule(128, 128, false);
    let root = &mut schedule.root.params.final_group.commitment;
    root.opening_method = crate::OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: 64,
    };
    root.fold_challenge_config =
        akita_challenges::SparseChallengeConfig::production_for_ring_dim(64).unwrap();
    use_expected_producer_encodings(&mut schedule, 4);
    assert!(schedule.validate_nonterminal_opening_execution(4).is_err());
}

#[test]
fn rejects_unaudited_recursive_packing_family() {
    let mut schedule = recursive_schedule(64, 64, false);
    schedule.recursive_folds[0].params.witness.opening_method =
        crate::OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: 64,
        };
    schedule.recursive_folds[0]
        .params
        .witness
        .fold_challenge_config = akita_challenges::SparseChallengeConfig::pm1_only(1);
    assert!(schedule.validate_nonterminal_opening_execution(1).is_err());
}

#[test]
fn rejects_packing_over_tensor_projected_source() {
    let mut schedule = recursive_schedule(128, 128, false);
    let root = &mut schedule.root.params.final_group.commitment;
    root.opening_method = crate::OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: 64,
    };
    root.source_encoding = crate::CommittedSourceEncoding::TensorSubfieldProjection {
        extension_degree: 2,
    };
    root.fold_challenge_config =
        akita_challenges::SparseChallengeConfig::production_for_ring_dim(64).unwrap();
    use_expected_producer_encodings(&mut schedule, 2);
    schedule.root.params.final_group.commitment.source_encoding =
        crate::CommittedSourceEncoding::TensorSubfieldProjection {
            extension_degree: 2,
        };
    let error = schedule
        .validate_nonterminal_opening_execution(2)
        .expect_err("packing over a tensor source must reject");
    assert!(
        matches!(
            &error,
            AkitaError::InvalidSetup(message)
                if message.contains("canonical coefficient source encoding")
        ),
        "unexpected rejection: {error:?}",
    );
}

#[test]
fn rejects_active_root_tensor_gate_mutated_to_canonical() {
    let mut schedule = recursive_schedule(128, 128, false);
    use_expected_producer_encodings(&mut schedule, 2);
    schedule.root.params.final_group.commitment.source_encoding =
        crate::CommittedSourceEncoding::CanonicalCoefficientTable;

    assert!(matches!(
        schedule.validate_nonterminal_opening_execution(2),
        Err(AkitaError::InvalidSetup(message)) if message.contains("producer geometry")
    ));
}

#[test]
fn rejects_root_tensor_encoding_below_the_tensor_arity_gate() {
    let mut schedule = recursive_schedule(128, 128, false);
    schedule.root.input_witness_len = 64;
    use_expected_producer_encodings(&mut schedule, 2);
    schedule.root.params.final_group.commitment.source_encoding =
        crate::CommittedSourceEncoding::TensorSubfieldProjection {
            extension_degree: 2,
        };

    assert!(matches!(
        schedule.validate_nonterminal_opening_execution(2),
        Err(AkitaError::InvalidSetup(message)) if message.contains("producer geometry")
    ));
}

#[test]
fn rejects_extension_recursive_trace_mutated_to_canonical() {
    let mut schedule = recursive_schedule(128, 128, false);
    use_expected_producer_encodings(&mut schedule, 2);
    schedule.recursive_folds[0].params.witness.source_encoding =
        crate::CommittedSourceEncoding::CanonicalCoefficientTable;

    assert!(matches!(
        schedule.validate_nonterminal_opening_execution(2),
        Err(AkitaError::InvalidSetup(message)) if message.contains("producer geometry")
    ));
}

#[test]
fn rejects_tensor_degree_that_does_not_fit_half_the_a_ring() {
    let mut schedule = recursive_schedule(128, 128, false);
    schedule.root.params.final_group.commitment.source_encoding =
        crate::CommittedSourceEncoding::TensorSubfieldProjection {
            extension_degree: 128,
        };

    assert!(matches!(
        schedule.validate_structure(),
        Err(AkitaError::InvalidSetup(message)) if message.contains("half the A ring dimension")
    ));
}
