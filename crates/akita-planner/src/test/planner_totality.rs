use super::*;

use akita_config::{
    honest_fold_policy_of, policy_of,
    proof_optimized::{fp128::DenseMultiChunk, fp64::Dense},
    CommitmentConfig,
};

fn root_contracts(schedule: &PlannedFoldSchedule, field_bits: u32) -> bool {
    let root = &schedule.schedule.root;
    root.output_witness_len * (root.params.open().digits.log_basis as usize)
        < root.input_witness_len * field_bits as usize
}

fn root_candidate_classes<Cfg: CommitmentConfig>(
    num_vars: usize,
) -> Result<(bool, bool), AkitaError> {
    let policy = policy_of::<Cfg>();
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(num_vars));
    let input_bits = (1usize << num_vars) * policy.decomposition.field_bits() as usize;
    let mut has_contractive = false;
    let mut has_noncontractive = false;
    for dimensions in crate::schedule_params::dimension_candidates(
        &policy,
        0,
        crate::schedule_params::initial_dimension_ceiling(&policy)?,
    )? {
        for opening in PlannerOpeningCandidate::coefficient_packing_domain(
            0,
            policy.claim_ext_degree,
            dimensions,
        )? {
            for inner_basis in Cfg::inner_basis_range().0..=Cfg::inner_basis_range().1 {
                for opening_basis in Cfg::opening_basis_range().0..=Cfg::opening_basis_range().1 {
                    for (params, output_witness_len) in root_level_candidates_for_basis(
                        &key,
                        honest_fold_policy_of::<Cfg>(),
                        &[],
                        &policy,
                        dimensions,
                        opening,
                        &[],
                        inner_basis,
                        opening_basis,
                        None,
                    )? {
                        let contracts = output_witness_len
                            * (params.open().digits.log_basis as usize)
                            < input_bits;
                        has_contractive |= contracts;
                        has_noncontractive |= !contracts;
                    }
                    if has_contractive && has_noncontractive {
                        return Ok((true, true));
                    }
                }
            }
        }
    }
    Ok((has_contractive, has_noncontractive))
}

#[test]
fn contractive_winner_remains_selected() {
    let policy = policy_of::<Dense>();
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(14));
    let schedule = find_schedule(
        &key,
        honest_fold_policy_of::<Dense>(),
        &[],
        &policy,
        Dense::ring_challenge_config,
    )
    .expect("valid scalar root");

    assert!(root_contracts(&schedule, policy.decomposition.field_bits()));
    for fold in &schedule.schedule.recursive_folds {
        assert!(
            fold.output_witness_len < fold.input_witness_len,
            "recursive folds must preserve strict progress"
        );
    }
}

#[test]
fn noncontractive_root_is_selected_by_the_complete_policy() {
    let policy = policy_of::<Dense>();
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(9));
    let schedule = find_schedule(
        &key,
        honest_fold_policy_of::<Dense>(),
        &[],
        &policy,
        Dense::ring_challenge_config,
    )
    .expect("valid scalar root");

    assert!(
        !root_contracts(&schedule, policy.decomposition.field_bits()),
        "root contraction is not a complete selection coordinate"
    );
}

#[test]
fn noncontractive_multi_chunk_root_can_beat_contractive_candidates() {
    let policy = policy_of::<DenseMultiChunk>();
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(16));
    let schedule = find_schedule(
        &key,
        honest_fold_policy_of::<DenseMultiChunk>(),
        &[],
        &policy,
        DenseMultiChunk::ring_challenge_config,
    )
    .expect("valid multi-chunk root");
    assert_eq!(
        root_candidate_classes::<DenseMultiChunk>(16).unwrap(),
        (true, true)
    );
    assert!(
        !root_contracts(&schedule, policy.decomposition.field_bits()),
        "the actual complete objective must be allowed to select the better noncontractive root"
    );
}

#[test]
fn valid_small_scalar_root_has_a_schedule() {
    let policy = policy_of::<Dense>();
    for num_vars in 8..=9 {
        let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(num_vars));
        let schedule = find_schedule(
            &key,
            honest_fold_policy_of::<Dense>(),
            &[],
            &policy,
            Dense::ring_challenge_config,
        )
        .unwrap_or_else(|error| panic!("valid nv={num_vars} D64-root request: {error}"));

        if num_vars == 8 {
            assert_eq!(root_candidate_classes::<Dense>(8).unwrap(), (false, true));
            let root = &schedule.schedule.root;
            assert!(
                root.output_witness_len * root.params.open().digits.log_basis as usize
                    >= root.input_witness_len * policy.decomposition.field_bits() as usize,
                "the regression must exercise the previously rejected noncontractive root"
            );
            let cleartext_source_bytes =
                (1usize << num_vars) * (policy.decomposition.field_bits() as usize).div_ceil(8);
            assert!(
                schedule
                    .estimate
                    .estimated_proof_payload_bytes()
                    .expect("selected proof size")
                    > cleartext_source_bytes,
                "planner totality must not depend on beating cleartext transmission"
            );
        }
        schedule
            .schedule
            .validate_structure()
            .expect("the selected schedule must pass structural validation");
    }
}

#[test]
fn valid_small_grouped_root_has_a_schedule() {
    let precommitted_group = PolynomialGroupLayout::singleton(16);
    let policy = policy_of::<Dense>();
    let producer_key = AkitaScheduleLookupKey::single(precommitted_group);
    let producer_dimensions = CommitmentRingDims {
        inner: 128,
        outer: 64,
        opening: 64,
    };
    let producer_opening = PlannerOpeningCandidate::coefficient_packing(
        0,
        policy.claim_ext_degree,
        producer_dimensions,
        64,
    )
    .expect("valid D64 producer opening request")
    .expect("D64 producer opening");
    let producer = root_level_candidates_for_basis(
        &producer_key,
        honest_fold_policy_of::<Dense>(),
        &[],
        &policy,
        producer_dimensions,
        producer_opening,
        &[],
        Dense::inner_basis_range().0,
        Dense::opening_basis_range().0,
        None,
    )
    .expect("scalar producer candidates")
    .into_iter()
    .next()
    .expect("scalar producer candidate")
    .0;
    let precommitted_profile =
        GroupCommitPhaseParams::try_from_params(precommitted_group, &producer)
            .expect("scalar producer profile");
    let key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::singleton(8),
        precommitteds: vec![precommitted_profile],
    };
    let schedule = find_schedule(
        &key,
        honest_fold_policy_of::<Dense>(),
        &[honest_fold_policy_of::<Dense>()],
        &policy,
        Dense::ring_challenge_config,
    )
    .expect("a valid grouped D64-root request must have a schedule");

    assert_eq!(schedule.schedule.root.params.precommitted_groups().len(), 1);
    assert!(
        !schedule.schedule.recursive_folds.is_empty(),
        "a grouped root must retain its required child fold"
    );
    schedule
        .schedule
        .validate_structure()
        .expect("the grouped schedule must pass structural validation");
}
