use super::*;

#[test]
fn combined_terminal_and_fold_views_match_independent_searches() {
    use akita_config::{
        policy_of, proof_optimized::fp128::OneHot, CommitmentConfig, RecursiveCommitmentConfig,
    };
    use akita_types::InnerCommitSecurityRoute;

    type Recursive = RecursiveCommitmentConfig<OneHot>;
    let policy = policy_of::<Recursive>();
    let opening = PlannerOpeningCandidate::evaluation_trace(
        Recursive::ring_challenge_config(64).expect("challenge config"),
    );
    let dimensions = CommitmentRingDims::uniform(64);
    let source = crate::InnerBasisSource::BalancedDigits { log_basis: 4 };
    let source_moment = crate::response_model::SourceMomentEstimate::new(1_000_000);
    for retain_split_frontier in [false, true] {
        let request = RecursiveCandidateRequest {
            policy: &policy,
            payload_mode: akita_types::CommitmentPayloadMode::Compressed,
            opening,
            dimensions,
            current_witness_len: 948_672,
            source,
            log_basis_inner: 4,
            log_basis_open: 4,
            fold_level: 3,
            source_moment,
            relation_traversal_order: RelationTraversalOrder::Canonical,
        };
        let fold_policy = if retain_split_frontier {
            FoldCandidatePolicy::Frontier(SplitBoundPolicy::Enabled)
        } else {
            FoldCandidatePolicy::Best
        };
        let expected_terminal = derive_terminal_candidates(request).expect("terminal search");
        let expected_folds = derive_fold_candidates(
            request,
            RecursiveFoldWork::direct(RelationSearchDomain::QuotientOnly),
            fold_policy,
        )
        .expect("fold search");
        let actual = derive_recursive_candidate_views(
            request,
            fold_policy,
            RelationSearchDomain::QuotientOnly,
        )
        .expect("combined search");

        assert_eq!(actual.terminal, expected_terminal);
        assert_eq!(actual.folds, expected_folds);
        assert!(actual.folds.iter().any(|(candidate, _)| matches!(
            candidate.inner().matrix.security_route(),
            InnerCommitSecurityRoute::L2 { .. }
        )));
    }
}

#[test]
fn combined_relation_views_match_mode_specific_searches() {
    use akita_config::{
        policy_of, proof_optimized::fp128::OneHot, CommitmentConfig, RecursiveCommitmentConfig,
    };
    use akita_types::RingRelationMode::{QuotientLift, ReducedEvaluation};

    type Recursive = RecursiveCommitmentConfig<OneHot>;
    let policy = policy_of::<Recursive>();
    let request = RecursiveCandidateRequest {
        policy: &policy,
        payload_mode: akita_types::CommitmentPayloadMode::Compressed,
        opening: PlannerOpeningCandidate::evaluation_trace(
            Recursive::ring_challenge_config(64).expect("challenge config"),
        ),
        dimensions: CommitmentRingDims::uniform(64),
        current_witness_len: 948_672,
        source: crate::InnerBasisSource::BalancedDigits { log_basis: 4 },
        log_basis_inner: 4,
        log_basis_open: 4,
        fold_level: 3,
        source_moment: crate::response_model::SourceMomentEstimate::new(1_000_000),
        relation_traversal_order: RelationTraversalOrder::Canonical,
    };
    let relation_domain = RelationSearchDomain::for_topology(
        RingRelationPhase::QuotientPrefix,
        request.fold_level,
        RelationCandidateTopology::DirectEvaluationTrace,
        None,
    )
    .expect("direct relation transitions");
    let expected = [QuotientLift, ReducedEvaluation]
        .into_iter()
        .flat_map(|mode| {
            derive_fold_candidates(
                request,
                RecursiveFoldWork::direct(RelationSearchDomain::for_mode(mode)),
                FoldCandidatePolicy::Best,
            )
            .expect("mode-specific fold search")
        })
        .map(|(candidate, next)| (candidate.canonical_descriptor_bytes(), next))
        .collect::<std::collections::BTreeSet<_>>();
    let actual =
        derive_recursive_candidate_views(request, FoldCandidatePolicy::Best, relation_domain)
            .expect("shared relation search");
    let actual_folds = actual
        .folds
        .into_iter()
        .map(|(candidate, next)| (candidate.canonical_descriptor_bytes(), next))
        .collect::<std::collections::BTreeSet<_>>();

    assert_eq!(actual_folds, expected);
    assert!(actual
        .terminal
        .iter()
        .all(|params| params.ring_relation_mode == QuotientLift));
}

#[test]
fn reduced_only_views_keep_quotient_terminal_and_exclusively_reduced_folds() {
    use akita_config::{
        policy_of, proof_optimized::fp128::OneHot, CommitmentConfig, RecursiveCommitmentConfig,
    };
    use akita_types::RingRelationMode::{QuotientLift, ReducedEvaluation};

    type Recursive = RecursiveCommitmentConfig<OneHot>;
    let policy = policy_of::<Recursive>();
    let request = RecursiveCandidateRequest {
        policy: &policy,
        payload_mode: akita_types::CommitmentPayloadMode::Compressed,
        opening: PlannerOpeningCandidate::evaluation_trace(
            Recursive::ring_challenge_config(64).expect("challenge config"),
        ),
        dimensions: CommitmentRingDims::uniform(64),
        current_witness_len: 948_672,
        source: crate::InnerBasisSource::BalancedDigits { log_basis: 4 },
        log_basis_inner: 4,
        log_basis_open: 4,
        fold_level: 3,
        source_moment: crate::response_model::SourceMomentEstimate::new(1_000_000),
        relation_traversal_order: RelationTraversalOrder::Canonical,
    };
    let relation_domain = RelationSearchDomain::ReducedOnly;
    for fold_policy in [
        FoldCandidatePolicy::Best,
        FoldCandidatePolicy::Frontier(SplitBoundPolicy::Enabled),
    ] {
        let views = derive_recursive_candidate_views(request, fold_policy, relation_domain)
            .expect("reduced-only combined search");
        assert!(!views.terminal.is_empty());
        assert!(views
            .terminal
            .iter()
            .all(|params| params.ring_relation_mode == QuotientLift));
        assert!(!views.folds.is_empty());
        assert!(views.folds.iter().all(|(candidate, _)| {
            candidate.ring_relation_mode == ReducedEvaluation
                && relation_domain.admits(candidate.ring_relation_mode)
        }));
    }
}

#[test]
fn combined_views_keep_a_noncontracting_terminal_candidate() {
    use akita_config::{
        policy_of, proof_optimized::fp128::OneHot, CommitmentConfig, RecursiveCommitmentConfig,
    };

    type Recursive = RecursiveCommitmentConfig<OneHot>;
    let policy = policy_of::<Recursive>();
    let opening = PlannerOpeningCandidate::evaluation_trace(
        Recursive::ring_challenge_config(64).expect("challenge config"),
    );
    let source = crate::InnerBasisSource::BalancedDigits { log_basis: 4 };
    let mut witnessed_boundary = false;
    for current_witness_len in [1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 16] {
        let views = derive_recursive_candidate_views(
            RecursiveCandidateRequest {
                policy: &policy,
                payload_mode: akita_types::CommitmentPayloadMode::Raw,
                opening,
                dimensions: CommitmentRingDims::uniform(64),
                current_witness_len,
                source,
                log_basis_inner: 4,
                log_basis_open: 4,
                fold_level: 2,
                source_moment: None,
                relation_traversal_order: RelationTraversalOrder::Canonical,
            },
            FoldCandidatePolicy::Best,
            RelationSearchDomain::QuotientOnly,
        )
        .expect("combined search");
        if !views.terminal.is_empty() && views.folds.is_empty() {
            witnessed_boundary = true;
            break;
        }
    }
    assert!(
        witnessed_boundary,
        "the fixture must exercise a terminal winner rejected by fold contraction"
    );
}

#[test]
fn late_consumer_keeps_setup_prefix_slices_eligible() {
    use akita_config::{
        policy_of, proof_optimized::fp128::OneHot, CommitmentConfig, RecursiveCommitmentConfig,
    };

    type Recursive = RecursiveCommitmentConfig<OneHot>;
    let policy = policy_of::<Recursive>();
    let challenge = Recursive::ring_challenge_config(64).expect("challenge config");
    let mut cache = SetupPrefixSearchCache::default();
    let request = RecursiveCandidateRequest {
        policy: &policy,
        payload_mode: akita_types::CommitmentPayloadMode::Raw,
        opening: PlannerOpeningCandidate::evaluation_trace(challenge),
        dimensions: CommitmentRingDims::uniform(64),
        current_witness_len: 1 << 16,
        source: crate::InnerBasisSource::BalancedDigits { log_basis: 4 },
        log_basis_inner: 4,
        log_basis_open: 4,
        fold_level: 2,
        source_moment: None,
        relation_traversal_order: RelationTraversalOrder::Canonical,
    };
    let search = prepare_recursive_level_search(
        &request,
        RecursiveSetupPrefix::Search {
            cache: &mut cache,
            natural_len: 1 << 12,
        },
    )
    .expect("late consumer search")
    .expect("eligible recursive level");

    assert!(search
        .setup_prefixes
        .iter()
        .flatten()
        .any(|slot| { slot.profile.outer_slice_count > akita_types::CommitmentSliceCount::ONE }));
}
