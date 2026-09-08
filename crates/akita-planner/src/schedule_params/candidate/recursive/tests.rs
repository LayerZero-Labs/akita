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
            guide: None,
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
fn guided_search_forces_a_split_outside_the_bounded_domain() {
    use akita_config::{
        policy_of, proof_optimized::fp128::OneHot, CommitmentConfig, RecursiveCommitmentConfig,
    };

    type Recursive = RecursiveCommitmentConfig<OneHot>;
    let policy = policy_of::<Recursive>();
    let base_request = RecursiveCandidateRequest {
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
        guide: None,
    };
    let search = prepare_recursive_level_search(&base_request, RecursiveSetupPrefix::None)
        .expect("recursive search")
        .expect("eligible recursive level");
    let delta_commit = base_request
        .source
        .num_digits_inner(policy.decomposition, base_request.log_basis_inner)
        .expect("inner digit count");
    let delta_open = num_digits_open(DecompositionParams {
        log_basis: base_request.log_basis_open,
        ..policy.decomposition
    });
    let bounded = recursive_split_search_domain(
        policy.recursive_split_search_policy,
        search.num_ring_elems,
        search.reduced_vars,
        delta_commit,
        delta_open,
        search.num_chunks,
    );
    let guided_split = (1..search.reduced_vars)
        .find(|split| !bounded.contains(split))
        .expect("bounded domain must omit a split");
    let request = RecursiveCandidateRequest {
        guide: Some(CandidateLayoutGuide {
            position_index_bits: search.reduced_vars - guided_split,
            outer_slice_count: akita_types::CommitmentSliceCount::ONE,
            inner_route: CandidateInnerRoute::Linf,
            setup_prefix: None,
        }),
        ..base_request
    };
    let context = RecursiveCandidateContext {
        request: &request,
        search: &search,
        source_moment: request.source_moment,
        successor_policy: SuccessorPolicy::RequireContraction,
    };
    let mut admitted = Vec::new();
    context
        .walk_splits(
            RelationSearchDomain::QuotientOnly,
            |split, _| {
                admitted.push(split);
                false
            },
            |_, _, _, _| unreachable!("rejected split must not materialize"),
        )
        .expect("guided split walk");
    assert_eq!(admitted, vec![guided_split]);
}

#[test]
fn guided_recursive_l2_survives_the_profitability_filter() {
    use akita_config::{
        policy_of, proof_optimized::fp128::OneHot, CommitmentConfig, RecursiveCommitmentConfig,
    };
    use akita_types::InnerCommitSecurityRoute;

    type Recursive = RecursiveCommitmentConfig<OneHot>;
    let policy = policy_of::<Recursive>();
    for source_bound in [
        1_000,
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        100_000_000,
        1_000_000_000,
        10_000_000_000,
        100_000_000_000,
        1_000_000_000_000,
    ] {
        for log_basis in 2..=8 {
            let request = RecursiveCandidateRequest {
                policy: &policy,
                payload_mode: akita_types::CommitmentPayloadMode::Compressed,
                opening: PlannerOpeningCandidate::evaluation_trace(
                    Recursive::ring_challenge_config(64).expect("challenge config"),
                ),
                dimensions: CommitmentRingDims::uniform(64),
                current_witness_len: 948_672,
                source: crate::InnerBasisSource::BalancedDigits { log_basis },
                log_basis_inner: log_basis,
                log_basis_open: log_basis,
                fold_level: 3,
                source_moment: crate::response_model::SourceMomentEstimate::new(source_bound),
                relation_traversal_order: RelationTraversalOrder::Canonical,
                guide: None,
            };
            let linf_candidates = derive_fold_candidates(
                request,
                RecursiveFoldWork::direct(RelationSearchDomain::QuotientOnly),
                FoldCandidatePolicy::Frontier(SplitBoundPolicy::Enabled),
            )
            .expect("unguided recursive candidates");
            for (linf, _) in linf_candidates.iter().filter(|(candidate, _)| {
                matches!(
                    candidate.inner().matrix.security_route(),
                    InnerCommitSecurityRoute::Linf(_)
                )
            }) {
                let guided_request = RecursiveCandidateRequest {
                    guide: Some(CandidateLayoutGuide {
                        position_index_bits: linf.blocks().position_index_bits(),
                        outer_slice_count: linf.outer_slice_count(),
                        inner_route: CandidateInnerRoute::L2,
                        setup_prefix: None,
                    }),
                    ..request
                };
                let guided = derive_fold_candidates(
                    guided_request,
                    RecursiveFoldWork::direct(RelationSearchDomain::QuotientOnly),
                    FoldCandidatePolicy::Best,
                )
                .expect("guided recursive candidates");
                let guided_linf_rank = guided.iter().find_map(|(candidate, _)| {
                    matches!(
                        candidate.inner().matrix.security_route(),
                        InnerCommitSecurityRoute::Linf(_)
                    )
                    .then_some(candidate.inner().matrix.output_rank())
                });
                let guided_l2_rank = guided.iter().find_map(|(candidate, _)| {
                    matches!(
                        candidate.inner().matrix.security_route(),
                        InnerCommitSecurityRoute::L2 { .. }
                    )
                    .then_some(candidate.inner().matrix.output_rank())
                });
                if let (Some(linf_rank), Some(l2_rank)) = (guided_linf_rank, guided_l2_rank) {
                    if l2_rank >= linf_rank {
                        return;
                    }
                }
            }
        }
    }
    panic!("fixture must contain a feasible L2 route that is not the greedy rank winner");
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
        guide: None,
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
        guide: None,
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
                guide: None,
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
        guide: None,
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
