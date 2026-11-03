use super::*;
use crate::diagnostics::{capture, PlannerDiagnosticsSnapshot};

fn plan_with_diagnostics<C: akita_config::CommitmentConfig>(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
) -> (PlannedFoldSchedule, PlannerDiagnosticsSnapshot) {
    let (planned, snapshot) = capture(true, || {
        find_schedule(
            key,
            policy,
            C::committed_source_contract().unwrap(),
            C::ring_challenge_config,
        )
    });
    (
        planned.expect("setup search"),
        snapshot.expect("enabled capture"),
    )
}

/// Plan `key` with widening setup budgets and with one pass whose explicit
/// budget never binds, and require the same schedule.
fn assert_budgeted_search_matches_unbounded_pass<C: akita_config::CommitmentConfig>(
    policy: PlannerPolicy,
    key: PolynomialGroupLayout,
    label: &str,
) {
    assert_eq!(policy.setup_field_budget, None, "{label}");
    assert_eq!(
        policy.selection_policy,
        crate::SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenExactProofAndWorkV6,
        "{label}"
    );
    let mut unbounded = policy;
    unbounded.setup_field_budget = Some(usize::MAX);
    let (budgeted, budgeted_passes) = plan_with_diagnostics::<C>(key, &policy);
    let (single, single_passes) = plan_with_diagnostics::<C>(key, &unbounded);

    assert_eq!(
        budgeted.schedule.canonical_descriptor_bytes(),
        single.schedule.canonical_descriptor_bytes(),
        "{label}"
    );
    assert_eq!(budgeted.estimate, single.estimate, "{label}");

    let budgets = budgeted_passes
        .setup_budget_passes
        .iter()
        .map(|pass| pass.budget)
        .collect::<Vec<_>>();
    let padded_setup =
        akita_params::padded_setup_prefix_len(budgeted.estimate.estimated_num_setup_field_elements);
    let (last, earlier) = budgets.split_last().expect("at least one pass");
    assert_eq!(*last, Some(padded_setup), "{label}: {budgets:?}");
    assert!(
        earlier.windows(2).all(|pair| pair[0]
            .zip(pair[1])
            .is_some_and(|(left, right)| left < right)),
        "{label}: {budgets:?}"
    );
    assert!(
        earlier
            .iter()
            .all(|budget| budget.is_some_and(|budget| budget < padded_setup)),
        "{label}: {budgets:?}"
    );
    assert_eq!(
        single_passes
            .setup_budget_passes
            .iter()
            .map(|pass| pass.budget)
            .collect::<Vec<_>>(),
        [Some(usize::MAX)],
        "{label}"
    );
}

#[test]
fn padded_setup_budget_passes_match_one_unbounded_pass() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, RecursiveCommitmentConfig};

    type Recursive = RecursiveCommitmentConfig<OneHot>;

    // Restricted domains from the unpruned-oracle fixtures. The oracle itself
    // does not model setup-prefix offloading, so it cannot rank V6 schedules.
    let oracle_domains = [
        RingDimensionSearchDomain::new([
            CommitmentRingDims::uniform(64),
            CommitmentRingDims {
                inner: 128,
                outer: 64,
                opening: 64,
            },
        ])
        .unwrap(),
        RingDimensionSearchDomain::new([
            CommitmentRingDims::uniform(64),
            CommitmentRingDims {
                inner: 128,
                outer: 64,
                opening: 64,
            },
            CommitmentRingDims {
                inner: 128,
                outer: 128,
                opening: 64,
            },
        ])
        .unwrap(),
        RingDimensionSearchDomain::new([
            CommitmentRingDims::uniform(64),
            CommitmentRingDims {
                inner: 128,
                outer: 64,
                opening: 64,
            },
            CommitmentRingDims {
                inner: 128,
                outer: 64,
                opening: 128,
            },
        ])
        .unwrap(),
    ];
    for (domain_index, domain) in oracle_domains.iter().enumerate() {
        let mut policy = policy_for_domain(
            policy_of::<Recursive>(akita_params::ChunkedWitnessCfg::default_non_chunked()),
            domain,
        );
        policy.selective_l2_response_model = crate::SelectiveL2ResponseModelId::Disabled;
        for num_vars in [14, 16, 18] {
            assert_budgeted_search_matches_unbounded_pass::<Recursive>(
                policy,
                onehot_group(num_vars, 1),
                &format!("one-hot domain {domain_index} nv={num_vars}"),
            );
        }
    }
}

// The drift job already checks every V6 catalog row against artifacts that the
// unbudgeted search produced. Only the unbounded comparison passes are slow.
#[test]
#[ignore = "plans three catalog rows without a setup budget; takes about 100 s"]
fn padded_setup_budget_matches_one_unbounded_pass_on_catalog_rows() {
    use akita_config::{
        policy_of,
        proof_optimized::{fp128, fp32},
        RecursiveCommitmentConfig,
    };

    type Fp32Dense = RecursiveCommitmentConfig<fp32::Dense>;
    type Fp128Dense = RecursiveCommitmentConfig<fp128::Dense>;
    for num_vars in [20, 22] {
        assert_budgeted_search_matches_unbounded_pass::<Fp32Dense>(
            policy_of::<Fp32Dense>(akita_params::ChunkedWitnessCfg::default_non_chunked()),
            PolynomialGroupLayout::singleton(num_vars),
            &format!("fp32_dense_recursive nv={num_vars}"),
        );
    }
    assert_budgeted_search_matches_unbounded_pass::<Fp128Dense>(
        policy_of::<Fp128Dense>(akita_params::ChunkedWitnessCfg::default_non_chunked()),
        PolynomialGroupLayout::singleton(20),
        "fp128_dense_recursive nv=20",
    );
}
