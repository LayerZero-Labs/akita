use akita_config::CommitmentConfig;
use akita_types::{FoldSchedule, PolynomialGroupLayout, SetupContributionMode};

pub(super) struct ProofSizeBudgets {
    planner_estimate: usize,
    native_bound: usize,
}

pub(super) fn proof_size_budgets<Cfg: CommitmentConfig>(
    schedule: &FoldSchedule,
    final_group: PolynomialGroupLayout,
) -> ProofSizeBudgets {
    let key = akita_types::AkitaScheduleLookupKey {
        final_group,
        precommitteds: schedule
            .root
            .params
            .precommitted_groups()
            .iter()
            .map(|group| group.profile)
            .collect(),
    };
    let policy = akita_config::policy_of::<Cfg>();
    let planner_estimate =
        akita_schedules::expanded_schedule_native_proof_estimate_bytes(&key, schedule, &policy)
            .expect("expanded schedule estimate");
    let native_bound =
        akita_schedules::expanded_schedule_native_proof_bound(&key, schedule, &policy)
            .expect("native proof bound");
    ProofSizeBudgets {
        planner_estimate,
        native_bound,
    }
}

pub(super) fn assert_observed_proof_size(label: &str, proof: &[u8]) {
    assert!(
        !proof.is_empty(),
        "[{label}] native proof must not be empty"
    );
}

pub(super) fn report_proof_size_against_planner(
    label: &str,
    proof: &[u8],
    budgets: ProofSizeBudgets,
    source: &str,
    mode: SetupContributionMode,
) {
    let actual_bytes = proof.len();
    assert!(
        actual_bytes <= budgets.native_bound,
        "[{label}] native proof bytes {actual_bytes} exceed the schedule-derived native bound {}",
        budgets.native_bound,
    );
    let planner_delta = i128::try_from(actual_bytes).expect("proof size fits i128")
        - i128::try_from(budgets.planner_estimate).expect("planner estimate fits i128");
    tracing::info!(
        label,
        actual_bytes,
        planner_estimate = budgets.planner_estimate,
        native_bound = budgets.native_bound,
        planner_delta,
        ?mode,
        "native proof-size comparison"
    );
    eprintln!(
        "[{label}] proof_size: native={actual_bytes} bytes, planner_estimate={} bytes, \
         native_bound={} bytes, actual_minus_planner={planner_delta} bytes, \
         source={source}, setup_contribution_mode={mode:?}",
        budgets.planner_estimate, budgets.native_bound,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_underestimate_is_diagnostic_when_native_bound_holds() {
        report_proof_size_against_planner(
            "fixture",
            &[0; 2],
            ProofSizeBudgets {
                planner_estimate: 1,
                native_bound: 2,
            },
            "fixture",
            SetupContributionMode::Direct,
        );
    }
}
