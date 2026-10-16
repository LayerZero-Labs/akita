use akita_params::{FoldSchedule, PolynomialGroupLayout};
use akita_schedules::PlannerPolicy;

pub(super) struct ProofSizeBudgets {
    planner_estimate: usize,
    bound: usize,
}

pub(super) fn proof_size_budgets(
    schedule: &FoldSchedule,
    final_group: PolynomialGroupLayout,
    policy: &PlannerPolicy,
) -> ProofSizeBudgets {
    let key = akita_params::ScheduleLookupKey {
        final_group,
        precommitteds: schedule
            .root
            .params
            .precommitted_groups()
            .iter()
            .map(|group| group.profile)
            .collect(),
    };
    let planner_estimate =
        akita_schedules::expanded_schedule_proof_estimate_bytes(&key, schedule, policy)
            .expect("expanded schedule estimate");
    let bound = akita_schedules::expanded_schedule_proof_bound(&key, schedule, policy)
        .expect("native proof bound");
    ProofSizeBudgets {
        planner_estimate,
        bound,
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
) {
    let actual_bytes = proof.len();
    assert!(
        actual_bytes <= budgets.bound,
        "[{label}] native proof bytes {actual_bytes} exceed the schedule-derived native bound {}",
        budgets.bound,
    );
    let planner_delta = i128::try_from(actual_bytes).expect("proof size fits i128")
        - i128::try_from(budgets.planner_estimate).expect("planner estimate fits i128");
    tracing::info!(
        label,
        actual_bytes,
        planner_estimate = budgets.planner_estimate,
        native_bound = budgets.bound,
        planner_delta,
        "native proof-size comparison"
    );
    eprintln!(
        "[{label}] proof_size: native={actual_bytes} bytes, planner_estimate={} bytes, \
         native_bound={} bytes, actual_minus_planner={planner_delta} bytes, \
         source={source}",
        budgets.planner_estimate, budgets.bound,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_underestimate_is_diagnostic_when_bound_holds() {
        report_proof_size_against_planner(
            "fixture",
            &[0; 2],
            ProofSizeBudgets {
                planner_estimate: 1,
                bound: 2,
            },
            "fixture",
        );
    }
}
