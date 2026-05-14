//! Inspect small-field schedule candidates before baking generated tables.
//!
//! This binary is deliberately diagnostic: it runs the same config-backed
//! planner that table generation uses, but reports failures instead of
//! treating every candidate as a production row.

use std::env;

use akita_config::proof_optimized::{fp32, fp64};
use akita_config::CommitmentConfig;
use akita_planner::schedule_params::find_optimal_schedule;
use akita_types::{AkitaScheduleLookupKey, ClaimIncidenceSummary, Schedule, Step};

#[derive(Clone, Copy)]
enum Workload {
    Singleton,
    Batch4SamePoint,
}

impl Workload {
    fn name(self) -> &'static str {
        match self {
            Self::Singleton => "singleton",
            Self::Batch4SamePoint => "batch4_same_point",
        }
    }

    fn incidence(self, num_vars: usize) -> Result<ClaimIncidenceSummary, akita_field::AkitaError> {
        match self {
            Self::Singleton => ClaimIncidenceSummary::same_point(num_vars, 1),
            Self::Batch4SamePoint => ClaimIncidenceSummary::same_point(num_vars, 4),
        }
    }
}

fn parse_num_vars() -> Vec<usize> {
    env::var("HACHI_TUNE_NVS")
        .ok()
        .map(|raw| {
            raw.split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| vec![20, 25, 26, 30, 32])
}

fn first_fold_summary(schedule: &Schedule) -> String {
    match schedule.steps.first() {
        Some(Step::Fold(step)) => format!(
            "folds={} root=(D{} lb{} m{} r{} na{} nb{} nd{} next_w{})",
            schedule
                .steps
                .iter()
                .filter(|step| matches!(step, Step::Fold(_)))
                .count(),
            step.params.ring_dimension,
            step.params.log_basis,
            step.params.log_block_len(),
            step.params.log_num_blocks(),
            step.params.a_key.row_len(),
            step.params.b_key.row_len(),
            step.params.d_key.row_len(),
            step.next_w_len
        ),
        Some(Step::Direct(step)) => format!("direct bytes={}", step.direct_bytes),
        None => "empty".to_string(),
    }
}

fn report_cfg<Cfg: CommitmentConfig>(family: &str, d_label: &str, shape: &str, num_vars: &[usize]) {
    for &nv in num_vars {
        for workload in [Workload::Singleton, Workload::Batch4SamePoint] {
            let incidence = match workload.incidence(nv) {
                Ok(incidence) => incidence,
                Err(err) => {
                    println!(
                        "{family},{d_label},{shape},{},{nv},ERROR,{err}",
                        workload.name()
                    );
                    continue;
                }
            };
            let key = match AkitaScheduleLookupKey::new_from_incidence(&incidence) {
                Ok(key) => key,
                Err(err) => {
                    println!(
                        "{family},{d_label},{shape},{},{nv},ERROR,{err}",
                        workload.name()
                    );
                    continue;
                }
            };
            match find_optimal_schedule::<Cfg>(key) {
                Ok(schedule) => println!(
                    "{family},{d_label},{shape},{},{nv},OK,total_bytes={},{}",
                    workload.name(),
                    schedule.total_bytes,
                    first_fold_summary(&schedule)
                ),
                Err(err) => println!(
                    "{family},{d_label},{shape},{},{nv},ERROR,{err}",
                    workload.name()
                ),
            }
        }
    }
}

fn main() {
    let num_vars = parse_num_vars();
    println!("family,d,shape,workload,num_vars,status,total/summary");
    report_cfg::<fp32::D32Full>("fp32", "D32", "full", &num_vars);
    report_cfg::<fp32::D32OneHot>("fp32", "D32", "onehot", &num_vars);
    report_cfg::<fp32::D64Full>("fp32", "D64", "full", &num_vars);
    report_cfg::<fp32::D64OneHot>("fp32", "D64", "onehot", &num_vars);
    report_cfg::<fp32::D128Full>("fp32", "D128", "full", &num_vars);
    report_cfg::<fp32::D128OneHot>("fp32", "D128", "onehot", &num_vars);
    report_cfg::<fp32::D256Full>("fp32", "D256", "full", &num_vars);
    report_cfg::<fp32::D256OneHot>("fp32", "D256", "onehot", &num_vars);
    report_cfg::<fp32::D512Full>("fp32", "D512", "full", &num_vars);
    report_cfg::<fp32::D512OneHot>("fp32", "D512", "onehot", &num_vars);
    report_cfg::<fp64::D32Full>("fp64", "D32", "full", &num_vars);
    report_cfg::<fp64::D32OneHot>("fp64", "D32", "onehot", &num_vars);
    report_cfg::<fp64::D64Full>("fp64", "D64", "full", &num_vars);
    report_cfg::<fp64::D64OneHot>("fp64", "D64", "onehot", &num_vars);
    report_cfg::<fp64::D128Full>("fp64", "D128", "full", &num_vars);
    report_cfg::<fp64::D128OneHot>("fp64", "D128", "onehot", &num_vars);
    report_cfg::<fp64::D256Full>("fp64", "D256", "full", &num_vars);
    report_cfg::<fp64::D256OneHot>("fp64", "D256", "onehot", &num_vars);
}
