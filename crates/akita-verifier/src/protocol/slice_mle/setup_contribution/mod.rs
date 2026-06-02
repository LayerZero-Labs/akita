mod evaluator;

pub use evaluator::SetupEvaluator;
pub(crate) use evaluator::{
    jolt_end_cycle_tracking, jolt_start_cycle_tracking, SetupEvalPlan, SetupEvaluation,
    SetupEvaluatorMode,
};

#[cfg(test)]
mod fixtures;
#[cfg(test)]
mod tests;
