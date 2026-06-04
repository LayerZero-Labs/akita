mod evaluator;

pub(crate) use akita_types::SetupContributionPlan as SetupEvalPlan;
pub use evaluator::SetupEvaluator;
pub(crate) use evaluator::{JoltCycleScope, SetupEvaluation, SetupEvaluatorMode};

#[cfg(test)]
mod fixtures;
#[cfg(test)]
mod tests;
