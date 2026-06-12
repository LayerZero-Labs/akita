mod evaluator;

pub(crate) use akita_types::SetupContributionPlan as SetupEvalPlan;
pub(crate) use evaluator::{SetupEvaluation, SetupEvaluator, SetupEvaluatorMode};

#[cfg(test)]
mod fixtures;
#[cfg(test)]
mod tests;
