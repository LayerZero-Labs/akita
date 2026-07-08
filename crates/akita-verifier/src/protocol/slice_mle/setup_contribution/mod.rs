mod evaluator;

pub(crate) use evaluator::SetupContributionEvaluator;
pub(crate) use evaluator::{SetupContributionEvalMode, SetupContributionEvaluation};

#[cfg(test)]
mod fixtures;
#[cfg(test)]
mod tests;
