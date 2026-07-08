mod evaluator;

pub(crate) use evaluator::SetupEvaluator;
pub(crate) use evaluator::{SetupEvaluation, SetupEvaluatorMode};

#[cfg(test)]
mod fixtures;
#[cfg(test)]
mod tests;
