mod evaluator;

pub(crate) use evaluator::{SetupEvaluation, SetupEvaluator, SetupEvaluatorMode};

#[cfg(test)]
mod fixtures;
#[cfg(test)]
mod tests;
