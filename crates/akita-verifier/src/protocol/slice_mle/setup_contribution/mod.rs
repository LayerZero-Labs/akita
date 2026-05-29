mod evaluator;

#[cfg(test)]
pub(crate) use evaluator::MaterializedSetupOmega;
pub(crate) use evaluator::{SetupEvaluation, SetupEvaluator, SetupEvaluatorMode};

#[cfg(test)]
mod fixtures;
#[cfg(test)]
mod tests;
