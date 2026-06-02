mod evaluator;

pub(crate) use evaluator::SetupEvaluator;

#[cfg(test)]
mod fixtures;
#[cfg(test)]
mod reference;
#[cfg(test)]
mod tests;
