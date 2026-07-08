mod evaluator;

pub(crate) use evaluator::evaluate_setup_contribution_direct;
#[cfg(test)]
pub(crate) use evaluator::evaluate_setup_contribution_recursive;

#[cfg(test)]
mod fixtures;
#[cfg(test)]
mod tests;
