use super::*;

mod suffix_sums;
mod tables;
mod tensor_factor;
mod term;
mod witness;

#[cfg(test)]
mod tests;

pub use term::ExtensionOpeningReductionTerm;
pub use witness::SparseExtensionOpeningWitness;

pub(in crate::protocol::extension_opening_reduction) use tables::ExtensionOpeningTables;
pub(in crate::protocol::extension_opening_reduction) use tensor_factor::{
    SparseFactor, TensorEqualityFactor,
};
