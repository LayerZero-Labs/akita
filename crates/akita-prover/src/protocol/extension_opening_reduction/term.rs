use super::*;

/// One witness member of a dense extension-opening reduction group.
#[derive(Debug, Clone)]
pub struct ExtensionOpeningReductionTerm<E: Field> {
    pub(in crate::protocol::extension_opening_reduction) witness: Vec<E>,
    pub(in crate::protocol::extension_opening_reduction) coeff: E,
    /// Coefficient-scaled next-round values produced by the fused witness fold.
    pub(in crate::protocol::extension_opening_reduction) cached_accumulate: Option<(E, E)>,
}

impl<E: Field> ExtensionOpeningReductionTerm<E> {
    /// Construct one group member `coeff * witness(x)`.
    ///
    /// The owning [`ExtensionOpeningReductionGroup`] supplies the common
    /// transparent factor and validates the witness table shape.
    pub fn new(witness_evals: Vec<E>, coeff: E) -> Self {
        Self {
            witness: witness_evals,
            coeff,
            cached_accumulate: None,
        }
    }

    /// Batching coefficient multiplying this member.
    pub fn coeff(&self) -> E {
        self.coeff
    }
}
