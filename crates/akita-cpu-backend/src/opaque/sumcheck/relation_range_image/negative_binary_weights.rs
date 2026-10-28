//! Direct public compression-bit weights in the shared Stage 2 engine.

use super::{AdditionalRelationTerms, SparseWeight};
use crate::opaque::sumcheck::relation_range_image::RelationRangeImageProver;
use akita_error::{checked, AkitaError};
use jolt_field::{Field, Ring};

impl<E: Field + Ring> RelationRangeImageProver<E> {
    /// Install independently batched negative-binary equations before any round.
    pub(crate) fn attach_negative_binary_weights(
        &mut self,
        weights: Vec<(usize, E)>,
    ) -> Result<(), AkitaError> {
        if weights.is_empty() {
            return Ok(());
        }
        if self.state.rounds_completed != 0 || self.state.additional_relation_terms.is_some() {
            return Err(AkitaError::InvalidInput(
                "negative-binary terms require a fresh relation session".into(),
            ));
        }
        let domain_len = checked::pow2(self.state.num_vars).ok_or(AkitaError::InvalidProof)?;
        let mut previous = None;
        let mut sparse = Vec::new();
        sparse
            .try_reserve_exact(weights.len())
            .map_err(|_| AkitaError::InvalidProof)?;
        for (index, binary) in weights {
            if index >= domain_len || previous.is_some_and(|previous| index <= previous) {
                return Err(AkitaError::InvalidInput(
                    "negative-binary weights are outside the domain or not strictly sorted".into(),
                ));
            }
            previous = Some(index);
            sparse.push(SparseWeight {
                index,
                linear: E::zero(),
                binary,
            });
        }
        self.state.additional_relation_terms = Some(AdditionalRelationTerms {
            weights: sparse,
            binary_batching: E::one(),
            domain_len,
        });
        Ok(())
    }
}
