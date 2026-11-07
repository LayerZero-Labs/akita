//! Direct public compression-bit weights in the shared Stage 2 engine.

use super::{AdditionalRelationTerms, SparseWeight};
use crate::opaque::sumcheck::relation_range_image::RelationRangeImageProver;
use akita_error::{checked, AkitaError};
#[cfg(feature = "parallel")]
use jolt_field::solinas::parallel::*;
use jolt_field::{cfg_chunks, Field, Ring};

impl<E: Field + Ring> RelationRangeImageProver<E> {
    /// Install independently batched negative-binary equations before any round.
    #[tracing::instrument(skip_all, name = "root_prepare_negative_binary_weights", fields(count = weights.len()))]
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
        let validation = tracing::info_span!(
            "root_prepare_validate_negative_weights",
            count = weights.len(),
            domain_len
        )
        .entered();
        // Include the preceding chunk's final index, so every adjacency is
        // checked once even when the two entries belong to different workers.
        let invalid = cfg_chunks!(weights, 1 << 14)
            .enumerate()
            .any(|(chunk_index, chunk)| {
                let mut previous = checked::product([chunk_index, 1 << 14])
                    .and_then(|start| start.checked_sub(1))
                    .and_then(|index| weights.get(index))
                    .map(|&(index, _)| index);
                for &(index, _) in chunk {
                    if index >= domain_len || previous.is_some_and(|previous| index <= previous) {
                        return true;
                    }
                    previous = Some(index);
                }
                false
            });
        if invalid {
            return Err(AkitaError::InvalidInput(
                "negative-binary weights are outside the domain or not strictly sorted".into(),
            ));
        }
        drop(validation);
        let _fill = tracing::info_span!(
            "root_prepare_convert_negative_weights",
            count = weights.len()
        )
        .entered();
        let mut sparse = Vec::new();
        sparse
            .try_reserve_exact(weights.len())
            .map_err(|_| AkitaError::InvalidProof)?;
        // All addresses are validated before workers fill the ordered output.
        // The input weights and their independently drawn coefficients retain
        // their original pairing; no field reduction or transcript runs here.
        let convert = |(index, binary)| SparseWeight {
            index,
            linear: E::zero(),
            binary,
        };
        #[cfg(feature = "parallel")]
        sparse.par_extend(weights.into_par_iter().map(convert));
        #[cfg(not(feature = "parallel"))]
        sparse.extend(weights.into_iter().map(convert));
        self.state.additional_relation_terms = Some(AdditionalRelationTerms {
            weights: sparse,
            binary_batching: E::one(),
            domain_len,
        });
        Ok(())
    }
}
