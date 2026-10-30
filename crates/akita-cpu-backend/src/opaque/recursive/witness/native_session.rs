//! Native compression equations attached to the ordinary Stage 2 session.

use super::CpuStage2SessionHandle;
use akita_error::AkitaError;
use jolt_field::{Field, Ring};

impl<E: Field + Ring> CpuStage2SessionHandle<E> {
    pub(crate) fn with_negative_binary_weights(
        mut self,
        weights: Vec<(usize, E)>,
    ) -> Result<Self, AkitaError> {
        if self.next_round != 0 || self.pending.is_some() {
            return Err(AkitaError::InvalidInput(
                "negative-binary terms require a fresh session".into(),
            ));
        }
        self.prover.attach_negative_binary_weights(weights)?;
        Ok(self)
    }
}
