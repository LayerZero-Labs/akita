//! Atomic replay of the inactive, group-indexed binary column plan.

use super::*;
use crate::binary_source::{
    column_nonce_widths, BinaryColumnPlan, BinaryColumnPrefix, BinaryColumnResponse,
};
use akita_challenges::BinaryChallenge;
use akita_params::GrindingRun;

impl GrindingPlanCursor<'_> {
    fn consume_binary_columns(&mut self, plan: &BinaryColumnPlan) -> Result<(), AkitaError> {
        for group in plan.groups() {
            for expected in [group.response_run(), group.work_run(), group.draw_run()] {
                self.consume_exact_run(expected)?;
            }
        }
        Ok(())
    }

    fn consume_exact_run(&mut self, expected: GrindingRun) -> Result<(), AkitaError> {
        if self.run_offset != 0 || self.plan.runs().get(self.run_index) != Some(&expected) {
            return Err(AkitaError::InvalidInput(
                "binary column replay plan mismatch".into(),
            ));
        }
        self.run_index = self
            .run_index
            .checked_add(1)
            .ok_or(AkitaError::InvalidProof)?;
        Ok(())
    }
}

#[cfg(feature = "prover")]
impl ProverGrinding<'_> {
    /// Bind all accepted nonce tuples and supports to the live cursor, in
    /// group/owner/column order. Search uses only the immutable prefix.
    /// The caller supplies Pi captured from this channel at the all-D barrier
    /// and leaves its live cursor unchanged while searching detached candidates.
    pub fn commit_binary_column_responses(
        &mut self,
        prefix: &BinaryColumnPrefix,
        plan: &BinaryColumnPlan,
        responses: &[BinaryColumnResponse],
    ) -> Result<(), AkitaError> {
        let result = (|| {
            let (nonces, binding) = prefix.bind_responses(plan, responses)?;
            self.cursor.consume_binary_columns(plan)?;
            let widths = column_nonce_widths(plan);
            akita_transcript::send_indexed_nonces(self.state, &nonces, &widths, binding)?;
            let bytes = nonces
                .iter()
                .try_fold(0usize, |sum, &nonce| {
                    sum.checked_add(nonce_encoded_len(nonce))
                })
                .ok_or(AkitaError::InvalidProof)?;
            self.serialized_nonce_bytes = self
                .serialized_nonce_bytes
                .checked_add(bytes)
                .ok_or(AkitaError::InvalidProof)?;
            Ok(())
        })();
        if result.is_err() {
            self.poison();
        }
        result
    }
}

impl VerifierGrinding<'_, '_> {
    /// Read every group's canonical response/work nonces and replay each
    /// Pi-local work and support stream. Admission and the entire cursor check
    /// precede live absorption; a failure permanently rejects this replay.
    /// The caller supplies this channel's all-D Pi and has not advanced the
    /// channel since capture. Plan replay does not enforce that message barrier.
    pub fn read_binary_column_responses(
        &mut self,
        prefix: &BinaryColumnPrefix,
        plan: &BinaryColumnPlan,
        mut admit: impl FnMut(u32, u32, &[BinaryChallenge]) -> Result<bool, AkitaError>,
    ) -> Result<Vec<BinaryColumnResponse>, AkitaError> {
        let result = (|| {
            self.cursor.consume_binary_columns(plan)?;
            akita_transcript::receive_indexed_nonces(
                self.state,
                &column_nonce_widths(plan),
                |nonces| {
                    let mut nonces = nonces.iter().copied();
                    let mut responses = Vec::new();
                    responses
                        .try_reserve_exact(plan.groups().len())
                        .map_err(|_| AkitaError::InvalidProof)?;
                    for group in plan.groups() {
                        let nonce = nonces.next().ok_or(AkitaError::InvalidProof)?;
                        let work = if group.work_run().nonce_bits() == 0 {
                            None
                        } else {
                            Some(nonces.next().ok_or(AkitaError::InvalidProof)?)
                        };
                        responses.push(prefix.replay_group(
                            group,
                            nonce,
                            work,
                            |nonce, supports| admit(group.group(), nonce, supports),
                        )?);
                    }
                    if nonces.next().is_some() {
                        return Err(AkitaError::InvalidProof);
                    }
                    let (_, binding) = prefix.bind_responses(plan, &responses)?;
                    Ok((responses, binding))
                },
            )
        })();
        if result.is_err() {
            self.poison();
        }
        result
    }
}
