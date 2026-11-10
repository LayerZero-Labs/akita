//! Inactive multigroup column-address seam; the root drivers do not select it.

use super::super::{BinaryColumnGroupPlan, BinaryColumnPlan};
use akita_challenges::{
    BinaryChallenge, ColumnChallengeAddress, MultigroupColumnSampler, MULTIGROUP_COLUMN_DOMAIN,
};
use akita_error::{checked, AkitaError};
use akita_params::FOLD_RESPONSE_ATTEMPTS;
use akita_transcript::{IndexedTranscriptPrefix, NonceAtom, ProofChannel};

/// The immutable common prefix through the final D payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryColumnPrefix(IndexedTranscriptPrefix);

/// One accepted group's canonical supports and its response/work nonce tuple.
/// Private fields prevent combining supports with a different prefix or plan.
#[derive(Clone, Debug)]
pub struct BinaryColumnResponse {
    prefix: BinaryColumnPrefix,
    plan: BinaryColumnGroupPlan,
    response_nonce: u32,
    work_nonce: Option<u32>,
    supports: Vec<BinaryChallenge>,
}

impl BinaryColumnResponse {
    /// Accepted response counter in 0..4096.
    pub const fn response_nonce(&self) -> u32 {
        self.response_nonce
    }

    /// Omitted for a zero-bit work site.
    pub const fn work_nonce(&self) -> Option<u32> {
        self.work_nonce
    }

    /// Supports in owner-major, then column order, with deterministic signs.
    pub fn supports(&self) -> &[BinaryChallenge] {
        &self.supports
    }

    /// Read one validated owner/column coordinate.
    pub fn support(&self, owner: usize, column: usize) -> Result<&BinaryChallenge, AkitaError> {
        self.supports
            .get(self.plan.coordinate_index(owner, column)?)
            .ok_or(AkitaError::InvalidProof)
    }
}

impl BinaryColumnPrefix {
    /// Freeze the common prefix after **all** D payloads. The later driver
    /// enforces that barrier; this seam does not change driver admission.
    /// The 256-bit representation carries the spec's separate hash charge.
    pub fn capture(state: &mut impl ProofChannel) -> Result<Self, AkitaError> {
        state.public(b"akita/binary-source/multigroup/post-d/v1\0\0\0");
        IndexedTranscriptPrefix::capture(state).map(Self)
    }

    /// Common Pi binding; accepted nonces never redefine it.
    pub const fn binding(&self) -> &[u8; 32] {
        self.0.binding()
    }

    fn draw(
        &self,
        plan: &BinaryColumnGroupPlan,
        response_nonce: u32,
        work_nonce: Option<u32>,
        column_work_binding: [u8; 32],
    ) -> Result<BinaryColumnResponse, AkitaError> {
        let sampler = MultigroupColumnSampler::new()?;
        let mut supports = Vec::new();
        supports
            .try_reserve_exact(plan.coordinate_count())
            .map_err(|_| AkitaError::InvalidProof)?;
        for owner in 0..plan.owners() {
            for column in 0..plan.columns() {
                supports.push(sampler.sample(&ColumnChallengeAddress {
                    common_prefix: *self.binding(),
                    level: 0,
                    group: u64::from(plan.group()),
                    owner: u64::try_from(owner).map_err(|_| AkitaError::InvalidProof)?,
                    column: u64::try_from(column).map_err(|_| AkitaError::InvalidProof)?,
                    response_nonce: u64::from(response_nonce),
                    column_work_binding,
                })?);
            }
        }
        Ok(BinaryColumnResponse {
            prefix: self.clone(),
            plan: *plan,
            response_nonce,
            work_nonce,
            supports,
        })
    }

    /// Replay one group independently from Pi, checking work before its draws.
    /// The admission callback checks that group's bounded response using its
    /// own source data; nonce admissibility alone does not prove that bound.
    pub fn replay_group(
        &self,
        plan: &BinaryColumnGroupPlan,
        response_nonce: u32,
        work_nonce: Option<u32>,
        admit: impl FnOnce(u32, &[BinaryChallenge]) -> Result<bool, AkitaError>,
    ) -> Result<BinaryColumnResponse, AkitaError> {
        if response_nonce >= FOLD_RESPONSE_ATTEMPTS {
            return Err(AkitaError::InvalidProof);
        }
        let omega = self.0.work_binding(
            MULTIGROUP_COLUMN_DOMAIN,
            0,
            u64::from(plan.group()),
            response_nonce,
            plan.work_run().grind_bits(),
            work_nonce,
        )?;
        let response = self.draw(plan, response_nonce, work_nonce, omega)?;
        if !admit(response_nonce, response.supports())? {
            return Err(AkitaError::InvalidProof);
        }
        Ok(response)
    }

    /// Search only this group's source response; no other group's address or
    /// challenge stream is consulted. Exhaustion is a typed completeness abort.
    #[cfg(feature = "prover")]
    pub fn search_group(
        &self,
        plan: &BinaryColumnGroupPlan,
        mut admit: impl FnMut(u32, &[BinaryChallenge]) -> Result<bool, AkitaError>,
    ) -> Result<BinaryColumnResponse, AkitaError> {
        for nonce in 0..FOLD_RESPONSE_ATTEMPTS {
            let (work, omega) = self.0.search_work(
                MULTIGROUP_COLUMN_DOMAIN,
                0,
                u64::from(plan.group()),
                nonce,
                plan.work_run().grind_bits(),
            )?;
            let response = self.draw(plan, nonce, work, omega)?;
            if admit(nonce, response.supports())? {
                return Ok(response);
            }
        }
        Err(AkitaError::InvalidInput(
            "binary group response search exhausted".into(),
        ))
    }

    pub(crate) fn bind_responses(
        &self,
        plan: &BinaryColumnPlan,
        responses: &[BinaryColumnResponse],
    ) -> Result<(Vec<u32>, Vec<u8>), AkitaError> {
        if responses.len() != plan.groups().len() {
            return Err(AkitaError::InvalidInput(
                "binary group response count mismatch".into(),
            ));
        }
        let sampler = MultigroupColumnSampler::new()?;
        let count = plan
            .groups()
            .iter()
            .try_fold(0usize, |sum, group| {
                checked::sum([sum, checked::product([21, group.coordinate_count()])?, 16])
            })
            .ok_or(AkitaError::InvalidProof)?;
        let mut binding = Vec::new();
        binding
            .try_reserve_exact(count)
            .map_err(|_| AkitaError::InvalidProof)?;
        let mut nonces = Vec::new();
        nonces
            .try_reserve_exact(
                checked::product([2, responses.len()]).ok_or(AkitaError::InvalidProof)?,
            )
            .map_err(|_| AkitaError::InvalidProof)?;
        for (group, response) in plan.groups().iter().zip(responses) {
            if response.prefix != *self
                || response.plan != *group
                || response.supports.len() != group.coordinate_count()
            {
                return Err(AkitaError::InvalidInput(
                    "binary group response prefix or plan mismatch".into(),
                ));
            }
            nonces.push(response.response_nonce);
            binding.extend_from_slice(&NonceAtom::new(response.response_nonce).word());
            if let Some(nonce) = response.work_nonce {
                nonces.push(nonce);
                binding.extend_from_slice(&NonceAtom::new(nonce).word());
            }
            for support in response.supports() {
                binding.extend_from_slice(&support.canonical_support_encoding(sampler.profile())?);
            }
        }
        Ok((nonces, binding))
    }
}

pub(crate) fn column_nonce_widths(plan: &BinaryColumnPlan) -> Vec<u8> {
    plan.groups()
        .iter()
        .flat_map(|group| {
            [
                Some(group.response_run().nonce_bits()),
                (group.work_run().nonce_bits() != 0).then_some(group.work_run().nonce_bits()),
            ]
            .into_iter()
            .flatten()
        })
        .collect()
}

#[cfg(all(test, feature = "prover"))]
mod tests;
