//! Versioned, independently addressed column challenges inside one transcript.

use akita_error::AkitaError;
use num_bigint::BigUint;
use shake::digest::{ExtendableOutput, Update, XofReader};
use shake::Shake256;

use super::{root_family, BinaryChallenge, BinaryChallengeProfile};

/// Domain of the addressed binary-source column sampler.
pub const MULTIGROUP_COLUMN_DOMAIN: &[u8] = b"akita/binary-source/multigroup/column/v1";
/// The size/lexicographic bounded-46 support family with the existing sign rule.
pub const MULTIGROUP_COLUMN_FAMILY_ID: u64 = 1;
const CANDIDATE_BYTES: usize = 18;
const CANDIDATE_LIMIT: u64 = 256;
const BITMAP_BYTES: usize = 21;

/// One coordinate bound to the common post-D prefix and checked group work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnChallengeAddress {
    /// Immutable 256-bit binding of the transcript through the final D payload.
    pub common_prefix: [u8; 32],
    /// Root level, canonically zero for this profile.
    pub level: u64,
    /// Binary group occurrence in the opening.
    pub group: u64,
    /// Polynomial owner within that group.
    pub owner: u64,
    /// Column within that owner's source image.
    pub column: u64,
    /// Response candidate in `0..4096` for this group.
    pub response_nonce: u64,
    /// Binding derived after checking this group's scheduled column work.
    pub column_work_binding: [u8; 32],
}

impl ColumnChallengeAddress {
    fn validate(&self) -> Result<(), AkitaError> {
        if self.level != 0
            || self.group >= 64
            || self.owner >= 4096
            || self.column >= 256
            || self.response_nonce >= 4096
        {
            return Err(AkitaError::InvalidInput(
                "binary column address exceeds profile bounds".into(),
            ));
        }
        Ok(())
    }

    fn xof_prefix(&self) -> Shake256 {
        let mut xof = Shake256::default();
        xof.update(MULTIGROUP_COLUMN_DOMAIN);
        xof.update(&self.common_prefix);
        xof.update(&self.level.to_le_bytes());
        xof.update(&self.group.to_le_bytes());
        xof.update(&self.owner.to_le_bytes());
        xof.update(&self.column.to_le_bytes());
        xof.update(&self.response_nonce.to_le_bytes());
        xof.update(&self.column_work_binding);
        xof.update(&MULTIGROUP_COLUMN_FAMILY_ID.to_le_bytes());
        xof
    }
}

/// Exact finite sampler for the selected multigroup column profile.
pub struct MultigroupColumnSampler {
    profile: BinaryChallengeProfile,
}

impl MultigroupColumnSampler {
    /// Construct the fixed degree-162 bounded-weight-46 profile.
    pub fn new() -> Result<Self, AkitaError> {
        Ok(Self {
            profile: root_family::profile()?,
        })
    }

    /// The existing deterministic sign family and its exact support bounds.
    pub const fn profile(&self) -> &BinaryChallengeProfile {
        &self.profile
    }

    /// Draw one support with at most 256 independently indexed 137-bit candidates.
    pub fn sample(&self, address: &ColumnChallengeAddress) -> Result<BinaryChallenge, AkitaError> {
        address.validate()?;
        let prefix = address.xof_prefix();
        self.sample_with_blocks(|block, bytes| {
            let mut xof = prefix.clone();
            xof.update(&block.to_le_bytes());
            xof.finalize_xof().read(bytes);
        })
    }

    fn sample_with_blocks(
        &self,
        mut fill: impl FnMut(u64, &mut [u8; CANDIDATE_BYTES]),
    ) -> Result<BinaryChallenge, AkitaError> {
        for block in 0..CANDIDATE_LIMIT {
            let mut bytes = [0u8; CANDIDATE_BYTES];
            fill(block, &mut bytes);
            // 18 bytes carry 144 bits; only the low 137 bits belong to the rank.
            if let Some(high) = bytes.last_mut() {
                *high &= 1;
            }
            let rank = BigUint::from_bytes_le(&bytes);
            if &rank < self.profile.cardinality() {
                let support = self
                    .profile
                    .scalar_ring()
                    .counts()
                    .unrank_lex(&rank, self.profile.weight_cap())?;
                return Ok(BinaryChallenge::from_support(
                    &self.profile,
                    &support,
                    &mut Vec::new(),
                    &mut Vec::new(),
                ));
            }
        }
        Err(AkitaError::InvalidInput(
            "binary column support sampler exhausted".into(),
        ))
    }

    /// Decode a canonical 21-byte support bitmap and reproduce its fixed signs.
    pub fn decode_support_bitmap(&self, bitmap: &[u8]) -> Result<BinaryChallenge, AkitaError> {
        if bitmap.len() != BITMAP_BYTES || bitmap.last().is_none_or(|byte| byte & !3 != 0) {
            return Err(AkitaError::InvalidInput(
                "binary column support bitmap is not canonical".into(),
            ));
        }
        let mut support = Vec::new();
        support
            .try_reserve_exact(self.profile.weight_cap())
            .map_err(|_| AkitaError::InvalidInput("binary support allocation failed".into()))?;
        for (byte_index, &byte) in bitmap.iter().enumerate() {
            for bit in 0..8 {
                if byte & (1 << bit) != 0 {
                    if support.len() == self.profile.weight_cap() {
                        return Err(AkitaError::InvalidInput(
                            "binary column support exceeds weight cap".into(),
                        ));
                    }
                    support.push(u16::try_from(byte_index * 8 + bit).map_err(|_| {
                        AkitaError::InvalidInput("binary support position exceeds u16".into())
                    })?);
                }
            }
        }
        Ok(BinaryChallenge::from_support(
            &self.profile,
            &support,
            &mut Vec::new(),
            &mut Vec::new(),
        ))
    }
}

#[cfg(test)]
mod tests;
