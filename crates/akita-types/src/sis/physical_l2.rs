//! Schedule-owned physical L2 proof geometry.

use std::ops::Range;

/// Canonical work cap for one small-field limb inner-product subclaim.
///
/// The integer no-wrap limit can be much larger than a practical sumcheck
/// block. Keeping blocks at 4 Ki coefficients makes the proof geometry and
/// verifier work bounded while remaining comfortably below every derived
/// field-specific no-wrap ceiling.
const MAX_LIMB_GRAM_BLOCK_LEN: usize = 1 << 12;

use akita_error::AkitaError;

use super::ajtai_key::{SisModulusProfileId, SisSecurityPolicyId, SisTableKey};
use super::l2_table::SisL2TableKey;
use crate::descriptor_bytes::push_usize;

/// Checked block and upper-triangular limb-pair layout for a LimbGram proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LimbGramLayout {
    physical_response_len: usize,
    block_len: usize,
    limb_count: usize,
    block_count: usize,
    pair_count: usize,
}

impl LimbGramLayout {
    /// Build a checked layout from explicit public shape values.
    pub fn new(
        physical_response_len: usize,
        block_len: usize,
        limb_count: usize,
    ) -> Result<Self, AkitaError> {
        if physical_response_len == 0
            || block_len == 0
            || limb_count == 0
            || block_len > physical_response_len
        {
            return Err(AkitaError::InvalidSetup(
                "L2 limb-Gram shape has invalid response, block, or limb count".into(),
            ));
        }
        let block_count = physical_response_len.div_ceil(block_len);
        let pair_count =
            limb_count
                .checked_mul(limb_count.checked_add(1).ok_or_else(|| {
                    AkitaError::InvalidSetup("L2 limb-pair count overflow".into())
                })?)
                .and_then(|count| count.checked_div(2))
                .ok_or_else(|| AkitaError::InvalidSetup("L2 limb-pair count overflow".into()))?;
        block_count.checked_mul(pair_count).ok_or_else(|| {
            AkitaError::InvalidSetup("L2 limb-Gram subclaim count overflow".into())
        })?;
        Ok(Self {
            physical_response_len,
            block_len,
            limb_count,
            block_count,
            pair_count,
        })
    }

    #[must_use]
    pub const fn physical_response_len(self) -> usize {
        self.physical_response_len
    }

    #[must_use]
    pub const fn block_len(self) -> usize {
        self.block_len
    }

    #[must_use]
    pub const fn limb_count(self) -> usize {
        self.limb_count
    }

    #[must_use]
    pub const fn block_count(self) -> usize {
        self.block_count
    }

    #[must_use]
    pub const fn pair_count(self) -> usize {
        self.pair_count
    }

    #[must_use]
    pub fn subclaim_count(self) -> usize {
        self.block_count * self.pair_count
    }

    /// Consecutive response ranges, including the final short block.
    pub fn block_ranges(self) -> impl ExactSizeIterator<Item = Range<usize>> {
        (0..self.block_count).map(move |block_index| {
            let start = block_index * self.block_len;
            let end = start
                .saturating_add(self.block_len)
                .min(self.physical_response_len);
            start..end
        })
    }

    /// Canonical upper-triangular limb-pair order.
    pub fn limb_pairs(self) -> impl Iterator<Item = (usize, usize)> + Clone {
        (0..self.limb_count)
            .flat_map(move |left| (left..self.limb_count).map(move |right| (left, right)))
    }

    /// Canonical flat subclaim index for one block and limb pair.
    pub fn subclaim_index(
        self,
        block_index: usize,
        left_limb: usize,
        right_limb: usize,
    ) -> Option<usize> {
        if block_index >= self.block_count
            || left_limb > right_limb
            || right_limb >= self.limb_count
        {
            return None;
        }
        let preceding_rows = left_limb
            .checked_mul(self.limb_count)?
            .checked_sub(left_limb.checked_mul(left_limb.saturating_sub(1))? / 2)?;
        let pair_index = preceding_rows.checked_add(right_limb.checked_sub(left_limb)?)?;
        block_index
            .checked_mul(self.pair_count)?
            .checked_add(pair_index)
    }
}

/// Schedule-owned shape of the integer norm proof for one L2-selected A
/// matrix.
///
/// The proof serializes no block or limb-pair identifiers. This shape and
/// [`LimbGramLayout`] derive their complete canonical order.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum PhysicalL2NormProofShape {
    /// One direct square-sum claim over every physical response coefficient.
    Direct { physical_response_len: usize },
    /// Blockwise balanced-limb Gram claims used when the direct integer sum
    /// could wrap the base field.
    LimbGram {
        physical_response_len: usize,
        block_len: usize,
        limb_count: usize,
    },
}

/// The single selected security route for an A commitment matrix.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum InnerCommitSecurityRoute {
    /// Existing coefficient-L-infinity sizing and digit-range proof.
    Linf(SisTableKey),
    /// Complete physical L2 sizing with the scheduled integer norm proof.
    L2 {
        table_key: SisL2TableKey,
        response_l2_sq_cap: u128,
        norm_proof_shape: PhysicalL2NormProofShape,
    },
}

impl InnerCommitSecurityRoute {
    #[must_use]
    pub const fn modulus_profile(self) -> SisModulusProfileId {
        match self {
            Self::Linf(key) => key.modulus_profile,
            Self::L2 { table_key, .. } => table_key.modulus_profile,
        }
    }

    #[must_use]
    pub const fn policy(self) -> SisSecurityPolicyId {
        match self {
            Self::Linf(key) => key.policy,
            Self::L2 { table_key, .. } => table_key.policy,
        }
    }

    #[must_use]
    pub const fn ring_dimension(self) -> u32 {
        match self {
            Self::Linf(key) => key.ring_dimension,
            Self::L2 { table_key, .. } => table_key.ring_dimension,
        }
    }
}

impl PhysicalL2NormProofShape {
    /// Derive the canonical no-wrap proof shape for one physical response
    /// domain and its existing balanced digit decomposition.
    pub fn derive(
        modulus_profile: SisModulusProfileId,
        physical_response_len: usize,
        fold_basis: usize,
        fold_digit_count: usize,
    ) -> Result<Self, AkitaError> {
        if physical_response_len == 0
            || fold_digit_count == 0
            || fold_basis < 2
            || !fold_basis.is_power_of_two()
        {
            return Err(AkitaError::InvalidSetup(
                "L2 norm shape requires a nonempty response and balanced power-of-two digits"
                    .into(),
            ));
        }
        let direct = Self::Direct {
            physical_response_len,
        };
        if direct
            .validate_integer_soundness(modulus_profile, fold_basis, fold_digit_count)
            .is_ok()
        {
            return Ok(direct);
        }
        let modulus = modulus_profile.modulus();
        let digit_abs = (fold_basis / 2) as u128;
        if modulus > i128::MAX as u128 {
            return Err(AkitaError::InvalidSetup(
                "L2 response is too wide for direct proof and its modulus has no centered-limb path"
                    .into(),
            ));
        }
        let digit_square = digit_abs
            .checked_mul(digit_abs)
            .ok_or_else(|| AkitaError::InvalidSetup("L2 limb digit square overflow".into()))?;
        let max_block = modulus
            .checked_div(2)
            .and_then(|half| half.checked_sub(1))
            .and_then(|limit| limit.checked_div(digit_square))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("L2 limb alphabet cannot fit a centered block".into())
            })?;
        let shape = Self::LimbGram {
            physical_response_len,
            block_len: usize::try_from(max_block)
                .unwrap_or(usize::MAX)
                .min(MAX_LIMB_GRAM_BLOCK_LEN)
                .min(physical_response_len),
            limb_count: fold_digit_count,
        };
        shape.validate_integer_soundness(modulus_profile, fold_basis, fold_digit_count)?;
        Ok(shape)
    }

    /// Validate that the shape rules out field wraparound using public bounds.
    pub fn validate_integer_soundness(
        self,
        modulus_profile: SisModulusProfileId,
        fold_basis: usize,
        fold_digit_count: usize,
    ) -> Result<(), AkitaError> {
        self.validate()?;
        if fold_digit_count == 0 || fold_basis < 2 || !fold_basis.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "L2 norm shape has an invalid balanced digit decomposition".into(),
            ));
        }
        let modulus = modulus_profile.modulus();
        let digit_abs = (fold_basis / 2) as u128;
        match self {
            Self::Direct {
                physical_response_len,
            } => {
                let mut max_response = 0u128;
                let mut power = 1u128;
                for _ in 0..fold_digit_count {
                    max_response = max_response
                        .checked_add(digit_abs.checked_mul(power).ok_or_else(|| {
                            AkitaError::InvalidSetup("direct norm response bound overflow".into())
                        })?)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("direct norm response bound overflow".into())
                        })?;
                    power = power.checked_mul(fold_basis as u128).ok_or_else(|| {
                        AkitaError::InvalidSetup("direct norm basis power overflow".into())
                    })?;
                }
                let worst = (physical_response_len as u128)
                    .checked_mul(max_response.checked_mul(max_response).ok_or_else(|| {
                        AkitaError::InvalidSetup("direct norm worst-case overflow".into())
                    })?)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("direct norm worst-case overflow".into())
                    })?;
                if worst >= modulus {
                    return Err(AkitaError::InvalidSetup(
                        "direct norm shape does not rule out field wraparound".into(),
                    ));
                }
            }
            Self::LimbGram {
                block_len,
                limb_count,
                ..
            } => {
                if limb_count != fold_digit_count || modulus > i128::MAX as u128 {
                    return Err(AkitaError::InvalidSetup(
                        "L2 limb-Gram shape disagrees with its field or digit count".into(),
                    ));
                }
                let claim_abs_bound =
                    (block_len as u128)
                        .checked_mul(digit_abs.checked_mul(digit_abs).ok_or_else(|| {
                            AkitaError::InvalidSetup("L2 limb bound overflow".into())
                        })?)
                        .ok_or_else(|| AkitaError::InvalidSetup("L2 limb bound overflow".into()))?;
                if claim_abs_bound >= modulus / 2 {
                    return Err(AkitaError::InvalidSetup(
                        "L2 limb block does not rule out centered-lift ambiguity".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Validate nonzero, bounded arithmetic for the schedule-derived shape.
    pub fn validate(self) -> Result<(), AkitaError> {
        if self.physical_response_len() == 0 {
            return Err(AkitaError::InvalidSetup(
                "L2 norm proof requires a nonempty physical response".into(),
            ));
        }
        if let Self::LimbGram {
            physical_response_len,
            block_len,
            limb_count,
        } = self
        {
            LimbGramLayout::new(physical_response_len, block_len, limb_count)?;
        }
        Ok(())
    }

    #[must_use]
    pub const fn physical_response_len(self) -> usize {
        match self {
            Self::Direct {
                physical_response_len,
            }
            | Self::LimbGram {
                physical_response_len,
                ..
            } => physical_response_len,
        }
    }

    /// Checked LimbGram layout, or `None` for the direct shape.
    pub fn limb_gram_layout(self) -> Result<Option<LimbGramLayout>, AkitaError> {
        match self {
            Self::Direct { .. } => Ok(None),
            Self::LimbGram {
                physical_response_len,
                block_len,
                limb_count,
            } => LimbGramLayout::new(physical_response_len, block_len, limb_count).map(Some),
        }
    }

    #[must_use]
    pub fn subclaim_count(self) -> Option<usize> {
        match self.limb_gram_layout().ok()? {
            None => Some(usize::default()),
            Some(layout) => Some(layout.subclaim_count()),
        }
    }

    #[must_use]
    pub const fn virtual_evaluation_count(self) -> usize {
        match self {
            Self::Direct { .. } => 1,
            Self::LimbGram { limb_count, .. } => limb_count,
        }
    }

    pub(crate) fn append_descriptor_bytes(self, bytes: &mut Vec<u8>) {
        match self {
            Self::Direct {
                physical_response_len,
            } => {
                bytes.push(1);
                push_usize(bytes, physical_response_len);
            }
            Self::LimbGram {
                physical_response_len,
                block_len,
                limb_count,
            } => {
                bytes.push(2);
                push_usize(bytes, physical_response_len);
                push_usize(bytes, block_len);
                push_usize(bytes, limb_count);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limb_gram_layout_owns_block_and_pair_order() {
        let layout = LimbGramLayout::new(10, 4, 3).expect("layout");
        assert_eq!(
            layout.block_ranges().collect::<Vec<_>>(),
            vec![0..4, 4..8, 8..10]
        );
        assert_eq!(
            layout.limb_pairs().collect::<Vec<_>>(),
            vec![(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        );
        for block in 0..layout.block_count() {
            for (pair, (left, right)) in layout.limb_pairs().enumerate() {
                assert_eq!(
                    layout.subclaim_index(block, left, right),
                    Some(block * layout.pair_count() + pair)
                );
            }
        }
        assert_eq!(layout.subclaim_count(), 18);
        assert_eq!(layout.subclaim_index(0, 2, 1), None);
        assert_eq!(layout.subclaim_index(3, 0, 0), None);
    }
}
