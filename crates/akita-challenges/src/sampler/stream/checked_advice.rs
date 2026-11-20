//! Eligibility and authenticated decoding for packed checked advice.

use super::{read_bits, SparseChallengeStream};
use crate::checked_hint::AdviceEncoding;
use akita_error::AkitaError;

impl SparseChallengeStream {
    /// The schedule selects the record encoding without reading a tag.
    pub(crate) const fn advice_encoding(&self) -> AdviceEncoding {
        if self.count < 1024 || self.shell.rejection.is_some() {
            return AdviceEncoding::Positions;
        }
        match (
            self.shell.ring_d,
            self.shell.config.count_pm1,
            self.shell.config.count_pm2,
        ) {
            (64, 31, 10) => AdviceEncoding::Quartets,
            (128, 31, 0) => AdviceEncoding::Ternary,
            _ => AdviceEncoding::Positions,
        }
    }

    pub(crate) fn decode_checked_hints(&self, message: &[u8]) -> Result<Vec<[u64; 4]>, AkitaError> {
        if message.len() != self.message_len
            || self.advice_encoding() != AdviceEncoding::Quartets
            || self.shell.candidate_bits != 128
        {
            return Err(AkitaError::InvalidProof);
        }
        let message_words = crate::checked_hint::native_words(message)?;
        crate::checked_hint::take(
            self.count.checked_mul(4).ok_or(AkitaError::InvalidProof)?,
            |hints| {
                let mut output = Vec::new();
                output
                    .try_reserve_exact(self.count)
                    .map_err(|_| AkitaError::InvalidProof)?;
                for index in 0..self.count {
                    let low = u64::from_le(message_words[index * 2]);
                    let high = u64::from_le(message_words[index * 2 + 1]);
                    let rank = ((high as u128) << 23) | ((low >> 41) as u128);
                    if rank >= self.shell.rank_space {
                        return Err(AkitaError::InvalidProof);
                    }
                    let signs = low & ((1u64 << 41) - 1);
                    let pos = index * 4;
                    let words = [hints[pos], hints[pos + 1], hints[pos + 2], hints[pos + 3]];
                    // Upper padding is a canonical zero, so unused bits cannot carry unchecked state.
                    if (words[0] | words[1] | words[2] | words[3]) >> 40 != 0 {
                        return Err(AkitaError::InvalidProof);
                    }
                    crate::checked_hint::validate(&words, rank, self.shell.pm1_rank_space, signs)?;
                    output.push(words);
                }
                Ok(output)
            },
        )
    }

    pub(crate) fn decode_ternary_hints(&self, message: &[u8]) -> Result<Vec<[u64; 4]>, AkitaError> {
        if message.len() != self.message_len || self.advice_encoding() != AdviceEncoding::Ternary {
            return Err(AkitaError::InvalidProof);
        }
        crate::checked_hint::take(
            self.count.checked_mul(4).ok_or(AkitaError::InvalidProof)?,
            |hints| {
                let mut output = Vec::new();
                output
                    .try_reserve_exact(self.count)
                    .map_err(|_| AkitaError::InvalidProof)?;
                for (index, chunk) in hints.chunks_exact(4).enumerate() {
                    let start = index
                        .checked_mul(self.shell.candidate_bits)
                        .ok_or(AkitaError::InvalidProof)?;
                    let signs = read_bits(message, start, 31)? as u64;
                    let rank = read_bits(
                        message,
                        start.checked_add(31).ok_or(AkitaError::InvalidProof)?,
                        self.shell.rank_bits,
                    )?;
                    let words = chunk.try_into().map_err(|_| AkitaError::InvalidProof)?;
                    crate::checked_hint::ternary::validate(&words, rank, signs)?;
                    output.push(words);
                }
                Ok(output)
            },
        )
    }
}
