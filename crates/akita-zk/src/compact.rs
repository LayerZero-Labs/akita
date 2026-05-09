//! Compact signed-coefficient encodings for ZK responses.

use crate::error::ZkResult;
use crate::norm::{centered_i128, field_from_centered_i128};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore, PseudoMersenneField};
use akita_serialization::{AkitaSerialize, Compress, SerializationError};
use std::io::Write;

/// Compact two's-complement encoding for a vector of ring coefficients.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactRingVec {
    /// Number of ring elements represented.
    pub ring_len: usize,
    /// Ring degree.
    pub ring_degree: usize,
    /// Number of bits used per signed coefficient.
    pub bits_per_coeff: u32,
    /// Packed coefficient data.
    pub data: Vec<u8>,
}

impl CompactRingVec {
    /// Compute the smallest two's-complement width for `[-bound, bound]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the bound exceeds the supported packing range.
    pub fn bits_for_bound(bound: u128) -> ZkResult<u32> {
        if bound > i128::MAX as u128 {
            return Err(AkitaError::InvalidInput(
                "compact bound exceeds i128".to_string(),
            ));
        }
        let magnitude_bits = if bound == 0 {
            0
        } else {
            u128::BITS - bound.leading_zeros()
        };
        let bits = magnitude_bits + 1;
        if bits > 127 {
            return Err(AkitaError::InvalidInput(
                "compact coefficient width must be <= 127 bits".to_string(),
            ));
        }
        Ok(bits)
    }

    /// Pack a ring vector with the minimal width for `bound`.
    ///
    /// # Errors
    ///
    /// Returns an error if any coefficient lies outside `[-bound, bound]`.
    pub fn pack_with_bound<F, const D: usize>(
        values: &[CyclotomicRing<F, D>],
        bound: u128,
    ) -> ZkResult<Self>
    where
        F: FieldCore + CanonicalField + PseudoMersenneField,
    {
        let bits = Self::bits_for_bound(bound)?;
        Self::pack(values, bits, bound)
    }

    /// Pack a ring vector with an explicit two's-complement width.
    ///
    /// # Errors
    ///
    /// Returns an error if any coefficient lies outside `[-bound, bound]`, or
    /// if `bits_per_coeff` cannot represent the bound.
    pub fn pack<F, const D: usize>(
        values: &[CyclotomicRing<F, D>],
        bits_per_coeff: u32,
        bound: u128,
    ) -> ZkResult<Self>
    where
        F: FieldCore + CanonicalField + PseudoMersenneField,
    {
        if bits_per_coeff == 0 || bits_per_coeff > 127 {
            return Err(AkitaError::InvalidInput(
                "bits_per_coeff must be in 1..=127".to_string(),
            ));
        }
        if Self::bits_for_bound(bound)? > bits_per_coeff {
            return Err(AkitaError::InvalidInput(
                "bits_per_coeff cannot represent the requested bound".to_string(),
            ));
        }
        let total_coeffs = values
            .len()
            .checked_mul(D)
            .ok_or_else(|| AkitaError::InvalidInput("coefficient count overflow".to_string()))?;
        let total_bits = total_coeffs
            .checked_mul(bits_per_coeff as usize)
            .ok_or_else(|| AkitaError::InvalidInput("packed bit count overflow".to_string()))?;
        let mut data = vec![0u8; total_bits.div_ceil(8)];
        let mask = (1u128 << bits_per_coeff) - 1;

        for (coeff_idx, coeff) in values
            .iter()
            .flat_map(|ring| ring.coefficients().iter().copied())
            .enumerate()
        {
            let signed = centered_i128(coeff)?;
            if signed.unsigned_abs() > bound {
                return Err(AkitaError::InvalidInput(format!(
                    "coefficient {signed} exceeds compact bound {bound}"
                )));
            }
            let encoded = (signed as u128) & mask;
            write_bits(
                &mut data,
                coeff_idx * bits_per_coeff as usize,
                bits_per_coeff,
                encoded,
            );
        }

        Ok(Self {
            ring_len: values.len(),
            ring_degree: D,
            bits_per_coeff,
            data,
        })
    }

    /// Decode into a ring vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored shape is incompatible with `D`.
    pub fn unpack<F, const D: usize>(&self) -> ZkResult<Vec<CyclotomicRing<F, D>>>
    where
        F: FieldCore + CanonicalField,
    {
        if self.ring_degree != D {
            return Err(AkitaError::InvalidInput(format!(
                "compact ring degree {} does not match requested D={D}",
                self.ring_degree
            )));
        }
        if self.bits_per_coeff == 0 || self.bits_per_coeff > 127 {
            return Err(AkitaError::InvalidInput(
                "bits_per_coeff must be in 1..=127".to_string(),
            ));
        }
        let total_coeffs = self
            .ring_len
            .checked_mul(D)
            .ok_or_else(|| AkitaError::InvalidInput("coefficient count overflow".to_string()))?;
        let total_bits = total_coeffs
            .checked_mul(self.bits_per_coeff as usize)
            .ok_or_else(|| AkitaError::InvalidInput("packed bit count overflow".to_string()))?;
        let expected_bytes = total_bits.div_ceil(8);
        if self.data.len() != expected_bytes {
            return Err(AkitaError::InvalidInput(format!(
                "compact data length {} does not match expected {expected_bytes}",
                self.data.len()
            )));
        }
        validate_zero_padding_bits(&self.data, total_bits)?;

        let mut out = Vec::with_capacity(self.ring_len);
        for ring_idx in 0..self.ring_len {
            let mut coeffs = [F::zero(); D];
            for (j, coeff) in coeffs.iter_mut().enumerate() {
                let coeff_idx = ring_idx * D + j;
                let raw = read_bits(
                    &self.data,
                    coeff_idx * self.bits_per_coeff as usize,
                    self.bits_per_coeff,
                );
                let signed = decode_twos_complement(raw, self.bits_per_coeff)?;
                *coeff = field_from_centered_i128(signed)?;
            }
            out.push(CyclotomicRing::from_coefficients(coeffs));
        }
        Ok(out)
    }

    /// Number of packed response data bytes.
    pub fn packed_byte_len(&self) -> usize {
        self.data.len()
    }
}

impl AkitaSerialize for CompactRingVec {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        writer.write_all(&self.data)?;
        Ok(())
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        self.data.len()
    }
}

fn write_bits(data: &mut [u8], bit_offset: usize, bits: u32, mut value: u128) {
    let mut remaining = bits as usize;
    let mut offset = bit_offset;
    while remaining > 0 {
        let byte_idx = offset / 8;
        let bit_idx = offset % 8;
        let take = remaining.min(8 - bit_idx);
        let mask = (1u16 << take) - 1;
        data[byte_idx] |= ((value as u16 & mask) as u8) << bit_idx;
        value >>= take;
        offset += take;
        remaining -= take;
    }
}

fn read_bits(data: &[u8], bit_offset: usize, bits: u32) -> u128 {
    let mut remaining = bits as usize;
    let mut offset = bit_offset;
    let mut shift = 0usize;
    let mut out = 0u128;
    while remaining > 0 {
        let byte_idx = offset / 8;
        let bit_idx = offset % 8;
        let take = remaining.min(8 - bit_idx);
        let mask = (1u16 << take) - 1;
        let part = ((data[byte_idx] >> bit_idx) as u16) & mask;
        out |= (part as u128) << shift;
        offset += take;
        shift += take;
        remaining -= take;
    }
    out
}

fn validate_zero_padding_bits(data: &[u8], total_bits: usize) -> ZkResult<()> {
    let used_bits_in_last_byte = total_bits % 8;
    if used_bits_in_last_byte == 0 {
        return Ok(());
    }
    let Some(&last_byte) = data.last() else {
        return Ok(());
    };
    let used_mask = (1u8 << used_bits_in_last_byte) - 1;
    if last_byte & !used_mask != 0 {
        return Err(AkitaError::InvalidInput(
            "compact data has non-zero padding bits".to_string(),
        ));
    }
    Ok(())
}

fn decode_twos_complement(raw: u128, bits: u32) -> ZkResult<i128> {
    let sign_bit = 1u128 << (bits - 1);
    if raw & sign_bit == 0 {
        i128::try_from(raw)
            .map_err(|_| AkitaError::InvalidInput("positive compact value overflow".to_string()))
    } else {
        let modulus = 1u128 << bits;
        let magnitude = modulus - raw;
        let signed = i128::try_from(magnitude)
            .map_err(|_| AkitaError::InvalidInput("negative compact value overflow".to_string()))?;
        Ok(-signed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128OffsetA7F7;

    type F = Prime128OffsetA7F7;

    #[test]
    fn compact_ring_vec_roundtrips_29_bit_values() {
        const D: usize = 4;
        let mut coeffs = [F::zero(); D];
        coeffs[0] = field_from_centered_i128(144_880_516).unwrap();
        coeffs[1] = field_from_centered_i128(-144_880_516).unwrap();
        coeffs[2] = field_from_centered_i128(17).unwrap();
        coeffs[3] = field_from_centered_i128(-9).unwrap();
        let rings = vec![CyclotomicRing::from_coefficients(coeffs)];

        let compact = CompactRingVec::pack_with_bound(&rings, 144_880_516).unwrap();
        assert_eq!(compact.bits_per_coeff, 29);
        assert_eq!(compact.packed_byte_len(), 15);
        assert_eq!(compact.unpack::<F, D>().unwrap(), rings);
    }

    #[test]
    fn unpack_rejects_non_zero_padding_bits() {
        const D: usize = 4;
        let mut coeffs = [F::zero(); D];
        coeffs[0] = field_from_centered_i128(144_880_516).unwrap();
        let rings = vec![CyclotomicRing::from_coefficients(coeffs)];

        let mut compact = CompactRingVec::pack_with_bound(&rings, 144_880_516).unwrap();
        assert_eq!(compact.bits_per_coeff, 29);
        *compact.data.last_mut().unwrap() |= 0x80;

        assert!(compact.unpack::<F, D>().is_err());
    }
}
