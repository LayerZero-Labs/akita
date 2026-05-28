use super::*;

/// Bit-packed balanced digits for the final-level witness vector.
///
/// Each element is a signed value in `[-b/2, b/2)` where `b = 2^bits_per_elem`,
/// stored in two's-complement using exactly `bits_per_elem` bits per value.
/// This reduces proof size by ~32x compared to storing full field elements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedDigits {
    /// Number of logical elements.
    pub num_elems: usize,
    /// Bits per element used for packing.
    pub bits_per_elem: u32,
    /// Bit-packed two's-complement data.
    pub data: Vec<u8>,
}

/// Terminal direct witness payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirectWitnessProof<F: FieldCore> {
    /// Packed small signed digits, used by the current recursive terminal
    /// witness.
    PackedDigits(PackedDigits),
    /// Raw field elements, for direct witnesses that are not naturally digit
    /// bounded.
    FieldElements(FlatRingVec<F>),
}

impl<F: FieldCore> DirectWitnessProof<F> {
    /// Borrow the packed-digits payload, if present.
    pub fn as_packed_digits(&self) -> Option<&PackedDigits> {
        match self {
            Self::PackedDigits(packed) => Some(packed),
            Self::FieldElements(_) => None,
        }
    }

    /// Borrow the raw field-element payload, if present.
    pub fn as_field_elements(&self) -> Option<&FlatRingVec<F>> {
        match self {
            Self::PackedDigits(_) => None,
            Self::FieldElements(field_elems) => Some(field_elems),
        }
    }

    /// Shape descriptor for this direct witness payload.
    pub fn shape(&self) -> DirectWitnessShape {
        match self {
            Self::PackedDigits(packed) => {
                DirectWitnessShape::PackedDigits((packed.num_elems, packed.bits_per_elem))
            }
            Self::FieldElements(field_elems) => {
                DirectWitnessShape::FieldElements(field_elems.coeff_len())
            }
        }
    }

    /// Number of logical field elements carried by this witness payload.
    pub fn num_elems(&self) -> usize {
        match self {
            Self::PackedDigits(packed) => packed.num_elems,
            Self::FieldElements(field_elems) => field_elems.coeff_len(),
        }
    }

    /// Decode packed terminal-witness digits into their logical signed stream.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] when the witness is not the
    /// canonical packed-digit representation or when the bit-packed payload is
    /// malformed.
    pub fn packed_i8_digits(&self) -> Result<Vec<i8>, AkitaError> {
        let Self::PackedDigits(packed) = self else {
            return Err(AkitaError::InvalidProof);
        };
        packed.check().map_err(|_| AkitaError::InvalidProof)?;
        (0..packed.num_elems)
            .map(|idx| packed.digit_at(idx).ok_or(AkitaError::InvalidProof))
            .collect()
    }

    /// Split this terminal direct witness into transcript-bound byte slices.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] when the witness is not canonical
    /// packed-digits form or the descriptor-bound terminal segment is invalid.
    pub fn terminal_transcript_parts(
        &self,
        layout: TerminalWitnessSegmentLayout,
    ) -> Result<TerminalWitnessTranscriptParts, AkitaError> {
        terminal_witness_transcript_parts(&self.packed_i8_digits()?, layout)
    }
}

/// Shape descriptor for deserializing a direct witness payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirectWitnessShape {
    /// Packed balanced digits.
    PackedDigits((usize, u32)),
    /// Raw field elements.
    FieldElements(usize),
}

impl PackedDigits {
    /// Smallest `bits_per_elem` that can encode every signed digit in `w`.
    pub fn required_bits_per_elem(w: &[i8]) -> u32 {
        let required_half_b = w.iter().fold(1i16, |acc, &signed| {
            let needed = if signed >= 0 {
                signed as i16 + 1
            } else {
                -(signed as i16)
            };
            acc.max(needed)
        });

        let mut bits = 1u32;
        let mut half_b = 1i16;
        while half_b < required_half_b {
            bits += 1;
            half_b <<= 1;
        }
        bits
    }

    /// Pack balanced i8 digits into bit-packed form.
    ///
    /// Each element must be in `[-b/2, b/2)` where `b = 2^log_basis`.
    ///
    /// # Panics
    ///
    /// Panics (in debug) if any element does not fit in `log_basis` bits.
    pub fn from_i8_digits(w: &[i8], log_basis: u32) -> Self {
        assert!(log_basis > 0 && log_basis <= 6, "log_basis out of range");
        let half_b = 1i8 << (log_basis - 1);

        let bits = log_basis as usize;
        let total_bits = w.len() * bits;
        let num_bytes = total_bits.div_ceil(8);
        let mut data = vec![0u8; num_bytes];

        for (i, &signed) in w.iter().enumerate() {
            debug_assert!(
                signed >= -half_b && signed < half_b,
                "digit {signed} out of range for log_basis={log_basis}"
            );
            let unsigned = (signed as u8) & ((1u8 << bits) - 1);
            let bit_offset = i * bits;
            let byte_idx = bit_offset / 8;
            let bit_idx = bit_offset % 8;
            data[byte_idx] |= unsigned << bit_idx;
            if bit_idx + bits > 8 {
                data[byte_idx + 1] |= unsigned >> (8 - bit_idx);
            }
        }

        Self {
            num_elems: w.len(),
            bits_per_elem: log_basis,
            data,
        }
    }

    /// Pack digits using at least `min_bits_per_elem`, widening if needed so
    /// every element in `w` fits the chosen two's-complement range.
    pub fn from_i8_digits_with_min_bits(w: &[i8], min_bits_per_elem: u32) -> Self {
        let bits_per_elem = min_bits_per_elem.max(Self::required_bits_per_elem(w));
        Self::from_i8_digits(w, bits_per_elem)
    }

    /// Decode a single packed signed digit.
    pub fn digit_at(&self, idx: usize) -> Option<i8> {
        if idx >= self.num_elems {
            return None;
        }

        let bits = self.bits_per_elem as usize;
        if bits == 0 || bits > 6 {
            return None;
        }
        let mask = (1u8 << bits) - 1;
        let sign_bit = 1u8 << (bits - 1);
        let bit_offset = idx.checked_mul(bits)?;
        let byte_idx = bit_offset / 8;
        let bit_idx = bit_offset % 8;
        let mut raw = (self.data.get(byte_idx)? >> bit_idx) & mask;
        if bit_idx + bits > 8 {
            raw |= (self.data.get(byte_idx + 1)? << (8 - bit_idx)) & mask;
        }

        Some(if raw & sign_bit != 0 {
            raw as i8 | !(mask as i8)
        } else {
            raw as i8
        })
    }

    /// Unpack to field elements.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the packed byte buffer is
    /// malformed relative to `num_elems`/`bits_per_elem`.
    pub fn to_field_elems<F: FieldCore + FromPrimitiveInt>(&self) -> Result<Vec<F>, AkitaError> {
        let mut out = Vec::with_capacity(self.num_elems);
        for i in 0..self.num_elems {
            let signed = self.digit_at(i).ok_or(AkitaError::InvalidProof)?;
            out.push(F::from_i64(signed as i64));
        }
        Ok(out)
    }

    /// Number of packed data bytes.
    pub fn packed_byte_len(&self) -> usize {
        self.data.len()
    }
}

impl AkitaSerialize for PackedDigits {
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

impl Valid for PackedDigits {
    fn check(&self) -> Result<(), SerializationError> {
        if self.bits_per_elem == 0 || self.bits_per_elem > 6 {
            return Err(SerializationError::InvalidData(
                "bits_per_elem out of range".to_string(),
            ));
        }
        let expected_bits = self
            .num_elems
            .checked_mul(self.bits_per_elem as usize)
            .ok_or(SerializationError::LengthLimitExceeded {
                len: u64::MAX,
                max: DEFAULT_MAX_SEQUENCE_LEN,
            })?;
        let expected_bytes = expected_bits.div_ceil(8);
        if self.data.len() != expected_bytes {
            return Err(SerializationError::InvalidData(
                "packed data length mismatch".to_string(),
            ));
        }
        Ok(())
    }
}

impl AkitaDeserialize for PackedDigits {
    /// `(num_elems, bits_per_elem)` — shape of the packed digit vector.
    type Context = (usize, u32);
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        _validate: Validate,
        ctx: &(usize, u32),
    ) -> Result<Self, SerializationError> {
        let (num_elems, bits_per_elem) = *ctx;
        if matches!(_validate, Validate::Yes) {
            DirectWitnessShape::PackedDigits(*ctx).check()?;
        }
        let num_bits = num_elems.checked_mul(bits_per_elem as usize).ok_or(
            SerializationError::LengthLimitExceeded {
                len: u64::MAX,
                max: DEFAULT_MAX_SEQUENCE_LEN,
            },
        )?;
        let num_bytes = num_bits.div_ceil(8);
        let mut data = vec![0u8; num_bytes];
        reader.read_exact(&mut data)?;
        let out = Self {
            num_elems,
            bits_per_elem,
            data,
        };
        out.check()?;
        Ok(out)
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for DirectWitnessProof<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::PackedDigits(packed) => packed.serialize_with_mode(&mut writer, compress),
            Self::FieldElements(field_elems) => {
                field_elems.serialize_with_mode(&mut writer, compress)
            }
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            Self::PackedDigits(packed) => packed.serialized_size(compress),
            Self::FieldElements(field_elems) => field_elems.serialized_size(compress),
        }
    }
}

impl<F: FieldCore + Valid> Valid for DirectWitnessProof<F> {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            Self::PackedDigits(packed) => packed.check(),
            Self::FieldElements(field_elems) => field_elems.check(),
        }
    }
}

impl<F: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for DirectWitnessProof<F>
{
    type Context = DirectWitnessShape;

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &DirectWitnessShape,
    ) -> Result<Self, SerializationError> {
        let out = match ctx {
            DirectWitnessShape::PackedDigits(shape) => Self::PackedDigits(
                PackedDigits::deserialize_with_mode(&mut reader, compress, validate, shape)?,
            ),
            DirectWitnessShape::FieldElements(num_coeffs) => Self::FieldElements(
                FlatRingVec::deserialize_with_mode(&mut reader, compress, validate, num_coeffs)?,
            ),
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}
