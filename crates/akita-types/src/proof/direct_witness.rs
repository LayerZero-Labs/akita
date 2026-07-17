use super::*;
use crate::proof::tail_segments::{
    expand_segment_typed_to_i8_digits, tail_segment_layout_from_groups, SegmentTypedWitness,
    SegmentTypedWitnessShape, TerminalQuotientMode,
};
use crate::{LevelParams, LevelParamsLike};

/// Terminal direct witness payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CleartextWitnessProof<F: FieldCore> {
    /// Raw field elements, for direct witnesses that are not naturally digit
    /// bounded.
    FieldElements(RingVec<F>),
    /// Segment-typed terminal witness (`e`/`t`/`r` raw field, `z` Golomb-Rice).
    SegmentTyped(SegmentTypedWitness<F>),
}

impl<F: FieldCore> CleartextWitnessProof<F> {
    /// Borrow the segment-typed payload, if present.
    pub fn as_segment_typed(&self) -> Option<&SegmentTypedWitness<F>> {
        match self {
            Self::SegmentTyped(witness) => Some(witness),
            Self::FieldElements(_) => None,
        }
    }

    /// Borrow the raw field-element payload, if present.
    pub fn as_field_elements(&self) -> Option<&RingVec<F>> {
        match self {
            Self::SegmentTyped(_) => None,
            Self::FieldElements(field_elems) => Some(field_elems),
        }
    }

    /// Shape descriptor for this direct witness payload.
    pub fn shape(&self) -> CleartextWitnessShape {
        match self {
            Self::FieldElements(field_elems) => {
                CleartextWitnessShape::FieldElements(field_elems.coeff_len())
            }
            Self::SegmentTyped(witness) => {
                CleartextWitnessShape::SegmentTyped(SegmentTypedWitnessShape {
                    layout: witness.layout.clone(),
                })
            }
        }
    }

    /// Number of logical field elements carried by this witness payload.
    pub fn num_elems(&self) -> usize {
        match self {
            Self::FieldElements(field_elems) => field_elems.coeff_len(),
            Self::SegmentTyped(witness) => witness.layout.logical_num_elems,
        }
    }

    /// Decode the logical digit stream for stage-2 replay.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] when the witness cannot be decoded.
    pub fn logical_i8_digits<const D: usize>(
        &self,
        lp: &LevelParams,
        num_segments: usize,
    ) -> Result<Vec<i8>, AkitaError>
    where
        F: CanonicalField + HalvingField,
    {
        match self {
            Self::SegmentTyped(witness) => {
                expand_segment_typed_to_i8_digits::<D, F>(witness, lp, num_segments)
            }
            Self::FieldElements(_) => Err(AkitaError::InvalidProof),
        }
    }

    /// Split this terminal direct witness into transcript-bound byte slices.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] when the witness is not segment-typed
    /// or its canonical terminal segments cannot be encoded.
    pub fn terminal_transcript_parts(&self) -> Result<TerminalWitnessTranscriptParts, AkitaError>
    where
        F: CanonicalField + AkitaSerialize,
    {
        match self {
            Self::SegmentTyped(witness) => witness.terminal_transcript_parts(),
            Self::FieldElements(_) => Err(AkitaError::InvalidProof),
        }
    }
}

/// Shape descriptor for deserializing a direct witness payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CleartextWitnessShape {
    /// Raw field elements.
    FieldElements(usize),
    /// Segment-typed terminal witness.
    SegmentTyped(SegmentTypedWitnessShape),
}

impl CleartextWitnessShape {
    /// Whether `realized` is admitted by the scheduled witness shape.
    ///
    /// Segment-typed tails may serialize the Golomb `z` segment at its exact
    /// encoded length (prefixed on the wire) while the schedule carries the
    /// public upper bound.
    #[must_use]
    pub fn admits_realized(&self, realized: &Self) -> bool {
        match (self, realized) {
            (
                Self::SegmentTyped(scheduled),
                Self::SegmentTyped(SegmentTypedWitnessShape { layout }),
            ) => scheduled.layout.admits_realized(layout),
            (scheduled, other) => scheduled == other,
        }
    }
}

pub fn segment_typed_witness_shape_from_groups<'a>(
    terminal_lp: &LevelParams,
    field_bits: u32,
    groups: impl IntoIterator<Item = (&'a dyn LevelParamsLike, usize, usize, usize)>,
    num_segments: usize,
    quotient_mode: TerminalQuotientMode,
) -> Result<CleartextWitnessShape, AkitaError> {
    let layout = tail_segment_layout_from_groups(
        terminal_lp,
        groups,
        num_segments,
        field_bits,
        quotient_mode,
    )?;
    Ok(CleartextWitnessShape::SegmentTyped(
        SegmentTypedWitnessShape { layout },
    ))
}

impl<F: FieldCore + CanonicalField + AkitaSerialize> AkitaSerialize for CleartextWitnessProof<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::FieldElements(field_elems) => {
                field_elems.serialize_with_mode(&mut writer, compress)
            }
            Self::SegmentTyped(witness) => witness.serialize_with_mode(&mut writer, compress),
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            Self::FieldElements(field_elems) => field_elems.serialized_size(compress),
            Self::SegmentTyped(witness) => witness.serialized_size(compress),
        }
    }
}

impl<F: FieldCore + Valid> Valid for CleartextWitnessProof<F> {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            Self::FieldElements(field_elems) => field_elems.check(),
            Self::SegmentTyped(witness) => witness.check(),
        }
    }
}

impl<F: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for CleartextWitnessProof<F>
{
    type Context = CleartextWitnessShape;

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &CleartextWitnessShape,
    ) -> Result<Self, SerializationError> {
        let out = match ctx {
            CleartextWitnessShape::FieldElements(num_coeffs) => Self::FieldElements(
                RingVec::deserialize_with_mode(&mut reader, compress, validate, num_coeffs)?,
            ),
            CleartextWitnessShape::SegmentTyped(shape) => Self::SegmentTyped(
                SegmentTypedWitness::deserialize_with_mode(&mut reader, compress, validate, shape)?,
            ),
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}
