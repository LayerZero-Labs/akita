use super::*;
use crate::proof::tail_segments::{
    tail_segment_layout_from_groups, SegmentTypedWitness, SegmentTypedWitnessShape,
};
use crate::{LevelParams, LevelParamsLike};

/// The sole supported terminal cleartext witness representation.
pub type CleartextWitnessProof<F> = SegmentTypedWitness<F>;

/// Shape descriptor for the sole supported terminal witness representation.
pub type CleartextWitnessShape = SegmentTypedWitnessShape;

pub fn segment_typed_witness_shape_from_groups<'a>(
    terminal_lp: &LevelParams,
    field_bits: u32,
    groups: impl IntoIterator<Item = (&'a dyn LevelParamsLike, usize, usize, usize)>,
) -> Result<CleartextWitnessShape, AkitaError> {
    Ok(SegmentTypedWitnessShape {
        layout: tail_segment_layout_from_groups(terminal_lp, groups, field_bits)?,
    })
}
