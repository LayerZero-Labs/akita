#![no_main]

use akita_serialization::AkitaDeserialize;
use akita_types::{
    AkitaBatchedProofShape, LevelProofShape, SegmentTypedWitnessShape, TerminalLevelProofShape,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = SegmentTypedWitnessShape::deserialize_compressed(data, &());
    let _ = LevelProofShape::deserialize_compressed(data, &());
    let _ = TerminalLevelProofShape::deserialize_compressed(data, &());
    let _ = AkitaBatchedProofShape::deserialize_compressed(data, &());
});
