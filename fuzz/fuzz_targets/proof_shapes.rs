#![no_main]

use akita_serialization::AkitaDeserialize;
use akita_types::{
    AkitaBatchedProofShape, CleartextWitnessShape, LevelProofShape, TerminalLevelProofShape,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = CleartextWitnessShape::deserialize_compressed(data, &());
    let _ = LevelProofShape::deserialize_compressed(data, &());
    let _ = TerminalLevelProofShape::deserialize_compressed(data, &());
    let _ = AkitaBatchedProofShape::deserialize_compressed(data, &());
});
