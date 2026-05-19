#![no_main]

use akita_serialization::AkitaDeserialize;
use akita_types::{AkitaBatchedProofShape, AkitaProofStepShape, DirectWitnessShape, LevelProofShape};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = DirectWitnessShape::deserialize_compressed(data, &());
    let _ = LevelProofShape::deserialize_compressed(data, &());
    let _ = AkitaProofStepShape::deserialize_compressed(data, &());
    let _ = AkitaBatchedProofShape::deserialize_compressed(data, &());
});
