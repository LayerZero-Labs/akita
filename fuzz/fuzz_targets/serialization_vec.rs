#![no_main]

use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = Vec::<u8>::deserialize_compressed(data, &());
    let _ = Vec::<bool>::deserialize_compressed(data, &());

    if let Ok(decoded) = Vec::<u8>::deserialize_compressed(data, &()) {
        let mut encoded = Vec::new();
        if decoded.serialize_compressed(&mut encoded).is_ok() {
            let reparsed = Vec::<u8>::deserialize_compressed(&encoded[..], &());
            assert_eq!(reparsed.ok(), Some(decoded));
        }
    }
});
