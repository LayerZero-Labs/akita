#![no_main]

use akita_transcript::{
    native_prover_field_challenge, new_native_prover, public_native_bytes_prover, ProtocolSiteId,
};
use jolt_field::Prime128Offset275;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let split = data.len().min(255);
    let (label, bytes) = data.split_at(split);

    let Ok(mut transcript) = new_native_prover(b"akita-fuzz", label) else {
        return;
    };
    let _ = public_native_bytes_prover(&mut transcript, ProtocolSiteId::default(), bytes);
    let _: Prime128Offset275 = native_prover_field_challenge(&mut transcript).unwrap();
});
