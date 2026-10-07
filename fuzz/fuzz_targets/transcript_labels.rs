#![no_main]

use akita_transcript::{new_prover_channel, prover_field_challenge, public_bytes, ProtocolSiteId};
use jolt_field::Prime128Offset275;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let split = data.len().min(255);
    let (label, bytes) = data.split_at(split);

    let Ok(mut transcript) = new_prover_channel(b"akita-fuzz", label) else {
        return;
    };
    let _ = public_bytes(&mut transcript, ProtocolSiteId::default(), bytes);
    let _: Prime128Offset275 = prover_field_challenge(&mut transcript).unwrap();
});
