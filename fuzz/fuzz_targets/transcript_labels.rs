#![no_main]

use jolt_field::Prime128Offset275;
use akita_transcript::{AkitaTranscript, Transcript};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let split = data.len().min(255);
    let (label, bytes) = data.split_at(split);

    let mut transcript = AkitaTranscript::<Prime128Offset275>::new(b"akita-fuzz");
    transcript.append_bytes(label, bytes);
    let _ = transcript.challenge_scalar(label);
    let _ = transcript.challenge_bytes(label, bytes.len().min(96));
});
