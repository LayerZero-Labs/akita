#![no_main]

use akita_field::Prime128Offset275;
use akita_transcript::{Blake2bTranscript, KeccakTranscript, Transcript};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let split = data.len() / 2;
    let (label, bytes) = data.split_at(split);

    let mut blake = Blake2bTranscript::<Prime128Offset275>::new(b"akita-fuzz");
    blake.append_bytes(label, bytes);
    let _ = blake.challenge_scalar(label);
    let _ = blake.challenge_bytes(label, bytes.len().min(96));

    let mut keccak = KeccakTranscript::<Prime128Offset275>::new(b"akita-fuzz");
    keccak.append_bytes(label, bytes);
    let _ = keccak.challenge_scalar(label);
    let _ = keccak.challenge_bytes(label, bytes.len().min(96));
});
