#![allow(missing_docs)]

#[cfg(feature = "logging-transcript")]
use akita_transcript::TranscriptEvent;

#[cfg(feature = "logging-transcript")]
fn main() {
    use akita_config::proof_optimized::fp128;
    use akita_config::CommitmentConfig;
    use akita_field::{CanonicalField, Fp64};
    use akita_transcript::{labels, AkitaTranscript, LoggingTranscript, Transcript};

    type F = Fp64<4294967197>;

    let mut transcript =
        LoggingTranscript::wrap(AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL));
    transcript.bind_instance_bytes(b"transcript-schedule/example-descriptor");
    transcript.append_bytes(labels::ABSORB_COMMITMENT, b"commitment");
    transcript.append_field(
        labels::ABSORB_EVALUATION_CLAIMS,
        &F::from_canonical_u128_reduced(42),
    );
    let _ = transcript.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
    transcript.append_bytes(labels::ABSORB_TERMINAL_E_HAT, b"terminal-e-hat");
    let _ = transcript.challenge_scalar(labels::CHALLENGE_SPARSE_CHALLENGE);
    transcript.append_bytes(labels::ABSORB_TERMINAL_W_REMAINDER, b"terminal-w-remainder");
    let _ = transcript.challenge_scalar(labels::CHALLENGE_RING_SWITCH);
    let _ = transcript.challenge_scalar(labels::CHALLENGE_TAU1);

    transcript.assert_smell_checks();
    println!(
        "Akita transcript schedule example using D={}",
        fp128::D64OneHot::D
    );
    for (index, event) in transcript.events().iter().enumerate() {
        println!("{index:02}: {}", format_event(event));
    }
}

#[cfg(feature = "logging-transcript")]
fn format_event(event: &TranscriptEvent) -> String {
    match event {
        TranscriptEvent::Preamble {
            bytes_digest,
            bytes_len,
        } => format!(
            "preamble descriptor len={bytes_len} digest={}",
            hex_digest(bytes_digest)
        ),
        TranscriptEvent::Absorb {
            label,
            bytes_digest,
            bytes_len,
        } => format!(
            "absorb label={} len={bytes_len} digest={}",
            label_text(label),
            hex_digest(bytes_digest)
        ),
        TranscriptEvent::Squeeze { label, len } => {
            format!("squeeze label={} len={len}", label_text(label))
        }
        TranscriptEvent::Wire {
            label,
            bytes_digest,
            bytes_len,
        } => format!(
            "wire label={} len={bytes_len} digest={}",
            label_text(label),
            hex_digest(bytes_digest)
        ),
    }
}

#[cfg(feature = "logging-transcript")]
fn label_text(label: &[u8]) -> String {
    std::str::from_utf8(label)
        .map(str::to_owned)
        .unwrap_or_else(|_| hex_digest(label))
}

#[cfg(feature = "logging-transcript")]
fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(not(feature = "logging-transcript"))]
fn main() {
    eprintln!("enable with `cargo run -p akita-pcs --features logging-transcript --example transcript_schedule`");
}
