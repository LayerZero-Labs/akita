//! Transcript encoding for the new-field synthetic fixture (Jolt 790c3fe2).
use jolt_algebra::CanonicalEncoding;
use jolt_transcript::{BytesMsg, ProverTranscript, VerifierTranscript};
use jolt_zk::masking::ZkError;

/// Sends `values` as one message of their canonical bytes.
pub(super) fn send_values<F: CanonicalEncoding, T: ProverTranscript + ?Sized>(
    transcript: &mut T,
    values: &[F],
) {
    let mut bytes = vec![0u8; values.len() * F::NUM_BYTES];
    for (chunk, value) in bytes.chunks_exact_mut(F::NUM_BYTES).zip(values) {
        value.to_bytes_le(chunk);
    }
    transcript.send(&BytesMsg(bytes));
}

/// Reads one message of exactly `len` canonical field elements.
///
/// # Errors
///
/// [`ZkError::TranscriptRejected`] on a truncated string, a message of
/// another length, or a non-canonical element.
pub(super) fn receive_values<F: CanonicalEncoding, T: VerifierTranscript + ?Sized>(
    transcript: &mut T,
    len: usize,
) -> Result<Vec<F>, ZkError> {
    let BytesMsg(bytes) = transcript.prover_message::<BytesMsg>()?;
    let expected = len
        .checked_mul(F::NUM_BYTES)
        .ok_or(ZkError::TranscriptRejected)?;
    if bytes.len() != expected {
        return Err(ZkError::TranscriptRejected);
    }
    bytes
        .chunks_exact(F::NUM_BYTES)
        .map(|chunk| F::from_bytes_le_checked(chunk).ok_or(ZkError::TranscriptRejected))
        .collect()
}

/// Sends `bytes` as one message.
pub(super) fn send_bytes<T: ProverTranscript + ?Sized>(transcript: &mut T, bytes: &[u8]) {
    transcript.send(&BytesMsg(bytes));
}

/// Reads one message of exactly `len` bytes.
///
/// # Errors
///
/// [`ZkError::TranscriptRejected`] on a truncated string or another length.
pub(super) fn receive_bytes<T: VerifierTranscript + ?Sized>(
    transcript: &mut T,
    len: usize,
) -> Result<Vec<u8>, ZkError> {
    let BytesMsg(bytes) = transcript.prover_message::<BytesMsg>()?;
    if bytes.len() == len {
        Ok(bytes)
    } else {
        Err(ZkError::TranscriptRejected)
    }
}
