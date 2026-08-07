//! Traits for appending commitment objects to protocol transcripts.

use akita_transcript::Transcript;
use jolt_field::{CanonicalEncoding, Field};

/// Protocol object that can be absorbed into a transcript.
pub trait AppendToTranscript<F>
where
    F: Field + CanonicalEncoding,
{
    /// Append this object to a transcript using the provided event label.
    fn append_to_transcript<T: Transcript<F>>(&self, label: &[u8], transcript: &mut T);
}

impl<F, A> AppendToTranscript<F> for &A
where
    F: Field + CanonicalEncoding,
    A: AppendToTranscript<F> + ?Sized,
{
    fn append_to_transcript<T: Transcript<F>>(&self, label: &[u8], transcript: &mut T) {
        (*self).append_to_transcript(label, transcript);
    }
}
