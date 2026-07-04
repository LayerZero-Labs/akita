//! Traits for appending commitment objects to protocol transcripts.

use akita_field::{CanonicalField, FieldCore};
use akita_transcript::Transcript;

/// Protocol object that can be absorbed into a transcript.
pub trait AppendToTranscript<F>
where
    F: FieldCore + CanonicalField,
{
    /// Append this object to a transcript using the provided event label.
    fn append_to_transcript<T: Transcript<F>>(&self, label: &[u8], transcript: &mut T);
}

impl<F, A> AppendToTranscript<F> for &A
where
    F: FieldCore + CanonicalField,
    A: AppendToTranscript<F> + ?Sized,
{
    fn append_to_transcript<T: Transcript<F>>(&self, label: &[u8], transcript: &mut T) {
        (*self).append_to_transcript(label, transcript);
    }
}
