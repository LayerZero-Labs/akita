//! Protocol transcript contracts and implementations.

mod label;
pub mod labels;
#[cfg(feature = "logging-transcript")]
mod logging;
mod sponge;

use akita_serialization::AkitaSerialize;
use jolt_field::{CanonicalField, ExtField, FieldCore};

pub use label::Label;
#[cfg(feature = "logging-transcript")]
pub use logging::{clear_thread_events, thread_events, LoggingTranscript, TranscriptEvent};
pub use sponge::{AkitaTranscript, TranscriptSponge, PROTOCOL_TAG};

/// Transcript interface for protocol Fiat-Shamir transforms.
///
/// The protocol layer is label-aware and uses deterministic byte encoding for
/// all absorbed values.
pub trait Transcript<F>: Send
where
    F: FieldCore + CanonicalField,
{
    /// Construct a new transcript under a domain label.
    fn new(domain_label: &[u8]) -> Self;

    /// Bind canonical instance-descriptor bytes before replaying a proof.
    ///
    /// Implementations must absorb these bytes with transcript-specific domain
    /// separation. The method is required so custom transcript backends cannot
    /// accidentally skip Akita instance binding.
    fn bind_instance_bytes(&mut self, instance_bytes: &[u8]);

    /// Record a verifier-side structured proof-field use for logging checks.
    fn record_wire_serde<S: AkitaSerialize>(&mut self, _label: &[u8], _s: &S) {}

    /// Record verifier-side canonical bytes for logging checks.
    fn record_wire_bytes(&mut self, _label: &[u8], _bytes: &[u8]) {}

    /// Record a structured proof field for logging checks *and* absorb it into
    /// the transcript, in one call.
    ///
    /// `record_wire_*` alone is a no-op in production — only the paired
    /// `append_*` binds the value into the sponge / Fiat-Shamir state. Keeping
    /// the two as separate adjacent calls means a future edit can silently drop
    /// the `append_*` and remove a value from the transcript with no compile
    /// error and no failure outside the `logging-transcript` feature. Prefer
    /// this helper at every wire-value absorb site so the pair cannot drift.
    fn absorb_and_record_serde<S: AkitaSerialize>(&mut self, label: &[u8], s: &S) {
        self.record_wire_serde(label, s);
        self.append_serde(label, s);
    }

    /// Bytes counterpart of [`Self::absorb_and_record_serde`].
    fn absorb_and_record_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        self.record_wire_bytes(label, bytes);
        self.append_bytes(label, bytes);
    }

    /// Append labeled raw bytes.
    fn append_bytes(&mut self, label: &[u8], bytes: &[u8]);

    /// Append a field element with deterministic encoding.
    fn append_field(&mut self, label: &[u8], x: &F);

    /// Append a serializable protocol value.
    fn append_serde<S: AkitaSerialize>(&mut self, label: &[u8], s: &S);

    /// Derive a challenge scalar under the provided label.
    fn challenge_scalar(&mut self, label: &[u8]) -> F;

    /// Squeeze `len` challenge bytes under the provided label.
    fn challenge_bytes(&mut self, label: &[u8], len: usize) -> Vec<u8>;
}

/// Preview-only seed derivation for prover-side fold Fiat–Shamir grinding.
pub trait FoldChallengeSeedPreview {
    /// Derive challenge bytes after a hypothetical absorb.
    fn preview_challenge_bytes_after_absorb(&self, absorb_payload: &[u8], len: usize) -> Vec<u8>;

    /// Derive challenge bytes after a hypothetical absorb/squeeze chain.
    fn preview_challenge_bytes_after_absorb_chain(
        &self,
        absorbs: &[&[u8]],
        squeeze_lens: &[usize],
    ) -> Vec<u8>;
}

/// Append an extension-field element by absorbing its base-field coordinates.
pub fn append_ext_field<F, E, T>(transcript: &mut T, label: &[u8], x: &E)
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
    T: Transcript<F>,
{
    let coeffs = x.to_base_vec();
    if E::EXT_DEGREE == 1 {
        for coeff in coeffs.iter().take(1) {
            transcript.append_field(label, coeff);
        }
        return;
    }

    for (limb, coeff) in coeffs.iter().enumerate() {
        transcript.append_field(&ext_limb_label(label, limb), coeff);
    }
}

/// Sample an extension-field challenge from base-field transcript limbs.
///
/// This draws `E::EXT_DEGREE` base-field challenges under distinct limb labels
/// and assembles the extension element with [`ExtField::from_base_slice`].
pub fn sample_ext_challenge<F, E, T>(transcript: &mut T, label: &[u8]) -> E
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
    T: Transcript<F>,
{
    if E::EXT_DEGREE == 1 {
        let coeff = transcript.challenge_scalar(label);
        return E::from_base_slice(&[coeff]);
    }

    let coeffs = (0..E::EXT_DEGREE)
        .map(|limb| transcript.challenge_scalar(&ext_limb_label(label, limb)))
        .collect::<Vec<_>>();
    E::from_base_slice(&coeffs)
}

const EXT_LIMB_LABEL_SUFFIX_LEN: usize = 12;

/// Return the diagnostic label used for an extension-field limb.
///
/// Production [`AkitaTranscript`] bytes remain positional; this helper exists
/// so logging tests and label validators do not duplicate the limb-label wire
/// format.
#[must_use]
pub fn ext_limb_label(label: &[u8], limb: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(label.len() + EXT_LIMB_LABEL_SUFFIX_LEN);
    out.extend_from_slice(label);
    out.push(0xff);
    out.extend_from_slice(&(limb as u64).to_le_bytes());
    out.extend_from_slice(b"ext");
    out
}

/// Return the base diagnostic label when `label` names an extension-field
/// limb, otherwise `None`.
#[must_use]
pub fn ext_limb_base_label(label: &[u8]) -> Option<&[u8]> {
    let suffix_start = label.len().checked_sub(EXT_LIMB_LABEL_SUFFIX_LEN)?;
    let (&marker, rest) = label[suffix_start..].split_first()?;
    (marker == 0xff && rest.len() == 11 && rest[8..] == *b"ext").then_some(&label[..suffix_start])
}

/// Return whether `candidate` is an extension-field limb label for `base`.
#[must_use]
pub fn is_ext_limb_label(candidate: &[u8], base: &[u8]) -> bool {
    ext_limb_base_label(candidate).is_some_and(|candidate_base| candidate_base == base)
}
