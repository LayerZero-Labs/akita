//! Protocol transcript contracts and implementations.

mod grinding;
mod label;
pub mod labels;
#[cfg(feature = "logging-transcript")]
mod logging;
#[cfg(any(
    all(feature = "transcript-blake2b", not(feature = "transcript-keccak")),
    all(feature = "transcript-keccak", not(feature = "transcript-blake2b"))
))]
mod sponge;

#[cfg(not(any(feature = "transcript-blake2b", feature = "transcript-keccak")))]
compile_error!("enable exactly one transcript backend: transcript-blake2b or transcript-keccak");

#[cfg(all(feature = "transcript-blake2b", feature = "transcript-keccak"))]
compile_error!("enable exactly one transcript backend: transcript-blake2b or transcript-keccak");

use akita_field::{CanonicalField, ExtField, FieldCore};
use akita_serialization::AkitaSerialize;

pub use grinding::{
    grinding_payload, grinding_predicate_accepts, preview_grinding_predicate,
    search_grinding_nonce, TranscriptChallengePreview, GRINDING_LITTLE_ENDIAN_BIT_ORDER,
    GRINDING_NONCE_SLACK_BITS, GRINDING_PREDICATE_BYTES, GRINDING_PREDICATE_LEN, MAX_GRINDING_BITS,
};
pub use label::Label;
#[cfg(feature = "logging-transcript")]
pub use logging::{clear_thread_events, thread_events, LoggingTranscript, TranscriptEvent};
#[cfg(any(
    all(feature = "transcript-blake2b", not(feature = "transcript-keccak")),
    all(feature = "transcript-keccak", not(feature = "transcript-blake2b"))
))]
pub use sponge::{AkitaTranscript, TranscriptSponge, PROTOCOL_TAG};

/// Transcript interface for protocol Fiat-Shamir transforms.
///
/// The protocol layer is label-aware and uses deterministic byte encoding for
/// all absorbed values.
pub trait Transcript<F>: Send
where
    F: FieldCore + CanonicalField,
{
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

    /// Record one compact public grinding-plan run for feature-gated audits.
    #[cfg(feature = "logging-transcript")]
    fn record_grinding_plan_query(&mut self, _site: &[u8], _multiplicity: u64) {}

    /// Record the protected challenge that discharged one pending PoW site.
    #[cfg(feature = "logging-transcript")]
    fn record_grinding_actual_query(&mut self, _site: &[u8], _label: &[u8]) {}

    /// Record the indexed coordinate range sampled by one live fold draw.
    #[cfg(feature = "logging-transcript")]
    fn record_fold_challenge_range(&mut self, _group_index: usize, _coordinate_count: usize) {}

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

    /// Squeeze one native 32-byte transcript block.
    fn challenge_block(&mut self, label: &[u8]) -> [u8; TRANSCRIPT_CHALLENGE_BLOCK_LEN];

    /// Apply one public transcript proof-of-work transition.
    ///
    /// A zero-bit target is an explicit no-op and returns `None`. A nonzero
    /// target absorbs the canonical payload and consumes one 32-byte predicate
    /// block. The site label is diagnostic and is not part of the production
    /// sponge input.
    fn grinding_predicate(
        &mut self,
        _site_label: &[u8],
        grind_bits: u8,
        nonce_bits: u8,
        nonce: u32,
    ) -> Option<[u8; GRINDING_PREDICATE_LEN]> {
        let grind_bits = std::num::NonZeroU8::new(grind_bits)?;
        let payload = grinding_payload(grind_bits, nonce_bits, nonce);
        self.append_bytes(labels::ABSORB_TRANSCRIPT_GRINDING, &payload);
        Some(self.challenge_block(labels::CHALLENGE_GRINDING_PREDICATE))
    }
}

/// Construction contract for owned transcript implementations.
///
/// Keeping construction separate from replay lets protocol adapters borrow an
/// existing transcript while still implementing [`Transcript`].
pub trait TranscriptFactory<F>: Transcript<F>
where
    F: FieldCore + CanonicalField,
{
    /// Construct a new transcript under a domain label.
    fn new(domain_label: &[u8]) -> Self;
}

/// Byte length of one native transcript squeeze block.
pub const TRANSCRIPT_CHALLENGE_BLOCK_LEN: usize = 32;
/// Byte length of every fold-challenge seed.
pub const FOLD_CHALLENGE_SEED_LEN: usize = TRANSCRIPT_CHALLENGE_BLOCK_LEN;

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
