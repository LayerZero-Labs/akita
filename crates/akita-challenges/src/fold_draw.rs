//! Fold-challenge drawing: one challenge message per commitment group, over
//! the live channel or a prover-side preview.
//!
//! A draw absorbs nothing. The schedule bound in the instance descriptor fixes
//! the family, and the group's position in the transcript fixes its index and
//! counts, so the message continues the squeeze run before it.

use akita_error::AkitaError;
use akita_transcript::ProofChannel;

pub trait FoldDraw {
    /// Squeeze one group's challenge message of exactly `len` bytes and
    /// decode it. `decode` returns `None` when an operator-norm pool runs out;
    /// the verifier adapter rejects then.
    fn draw_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<T>, AkitaError>,
    ) -> Result<Option<T>, AkitaError>;
}

fn fold_record(
    level: u32,
    group: u32,
    message_len: usize,
) -> akita_transcript::ProtocolContextRecord {
    akita_transcript::ProtocolContextRecord::new(
        akita_transcript::ProtocolSiteId {
            family: akita_transcript::SITE_FAMILY_FOLD_CHALLENGE,
            level,
            group,
            ..akita_transcript::ProtocolSiteId::default()
        }
        .to_bytes(),
        akita_transcript::ProtocolMessageKind::Challenge as u32,
        0,
        0,
        message_len as u64,
    )
}

/// One group-local fold draw against a candidate public state.
#[cfg(feature = "prover")]
pub struct PreviewFoldDraw<'a> {
    preview: &'a mut akita_transcript::FoldPreview,
}

#[cfg(feature = "prover")]
impl<'a> PreviewFoldDraw<'a> {
    /// Bind this short-lived draw adapter to one schedule-derived fold group.
    #[must_use]
    pub const fn new(preview: &'a mut akita_transcript::FoldPreview) -> Self {
        Self { preview }
    }
}

#[cfg(feature = "prover")]
impl FoldDraw for PreviewFoldDraw<'_> {
    fn draw_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<T>, AkitaError>,
    ) -> Result<Option<T>, AkitaError> {
        self.preview.challenge_message(len, decode)
    }
}

/// One group-local fold draw against the live prover channel.
pub struct ProverFoldDraw<'a> {
    state: &'a mut akita_transcript::ProverChannel,
    level: u32,
    group: u32,
}

impl<'a> ProverFoldDraw<'a> {
    /// Bind this short-lived draw adapter to one schedule-derived fold group.
    #[must_use]
    pub const fn new(
        state: &'a mut akita_transcript::ProverChannel,
        level: u32,
        group: u32,
    ) -> Self {
        Self {
            state,
            level,
            group,
        }
    }
}

impl FoldDraw for ProverFoldDraw<'_> {
    fn draw_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<T>, AkitaError>,
    ) -> Result<Option<T>, AkitaError> {
        self.state.context(fold_record(self.level, self.group, len));
        self.state.challenge_message(len, decode)
    }
}

/// One group-local fold draw against the live verifier channel, clear or
/// masked: fold challenges are public in both modes.
pub struct VerifierFoldDraw<'a, S> {
    state: &'a mut S,
    level: u32,
    group: u32,
}

impl<'a, S: akita_transcript::ProofChannel> VerifierFoldDraw<'a, S> {
    /// Bind this short-lived draw adapter to one schedule-derived fold group.
    #[must_use]
    pub const fn new(state: &'a mut S, level: u32, group: u32) -> Self {
        Self {
            state,
            level,
            group,
        }
    }
}

impl<S: akita_transcript::ProofChannel> FoldDraw for VerifierFoldDraw<'_, S> {
    fn draw_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<T>, AkitaError>,
    ) -> Result<Option<T>, AkitaError> {
        self.state.context(fold_record(self.level, self.group, len));
        self.state.challenge_message(len, |message| {
            decode(message)?.map(Some).ok_or(AkitaError::InvalidProof)
        })
    }
}
