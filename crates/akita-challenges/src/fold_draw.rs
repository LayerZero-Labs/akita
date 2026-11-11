//! Fold-challenge drawing: one challenge message per commitment group, over
//! the live channel or a prover-side preview.
//!
//! A draw absorbs nothing. The schedule bound in the instance descriptor fixes
//! the family, and the group's position in the transcript fixes its index and
//! counts, so the message continues the squeeze run before it.

use akita_error::AkitaError;
use akita_transcript::ProofChannel;

/// Diagnostic capture at the shared fold-draw boundary. Sparse decoding is
/// owned by Jolt on preview; capture the public message without replacing it.
#[cfg(all(not(target_arch = "riscv64"), feature = "experiment-capture"))]
fn capture_message(message: &[u8]) {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ENABLED.get_or_init(|| std::env::var_os("AKITA_CAPTURE_CHALLENGES").is_some()) {
        use std::fmt::Write;
        let mut text = String::from("EXPERIMENT_FOLD_MESSAGE ");
        for byte in message {
            let _ = write!(text, "{byte:02x}");
        }
        eprintln!("{text}");
    }
}

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
        self.preview.challenge_message(len, |message| {
            #[cfg(all(not(target_arch = "riscv64"), feature = "experiment-capture"))]
            capture_message(message);
            decode(message)
        })
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
        self.state.challenge_message(len, |message| {
            #[cfg(all(not(target_arch = "riscv64"), feature = "experiment-capture"))]
            capture_message(message);
            decode(message)
        })
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
            #[cfg(all(not(target_arch = "riscv64"), feature = "experiment-capture"))]
            capture_message(message);
            decode(message)?.map(Some).ok_or(AkitaError::InvalidProof)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_transcript::{new_prover_channel, new_verifier_channel};

    #[test]
    fn capture_keeps_live_draws_equal_and_rejects_exhaustion() {
        let mut prover = new_prover_channel(b"fold-capture-parity");
        let mut verifier = new_verifier_channel(b"fold-capture-parity", &[]);
        let mut reference = new_prover_channel(b"fold-capture-parity");
        for (group, len) in [0, 1, 40, 513].into_iter().enumerate() {
            reference.context(fold_record(2, group as u32, len));
            let expected = reference
                .challenge_message(len, |m| Ok(m.to_vec()))
                .unwrap();
            let actual = ProverFoldDraw::new(&mut prover, 2, group as u32)
                .draw_message(len, |m| Ok(Some(m.to_vec())))
                .unwrap()
                .unwrap();
            let verified = VerifierFoldDraw::new(&mut verifier, 2, group as u32)
                .draw_message(len, |m| Ok(Some(m.to_vec())))
                .unwrap()
                .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(verified, expected);
        }
        assert!(prover.narg_string().is_empty());
        let exhausted =
            VerifierFoldDraw::new(&mut verifier, 2, 4).draw_message::<()>(40, |_| Ok(None));
        assert!(matches!(exhausted, Err(AkitaError::InvalidProof)));
    }
}
