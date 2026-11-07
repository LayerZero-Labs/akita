//! Lazy fixed-width draws with lossless deferred diagnostic recording.

use crate::{ProofChannel, ProtocolContextRecord, ProtocolMessageKind, ProtocolSiteId};
use akita_error::AkitaError;

/// Consecutive 32-byte messages at one site, with no intervening channel work.
///
/// Every call squeezes exactly one original message. Contexts are flushed on
/// drop, including a failed draw's context. The exclusive channel borrow keeps
/// their order and proof boundary unchanged while allowing native channels to
/// record one counted run instead of touching thread-local logs for each draw.
pub struct ChallengeBlocks<'a, C: ProofChannel> {
    channel: &'a mut C,
    context: ProtocolContextRecord,
    draws: usize,
}

impl<'a, C: ProofChannel> ChallengeBlocks<'a, C> {
    /// Start a lazy run. An unused run records no context and draws no bytes.
    pub fn new(channel: &'a mut C, site: ProtocolSiteId) -> Self {
        Self {
            channel,
            context: ProtocolContextRecord::new(
                site.to_bytes(),
                ProtocolMessageKind::Challenge as u32,
                0,
                0,
                32,
            ),
            draws: 0,
        }
    }

    /// Draw the next message, retaining its context even if the channel fails.
    pub fn next_block(&mut self) -> Result<[u8; 32], AkitaError> {
        self.draws = self.draws.checked_add(1).ok_or(AkitaError::InvalidProof)?;
        self.channel.challenge_array()
    }

    /// Number of attempted 32-byte draws, including a channel failure if any.
    pub fn draws(&self) -> usize {
        self.draws
    }
}

impl<C: ProofChannel> Drop for ChallengeBlocks<'_, C> {
    fn drop(&mut self) {
        self.channel.context_repeated(self.context, self.draws);
    }
}

#[cfg(all(test, feature = "logging-transcript"))]
mod tests {
    use super::*;
    use crate::{clear_thread_events, new_prover_channel, new_verifier_channel, thread_events};

    #[test]
    fn lazy_runs_match_individual_messages_and_error_prefixes() {
        let site = ProtocolSiteId::default();
        let context = ProtocolContextRecord::new(
            site.to_bytes(),
            ProtocolMessageKind::Challenge as u32,
            0,
            0,
            32,
        );
        for count in [0, 1, 65, 16_385] {
            clear_thread_events();
            let mut original = new_prover_channel(b"lazy-blocks");
            let expected = (0..count)
                .map(|_| {
                    original.context(context);
                    original.challenge_array::<32>().unwrap()
                })
                .collect::<Vec<_>>();
            let events = thread_events();
            clear_thread_events();
            let mut candidate = new_prover_channel(b"lazy-blocks");
            let mut blocks = ChallengeBlocks::new(&mut candidate, site);
            for expected in expected {
                assert_eq!(blocks.next_block().unwrap(), expected);
            }
            assert_eq!(blocks.draws(), count);
            drop(blocks);
            assert_eq!(thread_events(), events);
            let log = crate::thread_event_log().unwrap();
            assert_eq!(log.len(), events.len());
            assert_eq!(log.events().collect::<Vec<_>>(), events);
            candidate.public(b"continue");
            original.public(b"continue");
            assert_eq!(
                candidate.challenge_array::<32>().unwrap(),
                original.challenge_array::<32>().unwrap()
            );
        }
        clear_thread_events();
        let mut verifier = new_verifier_channel(b"lazy-blocks", &[]);
        assert!(verifier.guard::<()>(Err(AkitaError::InvalidProof)).is_err());
        let mut blocks = ChallengeBlocks::new(&mut verifier, site);
        assert!(blocks.next_block().is_err());
        drop(blocks);
        assert_eq!(
            thread_events(),
            vec![crate::TranscriptEvent::Context(context)]
        );
        let log = crate::thread_event_log().unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(log.runs().collect::<Vec<_>>(), vec![(context, 1)]);
        assert_eq!(log.events().collect::<Vec<_>>(), thread_events());
        clear_thread_events();
    }
}
