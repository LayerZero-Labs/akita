//! Canonical transcript proof-of-work transition and prover preview search.

use std::num::NonZeroU8;

/// Extra nonce bits used to make honest proof-of-work exhaustion negligible.
pub const GRINDING_NONCE_SLACK_BITS: u8 = 7;
/// Largest proof-of-work target supported by the current transition.
pub const MAX_GRINDING_BITS: u8 = 25;
/// Byte length of one transcript proof-of-work predicate in array contexts.
pub const GRINDING_PREDICATE_LEN: usize = crate::TRANSCRIPT_CHALLENGE_BLOCK_LEN;
/// Byte length bound into the grinding policy encoding.
pub const GRINDING_PREDICATE_BYTES: u8 = GRINDING_PREDICATE_LEN as u8;
/// Low-bit-first predicate bit order.
pub const GRINDING_LITTLE_ENDIAN_BIT_ORDER: u8 = 0;

const GRINDING_PAYLOAD_DOMAIN: &[u8; 28] = b"akita/transcript-grinding/v1";
const GRINDING_PAYLOAD_BYTES: usize = GRINDING_PAYLOAD_DOMAIN.len() + 6;

/// Prover-only challenge preview over a scratch copy of the transcript state.
pub trait TranscriptChallengePreview {
    /// Absorb each hypothetical payload and squeeze one 32-byte block after
    /// each one. Return the final block without changing the live transcript.
    fn preview_challenge_block(
        &self,
        absorb_payloads: &[&[u8]],
    ) -> [u8; crate::TRANSCRIPT_CHALLENGE_BLOCK_LEN];
}

/// Build the canonical fixed-width payload for one nonzero grinding target.
#[must_use]
pub fn grinding_payload(
    grind_bits: NonZeroU8,
    nonce_bits: u8,
    counter: u32,
) -> [u8; GRINDING_PAYLOAD_BYTES] {
    let mut payload = [0u8; GRINDING_PAYLOAD_BYTES];
    payload[..GRINDING_PAYLOAD_DOMAIN.len()].copy_from_slice(GRINDING_PAYLOAD_DOMAIN);
    payload[GRINDING_PAYLOAD_DOMAIN.len()] = grind_bits.get();
    payload[GRINDING_PAYLOAD_DOMAIN.len() + 1] = nonce_bits;
    payload[(GRINDING_PAYLOAD_DOMAIN.len() + 2)..].copy_from_slice(&counter.to_le_bytes());
    payload
}

/// Return whether the first `grind_bits` low-order predicate bits are zero.
#[must_use]
pub fn grinding_predicate_accepts(
    predicate: &[u8; GRINDING_PREDICATE_LEN],
    grind_bits: NonZeroU8,
) -> bool {
    // `grind_bits` is a `u8`, so `whole_bytes` is at most `u8::MAX / 8 == 31`,
    // which indexes a `GRINDING_PREDICATE_LEN`-byte predicate in bounds.
    const _: () = assert!((u8::MAX as usize / u8::BITS as usize) < GRINDING_PREDICATE_LEN);
    let grind_bits = usize::from(grind_bits.get());
    let whole_bytes = grind_bits / u8::BITS as usize;
    let remaining_bits = grind_bits % u8::BITS as usize;
    predicate[..whole_bytes].iter().all(|&byte| byte == 0)
        && (remaining_bits == 0 || predicate[whole_bytes] & ((1u8 << remaining_bits) - 1) == 0)
}

/// Preview one candidate predicate without changing the live transcript.
#[must_use]
pub fn preview_grinding_predicate(
    preview: &(impl TranscriptChallengePreview + ?Sized),
    grind_bits: u8,
    nonce_bits: u8,
    counter: u32,
) -> Option<[u8; GRINDING_PREDICATE_LEN]> {
    let grind_bits = NonZeroU8::new(grind_bits)?;
    let payload = grinding_payload(grind_bits, nonce_bits, counter);
    Some(preview.preview_challenge_block(&[payload.as_slice()]))
}

/// Return the first accepted bounded nonce, or `None` on exhaustion.
///
/// A zero-bit target is a no-op and returns zero without previewing the
/// transcript. Nonzero targets require the canonical `g + 7` width. The loop
/// uses a `u64` endpoint so a 32-bit nonce range does not overflow.
#[must_use]
pub fn search_grinding_nonce(
    preview: &(impl TranscriptChallengePreview + ?Sized),
    grind_bits: u8,
    nonce_bits: u8,
) -> Option<u32> {
    let Some(grind_bits_nonzero) = NonZeroU8::new(grind_bits) else {
        return (nonce_bits == 0).then_some(u32::default());
    };
    if grind_bits > MAX_GRINDING_BITS
        || nonce_bits != grind_bits.checked_add(GRINDING_NONCE_SLACK_BITS)?
        || nonce_bits > u32::BITS as u8
    {
        return None;
    }
    let attempts = 1u64.checked_shl(u32::from(nonce_bits))?;
    (0..attempts).find_map(|candidate| {
        let counter = u32::try_from(candidate).ok()?;
        let predicate = preview_grinding_predicate(preview, grind_bits, nonce_bits, counter)?;
        grinding_predicate_accepts(&predicate, grind_bits_nonzero).then_some(counter)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    struct FixedPreview {
        predicate: [u8; GRINDING_PREDICATE_LEN],
        calls: Cell<usize>,
    }

    impl TranscriptChallengePreview for FixedPreview {
        fn preview_challenge_block(
            &self,
            _absorb_payloads: &[&[u8]],
        ) -> [u8; GRINDING_PREDICATE_LEN] {
            self.calls.set(self.calls.get() + 1);
            self.predicate
        }
    }

    #[test]
    fn payload_encoding_is_exact() {
        let nonce_bytes = [0x12, 0x34, 0x56, 0x78];
        let payload = grinding_payload(
            NonZeroU8::new(9).unwrap(),
            16,
            u32::from_le_bytes(nonce_bytes),
        );
        let mut expected = b"akita/transcript-grinding/v1".to_vec();
        expected.extend_from_slice(&[9, 16]);
        expected.extend_from_slice(&nonce_bytes);
        assert_eq!(payload.as_slice(), expected);
    }

    #[test]
    fn predicate_checks_low_bits_at_byte_boundaries() {
        for grind_bits in [1, 8, 9, 16, 17, MAX_GRINDING_BITS] {
            let grind_bits_nonzero = NonZeroU8::new(grind_bits).unwrap();
            let mut predicate = [0u8; GRINDING_PREDICATE_LEN];
            assert!(grinding_predicate_accepts(&predicate, grind_bits_nonzero));
            let rejected_bit = usize::from(grind_bits - 1);
            predicate[rejected_bit / 8] = 1 << (rejected_bit % 8);
            assert!(!grinding_predicate_accepts(&predicate, grind_bits_nonzero));
            predicate.fill(0);
            let first_unchecked_bit = usize::from(grind_bits);
            predicate[first_unchecked_bit / 8] = 1 << (first_unchecked_bit % 8);
            assert!(grinding_predicate_accepts(&predicate, grind_bits_nonzero));
        }
    }

    #[test]
    fn zero_bit_search_is_a_no_op() {
        let preview = FixedPreview {
            predicate: [u8::MAX; GRINDING_PREDICATE_LEN],
            calls: Cell::new(0),
        };
        assert_eq!(search_grinding_nonce(&preview, 0, 0), Some(0));
        assert_eq!(preview.calls.get(), 0);
    }

    #[test]
    fn bounded_search_reports_exhaustion() {
        let preview = FixedPreview {
            predicate: [u8::MAX; GRINDING_PREDICATE_LEN],
            calls: Cell::new(0),
        };
        assert_eq!(search_grinding_nonce(&preview, 1, 8), None);
        assert_eq!(preview.calls.get(), 1 << 8);
    }

    #[test]
    fn search_accepts_maximum_canonical_width() {
        let preview = FixedPreview {
            predicate: [0; GRINDING_PREDICATE_LEN],
            calls: Cell::new(0),
        };
        let nonce_bits = MAX_GRINDING_BITS + GRINDING_NONCE_SLACK_BITS;
        assert_eq!(nonce_bits, u32::BITS as u8);
        assert_eq!(
            search_grinding_nonce(&preview, MAX_GRINDING_BITS, nonce_bits),
            Some(0)
        );
        assert_eq!(preview.calls.get(), 1);
    }

    #[test]
    fn search_rejects_noncanonical_widths() {
        let preview = FixedPreview {
            predicate: [0; GRINDING_PREDICATE_LEN],
            calls: Cell::new(0),
        };
        assert_eq!(search_grinding_nonce(&preview, 1, 7), None);
        assert_eq!(search_grinding_nonce(&preview, 1, 9), None);
        assert_eq!(
            search_grinding_nonce(&preview, MAX_GRINDING_BITS + 1, 32),
            None
        );
        assert_eq!(preview.calls.get(), 0);
    }
}
