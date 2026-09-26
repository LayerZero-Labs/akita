//! Akita proof-of-work policy shared by native Spongefish replay.

use std::num::NonZeroU8;

/// Extra nonce bits used to make honest proof-of-work exhaustion negligible.
pub const GRINDING_NONCE_SLACK_BITS: u8 = 7;
/// Largest proof-of-work target supported by the current transition.
pub const MAX_GRINDING_BITS: u8 = 25;
/// Byte length of one proof-of-work predicate.
pub const GRINDING_PREDICATE_LEN: usize = crate::TRANSCRIPT_CHALLENGE_BLOCK_LEN;
/// Byte length bound into the grinding policy encoding.
pub const GRINDING_PREDICATE_BYTES: u8 = GRINDING_PREDICATE_LEN as u8;
/// Low-bit-first predicate bit order.
pub const GRINDING_LITTLE_ENDIAN_BIT_ORDER: u8 = 0;

/// Return whether the first `grind_bits` low-order predicate bits are zero.
#[must_use]
pub fn grinding_predicate_accepts(
    predicate: &[u8; GRINDING_PREDICATE_LEN],
    grind_bits: NonZeroU8,
) -> bool {
    const _: () = assert!((u8::MAX as usize / u8::BITS as usize) < GRINDING_PREDICATE_LEN);
    let grind_bits = usize::from(grind_bits.get());
    let whole_bytes = grind_bits / u8::BITS as usize;
    let remaining_bits = grind_bits % u8::BITS as usize;
    predicate[..whole_bytes].iter().all(|&byte| byte == 0)
        && (remaining_bits == 0 || predicate[whole_bytes] & ((1u8 << remaining_bits) - 1) == 0)
}

pub(crate) fn search_grinding_nonce_with(
    grind_bits: u8,
    nonce_bits: u8,
    mut predicate_for_candidate: impl FnMut(u32) -> Option<[u8; GRINDING_PREDICATE_LEN]>,
) -> Option<(u32, [u8; GRINDING_PREDICATE_LEN])> {
    let Some(grind_bits_nonzero) = NonZeroU8::new(grind_bits) else {
        return (nonce_bits == 0).then_some((u32::default(), [0; GRINDING_PREDICATE_LEN]));
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
        let predicate = predicate_for_candidate(counter)?;
        grinding_predicate_accepts(&predicate, grind_bits_nonzero).then_some((counter, predicate))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicate_checks_low_bits_at_byte_boundaries() {
        for grind_bits in [1, 8, 9, 16, 17, MAX_GRINDING_BITS] {
            let bits = NonZeroU8::new(grind_bits).unwrap();
            let mut predicate = [0u8; GRINDING_PREDICATE_LEN];
            assert!(grinding_predicate_accepts(&predicate, bits));
            let rejected_bit = usize::from(grind_bits - 1);
            predicate[rejected_bit / 8] = 1 << (rejected_bit % 8);
            assert!(!grinding_predicate_accepts(&predicate, bits));
        }
    }

    #[test]
    fn bounded_search_enforces_canonical_width_and_exhaustion() {
        assert_eq!(
            search_grinding_nonce_with(0, 0, |_| unreachable!()),
            Some((0, [0; GRINDING_PREDICATE_LEN]))
        );
        assert_eq!(search_grinding_nonce_with(1, 7, |_| Some([0; 32])), None);
        assert_eq!(
            search_grinding_nonce_with(1, 8, |_| Some([u8::MAX; 32])),
            None
        );
        assert_eq!(
            search_grinding_nonce_with(1, 8, |_| Some([0; 32])),
            Some((0, [0; 32]))
        );
    }
}
