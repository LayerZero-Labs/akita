//! Akita's field challenges are `jolt_field::Field::random` draws on the
//! transcript's squeeze stream. This pins, for every production base field,
//! that the draw is exact rejection sampling at the field's own width: a
//! candidate at the modulus is rejected and the next one is used, and the
//! largest residue is accepted.

#![expect(
    clippy::indexing_slicing,
    reason = "the replayed stream is sized by the test"
)]

use jolt_field::{
    Field, One, Prime128Offset275, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59, Ring,
};
use rand_core::{Error, RngCore};

/// Replays fixed bytes and counts how many a draw consumed.
struct Replay {
    bytes: Vec<u8>,
    consumed: usize,
}

impl RngCore for Replay {
    fn next_u32(&mut self) -> u32 {
        let mut bytes = [0; 4];
        self.fill_bytes(&mut bytes);
        u32::from_le_bytes(bytes)
    }

    fn next_u64(&mut self) -> u64 {
        let mut bytes = [0; 8];
        self.fill_bytes(&mut bytes);
        u64::from_le_bytes(bytes)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        dest.copy_from_slice(&self.bytes[self.consumed..self.consumed + dest.len()]);
        self.consumed += dest.len();
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

fn draw<F: Field>(bytes: Vec<u8>) -> (F, usize) {
    let mut rng = Replay { bytes, consumed: 0 };
    (F::random(&mut rng), rng.consumed)
}

/// `F` has modulus `2^bits - offset`.
fn check<F: Field + Ring + One>(bits: u32, offset: u128) {
    let width = (bits / 8) as usize;
    let modulus = 1u128.checked_shl(bits).unwrap_or(0).wrapping_sub(offset);
    let encode = |value: u128| value.to_le_bytes()[..width].to_vec();

    let (value, consumed) = draw::<F>([encode(modulus), encode(5)].concat());
    assert_eq!(
        (value, consumed),
        (F::from_u64(5), 2 * width),
        "{bits}-bit: p rejects"
    );
    let (value, consumed) = draw::<F>(encode(modulus - 1));
    assert_eq!(
        (value, consumed),
        (-F::one(), width),
        "{bits}-bit: p - 1 accepts"
    );
}

#[test]
fn production_fields_draw_by_exact_rejection_at_their_width() {
    check::<Prime32Offset99>(32, 99);
    check::<Prime64Offset59>(64, 59);
    check::<Prime128OffsetA7F7>(128, 0xFFFF_A7F7);
    check::<Prime128Offset275>(128, 275);
}
