//! Arithmetic fixtures independent of the withdrawn root protocol.

use akita_algebra::binary::{field_switch::SwitchField, BinaryField128, BinaryField192};
use akita_challenges::FoldDraw;
use akita_error::AkitaError;
use rand::{rngs::StdRng, RngCore};

pub(crate) const Q: u32 = 33_568_993;

pub(crate) trait TestHost: SwitchField {
    fn random_source(rng: &mut StdRng) -> Self::Source;
}

impl TestHost for BinaryField128 {
    fn random_source(rng: &mut StdRng) -> Self::Source {
        u128::from(rng.next_u64()) | (u128::from(rng.next_u64()) << 64)
    }
}

impl TestHost for BinaryField192 {
    fn random_source(rng: &mut StdRng) -> Self::Source {
        rng.next_u64()
    }
}

pub(crate) struct FixedDraw;
impl FoldDraw for FixedDraw {
    fn draw_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<T>, AkitaError>,
    ) -> Result<Option<T>, AkitaError> {
        decode(&vec![0x53; len])
    }
}
