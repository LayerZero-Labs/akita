//! Four-way bit interleaving for generic x86-64 builds.

#[target_feature(enable = "bmi2")]
#[inline]
pub(super) unsafe fn pack_bmi2(words: [u128; 4]) -> [u64; 11] {
    use std::arch::x86_64::_pdep_u64;
    let mut packed = [0; 11];
    for (component, mut word) in words.into_iter().enumerate() {
        for destination in packed.iter_mut().take(8) {
            *destination |= _pdep_u64(word as u64, 0x1111_1111_1111_1111) << component;
            word >>= 16;
        }
    }
    packed
}
