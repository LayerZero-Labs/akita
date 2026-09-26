//! Scalar bit-matrix operations shared by the hardware kernels.

/// Transpose one 8-by-8 bit matrix stored as eight row bytes.
#[inline(always)]
pub(super) fn transpose8(mut value: u64) -> u64 {
    let mut swap = (value ^ (value >> 7)) & 0x00aa_00aa_00aa_00aa;
    value ^= swap ^ (swap << 7);
    swap = (value ^ (value >> 14)) & 0x0000_cccc_0000_cccc;
    value ^= swap ^ (swap << 14);
    swap = (value ^ (value >> 28)) & 0x0000_0000_f0f0_f0f0;
    value ^ swap ^ (swap << 28)
}
