use super::{reduce128, reduce64, BinaryField128, BinaryField192};

#[target_feature(enable = "aes")]
pub(super) unsafe fn multiply128(a: BinaryField128, b: BinaryField128) -> BinaryField128 {
    let [a0, a1] = a.0;
    let [b0, b1] = b.0;
    let d0 = std::arch::aarch64::vmull_p64(a0, b0);
    let d1 = std::arch::aarch64::vmull_p64(a1, b1);
    let cross = std::arch::aarch64::vmull_p64(a0 ^ a1, b0 ^ b1) ^ d0 ^ d1;
    reduce128([
        d0 as u64,
        (d0 >> 64) as u64 ^ cross as u64,
        d1 as u64 ^ (cross >> 64) as u64,
        (d1 >> 64) as u64,
    ])
}

#[target_feature(enable = "aes")]
pub(super) unsafe fn multiply192(a: BinaryField192, b: BinaryField192) -> BinaryField192 {
    let [a0, a1, a2] = a.0;
    let [b0, b1, b2] = b.0;
    let d0 = std::arch::aarch64::vmull_p64(a0, b0);
    let d1 = std::arch::aarch64::vmull_p64(a1, b1);
    let d2 = std::arch::aarch64::vmull_p64(a2, b2);
    let c01 = std::arch::aarch64::vmull_p64(a0 ^ a1, b0 ^ b1) ^ d0 ^ d1;
    let c02 = std::arch::aarch64::vmull_p64(a0 ^ a2, b0 ^ b2) ^ d0 ^ d2;
    let c12 = std::arch::aarch64::vmull_p64(a1 ^ a2, b1 ^ b2) ^ d1 ^ d2;
    BinaryField192([
        reduce64(d0 ^ c12),
        reduce64(c01 ^ c12 ^ d2),
        reduce64(d1 ^ c02 ^ d2),
    ])
}

#[target_feature(enable = "aes")]
pub(super) unsafe fn equality128(point: &[BinaryField128], output: &mut [BinaryField128]) {
    if output.is_empty() {
        return;
    }
    output[0] = BinaryField128::ONE;
    for (axis, &r) in point.iter().enumerate() {
        let width = 1 << axis;
        let (low, high) = output.split_at_mut(width);
        for j in 0..width {
            // SAFETY: the enclosing function carries the same target feature.
            let hi = unsafe { multiply128(low[j], r) };
            high[j] = hi;
            low[j] += hi;
        }
    }
}

#[target_feature(enable = "aes")]
pub(super) unsafe fn equality192(point: &[BinaryField192], output: &mut [BinaryField192]) {
    if output.is_empty() {
        return;
    }
    output[0] = BinaryField192::ONE;
    for (axis, &r) in point.iter().enumerate() {
        let width = 1 << axis;
        let (low, high) = output.split_at_mut(width);
        for j in 0..width {
            // SAFETY: the enclosing function carries the same target feature.
            let hi = unsafe { multiply192(low[j], r) };
            high[j] = hi;
            low[j] += hi;
        }
    }
}
