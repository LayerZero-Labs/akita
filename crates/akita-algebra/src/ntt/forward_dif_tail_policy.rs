//! Policy for when i32 forward-DIF NTT may use the vectorized tail kernel.

/// Whether ring degree `D` is eligible for the vectorized forward-DIF tail path.
#[inline]
pub const fn forward_dif_tail_eligible<const D: usize>() -> bool {
    D.is_multiple_of(16)
}
