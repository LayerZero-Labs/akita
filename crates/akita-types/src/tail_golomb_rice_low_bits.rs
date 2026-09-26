//! Offline selection of the terminal Golomb-Rice low-bit width.
//!
//! Schedule generation stores the selected width in the terminal response
//! shape. Runtime encoding and decoding consume that exact stored value.

/// Profile-calibrated tightening vs [`rice_low_bits_for_cap`]: honest tails sit ~2 low bits below
/// `floor(log2(cap))` on CI profile cells.
const OFFLINE_RICE_LOW_BITS_DELTA: u32 = 2;

/// Cap-derived Rice low-bit width: `floor(log2(cap))` for worst-case `|n| ≤ cap`.
#[must_use]
pub fn cap_rice_low_bits(cap: u128) -> u32 {
    rice_low_bits_for_cap(cap)
}

/// Select the Rice low-bit width emitted into a terminal response shape.
#[must_use]
pub fn wire_rice_low_bits(cap: u128) -> u32 {
    cap_rice_low_bits(cap).saturating_sub(OFFLINE_RICE_LOW_BITS_DELTA)
}

/// Rice low-bit width from a per-coordinate magnitude scale (e.g. fold `‖z‖_inf` cap).
///
/// Equals `floor(log2(scale))` for `scale > 1`; divisor is `2^rice_low_bits`.
#[must_use]
pub fn rice_low_bits_for_cap(scale: u128) -> u32 {
    if scale <= 1 {
        return 0;
    }
    u128::BITS - 1 - scale.leading_zeros()
}

/// Average-case planner bit budget per `z` coordinate from cap-derived low-bit width.
#[must_use]
pub fn tail_z_planner_bits_per_coord(cap_rice_low_bits: u32) -> usize {
    (cap_rice_low_bits as usize).saturating_add(2)
}

/// Conservative planner estimate for an L2-bounded Golomb-Rice payload.
///
/// The zigzag magnitude is at most `2 * |z_i|`. Cauchy-Schwarz gives
/// `sum_i |z_i| <= floor(sqrt(num_values * l2_sq_cap))`, which bounds the
/// complete unary-quotient contribution without a distributional assumption.
/// This estimate does not replace the scheduled payload cap enforced on wire.
#[must_use]
pub fn golomb_rice_l2_planner_payload_bytes(
    num_values: usize,
    l2_sq_cap: u128,
    rice_low_bits: u32,
) -> Option<usize> {
    let num_values_u128 = u128::try_from(num_values).ok()?;
    let sum_abs_bound = num_values_u128.checked_mul(l2_sq_cap)?.isqrt();
    let quotient_sum_bound = sum_abs_bound.checked_mul(2)?.checked_shr(rice_low_bits)?;
    let fixed_bits = num_values.checked_mul(rice_low_bits as usize + 1)?;
    let quotient_bits = usize::try_from(quotient_sum_bound).ok()?;
    fixed_bits
        .checked_add(quotient_bits)?
        .checked_add(7)
        .map(|bits| bits / 8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golomb_rice::{
        golomb_rice_decode_vec, golomb_rice_encode_vec, golomb_rice_max_quotient_for_cap,
        golomb_rice_zigzag_width, zigzag_encode,
    };

    fn max_quotient_in_cap_range(cap: u128, rice_low_bits: u32, zigzag_w: u32) -> u64 {
        let cap_i64 = cap as i64;
        (-cap_i64..=cap_i64)
            .map(|n| {
                let u = zigzag_encode(n, zigzag_w).unwrap();
                if rice_low_bits == 0 {
                    u
                } else {
                    u >> rice_low_bits
                }
            })
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn wire_rice_low_bits_is_cap_minus_delta_on_shipping_caps() {
        for cap in [504u128, 1008, 1568, 2016] {
            assert_eq!(
                wire_rice_low_bits(cap),
                cap_rice_low_bits(cap).saturating_sub(OFFLINE_RICE_LOW_BITS_DELTA),
                "cap={cap}"
            );
        }
    }

    #[test]
    fn wire_rice_low_bits_round_trips_full_fold_cap_range() {
        for cap in [504u128, 1008, 1568, 2016] {
            let rice_low_bits = wire_rice_low_bits(cap);
            let zigzag_w = golomb_rice_zigzag_width(cap);
            let max_quotient =
                golomb_rice_max_quotient_for_cap(cap, rice_low_bits, zigzag_w).expect("max q");
            assert_eq!(
                max_quotient,
                max_quotient_in_cap_range(cap, rice_low_bits, zigzag_w),
                "cap={cap}"
            );
            let cap_i64 = cap as i64;
            for n in [-cap_i64, -1, 0, 1, cap_i64] {
                let encoded =
                    golomb_rice_encode_vec(&[n], rice_low_bits, zigzag_w).expect("encode");
                let decoded =
                    golomb_rice_decode_vec(&encoded, 1, rice_low_bits, zigzag_w, max_quotient, Ok)
                        .expect("decode");
                assert_eq!(decoded, [n], "cap={cap} n={n}");
            }
        }
    }
}
