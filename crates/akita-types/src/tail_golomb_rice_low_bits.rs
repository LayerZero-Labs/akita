//! Cap-derived Rice low-bit width for terminal Golomb `z` (descriptor-bound wire rule).

use akita_field::AkitaError;

use crate::golomb_rice::rice_low_bits_for_cap;

/// Wire low-bits rule: `wire = cap_rice_low_bits - delta`.
pub const WIRE_RICE_LOW_BITS_RULE_SECURITY_MINUS_DELTA: u8 = 1;

/// Profile-calibrated tightening vs [`rice_low_bits_for_cap`]: honest tails sit ~2 low bits below
/// `floor(log2(cap))` on CI profile cells.
pub const WIRE_RICE_LOW_BITS_DELTA: u8 = 2;

/// Cap-derived Rice low-bit width: `floor(log2(cap))` for worst-case `|n| ≤ cap`.
#[must_use]
pub fn cap_rice_low_bits(cap: u128) -> u32 {
    rice_low_bits_for_cap(cap)
}

/// Wire Rice low-bit width from a descriptor-bound cap→wire rule.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when `rule_id` is unsupported.
pub fn wire_rice_low_bits_from_rule(
    cap: u128,
    rule_id: u8,
    delta: u8,
) -> Result<u32, AkitaError> {
    match rule_id {
        WIRE_RICE_LOW_BITS_RULE_SECURITY_MINUS_DELTA => {
            Ok(cap_rice_low_bits(cap).saturating_sub(u32::from(delta)))
        }
        other => Err(AkitaError::InvalidSetup(format!(
            "unsupported terminal z wire rice low-bits rule id {other}"
        ))),
    }
}

/// Wire Rice low-bit width under [`crate::FoldLinfProtocolBinding::CURRENT`].
///
/// Under [`WIRE_RICE_LOW_BITS_RULE_SECURITY_MINUS_DELTA`]: `rice_low_bits_for_cap(cap) - δ` with
/// δ = [`WIRE_RICE_LOW_BITS_DELTA`]. Typical coefficients pay `wire_rice_low_bits` fixed low
/// bits; coefficients near `±cap` pay longer unary runs.
#[must_use]
pub fn wire_rice_low_bits(cap: u128) -> u32 {
    wire_rice_low_bits_from_rule(
        cap,
        WIRE_RICE_LOW_BITS_RULE_SECURITY_MINUS_DELTA,
        WIRE_RICE_LOW_BITS_DELTA,
    )
    .expect("active wire rule is valid")
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
                cap_rice_low_bits(cap).saturating_sub(u32::from(WIRE_RICE_LOW_BITS_DELTA)),
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
                let decoded = golomb_rice_decode_vec(
                    &encoded,
                    1,
                    rice_low_bits,
                    zigzag_w,
                    max_quotient,
                )
                .expect("decode");
                assert_eq!(decoded, [n], "cap={cap} n={n}");
            }
        }
    }
}
