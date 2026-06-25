//! Cap-derived Rice low-bit width for terminal Golomb `z` (code-constant rule; not descriptor-bound).

use crate::golomb_rice::rice_low_bits_for_cap;

/// Wire low-bits rule tag for logs/tests (not transcript-bound).
pub const WIRE_RICE_LOW_BITS_RULE_SECURITY_MINUS_DELTA: u8 = 1;

/// Active cap→wire low-bits rule ([`WIRE_RICE_LOW_BITS_RULE_SECURITY_MINUS_DELTA`] today).
pub const WIRE_RICE_LOW_BITS_ACTIVE_RULE: u8 = WIRE_RICE_LOW_BITS_RULE_SECURITY_MINUS_DELTA;

/// Profile-calibrated tightening vs [`rice_low_bits_for_cap`]: honest tails sit ~2 low bits below
/// `floor(log2(cap))` on CI profile cells.
pub const WIRE_RICE_LOW_BITS_DELTA: u32 = 2;

/// Cap-derived Rice low-bit width: `floor(log2(cap))` for worst-case `|n| ≤ cap`.
#[must_use]
pub fn cap_rice_low_bits(cap: u128) -> u32 {
    rice_low_bits_for_cap(cap)
}

/// Wire Rice low-bit width for terminal `z` Golomb encode/decode.
///
/// This is **not** [`crate::golomb_rice::sample_optimal_rice_low_bits`]: that minimizes total
/// bits on a realized witness sample and is witness-dependent. It is also **not** a
/// `min_sound_low_bits` search: the codec round-trips every `n ∈ [-cap, cap]` even at `0`
/// via the escape path, so `min_sound_low_bits` is always `0` and carries no pricing signal.
///
/// Under [`WIRE_RICE_LOW_BITS_RULE_SECURITY_MINUS_DELTA`]: `rice_low_bits_for_cap(cap) - δ` with
/// δ = [`WIRE_RICE_LOW_BITS_DELTA`]. Typical coefficients pay `wire_rice_low_bits` fixed low
/// bits; coefficients near `±cap` pay longer unary runs but stay below
/// [`crate::golomb_rice::GOLOMB_RICE_Q_MAX`] at shipping caps.
#[must_use]
pub fn wire_rice_low_bits(cap: u128) -> u32 {
    match WIRE_RICE_LOW_BITS_ACTIVE_RULE {
        WIRE_RICE_LOW_BITS_RULE_SECURITY_MINUS_DELTA => {
            cap_rice_low_bits(cap).saturating_sub(WIRE_RICE_LOW_BITS_DELTA)
        }
        _ => cap_rice_low_bits(cap),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golomb_rice::{
        golomb_rice_decode_vec, golomb_rice_encode_vec, golomb_rice_zigzag_width, zigzag_encode,
        GOLOMB_RICE_Q_MAX,
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
                cap_rice_low_bits(cap).saturating_sub(WIRE_RICE_LOW_BITS_DELTA),
                "cap={cap}"
            );
        }
    }

    #[test]
    fn wire_rice_low_bits_encodes_full_fold_cap_range_without_escape_on_shipping_caps() {
        for cap in [504u128, 1008, 1568, 2016] {
            let rice_low_bits = wire_rice_low_bits(cap);
            let zigzag_w = golomb_rice_zigzag_width(cap);
            assert!(
                max_quotient_in_cap_range(cap, rice_low_bits, zigzag_w)
                    < u64::from(GOLOMB_RICE_Q_MAX),
                "cap={cap} rice_low_bits={rice_low_bits} needs escape at some legal coefficient"
            );
            let cap_i64 = cap as i64;
            for n in [-cap_i64, -1, 0, 1, cap_i64] {
                let encoded =
                    golomb_rice_encode_vec(&[n], rice_low_bits, zigzag_w).expect("encode");
                let decoded =
                    golomb_rice_decode_vec(&encoded, 1, rice_low_bits, zigzag_w).expect("decode");
                assert_eq!(decoded, [n], "cap={cap} n={n}");
            }
        }
    }
}
