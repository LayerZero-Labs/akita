//! Cap → live Rice `k` for terminal Golomb `z` (code-constant rule; not descriptor-bound).

use crate::golomb_rice::optimal_rice_k;

/// Audit id when the cap→live-`k` map changes (logs/tests only; not transcript-bound).
pub const TAIL_GOLOMB_CAP_TO_K_RULE_ID: u8 = 1;

/// Profile-calibrated tightening vs [`optimal_rice_k`]: honest tails sit ~2 Rice bits below
/// the security `floor(log2(cap))` remainder width on CI profile cells.
pub const TAIL_GOLOMB_LIVE_K_DELTA: u32 = 2;

/// Security / legacy Rice parameter: `floor(log2(cap))` remainder width for worst-case `|n| ≤ cap`.
#[must_use]
pub fn security_rice_k_for_fold_cap(cap: u128) -> u32 {
    optimal_rice_k(cap)
}

/// Live Rice `k` used by prover and verifier for terminal `z` Golomb encode/decode.
///
/// This is **not** [`crate::golomb_rice::empirical_optimal_rice_k`]: that minimizes total
/// bits on a realized witness sample and is witness-dependent. It is also **not** a
/// `min_sound_k` search: the codec round-trips every `n ∈ [-cap, cap]` even at `k = 0`
/// via the escape path, so `min_sound_k` is always `0` and carries no pricing signal.
///
/// Rule v1: `optimal_rice_k(cap) - δ` with δ = [`TAIL_GOLOMB_LIVE_K_DELTA`]. Typical
/// coefficients pay `k_live` fixed remainder bits; coefficients near `±cap` pay longer
/// unary runs but stay below [`crate::golomb_rice::GOLOMB_RICE_Q_MAX`] at shipping caps.
#[must_use]
pub fn live_rice_k_for_fold_cap(cap: u128) -> u32 {
    match TAIL_GOLOMB_CAP_TO_K_RULE_ID {
        1 => security_rice_k_for_fold_cap(cap).saturating_sub(TAIL_GOLOMB_LIVE_K_DELTA),
        _ => security_rice_k_for_fold_cap(cap),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golomb_rice::{
        golomb_rice_decode_vec, golomb_rice_encode_vec, golomb_rice_zigzag_width, GOLOMB_RICE_Q_MAX,
    };

    fn max_quotient_in_cap_range(cap: u128, k: u32, w: u32) -> u64 {
        let cap_i64 = cap as i64;
        (-cap_i64..=cap_i64)
            .map(|n| {
                let u = crate::golomb_rice::zigzag_encode(n, w).unwrap();
                if k == 0 {
                    u
                } else {
                    u >> k
                }
            })
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn live_k_is_security_k_minus_delta_on_shipping_caps() {
        for cap in [504u128, 1008, 1568, 2016] {
            assert_eq!(
                live_rice_k_for_fold_cap(cap),
                security_rice_k_for_fold_cap(cap).saturating_sub(TAIL_GOLOMB_LIVE_K_DELTA),
                "cap={cap}"
            );
        }
    }

    #[test]
    fn live_k_encodes_full_fold_cap_range_without_escape_on_shipping_caps() {
        for cap in [504u128, 1008, 1568, 2016] {
            let k = live_rice_k_for_fold_cap(cap);
            let w = golomb_rice_zigzag_width(cap);
            assert!(
                max_quotient_in_cap_range(cap, k, w) < u64::from(GOLOMB_RICE_Q_MAX),
                "cap={cap} k={k} needs escape at some legal coefficient"
            );
            let cap_i64 = cap as i64;
            for n in [-cap_i64, -1, 0, 1, cap_i64] {
                let encoded = golomb_rice_encode_vec(&[n], k, w).expect("encode");
                let decoded = golomb_rice_decode_vec(&encoded, 1, k, w).expect("decode");
                assert_eq!(decoded, [n], "cap={cap} n={n}");
            }
        }
    }
}
