//! Layout/search helpers that compose the SIS/Ajtai primitives in
//! [`crate::sis`].
//!
//! The SIS/Ajtai *leaf* primitives (digit counts, collision norms, secure-rank
//! lookup, per-role widths) all live in [`crate::sis`]. This module holds only
//! the gadget row scalars and the `(m, r)`-split search, which *compose* those
//! primitives but contain no SIS formula of their own.

use akita_field::{CanonicalField, FieldCore};

use crate::sis::{
    committed_fold_collision_l2_sq, min_secure_rank, num_digits_fold, num_digits_for_bound,
    FoldChallengeNorms, FoldWitnessNorms, SisModulusFamily,
};

/// Return the row gadget scalars `1, b, b^2, ...` for `b = 2^log_basis`.
pub fn gadget_row_scalars<F: FieldCore + CanonicalField>(levels: usize, log_basis: u32) -> Vec<F> {
    let base = F::from_canonical_u128_reduced(1u128 << log_basis);
    let mut out = Vec::with_capacity(levels);
    let mut power = F::one();
    for _ in 0..levels {
        out.push(power);
        power *= base;
    }
    out
}

/// Find the `(m, r)` split of `reduced_vars` that minimizes next-level witness
/// size.
///
/// # Background (Akita paper, Section 4.5)
///
/// After removing the ring dimension (`α = log2(D)` variables), the remaining
/// `reduced_vars = ℓ - α` variables are partitioned as `m + r = reduced_vars`.
/// The witness is a matrix: `2^r` block-columns and `m_eff` rows. The witness
/// size (ring elements, dropping the split-independent quotient term) is:
///
/// ```text
///   witness_size = (1 + n_A) · δ_open · 2^r  +  δ_commit · δ_fold · m_eff
///              ─────────────────────────     ────────────────────────
///              |t̂| + |ŵ|  (opening)         |ẑ|  (folded witness)
/// ```
///
/// `n_A` is the per-`r` minimum SIS-secure A-rank for the candidate's
/// `inner_width(r) = block_len(r) · δ_commit` (via [`min_secure_rank`]). The
/// A collision is itself recomputed per `r` via
/// [`crate::sis::committed_fold_collision_l2_sq`], because the committed-level
/// weak-binding norm grows with the fold arity `num_claims · 2^r`; scoring
/// every split against a single bucket would rank the larger-`r` splits wrong.
///
/// As `r` grows, the opening term grows `~2^r` while the folding term shrinks
/// with `m_eff`; `δ_fold` and the A collision (hence `n_A`) also grow with `r`.
/// There is no closed form, so all valid splits are brute-forced.
///
/// # Tight z mode
///
/// - `num_ring > 0`: `m_eff = ⌈num_ring / 2^r⌉` (actual occupied rows).
/// - `num_ring = 0`: `m_eff = 2^m` (power-of-two upper bound).
///
/// # Return value
///
/// `(m_vars, r_vars, n_a)` — the chosen split plus its per-`r` SIS-secure
/// A-rank. Callers building a `LevelParams` should use this `n_a` so the
/// derived `b_key.col_len` matches the cost the optimizer scored.
///
/// # Fallback
///
/// If `reduced_vars` is too small/large to brute-force, or every `r` falls off
/// the SIS-floor table, returns the paper's symmetric split `m = r` with
/// `n_a = 1` (a cost-estimate fallback; downstream re-derives the SIS-strict
/// layout for selected candidates).
///
/// # Panics
///
/// Panics if `log_basis` is 0 or at least 128.
#[allow(clippy::too_many_arguments)]
pub fn optimal_m_r_split(
    sis_family: SisModulusFamily,
    d: u32,
    num_claims: usize,
    ring_subfield_norm_bound: u32,
    fold_challenge: FoldChallengeNorms,
    fold_witness: FoldWitnessNorms,
    log_commit_bound: u32,
    log_basis: u32,
    reduced_vars: usize,
    num_ring: usize,
    field_bits: u32,
) -> (usize, usize, u32) {
    // Too few variables to optimize; too many would overflow `2^r` in u64.
    if reduced_vars <= 2 || reduced_vars >= 53 {
        let r = reduced_vars / 2;
        return (reduced_vars - r, r, 1);
    }

    let open_bound = log_commit_bound.max(field_bits);
    let delta_open = num_digits_for_bound(open_bound, field_bits, log_basis) as u64;
    let delta_commit = num_digits_for_bound(log_commit_bound, field_bits, log_basis) as u64;

    let mut best: Option<(u64, usize, u32)> = None;

    for r in (1..reduced_vars).rev() {
        let num_blocks = 1u64 << r;
        let block_len: u64 = if num_ring > 0 {
            num_ring.div_ceil(1usize << r) as u64
        } else {
            1u64 << (reduced_vars - r)
        };
        let m_eff = block_len;

        let Some(inner_width) = (block_len as usize).checked_mul(delta_commit as usize) else {
            continue;
        };
        // The committed-level A collision is fold-priced, so it grows with `r`
        // (and `num_claims`); recompute its bucket per split rather than reusing
        // one fixed bucket.
        let Some(a_collision) = committed_fold_collision_l2_sq(
            sis_family,
            d,
            fold_challenge,
            fold_witness,
            r,
            num_claims,
            ring_subfield_norm_bound,
        ) else {
            continue;
        };
        let Some(n_a) = min_secure_rank(sis_family, d, a_collision, inner_width as u64) else {
            continue;
        };
        let n_a_u32 = n_a as u32;

        // δ_fold grows with r and num_claims: num_digits_fold derives
        // β = num_claims · 2^r · min(||c||_inf·||s||_1, ||c||_1·||s||_inf).
        // An overflowing/degenerate β makes this `r` infeasible — skip it.
        let Ok(delta_fold) = num_digits_fold(
            r,
            num_claims,
            field_bits,
            log_basis,
            fold_challenge,
            fold_witness,
        ) else {
            continue;
        };
        let delta_fold = delta_fold as u64;

        let per_block_cost = delta_open.saturating_add((n_a as u64).saturating_mul(delta_open));
        let opening_cost = per_block_cost.saturating_mul(num_blocks);
        let folding_cost = delta_commit
            .saturating_mul(delta_fold)
            .saturating_mul(m_eff);
        let total = opening_cost.saturating_add(folding_cost);

        if best.is_none_or(|(c, _, _)| total < c) {
            best = Some((total, r, n_a_u32));
        }
    }

    match best {
        Some((_, r, n_a)) => (reduced_vars - r, r, n_a),
        None => {
            let r = reduced_vars / 2;
            (reduced_vars - r, r, 1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sis::FoldWitnessNorms;

    #[test]
    fn optimal_m_r_split_uses_num_claims_in_fold_digit_scoring() {
        let fold_challenge = FoldChallengeNorms {
            infinity_norm: 8,
            l1_norm: 54,
        };
        let fold_witness = FoldWitnessNorms::new(3, 64, 64, true);
        let singleton = optimal_m_r_split(
            SisModulusFamily::Q32,
            64,
            1,
            1,
            fold_challenge,
            fold_witness,
            128,
            3,
            20,
            0,
            32,
        );
        let batched = optimal_m_r_split(
            SisModulusFamily::Q32,
            64,
            4,
            1,
            fold_challenge,
            fold_witness,
            128,
            3,
            20,
            0,
            32,
        );
        assert_ne!(
            singleton, batched,
            "batched roots must not score fold digits as singleton"
        );
    }
}
