//! Layout/search helpers that compose the SIS/Ajtai primitives in
//! [`crate::sis`].
//!
//! The SIS/Ajtai *leaf* primitives (digit counts, collision norms, secure-rank
//! lookup, per-role widths) all live in [`crate::sis`]. This module holds only
//! the gadget row scalars and the `(m, r)`-split search, which *compose* those
//! primitives but contain no SIS formula of their own.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::{CanonicalField, FieldCore};

use crate::sis::{
    choose_op_norm_rejection_for_a_role, fold_level_witness_scoring_cost, num_digits_for_bound,
    FoldChallengeNorms, FoldWitnessNorms, SisModulusFamily,
};
use crate::DecompositionParams;

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
/// `inner_width(r) = block_len(r) · δ_commit` (via [`crate::sis::min_secure_rank`]). The
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
    stage1_config: &SparseChallengeConfig,
    fold_challenge_shape: TensorChallengeShape,
    log_commit_bound: u32,
    log_basis: u32,
    onehot_chunk_size: usize,
    reduced_vars: usize,
    num_ring: usize,
    field_bits: u32,
    fold_level: usize,
    grind_schedule: crate::sis::GrindTargetAcceptSchedule,
) -> (usize, usize, u32) {
    // Too few variables to optimize; too many would overflow `2^r` in u64.
    if reduced_vars <= 2 || reduced_vars >= 53 {
        let r = reduced_vars / 2;
        return (reduced_vars - r, r, 1);
    }

    let delta_commit = num_digits_for_bound(log_commit_bound, field_bits, log_basis) as u64;

    let mut best: Option<(u64, usize, u32)> = None;

    for r in (1..reduced_vars).rev() {
        let block_len: u64 = if num_ring > 0 {
            num_ring.div_ceil(1usize << r) as u64
        } else {
            1u64 << (reduced_vars - r)
        };

        let Some(inner_width) = (block_len as usize).checked_mul(delta_commit as usize) else {
            continue;
        };
        let Some((op_norm_rejection, _a_collision, n_a)) = choose_op_norm_rejection_for_a_role(
            sis_family,
            d as usize,
            DecompositionParams {
                log_basis,
                log_commit_bound,
                log_open_bound: None,
            },
            stage1_config,
            fold_challenge_shape,
            log_commit_bound == 1,
            onehot_chunk_size,
            ring_subfield_norm_bound,
            r,
            num_claims,
            inner_width as u64,
            fold_level,
            grind_schedule,
        ) else {
            continue;
        };
        let n_a_u32 = n_a as u32;

        let Some(total) = fold_level_witness_scoring_cost(
            n_a,
            op_norm_rejection,
            r,
            num_claims,
            inner_width,
            DecompositionParams {
                log_basis,
                log_commit_bound,
                log_open_bound: None,
            },
            stage1_config,
            fold_challenge_shape,
            d as usize,
            fold_challenge,
            fold_witness,
            fold_level,
            grind_schedule,
        ) else {
            continue;
        };

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
    use crate::sis::{num_digits_fold, FoldWitnessLinfCapConfig, FoldWitnessNorms};
    use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};

    #[test]
    fn optimal_m_r_split_uses_num_claims_in_fold_digit_scoring() {
        use akita_challenges::{
            D64_PRODUCTION_EXACT_SHELL_MAG1, D64_PRODUCTION_EXACT_SHELL_MAG2,
            D64_PRODUCTION_OPERATOR_NORM_THRESHOLD,
        };
        let stage1_config = SparseChallengeConfig::ExactShell {
            count_mag1: D64_PRODUCTION_EXACT_SHELL_MAG1,
            count_mag2: D64_PRODUCTION_EXACT_SHELL_MAG2,
            operator_norm_threshold: D64_PRODUCTION_OPERATOR_NORM_THRESHOLD,
        };
        let fold_challenge =
            crate::sis::fold_challenge_norms(&stage1_config, TensorChallengeShape::Flat);
        let fold_witness = FoldWitnessNorms::new(3, 64, 64, true);
        let cap_config = FoldWitnessLinfCapConfig::for_fold_level_scoring(
            crate::sis::fold_witness_linf_cap_policy(
                &stage1_config,
                TensorChallengeShape::Flat,
                64,
            ),
            &stage1_config,
            TensorChallengeShape::Flat,
            64,
            false,
            64,
            crate::FoldLinfProtocolBinding::CURRENT
                .grind_target_accept_prob()
                .0,
            crate::FoldLinfProtocolBinding::CURRENT
                .grind_target_accept_prob()
                .1,
        )
        .unwrap();
        let singleton_fold_digits =
            num_digits_fold(5, 1, 128, 3, fold_challenge, fold_witness, cap_config)
                .expect("singleton fold digits");
        let batched_fold_digits =
            num_digits_fold(5, 4, 128, 3, fold_challenge, fold_witness, cap_config)
                .expect("batched fold digits");
        assert_ne!(
            singleton_fold_digits, batched_fold_digits,
            "fold digit depth must grow with batched num_claims"
        );
    }
}
