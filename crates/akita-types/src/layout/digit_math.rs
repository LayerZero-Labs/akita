//! Layout/search helpers that compose the SIS/Ajtai primitives in
//! [`crate::sis`].
//!
//! The SIS/Ajtai *leaf* primitives (digit counts, collision norms, secure-rank
//! lookup, per-role widths) all live in [`crate::sis`]. This module holds only
//! the gadget row scalars and the `(r_pos, r_blk)`-split search, which *compose* those
//! primitives but contain no SIS formula of their own.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::{CanonicalField, FieldCore};

use crate::sis::{
    fold_witness_digit_plan, fold_witness_linf_cap_policy, min_secure_rank, num_digits_for_bound,
    rounded_up_role_a_inf_norm, FoldChallengeNorms, FoldWitnessLinfCapConfig, FoldWitnessNorms,
    SisModulusProfileId, SisSecurityPolicyId, SisTableKey,
};
use crate::DecompositionParams;

/// Smallest integer `s` with `s^2 >= v`.
#[inline]
#[must_use]
pub fn isqrt_ceil(v: u128) -> u128 {
    let s = v.isqrt();
    s + u128::from(s.saturating_mul(s) < v)
}

/// Return the row gadget scalars `1, b, b^2, ...` for `b = 2^log_basis`.
pub fn gadget_row_scalars<F: FieldCore + CanonicalField>(levels: usize, log_basis: u32) -> Vec<F> {
    let base = F::from_canonical_u128_reduced(1u128 << log_basis);
    let mut out = Vec::with_capacity(levels);
    let mut power = F::one();
    for i in 0..levels {
        if i > 0 {
            power *= base;
        }
        out.push(power);
    }
    out
}

/// Find the `(r_pos, r_blk)` split of `reduced_vars` that minimizes next-level witness
/// size.
///
/// # Background (Akita paper, Section 4.5)
///
/// After removing the ring dimension (`α = log2(D)` variables), the remaining
/// `reduced_vars = ℓ - α` variables are partitioned as
/// `r_pos + r_blk = reduced_vars`.
/// The witness is a matrix: `B` exact live block-columns and `M` rows. The witness
/// size (ring elements, dropping the split-independent quotient term) is:
///
/// ```text
///   witness_size = (1 + n_A) · δ_open · B  +  δ_commit · δ_fold · M
///              ─────────────────────────     ────────────────────────
///              |t̂| + |ŵ|  (opening)         |ẑ|  (folded witness)
/// ```
///
/// `n_A` is the per-candidate minimum SIS-secure A-rank for
/// `inner_width = M · δ_commit` (via [`crate::sis::min_secure_rank`]). The
/// A collision is itself recomputed for every candidate via
/// [`crate::sis::rounded_up_role_a_inf_norm`], because the committed-level
/// weak-binding norm grows with the exact fold arity `num_claims · B`; scoring
/// every split against a single bucket would rank candidates with different `B` wrong.
///
/// As `r_blk` grows, the opening term grows with exact `B` while the folding term
/// shrinks with `M`; `δ_fold` and the A collision (hence `n_A`) also grow with `B`.
/// There is no closed form, so all valid splits are brute-forced.
///
/// # Tight z mode
///
/// - `num_ring > 0`: `M = 2^r_pos` and `B = ⌈num_ring / M⌉` uses the exact live
///   block count.
/// - `num_ring = 0`: `M = 2^r_pos` and `B = 2^r_blk` use the root capacity.
///
/// # Return value
///
/// `(position_index_bits, block_index_bits, n_a)` — the chosen split plus its
/// per-candidate SIS-secure A-rank. Callers building a `LevelParams` should use
/// this `n_a` so the
/// derived `b_key.col_len` matches the cost the optimizer scored.
///
/// # Fallback
///
/// If `reduced_vars` is too small/large to brute-force, or every candidate falls off
/// the SIS-floor table, returns the paper's symmetric bit split with
/// `n_a = 1` (a cost-estimate fallback; downstream re-derives the SIS-strict
/// layout for selected candidates).
///
/// # Panics
///
/// Panics if `log_basis` is 0 or at least 128.
#[allow(clippy::too_many_arguments)]
pub fn optimal_block_geometry_split(
    policy: SisSecurityPolicyId,
    sis_modulus_profile: SisModulusProfileId,
    d: u32,
    num_claims: usize,
    ring_subfield_norm_bound: u32,
    fold_challenge: FoldChallengeNorms,
    fold_witness: FoldWitnessNorms,
    fold_challenge_config: &SparseChallengeConfig,
    fold_challenge_shape: TensorChallengeShape,
    decomposition: DecompositionParams,
    onehot_chunk_size: usize,
    reduced_vars: usize,
    num_ring: usize,
) -> (usize, usize, u32) {
    // Too few variables to optimize; too many would overflow `2^r` in u64.
    if reduced_vars <= 2 || reduced_vars >= 53 {
        let r = reduced_vars / 2;
        return (reduced_vars - r, r, 1);
    }

    let field_bits = decomposition.field_bits();
    let log_commit_bound = decomposition.log_commit_bound;
    let delta_commit =
        num_digits_for_bound(log_commit_bound, field_bits, decomposition.log_basis) as u64;

    let mut best: Option<(u64, usize, u32)> = None;

    for r in (1..reduced_vars).rev() {
        let num_positions_per_block = 1u64 << (reduced_vars - r);
        let num_live_blocks = if num_ring > 0 {
            num_ring.div_ceil(num_positions_per_block as usize)
        } else {
            1usize << r
        };

        let Some(inner_width) =
            (num_positions_per_block as usize).checked_mul(delta_commit as usize)
        else {
            continue;
        };
        let Some(a_collision) = rounded_up_role_a_inf_norm(
            policy,
            sis_modulus_profile,
            d as usize,
            decomposition,
            decomposition.log_basis,
            fold_challenge_config,
            fold_challenge_shape,
            log_commit_bound == 1,
            onehot_chunk_size,
            ring_subfield_norm_bound,
            num_live_blocks,
            num_claims,
            inner_width as u64,
        ) else {
            continue;
        };
        let Some(n_a) = min_secure_rank(
            SisTableKey {
                policy,
                table_digest: crate::sis::SisTableDigest::CURRENT,
                modulus_profile: sis_modulus_profile,
                role: crate::sis::SisMatrixRole::A,
                ring_dimension: d,
                coeff_linf_bound: a_collision,
            },
            inner_width as u64,
        ) else {
            continue;
        };
        let n_a_u32 = n_a as u32;

        let Some(total) = fold_level_witness_scoring_cost(
            n_a,
            num_live_blocks,
            num_claims,
            inner_width,
            decomposition,
            fold_challenge_config,
            fold_challenge_shape,
            d as usize,
            fold_challenge,
            fold_witness,
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

/// Next-level witness scoring cost for one block geometry, matching
/// [`optimal_block_geometry_split`]:
///
/// ```text
///   (1 + n_a) · δ_open · B  +  δ_commit · δ_fold · M
/// ```
#[allow(clippy::too_many_arguments)]
pub fn fold_level_witness_scoring_cost(
    n_a: usize,
    num_live_blocks: usize,
    num_claims: usize,
    inner_width: usize,
    decomposition: DecompositionParams,
    fold_challenge_config: &SparseChallengeConfig,
    fold_shape: TensorChallengeShape,
    ring_dimension: usize,
    fold_challenge: FoldChallengeNorms,
    fold_witness: FoldWitnessNorms,
) -> Option<u64> {
    let field_bits = decomposition.field_bits();
    let log_basis = decomposition.log_basis;
    let log_commit_bound = decomposition.log_commit_bound;
    let open_bound = log_commit_bound.max(field_bits);
    let delta_open = num_digits_for_bound(open_bound, field_bits, log_basis) as u64;
    let delta_commit = num_digits_for_bound(log_commit_bound, field_bits, log_basis) as u64;
    let num_positions_per_block = inner_width.checked_div(delta_commit as usize)?;
    if num_positions_per_block == 0 {
        return None;
    }
    let num_live_blocks_u64 = u64::try_from(num_live_blocks).ok()?;
    let num_positions_per_block_u64 = num_positions_per_block as u64;
    let cap_policy =
        fold_witness_linf_cap_policy(fold_challenge_config, fold_shape, ring_dimension);
    let binding = crate::FoldLinfProtocolBinding::CURRENT;
    let (grind_target_accept_num, grind_target_accept_den) = binding.grind_target_accept_prob();
    let cap_config = FoldWitnessLinfCapConfig::for_fold_level_scoring(
        cap_policy,
        fold_challenge_config,
        fold_shape,
        ring_dimension,
        inner_width,
        grind_target_accept_num,
        grind_target_accept_den,
    )
    .ok()?;
    let (decomposed_fold_digits, _) = fold_witness_digit_plan(
        num_live_blocks,
        num_claims,
        field_bits,
        log_basis,
        fold_challenge,
        fold_witness,
        &cap_config,
    )
    .ok()?;
    let per_block_cost = delta_open.saturating_add((n_a as u64).saturating_mul(delta_open));
    let opening_cost = per_block_cost.saturating_mul(num_live_blocks_u64);
    let folding_cost = delta_commit
        .saturating_mul(decomposed_fold_digits as u64)
        .saturating_mul(num_positions_per_block_u64);
    Some(opening_cost.saturating_add(folding_cost))
}

#[cfg(test)]
mod tests {
    use crate::sis::{fold_witness_digit_plan, FoldWitnessLinfCapConfig, FoldWitnessNorms};
    use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};

    #[test]
    fn optimal_block_geometry_split_uses_num_claims_in_fold_digit_scoring() {
        use akita_challenges::{D64_PRODUCTION_PM1_COUNT, D64_PRODUCTION_PM2_COUNT};
        let fold_challenge_config = SparseChallengeConfig {
            count_pm1: D64_PRODUCTION_PM1_COUNT,
            count_pm2: D64_PRODUCTION_PM2_COUNT,
        };
        let fold_challenge =
            crate::sis::FoldChallengeNorms::new(&fold_challenge_config, TensorChallengeShape::Flat);
        let fold_witness = FoldWitnessNorms::new(3, 64, 64, true);
        let cap_config = FoldWitnessLinfCapConfig::for_fold_level_scoring(
            crate::sis::fold_witness_linf_cap_policy(
                &fold_challenge_config,
                TensorChallengeShape::Flat,
                64,
            ),
            &fold_challenge_config,
            TensorChallengeShape::Flat,
            64,
            64,
            crate::FoldLinfProtocolBinding::CURRENT
                .grind_target_accept_prob()
                .0,
            crate::FoldLinfProtocolBinding::CURRENT
                .grind_target_accept_prob()
                .1,
        )
        .unwrap();
        let (singleton_fold_digits, _) =
            fold_witness_digit_plan(5, 1, 128, 3, fold_challenge, fold_witness, &cap_config)
                .expect("singleton fold digits");
        let (batched_fold_digits, _) =
            fold_witness_digit_plan(5, 4, 128, 3, fold_challenge, fold_witness, &cap_config)
                .expect("batched fold digits");
        assert!(
            batched_fold_digits >= singleton_fold_digits,
            "snapped fold digit depth must not shrink with batched num_claims"
        );
    }
}
