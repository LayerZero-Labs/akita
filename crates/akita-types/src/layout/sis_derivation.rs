//! Pure layout helpers shared by config, scheme, and planner code.
//!
//! The verifier-reachable layout builders (`level_layout_from_params`,
//! `recursive_level_layout_from_params`) live here. They compose the SIS/Ajtai
//! leaf primitives in [`crate::sis`] (digit counts, collision norms, secure
//! ranks) — they contain no SIS formula of their own.

use crate::layout::digit_math::optimal_m_r_split;
use crate::sis::{decomp_depths, AjtaiKeyParams, FoldChallengeNorms, FoldWitnessNorms};
use crate::{DecompositionParams, LevelParams};
use akita_field::AkitaError;

/// Apply layout coordinates and decomposition depths to a parameter-only level.
///
/// # Errors
///
/// Returns an error when the resulting layout is internally inconsistent.
pub fn level_layout_from_params(
    m_vars: usize,
    r_vars: usize,
    lp: &LevelParams,
    decomp: DecompositionParams,
    num_ring: usize,
) -> Result<LevelParams, AkitaError> {
    let (depth_commit, depth_open) = decomp_depths(decomp);
    lp.with_decomp(m_vars, r_vars, depth_commit, depth_open, num_ring)
}

/// Derive a recursive `w`-opening layout from the active level params.
///
/// # Errors
///
/// Returns an error if the witness length is incompatible with `params.d` or if
/// the recursive layout derivation overflows.
pub fn recursive_level_layout_from_params(
    lp: &LevelParams,
    current_w_len: usize,
    root_decomp: DecompositionParams,
) -> Result<LevelParams, AkitaError> {
    if !current_w_len.is_multiple_of(lp.ring_dimension) {
        return Err(AkitaError::InvalidInput(format!(
            "witness length {current_w_len} is not divisible by D={}",
            lp.ring_dimension
        )));
    }
    let num_ring_elems = current_w_len / lp.ring_dimension;
    let total = num_ring_elems.next_power_of_two().max(1);
    let alpha = lp.ring_dimension.trailing_zeros() as usize;
    let reduced_vars = total.trailing_zeros() as usize;
    let max_num_vars = reduced_vars + alpha;
    // Recursive level: balanced-digit `w` entries collapse `log_commit_bound`
    // to `log_basis`; opening folds inherit the parent's open bound.
    let decomp = DecompositionParams {
        log_basis: lp.log_basis,
        log_commit_bound: lp.log_basis,
        log_open_bound: Some(
            root_decomp
                .log_open_bound
                .unwrap_or(root_decomp.log_commit_bound),
        ),
    };
    // `optimal_m_r_split` derives `n_a` per `r` from the SIS-floor table
    // for `(sis_family, ring_dimension, a_collision)`. All three live on
    // `lp.a_key` once `lp` has gone through `sis_secure_level_params`
    // (or any planner / materializer code path that constructs a
    // SIS-typed `LevelParams`). Seed-only `LevelParams::params_only`
    // forms with `collision_inf = 0` would land on the symmetric-split
    // fallback inside `optimal_m_r_split` — those seeds should be
    // populated with the audited bucket before calling here.
    // Recursive levels always commit a dense balanced-digit witness
    // (`||s||_inf = b/2`, `nonzeros = D`), never one-hot, so the fold-witness
    // sparsity is dense regardless of the root config.
    let fold_challenge = FoldChallengeNorms {
        infinity_norm: lp.challenge_infinity_norm() as u128,
        l1_norm: lp.challenge_l1_mass() as u128,
    };
    let fold_witness = FoldWitnessNorms::new(decomp.log_basis, lp.ring_dimension, 1, false);
    let (m_vars, r_vars, n_a) = optimal_m_r_split(
        lp.a_key.sis_family(),
        lp.ring_dimension as u32,
        lp.a_key.collision_inf(),
        fold_challenge,
        fold_witness,
        decomp.log_commit_bound,
        decomp.log_basis,
        reduced_vars,
        num_ring_elems,
        decomp.field_bits(),
    );
    // Sync the seed's A-row rank with the per-`r` SIS-secure rank
    // chosen by `optimal_m_r_split` so `with_decomp`'s derived
    // `b_key.col_len = n_a · δ_open · num_blocks` agrees with the
    // cost the optimizer scored. No rank floor: the SIS-floor lookup
    // inside `optimal_m_r_split` gives the tight secure minimum, and
    // every caller now constructs layouts off that minimum (no
    // envelope-driven bumps remain).
    let mut layout_seed = lp.clone();
    layout_seed.a_key = AjtaiKeyParams::new_unchecked(
        lp.a_key.sis_family(),
        n_a as usize,
        lp.a_key.col_len(),
        lp.a_key.collision_inf(),
        lp.ring_dimension,
    );
    let layout = level_layout_from_params(m_vars, r_vars, &layout_seed, decomp, num_ring_elems)?;
    debug_assert_eq!(layout.m_vars + layout.r_vars + alpha, max_num_vars);
    Ok(layout)
}
