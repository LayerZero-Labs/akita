//! Pure layout helpers shared by config, scheme, and planner code.
//!
//! The verifier-reachable layout helpers
//! (`level_layout_from_params`, `recursive_level_layout_from_params`,
//! `decomp_depths`) live here. Search and SIS-derivation loops moved
//! to `akita_derive::derivation`. The fast-verify tier helpers
//! (`tiered_b_prime_rank`, `tiered_f_rank`, `apply_dynamic_tier`, etc.)
//! stay here because they are verifier-reachable through the table
//! materializer's tier post-processing.

use crate::generated::sis_floor::{ceil_supported_collision, min_rank_for_secure_width};
use crate::layout::digit_math::{
    compute_num_digits_fold_with_claims, num_digits_for_bound, optimal_m_r_split,
};
use crate::{AjtaiKeyParams, DecompositionParams, LevelParams, SisModulusFamily};
use akita_field::AkitaError;

/// Inclusive upper bound on `outer_log_basis` for the tiered root
/// commitment. Today's balanced i8 `FlatDigitBlocks` storage only
/// supports `log_basis ≤ 6`; the tiered outer digit basis must obey the
/// same bound until a non-i8 digit type is introduced.
pub const MAX_TIERED_OUTER_LOG_BASIS: u32 = 6;
/// Inclusive lower bound on `outer_log_basis`. Bases below 2 would not
/// actually compress and are rejected by the planner.
pub const MIN_TIERED_OUTER_LOG_BASIS: u32 = 2;

/// Collision-`inf` bound used by the SIS floor table for a balanced
/// i8 gadget decomposition with basis `2^log_basis`.
///
/// Two valid digit vectors `x, x'` with `‖x‖_∞, ‖x'‖_∞ < 2^log_basis`
/// have `‖x − x'‖_∞ ≤ 2^log_basis − 1`. Existing code already adopts
/// this convention (see `sis_derived_recursive_params_for_layout`'s
/// `bd_collision = (1u32 << log_basis) - 1`); the tiered helpers below
/// follow it so all SIS sizing speaks one bound convention.
#[inline]
pub fn balanced_digit_delta_bound(log_basis: u32) -> u32 {
    (1u32 << log_basis).saturating_sub(1)
}

/// Validate that `outer_log_basis` falls in the supported range. Returns
/// `Err` if it does not.
///
/// # Errors
///
/// Returns an error when `outer_log_basis` is outside
/// `[MIN_TIERED_OUTER_LOG_BASIS, MAX_TIERED_OUTER_LOG_BASIS]`.
pub fn validate_tiered_outer_log_basis(outer_log_basis: u32) -> Result<(), AkitaError> {
    if !(MIN_TIERED_OUTER_LOG_BASIS..=MAX_TIERED_OUTER_LOG_BASIS).contains(&outer_log_basis) {
        return Err(AkitaError::InvalidSetup(format!(
            "tiered outer_log_basis = {outer_log_basis} is outside the supported \
             range [{MIN_TIERED_OUTER_LOG_BASIS}, {MAX_TIERED_OUTER_LOG_BASIS}] \
             (current balanced i8 storage limits log_basis to 6)"
        )));
    }
    Ok(())
}

/// Compute the SIS rank for the tier-1 `B'` matrix.
///
/// Implements the case-2 branch of the two-case binding proof in
/// `specs/tiered_commit.md` §6: a `B'` collision is exhibited by a
/// non-zero `Δt_i` with `‖Δt_i‖_∞ ≤ 2 · t_inf_bound`. The planner sizes
/// `n_b'` so that no such collision exists within the chunk width
/// `outer_width / split_factor`.
///
/// `t_inf_bound` is the existing per-cell `t̂` infinity-norm bound (the
/// inner gadget bound used today for the legacy `B` key).
///
/// # Errors
///
/// Returns `Err` when the generated SIS floor table has no entry for the
/// requested `(family, d, collision, width)` tuple.
pub fn tiered_b_prime_rank(
    family: SisModulusFamily,
    d: u32,
    t_inf_bound: u32,
    outer_width: usize,
    split_factor: usize,
) -> Result<u32, AkitaError> {
    if split_factor < 1 {
        return Err(AkitaError::InvalidSetup(
            "tiered_b_prime_rank: split_factor must be ≥ 1".to_string(),
        ));
    }
    if !outer_width.is_multiple_of(split_factor) {
        return Err(AkitaError::InvalidSetup(format!(
            "tiered_b_prime_rank: outer_width = {outer_width} not divisible by \
             split_factor = {split_factor}"
        )));
    }
    let chunk_width = outer_width / split_factor;
    let collision_raw = t_inf_bound
        .checked_mul(2)
        .ok_or_else(|| AkitaError::InvalidSetup("Δt collision bound overflow".to_string()))?;
    let collision = ceil_supported_collision(family, d, collision_raw).ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "tiered_b_prime_rank: no supported SIS collision bucket covers Δt bound \
             {collision_raw} for family={family:?}, D={d}"
        ))
    })?;
    let rank =
        min_rank_for_secure_width(family, d, collision, chunk_width as u64).ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "tiered_b_prime_rank: no generated SIS row covers \
                 family={family:?}, D={d}, collision={collision}, width={chunk_width}"
            ))
        })?;
    u32::try_from(rank).map_err(|_| {
        AkitaError::InvalidSetup("tiered_b_prime_rank: SIS rank exceeds u32".to_string())
    })
}

/// Compute the SIS rank for the tier-1 `F` matrix.
///
/// Implements the case-1 branch of the two-case binding proof in
/// `specs/tiered_commit.md` §6: an `F` collision is exhibited by a
/// non-zero `Δû_concat` with `‖Δû_concat‖_∞ ≤ 2 · floor(2^outer_log_basis / 2)`.
/// We use `2^outer_log_basis − 1` here for consistency with the existing
/// balanced-digit collision convention (see `balanced_digit_delta_bound`).
///
/// `F` has width `n_b' · split_factor · num_digits_outer`.
///
/// # Errors
///
/// Returns `Err` when `outer_log_basis` is outside the supported range or
/// when the generated SIS floor table has no entry for the requested
/// `(family, d, collision, width)` tuple.
pub fn tiered_f_rank(
    family: SisModulusFamily,
    d: u32,
    outer_log_basis: u32,
    n_b_prime: u32,
    split_factor: usize,
    num_digits_outer: usize,
) -> Result<u32, AkitaError> {
    validate_tiered_outer_log_basis(outer_log_basis)?;
    if split_factor < 2 {
        return Err(AkitaError::InvalidSetup(
            "tiered_f_rank: split_factor must be ≥ 2 for the tiered path".to_string(),
        ));
    }
    if num_digits_outer == 0 {
        return Err(AkitaError::InvalidSetup(
            "tiered_f_rank: num_digits_outer must be ≥ 1".to_string(),
        ));
    }
    let width = (n_b_prime as usize)
        .checked_mul(split_factor)
        .and_then(|x| x.checked_mul(num_digits_outer))
        .ok_or_else(|| AkitaError::InvalidSetup("F width overflow".to_string()))?;
    let collision = balanced_digit_delta_bound(outer_log_basis);
    let collision = ceil_supported_collision(family, d, collision).ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "tiered_f_rank: no supported SIS collision bucket covers Δû bound \
             {collision} for family={family:?}, D={d}"
        ))
    })?;
    let rank = min_rank_for_secure_width(family, d, collision, width as u64).ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "tiered_f_rank: no generated SIS row covers family={family:?}, D={d}, \
             collision={collision}, width={width}"
        ))
    })?;
    u32::try_from(rank)
        .map_err(|_| AkitaError::InvalidSetup("tiered_f_rank: SIS rank exceeds u32".to_string()))
}

/// Compute the smallest tier split factor `f >= ceil(|B| / |A|)` such that
/// `outer_width % f == 0`, treating `|A|` and `|B|` as the cell counts of the
/// inner and outer SIS rectangles respectively (`|A| = n_a * inner_width`,
/// `|B| = n_b * outer_width`).
///
/// Returns `Some(1)` when `|B| <= |A|` (no tiering needed) and `None` when
/// no divisor of `outer_width` in `[ceil(|B|/|A|), outer_width]` exists.
///
/// `n_b` here is the *legacy* (unchunked) B SIS rank, not the post-split
/// `n_b'`. Callers building tier metadata must use the legacy rank from
/// [`min_rank_for_secure_width`] / [`tiered_b_prime_rank`] with
/// `split_factor = 1`, *not* the chunk-width rank.
#[inline]
pub fn dynamic_tier_split_factor(
    n_a: u32,
    n_b: u32,
    inner_width: usize,
    outer_width: usize,
) -> Option<usize> {
    if outer_width == 0 || inner_width == 0 {
        return None;
    }
    let a_size = (n_a as usize).checked_mul(inner_width)?;
    let b_size = (n_b as usize).checked_mul(outer_width)?;
    if b_size <= a_size {
        return Some(1);
    }
    let min_f = b_size.div_ceil(a_size);
    if min_f > outer_width {
        return None;
    }
    (min_f..=outer_width).find(|&f| outer_width.is_multiple_of(f))
}

/// Compute `num_digits_outer` so the balanced gadget of basis
/// `b = 2^outer_log_basis` covers the full centered range `[-q/2, q/2)` for a
/// `field_bits`-bit modulus.
///
/// Closed form `delta = ceil((field_bits + 2) / outer_log_basis)`
/// over-provisions by at most one digit and matches the bench's manually
/// tuned `(lb=2, delta=65)` choice for `Q128`.
#[inline]
pub fn dynamic_tier_num_digits_outer(field_bits: u32, outer_log_basis: u32) -> usize {
    let numerator = (field_bits as usize) + 2;
    numerator.div_ceil(outer_log_basis as usize)
}

/// Layer dynamic tier metadata onto a fully-laid-out *legacy* root
/// `LevelParams` and return the corresponding tiered `LevelParams`.
///
/// Expects `legacy_lp` to be the unchunked root layout: `b_key.col_len` must
/// equal the full outer width and `b_key.row_len` must equal the legacy SIS
/// rank for that width. The function then:
///
/// 1. Computes `split_factor` via [`dynamic_tier_split_factor`].
/// 2. If `split_factor == 1` (no tiering), returns a clone of the input with
///    its tier fields cleared (`split_factor = 1`, empty `f_key`, etc.).
/// 3. Otherwise re-derives `n_b' = tiered_b_prime_rank`,
///    `n_F = tiered_f_rank`, `num_digits_outer = dynamic_tier_num_digits_outer`,
///    and constructs a `b_key` with `col_len = outer_width / split_factor`
///    plus a populated `f_key`.
///
/// # Errors
///
/// Returns an error if no valid divisor of `outer_width` in
/// `[ceil(|B|/|A|), outer_width]` exists, or if the SIS floor tables don't
/// cover the requested `(family, D, collision, width)` tuples.
pub fn apply_dynamic_tier(
    legacy_lp: &LevelParams,
    field_bits: u32,
) -> Result<LevelParams, AkitaError> {
    let n_a = legacy_lp.a_key.row_len() as u32;
    let n_b_legacy = legacy_lp.b_key.row_len() as u32;
    let inner_width = legacy_lp.inner_width();
    let outer_width = legacy_lp.full_outer_width();

    let split_factor = dynamic_tier_split_factor(n_a, n_b_legacy, inner_width, outer_width)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "apply_dynamic_tier: no valid split for outer_width={outer_width}, \
                 n_a={n_a}, n_b={n_b_legacy}, inner_width={inner_width}"
            ))
        })?;
    if split_factor == 1 {
        // No tiering needed at this shape. Return the LP with cleared tier
        // fields so downstream consumers see a clean legacy LP.
        return Ok(LevelParams {
            split_factor: 1,
            outer_log_basis: 0,
            num_digits_outer: 0,
            f_key: AjtaiKeyParams::new_unchecked(
                legacy_lp.b_key.sis_family(),
                0,
                0,
                0,
                legacy_lp.ring_dimension,
            ),
            ..legacy_lp.clone()
        });
    }

    let family = legacy_lp.b_key.sis_family();
    let d = legacy_lp.ring_dimension;
    let outer_log_basis = legacy_lp.log_basis;
    let num_digits_outer = dynamic_tier_num_digits_outer(field_bits, outer_log_basis);
    let chunk_width = outer_width / split_factor;
    let t_inf_bound = legacy_lp.b_key.collision_inf();

    let n_b_prime = tiered_b_prime_rank(family, d as u32, t_inf_bound, outer_width, split_factor)?;
    let n_f = tiered_f_rank(
        family,
        d as u32,
        outer_log_basis,
        n_b_prime,
        split_factor,
        num_digits_outer,
    )?;
    let f_width = (n_b_prime as usize)
        .checked_mul(split_factor)
        .and_then(|w| w.checked_mul(num_digits_outer))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("apply_dynamic_tier F width overflow".to_string())
        })?;
    let f_collision = balanced_digit_delta_bound(outer_log_basis);
    let tiered_b_key =
        AjtaiKeyParams::new_unchecked(family, n_b_prime as usize, chunk_width, t_inf_bound, d);
    let f_key = AjtaiKeyParams::new_unchecked(family, n_f as usize, f_width, f_collision, d);

    Ok(LevelParams {
        split_factor,
        outer_log_basis,
        num_digits_outer,
        f_key,
        b_key: tiered_b_key,
        ..legacy_lp.clone()
    })
}

/// Compute `(depth_commit, depth_open)` for one decomposition.
pub fn decomp_depths(decomp: DecompositionParams) -> (usize, usize) {
    let field_bits = decomp.field_bits();
    let depth_commit = num_digits_for_bound(decomp.log_commit_bound, field_bits, decomp.log_basis);
    let open_bound = decomp.log_open_bound.unwrap_or(decomp.log_commit_bound);
    let depth_open = num_digits_for_bound(open_bound, field_bits, decomp.log_basis);
    (depth_commit, depth_open)
}

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
    let depth_fold = compute_num_digits_fold_with_claims(
        r_vars,
        lp.challenge_l1_mass(),
        decomp.log_basis,
        1,
        decomp.field_bits(),
    );
    lp.with_decomp(
        m_vars,
        r_vars,
        depth_commit,
        depth_open,
        depth_fold,
        num_ring,
    )
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
    let (m_vars, r_vars) = optimal_m_r_split(
        lp.a_key.row_len() as u32,
        lp.challenge_l1_mass(),
        decomp.log_commit_bound,
        decomp.log_basis,
        reduced_vars,
        num_ring_elems,
        decomp.field_bits(),
    );
    let layout = level_layout_from_params(m_vars, r_vars, lp, decomp, num_ring_elems)?;
    debug_assert_eq!(layout.m_vars + layout.r_vars + alpha, max_num_vars);
    Ok(layout)
}
