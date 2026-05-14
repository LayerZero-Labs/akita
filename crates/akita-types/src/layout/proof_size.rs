//! Header-stripped proof-size and planned-witness sizing formulas.

use akita_field::AkitaError;

use crate::layout::digit_math::{
    compute_num_digits_fold_with_claims, compute_num_digits_full_field,
};
use crate::stage1_tree_stage_shapes;
use crate::{DirectWitnessShape, LevelParams, TieredSetupParams};

/// Field element size in bytes for a field with `field_bits` bits.
pub fn field_bytes(field_bits: u32) -> usize {
    (field_bits as usize).div_ceil(8)
}

/// Ring vector bytes without a length prefix.
pub fn proof_ring_vec_bytes(ring_len: usize, ring_dim: usize, elem_bytes: usize) -> usize {
    ring_len.saturating_mul(ring_dim).saturating_mul(elem_bytes)
}

/// Packed digit bytes without a length/tag prefix.
pub fn packed_digits_bytes(num_elems: usize, bits_per_elem: u32) -> usize {
    num_elems.saturating_mul(bits_per_elem as usize).div_ceil(8)
}

/// Serialized byte size for a terminal direct witness shape.
pub fn direct_witness_bytes(field_bits: u32, shape: &DirectWitnessShape) -> usize {
    match shape {
        DirectWitnessShape::PackedDigits((num_elems, bits_per_elem)) => {
            packed_digits_bytes(*num_elems, *bits_per_elem)
        }
        DirectWitnessShape::FieldElements(num_coeffs) => {
            num_coeffs.saturating_mul(field_bytes(field_bits))
        }
    }
}

fn compressed_unipoly_bytes(degree: usize, elem_bytes: usize) -> usize {
    degree * elem_bytes
}

fn sumcheck_bytes(rounds: usize, degree: usize, elem_bytes: usize) -> usize {
    rounds * compressed_unipoly_bytes(degree, elem_bytes)
}

fn stage1_proof_bytes(rounds: usize, b: usize, elem_bytes: usize) -> usize {
    stage1_tree_stage_shapes(rounds, b)
        .into_iter()
        .map(|stage| {
            sumcheck_bytes(rounds, stage.sumcheck.1, elem_bytes) + stage.child_claims * elem_bytes
        })
        .sum::<usize>()
        + elem_bytes
}

/// Planned recursive witness size in ring elements for a singleton fold.
pub fn planned_w_ring_element_count(field_bits: u32, lp: &LevelParams) -> usize {
    planned_w_ring_element_count_with_claims(field_bits, lp, 1)
}

/// Planned recursive witness size in ring elements when this level
/// jointly opens `num_claims` polynomials under one shared LP.
///
/// Phase D-full: when the previous level emits a setup-claim-reduction
/// payload AND routes `S` recursively, this level sees `num_claims = 2`
/// (the folded witness and the routed `S`). The recursive witness
/// produced here has `w_hat` and `t_hat` sized per-claim; `z_pre` is
/// per-point (`num_points` distinct opening points); `r` rows scale
/// with `(num_commitment_groups, num_points)`.
///
/// For the `k = 1` routing the per-point and per-group counts equal
/// `num_claims` (one commitment per claim, one opening point per
/// claim), so the standard joint-open shape `(num_claims, num_claims,
/// num_claims)` flows through.
pub fn planned_w_ring_element_count_with_claims(
    field_bits: u32,
    lp: &LevelParams,
    num_claims: usize,
) -> usize {
    let w_hat_count = num_claims * lp.num_blocks * lp.num_digits_open;
    let t_hat_count = num_claims * lp.num_blocks * lp.a_key.row_len() * lp.num_digits_open;
    let z_pre_count = num_claims * lp.inner_width() * lp.num_digits_fold;
    let r_count = lp.m_row_count(num_claims, num_claims)
        * compute_num_digits_full_field(field_bits, lp.log_basis);
    w_hat_count + t_hat_count + z_pre_count + r_count
}

/// Planned recursive witness size in field elements for a singleton fold.
pub fn planned_next_w_len(field_bits: u32, lp: &LevelParams) -> usize {
    planned_w_ring_element_count(field_bits, lp) * lp.ring_dimension
}

/// Planned recursive witness size in field elements for a multi-claim
/// fold; see [`planned_w_ring_element_count_with_claims`].
pub fn planned_next_w_len_with_claims(
    field_bits: u32,
    lp: &LevelParams,
    num_claims: usize,
) -> usize {
    planned_w_ring_element_count_with_claims(field_bits, lp, num_claims) * lp.ring_dimension
}

/// Derive the per-group `LevelParams` for the un-tiered setup-polynomial
/// (`S`) handle (book §5.3 "split commitment", un-tiered `f = 1`) attached
/// to a level whose outer LP is `base`.
///
/// `setup_field_len` is the number of field-element coefficients that the
/// setup polynomial `S` occupies at the source level's M-table view; it
/// must be a multiple of `base.ring_dimension`. The returned LP keeps
/// the shared outer fields (`D`, `A`, ring dimension, log basis, stage-1
/// challenge config) and overrides the per-group `(m, r, B, digit_count)`
/// with values sized for the `S` polynomial. Both `num_digits_open` and
/// `num_digits_commit` are set to the full-field digit count
/// `⌈128 / log_2 b⌉` (book §5.3 line 637).
///
/// # Errors
///
/// Returns an error if `setup_field_len` is not a multiple of
/// `base.ring_dimension` or `with_decomp` rejects the derived layout.
pub fn untiered_setup_group_lp(
    base: &LevelParams,
    setup_field_len: usize,
) -> Result<LevelParams, AkitaError> {
    if !setup_field_len.is_multiple_of(base.ring_dimension) {
        return Err(AkitaError::InvalidSetup(
            "setup polynomial length is not divisible by ring dimension".to_string(),
        ));
    }
    let num_ring = setup_field_len / base.ring_dimension;
    let reduced_vars = num_ring.next_power_of_two().max(1).trailing_zeros() as usize;
    let r_vars = base.r_vars.min(reduced_vars);
    let m_vars = reduced_vars - r_vars;
    let full_digits = compute_num_digits_full_field(128, base.log_basis);
    let fold_digits = compute_num_digits_fold_with_claims(
        r_vars,
        base.challenge_l1_mass(),
        base.log_basis,
        1,
        128,
    );
    base.with_decomp(
        m_vars,
        r_vars,
        full_digits,
        full_digits,
        fold_digits,
        num_ring,
    )
}

/// Planned setup-polynomial field-element length emitted by level `lp`'s
/// M-table when the level's setup-claim reduction routes `S` to the next
/// fold (book §5.3 lines 627-642).
///
/// `s_lp_in` is the `S`-group LP carried in from the previous level when
/// the cascade is already active (so the current level's M-table sees
/// 2 groups under multi-group batched Hachi); pass `None` when the level
/// runs single-group (only `W`).
///
/// Mirrors `PreparedMEval::setup_polynomial_row_count *
/// setup_polynomial_col_count_padded() * D` so the planner can reason
/// about the cascade growth without materializing a full `PreparedMEval`.
pub fn planned_setup_field_len(
    lp: &LevelParams,
    s_lp_in: Option<&LevelParams>,
    num_eval_rows: usize,
    num_commitment_groups: usize,
) -> usize {
    let n_a = lp.a_key.row_len();
    let n_b_outer = lp.b_key.row_len();
    let n_d = lp.d_key.row_len();
    let row_count = if let Some(s_lp) = s_lp_in {
        let max_b = n_b_outer.max(s_lp.b_key.row_len());
        n_a.max(n_d).max(max_b).max(1)
    } else {
        n_a.max(n_b_outer).max(n_d).max(1)
    };
    let col_count = if let Some(s_lp) = s_lp_in {
        let w_len = lp.num_blocks * lp.num_digits_open + s_lp.num_blocks * s_lp.num_digits_open;
        let b_w = lp.num_blocks * n_a * lp.num_digits_open;
        let b_s = s_lp.num_blocks * n_a * s_lp.num_digits_open;
        let max_b_cols = b_w.max(b_s);
        let a_cols_w = num_eval_rows * lp.inner_width();
        let a_cols_s = num_eval_rows * s_lp.inner_width();
        let a_cols = a_cols_w.saturating_add(a_cols_s);
        w_len.max(max_b_cols).max(a_cols).max(1)
    } else {
        let total_blocks = num_commitment_groups * lp.num_blocks;
        let d_cols = lp.num_digits_open * total_blocks;
        let b_cols = lp.num_digits_open * n_a * total_blocks.max(lp.num_blocks);
        let a_cols = num_eval_rows * lp.inner_width();
        d_cols.max(b_cols).max(a_cols).max(1)
    };
    let col_count_padded = col_count.next_power_of_two();
    row_count
        .saturating_mul(col_count_padded)
        .saturating_mul(lp.ring_dimension)
}

/// Planned multi-group joint fold output (ring-element count) when level
/// `lp` jointly opens `(W, S)` as two commitment groups under shared
/// outer `(D, A)` (book §5.3 "split commitment", un-tiered `f = 1`).
///
/// `outer_lp` carries the shared outer fields and the `W` group's
/// `(m, r, B, digit_count)`; `s_lp` carries the `S` group's
/// `(m, r, B, digit_count)`. `num_eval_rows` is the number of distinct
/// opening points (= 2 under the 1-claim-per-point inference rule).
pub fn planned_joint_w_ring_with_setup_group(
    field_bits: u32,
    outer_lp: &LevelParams,
    s_lp: &LevelParams,
    num_eval_rows: usize,
) -> usize {
    let n_a = outer_lp.a_key.row_len();
    let w_hat_w = outer_lp.num_blocks * outer_lp.num_digits_open;
    let t_hat_w = outer_lp.num_blocks * n_a * outer_lp.num_digits_open;
    let z_pre_w = num_eval_rows * outer_lp.inner_width() * outer_lp.num_digits_fold;
    let w_hat_s = s_lp.num_blocks * s_lp.num_digits_open;
    let t_hat_s = s_lp.num_blocks * n_a * s_lp.num_digits_open;
    let z_pre_s = num_eval_rows * s_lp.inner_width() * s_lp.num_digits_fold;
    let r_rows = outer_lp.m_row_count(2, num_eval_rows);
    let r_count = r_rows * compute_num_digits_full_field(field_bits, outer_lp.log_basis);
    w_hat_w + t_hat_w + z_pre_w + w_hat_s + t_hat_s + z_pre_s + r_count
}

/// Planned multi-group joint fold output in field elements; see
/// [`planned_joint_w_ring_with_setup_group`].
pub fn planned_joint_next_w_len_with_setup_group(
    field_bits: u32,
    outer_lp: &LevelParams,
    s_lp: &LevelParams,
    num_eval_rows: usize,
) -> usize {
    planned_joint_w_ring_with_setup_group(field_bits, outer_lp, s_lp, num_eval_rows)
        * outer_lp.ring_dimension
}

/// Derive the **per-chunk** `LevelParams` for the tiered setup-polynomial
/// (`S`) handle (book §5.4 "tiered commitment design", `f ≥ 2`).
///
/// Per book lines 686–699: `S` is split row-major into `k = f²` chunks of
/// `2^{r_chunk}` blocks each, where `r_chunk = r_S − log_2 f` and each
/// chunk is committed under shared per-chunk matrices `(D_chunk,
/// B_chunk)` that are `1/f` the column width of the baseline (line 702).
/// The returned LP therefore has `(m_chunk, r_chunk) = (m_S − log_2 f,
/// r_S − log_2 f)`, leaving the shared outer fields (`D`, `A`, ring
/// dimension, log basis, stage-1 challenge config) intact.
///
/// `tier.shrink_factor == 1` is the un-tiered case and degenerates to
/// [`untiered_setup_group_lp`]; callers that know they are un-tiered
/// should use that function directly.
///
/// # Errors
///
/// Returns an error if `setup_field_len` is not a multiple of
/// `base.ring_dimension * tier.num_chunks`, if the un-tiered `(m_S,
/// r_S)` cannot absorb `log_2 f` worth of shrinkage on each axis, or if
/// `with_decomp` rejects the derived per-chunk layout.
pub fn tiered_setup_group_lp(
    base: &LevelParams,
    setup_field_len: usize,
    tier: TieredSetupParams,
) -> Result<LevelParams, AkitaError> {
    if tier.shrink_factor == 1 {
        return untiered_setup_group_lp(base, setup_field_len);
    }
    let untiered = untiered_setup_group_lp(base, setup_field_len)?;
    let log_shrink = tier.log2_shrink()? as usize;
    if untiered.r_vars < log_shrink || untiered.m_vars < log_shrink {
        return Err(AkitaError::InvalidSetup(format!(
            "tiered shrink factor f = {} requires (m_S, r_S) >= log2(f) = {} on both axes; \
             got (m_S, r_S) = ({}, {})",
            tier.shrink_factor, log_shrink, untiered.m_vars, untiered.r_vars
        )));
    }
    let r_vars_chunk = untiered.r_vars - log_shrink;
    let m_vars_chunk = untiered.m_vars - log_shrink;
    let num_ring_total = setup_field_len / base.ring_dimension;
    if !num_ring_total.is_multiple_of(tier.num_chunks) {
        return Err(AkitaError::InvalidSetup(format!(
            "tiered setup: {num_ring_total} ring elements does not divide into {} chunks",
            tier.num_chunks
        )));
    }
    let chunk_num_ring = num_ring_total / tier.num_chunks;
    let full_digits = compute_num_digits_full_field(128, base.log_basis);
    let fold_digits = compute_num_digits_fold_with_claims(
        r_vars_chunk,
        base.challenge_l1_mass(),
        base.log_basis,
        1,
        128,
    );
    base.with_decomp(
        m_vars_chunk,
        r_vars_chunk,
        full_digits,
        full_digits,
        fold_digits,
        chunk_num_ring,
    )
}

/// Planned multi-group joint fold output (ring-element count) for the
/// tiered case from book §5.4 (groups 1–10).
///
/// The L+1 commit jointly opens `(W, S)` where the `S` group is split
/// into `k = f²` chunks under shared `(D_chunk, B_chunk)` plus a
/// tier-3 meta-commit. `s_lp` is the per-chunk-shaped LP (the output of
/// [`tiered_setup_group_lp`]); `tier` carries `(f, k)`.
///
/// Per book line 762 (T2 ratio) the cascade penalty from tiered S is
/// `|S|/f` — at `f = 8` the table-1 sweet spot makes T2 ratio ≈ 1, so
/// cascading remains viable. This helper computes the joint output
/// exactly by summing the W contribution, `k ×` per-chunk S
/// contribution, the meta-tier contribution, and the augmented
/// `m_row_count` for the 10-check-group relation.
///
/// `num_eval_rows` is the number of distinct opening points (= 2 under
/// the 1-claim-per-point inference rule, matching the un-tiered case).
pub fn planned_joint_w_ring_with_setup_group_tiered(
    field_bits: u32,
    outer_lp: &LevelParams,
    s_lp: &LevelParams,
    tier: TieredSetupParams,
    num_eval_rows: usize,
) -> usize {
    if tier.shrink_factor == 1 {
        return planned_joint_w_ring_with_setup_group(field_bits, outer_lp, s_lp, num_eval_rows);
    }
    let n_a = outer_lp.a_key.row_len();
    // W group: unchanged from un-tiered (book §5.4 line 711: W-group is
    // unaffected by tiering).
    let w_hat_w = outer_lp.num_blocks * outer_lp.num_digits_open;
    let t_hat_w = outer_lp.num_blocks * n_a * outer_lp.num_digits_open;
    let z_pre_w = num_eval_rows * outer_lp.inner_width() * outer_lp.num_digits_fold;
    // S group: k chunks under shared per-chunk matrices. Each chunk
    // contributes its own (w_hat, t_hat, z_pre) at the per-chunk shape.
    let k = tier.num_chunks;
    let w_hat_s = k * s_lp.num_blocks * s_lp.num_digits_open;
    let t_hat_s = k * s_lp.num_blocks * n_a * s_lp.num_digits_open;
    let z_pre_s = k * num_eval_rows * s_lp.inner_width() * s_lp.num_digits_fold;
    // Meta-tier (groups 6–10): D_meta, B_meta, A_meta sized for the
    // collection of k per-chunk commitment vectors. Per book line 698
    // the on-wire contribution is independent of k. For planner sizing
    // we approximate the meta-tier z_pre by the per-chunk count of
    // u_{S,j} ring elements (which is what t̂_meta digit-decomposes)
    // plus a single fold-block row. The exact meta-tier sizing depends
    // on the meta-tier `(D_meta, B_meta, A_meta)` rank derivation which
    // the planner picks separately; here we keep the planner sizing
    // conservative by adding the meta-tier rows to `r_rows` below.
    let meta_z_pre = num_eval_rows * s_lp.b_key.row_len() * s_lp.num_digits_fold;
    // The 10-check-group relation rows (book §5.4 lines 709–750):
    // - 1 consistency row
    // - `num_eval_rows` y-rows
    // - n_D rows (joint D)
    // - n_B rows per commitment group (W and S share outer B-key)
    // - 5 meta-tier groups add: n_D_meta + n_B_meta + 1 + 1 + n_A_meta rows
    // For planner sizing we treat the meta-tier rank as the per-chunk
    // rank (a conservative upper bound the planner refines once Slice H
    // regenerates the SIS tables with the meta-tier roles).
    let meta_extra_rows = s_lp.d_key.row_len() + s_lp.b_key.row_len() + s_lp.a_key.row_len() + 2;
    let r_rows = outer_lp.m_row_count(2, num_eval_rows) + meta_extra_rows;
    let r_count = r_rows * compute_num_digits_full_field(field_bits, outer_lp.log_basis);
    w_hat_w + t_hat_w + z_pre_w + w_hat_s + t_hat_s + z_pre_s + meta_z_pre + r_count
}

/// Planned multi-group joint fold output in field elements for the
/// tiered case; see [`planned_joint_w_ring_with_setup_group_tiered`].
pub fn planned_joint_next_w_len_with_setup_group_tiered(
    field_bits: u32,
    outer_lp: &LevelParams,
    s_lp: &LevelParams,
    tier: TieredSetupParams,
    num_eval_rows: usize,
) -> usize {
    planned_joint_w_ring_with_setup_group_tiered(field_bits, outer_lp, s_lp, tier, num_eval_rows)
        * outer_lp.ring_dimension
}

/// Total sumcheck rounds (`col_bits + ring_bits`) for one fold level.
pub fn sumcheck_rounds(level_d: usize, next_w_len: usize) -> usize {
    let ring_bits = level_d.trailing_zeros() as usize;
    let num_ring_elems = next_w_len / level_d;
    let col_bits = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    col_bits + ring_bits
}

/// Header-stripped byte size of one folded proof level.
pub fn level_proof_bytes(
    field_bits: u32,
    lp: &LevelParams,
    level_lp: &LevelParams,
    next_lp: &LevelParams,
    next_w_len: usize,
    num_claims: usize,
) -> usize {
    let elem_bytes = field_bytes(field_bits);
    let y_bytes = proof_ring_vec_bytes(num_claims, lp.ring_dimension, elem_bytes);
    let v_bytes = proof_ring_vec_bytes(lp.d_key.row_len(), lp.ring_dimension, elem_bytes);
    let next_commit_bytes =
        proof_ring_vec_bytes(next_lp.b_key.row_len(), next_lp.ring_dimension, elem_bytes);
    let next_eval_bytes = elem_bytes;
    let rounds = sumcheck_rounds(lp.ring_dimension, next_w_len);
    let b = 1usize << level_lp.log_basis;
    let stage1_bytes = stage1_proof_bytes(rounds, b, elem_bytes);

    y_bytes
        + v_bytes
        + stage1_bytes
        + sumcheck_bytes(rounds, 3, elem_bytes)
        + next_commit_bytes
        + next_eval_bytes
}

/// Header-stripped byte size of a singleton recursive proof level.
pub fn recursive_level_proof_bytes(
    field_bits: u32,
    lp: &LevelParams,
    next_lp: &LevelParams,
    next_w_len: usize,
) -> usize {
    level_proof_bytes(field_bits, lp, lp, next_lp, next_w_len, 1)
}
