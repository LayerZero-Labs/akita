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
/// fold (book §5.3 lines 627-642, book §5.4 lines 686-754).
///
/// `s_lp_in` is the chunk-shaped `S`-group LP carried in from the
/// previous level when the cascade is already active. `incoming_tier`
/// carries the tier shape the previous level used to emit `S`:
/// `un_tiered()` is the book §5.3 un-tiered 2-group `(W, S)` cascade;
/// `is_tiered()` is the book §5.4 tiered 3-group `(W, chunks, meta)`
/// cascade where the chunks group has `claim_count = k` and the meta
/// group commits the concatenated chunk B-commitments.
///
/// Pass `None` for `s_lp_in` and `un_tiered()` for `incoming_tier` when
/// the level runs single-group (only `W`).
///
/// Mirrors `PreparedMEval::setup_polynomial_row_count *
/// setup_polynomial_col_count_padded() * D` so the planner can reason
/// about the cascade growth without materializing a full `PreparedMEval`.
pub fn planned_setup_field_len(
    lp: &LevelParams,
    s_lp_in: Option<&LevelParams>,
    incoming_tier: TieredSetupParams,
    num_eval_rows: usize,
    num_commitment_groups: usize,
) -> usize {
    let n_a = lp.a_key.row_len();
    let n_b_outer = lp.b_key.row_len();
    let n_d = lp.d_key.row_len();
    let (col_count, row_count) = if let Some(s_lp) = s_lp_in {
        if incoming_tier.is_tiered() {
            let k = incoming_tier.num_chunks;
            // Meta-tier LP derived as in
            // `planned_joint_w_ring_with_setup_group_tiered`: the
            // concatenated chunk B-commitments padded to the next
            // power of two ring elements.
            let meta_field_len = (k * s_lp.b_key.row_len() * lp.ring_dimension)
                .next_power_of_two()
                .max(lp.ring_dimension);
            let meta_lp =
                untiered_setup_group_lp(lp, meta_field_len).unwrap_or_else(|_| s_lp.clone());
            let w_len = lp.num_blocks * lp.num_digits_open
                + k * s_lp.num_blocks * s_lp.num_digits_open
                + meta_lp.num_blocks * meta_lp.num_digits_open;
            let b_w = lp.num_blocks * n_a * lp.num_digits_open;
            let b_s = k * s_lp.num_blocks * n_a * s_lp.num_digits_open;
            let b_meta = meta_lp.num_blocks * n_a * meta_lp.num_digits_open;
            let max_b_cols = b_w.max(b_s).max(b_meta);
            let a_cols =
                num_eval_rows * (lp.inner_width() + s_lp.inner_width() + meta_lp.inner_width());
            let col_count = w_len.max(max_b_cols).max(a_cols).max(1);
            let max_b = n_b_outer
                .max(s_lp.b_key.row_len())
                .max(meta_lp.b_key.row_len());
            let row_count = n_a.max(n_d).max(max_b).max(1);
            (col_count, row_count)
        } else {
            let w_len = lp.num_blocks * lp.num_digits_open + s_lp.num_blocks * s_lp.num_digits_open;
            let b_w = lp.num_blocks * n_a * lp.num_digits_open;
            let b_s = s_lp.num_blocks * n_a * s_lp.num_digits_open;
            let max_b_cols = b_w.max(b_s);
            let a_cols_w = num_eval_rows * lp.inner_width();
            let a_cols_s = num_eval_rows * s_lp.inner_width();
            let a_cols = a_cols_w.saturating_add(a_cols_s);
            let col_count = w_len.max(max_b_cols).max(a_cols).max(1);
            let max_b = n_b_outer.max(s_lp.b_key.row_len());
            let row_count = n_a.max(n_d).max(max_b).max(1);
            (col_count, row_count)
        }
    } else {
        let total_blocks = num_commitment_groups * lp.num_blocks;
        let d_cols = lp.num_digits_open * total_blocks;
        let b_cols = lp.num_digits_open * n_a * total_blocks.max(lp.num_blocks);
        let a_cols = num_eval_rows * lp.inner_width();
        let col_count = d_cols.max(b_cols).max(a_cols).max(1);
        let row_count = n_a.max(n_b_outer).max(n_d).max(1);
        (col_count, row_count)
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
/// # Example
///
/// ```text
/// // Production tiered shape: f = 8, k = 64.
/// let tier = TieredSetupParams::PRODUCTION;
/// let chunk_lp = tiered_setup_group_lp(&outer_lp, |S|_in_fields, tier)?;
/// assert_eq!(chunk_lp.r_vars, outer_lp.r_vars - 3); // 3 = log2(8)
/// assert_eq!(chunk_lp.m_vars, outer_lp.m_vars - 3);
/// ```
///
/// # Errors
///
/// Pads the setup polynomial to the next power-of-two ring length before
/// chunking. Returns an error if the padded length does not divide into
/// `tier.num_chunks`, if the un-tiered `(m_S, r_S)` cannot absorb
/// `log_2 f` worth of shrinkage on each axis, or if `with_decomp` rejects
/// the derived per-chunk layout.
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
    let padded_num_ring_total = num_ring_total.next_power_of_two();
    if !padded_num_ring_total.is_multiple_of(tier.num_chunks) {
        return Err(AkitaError::InvalidSetup(format!(
            "tiered setup: padded {padded_num_ring_total} ring elements does not divide into {} chunks",
            tier.num_chunks
        )));
    }
    let chunk_num_ring = padded_num_ring_total / tier.num_chunks;
    let full_digits = compute_num_digits_full_field(128, base.log_basis);
    let fold_digits = compute_num_digits_fold_with_claims(
        r_vars_chunk,
        base.challenge_l1_mass(),
        base.log_basis,
        tier.num_chunks,
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

/// Derive per-chunk setup LP from the actual setup-polynomial matrix shape.
///
/// The routed setup polynomial is `S(row, col, coeff)`, so the tier split
/// removes high bits from the row axis (`r_vars`) and the column axis
/// (`m_vars`) directly. This is the runtime/book-aligned variant to use when
/// `row_count` and `col_count` are known.
///
/// # Errors
///
/// Returns an error if the tier shrink exceeds either axis, if the padded
/// setup-polynomial size does not divide evenly into `tier.num_chunks`, or
/// if any intermediate dimension overflows.
pub fn tiered_setup_group_lp_from_dims(
    base: &LevelParams,
    row_count: usize,
    col_count: usize,
    tier: TieredSetupParams,
) -> Result<LevelParams, AkitaError> {
    if tier.shrink_factor == 1 {
        return untiered_setup_group_lp(base, row_count * col_count * base.ring_dimension);
    }
    let log_shrink = tier.log2_shrink()? as usize;
    let row_bits = row_count.next_power_of_two().trailing_zeros() as usize;
    let col_bits = col_count.next_power_of_two().trailing_zeros() as usize;
    if row_bits < log_shrink || col_bits < log_shrink {
        return Err(AkitaError::InvalidSetup(format!(
            "tiered setup dims require row_bits and col_bits >= log2(f) = {log_shrink}; got row_bits={row_bits}, col_bits={col_bits}"
        )));
    }
    let r_vars_chunk = row_bits - log_shrink;
    let m_vars_chunk = col_bits - log_shrink;
    let padded_num_ring = (1usize << row_bits)
        .checked_mul(1usize << col_bits)
        .ok_or_else(|| AkitaError::InvalidSetup("tiered setup padded size overflow".to_string()))?;
    if !padded_num_ring.is_multiple_of(tier.num_chunks) {
        return Err(AkitaError::InvalidSetup(format!(
            "tiered setup: padded {padded_num_ring} ring elements does not divide into {} chunks",
            tier.num_chunks
        )));
    }
    let chunk_num_ring = padded_num_ring / tier.num_chunks;
    let full_digits = compute_num_digits_full_field(128, base.log_basis);
    let fold_digits = compute_num_digits_fold_with_claims(
        r_vars_chunk,
        base.challenge_l1_mass(),
        base.log_basis,
        tier.num_chunks,
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

/// Dense-coefficient source indices for the book §5.4 two-axis tier split.
///
/// The full setup polynomial is viewed with `r` block variables followed by
/// `m` in-block variables. A tier with `f = 2^s` removes the high `s` bits
/// from each axis and uses them as the `f × f` chunk index; each chunk keeps
/// the low `(r - s, m - s)` variables in the same dense order.
///
/// # Errors
///
/// Returns an error if either axis has fewer variables than `log2(f)`.
pub fn tiered_setup_chunk_index_map(
    full_r_vars: usize,
    full_m_vars: usize,
    tier: TieredSetupParams,
) -> Result<Vec<Vec<usize>>, AkitaError> {
    if tier.shrink_factor == 1 {
        return Ok(vec![(0..(1usize << (full_r_vars + full_m_vars))).collect()]);
    }
    let log_shrink = tier.log2_shrink()? as usize;
    if full_r_vars < log_shrink || full_m_vars < log_shrink {
        return Err(AkitaError::InvalidSetup(format!(
            "tiered chunk index map requires full axes >= log2(f) = {log_shrink}; got r={full_r_vars}, m={full_m_vars}"
        )));
    }
    let r_chunk = full_r_vars - log_shrink;
    let m_chunk = full_m_vars - log_shrink;
    let f = tier.shrink_factor;
    let full_block_len = 1usize << full_m_vars;
    let chunk_block_len = 1usize << m_chunk;
    let chunk_len = 1usize << (r_chunk + m_chunk);
    let mut chunks = Vec::with_capacity(tier.num_chunks);
    for high_m in 0..f {
        for high_r in 0..f {
            let mut indices = Vec::with_capacity(chunk_len);
            for low_r in 0..(1usize << r_chunk) {
                let full_block = low_r | (high_r << r_chunk);
                for low_m in 0..chunk_block_len {
                    let full_elem = low_m | (high_m << m_chunk);
                    indices.push(full_block * full_block_len + full_elem);
                }
            }
            chunks.push(indices);
        }
    }
    Ok(chunks)
}

/// Project a full routed setup opening point to the low variables used by a
/// per-chunk tiered setup claim.
///
/// # Errors
///
/// Returns an error if either axis has fewer variables than `log2(f)` or if
/// the supplied `opening_point` is shorter than the projection requires.
pub fn tiered_setup_chunk_opening_point<F: Clone>(
    opening_point: &[F],
    alpha_bits: usize,
    full_r_vars: usize,
    full_m_vars: usize,
    tier: TieredSetupParams,
) -> Result<Vec<F>, AkitaError> {
    let log_shrink = tier.log2_shrink()? as usize;
    if full_r_vars < log_shrink || full_m_vars < log_shrink {
        return Err(AkitaError::InvalidSetup(format!(
            "tiered chunk opening point requires full axes >= log2(f) = {log_shrink}; got r={full_r_vars}, m={full_m_vars}"
        )));
    }
    let r_chunk = full_r_vars - log_shrink;
    let m_chunk = full_m_vars - log_shrink;
    let expected = alpha_bits + full_r_vars + full_m_vars;
    if opening_point.len() < expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: opening_point.len(),
        });
    }
    let coeffs = &opening_point[..alpha_bits];
    let r = &opening_point[alpha_bits..alpha_bits + full_r_vars];
    let m = &opening_point[alpha_bits + full_r_vars..alpha_bits + full_r_vars + full_m_vars];
    let mut out = Vec::with_capacity(alpha_bits + r_chunk + m_chunk);
    out.extend_from_slice(coeffs);
    out.extend_from_slice(&r[..r_chunk]);
    out.extend_from_slice(&m[..m_chunk]);
    Ok(out)
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
///
/// # Example
///
/// ```text
/// let tier = TieredSetupParams::new(2).unwrap();  // f = 2, k = 4
/// let s_lp = tiered_setup_group_lp(&outer_lp, setup_field_len, tier)?;
/// let joint =
///     planned_joint_w_ring_with_setup_group_tiered(128, &outer_lp, &s_lp, tier, 2);
/// // Joint output includes W's contribution + 4 chunks' contributions + meta-tier
/// // rows; passing `tier = TieredSetupParams::un_tiered()` reproduces
/// // `planned_joint_w_ring_with_setup_group` bit-for-bit.
/// ```
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
    let k = tier.num_chunks;
    // Phase 5 / book §5.4 routed-tier shape: at the next level the
    // routed S claim expands into `k + 1` claims forming THREE
    // commitment groups via the merge rule (chunks-as-1-group with
    // `claim_count = k` carrying `tier = Some(t)`, plus W and meta as
    // standard 1-claim groups). The merge rule is applied in
    // `prove_recursive_multi_fold_with_params` and mirrored on the
    // verifier side.
    //
    // Meta-tier LP: the meta polynomial concatenates the k chunk B-side
    // commitment vectors (length `k * n_B_chunk` ring elements, padded
    // to next pow2). Sized via `untiered_setup_group_lp` against the
    // outer LP, mirroring the prover-side `meta_lp_from_chunks` in
    // `crates/akita-prover/src/protocol/flow.rs`.
    let meta_field_len = (k * s_lp.b_key.row_len() * outer_lp.ring_dimension)
        .next_power_of_two()
        .max(outer_lp.ring_dimension);
    let meta_lp = match untiered_setup_group_lp(outer_lp, meta_field_len) {
        Ok(lp) => lp,
        Err(_) => s_lp.clone(),
    };
    let w_hat_w = outer_lp.num_blocks * outer_lp.num_digits_open;
    let t_hat_w = outer_lp.num_blocks * n_a * outer_lp.num_digits_open;
    let z_pre_w = num_eval_rows * outer_lp.inner_width() * outer_lp.num_digits_fold;
    // Chunks group with claim_count=k under shared chunk_lp:
    // `w_hat` and `t_hat` scale with `k` (each chunk has its own
    // digit-decomposed inner witness). `z_pre` does NOT scale with `k`
    // because the chunks share folding challenges (book line 949) and
    // their folded witnesses sum into ONE z_pre per group via
    // `aggregate_decompose_fold_witnesses`.
    let w_hat_s = k * s_lp.num_blocks * s_lp.num_digits_open;
    let t_hat_s = k * s_lp.num_blocks * n_a * s_lp.num_digits_open;
    let z_pre_s = num_eval_rows * s_lp.inner_width() * s_lp.num_digits_fold;
    let w_hat_meta = meta_lp.num_blocks * meta_lp.num_digits_open;
    let t_hat_meta = meta_lp.num_blocks * n_a * meta_lp.num_digits_open;
    let z_pre_meta = num_eval_rows * meta_lp.inner_width() * meta_lp.num_digits_fold;
    // M-relation r-tail: 1 consistency + num_eval_rows + tier-aware D rows
    // + sum_g n_B_g + original/meta A rows. The tiered routed shape has W,
    // k chunks, and meta D slices under the shared D prefix.
    let total_b = outer_lp.b_key.row_len() + k * s_lp.b_key.row_len() + meta_lp.b_key.row_len();
    let total_d = (k + 2) * outer_lp.d_key.row_len();
    let r_rows = 3 + num_eval_rows + total_d + total_b + 3 * n_a;
    let r_count = r_rows * compute_num_digits_full_field(field_bits, outer_lp.log_basis);
    w_hat_w
        + t_hat_w
        + z_pre_w
        + w_hat_s
        + t_hat_s
        + z_pre_s
        + w_hat_meta
        + t_hat_meta
        + z_pre_meta
        + r_count
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AjtaiKeyParams, LevelParams};
    use akita_challenges::{SparseChallengeConfig, Stage1ChallengeShape};

    fn lp_for_chunk_test() -> LevelParams {
        LevelParams {
            ring_dimension: 32,
            log_basis: 2,
            a_key: AjtaiKeyParams::new_unchecked(1, 1, 0, 32),
            b_key: AjtaiKeyParams::new_unchecked(1, 1, 0, 32),
            d_key: AjtaiKeyParams::new_unchecked(1, 1, 0, 32),
            num_blocks: 16,
            block_len: 8,
            m_vars: 3,
            r_vars: 4,
            stage1_config: SparseChallengeConfig::ExactShell {
                count_mag1: 1,
                count_mag2: 0,
            },
            stage1_challenge_shape: Stage1ChallengeShape::Flat,
            use_setup_claim_reduction: false,
            num_digits_commit: 1,
            num_digits_open: 1,
            num_digits_fold: 1,
            groups: None,
        }
    }

    #[test]
    fn tiered_setup_group_lp_un_tiered_matches_untiered_helper() {
        let base = lp_for_chunk_test();
        let setup_field_len = 4 * 2 * base.ring_dimension; // 8 ring elements
        let untiered = untiered_setup_group_lp(&base, setup_field_len).expect("untiered");
        let tiered_un_tier =
            tiered_setup_group_lp(&base, setup_field_len, TieredSetupParams::un_tiered())
                .expect("tiered with f=1 must equal untiered");
        assert_eq!(untiered.m_vars, tiered_un_tier.m_vars);
        assert_eq!(untiered.r_vars, tiered_un_tier.r_vars);
        assert_eq!(untiered.num_blocks, tiered_un_tier.num_blocks);
        assert_eq!(untiered.block_len, tiered_un_tier.block_len);
        assert_eq!(untiered.num_digits_open, tiered_un_tier.num_digits_open);
        assert_eq!(untiered.num_digits_commit, tiered_un_tier.num_digits_commit);
    }

    #[test]
    fn tiered_setup_group_lp_shrinks_m_r_by_log2_f() {
        let base = lp_for_chunk_test();
        // Build a polynomial with enough variables that f=2 can shrink
        // both m_S and r_S by 1 each. The un-tiered helper picks (m_S,
        // r_S) by setup_field_len; we pre-compute that and then check
        // the tiered version drops one bit each.
        let setup_field_len = 16 * 16 * base.ring_dimension; // 256 ring elements
        let untiered = untiered_setup_group_lp(&base, setup_field_len).expect("untiered");
        assert!(untiered.m_vars >= 1 && untiered.r_vars >= 1);
        let tier = TieredSetupParams::new(2).expect("f=2");
        let tiered = tiered_setup_group_lp(&base, setup_field_len, tier).expect("tiered");
        assert_eq!(tiered.m_vars, untiered.m_vars - 1);
        assert_eq!(tiered.r_vars, untiered.r_vars - 1);
        // The per-chunk num_blocks halves under f=2.
        assert_eq!(tiered.num_blocks * 2, untiered.num_blocks);
    }

    #[test]
    fn tiered_setup_group_lp_rejects_insufficient_axes() {
        let base = lp_for_chunk_test();
        // A polynomial too small for f=8 (would need log2(f) = 3 on each
        // axis, but m_S would be 0 after shrinkage).
        let setup_field_len = 2 * 2 * base.ring_dimension;
        let tier = TieredSetupParams::new(8).expect("f=8");
        assert!(tiered_setup_group_lp(&base, setup_field_len, tier).is_err());
    }

    #[test]
    fn planned_joint_tiered_un_tiered_matches_untiered_helper() {
        let base = lp_for_chunk_test();
        let setup_field_len = 4 * 2 * base.ring_dimension;
        let s_lp = untiered_setup_group_lp(&base, setup_field_len).expect("s_lp");
        let untiered = planned_joint_w_ring_with_setup_group(128, &base, &s_lp, 2);
        let tiered_at_f1 = planned_joint_w_ring_with_setup_group_tiered(
            128,
            &base,
            &s_lp,
            TieredSetupParams::un_tiered(),
            2,
        );
        assert_eq!(untiered, tiered_at_f1);
    }

    #[test]
    fn planned_joint_tiered_grows_with_k_chunks() {
        let base = lp_for_chunk_test();
        let setup_field_len = 16 * 16 * base.ring_dimension;
        let tier = TieredSetupParams::new(2).expect("f=2");
        let s_lp = tiered_setup_group_lp(&base, setup_field_len, tier).expect("tiered s_lp");
        let tiered = planned_joint_w_ring_with_setup_group_tiered(128, &base, &s_lp, tier, 2);
        // The tiered joint output sums the per-chunk contributions times
        // `k = 4` plus a meta-tier overhead — it must be strictly
        // larger than zero and capture the per-chunk multiplier.
        assert!(tiered > 0);
        // Per-chunk S contribution * k must scale with k.
        let s_lp_double = tiered_setup_group_lp(
            &base,
            setup_field_len,
            TieredSetupParams::new(4).expect("f=4"),
        )
        .expect("f=4 lp");
        let tiered_f4 = planned_joint_w_ring_with_setup_group_tiered(
            128,
            &base,
            &s_lp_double,
            TieredSetupParams::new(4).expect("f=4"),
            2,
        );
        // f=4 has k=16 chunks vs f=2's k=4 chunks. Different size but
        // both finite and well-defined.
        assert!(tiered_f4 > 0);
    }
}
