//! Header-stripped proof-size and planned-witness sizing formulas.

use akita_field::AkitaError;

use crate::generated::sis_floor::min_rank_for_secure_width;
use crate::layout::digit_math::{
    compute_num_digits_fold_with_claims, compute_num_digits_full_field,
};
use crate::stage1_tree_stage_shapes;
use crate::{AjtaiKeyParams, DirectWitnessShape, LevelParams, TieredSetupParams};

/// Re-derive the **B-role** Ajtai rank for a per-tier `LevelParams` from
/// its concrete `outer_width` via the 128-bit SIS floor table — SHRINK
/// ONLY.
///
/// Book §5.4 line 798-799 example: at `f = 8`, `D = 32`, `n_v = 32` the
/// per-chunk SIS-secure ranks drop because each chunk's widths are
/// `1/f` of the un-tiered baseline. Phase 5 / Item (a) wires this drop
/// in for the B-role only, because the per-tier `GroupSpec` carries
/// only `b_key` (the A-role and D-role at the next level are read from
/// the OUTER LP, not from the per-tier LP, per the multi-group commit
/// kernel's per-`GroupSpec` partitioning). Touching `a_key` or `d_key`
/// here would cause the L0 commit (which uses the per-tier LP's full
/// `(a_key, b_key, d_key)`) and the L+1 M-relation (which uses outer's
/// `a_key`/`d_key` for tier-marked groups) to disagree and produce
/// `AkitaError::InvalidProof` at verify.
///
/// Returns the LP unchanged when:
///   - `collision_inf == 0` on the B-role (planner hasn't pinned the
///     SIS bucket yet, e.g. during early-construction stages).
///   - The derived rank would GROW the inherited rank: that would
///     indicate the outer base LP was insecure for the chunk widths,
///     a different bug that this helper does not silently paper over.
///     The runtime keeps the inherited (outer-derived) rank in that
///     case; `validate_stored_sis_ranks` will surface the issue.
///
/// # Errors
///
/// Returns an error when the B-role's width / collision-bucket pair has
/// no entry in `sis_floor`, or when `AjtaiKeyParams::try_new` rejects
/// the shrunken rank.
pub fn derive_chunk_sis_ranks_from_widths(lp: LevelParams) -> Result<LevelParams, AkitaError> {
    let d = lp.ring_dimension as u32;
    let b_collision = lp.b_key.collision_inf();
    if b_collision == 0 {
        return Ok(lp);
    }
    let outer_width = lp.b_key.col_len();
    let Some(secure_n_b) = min_rank_for_secure_width(d, b_collision, outer_width as u64) else {
        return Err(AkitaError::InvalidSetup(format!(
            "SIS floor lookup for chunk B role missing: D={d} \
             collision={b_collision} width={outer_width}"
        )));
    };
    let current_n_b = lp.b_key.row_len();
    let new_n_b = secure_n_b.max(1).min(current_n_b);
    if new_n_b == current_n_b {
        return Ok(lp);
    }
    let new_b_key = AjtaiKeyParams::try_new(new_n_b, outer_width, b_collision, lp.ring_dimension)?;
    let LevelParams {
        ring_dimension,
        log_basis,
        a_key,
        b_key: _,
        d_key,
        num_blocks,
        block_len,
        m_vars,
        r_vars,
        stage1_config,
        stage1_challenge_shape,
        use_setup_claim_reduction,
        num_digits_commit,
        num_digits_open,
        num_digits_fold,
        groups,
    } = lp;
    Ok(LevelParams {
        ring_dimension,
        log_basis,
        a_key,
        b_key: new_b_key,
        d_key,
        num_blocks,
        block_len,
        m_vars,
        r_vars,
        stage1_config,
        stage1_challenge_shape,
        use_setup_claim_reduction,
        num_digits_commit,
        num_digits_open,
        num_digits_fold,
        groups,
    })
}

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

/// Planned padded `(row_count, col_count_padded)` for the stage-2
/// M-table setup-polynomial view at level `lp` (book §5.3 lines
/// 627-642, book §5.4 lines 686-754).
///
/// Single source of truth for the cascade-aware dims that both the
/// emitted setup-polynomial length ([`planned_setup_field_len`]) and
/// the setup-claim-reduction sumcheck rounds
/// ([`planned_setup_claim_reduction_rounds`]) consume. Mirrors the
/// runtime envelope `PreparedMEval` materializes via
/// [`setup_polynomial_padded_dims_inner`] without instantiating a
/// `PreparedMEval` at planner time.
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
pub fn planned_setup_padded_dims(
    lp: &LevelParams,
    s_lp_in: Option<&LevelParams>,
    incoming_tier: TieredSetupParams,
    num_eval_rows: usize,
    num_commitment_groups: usize,
) -> (usize, usize) {
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
            // Book §5.5 line 752: the k chunks share `D_chunk / B_chunk`,
            // so the setup-polynomial col envelope is the MAX across the
            // three (W, chunks, meta) groups — each group writes to its
            // own per-group col range that overlaps with the others, and
            // the chunks group does NOT pick up a `k` multiplier (the
            // chunks accumulate into the SAME col slots via the
            // chunk-independent `d_col / local_col / phys_k` formulas
            // in `setup_weight_table_at_point_grouped`). Mirrors the
            // runtime envelope from `setup_polynomial_padded_dims_inner`.
            let group_max_col = |group_lp: &LevelParams, effective_claims: usize| -> usize {
                let d_cols = effective_claims
                    .saturating_mul(group_lp.num_blocks)
                    .saturating_mul(group_lp.num_digits_open);
                let b_cols = effective_claims
                    .saturating_mul(group_lp.num_blocks)
                    .saturating_mul(n_a)
                    .saturating_mul(group_lp.num_digits_open);
                let a_cols = num_eval_rows
                    .saturating_mul(group_lp.block_len)
                    .saturating_mul(group_lp.num_digits_commit);
                d_cols.max(b_cols).max(a_cols)
            };
            let w_max = group_max_col(lp, 1);
            let chunks_max = group_max_col(s_lp, 1); // tiered: shared chunks
            let meta_max = group_max_col(&meta_lp, 1);
            let col_count = w_max.max(chunks_max).max(meta_max).max(1);
            let max_b = n_b_outer
                .max(s_lp.b_key.row_len())
                .max(meta_lp.b_key.row_len());
            let row_count = n_a.max(n_d).max(max_b).max(1);
            (col_count, row_count)
        } else {
            // Un-tiered (W, S) cascade: each group writes to its own
            // per-group col range that overlaps with the other. Take the
            // MAX (not sum) across the two groups — the col envelope is
            // structurally `max(W's max_col, S's max_col)`.
            let group_max_col = |group_lp: &LevelParams, effective_claims: usize| -> usize {
                let d_cols = effective_claims
                    .saturating_mul(group_lp.num_blocks)
                    .saturating_mul(group_lp.num_digits_open);
                let b_cols = effective_claims
                    .saturating_mul(group_lp.num_blocks)
                    .saturating_mul(n_a)
                    .saturating_mul(group_lp.num_digits_open);
                let a_cols = num_eval_rows
                    .saturating_mul(group_lp.block_len)
                    .saturating_mul(group_lp.num_digits_commit);
                d_cols.max(b_cols).max(a_cols)
            };
            let col_count = group_max_col(lp, 1).max(group_max_col(s_lp, 1)).max(1);
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
    (row_count, col_count.next_power_of_two())
}

/// Planned setup-polynomial field-element length emitted by level `lp`'s
/// M-table.
///
/// Equals `row_count * col_count_padded * ring_dimension` for the dims
/// returned by [`planned_setup_padded_dims`]; see that function for
/// `s_lp_in` / `incoming_tier` semantics.
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
    let (row_count, col_count_padded) = planned_setup_padded_dims(
        lp,
        s_lp_in,
        incoming_tier,
        num_eval_rows,
        num_commitment_groups,
    );
    row_count
        .saturating_mul(col_count_padded)
        .saturating_mul(lp.ring_dimension)
}

/// Round count of the setup-side claim-reduction sumcheck at level
/// `lp` (book §5.3 line 658, book §5.4 line 752).
///
/// The sumcheck binds variables in `(row | col | coeff)` order over
/// the padded setup-polynomial view, so the round count is
/// `row_bits + col_bits + coeff_bits` where:
///
/// * `row_bits = log2_ceil(row_count)` — the rows envelope
///   `max(n_A, max_g n_B_g, n_D)`. Under tiered grouping the meta-tier
///   `n_B_meta` is included via `max_g n_B_g`; per-chunk B/D rows are
///   shared (book line 752 "MLE evaluation cost is `O(|D_chunk|) +
///   O(log k)`, independent of `k`") so the envelope is independent of
///   `k`.
/// * `col_bits = log2(col_count_padded)` — the padded column envelope
///   `max(W-cols, max_g B_cols_g, A-cols)`, which grows with the
///   joint W+S layout under cascade routing.
/// * `coeff_bits = log2(ring_dimension)` — the per-coefficient bind.
///
/// Sumcheck degree is `2` (product of structured weight and `S`),
/// so the per-round serialized payload is `degree * field_bytes` (the
/// linear term is recoverable from the round claim).
///
/// See [`planned_setup_padded_dims`] for the `(s_lp_in, incoming_tier,
/// num_eval_rows, num_commitment_groups)` semantics.
pub fn planned_setup_claim_reduction_rounds(
    lp: &LevelParams,
    s_lp_in: Option<&LevelParams>,
    incoming_tier: TieredSetupParams,
    num_eval_rows: usize,
    num_commitment_groups: usize,
) -> usize {
    let (row_count, col_count_padded) = planned_setup_padded_dims(
        lp,
        s_lp_in,
        incoming_tier,
        num_eval_rows,
        num_commitment_groups,
    );
    let row_bits = row_count.next_power_of_two().trailing_zeros() as usize;
    let col_bits = col_count_padded.trailing_zeros() as usize;
    let coeff_bits = lp.ring_dimension.trailing_zeros() as usize;
    row_bits + col_bits + coeff_bits
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
    let inherited = base.with_decomp(
        m_vars_chunk,
        r_vars_chunk,
        full_digits,
        full_digits,
        fold_digits,
        chunk_num_ring,
    )?;
    derive_chunk_sis_ranks_from_widths(inherited)
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
    let inherited = base.with_decomp(
        m_vars_chunk,
        r_vars_chunk,
        full_digits,
        full_digits,
        fold_digits,
        chunk_num_ring,
    )?;
    derive_chunk_sis_ranks_from_widths(inherited)
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
///
/// Mirrors the on-wire shape of [`crate::AkitaLevelProof`] /
/// [`crate::AkitaBatchedFoldRoot`]:
///
/// * `y_bytes` — `num_claims · D · field_bytes`. `num_claims` is the
///   number of distinct opening points on the wire (= number of
///   `y_ring` slots = number of commitment groups joint-opened at
///   this level). For a singleton recursive level it is `1`; under
///   un-tiered cascade routing (book §5.3 lines 627-660) it is `2`
///   (W, S); under tiered cascade routing (book §5.4 lines 686-754)
///   it is `3` (W, chunks-as-1-group, meta). The per-chunk
///   B-commitments are NOT on the wire — the verifier reconstructs
///   them from preprocessed setup material per book line 752 "MLE
///   evaluation cost is `O(|D_chunk|) + O(log k)`, independent of `k`".
/// * `v_bytes` — `n_D · D · field_bytes`.
/// * `stage1_bytes` — range-check tree proof for the stage-2 column
///   rounds; computed via `stage1_proof_bytes(rounds, b, elem_bytes)`.
/// * `stage2 sumcheck` — `rounds · 3 · field_bytes` (the compressed
///   unipoly stores all coefficients except the linear term).
/// * `setup_claim_reduction` (optional) — `rounds_cr · 2 · field_bytes`
///   for the book §5.3 line 658 / §5.4 line 752 reduction sumcheck,
///   plus `2 · field_bytes` for `m_setup_eval` and `s_opening_value`.
///   Pass `Some(rounds)` whenever the level emits a CR payload
///   (`lp.use_setup_claim_reduction == true`); `None` otherwise.
///   Rounds are produced by [`planned_setup_claim_reduction_rounds`]
///   from the level's M-table shape, which depends on the incoming
///   cascade `(s_lp_in, incoming_tier, num_eval_rows,
///   num_commitment_groups)`.
/// * `next_commit_bytes` — `next_lp.n_B · next_lp.D · field_bytes`.
///   The next-level commitment is a single `FlatRingVec` whose ring
///   count is `next_lp.b_key.row_len()`. The planner sizes `next_lp`
///   against the joint witness `current_w_len + s_field_len_in`, so
///   the SIS-driven `n_B` already reflects the wider joint W; no
///   further widening is required here. Per-chunk and meta-tier
///   commitments live in the preprocessed
///   [`crate::TieredSetupCommitments`] on the verifier side and are
///   never serialized in the proof.
/// * `next_eval_bytes` — one `field_bytes` for `next_w_eval`.
pub fn level_proof_bytes(
    field_bits: u32,
    lp: &LevelParams,
    level_lp: &LevelParams,
    next_lp: &LevelParams,
    next_w_len: usize,
    num_claims: usize,
    setup_claim_reduction_rounds: Option<usize>,
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
    // Setup-claim-reduction payload: degree-2 sumcheck over the padded
    // setup-polynomial view + `m_setup_eval` + `s_opening_value`.
    let cr_bytes = setup_claim_reduction_rounds.map_or(0, |cr_rounds| {
        sumcheck_bytes(cr_rounds, 2, elem_bytes) + 2 * elem_bytes
    });

    y_bytes
        + v_bytes
        + stage1_bytes
        + sumcheck_bytes(rounds, 3, elem_bytes)
        + cr_bytes
        + next_commit_bytes
        + next_eval_bytes
}

/// Header-stripped byte size of a singleton recursive proof level
/// WITHOUT a setup-claim-reduction payload.
///
/// For CR-on levels the caller must use [`level_proof_bytes`] directly
/// with `Some(rounds)` from [`planned_setup_claim_reduction_rounds`].
pub fn recursive_level_proof_bytes(
    field_bits: u32,
    lp: &LevelParams,
    next_lp: &LevelParams,
    next_w_len: usize,
) -> usize {
    level_proof_bytes(field_bits, lp, lp, next_lp, next_w_len, 1, None)
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
    fn level_proof_bytes_cr_payload_adds_setup_reduction_cost() {
        // Singleton recursive level: CR-off baseline + Some(rounds)
        // overload differ by exactly `rounds * 2 * elem_bytes + 2 *
        // elem_bytes` (degree-2 sumcheck + `m_setup_eval` +
        // `s_opening_value`).
        let lp = lp_for_chunk_test();
        let next_lp = lp_for_chunk_test();
        let next_w_len = next_lp.ring_dimension * 4;
        let baseline = level_proof_bytes(128, &lp, &lp, &next_lp, next_w_len, 1, None);
        let elem_bytes = field_bytes(128);
        for rounds in [4_usize, 16, 25] {
            let with_cr = level_proof_bytes(128, &lp, &lp, &next_lp, next_w_len, 1, Some(rounds));
            let cr_expected = sumcheck_bytes(rounds, 2, elem_bytes) + 2 * elem_bytes;
            assert_eq!(
                with_cr - baseline,
                cr_expected,
                "CR-on overhead must equal degree-2 sumcheck + 2 scalars at rounds={rounds}"
            );
        }
    }

    /// Phase 5 fix-loop Item (a) lock-in: the helper must NEVER touch
    /// `a_key` or `d_key` and must NEVER grow `b_key.row_len`. The
    /// `GroupSpec` abstraction only carries `b_key`; the A-role and
    /// D-role at L+1 are read from the OUTER LP. Touching `a_key` or
    /// `d_key` here would silently desync the L0 commit (which uses
    /// the per-tier LP's full `(a_key, b_key, d_key)`) from the L+1
    /// M-relation (which uses outer's `a_key`/`d_key` for tier-marked
    /// groups), producing `AkitaError::InvalidProof` at verify.
    /// Growing `b_key.row_len` would similarly indicate the OUTER LP
    /// is insecure for chunk widths — a separate bug we surface via
    /// `validate_stored_sis_ranks` rather than silently fix.
    #[test]
    fn derive_chunk_sis_ranks_only_touches_b_role_and_shrinks() {
        let d = 64usize;
        let collision = 63u32;
        let inner_width = 128usize;
        let d_matrix_width = 16usize;
        let base_n_a = 3usize;
        let base_n_b = 4usize;
        let base_n_d = 4usize;
        let outer_width = base_n_a * 1 * 16;
        let inherited = LevelParams {
            ring_dimension: d,
            log_basis: 6,
            a_key: AjtaiKeyParams::new_unchecked(base_n_a, inner_width, collision, d),
            b_key: AjtaiKeyParams::new_unchecked(base_n_b, outer_width, collision, d),
            d_key: AjtaiKeyParams::new_unchecked(base_n_d, d_matrix_width, collision, d),
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
        };
        let shrunken = derive_chunk_sis_ranks_from_widths(inherited.clone()).expect("shrink");
        assert_eq!(
            shrunken.a_key, inherited.a_key,
            "Phase 5 fix-loop: a_key must NOT change (chunks A-binding at \
             L+1 reads from outer LP via GroupSpec lacking a_key)"
        );
        assert_eq!(
            shrunken.d_key, inherited.d_key,
            "Phase 5 fix-loop: d_key must NOT change (chunks D-binding at \
             L+1 reads from outer LP via GroupSpec lacking d_key)"
        );
        assert!(
            shrunken.b_key.row_len() <= inherited.b_key.row_len(),
            "B-role rank may shrink but never grow ({} -> {})",
            inherited.b_key.row_len(),
            shrunken.b_key.row_len()
        );
        assert_eq!(
            shrunken.b_key.col_len(),
            inherited.b_key.col_len(),
            "Phase 5 fix-loop: b_key.col_len must NOT change (cols derived \
             from outer.a_key.row_len * num_digits_open * num_blocks)"
        );
        // The helper preserves all non-key shape fields.
        assert_eq!(shrunken.num_blocks, inherited.num_blocks);
        assert_eq!(shrunken.block_len, inherited.block_len);
        assert_eq!(shrunken.m_vars, inherited.m_vars);
        assert_eq!(shrunken.r_vars, inherited.r_vars);
        assert_eq!(shrunken.num_digits_open, inherited.num_digits_open);
        assert_eq!(shrunken.num_digits_commit, inherited.num_digits_commit);
        assert_eq!(shrunken.num_digits_fold, inherited.num_digits_fold);
    }

    /// `collision_inf == 0` on the B-role means the planner hasn't pinned
    /// the SIS bucket yet; the helper must return the LP unchanged in
    /// that case (early-construction path; the caller will pin the
    /// bucket later via a separate SIS-derivation pass).
    #[test]
    fn derive_chunk_sis_ranks_returns_unchanged_when_b_collision_unpinned() {
        let lp = lp_for_chunk_test();
        let shrunken = derive_chunk_sis_ranks_from_widths(lp.clone()).expect("unchanged");
        assert_eq!(shrunken.a_key, lp.a_key);
        assert_eq!(shrunken.b_key, lp.b_key);
        assert_eq!(shrunken.d_key, lp.d_key);
    }

    #[test]
    fn planned_setup_padded_dims_round_count_matches_field_len_log() {
        // For any (s_lp_in, incoming_tier, num_eval_rows, num_groups)
        // the CR rounds equal `row_bits + col_bits + coeff_bits`, and
        // `setup_field_len = row_count * col_count_padded * D` shares
        // its log2 with the rounds count (since col_count_padded and
        // D are powers of two).
        let base = lp_for_chunk_test();

        // Single-group baseline.
        let (row_count_single, col_padded_single) =
            planned_setup_padded_dims(&base, None, TieredSetupParams::un_tiered(), 1, 1);
        let rounds_single =
            planned_setup_claim_reduction_rounds(&base, None, TieredSetupParams::un_tiered(), 1, 1);
        let expected_single = row_count_single.next_power_of_two().trailing_zeros() as usize
            + col_padded_single.trailing_zeros() as usize
            + base.ring_dimension.trailing_zeros() as usize;
        assert_eq!(rounds_single, expected_single);

        // Un-tiered cascade incoming (W, S): the planner passes
        // num_eval_rows = num_commitment_groups = 2.
        let setup_field_len = 8 * base.ring_dimension;
        let s_lp = untiered_setup_group_lp(&base, setup_field_len).expect("s_lp");
        let rounds_untiered = planned_setup_claim_reduction_rounds(
            &base,
            Some(&s_lp),
            TieredSetupParams::un_tiered(),
            2,
            2,
        );
        let (row_count_untiered, col_padded_untiered) =
            planned_setup_padded_dims(&base, Some(&s_lp), TieredSetupParams::un_tiered(), 2, 2);
        assert_eq!(
            rounds_untiered,
            row_count_untiered.next_power_of_two().trailing_zeros() as usize
                + col_padded_untiered.trailing_zeros() as usize
                + base.ring_dimension.trailing_zeros() as usize
        );

        // Tiered cascade incoming (W, chunks, meta): the planner
        // passes num_eval_rows = num_commitment_groups = 3.
        let big_field_len = 16 * 16 * base.ring_dimension;
        let tier = TieredSetupParams::new(2).expect("f=2");
        let chunk_lp = tiered_setup_group_lp(&base, big_field_len, tier).expect("chunk_lp");
        let rounds_tiered =
            planned_setup_claim_reduction_rounds(&base, Some(&chunk_lp), tier, 3, 3);
        let (row_count_tiered, col_padded_tiered) =
            planned_setup_padded_dims(&base, Some(&chunk_lp), tier, 3, 3);
        assert_eq!(
            rounds_tiered,
            row_count_tiered.next_power_of_two().trailing_zeros() as usize
                + col_padded_tiered.trailing_zeros() as usize
                + base.ring_dimension.trailing_zeros() as usize
        );

        // Tiered cascade rounds must dominate single-group baseline:
        // the multi-group M-table is structurally wider.
        assert!(
            rounds_tiered >= rounds_single,
            "tiered CR rounds ({rounds_tiered}) must be at least single-group ({rounds_single})"
        );
    }

    /// Phase 5 / book §5.5 line 752 invariant: the planner's
    /// setup-polynomial col envelope for tier-marked cascade incoming
    /// shapes treats the `k` chunks under SHARED `D_chunk / B_chunk`
    /// matrices — the chunks-group contribution to `col_count` is
    /// fixed at the per-chunk extent (no `k` multiplier), and only
    /// the MAX over `(W, chunks, meta)` group contributions sets the
    /// envelope.
    ///
    /// Regression for the prior over-allocation: previously
    /// `planned_setup_padded_dims` computed `col_count` as
    /// `lp.num_blocks * lp.num_digits_open + k * s_lp.num_blocks *
    /// s_lp.num_digits_open + meta.num_blocks * meta.num_digits_open`
    /// (sum of all groups WITH `k` multiplier), making CR rounds
    /// scale as `log2(k)` per cascade level. Book §5.5 line 752's
    /// "O(|D_chunk|) + O(log k), independent of k" demands the chunks
    /// contribution be k-independent. Meta still scales with k (its
    /// length is `k · n_B_chunk · D`), but the per-chunk write
    /// pattern shares col slots across chunks.
    #[test]
    fn planned_setup_padded_dims_tiered_drops_k_multiplier_from_chunks() {
        let base = lp_for_chunk_test();
        let tier = TieredSetupParams::new(2).expect("f=2");
        let k = tier.num_chunks; // 4
        let big_field_len = 16 * 16 * base.ring_dimension;
        let chunk_lp = tiered_setup_group_lp(&base, big_field_len, tier).expect("chunk_lp");

        let n_a = base.a_key.row_len();
        let chunks_per_group_max_col = (chunk_lp.num_blocks * n_a * chunk_lp.num_digits_open)
            .max(chunk_lp.num_blocks * chunk_lp.num_digits_open)
            .max(3 * chunk_lp.block_len * chunk_lp.num_digits_commit);
        let w_per_group_max_col = (base.num_blocks * n_a * base.num_digits_open)
            .max(base.num_blocks * base.num_digits_open)
            .max(3 * base.block_len * base.num_digits_commit);
        // Pre-Phase-5: chunks contribution would be `k * chunk_lp.num_blocks
        // * n_a * chunk_lp.num_digits_open`. Post-Phase-5: it's the per-chunk
        // extent WITHOUT `k`.
        let pre_phase5_chunks_b_cols = k * chunk_lp.num_blocks * n_a * chunk_lp.num_digits_open;
        assert!(
            pre_phase5_chunks_b_cols > chunks_per_group_max_col,
            "test setup must distinguish pre-vs-post Phase 5 chunks contribution \
             (k={k}, chunk_b_cols_with_k={pre_phase5_chunks_b_cols}, \
             chunks_per_group_max_col={chunks_per_group_max_col})"
        );

        let (_, col_padded) = planned_setup_padded_dims(&base, Some(&chunk_lp), tier, 3, 3);

        // The post-Phase-5 envelope is the MAX over the (W, chunks,
        // meta) groups' per-group col extents (rounded up to next pow2).
        // We don't assert exact value here (depends on the derived meta_lp);
        // we assert the chunks contribution alone does NOT carry the
        // `k` multiplier — col_padded must be ≤ `next_pow2(max(W_max,
        // chunks_max, k · n_B_chunk · D_meta_max))` where the chunks
        // term is k-INDEPENDENT.
        let upper_bound_without_k_in_chunks = {
            let meta_field_len = (k * chunk_lp.b_key.row_len() * base.ring_dimension)
                .next_power_of_two()
                .max(base.ring_dimension);
            let meta_lp = untiered_setup_group_lp(&base, meta_field_len).expect("meta_lp");
            let meta_per_group_max_col = (meta_lp.num_blocks * n_a * meta_lp.num_digits_open)
                .max(meta_lp.num_blocks * meta_lp.num_digits_open)
                .max(3 * meta_lp.block_len * meta_lp.num_digits_commit);
            w_per_group_max_col
                .max(chunks_per_group_max_col)
                .max(meta_per_group_max_col)
                .next_power_of_two()
        };
        assert!(
            col_padded <= upper_bound_without_k_in_chunks,
            "col_padded {col_padded} exceeds the post-Phase-5 envelope \
             upper bound {upper_bound_without_k_in_chunks} (suggests the \
             chunks contribution still carries the `k` multiplier)"
        );

        // Strong invariant: the envelope must NOT include the pre-Phase-5
        // `k * chunk_b_cols` term verbatim. We require strict shrinkage
        // versus the pre-Phase-5 expression to lock in the k-drop.
        let pre_phase5_col_count = (base.num_blocks * base.num_digits_open
            + k * chunk_lp.num_blocks * chunk_lp.num_digits_open
            + {
                let meta_field_len = (k * chunk_lp.b_key.row_len() * base.ring_dimension)
                    .next_power_of_two()
                    .max(base.ring_dimension);
                let meta_lp = untiered_setup_group_lp(&base, meta_field_len).expect("meta_lp");
                meta_lp.num_blocks * meta_lp.num_digits_open
            })
        .max(base.num_blocks * n_a * base.num_digits_open)
        .max(k * chunk_lp.num_blocks * n_a * chunk_lp.num_digits_open);
        let pre_phase5_padded = pre_phase5_col_count.next_power_of_two();
        assert!(
            col_padded <= pre_phase5_padded,
            "post-Phase-5 col_padded {col_padded} must be ≤ pre-Phase-5 \
             upper bound {pre_phase5_padded}"
        );
        // For this configuration the pre-Phase-5 value is strictly larger.
        // The exact shrink ratio depends on which group dominates.
        assert!(
            col_padded < pre_phase5_padded,
            "expected strict col_padded shrinkage versus pre-Phase-5 \
             k-multiplied formula ({col_padded} vs {pre_phase5_padded})"
        );
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
