use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use jolt_field::{CanonicalField, ExtField, FieldCore, FromPrimitiveInt, Invertible};
use std::marker::PhantomData;

use crate::field_reduction::trace_open_folded_ring_mle_dot;
use crate::{gadget_row_scalars, lagrange_weights, BasisMode, FpExtEncoding};

use super::layout::TraceWeightLayout;

/// One scalar-block trace term for a contiguous range of logical trace blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceFieldBlockOpening<F: FieldCore, const D: usize> {
    pub block_offset: usize,
    pub block_weights: Vec<F>,
    pub inner_opening_ring: CyclotomicRing<F, D>,
}

/// One ring-valued trace term for a contiguous range of logical trace blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceRingBlockOpening<F: FieldCore, const D: usize> {
    pub block_offset: usize,
    pub block_rings: Vec<CyclotomicRing<F, D>>,
    pub packed_inner_point: CyclotomicRing<F, D>,
}

/// Opening weights consumed by [`eval_trace_weight_at_point`].
pub enum TraceOpeningAtPoint<'a, F: FieldCore, E: FieldCore, const D: usize> {
    /// `K = 1`: scalar block weights with one packed inner opening per term.
    Field {
        terms: &'a [TraceFieldBlockOpening<F, D>],
    },
    /// `K > 1`: embedded block rings and ψ-packed inner point.
    Ring {
        terms: &'a [TraceRingBlockOpening<F, D>],
        _ext: PhantomData<E>,
    },
}

fn lift_gadget_row<F, E>(gadget_scalars: &[F]) -> Vec<E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    gadget_scalars.iter().copied().map(E::lift_base).collect()
}

fn validate_eval_point(
    layout: &TraceWeightLayout,
    ring_point_len: usize,
    col_point_len: usize,
) -> Result<(), AkitaError> {
    if ring_point_len != layout.ring_bits || col_point_len != layout.col_bits {
        return Err(AkitaError::InvalidSize {
            expected: layout.col_bits + layout.ring_bits,
            actual: col_point_len + ring_point_len,
        });
    }
    layout.validate_opening_digit_segment()
}

#[inline]
fn eq_weight_at_index<E: FieldCore>(point: &[E], index: usize) -> E {
    let mut weight = E::one();
    for (bit, &coord) in point.iter().enumerate() {
        if ((index >> bit) & 1) == 1 {
            weight *= coord;
        } else {
            weight *= E::one() - coord;
        }
    }
    weight
}

/// Evaluate the trace-weight MLE at `(ring_point, col_point)`.
///
/// `K` must match the claim-field extension degree (`1` for base-field claims,
/// `2`/`4`/`8` for ring-subfield extension claims). The compiler monomorphizes
/// the `K = 1` tensor path separately from the extension trace path.
pub fn eval_trace_weight_at_point<F, E, const D: usize, const K: usize>(
    layout: &TraceWeightLayout,
    ring_point: &[E],
    col_point: &[E],
    opening: TraceOpeningAtPoint<'_, F, E, D>,
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt + FieldCore,
{
    match opening {
        TraceOpeningAtPoint::Field { terms } => {
            if K != 1 {
                return Err(AkitaError::InvalidInput(
                    "field opening weights require K = 1".to_string(),
                ));
            }
            eval_at_point_k1::<F, E, D>(layout, ring_point, col_point, terms)
        }
        TraceOpeningAtPoint::Ring { terms, .. } => {
            if K == 1 {
                return Err(AkitaError::InvalidInput(
                    "ring opening weights require K > 1".to_string(),
                ));
            }
            eval_at_point_k_extension::<F, E, D>(layout, ring_point, col_point, terms)
        }
    }
}

#[inline]
fn eval_at_point_k1<F, E, const D: usize>(
    layout: &TraceWeightLayout,
    ring_point: &[E],
    col_point: &[E],
    terms: &[TraceFieldBlockOpening<F, D>],
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F> + FromPrimitiveInt + FieldCore,
{
    if terms.is_empty() {
        return Err(AkitaError::InvalidInput(
            "field opening terms must be non-empty".to_string(),
        ));
    }
    validate_eval_point(layout, ring_point.len(), col_point.len())?;
    let gadget_scalars = gadget_row_scalars::<F>(layout.num_digits_open, layout.log_basis);
    let gadget_row = lift_gadget_row::<F, E>(&gadget_scalars);
    let ring_eq = lagrange_weights(ring_point)?;
    let mut out = E::zero();

    for term in terms {
        layout.validate_trace_term_block_range(term.block_offset, term.block_weights.len())?;
        let mut col_factor = E::zero();
        for (local_block, &block_weight) in term.block_weights.iter().enumerate() {
            let block = term.block_offset + local_block;
            for (plane, &gadget) in gadget_row.iter().enumerate() {
                let col = layout.opening_digit_col_index(block, plane);
                col_factor +=
                    eq_weight_at_index(col_point, col) * E::lift_base(block_weight) * gadget;
            }
        }
        let inner_factor = term
            .inner_opening_ring
            .coefficients()
            .iter()
            .zip(ring_eq.iter())
            .fold(E::zero(), |acc, (&coeff, &weight)| {
                acc + E::lift_base(coeff) * weight
            });
        out += col_factor * inner_factor;
    }

    Ok(out)
}

fn eval_at_point_k_extension<F, E, const D: usize>(
    layout: &TraceWeightLayout,
    ring_point: &[E],
    col_point: &[E],
    terms: &[TraceRingBlockOpening<F, D>],
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt + FieldCore,
{
    if terms.is_empty() {
        return Err(AkitaError::InvalidInput(
            "ring opening terms must be non-empty".to_string(),
        ));
    }
    validate_eval_point(layout, ring_point.len(), col_point.len())?;
    let gadget_scalars = gadget_row_scalars::<F>(layout.num_digits_open, layout.log_basis);
    let gadget_row = lift_gadget_row::<F, E>(&gadget_scalars);

    let ring_eq = lagrange_weights(ring_point)?;
    let mut out = E::zero();
    for term in terms {
        layout.validate_trace_term_block_range(term.block_offset, term.block_rings.len())?;
        // The trace-open pipeline is E-linear in the fold-block ring, so fold
        // every block of this term into one ring element first (weighted by its
        // column factor) and take a single `Tr_H` of one ring product, instead
        // of one ring trace per fold block.
        let mut folded = CyclotomicRing::<E, D>::zero();
        for (local_block, block_ring) in term.block_rings.iter().enumerate() {
            let block = term.block_offset + local_block;
            let mut col_factor = E::zero();
            for (plane, &gadget) in gadget_row.iter().enumerate() {
                let col = layout.opening_digit_col_index(block, plane);
                col_factor += eq_weight_at_index(col_point, col) * gadget;
            }
            if col_factor.is_zero() {
                continue;
            }
            let folded_coeffs = folded.coefficients_mut();
            for (coeff, &base) in folded_coeffs
                .iter_mut()
                .zip(block_ring.coefficients().iter())
            {
                *coeff += col_factor * E::lift_base(base);
            }
        }
        out += trace_open_folded_ring_mle_dot::<F, E, D>(
            &folded,
            &ring_eq,
            &term.packed_inner_point,
            layout.ring_bits,
        )?;
    }
    Ok(out)
}

/// One closed-form trace term for one claim opening over a contiguous block run.
///
/// Unlike [`TraceFieldBlockOpening`]/[`TraceRingBlockOpening`] (which the prover
/// expands into a dense table), this carries only the short opening data the
/// verifier already holds:
///
/// - `b_open`: the block-axis opening coordinates (length `r_pc`, in the
///   evaluation field `E`). The fold-block weight for block `j` is the basis
///   weight `∏_t w_t(j_t)` of `b_open` (`w_t(0), w_t(1) = (1 − b_t, b_t)` for
///   Lagrange, `(1, b_t)` for Monomial), so `b_open` plus `basis` reconstruct
///   every block multiplier.
/// - `basis`: the opening basis that fixes those per-bit weights.
/// - `packed_inner_point`: the ψ-packed inner opening over `F`.
/// - `block_offset`: where this claim's `2^{r_pc}` blocks start inside the
///   (possibly claim-batched) block row.
/// - `coefficient`: the public scalar applied to this term (per-claim row
///   weight times any end-of-round tensor factor), in `E`.
///
/// [`eval_trace_terms_closed`] evaluates the fused trace MLE from these in
/// `O(num_digits_open · (r_pc · K³ + col_bits) + D² / K)` per term, i.e. one
/// `Tr_H` per claim, with no dependence on the number of fold blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceTerm<F: FieldCore, E: FieldCore, const D: usize> {
    pub block_offset: usize,
    pub b_open: Vec<E>,
    pub basis: BasisMode,
    pub packed_inner_point: CyclotomicRing<F, D>,
    pub coefficient: E,
}

/// Structure constants of the ring-subfield basis of `E` over `F`.
///
/// `gamma[i][j][m]` is the `m`-th ring-subfield coordinate of `beta_i · beta_j`,
/// where `beta_i` is the basis element with coordinate vector `e_i`. Because the
/// `FpExtEncoding` types use the same coordinates for `from_base_slice`,
/// `to_base_vec`, and `to_ext_coords`, this is exactly the
/// multiplication table the `psi`-embedding respects, so folding block weights in
/// this `K`-dimensional coordinate algebra agrees with folding the embedded ring
/// elements in `R_q` (but costs `O(K²)` instead of `O(D²)`).
fn ring_subfield_struct_consts<F, E>(k: usize) -> Vec<Vec<Vec<F>>>
where
    F: FieldCore,
    E: FpExtEncoding<F> + ExtField<F> + FieldCore,
{
    let mut gamma = vec![vec![vec![F::zero(); k]; k]; k];
    let mut unit = vec![F::zero(); k];
    for i in 0..k {
        unit[i] = F::one();
        let beta_i = E::from_base_slice(&unit);
        unit[i] = F::zero();
        for j in 0..k {
            unit[j] = F::one();
            let beta_j = E::from_base_slice(&unit);
            unit[j] = F::zero();
            let prod = (beta_i * beta_j).to_ext_coords();
            for (m, slot) in gamma[i][j].iter_mut().enumerate() {
                *slot = prod[m];
            }
        }
    }
    gamma
}

/// Multiply two ring-subfield coordinate vectors in the `E`-coefficient algebra
/// `E ⊗_F E` using the structure constants `gamma`.
fn cstar_mul<F, E>(u: &[E], w: &[E], gamma: &[Vec<Vec<F>>]) -> Vec<E>
where
    F: FieldCore,
    E: ExtField<F> + FieldCore,
{
    let k = u.len();
    let mut out = vec![E::zero(); k];
    for (i, &ui) in u.iter().enumerate() {
        if ui.is_zero() {
            continue;
        }
        for (j, &wj) in w.iter().enumerate() {
            if wj.is_zero() {
                continue;
            }
            let uw = ui * wj;
            for (m, slot) in out.iter_mut().enumerate() {
                let g = gamma[i][j][m];
                if !g.is_zero() {
                    *slot += uw * E::lift_base(g);
                }
            }
        }
    }
    out
}

/// Lifted ring-subfield coordinates of the two per-bit block weights `w_t(0)`
/// and `w_t(1)` for opening coordinate `b` under `basis`.
fn block_weight_coords<F, E>(b: E, basis: BasisMode, k: usize) -> (Vec<E>, Vec<E>)
where
    F: FieldCore,
    E: FpExtEncoding<F> + ExtField<F> + FieldCore,
{
    let (w0, w1) = match basis {
        BasisMode::Lagrange => (E::one() - b, b),
        BasisMode::Monomial => (E::one(), b),
    };
    let lift = |w: E| -> Vec<E> {
        let mut coords: Vec<E> = w.to_ext_coords().into_iter().map(E::lift_base).collect();
        coords.resize(k, E::zero());
        coords
    };
    (lift(w0), lift(w1))
}

/// Fold the block weights over one opening-digit plane into ring-subfield
/// coordinates: returns `Σ_j eq(col_point, off + j) · coords(b_j)` for
/// `j ∈ [0, 2^{r_pc})`, where `b_j = ∏_t w_t(j_t)` is the basis weight and
/// `coords` are the ring-subfield coordinates the ψ-embedding respects.
///
/// The column index `off + j` mixes the block index `j` with the plane offset
/// `off`, so the low `r_pc` bits can carry into the high bits. A two-state
/// carry DP over the block bits handles this in `O(r_pc · K³)`: state `S_c`
/// accumulates `(∏ eq-bit) ⊛ (⊛ coords)` for paths whose low bits carry `c`
/// into bit `r_pc`; the high bits then contribute `eq(col_high, off≫r_pc + c)`.
#[allow(clippy::too_many_arguments)]
fn fold_blocks_for_plane<F, E>(
    col_low: &[E],
    col_high: &[E],
    bit_coords: &[(Vec<E>, Vec<E>)],
    off: usize,
    k: usize,
    gamma: &[Vec<Vec<F>>],
) -> Vec<E>
where
    F: FieldCore,
    E: ExtField<F> + FieldCore,
{
    let r_pc = col_low.len();
    let off_low = off & ((1usize << r_pc) - 1);
    let off_high = off >> r_pc;

    // `state[c]` holds the running `E^K` accumulator for carry `c`. The empty
    // product is the coordinate-algebra identity (`unit_0`) with eq weight 1.
    let mut state = [vec![E::zero(); k], vec![E::zero(); k]];
    state[0][0] = E::one();
    for (t, (cw0, cw1)) in bit_coords.iter().enumerate() {
        let off_bit = (off_low >> t) & 1;
        let r = col_low[t];
        let one_minus_r = E::one() - r;
        let mut next = [vec![E::zero(); k], vec![E::zero(); k]];
        for (carry_in, state_row) in state.iter().enumerate() {
            if state_row.iter().all(|value| value.is_zero()) {
                continue;
            }
            for (j_bit, cw) in [(0usize, cw0), (1usize, cw1)] {
                let sum = off_bit + j_bit + carry_in;
                let result_bit = sum & 1;
                let carry_out = sum >> 1;
                let eq_bit = if result_bit == 1 { r } else { one_minus_r };
                let folded = cstar_mul::<F, E>(state_row, cw, gamma);
                for (slot, value) in next[carry_out].iter_mut().zip(folded) {
                    *slot += eq_bit * value;
                }
            }
        }
        state = next;
    }

    let eq_high_0 = eq_eval_at_index(col_high, off_high);
    let eq_high_1 = eq_eval_at_index(col_high, off_high + 1);
    (0..k)
        .map(|m| state[0][m] * eq_high_0 + state[1][m] * eq_high_1)
        .collect()
}

/// Ring-subfield coordinates of the product of high-bit block weights selecting
/// chunk `chunk`: `⊛_{s} coords(w_s(chunk_s))` over the high block bits `b_high`.
/// For `k = 1` this is the scalar `∏_s w_s(chunk_s)`; the empty product (no high
/// bits) is the coordinate-algebra identity.
fn high_chunk_weight_coords<F, E>(
    b_high: &[E],
    chunk: usize,
    basis: BasisMode,
    k: usize,
    gamma: &[Vec<Vec<F>>],
) -> Vec<E>
where
    F: FieldCore,
    E: FpExtEncoding<F> + ExtField<F> + FieldCore,
{
    let mut acc = vec![E::zero(); k];
    acc[0] = E::one();
    for (s, &b) in b_high.iter().enumerate() {
        let (cw0, cw1) = block_weight_coords::<F, E>(b, basis, k);
        let bit = (chunk >> s) & 1;
        let cw = if bit == 1 { &cw1 } else { &cw0 };
        acc = cstar_mul::<F, E>(&acc, cw, gamma);
    }
    acc
}

/// Evaluate the fused trace-weight MLE at `(ring_point, col_point)` in closed
/// form from short per-claim opening data.
///
/// This is the verifier-side evaluator. It produces the same value as the dense
/// trace-weight table built by the prover (see `build_trace_weight_table_*`),
/// but contracts the block axis analytically: per term it folds the block
/// weights into ring-subfield coordinates `V`, embeds them into one ψ-packed
/// ring `B_blk = embed(V)`, and takes a single `Tr_H`, instead of one trace per
/// fold block. See [`TraceTerm`] for the cost.
pub fn eval_trace_terms_closed<F, E, const D: usize>(
    layout: &TraceWeightLayout,
    ring_point: &[E],
    col_point: &[E],
    terms: &[TraceTerm<F, E, D>],
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt + FieldCore,
{
    if terms.is_empty() {
        return Err(AkitaError::InvalidInput(
            "trace terms must be non-empty".to_string(),
        ));
    }
    validate_eval_point(layout, ring_point.len(), col_point.len())?;

    let k = E::EXT_DEGREE;
    let gadget_scalars = gadget_row_scalars::<F>(layout.num_digits_open, layout.log_basis);
    let gadget_row = lift_gadget_row::<F, E>(&gadget_scalars);
    let ring_eq = lagrange_weights(ring_point)?;
    // Structure constants of the ring-subfield basis (`[[[1]]]` when `k == 1`).
    let gamma = ring_subfield_struct_consts::<F, E>(k);

    let mut out = E::zero();
    for term in terms {
        let r_pc = term.b_open.len();
        if r_pc > layout.col_bits {
            return Err(AkitaError::InvalidInput(
                "trace term block-opening width exceeds column dimension".to_string(),
            ));
        }
        let block_span = 1usize.checked_shl(r_pc as u32).ok_or_else(|| {
            AkitaError::InvalidInput("trace term block span overflow".to_string())
        })?;
        layout.validate_trace_term_block_range(term.block_offset, block_span)?;

        // `V = Σ_plane gadget[plane] · Σ_j eq(col, col(j, plane)) · coords(b_j)`.
        // Single-chunk: blocks map to contiguous columns `base + plane·num_blocks
        // + j`. Multi-chunk: the block axis splits into a chunk (high bits) and a
        // block-local window (low bits) at distinct chunk offsets, so we fold the
        // low bits per chunk and weight each chunk by its high-bit block weight.
        let mut v = vec![E::zero(); k];
        if layout.chunk.num_chunks <= 1 {
            let base = layout
                .opening_digit_offset
                .checked_add(term.block_offset)
                .ok_or_else(|| {
                    AkitaError::InvalidInput("trace term column base overflow".to_string())
                })?;
            let col_low = &col_point[..r_pc];
            let col_high = &col_point[r_pc..];
            let bit_coords: Vec<(Vec<E>, Vec<E>)> = term
                .b_open
                .iter()
                .map(|&b| block_weight_coords::<F, E>(b, term.basis, k))
                .collect();
            for (plane, &gadget) in gadget_row.iter().enumerate() {
                let off = plane
                    .checked_mul(layout.num_blocks)
                    .and_then(|plane_offset| plane_offset.checked_add(base))
                    .ok_or_else(|| {
                        AkitaError::InvalidInput("trace term column index overflow".to_string())
                    })?;
                let plane_fold =
                    fold_blocks_for_plane::<F, E>(col_low, col_high, &bit_coords, off, k, &gamma);
                for (slot, value) in v.iter_mut().zip(plane_fold) {
                    *slot += gadget * value;
                }
            }
        } else {
            let c = &layout.chunk;
            let rb_low = c.blocks_per_chunk.trailing_zeros() as usize;
            let rb_high = c.num_chunks.trailing_zeros() as usize;
            if rb_low + rb_high != r_pc {
                return Err(AkitaError::InvalidInput(
                    "trace term block bits do not match chunked block axis".to_string(),
                ));
            }
            let claim = term
                .block_offset
                .checked_div(c.num_blocks_global)
                .unwrap_or(0);
            let plane_stride = c.num_claims * c.blocks_per_chunk;
            let claim_base = claim * c.blocks_per_chunk;
            let b_low = &term.b_open[..rb_low];
            let b_high = &term.b_open[rb_low..];
            let bit_coords: Vec<(Vec<E>, Vec<E>)> = b_low
                .iter()
                .map(|&b| block_weight_coords::<F, E>(b, term.basis, k))
                .collect();
            let col_low = &col_point[..rb_low];
            let col_high = &col_point[rb_low..];
            for chunk in 0..c.num_chunks {
                let high_w = high_chunk_weight_coords::<F, E>(b_high, chunk, term.basis, k, &gamma);
                if high_w.iter().all(|x| x.is_zero()) {
                    continue;
                }
                let chunk_offset_e = chunk
                    .checked_mul(c.chunk_stride)
                    .and_then(|o| o.checked_add(layout.opening_digit_offset))
                    .and_then(|o| o.checked_add(claim_base))
                    .ok_or_else(|| {
                        AkitaError::InvalidInput("chunked trace column base overflow".to_string())
                    })?;
                let mut v_chunk = vec![E::zero(); k];
                for (plane, &gadget) in gadget_row.iter().enumerate() {
                    let off = plane
                        .checked_mul(plane_stride)
                        .and_then(|p| p.checked_add(chunk_offset_e))
                        .ok_or_else(|| {
                            AkitaError::InvalidInput(
                                "chunked trace column index overflow".to_string(),
                            )
                        })?;
                    let plane_fold = fold_blocks_for_plane::<F, E>(
                        col_low,
                        col_high,
                        &bit_coords,
                        off,
                        k,
                        &gamma,
                    );
                    for (slot, value) in v_chunk.iter_mut().zip(plane_fold) {
                        *slot += gadget * value;
                    }
                }
                // Weight this chunk by its high-bit block weight (coordinate-algebra
                // product), then accumulate into the term's folded `V`.
                let combined = cstar_mul::<F, E>(&high_w, &v_chunk, &gamma);
                for (slot, value) in v.iter_mut().zip(combined) {
                    *slot += value;
                }
            }
        }

        let trace_factor = if k == 1 {
            // Scalar block weights pull straight through the trace, which
            // collapses to `<ring_eq, packed_inner>`.
            let inner = ring_eq
                .iter()
                .zip(term.packed_inner_point.coefficients().iter())
                .fold(E::zero(), |acc, (&w, &c)| acc + w * E::lift_base(c));
            v[0] * inner
        } else {
            // `B_blk = embed_E(V)`: place the K coordinates at the ψ positions.
            let step = D / (2 * k);
            let mut coeffs = [E::zero(); D];
            coeffs[0] = v[0];
            for (i, &vi) in v.iter().enumerate().skip(1) {
                coeffs[i * step] = vi;
                coeffs[D - i * step] = -vi;
            }
            let b_blk = CyclotomicRing::<E, D>::from_coefficients(coeffs);
            trace_open_folded_ring_mle_dot::<F, E, D>(
                &b_blk,
                &ring_eq,
                &term.packed_inner_point,
                layout.ring_bits,
            )?
        };

        out += term.coefficient * trace_factor;
    }
    Ok(out)
}
