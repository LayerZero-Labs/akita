use akita_algebra::offset_eq::{eval_affine_digit_interval, AffineWeight};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, Invertible};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::field_reduction::trace_open_folded_ring_mle_dot;
use crate::{gadget_row_scalars, lagrange_weights, BasisMode, FpExtEncoding};

use super::layout::TraceWeightLayout;

/// One scalar-block trace term for a contiguous range of logical trace blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceFieldBlockOpening<F: FieldCore, const D: usize> {
    pub block_offset: usize,
    pub live_block_weights: Vec<F>,
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
        layout.validate_trace_term_block_range(term.block_offset, term.live_block_weights.len())?;
        let mut col_factor = E::zero();
        for (local_block, &block_weight) in term.live_block_weights.iter().enumerate() {
            let block = term.block_offset + local_block;
            for (plane, &gadget) in gadget_row.iter().enumerate() {
                let col = layout.opening_digit_col_index(block, plane)?;
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
                let col = layout.opening_digit_col_index(block, plane)?;
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
/// - `block_offset`: where this claim's exact live block run starts inside the
///   (possibly claim-batched) block row. `b_open` addresses the enclosing
///   power-of-two capacity; the layout supplies the exact live run length.
/// - `coefficient`: the public scalar applied to this term (per-claim row
///   weight times any end-of-round tensor factor), in `E`.
///
/// [`eval_trace_terms_closed`] evaluates the fused trace MLE from balanced
/// high/low factors in `O((H + Q) · poly(K, num_digits_open, col_bits) + D²/K)`
/// per term, where `H · Q` is the block-index domain size. It performs one `Tr_H` per
/// claim and never materializes or enumerates the Cartesian block domain.
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

#[derive(Clone)]
struct TraceAffineWeight<F: FieldCore, E: FieldCore> {
    coordinates: Vec<E>,
    gamma: Arc<Vec<Vec<Vec<F>>>>,
}

impl<F, E> AffineWeight<E> for TraceAffineWeight<F, E>
where
    F: FieldCore,
    E: ExtField<F> + FieldCore,
{
    fn zero_like(&self) -> Self {
        Self {
            coordinates: vec![E::zero(); self.coordinates.len()],
            gamma: Arc::clone(&self.gamma),
        }
    }

    fn add_scaled(&mut self, factor: &Self, scale: E) {
        for (slot, &value) in self.coordinates.iter_mut().zip(&factor.coordinates) {
            *slot += value * scale;
        }
    }

    fn multiply(&self, rhs: &Self) -> Self {
        Self {
            coordinates: cstar_mul::<F, E>(&self.coordinates, &rhs.coordinates, &self.gamma),
            gamma: Arc::clone(&self.gamma),
        }
    }
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

/// Ring-subfield coordinates of one block's basis weight.
fn block_weight_at_index_coords<F, E>(
    block_point: &[E],
    block: usize,
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
    for (s, &b) in block_point.iter().enumerate() {
        let (cw0, cw1) = block_weight_coords::<F, E>(b, basis, k);
        let bit = (block >> s) & 1;
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
    let gamma = Arc::new(ring_subfield_struct_consts::<F, E>(k));

    let mut out = E::zero();
    for term in terms {
        let r_pc = term.b_open.len();
        if r_pc > layout.col_bits {
            return Err(AkitaError::InvalidInput(
                "trace term block-opening width exceeds column dimension".to_string(),
            ));
        }
        let block_index_domain_size = 1usize.checked_shl(r_pc as u32).ok_or_else(|| {
            AkitaError::InvalidInput("trace term block span overflow".to_string())
        })?;
        let block_span = layout
            .witness_layout
            .group_live_block_count(layout.group_id)?;
        if block_span > block_index_domain_size {
            return Err(AkitaError::InvalidInput(
                "trace term live_block_count exceeds block-opening capacity".to_string(),
            ));
        }
        layout.validate_trace_term_block_range(term.block_offset, block_span)?;
        if !term.block_offset.is_multiple_of(block_span) {
            return Err(AkitaError::InvalidInput(
                "trace term must begin at a claim boundary".to_string(),
            ));
        }

        let low_bits = r_pc / 2;
        let low_len = 1usize.checked_shl(low_bits as u32).ok_or_else(|| {
            AkitaError::InvalidInput("trace low-factor length overflow".to_string())
        })?;
        let high_len = 1usize
            .checked_shl((r_pc - low_bits) as u32)
            .ok_or_else(|| {
                AkitaError::InvalidInput("trace high-factor length overflow".to_string())
            })?;
        let factor_count = low_len.checked_add(high_len).ok_or_else(|| {
            AkitaError::InvalidInput("trace affine-factor count overflow".to_string())
        })?;
        if factor_count > akita_algebra::offset_eq::MAX_COMPACT_STRIDE_TERMS {
            return Err(AkitaError::InvalidSize {
                expected: akita_algebra::offset_eq::MAX_COMPACT_STRIDE_TERMS,
                actual: factor_count,
            });
        }
        let mut low_weights = Vec::new();
        low_weights.try_reserve_exact(low_len).map_err(|_| {
            AkitaError::InvalidInput("trace low-factor allocation failed".to_string())
        })?;
        for low in 0..low_len {
            low_weights.push(TraceAffineWeight {
                coordinates: block_weight_at_index_coords::<F, E>(
                    &term.b_open[..low_bits],
                    low,
                    term.basis,
                    k,
                    &gamma,
                ),
                gamma: Arc::clone(&gamma),
            });
        }
        let mut high_weights = Vec::new();
        high_weights.try_reserve_exact(high_len).map_err(|_| {
            AkitaError::InvalidInput("trace high-factor allocation failed".to_string())
        })?;
        for high in 0..high_len {
            high_weights.push(TraceAffineWeight {
                coordinates: block_weight_at_index_coords::<F, E>(
                    &term.b_open[low_bits..],
                    high,
                    term.basis,
                    k,
                    &gamma,
                ),
                gamma: Arc::clone(&gamma),
            });
        }

        // `V = Σ_plane gadget[plane] · Σ_j eq(col, e_index(j, plane)) · coords(b_j)`.
        // The descriptor owns the physical E address for every logical block.
        let mut v = vec![E::zero(); k];
        for unit in layout.witness_layout.units_for_group(layout.group_id)? {
            let block = term
                .block_offset
                .checked_add(unit.global_block_start())
                .ok_or_else(|| {
                    AkitaError::InvalidInput("trace term block index overflow".to_string())
                })?;
            let base = layout.opening_digit_col_index(block, 0)?;
            let contribution = eval_affine_digit_interval(
                col_point,
                base,
                unit.global_block_start(),
                unit.live_block_count(),
                layout.num_digits_open,
                &gadget_row,
                &high_weights,
                &low_weights,
            )?;
            for (slot, &value) in v.iter_mut().zip(&contribution.coordinates) {
                *slot += value;
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
