use super::*;
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::scalar_powers;
use akita_types::{
    build_grouped_root_stage2_trace_table, build_trace_table_scaled,
    trace_public_weights_recursive, trace_public_weights_root_terms, CommitmentRingDims,
    LevelParams, MRowLayout, OpeningClaimsLayout, PreparedOpeningPoint, RingRelationInstance,
    TraceWeightLayout,
};

pub use akita_types::compute_grouped_m_evals_x;

/// Produce the compact `Vec<i8>` eval table of `w` for the fused prover.
///
/// The compact witness stays in the raw flat [`build_w_coeffs`] order.
///
/// # Errors
///
/// Returns an error if the witness length is not divisible by the ring
/// dimension.
pub fn build_w_evals_compact(w: &[i8], d: usize) -> Result<(Vec<i8>, usize, usize), AkitaError> {
    if !w.len().is_multiple_of(d) {
        return Err(AkitaError::InvalidSize {
            expected: d,
            actual: w.len(),
        });
    }
    let live_ring_count = w.len() / d;
    let col_bits = live_ring_count.next_power_of_two().trailing_zeros() as usize;
    let ring_bits = d.trailing_zeros() as usize;
    Ok((w.to_vec(), col_bits, ring_bits))
}

/// Trace segment inputs for unified relation-weight construction.
#[derive(Clone)]
pub enum RelationWeightTraceBuild<F: FieldCore, E: FieldCore> {
    /// Root fold: batched public row coefficients and block openings.
    Root {
        ring_d: usize,
        num_blocks: usize,
        layout: TraceWeightLayout,
        opening_batch: OpeningClaimsLayout,
        prepared_point: PreparedOpeningPoint<F, E>,
        row_coefficients: Vec<E>,
        trace_claim_scales: Option<Vec<E>>,
    },
    /// Recursive suffix: singleton fold with scaled trace weights.
    Recursive {
        ring_d: usize,
        layout: TraceWeightLayout,
        prepared: PreparedOpeningPoint<F, E>,
        trace_scale: E,
    },
    /// Grouped root: dense trace table with per-group block geometry.
    GroupedRoot {
        ring_d: usize,
        lp: Box<LevelParams>,
        layout: TraceWeightLayout,
        opening_batch: OpeningClaimsLayout,
        prepared_points: Vec<PreparedOpeningPoint<F, E>>,
        row_coefficients: Vec<E>,
        trace_claim_scales: Option<Vec<E>>,
        live_x_cols: usize,
    },
}

/// Build relation-weight evaluations from a full [`RingRelationInstance`].
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "build_relation_weight_evals_from_instance")]
pub fn build_relation_weight_evals_from_instance<F, E>(
    setup: &AkitaExpandedSetup<F>,
    instance: &RingRelationInstance<F>,
    alpha: E,
    role_dims: CommitmentRingDims,
    lp: &LevelParams,
    tau1: &[E],
    gamma: &[E],
    m_row_layout: MRowLayout,
    ring_bits: usize,
    live_x_cols: usize,
    trace: Option<RelationWeightTraceBuild<F, E>>,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + MulBase<F> + ExtField<F>,
{
    let d_a = role_dims.d_a();
    let ring_alpha_evals_y = scalar_powers(alpha, d_a);
    let alpha_evals_y = scalar_powers(alpha, 1usize << ring_bits);
    let relation_column_weights = compute_grouped_m_evals_x::<F, E>(
        setup,
        instance,
        alpha,
        &ring_alpha_evals_y,
        role_dims,
        lp,
        tau1,
        gamma,
        m_row_layout,
    )?;
    let y_len = alpha_evals_y.len();
    if relation_column_weights.len() < live_x_cols {
        return Err(AkitaError::InvalidSize {
            expected: live_x_cols,
            actual: relation_column_weights.len(),
        });
    }
    let trace_row_weight = {
        let eq_tau1 = EqPolynomial::evals(tau1)?;
        eq_tau1.first().copied().unwrap_or(E::zero())
    };
    let trace_dense = if let Some(trace) = trace {
        let table = match trace {
            RelationWeightTraceBuild::Root {
                ring_d,
                num_blocks,
                layout,
                opening_batch,
                prepared_point,
                row_coefficients,
                trace_claim_scales,
            } => akita_types::dispatch_ring_dim_result!(ring_d, |D| {
                let public_weights = trace_public_weights_root_terms::<F, E, D>(
                    num_blocks,
                    &opening_batch,
                    &prepared_point,
                    &row_coefficients,
                    trace_claim_scales.as_deref(),
                )?;
                build_trace_table_scaled(&layout, &public_weights, live_x_cols, E::one())
            })?,
            RelationWeightTraceBuild::Recursive {
                ring_d,
                layout,
                prepared,
                trace_scale,
            } => akita_types::dispatch_ring_dim_result!(ring_d, |D| {
                let public_weights =
                    trace_public_weights_recursive::<F, E, D>(&prepared, trace_scale)?;
                build_trace_table_scaled(&layout, &public_weights, live_x_cols, E::one())
            })?,
            RelationWeightTraceBuild::GroupedRoot {
                ring_d,
                lp,
                layout: _,
                opening_batch,
                prepared_points,
                row_coefficients,
                trace_claim_scales,
                live_x_cols: grouped_live_x_cols,
            } => akita_types::dispatch_ring_dim_result!(ring_d, |D| {
                build_grouped_root_stage2_trace_table::<F, E>(
                    ring_d,
                    lp.as_ref(),
                    &opening_batch,
                    &prepared_points,
                    &row_coefficients,
                    trace_claim_scales.as_deref(),
                    E::one(),
                    grouped_live_x_cols,
                )
            })?,
        };
        Some(
            table
                .materialize_dense(live_x_cols, y_len)
                .into_iter()
                .map(|v| v * trace_row_weight)
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    let trace_dense = trace_dense.as_deref();
    let mut out = Vec::with_capacity(live_x_cols * y_len);
    for (x, column_weight) in relation_column_weights
        .iter()
        .copied()
        .enumerate()
        .take(live_x_cols)
    {
        let trace_column = trace_dense.map(|trace| &trace[x * y_len..(x + 1) * y_len]);
        for (y, alpha_y) in alpha_evals_y.iter().copied().enumerate() {
            let trace = trace_column.map(|column| column[y]).unwrap_or(E::zero());
            out.push(alpha_y * column_weight + trace);
        }
    }
    Ok(out)
}
