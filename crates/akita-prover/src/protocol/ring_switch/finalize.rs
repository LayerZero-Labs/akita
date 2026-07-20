use super::*;
use akita_field::MulBaseUnreduced;
use akita_types::dispatch_for_field;

/// Complete the ring switch after the caller has bound the next witness.
///
/// Samples challenges and builds the evaluation tables for the fused sumcheck.
/// The caller must first absorb either the next-witness commitment or the
/// terminal cleartext witness bytes into `transcript`.
///
/// Only the current level's inner ring dimension is needed to expand the
/// full relation-weight table.
///
/// # Errors
///
/// Returns an error if the supplied gamma vector does not match the claim
/// count or if matrix expansion or evaluation-table construction fails.
#[tracing::instrument(skip_all, name = "ring_switch_finalize")]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn ring_switch_finalize<F, E, T>(
    instance: &RingRelationInstance<F>,
    setup: &AkitaExpandedSetup<F>,
    transcript: &mut T,
    w: &RecursiveWitnessFlat,
    lp: &LevelParams,
    opening_source_len: usize,
    opening_ring_dim: usize,
    gamma: Option<&[E]>,
    relation_matrix_row_layout: RelationMatrixRowLayout,
) -> Result<RingSwitchOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    let dims = instance.role_dims();
    let d_a = dims.d_a();
    dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Inner), F, d_a, |D| {
        let default_gamma;
        let gamma = if let Some(gamma) = gamma {
            gamma
        } else {
            default_gamma = instance
                .gamma()
                .iter()
                .copied()
                .map(E::lift_base)
                .collect::<Vec<_>>();
            &default_gamma
        };
        let alpha: E = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_RING_SWITCH);

        let opening_batch = instance.opening_batch();

        let opening_capacity = opening_source_len
            .checked_mul(opening_ring_dim)
            .ok_or_else(|| AkitaError::InvalidSetup("opening capacity overflow".into()))?;
        if opening_ring_dim == 0
            || !opening_ring_dim.is_power_of_two()
            || !w.len().is_multiple_of(opening_ring_dim)
            || w.len() > opening_capacity
        {
            return Err(AkitaError::InvalidInput(format!(
                "witness length {} does not fit opening capacity {} at ring dimension {}",
                w.len(),
                opening_capacity,
                opening_ring_dim,
            )));
        }
        let semantic_ring_elems = w.len() / D;
        let witness_layout = instance.segment_layout(lp, None).map_err(|err| {
            AkitaError::InvalidInput(format!("relation witness layout failed: {err:?}"))
        })?;
        if semantic_ring_elems != witness_layout.total_len() {
            return Err(AkitaError::InvalidSize {
                expected: witness_layout.total_len(),
                actual: semantic_ring_elems,
            });
        }
        // Uniform ring geometry retains the current separable (x, y) opening domain:
        // `col_bits` addresses the source columns and `ring_bits` addresses the
        // inner ring coefficients. This keeps the relation weights as a compact
        // per-column table `M(x)` from the semantic relation events instead of
        // the flattened field domain. Non-uniform role dimensions use the
        // flattened single-domain layout (`ring_bits = 0`).
        let x_capacity = akita_types::opening_domain_len(opening_source_len)?;
        let uniform = dims == akita_types::CommitmentRingDims::uniform(opening_ring_dim);
        let (live_x_cols, col_bits, ring_bits) = if uniform {
            (
                w.len() / opening_ring_dim,
                x_capacity.trailing_zeros() as usize,
                opening_ring_dim.trailing_zeros() as usize,
            )
        } else {
            let flat = x_capacity
                .checked_mul(opening_ring_dim)
                .ok_or_else(|| AkitaError::InvalidSetup("stage-2 domain overflow".into()))?;
            (w.len(), flat.trailing_zeros() as usize, 0usize)
        };
        let num_sc_vars = col_bits + ring_bits;
        let num_i =
            lp.relation_row_index_num_vars_for_layout(relation_matrix_row_layout, opening_batch)?;

        let tau0: Vec<E> = match relation_matrix_row_layout {
            RelationMatrixRowLayout::WithDBlock => (0..num_sc_vars)
                .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU0))
                .collect(),
            RelationMatrixRowLayout::WithoutCommitmentBlocks => Vec::new(),
        };
        let tau1: Vec<E> = (0..num_i)
            .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU1))
            .collect();
        if gamma.len() != instance.opening_batch().num_total_polynomials() {
            return Err(AkitaError::InvalidInput(
                "ring-switch gamma length does not match claim count".to_string(),
            ));
        }

        let build_relation_weights = || {
            let events = build_relation_weight_events(RelationWeightEventInputs {
                setup: RelationSetupSource::Matrix(setup),
                instance,
                alpha,
                level_params: lp,
                relation_row_point: &tau1,
                claim_coefficients: gamma,
                relation_matrix_row_layout,
                opening_source_len,
                opening_ring_dim,
            })?;
            if uniform {
                events.materialize_uniform_columns()
            } else {
                events.materialize_dense()
            }
        };

        #[cfg(feature = "parallel")]
        let (relation_weight_evals_result, w_result) = rayon::join(build_relation_weights, || {
            build_w_evals_compact(
                w.shared_i8_digits(),
                opening_ring_dim,
                1,
                opening_source_len,
            )
        });
        #[cfg(not(feature = "parallel"))]
        let (relation_weight_evals_result, w_result) = {
            let relation_weight_evals = build_relation_weights();
            let w_compact = build_w_evals_compact(
                w.shared_i8_digits(),
                opening_ring_dim,
                1,
                opening_source_len,
            );
            (relation_weight_evals, w_compact)
        };

        let relation_weight_evals = relation_weight_evals_result.map_err(|err| {
            AkitaError::InvalidInput(format!("relation-weight materialization failed: {err:?}"))
        })?;
        let (w_evals_compact, _, _) = w_result.map_err(|err| {
            AkitaError::InvalidInput(format!("witness opening materialization failed: {err:?}"))
        })?;

        Ok(RingSwitchOutput {
            w_evals_compact,
            live_x_cols,
            relation_weight_evals,
            col_bits,
            ring_bits,
            tau0,
            tau1,
            b: 1usize << lp.log_basis,
            alpha,
        })
    })
}
