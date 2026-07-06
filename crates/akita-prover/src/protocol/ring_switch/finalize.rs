use super::*;

/// Inputs for building the unified relation-weight handoff during finalize.
pub struct RelationWeightFinalizeInputs<F: FieldCore, E: FieldCore> {
    pub trace_eval_target: E,
    pub trace: Option<super::evals::RelationWeightTraceBuild<F, E>>,
}

/// Complete the ring switch after the caller has bound the next witness.
///
/// Samples challenges and builds the evaluation tables for the fused sumcheck.
/// The caller must first absorb either the next-witness commitment or the
/// terminal cleartext witness bytes into `transcript`.
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
    gamma: Option<&[E]>,
    m_row_layout: MRowLayout,
    commitment: &RingVec<F>,
    relation_weight: RelationWeightFinalizeInputs<F, E>,
) -> Result<RingSwitchOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + Invertible,
    E: FpExtEncoding<F> + FromPrimitiveInt + ExtField<F>,
    T: Transcript<F>,
{
    let dims = instance.role_dims();
    let d_a = dims.d_a();
    dispatch_ring_dim_result!(d_a, |D| {
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
        let num_polys = opening_batch.num_total_polynomials();

        let num_ring_elems = w.len() / D;
        let live_x_cols = num_ring_elems;
        let col_bits = num_ring_elems
            .checked_next_power_of_two()
            .ok_or_else(|| {
                AkitaError::InvalidSetup("ring-switch column count overflow".to_string())
            })?
            .trailing_zeros() as usize;
        let ring_bits = D.trailing_zeros() as usize;
        let row_layout = instance.relation_row_layout(lp)?;
        let m_rows = row_layout.total_row_count();
        let num_sc_vars = col_bits + ring_bits;
        let num_i = m_rows
            .checked_next_power_of_two()
            .ok_or_else(|| AkitaError::InvalidSetup("ring-switch row count overflow".to_string()))?
            .trailing_zeros() as usize;

        let tau0: Vec<E> = match m_row_layout {
            MRowLayout::WithDBlock => (0..num_sc_vars)
                .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU0))
                .collect(),
            MRowLayout::WithoutDBlock => Vec::new(),
        };
        let tau1: Vec<E> = (0..num_i)
            .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU1))
            .collect();

        let challenges = &instance.challenges;
        if gamma.len() != instance.opening_batch().num_total_polynomials() {
            return Err(AkitaError::InvalidInput(
                "ring-switch gamma length does not match claim count".to_string(),
            ));
        }

        let relation_weight_claim =
            akita_types::relation_weight_claim_from_rows_extension_at_dims::<F, E>(
                dims,
                &tau1,
                alpha,
                lp.a_key.row_len(),
                relation_weight.trace_eval_target,
                instance.v(),
                commitment,
            )?;

        #[cfg(feature = "parallel")]
        let (relation_weight_evals_result, w_result) = rayon::join(
            || {
                super::evals::build_relation_weight_evals::<F, E>(
                    setup,
                    instance.opening_point(),
                    instance.ring_multiplier_point(),
                    challenges,
                    alpha,
                    dims,
                    lp,
                    &tau1,
                    num_polys,
                    opening_batch.num_groups(),
                    gamma,
                    m_row_layout,
                    ring_bits,
                    live_x_cols,
                    relation_weight.trace.clone(),
                )
            },
            || build_w_evals_compact(w.as_i8_digits(), D),
        );
        #[cfg(not(feature = "parallel"))]
        let (relation_weight_evals_result, w_result) = {
            let relation_weight_evals = super::evals::build_relation_weight_evals::<F, E>(
                setup,
                instance.opening_point(),
                instance.ring_multiplier_point(),
                challenges,
                alpha,
                dims,
                lp,
                &tau1,
                num_polys,
                opening_batch.num_groups(),
                gamma,
                m_row_layout,
                ring_bits,
                live_x_cols,
                relation_weight.trace.clone(),
            )?;
            let w_compact = build_w_evals_compact(w.as_i8_digits(), D);
            (Ok(relation_weight_evals), w_compact)
        };

        let relation_weight_evals = relation_weight_evals_result?;
        let (w_evals_compact, _, _) = w_result?;

        Ok(RingSwitchOutput {
            w_evals_compact,
            live_x_cols,
            relation_weight_evals,
            relation_weight_claim,
            col_bits,
            ring_bits,
            tau0,
            tau1,
            b: 1usize << lp.log_basis,
            alpha,
        })
    })
}
