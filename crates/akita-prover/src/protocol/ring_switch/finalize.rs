use super::*;
use akita_types::dispatch_for_field;
use jolt_field::MulBaseUnreduced;

/// Complete the ring switch after the caller has bound the next witness.
///
/// Samples challenges and builds the evaluation tables for the fused sumcheck.
/// The caller must first absorb either the next-witness commitment or the
/// terminal cleartext witness bytes into `transcript`.
///
/// Only the current level's `D` is needed for M-alpha expansion and
/// `alpha_evals_y`.
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

        let num_ring_elems = w.len() / D;
        let live_x_cols = num_ring_elems;
        let col_bits = num_ring_elems
            .checked_next_power_of_two()
            .ok_or_else(|| {
                AkitaError::InvalidSetup("ring-switch column count overflow".to_string())
            })?
            .trailing_zeros() as usize;
        let ring_bits = D.trailing_zeros() as usize;
        let num_sc_vars = col_bits + ring_bits;
        let num_i =
            lp.relation_row_index_num_vars_for_layout(relation_matrix_row_layout, opening_batch)?;

        let tau0: Vec<E> = match relation_matrix_row_layout {
            RelationMatrixRowLayout::WithDBlock => (0..num_sc_vars)
                .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU0))
                .collect(),
            RelationMatrixRowLayout::WithoutDBlock => Vec::new(),
        };
        let tau1: Vec<E> = (0..num_i)
            .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU1))
            .collect();
        let ring_alpha_evals_y = scalar_powers(alpha, D);
        let alpha_evals_y = scalar_powers(alpha, D);

        if gamma.len() != instance.opening_batch().num_total_polynomials() {
            return Err(AkitaError::InvalidInput(
                "ring-switch gamma length does not match claim count".to_string(),
            ));
        }

        #[cfg(feature = "parallel")]
        let (relation_matrix_col_evals_result, w_result) = rayon::join(
            || {
                compute_relation_matrix_col_evals::<F, E>(
                    setup,
                    instance,
                    alpha,
                    &ring_alpha_evals_y,
                    dims,
                    lp,
                    &tau1,
                    gamma,
                    relation_matrix_row_layout,
                )
            },
            || build_w_evals_compact(w.as_i8_digits(), D, 1),
        );
        #[cfg(not(feature = "parallel"))]
        let (relation_matrix_col_evals_result, w_result) = {
            let relation_matrix_col_evals = compute_relation_matrix_col_evals::<F, E>(
                setup,
                instance,
                alpha,
                &ring_alpha_evals_y,
                dims,
                lp,
                &tau1,
                gamma,
                relation_matrix_row_layout,
            )?;
            let w_compact = build_w_evals_compact(w.as_i8_digits(), D, 1);
            (Ok(relation_matrix_col_evals), w_compact)
        };

        let relation_matrix_col_evals = relation_matrix_col_evals_result?;
        let (w_evals_compact, _, _) = w_result?;

        Ok(RingSwitchOutput {
            w_evals_compact,
            live_x_cols,
            relation_matrix_col_evals,
            alpha_evals_y,
            col_bits,
            ring_bits,
            tau0,
            tau1,
            b: 1usize << lp.log_basis,
            alpha,
        })
    })
}
