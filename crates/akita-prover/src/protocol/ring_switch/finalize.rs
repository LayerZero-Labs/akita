use super::*;
use akita_field::MulBaseUnreduced;
use akita_types::dispatch_for_field;

/// Complete the ring switch after the caller has bound the next witness.
///
/// Samples challenges and builds the evaluation tables for the fused sumcheck.
/// The caller must first absorb the next-witness binding into `transcript`.
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
    lp: &CommittedGroupParams,
    opening_source_len: usize,
    opening_ring_dim: usize,
    gamma: Option<&[E]>,
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
        // Bind the low coefficient block shared by every role first, then the
        // remaining relation lanes. The flat challenge order is unchanged: the
        // common coefficients are the low Boolean coordinates.
        let x_capacity = akita_types::opening_domain_len(opening_source_len)?;
        let coeff_count = dims.common_relation_witness_coeff_count(opening_ring_dim);
        if coeff_count == 0
            || !coeff_count.is_power_of_two()
            || !w.len().is_multiple_of(coeff_count)
            || !opening_ring_dim.is_multiple_of(coeff_count)
        {
            return Err(AkitaError::InvalidSetup(
                "relation and outgoing witness do not admit a common coefficient block".into(),
            ));
        }
        let common_opening_source_len = opening_source_len
            .checked_mul(opening_ring_dim / coeff_count)
            .ok_or_else(|| AkitaError::InvalidSetup("common opening domain overflow".into()))?;
        let lane_capacity = x_capacity
            .checked_mul(opening_ring_dim / coeff_count)
            .ok_or_else(|| AkitaError::InvalidSetup("stage-2 lane domain overflow".into()))?;
        let live_x_cols = w.len() / coeff_count;
        let col_bits = lane_capacity.trailing_zeros() as usize;
        let ring_bits = coeff_count.trailing_zeros() as usize;
        // This is the Stage-1 transcript permutation boundary, not the Stage-2
        // coefficient split. On mixed paths tau0 is already sampled in flat
        // physical-address order, so zero means "no permutation," not "no low bits."
        let digit_range_equality_low_variable_count =
            if dims == akita_types::CommitmentRingDims::uniform(opening_ring_dim) {
                opening_ring_dim.trailing_zeros() as usize
            } else {
                0
            };
        let num_sc_vars = col_bits + ring_bits;
        let num_i = lp.relation_row_index_num_vars(opening_batch)?;

        let tau0: Vec<E> = (0..num_sc_vars)
            .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU0))
            .collect();
        let tau1: Vec<E> = (0..num_i)
            .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU1))
            .collect();
        if gamma.len() != instance.opening_batch().num_total_polynomials() {
            return Err(AkitaError::InvalidInput(
                "ring-switch gamma length does not match claim count".to_string(),
            ));
        }

        let prepare_relation_weight_factorization = || {
            let _span = tracing::info_span!("relation_weight_compilation").entered();
            let events = build_relation_weight_events(RelationWeightEventInputs {
                setup: RelationSetupSource::Matrix(setup),
                instance,
                alpha,
                level_params: lp,
                relation_row_point: &tau1,
                claim_coefficients: gamma,
                opening_source_len,
                opening_ring_dim,
            })?;
            events.factor_common_alpha()
        };

        #[cfg(feature = "parallel")]
        let (relation_weight_factorization_result, w_result) =
            rayon::join(prepare_relation_weight_factorization, || {
                build_w_evals_compact(
                    w.shared_i8_digits(),
                    coeff_count,
                    1,
                    common_opening_source_len,
                )
            });
        #[cfg(not(feature = "parallel"))]
        let (relation_weight_factorization_result, w_result) = {
            let relation_weight_factorization = prepare_relation_weight_factorization();
            let w_compact = build_w_evals_compact(
                w.shared_i8_digits(),
                coeff_count,
                1,
                common_opening_source_len,
            );
            (relation_weight_factorization, w_compact)
        };

        let relation_weight_factorization =
            relation_weight_factorization_result.map_err(|err| {
                AkitaError::InvalidInput(format!("relation-weight compilation failed: {err:?}"))
            })?;
        let (w_evals_compact, witness_col_bits, witness_ring_bits) = w_result.map_err(|err| {
            AkitaError::InvalidInput(format!("witness opening materialization failed: {err:?}"))
        })?;
        if witness_col_bits != col_bits || witness_ring_bits != ring_bits {
            return Err(AkitaError::InvalidSetup(
                "prepared witness geometry disagrees with the joint relation-witness split".into(),
            ));
        }

        Ok(RingSwitchOutput {
            w_evals_compact,
            live_x_cols,
            relation_weight_factorization,
            col_bits,
            ring_bits,
            digit_range_equality_low_variable_count,
            tau0,
            tau1,
            b: 1usize << lp.log_basis_open,
            alpha,
        })
    })
}
